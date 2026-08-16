//! Phase 1/2 of `docs/MEMORY_STREAMING_PLAN.md`: an on-demand, LRU-bounded pool of routed-expert
//! weights read directly from the original (unmodified) sharded safetensors checkpoint via
//! [`crate::nvidia_ckpt::ShardReader::read_expert_slice`] — no offline repack step, no
//! dequant/requant, no full-checkpoint materialization.
//!
//! This is the read/cache primitive only. It reproduces the same gate/up/down GEMM math as
//! `Qwen3_5SparseMoeBlock::expert_forward`'s bf16 fused-expert path
//! (`gate = silu(x @ gate_w^T)`, `up = x @ up_w^T`, `out = (gate*up) @ down_w^T`), so a caller can
//! compute one routed expert's MLP output while only ever holding `capacity` experts' worth of
//! weights resident, instead of all `num_experts`. It is not wired into the model's forward pass
//! yet — see `examples/expert_stream_probe.rs` for a standalone correctness/memory check against the
//! real checkpoint, and the plan doc's "phased rollout" section for what's still needed to use this
//! from `moe_grouped.rs` / `capture.rs`.

use std::collections::BTreeMap;
use std::num::NonZeroUsize;
use std::path::Path;

use burn::prelude::Device;
use burn::tensor::activation::silu;
use burn::tensor::{DType, Tensor, TensorData};
use lru::LruCache;

use crate::linear2d::Precision;
#[cfg(feature = "cubecl-gpu")]
use crate::nvfp4::quantize_nvfp4_from_nk_bf16;
#[cfg(feature = "cubecl-gpu")]
use crate::nvfp4_linear::Nvfp4Linear;
use crate::nvidia_ckpt::ShardReader;

/// Which fused projection tensor within a layer's routed-expert stack.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Proj {
    /// `experts.gate_up_proj`, shape `[num_experts, 2*inner, hidden]`.
    GateUp,
    /// `experts.down_proj`, shape `[num_experts, hidden, inner]`.
    Down,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct SlotKey {
    layer: usize,
    proj: Proj,
    expert: usize,
}

enum CachedSlot {
    Dense(Tensor<2>),
    #[cfg(feature = "cubecl-gpu")]
    Nvfp4(Nvfp4Linear),
}

#[cfg(feature = "cubecl-gpu")]
fn stream_nvfp4() -> bool {
    std::env::var("QWEN35_STREAM_NVFP4").ok().as_deref() != Some("0")
}

/// Whether `expert_forward` blocks on `device.sync()` after each individual routed expert.
///
/// Defaults to on, preserving the original behaviour: the sync bounds the in-flight async dispatch
/// backlog, which previously kept resident/swap memory from ballooning past the resident-core +
/// pool-capacity bound. But it also serialises ~320 tiny per-expert GEMVs per token against the
/// host, and measurement shows that costs ~5 ms per expert -- far more than the 1-row GEMV math
/// itself. `QWEN35_STREAM_SYNC=0` drops to one sync per layer (issued by the caller) so the
/// dispatches can pipeline; watch memory when enabling it. NVFP4 slots are ~4x smaller than the
/// bf16 tensors this guard was originally sized against, so the backlog risk is correspondingly lower.
fn sync_per_expert() -> bool {
    std::env::var("QWEN35_STREAM_SYNC").ok().as_deref() != Some("0")
}

#[cfg(feature = "cubecl-gpu")]
fn pack_out_in_nvfp4(
    bf16_bytes: &[u8],
    n: usize,
    k: usize,
    device: &Device,
) -> Result<Nvfp4Linear, String> {
    if k % 16 != 0 {
        return Err(format!("NVFP4 stream pack requires K%16==0, got K={k}"));
    }
    if bf16_bytes.len() != n * k * 2 {
        return Err(format!(
            "NVFP4 stream pack: byte length {} != N*K*2 = {}",
            bf16_bytes.len(),
            n * k * 2
        ));
    }
    // Fused transpose+quantize straight from the [N,K] bf16 source. Bit-identical to the previous
    // "transpose into a [K,N] f32 scratch, then quantize_nvfp4" pair (asserted by
    // `nvfp4::tests::fused_nk_bf16_quantize_is_bit_identical_to_transpose_then_quantize`) but
    // without the two stride-`n` passes and the k*n f32 scratch allocation -- this runs on every
    // routed expert on every decode step, so it is the hot path.
    let (qw, bs, gscale) = quantize_nvfp4_from_nk_bf16(bf16_bytes, k, n);
    Ok(Nvfp4Linear::from_packed_parts(qw, bs, gscale, k, n, device).with_m_max(8))
}

/// LRU-bounded cache of decoded expert weight tensors, backed by on-demand mmap reads of the
/// original checkpoint shards.
///
/// Uses `lru::LruCache`, an O(1)-per-access intrusive linked-hashmap, instead of a hand-rolled
/// `HashMap` + `VecDeque` (the latter's `touch()` did an O(capacity) linear scan on *every* fetch,
/// hit or miss, which made larger pool capacities pathologically slower overall despite fewer
/// cache misses — measured 44.4s decode at capacity=512 vs 109.0s at capacity=8192 on the real
/// checkpoint before this fix).
pub struct ExpertSlotPool<'a> {
    reader: ShardReader<'a>,
    cache: LruCache<SlotKey, CachedSlot>,
    pub hits: usize,
    pub misses: usize,
    /// Cumulative wall time inside `read_expert_slice` (mmap read + byte-slice copy), nanoseconds.
    pub io_ns: u64,
    /// Cumulative wall time inside `Tensor::from_data` (host->device upload call), nanoseconds.
    pub upload_ns: u64,
    /// Cumulative wall time inside `expert_forward`'s matmul/activation calls, nanoseconds.
    pub compute_ns: u64,
}

impl<'a> ExpertSlotPool<'a> {
    /// `capacity` is the max number of (layer, projection, expert) slots kept resident at once —
    /// e.g. `2 * num_experts_per_tok` gives headroom for a token's routed set changing between
    /// consecutive decode steps without evicting mid-step.
    pub fn new(dir: &'a Path, index: &'a BTreeMap<String, String>, capacity: usize) -> Self {
        Self {
            reader: ShardReader::new(dir, index),
            cache: LruCache::new(NonZeroUsize::new(capacity.max(1)).unwrap()),
            hits: 0,
            misses: 0,
            io_ns: 0,
            upload_ns: 0,
            compute_ns: 0,
        }
    }

    fn fetch_slot(
        &mut self,
        key: SlotKey,
        tensor_key: &str,
        device: &Device,
    ) -> Result<CachedSlot, String> {
        if let Some(t) = self.cache.get(&key) {
            self.hits += 1;
            return Ok(match t {
                CachedSlot::Dense(tensor) => CachedSlot::Dense(tensor.clone()),
                #[cfg(feature = "cubecl-gpu")]
                CachedSlot::Nvfp4(lin) => CachedSlot::Nvfp4(lin.clone()),
            });
        }
        self.misses += 1;
        let t0 = std::time::Instant::now();
        let raw = self.reader.read_expert_slice(tensor_key, key.expert)?;
        let dims2: Vec<usize> = raw.shape[1..].to_vec();
        let expected = dims2.iter().product::<usize>() * 2;
        if raw.data.len() != expected {
            return Err(format!(
                "{tensor_key} expert {}: byte length {} != shape {:?} * 2 (BF16)",
                key.expert,
                raw.data.len(),
                dims2
            ));
        }
        self.io_ns += t0.elapsed().as_nanos() as u64;
        let t1 = std::time::Instant::now();
        #[cfg(feature = "cubecl-gpu")]
        if stream_nvfp4() {
            let (n, k) = match dims2.as_slice() {
                [n, k] => (*n, *k),
                _ => {
                    return Err(format!(
                        "{tensor_key} expert {}: expected rank-2 [N,K], got {dims2:?}",
                        key.expert
                    ));
                }
            };
            let packed = pack_out_in_nvfp4(&raw.data, n, k, device)?;
            self.upload_ns += t1.elapsed().as_nanos() as u64;
            self.cache.put(key, CachedSlot::Nvfp4(packed.clone()));
            return Ok(CachedSlot::Nvfp4(packed));
        }
        let data = TensorData::from_bytes_vec(raw.data, dims2, DType::BF16);
        let tensor = Tensor::<2>::from_data(data, (device, DType::BF16));
        self.upload_ns += t1.elapsed().as_nanos() as u64;
        self.cache.put(key, CachedSlot::Dense(tensor.clone()));
        Ok(CachedSlot::Dense(tensor))
    }

    /// Batched prefetch: for a given layer and a set of distinct expert indices, fetch every
    /// still-missing `gate_up`/`down` slot in two batched reader calls (one per projection)
    /// instead of `2 * experts.len()` separate `read_expert_slice` calls. Populates the cache so
    /// subsequent `gate_up`/`down`/`expert_forward` calls for these (layer, expert) pairs hit
    /// cache. Already-cached experts are skipped (touched via `cache.get`, promoting them in the
    /// LRU without a redundant fetch). Caller should size `experts` to fit within `capacity()`
    /// (headroom = `2 * experts.len()` slots) to avoid self-eviction mid-batch.
    pub fn prefetch_layer(
        &mut self,
        layer: usize,
        experts: &[usize],
        device: &Device,
    ) -> Result<(), String> {
        self.prefetch_proj(layer, Proj::GateUp, experts, device)?;
        self.prefetch_proj(layer, Proj::Down, experts, device)?;
        Ok(())
    }

    fn prefetch_proj(
        &mut self,
        layer: usize,
        proj: Proj,
        experts: &[usize],
        device: &Device,
    ) -> Result<(), String> {
        let missing: Vec<usize> = experts
            .iter()
            .copied()
            .filter(|&e| {
                let key = SlotKey {
                    layer,
                    proj,
                    expert: e,
                };
                if self.cache.get(&key).is_some() {
                    self.hits += 1;
                    false
                } else {
                    true
                }
            })
            .collect();
        if missing.is_empty() {
            return Ok(());
        }
        let tensor_key = match proj {
            Proj::GateUp => Self::gate_up_key(layer),
            Proj::Down => Self::down_key(layer),
        };
        let t0 = std::time::Instant::now();
        let raws = self.reader.read_expert_slices(&tensor_key, &missing)?;
        self.io_ns += t0.elapsed().as_nanos() as u64;
        for (expert_idx, raw) in raws {
            self.misses += 1;
            let dims2: Vec<usize> = raw.shape[1..].to_vec();
            let expected = dims2.iter().product::<usize>() * 2; // BF16 = 2 bytes/elem
            if raw.data.len() != expected {
                return Err(format!(
                    "{tensor_key} expert {}: byte length {} != shape {:?} * 2 (BF16)",
                    expert_idx,
                    raw.data.len(),
                    dims2
                ));
            }
            let key = SlotKey {
                layer,
                proj,
                expert: expert_idx,
            };
            #[cfg(feature = "cubecl-gpu")]
            if stream_nvfp4() {
                let (n, k) = match dims2.as_slice() {
                    [n, k] => (*n, *k),
                    _ => {
                        return Err(format!(
                            "{tensor_key} expert {expert_idx}: expected rank-2 [N,K], got {dims2:?}"
                        ));
                    }
                };
                let t1 = std::time::Instant::now();
                let packed = pack_out_in_nvfp4(&raw.data, n, k, device)?;
                self.upload_ns += t1.elapsed().as_nanos() as u64;
                self.cache.put(key, CachedSlot::Nvfp4(packed));
                continue;
            }
            let data = TensorData::from_bytes_vec(raw.data, dims2, DType::BF16);
            let t1 = std::time::Instant::now();
            let tensor = Tensor::<2>::from_data(data, (device, DType::BF16));
            self.upload_ns += t1.elapsed().as_nanos() as u64;
            self.cache.put(key, CachedSlot::Dense(tensor));
        }
        Ok(())
    }

    fn gate_up_key(layer: usize) -> String {
        format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj")
    }

    fn down_key(layer: usize) -> String {
        format!("model.language_model.layers.{layer}.mlp.experts.down_proj")
    }

    /// Fetch (from cache or disk) the `[2*inner, hidden]` gate_up weight for one expert.
    pub fn gate_up(
        &mut self,
        layer: usize,
        expert: usize,
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        let key = SlotKey {
            layer,
            proj: Proj::GateUp,
            expert,
        };
        let tensor_key = Self::gate_up_key(layer);
        match self.fetch_slot(key, &tensor_key, device)? {
            CachedSlot::Dense(t) => Ok(t),
            #[cfg(feature = "cubecl-gpu")]
            CachedSlot::Nvfp4(_) => Err("gate_up: slot is NVFP4-packed; use expert_forward".into()),
        }
    }

    /// Fetch (from cache or disk) the `[hidden, inner]` down weight for one expert.
    pub fn down(
        &mut self,
        layer: usize,
        expert: usize,
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        let key = SlotKey {
            layer,
            proj: Proj::Down,
            expert,
        };
        let tensor_key = Self::down_key(layer);
        match self.fetch_slot(key, &tensor_key, device)? {
            CachedSlot::Dense(t) => Ok(t),
            #[cfg(feature = "cubecl-gpu")]
            CachedSlot::Nvfp4(_) => Err("down: slot is NVFP4-packed; use expert_forward".into()),
        }
    }

    /// Compute one routed expert's MLP output for a `[tokens, hidden]` input, streaming that
    /// expert's weights through this pool. Reproduces
    /// `Qwen3_5SparseMoeBlock::expert_forward`'s bf16 fused-expert math exactly (see module docs).
    pub fn expert_forward(
        &mut self,
        layer: usize,
        expert: usize,
        x2: Tensor<2>,
        prec: Precision,
        device: &Device,
    ) -> Result<Tensor<2>, String> {
        #[cfg(feature = "cubecl-gpu")]
        if stream_nvfp4() {
            let gu_key = SlotKey {
                layer,
                proj: Proj::GateUp,
                expert,
            };
            let dn_key = SlotKey {
                layer,
                proj: Proj::Down,
                expert,
            };
            let gate_up = match self.fetch_slot(gu_key, &Self::gate_up_key(layer), device)? {
                CachedSlot::Nvfp4(lin) => lin,
                CachedSlot::Dense(_) => {
                    return Err("expected NVFP4 gate_up slot".into());
                }
            };
            let down = match self.fetch_slot(dn_key, &Self::down_key(layer), device)? {
                CachedSlot::Nvfp4(lin) => lin,
                CachedSlot::Dense(_) => {
                    return Err("expected NVFP4 down slot".into());
                }
            };
            let t = std::time::Instant::now();
            let x = x2.cast(DType::F32);
            let gu = gate_up.forward(x);
            let inner = gu.dims()[1] / 2;
            let m = gu.dims()[0];
            let gate = silu(gu.clone().slice([0..m, 0..inner]));
            let up = gu.slice([0..m, inner..inner * 2]);
            let out = down.forward(gate * up);
            if sync_per_expert() {
                let _ = device.sync();
            }
            self.compute_ns += t.elapsed().as_nanos() as u64;
            return Ok(out);
        }

        let gate_up = self.gate_up(layer, expert, device)?;
        let [two_inner, hidden] = gate_up.dims();
        let inner = two_inner / 2;
        let gate_w = gate_up.clone().slice([0..inner, 0..hidden]);
        let up_w = gate_up.slice([inner..two_inner, 0..hidden]);
        let down_w = self.down(layer, expert, device)?;

        let t = std::time::Instant::now();
        let gate = silu(matmul_out_in(x2.clone(), gate_w, prec));
        let up = matmul_out_in(x2, up_w, prec);
        let out = matmul_out_in(gate * up, down_w, prec);
        if !sync_per_expert() {
            self.compute_ns += t.elapsed().as_nanos() as u64;
            return Ok(out);
        }
        // Bound the in-flight async dispatch backlog: without this, ~1280 tiny per-expert matmuls
        // per decode step queue up unsynced (cubecl/wgpu buffers work async), which was suspected
        // to be inflating resident/swap memory far past the resident-core + pool-capacity bound --
        // see docs/MEMORY_STREAMING_PLAN.md's perf notes for the measured before/after.
        let _ = device.sync();
        self.compute_ns += t.elapsed().as_nanos() as u64;
        Ok(out)
    }

    /// Currently-resident slot count (<= capacity).
    pub fn resident_slots(&self) -> usize {
        self.cache.len()
    }

    /// Currently-configured slot capacity.
    pub fn capacity(&self) -> usize {
        self.cache.cap().get()
    }

    /// Human-readable breakdown of where fetch/compute time went (mmap read+copy, host->device
    /// upload, matmul/activation compute), for `QWEN35_STREAM_PROFILE=1`-style diagnostics.
    pub fn timing_report(&self) -> String {
        let ms = |ns: u64| ns as f64 / 1e6;
        format!(
            "io={:.1}ms upload={:.1}ms compute={:.1}ms (misses={})",
            ms(self.io_ns),
            ms(self.upload_ns),
            ms(self.compute_ns),
            self.misses
        )
    }
}

/// Duplicated from `qwen3_5::matmul_out_in` (private to that module) so this file has no
/// dependency on `qwen3_5`'s internals beyond the public `Precision` type.
fn matmul_out_in(x: Tensor<2>, weight_out_in: Tensor<2>, prec: Precision) -> Tensor<2> {
    let weight_in_out = weight_out_in.transpose();
    let xdt = x.dtype();
    match prec {
        Precision::F32 => x.matmul(weight_in_out.cast(xdt)),
        Precision::Bf16 => x
            .cast(DType::BF16)
            .matmul(weight_in_out.cast(DType::BF16))
            .cast(DType::F32),
        Precision::F16 => x
            .cast(DType::F16)
            .matmul(weight_in_out.cast(DType::F16))
            .cast(DType::F32),
    }
}
