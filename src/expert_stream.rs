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
#[cfg(feature = "cubecl-gpu")]
use std::sync::Arc;

use burn::prelude::Device;
use burn::tensor::activation::silu;
use burn::tensor::{DType, Tensor, TensorData};
use lru::LruCache;

use crate::linear2d::Precision;
#[cfg(feature = "cubecl-gpu")]
use crate::nvfp4::quantize_nvfp4_from_nk_bf16;
#[cfg(feature = "cubecl-gpu")]
use crate::nvfp4_blob::{BlobManifest, BlobProj};
#[cfg(feature = "cubecl-gpu")]
use crate::nvfp4_linear::Nvfp4Linear;
use crate::nvidia_ckpt::ShardReader;

/// Offline NVFP4 expert store (see `crate::nvfp4_blob`), read via bounded parallel `pread`.
///
/// Earlier version of this used `mmap`. turbo-fieldfare's own I/O experiments
/// (`turbo-fieldfare/docs/experiments/summaries/01-model-install-and-expert-io.md`, IO-01) measured
/// `mmap` as **3.5x slower per cold read** than explicit `pread` in the same regime we're in --
/// working set (17 GiB) exceeds usable RAM (16 GiB), so every miss is a genuine cold fault either
/// way, and the VM layer adds overhead `pread` doesn't pay. Their end-to-end simulator: 0.50 tok/s
/// with `mmap` vs 3.97 tok/s with parallel `pread`. `File::read_exact_at` is positional (no shared
/// seek cursor), so many threads can safely read the same `File` concurrently -- that's what makes
/// the bounded parallel read pool below possible without any locking.
#[cfg(feature = "cubecl-gpu")]
struct BlobSource {
    dir: std::path::PathBuf,
    manifest: BlobManifest,
    files: std::collections::HashMap<usize, Arc<std::fs::File>>,
}

#[cfg(feature = "cubecl-gpu")]
impl BlobSource {
    fn open(dir: std::path::PathBuf) -> Result<Self, String> {
        let manifest = BlobManifest::load(&dir)?;
        Ok(Self {
            dir,
            manifest,
            files: std::collections::HashMap::new(),
        })
    }

    fn file_for_layer(&mut self, layer: usize) -> Result<Arc<std::fs::File>, String> {
        if let Some(f) = self.files.get(&layer) {
            return Ok(f.clone());
        }
        let path = BlobManifest::layer_path(&self.dir, layer);
        let file = std::fs::File::open(&path)
            .map_err(|e| format!("open NVFP4 blob {}: {e}", path.display()))?;
        let want = self.manifest.layer_file_len() as u64;
        let got = file
            .metadata()
            .map_err(|e| format!("stat {}: {e}", path.display()))?
            .len();
        if got != want {
            return Err(format!(
                "NVFP4 blob {} is {got} bytes, expected {want}; rebuild the store",
                path.display()
            ));
        }
        let file = Arc::new(file);
        self.files.insert(layer, file.clone());
        Ok(file)
    }
}

/// One expert's raw NVFP4 record, read off disk with no decode/quantize step.
#[cfg(feature = "cubecl-gpu")]
type RawRecord = (Vec<u8>, Vec<u8>, f32); // (qw, block_scales, gscale)

/// `pread` one expert's record out of an open layer file. Positional (`read_exact_at`), so this is
/// safe to call concurrently from multiple threads against the same `File`.
#[cfg(feature = "cubecl-gpu")]
fn pread_record(
    file: &std::fs::File,
    layout: &crate::nvfp4_blob::ProjLayout,
    expert: usize,
) -> Result<RawRecord, String> {
    use std::os::unix::fs::FileExt;
    let base = layout.record_offset(expert) as u64;
    let mut buf = vec![0u8; layout.stride];
    file.read_exact_at(&mut buf, base)
        .map_err(|e| format!("pread NVFP4 record (expert {expert}): {e}"))?;
    let (qw, rest) = buf.split_at(layout.qw_len());
    let (block_scales, gs) = rest.split_at(layout.bs_len());
    let gscale = f32::from_le_bytes([gs[0], gs[1], gs[2], gs[3]]);
    Ok((qw.to_vec(), block_scales.to_vec(), gscale))
}

/// How many OS threads read misses concurrently for one layer's prefetch. turbo-fieldfare's own
/// sweep (DEC/IO summaries) tested 4 and 8 workers; 8 is I/O-bound (blocking syscalls, not spinning)
/// so it doesn't compete with GPU dispatch on the main thread the way the earlier CPU-bound
/// `std::thread::scope` quantize parallelization did (that oversubscribed P-cores; this doesn't).
#[cfg(feature = "cubecl-gpu")]
const PREAD_WORKERS: usize = 8;

/// A layer's misses, currently being read off disk in the background while the caller can dispatch
/// that layer's cache *hits* on the GPU. See `ExpertSlotPool::prefetch_layer_begin`.
#[cfg(feature = "cubecl-gpu")]
struct PendingReads {
    layer: usize,
    proj: Proj,
    handle: std::thread::JoinHandle<Vec<(usize, Result<RawRecord, String>)>>,
}

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
pub(crate) fn stream_nvfp4() -> bool {
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
/// Defaults to false (pipelined dispatch): syncing per-expert forces ~320 GPU pipeline drains
/// per token, adding ~160-320 ms of pure synchronization stall. Set QWEN35_STREAM_SYNC=1 only for debugging.
fn sync_per_expert() -> bool {
    std::env::var("QWEN35_STREAM_SYNC").ok().as_deref() == Some("1")
}

/// Comptime batch bound baked into the NVFP4 GEMV (`nvfp4_decode_gemv`'s `m_max`).
///
/// This is NOT free headroom: `m_max` is `#[comptime]`, so the kernel unrolls `m_max` accumulator
/// copies AND issues `m_max` `plane_sum` warp reductions per output column regardless of the actual
/// row count. At decode `m_dim == 1`, so the long-standing `m_max = 8` did 8x the reductions and
/// carried 8x the register pressure to serve one row. Rows beyond `m_max` are handled by chunking
/// in `Nvfp4Linear::forward`, so a smaller bound only costs extra launches during prefill (~14 us
/// each), which is far cheaper than the per-column waste at decode.
/// Override with QWEN35_STREAM_M_MAX (1..=8).
fn stream_m_max() -> usize {
    std::env::var("QWEN35_STREAM_M_MAX")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|v| (1..=8).contains(v))
        .unwrap_or(1)
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
    Ok(Nvfp4Linear::from_packed_parts(qw, bs, gscale, k, n, device).with_m_max(stream_m_max()))
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
    /// Offline NVFP4 store, when `QWEN35_NVFP4_BLOB_DIR` points at a valid one.
    #[cfg(feature = "cubecl-gpu")]
    blob: Option<BlobSource>,
    /// In-flight background reads started by `prefetch_layer_begin`, joined by `prefetch_layer_finish`.
    #[cfg(feature = "cubecl-gpu")]
    pending: Vec<PendingReads>,
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
        #[cfg(feature = "cubecl-gpu")]
        let blob = match std::env::var("QWEN35_NVFP4_BLOB_DIR") {
            Ok(p) if !p.is_empty() => match BlobSource::open(std::path::PathBuf::from(&p)) {
                Ok(src) => {
                    eprintln!("streamed experts: using offline NVFP4 blob store at {p}");
                    Some(src)
                }
                // A misconfigured store must not silently fall back to the 10x slower bf16 path.
                Err(e) => panic!("QWEN35_NVFP4_BLOB_DIR={p} could not be opened: {e}"),
            },
            _ => {
                let candidates = [
                    std::path::PathBuf::from("models-nvfp4"),
                    dir.join("models-nvfp4"),
                    dir.to_path_buf(),
                ];
                candidates.into_iter().find_map(|p| {
                    if p.join("manifest.txt").exists() {
                        match BlobSource::open(p.clone()) {
                            Ok(src) => {
                                eprintln!("streamed experts: auto-detected offline NVFP4 blob store at {}", p.display());
                                Some(src)
                            }
                            Err(_) => None,
                        }
                    } else {
                        None
                    }
                })
            }
        };
        Self {
            reader: ShardReader::new(dir, index),
            #[cfg(feature = "cubecl-gpu")]
            blob,
            #[cfg(feature = "cubecl-gpu")]
            pending: Vec::new(),
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
        self.prefetch_layer_begin(layer, experts, device)?;
        self.prefetch_layer_finish(device)
    }

    /// Hit-first split, phase 1: classify `experts` into cache hits (already resident, promoted in
    /// the LRU, ready to dispatch immediately) versus misses, and for the offline NVFP4 blob store
    /// (`QWEN35_NVFP4_BLOB_DIR`), start reading the misses on a bounded background thread pool
    /// (`pread`, not `mmap` -- see `BlobSource`'s doc comment for why) instead of blocking on them.
    ///
    /// Caller should run `expert_forward` for the returned hits *before* calling
    /// [`Self::prefetch_layer_finish`], so that GPU dispatch of the already-resident experts
    /// overlaps the background disk reads instead of waiting behind them -- this is the same
    /// "hit-first execution" split turbo-fieldfare measured as a 14.4% win over blocking every
    /// expert on I/O (`turbo-fieldfare/docs/experiments/summaries/02-decode-moe-int4-and-router.md`,
    /// DEC-18).
    ///
    /// Without the blob store (bf16 fallback), this eagerly does the old synchronous read+quantize
    /// for all misses and returns only the true hits -- still correct, just without the overlap,
    /// since `ShardReader::read_expert_slices` takes `&mut self` and isn't safely parallelizable
    /// without a larger refactor.
    pub fn prefetch_layer_begin(
        &mut self,
        layer: usize,
        experts: &[usize],
        device: &Device,
    ) -> Result<Vec<usize>, String> {
        #[cfg(feature = "cubecl-gpu")]
        {
            use std::collections::HashSet;
            let mut missed: HashSet<usize> = HashSet::new();
            for proj in [Proj::GateUp, Proj::Down] {
                let missing = self.classify_and_spawn(layer, proj, experts, device)?;
                missed.extend(missing);
            }
            return Ok(experts
                .iter()
                .copied()
                .filter(|e| !missed.contains(e))
                .collect());
        }
        #[cfg(not(feature = "cubecl-gpu"))]
        {
            self.prefetch_proj(layer, Proj::GateUp, experts, device)?;
            self.prefetch_proj(layer, Proj::Down, experts, device)?;
            Ok(Vec::new())
        }
    }

    /// Hit-first split, phase 2: join the background reads started by
    /// [`Self::prefetch_layer_begin`] and upload them into the cache. After this returns, every
    /// expert passed to `prefetch_layer_begin` is resident.
    #[cfg(feature = "cubecl-gpu")]
    pub fn prefetch_layer_finish(&mut self, device: &Device) -> Result<(), String> {
        let t0 = std::time::Instant::now();
        let pending = std::mem::take(&mut self.pending);
        let mut joined: Vec<(SlotKey, RawRecord)> = Vec::new();
        for pr in pending {
            let results = pr
                .handle
                .join()
                .map_err(|_| "NVFP4 blob read worker thread panicked".to_string())?;
            for (expert, res) in results {
                let rec = res?;
                joined.push((
                    SlotKey {
                        layer: pr.layer,
                        proj: pr.proj,
                        expert,
                    },
                    rec,
                ));
            }
        }
        self.io_ns += t0.elapsed().as_nanos() as u64;

        let t1 = std::time::Instant::now();
        for (key, (qw, bs, gscale)) in joined {
            let layout = *self
                .blob
                .as_ref()
                .expect("pending reads imply blob is present")
                .manifest
                .layout(match key.proj {
                    Proj::GateUp => BlobProj::GateUp,
                    Proj::Down => BlobProj::Down,
                });
            let packed = Nvfp4Linear::from_packed_parts(qw, bs, gscale, layout.k, layout.n, device)
                .with_m_max(stream_m_max());
            self.cache.put(key, CachedSlot::Nvfp4(packed));
        }
        self.upload_ns += t1.elapsed().as_nanos() as u64;
        Ok(())
    }

    #[cfg(not(feature = "cubecl-gpu"))]
    pub fn prefetch_layer_finish(&mut self, _device: &Device) -> Result<(), String> {
        Ok(())
    }

    /// Classify one projection's experts into hits/misses (touching the LRU for hits) and, if the
    /// blob store is active, spawn [`PREAD_WORKERS`] threads to read the misses in the background.
    /// Returns the miss list (bf16 fallback: misses are read synchronously here instead, so the
    /// returned list can be treated as "no longer missing" by the caller either way).
    #[cfg(feature = "cubecl-gpu")]
    fn classify_and_spawn(
        &mut self,
        layer: usize,
        proj: Proj,
        experts: &[usize],
        device: &Device,
    ) -> Result<Vec<usize>, String> {
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
            return Ok(missing);
        }

        if let Some(blob) = self.blob.as_mut() {
            self.misses += missing.len();
            let file = blob.file_for_layer(layer)?;
            let blob_proj = match proj {
                Proj::GateUp => BlobProj::GateUp,
                Proj::Down => BlobProj::Down,
            };
            let layout = *blob.manifest.layout(blob_proj);
            let workers = PREAD_WORKERS.min(missing.len()).max(1);
            // Chunk (not one-thread-per-expert): bounds thread count regardless of miss count, and
            // these are blocking `pread` syscalls, so a chunk's threads sleep while waiting on disk
            // rather than competing with the main thread's GPU dispatch for a P-core.
            let chunks: Vec<Vec<usize>> = {
                let mut out = vec![Vec::new(); workers];
                for (i, e) in missing.iter().enumerate() {
                    out[i % workers].push(*e);
                }
                out
            };
            for chunk in chunks {
                if chunk.is_empty() {
                    continue;
                }
                let file = file.clone();
                let layout = layout;
                let handle = std::thread::spawn(move || {
                    chunk
                        .into_iter()
                        .map(|e| (e, pread_record(&file, &layout, e)))
                        .collect::<Vec<_>>()
                });
                self.pending.push(PendingReads {
                    layer,
                    proj,
                    handle,
                });
            }
            return Ok(missing);
        }

        // No blob store: fall back to the old synchronous bf16 read+quantize path so behaviour is
        // unchanged. `prefetch_proj` recomputes its own hit/miss filter over `missing`, but since
        // every key here is a genuine miss that filter is a no-op re-check (0 additional hits) --
        // the real miss count still comes from its read loop, so this doesn't double-count.
        self.prefetch_proj(layer, proj, &missing, device)?;
        Ok(missing)
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
        // The offline blob store is handled entirely by `classify_and_spawn` / `prefetch_layer_begin`
        // + `prefetch_layer_finish` (bounded parallel `pread`, hit-first overlap). `prefetch_proj` is
        // now only the bf16-checkpoint fallback path used when no blob store is configured.

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
            CachedSlot::Nvfp4(_) => {
                Err("gate_up: slot is NVFP4-packed; use expert_forward or gate_up_nvfp4".into())
            }
        }
    }

    /// Fetch (from cache or disk) the NVFP4 `gate_up` weight for one expert.
    #[cfg(feature = "cubecl-gpu")]
    pub fn gate_up_nvfp4(
        &mut self,
        layer: usize,
        expert: usize,
        device: &Device,
    ) -> Result<Nvfp4Linear, String> {
        let key = SlotKey {
            layer,
            proj: Proj::GateUp,
            expert,
        };
        let tensor_key = Self::gate_up_key(layer);
        match self.fetch_slot(key, &tensor_key, device)? {
            CachedSlot::Nvfp4(lin) => Ok(lin),
            CachedSlot::Dense(_) => Err("gate_up_nvfp4: slot is Dense; expected NVFP4".into()),
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
            CachedSlot::Nvfp4(_) => {
                Err("down: slot is NVFP4-packed; use expert_forward or down_nvfp4".into())
            }
        }
    }

    /// Fetch (from cache or disk) the NVFP4 `down` weight for one expert.
    #[cfg(feature = "cubecl-gpu")]
    pub fn down_nvfp4(
        &mut self,
        layer: usize,
        expert: usize,
        device: &Device,
    ) -> Result<Nvfp4Linear, String> {
        let key = SlotKey {
            layer,
            proj: Proj::Down,
            expert,
        };
        let tensor_key = Self::down_key(layer);
        match self.fetch_slot(key, &tensor_key, device)? {
            CachedSlot::Nvfp4(lin) => Ok(lin),
            CachedSlot::Dense(_) => Err("down_nvfp4: slot is Dense; expected NVFP4".into()),
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
