//! Reusable CUDA-graph harness for single-token static decode.
//!
//! This module intentionally lives below Burn Fusion: captured decode must use the raw
//! `CubeBackend<CudaRuntime, f32, i32, u8>` so the CUDA graph records the real launch list. The
//! model-specific closure passed to [`CapturedDecoder::build`] must be a fixed-shape, host-sync-free
//! single-token step: read `tok`/`pos`/`last`, sample on-device, write every persistent buffer in
//! place with `Option::take()` plus one tensor op, run the static cached forward, and advance `pos`
//! on device. Do not read tensors, synchronize, allocate persistent outputs, or branch on host data
//! inside that step.

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, Tensor};
#[cfg(feature = "cuda")]
use burn::tensor::{DType, IndexingUpdateOp, TensorPrimitive};
#[cfg(feature = "cuda")]
use burn_cubecl::CubeBackend;
#[cfg(feature = "cuda")]
use cubecl::Runtime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;

#[cfg(feature = "cuda")]
use crate::ModelCache;
use crate::{Qwen3_5HybridCache, Qwen3_5HybridLayerCache};

#[cfg(feature = "cuda")]
pub type CaptureBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

#[cfg(feature = "cuda")]
type Client = cubecl::client::ComputeClient<CudaRuntime>;

#[cfg(feature = "cuda")]
fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

/// Persistent device buffers used by the Qwen3.5 hybrid static decode path.
///
/// Callers must ensure the eventual prompt length plus `max_new` is within `t_max`; that assertion
/// belongs at the capture-driver point where the prompt length is known.
pub struct Qwen35DecodeState<B: Backend> {
    pub tok: Option<Tensor<B, 2, Int>>,
    pub pos: Option<Tensor<B, 1, Int>>,
    pub finished: Option<Tensor<B, 1, Bool>>,
    pub last: Option<Tensor<B, 2>>,
    pub cache: Qwen3_5HybridCache<B>,
    pub emit: Option<Tensor<B, 2, Int>>,
    pub batch: usize,
    pub vocab: usize,
    pub t_max: usize,
    pub max_new: usize,
}

impl<B: Backend> Qwen35DecodeState<B> {
    /// Allocate all persistent Qwen3.5 decode buffers outside the captured region.
    ///
    /// `t_max` is retained for capture-driver budget checks; callers with a concrete prompt length
    /// must assert `prompt_len + max_new <= t_max` before replay.
    pub fn new(
        batch: usize,
        vocab: usize,
        t_max: usize,
        max_new: usize,
        device: &B::Device,
        cache: Qwen3_5HybridCache<B>,
    ) -> Self {
        let tok = Tensor::<B, 2, Int>::zeros([batch, 1], device);
        let pos = Tensor::<B, 1, Int>::zeros([1], device);
        let finished = Tensor::<B, 1, Int>::zeros([batch], device).equal_elem(1i64);
        let last = Tensor::<B, 2>::zeros([batch, vocab], device);
        let emit = Tensor::<B, 2, Int>::zeros([batch, max_new], device);
        Self {
            tok: Some(tok),
            pos: Some(pos),
            finished: Some(finished),
            last: Some(last),
            cache,
            emit: Some(emit),
            batch,
            vocab,
            t_max,
            max_new,
        }
    }

    /// Reset every persistent replay buffer and the hybrid cache in place.
    pub fn reset_for_replay(&mut self) {
        self.tok = Some(self.tok.take().expect("tok buffer missing").mul_scalar(0));
        self.pos = Some(self.pos.take().expect("pos buffer missing").mul_scalar(0));
        self.last = Some(
            self.last
                .take()
                .expect("last buffer missing")
                .mul_scalar(0.0),
        );
        self.emit = Some(self.emit.take().expect("emit buffer missing").mul_scalar(0));

        let finished = self.finished.take().expect("finished buffer missing");
        let device = finished.device();
        let false_values = Tensor::<B, 1, Int>::zeros([self.batch], &device).equal_elem(1i64);
        self.finished = Some(finished.slice_assign([0..self.batch], false_values));

        self.cache.reset_for_replay();
    }
}

/// Persistent device buffers used by a captured decode step.
///
/// `tok`, `pos`, `last`, `finished`, and every KV tensor in `cache` are allocated before capture and
/// must keep the same device virtual addresses for the captured graph's lifetime. Step closures must
/// update these fields in place via `take()` + a single tensor operation; replacing a field with a
/// clone/copy can relocate the baked VA and make replay read stale memory.
#[cfg(feature = "cuda")]
pub struct DecodeState<B: Backend> {
    pub tok: Option<Tensor<B, 2, Int>>,
    pub pos: Option<Tensor<B, 1, Int>>,
    pub last: Option<Tensor<B, 2>>,
    pub finished: Option<Tensor<B, 2, Int>>,
    pub cache: ModelCache<B>,
    pub input_ids: Tensor<B, 2, Int>,
    pub pad: Tensor<B, 2, Int>,
    pub eos: Vec<i64>,
    pub batch: usize,
    pub prompt_len: usize,
    pub max_new: usize,
    pub total: usize,
    pub vocab: usize,
}

#[cfg(feature = "cuda")]
impl DecodeState<CaptureBackend> {
    /// Allocate every persistent buffer outside the captured region.
    pub fn new(
        input_ids: Tensor<CaptureBackend, 2, Int>,
        cache: ModelCache<CaptureBackend>,
        max_new: usize,
        vocab: usize,
        eos: Vec<i64>,
    ) -> Self {
        let device = input_ids.device();
        let [batch, prompt_len] = input_ids.dims();
        let total = prompt_len + max_new;
        let eos0 = eos.first().copied().unwrap_or(0);
        let tok = Tensor::<CaptureBackend, 2, Int>::zeros([batch, total], &device)
            .slice_assign([0..batch, 0..prompt_len], input_ids.clone());
        let pos = Tensor::<CaptureBackend, 1, Int>::full([1], prompt_len as i64, &device);
        let finished = Tensor::<CaptureBackend, 2, Int>::zeros([batch, 1], &device);
        let pad = Tensor::<CaptureBackend, 2, Int>::full([batch, 1], eos0, &device);
        let last = Tensor::<CaptureBackend, 2>::zeros([batch, vocab], &device);
        Self {
            tok: Some(tok),
            pos: Some(pos),
            last: Some(last),
            finished: Some(finished),
            cache,
            input_ids,
            pad,
            eos,
            batch,
            prompt_len,
            max_new,
            total,
            vocab,
        }
    }

    /// Current device virtual addresses of the persistent decode buffers, including KV.
    pub fn va_snapshot(&self) -> VaSnapshot {
        VaSnapshot::from_state(self)
    }
}

/// Device virtual-address snapshot for the persistent decode buffers.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VaSnapshot {
    tok: u64,
    pos: u64,
    finished: u64,
    last: u64,
    kv: Vec<(u64, u64)>,
}

/// Device virtual-address snapshot for Qwen3.5 hybrid captured decode.
#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen35VaSnapshot {
    tok: u64,
    pos: u64,
    finished: u64,
    last: u64,
    emit: u64,
    layers: Vec<Qwen35LayerVa>,
}

#[cfg(feature = "cuda")]
#[derive(Clone, Debug, Eq, PartialEq)]
enum Qwen35LayerVa {
    Linear { state: u64, conv: u64 },
    Full { key: u64, value: u64 },
}

#[cfg(feature = "cuda")]
impl Qwen35VaSnapshot {
    /// Capture every persistent VA touched by the Qwen3.5 hybrid decode graph.
    pub fn from_hybrid(state: &Qwen35DecodeState<CaptureBackend>) -> Self {
        let layers = state
            .cache
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| match layer {
                Qwen3_5HybridLayerCache::Linear(cache) => {
                    let state = cache
                        .state
                        .as_ref()
                        .unwrap_or_else(|| panic!("Qwen3.5 layer {i} GDN state buffer missing"));
                    let conv = cache
                        .conv
                        .as_ref()
                        .unwrap_or_else(|| panic!("Qwen3.5 layer {i} GDN conv buffer missing"));
                    Qwen35LayerVa::Linear {
                        state: float_va(state),
                        conv: float_va(conv),
                    }
                }
                Qwen3_5HybridLayerCache::Full(cache) => {
                    let key = cache.key.as_ref().unwrap_or_else(|| {
                        panic!("Qwen3.5 layer {i} full-attn key buffer missing")
                    });
                    let value = cache.value.as_ref().unwrap_or_else(|| {
                        panic!("Qwen3.5 layer {i} full-attn value buffer missing")
                    });
                    Qwen35LayerVa::Full {
                        key: float_va(key),
                        value: float_va(value),
                    }
                }
            })
            .collect();
        Self {
            tok: int_va(state.tok.as_ref().expect("tok buffer missing")),
            pos: int_va(state.pos.as_ref().expect("pos buffer missing")),
            finished: bool_va(state.finished.as_ref().expect("finished buffer missing")),
            last: float_va(state.last.as_ref().expect("last buffer missing")),
            emit: int_va(state.emit.as_ref().expect("emit buffer missing")),
            layers,
        }
    }

    /// Panic if any graph-baked persistent VA has moved.
    pub fn assert_unchanged(&self, state: &Qwen35DecodeState<CaptureBackend>, context: &str) {
        let after = Self::from_hybrid(state);
        assert_same_va("tok", self.tok, after.tok, context);
        assert_same_va("pos", self.pos, after.pos, context);
        assert_same_va("finished", self.finished, after.finished, context);
        assert_same_va("last", self.last, after.last, context);
        assert_same_va("emit", self.emit, after.emit, context);
        assert_eq!(
            self.layers.len(),
            after.layers.len(),
            "VA-STABILITY VIOLATION ({context}): Qwen3.5 hybrid layer count changed ({} -> {})",
            self.layers.len(),
            after.layers.len()
        );
        for (i, (before, after)) in self.layers.iter().zip(after.layers.iter()).enumerate() {
            match (before, after) {
                (
                    Qwen35LayerVa::Linear {
                        state: bs,
                        conv: bc,
                    },
                    Qwen35LayerVa::Linear {
                        state: as_,
                        conv: ac,
                    },
                ) => {
                    assert_layer_va(i, "GDN state", *bs, *as_, context);
                    assert_layer_va(i, "GDN conv", *bc, *ac, context);
                }
                (
                    Qwen35LayerVa::Full { key: bk, value: bv },
                    Qwen35LayerVa::Full { key: ak, value: av },
                ) => {
                    assert_layer_va(i, "full-attn key", *bk, *ak, context);
                    assert_layer_va(i, "full-attn value", *bv, *av, context);
                }
                _ => panic!(
                    "VA-STABILITY VIOLATION ({context}): Qwen3.5 hybrid layer {i} kind changed"
                ),
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn assert_same_va(kind: &str, before: u64, after: u64, context: &str) {
    assert_eq!(
        before, after,
        "VA-STABILITY VIOLATION ({context}): persistent buffer '{kind}' relocated \
         ({before:#x} -> {after:#x}); a non-in-place update broke the graph-baked address"
    );
}

#[cfg(feature = "cuda")]
fn assert_layer_va(layer: usize, kind: &str, before: u64, after: u64, context: &str) {
    assert_eq!(
        before, after,
        "VA-STABILITY VIOLATION ({context}): Qwen3.5 layer {layer} {kind} relocated \
         ({before:#x} -> {after:#x}); reset_for_replay/prefill must preserve graph-baked cache \
         addresses"
    );
}

#[cfg(feature = "cuda")]
impl VaSnapshot {
    fn from_state(state: &DecodeState<CaptureBackend>) -> Self {
        let kv = state
            .cache
            .layers
            .iter()
            .enumerate()
            .map(|(i, layer)| {
                let key = layer
                    .key
                    .as_ref()
                    .unwrap_or_else(|| panic!("KV layer {i} key buffer missing"));
                let value = layer
                    .value
                    .as_ref()
                    .unwrap_or_else(|| panic!("KV layer {i} value buffer missing"));
                (float_va(key), float_va(value))
            })
            .collect();
        Self {
            tok: int_va(state.tok.as_ref().expect("tok buffer missing")),
            pos: int_va(state.pos.as_ref().expect("pos buffer missing")),
            finished: int_va(state.finished.as_ref().expect("finished buffer missing")),
            last: float_va(state.last.as_ref().expect("last buffer missing")),
            kv,
        }
    }

    fn assert_unchanged(&self, after: Self) {
        let labels = ["tok", "pos", "finished", "last"];
        let before = [self.tok, self.pos, self.finished, self.last];
        let after_buffers = [after.tok, after.pos, after.finished, after.last];
        for (i, (b, a)) in before.iter().zip(after_buffers.iter()).enumerate() {
            assert_eq!(
                b, a,
                "VA-STABILITY VIOLATION: persistent buffer '{}' relocated ({b:#x} -> {a:#x}); \
                 a non-in-place update broke the graph-baked address",
                labels[i]
            );
        }
        assert_eq!(
            self.kv.len(),
            after.kv.len(),
            "VA-STABILITY VIOLATION: KV layer count changed ({} -> {})",
            self.kv.len(),
            after.kv.len()
        );
        for (i, ((bk, bv), (ak, av))) in self.kv.iter().zip(after.kv.iter()).enumerate() {
            assert_eq!(
                bk, ak,
                "VA-STABILITY VIOLATION: KV layer {i} key relocated ({bk:#x} -> {ak:#x}); \
                 reset_for_replay/prefill must preserve graph-baked cache addresses"
            );
            assert_eq!(
                bv, av,
                "VA-STABILITY VIOLATION: KV layer {i} value relocated ({bv:#x} -> {av:#x}); \
                 reset_for_replay/prefill must preserve graph-baked cache addresses"
            );
        }
    }
}

/// Capture-once / replay-per-token driver for greedy static decode.
///
/// `Prefill` is the model-specific eager prefill callback. It must refill `state.cache` and restore
/// `state.last` in place after `cache.reset_for_replay()`. The captured `step` callback is used only
/// during build; it must not perform host reads or syncs.
#[cfg(feature = "cuda")]
pub struct CapturedDecoder<Prefill>
where
    Prefill: FnMut(&mut DecodeState<CaptureBackend>),
{
    graph: cubecl::client::CapturedGraph<CudaRuntime>,
    pub state: DecodeState<CaptureBackend>,
    prefill: Prefill,
    va: VaSnapshot,
    client: Client,
}

#[cfg(feature = "cuda")]
impl<Prefill> CapturedDecoder<Prefill>
where
    Prefill: FnMut(&mut DecodeState<CaptureBackend>),
{
    /// Build state, run eager prefill, warm up at least three decode passes, and capture one step.
    ///
    /// `warmup` must be at least 3 and strictly less than `max_new`; the warmup/capture pass writes
    /// column `prompt_len + warmup`, so `warmup >= max_new` would write past the token buffer.
    pub fn build<Step>(
        input_ids: Tensor<CaptureBackend, 2, Int>,
        cache: ModelCache<CaptureBackend>,
        max_new: usize,
        vocab: usize,
        eos: Vec<i64>,
        warmup: usize,
        mut prefill: Prefill,
        mut step: Step,
    ) -> Self
    where
        Step: FnMut(&mut DecodeState<CaptureBackend>),
    {
        assert!(
            warmup >= 3,
            "captured decode requires warmup >= 3 eager passes"
        );
        assert!(
            warmup < max_new,
            "capture warmup pass writes column lp+warmup; warmup must be < max_new ({max_new})"
        );

        let device = input_ids.device();
        let client = CudaRuntime::client(&device);
        let mut state = DecodeState::new(input_ids, cache, max_new, vocab, eos);

        prefill(&mut state);
        block_sync(&client);

        let va = state.va_snapshot();
        let graph = unsafe { client.capture_arena(warmup, || step(&mut state)) };
        block_sync(&client);

        va.assert_unchanged(state.va_snapshot());

        let mut decoder = Self {
            graph,
            state,
            prefill,
            va,
            client,
        };
        decoder.reset_and_prefill_current();
        decoder
    }

    /// Reset the KV cache and persistent buffers in place, then prefill from `prompt_ids`.
    pub fn reset_and_prefill(&mut self, prompt_ids: &[i64]) {
        assert_eq!(
            prompt_ids.len(),
            self.state.batch * self.state.prompt_len,
            "prompt_ids length must match the captured [batch, prompt_len] shape"
        );
        let device = self.state.input_ids.device();
        self.state.input_ids = Tensor::<CaptureBackend, 1, Int>::from_data(prompt_ids, &device)
            .reshape([self.state.batch, self.state.prompt_len]);
        self.reset_and_prefill_current();
    }

    /// Reset in place and reuse the currently stored prompt tensor.
    pub fn reset_and_prefill_current(&mut self) {
        self.state.cache.reset_for_replay();
        (self.prefill)(&mut self.state);
        let b = self.state.batch;
        let lp = self.state.prompt_len;
        self.state.tok = Some(
            self.state
                .tok
                .take()
                .expect("tok buffer missing")
                .mul_scalar(0),
        );
        self.state.tok = Some(
            self.state
                .tok
                .take()
                .expect("tok buffer missing")
                .slice_assign([0..b, 0..lp], self.state.input_ids.clone()),
        );
        self.state.finished = Some(
            self.state
                .finished
                .take()
                .expect("finished buffer missing")
                .mul_scalar(0),
        );
        self.state.pos = Some(
            self.state
                .pos
                .take()
                .expect("pos buffer missing")
                .mul_scalar(0)
                .add_scalar(lp as i64),
        );
        block_sync(&self.client);
        self.verify_va();
    }

    /// Replay `n` captured single-token steps and read the full `[batch, prompt_len + max_new]`
    /// token buffer once at the end.
    pub fn decode_n(&mut self, n: usize, eos: &[i64]) -> Vec<i64> {
        assert_eq!(
            eos,
            self.state.eos.as_slice(),
            "EOS set is baked into the captured step"
        );
        assert!(
            n <= self.state.max_new,
            "decode length {n} exceeds max_new {}",
            self.state.max_new
        );
        for _ in 0..n {
            self.graph.replay();
        }
        block_sync(&self.client);
        self.state
            .tok
            .as_ref()
            .expect("tok buffer missing")
            .clone()
            .into_data()
            .to_vec::<i32>()
            .expect("token read failed")
            .into_iter()
            .map(|x| x as i64)
            .collect()
    }

    /// Reset and time pure replay, returning the median milliseconds per token.
    pub fn replay_ms_per_token(&mut self, n: usize, reps: usize) -> f64 {
        assert!(
            n <= self.state.max_new,
            "decode length {n} exceeds max_new {}",
            self.state.max_new
        );
        let mut xs = Vec::with_capacity(reps);
        for _ in 0..reps {
            self.reset_and_prefill_current();
            let t0 = std::time::Instant::now();
            for _ in 0..n {
                self.graph.replay();
            }
            block_sync(&self.client);
            xs.push(t0.elapsed().as_secs_f64() * 1e3 / n as f64);
        }
        median(&xs)
    }

    pub fn arena_bytes(&self) -> u64 {
        self.graph.arena_bytes()
    }

    /// Mutation-test hook: panics if any persistent decode buffer moved from its captured VA.
    pub fn verify_va(&self) {
        self.va.assert_unchanged(self.state.va_snapshot());
    }
}

#[cfg(feature = "cuda")]
pub fn float_va<const D: usize>(t: &Tensor<CaptureBackend, D>) -> u64 {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(ct) => {
            ct.client
                .get_resource(ct.handle.clone().binding())
                .resource()
                .ptr
        }
        TensorPrimitive::QFloat(_) => unreachable!("persistent buffers are never quantized"),
    }
}

#[cfg(feature = "cuda")]
pub fn int_va<const D: usize>(t: &Tensor<CaptureBackend, D, Int>) -> u64 {
    let ct = t.clone().into_primitive();
    ct.client
        .get_resource(ct.handle.clone().binding())
        .resource()
        .ptr
}

#[cfg(feature = "cuda")]
pub fn bool_va<const D: usize>(t: &Tensor<CaptureBackend, D, Bool>) -> u64 {
    let ct = t.clone().into_primitive();
    ct.client
        .get_resource(ct.handle.clone().binding())
        .resource()
        .ptr
}

/// Minimal allocator telemetry used by captured replay gates.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MemoryUsageSnapshot {
    pub number_allocs: u64,
    pub bytes_in_use: u64,
}

#[cfg(feature = "cuda")]
pub fn memory_usage_snapshot<R: Runtime>(
    client: &cubecl::client::ComputeClient<R>,
) -> MemoryUsageSnapshot {
    let usage = client.memory_usage();
    MemoryUsageSnapshot {
        number_allocs: usage.number_allocs,
        bytes_in_use: usage.bytes_in_use,
    }
}

#[cfg(feature = "cuda")]
pub fn assert_no_new_allocs(
    before: MemoryUsageSnapshot,
    after: MemoryUsageSnapshot,
    context: &str,
) {
    assert_eq!(
        before.number_allocs, after.number_allocs,
        "ALLOCATION-STABILITY VIOLATION ({context}): number_allocs changed ({} -> {})",
        before.number_allocs, after.number_allocs
    );
    assert_eq!(
        before.bytes_in_use, after.bytes_in_use,
        "ALLOCATION-STABILITY VIOLATION ({context}): bytes_in_use changed ({} -> {})",
        before.bytes_in_use, after.bytes_in_use
    );
}

#[cfg(feature = "cuda")]
pub fn write_last_in_place(
    last: Tensor<CaptureBackend, 2>,
    logits_3d: Tensor<CaptureBackend, 3>,
    batch: usize,
    vocab: usize,
) -> Tensor<CaptureBackend, 2> {
    let new_last = logits_3d
        .slice([0..batch, 0..1, 0..vocab])
        .reshape([batch, vocab])
        .cast(DType::F32);
    last.slice_assign([0..batch, 0..vocab], new_last)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Qwen3_5LayerType;
    use burn::{backend::NdArray, tensor::TensorData};

    type B = NdArray;

    fn device() -> <B as Backend>::Device {
        Default::default()
    }

    fn vec_i(t: Tensor<B, 2, Int>) -> Vec<i64> {
        t.into_data().to_vec::<i64>().unwrap()
    }

    fn vec_f<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    fn vec_b(t: Tensor<B, 1, Bool>) -> Vec<bool> {
        t.into_data().to_vec::<bool>().unwrap()
    }

    #[test]
    fn qwen35_decode_state_reset_preserves_shapes_and_zeroes_contents() {
        let device = device();
        let layer_types = [
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ];
        let mut cache = Qwen3_5HybridCache::<B>::with_capacity(&layer_types, 2, 2, 2, 3, 3, 6);

        if let Qwen3_5HybridLayerCache::Linear(gdn) = &mut cache.layers[0] {
            gdn.init_static(2, &device);
            gdn.set_state_static(
                Tensor::<B, 1>::from_floats(
                    [
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                        15.0, 16.0,
                    ],
                    &device,
                )
                .reshape([2, 2, 2, 2]),
            );
            let _ = gdn.push_conv(
                Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &device)
                    .reshape([2, 3]),
            );
            let _ = gdn.push_conv(
                Tensor::<B, 1>::from_floats([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &device)
                    .reshape([2, 3]),
            );
        } else {
            panic!("test layer 0 must be GDN");
        }
        if let Qwen3_5HybridLayerCache::Full(kv) = &mut cache.layers[1] {
            let key =
                Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0, 4.0], &device).reshape([2, 2, 1, 1]);
            let value =
                Tensor::<B, 1>::from_floats([5.0, 6.0, 7.0, 8.0], &device).reshape([2, 2, 1, 1]);
            let _ = kv.update(key, value);
            assert_eq!(kv.filled(), 2);
        } else {
            panic!("test layer 1 must be full attention");
        }

        let mut state = Qwen35DecodeState::new(2, 5, 6, 3, &device, cache);
        state.tok = Some(Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![9i64, 8], [2, 1]),
            &device,
        ));
        state.pos = Some(Tensor::<B, 1, Int>::from_data([4], &device));
        state.finished = Some(Tensor::<B, 1, Bool>::from_data([true, true], &device));
        state.last = Some(
            Tensor::<B, 1>::from_floats(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                &device,
            )
            .reshape([2, 5]),
        );
        state.emit = Some(Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![1i64, 2, 3, 4, 5, 6], [2, 3]),
            &device,
        ));

        state.reset_for_replay();

        assert_eq!(state.tok.as_ref().unwrap().dims(), [2, 1]);
        assert_eq!(state.pos.as_ref().unwrap().dims(), [1]);
        assert_eq!(state.finished.as_ref().unwrap().dims(), [2]);
        assert_eq!(state.last.as_ref().unwrap().dims(), [2, 5]);
        assert_eq!(state.emit.as_ref().unwrap().dims(), [2, 3]);
        assert_eq!(vec_i(state.tok.as_ref().unwrap().clone()), vec![0, 0]);
        assert_eq!(
            state
                .pos
                .as_ref()
                .unwrap()
                .clone()
                .into_data()
                .to_vec::<i64>()
                .unwrap(),
            vec![0]
        );
        assert_eq!(
            vec_b(state.finished.as_ref().unwrap().clone()),
            vec![false, false]
        );
        assert!(
            vec_f(state.last.as_ref().unwrap().clone())
                .iter()
                .all(|x| *x == 0.0)
        );
        assert_eq!(
            vec_i(state.emit.as_ref().unwrap().clone()),
            vec![0, 0, 0, 0, 0, 0]
        );

        match &state.cache.layers[0] {
            Qwen3_5HybridLayerCache::Linear(gdn) => {
                assert!(
                    vec_f(gdn.state.as_ref().unwrap().clone())
                        .iter()
                        .all(|x| *x == 0.0)
                );
                assert!(
                    vec_f(gdn.conv.as_ref().unwrap().clone())
                        .iter()
                        .all(|x| *x == 0.0)
                );
            }
            Qwen3_5HybridLayerCache::Full(_) => panic!("test layer 0 must be GDN"),
        }
        match &state.cache.layers[1] {
            Qwen3_5HybridLayerCache::Full(kv) => {
                assert_eq!(kv.filled(), 0);
                assert!(
                    vec_f(kv.key.as_ref().unwrap().clone())
                        .iter()
                        .all(|x| *x == 0.0)
                );
                assert!(
                    vec_f(kv.value.as_ref().unwrap().clone())
                        .iter()
                        .all(|x| *x == 0.0)
                );
            }
            Qwen3_5HybridLayerCache::Linear(_) => panic!("test layer 1 must be full attention"),
        }
    }
}

#[cfg(feature = "cuda")]
pub fn scatter_emit_to_tok(
    tok: Tensor<CaptureBackend, 2, Int>,
    pos: Tensor<CaptureBackend, 1, Int>,
    emit: Tensor<CaptureBackend, 2, Int>,
) -> Tensor<CaptureBackend, 2, Int> {
    tok.select_assign(1, pos, emit, IndexingUpdateOp::Add)
}

#[cfg(feature = "cuda")]
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}
