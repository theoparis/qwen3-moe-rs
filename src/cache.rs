//! KV Cache for efficient autoregressive generation.
//!
//! During autoregressive generation, we only need to compute attention for the
//! new token while reusing the cached key-value pairs from previous tokens.
//!
//! Two layouts (Phase 2, docs/VLLM_PARITY_PLAN.md):
//!  * **legacy `cat`** (`new` / capacity `None`): `Tensor::cat` the cache each step — O(T^2) realloc+copy.
//!  * **static** (`with_capacity`): a pre-allocated `[B, T_max, kv_heads, head_dim]` buffer written in
//!    place via `slice_assign` at the write offset, returning the valid `[.., 0..filled, ..]` prefix.
//!    No per-step reallocation, and fixed shapes (the prerequisite for later CUDA-graph capture).
//!    Numerically identical to the `cat` path (pinned by `cached_matches_uncached_greedy` +
//!    `canonical_gate_long_context_parity`).

use std::sync::Arc;

use burn::{
    prelude::Backend,
    tensor::{DType, IndexingUpdateOp, Int, Tensor},
};

use crate::qwen3_5::Qwen3_5LayerType;

/// KV cache for a single attention layer.
#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    /// Cached keys: `[batch, seq_len, num_kv_heads, head_dim]` (the static buffer when `capacity` is set).
    pub key: Option<Tensor<B, 4>>,
    /// Cached values: `[batch, seq_len, num_kv_heads, head_dim]`.
    pub value: Option<Tensor<B, 4>>,
    /// `Some(T_max)` ⇒ static pre-allocated buffer of this capacity; `None` ⇒ legacy `cat`.
    capacity: Option<usize>,
    /// Number of valid positions written into the static buffer.
    filled: usize,
}

impl<B: Backend> Default for KVCache<B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> KVCache<B> {
    /// Create a new empty cache using the legacy `cat` path (O(T^2)).
    pub fn new() -> Self {
        KVCache {
            key: None,
            value: None,
            capacity: None,
            filled: 0,
        }
    }

    /// Create a static cache pre-allocated to `capacity` (= max sequence length, e.g. `prompt_len +
    /// max_new_tokens`). Writes in place via `slice_assign`; no per-step reallocation.
    pub fn with_capacity(capacity: usize) -> Self {
        KVCache {
            key: None,
            value: None,
            capacity: Some(capacity),
            filled: 0,
        }
    }

    /// Update the cache with new key-value pairs and return the full (valid) K/V for attention.
    ///
    /// * `new_key` / `new_value` — `[batch, new_seq_len, num_kv_heads, head_dim]`.
    pub fn update(
        &mut self,
        new_key: Tensor<B, 4>,
        new_value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match self.capacity {
            // ---- legacy cat path (unchanged) ----
            None => {
                let (key, value) = match (&self.key, &self.value) {
                    (Some(ck), Some(cv)) => (
                        Tensor::cat(vec![ck.clone(), new_key], 1),
                        Tensor::cat(vec![cv.clone(), new_value], 1),
                    ),
                    _ => (new_key, new_value),
                };
                self.key = Some(key.clone());
                self.value = Some(value.clone());
                (key, value)
            }
            // ---- static buffer: slice_assign at the write offset ----
            Some(cap) => {
                let [b, s, h, d] = new_key.dims();
                let off = self.filled;
                debug_assert!(
                    off + s <= cap,
                    "KV cache overflow: {off}+{s} > capacity {cap}"
                );
                if self.key.is_none() {
                    // lazy-allocate once we know [B, kv_heads, head_dim] + the dtype (bf16 on the real model).
                    let device = new_key.device();
                    let dtype = new_key.dtype();
                    self.key = Some(Tensor::<B, 4>::zeros([b, cap, h, d], &device).cast(dtype));
                    self.value = Some(Tensor::<B, 4>::zeros([b, cap, h, d], &device).cast(dtype));
                }
                // `take()` so the buffer is the sole owner during slice_assign ⇒ Burn can write in place.
                let kbuf = self
                    .key
                    .take()
                    .unwrap()
                    .slice_assign([0..b, off..off + s, 0..h, 0..d], new_key);
                let vbuf = self
                    .value
                    .take()
                    .unwrap()
                    .slice_assign([0..b, off..off + s, 0..h, 0..d], new_value);
                self.key = Some(kbuf);
                self.value = Some(vbuf);
                self.filled = off + s;
                // Valid prefix for attention (read-only; does not consume the buffer's ownership).
                let f = self.filled;
                let key = self
                    .key
                    .as_ref()
                    .unwrap()
                    .clone()
                    .slice([0..b, 0..f, 0..h, 0..d]);
                let value = self
                    .value
                    .as_ref()
                    .unwrap()
                    .clone()
                    .slice([0..b, 0..f, 0..h, 0..d]);
                (key, value)
            }
        }
    }

    /// Number of valid positions written into the static buffer.
    pub fn filled(&self) -> usize {
        self.filled
    }

    /// Whether this is a static (pre-allocated `with_capacity`) cache — the prerequisite for the
    /// device-`pos` static write path ([`Self::update_static`]) and CUDA-graph capture. Mirrors
    /// [`GdnStateCache::is_static`] so a model-level static preflight can assert every layer cache.
    pub fn is_static(&self) -> bool {
        self.capacity.is_some()
    }

    /// Rewind the valid prefix length for speculative-decode rollback.
    ///
    /// This is safe only for the static [`Self::update`] path: attention reads the `0..filled`
    /// prefix, and later `slice_assign` writes overwrite columns at the rewound offset. It is not
    /// valid for [`Self::update_static`], whose `select_assign(Add)` write requires single-write,
    /// zero-initialized columns; rewinding that path would need zeroing before reuse.
    pub fn rewind(&mut self, new_filled: usize) {
        assert!(
            self.capacity.is_some(),
            "KVCache::rewind requires a static cache using update/slice_assign"
        );
        assert!(new_filled <= self.filled);
        self.filled = new_filled;
    }

    /// **Phase-2 device-`pos`-indexed static write** (docs/cudagraph/DESIGN.md §0b P0-A / §7). Writes
    /// the new step's K/V into the static `[B, T_max, kv_heads, head_dim]` buffer at a DEVICE position
    /// index `pos` and returns the **FULL fixed-shape buffer** `[B, T_max, ..]` (constant every step) for
    /// masked attention — NOT the growing `0..filled` prefix, and NOT a host `slice_assign([.., off..]..)`.
    ///
    /// The write offset is a `[1]` Int DEVICE tensor, so a CUDA graph captured at one step replays into
    /// the correct (incremented) column at every step instead of baking the host loop index `t`. The
    /// scatter is `select_assign(dim=1, pos, new_kv, Add)`: each absolute position is written EXACTLY
    /// ONCE (the decode counter is monotone) into a column that is still zero (the buffer is zero-init in
    /// the unwritten region), so `0 + x == x` exactly ⇒ `Add` is a bit-exact assign (same trick as the
    /// MoE scatter, src/moe_grouped.rs). The unwritten columns `> pos` stay zero and are masked to `-inf`
    /// by the attention's position mask, so they never contribute — the full-`T_max` read is numerically
    /// identical to the `filled`-prefix read.
    ///
    /// * `pos` — `[1]` Int DEVICE write index (`= prompt_len + step`).
    /// * `new_key` / `new_value` — `[B, 1, kv_heads, head_dim]` (one decode token).
    pub fn update_static(
        &mut self,
        pos: &Tensor<B, 1, Int>,
        new_key: Tensor<B, 4>,
        new_value: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let cap = self
            .capacity
            .expect("update_static requires a static (with_capacity) cache");
        let [b, _s, h, d] = new_key.dims();
        // `Add`==assign holds ONLY because each column is written EXACTLY ONCE over a zero-init buffer
        // (`pos` is strictly monotone in the decode loop). update_static must be used AFTER a prefill
        // (`update` writes cols `0..lp`) and NEVER mixed with `seq_len()`/`filled` reads (it does not
        // advance `filled`). It also assumes single-write columns: speculative-decode / beam-rollback /
        // KV-eviction would make `Add` accumulate — do not use it there. For CUDA-graph REPLAY, the
        // buffers must be re-zeroed per replay (or the zero-init captured), or `Add` accumulates across
        // replays (P-final constraint, 3-voice review).
        debug_assert!(
            self.key.is_some(),
            "update_static called as the FIRST write (no prefill): cols 0..lp would be left empty + \
             `filled` stays 0. Prefill via `update` before device-pos decode."
        );
        if self.key.is_none() {
            // lazy-allocate once we know [B, kv_heads, head_dim] + the dtype (bf16 on the real model).
            let device = new_key.device();
            let dtype = new_key.dtype();
            self.key = Some(Tensor::<B, 4>::zeros([b, cap, h, d], &device).cast(dtype));
            self.value = Some(Tensor::<B, 4>::zeros([b, cap, h, d], &device).cast(dtype));
        }
        // Device-pos scatter: buffer[:, pos, :, :] += new (== assign; the column is still zero).
        // `take()` so the buffer is the sole owner ⇒ Burn can write in place. Returns the FULL buffer.
        let kbuf =
            self.key
                .take()
                .unwrap()
                .select_assign(1, pos.clone(), new_key, IndexingUpdateOp::Add);
        let vbuf = self.value.take().unwrap().select_assign(
            1,
            pos.clone(),
            new_value,
            IndexingUpdateOp::Add,
        );
        self.key = Some(kbuf.clone());
        self.value = Some(vbuf.clone());
        (kbuf, vbuf)
    }

    /// Select (gather) a SUBSET of the batch rows of this layer's K/V buffers — the per-layer
    /// primitive for dynamic batch-shrink (docs/VLLM_PARITY_PLAN.md Phase 3). `row_idx` are LOCAL
    /// row positions into the current batch dim (`0..batch`), gathered with `Tensor::select(0, ..)`
    /// so the result is `[len(row_idx), T_max, kv_heads, head_dim]`.
    ///
    /// Only dim 0 (batch) changes; the time dim (`filled` written positions) and `capacity` are
    /// preserved, so a subsequent `update` keeps writing at the same offset `filled`. A no-op until
    /// the buffer is allocated (i.e. before the first `update`). Kept rows' cached K/V are bit-for-bit
    /// unchanged, which is exactly why a shrunk decode is numerically identical for the rows it keeps.
    pub fn select_rows(&mut self, row_idx: &Tensor<B, 1, Int>) {
        if let Some(k) = self.key.take() {
            self.key = Some(k.select(0, row_idx.clone()));
        }
        if let Some(v) = self.value.take() {
            self.value = Some(v.select(0, row_idx.clone()));
        }
    }

    /// Get the current sequence length in the cache.
    pub fn seq_len(&self) -> usize {
        match self.capacity {
            Some(_) => self.filled,
            None => self.key.as_ref().map(|k| k.dims()[1]).unwrap_or(0),
        }
    }

    /// Clear the cache (keeps the capacity setting; the buffer re-allocates on the next update).
    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
        self.filled = 0;
    }

    /// **CUDA-graph reset** (P-final): zero the K/V buffers IN PLACE and reset `filled = 0`, WITHOUT
    /// reallocating — the buffer storage (its device address) is preserved. Unlike [`Self::clear`]
    /// (which drops the buffers, so the next `update` allocates a FRESH address), this keeps the exact
    /// addresses a captured graph baked, so a subsequent eager re-prefill (`update`, cols `0..lp`) +
    /// captured-step replays write into the same buffers the graph recorded. `mul_scalar(0)` is in
    /// place when the buffer is uniquely owned (it is — the cache holds the only reference after
    /// capture). No-op until the buffers exist.
    pub fn reset_for_replay(&mut self) {
        if let Some(k) = self.key.take() {
            self.key = Some(k.mul_scalar(0));
        }
        if let Some(v) = self.value.take() {
            self.value = Some(v.mul_scalar(0));
        }
        self.filled = 0;
    }
}

/// Cache for all layers in the model.
#[derive(Debug)]
pub struct ModelCache<B: Backend> {
    /// Per-layer KV caches.
    pub layers: Vec<KVCache<B>>,
}

/// Recurrent Gated-DeltaNet state for one Qwen3.6/Qwen3.5-MoE linear-attention layer.
///
/// The delta-rule matrix is kept in f32 regardless of activation dtype. The convolution buffer keeps
/// the native projection dtype and stores the last `kernel_dim - 1` unactivated qkv projections.
#[derive(Debug, Clone)]
pub struct GdnStateCache<B: Backend> {
    /// Delta-rule state: `[batch, num_value_heads, key_dim, value_dim]`, always f32.
    pub state: Option<Tensor<B, 4>>,
    /// Depthwise causal-conv history: `[batch, kernel_dim - 1, qkv_dim]`.
    pub conv: Option<Tensor<B, 3>>,
    pub num_value_heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub qkv_dim: usize,
    pub kernel_dim: usize,
    static_mode: bool,
    snap_token: Arc<()>,
}

/// Snapshot of one Gated-DeltaNet recurrent cache.
///
/// Cloning tensors is a cheap handle copy, but holding a snapshot keeps an additional reference to
/// the buffers. The next in-place-ish [`GdnStateCache::push_conv`] or [`GdnStateCache::set_state`]
/// mutation may therefore copy-on-write to preserve the snapshot's exact contents.
#[derive(Debug, Clone)]
pub struct GdnStateSnapshot<B: Backend> {
    pub state: Option<Tensor<B, 4>>,
    pub conv: Option<Tensor<B, 3>>,
    snap_token: Arc<()>,
}

impl<B: Backend> GdnStateCache<B> {
    pub fn new(
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        qkv_dim: usize,
        kernel_dim: usize,
    ) -> Self {
        debug_assert!(kernel_dim >= 1, "GDN conv kernel must be non-empty");
        Self {
            state: None,
            conv: None,
            num_value_heads,
            key_dim,
            value_dim,
            qkv_dim,
            kernel_dim,
            static_mode: false,
            snap_token: Arc::new(()),
        }
    }

    pub fn qwen3_5_default() -> Self {
        Self::new(32, 128, 128, 8192, 4)
    }

    pub fn ensure_allocated(
        &mut self,
        batch: usize,
        device: &B::Device,
        conv_dtype: burn::tensor::DType,
    ) {
        if self.state.is_none() {
            self.state = Some(Tensor::<B, 4>::zeros(
                [batch, self.num_value_heads, self.key_dim, self.value_dim],
                device,
            ));
        }
        if self.conv.is_none() {
            self.conv = Some(
                Tensor::<B, 3>::zeros([batch, self.kernel_dim - 1, self.qkv_dim], device)
                    .cast(conv_dtype),
            );
        }
    }

    /// Pre-allocate capture-stable recurrent buffers.
    ///
    /// Static mode keeps the delta-rule state storage alive and requires all future state updates to
    /// use [`Self::set_state_static`], which copies into this buffer instead of replacing it.
    pub fn init_static(&mut self, batch: usize, device: &B::Device) {
        let state_dims = [batch, self.num_value_heads, self.key_dim, self.value_dim];
        if let Some(state) = &self.state {
            assert_eq!(
                state.dims(),
                state_dims,
                "GdnStateCache::init_static called with a shape that does not match the existing \
                 static state buffer"
            );
        } else {
            self.state = Some(Tensor::<B, 4>::zeros(state_dims, device));
        }
        let conv_dims = [batch, self.kernel_dim - 1, self.qkv_dim];
        if let Some(conv) = &self.conv {
            assert_eq!(
                conv.dims(),
                conv_dims,
                "GdnStateCache::init_static called with a shape that does not match the existing \
                 conv ring buffer"
            );
        } else {
            self.conv = Some(Tensor::<B, 3>::zeros(conv_dims, device).cast(DType::F32));
        }
        self.static_mode = true;
    }

    pub fn is_static(&self) -> bool {
        self.static_mode
    }

    /// Snapshot recurrent state and convolution history for exact rollback.
    ///
    /// Tensor clones are cheap handle copies. While the snapshot is alive, later mutations may
    /// trigger copy-on-write because the cache no longer owns the only reference.
    pub fn snapshot(&self) -> GdnStateSnapshot<B> {
        assert!(
            !self.static_mode,
            "GdnStateCache::snapshot is incompatible with init_static; use non-static cache mode \
             for snapshot/rollback"
        );
        GdnStateSnapshot {
            state: self.state.clone(),
            conv: self.conv.clone(),
            snap_token: self.snap_token.clone(),
        }
    }

    /// Restore a previously captured recurrent-state snapshot.
    pub fn restore(&mut self, snap: GdnStateSnapshot<B>) {
        assert!(
            !self.static_mode,
            "GdnStateCache::restore is incompatible with init_static; use non-static cache mode \
             for snapshot/rollback"
        );
        let GdnStateSnapshot {
            state,
            conv,
            snap_token,
        } = snap;
        self.state = state;
        self.conv = conv;
        drop(snap_token);
    }

    /// Shift in the current unactivated qkv projection and return the full history buffer.
    pub fn push_conv(&mut self, current_qkv: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, qkv_dim] = current_qkv.dims();
        debug_assert_eq!(qkv_dim, self.qkv_dim);
        let device = current_qkv.device();
        let dtype = current_qkv.dtype();
        self.ensure_allocated(batch, &device, dtype);

        let history = self.kernel_dim - 1;
        let mut buf = self.conv.take().expect("GDN conv cache must be allocated");
        if history > 1 {
            let shifted = buf.clone().slice([0..batch, 1..history, 0..qkv_dim]);
            buf = buf.slice_assign([0..batch, 0..(history - 1), 0..qkv_dim], shifted);
        }
        buf = buf.slice_assign(
            [0..batch, (history - 1)..history, 0..qkv_dim],
            current_qkv.unsqueeze_dim::<3>(1),
        );
        self.conv = Some(buf.clone());
        buf
    }

    /// Static-mode convolution push that preserves the persistent buffer address.
    pub fn push_conv_static(&mut self, current_qkv: Tensor<B, 2>) -> Tensor<B, 3> {
        assert!(
            self.static_mode,
            "GdnStateCache::push_conv_static requires init_static(batch, device) before capture; \
             use push_conv for the functional eager path"
        );
        assert_eq!(
            Arc::strong_count(&self.snap_token),
            1,
            "GdnStateCache::push_conv_static cannot run with a live snapshot token: live snapshot \
             would COW-move the conv VA under capture"
        );

        let [batch, qkv_dim] = current_qkv.dims();
        debug_assert_eq!(qkv_dim, self.qkv_dim);
        let device = current_qkv.device();
        let dtype = current_qkv.dtype();
        self.ensure_allocated(batch, &device, dtype);

        let old = self
            .conv
            .as_ref()
            .expect("GDN conv cache must be allocated")
            .clone();
        let [conv_batch, history, conv_qkv_dim] = old.dims();
        assert_eq!(
            [conv_batch, conv_qkv_dim],
            [batch, qkv_dim],
            "GdnStateCache::push_conv_static conv shape must match the init_static buffer"
        );

        let current = current_qkv.unsqueeze_dim::<3>(1);
        let new_window = if history > 1 {
            Tensor::cat(
                vec![
                    old.clone().slice([0..batch, 1..history, 0..qkv_dim]),
                    current,
                ],
                1,
            )
        } else {
            current
        };
        // Dropping the read alias before copy-back keeps the persistent conv buffer uniquely owned.
        drop(old);

        let buf = self
            .conv
            .take()
            .expect("GdnStateCache::push_conv_static requires init_static to allocate conv");
        self.conv = Some(buf.slice_assign([0..batch, 0..history, 0..qkv_dim], new_window.clone()));
        new_window
    }

    pub fn set_state(&mut self, state: Tensor<B, 4>) {
        self.state = Some(state.cast(DType::F32));
    }

    /// Copy a newly computed state into the capture-stable static state buffer.
    pub fn set_state_static(&mut self, new_state: Tensor<B, 4>) {
        assert!(
            self.static_mode,
            "GdnStateCache::set_state_static requires init_static(batch, device) before capture; \
             use set_state for the functional eager path"
        );
        assert_eq!(
            Arc::strong_count(&self.snap_token),
            1,
            "GdnStateCache::set_state_static cannot run with a live snapshot token: live snapshot \
             would COW-move the state VA under capture"
        );
        let buf = self
            .state
            .take()
            .expect("GdnStateCache::set_state_static requires init_static to allocate state");
        let [batch, heads, key_dim, value_dim] = buf.dims();
        assert_eq!(
            new_state.dims(),
            [batch, heads, key_dim, value_dim],
            "GdnStateCache::set_state_static state shape must match the init_static buffer"
        );
        self.state = Some(buf.slice_assign(
            [0..batch, 0..heads, 0..key_dim, 0..value_dim],
            new_state.cast(DType::F32),
        ));
    }

    pub fn select_rows(&mut self, row_idx: &Tensor<B, 1, Int>) {
        if let Some(state) = self.state.take() {
            self.state = Some(state.select(0, row_idx.clone()));
        }
        if let Some(conv) = self.conv.take() {
            self.conv = Some(conv.select(0, row_idx.clone()));
        }
    }

    pub fn clear(&mut self) {
        self.state = None;
        self.conv = None;
        self.static_mode = false;
        self.snap_token = Arc::new(());
    }

    pub fn reset_for_replay(&mut self) {
        if let Some(state) = self.state.take() {
            self.state = Some(state.mul_scalar(0.0));
        }
        if let Some(conv) = self.conv.take() {
            self.conv = Some(conv.mul_scalar(0.0));
        }
    }
}

/// Per-layer recurrent GDN state cache.
#[derive(Debug)]
pub struct GdnModelCache<B: Backend> {
    pub layers: Vec<GdnStateCache<B>>,
}

/// Per-layer cache for the Qwen3.6/Qwen3.5-MoE hybrid tower.
#[derive(Debug)]
pub enum Qwen3_5HybridLayerCache<B: Backend> {
    Linear(GdnStateCache<B>),
    Full(KVCache<B>),
}

/// Hybrid cache matching `Qwen3_5MoeConfig::layer_types`.
#[derive(Debug)]
pub struct Qwen3_5HybridCache<B: Backend> {
    pub layers: Vec<Qwen3_5HybridLayerCache<B>>,
}

impl<B: Backend> GdnModelCache<B> {
    pub fn new_qwen3_5(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| GdnStateCache::qwen3_5_default())
                .collect(),
        }
    }

    pub fn select_rows(&mut self, row_idx: &Tensor<B, 1, Int>) {
        for layer in &mut self.layers {
            layer.select_rows(row_idx);
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.clear();
        }
    }

    pub fn reset_for_replay(&mut self) {
        for layer in &mut self.layers {
            layer.reset_for_replay();
        }
    }
}

impl<B: Backend> Qwen3_5HybridCache<B> {
    pub fn new(
        layer_types: &[Qwen3_5LayerType],
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        qkv_dim: usize,
        kernel_dim: usize,
    ) -> Self {
        Self::with_optional_capacity(
            layer_types,
            num_value_heads,
            key_dim,
            value_dim,
            qkv_dim,
            kernel_dim,
            None,
        )
    }

    pub fn with_capacity(
        layer_types: &[Qwen3_5LayerType],
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        qkv_dim: usize,
        kernel_dim: usize,
        capacity: usize,
    ) -> Self {
        Self::with_optional_capacity(
            layer_types,
            num_value_heads,
            key_dim,
            value_dim,
            qkv_dim,
            kernel_dim,
            Some(capacity),
        )
    }

    fn with_optional_capacity(
        layer_types: &[Qwen3_5LayerType],
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        qkv_dim: usize,
        kernel_dim: usize,
        capacity: Option<usize>,
    ) -> Self {
        let layers = layer_types
            .iter()
            .map(|kind| match kind {
                Qwen3_5LayerType::LinearAttention => Qwen3_5HybridLayerCache::Linear(
                    GdnStateCache::new(num_value_heads, key_dim, value_dim, qkv_dim, kernel_dim),
                ),
                Qwen3_5LayerType::FullAttention => Qwen3_5HybridLayerCache::Full(match capacity {
                    Some(capacity) => KVCache::with_capacity(capacity),
                    None => KVCache::new(),
                }),
            })
            .collect();
        Self { layers }
    }

    pub fn select_rows(&mut self, row_idx: &Tensor<B, 1, Int>) {
        for layer in &mut self.layers {
            match layer {
                Qwen3_5HybridLayerCache::Linear(cache) => cache.select_rows(row_idx),
                Qwen3_5HybridLayerCache::Full(cache) => cache.select_rows(row_idx),
            }
        }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Qwen3_5HybridLayerCache::Linear(cache) => cache.clear(),
                Qwen3_5HybridLayerCache::Full(cache) => cache.clear(),
            }
        }
    }

    pub fn reset_for_replay(&mut self) {
        for layer in &mut self.layers {
            match layer {
                Qwen3_5HybridLayerCache::Linear(cache) => cache.reset_for_replay(),
                Qwen3_5HybridLayerCache::Full(cache) => cache.reset_for_replay(),
            }
        }
    }

    pub fn snapshot_gdn(&self) -> Vec<Option<GdnStateSnapshot<B>>> {
        self.layers
            .iter()
            .map(|layer| match layer {
                Qwen3_5HybridLayerCache::Linear(cache) => Some(cache.snapshot()),
                Qwen3_5HybridLayerCache::Full(_) => None,
            })
            .collect()
    }

    pub fn restore_gdn(&mut self, snaps: Vec<Option<GdnStateSnapshot<B>>>) {
        assert_eq!(
            snaps.len(),
            self.layers.len(),
            "GDN snapshot layer count must match hybrid cache"
        );
        for (layer, snap) in self.layers.iter_mut().zip(snaps.into_iter()) {
            match (layer, snap) {
                (Qwen3_5HybridLayerCache::Linear(cache), Some(snap)) => cache.restore(snap),
                (Qwen3_5HybridLayerCache::Full(_), None) => {}
                (Qwen3_5HybridLayerCache::Linear(_), None) => {
                    panic!("missing GDN snapshot for linear-attention layer")
                }
                (Qwen3_5HybridLayerCache::Full(_), Some(_)) => {
                    panic!("unexpected GDN snapshot for full-attention layer")
                }
            }
        }
    }

    pub fn rewind_kv(&mut self, new_filled: usize) {
        for layer in &mut self.layers {
            if let Qwen3_5HybridLayerCache::Full(cache) = layer {
                cache.rewind(new_filled);
            }
        }
    }
}

impl<B: Backend> ModelCache<B> {
    /// Create a new (legacy `cat`) cache for a model with the given number of layers.
    pub fn new(num_layers: usize) -> Self {
        ModelCache {
            layers: (0..num_layers).map(|_| KVCache::new()).collect(),
        }
    }

    /// Create a STATIC pre-allocated cache (Phase 2) — each layer's buffer holds up to `capacity`
    /// positions (= `prompt_len + max_new_tokens`), written in place with no per-step reallocation.
    pub fn with_capacity(num_layers: usize, capacity: usize) -> Self {
        ModelCache {
            layers: (0..num_layers)
                .map(|_| KVCache::with_capacity(capacity))
                .collect(),
        }
    }

    /// Get the current sequence length (from the first layer's cache).
    pub fn seq_len(&self) -> usize {
        self.layers.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    /// Batch-shrink: gather the same subset of batch rows (`row_idx`, LOCAL positions into the
    /// current batch) in EVERY layer's K/V buffer, so the decode loop forwards only the still-active
    /// sequences. See [`KVCache::select_rows`].
    pub fn select_rows(&mut self, row_idx: &Tensor<B, 1, Int>) {
        for cache in &mut self.layers {
            cache.select_rows(row_idx);
        }
    }

    /// Clear all layer caches.
    pub fn clear(&mut self) {
        for cache in &mut self.layers {
            cache.clear();
        }
    }

    /// CUDA-graph reset (P-final): zero every layer's K/V IN PLACE + `filled = 0`, preserving the
    /// buffer addresses a captured graph baked. See [`KVCache::reset_for_replay`].
    pub fn reset_for_replay(&mut self) {
        for cache in &mut self.layers {
            cache.reset_for_replay();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type B = burn::backend::NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    fn t2(vals: &[f32], dims: [usize; 2], device: &<B as Backend>::Device) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(vals, device).reshape(dims)
    }

    fn t4(vals: &[f32], dims: [usize; 4], device: &<B as Backend>::Device) -> Tensor<B, 4> {
        Tensor::<B, 1>::from_floats(vals, device).reshape(dims)
    }

    fn vec3(t: Tensor<B, 3>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    fn vec4(t: Tensor<B, 4>) -> Vec<f32> {
        t.into_data().to_vec::<f32>().unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn assert_exact(a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());
        let diff = max_abs_diff(a, b);
        assert_eq!(diff, 0.0, "max_abs diff must be exactly zero");
    }

    #[test]
    fn kvcache_rewind_reuses_columns() {
        let device = dev();
        let mut cache = KVCache::<B>::with_capacity(6);

        let (k, v) = cache.update(
            t4(&[1.0, 2.0], [1, 2, 1, 1], &device),
            t4(&[10.0, 20.0], [1, 2, 1, 1], &device),
        );
        assert_eq!(cache.filled(), 2);
        assert_exact(&vec4(k), &[1.0, 2.0]);
        assert_exact(&vec4(v), &[10.0, 20.0]);

        cache.update(
            t4(&[3.0, 4.0], [1, 2, 1, 1], &device),
            t4(&[30.0, 40.0], [1, 2, 1, 1], &device),
        );
        assert_eq!(cache.filled(), 4);

        cache.rewind(2);
        assert_eq!(cache.filled(), 2);

        let (k, v) = cache.update(
            t4(&[7.0, 8.0], [1, 2, 1, 1], &device),
            t4(&[70.0, 80.0], [1, 2, 1, 1], &device),
        );
        assert_eq!(cache.filled(), 4);
        assert_exact(&vec4(k), &[1.0, 2.0, 7.0, 8.0]);
        assert_exact(&vec4(v), &[10.0, 20.0, 70.0, 80.0]);
    }

    #[test]
    fn gdn_snapshot_restore_exact() {
        let device = dev();
        let mut cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);

        let state_a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let conv_a = vec![11.0, 12.0, 13.0, 21.0, 22.0, 23.0];
        cache.set_state(t4(&state_a, [1, 2, 2, 2], &device));
        cache.push_conv(t2(&conv_a[0..3], [1, 3], &device));
        cache.push_conv(t2(&conv_a[3..6], [1, 3], &device));
        let snap = cache.snapshot();

        cache.set_state(t4(
            &[101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            [1, 2, 2, 2],
            &device,
        ));
        cache.push_conv(t2(&[31.0, 32.0, 33.0], [1, 3], &device));
        cache.push_conv(t2(&[41.0, 42.0, 43.0], [1, 3], &device));

        cache.restore(snap);

        assert_exact(&vec4(cache.state.clone().unwrap()), &state_a);
        assert_exact(&vec3(cache.conv.clone().unwrap()), &conv_a);
    }

    #[test]
    fn gdn_static_set_state_matches_functional_sequence() {
        let device = dev();
        let mut static_cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);
        let mut functional_cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);
        static_cache.init_static(2, &device);

        let states = [
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![
                -1.0, -2.0, 0.5, 0.25, 3.5, 4.5, 5.5, 6.5, -7.0, -8.0, 9.25, 10.25, 11.5, 12.5,
                -13.0, -14.0,
            ],
            vec![
                16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
                1.0,
            ],
        ];

        for vals in states {
            functional_cache.set_state(t4(&vals, [2, 2, 2, 2], &device));
            static_cache.set_state_static(t4(&vals, [2, 2, 2, 2], &device));
            assert_exact(
                &vec4(static_cache.state.clone().unwrap()),
                &vec4(functional_cache.state.clone().unwrap()),
            );
        }
    }

    #[test]
    fn gdn_static_push_conv_matches_functional_kernel_dim_4() {
        let device = dev();
        let mut static_cache = GdnStateCache::<B>::new(1, 2, 2, 3, 4);
        let mut functional_cache = GdnStateCache::<B>::new(1, 2, 2, 3, 4);
        static_cache.init_static(2, &device);

        let pushes: [[f32; 6]; 5] = [
            [1.0, 2.0, 3.0, 11.0, 12.0, 13.0],
            [21.0, 22.0, 23.0, 31.0, 32.0, 33.0],
            [41.0, 42.0, 43.0, 51.0, 52.0, 53.0],
            [61.0, 62.0, 63.0, 71.0, 72.0, 73.0],
            [81.0, 82.0, 83.0, 91.0, 92.0, 93.0],
        ];

        for vals in pushes {
            let functional_window = functional_cache.push_conv(t2(&vals, [2, 3], &device));
            let static_window = static_cache.push_conv_static(t2(&vals, [2, 3], &device));

            let expected_window = vec3(functional_window);
            assert_exact(&vec3(static_window), &expected_window);
            assert_exact(
                &vec3(static_cache.conv.clone().unwrap()),
                &vec3(functional_cache.conv.clone().unwrap()),
            );
        }
    }

    #[test]
    #[should_panic(expected = "GdnStateCache::snapshot is incompatible with init_static")]
    fn gdn_static_snapshot_panics() {
        let device = dev();
        let mut cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);
        cache.init_static(1, &device);
        let _ = cache.snapshot();
    }

    #[test]
    #[should_panic(expected = "live snapshot token")]
    fn gdn_static_set_state_with_live_snapshot_token_panics() {
        let device = dev();
        let mut cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);
        cache.init_static(1, &device);
        let _live_snapshot_token = cache.snap_token.clone();
        cache.set_state_static(t4(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 2, 2, 2],
            &device,
        ));
    }

    #[test]
    fn gdn_static_reset_for_replay_zeros_and_keeps_usable() {
        let device = dev();
        let mut cache = GdnStateCache::<B>::new(2, 2, 2, 3, 3);
        cache.init_static(1, &device);
        cache.set_state_static(t4(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [1, 2, 2, 2],
            &device,
        ));
        cache.push_conv(t2(&[11.0, 12.0, 13.0], [1, 3], &device));
        cache.push_conv(t2(&[21.0, 22.0, 23.0], [1, 3], &device));

        cache.reset_for_replay();

        assert_exact(
            &vec4(cache.state.clone().unwrap()),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_exact(
            &vec3(cache.conv.clone().unwrap()),
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );

        let after_reset = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        cache.set_state_static(t4(&after_reset, [1, 2, 2, 2], &device));
        assert_exact(&vec4(cache.state.clone().unwrap()), &after_reset);
    }

    #[test]
    fn hybrid_snapshot_rewind_roundtrip() {
        let device = dev();
        let layer_types = [
            Qwen3_5LayerType::FullAttention,
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
            Qwen3_5LayerType::LinearAttention,
        ];
        let mut cache = Qwen3_5HybridCache::<B>::with_capacity(&layer_types, 2, 2, 2, 3, 3, 6);

        for (i, layer) in cache.layers.iter_mut().enumerate() {
            match layer {
                Qwen3_5HybridLayerCache::Full(kv) => {
                    let base = 10.0 + i as f32;
                    kv.update(
                        t4(&[base, base + 1.0], [1, 2, 1, 1], &device),
                        t4(&[base + 100.0, base + 101.0], [1, 2, 1, 1], &device),
                    );
                }
                Qwen3_5HybridLayerCache::Linear(gdn) => {
                    let base = 20.0 + i as f32;
                    gdn.set_state(t4(
                        &[
                            base,
                            base + 1.0,
                            base + 2.0,
                            base + 3.0,
                            base + 4.0,
                            base + 5.0,
                            base + 6.0,
                            base + 7.0,
                        ],
                        [1, 2, 2, 2],
                        &device,
                    ));
                    gdn.push_conv(t2(
                        &[base + 10.0, base + 11.0, base + 12.0],
                        [1, 3],
                        &device,
                    ));
                    gdn.push_conv(t2(
                        &[base + 20.0, base + 21.0, base + 22.0],
                        [1, 3],
                        &device,
                    ));
                }
            }
        }

        let expected_gdn: Vec<Option<(Vec<f32>, Vec<f32>)>> = cache
            .layers
            .iter()
            .map(|layer| match layer {
                Qwen3_5HybridLayerCache::Linear(gdn) => Some((
                    vec4(gdn.state.clone().unwrap()),
                    vec3(gdn.conv.clone().unwrap()),
                )),
                Qwen3_5HybridLayerCache::Full(_) => None,
            })
            .collect();
        let snaps = cache.snapshot_gdn();
        let filled = cache
            .layers
            .iter()
            .find_map(|layer| match layer {
                Qwen3_5HybridLayerCache::Full(kv) => Some(kv.filled()),
                Qwen3_5HybridLayerCache::Linear(_) => None,
            })
            .unwrap();

        for (i, layer) in cache.layers.iter_mut().enumerate() {
            match layer {
                Qwen3_5HybridLayerCache::Full(kv) => {
                    let base = 200.0 + i as f32;
                    kv.update(
                        t4(&[base], [1, 1, 1, 1], &device),
                        t4(&[base + 100.0], [1, 1, 1, 1], &device),
                    );
                    assert_eq!(kv.filled(), filled + 1);
                }
                Qwen3_5HybridLayerCache::Linear(gdn) => {
                    let base = 300.0 + i as f32;
                    gdn.set_state(t4(
                        &[
                            base,
                            base + 1.0,
                            base + 2.0,
                            base + 3.0,
                            base + 4.0,
                            base + 5.0,
                            base + 6.0,
                            base + 7.0,
                        ],
                        [1, 2, 2, 2],
                        &device,
                    ));
                    gdn.push_conv(t2(
                        &[base + 10.0, base + 11.0, base + 12.0],
                        [1, 3],
                        &device,
                    ));
                }
            }
        }

        cache.restore_gdn(snaps);
        cache.rewind_kv(filled);

        for (layer, expected) in cache.layers.iter().zip(expected_gdn.iter()) {
            match (layer, expected) {
                (Qwen3_5HybridLayerCache::Full(kv), None) => {
                    assert_eq!(kv.filled(), filled);
                }
                (Qwen3_5HybridLayerCache::Linear(gdn), Some((state, conv))) => {
                    assert_exact(&vec4(gdn.state.clone().unwrap()), state);
                    assert_exact(&vec3(gdn.conv.clone().unwrap()), conv);
                }
                _ => panic!("hybrid layer kind changed"),
            }
        }
    }
}
