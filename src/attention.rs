//! Grouped Query Attention for Qwen3.

use burn::{
    Tensor,
    config::Config,
    module::Module,
    nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Device,
    // We use `attention_fallback` (the reference scaled-dot-product implementation) rather
    // than the fused `attention` kernel: on the CubeCL CUDA backend (sm_121 / NVIDIA GB10)
    // the fused kernel mishandles the broadcast causal mask of shape [1, 1, seq, seq]
    // (not expanded over batch/num_heads), producing wrong logits (argmax 15738 vs the
    // correct 279). The fallback honours the broadcast mask correctly and matches the HF
    // reference (cosine 0.99996). It is also correct on the CPU backends used elsewhere.
    tensor::{Bool, Int, module::attention_fallback as attention, ops::AttentionModuleOptions},
};

use super::cache::KVCache;
use super::linear2d::{Precision, linear3};
use super::rope::{apply_rope, compute_rope_embeddings, compute_rope_embeddings_pre};

/// Configuration for Qwen3 attention.
#[derive(Config, Debug)]
pub struct Qwen3AttentionConfig {
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_key_value_heads: usize,
    /// Explicit head dimension. If None, computed as hidden_size / num_attention_heads.
    pub head_dim: Option<usize>,
    /// RoPE theta parameter.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// RMSNorm epsilon.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
}

impl Qwen3AttentionConfig {
    /// Initialize the attention module.
    pub fn init(&self, device: &Device) -> Qwen3Attention {
        let head_dim = self
            .head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads);

        Qwen3Attention {
            num_heads: (self.num_attention_heads),
            num_kv_heads: (self.num_key_value_heads),
            head_dim: (head_dim),
            rope_theta: (self.rope_theta),
            q_proj: LinearConfig::new(self.hidden_size, self.num_attention_heads * head_dim)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(self.hidden_size, self.num_key_value_heads * head_dim)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(self.num_attention_heads * head_dim, self.hidden_size)
                .with_bias(false)
                .init(device),
            q_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            k_norm: RmsNormConfig::new(head_dim)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// Grouped Query Attention module for Qwen3.
#[derive(Module, Debug)]
pub struct Qwen3Attention {
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f64,

    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
}

impl Qwen3Attention {
    /// Per-head dimension `head_dim` (also the QK-RMSNorm dimension — `q_norm`/`k_norm` normalize each
    /// head of `head_dim`). Exposed so a reused decode path can assert its precomputed RoPE table and
    /// QK-norm assumptions match this attention (see `Qwen3MoeForCausalLM::build_static_decode`).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// RoPE theta. Exposed for the same precomputed-RoPE config-equality check.
    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// The QK-RMSNorm normalized dimension (`q_norm` gamma length). Equals `head_dim` for Qwen3 (the
    /// gamma is `head_dim` small). Used to assert QK-norm shape parity on a reused decode path.
    pub fn qk_norm_dim(&self) -> usize {
        self.q_norm.gamma.val().dims()[0]
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden_size]
    /// * `attention_mask` - Optional attention mask [batch, seq]
    /// * `position_ids` - Position indices [batch, seq]
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Tensor<2, Int>,
        prec: Precision,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Project to Q, K, V. Use the batch-safe 2-D Linear (see linear2d.rs): the CubeCL
        // batched matmul corrupts rows past the first at batch>1 for some shapes on sm_121.
        let query = linear3(&self.q_proj, hidden_states.clone(), prec);
        let key = linear3(&self.k_proj, hidden_states.clone(), prec);
        let value = linear3(&self.v_proj, hidden_states, prec);

        // Reshape to [batch, seq, n_heads, head_dim]
        let query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let key = key.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        // Apply QK normalization
        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // Compute and apply RoPE
        let (cos, sin) =
            compute_rope_embeddings(position_ids, self.head_dim, self.rope_theta, &device);
        let (query, key) = apply_rope(query, key, cos, sin);

        // Expand KV heads for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // Attention: transpose to [batch, n_heads, seq, head_dim]
        // Create causal mask: lower triangular matrix where true = attend
        // Use float operations to avoid I64 issues on Metal
        let row_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let col_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();

        let rows = Tensor::<1>::from_floats(row_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, seq_len]); // [seq, seq]
        let cols = Tensor::<1>::from_floats(col_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[seq_len, 1]); // [seq, seq]

        // Causal mask: rows < cols (upper triangular excluding diagonal) = positions to MASK OUT
        // Burn attention uses true = mask out (fill with -inf)
        let causal_mask = rows.clone().lower(cols.clone()); // [seq, seq] Bool, true = future positions to mask

        // Combine with optional attention mask (padding mask)
        let combined_mask = match attention_mask {
            Some(pad_mask) => {
                // pad_mask: [batch, seq] where true = valid token, false = padding
                // We need to mask where pad_mask is false, so invert: true = masked position
                let pad_mask_inverted = pad_mask.bool_not(); // true = padding (mask out)
                // Expand to [batch, 1, seq_q, seq_k] for attention scores
                // For self-attention, seq_q = seq_k = seq
                let pad_expanded = pad_mask_inverted
                    .unsqueeze_dims::<4>(&[1, 2]) // [batch, 1, 1, seq_k]
                    .repeat(&[1, 1, seq_len, 1]); // [batch, 1, seq_q, seq_k]
                let causal_expanded = causal_mask.unsqueeze_dims::<4>(&[0, 1]); // [1, 1, seq_q, seq_k]
                // Combined: mask where either is true (padding OR future)
                pad_expanded.bool_or(causal_expanded)
            }
            None => causal_mask.unsqueeze_dims::<4>(&[0, 1]), // [1, 1, seq, seq]
        };

        // No query row may be fully masked. A left-padded query whose only causally-visible keys
        // are masked pad tokens would attend to NOTHING -> softmax(all -inf) = NaN, which then
        // poisons real positions through later layers (attention mixes positions). Let every
        // position attend to ITSELF (the diagonal): real queries are unaffected (they never attend
        // to pad keys off-diagonal), and dead pad-query rows become finite garbage that the
        // completion mask discards downstream. No-op for the unmasked path (causal already keeps
        // the diagonal open).
        let diag = rows.equal(cols).unsqueeze_dims::<4>(&[0, 1]); // [1,1,seq,seq], true on diagonal
        let combined_mask = combined_mask.bool_and(diag.bool_not());

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            Some(combined_mask),
            None,
            AttentionModuleOptions::default(),
        );

        // Reshape back to [batch, seq, hidden_size]
        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (self.num_heads * self.head_dim) as i64,
        ]);

        linear3(&self.o_proj, attn_output, prec)
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    ///
    /// # Arguments
    /// * `hidden_states` - Input tensor [batch, seq, hidden_size] (usually seq=1 for generation)
    /// * `attention_mask` - Optional attention mask [batch, total_seq] (including cached positions)
    /// * `position_ids` - Position indices [batch, seq] (positions for new tokens only)
    /// * `cache` - Mutable reference to KV cache for this layer
    ///
    /// # Returns
    /// Output tensor [batch, seq, hidden_size]
    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<3>,
        attention_mask: Option<Tensor<2, Bool>>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Project to Q, K, V for new tokens only (batch-safe 2-D Linear; see linear2d.rs)
        let query = linear3(&self.q_proj, hidden_states.clone(), prec);
        let key = linear3(&self.k_proj, hidden_states.clone(), prec);
        let value = linear3(&self.v_proj, hidden_states, prec);

        // Reshape to [batch, seq, n_heads, head_dim]
        let query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let key = key.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        // Apply QK normalization
        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // Compute and apply RoPE (only for new positions)
        let (cos, sin) =
            compute_rope_embeddings(position_ids, self.head_dim, self.rope_theta, &device);
        let (query, key) = apply_rope(query, key, cos, sin);

        // Update cache and get full K, V (including past)
        let (key, value) = cache.update(key, value);

        // Expand KV heads for GQA
        let n_rep = self.num_heads / self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // Attention: transpose to [batch, n_heads, seq, head_dim]
        // Query is [batch, new_seq, n_heads, head_dim]
        // Key/Value are [batch, total_seq, n_heads, head_dim]
        //
        // For the prefill phase (seq_len > 1), we need a causal mask to prevent
        // tokens from attending to future positions.
        // For the decode phase (seq_len == 1), no causal mask is needed.
        let [_, total_seq, _, _] = key.dims();

        let combined_mask = if seq_len > 1 {
            // Prefill: create causal mask [seq_len, total_seq]
            // Causal mask: rows < cols = positions to MASK OUT (future tokens).
            // Query positions are OFFSET by the cached prefix: with a non-empty KV cache the
            // queries occupy the LAST `seq_len` of `total_seq`, so row indices start at
            // `total_seq - seq_len` (else new tokens would mask themselves out).
            let q_offset = total_seq - seq_len;
            let row_idx: Vec<f32> = (0..seq_len).map(|i| (q_offset + i) as f32).collect();
            let col_idx: Vec<f32> = (0..total_seq).map(|i| i as f32).collect();

            let rows = Tensor::<1>::from_floats(row_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(1)
                .repeat(&[1, total_seq]); // [seq_len, total_seq]
            let cols = Tensor::<1>::from_floats(col_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[seq_len, 1]); // [seq_len, total_seq]

            // Causal mask: rows < cols = future positions to mask
            let causal_mask = rows.clone().lower(cols.clone()); // [seq_len, total_seq] Bool

            // Combine with optional attention mask (padding mask)
            let prefill_mask = match attention_mask {
                Some(pad_mask) => {
                    let pad_mask_inverted = pad_mask.bool_not();
                    let pad_expanded = pad_mask_inverted
                        .unsqueeze_dims::<4>(&[1, 2])
                        .repeat(&[1, 1, seq_len, 1]);
                    let causal_expanded = causal_mask.unsqueeze_dims::<4>(&[0, 1]);
                    pad_expanded.bool_or(causal_expanded)
                }
                None => causal_mask.unsqueeze_dims::<4>(&[0, 1]),
            };
            // Same diagonal-unmask as the no-cache `forward` path: a left-padded prompt's pad-query
            // row whose only causally-visible keys are masked pad would softmax(all -inf) -> NaN and
            // poison the cached states / later layers. `rows`/`cols` carry GLOBAL indices (prefill
            // queries are offset by `total_seq - seq_len`), so `rows.equal(cols)` is the true
            // self-attention diagonal. No-op for the unmasked path (causal already opens it).
            let diag = rows.equal(cols).unsqueeze_dims::<4>(&[0, 1]);
            Some(prefill_mask.bool_and(diag.bool_not()))
        } else {
            // Decode phase: no causal mask needed (query is single token). Invert the padding
            // mask (true=valid -> true=mask-out) to match the prefill path's `bool_not` polarity,
            // so a padding mask during cached decode doesn't mask out the VALID tokens.
            attention_mask.map(|m| m.bool_not().unsqueeze_dims(&[1, 2]))
        };

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            combined_mask,
            None,
            AttentionModuleOptions::default(),
        );

        // Reshape back to [batch, seq, hidden_size]
        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (self.num_heads * self.head_dim) as i64,
        ]);

        linear3(&self.o_proj, attn_output, prec)
    }

    /// **Phase-2 fixed-shape, device-`pos`-indexed decode attention** (docs/cudagraph/DESIGN.md §0b
    /// P0-A / §7). The capture-ready sibling of [`Qwen3Attention::forward_with_cache`]'s decode branch:
    /// it forwards exactly ONE token (`seq == 1`) over the **FULL static `[B, T_max, ..]` K/V buffer**
    /// (constant shape every step) with a **position mask** that `-inf`s key columns `idx > pos`, instead
    /// of reading the growing `0..filled` prefix. Every per-step op is fixed-shape and indexed by the
    /// DEVICE counter `pos` (a `[1]` Int tensor), so nothing bakes the host loop index `t`.
    ///
    /// Numerically identical to the growing-prefix decode: the masked columns become `exp(-inf) == 0`,
    /// so the softmax sees exactly the filled `0..=pos` keys (verified bit-exact on NdArray CPU by
    /// `static_matches_device_loop_greedy`). The RoPE position of the new token IS `pos` (device), and
    /// the KV write goes to the device index `pos` via [`KVCache::update_static`].
    ///
    /// * `hidden_states` — `[B, 1, hidden]` (one new token).
    /// * `pos` — `[1]` Int DEVICE counter: the KV write column, the RoPE position, and the mask boundary.
    pub fn forward_with_cache_static(
        &self,
        hidden_states: Tensor<3>,
        pos: Tensor<1, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _] = hidden_states.dims(); // seq_len == 1 (decode)
        let device = hidden_states.device();

        // Project to Q, K, V for the new token (batch-safe 2-D Linear; see linear2d.rs).
        let query = linear3(&self.q_proj, hidden_states.clone(), prec);
        let key = linear3(&self.k_proj, hidden_states.clone(), prec);
        let value = linear3(&self.v_proj, hidden_states, prec);

        let query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let key = key.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // RoPE position of the new token = `pos` (DEVICE), broadcast to [B, 1]. No host index.
        let position_ids = pos.clone().reshape([1, 1]).repeat(&[batch_size, 1]); // [B, 1] Int
        let (cos, sin) =
            compute_rope_embeddings(position_ids, self.head_dim, self.rope_theta, &device);
        let (query, key) = apply_rope(query, key, cos, sin);

        // Device-pos KV write into the static buffer; read back the FULL [B, T_max, kv_heads, head_dim].
        let (key, value) = cache.update_static(&pos, key, value);
        let t_max = key.dims()[1];

        // Expand KV heads for GQA (over the full T_max buffer — constant shape).
        let n_rep = self.num_heads / self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // POSITION MASK (the load-bearing correctness piece). Mask out (= true ⇒ -inf before softmax)
        // every key column `idx > pos`, built from a DEVICE `arange(T_max)` vs the device `pos` counter
        // (no host offset). Shape [1, 1, 1, T_max] broadcasts over batch + heads + the single query row
        // against the scores [B, n_heads, 1, T_max]. The new token's own column (idx == pos) stays
        // visible; the unwritten / future columns become -inf ⇒ exp == 0 ⇒ identical to reading only the
        // `0..=pos` prefix.
        let idx = Tensor::<1, Int>::arange(0..t_max as i64, &device).reshape([1, 1, 1, t_max]);
        let pos_mask = idx.greater(pos.reshape([1, 1, 1, 1])); // [1,1,1,T_max] Bool, true above pos

        let attn_output = attention(
            query.movedim(1, 2), // [B, n_heads, 1, head_dim]
            key.movedim(1, 2),   // [B, n_heads, T_max, head_dim]
            value.movedim(1, 2),
            Some(pos_mask),
            None,
            AttentionModuleOptions::default(),
        );

        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (self.num_heads * self.head_dim) as i64,
        ]);

        linear3(&self.o_proj, attn_output, prec)
    }

    /// CUDA-graph-CAPTURABLE sibling of [`Qwen3Attention::forward_with_cache_static`] (P-final). Same
    /// fixed-shape, device-`pos`-indexed decode attention, but every per-step HOST->DEVICE STAGING is
    /// hoisted out so the body is capturable below Fusion: the RoPE frequency table (`freqs`, was a
    /// per-call `from_floats`) and the `arange(T_max)` mask index (`arange_tmax`, was a per-call
    /// `Tensor::arange`) are PRECOMPUTED ONCE by the caller and passed in. Numerically identical to
    /// [`Qwen3Attention::forward_with_cache_static`].
    ///
    /// * `freqs` — `[head_dim/2]` RoPE inv-freq table (see [`crate::rope::rope_freqs`]).
    /// * `arange_tmax` — `[T_max]` Int `0..T_max` (the cache capacity), the position-mask index.
    pub fn forward_with_cache_static_pre(
        &self,
        hidden_states: Tensor<3>,
        pos: Tensor<1, Int>,
        cache: &mut KVCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
    ) -> Tensor<3> {
        self.forward_with_cache_static_pre_lp(
            hidden_states,
            pos,
            cache,
            prec,
            freqs,
            arange_tmax,
            None,
        )
    }

    /// LEFT-PAD-aware variant of [`Self::forward_with_cache_static_pre`] (P4 — prompt-length buckets).
    /// A length-`L` prompt routed to a bucket-`B` graph is LEFT-PADDED to `B`; the `lo` device counter
    /// (`= B - L`, the number of left-pad columns) masks attention columns `idx < lo` so real tokens
    /// never attend to pad keys — combined with the existing `idx > pos` future mask. Because RoPE is
    /// relative and every real/decode position is shifted uniformly by `lo`, masking the pad makes the
    /// bucketized decode invariant to the left-pad (it matches the prompt run at its true length `L`).
    /// `lo = None` (or a zero counter) is the no-padding case == [`Self::forward_with_cache_static_pre`].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache_static_pre_lp(
        &self,
        hidden_states: Tensor<3>,
        pos: Tensor<1, Int>,
        cache: &mut KVCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
        lo: Option<&Tensor<1, Int>>,
    ) -> Tensor<3> {
        let [batch_size, seq_len, _] = hidden_states.dims(); // seq_len == 1 (decode)

        let query = linear3(&self.q_proj, hidden_states.clone(), prec);
        let key = linear3(&self.k_proj, hidden_states.clone(), prec);
        let value = linear3(&self.v_proj, hidden_states, prec);

        let query = query.reshape([batch_size, seq_len, self.num_heads, self.head_dim]);
        let key = key.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);
        let value = value.reshape([batch_size, seq_len, self.num_kv_heads, self.head_dim]);

        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        // RoPE position of the new token = `pos` (DEVICE); freqs PRECOMPUTED (no per-step from_floats).
        let position_ids = pos.clone().reshape([1, 1]).repeat(&[batch_size, 1]); // [B, 1] Int
        let (cos, sin) = compute_rope_embeddings_pre(position_ids, freqs.clone());
        let (query, key) = apply_rope(query, key, cos, sin);

        let (key, value) = cache.update_static(&pos, key, value);
        let t_max = key.dims()[1];

        let n_rep = self.num_heads / self.num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        // Position mask from the PRECOMPUTED arange (no per-step Tensor::arange alloc/stage): true
        // (⇒ -inf) for FUTURE columns `idx > pos`, and — under left-pad (P4) — also for the LEFT-PAD
        // columns `idx < lo` so real tokens ignore the pad keys.
        let idx = arange_tmax.clone().reshape([1, 1, 1, t_max]);
        let pos_mask = match lo {
            Some(lo) => idx
                .clone()
                .greater(pos.reshape([1, 1, 1, 1]))
                .bool_or(idx.lower(lo.clone().reshape([1, 1, 1, 1]))),
            None => idx.greater(pos.reshape([1, 1, 1, 1])), // [1,1,1,T_max] Bool, true above pos
        };

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            Some(pos_mask),
            None,
            AttentionModuleOptions::default(),
        );

        let attn_output = attn_output.movedim(1, 2).reshape([
            batch_size as i64,
            seq_len as i64,
            (self.num_heads * self.head_dim) as i64,
        ]);

        linear3(&self.o_proj, attn_output, prec)
    }
}
