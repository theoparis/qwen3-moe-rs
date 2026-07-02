//! Qwen3 decoder model.
//!
//! This module provides both the base model (for embeddings) and the causal LM
//! model (for text generation).

use burn::{
    Tensor,
    config::Config,
    module::{Ignored, Module},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{Bool, Int, activation::{silu, softmax}},
};

use super::attention::{Qwen3Attention, Qwen3AttentionConfig};
use super::cache::{KVCache, ModelCache};
use super::linear2d::{linear3, Precision};

/// Configuration for Qwen3 model.
#[derive(Config, Debug)]
pub struct Qwen3Config {
    /// Vocabulary size.
    #[config(default = 151936)]
    pub vocab_size: usize,
    /// Hidden dimension.
    #[config(default = 2560)]
    pub hidden_size: usize,
    /// Intermediate FFN dimension.
    #[config(default = 9728)]
    pub intermediate_size: usize,
    /// Number of transformer layers.
    #[config(default = 36)]
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    #[config(default = 32)]
    pub num_attention_heads: usize,
    /// Number of key-value heads for GQA.
    #[config(default = 8)]
    pub num_key_value_heads: usize,
    /// Dimension per attention head.
    /// If not set, defaults to hidden_size / num_attention_heads.
    pub head_dim: Option<usize>,
    /// RMSNorm epsilon.
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    /// RoPE theta.
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    /// Maximum sequence length.
    #[config(default = 40960)]
    pub max_position_embeddings: usize,
    /// Whether the input embedding and the output head share one matrix. Qwen3 0.6/1.7/4B = true
    /// (tied); 8B/14B/32B = false (a separate `lm_head.weight`).
    #[config(default = true)]
    pub tie_word_embeddings: bool,
}

impl Default for Qwen3Config {
    fn default() -> Self {
        Qwen3Config::new()
    }
}

impl Qwen3Config {
    /// Get the effective head dimension.
    pub fn get_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Configuration for Qwen3-0.6B model.
    /// Note: Qwen3-0.6B uses head_dim=128 (not hidden_size/num_heads=64).
    /// q_proj: [2048, 1024] = 16 heads × 128 head_dim
    /// k_proj/v_proj: [1024, 1024] = 8 kv_heads × 128 head_dim
    pub fn qwen3_0_6b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(1024)
            .with_intermediate_size(3072)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(16)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }

    /// Configuration for Qwen3-1.7B model.
    pub fn qwen3_1_7b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(2048)
            .with_intermediate_size(6144)
            .with_num_hidden_layers(28)
            .with_num_attention_heads(16)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }

    /// Configuration for Qwen3-4B model (default).
    /// Note: Standard Qwen3-4B has head_dim = 80 (2560/32).
    pub fn qwen3_4b() -> Self {
        Qwen3Config::new()
        // Uses default values: 2560 hidden, 9728 intermediate, 36 layers, 32 heads, 8 kv heads
        // head_dim = 2560/32 = 80
    }

    /// Configuration for Z-Image text encoder variant.
    /// This uses Qwen3-4B architecture but with head_dim=128 instead of 80.
    /// q_proj: [4096, 2560] = 32 heads × 128 head_dim
    /// k_proj/v_proj: [1024, 2560] = 8 kv_heads × 128 head_dim
    pub fn z_image_text_encoder() -> Self {
        Qwen3Config::new()
            .with_hidden_size(2560)
            .with_intermediate_size(9728)
            .with_num_hidden_layers(36)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128)) // Key difference: 128 instead of 80
    }

    /// Configuration for Qwen3-8B model.
    pub fn qwen3_8b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(4096)
            .with_intermediate_size(12288)
            .with_num_hidden_layers(36)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
    }

    /// Qwen3-14B preset. UNTIED embeddings (a separate `lm_head.weight`).
    pub fn qwen3_14b() -> Self {
        Qwen3Config::new()
            .with_hidden_size(5120)
            .with_intermediate_size(17408)
            .with_num_hidden_layers(40)
            .with_num_attention_heads(40)
            .with_num_key_value_heads(8)
            .with_head_dim(Some(128))
            .with_tie_word_embeddings(false)
    }
}

impl Qwen3Config {
    /// Initialize the model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Qwen3Model<B> {
        let layers: Vec<Qwen3DecoderLayer<B>> = (0..self.num_hidden_layers)
            .map(|_| Qwen3DecoderLayerConfig::from_model_config(self).init(device))
            .collect();

        Qwen3Model {
            config: Ignored(self.clone()),
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// Qwen3 decoder-only transformer model.
#[derive(Module, Debug)]
pub struct Qwen3Model<B: Backend> {
    config: Ignored<Qwen3Config>,
    embed_tokens: Embedding<B>,
    layers: Vec<Qwen3DecoderLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3Model<B> {
    /// Get embedding weight tensor for debugging.
    pub fn embed_tokens_weight(&self) -> Tensor<B, 2> {
        self.embed_tokens.weight.val()
    }

    /// Forward pass returning all hidden states.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Attention mask [batch, seq]
    ///
    /// # Returns
    /// A vector of hidden states from each layer, plus the final normalized output.
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        prec: Precision,
    ) -> Vec<Tensor<B, 3>> {
        let [batch_size, seq_len] = input_ids.dims();
        let device = input_ids.device();

        // Default RoPE positions are the column indices. `forward_with_positions` takes explicit
        // ones for left-padded / packed sequences.
        let position_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch_size, 1]);
        self.forward_with_positions(input_ids, attention_mask, position_ids, prec)
    }

    /// Forward returning all hidden states, with EXPLICIT RoPE position ids `[batch, seq]`.
    ///
    /// Use this for left-padded or packed sequences where a token's RoPE position differs from its
    /// column index (e.g. a left-padded prompt batch: `position_ids = cumsum(pad_mask) - 1`, the pad
    /// clamped to 0). `forward` is exactly this with `position_ids = arange(0..seq_len)`.
    pub fn forward_with_positions(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        prec: Precision,
    ) -> Vec<Tensor<B, 3>> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Collect hidden states from each layer
        // We push AFTER processing each layer to match HuggingFace behavior
        // hidden_states[0] = embedding output
        // hidden_states[i] = output of layer i-1 (for i > 0)
        // hidden_states[-1] = final norm output
        let mut all_hidden_states = Vec::with_capacity(self.layers.len() + 2);
        all_hidden_states.push(hidden_states.clone()); // Embedding output

        // Pass through decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(hidden_states, attention_mask.clone(), position_ids.clone(), prec);
            all_hidden_states.push(hidden_states.clone()); // Push AFTER processing

            // Debug: print first layer output
            if i == 0 && std::env::var("QWEN3_DEBUG").is_ok() {
                let debug_vals: Vec<f32> = hidden_states.clone()
                    .cast(burn::tensor::DType::F32)
                    .slice([0..1, 0..1, 0..10])
                    .reshape([10])
                    .into_data()
                    .as_slice::<f32>()
                    .unwrap_or(&[])
                    .to_vec();
                eprintln!("[DEBUG] Layer 0 output[0,0,:10]: {:?}", debug_vals);
            }
        }

        // Note: We do NOT push final norm output to match HuggingFace behavior
        // HuggingFace hidden_states has 37 elements (1 embedding + 36 layers)
        // hidden_states[-1] = layer 35 output (before final norm)
        // hidden_states[-2] = layer 34 output
        all_hidden_states
    }

    /// Get text embeddings for Z-Image.
    ///
    /// Returns the second-to-last layer hidden states, masked by attention mask.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Attention mask [batch, seq] (true = valid token)
    pub fn encode(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Tensor<B, 2, Bool>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        // Explicit precision (Gemini review): callers choose; inference embeddings pass `F32`.
        let all_hidden_states = self.forward(input_ids, Some(attention_mask.clone()), prec);

        // Get second-to-last layer (index -2)
        let hidden_states = all_hidden_states[all_hidden_states.len() - 2].clone();

        // The Python code does: prompt_embed[prompt_mask]
        // This extracts only the valid tokens. For simplicity, we'll mask and let
        // the caller handle extraction if needed.
        hidden_states
    }

    /// Forward pass with KV cache for efficient autoregressive generation.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq] (new tokens only during generation)
    /// * `attention_mask` - Optional attention mask [batch, total_seq]
    /// * `position_ids` - Position indices [batch, seq] (positions for new tokens)
    /// * `cache` - Mutable reference to model cache
    ///
    /// # Returns
    /// Final hidden states tensor [batch, seq, hidden_size]
    pub fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        // Token embeddings
        let mut hidden_states = self.embed_tokens.forward(input_ids);

        // Pass through decoder layers with cache
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden_states = layer.forward_with_cache(
                hidden_states,
                attention_mask.clone(),
                position_ids.clone(),
                layer_cache,
                prec,
            );
        }

        // Final layer norm
        self.norm.forward(hidden_states)
    }

    /// Phase-2 fixed-shape, device-`pos`-indexed decode forward (docs/cudagraph/DESIGN.md §7). Forwards
    /// ONE token through every layer's [`Qwen3DecoderLayer::forward_with_cache_static`]: the KV write
    /// offset, the RoPE position, and the attention position mask all come from the `[1]` Int DEVICE
    /// counter `pos`, so no per-step op bakes the host loop index. Returns the final hidden states
    /// `[B, 1, hidden]`.
    pub fn forward_with_cache_static(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden_states = layer.forward_with_cache_static(hidden_states, pos.clone(), layer_cache, prec);
        }
        self.norm.forward(hidden_states)
    }

    /// CUDA-graph-CAPTURABLE decode forward (P-final): [`Self::forward_with_cache_static`] with the RoPE
    /// freq table + arange(T_max) precomputed once and threaded through every layer (no per-step host
    /// staging). Numerically identical.
    pub fn forward_with_cache_static_pre(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        prec: Precision,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
    ) -> Tensor<B, 3> {
        self.forward_with_cache_static_pre_lp(input_ids, pos, cache, prec, freqs, arange_tmax, None)
    }

    /// LEFT-PAD-aware decode forward (P4): [`Self::forward_with_cache_static_pre`] threading the `lo`
    /// left-pad-column counter into every layer's masked attention. `lo = Some(pad_len)` makes a
    /// bucket-`B` graph mask the left-pad of a true-length-`L` prompt (`pad_len = B - L`).
    #[allow(clippy::too_many_arguments)]
    pub fn forward_with_cache_static_pre_lp(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        prec: Precision,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
        lo: Option<&Tensor<B, 1, Int>>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden_states = layer.forward_with_cache_static_pre_lp(
                hidden_states,
                pos.clone(),
                layer_cache,
                prec,
                freqs,
                arange_tmax,
                lo,
            );
        }
        self.norm.forward(hidden_states)
    }

    /// Create a new cache for this model.
    pub fn new_cache(&self) -> ModelCache<B> {
        ModelCache::new(self.layers.len())
    }

    /// Create a STATIC pre-allocated cache (Phase 2) for `capacity = prompt_len + max_new_tokens`.
    pub fn new_cache_with_capacity(&self, capacity: usize) -> ModelCache<B> {
        ModelCache::with_capacity(self.layers.len(), capacity)
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Configuration for a single decoder layer.
#[derive(Config, Debug)]
struct Qwen3DecoderLayerConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
}

impl Qwen3DecoderLayerConfig {
    fn from_model_config(config: &Qwen3Config) -> Self {
        Qwen3DecoderLayerConfig::new(
            config.hidden_size,
            config.intermediate_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.get_head_dim(),
            config.rms_norm_eps,
            config.rope_theta,
        )
    }

    fn init<B: Backend>(&self, device: &B::Device) -> Qwen3DecoderLayer<B> {
        Qwen3DecoderLayer {
            self_attn: Qwen3AttentionConfig::new(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
            )
            .with_head_dim(Some(self.head_dim))
            .with_rope_theta(self.rope_theta)
            .with_rms_norm_eps(self.rms_norm_eps)
            .init(device),
            mlp: Qwen3MLP::new(self.hidden_size, self.intermediate_size, device),
            input_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size)
                .with_epsilon(self.rms_norm_eps)
                .init(device),
        }
    }
}

/// A single Qwen3 decoder layer.
#[derive(Module, Debug)]
struct Qwen3DecoderLayer<B: Backend> {
    self_attn: Qwen3Attention<B>,
    mlp: Qwen3MLP<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3DecoderLayer<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        // Self attention with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids, prec);
        let hidden_states = residual + hidden_states;

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    fn forward_with_cache(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut KVCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        // Self attention with pre-norm and cache
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache(hidden_states, attention_mask, position_ids, cache, prec);
        let hidden_states = residual + hidden_states;

        // MLP with pre-norm
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    /// Phase-2 fixed-shape decode layer (docs/cudagraph/DESIGN.md §7): the device-`pos`-indexed
    /// counterpart of [`forward_with_cache`], over [`Qwen3Attention::forward_with_cache_static`].
    fn forward_with_cache_static(
        &self,
        hidden_states: Tensor<B, 3>,
        pos: Tensor<B, 1, Int>,
        cache: &mut KVCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache_static(hidden_states, pos, cache, prec);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    /// CUDA-graph-CAPTURABLE decode layer (P-final): [`Self::forward_with_cache_static`] over
    /// [`Qwen3Attention::forward_with_cache_static_pre`] (precomputed RoPE freqs + arange, no per-step
    /// host staging). Numerically identical.
    /// LEFT-PAD-aware decode layer (P4): the device-`pos` static decode threading the `lo` left-pad-
    /// column counter into [`Qwen3Attention::forward_with_cache_static_pre_lp`]. `lo = None` is the
    /// no-padding case (== the P-final `forward_with_cache_static_pre`).
    #[allow(clippy::too_many_arguments)]
    fn forward_with_cache_static_pre_lp(
        &self,
        hidden_states: Tensor<B, 3>,
        pos: Tensor<B, 1, Int>,
        cache: &mut KVCache<B>,
        prec: Precision,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
        lo: Option<&Tensor<B, 1, Int>>,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states =
            self.self_attn.forward_with_cache_static_pre_lp(hidden_states, pos, cache, prec, freqs, arange_tmax, lo);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }
}

/// Qwen3 MLP with SiLU gating (SwiGLU style). `pub(crate)` so the MoE block (src/moe.rs) reuses it
/// verbatim as a single expert.
#[derive(Module, Debug)]
pub(crate) struct Qwen3MLP<B: Backend> {
    gate_proj: Linear<B>,
    up_proj: Linear<B>,
    down_proj: Linear<B>,
}

impl<B: Backend> Qwen3MLP<B> {
    pub(crate) fn new(hidden_size: usize, intermediate_size: usize, device: &B::Device) -> Self {
        Qwen3MLP {
            gate_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(hidden_size, intermediate_size)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(intermediate_size, hidden_size)
                .with_bias(false)
                .init(device),
        }
    }

    pub(crate) fn forward(&self, x: Tensor<B, 3>, prec: Precision) -> Tensor<B, 3> {
        // Batch-safe 2-D Linear (see linear2d.rs): CubeCL's batched matmul corrupts rows
        // past the first at batch>1 for some shapes on sm_121.
        let gate = silu(linear3(&self.gate_proj, x.clone(), prec));
        let up = linear3(&self.up_proj, x, prec);
        linear3(&self.down_proj, gate * up, prec)
    }
}

// ============================================================================
// Causal Language Model
// ============================================================================

impl Qwen3Config {
    /// Initialize a causal language model (for text generation).
    pub fn init_causal_lm<B: Backend>(&self, device: &B::Device) -> Qwen3ForCausalLM<B> {
        // Untied models (e.g. Qwen3-14B) get a separate no-bias lm_head; tied models keep `None`
        // and project from the embedding. See `lm_logits`.
        let lm_head = if self.tie_word_embeddings {
            None
        } else {
            Some(LinearConfig::new(self.hidden_size, self.vocab_size).with_bias(false).init(device))
        };
        Qwen3ForCausalLM {
            model: self.init(device),
            lm_head,
            train_precision: Ignored(Precision::F32),
            infer_precision: Ignored(Precision::F32),
        }
    }
}

/// Qwen3 model with a language modeling head for text generation.
///
/// The LM head is TIED to the token embedding (Qwen3 shares one matrix), so logits are
/// projected from `model.embed_tokens.weight` directly in `forward` (`tied_logits`). There is
/// deliberately no separate `lm_head` parameter — a detached copy would train untied.
#[derive(Module, Debug)]
pub struct Qwen3ForCausalLM<B: Backend> {
    /// The base transformer model.
    pub model: Qwen3Model<B>,
    /// Separate output head for UNTIED models (`tie_word_embeddings = false`, e.g. Qwen3-14B).
    /// `None` for tied models, where logits project from `embed_tokens.weight` (see `lm_logits`).
    lm_head: Option<Linear<B>>,
    /// Compute precision for the TRAINING forward (no cache). Default `F32`.
    ///
    /// RUNTIME CONFIG, not model state: `Ignored<_>` is not serialized — `into_record` drops the
    /// value and `load_record` retains the in-memory one (verified: burn-core constant.rs:281,285).
    /// So `load_record` does NOT wipe an already-set precision, but the value also does not travel
    /// with a checkpoint: a freshly built + weight-loaded model defaults to `F32`, so set precision
    /// AFTER building (as the training examples do).
    ///
    /// MULTI-GPU / RESUME WARNING (review, cross-model): because the value does not travel with a
    /// record, a record-based model COPY — Burn `Learner` multi-device worker sync, or
    /// resume-from-checkpoint — silently reverts to `F32`. A master training in bf16 while workers
    /// default to f32 diverges silently. This pipeline is single-GPU + manual loop (no `Learner`),
    /// so it is safe today; BEFORE going multi-GPU, make precision serializable (move into
    /// `Qwen3Config`) so it syncs to workers.
    train_precision: Ignored<Precision>,
    /// Compute precision for INFERENCE (cached generation + non-cache `generate`). Default `F32`,
    /// decoupled from training so bf16 training never silently changes generation or HF parity.
    /// Same `Ignored` runtime-config semantics as `train_precision`.
    infer_precision: Ignored<Precision>,
}

impl<B: Backend> Qwen3ForCausalLM<B> {
    /// Forward pass returning logits over the vocabulary.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Optional attention mask [batch, seq]
    ///
    /// # Returns
    /// Logits tensor of shape [batch, seq, vocab_size]
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        // Training entry point: uses `train_precision` (default f32; set bf16 for mixed precision).
        self.forward_with_precision(input_ids, attention_mask, *self.train_precision)
    }

    /// Forward (no cache) at an explicit precision. `forward` uses `train_precision`; inference
    /// callers (e.g. `generate`) use `infer_precision`, so bf16 training never leaks into generation.
    fn forward_with_precision(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let all_hidden_states = self.model.forward(input_ids, attention_mask, prec);
        // Get the last layer's hidden states. `Qwen3Model::forward` deliberately does
        // NOT apply the final RMSNorm (a Z-Image text-encoder artifact: it stops at the
        // last decoder layer). For causal LM logits we MUST apply `model.norm` before the
        // lm_head, exactly as the reference HF model (and `forward_with_cache`) does.
        let hidden_states = all_hidden_states.last().unwrap().clone();
        let hidden_states = self.model.norm.forward(hidden_states);
        self.lm_logits(hidden_states)
    }

    /// Logits (no cache) at EXPLICIT RoPE positions, at `train_precision`.
    ///
    /// For a left-padded prompt batch: pass the prompt padding mask (`true` = real token) and the
    /// matching `position_ids` (`cumsum(mask) - 1`, pad clamped to 0) so RoPE numbers the real
    /// tokens from 0 and attention ignores the pad. Otherwise identical to `forward`; with
    /// `attention_mask = None` and `position_ids = arange` it equals `forward`.
    pub fn forward_with_positions(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let all_hidden_states =
            self.model.forward_with_positions(input_ids, attention_mask, position_ids, *self.train_precision);
        let hidden_states = all_hidden_states.last().unwrap().clone();
        let hidden_states = self.model.norm.forward(hidden_states);
        self.lm_logits(hidden_states)
    }

    /// Forward pass with KV cache returning logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq]
    /// * `attention_mask` - Optional attention mask [batch, total_seq]
    /// * `position_ids` - Position indices [batch, seq]
    /// * `cache` - Mutable reference to model cache
    ///
    /// # Returns
    /// Logits tensor of shape [batch, seq, vocab_size]
    pub fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
    ) -> Tensor<B, 3> {
        let hidden_states = self.model.forward_with_cache(input_ids, attention_mask, position_ids, cache, *self.infer_precision);
        self.lm_logits(hidden_states)
    }

    /// Phase-2 fixed-shape, device-`pos`-indexed decode forward returning logits (docs/cudagraph/
    /// DESIGN.md §7). The capture-ready decode step: ONE token `[B, 1]` in, logits `[B, 1, vocab]` out,
    /// every per-step op fixed-shape and indexed by the `[1]` Int DEVICE counter `pos` (KV write offset
    /// + RoPE position + attention position mask). Numerically identical to the `forward_with_cache`
    /// decode branch (the growing-prefix path); see `group_sample_cached_device_static`.
    pub fn forward_with_cache_static(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
    ) -> Tensor<B, 3> {
        let hidden_states = self.model.forward_with_cache_static(input_ids, pos, cache, *self.infer_precision);
        self.lm_logits(hidden_states)
    }

    /// CUDA-graph-CAPTURABLE decode forward returning logits (P-final): [`Self::forward_with_cache_static`]
    /// with precomputed RoPE freqs + arange(T_max) threaded through (no per-step host staging), so the
    /// whole step is recordable below Fusion. Numerically identical.
    pub fn forward_with_cache_static_pre(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
    ) -> Tensor<B, 3> {
        self.forward_with_cache_static_pre_lp(input_ids, pos, cache, freqs, arange_tmax, None)
    }

    /// LEFT-PAD-aware capturable decode forward returning logits (P4 — prompt-length buckets): same as
    /// [`Self::forward_with_cache_static_pre`] but with the `lo` left-pad-column counter so a bucket-`B`
    /// graph masks the left-pad of a true-length-`L` prompt. See
    /// [`Qwen3Attention::forward_with_cache_static_pre_lp`] for the invariance argument.
    pub fn forward_with_cache_static_pre_lp(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
        lo: Option<&Tensor<B, 1, Int>>,
    ) -> Tensor<B, 3> {
        let hidden_states = self.model.forward_with_cache_static_pre_lp(
            input_ids,
            pos,
            cache,
            *self.infer_precision,
            freqs,
            arange_tmax,
            lo,
        );
        self.lm_logits(hidden_states)
    }

    /// Project hidden states to vocab logits using the TIED token-embedding matrix.
    ///
    /// Qwen3 ties the input embedding and the output head into ONE matrix, so we matmul
    /// `hidden @ embed_tokens.weight^T` directly. This keeps the head truly tied under autodiff:
    /// the full LM gradient flows back to the single shared embedding. A separate `lm_head`
    /// param — even one initialized equal via `tie_lm_head_to_embeddings` — trains UNTIED (the
    /// dense output gradient goes to the copy while the embedding gets only sparse input-side
    /// gradients) and is discarded on tied-format export. So the training path MUST project from
    /// the embedding. The `lm_head` field is now vestigial (kept only so other examples compile).
    /// `[B*S,H] @ [H,V]` is 2-D, which dodges the CubeCL broadcast batched-matmul bug like `linear3`.
    fn tied_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, h] = hidden.dims();
        // Keep the GEMM uniform-dtype: cast the tied embedding to the ACTIVATION's dtype (a mixed
        // bf16/f32 matmul silently corrupts on the CubeCL CUDA backend in BOTH directions — see the
        // linear3 F32-arm invariant in linear2d.rs). No-op when hidden and the embedding already
        // match (the f32-checkpoint GRPO setup); robust if a bf16 checkpoint is ever loaded.
        let w = self.model.embed_tokens.weight.val().transpose().cast(hidden.dtype());
        let vocab = w.dims()[1];
        hidden.reshape([b * s, h]).matmul(w).reshape([b, s, vocab])
    }

    /// Project final hidden states to vocab logits. Untied models (Qwen3-14B, `tie_word_embeddings
    /// = false`) use the separate `lm_head`; tied models project from the token embedding. Both go
    /// through the batch-safe 2-D GEMM (the head via `linear3`, tied via the reshape in
    /// `tied_logits`) to dodge the CubeCL broadcast batched-matmul bug; the head stays f32.
    fn lm_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        match &self.lm_head {
            Some(head) => linear3(head, hidden, Precision::F32),
            None => self.tied_logits(hidden),
        }
    }

    /// Dtype of the (tied) token-embedding weight — the matrix that also serves as the LM head.
    /// Cheap (metadata only); used to assert the loaded weights are f32 before training.
    pub fn embed_weight_dtype(&self) -> burn::tensor::DType {
        self.model.embed_tokens.weight.val().dtype()
    }

    /// `ParamId` of the tied token embedding (which is also the LM head). Used by the freeze-head
    /// ablation (G4d): drop this gradient from `GradientsParams` before `optim.step` to freeze the
    /// embedding + tied head and train ONLY the bf16 Linears — isolating the bf16 gradient path.
    ///
    /// WARNING (untied models): for `tie_word_embeddings = false` (Qwen3-8B/14B/32B) the OUTPUT head
    /// is the separate `lm_head`, NOT the embedding — freezing THIS id leaves `lm_head` trainable.
    /// The G4d ablation is a tied-model tool; untied freeze / head-targeting must use the lm_head id.
    /// (Normal GRPO is unaffected: it trains all params and projects logits via `lm_logits`.)
    pub fn embed_param_id(&self) -> burn::module::ParamId {
        self.model.embed_tokens.weight.id
    }

    /// Set the TRAINING compute precision (default f32). `Precision::Bf16` runs the 7 Linear GEMMs
    /// in bf16 (f32 accumulation) while master weights, optimizer, norms, softmax and the tied
    /// LM head stay f32 — the mixed-precision recipe.
    pub fn with_train_precision(mut self, prec: Precision) -> Self {
        self.train_precision = Ignored(prec);
        self
    }

    /// Set the INFERENCE compute precision. Only `Precision::F32` is accepted (the default).
    ///
    /// bf16 inference panics with a `DTypeMismatch` in `RmsNorm` inside `forward_with_cache` on the
    /// Fusion backend, so this setter REJECTS `Bf16` up front (fail fast) rather than letting you
    /// build a model that dies during generation. bf16 is supported for TRAINING via
    /// `with_train_precision`. Remove this guard once the Fusion RmsNorm dtype issue is fixed.
    ///
    /// # Panics
    /// Panics if `prec` is `Precision::Bf16`.
    pub fn with_infer_precision(mut self, prec: Precision) -> Self {
        assert!(
            prec == Precision::F32,
            "bf16 inference is unsupported (forward_with_cache panics with DTypeMismatch in RmsNorm \
             on the Fusion backend). Use Precision::F32 for inference; bf16 is for TRAINING via \
             with_train_precision."
        );
        self.infer_precision = Ignored(prec);
        self
    }

    /// Deprecated no-op. Logits are now projected from the SHARED token embedding inside
    /// `forward` (`tied_logits`), so there is no separate `lm_head` to tie — keeping callers
    /// compiling. (The old version copied a DETACHED head that trained UNTIED, dropping the LM
    /// gradient from the embedding and discarding the head on tied-format export.)
    pub fn tie_lm_head_to_embeddings(self) -> Self {
        self
    }

    /// Generate text autoregressively (without KV cache - slower but simpler).
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch_size, _] = input_ids.dims();

        let mut generated = input_ids;

        for _ in 0..max_new_tokens {
            // Get logits for the last position (inference precision, default f32)
            let logits = self.forward_with_precision(generated.clone(), None, *self.infer_precision);
            let [_, seq_len, vocab_size] = logits.dims();

            // Extract last token logits: [batch, vocab_size]
            let next_token_logits = logits.slice([0..batch_size, (seq_len - 1)..seq_len, 0..vocab_size])
                .reshape([batch_size, vocab_size]);

            // Apply temperature
            let next_token_logits = if temperature != 1.0 {
                next_token_logits / temperature
            } else {
                next_token_logits
            };

            // Sample next token
            // argmax(1) returns [batch, 1], flatten to [batch]
            let next_token: Tensor<B, 1, Int> = if temperature == 0.0 {
                // Greedy decoding
                next_token_logits.argmax(1).flatten(0, 1)
            } else {
                // Apply top-k and top-p, then sample
                let probs = softmax(next_token_logits, 1);
                sample_from_probs(probs, top_k, top_p, &device)
            };

            // Append to generated sequence
            generated = Tensor::cat(vec![generated, next_token.unsqueeze_dim(1)], 1);
        }

        generated
    }

    /// Generate text autoregressively with KV cache (faster).
    ///
    /// This is more efficient than `generate()` as it only computes attention
    /// for new tokens while reusing cached key-value pairs from previous tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
    ) -> Tensor<B, 2, Int> {
        // Use default EOS tokens for Qwen3
        self.generate_with_cache_eos(input_ids, max_new_tokens, temperature, top_p, top_k, &[151643, 151645])
    }

    /// Generate text autoregressively with KV cache and custom EOS tokens.
    ///
    /// # Arguments
    /// * `input_ids` - Initial token IDs [batch, seq]
    /// * `max_new_tokens` - Maximum number of tokens to generate
    /// * `temperature` - Sampling temperature (1.0 = no change, <1 = sharper, >1 = flatter)
    /// * `top_p` - Nucleus sampling threshold (1.0 = disabled)
    /// * `top_k` - Top-k sampling (0 = disabled)
    /// * `eos_token_ids` - Token IDs that signal end of generation (e.g., [151643, 151645] for <|endoftext|> and <|im_end|>)
    ///
    /// # Returns
    /// Generated token IDs [batch, seq + generated]
    pub fn generate_with_cache_eos(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        eos_token_ids: &[i64],
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch_size, initial_seq_len] = input_ids.dims();

        // Create cache
        let mut cache = self.model.new_cache();

        // First forward pass: process all input tokens
        let position_ids = Tensor::<B, 1, Int>::arange(0..initial_seq_len as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[batch_size, 1]);

        let logits = self.forward_with_cache(input_ids.clone(), None, position_ids, &mut cache);
        let [_, _, vocab_size] = logits.dims();

        // Get first new token from the last position
        let next_token_logits = logits.slice([0..batch_size, (initial_seq_len - 1)..initial_seq_len, 0..vocab_size])
            .reshape([batch_size, vocab_size]);

        let next_token_logits = if temperature != 1.0 && temperature != 0.0 {
            next_token_logits / temperature
        } else {
            next_token_logits
        };

        // argmax(1) returns [batch, 1], reshape to [batch]
        let mut next_token: Tensor<B, 1, Int> = if temperature == 0.0 {
            next_token_logits.argmax(1).flatten(0, 1)
        } else {
            let probs = softmax(next_token_logits, 1);
            sample_from_probs(probs, top_k, top_p, &device)
        };

        // Check if first token is EOS
        // Cast to I64 first: CUDA Int tensors are i32, so reading them directly as i64 fails.
        let token_id = next_token.clone().cast(burn::tensor::DType::I64).into_data().as_slice::<i64>().map(|s| s[0]).unwrap_or(0);
        if eos_token_ids.contains(&token_id) {
            return input_ids;
        }

        let mut generated = Tensor::cat(vec![input_ids, next_token.clone().unsqueeze_dim(1)], 1);
        let mut current_pos = initial_seq_len;

        // Generate remaining tokens one at a time
        for _ in 1..max_new_tokens {
            current_pos += 1;

            // Position for the new token
            let position_ids = Tensor::<B, 1, Int>::from_data([current_pos as i64 - 1], &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[batch_size, 1]);

            // Forward only the new token
            let logits = self.forward_with_cache(
                next_token.clone().unsqueeze_dim(1),
                None,
                position_ids,
                &mut cache,
            );

            // Extract logits for the single new token
            let next_token_logits = logits.slice([0..batch_size, 0..1, 0..vocab_size])
                .reshape([batch_size, vocab_size]);

            let next_token_logits = if temperature != 1.0 && temperature != 0.0 {
                next_token_logits / temperature
            } else {
                next_token_logits
            };

            // argmax(1) returns [batch, 1], flatten to [batch]
            next_token = if temperature == 0.0 {
                next_token_logits.argmax(1).flatten(0, 1)
            } else {
                let probs = softmax(next_token_logits, 1);
                sample_from_probs(probs, top_k, top_p, &device)
            };

            // Check for EOS token
            // Cast to I64 first: CUDA Int tensors are i32, so reading them directly as i64 fails.
            let token_id = next_token.clone().cast(burn::tensor::DType::I64).into_data().as_slice::<i64>().map(|s| s[0]).unwrap_or(0);
            if eos_token_ids.contains(&token_id) {
                break;
            }

            generated = Tensor::cat(vec![generated, next_token.clone().unsqueeze_dim(1)], 1);
        }

        generated
    }

    /// Create a new cache for this model.
    pub fn new_cache(&self) -> ModelCache<B> {
        self.model.new_cache()
    }

    /// Create a STATIC pre-allocated cache (Phase 2) for `capacity = prompt_len + max_new_tokens`.
    pub fn new_cache_with_capacity(&self, capacity: usize) -> ModelCache<B> {
        self.model.new_cache_with_capacity(capacity)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.model.config.0.vocab_size
    }

    /// Get the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.model.config.0.hidden_size
    }
}

/// Sample a token from probability distribution.
///
/// Uses CPU-based sampling to avoid slow GPU random operations.
/// Falls back to argmax if sampling fails.
fn sample_from_probs<B: Backend>(
    probs: Tensor<B, 2>,
    top_k: usize,
    top_p: f32,
    device: &B::Device,
) -> Tensor<B, 1, Int> {
    use rand::Rng;

    let [batch_size, vocab_size] = probs.dims();

    // Transfer probs to CPU for sampling (GPU random is extremely slow)
    let probs_data = probs.into_data();
    let probs_slice: Vec<f32> = probs_data.as_slice::<half::bf16>()
        .map(|s| s.iter().map(|x| x.to_f32()).collect())
        .or_else(|_| probs_data.as_slice::<f32>().map(|s| s.to_vec()))
        .unwrap_or_default();

    if probs_slice.is_empty() {
        // Fallback: return 0
        return Tensor::zeros([batch_size], device);
    }

    let mut rng = rand::rng();
    let mut sampled_tokens = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row = &probs_slice[b * vocab_size..(b + 1) * vocab_size];
        // Shared, unit-tested sampler (`crate::sampling`): top-k AND top-p are BOTH applied now.
        // The previous inline sampler silently ignored top_p; generation and GRPO rollouts now
        // sample identically and correctly.
        let r: f32 = rng.random();
        sampled_tokens.push(crate::sampling::sample_index(row, top_k, top_p, r) as i64);
    }

    Tensor::from_data(sampled_tokens.as_slice(), device)
}
