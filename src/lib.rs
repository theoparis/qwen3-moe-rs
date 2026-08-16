//! Qwen3 language model implementation using the Burn deep learning framework.
//!
//! This crate provides a Rust implementation of the Qwen3 decoder-only transformer
//! architecture. Qwen3 is a large language model developed by Alibaba that features:
//!
//! - Grouped Query Attention (GQA) for efficient KV caching
//! - RoPE (Rotary Position Embeddings) with theta=1,000,000
//! - RMSNorm for layer normalization
//! - SwiGLU activation in feed-forward layers
//! - QK normalization in attention
//!
//! # Models
//!
//! This crate provides two model variants:
//!
//! - [`Qwen3Model`] - Base transformer model for embeddings and hidden states
//! - [`Qwen3ForCausalLM`] - Full causal language model with generation capabilities
//!
//! # Features
//!
//! - Load pretrained weights from HuggingFace safetensors format
//! - Tokenizer support via the `tokenizers` crate
//! - Text generation with temperature, top-k, and top-p sampling
//! - Extract hidden states for text embeddings
//! - Tied and untied output embeddings; single-file and sharded safetensors loading
//! - GRPO reinforcement-learning training (see the [`grpo`] module)
//! - Compatible with all Burn backends (CPU, CUDA, Metal, etc.)
//!
//! # Text Generation Example
//!
//! ```ignore
//! use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};
//!
//! // Load tokenizer and model
//! let tokenizer = Qwen3Tokenizer::from_file("tokenizer.json")?;
//! let mut model = Qwen3Config::default().init_causal_lm::<Backend>(&device);
//! model.load_weights("model.safetensors")?;
//!
//! // Tokenize prompt
//! let (input_ids, _) = tokenizer.encode("Once upon a time")?;
//! let input_tensor = Tensor::from_data(&input_ids, &device).unsqueeze();
//!
//! // Generate text
//! let output = model.generate(
//!     input_tensor,
//!     50,    // max_new_tokens
//!     0.7,   // temperature
//!     0.9,   // top_p
//!     50,    // top_k
//! );
//!
//! // Decode output tokens
//! let generated_text = tokenizer.decode(output)?;
//! ```
//!
//! # Text Embedding Example
//!
//! ```ignore
//! use qwen3_burn::{Qwen3Config, Qwen3Model, Qwen3Tokenizer};
//!
//! // Load tokenizer and base model
//! let tokenizer = Qwen3Tokenizer::from_file("tokenizer.json")?;
//! let mut model = Qwen3Config::default().init::<Backend>(&device);
//! model.load_weights("model.safetensors")?;
//!
//! // Get embeddings (second-to-last layer hidden states)
//! let (input_ids, attention_mask) = tokenizer.encode_prompt("Hello, world!")?;
//! let embeddings = model.encode(input_ids_tensor, attention_mask_tensor, Precision::F32);
//! ```
//!
//! # Model Configurations
//!
//! The default configuration matches the Z-Image text encoder variant:
//! - 36 layers, 2560 hidden size, 32 attention heads, 8 KV heads
//!
//! Common Qwen3 model sizes:
//!
//! | Model | Layers | Hidden | Heads | KV Heads |
//! |-------|--------|--------|-------|----------|
//! | 0.6B  | 28     | 1024   | 16    | 8        |
//! | 1.7B  | 28     | 2048   | 16    | 8        |
//! | 4B    | 36     | 2560   | 32    | 8        |
//! | 8B    | 36     | 4096   | 32    | 8        |
//! | 14B   | 40     | 5120   | 40    | 8        |
//! | 32B   | 64     | 5120   | 40    | 8        |
//!
//! Presets exist through 14B (`Qwen3Config::qwen3_0_6b()` .. `qwen3_14b()`); 0.6B-4B use tied
//! embeddings and 8B+ are untied (a separate `lm_head.weight`). There is no `qwen3_32b()` preset
//! yet — its config is expressible by hand from the row above.

mod attention;
mod cache;
/// CUDA-graph capture/replay harness for raw-CubeBackend static decode plus generic persistent
/// decode-state containers used by capture tests.
pub mod capture;
/// Typed, safe wrapper around the Burn-Fusion custom-op bridge (CUDA only). Foundation for the
/// custom CubeCL kernels (fp8 GEMM, MoE grouped-GEMM); enforces the §0b production rules.
#[cfg(feature = "cuda")]
pub mod cube_custom_op;
#[cfg(feature = "cubecl-gpu")]
pub(crate) mod cubecl_rt;
mod decoder;
pub mod expert_stream;
/// Custom CubeCL FlashAttention-style attention kernel (online softmax, f32 accum; CUDA only).
/// Validated on the GB10 vs an NdArray CPU oracle (see `examples/attn_kernel_spike.rs`).
#[cfg(feature = "cuda")]
pub mod flash_attn;
/// L2A.2 split-K online-softmax flash-decode (raw CubeBackend, capture-ready). CUDA only.
#[cfg(feature = "cuda")]
pub mod flash_decode;
pub mod grpo;
mod linear2d;
pub mod load;
mod moe;
/// CAPTURABLE top-k MoE DECODE block (keystone lever A of `docs/PERF_80TOKS_PLAN.md`). A post-load
/// pre-stacked contiguous expert-weight cache + a fixed-shape, NO-host-sync single-token top-k gather
/// decode that at `T=1` reads ONLY the `k` routed experts' weight slabs (8 of 128), not all `E`.
/// Backend-generic (pure Burn ops); validated on NdArray vs the `forward_oracle`/`forward_routed`.
pub mod moe_decode;
/// DROPLESS MoE grouped-GEMM fast path (CUDA only) — `docs/VLLM_KERNELS.md` §3. Computes EXACTLY the
/// `k*T` routed (token,expert) pairs via the vLLM `moe_align_block_size` layout (`sorted_token_ids`
/// + per-block `expert_ids`) and a block-per-segment SwiGLU GEMM with i64 global offsets — no token
/// drop (unlike `forward_routed_ondevice`'s capacity-padded path). Validated on the GB10 vs an
/// NdArray CPU oracle (see `examples/moe_grouped_spike.rs`).
#[cfg(feature = "cuda")]
pub mod moe_grouped;
/// L2C NVFP4 host codec plus CubeCL decode-GEMV / fused MoE kernels (CUDA and wgpu/Metal/Vulkan).
pub mod nvfp4;
#[cfg(feature = "cubecl-gpu")]
mod nvfp4_kernels;
pub mod nvfp4_linear;
pub mod nvidia_ckpt;
pub mod quant_gate;
pub mod qwen3_5;
mod rope;
/// Host token sampling: temperature + top-k + top-p (nucleus) filtering then categorical draw
/// ([`sampling::sample_index`]). Public so inference consumers (e.g. `examples/vllm_infer.rs`) can call
/// the one canonical sampler instead of copying it.
pub mod sampling;
/// Device-side (GPU) sampling + raw log-prob for the GRPO rollout decode step (lever §0-A): argmax /
/// logsumexp / Gumbel-max categorical sampling in pure Burn tensor ops, copying back only `[N]` tokens
/// + `[N]` log-probs instead of the `[N, V]` logits. See [`sampling_device`].
pub mod sampling_device;
/// M-S: OpenAI-compatible single-stream server (docs/SERVE_PLAN.md).
#[cfg(feature = "serve")]
pub mod serve;
mod tokenizer;
/// Fused fp8 W8A16 (e4m3 weight-only) GEMM kernel (CUDA only) — `docs/VLLM_KERNELS.md` §2. Reads
/// packed e4m3 weight BYTES from HBM and dequants in the GEMM load path. Validated on the GB10 vs an
/// NdArray CPU oracle + OCP golden vectors (see `examples/w8a16_spike.rs`).
#[cfg(feature = "cuda")]
pub mod w8a16;
/// Drop-in fp8 (e4m3) weight-only `W8A16Linear` (CUDA only) — `docs/PERF_80TOKS_PLAN.md` §2 lever B.
/// Wraps the `w8a16` fused dequant-in-GEMM kernel with a quantize-on-load path (`from_linear`) and a
/// `Linear`-shaped `forward` (M=1 decode + M>1). Validated vs the bf16 `Linear` (cosine > 0.999).
#[cfg(feature = "cuda")]
pub mod w8a16_linear;

pub use cache::{
    GdnModelCache, GdnStateCache, KVCache, ModelCache, Qwen3_5HybridCache, Qwen3_5HybridLayerCache,
};
pub use decoder::{Qwen3Config, Qwen3ForCausalLM, Qwen3Model};
pub use grpo::{GrpoConfig, GrpoMetrics, group_norm_advantage, grpo_loss, token_logprobs};
pub use linear2d::{Precision, linear3};
pub use moe::{
    MoeStaticDecode, Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeModel, Qwen3MoeSparseBlock,
};
pub use moe_decode::MoeExpertCache;
pub use qwen3_5::{Qwen3_5LayerType, Qwen3_5Model, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM};
pub use rope::rope_freqs;
pub use sampling_device::device_sample_step;
pub use tokenizer::Qwen3Tokenizer;
