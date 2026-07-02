//! Qwen3-MoE (Mixture-of-Experts) — Phase B1: the correct forward (the "Tier-1 oracle").
//!
//! Mirrors the dense `decoder.rs` model, swapping the dense `Qwen3MLP` for a sparse MoE block. The
//! attention (GQA + per-head QK-RMSNorm + RoPE), norms, KV cache, and the batch-safe `linear3` GEMM
//! are REUSED unchanged. Target: Qwen3-30B-A3B (128 experts, top-8, no shared expert, every layer
//! sparse, untied lm_head). See `docs/MOE_PLAN.md`.
//!
//! ## Routing (parity-critical, verified vs transformers v4.51.0)
//! What we compute, matching HF literally: `softmax` over ALL experts in fp32, then take the top-k by
//! **iterated argmax over those softmax probs** (`topk`/`sort` are HOST-SYNC on CubeCL — see
//! docs/MOE_PLAN.md §4; argmax over the monotone softmax returns HF's `topk` set), then renormalize
//! the kept k by their own sum (`norm_topk_prob`). Combine = Σ_{e∈top-k} w_e · SwiGLU_e(x), with NO
//! shared/dense term. (Equivalent and cheaper for B2: argmax over the raw logits then softmax only the
//! kept k, since `renorm(softmax_N(top_k))` == `softmax(top_k logits)` — deferred to the fast path.)
//!
//! ## This is the Tier-1 ORACLE
//! The combine is a dense-masked sum: every expert is evaluated over every token, weighted by a
//! per-(token,expert) gate matrix. Cost is `E×dense` — correct and the numerical reference, but NOT
//! the fast path. The fast path (a custom CubeCL grouped-GEMM kernel) is Phase B2, gated on a spike.

use burn::{
    config::Config,
    module::{Ignored, Module, Param, ParamId},
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
    prelude::Backend,
    tensor::{activation::{silu, softmax}, Bool, DType, Distribution, IndexingUpdateOp, Int, Shape, Tensor},
};

use crate::attention::{Qwen3Attention, Qwen3AttentionConfig};
use crate::cache::ModelCache;
use crate::linear2d::{linear3, Precision};
use crate::moe_decode::MoeExpertCache;
use crate::rope::rope_freqs;

#[cfg(feature = "cuda")]
use crate::moe_grouped::FusedSwigluBackend;

/// Build a LAZY `[d0, d1, d2]` stacked-expert `Param` (the deferred init only runs if the param is
/// ever materialized WITHOUT being loaded — e.g. the tiny CI model with no checkpoint). On the real
/// 30B the loader REPLACES these params before first access, so the random init never allocates
/// (mirrors how the per-expert `Linear` params used to stay lazy until load — the reason a 30B `init`
/// does not OOM). Small Normal(0, 0.02) init keeps the tiny-model parity tests non-degenerate.
fn lazy_expert_stack<B: Backend>(shape: [usize; 3], device: &B::Device) -> Param<Tensor<B, 3>> {
    Param::uninitialized(
        ParamId::new(),
        move |dev: &B::Device, _req_grad: bool| {
            Tensor::<B, 3>::random(shape, Distribution::Normal(0.0, 0.02), dev)
        },
        device.clone(),
        false,
        Shape::from(shape),
    )
}

/// Bias-free 2-D GEMM with the same precision semantics as [`crate::linear2d::linear3`] (the 2-D
/// flatten that dodges the sm_121 broadcast batched-matmul bug), for a raw `[d_in, d_out]` weight
/// SLAB sliced out of an expert stack (there is no `Linear` to call `linear3` on). `F32` = a plain
/// matmul (matching `Linear::forward` on a bias-free Linear); `Bf16` = bf16 inputs, f32 accumulation
/// (cubek `Acc=(bf16,f32)`), widened back to f32 — identical to `linear3`'s `matmul_bf16`.
fn matmul2<B: Backend>(x: Tensor<B, 2>, w: Tensor<B, 2>, prec: Precision) -> Tensor<B, 2> {
    match prec {
        // Keep the GEMM uniform-dtype; see the linear3 F32 invariant.
        Precision::F32 => {
            let xdt = x.dtype();
            x.matmul(w.cast(xdt))
        }
        Precision::Bf16 => x.cast(DType::BF16).matmul(w.cast(DType::BF16)).cast(DType::F32),
    }
}

/// Configuration for a Qwen3-MoE model (homogeneous all-sparse; every layer is a MoE block).
#[derive(Config, Debug)]
pub struct Qwen3MoeConfig {
    #[config(default = 151936)]
    pub vocab_size: usize,
    #[config(default = 2048)]
    pub hidden_size: usize,
    #[config(default = 48)]
    pub num_hidden_layers: usize,
    #[config(default = 32)]
    pub num_attention_heads: usize,
    #[config(default = 4)]
    pub num_key_value_heads: usize,
    /// Per-head dim. `None` => `hidden_size / num_attention_heads`. Qwen3-MoE sets it explicitly (128).
    pub head_dim: Option<usize>,
    #[config(default = 1e-6)]
    pub rms_norm_eps: f64,
    #[config(default = 1_000_000.0)]
    pub rope_theta: f64,
    #[config(default = 40960)]
    pub max_position_embeddings: usize,
    /// Total number of experts (Qwen3-MoE: 128).
    #[config(default = 128)]
    pub num_experts: usize,
    /// Experts activated per token, `top_k` (Qwen3-MoE: 8).
    #[config(default = 8)]
    pub num_experts_per_tok: usize,
    /// Per-expert SwiGLU inner dim (Qwen3-30B-A3B: 768).
    #[config(default = 768)]
    pub moe_intermediate_size: usize,
    /// Renormalize the kept top-k routing weights to sum to 1 (Qwen3-MoE: true).
    #[config(default = true)]
    pub norm_topk_prob: bool,
}

impl Qwen3MoeConfig {
    /// Effective per-head dimension.
    pub fn get_head_dim(&self) -> usize {
        self.head_dim.unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// Qwen3-30B-A3B preset (30B total / ~3.3B active). 128 experts, top-8, untied head.
    pub fn qwen3_30b_a3b() -> Self {
        Qwen3MoeConfig::new()
            .with_hidden_size(2048)
            .with_num_hidden_layers(48)
            .with_num_attention_heads(32)
            .with_num_key_value_heads(4)
            .with_head_dim(Some(128))
            .with_num_experts(128)
            .with_num_experts_per_tok(8)
            .with_moe_intermediate_size(768)
    }

    /// Tiny config for tests/CI (CPU): runs the full MoE path without a GPU or 30B weights.
    pub fn tiny() -> Self {
        Qwen3MoeConfig::new()
            .with_vocab_size(256)
            .with_hidden_size(64)
            .with_num_hidden_layers(2)
            .with_num_attention_heads(8)
            .with_num_key_value_heads(4)
            .with_head_dim(Some(16))
            .with_num_experts(8)
            .with_num_experts_per_tok(2)
            .with_moe_intermediate_size(32)
    }

    /// Initialize a Qwen3-MoE causal language model.
    pub fn init_causal_lm<B: Backend>(&self, device: &B::Device) -> Qwen3MoeForCausalLM<B> {
        let layers: Vec<Qwen3MoeDecoderLayer<B>> =
            (0..self.num_hidden_layers).map(|_| self.init_layer(device)).collect();
        let model = Qwen3MoeModel {
            config: Ignored(self.clone()),
            embed_tokens: EmbeddingConfig::new(self.vocab_size, self.hidden_size).init(device),
            layers,
            norm: RmsNormConfig::new(self.hidden_size).with_epsilon(self.rms_norm_eps).init(device),
        };
        // Qwen3-MoE is always UNTIED: a separate lm_head.weight is always present.
        let lm_head = LinearConfig::new(self.hidden_size, self.vocab_size).with_bias(false).init(device);
        Qwen3MoeForCausalLM { model, lm_head, infer_precision: Ignored(Precision::F32) }
    }

    fn init_layer<B: Backend>(&self, device: &B::Device) -> Qwen3MoeDecoderLayer<B> {
        Qwen3MoeDecoderLayer {
            self_attn: Qwen3AttentionConfig::new(
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
            )
            .with_head_dim(Some(self.get_head_dim()))
            .with_rope_theta(self.rope_theta)
            .with_rms_norm_eps(self.rms_norm_eps)
            .init(device),
            mlp: Qwen3MoeSparseBlock::new(
                self.hidden_size,
                self.moe_intermediate_size,
                self.num_experts,
                self.num_experts_per_tok,
                self.norm_topk_prob,
                device,
            ),
            input_layernorm: RmsNormConfig::new(self.hidden_size).with_epsilon(self.rms_norm_eps).init(device),
            post_attention_layernorm: RmsNormConfig::new(self.hidden_size).with_epsilon(self.rms_norm_eps).init(device),
        }
    }
}

/// A sparse MoE block: a router (`gate`) + the `num_experts` SwiGLU experts stored as ONE contiguous
/// `[E,..]` stack per projection (the vLLM `FusedMoE` `w13_weight`/`w2_weight` layout). The stacks are
/// the SINGLE OWNER of the expert weights — there is no per-expert `Linear` and no `cat`-built copy, so
/// loading the 30B holds exactly one resident copy (see `docs/WAVE2_STATIC_DECODE.md`, fix (b)). The
/// router HF key `mlp.gate.weight` loads declaratively; the per-expert HF keys
/// `mlp.experts.{j}.{gate,up,down}_proj.weight` are slice-written into slot `j` of the stacks by the
/// custom loader (`Qwen3MoeForCausalLM::load_weights_sharded`, the Burn analogue of vLLM's
/// `param.data[expert_id].copy_(shard)`).
#[derive(Module, Debug)]
pub struct Qwen3MoeSparseBlock<B: Backend> {
    /// Router: `Linear(hidden -> num_experts)`, bias-free. HF key `mlp.gate.weight`.
    gate: Linear<B>,
    /// `[E, H, I]` — stacked `gate_proj` weights (Burn `Linear` layout `[d_in=H, d_out=I]`). One of the
    /// three SINGLE-OWNER expert stacks. Lazy at `init`; replaced slot-wise by the loader.
    gate_stack: Param<Tensor<B, 3>>,
    /// `[E, H, I]` — stacked `up_proj` weights.
    up_stack: Param<Tensor<B, 3>>,
    /// `[E, I, H]` — stacked `down_proj` weights (Burn `Linear` layout `[d_in=I, d_out=H]`).
    down_stack: Param<Tensor<B, 3>>,
    num_experts: Ignored<usize>,
    top_k: Ignored<usize>,
    norm_topk_prob: Ignored<bool>,
    hidden_size: Ignored<usize>,
    moe_intermediate_size: Ignored<usize>,
}

impl<B: Backend> Qwen3MoeSparseBlock<B> {
    /// Number of experts `E`. Exposed for the dropless grouped-GEMM fast path (`moe_grouped`).
    pub fn num_experts(&self) -> usize {
        *self.num_experts
    }

    /// Experts per token `top_k`. Exposed for the dropless grouped-GEMM fast path (`moe_grouped`).
    pub fn top_k(&self) -> usize {
        *self.top_k
    }

    /// Hidden size `H`.
    pub fn hidden_size(&self) -> usize {
        *self.hidden_size
    }

    /// Per-expert SwiGLU inner dim `I`.
    pub fn inner_size(&self) -> usize {
        *self.moe_intermediate_size
    }

    /// Stacked bias-free expert weights `(gate [E,H,I], up [E,H,I], down [E,I,H])` for the dropless
    /// grouped-GEMM fast path (`moe_grouped`). These are the block's OWN stacked params — the returned
    /// tensors SHARE the resident buffers (Burn `Tensor::clone`/`Param::val` are refcounted handles, not
    /// copies), so there is NO duplicate allocation (unlike the old per-call `cat`).
    pub fn stacked_experts_pub(&self) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        self.stacked_experts()
    }

    /// REPLACE the three expert stacks with the loaded contiguous tensors (the loader's slot-write
    /// result). `gate`/`up` must be `[E,H,I]` and `down` `[E,I,H]`. Used by the custom MoE loader.
    pub(crate) fn load_expert_stacks(&mut self, gate: Tensor<B, 3>, up: Tensor<B, 3>, down: Tensor<B, 3>) {
        let e = *self.num_experts;
        let h = *self.hidden_size;
        let i = *self.moe_intermediate_size;
        assert_eq!(gate.dims(), [e, h, i], "gate_stack must be [E,H,I]=[{e},{h},{i}]");
        assert_eq!(up.dims(), [e, h, i], "up_stack must be [E,H,I]=[{e},{h},{i}]");
        assert_eq!(down.dims(), [e, i, h], "down_stack must be [E,I,H]=[{e},{i},{h}]");
        self.gate_stack = Param::initialized(ParamId::new(), gate);
        self.up_stack = Param::initialized(ParamId::new(), up);
        self.down_stack = Param::initialized(ParamId::new(), down);
    }

    pub fn new(
        hidden_size: usize,
        moe_intermediate_size: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
        device: &B::Device,
    ) -> Self {
        assert!(top_k >= 1 && top_k <= num_experts, "top_k ({top_k}) must be in 1..=num_experts ({num_experts})");
        Qwen3MoeSparseBlock {
            gate: LinearConfig::new(hidden_size, num_experts).with_bias(false).init(device),
            gate_stack: lazy_expert_stack([num_experts, hidden_size, moe_intermediate_size], device),
            up_stack: lazy_expert_stack([num_experts, hidden_size, moe_intermediate_size], device),
            down_stack: lazy_expert_stack([num_experts, moe_intermediate_size, hidden_size], device),
            num_experts: Ignored(num_experts),
            top_k: Ignored(top_k),
            norm_topk_prob: Ignored(norm_topk_prob),
            hidden_size: Ignored(hidden_size),
            moe_intermediate_size: Ignored(moe_intermediate_size),
        }
    }

    /// Single-expert SwiGLU from the stacks: `down_e( silu(x·gate_e) * (x·up_e) )`, slicing expert `ei`
    /// out of the `[E,..]` stacks (`select`-equivalent slab read) and running the batch-safe 2-D GEMMs.
    /// Numerically identical to the old per-expert `Qwen3MLP::forward` (same bias-free weights, same
    /// `linear3` 2-D-flatten that dodges the sm_121 broadcast-matmul bug). The eager paths
    /// ([`forward_oracle`](Self::forward_oracle)/[`forward_routed`](Self::forward_routed)) call this.
    fn expert_forward(&self, ei: usize, x: Tensor<B, 3>, prec: Precision) -> Tensor<B, 3> {
        let [b, s, h] = x.dims();
        let t = b * s;
        let i = *self.moe_intermediate_size;
        let g_w = self.gate_stack.val().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]); // [H,I]
        let u_w = self.up_stack.val().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]); // [H,I]
        let d_w = self.down_stack.val().slice([ei..ei + 1, 0..i, 0..h]).reshape([i, h]); // [I,H]
        let x2 = x.reshape([t, h]); // [T,H]
        let gate = silu(matmul2(x2.clone(), g_w, prec)); // [T,I]
        let up = matmul2(x2, u_w, prec); // [T,I]
        let y = matmul2(gate * up, d_w, prec); // [T,H]
        y.reshape([b, s, h])
    }

    /// Routing only. Returns `(router_logits [T,E], gate_weights [T,E])`, where `gate_weights[t,e]`
    /// is the renormalized softmax weight of expert `e` for token `t` (0 if `e` is not in the token's
    /// top-k). Exposed for tests/debugging (Gate 1a parity vs a host-side reference).
    ///
    /// Top-k is taken by ITERATED ARGMAX (no host-sync `topk`): pick the max, record its prob, then
    /// push it to -inf via a scatter-Add so the next round skips it. `renorm(softmax_E top-k)` is
    /// exactly HF's path, and iterated-argmax on the (monotone) softmax returns HF's top-k set+values.
    pub fn route(&self, x: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let [b, s, _h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let device = x.device();

        // Router GEMM. `Precision::F32` sets the GEMM COMPUTE precision; the OUTPUT dtype is the
        // backend `FloatElem`. Cast to F32 explicitly before softmax so the fp32 router (HF:
        // `F.softmax(.., dtype=float32)`) holds even if a future backend's `FloatElem` is f16/bf16.
        let logits = linear3(&self.gate, x, Precision::F32).reshape([t, e]); // [T, E]
        let probs = softmax(logits.clone().cast(DType::F32), 1); // fp32 router softmax over the E experts
        let (sel_idx, sel_w) = self.topk_select(probs, t);

        // Dense per-(token,expert) gate matrix [T, E] via k scatter-Adds. The k picks are distinct
        // experts per token, so each cell is written once — Add is exact (no double-count).
        let mut gate_w = Tensor::<B, 2>::zeros([t, e], &device);
        for slot in 0..*self.top_k {
            gate_w = gate_w.scatter(1, sel_idx[slot].clone(), sel_w[slot].clone(), IndexingUpdateOp::Add);
        }
        (logits, gate_w)
    }

    /// Top-k routing in the COMPACT form for the gather fast path: `(sel_idx [T,k] Int, sel_w [T,k])`
    /// — each token's selected expert ids and their renormalized gate weights (the same selection as
    /// [`route`](Self::route), without materializing the dense `[T,E]` gate matrix).
    pub fn route_topk(&self, x: Tensor<B, 3>) -> (Tensor<B, 2, Int>, Tensor<B, 2>) {
        let [b, s, _h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let logits = linear3(&self.gate, x, Precision::F32).reshape([t, e]);
        let probs = softmax(logits.cast(DType::F32), 1);
        let (sel_idx, sel_w) = self.topk_select(probs, t);
        (Tensor::cat(sel_idx, 1), Tensor::cat(sel_w, 1)) // [T,k], [T,k]
    }

    /// Top-k by ITERATED ARGMAX over the fp32 router probs (no host-sync `topk`): pick the max, record
    /// its prob, push it to -inf via a scatter-Add so the next round skips it; then renormalize the kept
    /// k (`norm_topk_prob`). Returns the k per-slot `[T,1]` index/weight tensors. Shared by `route` and
    /// `route_topk`. (Equivalent to HF's softmax-over-E -> topk -> renorm; argmax over the monotone
    /// softmax returns HF's top-k set + weights.)
    fn topk_select(&self, probs: Tensor<B, 2>, t: usize) -> (Vec<Tensor<B, 2, Int>>, Vec<Tensor<B, 2>>) {
        let k = *self.top_k;
        let device = probs.device();
        let mut masked = probs;
        let mut sel_idx: Vec<Tensor<B, 2, Int>> = Vec::with_capacity(k); // each [T,1]
        let mut sel_w: Vec<Tensor<B, 2>> = Vec::with_capacity(k); // each [T,1]
        for _ in 0..k {
            let idx = masked.clone().argmax(1);
            let w = masked.clone().gather(1, idx.clone());
            let neg = Tensor::<B, 2>::full([t, 1], -1.0e30, &device);
            masked = masked.scatter(1, idx.clone(), neg, IndexingUpdateOp::Add);
            sel_idx.push(idx);
            sel_w.push(w);
        }
        if *self.norm_topk_prob {
            let mut wsum = sel_w[0].clone();
            for w in sel_w.iter().skip(1) {
                wsum = wsum + w.clone();
            }
            let wsum = wsum.clamp_min(1e-20);
            for w in sel_w.iter_mut() {
                *w = w.clone() / wsum.clone();
            }
        }
        (sel_idx, sel_w)
    }

    /// MoE block forward. Dispatches to the token-routing fast path ([`forward_routed`](Self::forward_routed))
    /// when env `QWEN3_MOE_ROUTED=1`, else the dense-masked oracle ([`forward_oracle`](Self::forward_oracle));
    /// the two are numerically identical (`forward_routed_equals_oracle`). The env toggle is for A/B
    /// benchmarking; production should choose the path explicitly.
    pub fn forward(&self, x: Tensor<B, 3>, prec: Precision) -> Tensor<B, 3> {
        if std::env::var("QWEN3_MOE_ONDEVICE").is_ok() {
            let t = x.dims()[0] * x.dims()[1];
            return self.forward_routed_ondevice(x, t); // C=T (exact); full-dense at T=1 decode
        }
        if std::env::var("QWEN3_MOE_ROUTED").is_ok() {
            self.forward_routed(x, prec)
        } else {
            self.forward_oracle(x, prec)
        }
    }

    /// Tier-1 ORACLE: route, then a dense-masked weighted SwiGLU sum (every expert over every token,
    /// weighted by its gate column). `E×dense` cost — correct and the numerical reference. The fast
    /// path is [`forward_routed`](Self::forward_routed). `prec` = expert GEMM precision.
    pub fn forward_oracle(&self, x: Tensor<B, 3>, prec: Precision) -> Tensor<B, 3> {
        let [b, s, h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let device = x.device();
        let dtype = x.dtype();
        // Routing runs in fp32 (route() casts the router logits to F32 before softmax). Cast the gate
        // weights back to the COMPUTE dtype before the combine — exactly HF's
        // `routing_weights.to(hidden_states.dtype)` after the renorm. Without this, a bf16-weighted
        // model crashes: bf16 expert output * f32 gate weight => DTypeMismatch (caught loading the real
        // 15B-A2B checkpoint, which the f32-only unit tests could not surface).
        let (_logits, gate_w) = self.route(x.clone());
        let gate_w = gate_w.cast(dtype);

        let mut acc = Tensor::<B, 3>::zeros([b, s, h], &device).cast(dtype);
        for ei in 0..e {
            let y_e = self.expert_forward(ei, x.clone(), prec); // [B, S, H] — slab ei of the stacks
            let w_e = gate_w.clone().slice([0..t, ei..ei + 1]).reshape([b, s, 1]); // [B, S, 1]
            acc = acc + y_e * w_e;
        }
        acc
    }

    /// Experimental stacked-expert BATCHED-GEMM forward: computes all experts as 3 batched matmuls
    /// (`[E,T,H]@[E,H,I]` ...) instead of 128×3 per-expert 2-D GEMMs. Numerically equal to the oracle
    /// [`forward`](Self::forward) — pinned by `forward_fast_equals_oracle`.
    ///
    /// HONEST PERF NOTE (`examples/moe_probe.rs`, GB10/sm_121): a true per-expert-batched matmul is
    /// bit-exact and ~10× faster than the looped 2-D GEMMs *in isolation*, BUT this full-pipeline form
    /// is NOT a speedup (~0.8-0.9×): to dodge the buggy *broadcast* matmul we must `repeat` x into
    /// `[E,T,H]` (E copies), and that materialization + CubeCL's batched-GEMV handling negate the win.
    /// It also stacks the expert weights per call (huge copies on the real model). So this is a
    /// reference/experiment, NOT the production fast path — the real B2 win needs token-routing (gather
    /// to run only the top-k experts, ~16× fewer FLOPs) or a custom fused kernel (docs/MOE_PLAN.md §4b).
    pub fn forward_fast(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let dtype = x.dtype();
        let (_logits, gate_w) = self.route(x.clone());
        let gate_w = gate_w.cast(dtype); // [T, E]

        // The bias-free expert weights ARE the resident stacks now (no per-call re-stack / `cat`):
        // gate/up `[E,H,I]`, down `[E,I,H]`. `val()` returns refcounted handles, not copies.
        let (gate_stack, up_stack, down_stack) = self.stacked_experts();

        // Broadcast tokens to every expert and run the batched SwiGLU.
        let xe = x.reshape([t, h]).unsqueeze::<3>().repeat(&[e, 1, 1]); // [E,T,H]
        let g = silu(xe.clone().matmul(gate_stack)); // [E,T,I]
        let u = xe.matmul(up_stack); // [E,T,I]
        let y = (g * u).matmul(down_stack); // [E,T,H]

        // Combine: out[t] = Σ_e gate_w[t,e] · y[e,t].
        let gw = gate_w.transpose().reshape([e, t, 1]); // [E,T,1]
        (y * gw).sum_dim(0).reshape([b, s, h])
    }

    /// FAST forward via TOKEN ROUTING: computes ONLY the top-k experts per token (≈ E/k = 16× fewer
    /// expert FLOPs and 16× fewer GEMM launches than the dense-masked oracle). The candle/Mixtral
    /// `index_select` + `index_add` pattern: route → read the top-k expert ids+weights to host → group
    /// tokens by expert → for each TOUCHED expert, `select` its token rows, run the expert on only
    /// those, and `select_assign(Add)` the gate-weighted output back to token positions. A token routed
    /// to k experts accumulates k scatter-Adds (correct, since its experts are distinct). Numerically
    /// equal to [`forward`](Self::forward) — pinned by `forward_routed_equals_oracle`.
    ///
    /// One small host read of the routing per call (`T*k` ids+weights). Cheap for decode (T small); for
    /// large-T prefill the host grouping grows, and a fully on-device index build (one_hot + cumsum)
    /// is the further step. Expert GEMMs use the batch-safe 2-D `linear3` (no batched/broadcast matmul,
    /// so it dodges the sm_121 corruption bug — unlike `forward_fast`).
    pub fn forward_routed(&self, x: Tensor<B, 3>, prec: Precision) -> Tensor<B, 3> {
        let [b, s, h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let k = *self.top_k;
        let device = x.device();
        let dtype = x.dtype();

        // Route, then read the compact top-k to host and group token indices (+ weights) by expert.
        let (sel_idx, sel_w) = self.route_topk(x.clone()); // [T,k] Int, [T,k] f32
        let idx_host: Vec<i64> = sel_idx.cast(DType::I64).into_data().to_vec().unwrap(); // T*k, row-major
        let w_host: Vec<f32> = sel_w.into_data().to_vec().unwrap(); // T*k
        let mut by_expert: Vec<(Vec<i64>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); e];
        for ti in 0..t {
            for slot in 0..k {
                let ei = idx_host[ti * k + slot] as usize;
                by_expert[ei].0.push(ti as i64);
                by_expert[ei].1.push(w_host[ti * k + slot]);
            }
        }

        // Run ONLY the touched experts, each on only its routed tokens; scatter-add the result back.
        let x2 = x.reshape([t, h]); // [T, H]
        let mut acc = Tensor::<B, 2>::zeros([t, h], &device).cast(dtype); // [T, H]
        for ei in 0..e {
            let (toks, ws) = &by_expert[ei];
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let idx_t = Tensor::<B, 1, Int>::from_data(toks.as_slice(), &device); // [n]
            let w_t = Tensor::<B, 1>::from_data(ws.as_slice(), &device).reshape([n, 1]).cast(dtype); // [n,1]
            let xe = x2.clone().select(0, idx_t.clone()); // [n, H] — gather this expert's tokens
            let ye = self.expert_forward(ei, xe.reshape([n, 1, h]), prec).reshape([n, h]); // [n, H]
            acc = acc.select_assign(0, idx_t, ye * w_t, IndexingUpdateOp::Add); // scatter-add weighted
        }
        acc.reshape([b, s, h])
    }

    /// The bias-free expert weight stacks `(gate [E,H,I], up [E,H,I], down [E,I,H])`. These ARE the
    /// block's resident single-owner params (vLLM `FusedMoE` layout); `Param::val` returns refcounted
    /// handles that SHARE the buffers — NO `cat`, NO copy (the old per-call `cat` was the ~58 GB
    /// duplicate that OOM'd the 30B; see `docs/WAVE2_STATIC_DECODE.md`).
    fn stacked_experts(&self) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        (self.gate_stack.val(), self.up_stack.val(), self.down_stack.val())
    }

    /// FULLY ON-DEVICE token-routing fast path (no host sync, fixed shapes → CUDA-graph-friendly).
    /// Builds the per-expert dispatch entirely with device ops — NOT `Tensor::one_hot` (which does an
    /// internal host round-trip); instead an `arange == expert` mask + `cumsum` segmented prefix-sum
    /// gives each (token,expert) assignment its within-expert position, scattered into a fixed
    /// capacity-`C` `[E,C]` buffer; then a stacked batched SwiGLU `[E,C,H]@[E,H,I]` and a scatter-add
    /// combine. Numerically equal to the oracle when `C ≥ max per-expert count` — pinned by
    /// `forward_routed_ondevice_equals_oracle` at `C=T` (where overflow is impossible).
    ///
    /// PERF (regime-dependent, benchmark — never assert): for the BATCHED GRPO-rollout decode shape
    /// (T = P·G tokens/step) with `C ≈ k·T/E` this computes `E·C ≈ k·T` FFNs → toward the `E/k = 16×`
    /// FLOP win (~9× at T=256, ~14× at T≥16k as the capacity slack shrinks). For single-stream `T=1`,
    /// `C=1` makes it `[E,1,H]` = full dense (no FLOP win) → prefer [`forward_routed`](Self::forward_routed)
    /// there. Exact-no-drop + compact + fixed-shape together need a custom grouped-GEMM kernel; this is
    /// the standard-op stand-in (`C<T` is probabilistic-no-drop — guard overflow for GRPO correctness).
    pub fn forward_routed_ondevice(&self, x: Tensor<B, 3>, capacity: usize) -> Tensor<B, 3> {
        let [b, s, h] = x.dims();
        let t = b * s;
        let e = *self.num_experts;
        let k = *self.top_k;
        let c = capacity.max(1); // C=0 would make the buffers degenerate; never drop below 1
        let n = t * k;
        let device = x.device();
        let dtype = x.dtype();

        let (sel_idx, sel_w) = self.route_topk(x.clone()); // [T,k] Int, [T,k] f32
        let assign_e = sel_idx.reshape([n]); // [N] expert id per assignment
        let assign_tok = Tensor::<B, 1, Int>::arange(0..t as i64, &device).reshape([t, 1]).repeat(&[1, k]).reshape([n]); // [N] token per assignment
        let assign_w = sel_w.reshape([n]); // [N] f32

        // ON-DEVICE one-hot via (arange == expert) — NOT Tensor::one_hot (host round-trip).
        let experts_row = Tensor::<B, 1, Int>::arange(0..e as i64, &device).reshape([1, e]); // [1,E]
        let oh = assign_e.clone().reshape([n, 1]).equal(experts_row).int(); // [N,E] 0/1
        // within-expert rank = inclusive cumsum down N, read at own expert, minus 1.
        let run = oh.cumsum(0); // [N,E]
        let pos = run.gather(1, assign_e.clone().reshape([n, 1])).reshape([n]).add_scalar(-1i64); // [N], 0-indexed
        // dest slot = expert*C + pos; overflow (pos>=C) → dummy slot E*C (sliced off later).
        let over = pos.clone().greater_equal_elem(c as i64); // [N] Bool
        let dest = (assign_e.mul_scalar(c as i64) + pos).mask_fill(over, (e * c) as i64); // [N]

        // Build the [E*C+1] dispatch buffers via scatter-Add into zeros (real dests are unique → Add ==
        // assign). Store token+1 so an UNWRITTEN (empty) slot reads 0 and is distinguishable from token 0
        // after a -1; the dummy slot E*C absorbs overflow assignments and is sliced off.
        let tok_buf = Tensor::<B, 1, Int>::zeros([e * c + 1], &device).select_assign(0, dest.clone(), assign_tok.add_scalar(1i64), IndexingUpdateOp::Add);
        let w_buf = Tensor::<B, 1>::zeros([e * c + 1], &device).select_assign(0, dest, assign_w, IndexingUpdateOp::Add);
        // Two kinds of non-real slots, handled differently (Gemini review #2): OVERFLOW assignments
        // (pos≥C) were routed to dest=E*C and are SEVERED here by the [0..E*C] slice — never gathered or
        // computed. EMPTY slots (a real [0,E*C) slot no assignment filled) read 0 → −1 below.
        let tok_raw = tok_buf.slice([0..e * c]).add_scalar(-1i64); // [E*C] real=token, empty=−1
        // Empty slots → an appended ZERO row at index T (not a real token): SwiGLU(0)=0 is finite, so
        // there is no `0*NaN` hole even if a real token's activation were non-finite (Codex review #5).
        let tokens = tok_raw.clone().mask_fill(tok_raw.lower_elem(0i64), t as i64); // [E*C]
        let weights = w_buf.slice([0..e * c]).reshape([e, c, 1]).cast(dtype); // [E,C,1] (empty/dropped = 0)

        // Gather token rows (x plus the appended zero row) → [E,C,H], stacked batched SwiGLU, weight, scatter back.
        let (gate_stack, up_stack, down_stack) = self.stacked_experts();
        let x_pad = Tensor::cat(vec![x.reshape([t, h]), Tensor::<B, 2>::zeros([1, h], &device).cast(dtype)], 0); // [T+1,H]
        let xe = x_pad.select(0, tokens.clone()).reshape([e, c, h]); // [E,C,H]
        let g = silu(xe.clone().matmul(gate_stack)); // [E,C,I]
        let u = xe.matmul(up_stack); // [E,C,I]
        let y = (g * u).matmul(down_stack); // [E,C,H]
        let y_w = (y * weights).reshape([e * c, h]); // [E*C,H] (empty/dropped: zero row × weight 0 = 0)
        // acc has an extra dummy row [T] absorbing empty/dropped slots' (zero) contributions; sliced off.
        let acc = Tensor::<B, 2>::zeros([t + 1, h], &device).cast(dtype).select_assign(0, tokens, y_w, IndexingUpdateOp::Add);
        acc.slice([0..t]).reshape([b, s, h])
    }
}

/// One Qwen3-MoE decoder layer: attention (reused) + a sparse MoE block, pre-norm + residuals.
#[derive(Module, Debug)]
pub struct Qwen3MoeDecoderLayer<B: Backend> {
    self_attn: Qwen3Attention<B>,
    mlp: Qwen3MoeSparseBlock<B>,
    input_layernorm: RmsNorm<B>,
    post_attention_layernorm: RmsNorm<B>,
}

impl<B: Backend> Qwen3MoeDecoderLayer<B> {
    fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward(hidden_states, attention_mask, position_ids, prec);
        let hidden_states = residual + hidden_states;

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
        cache: &mut crate::cache::KVCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache(hidden_states, attention_mask, position_ids, cache, prec);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    /// CUDA-graph-CAPTURABLE fixed-shape decode layer (WAVE-2 STEP 1, docs/PERF_80TOKS_PLAN.md §2/§5):
    /// the MoE counterpart of the dense [`crate::decoder`]'s `forward_with_cache_static_pre_lp`. Attention
    /// is the SAME GQA, so the dense static-decode attention path is REUSED VERBATIM
    /// ([`Qwen3Attention::forward_with_cache_static_pre`]: full-`T_max` masked attention, device-`pos` KV
    /// write, precomputed RoPE freqs + arange). Only the MLP differs: instead of the legacy
    /// `Qwen3MoeSparseBlock::forward` (oracle/routed/ondevice — all of which either host-sync every layer or
    /// re-stack all `E` experts at `T=1`), it runs the pre-stacked top-`k` gather decode
    /// ([`MoeExpertCache::decode_topk`], Block A) which reads ONLY the `k` routed experts' weight slabs and
    /// has NO host sync. Fixed-shape `[B,1,H]` in/out; every per-step index comes from the `[1]` Int DEVICE
    /// counter `pos`. `expert_cache` is the per-layer cache built ONCE post-load (see [`MoeStaticDecode`]).
    #[allow(clippy::too_many_arguments)]
    fn forward_with_cache_static_pre(
        &self,
        hidden_states: Tensor<B, 3>,
        pos: Tensor<B, 1, Int>,
        cache: &mut crate::cache::KVCache<B>,
        expert_cache: &MoeExpertCache<B>,
        prec: Precision,
        freqs: &Tensor<B, 1>,
        arange_tmax: &Tensor<B, 1, Int>,
        assign_tok: &Tensor<B, 1, Int>,
        decode_fn: MoeDecodeFn<B>,
    ) -> Tensor<B, 3> {
        // Self-attention: the dense static decode path, reused unchanged (same GQA + QK-norm + RoPE).
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states =
            self.self_attn.forward_with_cache_static_pre(hidden_states, pos, cache, prec, freqs, arange_tmax);
        let hidden_states = residual + hidden_states;

        // Sparse MoE block: the capturable, host-sync-free top-k decode. `decode_fn` selects the
        // materializing oracle (`decode_topk_pre`) or the FUSED gather-GEMV (`decode_topk_fused`, lever
        // (c)); both route on-device + touch only the k routed experts, with the per-assignment token
        // index HOISTED (no per-call `arange` H2D) so the step stays CUDA-graph capturable.
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = decode_fn(expert_cache, &self.mlp, hidden_states, prec, assign_tok);
        residual + hidden_states
    }
}

/// The Qwen3-MoE base transformer (embedding + sparse layers + final norm).
#[derive(Module, Debug)]
pub struct Qwen3MoeModel<B: Backend> {
    config: Ignored<Qwen3MoeConfig>,
    pub(crate) embed_tokens: Embedding<B>,
    layers: Vec<Qwen3MoeDecoderLayer<B>>,
    norm: RmsNorm<B>,
}

impl<B: Backend> Qwen3MoeModel<B> {
    /// Last decoder-layer output (no final norm), default RoPE positions.
    fn forward(&self, input_ids: Tensor<B, 2, Int>, attention_mask: Option<Tensor<B, 2, Bool>>, prec: Precision) -> Tensor<B, 3> {
        let [batch, seq] = input_ids.dims();
        let device = input_ids.device();
        let position_ids = Tensor::<B, 1, Int>::arange(0..seq as i64, &device).unsqueeze_dim::<2>(0).repeat(&[batch, 1]);
        self.forward_with_positions(input_ids, attention_mask, position_ids, prec)
    }

    /// Last decoder-layer output (no final norm), EXPLICIT RoPE positions.
    fn forward_with_positions(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for layer in self.layers.iter() {
            hidden_states = layer.forward(hidden_states, attention_mask.clone(), position_ids.clone(), prec);
        }
        hidden_states
    }

    /// Cached decode: applies the final norm (mirrors the dense `forward_with_cache`).
    fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        let mut hidden_states = self.embed_tokens.forward(input_ids);
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden_states = layer.forward_with_cache(hidden_states, attention_mask.clone(), position_ids.clone(), layer_cache, prec);
        }
        self.norm.forward(hidden_states)
    }

    /// A fresh KV cache for this model.
    pub fn new_cache(&self) -> ModelCache<B> {
        ModelCache::new(self.layers.len())
    }

    /// A STATIC pre-allocated KV cache (capacity = `prompt_len + max_new_tokens`) for the device-`pos`
    /// fixed-shape decode (mirrors the dense `new_cache_with_capacity`).
    pub fn new_cache_with_capacity(&self, capacity: usize) -> ModelCache<B> {
        ModelCache::with_capacity(self.layers.len(), capacity)
    }

    /// Build the per-layer pre-stacked expert-weight caches ONCE (one [`MoeExpertCache::from_block`] per
    /// MoE layer) for the static top-k decode. MUST be called AFTER weight load — `from_block` borrows
    /// the (loaded) expert stacks as refcounted handles (no copy). Never rebuilt per step.
    pub fn build_expert_caches(&self) -> Vec<MoeExpertCache<B>> {
        self.layers.iter().map(|l| MoeExpertCache::from_block(&l.mlp)).collect()
    }

    /// `(num_layers, num_experts E, hidden H, moe_intermediate I)` — the expert-stack geometry the
    /// custom MoE loader needs to pre-size and slot-fill the `[E,..]` stacks. (`pub(crate)` for `load`.)
    pub(crate) fn expert_layout(&self) -> (usize, usize, usize, usize) {
        let cfg = &self.config.0;
        (self.layers.len(), cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size)
    }

    /// The device the model lives on (without materializing any param — lazy device of the embedding).
    pub(crate) fn device(&self) -> B::Device {
        self.embed_tokens.weight.lazy_device()
    }

    /// REPLACE layer `l`'s three expert stacks with the loaded contiguous tensors (the loader's
    /// slot-write result). The Burn analogue of vLLM's `param.data[expert_id].copy_` finalized into the
    /// single-owner `[E,..]` params. (`pub(crate)` for the custom loader in `load.rs`.)
    pub(crate) fn set_layer_expert_stacks(
        &mut self,
        l: usize,
        gate: Tensor<B, 3>,
        up: Tensor<B, 3>,
        down: Tensor<B, 3>,
    ) {
        self.layers[l].mlp.load_expert_stacks(gate, up, down);
    }

    /// Fixed-shape, device-`pos`-indexed, HOST-SYNC-FREE decode forward (WAVE-2 STEP 1). Mirrors the dense
    /// `Qwen3Model::forward_with_cache_static_pre`: ONE token `[B,1]` in → final hidden `[B,1,H]` out, every
    /// per-step op fixed-shape and indexed by the `[1]` Int DEVICE counter `pos`. Each layer runs the
    /// reused dense static attention + Block A's [`MoeExpertCache::decode_topk`]. The final norm IS applied
    /// (like `forward_with_cache`). `sd` holds the per-layer expert caches + precomputed RoPE freqs +
    /// arange(T_max) (built once via [`MoeStaticDecode::new`]).
    fn forward_with_cache_static_pre(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        sd: &MoeStaticDecode<B>,
        prec: Precision,
    ) -> Tensor<B, 3> {
        // MUST-FIX (Codex review): zip silently truncates to the SHORTEST iterator, so a short
        // `cache.layers` or `expert_caches` would skip trailing layers and return plausible-but-wrong
        // logits. Assert all three lengths match up front — a mismatch PANICS, never silently skips.
        assert_eq!(
            self.layers.len(),
            cache.layers.len(),
            "layer/KV-cache length mismatch: {} layers vs {} caches",
            self.layers.len(),
            cache.layers.len(),
        );
        assert_eq!(
            self.layers.len(),
            sd.expert_caches.len(),
            "layer/expert-cache length mismatch: {} layers vs {} expert caches (rebuild MoeStaticDecode)",
            self.layers.len(),
            sd.expert_caches.len(),
        );
        let mut hidden = self.embed_tokens.forward(input_ids);
        for ((layer, layer_cache), ec) in
            self.layers.iter().zip(cache.layers.iter_mut()).zip(sd.expert_caches.iter())
        {
            hidden = layer.forward_with_cache_static_pre(
                hidden,
                pos.clone(),
                layer_cache,
                ec,
                prec,
                &sd.freqs,
                &sd.arange_tmax,
                &sd.assign_tok,
                sd.decode_fn,
            );
        }
        self.norm.forward(hidden)
    }
}

/// Pre-built, "once post-load" companion for the static MoE decode (WAVE-2 STEP 1). Bundles the three
/// things the capturable fixed-shape decode needs precomputed ONCE (none per step):
///  * `expert_caches` — one [`MoeExpertCache`] per MoE layer (the pre-stacked contiguous expert weights;
///    the per-step decode gathers only the `k` routed slabs out of these — never re-stacks all `E`).
///  * `freqs` — the RoPE inverse-frequency table `[head_dim/2]` (hoisted out of the per-step attention so
///    the captured region contains no host->device `from_floats` staging).
///  * `arange_tmax` — `[T_max]` Int `0..capacity`, the attention position-mask index (hoisted out of the
///    per-step `Tensor::arange`).
///  * `assign_tok` — `[N=T*k]` Int per-assignment token index for [`MoeExpertCache::decode_topk_pre`]
///    (hoisted out of its per-call `Tensor::arange`, which would stage a host→device copy uncapturable
///    inside a CUDA graph). Built for the single-token decode (`T=1`) → `[0; k]`.
///
/// Build it for `capacity = prompt_len + max_new_tokens` (the static KV width); the decode driver asserts
/// the cache capacity matches. It is `Clone` but heavy (holds all expert weights' stacked views) — build
/// it ONCE and reuse across decode steps / generations.
#[derive(Debug, Clone)]
pub struct MoeStaticDecode<B: Backend> {
    expert_caches: Vec<MoeExpertCache<B>>,
    freqs: Tensor<B, 1>,
    arange_tmax: Tensor<B, 1, Int>,
    /// `[N=T*k]` per-assignment token index for the top-k gather/scatter (T=1 single-token decode).
    assign_tok: Tensor<B, 1, Int>,
    /// The per-layer MoE-decode kernel the static path dispatches to. Defaults to the materializing
    /// oracle [`MoeExpertCache::decode_topk_pre`]; switched to the FUSED gather-GEMV
    /// [`MoeExpertCache::decode_topk_fused`] (lever (c)) via [`MoeStaticDecode::with_fused`] (Cuda only).
    decode_fn: MoeDecodeFn<B>,
}

/// The per-layer MoE decode the static path runs. A plain fn pointer so the GENERIC static-decode loop
/// (`forward_with_cache_static_pre`) can dispatch the materializing oracle vs the fused gather-GEMV
/// without a backend-specialized clone of the whole forward. Both alternatives share this signature
/// (`&cache, &block, x, prec, &assign_tok -> y`): [`MoeExpertCache::decode_topk_pre`] and (Cuda only)
/// [`MoeExpertCache::decode_topk_fused`].
pub type MoeDecodeFn<B> = fn(
    &MoeExpertCache<B>,
    &Qwen3MoeSparseBlock<B>,
    Tensor<B, 3>,
    Precision,
    &Tensor<B, 1, Int>,
) -> Tensor<B, 3>;

impl<B: Backend> MoeStaticDecode<B> {
    /// The static KV width `T_max` this was built for (= `prompt_len + max_new_tokens`).
    pub fn capacity(&self) -> usize {
        self.arange_tmax.dims()[0]
    }

    /// Number of MoE layers (one expert cache each).
    pub fn num_layers(&self) -> usize {
        self.expert_caches.len()
    }
}

#[cfg(feature = "cuda")]
impl<B: FusedSwigluBackend> MoeStaticDecode<B> {
    /// Select the per-layer MoE decode kernel: `fused = true` ⇒ the FUSED gather-GEMV
    /// [`MoeExpertCache::decode_topk_fused`] (lever (c), reads each routed expert's weights ONCE from
    /// the persistent stacks — no `[N,H,I]` materialization); `false` ⇒ the materializing oracle
    /// [`MoeExpertCache::decode_topk_pre`] (the default `build_static_decode` already sets). Same
    /// fixed-shape, host-sync-free, capturable contract either way; only the SwiGLU kernel differs.
    pub fn with_fused(mut self, fused: bool) -> Self {
        if fused {
            self.decode_fn = |c, b, x, p, a| c.decode_topk_fused(b, x, p, a);
        }
        self
    }
}

/// Qwen3-MoE with an (always-untied) LM head for text generation.
#[derive(Module, Debug)]
pub struct Qwen3MoeForCausalLM<B: Backend> {
    pub model: Qwen3MoeModel<B>,
    /// Separate output head — Qwen3-MoE is always untied (`tie_word_embeddings = false`).
    lm_head: Linear<B>,
    /// Inference compute precision. Default `F32`.
    infer_precision: Ignored<Precision>,
}

impl<B: Backend> Qwen3MoeForCausalLM<B> {
    /// Logits `[B, S, vocab]` (no cache), default RoPE positions, at f32.
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>, attention_mask: Option<Tensor<B, 2, Bool>>) -> Tensor<B, 3> {
        let hidden = self.model.forward(input_ids, attention_mask, *self.infer_precision);
        let hidden = self.model.norm.forward(hidden);
        linear3(&self.lm_head, hidden, Precision::F32)
    }

    /// Logits with KV cache (the model already applies the final norm).
    pub fn forward_with_cache(
        &self,
        input_ids: Tensor<B, 2, Int>,
        attention_mask: Option<Tensor<B, 2, Bool>>,
        position_ids: Tensor<B, 2, Int>,
        cache: &mut ModelCache<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.model.forward_with_cache(input_ids, attention_mask, position_ids, cache, *self.infer_precision);
        linear3(&self.lm_head, hidden, Precision::F32)
    }

    /// Greedy KV-cached generation. Intended for a SINGLE sequence (`batch = 1`); the EOS check reads
    /// row 0 only (matching the dense model's own single-row EOS handling), so for `batch > 1` it stops
    /// the whole batch on row 0's EOS. Returns the full token sequence `[B, S+gen]`. (Sampling
    /// generation can be added by reusing `crate::sampling::sample_index` as the dense model does;
    /// greedy is the minimal path that exercises the full cached MoE forward.)
    pub fn generate_greedy(&self, input_ids: Tensor<B, 2, Int>, max_new_tokens: usize, eos_token_ids: &[i64]) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch, init_len] = input_ids.dims();
        let mut cache = self.model.new_cache();

        let pos = Tensor::<B, 1, Int>::arange(0..init_len as i64, &device).unsqueeze_dim::<2>(0).repeat(&[batch, 1]);
        let logits = self.forward_with_cache(input_ids.clone(), None, pos, &mut cache);
        let [_, _, vocab] = logits.dims();
        let mut next: Tensor<B, 1, Int> =
            logits.slice([0..batch, (init_len - 1)..init_len, 0..vocab]).reshape([batch, vocab]).argmax(1).flatten(0, 1);

        let first = next.clone().cast(DType::I64).into_data().as_slice::<i64>().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0);
        if eos_token_ids.contains(&first) {
            return input_ids;
        }
        let mut generated = Tensor::cat(vec![input_ids, next.clone().unsqueeze_dim(1)], 1);
        let mut cur = init_len;
        for _ in 1..max_new_tokens {
            cur += 1;
            let pos = Tensor::<B, 1, Int>::from_data([cur as i64 - 1], &device).unsqueeze_dim::<2>(0).repeat(&[batch, 1]);
            let logits = self.forward_with_cache(next.clone().unsqueeze_dim(1), None, pos, &mut cache);
            next = logits.slice([0..batch, 0..1, 0..vocab]).reshape([batch, vocab]).argmax(1).flatten(0, 1);
            let id = next.clone().cast(DType::I64).into_data().as_slice::<i64>().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0);
            generated = Tensor::cat(vec![generated, next.clone().unsqueeze_dim(1)], 1);
            if eos_token_ids.contains(&id) {
                break;
            }
        }
        generated
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.model.layers.len()
    }

    /// Build the "once post-load" [`MoeStaticDecode`] companion for the static decode at this `capacity`
    /// (= `prompt_len + max_new_tokens`, the static KV width). Builds one [`MoeExpertCache`] per layer + the
    /// RoPE freq table + `arange(capacity)`. MUST be called AFTER weight load. Reuse it across decode steps
    /// / generations of the same capacity; do NOT rebuild per step.
    pub fn build_static_decode(&self, capacity: usize) -> MoeStaticDecode<B> {
        let device = self.model.embed_tokens.weight.val().device();
        let cfg = &self.model.config.0;

        // MUST-FIX (Codex review): the static decode REUSES the dense static-attention path with a
        // RoPE table + arange precomputed HERE from the MoE config. That reuse is only correct if every
        // attention layer was actually built with the SAME head_dim, rope_theta, AND QK-norm dim (not
        // just "same GQA"). Assert config equality per layer so a future config divergence PANICS at
        // build time rather than silently decoding with the wrong RoPE/QK-norm geometry.
        let head_dim = cfg.get_head_dim();
        for (li, layer) in self.model.layers.iter().enumerate() {
            let attn = &layer.self_attn;
            assert_eq!(
                attn.head_dim(), head_dim,
                "layer {li}: attention head_dim {} != MoE config head_dim {head_dim} (RoPE table mismatch)",
                attn.head_dim(),
            );
            assert_eq!(
                attn.qk_norm_dim(), head_dim,
                "layer {li}: QK-norm dim {} != head_dim {head_dim} (QK-RMSNorm geometry mismatch)",
                attn.qk_norm_dim(),
            );
            assert!(
                (attn.rope_theta() - cfg.rope_theta).abs() < 1e-6,
                "layer {li}: attention rope_theta {} != MoE config rope_theta {} (RoPE freqs mismatch)",
                attn.rope_theta(), cfg.rope_theta,
            );
        }

        let freqs = rope_freqs::<B>(head_dim, cfg.rope_theta, &device);
        let arange_tmax = Tensor::<B, 1, Int>::arange(0..capacity as i64, &device);
        let expert_caches = self.model.build_expert_caches();
        // The per-layer expert cache count MUST equal the layer count (one cache per MoE layer) — a
        // short list would later silently skip layers in `forward_with_cache_static_pre`'s zip.
        assert_eq!(
            expert_caches.len(),
            self.model.layers.len(),
            "built {} expert caches for {} layers",
            expert_caches.len(),
            self.model.layers.len(),
        );
        // Hoist the per-assignment token index OUT of the per-step decode (its `arange` stages a
        // host→device copy that is uncapturable inside a CUDA graph). Built for the single-token decode
        // (T=1) the static driver runs → `[0; k]`. k comes from the (identical) per-layer expert caches.
        let k = expert_caches[0].top_k();
        let assign_tok = MoeExpertCache::<B>::assign_tok(1, k, &device);
        // Default to the materializing oracle decode; `with_fused(true)` (Cuda) swaps in lever (c).
        let decode_fn: MoeDecodeFn<B> = MoeExpertCache::<B>::decode_topk_pre;
        MoeStaticDecode { expert_caches, freqs, arange_tmax, assign_tok, decode_fn }
    }

    /// CUDA-graph-CAPTURABLE fixed-shape decode forward returning logits `[B,1,vocab]` (WAVE-2 STEP 1).
    /// The MoE sibling of the dense `Qwen3ForCausalLM::forward_with_cache_static_pre`: ONE token `[B,1]` in,
    /// every per-step op fixed-shape + indexed by the `[1]` Int DEVICE counter `pos`, NO host sync anywhere
    /// (no `into_data`/`to_vec`/`into_scalar`). Numerically equal to the `forward_with_cache` decode branch
    /// (same routing + combine via `decode_topk`; same masked attention). Reuses the pre-built `sd`.
    pub fn forward_with_cache_static_pre(
        &self,
        input_ids: Tensor<B, 2, Int>,
        pos: Tensor<B, 1, Int>,
        cache: &mut ModelCache<B>,
        sd: &MoeStaticDecode<B>,
    ) -> Tensor<B, 3> {
        let hidden =
            self.model.forward_with_cache_static_pre(input_ids, pos, cache, sd, *self.infer_precision);
        linear3(&self.lm_head, hidden, Precision::F32)
    }

    /// Greedy generation driven ENTIRELY through the fixed-shape static decode (WAVE-2 STEP 1). Same
    /// contract/shape as [`generate_greedy`] (single sequence intended; `[B, lp+max_new]` out), but the
    /// per-token decode runs [`Self::forward_with_cache_static_pre`] (Block A + reused static attention)
    /// over a device-`pos` counter and a STATIC KV cache.
    ///
    /// HOST-SYNC-FREE decode region (the capture prerequisite): the prompt is prefilled ONCE (eager,
    /// variable-shape — excluded from the capturable region), then the loop body has ZERO host reads —
    /// greedy argmax, EOS bookkeeping (a device `finished` Bool + pad-token emit), and the token write
    /// (`select_assign` at device `pos`) are all on-device. Like the dense `group_sample_cached_device_static`
    /// it runs the FULL `max_new_tokens` (no host-read early break) and pads finished rows, so the only
    /// device->host transfer is the single read of the final token buffer by the caller.
    pub fn generate_greedy_static(
        &self,
        input_ids: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        eos_token_ids: &[i64],
        sd: &MoeStaticDecode<B>,
    ) -> Tensor<B, 2, Int> {
        let device = input_ids.device();
        let [batch, lp] = input_ids.dims();
        let total = lp + max_new_tokens;
        assert_eq!(
            sd.capacity(),
            total,
            "MoeStaticDecode capacity {} != prompt_len + max_new_tokens {} (rebuild with build_static_decode(total))",
            sd.capacity(),
            total,
        );
        let mut cache = self.model.new_cache_with_capacity(total);

        // ---- device EOS state: `finished` [B,1] Bool (all-false) + a constant pad token [B,1] Int. ----
        let eos0 = eos_token_ids.first().copied().unwrap_or(0);
        let mut finished = Tensor::<B, 2, Int>::zeros([batch, 1], &device).equal_elem(1i64); // 0 != 1 ⇒ false
        let pad = Tensor::<B, 2, Int>::full([batch, 1], eos0, &device);

        // ---- fixed-shape token buffer [B, total]: prompt written ONCE; completion scattered at col `pos`. ----
        let mut tok_buf =
            Tensor::<B, 2, Int>::zeros([batch, total], &device).slice_assign([0..batch, 0..lp], input_ids.clone());

        // ---- prefill (eager, variable-shape — NOT part of the capturable decode region) → first logits. ----
        let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, &device).unsqueeze_dim::<2>(0).repeat(&[batch, 1]);
        let logits = self.forward_with_cache(input_ids, None, pos0, &mut cache); // [B, lp, v]
        let v = logits.dims()[2];
        let mut last = logits.slice([0..batch, (lp - 1)..lp, 0..v]).reshape([batch, v]);

        // ---- device position counter: starts at `lp` (absolute position of completion token 0). ----
        let mut pos = Tensor::<B, 1, Int>::full([1], lp as i64, &device);

        for _ in 0..max_new_tokens {
            // greedy argmax (device); finished rows emit pad.
            let sampled = last.clone().argmax(1); // [B,1] Int
            let emit = sampled.mask_where(finished.clone(), pad.clone()); // pad where finished, else sampled

            // is_eos = OR over the eos set (empty set ⇒ all-false ⇒ never finishes / pads).
            let mut is_eos = Tensor::<B, 2, Int>::zeros([batch, 1], &device).equal_elem(1i64); // all false
            for &e in eos_token_ids {
                is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
            }

            // device-`pos` scatter into the fixed token buffer (zero-init completion region ⇒ Add == assign).
            tok_buf = tok_buf.select_assign(1, pos.clone(), emit.clone(), IndexingUpdateOp::Add);
            finished = finished.bool_or(is_eos);

            // decode the NEXT token through the STATIC path at device `pos` (uniform body; last step unused).
            let lg = self.forward_with_cache_static_pre(emit, pos.clone(), &mut cache, sd); // [B,1,v]
            last = lg.slice([0..batch, 0..1, 0..v]).reshape([batch, v]);

            pos = pos.add_scalar(1i64); // device add of constant 1 (never a host int)
        }
        tok_buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type B = burn::backend::NdArray;

    fn dev() -> <B as Backend>::Device {
        Default::default()
    }

    /// Gate 1a: the tensor router (fp32 softmax over E → iterated-argmax top-k → renorm → scatter)
    /// must match a plain-Rust reference (softmax → sort top-k → divide by their sum) on the SAME
    /// router logits. This locks the parity-critical routing math against a regression.
    #[test]
    fn route_matches_host_reference() {
        let device = dev();
        let (hidden, e, k) = (32usize, 8usize, 3usize);
        let (b, s) = (2usize, 5usize);
        let t = b * s;
        let block = Qwen3MoeSparseBlock::<B>::new(hidden, 16, e, k, true, &device);
        let x = Tensor::<B, 3>::random([b, s, hidden], Distribution::Normal(0.0, 1.0), &device);

        let (logits, gate_w) = block.route(x);
        let lv: Vec<f32> = logits.into_data().to_vec().unwrap(); // [T*E] row-major
        let gv: Vec<f32> = gate_w.into_data().to_vec().unwrap(); // [T*E]

        for ti in 0..t {
            let row = &lv[ti * e..(ti + 1) * e];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&l| (l - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&p| p / z).collect();
            let mut idx: Vec<usize> = (0..e).collect();
            idx.sort_by(|&a, &c| probs[c].partial_cmp(&probs[a]).unwrap());
            let sumk: f32 = idx[..k].iter().map(|&i| probs[i]).sum();
            let mut expect = vec![0f32; e];
            for &i in &idx[..k] {
                expect[i] = probs[i] / sumk;
            }
            for ei in 0..e {
                assert!(
                    (gv[ti * e + ei] - expect[ei]).abs() < 1e-4,
                    "token {ti} expert {ei}: got {} want {}",
                    gv[ti * e + ei],
                    expect[ei]
                );
            }
        }
    }

    /// Routing invariants: each token routes to EXACTLY top_k experts, and the kept weights sum to 1.
    #[test]
    fn gate_weights_invariants() {
        let device = dev();
        let (e, k) = (8usize, 2usize);
        let (b, s) = (3usize, 4usize);
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, e, k, true, &device);
        let x = Tensor::<B, 3>::random([b, s, 32], Distribution::Normal(0.0, 1.0), &device);
        let (_logits, gate_w) = block.route(x);
        let gv: Vec<f32> = gate_w.into_data().to_vec().unwrap();
        for ti in 0..(b * s) {
            let row = &gv[ti * e..(ti + 1) * e];
            let nz = row.iter().filter(|&&w| w > 0.0).count();
            assert_eq!(nz, k, "token {ti}: {nz} experts active, want {k}");
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "token {ti}: weights sum {sum}, want 1.0");
        }
    }

    /// The sparse block forward is shape-preserving and deterministic.
    #[test]
    fn block_forward_shape_and_deterministic() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, 8, 2, true, &device);
        let x = Tensor::<B, 3>::random([2, 4, 32], Distribution::Normal(0.0, 1.0), &device);
        let y1 = block.forward(x.clone(), Precision::F32);
        let y2 = block.forward(x, Precision::F32);
        assert_eq!(y1.dims(), [2, 4, 32]);
        let d: f32 = (y1 - y2).abs().sum().into_scalar();
        assert!(d < 1e-6, "forward not deterministic: |diff|={d}");
    }

    /// Combine numeric check (independent of forward's loop): with top_k=1 + renorm, each token routes
    /// to ONE expert at weight 1.0, so forward(x)[token] must equal that expert's output[token]. Catches
    /// a dropped weight, wrong expert, double-count, or a per-token mis-slice in the combine.
    #[test]
    fn forward_top1_equals_selected_expert() {
        let device = dev();
        let (hidden, e) = (32usize, 8usize);
        let (b, s) = (2usize, 4usize);
        let t = b * s;
        let block = Qwen3MoeSparseBlock::<B>::new(hidden, 16, e, 1, true, &device);
        let x = Tensor::<B, 3>::random([b, s, hidden], Distribution::Normal(0.0, 1.0), &device);
        let (_l, gate_w) = block.route(x.clone());
        let y = block.forward(x.clone(), Precision::F32);
        let gv: Vec<f32> = gate_w.into_data().to_vec().unwrap();
        let yv: Vec<f32> = y.into_data().to_vec().unwrap();
        let expert_outs: Vec<Vec<f32>> = (0..e)
            .map(|ei| block.expert_forward(ei, x.clone(), Precision::F32).into_data().to_vec().unwrap())
            .collect();
        for ti in 0..t {
            let sel = (0..e).max_by(|&a, &c| gv[ti * e + a].partial_cmp(&gv[ti * e + c]).unwrap()).unwrap();
            assert!((gv[ti * e + sel] - 1.0).abs() < 1e-5, "top-1 weight not 1.0");
            for hi in 0..hidden {
                let (got, want) = (yv[ti * hidden + hi], expert_outs[sel][ti * hidden + hi]);
                assert!((got - want).abs() < 1e-4, "token {ti} dim {hi}: forward {got} != expert {want}");
            }
        }
    }

    /// The norm_topk_prob=FALSE branch keeps the RAW top-k softmax probs (not divided by the kept sum).
    /// Exercises the otherwise-dead no-renorm path.
    #[test]
    fn no_renorm_keeps_raw_topk_probs() {
        let device = dev();
        let (e, k) = (8usize, 3usize);
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, e, k, false, &device);
        let x = Tensor::<B, 3>::random([1, 4, 32], Distribution::Normal(0.0, 1.0), &device);
        let (logits, gate_w) = block.route(x);
        let lv: Vec<f32> = logits.into_data().to_vec().unwrap();
        let gv: Vec<f32> = gate_w.into_data().to_vec().unwrap();
        for ti in 0..4 {
            let row = &lv[ti * e..(ti + 1) * e];
            let m = row.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = row.iter().map(|&l| (l - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            for ei in 0..e {
                let g = gv[ti * e + ei];
                if g > 0.0 {
                    let raw = exps[ei] / z;
                    assert!((g - raw).abs() < 1e-4, "no-renorm gate {g} != raw softmax prob {raw}");
                }
            }
        }
    }

    /// forward_with_cache (cached prefill) must match the no-cache forward logits on the same prompt —
    /// verifies the cached MoE decode path's numerics, not just shapes.
    #[test]
    fn cache_matches_no_cache_logits() {
        let device = dev();
        let model = Qwen3MoeConfig::tiny().init_causal_lm::<B>(&device);
        let ids = Tensor::<B, 2, Int>::from_data([[1i64, 5, 9, 3, 7]], &device);
        let no_cache = model.forward(ids.clone(), None);
        let mut cache = model.model.new_cache();
        let pos = Tensor::<B, 1, Int>::arange(0..5, &device).unsqueeze_dim::<2>(0);
        let cached = model.forward_with_cache(ids, None, pos, &mut cache);
        assert_eq!(no_cache.dims(), cached.dims());
        let d: f32 = (no_cache - cached).abs().mean().into_scalar();
        assert!(d < 1e-3, "cache vs no-cache logits diverge: mean|diff|={d}");
    }

    /// The stacked-expert fast path (`forward_fast`, batched GEMMs) is numerically equal to the
    /// dense-masked oracle (`forward`). Pins the fast path against the reference.
    #[test]
    fn forward_fast_equals_oracle() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, 8, 3, true, &device);
        let x = Tensor::<B, 3>::random([2, 4, 32], Distribution::Normal(0.0, 1.0), &device);
        let y_oracle = block.forward_oracle(x.clone(), Precision::F32);
        let y_fast = block.forward_fast(x);
        let d: f32 = (y_oracle - y_fast).abs().max().into_scalar();
        assert!(d < 1e-4, "forward_fast != oracle: max|diff|={d}");
    }

    /// The token-routing fast path (`forward_routed`, computes only top-k experts) is numerically
    /// equal to the dense-masked oracle (`forward`, all experts). Pins the gather/scatter combine.
    #[test]
    fn forward_routed_equals_oracle() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, 8, 3, true, &device);
        let x = Tensor::<B, 3>::random([2, 5, 32], Distribution::Normal(0.0, 1.0), &device);
        let y_oracle = block.forward_oracle(x.clone(), Precision::F32);
        let y_routed = block.forward_routed(x, Precision::F32);
        let d: f32 = (y_oracle - y_routed).abs().max().into_scalar();
        assert!(d < 1e-4, "forward_routed != oracle: max|diff|={d}");
    }

    /// The fully-on-device routed path (`forward_routed_ondevice`) at capacity C=T (no drop possible)
    /// equals the oracle. Pins the on-device dispatch (arange-onehot + cumsum + scatter) and the
    /// capacity-padded batched grouped SwiGLU + scatter-add combine.
    #[test]
    fn forward_routed_ondevice_equals_oracle() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, 8, 3, true, &device);
        let (b, s) = (2usize, 5usize);
        let t = b * s;
        let x = Tensor::<B, 3>::random([b, s, 32], Distribution::Normal(0.0, 1.0), &device);
        let y_oracle = block.forward_oracle(x.clone(), Precision::F32);
        let y_ondevice = block.forward_routed_ondevice(x, t); // C = T → exact, overflow impossible
        let d: f32 = (y_oracle - y_ondevice).abs().max().into_scalar();
        assert!(d < 1e-4, "forward_routed_ondevice != oracle: max|diff|={d}");
    }

    /// At a SMALL capacity (C ≪ T), experts overflow and assignments are dropped — the path must stay
    /// safe: finite output (no `0*NaN` from empty/dropped slots gathering the zero row), correct shape,
    /// no panic. Pins the drop-safe design (Codex review #5/#10). Exactness is only claimed at C=T.
    #[test]
    fn forward_routed_ondevice_capacity_drop_is_finite() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::<B>::new(32, 16, 8, 3, true, &device);
        let (b, s) = (2usize, 8usize); // T=16, top_k=3, E=8 → mean load 6; C=2 forces heavy drops
        let x = Tensor::<B, 3>::random([b, s, 32], Distribution::Normal(0.0, 1.0), &device);
        let y = block.forward_routed_ondevice(x, 2);
        assert_eq!(y.dims(), [b, s, 32]);
        // any NaN/Inf would make (y*0).sum() non-zero (0*NaN=NaN); finite ⇒ exactly 0.
        let finite: f32 = y.mul_scalar(0.0f32).sum().into_scalar();
        assert!(finite == 0.0, "capacity-drop path produced non-finite output: {finite}");
    }

    /// End-to-end: a tiny Qwen3-MoE causal LM forwards to vocab logits and generates greedily.
    #[test]
    fn causal_lm_forward_and_generate() {
        let device = dev();
        let cfg = Qwen3MoeConfig::tiny();
        let model = cfg.init_causal_lm::<B>(&device);
        assert_eq!(model.num_layers(), cfg.num_hidden_layers);

        let ids = Tensor::<B, 2, Int>::from_data([[1i64, 5, 9, 3]], &device);
        let logits = model.forward(ids.clone(), None);
        assert_eq!(logits.dims(), [1, 4, cfg.vocab_size]);

        // greedy generate with no EOS in the tiny vocab range we feed -> extends the sequence.
        let out = model.generate_greedy(ids, 6, &[]);
        let [bo, so] = out.dims();
        assert_eq!(bo, 1);
        assert!(so > 4 && so <= 4 + 6, "generated length {so} out of range");
    }

    /// WAVE-2 STEP 1 PARITY (tiny synthetic, CPU/NdArray, f32): the fixed-shape, device-`pos`, host-sync-free
    /// static decode (`generate_greedy_static`: Block A `decode_topk` + reused dense static attention) must
    /// produce GREEDY-TOKEN-IDENTICAL output to the EAGER `generate_greedy` (oracle MoE + growing-prefix
    /// attention) on the same prompt. This is the CPU half of the bf16-parity gate the Block-A review
    /// flagged; the CUDA/bf16 half is `examples/moe_static_decode.rs` on the real 30B.
    #[test]
    fn static_decode_matches_eager_greedy_tiny() {
        let device = dev();
        let cfg = Qwen3MoeConfig::tiny();
        let model = cfg.init_causal_lm::<B>(&device);

        let ids = Tensor::<B, 2, Int>::from_data([[1i64, 5, 9, 3, 7, 2]], &device);
        let lp = 6usize;
        let max_new = 12usize;

        // eos=[] so neither path early-stops → both produce [1, lp+max_new] for a total-length compare.
        let eager = model.generate_greedy(ids.clone(), max_new, &[]);
        let sd = model.build_static_decode(lp + max_new);
        let stat = model.generate_greedy_static(ids, max_new, &[], &sd);

        assert_eq!(eager.dims(), [1, lp + max_new]);
        assert_eq!(stat.dims(), [1, lp + max_new]);
        let ev: Vec<i64> = eager.cast(DType::I64).into_data().to_vec().unwrap();
        let sv: Vec<i64> = stat.cast(DType::I64).into_data().to_vec().unwrap();
        assert_eq!(ev, sv, "static decode greedy tokens != eager generate_greedy tokens\n eager={ev:?}\nstatic={sv:?}");
    }

    /// The single-step static logits forward equals the eager cached decode logits step-for-step (tiny,
    /// f32). Isolates `forward_with_cache_static_pre` numerics (decode_topk + masked static attention)
    /// from the greedy driver / argmax — a tighter check than token-identity.
    #[test]
    fn static_forward_matches_eager_cache_logits_tiny() {
        let device = dev();
        let cfg = Qwen3MoeConfig::tiny();
        let model = cfg.init_causal_lm::<B>(&device);
        let prompt = Tensor::<B, 2, Int>::from_data([[4i64, 1, 7, 2, 5]], &device);
        let lp = 5usize;
        let steps = 4usize;
        let total = lp + steps;

        // --- eager reference: prefill + growing-prefix cached decode, one fixed token sequence. ---
        let mut ec = model.model.new_cache();
        let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, &device).unsqueeze_dim::<2>(0);
        let l0 = model.forward_with_cache(prompt.clone(), None, pos0, &mut ec);
        let v = l0.dims()[2];
        // feed a fixed token stream (token id `i`) so both paths decode the SAME inputs.
        let feed: Vec<i64> = (0..steps as i64).map(|i| (i * 13 + 3) % cfg.vocab_size as i64).collect();
        let mut eager_logits: Vec<Vec<f32>> = Vec::new();
        for (t, &tok) in feed.iter().enumerate() {
            let tt = Tensor::<B, 2, Int>::from_data([[tok]], &device);
            let p = Tensor::<B, 1, Int>::from_data([(lp + t) as i64], &device).unsqueeze_dim::<2>(0);
            let lg = model.forward_with_cache(tt, None, p, &mut ec);
            eager_logits.push(lg.slice([0..1, 0..1, 0..v]).reshape([v]).into_data().to_vec().unwrap());
        }

        // --- static path: same prefill into a STATIC cache, then device-`pos` static forwards. ---
        let sd = model.build_static_decode(total);
        let mut sc = model.model.new_cache_with_capacity(total);
        let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, &device).unsqueeze_dim::<2>(0);
        let _ = model.forward_with_cache(prompt, None, pos0, &mut sc); // prefill cols 0..lp
        for (t, &tok) in feed.iter().enumerate() {
            let tt = Tensor::<B, 2, Int>::from_data([[tok]], &device);
            let pos = Tensor::<B, 1, Int>::full([1], (lp + t) as i64, &device);
            let lg = model.forward_with_cache_static_pre(tt, pos, &mut sc, &sd);
            let sv: Vec<f32> = lg.slice([0..1, 0..1, 0..v]).reshape([v]).into_data().to_vec().unwrap();
            let mut maxe = 0.0f32;
            for (a, b) in eager_logits[t].iter().zip(sv.iter()) {
                maxe = maxe.max((a - b).abs());
            }
            assert!(maxe < 1e-3, "step {t}: static logits diverge from eager cache, max|diff|={maxe}");
        }
    }
}
