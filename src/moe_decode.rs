//! CAPTURABLE top-8 MoE DECODE block — the keystone lever (A) of `docs/PERF_80TOKS_PLAN.md` §2-3.
//!
//! ## Why this exists (the council finding)
//! At single-token decode (`T=1`) the existing MoE forwards are all wrong for a *captured* path:
//!  * [`Qwen3MoeSparseBlock::forward_routed`](crate::Qwen3MoeSparseBlock::forward_routed) reads only the
//!    top-`k` experts ✓ but does a **device→host sync EVERY layer** (`sel_idx.into_data().to_vec()`,
//!    src/moe.rs) to group tokens on the host → **uncapturable** (48 syncs/token on the 30B).
//!  * [`forward_routed_ondevice`](crate::Qwen3MoeSparseBlock::forward_routed_ondevice) and
//!    [`forward_grouped`](crate::moe_grouped::forward_grouped) **re-stack ALL `E` experts every call**
//!    (and grouped re-casts to f32), so at `T=1` they touch ≥`E` experts' weights — *worse* than dense.
//!
//! This module builds the missing path: a **post-load pre-stacked contiguous expert-weight cache**
//! (built ONCE) + a **fixed-shape, no-host-sync, top-`k` single-token gather decode**. At `T=1` it reads
//! exactly the `k` routed experts' weight slabs (8 of 128), never the other 120.
//!
//! ## The design
//! ### 1. Pre-stacked contiguous cache ([`MoeExpertCache`])
//! Built once after load from the per-expert `Qwen3MLP` weights into three CONTIGUOUS batched tensors —
//! `gate [E,H,I]`, `up [E,H,I]`, `down [E,I,H]` (Burn `Linear` stores `[d_in, d_out]`, so gate/up are
//! `[H,I]` and down is `[I,H]`). `Tensor::cat` along a new leading axis allocates a fresh contiguous
//! buffer, so the result is gather-ready and is built ONCE, never per call (unlike `stacked_experts`,
//! which the existing fast paths rebuild every forward). Mirrors `moe_grouped`'s stack layout exactly so
//! the same cache could later feed the custom grouped-GEMM kernel.
//!
//! ### 2. Fixed-shape top-k gather decode ([`MoeExpertCache::decode_topk`])
//! Input `x:[B,S,H]` (decode: `S=1`, `T=B*S`). Steps, ALL on-device (no `into_data`, no `to_vec`):
//!   1. `route_topk(x)` → `sel_idx:[T,k]` (Int, the chosen expert ids) + `sel_w:[T,k]` (renormalized
//!      gate weights). This already exists and is host-sync-free (iterated-argmax, no `topk`).
//!   2. Flatten the `N = T*k` `(token, expert)` assignments. **GATHER** each assignment's expert weight
//!      slab from the cache along the `E` axis with `Tensor::select(0, expert_ids)`:
//!      `gate.select(0, ·) → [N,H,I]`, same for `up`, `down → [N,I,H]`. `select` is a device gather
//!      kernel — it reads ONLY the `N` indexed rows (at `T=1`, `N=k=8` slabs), never the full `E`.
//!   3. Gather each assignment's token row `x[token] → [N,1,H]` and run a **true batched** SwiGLU
//!      `silu([N,1,H]@[N,H,I]) * ([N,1,H]@[N,H,I]) → [N,1,I]`, then `@[N,I,H] → [N,1,H]`. Not a
//!      broadcast matmul (each batch element has its own weights, like `forward_fast`'s `[E,T,H]@[E,H,I]`),
//!      so it dodges the sm_121 broadcast-matmul corruption.
//!   4. Weight each row by its gate weight and **scatter-add** back to its token
//!      (`zeros[T,H].select_assign(0, token, y·w, Add)`). A token's `k` distinct experts accumulate by
//!      Add — exactly the oracle combine `out[t] = Σ_{e∈topk(t)} w_{t,e}·SwiGLU_e(x_t)`.
//!
//! Every shape is a function of `(T, k, H, I, E)` only — none depend on tensor VALUES — so for fixed `T`
//! (e.g. `T=1` single-stream or `T=B` batched decode) the whole block is fixed-shape and CUDA-graph
//! capturable. Numerically equal to [`forward_oracle`](crate::Qwen3MoeSparseBlock::forward_oracle) /
//! [`forward_routed`](crate::Qwen3MoeSparseBlock::forward_routed) — pinned by `decode_topk_equals_oracle`.
//!
//! ## Backend-generic
//! Pure Burn tensor ops (no custom kernel), so this is `B: Backend` generic and the unit tests run on
//! NdArray (CPU). On CUDA the batched matmul is a per-`(token,slot)` GEMV; the M=1 batched-GEMV shape is
//! the regime to profile on the real GB10 (Wave 2). The bf16 compute path is provided for completeness
//! but only the f32 path is parity-validated here.
//!
//! ## Status / Wave 2 (integration, NOT in this file)
//! This is the BLOCK. Wiring it into `Qwen3MoeForCausalLM` (a `forward_with_cache_static*` MoE decode +
//! holding one `MoeExpertCache` per layer + CUDA-graph capture) is the remaining port — see the return
//! notes and `docs/PERF_80TOKS_PLAN.md` §3 levers (D)/(E)/(F).

use burn::{
    prelude::Device,
    tensor::{DType, IndexingUpdateOp, Int, Tensor, activation::silu},
};

use crate::Qwen3MoeSparseBlock;
use crate::linear2d::Precision;

#[cfg(feature = "cuda")]
use crate::moe_grouped::FusedSwigluBackend;

/// Pre-stacked, contiguous, one-time expert-weight cache for the capturable top-k decode.
///
/// Holds the `E` experts' bias-free SwiGLU weights as three contiguous batched tensors, built ONCE from
/// a [`Qwen3MoeSparseBlock`] after load. The decode gathers `k` slabs out of these along the `E` axis;
/// the per-token path never re-stacks (the whole point — `stacked_experts` per-call copies hundreds of
/// MB on the 30B every forward).
#[derive(Debug, Clone)]
pub struct MoeExpertCache {
    /// `[E, H, I]` — stacked `gate_proj` weights (Burn `Linear` `[d_in=H, d_out=I]`).
    gate: Tensor<3>,
    /// `[E, H, I]` — stacked `up_proj` weights.
    up: Tensor<3>,
    /// `[E, I, H]` — stacked `down_proj` weights (Burn `Linear` `[d_in=I, d_out=H]`).
    down: Tensor<3>,
    /// Number of experts `E`.
    e: usize,
    /// Hidden size `H`.
    h: usize,
    /// Per-expert SwiGLU inner dim `I`.
    i: usize,
    /// Experts per token `top_k`.
    k: usize,
}

impl MoeExpertCache {
    /// Build a thin gather handle over a sparse block's expert stacks. The block now OWNS the contiguous
    /// `gate [E,H,I]`/`up [E,H,I]`/`down [E,I,H]` stacks (the vLLM `FusedMoE` single-owner layout), so
    /// `stacked_experts_pub` returns refcounted handles that SHARE those buffers — this cache holds the
    /// shared handles, it does NOT allocate a second copy. (The old path `cat`-built a fresh ~58 GB
    /// duplicate here, which OOM'd the 30B; see `docs/WAVE2_STATIC_DECODE.md`.) Gather-ready, never
    /// rebuilt per forward.
    pub fn from_block(block: &Qwen3MoeSparseBlock) -> Self {
        let (gate, up, down) = block.stacked_experts_pub(); // [E,H,I],[E,H,I],[E,I,H]
        let [e, h, i] = gate.dims();
        debug_assert_eq!(down.dims(), [e, i, h], "down must be [E,I,H]");
        MoeExpertCache {
            gate,
            up,
            down,
            e,
            h,
            i,
            k: block.top_k(),
        }
    }

    /// Number of experts `E` in the cache.
    pub fn num_experts(&self) -> usize {
        self.e
    }

    /// Experts per token `top_k`.
    pub fn top_k(&self) -> usize {
        self.k
    }

    /// Per-expert SwiGLU inner dim `I` (useful for the Wave-2 model integration / kernel dispatch).
    pub fn inner_size(&self) -> usize {
        self.i
    }

    /// CAPTURABLE, fixed-shape, NO-host-sync top-`k` decode. `x:[B,S,H]` (decode: `S=1`). Routes each of
    /// the `T=B*S` tokens to its top-`k` experts, gathers ONLY those experts' weight slabs from the
    /// pre-stacked cache (at `T=1`, exactly `k` of `E`), runs a batched SwiGLU, weights by the gate
    /// weights, and scatter-adds the result back per token. Output `[B,S,H]`.
    ///
    /// Numerically equal to [`Qwen3MoeSparseBlock::forward_oracle`] /
    /// [`Qwen3MoeSparseBlock::forward_routed`] (the same routing + the same combine), but with the
    /// host-side grouping replaced by an on-device expert-axis gather — so there is no `into_data`/
    /// `to_vec` anywhere on this path. `prec` selects the SwiGLU GEMM compute precision (f32 default;
    /// f32 is the parity-validated path).
    pub fn decode_topk(
        &self,
        block: &Qwen3MoeSparseBlock,
        x: Tensor<3>,
        prec: Precision,
    ) -> Tensor<3> {
        let [b, s, _h] = x.dims();
        let t = b * s;
        let device = x.device();
        // The per-assignment token index `[N=T*k]` (each token id repeated `k` times). `arange` stages a
        // host→device copy, so it is built HERE (eager / non-captured) and handed to the capturable core.
        let assign_tok = Self::assign_tok(t, self.k, &device);
        self.decode_topk_pre(block, x, prec, &assign_tok)
    }

    /// Build the per-assignment token index `[N=T*k]` for a `T`-token decode — the constant the gather/
    /// scatter in [`Self::decode_topk_pre`] needs. `Tensor::arange` stages a host→device copy (uncapturable
    /// inside a CUDA graph), so the CAPTURED static decode must precompute this ONCE outside the capture
    /// region (hoisted into [`crate::MoeStaticDecode`], exactly like the RoPE `freqs`/`arange_tmax`) and
    /// pass it to [`Self::decode_topk_pre`]. For the single-token decode (`T=1`) this is just `[0; k]`.
    pub fn assign_tok(t: usize, k: usize, device: &Device) -> Tensor<1, Int> {
        Tensor::<1, Int>::arange(0..t as i64, device)
            .reshape([t, 1])
            .repeat(&[1, k])
            .reshape([t * k])
    }

    /// CAPTURABLE core of [`Self::decode_topk`] with the `arange`-built per-assignment token index
    /// `assign_tok` (`[N=T*k]`, from [`Self::assign_tok`]) HOISTED OUT, so the captured region stages no
    /// host→device `arange`. Identical math to [`Self::decode_topk`]; the only difference is `assign_tok`
    /// is supplied instead of built per call. This is the entry point the CUDA-graph static decode captures.
    pub fn decode_topk_pre(
        &self,
        block: &Qwen3MoeSparseBlock,
        x: Tensor<3>,
        prec: Precision,
        assign_tok: &Tensor<1, Int>,
    ) -> Tensor<3> {
        let [b, s, h] = x.dims();
        assert_eq!(h, self.h, "x hidden {h} != cache H {}", self.h);
        let t = b * s;
        let k = self.k;
        let n = t * k;
        let device = x.device();
        let dtype = x.dtype();
        assert_eq!(
            assign_tok.dims()[0],
            n,
            "assign_tok len {} != N=T*k={n} (T={t}, k={k}) — rebuild via MoeExpertCache::assign_tok(T, k)",
            assign_tok.dims()[0],
        );

        // 1. ROUTE → compact top-k (on-device: iterated-argmax + renorm, no host sync, no `topk`).
        let (sel_idx, sel_w) = block.route_topk(x.clone()); // [T,k] Int, [T,k] f32

        // 2. Flatten the N = T*k (token, expert) assignments.
        let assign_e = sel_idx.reshape([n]); // [N] expert id per assignment
        let assign_tok = assign_tok.clone(); // [N] token id per assignment (HOISTED — no per-call arange)

        // 2a. GATHER the selected experts' weight slabs along the E axis — reads ONLY the N indexed rows
        //     (at T=1, N=k slabs out of E), NOT the full stack. This is the keystone: select(0, ·) is a
        //     device gather kernel; the 120 unrouted experts are never touched.
        let gate_sel = self.gate.clone().select(0, assign_e.clone()); // [N,H,I]
        let up_sel = self.up.clone().select(0, assign_e.clone()); // [N,H,I]
        let down_sel = self.down.clone().select(0, assign_e); // [N,I,H]

        // 3. Gather each assignment's token row → [N,1,H]; batched SwiGLU on the gathered weights.
        let x2 = x.reshape([t, h]); // [T,H]
        let x_sel = x2.select(0, assign_tok.clone()).reshape([n, 1, h]); // [N,1,H]

        let comp = match prec {
            Precision::F32 => DType::F32,
            Precision::Bf16 => DType::BF16,
            Precision::F16 => DType::F16,
        };
        let xc = x_sel.cast(comp); // [N,1,H]
        let g = silu(xc.clone().matmul(gate_sel.cast(comp))); // [N,1,I] true batched matmul (not broadcast)
        let u = xc.matmul(up_sel.cast(comp)); // [N,1,I]
        let y = (g * u).matmul(down_sel.cast(comp)); // [N,1,H]
        let y = y.reshape([n, h]).cast(dtype); // back to model dtype for the combine

        // 4. Weight by the (renormalized) gate weight and SCATTER-ADD back per token. A token's k
        //    distinct experts accumulate by Add — exactly the oracle's weighted top-k sum.
        let w = sel_w.reshape([n, 1]).cast(dtype); // [N,1]
        let y_w = y * w; // [N,H]
        let acc = Tensor::<2>::zeros([t, h], &device)
            .cast(dtype)
            .select_assign(0, assign_tok, y_w, IndexingUpdateOp::Add); // [T,H]
        acc.reshape([b, s, h])
    }

    /// TEST/VERIFICATION hook: the gathered expert-weight slab tensor `[N,H,I]` for `x` (N = T*k). Its
    /// leading dim is the number of expert weight slabs the decode reads — `k` at `T=1`, i.e. ONLY the
    /// routed experts, never the full `E`. Used by `decode_reads_only_topk_experts` to prove the read set.
    #[cfg(test)]
    fn gathered_gate_slabs(&self, block: &Qwen3MoeSparseBlock, x: Tensor<3>) -> Tensor<3> {
        let [b, s, _h] = x.dims();
        let n = b * s * self.k;
        let (sel_idx, _sel_w) = block.route_topk(x);
        self.gate.clone().select(0, sel_idx.reshape([n]))
    }
}

#[cfg(feature = "cuda")]
impl MoeExpertCache {
    /// LEVER (c) of `docs/PERF_80TOKS_PLAN.md`: the FUSED gather-GEMV decode. Numerically EQUAL to
    /// [`Self::decode_topk_pre`] (the materializing oracle) for both `T=1` and `T>1`, but instead of
    /// `select(0, ids)`-materializing the `k` expert weight slabs `[N,H,I]`/`[N,I,H]` (a write +
    /// re-read = the ~3× round-trip that caps `decode_topk` at ~15% of peak), it reads each routed
    /// expert's gate/up/down weights ONCE directly from the persistent `[E,H,I]`/`[E,I,H]` stacks by
    /// `expert_id*stride` IN-KERNEL (bf16 decoded to f32 in-register — no host f32 cast of the stacks).
    ///
    /// Same routing (`route_topk`) and same scatter-ADD combine as the oracle — only the per-assignment
    /// SwiGLU is the fused kernel ([`crate::moe_grouped::fused_gather_swiglu`]) instead of the
    /// `select`+batched-matmul. Fixed-shape for fixed `T` (grid `CubeCount::Static`), NO host sync ⇒
    /// CUDA-graph capturable, exactly like `decode_topk_pre`. `prec` is accepted for signature parity
    /// with `decode_topk_pre`; the kernel always accumulates in f32 (matching the `Precision::F32`
    /// inference default the static decode runs at), so the bf16 weight bytes are read once into an f32
    /// MAC — never widened to a full bf16/f32 weight tensor.
    pub fn decode_topk_fused(
        &self,
        block: &Qwen3MoeSparseBlock,
        x: Tensor<3>,
        prec: Precision,
        assign_tok: &Tensor<1, Int>,
    ) -> Tensor<3> {
        // PRECISION CONTRACT: the fused kernel always reads bf16/f32 weights into an f32 MAC (f32
        // accumulation), matching the oracle ONLY at `Precision::F32`. A `Bf16` caller would silently
        // diverge from `decode_topk_pre`'s bf16-matmul path — the GRPO rollout-vs-recompute trap. Fail
        // loud rather than corrupt. (parity is pinned at F32 only; honor `prec` before lifting this.)
        assert!(
            matches!(prec, Precision::F32),
            "decode_topk_fused only supports Precision::F32 (it always f32-accumulates); Bf16 would \
             diverge from the oracle. Run F32 or extend the kernel to honor prec."
        );
        // BOUNDS INVARIANT: the kernel indexes `gate/up/down[expert_id*stride + ..]` without an in-kernel
        // `e < E` clamp (a clamp would need a host read → breaks capture). Safe by construction: `route_topk`
        // is an argmax over the E experts, so every `assign_e` ∈ 0..E-1. The stacks must also be whole,
        // offset-0, stride-indexable buffers (true for the persistent slot-loaded cache, fix (b)).
        let [b, s, h] = x.dims();
        assert_eq!(h, self.h, "x hidden {h} != cache H {}", self.h);
        let t = b * s;
        let k = self.k;
        let n = t * k;
        let device = x.device();
        let dtype = x.dtype();
        assert_eq!(
            assign_tok.dims()[0],
            n,
            "assign_tok len {} != N=T*k={n} (T={t}, k={k}) — rebuild via MoeExpertCache::assign_tok(T, k)",
            assign_tok.dims()[0],
        );

        // 1. ROUTE → compact top-k (on-device iterated-argmax + renorm; identical to the oracle).
        let (sel_idx, sel_w) = block.route_topk(x.clone()); // [T,k] Int, [T,k] f32

        // 2. Flatten the N=T*k assignments. assign_tok is HOISTED (capturable — no per-call arange).
        let assign_e = sel_idx.reshape([n]); // [N] expert id
        let sel_w_flat = sel_w.reshape([n]).cast(DType::F32); // [N] router weight (f32 for the kernel)

        // 3. FUSED gather-GEMV: reads gate/up/down ONCE by expert id from the persistent stacks (no
        //    `[N,H,I]` slab, no re-stack), returns the router-WEIGHTED per-assignment output [N,H] f32.
        let x2 = x.reshape([t, h]).cast(DType::F32); // [T,H] f32 (tiny — activations, not weights)
        let y = crate::moe_grouped::fused_gather_swiglu(
            x2,
            self.gate.clone(),
            self.up.clone(),
            self.down.clone(),
            assign_e,
            assign_tok.clone(),
            sel_w_flat,
            h,
            self.i,
            n,
        ); // [N,H] f32, router-weighted
        let y = y.cast(dtype); // back to the model dtype for the combine

        // 4. SCATTER-ADD per token — exactly the oracle combine `out[t]=Σ_{e∈topk(t)} w·SwiGLU_e(x_t)`.
        let acc = Tensor::<2>::zeros([t, h], &device)
            .cast(dtype)
            .select_assign(0, assign_tok.clone(), y, IndexingUpdateOp::Add); // [T,H]
        acc.reshape([b, s, h])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    fn dev() -> Device {
        Default::default()
    }

    /// KEYSTONE PARITY: the capturable top-k decode equals the dense-masked oracle (and the routed fast
    /// path) for BOTH a single token (`T=1`, the decode shape) and a multi-token batch (`T>1`). Mirrors
    /// `forward_routed_equals_oracle`. Locks the gather/batched-SwiGLU/scatter-add combine to the
    /// reference routing math.
    #[test]
    fn decode_topk_equals_oracle() {
        let device = dev();
        let (h, i, e, k) = (32usize, 16usize, 8usize, 3usize);
        let block = Qwen3MoeSparseBlock::new(h, i, e, k, true, &device);
        let cache = MoeExpertCache::from_block(&block);

        // T=1: the single-token decode shape (reads exactly k experts).
        let x1 = Tensor::<3>::random([1, 1, h], Distribution::Normal(0.0, 1.0), &device);
        let oracle1 = block.forward_oracle(x1.clone(), Precision::F32);
        let routed1 = block.forward_routed(x1.clone(), Precision::F32);
        let decode1 = cache.decode_topk(&block, x1, Precision::F32);
        let d_o1: f32 = (oracle1.clone() - decode1.clone())
            .abs()
            .max()
            .into_scalar();
        let d_r1: f32 = (routed1 - decode1).abs().max().into_scalar();
        assert!(d_o1 < 1e-4, "T=1 decode != oracle: max|diff|={d_o1}");
        assert!(
            d_r1 < 1e-4,
            "T=1 decode != forward_routed: max|diff|={d_r1}"
        );

        // T>1: several token rows (B=2,S=5 => T=10).
        let x = Tensor::<3>::random([2, 5, h], Distribution::Normal(0.0, 1.0), &device);
        let oracle = block.forward_oracle(x.clone(), Precision::F32);
        let decode = cache.decode_topk(&block, x, Precision::F32);
        let d_o: f32 = (oracle - decode).abs().max().into_scalar();
        assert!(d_o < 1e-4, "T>1 decode != oracle: max|diff|={d_o}");
    }

    /// HOIST PARITY: the capturable `decode_topk_pre` (precomputed `assign_tok` handed in — the CUDA-graph
    /// path) is BIT-IDENTICAL to the eager `decode_topk` (which builds `assign_tok` via the per-call
    /// `arange`). Pins that hoisting the per-assignment token index out of the captured region changed
    /// nothing numerically, for both the single-token decode (`T=1`) and a multi-token batch (`T>1`).
    #[test]
    fn decode_topk_pre_equals_decode_topk() {
        let device = dev();
        let (h, i, e, k) = (32usize, 16usize, 8usize, 3usize);
        let block = Qwen3MoeSparseBlock::new(h, i, e, k, true, &device);
        let cache = MoeExpertCache::from_block(&block);

        // T=1: assign_tok = [0; k]. decode_topk (arange-built) must equal decode_topk_pre (hoisted).
        let x1 = Tensor::<3>::random([1, 1, h], Distribution::Normal(0.0, 1.0), &device);
        let at1 = MoeExpertCache::assign_tok(1, k, &device);
        let eager1 = cache.decode_topk(&block, x1.clone(), Precision::F32);
        let pre1 = cache.decode_topk_pre(&block, x1, Precision::F32, &at1);
        let d1: f32 = (eager1 - pre1).abs().max().into_scalar();
        assert!(
            d1 == 0.0,
            "T=1 decode_topk_pre != decode_topk: max|diff|={d1} (must be bit-identical)"
        );

        // T>1 (B=2,S=5 => T=10): same with the full repeated index.
        let (b, s) = (2usize, 5usize);
        let t = b * s;
        let x = Tensor::<3>::random([b, s, h], Distribution::Normal(0.0, 1.0), &device);
        let at = MoeExpertCache::assign_tok(t, k, &device);
        let eager = cache.decode_topk(&block, x.clone(), Precision::F32);
        let pre = cache.decode_topk_pre(&block, x, Precision::F32, &at);
        let d: f32 = (eager - pre).abs().max().into_scalar();
        assert!(
            d == 0.0,
            "T>1 decode_topk_pre != decode_topk: max|diff|={d} (must be bit-identical)"
        );
    }

    /// PROOF the decode reads ONLY the top-k experts at T=1 (the bandwidth claim): poison every NON-routed
    /// expert's cached weights with NaN, then decode. If the gather touched any unrouted expert, NaN would
    /// propagate; finite output that still equals the (clean) oracle proves exactly the k routed slabs were
    /// read — 4 of 16 here, scaling to 8 of 128 on the real model. Also asserts the gathered slab count is
    /// k (not E).
    #[test]
    fn decode_reads_only_topk_experts() {
        let device = dev();
        let (h, i, e, k) = (24usize, 12usize, 16usize, 4usize); // 12 of 16 experts are UNROUTED at T=1
        let block = Qwen3MoeSparseBlock::new(h, i, e, k, true, &device);
        let clean = MoeExpertCache::from_block(&block);

        let x = Tensor::<3>::random([1, 1, h], Distribution::Normal(0.0, 1.0), &device);

        // The gather materializes exactly N = T*k = k slabs (NOT E) — the structural read-set proof.
        let slabs = clean.gathered_gate_slabs(&block, x.clone());
        assert_eq!(
            slabs.dims()[0],
            k,
            "decode gathered {} expert slabs, want k={k}",
            slabs.dims()[0]
        );

        // Which experts are routed (host read is fine in a test; the DECODE path itself never syncs).
        let (sel_idx, _w) = block.route_topk(x.clone());
        let routed: Vec<i64> = sel_idx.cast(DType::I64).into_data().to_vec().unwrap(); // k ids
        let routed_set: std::collections::HashSet<i64> = routed.iter().copied().collect();

        // Poison mask over the E axis: 1.0 for routed experts, NaN for the rest.
        let mask_vals: Vec<f32> = (0..e)
            .map(|ei| {
                if routed_set.contains(&(ei as i64)) {
                    1.0
                } else {
                    f32::NAN
                }
            })
            .collect();
        let mask = Tensor::<1>::from_data(mask_vals.as_slice(), &device).reshape([e, 1, 1]); // [E,1,1]
        let poisoned = MoeExpertCache {
            gate: clean.gate.clone() * mask.clone(),
            up: clean.up.clone() * mask.clone(),
            down: clean.down.clone() * mask,
            e: clean.e,
            h: clean.h,
            i: clean.i,
            k: clean.k,
        };

        let oracle = block.forward_oracle(x.clone(), Precision::F32);
        let decode = poisoned.decode_topk(&block, x, Precision::F32);

        // 1) finite (no NaN leaked from an unrouted, NaN-poisoned expert) and 2) equals the clean oracle.
        let finite: f32 = decode.clone().mul_scalar(0.0f32).sum().into_scalar(); // 0*NaN = NaN => nonzero
        assert!(
            finite == 0.0,
            "decode read a poisoned (unrouted) expert: non-finite output"
        );
        let d: f32 = (oracle - decode).abs().max().into_scalar();
        assert!(
            d < 1e-4,
            "poisoned-cache decode != clean oracle: max|diff|={d} (read an unrouted expert?)"
        );
    }

    /// top_k = 1 sanity: the decode must equal the single selected expert's output (weight renormalizes
    /// to 1.0). Catches a wrong expert id, a dropped weight, or a mis-scattered token.
    #[test]
    fn decode_top1_equals_selected_expert() {
        let device = dev();
        let (h, i, e) = (32usize, 16usize, 8usize);
        let block = Qwen3MoeSparseBlock::new(h, i, e, 1, true, &device);
        let cache = MoeExpertCache::from_block(&block);
        let x = Tensor::<3>::random([2, 3, h], Distribution::Normal(0.0, 1.0), &device);
        let oracle = block.forward_oracle(x.clone(), Precision::F32);
        let decode = cache.decode_topk(&block, x, Precision::F32);
        let d: f32 = (oracle - decode).abs().max().into_scalar();
        assert!(d < 1e-4, "top-1 decode != oracle: max|diff|={d}");
    }

    /// Shape-preserving and deterministic.
    #[test]
    fn decode_shape_and_deterministic() {
        let device = dev();
        let block = Qwen3MoeSparseBlock::new(32, 16, 8, 2, true, &device);
        let cache = MoeExpertCache::from_block(&block);
        let x = Tensor::<3>::random([1, 1, 32], Distribution::Normal(0.0, 1.0), &device);
        let y1 = cache.decode_topk(&block, x.clone(), Precision::F32);
        let y2 = cache.decode_topk(&block, x, Precision::F32);
        assert_eq!(y1.dims(), [1, 1, 32]);
        let d: f32 = (y1 - y2).abs().sum().into_scalar();
        assert!(d < 1e-6, "decode not deterministic: |diff|={d}");
    }
}

/// CUDA parity tests for the FUSED gather-GEMV decode (lever (c)). Run on the real GB10 — the kernel
/// is a custom CubeCL launch, so these are gated on the `cuda` feature (the backend-generic tests above
/// cover the materializing oracle on NdArray). Pin `decode_topk_fused == decode_topk_pre` (the oracle).
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use burn::prelude::Device;
    use burn::tensor::Distribution;

    type C = Cuda;

    /// KEYSTONE: the fused gather-GEMV decode equals the materializing oracle `decode_topk_pre` for both
    /// the single-token decode (`T=1`) and a multi-token batch (`T>1`), to f32 eps. Same routing + same
    /// scatter-add combine; only the per-assignment SwiGLU kernel differs (no `[N,H,I]` materialization).
    #[test]
    fn decode_topk_fused_equals_oracle_cuda() {
        let device = Device::cuda(0);
        let (h, i, e, k) = (64usize, 48usize, 16usize, 4usize);
        let block = Qwen3MoeSparseBlock::<C>::new(h, i, e, k, true, &device);
        let cache = MoeExpertCache::from_block(&block);

        // T=1 — the decode shape (reads exactly k experts of E).
        let x1 = Tensor::<C, 3>::random([1, 1, h], Distribution::Normal(0.0, 1.0), &device);
        let at1 = MoeExpertCache::<C>::assign_tok(1, k, &device);
        let oracle1 = cache.decode_topk_pre(&block, x1.clone(), Precision::F32, &at1);
        let fused1 = cache.decode_topk_fused(&block, x1, Precision::F32, &at1);
        let d1: f32 = (oracle1 - fused1).abs().max().into_scalar();
        assert!(d1 < 1e-4, "T=1 fused != oracle: max|diff|={d1}");

        // T>1 — several token rows (B=2,S=5 => T=10), exercises the per-token scatter-add.
        let (b, s) = (2usize, 5usize);
        let t = b * s;
        let x = Tensor::<C, 3>::random([b, s, h], Distribution::Normal(0.0, 1.0), &device);
        let at = MoeExpertCache::<C>::assign_tok(t, k, &device);
        let oracle = cache.decode_topk_pre(&block, x.clone(), Precision::F32, &at);
        let fused = cache.decode_topk_fused(&block, x, Precision::F32, &at);
        let d: f32 = (oracle - fused).abs().max().into_scalar();
        assert!(d < 1e-4, "T>1 fused != oracle: max|diff|={d}");
    }

    /// top_k = 1 sanity: with a single expert per token the fused decode must equal the oracle exactly
    /// (the router weight renormalizes to 1.0) — catches a wrong expert id or a mis-scattered token.
    #[test]
    fn decode_topk_fused_top1_cuda() {
        let device = Device::cuda(0);
        let (h, i, e) = (48usize, 32usize, 8usize);
        let block = Qwen3MoeSparseBlock::<C>::new(h, i, e, 1, true, &device);
        let cache = MoeExpertCache::from_block(&block);
        let x = Tensor::<C, 3>::random([2, 3, h], Distribution::Normal(0.0, 1.0), &device);
        let at = MoeExpertCache::<C>::assign_tok(6, 1, &device);
        let oracle = cache.decode_topk_pre(&block, x.clone(), Precision::F32, &at);
        let fused = cache.decode_topk_fused(&block, x, Precision::F32, &at);
        let d: f32 = (oracle - fused).abs().max().into_scalar();
        assert!(d < 1e-4, "top-1 fused != oracle: max|diff|={d}");
    }
}
