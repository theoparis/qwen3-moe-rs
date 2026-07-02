//! DROPLESS MoE grouped-GEMM fast path — `docs/VLLM_KERNELS.md` §3. CUDA only.
//!
//! The repo's [`Qwen3MoeSparseBlock::forward_routed_ondevice`](crate::Qwen3MoeSparseBlock::forward_routed_ondevice)
//! computes the MoE via a CAPACITY-padded `[E,C,H]` batched matmul: each expert gets a fixed `C`
//! slots and any assignment past `C` is DROPPED (corrupts GRPO parity) — and it computes `E*C` FFNs.
//! This module is the DROPLESS, COMPACT path: it computes EXACTLY the `k*T` routed `(token, expert)`
//! pairs, with no drop, via the vLLM `moe_align_block_size` layout + a block-per-expert-segment GEMM
//! with an indirection table (the vLLM `fused_moe` structure).
//!
//! ## The dropless align/sort (on-device, Burn ops — extends the existing one-hot + cumsum + scatter)
//! For the `N = T*k` flattened `(token, expert)` assignments:
//!   * `count_e   = Σ_t onehot[:,e]`                 — per-expert assignment count `[E]`
//!   * `padded_e  = ceil(count_e / BLOCK_M)*BLOCK_M` — each expert's segment is `BLOCK_M`-aligned `[E]`
//!   * `base_e    = ExclusiveCumsum(padded_e)`       — start slot of each expert's segment `[E]`
//!   * `rank      = within-expert cumsum − 1`        — each assignment's position inside its expert
//!   * `dest      = base_e[expert] + rank`           — UNIQUE slot per assignment ⇒ **no drop**
//! scattered into a fixed buffer of `num_blocks*BLOCK_M ≤ N + E*(BLOCK_M-1)` slots:
//! `sorted_token[dest]=token`, `sorted_weight[dest]=router_weight`, `sorted_expert[dest]=expert`
//! (empty slots read the `−1` sentinel). The per-block `expert_ids[blk] = sorted_expert[blk*BLOCK_M]`
//! (valid because, with dense within-expert ranks, the FIRST slot of every block of a non-empty
//! expert is always a real token); tail/padding blocks read `−1` and are skipped.
//!
//! ## The grouped GEMM ([`gpu::grouped_swiglu`])
//! One cube per `BLOCK_M` segment, one thread per slot-row. `e = expert_ids[blk]`; if `e<0` (or the
//! slot's token is the `−1` pad) the row writes zeros. Otherwise the thread gathers `x[token]` and
//! computes the expert's fused SwiGLU `down_e( silu(x·gate_e) * (x·up_e) )` reading the stacked
//! weights `gate/up/down[e]` via the `e` indirection, scales by the router weight, and writes the
//! per-slot result. f32 accumulate. The k contributions of a token are then combined by a
//! scatter-ADD (a separate, deterministic Burn reduction) back to `out[token]`.
//!
//! ### Structure built (be explicit): COMPUTE-then-SCATTER-ADD, scalar per-row GEMM
//! This is the correctness-first structure the §3 spec permits: a per-slot scalar SwiGLU (no
//! CMMA/tensor-core tiling, no `GROUP_SIZE_M` L2 reuse) writing a per-slot `[buffer,H]` output, with
//! the top-k scatter-ADD done by a Burn `select_assign(Add)` (avoids in-kernel global atomics; the
//! GEMM itself carries the full vLLM `sorted_token_ids`/`expert_ids` indirection). The PERF
//! follow-ons (a CMMA block-tiled GEMM with shared-memory weight reuse, `GROUP_SIZE_M` L2 locality,
//! a stream-K / persistent scheduler for expert load-imbalance, and an in-kernel atomic scatter-add)
//! are noted, NOT built — they do not change correctness.
//!
//! ## 64-bit offsets (§3, all reviewers)
//! Every GLOBAL element offset (`token*H`, `e*H*I + h*I + i`, `e*I*H + i*H + h`, `slot*H`) is computed
//! in **`i64`** — operands are cast to `i64` BEFORE the multiply, then the final offset is cast to
//! `usize` for indexing. CubeCL's `usize` is the kernel's address type (`u32` for buffers below
//! `u32::MAX`, auto-promoted to `u64` above), and a valid in-bounds offset always fits it; doing the
//! multiplies in `i64` is what prevents the silent `u32` wrap the spec warns about (`E*C*H = 4.3e9 >
//! 2^31`).
//!
//! Validated on the real GB10 against an INDEPENDENT NdArray (CPU f32) oracle by
//! `examples/moe_grouped_spike.rs` (the §0 cross-backend law).
//!
//! ⚠️ STATUS — CORRECT + DROPLESS + PARITY-SAFE, but a correctness-only prototype that would currently
//! make the MoE rollout SLOWER (3-voice review: Codex gpt-5.5 / Opus 4.8 / Gemini 3.1 Pro). The
//! dropless align/sort is PROVEN sound (Opus, full arithmetic walk); the numerics EQUAL the dense
//! oracle (cosine 1.0, ~1e-8 = fp accumulation order), so — UNLIKE the fp8 W8A16 kernel — rollout-grouped
//! vs recompute-oracle are parity-safe (`forward_grouped` is forward-only/no-backward, so it can only
//! live in the no-grad rollout; the grad recompute uses `forward_oracle`, which has a backward). What
//! blocks deployment:
//!  * **WRONG REGIME (P1) — the same trap as fp8.** This scalar, zero-weight-reuse kernel reads fewer
//!    weight bytes than dense ONLY when `k·T < E`, i.e. `T < E/k = 16`. The batched GRPO rollout decodes
//!    `T = prompts × group_size ≫ 16`, where it re-reads each expert's weights `k·T` times (no shared-mem
//!    reuse) → ~8× MORE weight traffic than `forward_routed_ondevice` (which reuses each expert weight
//!    once, ~2.4 GB/layer) and runs without tensor cores → ~10-50× SLOWER. It wins only at T=1
//!    single-stream serving — which the rollout never hits. Realizing the "k·T not E·dense" win needs a
//!    WEIGHT-STATIONARY CMMA rewrite (shared-mem weight tiles + I-tiling), NOT built.
//!  * **Per-call re-stacking (P1).** `forward_grouped` rebuilds the `[E,H,I]×3` stack + `.cast(f32)` +
//!    `into_contiguous` every forward (~7 GB of weight-shuffling/layer/step before any compute). Must
//!    become a one-time pre-stacked, contiguous, f32 weight cache after load.
//!  * **The `Array::<f32>::new(I)` per-thread local** (I=768 at 30B → ~3 KB/thread) spills to local mem;
//!    the padding combine routes all pad slots to one row (atomic-add contention). Fold into the rewrite.
//!  * **f32-regime enforcement (P1, latent).** `forward_grouped` hard-casts to f32; if the model ever runs
//!    `Precision::Bf16`, rollout(f32) vs recompute(bf16-oracle) would diverge ~1e-2/GEMM — the fp8 failure
//!    mode. Honor `prec` or assert f32 before wiring in. Gate on END-TO-END logprob parity at the real
//!    E=128/H=2048 shape (not GEMM cosine), which is not yet run.
//!  * **No MoE GRPO path exists.** The trainer/rollout are hardcoded to the dense `Qwen3ForCausalLM`;
//!    `forward_grouped` is `Cuda`-typed (not generic). Deploying needs generalizing the whole trainer +
//!    rollout over the model type first — a large gap independent of this kernel.
//!  Test follow-on: the spike validates E=8/E=32 (mostly-non-empty, even counts); add an E=128/T≈8 skew
//!  case to exercise the empty-`padded_e=0` / partial-last-block paths the real model hits.

use burn::backend::cuda::Cuda;
use burn::prelude::Backend;
use burn::tensor::{DType, IndexingUpdateOp, Int, Tensor, TensorPrimitive};

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ScalarArg;
use cubecl::tensor_line_size_parallel;
use cubecl::{CubeCount, CubeDim};

use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeBackend;

use crate::cube_custom_op::CubeCustomOp;
use crate::Qwen3MoeSparseBlock;

// =================================================================================================
// GPU kernel. Own module so `cubecl::prelude::Tensor` (GPU-side) does not clash with
// `burn::tensor::Tensor` (host-side).
// =================================================================================================
mod gpu {
    use cubecl::e4m3;
    use cubecl::prelude::*;

    /// DROPLESS grouped SwiGLU. One cube per `BLOCK_M` segment (`blk = CUBE_POS_X`), one thread per
    /// slot-row (`row = UNIT_POS_X`, `slot = blk*BLOCK_M + row`).
    ///
    /// * `x`            — `[T, H]` activations (contiguous).
    /// * `gate`,`up`    — `[E, H, I]` stacked expert weights (contiguous).
    /// * `down`         — `[E, I, H]` stacked expert weights (contiguous).
    /// * `sorted_token` — `[buffer]` i32: token id per slot, `−1` = padding.
    /// * `sorted_weight`— `[buffer]` f32: router weight per slot.
    /// * `expert_ids`   — `[num_blocks]` i32: expert id per block, `−1` = empty/tail block.
    /// * `out`          — `[buffer, H]` f32: per-slot weighted SwiGLU output (padding rows = 0).
    ///
    /// All global offsets are computed in `i64` (cast BEFORE the multiply) then narrowed to `usize`
    /// for indexing — see the module docs on 64-bit offsets.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn grouped_swiglu(
        x: &Tensor<f32>,
        gate: &Tensor<f32>,
        up: &Tensor<f32>,
        down: &Tensor<f32>,
        sorted_token: &Tensor<i32>,
        sorted_weight: &Tensor<f32>,
        expert_ids: &Tensor<i32>,
        out: &mut Tensor<f32>,
        h_dim: u32,
        i_dim: u32,
        block_m: u32,
        #[comptime] i_cap: usize,
    ) {
        // Positions/loop-bounds in `usize` (the kernel's index type); the GLOBAL offsets below are
        // built in `i64` (cast BEFORE the multiply) then narrowed to `usize` for indexing.
        let blk = CUBE_POS_X as usize;
        let row = UNIT_POS_X as usize;
        let bm = block_m as usize;
        let h_dim_u = h_dim as usize;
        let i_dim_u = i_dim as usize;
        let slot = blk * bm + row;

        if slot < out.shape(0) {
            let expert = expert_ids[blk];
            let token = sorted_token[slot];
            let weight = sorted_weight[slot];

            // ---- 64-bit GLOBAL offsets: cast to i64 BEFORE the multiply (§3 / all reviewers) ----
            let hi = i64::cast_from(h_dim);
            let ii = i64::cast_from(i_dim);
            let y_base = i64::cast_from(slot) * hi; // out[slot, :]

            if expert >= 0i32 && token >= 0i32 {
                let e_i = i64::cast_from(expert);
                let tok_i = i64::cast_from(token);
                let g_e_base = e_i * hi * ii; // gate/up [E,H,I] base for expert e
                let d_e_base = e_i * ii * hi; // down [E,I,H] base for expert e
                let x_base = tok_i * hi; // x[token, :]

                // gu[i] = silu(x·gate_e[:,i]) * (x·up_e[:,i]). Runtime loops (no comptime unroll
                // explosion); a comptime-sized local array holds the I intermediates.
                let mut gu = Array::<f32>::new(i_cap);
                for ci in 0..i_dim_u {
                    let i_i = i64::cast_from(ci);
                    let mut gacc = f32::new(0.0);
                    let mut uacc = f32::new(0.0);
                    for hh in 0..h_dim_u {
                        let h_i = i64::cast_from(hh);
                        let xv = x[usize::cast_from(x_base + h_i)];
                        let w_off = g_e_base + h_i * ii + i_i; // [E,H,I]: e*H*I + h*I + i
                        gacc += xv * gate[usize::cast_from(w_off)];
                        uacc += xv * up[usize::cast_from(w_off)];
                    }
                    // silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
                    let sig = 1.0f32 / (1.0f32 + (0.0f32 - gacc).exp());
                    gu[ci] = gacc * sig * uacc;
                }

                // y[h] = (Σ_i gu[i] * down_e[i,h]) * router_weight
                for hh in 0..h_dim_u {
                    let h_i = i64::cast_from(hh);
                    let mut acc = f32::new(0.0);
                    for ci in 0..i_dim_u {
                        let i_i = i64::cast_from(ci);
                        let d_off = d_e_base + i_i * hi + h_i; // [E,I,H]: e*I*H + i*H + h
                        acc += gu[ci] * down[usize::cast_from(d_off)];
                    }
                    out[usize::cast_from(y_base + h_i)] = acc * weight;
                }
            } else {
                // Padding slot / empty / tail block → zero row (no NaN holes downstream).
                for hh in 0..h_dim_u {
                    let h_i = i64::cast_from(hh);
                    out[usize::cast_from(y_base + h_i)] = f32::new(0.0);
                }
            }
        }
    }

    /// LEVER (c) phase 1 — FUSED gather-GEMV `gu`. One thread per `(n, i)` output element of
    /// `gu:[N,I]`: `gu[n,i] = silu( Σ_h x[tok,h]·gate[e,h,i] ) · ( Σ_h x[tok,h]·up[e,h,i] )`, where
    /// `e = assign_e[n]`, `tok = assign_tok[n]`. The expert's gate/up weight COLUMNS are read DIRECTLY
    /// from the persistent `[E,H,I]` stacks by `e` (via the tensor's own strides — NO materialized
    /// `[N,H,I]` slab, NO host re-stack) and decoded to f32 IN-REGISTER, so each weight element is read
    /// from HBM exactly once into the MAC. Weight element type `EW` (bf16 on the 30B, f32 on the
    /// synthetic CPU/Cuda oracle) is a kernel type-param so no host dtype cast is needed.
    ///
    /// All GLOBAL offsets are built in `i64` (cast BEFORE the multiply) then narrowed to `usize`, same
    /// 64-bit-offset rule as [`grouped_swiglu`] (`E*H*I` overflows `u32` on the 30B).
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused_swiglu_gu<EW: Float>(
        x: &Tensor<f32>,          // [T, H] activations (contiguous f32)
        gate: &Tensor<EW>,        // [E, H, I] stacked gate weights (read by expert id)
        up: &Tensor<EW>,          // [E, H, I] stacked up weights
        assign_e: &Tensor<i32>,   // [N] expert id per assignment
        assign_tok: &Tensor<i32>, // [N] token id per assignment
        gu: &mut Tensor<f32>,     // [N, I] silu(x·gate)·(x·up)
        h_dim: u32,
        i_dim: u32,
    ) {
        if ABSOLUTE_POS < gu.len() {
            let pos = ABSOLUTE_POS as usize;
            let i_dim_u = i_dim as usize;
            let h_dim_u = h_dim as usize;
            let n = pos / i_dim_u; // assignment
            let ci = pos % i_dim_u; // inner column

            let e = assign_e[n * assign_e.stride(0)];
            let tok = assign_tok[n * assign_tok.stride(0)];

            // ---- i64 global offsets (cast BEFORE the multiply); use the tensors' OWN strides so the
            //      persistent (un-`into_contiguous`'d) stacks are indexed correctly. ----
            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let gs0 = i64::cast_from(gate.stride(0));
            let gs1 = i64::cast_from(gate.stride(1));
            let gs2 = i64::cast_from(gate.stride(2));
            let us0 = i64::cast_from(up.stride(0));
            let us1 = i64::cast_from(up.stride(1));
            let us2 = i64::cast_from(up.stride(2));
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let ci_i = i64::cast_from(ci);
            let x_base = tok_i * xs0;
            let g_base = e_i * gs0 + ci_i * gs2; // gate[e, :, ci]
            let u_base = e_i * us0 + ci_i * us2; // up[e, :, ci]

            let mut gacc = f32::new(0.0);
            let mut uacc = f32::new(0.0);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                gacc += xv * f32::cast_from(gate[usize::cast_from(g_base + h_i * gs1)]);
                uacc += xv * f32::cast_from(up[usize::cast_from(u_base + h_i * us1)]);
            }
            // silu(g) = g * sigmoid(g) = g / (1 + exp(-g)); fused with the up gate.
            let sig = 1.0f32 / (1.0f32 + (0.0f32 - gacc).exp());
            gu[pos] = gacc * sig * uacc;
        }
    }

    /// LEVER (c) phase 2 — FUSED gather-GEMV `down` + router-weighted output. One thread per `(n, h)`
    /// element of `out:[N,H]`: `out[n,h] = sel_w[n] · Σ_i gu[n,i]·down[e,i,h]`, `e = assign_e[n]`. The
    /// expert's down weight COLUMN is read DIRECTLY from the persistent `[E,I,H]` stack by `e` (strides,
    /// no slab), decoded to f32 in-register (each element read once), and the router weight is applied
    /// in-register so the host combine is a pure scatter-ADD. Same 64-bit-offset rule.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused_swiglu_down<EW: Float>(
        gu: &Tensor<f32>,       // [N, I]
        down: &Tensor<EW>,      // [E, I, H] stacked down weights (read by expert id)
        assign_e: &Tensor<i32>, // [N] expert id per assignment
        sel_w: &Tensor<f32>,    // [N] router weight per assignment
        out: &mut Tensor<f32>,  // [N, H] weighted per-assignment SwiGLU output
        h_dim: u32,
        i_dim: u32,
    ) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let h_dim_u = h_dim as usize;
            let i_dim_u = i_dim as usize;
            let n = pos / h_dim_u; // assignment
            let hh = pos % h_dim_u; // output feature

            let e = assign_e[n * assign_e.stride(0)];
            let w = sel_w[n * sel_w.stride(0)];

            let gs0 = i64::cast_from(gu.stride(0));
            let gs1 = i64::cast_from(gu.stride(1));
            let ds0 = i64::cast_from(down.stride(0));
            let ds1 = i64::cast_from(down.stride(1));
            let ds2 = i64::cast_from(down.stride(2));
            let e_i = i64::cast_from(e);
            let n_i = i64::cast_from(n);
            let h_i = i64::cast_from(hh);
            let gu_base = n_i * gs0;
            let d_base = e_i * ds0 + h_i * ds2; // down[e, :, hh]

            let mut acc = f32::new(0.0);
            for ci in 0..i_dim_u {
                let i_i = i64::cast_from(ci);
                acc += gu[usize::cast_from(gu_base + i_i * gs1)]
                    * f32::cast_from(down[usize::cast_from(d_base + i_i * ds1)]);
            }
            out[pos] = acc * w;
        }
    }

    /// Qwen3.5 35B scaffold: combined gate/up stack layout `[E, 2I, H]`, one thread per `gu[n,i]`.
    /// The gate and up halves are addressed by two in-kernel offsets into the SAME tensor; callers must
    /// not slice the stack into gate/up views because sliced storage offsets are not honored here.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_bf16_scalar<EW: Float>(
        x: &Tensor<f32>,        // [T, H]
        gate_up: &Tensor<EW>,   // [E, 2I, H]
        assign_e: &Tensor<i32>, // [N]
        gu: &mut Tensor<f32>,   // [N, I]
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
    ) {
        if ABSOLUTE_POS < gu.len() {
            let pos = ABSOLUTE_POS as usize;
            let i_dim_u = i_dim as usize;
            let h_dim_u = h_dim as usize;
            let n = pos / i_dim_u;
            let ci = pos % i_dim_u;
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let gs0 = i64::cast_from(gate_up.stride(0));
            let gs1 = i64::cast_from(gate_up.stride(1));
            let gs2 = i64::cast_from(gate_up.stride(2));
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let ci_i = i64::cast_from(ci);
            let i_dim_i = i64::cast_from(i_dim_u);
            let x_base = tok_i * xs0;
            let g_base = e_i * gs0 + ci_i * gs1;
            let u_base = e_i * gs0 + (ci_i + i_dim_i) * gs1;

            let mut gacc = f32::new(0.0);
            let mut uacc = f32::new(0.0);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                gacc += xv * f32::cast_from(gate_up[usize::cast_from(g_base + h_i * gs2)]);
                uacc += xv * f32::cast_from(gate_up[usize::cast_from(u_base + h_i * gs2)]);
            }
            let sig = 1.0f32 / (1.0f32 + (0.0f32 - gacc).exp());
            gu[pos] = gacc * sig * uacc;
        }
    }

    /// Qwen3.5 35B scaffold: down stack layout `[E, H, I]`, contraction dim is `I`.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_down_bf16_scalar<EW: Float>(
        gu: &Tensor<f32>,       // [N, I]
        down: &Tensor<EW>,      // [E, H, I]
        assign_e: &Tensor<i32>, // [N]
        sel_w: &Tensor<f32>,    // [N]
        out: &mut Tensor<f32>,  // [N, H]
        h_dim: u32,
        i_dim: u32,
    ) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let h_dim_u = h_dim as usize;
            let i_dim_u = i_dim as usize;
            let n = pos / h_dim_u;
            let hh = pos % h_dim_u;

            let e = assign_e[n * assign_e.stride(0)];
            let w = sel_w[n * sel_w.stride(0)];

            let gs0 = i64::cast_from(gu.stride(0));
            let gs1 = i64::cast_from(gu.stride(1));
            let ds0 = i64::cast_from(down.stride(0));
            let ds1 = i64::cast_from(down.stride(1));
            let ds2 = i64::cast_from(down.stride(2));
            let e_i = i64::cast_from(e);
            let n_i = i64::cast_from(n);
            let h_i = i64::cast_from(hh);
            let gu_base = n_i * gs0;
            let d_base = e_i * ds0 + h_i * ds1;

            let mut acc = f32::new(0.0);
            for ci in 0..i_dim_u {
                let i_i = i64::cast_from(ci);
                acc += gu[usize::cast_from(gu_base + i_i * gs1)]
                    * f32::cast_from(down[usize::cast_from(d_base + i_i * ds2)]);
            }
            out[pos] = acc * w;
        }
    }

    #[cube(launch)]
    pub fn e4m3_line_decode_probe(q: &Tensor<Line<u8>>, out: &mut Tensor<Line<f32>>) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            out[pos] = Line::<f32>::cast_from(Line::<e4m3>::reinterpret(q[pos]));
        }
    }

    #[cube(launch)]
    pub fn e2m1_marlin_decode_probe(q: &Tensor<u8>, out: &mut Tensor<f32>) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let byte = u32::cast_from(q[pos / 2usize]);
            let code = if pos % 2usize == 0usize {
                byte & 15u32
            } else {
                (byte >> 4u32) & 15u32
            };
            out[pos] = e2m1_marlin_decode(code);
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_fp8_scalar(
        x: &Tensor<f32>,        // [T,H]
        q_gu: &Tensor<u8>,      // [E,H,2I] raw e4m3 bytes
        s_gu: &Tensor<f32>,     // [E,2I] per-output-channel scale
        assign_e: &Tensor<i32>, // [N]
        gu: &mut Tensor<f32>,   // [N,I]
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
    ) {
        if ABSOLUTE_POS < gu.len() {
            let pos = ABSOLUTE_POS as usize;
            let i_dim_u = i_dim as usize;
            let h_dim_u = h_dim as usize;
            let n = pos / i_dim_u;
            let ci = pos % i_dim_u;
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let qs0 = i64::cast_from(q_gu.stride(0));
            let qs1 = i64::cast_from(q_gu.stride(1));
            let qs2 = i64::cast_from(q_gu.stride(2));
            let ss0 = i64::cast_from(s_gu.stride(0));
            let ss1 = i64::cast_from(s_gu.stride(1));
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let ci_i = i64::cast_from(ci);
            let i_dim_i = i64::cast_from(i_dim_u);
            let x_base = tok_i * xs0;
            let g_base = e_i * qs0 + ci_i * qs2;
            let u_base = e_i * qs0 + (ci_i + i_dim_i) * qs2;
            let sg = s_gu[usize::cast_from(e_i * ss0 + ci_i * ss1)];
            let su = s_gu[usize::cast_from(e_i * ss0 + (ci_i + i_dim_i) * ss1)];

            let mut gacc = f32::new(0.0);
            let mut uacc = f32::new(0.0);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                let gb = q_gu[usize::cast_from(g_base + h_i * qs1)];
                let ub = q_gu[usize::cast_from(u_base + h_i * qs1)];
                gacc += xv * f32::cast_from(e4m3::reinterpret(gb)) * sg;
                uacc += xv * f32::cast_from(e4m3::reinterpret(ub)) * su;
            }
            let sig = 1.0f32 / (1.0f32 + (0.0f32 - gacc).exp());
            gu[pos] = gacc * sig * uacc;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_down_fp8_scalar(
        gu: &Tensor<f32>,       // [N,I]
        q_dn: &Tensor<u8>,      // [E,I,H] raw e4m3 bytes
        s_dn: &Tensor<f32>,     // [E,H] per-output-channel scale
        assign_e: &Tensor<i32>, // [N]
        sel_w: &Tensor<f32>,    // [N]
        out: &mut Tensor<f32>,  // [N,H]
        h_dim: u32,
        i_dim: u32,
    ) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let h_dim_u = h_dim as usize;
            let i_dim_u = i_dim as usize;
            let n = pos / h_dim_u;
            let hh = pos % h_dim_u;
            let e = assign_e[n * assign_e.stride(0)];
            let w = sel_w[n * sel_w.stride(0)];

            let gs0 = i64::cast_from(gu.stride(0));
            let gs1 = i64::cast_from(gu.stride(1));
            let qs0 = i64::cast_from(q_dn.stride(0));
            let qs1 = i64::cast_from(q_dn.stride(1));
            let qs2 = i64::cast_from(q_dn.stride(2));
            let ss0 = i64::cast_from(s_dn.stride(0));
            let ss1 = i64::cast_from(s_dn.stride(1));
            let e_i = i64::cast_from(e);
            let n_i = i64::cast_from(n);
            let h_i = i64::cast_from(hh);
            let gu_base = n_i * gs0;
            let q_base = e_i * qs0 + h_i * qs2;
            let sd = s_dn[usize::cast_from(e_i * ss0 + h_i * ss1)];

            let mut acc = f32::new(0.0);
            for ci in 0..i_dim_u {
                let i_i = i64::cast_from(ci);
                let qb = q_dn[usize::cast_from(q_base + i_i * qs1)];
                let dv = f32::cast_from(e4m3::reinterpret(qb)) * sd;
                acc += gu[usize::cast_from(gu_base + i_i * gs1)] * dv;
            }
            out[pos] = acc * w;
        }
    }

    // ============================================================================================
    // LEVER (c) phase 2 — SPLIT-K, VECTORIZED, REGISTER-BLOCKED gather-GEMV (the 3-voice-vetted
    // kernel). Same MATH as `fused_swiglu_gu` / `fused_swiglu_down`, same persistent-stack reads,
    // but it attacks the LATENCY-bound scalar GEMV with three levers at once, each weight byte still
    // read exactly once, coalesced:
    //   * VECTORIZE — the weight stacks ride as `&Tensor<Line<EW>>` (`as_tensor_arg(V)`), so each
    //     load is one `LDG.E.128` of `V` contiguous bf16 (V=8 → 128-bit) instead of `V` scalar
    //     16-bit loads. The `V` lanes ARE the register block over the CONTIGUOUS output axis.
    //   * BREAK THE FMA CHAIN — the accumulator is a `Line<f32>` of width `V` = `V` independent f32
    //     accumulators, so the loop-carried dependency the scalar kernel hit is `V`-way ILP.
    //   * SPLIT-K — `CubeDim::new_2d(BX, KSPLIT)`: the `BX` x-lanes tile the contiguous OUTPUT axis
    //     (one warp = BX contiguous output lines → fully coalesced 128-bit loads), the `KSPLIT`
    //     y-warps partition the STRIDED REDUCTION axis so each thread's dependency chain is only
    //     `K/KSPLIT` long. The `KSPLIT` partial `Line`s are summed across warps through
    //     `SharedMemory<Line<f32>>` + `sync_cube` (cross-warp, NOT plane_sum — the warp is on the
    //     output axis, not the reduction axis), deterministically (kk=0..KSPLIT), in f32.
    //
    // PARITY NOTE (P0, gated): the KSPLIT cross-warp sum reorders the f32 reduction vs the scalar
    // oracle's sequential h=0..H / i=0..I — stays f32 (within ~1e-6 rel), validated <1e-4 +
    // token-identical by `decode_topk_fused_equals_oracle_cuda`. CONTIGUITY PRECONDITION: the line
    // size is probed on the host with `try_tensor_line_size_parallel` (the INNERMOST stride-1
    // output axis); if a stack is not innermost-contiguous/V-aligned the probe returns 1 and the
    // host dispatches the SCALAR kernels above (correct fallback). V (=`gate.line_size()`) is
    // COMPTIME inside the kernel; the line size of the tensor arg monomorphizes one kernel per V.

    /// SPLIT-K `gu`: `gu[n,i] = silu(Σ_h x[tok,h]·gate[e,h,i]) · (Σ_h x[tok,h]·up[e,h,i])`.
    /// Output axis = I (vectorized, `BX` lanes × `V`); reduction axis = H (`KSPLIT`-split over y).
    /// `tiles_per_n = ceil((I/V) / BX)`; grid = `Static(N*tiles_per_n,1,1)`, block = `(BX, KSPLIT)`.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused_swiglu_gu_splitk<EW: Float>(
        x: &Tensor<f32>,            // [T,H] activations (contiguous f32, scalar reads)
        gate: &Tensor<Line<EW>>,    // [E,H,I] gate stack, lined V over the innermost (I) axis
        up: &Tensor<Line<EW>>,      // [E,H,I] up stack, lined V over I
        assign_e: &Tensor<i32>,     // [N] expert id per assignment
        assign_tok: &Tensor<i32>,   // [N] token id per assignment
        gu: &mut Tensor<Line<f32>>, // [N,I] silu(x·gate)·(x·up), lined V over I
        h_dim: u32,
        i_dim: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = gate.line_size(); // comptime line size (V)
        let lx = UNIT_POS_X; // 0..BX — output-line lane (tiles the contiguous I axis)
        let ky = UNIT_POS_Y; // 0..KSPLIT — reduction stripe over H
        let blk = CUBE_POS_X;

        let n = (blk / tiles_per_n) as usize; // assignment
        let tile = blk % tiles_per_n; // output-tile within the assignment
        let out_line = (tile * bx + lx) as usize; // output LINE index along I (V outputs)
        let o_lines = (i_dim as usize) / v; // total output lines per assignment
        let active = out_line < o_lines; // mask the partial last tile (no OOB)

        // V independent f32 accumulators (the register block + the ILP that breaks the FMA chain).
        let mut gacc = Line::<f32>::empty(v).fill(0f32);
        let mut uacc = Line::<f32>::empty(v).fill(0f32);

        if active {
            let e = assign_e[n * assign_e.stride(0)];
            let tok = assign_tok[n * assign_tok.stride(0)];
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let v_i = i64::cast_from(v);

            // x[tok, h] element offsets (x is scalar/line-1; its own strides).
            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let x_base = tok_i * xs0;

            // Weight strides in LINE units (exact: contiguous + V-aligned ⇒ stride(0..1) % V == 0).
            let gs0l = i64::cast_from(gate.stride(0)) / v_i;
            let gs1l = i64::cast_from(gate.stride(1)) / v_i;
            let us0l = i64::cast_from(up.stride(0)) / v_i;
            let us1l = i64::cast_from(up.stride(1)) / v_i;
            let out_line_i = i64::cast_from(out_line);
            let g_base = e_i * gs0l + out_line_i; // gate[e, h=0, out_line] (line index)
            let u_base = e_i * us0l + out_line_i; // up[e, h=0, out_line]

            // Reduction stripe: this thread sums h = ky, ky+KSPLIT, ... < H (chain H→H/KSPLIT).
            let h_u = h_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (h_u + ks - 1) / ks; // ceil(H/KSPLIT)
            for j in 0..n_steps {
                let h = (ky as usize) + j * ks;
                if h < h_u {
                    let h_i = i64::cast_from(h);
                    let xv = x[usize::cast_from(x_base + h_i * xs1)]; // scalar f32
                    let xl = Line::<f32>::empty(v).fill(xv); // broadcast scalar → V lanes (fix #2)
                    let gline = Line::<f32>::cast_from(gate[usize::cast_from(g_base + h_i * gs1l)]);
                    let uline = Line::<f32>::cast_from(up[usize::cast_from(u_base + h_i * us1l)]);
                    gacc += gline * xl; // V-wide f32 FMA (V independent accumulators = ILP)
                    uacc += uline * xl;
                }
            }
        }

        // Cross-warp KSPLIT reduction through shared memory (declared by all threads in the block).
        let mut shared_g = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let mut shared_u = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let sidx = (ky * bx + lx) as usize;
        shared_g[sidx] = gacc;
        shared_u[sidx] = uacc;
        sync_cube(); // ALL threads reach this (mask is outside the sync)

        if ky == 0 && active {
            let mut g = shared_g[lx as usize];
            let mut u = shared_u[lx as usize];
            for kk in 1..ksplit {
                let off = (kk * bx + lx) as usize;
                g += shared_g[off];
                u += shared_u[off];
            }
            // silu(g) = g / (1 + exp(-g)); fused with the up gate — vectorized over the V lanes.
            // Scalars broadcast via Line::empty(v).fill (the Line·scalar operators are unexpanded).
            let denom =
                (Line::<f32>::empty(v).fill(0.0f32) - g).exp() + Line::<f32>::empty(v).fill(1.0f32);
            let sig = Line::<f32>::empty(v).fill(1.0f32) / denom;
            gu[n * o_lines + out_line] = g * sig * u; // one V-wide store
        }
    }

    /// SPLIT-K `down`: `out[n,h] = sel_w[n] · Σ_i gu[n,i]·down[e,i,h]`.
    /// Output axis = H (vectorized, `BX` lanes × `V`); reduction axis = I (`KSPLIT`-split over y).
    /// `tiles_per_n = ceil((H/V) / BX)`; grid = `Static(N*tiles_per_n,1,1)`, block = `(BX, KSPLIT)`.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused_swiglu_down_splitk<EW: Float>(
        gu: &Tensor<f32>,            // [N,I] (scalar reads, reduction axis)
        down: &Tensor<Line<EW>>,     // [E,I,H] down stack, lined V over the innermost (H) axis
        assign_e: &Tensor<i32>,      // [N] expert id per assignment
        sel_w: &Tensor<f32>,         // [N] router weight per assignment
        out: &mut Tensor<Line<f32>>, // [N,H] weighted output, lined V over H
        h_dim: u32,
        i_dim: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = down.line_size(); // comptime line size (V)
        let lx = UNIT_POS_X; // 0..BX — output-line lane (tiles the contiguous H axis)
        let ky = UNIT_POS_Y; // 0..KSPLIT — reduction stripe over I
        let blk = CUBE_POS_X;

        let n = (blk / tiles_per_n) as usize;
        let tile = blk % tiles_per_n;
        let out_line = (tile * bx + lx) as usize; // output LINE index along H (V outputs)
        let o_lines = (h_dim as usize) / v;
        let active = out_line < o_lines;

        let mut acc = Line::<f32>::empty(v).fill(0f32);

        if active {
            let e = assign_e[n * assign_e.stride(0)];
            let e_i = i64::cast_from(e);
            let v_i = i64::cast_from(v);

            let gus0 = i64::cast_from(gu.stride(0));
            let gus1 = i64::cast_from(gu.stride(1));
            let gu_base = i64::cast_from(n) * gus0;

            let ds0l = i64::cast_from(down.stride(0)) / v_i;
            let ds1l = i64::cast_from(down.stride(1)) / v_i;
            let out_line_i = i64::cast_from(out_line);
            let d_base = e_i * ds0l + out_line_i; // down[e, i=0, out_line] (line index)

            let i_u = i_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (i_u + ks - 1) / ks; // ceil(I/KSPLIT)
            for j in 0..n_steps {
                let ii = (ky as usize) + j * ks;
                if ii < i_u {
                    let ii_i = i64::cast_from(ii);
                    let gv = gu[usize::cast_from(gu_base + ii_i * gus1)]; // scalar f32
                    let gl = Line::<f32>::empty(v).fill(gv); // broadcast scalar → V lanes (fix #2)
                    let dline =
                        Line::<f32>::cast_from(down[usize::cast_from(d_base + ii_i * ds1l)]);
                    acc += dline * gl;
                }
            }
        }

        let mut shared_a = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let sidx = (ky * bx + lx) as usize;
        shared_a[sidx] = acc;
        sync_cube();

        if ky == 0 && active {
            let mut a = shared_a[lx as usize];
            for kk in 1..ksplit {
                a += shared_a[(kk * bx + lx) as usize];
            }
            let w = sel_w[n * sel_w.stride(0)]; // router weight in-register (host combine = pure ADD)
            let wl = Line::<f32>::empty(v).fill(w); // broadcast scalar → V lanes (fix #2)
            out[n * o_lines + out_line] = a * wl; // one V-wide store
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_fp8_splitk(
        x: &Tensor<f32>,            // [T,H]
        q_gu: &Tensor<Line<u8>>,    // [E,H,2I], lined over output axis
        s_gu: &Tensor<Line<f32>>,   // [E,2I], lined over output axis
        assign_e: &Tensor<i32>,     // [N]
        gu: &mut Tensor<Line<f32>>, // [N,I], lined over I
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = q_gu.line_size();
        let lx = UNIT_POS_X;
        let ky = UNIT_POS_Y;
        let blk = CUBE_POS_X;
        let n = (blk / tiles_per_n) as usize;
        let tile = blk % tiles_per_n;
        let out_line = (tile * bx + lx) as usize;
        let o_lines = (i_dim as usize) / v;
        let active = out_line < o_lines;

        let mut gacc = Line::<f32>::empty(v).fill(0f32);
        let mut uacc = Line::<f32>::empty(v).fill(0f32);
        if active {
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let v_i = i64::cast_from(v);

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let x_base = tok_i * xs0;
            let qs0l = i64::cast_from(q_gu.stride(0)) / v_i;
            let qs1l = i64::cast_from(q_gu.stride(1)) / v_i;
            let ss0l = i64::cast_from(s_gu.stride(0)) / v_i;
            let out_line_i = i64::cast_from(out_line);
            let o_lines_i = i64::cast_from(o_lines);
            let g_base = e_i * qs0l + out_line_i;
            let u_base = e_i * qs0l + out_line_i + o_lines_i;
            let sg = s_gu[usize::cast_from(e_i * ss0l + out_line_i)];
            let su = s_gu[usize::cast_from(e_i * ss0l + out_line_i + o_lines_i)];

            let h_u = h_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (h_u + ks - 1) / ks;
            for j in 0..n_steps {
                let h = (ky as usize) + j * ks;
                if h < h_u {
                    let h_i = i64::cast_from(h);
                    let xv = x[usize::cast_from(x_base + h_i * xs1)];
                    let xl = Line::<f32>::empty(v).fill(xv);
                    let gb = q_gu[usize::cast_from(g_base + h_i * qs1l)];
                    let ub = q_gu[usize::cast_from(u_base + h_i * qs1l)];
                    let gl = Line::<f32>::cast_from(Line::<e4m3>::reinterpret(gb)) * sg;
                    let ul = Line::<f32>::cast_from(Line::<e4m3>::reinterpret(ub)) * su;
                    gacc += gl * xl;
                    uacc += ul * xl;
                }
            }
        }

        let mut shared_g = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let mut shared_u = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let sidx = (ky * bx + lx) as usize;
        shared_g[sidx] = gacc;
        shared_u[sidx] = uacc;
        sync_cube();

        if ky == 0 && active {
            let mut g = shared_g[lx as usize];
            let mut u = shared_u[lx as usize];
            for kk in 1..ksplit {
                let off = (kk * bx + lx) as usize;
                g += shared_g[off];
                u += shared_u[off];
            }
            let denom =
                (Line::<f32>::empty(v).fill(0.0f32) - g).exp() + Line::<f32>::empty(v).fill(1.0f32);
            let sig = Line::<f32>::empty(v).fill(1.0f32) / denom;
            gu[n * o_lines + out_line] = g * sig * u;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_down_fp8_splitk(
        gu: &Tensor<f32>,            // [N,I]
        q_dn: &Tensor<Line<u8>>,     // [E,I,H], lined over H
        s_dn: &Tensor<Line<f32>>,    // [E,H], lined over H
        assign_e: &Tensor<i32>,      // [N]
        sel_w: &Tensor<f32>,         // [N]
        out: &mut Tensor<Line<f32>>, // [N,H], lined over H
        h_dim: u32,
        i_dim: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = q_dn.line_size();
        let lx = UNIT_POS_X;
        let ky = UNIT_POS_Y;
        let blk = CUBE_POS_X;
        let n = (blk / tiles_per_n) as usize;
        let tile = blk % tiles_per_n;
        let out_line = (tile * bx + lx) as usize;
        let o_lines = (h_dim as usize) / v;
        let active = out_line < o_lines;
        let mut acc = Line::<f32>::empty(v).fill(0f32);

        if active {
            let e = assign_e[n * assign_e.stride(0)];
            let e_i = i64::cast_from(e);
            let v_i = i64::cast_from(v);

            let gus0 = i64::cast_from(gu.stride(0));
            let gus1 = i64::cast_from(gu.stride(1));
            let gu_base = i64::cast_from(n) * gus0;
            let qs0l = i64::cast_from(q_dn.stride(0)) / v_i;
            let qs1l = i64::cast_from(q_dn.stride(1)) / v_i;
            let ss0l = i64::cast_from(s_dn.stride(0)) / v_i;
            let out_line_i = i64::cast_from(out_line);
            let q_base = e_i * qs0l + out_line_i;
            let sd = s_dn[usize::cast_from(e_i * ss0l + out_line_i)];

            let i_u = i_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (i_u + ks - 1) / ks;
            for j in 0..n_steps {
                let ii = (ky as usize) + j * ks;
                if ii < i_u {
                    let ii_i = i64::cast_from(ii);
                    let gv = gu[usize::cast_from(gu_base + ii_i * gus1)];
                    let gl = Line::<f32>::empty(v).fill(gv);
                    let qb = q_dn[usize::cast_from(q_base + ii_i * qs1l)];
                    let dl = Line::<f32>::cast_from(Line::<e4m3>::reinterpret(qb)) * sd;
                    acc += dl * gl;
                }
            }
        }

        let mut shared_a = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, v);
        let sidx = (ky * bx + lx) as usize;
        shared_a[sidx] = acc;
        sync_cube();

        if ky == 0 && active {
            let mut a = shared_a[lx as usize];
            for kk in 1..ksplit {
                a += shared_a[(kk * bx + lx) as usize];
            }
            let wl = Line::<f32>::empty(v).fill(sel_w[n * sel_w.stride(0)]);
            out[n * o_lines + out_line] = a * wl;
        }
    }

    /// Decode one E2M1 nibble using the Marlin NVFP4 trick: shift the 4-bit code into the high
    /// nibble, reinterpret the masked bits as E4M3, then apply the fixed exponent-bias correction.
    #[cube]
    pub fn e2m1_marlin_decode(code: u32) -> f32 {
        let top = (code & 15u32) << 4u32;
        let fp8_bits = (top & 128u32) | ((top & 112u32) >> 2u32);
        f32::cast_from(e4m3::reinterpret(u8::cast_from(fp8_bits))) * f32::new(64.0)
    }

    #[cube]
    pub fn nvfp4_dequant_nibble(packed: u32, high: bool, block_scale: u8, gscale: f32) -> f32 {
        let code = if high {
            (packed >> 4u32) & 15u32
        } else {
            packed & 15u32
        };
        e2m1_marlin_decode(code) * f32::cast_from(e4m3::reinterpret(block_scale)) * gscale
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_nvfp4_scalar(
        x: &Tensor<f32>,         // [T,H]
        q_gu: &Tensor<u8>,       // [E,H,I] output-major e2m1 bytes for logical [E,H,2I]
        bs_gu: &Tensor<u8>,      // [E,2I,H/16] raw e4m3 block scales
        gscale_gu: &Tensor<f32>, // [E,2] gate/up global scales
        assign_e: &Tensor<i32>,  // [N]
        gu: &mut Tensor<f32>,    // [N,I]
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
    ) {
        if ABSOLUTE_POS < gu.len() {
            let pos = ABSOLUTE_POS as usize;
            let i_dim_u = i_dim as usize;
            let h_dim_u = h_dim as usize;
            let n = pos / i_dim_u;
            let ci = pos % i_dim_u;
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let qs0 = i64::cast_from(q_gu.stride(0));
            let qs1 = i64::cast_from(q_gu.stride(1));
            let qs2 = i64::cast_from(q_gu.stride(2));
            let bs0 = i64::cast_from(bs_gu.stride(0));
            let bs1 = i64::cast_from(bs_gu.stride(1));
            let bs2 = i64::cast_from(bs_gu.stride(2));
            let gs0 = i64::cast_from(gscale_gu.stride(0));
            let gs1 = i64::cast_from(gscale_gu.stride(1));
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let ci_i = i64::cast_from(ci);
            let half_bytes = i64::cast_from(i_dim_u / 2usize);
            let byte_i = i64::cast_from(ci / 2usize);
            let high = (ci & 1usize) == 1usize;
            let x_base = tok_i * xs0;
            let g_q_base = e_i * qs0 + byte_i * qs2;
            let u_q_base = e_i * qs0 + (half_bytes + byte_i) * qs2;
            let g_bs_base = e_i * bs0 + ci_i * bs1;
            let u_bs_base = e_i * bs0 + (ci_i + i64::cast_from(i_dim_u)) * bs1;
            let g_gscale = gscale_gu[usize::cast_from(e_i * gs0)];
            let u_gscale = gscale_gu[usize::cast_from(e_i * gs0 + gs1)];

            let mut gacc = f32::new(0.0);
            let mut uacc = f32::new(0.0);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let block_i = i64::cast_from(hh / 16usize);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                let gb = u32::cast_from(q_gu[usize::cast_from(g_q_base + h_i * qs1)]);
                let ub = u32::cast_from(q_gu[usize::cast_from(u_q_base + h_i * qs1)]);
                let gs = bs_gu[usize::cast_from(g_bs_base + block_i * bs2)];
                let us = bs_gu[usize::cast_from(u_bs_base + block_i * bs2)];
                gacc += xv * nvfp4_dequant_nibble(gb, high, gs, g_gscale);
                uacc += xv * nvfp4_dequant_nibble(ub, high, us, u_gscale);
            }
            let sig = 1.0f32 / (1.0f32 + (0.0f32 - gacc).exp());
            gu[pos] = gacc * sig * uacc;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_down_nvfp4_scalar(
        gu: &Tensor<f32>,        // [N,I]
        q_dn: &Tensor<u8>,       // [E,I,H/2] output-major e2m1 bytes for logical [E,I,H]
        bs_dn: &Tensor<u8>,      // [E,H,I/16] raw e4m3 block scales
        gscale_dn: &Tensor<f32>, // [E]
        assign_e: &Tensor<i32>,  // [N]
        sel_w: &Tensor<f32>,     // [N]
        out: &mut Tensor<f32>,   // [N,H]
        h_dim: u32,
        i_dim: u32,
    ) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let h_dim_u = h_dim as usize;
            let i_dim_u = i_dim as usize;
            let n = pos / h_dim_u;
            let hh = pos % h_dim_u;
            let e = assign_e[n * assign_e.stride(0)];
            let w = sel_w[n * sel_w.stride(0)];

            let gus0 = i64::cast_from(gu.stride(0));
            let gus1 = i64::cast_from(gu.stride(1));
            let qs0 = i64::cast_from(q_dn.stride(0));
            let qs1 = i64::cast_from(q_dn.stride(1));
            let qs2 = i64::cast_from(q_dn.stride(2));
            let bs0 = i64::cast_from(bs_dn.stride(0));
            let bs1 = i64::cast_from(bs_dn.stride(1));
            let bs2 = i64::cast_from(bs_dn.stride(2));
            let e_i = i64::cast_from(e);
            let n_i = i64::cast_from(n);
            let h_i = i64::cast_from(hh);
            let byte_i = i64::cast_from(hh / 2usize);
            let high = (hh & 1usize) == 1usize;
            let gu_base = n_i * gus0;
            let q_base = e_i * qs0 + byte_i * qs2;
            let bs_base = e_i * bs0 + h_i * bs1;
            let gd = gscale_dn[e as usize * gscale_dn.stride(0)];

            let mut acc = f32::new(0.0);
            for ci in 0..i_dim_u {
                let ci_i = i64::cast_from(ci);
                let block_i = i64::cast_from(ci / 16usize);
                let qb = u32::cast_from(q_dn[usize::cast_from(q_base + ci_i * qs1)]);
                let sb = bs_dn[usize::cast_from(bs_base + block_i * bs2)];
                acc += gu[usize::cast_from(gu_base + ci_i * gus1)]
                    * nvfp4_dequant_nibble(qb, high, sb, gd);
            }
            out[pos] = acc * w;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_nvfp4_splitk(
        x: &Tensor<f32>,            // [T,H]
        q_gu: &Tensor<Line<u8>>,    // [E,H,I] bytes; V bytes => 2V logical channels
        bs_gu: &Tensor<u8>,         // [E,2I,H/16]
        gscale_gu: &Tensor<f32>,    // [E,2]
        assign_e: &Tensor<i32>,     // [N]
        gu: &mut Tensor<Line<f32>>, // [N,I], lined over 2V logical channels
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = q_gu.line_size(); // bytes per line
        let vo = gu.line_size(); // logical output channels accumulated/stored by this thread (=2*V)
        let lx = UNIT_POS_X;
        let ky = UNIT_POS_Y;
        let blk = CUBE_POS_X;
        let n = (blk / tiles_per_n) as usize;
        let tile = blk % tiles_per_n;
        let out_line = (tile * bx + lx) as usize; // line over 2V logical I channels
        let o_lines = (i_dim as usize) / vo;
        let active = out_line < o_lines;

        let mut gacc = Line::<f32>::empty(vo).fill(0f32);
        let mut uacc = Line::<f32>::empty(vo).fill(0f32);
        if active {
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let v_i = i64::cast_from(v);
            let vo_i = i64::cast_from(vo);

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let qs0l = i64::cast_from(q_gu.stride(0)) / v_i;
            let qs1l = i64::cast_from(q_gu.stride(1)) / v_i;
            let bs0 = i64::cast_from(bs_gu.stride(0));
            let bs1 = i64::cast_from(bs_gu.stride(1));
            let bs2 = i64::cast_from(bs_gu.stride(2));
            let gs0 = i64::cast_from(gscale_gu.stride(0));
            let gs1 = i64::cast_from(gscale_gu.stride(1));
            let x_base = tok_i * xs0;
            let out_line_i = i64::cast_from(out_line);
            let o_lines_i = i64::cast_from(o_lines);
            let chan0 = out_line_i * vo_i;
            let g_q_base = e_i * qs0l + out_line_i;
            let u_q_base = e_i * qs0l + out_line_i + o_lines_i;
            let g_gscale = gscale_gu[usize::cast_from(e_i * gs0)];
            let u_gscale = gscale_gu[usize::cast_from(e_i * gs0 + gs1)];

            let h_u = h_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (h_u + ks - 1) / ks;
            for j in 0..n_steps {
                let h = (ky as usize) + j * ks;
                if h < h_u {
                    let h_i = i64::cast_from(h);
                    let block_i = i64::cast_from(j); // KSPLIT==16, so h=ky+16*j is block j.
                    let xv = x[usize::cast_from(x_base + h_i * xs1)];
                    let gq = q_gu[usize::cast_from(g_q_base + h_i * qs1l)];
                    let uq = q_gu[usize::cast_from(u_q_base + h_i * qs1l)];
                    for lane in 0..v {
                        let byte_chan = chan0 + i64::cast_from(lane * 2usize);
                        let gp = u32::cast_from(gq[lane]);
                        let up = u32::cast_from(uq[lane]);
                        let gs0b =
                            bs_gu[usize::cast_from(e_i * bs0 + byte_chan * bs1 + block_i * bs2)];
                        let gs1b = bs_gu[usize::cast_from(
                            e_i * bs0 + (byte_chan + 1i64) * bs1 + block_i * bs2,
                        )];
                        let us0b = bs_gu[usize::cast_from(
                            e_i * bs0
                                + (byte_chan + i64::cast_from(i_dim as usize)) * bs1
                                + block_i * bs2,
                        )];
                        let us1b = bs_gu[usize::cast_from(
                            e_i * bs0
                                + (byte_chan + i64::cast_from(i_dim as usize) + 1i64) * bs1
                                + block_i * bs2,
                        )];
                        let lo = lane * 2usize;
                        gacc[lo] += xv * nvfp4_dequant_nibble(gp, false, gs0b, g_gscale);
                        gacc[lo + 1usize] += xv * nvfp4_dequant_nibble(gp, true, gs1b, g_gscale);
                        uacc[lo] += xv * nvfp4_dequant_nibble(up, false, us0b, u_gscale);
                        uacc[lo + 1usize] += xv * nvfp4_dequant_nibble(up, true, us1b, u_gscale);
                    }
                }
            }
        }

        let mut shared_g = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, vo);
        let mut shared_u = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, vo);
        let sidx = (ky * bx + lx) as usize;
        shared_g[sidx] = gacc;
        shared_u[sidx] = uacc;
        sync_cube();

        if ky == 0 && active {
            let mut g = shared_g[lx as usize];
            let mut u = shared_u[lx as usize];
            for kk in 1..ksplit {
                let off = (kk * bx + lx) as usize;
                g += shared_g[off];
                u += shared_u[off];
            }
            let denom = (Line::<f32>::empty(vo).fill(0.0f32) - g).exp()
                + Line::<f32>::empty(vo).fill(1.0f32);
            let sig = Line::<f32>::empty(vo).fill(1.0f32) / denom;
            gu[n * o_lines + out_line] = g * sig * u;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_nvfp4_projection_probe(
        x: &Tensor<f32>,         // [T,H]
        q_gu: &Tensor<u8>,       // [E,H,I] output-major e2m1 bytes for logical [E,H,2I]
        bs_gu: &Tensor<u8>,      // [E,2I,H/16] raw e4m3 block scales
        gscale_gu: &Tensor<f32>, // [E,2]
        assign_e: &Tensor<i32>,  // [N]
        gate: &mut Tensor<f32>,  // [N,I]
        up: &mut Tensor<f32>,    // [N,I]
        h_dim: u32,
        i_dim: u32,
        top_k: u32,
    ) {
        if ABSOLUTE_POS < gate.len() {
            let pos = ABSOLUTE_POS as usize;
            let i_dim_u = i_dim as usize;
            let h_dim_u = h_dim as usize;
            let n = pos / i_dim_u;
            let ci = pos % i_dim_u;
            let tok = n / (top_k as usize);
            let e = assign_e[n * assign_e.stride(0)];

            let xs0 = i64::cast_from(x.stride(0));
            let xs1 = i64::cast_from(x.stride(1));
            let qs0 = i64::cast_from(q_gu.stride(0));
            let qs1 = i64::cast_from(q_gu.stride(1));
            let qs2 = i64::cast_from(q_gu.stride(2));
            let bs0 = i64::cast_from(bs_gu.stride(0));
            let bs1 = i64::cast_from(bs_gu.stride(1));
            let bs2 = i64::cast_from(bs_gu.stride(2));
            let gs0 = i64::cast_from(gscale_gu.stride(0));
            let gs1 = i64::cast_from(gscale_gu.stride(1));
            let e_i = i64::cast_from(e);
            let tok_i = i64::cast_from(tok);
            let ci_i = i64::cast_from(ci);
            let half_bytes = i64::cast_from(i_dim_u / 2usize);
            let byte_i = i64::cast_from(ci / 2usize);
            let high = (ci & 1usize) == 1usize;
            let x_base = tok_i * xs0;
            let g_q_base = e_i * qs0 + byte_i * qs2;
            let u_q_base = e_i * qs0 + (half_bytes + byte_i) * qs2;
            let g_bs_base = e_i * bs0 + ci_i * bs1;
            let u_bs_base = e_i * bs0 + (ci_i + i64::cast_from(i_dim_u)) * bs1;
            let g_gscale = gscale_gu[usize::cast_from(e_i * gs0)];
            let u_gscale = gscale_gu[usize::cast_from(e_i * gs0 + gs1)];

            let mut gacc = f32::new(0.0);
            let mut uacc = f32::new(0.0);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let block_i = i64::cast_from(hh / 16usize);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                let gb = u32::cast_from(q_gu[usize::cast_from(g_q_base + h_i * qs1)]);
                let ub = u32::cast_from(q_gu[usize::cast_from(u_q_base + h_i * qs1)]);
                let gs = bs_gu[usize::cast_from(g_bs_base + block_i * bs2)];
                let us = bs_gu[usize::cast_from(u_bs_base + block_i * bs2)];
                gacc += xv * nvfp4_dequant_nibble(gb, high, gs, g_gscale);
                uacc += xv * nvfp4_dequant_nibble(ub, high, us, u_gscale);
            }
            gate[pos] = gacc;
            up[pos] = uacc;
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_down_nvfp4_splitk(
        gu: &Tensor<f32>,            // [N,I]
        q_dn: &Tensor<Line<u8>>,     // [E,I,H/2] bytes; V bytes => 2V logical H channels
        bs_dn: &Tensor<u8>,          // [E,H,I/16]
        gscale_dn: &Tensor<f32>,     // [E]
        assign_e: &Tensor<i32>,      // [N]
        sel_w: &Tensor<f32>,         // [N]
        out: &mut Tensor<Line<f32>>, // [N,H], lined over 2V logical channels
        h_dim: u32,
        i_dim: u32,
        tiles_per_n: u32,
        #[comptime] ksplit: u32,
        #[comptime] bx: u32,
    ) {
        let v = q_dn.line_size();
        let vo = out.line_size(); // logical output channels accumulated/stored by this thread (=2*V)
        let lx = UNIT_POS_X;
        let ky = UNIT_POS_Y;
        let blk = CUBE_POS_X;
        let n = (blk / tiles_per_n) as usize;
        let tile = blk % tiles_per_n;
        let out_line = (tile * bx + lx) as usize;
        let o_lines = (h_dim as usize) / vo;
        let active = out_line < o_lines;
        let mut acc = Line::<f32>::empty(vo).fill(0f32);

        if active {
            let e = assign_e[n * assign_e.stride(0)];
            let e_i = i64::cast_from(e);
            let v_i = i64::cast_from(v);
            let vo_i = i64::cast_from(vo);

            let gus0 = i64::cast_from(gu.stride(0));
            let gus1 = i64::cast_from(gu.stride(1));
            let qs0l = i64::cast_from(q_dn.stride(0)) / v_i;
            let qs1l = i64::cast_from(q_dn.stride(1)) / v_i;
            let bs0 = i64::cast_from(bs_dn.stride(0));
            let bs1 = i64::cast_from(bs_dn.stride(1));
            let bs2 = i64::cast_from(bs_dn.stride(2));
            let gu_base = i64::cast_from(n) * gus0;
            let out_line_i = i64::cast_from(out_line);
            let chan0 = out_line_i * vo_i;
            let q_base = e_i * qs0l + out_line_i;
            let gd = gscale_dn[e as usize * gscale_dn.stride(0)];

            let i_u = i_dim as usize;
            let ks = ksplit as usize;
            let n_steps = (i_u + ks - 1) / ks;
            for j in 0..n_steps {
                let ii = (ky as usize) + j * ks;
                if ii < i_u {
                    let ii_i = i64::cast_from(ii);
                    let block_i = i64::cast_from(j); // KSPLIT==16, so ii=ky+16*j is block j.
                    let gv = gu[usize::cast_from(gu_base + ii_i * gus1)];
                    let qv = q_dn[usize::cast_from(q_base + ii_i * qs1l)];
                    for lane in 0..v {
                        let byte_chan = chan0 + i64::cast_from(lane * 2usize);
                        let packed = u32::cast_from(qv[lane]);
                        let s0 =
                            bs_dn[usize::cast_from(e_i * bs0 + byte_chan * bs1 + block_i * bs2)];
                        let s1 = bs_dn[usize::cast_from(
                            e_i * bs0 + (byte_chan + 1i64) * bs1 + block_i * bs2,
                        )];
                        let lo = lane * 2usize;
                        acc[lo] += gv * nvfp4_dequant_nibble(packed, false, s0, gd);
                        acc[lo + 1usize] += gv * nvfp4_dequant_nibble(packed, true, s1, gd);
                    }
                }
            }
        }

        let mut shared_a = SharedMemory::<f32>::new_lined((ksplit * bx) as usize, vo);
        let sidx = (ky * bx + lx) as usize;
        shared_a[sidx] = acc;
        sync_cube();

        if ky == 0 && active {
            let mut a = shared_a[lx as usize];
            for kk in 1..ksplit {
                a += shared_a[(kk * bx + lx) as usize];
            }
            let wl = Line::<f32>::empty(vo).fill(sel_w[n * sel_w.stride(0)]);
            out[n * o_lines + out_line] = a * wl;
        }
    }
}

// =================================================================================================
// On-device dropless align/sort (Burn ops). Cross-backend-agnostic in spirit, but fixed to the
// default `Cuda` backend here since the grouped GEMM that consumes it is CUDA-only.
// =================================================================================================

/// The vLLM `moe_align_block_size` layout, built fully on-device (no host sync).
pub struct DroplessLayout {
    /// `[buffer]` i32 — token id per slot, `−1` for padding slots.
    pub sorted_token: Tensor<Cuda, 1, Int>,
    /// `[buffer]` f32 — router weight per slot, `0` for padding slots.
    pub sorted_weight: Tensor<Cuda, 1>,
    /// `[buffer]` i32 — expert id per slot, `−1` for padding slots.
    pub sorted_expert: Tensor<Cuda, 1, Int>,
    /// `[num_blocks]` i32 — expert id per `BLOCK_M`-block, `−1` for empty/tail blocks.
    pub expert_ids: Tensor<Cuda, 1, Int>,
    /// `[E]` i32 — per-expert assignment count (the dropless invariant: `Σ count_e == N`).
    pub count_e: Tensor<Cuda, 1, Int>,
    /// Number of `BLOCK_M`-blocks launched (a safe host upper bound `E + ceil(N/BLOCK_M)`).
    pub num_blocks: usize,
    /// `num_blocks * BLOCK_M`.
    pub buffer: usize,
    /// `T*k` — total routed assignments (no drop: all land in distinct real slots).
    pub n: usize,
}

/// Build the DROPLESS `moe_align_block_size` layout from the compact top-k routing
/// `(sel_idx [T,k], sel_w [T,k])`. `block_m` is the segment alignment (vLLM `BLOCK_M`).
pub fn dropless_align(
    sel_idx: Tensor<Cuda, 2, Int>,
    sel_w: Tensor<Cuda, 2>,
    num_experts: usize,
    top_k: usize,
    block_m: usize,
) -> DroplessLayout {
    let [t, k] = sel_idx.dims();
    assert_eq!(k, top_k, "sel_idx top-k dim ({k}) != top_k ({top_k})");
    let e = num_experts;
    let n = t * k;
    let bm = block_m as i64;
    let device = sel_idx.device();

    // Flatten the N=T*k assignments.
    let assign_e = sel_idx.reshape([n]); // [N] expert id per assignment
    let assign_tok = Tensor::<Cuda, 1, Int>::arange(0..t as i64, &device)
        .reshape([t, 1])
        .repeat(&[1, k])
        .reshape([n]); // [N] token per assignment
    let assign_w = sel_w.reshape([n]); // [N] router weight

    // On-device one-hot via (arange == expert) — NOT Tensor::one_hot (host round-trip).
    let experts_row = Tensor::<Cuda, 1, Int>::arange(0..e as i64, &device).reshape([1, e]); // [1,E]
    let oh = assign_e.clone().reshape([n, 1]).equal(experts_row).int(); // [N,E] 0/1

    // count_e[E], padded_e[E] = round_up(count_e, BLOCK_M), base_e[E] = ExclusiveCumsum(padded_e).
    let count_e = oh.clone().sum_dim(0).reshape([e]); // [E]
    let padded_e = count_e
        .clone()
        .add_scalar(bm - 1)
        .div_scalar(bm)
        .mul_scalar(bm); // ceil(count/BM)*BM
    let base_e = padded_e.clone().cumsum(0) - padded_e; // exclusive cumsum [E]

    // within-expert rank = (inclusive cumsum down N, read at own expert) − 1.
    let run = oh.cumsum(0); // [N,E]
    let rank = run
        .gather(1, assign_e.clone().reshape([n, 1]))
        .reshape([n])
        .add_scalar(-1i64); // [N], 0-indexed

    // dest = base_e[expert] + rank — UNIQUE per assignment ⇒ DROPLESS (every assignment placed).
    let dest = base_e.gather(0, assign_e.clone()) + rank; // [N]

    // Fixed buffer: num_blocks * BLOCK_M ≥ Σ padded_e ≤ N + E*(BLOCK_M-1). Safe host upper bound on
    // the block count: E + ceil(N/BLOCK_M).
    let num_blocks = e + n.div_ceil(block_m);
    let buffer = num_blocks * block_m;

    // Scatter into the buffer. Store token+1 / expert+1 so an UNWRITTEN slot reads 0 → −1 sentinel
    // after the shift (distinguishable from a real token/expert 0). Dests are unique → Add == assign.
    let sorted_token = Tensor::<Cuda, 1, Int>::zeros([buffer], &device)
        .select_assign(
            0,
            dest.clone(),
            assign_tok.add_scalar(1i64),
            IndexingUpdateOp::Add,
        )
        .add_scalar(-1i64); // real = token, empty = −1
    let sorted_weight = Tensor::<Cuda, 1>::zeros([buffer], &device).select_assign(
        0,
        dest.clone(),
        assign_w,
        IndexingUpdateOp::Add,
    ); // real = weight, empty = 0
    let sorted_expert = Tensor::<Cuda, 1, Int>::zeros([buffer], &device)
        .select_assign(0, dest, assign_e.add_scalar(1i64), IndexingUpdateOp::Add)
        .add_scalar(-1i64); // real = expert, empty = −1

    // expert_ids[blk] = sorted_expert[blk*BLOCK_M]: the first slot of every block of a non-empty
    // expert is always a real token (dense within-expert ranks), so this picks the right expert;
    // empty/tail blocks read −1.
    let expert_ids = sorted_expert
        .clone()
        .reshape([num_blocks, block_m])
        .slice([0..num_blocks, 0..1])
        .reshape([num_blocks]);

    DroplessLayout {
        sorted_token,
        sorted_weight,
        sorted_expert,
        expert_ids,
        count_e,
        num_blocks,
        buffer,
        n,
    }
}

// =================================================================================================
// Host dispatch (through the typed Fusion-bridge wrapper).
// =================================================================================================

/// Allocate a fresh contiguous f32 output `CubeTensor` of `shape` on the same client as `like`.
fn alloc_f32(like: &CubeTensor<CudaRuntime>, shape: &[usize]) -> CubeTensor<CudaRuntime> {
    let nelem: usize = shape.iter().product();
    let buffer = like.client.empty(nelem * DType::F32.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.to_vec().into(),
        buffer,
        DType::F32,
    )
}

/// Launch the dropless grouped SwiGLU GEMM. Returns the per-slot weighted output `y_sorted:[buffer,H]`
/// (padding rows = 0); the caller scatter-ADDs it back to `out[token]`.
#[allow(clippy::too_many_arguments)]
pub fn grouped_swiglu(
    x: Tensor<Cuda, 2>,                 // [T, H]
    gate: Tensor<Cuda, 3>,              // [E, H, I]
    up: Tensor<Cuda, 3>,                // [E, H, I]
    down: Tensor<Cuda, 3>,              // [E, I, H]
    sorted_token: Tensor<Cuda, 1, Int>, // [buffer]
    sorted_weight: Tensor<Cuda, 1>,     // [buffer]
    expert_ids: Tensor<Cuda, 1, Int>,   // [num_blocks]
    h: usize,
    i: usize,
    block_m: usize,
    num_blocks: usize,
) -> Tensor<Cuda, 2> {
    let buffer = num_blocks * block_m;
    assert_eq!(
        x.dtype(),
        DType::F32,
        "grouped_swiglu activations must be f32, got {:?}",
        x.dtype()
    );
    assert_eq!(
        sorted_token.dims()[0],
        buffer,
        "sorted_token length != num_blocks*BLOCK_M"
    );
    assert_eq!(
        sorted_weight.dims()[0],
        buffer,
        "sorted_weight length != num_blocks*BLOCK_M"
    );
    assert_eq!(
        expert_ids.dims()[0],
        num_blocks,
        "expert_ids length != num_blocks"
    );
    assert_eq!(gate.dims(), [gate.dims()[0], h, i], "gate must be [E,H,I]");
    assert_eq!(down.dims(), [down.dims()[0], i, h], "down must be [E,I,H]");

    let x_prim = x.into_primitive().tensor();
    let gate_prim = gate.into_primitive().tensor();
    let up_prim = up.into_primitive().tensor();
    let down_prim = down.into_primitive().tensor();
    let st_prim = sorted_token.into_primitive(); // Int handle (§0b rule 4)
    let sw_prim = sorted_weight.into_primitive().tensor();
    let eid_prim = expert_ids.into_primitive(); // Int handle

    let outputs = CubeCustomOp::<CudaRuntime>::new("moe_grouped_swiglu")
        .float_input(x_prim) // every read tensor is a declared input (rule 1 / no closure capture)
        .float_input(gate_prim)
        .float_input(up_prim)
        .float_input(down_prim)
        .int_input(st_prim)
        .float_input(sw_prim)
        .int_input(eid_prim)
        .float_output([buffer, h], DType::F32) // cross-validated vs the alloc (rule 2)
        .launch(move |inputs| {
            // Plain (non-packed) tensors → into_contiguous so the kernel's flat index math is valid.
            let x = into_contiguous(inputs[0].clone());
            let gate = into_contiguous(inputs[1].clone());
            let up = into_contiguous(inputs[2].clone());
            let down = into_contiguous(inputs[3].clone());
            let st = into_contiguous(inputs[4].clone());
            let sw = into_contiguous(inputs[5].clone());
            let eid = into_contiguous(inputs[6].clone());
            let out = alloc_f32(&x, &[buffer, h]);

            // One cube per BLOCK_M segment; BLOCK_M threads per cube (one per slot-row).
            gpu::grouped_swiglu::launch::<CudaRuntime>(
                &x.client,
                CubeCount::Static(num_blocks as u32, 1, 1),
                CubeDim {
                    x: block_m as u32,
                    y: 1,
                    z: 1,
                },
                x.as_tensor_arg(1),
                gate.as_tensor_arg(1),
                up.as_tensor_arg(1),
                down.as_tensor_arg(1),
                st.as_tensor_arg(1),
                sw.as_tensor_arg(1),
                eid.as_tensor_arg(1),
                out.as_tensor_arg(1),
                ScalarArg::new(h as u32),
                ScalarArg::new(i as u32),
                ScalarArg::new(block_m as u32),
                i, // comptime i_cap (local gu-array size)
            )
            .expect("grouped_swiglu launch failed");
            vec![out]
        });

    Tensor::from_primitive(TensorPrimitive::Float(
        outputs.into_iter().next().expect("one output"),
    ))
}

/// DROPLESS grouped-GEMM MoE forward for a [`Qwen3MoeSparseBlock`] on the default CUDA backend.
///
/// Mirrors [`Qwen3MoeSparseBlock::forward_oracle`]'s math — `out[t] = Σ_{e∈topk(t)} w_{t,e} ·
/// SwiGLU_e(x_t)` — but computes ONLY the `k*T` routed pairs (no per-expert dense pass, no capacity
/// drop). `block_m` is the vLLM `BLOCK_M` segment alignment (e.g. 16).
pub fn forward_grouped(
    block: &Qwen3MoeSparseBlock<Cuda>,
    x: Tensor<Cuda, 3>,
    block_m: usize,
) -> Tensor<Cuda, 3> {
    let [b, s, h] = x.dims();
    let t = b * s;
    let e = block.num_experts();
    let k = block.top_k();
    let device = x.device();

    // 1. Route → compact top-k. 2. Dropless align/sort (vLLM moe_align_block_size).
    let (sel_idx, sel_w) = block.route_topk(x.clone());
    let lay = dropless_align(sel_idx, sel_w, e, k, block_m);

    // 3. Stacked expert weights, fed to the kernel by the expert-id indirection.
    let (gate, up, down) = block.stacked_experts_pub(); // [E,H,I],[E,H,I],[E,I,H]
    let i = gate.dims()[2];

    // 4. The grouped GEMM → per-slot weighted output [buffer, H].
    let x2 = x.reshape([t, h]).cast(DType::F32);
    let y_sorted = grouped_swiglu(
        x2,
        gate.cast(DType::F32),
        up.cast(DType::F32),
        down.cast(DType::F32),
        lay.sorted_token.clone(),
        lay.sorted_weight,
        lay.expert_ids,
        h,
        i,
        block_m,
        lay.num_blocks,
    );

    // 5. Combine: scatter-ADD each slot's contribution to its token. Padding slots (token −1) are
    //    routed to a dummy row T (their y_sorted rows are 0) and sliced off — a deterministic Burn
    //    reduction; the k contributions of a token accumulate by Add.
    let mask_pad = lay.sorted_token.clone().lower_elem(0i64);
    let tokens_remap = lay.sorted_token.mask_fill(mask_pad, t as i64); // [buffer], −1 → T
    let out_pad = Tensor::<Cuda, 2>::zeros([t + 1, h], &device)
        .cast(DType::F32)
        .select_assign(0, tokens_remap, y_sorted, IndexingUpdateOp::Add);
    out_pad.slice([0..t, 0..h]).reshape([b, s, h])
}

// =================================================================================================
// LEVER (c): FUSED gather-GEMV MoE decode — read each routed expert's weights ONCE directly from the
// persistent contiguous stacks by `expert_id`, no materialized `[N,H,I]` slab (the ~3× round-trip
// `decode_topk`'s `select(0,ids)` does), no host re-stack, no f32 cast of the bf16 stacks.
// =================================================================================================

/// Shared kernel-launch core for the fused gather-GEMV, over RAW `CubeTensor` handles so the SAME
/// code drives both the Fusion-bridge launcher ([`fused_gather_swiglu`]) and a below-Fusion capture
/// path. `inputs = [x:[T,H] f32, gate:[E,H,I] EW, up:[E,H,I] EW, down:[E,I,H] EW, assign_e:[N] i32,
/// assign_tok:[N] i32, sel_w:[N] f32]`; returns the WEIGHTED per-assignment output `out:[N,H]` f32.
///
/// The expert stacks (`gate`/`up`/`down`) are passed AS-IS — NEVER `into_contiguous`'d (that would
/// COPY all `E` experts every call = the re-materialization lever (c) exists to kill); the kernels
/// index them by their own strides. The small/fresh tensors (x, the `[N]` index/weight vectors) ARE
/// made contiguous (cheap). Two `CubeCount::Static` launches (grid fixed for fixed `T,k` ⇒ no
/// `CubeCount::Dynamic` ⇒ CUDA-graph capturable), weight dtype (bf16/f32) dispatched in-register.
fn run_fused_swiglu(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
) -> CubeTensor<CudaRuntime> {
    let x = into_contiguous(inputs[0].clone()); // [T,H] f32 (small)
    let gate = inputs[1].clone(); // [E,H,I] — stays put (strided read in-kernel)
    let up = inputs[2].clone();
    let down = inputs[3].clone();
    let ae = into_contiguous(inputs[4].clone()); // [N] i32
    let at = into_contiguous(inputs[5].clone()); // [N] i32
    let sw = into_contiguous(inputs[6].clone()); // [N] f32
    let wdtype = gate.dtype;
    let gu = alloc_f32(&x, &[n, i]); // [N,I] intermediate (tiny — NOT a weight slab)
    let out = alloc_f32(&x, &[n, h]); // [N,H] weighted output

    // ---- LINE-SIZE PROBE (3-voice fix #1): the vectorized output axis is the INNERMOST stride-1
    //      axis (I for gate/up/gu, H for down/out), so use `try_tensor_line_size_parallel` (NOT
    //      perpendicular — perpendicular returns 1 on a stride-1 axis = silent scalar fallback). V==1
    //      ⇒ a stack isn't innermost-contiguous/V-aligned ⇒ keep the SCALAR kernels (correct, no
    //      speedup). V≥2 ⇒ the split-K vectorized kernel. ----
    let v_g = probe_parallel_line_size(&gate); // over I (gate/up/gu innermost)
    let v_d = probe_parallel_line_size(&down); // over H (down/out innermost)
    debug_report_line_sizes(v_g, v_d, wdtype);

    let n_u = n as u32;
    let i_u = i as u32;
    let h_u = h as u32;

    // Scalar fallbacks (the existing one-thread-per-output GEMVs), kept verbatim as the correctness
    // path whenever the line-size probe returns 1 (a non-vectorizable / non-contiguous stack).
    let threads = 256u32;
    let cdim1 = CubeDim {
        x: threads,
        y: 1,
        z: 1,
    };
    macro_rules! gu_scalar {
        ($ew:ty) => {{
            gpu::fused_swiglu_gu::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
                cdim1,
                x.as_tensor_arg(1),
                gate.as_tensor_arg(1),
                up.as_tensor_arg(1),
                ae.as_tensor_arg(1),
                at.as_tensor_arg(1),
                gu.as_tensor_arg(1),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
            )
            .expect("fused_swiglu_gu (scalar) launch failed");
        }};
    }
    macro_rules! down_scalar {
        ($ew:ty) => {{
            gpu::fused_swiglu_down::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(((n * h) as u32).div_ceil(threads), 1, 1),
                cdim1,
                gu.as_tensor_arg(1),
                down.as_tensor_arg(1),
                ae.as_tensor_arg(1),
                sw.as_tensor_arg(1),
                out.as_tensor_arg(1),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
            )
            .expect("fused_swiglu_down (scalar) launch failed");
        }};
    }

    // SPLIT-K vectorized path. `tiles_per_n = ceil((O/V)/BX)`, grid = `Static(N*tiles_per_n,1,1)` —
    // a pure function of fixed `(N,O,V,BX)` ⇒ CUDA-graph capturable (no `CubeCount::Dynamic`). Each
    // kernel uses its own KSPLIT (its own shared-mem budget) ⇒ its own block y-extent.
    let cdim_gu = CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_GU);
    let cdim_down = CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_DOWN);
    macro_rules! gu_splitk {
        ($ew:ty, $v:expr) => {{
            let vv = $v as u32;
            let tiles = (i_u / vv).div_ceil(SPLITK_BX); // output lines along I, tiled by BX
            gpu::fused_swiglu_gu_splitk::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(n_u * tiles, 1, 1),
                cdim_gu,
                x.as_tensor_arg(1),
                gate.as_tensor_arg($v),
                up.as_tensor_arg($v),
                ae.as_tensor_arg(1),
                at.as_tensor_arg(1),
                gu.as_tensor_arg($v),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
                ScalarArg::new(tiles),
                SPLITK_KSPLIT_GU,
                SPLITK_BX,
            )
            .expect("fused_swiglu_gu_splitk launch failed");
        }};
    }
    macro_rules! down_splitk {
        ($ew:ty, $v:expr) => {{
            let vv = $v as u32;
            let tiles = (h_u / vv).div_ceil(SPLITK_BX); // output lines along H, tiled by BX
            gpu::fused_swiglu_down_splitk::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(n_u * tiles, 1, 1),
                cdim_down,
                gu.as_tensor_arg(1),
                down.as_tensor_arg($v),
                ae.as_tensor_arg(1),
                sw.as_tensor_arg(1),
                out.as_tensor_arg($v),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
                ScalarArg::new(tiles),
                SPLITK_KSPLIT_DOWN,
                SPLITK_BX,
            )
            .expect("fused_swiglu_down_splitk launch failed");
        }};
    }

    macro_rules! dispatch {
        ($ew:ty) => {{
            match v_g {
                1 => gu_scalar!($ew),
                v => gu_splitk!($ew, v),
            }
            match v_d {
                1 => down_scalar!($ew),
                v => down_splitk!($ew, v),
            }
        }};
    }

    match wdtype {
        DType::F32 => dispatch!(f32),
        DType::BF16 => dispatch!(half::bf16),
        d => panic!(
            "fused gather-GEMV: unsupported expert-weight dtype {d:?} (expected bf16 or f32)"
        ),
    }
    out
}

/// Split-K tuning knobs (comptime in the kernel). `BX=32` = one warp of output lanes (coalesced
/// 128-bit weight loads over the contiguous output axis). `KSPLIT` = K-split warps that cut the
/// loop-carried FMA chain AND raise threads/block to hide latency on the occupancy-starved decode
/// grid (gu 24 blocks, down 64 at N=8). MEASURED & TUNED on the captured 30B decode:
///   KSPLIT(gu): 8 → 19.32 tok/s, 16 → 20.92, 20 → 20.95 (plateau) — gu is parallelism-limited and
///   wants the most warps it can fit; the win saturates at 16 (the design's predicted occupancy
///   lever). KSPLIT(down): 16 → 20.92, 32 → 20.95 (no real change) — down already has 64 blocks, so
///   its split-K is not the limiter. Defaults 16/16 sit at the ~20.9 tok/s / 46% peak plateau (vs the
///   19.38 / 43% scalar-gather baseline) at the smallest shared-mem footprint. Per-kernel because the
///   budgets differ at the ≤48 KB/block limit: gu uses TWO `SharedMemory<Line<f32>>` reduction
///   buffers (g+u) = `2·KSPLIT·BX·V·4` B (KSPLIT=16 → 32 KB), down uses ONE (a) = half that. Both
///   stay capturable (comptime smem, `CubeCount::Static`).
const SPLITK_BX: u32 = 32;
const SPLITK_KSPLIT_GU: u32 = 16;
const SPLITK_KSPLIT_DOWN: u32 = 16;

/// Probe the max PARALLEL (innermost stride-1 axis) line size for a stack — V for `as_tensor_arg(V)`.
/// Returns 1 when the innermost axis is not stride-1 / not V-aligned (⇒ scalar fallback). The output
/// axis (I for gate/up, H for down) IS the innermost contiguous axis of the persistent stacks.
fn probe_parallel_line_size(t: &CubeTensor<CudaRuntime>) -> usize {
    let shape = t.meta.shape();
    let strides = t.meta.strides();
    let axis = shape.len() - 1; // innermost output axis (stride 1 for a contiguous stack)
    let sizes = t.client.io_optimized_line_sizes(t.dtype.size());
    tensor_line_size_parallel(sizes, shape, strides, axis)
}

/// One-time diagnostic so a SILENT scalar fallback (V==1, no speedup) is visible (3-voice fix #3:
/// "verify V==8 is actually chosen, don't assume"). Prints once per process; no device sync.
fn debug_report_line_sizes(v_g: usize, v_d: usize, wdtype: DType) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let path = if v_g >= 2 && v_d >= 2 {
            "SPLIT-K vectorized"
        } else {
            "SCALAR fallback (V==1 on a stack — check contiguity!)"
        };
        eprintln!(
            "[moe fused gather-GEMV] weight dtype={wdtype:?} line-size V_gu(I)={v_g} V_down(H)={v_d} \
             -> {path} (BX={SPLITK_BX}, KSPLIT gu={SPLITK_KSPLIT_GU} down={SPLITK_KSPLIT_DOWN})"
        );
    });
}

/// Shared shape/dtype validation for [`FusedSwigluBackend::fused_gather_swiglu`] (both backends).
#[allow(clippy::too_many_arguments)]
fn assert_fused_shapes<B: Backend>(
    x: &Tensor<B, 2>,
    gate: &Tensor<B, 3>,
    up: &Tensor<B, 3>,
    down: &Tensor<B, 3>,
    assign_e: &Tensor<B, 1, Int>,
    assign_tok: &Tensor<B, 1, Int>,
    sel_w: &Tensor<B, 1>,
    h: usize,
    i: usize,
    n: usize,
) {
    assert_eq!(
        x.dtype(),
        DType::F32,
        "fused_gather_swiglu activations must be f32, got {:?}",
        x.dtype()
    );
    let wdtype = gate.dtype();
    assert!(
        matches!(wdtype, DType::F32 | DType::BF16),
        "fused_gather_swiglu expert weights must be bf16 or f32, got {wdtype:?}"
    );
    assert_eq!(
        up.dtype(),
        wdtype,
        "up dtype {:?} != gate dtype {wdtype:?}",
        up.dtype()
    );
    assert_eq!(
        down.dtype(),
        wdtype,
        "down dtype {:?} != gate dtype {wdtype:?}",
        down.dtype()
    );
    let e = gate.dims()[0];
    assert_eq!(gate.dims(), [e, h, i], "gate must be [E,H,I]");
    assert_eq!(up.dims(), [e, h, i], "up must be [E,H,I]");
    assert_eq!(down.dims(), [e, i, h], "down must be [E,I,H]");
    assert_eq!(assign_e.dims()[0], n, "assign_e len != N");
    assert_eq!(assign_tok.dims()[0], n, "assign_tok len != N");
    assert_eq!(sel_w.dims()[0], n, "sel_w len != N");
    assert_eq!(x.dims()[1], h, "x H != stack H");
}

/// Backends that can run lever (c)'s FUSED gather-GEMV decode core: compute the WEIGHTED per-assignment
/// SwiGLU output `y[N,H] = sel_w[n]·down_e(silu(x[tok]·gate_e)·(x[tok]·up_e))` for the `N=T*k` routed
/// `(token, expert)` assignments, reading the persistent `[E,H,I]`/`[E,I,H]` stacks DIRECTLY by
/// `expert_id` — NO materialized `[N,H,I]` weight slabs, NO host re-stack, NO f32 cast of the bf16
/// stacks. The caller scatter-ADDs `y[N,H]` back per token (the oracle combine).
///
/// Implemented for BOTH the default Fusion `Cuda` backend (through the `CubeCustomOp` bridge — the eager
/// decode path) AND the raw `CubeBackend<CudaRuntime,…>` below Fusion (a DIRECT `CubeTensor` launch — the
/// CUDA-graph capture path, which must run below Fusion). Same two `CubeCount::Static` kernels either way.
pub trait FusedSwigluBackend: Backend {
    /// See the trait docs. `x:[T,H]` (f32), persistent stacks `gate/up:[E,H,I]`, `down:[E,I,H]`,
    /// routing `assign_e/assign_tok:[N]` + `sel_w:[N]` → weighted per-assignment output `[N,H]` (f32).
    #[allow(clippy::too_many_arguments)]
    fn fused_gather_swiglu(
        x: Tensor<Self, 2>,
        gate: Tensor<Self, 3>,
        up: Tensor<Self, 3>,
        down: Tensor<Self, 3>,
        assign_e: Tensor<Self, 1, Int>,
        assign_tok: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2>;
}

impl FusedSwigluBackend for Cuda {
    fn fused_gather_swiglu(
        x: Tensor<Cuda, 2>,
        gate: Tensor<Cuda, 3>,
        up: Tensor<Cuda, 3>,
        down: Tensor<Cuda, 3>,
        assign_e: Tensor<Cuda, 1, Int>,
        assign_tok: Tensor<Cuda, 1, Int>,
        sel_w: Tensor<Cuda, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Cuda, 2> {
        assert_fused_shapes(
            &x,
            &gate,
            &up,
            &down,
            &assign_e,
            &assign_tok,
            &sel_w,
            h,
            i,
            n,
        );

        // Fusion path: declare every read tensor as an input (rule 1) and launch the two kernels INSIDE
        // the custom-op closure (on the raw client — rule 7). The stacks ride as float inputs and are
        // NEVER `into_contiguous`'d (run_fused_swiglu indexes them by stride — no re-stack copy).
        let x_prim = x.into_primitive().tensor();
        let gate_prim = gate.into_primitive().tensor();
        let up_prim = up.into_primitive().tensor();
        let down_prim = down.into_primitive().tensor();
        let ae_prim = assign_e.into_primitive(); // Int handle (rule 4)
        let at_prim = assign_tok.into_primitive(); // Int handle
        let sw_prim = sel_w.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("moe_fused_gather_swiglu")
            .float_input(x_prim)
            .float_input(gate_prim)
            .float_input(up_prim)
            .float_input(down_prim)
            .int_input(ae_prim)
            .int_input(at_prim)
            .float_input(sw_prim)
            .float_output([n, h], DType::F32) // cross-validated vs the alloc (rule 2)
            .launch(move |inputs| vec![run_fused_swiglu(inputs, h, i, n)]);

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }
}

impl FusedSwigluBackend for CubeBackend<CudaRuntime, f32, i32, u8> {
    fn fused_gather_swiglu(
        x: Tensor<Self, 2>,
        gate: Tensor<Self, 3>,
        up: Tensor<Self, 3>,
        down: Tensor<Self, 3>,
        assign_e: Tensor<Self, 1, Int>,
        assign_tok: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2> {
        assert_fused_shapes(
            &x,
            &gate,
            &up,
            &down,
            &assign_e,
            &assign_tok,
            &sel_w,
            h,
            i,
            n,
        );

        // RAW below-Fusion path (the CUDA-graph capture backend): the tensors' primitives ARE
        // `CubeTensor`s already, so launch the same two Static kernels DIRECTLY on the raw client — no
        // fusion stream, no custom-op bridge. This is what `cudagraph_moe_decode_bench` captures.
        let inputs = [
            x.into_primitive().tensor(),
            gate.into_primitive().tensor(),
            up.into_primitive().tensor(),
            down.into_primitive().tensor(),
            assign_e.into_primitive(), // Int primitive IS a CubeTensor
            assign_tok.into_primitive(),
            sel_w.into_primitive().tensor(),
        ];
        let out = run_fused_swiglu(&inputs, h, i, n);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

// =================================================================================================
// Qwen3.5 35B COMMIT-1 scaffold: fused bf16 combined gate_up `[E,2I,H]` + down `[E,H,I]`.
// This is intentionally a BF16 plumbing path only; fp8 numerics are separate.
// =================================================================================================

fn run_fused35_bf16(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
) -> CubeTensor<CudaRuntime> {
    let x = into_contiguous(inputs[0].clone()); // [T,H] f32
    let gate_up = inputs[1].clone(); // [E,2I,H] — stays put; stride-indexed in-kernel
    let down = inputs[2].clone(); // [E,H,I] — stays put; stride-indexed in-kernel
    let ae = into_contiguous(inputs[3].clone()); // [N] i32
    let sw = into_contiguous(inputs[4].clone()); // [N] f32
    let wdtype = gate_up.dtype;
    let gu = alloc_f32(&x, &[n, i]);
    let out = alloc_f32(&x, &[n, h]);

    let [t, x_h] = x.meta.shape().dims::<2>();
    assert_eq!(x_h, h, "fused35 bf16: x H != h");
    assert!(t > 0, "fused35 bf16: T must be nonzero");
    assert_eq!(n % t, 0, "fused35 bf16: N must be divisible by T");
    let top_k = (n / t) as u32;

    // The 35B bf16 stacks are transposed relative to the vectorized layout: gate/up output axis I is
    // axis 1 in `[E,2I,H]`, and down output axis H is axis 1 in `[E,H,I]`. Probe those axes so the
    // expected scalar fallback is visible instead of accidentally reporting the contiguous reduction
    // axis as vectorizable.
    let v_gu = probe_axis_line_size(&gate_up, 1);
    let v_down = probe_axis_line_size(&down, 1);
    debug_report_35_line_sizes(v_gu, v_down, wdtype);
    assert_eq!(
        v_gu, 1,
        "fused35 bf16 scaffold only supports scalar gate_up layout in Commit 1 (probe V={v_gu})"
    );
    assert_eq!(
        v_down, 1,
        "fused35 bf16 scaffold only supports scalar down layout in Commit 1 (probe V={v_down})"
    );

    let h_u = h as u32;
    let i_u = i as u32;
    let threads = 256u32;
    let cdim = CubeDim {
        x: threads,
        y: 1,
        z: 1,
    };

    macro_rules! launch_scalar {
        ($ew:ty) => {{
            gpu::fused35_gu_bf16_scalar::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
                cdim,
                x.as_tensor_arg(1),
                gate_up.as_tensor_arg(1),
                ae.as_tensor_arg(1),
                gu.as_tensor_arg(1),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
                ScalarArg::new(top_k),
            )
            .expect("fused35_gu_bf16_scalar launch failed");

            gpu::fused35_down_bf16_scalar::launch::<$ew, CudaRuntime>(
                &x.client,
                CubeCount::Static(((n * h) as u32).div_ceil(threads), 1, 1),
                cdim,
                gu.as_tensor_arg(1),
                down.as_tensor_arg(1),
                ae.as_tensor_arg(1),
                sw.as_tensor_arg(1),
                out.as_tensor_arg(1),
                ScalarArg::new(h_u),
                ScalarArg::new(i_u),
            )
            .expect("fused35_down_bf16_scalar launch failed");
        }};
    }

    match wdtype {
        DType::F32 => launch_scalar!(f32),
        DType::BF16 => launch_scalar!(half::bf16),
        d => panic!("fused35 bf16: unsupported expert-weight dtype {d:?} (expected bf16 or f32)"),
    }
    out
}

#[cfg(feature = "cuda")]
fn run_e4m3_line_decode_probe(
    inputs: &[CubeTensor<CudaRuntime>],
    len: usize,
) -> CubeTensor<CudaRuntime> {
    let q = into_contiguous(inputs[0].clone());
    let out = alloc_f32(&q, &[len]);
    let v = probe_parallel_line_size(&q);
    assert!(
        v > 1 && len % v == 0,
        "e4m3 line decode probe requires a vectorizable len, got len={len} V={v}"
    );
    gpu::e4m3_line_decode_probe::launch::<CudaRuntime>(
        &q.client,
        CubeCount::Static((len / v) as u32, 1, 1),
        CubeDim::new_1d(1),
        q.as_tensor_arg(v),
        out.as_tensor_arg(v),
    )
    .expect("e4m3_line_decode_probe launch failed");
    out
}

#[cfg(feature = "cuda")]
pub fn e4m3_line_decode_probe(q: Tensor<Cuda, 1, Int>) -> Tensor<Cuda, 1> {
    assert_eq!(
        q.dtype(),
        DType::I8,
        "e4m3_line_decode_probe expects raw e4m3 bytes carried as I8, got {:?}",
        q.dtype()
    );
    let len = q.dims()[0];
    let q_prim = q.into_primitive();
    let outputs = CubeCustomOp::<CudaRuntime>::new("e4m3_line_decode_probe")
        .int_input(q_prim)
        .float_output([len], DType::F32)
        .launch(move |inputs| vec![run_e4m3_line_decode_probe(inputs, len)]);
    Tensor::from_primitive(TensorPrimitive::Float(
        outputs.into_iter().next().expect("one output"),
    ))
}

#[cfg(feature = "cuda")]
fn run_e2m1_marlin_decode_probe(
    inputs: &[CubeTensor<CudaRuntime>],
    len: usize,
) -> CubeTensor<CudaRuntime> {
    let q = into_contiguous(inputs[0].clone());
    let out = alloc_f32(&q, &[len * 2]);
    let threads = 256u32;
    gpu::e2m1_marlin_decode_probe::launch::<CudaRuntime>(
        &q.client,
        CubeCount::Static(((len * 2) as u32).div_ceil(threads), 1, 1),
        CubeDim {
            x: threads,
            y: 1,
            z: 1,
        },
        q.as_tensor_arg(1),
        out.as_tensor_arg(1),
    )
    .expect("e2m1_marlin_decode_probe launch failed");
    out
}

#[cfg(feature = "cuda")]
pub fn e2m1_marlin_decode_probe(q: Tensor<Cuda, 1, Int>) -> Tensor<Cuda, 1> {
    assert_eq!(
        q.dtype(),
        DType::I8,
        "e2m1_marlin_decode_probe expects packed e2m1 bytes carried as I8, got {:?}",
        q.dtype()
    );
    let len = q.dims()[0];
    let q_prim = q.into_primitive();
    let outputs = CubeCustomOp::<CudaRuntime>::new("e2m1_marlin_decode_probe")
        .int_input(q_prim)
        .float_output([len * 2], DType::F32)
        .launch(move |inputs| vec![run_e2m1_marlin_decode_probe(inputs, len)]);
    Tensor::from_primitive(TensorPrimitive::Float(
        outputs.into_iter().next().expect("one output"),
    ))
}

#[cfg(all(test, feature = "cuda"))]
fn run_fused35_nvfp4_projection_probe(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
) -> Vec<CubeTensor<CudaRuntime>> {
    let x = into_contiguous(inputs[0].clone());
    let q_gu = inputs[1].clone();
    let bs_gu = inputs[2].clone();
    let gscale_gu = inputs[3].clone();
    let ae = into_contiguous(inputs[4].clone());
    let gate = alloc_f32(&x, &[n, i]);
    let up = alloc_f32(&x, &[n, i]);

    let [t, x_h] = x.meta.shape().dims::<2>();
    assert_eq!(x_h, h, "nvfp4 projection probe: x H != h");
    assert!(t > 0, "nvfp4 projection probe: T must be nonzero");
    assert_eq!(n % t, 0, "nvfp4 projection probe: N must be divisible by T");
    let top_k = (n / t) as u32;
    let threads = 256u32;
    gpu::fused35_gu_nvfp4_projection_probe::launch::<CudaRuntime>(
        &x.client,
        CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
        CubeDim {
            x: threads,
            y: 1,
            z: 1,
        },
        x.as_tensor_arg(1),
        q_gu.as_tensor_arg(1),
        bs_gu.as_tensor_arg(1),
        gscale_gu.as_tensor_arg(1),
        ae.as_tensor_arg(1),
        gate.as_tensor_arg(1),
        up.as_tensor_arg(1),
        ScalarArg::new(h as u32),
        ScalarArg::new(i as u32),
        ScalarArg::new(top_k),
    )
    .expect("fused35_gu_nvfp4_projection_probe launch failed");
    vec![gate, up]
}

#[cfg(all(test, feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn fused35_nvfp4_projection_probe_cuda(
    x: Tensor<Cuda, 2>,
    q_gu: Tensor<Cuda, 3, Int>,
    bs_gu: Tensor<Cuda, 3, Int>,
    gscale_gu: Tensor<Cuda, 2>,
    assign_e: Tensor<Cuda, 1, Int>,
    h: usize,
    i: usize,
    n: usize,
) -> (Tensor<Cuda, 2>, Tensor<Cuda, 2>) {
    assert_eq!(
        x.dtype(),
        DType::F32,
        "nvfp4 projection probe x must be f32"
    );
    assert_eq!(
        q_gu.dtype(),
        DType::I8,
        "nvfp4 projection probe q_gu must be I8"
    );
    assert_eq!(
        bs_gu.dtype(),
        DType::I8,
        "nvfp4 projection probe bs_gu must be I8"
    );
    assert_eq!(
        gscale_gu.dtype(),
        DType::F32,
        "nvfp4 projection probe gscale_gu must be f32"
    );
    assert_eq!(q_gu.dims()[1], h, "nvfp4 projection probe q_gu H != h");
    assert_eq!(q_gu.dims()[2], i, "nvfp4 projection probe q_gu bytes != I");
    assert_eq!(
        bs_gu.dims(),
        [q_gu.dims()[0], i * 2, h / 16],
        "nvfp4 projection probe bs_gu must be [E,2I,H/16]"
    );
    assert_eq!(
        gscale_gu.dims(),
        [q_gu.dims()[0], 2],
        "nvfp4 projection probe gscale_gu must be [E,2]"
    );
    assert_eq!(
        assign_e.dims()[0],
        n,
        "nvfp4 projection probe assign_e len != N"
    );

    let x_prim = x.into_primitive().tensor();
    let q_gu_prim = q_gu.into_primitive();
    let bs_gu_prim = bs_gu.into_primitive();
    let gscale_gu_prim = gscale_gu.into_primitive().tensor();
    let ae_prim = assign_e.into_primitive();
    let outputs = CubeCustomOp::<CudaRuntime>::new("qwen35_nvfp4_gu_projection_probe")
        .float_input(x_prim)
        .int_input(q_gu_prim)
        .int_input(bs_gu_prim)
        .float_input(gscale_gu_prim)
        .int_input(ae_prim)
        .float_output([n, i], DType::F32)
        .float_output([n, i], DType::F32)
        .launch(move |inputs| run_fused35_nvfp4_projection_probe(inputs, h, i, n));
    let mut outputs = outputs.into_iter();
    let gate = Tensor::from_primitive(TensorPrimitive::Float(
        outputs.next().expect("gate projection output"),
    ));
    let up = Tensor::from_primitive(TensorPrimitive::Float(
        outputs.next().expect("up projection output"),
    ));
    (gate, up)
}

fn run_fused35_fp8(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
) -> CubeTensor<CudaRuntime> {
    let x = into_contiguous(inputs[0].clone()); // [T,H] f32
    let q_gu = inputs[1].clone(); // [E,H,2I] raw e4m3 bytes, output axis innermost
    let s_gu = inputs[2].clone(); // [E,2I] f32, output axis innermost
    let q_dn = inputs[3].clone(); // [E,I,H] raw e4m3 bytes, output axis innermost
    let s_dn = inputs[4].clone(); // [E,H] f32, output axis innermost
    let ae = into_contiguous(inputs[5].clone()); // [N] i32
    let sw = into_contiguous(inputs[6].clone()); // [N] f32
    let gu = alloc_f32(&x, &[n, i]);
    let out = alloc_f32(&x, &[n, h]);

    let [t, x_h] = x.meta.shape().dims::<2>();
    assert_eq!(x_h, h, "fused35 fp8: x H != h");
    assert!(t > 0, "fused35 fp8: T must be nonzero");
    assert_eq!(n % t, 0, "fused35 fp8: N must be divisible by T");
    let top_k = (n / t) as u32;

    let v_gu_q = probe_parallel_line_size(&q_gu);
    let v_gu_s = probe_parallel_line_size(&s_gu);
    let v_down_q = probe_parallel_line_size(&q_dn);
    let v_down_s = probe_parallel_line_size(&s_dn);
    let v_gu = v_gu_q.min(v_gu_s);
    let v_down = v_down_q.min(v_down_s);
    debug_report_35_fp8_line_sizes(v_gu_q, v_gu_s, v_down_q, v_down_s, v_gu, v_down);

    let h_u = h as u32;
    let i_u = i as u32;
    let threads = 256u32;
    let cdim_scalar = CubeDim {
        x: threads,
        y: 1,
        z: 1,
    };

    if v_gu <= 1 || i % v_gu != 0 {
        eprintln!(
            "[qwen3.5 fused35 fp8] gate/up scalar fallback: V_q={v_gu_q} V_s={v_gu_s} \
             V_common={v_gu} I%V={}",
            if v_gu == 0 { i } else { i % v_gu }
        );
        gpu::fused35_gu_fp8_scalar::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
            cdim_scalar,
            x.as_tensor_arg(1),
            q_gu.as_tensor_arg(1),
            s_gu.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            gu.as_tensor_arg(1),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(top_k),
        )
        .expect("fused35_gu_fp8_scalar launch failed");
    } else {
        let tiles = ((i_u / v_gu as u32).div_ceil(SPLITK_BX)).max(1);
        gpu::fused35_gu_fp8_splitk::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(n as u32 * tiles, 1, 1),
            CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_GU),
            x.as_tensor_arg(1),
            q_gu.as_tensor_arg(v_gu),
            s_gu.as_tensor_arg(v_gu),
            ae.as_tensor_arg(1),
            gu.as_tensor_arg(v_gu),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(top_k),
            ScalarArg::new(tiles),
            SPLITK_KSPLIT_GU,
            SPLITK_BX,
        )
        .expect("fused35_gu_fp8_splitk launch failed");
    }

    if v_down <= 1 || h % v_down != 0 {
        eprintln!(
            "[qwen3.5 fused35 fp8] down scalar fallback: V_q={v_down_q} V_s={v_down_s} \
             V_common={v_down} H%V={}",
            if v_down == 0 { h } else { h % v_down }
        );
        gpu::fused35_down_fp8_scalar::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(((n * h) as u32).div_ceil(threads), 1, 1),
            cdim_scalar,
            gu.as_tensor_arg(1),
            q_dn.as_tensor_arg(1),
            s_dn.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            sw.as_tensor_arg(1),
            out.as_tensor_arg(1),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
        )
        .expect("fused35_down_fp8_scalar launch failed");
    } else {
        let tiles = ((h_u / v_down as u32).div_ceil(SPLITK_BX)).max(1);
        gpu::fused35_down_fp8_splitk::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(n as u32 * tiles, 1, 1),
            CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_DOWN),
            gu.as_tensor_arg(1),
            q_dn.as_tensor_arg(v_down),
            s_dn.as_tensor_arg(v_down),
            ae.as_tensor_arg(1),
            sw.as_tensor_arg(1),
            out.as_tensor_arg(v_down),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(tiles),
            SPLITK_KSPLIT_DOWN,
            SPLITK_BX,
        )
        .expect("fused35_down_fp8_splitk launch failed");
    }
    out
}

fn assert_nvfp4_splitk(splitk: u32) {
    assert_eq!(
        splitk, 16,
        "fused35 nvfp4 requires splitk == 16 to match the per-16-K block-scale layout, got {splitk}"
    );
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(dead_code)]
enum Nvfp4LaunchPath {
    Auto,
    Scalar,
    SplitK,
}

fn nvfp4_packed_byte_line_size(
    q: &CubeTensor<CudaRuntime>,
    out: &CubeTensor<CudaRuntime>,
) -> usize {
    let v_q = probe_parallel_line_size(q);
    let v_out = probe_parallel_line_size(out);
    v_q.min(v_out / 2)
}

fn debug_report_35_nvfp4_line_sizes(
    v_gu_q: usize,
    v_gu_out: usize,
    v_down_q: usize,
    v_down_out: usize,
    v_gu: usize,
    v_down: usize,
) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let path = if v_gu >= 2 && v_down >= 2 {
            "SPLIT-K vectorized"
        } else {
            "SCALAR fallback on at least one stack"
        };
        eprintln!(
            "[qwen3.5 fused35 nvfp4] line-size V_gu(q-bytes/out-f32/common-bytes)={v_gu_q}/{v_gu_out}/{v_gu} \
             V_down(q-bytes/out-f32/common-bytes)={v_down_q}/{v_down_out}/{v_down} -> {path} \
             (BX={SPLITK_BX}, KSPLIT gu={SPLITK_KSPLIT_GU} down={SPLITK_KSPLIT_DOWN}; \
             each thread accumulates 2*V logical channels)"
        );
    });
}

fn run_fused35_nvfp4(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
) -> CubeTensor<CudaRuntime> {
    run_fused35_nvfp4_with_path(inputs, h, i, n, Nvfp4LaunchPath::Auto)
}

fn run_fused35_nvfp4_with_path(
    inputs: &[CubeTensor<CudaRuntime>],
    h: usize,
    i: usize,
    n: usize,
    path: Nvfp4LaunchPath,
) -> CubeTensor<CudaRuntime> {
    assert_nvfp4_splitk(SPLITK_KSPLIT_GU);
    assert_nvfp4_splitk(SPLITK_KSPLIT_DOWN);

    let x = into_contiguous(inputs[0].clone()); // [T,H] f32
    let q_gu = inputs[1].clone(); // [E,H,I] raw e2m1 bytes, output-major
    let bs_gu = inputs[2].clone(); // [E,2I,H/16] raw e4m3 bytes
    let gscale_gu = inputs[3].clone(); // [E,2] f32
    let q_dn = inputs[4].clone(); // [E,I,H/2] raw e2m1 bytes, output-major
    let bs_dn = inputs[5].clone(); // [E,H,I/16] raw e4m3 bytes
    let gscale_dn = inputs[6].clone(); // [E] f32
    let ae = into_contiguous(inputs[7].clone()); // [N] i32
    let sw = into_contiguous(inputs[8].clone()); // [N] f32
    let gu = alloc_f32(&x, &[n, i]);
    let out = alloc_f32(&x, &[n, h]);

    let [t, x_h] = x.meta.shape().dims::<2>();
    assert_eq!(x_h, h, "fused35 nvfp4: x H != h");
    assert!(
        (1..=16).contains(&t),
        "fused35 nvfp4 decode path requires 1 <= T <= 16, got {t}"
    );
    assert_eq!(n % t, 0, "fused35 nvfp4: N must be divisible by T");
    let top_k = (n / t) as u32;

    let v_gu_q_probe = probe_parallel_line_size(&q_gu);
    let v_gu_out_probe = probe_parallel_line_size(&gu);
    let v_down_q_probe = probe_parallel_line_size(&q_dn);
    let v_down_out_probe = probe_parallel_line_size(&out);
    let v_gu = nvfp4_packed_byte_line_size(&q_gu, &gu);
    let v_down = nvfp4_packed_byte_line_size(&q_dn, &out);
    debug_report_35_nvfp4_line_sizes(
        v_gu_q_probe,
        v_gu_out_probe,
        v_down_q_probe,
        v_down_out_probe,
        v_gu,
        v_down,
    );

    let h_u = h as u32;
    let i_u = i as u32;
    let threads = 256u32;
    let cdim_scalar = CubeDim {
        x: threads,
        y: 1,
        z: 1,
    };

    let gu_scalar = match path {
        Nvfp4LaunchPath::Auto => v_gu <= 1 || i % (2 * v_gu) != 0,
        Nvfp4LaunchPath::Scalar => true,
        Nvfp4LaunchPath::SplitK => {
            assert!(
                v_gu > 1 && i % (2 * v_gu) == 0,
                "forced nvfp4 split-K gate/up path requires V>1 and I%(2V)==0, got V={v_gu}, I={i}"
            );
            false
        }
    };
    if gu_scalar {
        eprintln!(
            "[qwen3.5 fused35 nvfp4] gate/up scalar fallback: V_bytes={v_gu} I%(2V)={}",
            if v_gu == 0 { i } else { i % (2 * v_gu) }
        );
        gpu::fused35_gu_nvfp4_scalar::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
            cdim_scalar,
            x.as_tensor_arg(1),
            q_gu.as_tensor_arg(1),
            bs_gu.as_tensor_arg(1),
            gscale_gu.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            gu.as_tensor_arg(1),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(top_k),
        )
        .expect("fused35_gu_nvfp4_scalar launch failed");
    } else {
        let tiles = ((i_u / (2 * v_gu) as u32).div_ceil(SPLITK_BX)).max(1);
        gpu::fused35_gu_nvfp4_splitk::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(n as u32 * tiles, 1, 1),
            CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_GU),
            x.as_tensor_arg(1),
            q_gu.as_tensor_arg(v_gu),
            bs_gu.as_tensor_arg(1),
            gscale_gu.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            gu.as_tensor_arg(2 * v_gu),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(top_k),
            ScalarArg::new(tiles),
            SPLITK_KSPLIT_GU,
            SPLITK_BX,
        )
        .expect("fused35_gu_nvfp4_splitk launch failed");
    }

    let down_scalar = match path {
        Nvfp4LaunchPath::Auto => v_down <= 1 || h % (2 * v_down) != 0,
        Nvfp4LaunchPath::Scalar => true,
        Nvfp4LaunchPath::SplitK => {
            assert!(
                v_down > 1 && h % (2 * v_down) == 0,
                "forced nvfp4 split-K down path requires V>1 and H%(2V)==0, got V={v_down}, H={h}"
            );
            false
        }
    };
    if down_scalar {
        eprintln!(
            "[qwen3.5 fused35 nvfp4] down scalar fallback: V_bytes={v_down} H%(2V)={}",
            if v_down == 0 { h } else { h % (2 * v_down) }
        );
        gpu::fused35_down_nvfp4_scalar::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(((n * h) as u32).div_ceil(threads), 1, 1),
            cdim_scalar,
            gu.as_tensor_arg(1),
            q_dn.as_tensor_arg(1),
            bs_dn.as_tensor_arg(1),
            gscale_dn.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            sw.as_tensor_arg(1),
            out.as_tensor_arg(1),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
        )
        .expect("fused35_down_nvfp4_scalar launch failed");
    } else {
        let tiles = ((h_u / (2 * v_down) as u32).div_ceil(SPLITK_BX)).max(1);
        gpu::fused35_down_nvfp4_splitk::launch::<CudaRuntime>(
            &x.client,
            CubeCount::Static(n as u32 * tiles, 1, 1),
            CubeDim::new_2d(SPLITK_BX, SPLITK_KSPLIT_DOWN),
            gu.as_tensor_arg(1),
            q_dn.as_tensor_arg(v_down),
            bs_dn.as_tensor_arg(1),
            gscale_dn.as_tensor_arg(1),
            ae.as_tensor_arg(1),
            sw.as_tensor_arg(1),
            out.as_tensor_arg(2 * v_down),
            ScalarArg::new(h_u),
            ScalarArg::new(i_u),
            ScalarArg::new(tiles),
            SPLITK_KSPLIT_DOWN,
            SPLITK_BX,
        )
        .expect("fused35_down_nvfp4_splitk launch failed");
    }
    out
}

fn probe_axis_line_size(t: &CubeTensor<CudaRuntime>, axis: usize) -> usize {
    let shape = t.meta.shape();
    let strides = t.meta.strides();
    let sizes = t.client.io_optimized_line_sizes(t.dtype.size());
    tensor_line_size_parallel(sizes, shape, strides, axis)
}

fn debug_report_35_line_sizes(v_gu: usize, v_down: usize, wdtype: DType) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let path = if v_gu == 1 && v_down == 1 {
            "SCALAR fallback (expected for bf16 [E,2I,H]/[E,H,I] scaffold)"
        } else {
            "non-scalar probe (Commit 1 does not dispatch vectorized 35B bf16)"
        };
        eprintln!(
            "[qwen3.5 fused35 bf16] weight dtype={wdtype:?} line-size V_gu(I-axis)={v_gu} \
             V_down(H-axis)={v_down} -> {path}"
        );
    });
}

fn debug_report_35_fp8_line_sizes(
    v_gu_q: usize,
    v_gu_s: usize,
    v_down_q: usize,
    v_down_s: usize,
    v_gu: usize,
    v_down: usize,
) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let path = if v_gu >= 2 && v_down >= 2 {
            "SPLIT-K vectorized"
        } else {
            "SCALAR fallback on at least one stack"
        };
        eprintln!(
            "[qwen3.5 fused35 fp8] line-size V_gu(q/s/common)={v_gu_q}/{v_gu_s}/{v_gu} \
             V_down(q/s/common)={v_down_q}/{v_down_s}/{v_down} -> {path} (BX={SPLITK_BX}, \
             KSPLIT gu={SPLITK_KSPLIT_GU} down={SPLITK_KSPLIT_DOWN})"
        );
    });
}

#[allow(clippy::too_many_arguments)]
fn assert_fused35_shapes<B: Backend>(
    x: &Tensor<B, 2>,
    gate_up: &Tensor<B, 3>,
    down: &Tensor<B, 3>,
    assign_e: &Tensor<B, 1, Int>,
    sel_w: &Tensor<B, 1>,
    h: usize,
    i: usize,
    n: usize,
) {
    assert_eq!(
        x.dtype(),
        DType::F32,
        "fused35 bf16 activations must be f32, got {:?}",
        x.dtype()
    );
    let wdtype = gate_up.dtype();
    assert!(
        matches!(wdtype, DType::F32 | DType::BF16),
        "fused35 bf16 expert weights must be bf16 or f32, got {wdtype:?}"
    );
    assert_eq!(
        down.dtype(),
        wdtype,
        "down dtype {:?} != gate_up dtype {wdtype:?}",
        down.dtype()
    );
    let e = gate_up.dims()[0];
    assert_eq!(gate_up.dims(), [e, i * 2, h], "gate_up must be [E,2I,H]");
    assert_eq!(down.dims(), [e, h, i], "down must be [E,H,I]");
    assert_eq!(assign_e.dims()[0], n, "assign_e len != N");
    assert_eq!(sel_w.dims()[0], n, "sel_w len != N");
    assert_eq!(x.dims()[1], h, "x H != stack H");
}

#[allow(clippy::too_many_arguments)]
fn assert_fused35_fp8_shapes<B: Backend>(
    x: &Tensor<B, 2>,
    q_gu: &Tensor<B, 3, Int>,
    s_gu: &Tensor<B, 2>,
    q_dn: &Tensor<B, 3, Int>,
    s_dn: &Tensor<B, 2>,
    assign_e: &Tensor<B, 1, Int>,
    sel_w: &Tensor<B, 1>,
    h: usize,
    i: usize,
    n: usize,
) {
    assert_eq!(
        x.dtype(),
        DType::F32,
        "fused35 fp8 activations must be f32, got {:?}",
        x.dtype()
    );
    assert_eq!(
        q_gu.dtype(),
        DType::I8,
        "fused35 fp8 q_gu must be raw e4m3 bytes carried as I8, got {:?}",
        q_gu.dtype()
    );
    assert_eq!(
        q_dn.dtype(),
        DType::I8,
        "fused35 fp8 q_dn must be raw e4m3 bytes carried as I8, got {:?}",
        q_dn.dtype()
    );
    assert_eq!(s_gu.dtype(), DType::F32, "fused35 fp8 s_gu must be f32");
    assert_eq!(s_dn.dtype(), DType::F32, "fused35 fp8 s_dn must be f32");
    let e = q_gu.dims()[0];
    assert_eq!(q_gu.dims(), [e, h, i * 2], "q_gu must be [E,H,2I]");
    assert_eq!(s_gu.dims(), [e, i * 2], "s_gu must be [E,2I]");
    assert_eq!(q_dn.dims(), [e, i, h], "q_dn must be [E,I,H]");
    assert_eq!(s_dn.dims(), [e, h], "s_dn must be [E,H]");
    assert_eq!(assign_e.dims()[0], n, "assign_e len != N");
    assert_eq!(sel_w.dims()[0], n, "sel_w len != N");
    assert_eq!(x.dims()[1], h, "x H != sidecar H");
}

#[allow(clippy::too_many_arguments)]
fn assert_fused35_nvfp4_shapes<B: Backend>(
    x: &Tensor<B, 2>,
    q_gu: &Tensor<B, 3, Int>,
    bs_gu: &Tensor<B, 3, Int>,
    gscale_gu: &Tensor<B, 2>,
    q_dn: &Tensor<B, 3, Int>,
    bs_dn: &Tensor<B, 3, Int>,
    gscale_dn: &Tensor<B, 1>,
    assign_e: &Tensor<B, 1, Int>,
    sel_w: &Tensor<B, 1>,
    h: usize,
    i: usize,
    n: usize,
) {
    assert_eq!(
        x.dtype(),
        DType::F32,
        "fused35 nvfp4 activations must be f32, got {:?}",
        x.dtype()
    );
    assert_eq!(
        q_gu.dtype(),
        DType::I8,
        "fused35 nvfp4 q_gu must be raw e2m1 bytes carried as I8, got {:?}",
        q_gu.dtype()
    );
    assert_eq!(
        bs_gu.dtype(),
        DType::I8,
        "fused35 nvfp4 bs_gu must be raw e4m3 bytes carried as I8, got {:?}",
        bs_gu.dtype()
    );
    assert_eq!(
        q_dn.dtype(),
        DType::I8,
        "fused35 nvfp4 q_dn must be raw e2m1 bytes carried as I8, got {:?}",
        q_dn.dtype()
    );
    assert_eq!(
        bs_dn.dtype(),
        DType::I8,
        "fused35 nvfp4 bs_dn must be raw e4m3 bytes carried as I8, got {:?}",
        bs_dn.dtype()
    );
    assert_eq!(
        gscale_gu.dtype(),
        DType::F32,
        "fused35 nvfp4 gscale_gu must be f32"
    );
    assert_eq!(
        gscale_dn.dtype(),
        DType::F32,
        "fused35 nvfp4 gscale_dn must be f32"
    );
    assert_eq!(h % 16, 0, "fused35 nvfp4 requires H%16==0, got H={h}");
    assert_eq!(i % 16, 0, "fused35 nvfp4 requires I%16==0, got I={i}");
    assert_eq!(
        i % 2,
        0,
        "fused35 nvfp4 requires even I for gate/up nibble pairs, got I={i}"
    );
    assert_eq!(
        h % 2,
        0,
        "fused35 nvfp4 requires even H for down nibble pairs, got H={h}"
    );
    let e = q_gu.dims()[0];
    assert_eq!(q_gu.dims(), [e, h, i], "q_gu must be [E,H,I] bytes");
    assert_eq!(
        bs_gu.dims(),
        [e, i * 2, h / 16],
        "bs_gu must be [E,2I,H/16]"
    );
    assert_eq!(gscale_gu.dims(), [e, 2], "gscale_gu must be [E,2]");
    assert_eq!(q_dn.dims(), [e, i, h / 2], "q_dn must be [E,I,H/2] bytes");
    assert_eq!(bs_dn.dims(), [e, h, i / 16], "bs_dn must be [E,H,I/16]");
    assert_eq!(gscale_dn.dims(), [e], "gscale_dn must be [E]");
    assert_eq!(assign_e.dims()[0], n, "assign_e len != N");
    assert_eq!(sel_w.dims()[0], n, "sel_w len != N");
    assert_eq!(x.dims()[1], h, "x H != sidecar H");
}

pub trait Fused35MoeBackend: Backend {
    #[allow(clippy::too_many_arguments)]
    fn fused_moe_gu2_down_bf16(
        x: Tensor<Self, 2>,
        gate_up: Tensor<Self, 3>,
        down: Tensor<Self, 3>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2>;

    #[allow(clippy::too_many_arguments)]
    fn fused_moe_gu2_down_fp8(
        x: Tensor<Self, 2>,
        q_gu: Tensor<Self, 3, Int>,
        s_gu: Tensor<Self, 2>,
        q_dn: Tensor<Self, 3, Int>,
        s_dn: Tensor<Self, 2>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2>;

    #[allow(clippy::too_many_arguments)]
    fn fused_moe_gu2_down_nvfp4(
        x: Tensor<Self, 2>,
        q_gu: Tensor<Self, 3, Int>,
        bs_gu: Tensor<Self, 3, Int>,
        gscale_gu: Tensor<Self, 2>,
        q_dn: Tensor<Self, 3, Int>,
        bs_dn: Tensor<Self, 3, Int>,
        gscale_dn: Tensor<Self, 1>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2>;
}

impl Fused35MoeBackend for Cuda {
    fn fused_moe_gu2_down_bf16(
        x: Tensor<Cuda, 2>,
        gate_up: Tensor<Cuda, 3>,
        down: Tensor<Cuda, 3>,
        assign_e: Tensor<Cuda, 1, Int>,
        sel_w: Tensor<Cuda, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Cuda, 2> {
        assert_fused35_shapes(&x, &gate_up, &down, &assign_e, &sel_w, h, i, n);

        let x_prim = x.into_primitive().tensor();
        let gate_up_prim = gate_up.into_primitive().tensor();
        let down_prim = down.into_primitive().tensor();
        let ae_prim = assign_e.into_primitive();
        let sw_prim = sel_w.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("qwen35_fused_moe_gu2_down_bf16")
            .float_input(x_prim)
            .float_input(gate_up_prim)
            .float_input(down_prim)
            .int_input(ae_prim)
            .float_input(sw_prim)
            .float_output([n, h], DType::F32)
            .launch(move |inputs| vec![run_fused35_bf16(inputs, h, i, n)]);

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }

    fn fused_moe_gu2_down_fp8(
        x: Tensor<Cuda, 2>,
        q_gu: Tensor<Cuda, 3, Int>,
        s_gu: Tensor<Cuda, 2>,
        q_dn: Tensor<Cuda, 3, Int>,
        s_dn: Tensor<Cuda, 2>,
        assign_e: Tensor<Cuda, 1, Int>,
        sel_w: Tensor<Cuda, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Cuda, 2> {
        assert_fused35_fp8_shapes(&x, &q_gu, &s_gu, &q_dn, &s_dn, &assign_e, &sel_w, h, i, n);

        let x_prim = x.into_primitive().tensor();
        let q_gu_prim = q_gu.into_primitive();
        let s_gu_prim = s_gu.into_primitive().tensor();
        let q_dn_prim = q_dn.into_primitive();
        let s_dn_prim = s_dn.into_primitive().tensor();
        let ae_prim = assign_e.into_primitive();
        let sw_prim = sel_w.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("qwen35_fused_moe_gu2_down_fp8")
            .float_input(x_prim)
            .int_input(q_gu_prim)
            .float_input(s_gu_prim)
            .int_input(q_dn_prim)
            .float_input(s_dn_prim)
            .int_input(ae_prim)
            .float_input(sw_prim)
            .float_output([n, h], DType::F32)
            .launch(move |inputs| vec![run_fused35_fp8(inputs, h, i, n)]);

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }

    fn fused_moe_gu2_down_nvfp4(
        x: Tensor<Cuda, 2>,
        q_gu: Tensor<Cuda, 3, Int>,
        bs_gu: Tensor<Cuda, 3, Int>,
        gscale_gu: Tensor<Cuda, 2>,
        q_dn: Tensor<Cuda, 3, Int>,
        bs_dn: Tensor<Cuda, 3, Int>,
        gscale_dn: Tensor<Cuda, 1>,
        assign_e: Tensor<Cuda, 1, Int>,
        sel_w: Tensor<Cuda, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Cuda, 2> {
        assert_fused35_nvfp4_shapes(
            &x, &q_gu, &bs_gu, &gscale_gu, &q_dn, &bs_dn, &gscale_dn, &assign_e, &sel_w, h, i, n,
        );

        let x_prim = x.into_primitive().tensor();
        let q_gu_prim = q_gu.into_primitive();
        let bs_gu_prim = bs_gu.into_primitive();
        let gscale_gu_prim = gscale_gu.into_primitive().tensor();
        let q_dn_prim = q_dn.into_primitive();
        let bs_dn_prim = bs_dn.into_primitive();
        let gscale_dn_prim = gscale_dn.into_primitive().tensor();
        let ae_prim = assign_e.into_primitive();
        let sw_prim = sel_w.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("qwen35_fused_moe_gu2_down_nvfp4")
            .float_input(x_prim)
            .int_input(q_gu_prim)
            .int_input(bs_gu_prim)
            .float_input(gscale_gu_prim)
            .int_input(q_dn_prim)
            .int_input(bs_dn_prim)
            .float_input(gscale_dn_prim)
            .int_input(ae_prim)
            .float_input(sw_prim)
            .float_output([n, h], DType::F32)
            .launch(move |inputs| vec![run_fused35_nvfp4(inputs, h, i, n)]);

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }
}

#[cfg(all(test, feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn fused_moe_gu2_down_nvfp4_forced_cuda(
    x: Tensor<Cuda, 2>,
    q_gu: Tensor<Cuda, 3, Int>,
    bs_gu: Tensor<Cuda, 3, Int>,
    gscale_gu: Tensor<Cuda, 2>,
    q_dn: Tensor<Cuda, 3, Int>,
    bs_dn: Tensor<Cuda, 3, Int>,
    gscale_dn: Tensor<Cuda, 1>,
    assign_e: Tensor<Cuda, 1, Int>,
    sel_w: Tensor<Cuda, 1>,
    h: usize,
    i: usize,
    n: usize,
    path: Nvfp4LaunchPath,
) -> Tensor<Cuda, 2> {
    assert!(
        matches!(path, Nvfp4LaunchPath::Scalar | Nvfp4LaunchPath::SplitK),
        "test helper must force a concrete nvfp4 path, got {path:?}"
    );
    assert_fused35_nvfp4_shapes(
        &x, &q_gu, &bs_gu, &gscale_gu, &q_dn, &bs_dn, &gscale_dn, &assign_e, &sel_w, h, i, n,
    );

    let x_prim = x.into_primitive().tensor();
    let q_gu_prim = q_gu.into_primitive();
    let bs_gu_prim = bs_gu.into_primitive();
    let gscale_gu_prim = gscale_gu.into_primitive().tensor();
    let q_dn_prim = q_dn.into_primitive();
    let bs_dn_prim = bs_dn.into_primitive();
    let gscale_dn_prim = gscale_dn.into_primitive().tensor();
    let ae_prim = assign_e.into_primitive();
    let sw_prim = sel_w.into_primitive().tensor();

    let outputs = CubeCustomOp::<CudaRuntime>::new("qwen35_fused_moe_gu2_down_nvfp4_forced")
        .float_input(x_prim)
        .int_input(q_gu_prim)
        .int_input(bs_gu_prim)
        .float_input(gscale_gu_prim)
        .int_input(q_dn_prim)
        .int_input(bs_dn_prim)
        .float_input(gscale_dn_prim)
        .int_input(ae_prim)
        .float_input(sw_prim)
        .float_output([n, h], DType::F32)
        .launch(move |inputs| vec![run_fused35_nvfp4_with_path(inputs, h, i, n, path)]);

    Tensor::from_primitive(TensorPrimitive::Float(
        outputs.into_iter().next().expect("one output"),
    ))
}

impl Fused35MoeBackend for CubeBackend<CudaRuntime, f32, i32, u8> {
    fn fused_moe_gu2_down_bf16(
        x: Tensor<Self, 2>,
        gate_up: Tensor<Self, 3>,
        down: Tensor<Self, 3>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2> {
        assert_fused35_shapes(&x, &gate_up, &down, &assign_e, &sel_w, h, i, n);

        let inputs = [
            x.into_primitive().tensor(),
            gate_up.into_primitive().tensor(),
            down.into_primitive().tensor(),
            assign_e.into_primitive(),
            sel_w.into_primitive().tensor(),
        ];
        let out = run_fused35_bf16(&inputs, h, i, n);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    fn fused_moe_gu2_down_fp8(
        x: Tensor<Self, 2>,
        q_gu: Tensor<Self, 3, Int>,
        s_gu: Tensor<Self, 2>,
        q_dn: Tensor<Self, 3, Int>,
        s_dn: Tensor<Self, 2>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2> {
        assert_fused35_fp8_shapes(&x, &q_gu, &s_gu, &q_dn, &s_dn, &assign_e, &sel_w, h, i, n);
        let inputs = [
            x.into_primitive().tensor(),
            q_gu.into_primitive(),
            s_gu.into_primitive().tensor(),
            q_dn.into_primitive(),
            s_dn.into_primitive().tensor(),
            assign_e.into_primitive(),
            sel_w.into_primitive().tensor(),
        ];
        let out = run_fused35_fp8(&inputs, h, i, n);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }

    fn fused_moe_gu2_down_nvfp4(
        x: Tensor<Self, 2>,
        q_gu: Tensor<Self, 3, Int>,
        bs_gu: Tensor<Self, 3, Int>,
        gscale_gu: Tensor<Self, 2>,
        q_dn: Tensor<Self, 3, Int>,
        bs_dn: Tensor<Self, 3, Int>,
        gscale_dn: Tensor<Self, 1>,
        assign_e: Tensor<Self, 1, Int>,
        sel_w: Tensor<Self, 1>,
        h: usize,
        i: usize,
        n: usize,
    ) -> Tensor<Self, 2> {
        assert_fused35_nvfp4_shapes(
            &x, &q_gu, &bs_gu, &gscale_gu, &q_dn, &bs_dn, &gscale_dn, &assign_e, &sel_w, h, i, n,
        );
        let inputs = [
            x.into_primitive().tensor(),
            q_gu.into_primitive(),
            bs_gu.into_primitive(),
            gscale_gu.into_primitive().tensor(),
            q_dn.into_primitive(),
            bs_dn.into_primitive(),
            gscale_dn.into_primitive().tensor(),
            assign_e.into_primitive(),
            sel_w.into_primitive().tensor(),
        ];
        let out = run_fused35_nvfp4(&inputs, h, i, n);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

#[cfg(test)]
mod nvfp4_host_tests {
    fn e2m1_marlin_decode_host(code: u8) -> f32 {
        let top = (code & 0x0f) << 4;
        let fp8_bits = (top & 0x80) | ((top & 0x70) >> 2);
        crate::nvfp4::e4m3_to_f32(fp8_bits) * 64.0
    }

    #[test]
    fn e2m1_marlin_bit_trick_matches_lut_for_all_packed_bytes() {
        for byte in 0u16..=255 {
            let byte = byte as u8;
            let low = byte & 0x0f;
            let high = byte >> 4;
            let low_lut = crate::nvfp4::e2m1_bits_to_f32(low);
            let high_lut = crate::nvfp4::e2m1_bits_to_f32(high);
            let low_trick = e2m1_marlin_decode_host(low);
            let high_trick = e2m1_marlin_decode_host(high);
            assert!(
                low_lut.to_bits() == low_trick.to_bits(),
                "low nibble mismatch for byte 0x{byte:02x}: LUT={low_lut:?} trick={low_trick:?}"
            );
            assert!(
                high_lut.to_bits() == high_trick.to_bits(),
                "high nibble mismatch for byte 0x{byte:02x}: LUT={high_lut:?} trick={high_trick:?}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "splitk == 16")]
    fn fused35_nvfp4_rejects_non_16_splitk() {
        super::assert_nvfp4_splitk(8);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod fp8_tests {
    use super::*;
    use burn::{
        backend::cuda::{Cuda, CudaDevice},
        tensor::TensorData,
    };

    #[test]
    fn e4m3_line_reinterpret_matches_scalar_decode_cuda() {
        let device = CudaDevice::default();
        let bytes: Vec<u8> = vec![
            0x00, 0x01, 0x07, 0x08, 0x10, 0x20, 0x30, 0x38, 0x3f, 0x40, 0x48, 0x50, 0x58, 0x60,
            0x70, 0x7e, 0x80, 0x81, 0x87, 0x88, 0x90, 0xa0, 0xb0, 0xb8, 0xbf, 0xc0, 0xc8, 0xd0,
            0xd8, 0xe0, 0xf0, 0xfe,
        ];
        let q_i8: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let q = Tensor::<Cuda, 1, Int>::from_data_dtype(
            TensorData::new(q_i8, [bytes.len()]),
            &device,
            DType::I8,
        );
        let got = e4m3_line_decode_probe(q)
            .into_data()
            .to_vec::<f32>()
            .expect("line decode output");
        let want: Vec<f32> = bytes
            .iter()
            .map(|&b| crate::w8a16::e4m3_to_f32(b))
            .collect();
        assert_eq!(got.len(), want.len());
        for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert!(
                g.to_bits() == w.to_bits(),
                "lane {idx}: line reinterpret {g:?} != scalar decode {w:?} for byte 0x{:02x}",
                bytes[idx]
            );
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod nvfp4_decode_tests {
    use super::*;
    use burn::{
        backend::cuda::{Cuda, CudaDevice},
        tensor::TensorData,
    };

    fn marlin_e4m3_byte_for_e2m1(code: u8) -> u8 {
        let top = (code & 0x0f) << 4;
        (top & 0x80) | ((top & 0x70) >> 2)
    }

    #[test]
    fn e2m1_marlin_decode_probe_matches_host_lut_for_all_packed_bytes_cuda() {
        assert_eq!(marlin_e4m3_byte_for_e2m1(0x1), 0x04);
        assert_eq!(marlin_e4m3_byte_for_e2m1(0x9), 0x84);

        let device = CudaDevice::default();
        let bytes: Vec<u8> = (0u16..=255).map(|b| b as u8).collect();
        let q_i8: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        let q = Tensor::<Cuda, 1, Int>::from_data_dtype(
            TensorData::new(q_i8, [bytes.len()]),
            &device,
            DType::I8,
        );

        let got = e2m1_marlin_decode_probe(q)
            .into_data()
            .to_vec::<f32>()
            .expect("e2m1 decode probe output");
        let mut want = Vec::with_capacity(bytes.len() * 2);
        for &byte in &bytes {
            want.push(crate::nvfp4::e2m1_bits_to_f32(byte & 0x0f));
            want.push(crate::nvfp4::e2m1_bits_to_f32(byte >> 4));
        }

        assert_eq!(got.len(), want.len());
        for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            let byte = bytes[idx / 2];
            let nibble = if idx % 2 == 0 { "low" } else { "high" };
            assert!(
                g.to_bits() == w.to_bits(),
                "byte 0x{byte:02x} {nibble} nibble: kernel {g:?} != host LUT {w:?}"
            );
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod nvfp4_fused_tests {
    use super::*;
    use burn::{
        backend::cuda::{Cuda, CudaDevice},
        tensor::TensorData,
    };

    use crate::nvfp4::{dequant_nvfp4_outmajor, quantize_nvfp4, repack_kmajor_to_outmajor};

    const H: usize = 2048;
    const I: usize = 512;
    const TOP_K: usize = 8;

    #[derive(Clone)]
    struct Nvfp4ExpertFixture {
        qw_gu: Vec<u8>,
        bs_gu: Vec<u8>,
        gscale_gu: [f32; 2],
        qw_dn: Vec<u8>,
        bs_dn: Vec<u8>,
        gscale_dn: f32,
    }

    struct Nvfp4Fixture {
        experts: Vec<Nvfp4ExpertFixture>,
        q_gu: Tensor<Cuda, 3, Int>,
        bs_gu: Tensor<Cuda, 3, Int>,
        gscale_gu: Tensor<Cuda, 2>,
        q_dn: Tensor<Cuda, 3, Int>,
        bs_dn: Tensor<Cuda, 3, Int>,
        gscale_dn: Tensor<Cuda, 1>,
    }

    fn synth(seed: u64, idx: usize, scale: f32) -> f32 {
        let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
        (u * 2.0 - 1.0) * scale
    }

    fn synth_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|idx| synth(seed, idx, scale)).collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "cosine input lengths differ");
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "diff input lengths differ");
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn max_abs_value(xs: &[f32]) -> f32 {
        xs.iter().map(|x| x.abs()).fold(0.0f32, f32::max)
    }

    fn assert_rows_close(name: &str, got: &[f32], want: &[f32], rows: usize, cols: usize) {
        assert_eq!(got.len(), rows * cols, "{name}: got length mismatch");
        assert_eq!(want.len(), rows * cols, "{name}: want length mismatch");
        for row in 0..rows {
            let range = row * cols..(row + 1) * cols;
            let cos = cosine(&got[range.clone()], &want[range.clone()]);
            let max_abs = max_abs_diff(&got[range.clone()], &want[range.clone()]);
            let want_abs = max_abs_value(&want[range]);
            let max_allowed = 1.0e-3 * want_abs;
            assert!(cos > 0.9999, "{name} row {row}: cosine {cos:.9} <= 0.9999");
            assert!(
                max_abs <= max_allowed,
                "{name} row {row}: max|diff| {max_abs:.9} > 1e-3 * max|host| {max_allowed:.9} \
                 (max|host| {want_abs:.9})"
            );
        }
    }

    fn build_fixture(device: &CudaDevice, e: usize) -> Nvfp4Fixture {
        let mut experts = Vec::with_capacity(e);
        let mut q_gu_all = Vec::with_capacity(e * H * I);
        let mut bs_gu_all = Vec::with_capacity(e * 2 * I * (H / 16));
        let mut gscale_gu_all = Vec::with_capacity(e * 2);
        let mut q_dn_all = Vec::with_capacity(e * I * (H / 2));
        let mut bs_dn_all = Vec::with_capacity(e * H * (I / 16));
        let mut gscale_dn_all = Vec::with_capacity(e);

        for expert in 0..e {
            let scale_e = 1.0 + (expert % 5) as f32 * 0.11;
            let gate = synth_vec(0xB502_1000 + expert as u64, H * I, 0.020 * scale_e);
            let up = synth_vec(0xB502_2000 + expert as u64, H * I, 0.055 * scale_e);
            let down = synth_vec(0xB502_3000 + expert as u64, I * H, 0.025 * scale_e);

            let (q_gate, bs_gate, g_gate) = quantize_nvfp4(&gate, H, I);
            let (q_up, bs_up, g_up) = quantize_nvfp4(&up, H, I);
            assert_ne!(
                g_gate.to_bits(),
                g_up.to_bits(),
                "fixture expert {expert}: gate/up global scales must differ"
            );

            let mut q_gu_kmajor = q_gate;
            q_gu_kmajor.extend_from_slice(&q_up);
            let mut bs_gu = bs_gate;
            bs_gu.extend_from_slice(&bs_up);
            let qw_gu = repack_kmajor_to_outmajor(&q_gu_kmajor, H, 2 * I);

            let (q_dn, bs_dn, g_dn) = quantize_nvfp4(&down, I, H);
            let qw_dn = repack_kmajor_to_outmajor(&q_dn, I, H);

            q_gu_all.extend(qw_gu.iter().copied().map(|b| b as i8));
            bs_gu_all.extend(bs_gu.iter().copied().map(|b| b as i8));
            gscale_gu_all.extend([g_gate, g_up]);
            q_dn_all.extend(qw_dn.iter().copied().map(|b| b as i8));
            bs_dn_all.extend(bs_dn.iter().copied().map(|b| b as i8));
            gscale_dn_all.push(g_dn);
            experts.push(Nvfp4ExpertFixture {
                qw_gu,
                bs_gu,
                gscale_gu: [g_gate, g_up],
                qw_dn,
                bs_dn,
                gscale_dn: g_dn,
            });
        }

        Nvfp4Fixture {
            experts,
            q_gu: Tensor::<Cuda, 3, Int>::from_data_dtype(
                TensorData::new(q_gu_all, [e, H, I]),
                device,
                DType::I8,
            ),
            bs_gu: Tensor::<Cuda, 3, Int>::from_data_dtype(
                TensorData::new(bs_gu_all, [e, 2 * I, H / 16]),
                device,
                DType::I8,
            ),
            gscale_gu: Tensor::<Cuda, 2>::from_data(TensorData::new(gscale_gu_all, [e, 2]), device),
            q_dn: Tensor::<Cuda, 3, Int>::from_data_dtype(
                TensorData::new(q_dn_all, [e, I, H / 2]),
                device,
                DType::I8,
            ),
            bs_dn: Tensor::<Cuda, 3, Int>::from_data_dtype(
                TensorData::new(bs_dn_all, [e, H, I / 16]),
                device,
                DType::I8,
            ),
            gscale_dn: Tensor::<Cuda, 1>::from_data(TensorData::new(gscale_dn_all, [e]), device),
        }
    }

    fn routing(e: usize, t: usize) -> (Vec<i32>, Vec<f32>) {
        let n = t * TOP_K;
        let mut assign_e = Vec::with_capacity(n);
        let mut sel_w = Vec::with_capacity(n);
        for tok in 0..t {
            for k in 0..TOP_K {
                assign_e.push(((tok * 13 + k * 5 + 3) % e) as i32);
                sel_w.push(0.35 + 0.03 * k as f32 + 0.005 * tok as f32);
            }
        }
        (assign_e, sel_w)
    }

    fn dequant_gate_up(expert: &Nvfp4ExpertFixture) -> Vec<f32> {
        let mut gscale = vec![expert.gscale_gu[0]; I];
        gscale.extend(std::iter::repeat_n(expert.gscale_gu[1], I));
        dequant_nvfp4_outmajor(&expert.qw_gu, &expert.bs_gu, &gscale, H, 2 * I)
    }

    fn dequant_down(expert: &Nvfp4ExpertFixture) -> Vec<f32> {
        dequant_nvfp4_outmajor(&expert.qw_dn, &expert.bs_dn, &[expert.gscale_dn], I, H)
    }

    fn host_reference(
        fixture: &Nvfp4Fixture,
        x: &[f32],
        assign_e: &[i32],
        sel_w: &[f32],
        t: usize,
    ) -> Vec<f32> {
        let n = t * TOP_K;
        let mut gu_cache = vec![None; fixture.experts.len()];
        let mut dn_cache = vec![None; fixture.experts.len()];
        let mut mid = vec![0.0f32; I];
        let mut out = vec![0.0f32; n * H];

        for row in 0..n {
            let tok = row / TOP_K;
            let expert_idx = assign_e[row] as usize;
            if gu_cache[expert_idx].is_none() {
                gu_cache[expert_idx] = Some(dequant_gate_up(&fixture.experts[expert_idx]));
            }
            if dn_cache[expert_idx].is_none() {
                dn_cache[expert_idx] = Some(dequant_down(&fixture.experts[expert_idx]));
            }
            let gu = gu_cache[expert_idx].as_ref().expect("gate/up cache");
            let dn = dn_cache[expert_idx].as_ref().expect("down cache");

            for ci in 0..I {
                let mut gacc = 0.0f32;
                let mut uacc = 0.0f32;
                for hh in 0..H {
                    let xv = x[tok * H + hh];
                    let base = hh * 2 * I + ci;
                    gacc += xv * gu[base];
                    uacc += xv * gu[base + I];
                }
                let sig = 1.0f32 / (1.0f32 + (-gacc).exp());
                mid[ci] = gacc * sig * uacc;
            }

            for hh in 0..H {
                let mut acc = 0.0f32;
                for ci in 0..I {
                    acc += mid[ci] * dn[ci * H + hh];
                }
                out[row * H + hh] = acc * sel_w[row];
            }
        }
        out
    }

    fn host_gate_up_projection(
        fixture: &Nvfp4Fixture,
        x: &[f32],
        assign_e: &[i32],
        t: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let n = t * TOP_K;
        let mut gu_cache = vec![None; fixture.experts.len()];
        let mut gate = vec![0.0f32; n * I];
        let mut up = vec![0.0f32; n * I];

        for row in 0..n {
            let tok = row / TOP_K;
            let expert_idx = assign_e[row] as usize;
            if gu_cache[expert_idx].is_none() {
                gu_cache[expert_idx] = Some(dequant_gate_up(&fixture.experts[expert_idx]));
            }
            let gu = gu_cache[expert_idx].as_ref().expect("gate/up cache");
            for ci in 0..I {
                let mut gacc = 0.0f32;
                let mut uacc = 0.0f32;
                for hh in 0..H {
                    let xv = x[tok * H + hh];
                    let base = hh * 2 * I + ci;
                    gacc += xv * gu[base];
                    uacc += xv * gu[base + I];
                }
                gate[row * I + ci] = gacc;
                up[row * I + ci] = uacc;
            }
        }
        (gate, up)
    }

    fn run_forced_path(
        device: &CudaDevice,
        fixture: &Nvfp4Fixture,
        x: &[f32],
        assign_e: &[i32],
        sel_w: &[f32],
        t: usize,
        path: Nvfp4LaunchPath,
    ) -> Vec<f32> {
        let n = t * TOP_K;
        let x_t = Tensor::<Cuda, 2>::from_data(TensorData::new(x.to_vec(), [t, H]), device);
        let assign_t = Tensor::<Cuda, 1, Int>::from_data_dtype(
            TensorData::new(assign_e.to_vec(), [n]),
            device,
            DType::I32,
        );
        let sel_t = Tensor::<Cuda, 1>::from_data(TensorData::new(sel_w.to_vec(), [n]), device);
        fused_moe_gu2_down_nvfp4_forced_cuda(
            x_t,
            fixture.q_gu.clone(),
            fixture.bs_gu.clone(),
            fixture.gscale_gu.clone(),
            fixture.q_dn.clone(),
            fixture.bs_dn.clone(),
            fixture.gscale_dn.clone(),
            assign_t,
            sel_t,
            H,
            I,
            n,
            path,
        )
        .into_data()
        .to_vec::<f32>()
        .expect("fused nvfp4 output")
    }

    #[test]
    fn fused35_nvfp4_kernel_vs_host_parity_splitk_and_scalar_cuda() {
        let device = CudaDevice::default();
        for e in [8usize, 32] {
            let fixture = build_fixture(&device, e);
            for t in [1usize, 2, 8, 16] {
                let n = t * TOP_K;
                let x = synth_vec(0xB502_4000 + e as u64 * 17 + t as u64, t * H, 0.50);
                let (assign_e, sel_w) = routing(e, t);
                let want = host_reference(&fixture, &x, &assign_e, &sel_w, t);

                for path in [Nvfp4LaunchPath::SplitK, Nvfp4LaunchPath::Scalar] {
                    let got = run_forced_path(&device, &fixture, &x, &assign_e, &sel_w, t, path);
                    assert!(
                        got.iter().all(|v| v.is_finite()),
                        "E={e} T={t} path={path:?}: fused output contains NaN/Inf"
                    );
                    assert_rows_close(&format!("E={e} T={t} path={path:?}"), &got, &want, n, H);
                }
            }
        }
    }

    #[test]
    fn fused35_nvfp4_gate_up_projection_offsets_match_unfused_gemv_cuda() {
        let device = CudaDevice::default();
        let e = 8usize;
        let t = 2usize;
        let n = t * TOP_K;
        let fixture = build_fixture(&device, e);
        let x = synth_vec(0xB502_5000, t * H, 0.50);
        let (assign_e, _) = routing(e, t);
        let (gate_want, up_want) = host_gate_up_projection(&fixture, &x, &assign_e, t);

        let x_t = Tensor::<Cuda, 2>::from_data(TensorData::new(x, [t, H]), &device);
        let assign_t = Tensor::<Cuda, 1, Int>::from_data_dtype(
            TensorData::new(assign_e, [n]),
            &device,
            DType::I32,
        );
        let (gate_got, up_got) = fused35_nvfp4_projection_probe_cuda(
            x_t,
            fixture.q_gu,
            fixture.bs_gu,
            fixture.gscale_gu,
            assign_t,
            H,
            I,
            n,
        );
        let gate_got = gate_got
            .into_data()
            .to_vec::<f32>()
            .expect("gate projection output");
        let up_got = up_got
            .into_data()
            .to_vec::<f32>()
            .expect("up projection output");

        assert_rows_close("gate projection", &gate_got, &gate_want, n, I);
        assert_rows_close("up projection", &up_got, &up_want, n, I);
        assert!(
            max_abs_diff(&gate_want, &up_want) > 1.0e-3,
            "fixture must make gate/up projections distinct enough to catch half swaps"
        );
    }

    #[test]
    #[should_panic(expected = "splitk == 16")]
    fn fused35_nvfp4_cuda_rejects_non_16_splitk() {
        assert_nvfp4_splitk(8);
    }
}
