//! Batch-safe Linear application for the CubeCL CUDA backend, with optional bf16 compute.
//!
//! ## Why this exists
//! On NVIDIA GB10 / sm_121 (CUDA 13), CubeCL's **batched** matmul (`[batch, seq, K] @ [K, N]`,
//! which is what `burn::nn::Linear::forward` does for a 3-D input) selects, for some `(M, K, N)`
//! shapes, an autotuned kernel that produces **incorrect results for batch > 1**: rows past the
//! first are corrupted (verified: `|row0 - row1|` up to ~3-5 for IDENTICAL input rows, e.g.
//! `[2,128,1024] @ [1024,2048]` and `[2,255,1024] @ [1024,1024]`). The corruption is shape- and
//! autotune-cache-dependent and is NOT avoided by any `CUBECL_AUTOTUNE_LEVEL`. This was the true
//! root cause of "batch>1 training is wrong": at batch=1 the matmul is effectively 2-D and always
//! correct (which is why batch=1 overfit worked); at batch>1 the corruption appears in the very
//! first decoder layer's projections and the loss plateaus.
//!
//! ## The fix
//! Flatten the leading dims: `[batch, seq, K] -> [batch*seq, K]`, run a **2-D** GEMM
//! `[batch*seq, K] @ [K, N]` (the non-batched matmul path, which is correct for all tested
//! shapes), then reshape back to `[batch, seq, N]`. Mathematically identical to the batched form
//! (a Linear is applied independently per token), but it dodges the buggy batched kernel.
//! Verified: the 2-D path gives `|row0 - row1| = 0` for every shape where the 3-D path failed.
//!
//! ## bf16 mixed precision
//! `linear3` takes a [`Precision`]. With [`Precision::Bf16`] the GEMM runs in bf16 (cast both
//! operands to bf16; the CubeCL CUDA matmul accumulates in f32 — cubek `Acc=(bf16,f32)`) and the
//! bf16 output is widened back to f32 for the rest of the network. The 2-D flatten still dodges
//! the broadcast batched-matmul bug. Master weights stay f32: the per-forward weight cast is the
//! autodiff edge that routes an f32-typed gradient back to the f32 parameter (Burn `float_cast`
//! backward returns the gradient in the source dtype), so the optimizer state stays f32.

use burn::nn::Linear;
use burn::tensor::{DType, Tensor};

/// Compute precision for the Linear GEMMs. A NAMED type rather than a bare `bool`, so each of the
/// ~11 call sites reads `Precision::Bf16` instead of a mystery `true` (no boolean-trap footgun).
///
/// VALIDATED BACKEND (review): `Bf16` is validated ONLY on the CubeCL **CUDA** backend (NVIDIA
/// GB10 / sm_121), whose matmul accumulates in f32 (cubek `Acc=(bf16,f32)`). Other Burn backends
/// (NdArray CPU, WGPU) may not support a bf16 matmul, or may accumulate in bf16 — using `Bf16`
/// there can panic or silently change numerics. `F32` (the default) is safe on every backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Precision {
    /// Full f32 GEMM (the existing, always-correct path).
    #[default]
    F32,
    /// bf16 inputs, f32 accumulation (cubek `Acc=(bf16,f32)`), bf16 output widened back to f32.
    Bf16,
    /// f16 (IEEE half) inputs/output, widened back to f32. Unlike `Bf16`, confirmed to work on
    /// Metal (`examples/f16_matmul_probe.rs`): cubecl-wgpu/Metal's fused-matmul autotune has no
    /// candidate for BF16 lhs/rhs (panics with "required feature is unavailable"), but F16 matmul
    /// runs and gives correct results. Not validated for accumulation precision/numerics beyond
    /// basic correctness — use with care for training; fine for inference-only streamed experts.
    F16,
}

/// bf16 GEMM: cast both operands to bf16, matmul (accumulates in f32 on the CubeCL CUDA backend),
/// widen the bf16 output back to f32. Autodiff-safe: the per-forward weight cast is the gradient
/// edge that routes the **f32-typed** gradient back to the f32 master weight.
fn matmul_bf16(a: Tensor<2>, w: Tensor<2>) -> Tensor<2> {
    a.cast(DType::BF16)
        .matmul(w.cast(DType::BF16))
        .cast(DType::F32)
}

/// f16 GEMM: cast both operands to f16 (IEEE half), matmul, widen back to f32. Unlike bf16, this
/// is confirmed to actually execute on the Metal backend (see `Precision::F16` docs).
fn matmul_f16(a: Tensor<2>, w: Tensor<2>) -> Tensor<2> {
    a.cast(DType::F16)
        .matmul(w.cast(DType::F16))
        .cast(DType::F32)
}

/// Apply a [`Linear`] to a 3-D tensor `[batch, seq, d_input]` via a 2-D GEMM, returning
/// `[batch, seq, d_output]`. See module docs for why the 3-D batched path is avoided. `prec`
/// selects f32 (default) or bf16 compute for the GEMM.
pub fn linear3(lin: &Linear, x: Tensor<3>, prec: Precision) -> Tensor<3> {
    let [batch, seq, d_in] = x.dims();
    let x2 = x.reshape([batch * seq, d_in]); // [B*S, d_in]
    let xdt = x2.dtype();
    let y2 = match prec {
        // F32 precision means "preserve the activation stream dtype", not "force f32 operands": the GEMM
        // must be UNIFORM-dtype. Mixed bf16/f32 matmul silently corrupts on the CubeCL CUDA backend in
        // BOTH directions, proven by examples/matmul_mixed_probe.rs. Cast weight+bias to the ACTIVATION
        // dtype: f32 stream (35B/qwen3_5, GRPO f32 checkpoints) -> f32xf32; bf16 stream (30B moe) ->
        // bf16xbf16 (the pre-4ca3fb9 verified behavior). Casting to a FIXED dtype instead re-creates the
        // mixed pairing on the other stream (the 30B prefill-garbage regression, 2026-07-01).
        Precision::F32 => {
            let w = lin.weight.val().cast(xdt);
            let mut y = x2.matmul(w);
            if let Some(bias) = &lin.bias {
                y = y + bias.val().cast(xdt).unsqueeze();
            }
            y
        }
        // bf16 path: weight-only GEMM in bf16, then add the (f32) bias if present.
        Precision::Bf16 => {
            let mut y = matmul_bf16(x2, lin.weight.val()); // [B*S, d_out]
            if let Some(bias) = &lin.bias {
                y = y + bias.val().unsqueeze();
            }
            y
        }
        Precision::F16 => {
            let mut y = matmul_f16(x2, lin.weight.val());
            if let Some(bias) = &lin.bias {
                y = y + bias.val().unsqueeze();
            }
            y
        }
    };
    let d_out = y2.dims()[1];
    y2.reshape([batch, seq, d_out]) // [B, S, d_out]
}
