//! Fused **W8A16** (fp8 weight-only) GEMM — `docs/VLLM_KERNELS.md` §2. CUDA only.
//!
//! The decode rollout is weight-bandwidth-bound, so the win is reading **half the weight BYTES** from
//! HBM: store each weight as one OCP **E4M3** byte (per-output-channel symmetric scale) instead of two
//! bf16 bytes. The load-bearing rule (three independent reviews) is that the dequant must happen
//! **inside the GEMM's load path** — "dequant the whole weight to bf16, then call a normal GEMM"
//! round-trips HBM (fp8 read + bf16 write + bf16 read) and gives NO win. So [`gpu::w8a16_gemm`] reads
//! the packed e4m3 weight byte from HBM, dequants it **in-register** (`w = e4m3_to_f32(byte) * s[n]`),
//! and multiply-accumulates in f32 — the weight is NEVER materialized as a full bf16/f32 tensor.
//!
//! It reaches the GPU through the typed Fusion-bridge wrapper [`crate::cube_custom_op::CubeCustomOp`]
//! (§0b production rules): the packed e4m3 weight is an **`int_input`** (a 1-byte `i8` tensor — fp8 has
//! no Burn float DType and Burn's Int kind has no `u8`, so the raw e4m3 byte rides in an `i8`; §0b rule
//! 4 routes it through `get_int_tensor`), while the activations and the per-channel scale are
//! `float_input`s. The packed weight is passed **as-is** (no `into_contiguous`, §0b rule 5); the kernel
//! indexes it with the tensor's own strides.
//!
//! The e4m3 byte → f32 decode uses CubeCL's native `e4m3` minifloat
//! (`f32::cast_from(e4m3::reinterpret(byte))`, the same primitive `cubecl-std`'s dequant kernel uses) —
//! a hardware `cvt` on the GB10 (sm_121 / Blackwell). `reinterpret` is a pure same-size bitcast (no
//! sign-extension — verified in cubecl source), and `f32_to_e4m3` is OCP-faithful (SatFinite→448,
//! round-to-nearest-even). The host quantizer/dequantizer use the *same* OCP e4m3 codec, so a CPU oracle
//! that dequants the same bytes is bit-exact to the weights the kernel sees (modulo f32 accumulation
//! order). Validated against an independent NdArray (CPU f32) oracle + OCP golden vectors by
//! `examples/w8a16_spike.rs`.
//!
//! ⚠️ STATUS — a CORRECT, proven GEMM, but NOT a usable GRPO-rollout component yet (3-voice review,
//! Opus 4.8 tracing the trainer). Do NOT wire this into the rollout as-is:
//!  * **It breaks GRPO logprob parity (P0).** The trainer keeps `ratio = exp(logp_pi − logp_old) ≈ 1`
//!    by running the rollout AND the grad-tracked policy recompute through the SAME f32 `linear3`. This
//!    op is forward-ONLY (no autodiff backward), so it can only live in the no-grad rollout — making the
//!    rollout fp8 while the recompute stays f32. The ~per-layer quant error then compounds over all
//!    layers into a logprob shift that silently BIASES the policy gradient. The real gate is an
//!    end-to-end **logprob-parity test (fp8-rollout logp vs f32-recompute logp)**, not GEMM cosine.
//!  * **Wrong regime (P1).** The batched GRPO rollout decodes `[n,1]` with `n = prompts × group_size > 1`
//!    every step, so the GEMM is always M>1 — where this flat 1-thread-per-output kernel RE-READS the
//!    weight column M× (no shared-mem reuse) and runs CUDA-core f32 FMA instead of bf16 tensor cores →
//!    almost certainly SLOWER than bf16 `linear3`. The "half the bytes" win exists ONLY at true M=1
//!    (batch-1 serving), which the rollout never hits.
//!  * **It is W8A32, not W8A16** (f32 activations) — matches today's f32-typed model, but not "A16".
//!  * **Perf + integration unbuilt:** split-K/shared-mem/CMMA (the perf path), and the e4m3 load path /
//!    `W8A16Linear` swap (the layout `[d_in,d_out]=[K,N]` matches; Qwen3 is bias-free; the tied lm_head
//!    needs a contiguous materialize). Implication for docs/VLLM_PARITY_PLAN.md: fp8-storage is a batch-1
//!    serving lever, NOT the GRPO-rollout decode lever the plan assumed.

use burn::backend::cuda::Cuda;

use burn::tensor::{DType, Int, Tensor, TensorPrimitive};

use cubecl::cuda::CudaRuntime;
use cubecl::e4m3;
use cubecl::prelude::CubeElement;
use cubecl::{CubeCount, CubeDim};

use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;

use crate::capture::CaptureBackend;
use crate::cube_custom_op::CubeCustomOp;

/// Largest finite OCP E4M3 magnitude (`0x7E` = `1.75 * 2^8`).
pub const E4M3_MAX: f32 = 448.0;

// =================================================================================================
// Host codec (OCP E4M3) — shared by the quantizer, the CPU oracle, and the golden-vector test, so
// the bytes the GPU decodes and the bytes the oracle decodes go through one identical codec.
// =================================================================================================

/// Decode one OCP E4M3 byte to f32 on the host (lossless: every e4m3 value is exact in f32). This is
/// the SAME decode the GPU kernel performs (`e4m3::from_bits` + cast), so the oracle is faithful.
#[inline]
pub fn e4m3_to_f32(byte: u8) -> f32 {
    e4m3::from_bits(byte).to_f32()
}

/// Encode an f32 to its OCP E4M3 byte on the host (round-to-nearest, OCP saturation).
#[inline]
pub fn f32_to_e4m3(x: f32) -> u8 {
    e4m3::from_f32(x).to_bits()
}

/// Quantize a row-major weight `W:[K,N]` (f32) to packed e4m3 bytes `q:[K,N]` + a per-**output-channel**
/// symmetric scale `s:[N]`, with the canonical convention `s[n] = max_k|W[k,n]| / 448` and
/// `q[k,n] = e4m3(W[k,n] / s[n])`. (Dequant is `w = e4m3_to_f32(q[k,n]) * s[n]` — never the
/// `inv_scale=448/amax` variant; one canonical field, per §2.)
pub fn quantize_e4m3_per_channel(w: &[f32], k: usize, n: usize) -> (Vec<u8>, Vec<f32>) {
    assert_eq!(
        w.len(),
        k * n,
        "weight length {} != K*N = {}",
        w.len(),
        k * n
    );
    // Reject non-finite weights up front (Codex+Gemini review): a NaN never updates `amax`, and an Inf
    // makes `amax=Inf -> scale=Inf -> W/scale = NaN`, silently poisoning the column. Real checkpoints
    // are finite; fail loud if not.
    assert!(
        w.iter().all(|x| x.is_finite()),
        "quantize_e4m3_per_channel: weight contains non-finite values (NaN/Inf)"
    );

    // Per-output-channel amax over the K (input) dim.
    let mut amax = vec![0.0f32; n];
    for kk in 0..k {
        let row = kk * n;
        for nn in 0..n {
            let a = w[row + nn].abs();
            if a > amax[nn] {
                amax[nn] = a;
            }
        }
    }

    // scale[n] = amax/448, floored away from 0 so an all-zero column can't divide by zero.
    let scale: Vec<f32> = amax
        .iter()
        .map(|&a| (a / E4M3_MAX).max(f32::MIN_POSITIVE))
        .collect();

    // q[k,n] = e4m3(W[k,n] / scale[n]); |W/scale| <= 448 by construction, so no overflow to NaN.
    let mut q = vec![0u8; k * n];
    for kk in 0..k {
        let row = kk * n;
        for nn in 0..n {
            q[row + nn] = f32_to_e4m3(w[row + nn] / scale[nn]);
        }
    }
    (q, scale)
}

/// Dequantize packed e4m3 bytes `q:[K,N]` + scale `s:[N]` back to f32 `[K,N]` on the host — the CPU
/// oracle's weight (the exact bytes the GPU kernel decodes). `w[k,n] = e4m3_to_f32(q[k,n]) * s[n]`.
pub fn dequant_e4m3(q: &[u8], scale: &[f32], k: usize, n: usize) -> Vec<f32> {
    assert_eq!(q.len(), k * n, "q length {} != K*N = {}", q.len(), k * n);
    assert_eq!(scale.len(), n, "scale length {} != N = {}", scale.len(), n);
    let mut w = vec![0.0f32; k * n];
    for kk in 0..k {
        let row = kk * n;
        for nn in 0..n {
            w[row + nn] = e4m3_to_f32(q[row + nn]) * scale[nn];
        }
    }
    w
}

// =================================================================================================
// GPU kernels. In their own module so `cubecl::prelude::Tensor` (the GPU-side tensor) does not clash
// with `burn::tensor::Tensor` used on the host above.
// =================================================================================================
mod gpu {
    use cubecl::e4m3;
    use cubecl::prelude::*;

    /// STEP A — byte-level E4M3 → f32 decode micro-test. One thread per byte. Proves CubeCL can decode
    /// an e4m3 byte to f32 ON THE GPU: reinterpret the raw byte's bits as `e4m3` (`reinterpret` — a
    /// pure bitcast, the byte is carried in a 1-byte `i8` tensor since Burn's Int kind has no `u8`),
    /// then convert to f32 (`cast_from` → a hardware `cvt`). Same primitive the fused GEMM uses,
    /// de-risked in isolation against OCP golden vectors.
    #[cube(launch)]
    pub fn e4m3_decode(q: &Tensor<i8>, out: &mut Tensor<f32>) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            // reinterpret: the i8 byte's bits ARE the e4m3 (no conversion). cast_from: e4m3 -> f32 cvt.
            out[pos] = f32::cast_from(e4m3::reinterpret(q[pos]));
        }
    }

    /// STEP C — the FUSED W8A16 GEMM. `x:[M,K]` (f32), packed weight `q:[K,N]` (e4m3 BYTES, u8),
    /// per-output-channel scale `s:[N]` (f32) → `out:[M,N]` (f32). One thread per output element.
    ///
    /// The weight is read as e4m3 **bytes** straight from HBM and dequanted **in-register** in the
    /// accumulation loop — it is NOT pre-expanded to a full f32/bf16 weight tensor (that round-trip is
    /// exactly what §2 rejects). Indexing uses the tensors' own strides, so the packed weight needs no
    /// `into_contiguous` (§0b rule 5).
    #[cube(launch)]
    pub fn w8a16_gemm(
        x: &Tensor<f32>,       // [M, K]
        q: &Tensor<i8>,        // [K, N]  packed e4m3 weight bytes (1 byte/elem, raw e4m3 bits)
        s: &Tensor<f32>,       // [N]     per-output-channel scale
        out: &mut Tensor<f32>, // [M, N]
    ) {
        if ABSOLUTE_POS < out.len() {
            let pos = ABSOLUTE_POS as usize;
            let n_dim = out.shape(1);
            let k_dim = x.shape(1);
            let m = pos / n_dim;
            let n = pos % n_dim;

            // Per-output-channel scale, loaded ONCE into a register (constant over the K loop).
            let sn = s[n * s.stride(0)];

            let x_row = m * x.stride(0); // start of activation row m
            let q_col = n * q.stride(1); // column n offset into the packed weight

            // f32 accumulator. Reassigned scalar MUST be `f32::new(..)`, never a bare `0.0` literal
            // (a literal binds to an immutable const -> "Can't assign a value to a const variable").
            let mut acc = f32::new(0.0);
            for kk in 0..k_dim {
                let xv = x[x_row + kk * x.stride(1)];
                // ---- the FUSED dequant-in-load: read ONE e4m3 byte, decode + scale in-register ----
                let qb = q[kk * q.stride(0) + q_col]; // 1 byte from HBM (half of bf16's 2 bytes)
                let w = f32::cast_from(e4m3::reinterpret(qb)) * sn; // dequant in-register
                acc += xv * w; // f32 multiply-accumulate
            }
            out[pos] = acc;
        }
    }
}

// =================================================================================================
// Host dispatch (through the typed Fusion-bridge wrapper).
// =================================================================================================

/// Allocate a fresh contiguous f32 output `CubeTensor` of `shape` on the same client as `like`.
fn alloc_f32(like: &CubeTensor<CudaRuntime>, shape: &[usize]) -> CubeTensor<CudaRuntime> {
    let n: usize = shape.iter().product();
    let buffer = like.client.empty(n * DType::F32.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.to_vec().into(),
        buffer,
        DType::F32,
    )
}

/// STEP A — decode a 1-D tensor of e4m3 **bytes** (a 1-byte `I8` Int tensor) to f32 on the GPU.
///
/// fp8 has no Burn float DType and Burn's Int kind has no `u8`, so the raw e4m3 byte is carried in a
/// 1-byte `I8` tensor (bit-preserving `byte as i8`; the kernel reinterprets the bits as e4m3). Build it
/// with `Tensor::<1, Int>::from_data(TensorData::new(i8_bytes, ([n]), &dev, DType::I8))`.
pub fn e4m3_decode(q: Tensor<1, Int>) -> Tensor<1> {
    assert_eq!(
        q.dtype(),
        DType::I8,
        "e4m3_decode expects packed e4m3 bytes as a 1-byte I8 tensor, got {:?}",
        q.dtype()
    );
    let n = q.dims()[0];
    let q_prim = q.into_primitive(); // Int primitive (a FusionTensor) — routed via get_int_tensor

    let outputs = CubeCustomOp::<CudaRuntime>::new("e4m3_decode")
        .int_input(q_prim) // packed bytes: an INT handle (§0b rule 4)
        .float_output([n], DType::F32)
        .launch(move |inputs| {
            // The packed weight is passed AS-IS (no into_contiguous — §0b rule 5).
            let q = inputs[0].clone();
            let out = alloc_f32(&q, &[n]);

            let threads = 256u32;
            let blocks = (n as u32).div_ceil(threads);
            gpu::e4m3_decode::launch::<CudaRuntime>(
                &q.client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim {
                    x: threads,
                    y: 1,
                    z: 1,
                },
                q.as_tensor_arg(1),
                out.as_tensor_arg(1),
            )
            .expect("e4m3_decode launch failed");
            vec![out]
        });

    Tensor::from_primitive(TensorPrimitive::Float(
        outputs.into_iter().next().expect("one output"),
    ))
}

/// STEP C — the fused W8A16 GEMM on the default CUDA (Fusion) backend.
///
/// * `x` — activations `[M, K]` (f32).
/// * `q` — packed e4m3 weight `[K, N]`, a 1-byte `DType::I8` Int tensor (raw e4m3 bits; build with
///   `from_data(.., (DType::I8)`).
/// * `s` — per-output-channel scale `[N]` (f32), `s[n] = max_k|W[k,n]| / 448`.
///
/// Returns `y:[M,N]` (f32)), `y[m,n] = sum_k x[m,k] * (e4m3_to_f32(q[k,n]) * s[n])`.
fn run_w8a16_gemm_tensors(
    x: CubeTensor<CudaRuntime>,
    q: CubeTensor<CudaRuntime>,
    s: CubeTensor<CudaRuntime>,
) -> CubeTensor<CudaRuntime> {
    let x = into_contiguous(x);
    // The kernel reads q with CubeCL tensor strides (`q.stride(0/1)`), so q can remain a view in the
    // Fusion path. w8a16 q is plain [K,N] row-major, not swizzled; raw helpers upload it contiguous.
    let s = into_contiguous(s);

    let [m, k] = x.meta.shape().dims::<2>();
    let [qk, n] = q.meta.shape().dims::<2>();
    let [sn] = s.meta.shape().dims::<1>();
    assert_eq!(
        x.dtype,
        DType::F32,
        "w8a16_gemm activations must be f32, got {:?}",
        x.dtype
    );
    assert_eq!(
        q.dtype,
        DType::I8,
        "w8a16_gemm expects packed e4m3 weight as a 1-byte I8 tensor, got {:?}",
        q.dtype
    );
    assert_eq!(
        s.dtype,
        DType::F32,
        "w8a16_gemm scale must be f32, got {:?}",
        s.dtype
    );
    assert_eq!(qk, k, "weight K ({qk}) must match activation K ({k})");
    assert_eq!(sn, n, "scale length ({sn}) must match weight N ({n})");

    let out = alloc_f32(&x, &[m, n]);

    let total = (m * n) as u32;
    let threads = 256u32;
    let blocks = total.div_ceil(threads);
    gpu::w8a16_gemm::launch::<CudaRuntime>(
        &x.client,
        CubeCount::Static(blocks, 1, 1),
        CubeDim {
            x: threads,
            y: 1,
            z: 1,
        },
        x.as_tensor_arg(1),
        q.as_tensor_arg(1),
        s.as_tensor_arg(1),
        out.as_tensor_arg(1),
    )
    .expect("w8a16_gemm launch failed");

    out
}

#[cfg(feature = "cuda")]
pub trait W8A16GemvBackend: Backend {
    fn w8a16_gemv(x: Tensor<2>, q: Tensor<2, Int>, s: Tensor<1>) -> Tensor<2>;
}

#[cfg(feature = "cuda")]
impl W8A16GemvBackend for Cuda {
    fn w8a16_gemv(x: Tensor<2>, q: Tensor<2, Int>, s: Tensor<1>) -> Tensor<2> {
        let [m, n] = [x.dims()[0], q.dims()[1]];
        let x_prim = x.into_primitive().tensor();
        let q_prim = q.into_primitive(); // Int primitive routed via get_int_tensor (§0b rule 4)
        let s_prim = s.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("w8a16_gemm")
            .float_input(x_prim) // every read tensor is a declared input (rule 1 / no closure capture)
            .int_input(q_prim) // packed e4m3 weight = INT handle (rule 4)
            .float_input(s_prim)
            .float_output([m, n], DType::F32) // cross-validated vs the alloc (rule 2)
            .launch(move |inputs| {
                let x = inputs[0].clone();
                let q = inputs[1].clone();
                let s = inputs[2].clone();
                vec![run_w8a16_gemm_tensors(x, q, s)]
            });

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }
}

#[cfg(feature = "cuda")]
impl W8A16GemvBackend for CubeBackend<CudaRuntime, f32, i32, u8> {
    fn w8a16_gemv(x: Tensor<2>, q: Tensor<2, Int>, s: Tensor<1>) -> Tensor<2> {
        let x = x.into_primitive().tensor();
        let q = q.into_primitive();
        let s = s.into_primitive().tensor();
        let out = run_w8a16_gemm_tensors(x, q, s);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}

/// Run W8A16 GEMV on the raw CUDA `CubeBackend` below Fusion.
///
/// `x` is `[M,K]` f32, `q_bytes` is `[K,N]` raw OCP E4M3 bytes carried as `i8`, `scale` is `[N]`
/// f32, and the returned tensor is `[M,N]` f32. This helper is for probes and CUDA-graph capture
/// setup from host-side packed weights.
#[cfg(feature = "cuda")]
pub fn w8a16_gemv_raw(
    x: Tensor<2>,
    q_bytes: &[i8],
    scale: &[f32],
    k: usize,
    n: usize,
) -> Tensor<2> {
    let [_m, xk] = x.dims();
    assert_eq!(xk, k, "w8a16_gemv_raw: x K ({xk}) != requested K ({k})");
    assert_eq!(x.dtype(), DType::F32, "w8a16_gemv_raw: x must be f32");
    assert_eq!(
        q_bytes.len(),
        k * n,
        "w8a16_gemv_raw: q length {} != K*N = {}",
        q_bytes.len(),
        k * n
    );
    assert_eq!(
        scale.len(),
        n,
        "w8a16_gemv_raw: scale length {} != N = {}",
        scale.len(),
        n
    );
    assert!(
        scale.iter().all(|s| s.is_finite() && *s > 0.0),
        "w8a16_gemv_raw: scale must be finite and positive"
    );

    let x_ct = x.into_primitive().tensor();
    let client = x_ct.client.clone();
    let device = x_ct.device.clone();
    let q_raw: Vec<u8> = q_bytes.iter().map(|&b| b as u8).collect();
    let q_handle = client.create_from_slice(&q_raw);
    let s_handle = client.create_from_slice(f32::as_bytes(scale));
    let q = CubeTensor::new_contiguous(
        client.clone(),
        device.clone(),
        vec![k, n].into(),
        q_handle,
        DType::I8,
    );
    let s = CubeTensor::new_contiguous(client, device, vec![n].into(), s_handle, DType::F32);
    <CaptureBackend as W8A16GemvBackend>::w8a16_gemv(
        Tensor::from_primitive(TensorPrimitive::Float(x_ct)),
        Tensor::from_primitive(q),
        Tensor::from_primitive(TensorPrimitive::Float(s)),
    )
}

pub fn w8a16_gemm(x: Tensor<2>, q: Tensor<2, Int>, s: Tensor<1>) -> Tensor<2> {
    <Cuda as W8A16GemvBackend>::w8a16_gemv(x, q, s)
}
