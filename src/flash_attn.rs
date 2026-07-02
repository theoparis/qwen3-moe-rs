//! Custom CubeCL FlashAttention-style attention kernel (tiled online-softmax, f32 accumulation).
//! CUDA only — `docs/VLLM_KERNELS.md` §1. Reaches the GPU through the typed Fusion-bridge wrapper
//! [`crate::cube_custom_op::CubeCustomOp`], so EVERY tensor the kernel reads is a declared
//! `float_input` (never captured into the closure — that would be a use-after-free; §0b).
//!
//! Validated on the real GB10 against an INDEPENDENT NdArray (CPU f32) oracle by
//! `examples/attn_kernel_spike.rs`: cosine = 1.000000 (max_abs_diff ~1e-6) on decode (`q_len=1`)
//! AND prefill (`q_len=S`), short (64/512) and long (2048) context, GQA ratio 4, batch > 1,
//! head_dim 64 & 128 — tighter to the oracle than Burn's fused `attention_fallback` (~1e-4).
//!
//! Algorithm (FA-2 online softmax, the canonical recurrence; f32 state). Per query row `i`, scanning
//! the causally-visible keys `[0 ..= q_global]`:
//! `m_new = max(m, s_k)`; `alpha = exp(m - m_new)`; `p_k = exp(s_k - m_new)`; `l = alpha*l + p_k`;
//! `acc = alpha*acc + p_k*V_k` (the `acc *= alpha` rescale is the most-dropped FA-2 term); finally
//! once `O_i = acc / l`. Guards: only visible keys are scanned, so the diagonal key is ALWAYS present
//! and `exp(-inf - -inf) = NaN` cannot arise (a finite `-1e30` sentinel for `m`, never `-inf`); `l<=0
//! -> O=0`; `q_idx` is GLOBAL (`q_offset = kv_len - q_len`, KV-cache aware); GQA `kv_head = h /
//! n_rep` reads K/V un-expanded; offsets use the tensor's native `usize` strides.
//!
//! SCOPE: correctness-first. One cube per `(q-head, batch, query-row)`; a single thread per cube
//! scans the KV. Split-K / Flash-Decode (KV partition + cross-split LSE merge) and tensor-core MMA
//! tiling are PERF follow-ons. The 256-f32 per-thread state (`q_reg[128] + acc[128]`) spills to local
//! memory at head_dim=128 (correct, not fast); a tiled/MMA rewrite is the perf path.
//!
//! NOT A DROP-IN SDPA REPLACEMENT YET (3-voice review — Codex gpt-5.5 / Opus 4.8 / Gemini 3.1 Pro).
//! It is a VALIDATED-correct foundation (proves the bridge→wrapper→kernel path end-to-end), but to
//! wire into `src/attention.rs` it still needs:
//!  1. **A padding-mask input.** This kernel is CAUSAL-ONLY. On the left-padded ragged GRPO path
//!     (`grpo_step_ragged`, which forwards `Some(mask)`), real tokens would attend to pad keys →
//!     wrong logprobs/ratio/KL (Opus P0). The uniform/causal-only paths (mask `None`) are correct.
//!  2. **A bf16-input/f32-accumulate variant.** This is f32-only; the host fn asserts f32 so a bf16
//!     tensor fails LOUD, not silently reinterpreted. Latent today: every supported path is f32-typed
//!     (`linear3` casts back to f32; inference is forced f32), so the f32 kernel matches the current model.
//!  3. **Perf.** The single-thread serial scan is ~10-100× SLOWER than the cuBLAS/CMMA reference SDPA at
//!     decode (1 thread/cube → <2% GB10 occupancy) — a wall-clock REGRESSION. Flash-Decode split-K +
//!     warp-parallel + CMMA is the unbuilt perf path.
//!  4. **Drop the `into_contiguous` on K/V** (it copies the whole KV cache each call → O(T²) at decode)
//!     and restructure `attention.rs` to pass `[B,H,S,D]` + `n_rep` instead of GQA-expanding.

use burn::backend::cuda::Cuda;
use burn::tensor::{DType, Tensor, TensorPrimitive};

use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ScalarArg;
use cubecl::{CubeCount, CubeDim};

use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;

use crate::cube_custom_op::CubeCustomOp;

mod gpu {
    use cubecl::prelude::*;

    /// FlashAttention-2 online-softmax attention, f32 accumulation. One cube per
    /// `(q-head = CUBE_POS_X, batch = CUBE_POS_Y, query-row = CUBE_POS_Z)`, single thread.
    /// Contiguous tensors: `q:[B,Hq,Sq,D]`, `k,v:[B,Hkv,Sk,D]` (GQA: NOT expanded), `out:[B,Hq,Sq,D]`.
    #[cube(launch)]
    pub fn flash_attn(
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<f32>,
        out: &mut Tensor<f32>,
        scale: f32,
        n_rep: u32,
        #[comptime] head_dim: usize,
    ) {
        let h = CUBE_POS_X; // query head
        let b = CUBE_POS_Y; // batch
        let qi = CUBE_POS_Z; // query row within Sq
        let kv_h = h / n_rep; // GQA: query head -> kv head

        let sq = q.shape(2);
        let sk = k.shape(2);
        let q_offset = sk - sq; // KV-cache global offset: queries occupy the LAST Sq of Sk.

        // Base offsets via the tensor's own (usize) strides; last-dim stride is 1 (contiguous), so
        // element d is at base + d. usize keeps offsets at the backend's native pointer width.
        let q_base =
            (b as usize) * q.stride(0) + (h as usize) * q.stride(1) + (qi as usize) * q.stride(2);
        let k_base0 = (b as usize) * k.stride(0) + (kv_h as usize) * k.stride(1);
        let v_base0 = (b as usize) * v.stride(0) + (kv_h as usize) * v.stride(1);
        let o_base = (b as usize) * out.stride(0)
            + (h as usize) * out.stride(1)
            + (qi as usize) * out.stride(2);
        let ks2 = k.stride(2);
        let vs2 = v.stride(2);

        // Reassigned scalars/arrays use `f32::new(..)`, NOT a bare `0.0f32` literal: a bare literal
        // binds to an IMMUTABLE const element and a later `x = ..` panics in the JIT
        // ("Can't assign a value to a const variable"); `f32::new(..)` yields a mutable runtime local.
        let mut q_reg = Array::<f32>::new(head_dim);
        let mut acc = Array::<f32>::new(head_dim);
        for d in 0..head_dim {
            q_reg[d] = q[q_base + d];
            acc[d] = f32::new(0.0);
        }

        let q_global = q_offset + (qi as usize);
        let n_keys = q_global + 1; // causal: visible keys are [0 ..= q_global]

        let mut m = f32::new(-1.0e30); // running max (finite sentinel, never -inf)
        let mut l = f32::new(0.0); // running denominator (sum of exp)

        for kj in 0..n_keys {
            // s_k = scale * dot(q_row, K_k)
            let k_base = k_base0 + kj * ks2;
            let mut s = f32::new(0.0);
            for d in 0..head_dim {
                s += q_reg[d] * k[k_base + d];
            }
            s *= scale;

            // FA-2 online-softmax recurrence (f32). `max` is the cubecl free function (the `.max()`
            // method does not expand in a cube kernel on this rev).
            let m_new = max(m, s);
            let alpha = (m - m_new).exp(); // rescale factor for the PRIOR running state
            let p = (s - m_new).exp(); // this key's unnormalized weight
            l = alpha * l + p;

            // acc = alpha*acc + p*V_k — the alpha rescale of acc is the most-dropped FA-2 term.
            let v_base = v_base0 + kj * vs2;
            for d in 0..head_dim {
                acc[d] = alpha * acc[d] + p * v[v_base + d];
            }
            m = m_new;
        }

        // Normalize once; guard l<=0 -> O=0 (cannot occur here as >=1 key is always visible).
        if l > 0.0f32 {
            let inv = 1.0f32 / l;
            for d in 0..head_dim {
                out[o_base + d] = acc[d] * inv;
            }
        } else {
            for d in 0..head_dim {
                out[o_base + d] = 0.0f32;
            }
        }
    }
}

/// Causal grouped-query FlashAttention on the default CUDA (Fusion) backend.
///
/// * `q`   — `[batch, num_q_heads, q_len, head_dim]` (post-RoPE / post-QK-norm).
/// * `k`,`v` — `[batch, num_kv_heads, kv_len, head_dim]` (GQA, NOT expanded). `num_q_heads` must be a
///   multiple of `num_kv_heads`.
/// * `scale` — usually `1 / sqrt(head_dim)`.
///
/// Returns `[batch, num_q_heads, q_len, head_dim]`. Decode is `q_len = 1`; prefill is `q_len = kv_len`
/// (or any `q_len <= kv_len`, with the queries occupying the last `q_len` positions of the KV).
pub fn flash_attention(
    q: Tensor<Cuda, 4>,
    k: Tensor<Cuda, 4>,
    v: Tensor<Cuda, 4>,
    scale: f32,
) -> Tensor<Cuda, 4> {
    let [bsz, hq, sq, d] = q.dims();
    let [kb, hkv, sk, kd] = k.dims();
    let [vb, vhkv, vsk, vd] = v.dims();
    // FAIL LOUD on the gaps both external reviews (Codex gpt-5.5 + Gemini 3.1 Pro) flagged P0/P1 — this
    // is a CORRECTNESS-validated but NOT production-ready spike (f32-only, suffix-causal, single-thread):
    // * dtype: f32-only. The bf16 model path needs a `<F: Float>` bf16-input/f32-accumulate variant;
    //   without this assert a bf16 tensor would be silently reinterpreted as f32 → OOB / NaN.
    assert!(
        q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32,
        "flash_attention is f32-only; got q={:?} k={:?} v={:?}. The bf16 rollout path needs a \
         bf16-input/f32-accumulate kernel variant (docs/VLLM_KERNELS.md §1).",
        q.dtype(), k.dtype(), v.dtype(),
    );
    // * shapes: sq <= sk (else `q_offset = sk - sq` underflows the usize → OOB key scan), and k/v must
    //   share batch / kv-heads / kv-len / head_dim.
    assert!(sq <= sk, "flash_attention: q_len ({sq}) must be <= kv_len ({sk})");
    assert!(
        kb == bsz && vb == bsz && vhkv == hkv && vsk == sk && kd == d && vd == d,
        "flash_attention: q/k/v shape mismatch (q=[{bsz},{hq},{sq},{d}] k=[{kb},{hkv},{sk},{kd}] v=[{vb},{vhkv},{vsk},{vd}])"
    );
    // * CAUSAL CONTRACT: queries are assumed to be the SUFFIX of the KV (`q_offset = sk - sq`), i.e. the
    //   caller passes the FILLED K/V (length = current context), NOT a max-capacity pre-allocated buffer
    //   (the Phase-2 static cache returns the filled `[.., 0..filled, ..]` slice — pass THAT). A paged /
    //   non-suffix layout needs an explicit `q_start` param (follow-on).
    assert!(
        hkv != 0 && hq % hkv == 0,
        "flash_attention: num_q_heads ({hq}) must be a multiple of num_kv_heads ({hkv})"
    );
    let n_rep = (hq / hkv) as u32;

    let q_prim = q.into_primitive().tensor();
    let k_prim = k.into_primitive().tensor();
    let v_prim = v.into_primitive().tensor();

    let outputs = CubeCustomOp::<CudaRuntime>::new("flash_attn")
        .float_input(q_prim) // every read tensor is a declared input (rule 1 / no closure capture)
        .float_input(k_prim)
        .float_input(v_prim)
        .float_output([bsz, hq, sq, d], DType::F32) // cross-validated vs the alloc (rule 2)
        .launch(move |inputs| {
            let q = into_contiguous(inputs[0].clone());
            let k = into_contiguous(inputs[1].clone());
            let v = into_contiguous(inputs[2].clone());

            let n = bsz * hq * sq * d;
            let buffer = q.client.empty(n * DType::F32.size());
            let out = CubeTensor::new_contiguous(
                q.client.clone(),
                q.device.clone(),
                [bsz, hq, sq, d].into(),
                buffer,
                DType::F32,
            );

            // One cube per (q-head, batch, query-row); single thread per cube.
            gpu::flash_attn::launch::<CudaRuntime>(
                &q.client,
                CubeCount::Static(hq as u32, bsz as u32, sq as u32),
                CubeDim { x: 1, y: 1, z: 1 },
                q.as_tensor_arg(1),
                k.as_tensor_arg(1),
                v.as_tensor_arg(1),
                out.as_tensor_arg(1),
                ScalarArg::new(scale),
                ScalarArg::new(n_rep),
                d, // comptime head_dim (specializes the kernel per head_dim)
            )
            .expect("flash_attn launch failed");
            vec![out]
        });

    Tensor::from_primitive(TensorPrimitive::Float(outputs.into_iter().next().expect("one output")))
}

/// L2A.1 (plan §5 A3): the SAME proven FA-2 kernel launched on the RAW `CubeBackend` (below Fusion),
/// NOT through the `CubeCustomOp` Fusion bridge. Required for CUDA-graph capture (Lane 2B): the
/// captured region records the launch list from CODE, so the kernel must launch directly on the raw
/// client rather than being deferred into the lazy Fusion queue. Numerics are identical to
/// [`flash_attention`] (same `gpu::flash_attn` body); only the launch path differs.
///
/// `q:[B,Hq,Sq,D]`, `k`,`v:[B,Hkv,Sk,D]` (GQA NOT expanded) on [`crate::capture::CaptureBackend`].
#[cfg(feature = "cuda")]
pub fn flash_attention_raw(
    q: Tensor<crate::capture::CaptureBackend, 4>,
    k: Tensor<crate::capture::CaptureBackend, 4>,
    v: Tensor<crate::capture::CaptureBackend, 4>,
    scale: f32,
) -> Tensor<crate::capture::CaptureBackend, 4> {
    let [bsz, hq, sq, d] = q.dims();
    let [kb, hkv, sk, kd] = k.dims();
    let [vb, vhkv, vsk, vd] = v.dims();
    assert!(
        q.dtype() == DType::F32 && k.dtype() == DType::F32 && v.dtype() == DType::F32,
        "flash_attention_raw is f32-only; got q={:?} k={:?} v={:?}",
        q.dtype(), k.dtype(), v.dtype(),
    );
    assert!(sq <= sk, "flash_attention_raw: q_len ({sq}) must be <= kv_len ({sk})");
    assert!(
        kb == bsz && vb == bsz && vhkv == hkv && vsk == sk && kd == d && vd == d,
        "flash_attention_raw: q/k/v shape mismatch"
    );
    assert!(
        hkv != 0 && hq % hkv == 0,
        "flash_attention_raw: num_q_heads ({hq}) must be a multiple of num_kv_heads ({hkv})"
    );
    let n_rep = (hq / hkv) as u32;

    // Raw CubeTensor handles (below Fusion). into_contiguous so the kernel's usize-stride offsets hold.
    let q_ct = into_contiguous(q.into_primitive().tensor());
    let k_ct = into_contiguous(k.into_primitive().tensor());
    let v_ct = into_contiguous(v.into_primitive().tensor());

    let nelem = bsz * hq * sq * d;
    let buffer = q_ct.client.empty(nelem * DType::F32.size());
    let out = CubeTensor::new_contiguous(
        q_ct.client.clone(),
        q_ct.device.clone(),
        [bsz, hq, sq, d].into(),
        buffer,
        DType::F32,
    );

    gpu::flash_attn::launch::<CudaRuntime>(
        &q_ct.client,
        CubeCount::Static(hq as u32, bsz as u32, sq as u32),
        CubeDim { x: 1, y: 1, z: 1 },
        q_ct.as_tensor_arg(1),
        k_ct.as_tensor_arg(1),
        v_ct.as_tensor_arg(1),
        out.as_tensor_arg(1),
        ScalarArg::new(scale),
        ScalarArg::new(n_rep),
        d,
    )
    .expect("flash_attn raw launch failed");

    Tensor::from_primitive(TensorPrimitive::Float(out))
}
