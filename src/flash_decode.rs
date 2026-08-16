//! L2A.2 — split-K online-softmax FLASH-DECODE (q_len=1), per the workflow-vetted blueprint
//! `docs/specs/L2A.2-split-k-flash-decode-design.md`. Two raw-`CubeBackend` kernels (below Fusion,
//! capture-ready): `flash_decode_split` (one warp per (q-head, batch, kv-split); D-partitioned across
//! the 32 lanes; per-key QK dot reduced with `plane_sum` — validated on sm_121 by
//! `examples/plane_sum_probe.rs`; FA-2 recurrence over the split's KV chunk → per-split raw
//! `acc_g` + `(m_g, l_g)`) and `flash_decode_combine` (cross-split max-rescale merge → `out`).
//!
//! Merge math (avoids a `log`; exact global softmax): with per-split running-max `m_g`, denominator
//! `l_g` and un-normalized `acc_g = Σ_k exp(s_k - m_g) V_k`, the global output is
//! `O = Σ_g e^{m_g - m*} acc_g / Σ_g e^{m_g - m*} l_g`, `m* = max_g m_g`. Empty splits (`l_g=0`,
//! `m_g=-1e30`) contribute weight ~0.
//!
//! CORRECTNESS-FIRST scope: f32; scalar (non-`Line`) strided D-partition (lane ℓ owns dims
//! {ℓ, ℓ+32, …}); layout `q:[B,Hq,1,D]`, `k,v:[B,Hkv,Sk,D]` matching `flash_attention_raw` (the
//! cache-native `[B,T_max,Hkv,D]` + device-`pos` bound + `Line` vectorization + bf16 are the next
//! increments). Verified vs a CPU oracle in `examples/flash_raw_smoke.rs`.

use burn::backend::cuda::Cuda;

use burn::tensor::{DType, Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::ScalarArg;
use cubecl::{CubeCount, CubeDim};

use crate::capture::CaptureBackend;
use crate::cube_custom_op::CubeCustomOp;

mod gpu {
    use cubecl::prelude::*;

    /// Pass 1: one warp per `(q_head=CUBE_POS_X, batch=CUBE_POS_Y, kv_split=CUBE_POS_Z)`, 32 lanes.
    /// D is partitioned STRIDED across lanes: lane `ℓ` owns dims `{ℓ, ℓ+32, …}` (`dpl=head_dim/32`).
    /// The per-key dot is a lane-partial then `plane_sum` → the full scalar dot on all lanes. Emits
    /// the per-split RAW `acc_g` (each lane its dpl dims) and `(m_g, l_g)` (written by lane 0).
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn flash_decode_split<EW: Float>(
        q: &Tensor<f32>,           // [B, Hq, 1, D]
        k: &Tensor<EW>,            // [B, Hkv, Sk, D]
        v: &Tensor<EW>,            // [B, Hkv, Sk, D]
        acc_out: &mut Tensor<f32>, // [S, B, Hq, D]  un-normalized acc_g
        m_out: &mut Tensor<f32>,   // [S, B, Hq]     running max m_g
        l_out: &mut Tensor<f32>,   // [S, B, Hq]     denominator l_g
        scale: f32,
        n_rep: u32,
        split_len: u32, // RUNTIME (not comptime): one compiled kernel serves every sk — no JIT-per-token
        // recompile and no capture-time range bake-in (3-voice review, Gemini #1).
        #[comptime] head_dim: usize,
    ) {
        let h = CUBE_POS_X;
        let b = CUBE_POS_Y;
        let g = CUBE_POS_Z;
        let lane = UNIT_POS_X as usize;
        let kv_h = h / n_rep;
        let dpl = head_dim / 32; // comptime; head_dim % 32 == 0 asserted host-side

        let sk = k.shape(2);
        let start = (g * split_len) as usize;
        let end_raw = ((g + 1) * split_len) as usize;
        let end = if end_raw < sk { end_raw } else { sk }; // last split clamps to Sk (coverage)

        let q_base = (b as usize) * q.stride(0) + (h as usize) * q.stride(1);
        let mut q_reg = Array::<f32>::new(dpl);
        let mut acc = Array::<f32>::new(dpl);
        for r in 0..dpl {
            q_reg[r] = q[q_base + (lane + r * 32)];
            acc[r] = f32::new(0.0);
        }
        let mut m = f32::new(-3.4028235e38); // f32::MIN sentinel: a masked score of f32::MIN yields exp(0)=1
        // (uniform), not a silent zero as -1e30 did (3-voice review).
        let mut l = f32::new(0.0);

        let k_base0 = (b as usize) * k.stride(0) + (kv_h as usize) * k.stride(1);
        let v_base0 = (b as usize) * v.stride(0) + (kv_h as usize) * v.stride(1);
        let ks2 = k.stride(2);
        let vs2 = v.stride(2);

        for kj in start..end {
            let k_base = k_base0 + kj * ks2;
            let mut partial = f32::new(0.0);
            for r in 0..dpl {
                partial += q_reg[r] * f32::cast_from(k[k_base + (lane + r * 32)]);
            }
            let s = plane_sum(partial) * scale; // full dot, uniform on all 32 lanes
            let m_new = max(m, s);
            let alpha = (m - m_new).exp();
            let p = (s - m_new).exp();
            l = alpha * l + p;
            let v_base = v_base0 + kj * vs2;
            for r in 0..dpl {
                acc[r] = alpha * acc[r] + p * f32::cast_from(v[v_base + (lane + r * 32)]);
            }
            m = m_new;
        }

        let a_base = (g as usize) * acc_out.stride(0)
            + (b as usize) * acc_out.stride(1)
            + (h as usize) * acc_out.stride(2);
        for r in 0..dpl {
            acc_out[a_base + (lane + r * 32)] = acc[r];
        }
        if lane == 0 {
            let ml_idx = (g as usize) * m_out.stride(0)
                + (b as usize) * m_out.stride(1)
                + (h as usize) * m_out.stride(2);
            // empty split (no keys) ⇒ l stays 0, m stays -1e30 ⇒ combine weight ~0.
            m_out[ml_idx] = m;
            l_out[ml_idx] = l;
        }
    }

    /// Pass 2: one warp per `(q_head, batch)`, D-partitioned. Merge:
    /// `O = Σ_g e^{m_g-m*} acc_g / Σ_g e^{m_g-m*} l_g`, `m*=max_g m_g`. `n_splits` comptime → unrolls.
    #[cube(launch)]
    pub fn flash_decode_combine(
        acc_in: &Tensor<f32>,  // [S, B, Hq, D]
        m_in: &Tensor<f32>,    // [S, B, Hq]
        l_in: &Tensor<f32>,    // [S, B, Hq]
        out: &mut Tensor<f32>, // [B, Hq, 1, D]
        #[comptime] head_dim: usize,
        #[comptime] n_splits: usize,
    ) {
        let h = CUBE_POS_X;
        let b = CUBE_POS_Y;
        let lane = UNIT_POS_X as usize;
        let dpl = head_dim / 32;

        let mut m_star = f32::new(-3.4028235e38); // f32::MIN (see split kernel)
        for gg in 0..n_splits {
            let idx =
                gg * m_in.stride(0) + (b as usize) * m_in.stride(1) + (h as usize) * m_in.stride(2);
            m_star = max(m_star, m_in[idx]);
        }

        let mut num = Array::<f32>::new(dpl);
        for r in 0..dpl {
            num[r] = f32::new(0.0);
        }
        let mut den = f32::new(0.0);
        for gg in 0..n_splits {
            let ml =
                gg * m_in.stride(0) + (b as usize) * m_in.stride(1) + (h as usize) * m_in.stride(2);
            let c = (m_in[ml] - m_star).exp(); // 0 for empty (m=-1e30)
            den += c * l_in[ml];
            let a_base = gg * acc_in.stride(0)
                + (b as usize) * acc_in.stride(1)
                + (h as usize) * acc_in.stride(2);
            for r in 0..dpl {
                num[r] += c * acc_in[a_base + (lane + r * 32)];
            }
        }

        let out_base = (b as usize) * out.stride(0) + (h as usize) * out.stride(1);
        let inv = if den > 0.0f32 {
            1.0f32 / den
        } else {
            f32::new(0.0)
        };
        for r in 0..dpl {
            out[out_base + (lane + r * 32)] = num[r] * inv;
        }
    }
}

/// Split-K flash-decode on the raw `CubeBackend` (below Fusion). `q:[B,Hq,1,D]`, `k,v:[B,Hkv,Sk,D]`
/// (GQA NOT expanded). `n_splits` partitions the KV range. Correctness-first: f32, scalar strided
/// D-partition. Verified vs a CPU oracle in `examples/flash_raw_smoke.rs`.
pub fn flash_decode_raw(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f32,
    n_splits: usize,
) -> Tensor<4> {
    <CaptureBackend as FlashDecodeBackend>::flash_decode(q, k, v, scale, n_splits)
}

fn assert_flash_decode_shapes(
    op: &str,
    q: &Tensor<4>,
    k: &Tensor<4>,
    v: &Tensor<4>,
    n_splits: usize,
) {
    let [bsz, hq, sq, d] = q.dims();
    let [kb, hkv, sk, kd] = k.dims();
    let [vb, vhkv, vsk, vd] = v.dims();
    let kv_dtype = k.dtype();
    assert_eq!(sq, 1, "{op} is decode-only (q_len=1); got sq={sq}");
    assert!(q.dtype() == DType::F32, "{op}: q must be f32");
    assert_eq!(kv_dtype, v.dtype(), "{op}: k/v dtype mismatch");
    assert_eq!(
        d % 32,
        0,
        "{op}: head_dim ({d}) must be a multiple of 32 (warp D-partition)"
    );
    assert!(
        hkv != 0 && hq % hkv == 0,
        "{op}: hq ({hq}) must be a multiple of hkv ({hkv})"
    );
    assert!(
        kb == bsz && vb == bsz && vhkv == hkv && vsk == sk && kd == d && vd == d,
        "{op}: q/k/v shape mismatch"
    );
    assert!(n_splits >= 1, "{op}: n_splits must be >= 1");
}

#[cfg(feature = "cuda")]
fn run_flash_decode_tensors(
    q: CubeTensor<CudaRuntime>,
    k: CubeTensor<CudaRuntime>,
    v: CubeTensor<CudaRuntime>,
    scale: f32,
    n_splits: usize,
) -> CubeTensor<CudaRuntime> {
    let q_shape = q.meta.shape();
    let k_shape = k.meta.shape();
    let [bsz, hq, sq, d] = q_shape.dims();
    let [_, hkv, sk, _] = k_shape.dims();
    debug_assert_eq!(sq, 1);
    let n_rep = (hq / hkv) as u32;
    let split_len = sk.div_ceil(n_splits); // ceil(Sk / n_splits): full coverage, last split clamps

    let q_ct = into_contiguous(q);
    let k_ct = into_contiguous(k);
    let v_ct = into_contiguous(v);
    let client = q_ct.client.clone();
    let device = q_ct.device.clone();

    let mk = |shape: Vec<usize>| -> CubeTensor<CudaRuntime> {
        let nelem: usize = shape.iter().product();
        let buffer = client.empty(nelem * DType::F32.size());
        CubeTensor::new_contiguous(
            client.clone(),
            device.clone(),
            shape.into(),
            buffer,
            DType::F32,
        )
    };
    let acc_out = mk(vec![n_splits, bsz, hq, d]);
    let m_out = mk(vec![n_splits, bsz, hq]);
    let l_out = mk(vec![n_splits, bsz, hq]);
    let out = mk(vec![bsz, hq, 1, d]);

    match k_ct.dtype {
        DType::BF16 => gpu::flash_decode_split::launch::<half::bf16, CudaRuntime>(
            &client,
            CubeCount::Static(hq as u32, bsz as u32, n_splits as u32),
            CubeDim { x: 32, y: 1, z: 1 },
            q_ct.as_tensor_arg(1),
            k_ct.as_tensor_arg(1),
            v_ct.as_tensor_arg(1),
            acc_out.as_tensor_arg(1),
            m_out.as_tensor_arg(1),
            l_out.as_tensor_arg(1),
            ScalarArg::new(scale),
            ScalarArg::new(n_rep),
            ScalarArg::new(split_len as u32),
            d,
        )
        .expect("flash_decode_split launch failed"),
        DType::F32 => gpu::flash_decode_split::launch::<f32, CudaRuntime>(
            &client,
            CubeCount::Static(hq as u32, bsz as u32, n_splits as u32),
            CubeDim { x: 32, y: 1, z: 1 },
            q_ct.as_tensor_arg(1),
            k_ct.as_tensor_arg(1),
            v_ct.as_tensor_arg(1),
            acc_out.as_tensor_arg(1),
            m_out.as_tensor_arg(1),
            l_out.as_tensor_arg(1),
            ScalarArg::new(scale),
            ScalarArg::new(n_rep),
            ScalarArg::new(split_len as u32),
            d,
        )
        .expect("flash_decode_split launch failed"),
        d => panic!("flash_decode_raw: unsupported k/v dtype {d:?} (expected bf16 or f32)"),
    }

    gpu::flash_decode_combine::launch::<CudaRuntime>(
        &client,
        CubeCount::Static(hq as u32, bsz as u32, 1),
        CubeDim { x: 32, y: 1, z: 1 },
        acc_out.as_tensor_arg(1),
        m_out.as_tensor_arg(1),
        l_out.as_tensor_arg(1),
        out.as_tensor_arg(1),
        d,
        n_splits,
    )
    .expect("flash_decode_combine launch failed");

    out
}

#[cfg(feature = "cuda")]
pub trait FlashDecodeBackend: Backend {
    fn flash_decode(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        scale: f32,
        n_splits: usize,
    ) -> Tensor<4>;
}

#[cfg(feature = "cuda")]
impl FlashDecodeBackend for Cuda {
    fn flash_decode(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        scale: f32,
        n_splits: usize,
    ) -> Tensor<4> {
        assert_flash_decode_shapes("flash_decode", &q, &k, &v, n_splits);
        let [bsz, hq, _, d] = q.dims();
        let q = q.into_primitive().tensor();
        let k = k.into_primitive().tensor();
        let v = v.into_primitive().tensor();

        let outputs = CubeCustomOp::<CudaRuntime>::new("flash_decode")
            .float_input(q)
            .float_input(k)
            .float_input(v)
            .float_output([bsz, hq, 1, d], DType::F32)
            .launch(move |inputs| {
                vec![run_flash_decode_tensors(
                    inputs[0].clone(),
                    inputs[1].clone(),
                    inputs[2].clone(),
                    scale,
                    n_splits,
                )]
            });

        Tensor::from_primitive(TensorPrimitive::Float(
            outputs.into_iter().next().expect("one output"),
        ))
    }
}

#[cfg(feature = "cuda")]
impl FlashDecodeBackend for CubeBackend<CudaRuntime, f32, i32, u8> {
    fn flash_decode(
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        scale: f32,
        n_splits: usize,
    ) -> Tensor<4> {
        assert_flash_decode_shapes("flash_decode_raw", &q, &k, &v, n_splits);
        let q = q.into_primitive().tensor();
        let k = k.into_primitive().tensor();
        let v = v.into_primitive().tensor();
        let out = run_flash_decode_tensors(q, k, v, scale, n_splits);
        Tensor::from_primitive(TensorPrimitive::Float(out))
    }
}
