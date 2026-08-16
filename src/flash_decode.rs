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
//!
//! Backend-generic (`R: CubeRuntime`, same pattern as `gdn_kernel.rs`): runs on CUDA, Metal, Vulkan
//! and portable wgpu, not just CUDA — the split-K algorithm has no CUDA-specific dependency, only the
//! original wiring hardcoded `CudaRuntime`.

use burn::tensor::{DType, Tensor};
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{CubeCount, CubeDim};

use crate::cubecl_rt::alloc_f32;

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

/// Backend-generic split-K flash-decode over raw `CubeTensor<R>`s. `q:[B,Hq,1,D]`,
/// `k,v:[B,Hkv,Sk,D]` (GQA NOT expanded — the split kernel divides by `n_rep` internally, so the
/// caller must NOT pre-repeat K/V across query heads). Runs on any `CubeRuntime` (CUDA, Metal,
/// Vulkan, portable wgpu).
pub fn launch_flash_decode<R: CubeRuntime>(
    q: CubeTensor<R>,
    k: CubeTensor<R>,
    v: CubeTensor<R>,
    scale: f32,
    n_splits: usize,
) -> CubeTensor<R> {
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

    let acc_out = alloc_f32(&q_ct, &[n_splits, bsz, hq, d]);
    let m_out = alloc_f32(&q_ct, &[n_splits, bsz, hq]);
    let l_out = alloc_f32(&q_ct, &[n_splits, bsz, hq]);
    let out = alloc_f32(&q_ct, &[bsz, hq, 1, d]);

    match k_ct.dtype {
        DType::BF16 => gpu::flash_decode_split::launch::<half::bf16, R>(
            &client,
            CubeCount::Static(hq as u32, bsz as u32, n_splits as u32),
            CubeDim { x: 32, y: 1, z: 1 },
            q_ct.clone().into_tensor_arg(),
            k_ct.clone().into_tensor_arg(),
            v_ct.clone().into_tensor_arg(),
            acc_out.clone().into_tensor_arg(),
            m_out.clone().into_tensor_arg(),
            l_out.clone().into_tensor_arg(),
            scale.into(),
            n_rep.into(),
            (split_len as u32).into(),
            d,
        ),
        DType::F32 => gpu::flash_decode_split::launch::<f32, R>(
            &client,
            CubeCount::Static(hq as u32, bsz as u32, n_splits as u32),
            CubeDim { x: 32, y: 1, z: 1 },
            q_ct.clone().into_tensor_arg(),
            k_ct.clone().into_tensor_arg(),
            v_ct.clone().into_tensor_arg(),
            acc_out.clone().into_tensor_arg(),
            m_out.clone().into_tensor_arg(),
            l_out.clone().into_tensor_arg(),
            scale.into(),
            n_rep.into(),
            (split_len as u32).into(),
            d,
        ),
        dt => panic!("launch_flash_decode: unsupported k/v dtype {dt:?} (expected bf16 or f32)"),
    }

    gpu::flash_decode_combine::launch::<R>(
        &client,
        CubeCount::Static(hq as u32, bsz as u32, 1),
        CubeDim { x: 32, y: 1, z: 1 },
        acc_out.into_tensor_arg(),
        m_out.into_tensor_arg(),
        l_out.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        d,
        n_splits,
    );

    out
}

macro_rules! try_backend_flash_decode {
    ($B:ty, $q:expr, $k:expr, $v:expr, $scale:expr, $n_splits:expr) => {{
        if let (Ok(q), Ok(k), Ok(v)) = (
            $q.clone().try_into_primitive::<$B>(),
            $k.clone().try_into_primitive::<$B>(),
            $v.clone().try_into_primitive::<$B>(),
        ) {
            let out = launch_flash_decode(q, k, v, $scale, $n_splits);
            return Tensor::from_primitive::<$B>(out);
        }
    }};
}

/// Split-K flash-decode over standard `burn::tensor::Tensor<4>` handles, dispatched to whichever
/// GPU backend is compiled in (`cuda`, `metal`, `vulkan`, `wgpu`). `q:[B,Hq,1,D]`,
/// `k,v:[B,Hkv,Sk,D]` (GQA not pre-expanded). Panics if no matching backend feature is enabled or
/// primitive conversion fails (should not happen: called only under `cfg(feature = "cubecl-gpu")`
/// with tensors that live on a `CubeBackend`).
pub fn flash_decode(
    q: Tensor<4>,
    k: Tensor<4>,
    v: Tensor<4>,
    scale: f32,
    n_splits: usize,
) -> Tensor<4> {
    assert_flash_decode_shapes("flash_decode", &q, &k, &v, n_splits);

    #[cfg(all(feature = "metal", not(feature = "metal-fusion-diag")))]
    try_backend_flash_decode!(burn::backend::Metal, q, k, v, scale, n_splits);
    #[cfg(feature = "vulkan")]
    try_backend_flash_decode!(burn::backend::Vulkan, q, k, v, scale, n_splits);
    #[cfg(feature = "wgpu")]
    try_backend_flash_decode!(burn::backend::Wgpu, q, k, v, scale, n_splits);
    #[cfg(feature = "cuda")]
    {
        type CudaRaw = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        try_backend_flash_decode!(CudaRaw, q, k, v, scale, n_splits);
        try_backend_flash_decode!(burn::backend::Cuda, q, k, v, scale, n_splits);
    }

    panic!("flash_decode: backend not supported or primitive conversion failed");
}
