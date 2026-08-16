//! CubeCL NVFP4 kernels shared by CUDA and wgpu/Metal/Vulkan.
//!
//! Native `e4m3` buffer types are avoided so the same kernels lower through MSL/SPIR-V/WGSL.

use burn::tensor::{DType, Int, Tensor};
use burn_cubecl::CubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{CubeCount, CubeDim};

use crate::cubecl_rt::alloc_f32;

const NVFP4_COLS_PER_CTA: u32 = 4;

pub mod gpu {
    use cubecl::prelude::*;

    #[cube]
    pub fn e2m1_decode(code: u32) -> f32 {
        let mag = code & 7u32;
        let m = f32::cast_from(mag & 1u32);
        let e = (mag >> 1u32) & 3u32;
        let val_mag = if e == 0u32 {
            f32::new(0.5f32) * m
        } else {
            let pe = if e == 1u32 {
                f32::new(1.0f32)
            } else if e == 2u32 {
                f32::new(2.0f32)
            } else {
                f32::new(4.0f32)
            };
            pe * (f32::new(1.0f32) + f32::new(0.5f32) * m)
        };
        if (code >> 3u32) & 1u32 == 1u32 {
            -val_mag
        } else {
            val_mag
        }
    }

    #[cube]
    pub fn e4m3_decode(byte: u32) -> f32 {
        let exp = (byte >> 3u32) & 15u32;
        let mant = byte & 7u32;
        let mag = if exp == 0u32 {
            if mant == 0u32 {
                f32::new(0.0f32)
            } else {
                f32::cast_from(mant) * f32::new(0.001953125f32)
            }
        } else if exp == 15u32 && mant == 7u32 {
            f32::new(0.0f32)
        } else {
            let pe = if exp == 1u32 {
                f32::new(0.015625f32)
            } else if exp == 2u32 {
                f32::new(0.03125f32)
            } else if exp == 3u32 {
                f32::new(0.0625f32)
            } else if exp == 4u32 {
                f32::new(0.125f32)
            } else if exp == 5u32 {
                f32::new(0.25f32)
            } else if exp == 6u32 {
                f32::new(0.5f32)
            } else if exp == 7u32 {
                f32::new(1.0f32)
            } else if exp == 8u32 {
                f32::new(2.0f32)
            } else if exp == 9u32 {
                f32::new(4.0f32)
            } else if exp == 10u32 {
                f32::new(8.0f32)
            } else if exp == 11u32 {
                f32::new(16.0f32)
            } else if exp == 12u32 {
                f32::new(32.0f32)
            } else if exp == 13u32 {
                f32::new(64.0f32)
            } else {
                f32::new(128.0f32)
            };
            pe * (f32::new(1.0f32) + f32::cast_from(mant) * f32::new(0.125f32))
        };
        if (byte & 128u32) == 128u32 { -mag } else { mag }
    }

    #[cube]
    pub fn nvfp4_dequant_nibble(packed: u32, high: bool, block_scale: u32, gscale: f32) -> f32 {
        let code = if high {
            (packed >> 4u32) & 15u32
        } else {
            packed & 15u32
        };
        e2m1_decode(code) * e4m3_decode(block_scale) * gscale
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn nvfp4_decode_gemv(
        x: &Tensor<f32>,
        qw: &Tensor<u8>,
        bs: &Tensor<u8>,
        gscale: &Tensor<f32>,
        out: &mut Tensor<f32>,
        m_dim: u32,
        #[comptime] k: usize,
        #[comptime] blocks: usize,
        #[comptime] m_max: usize,
    ) {
        let col = (CUBE_POS_X * CUBE_DIM_Y + UNIT_POS_Y) as usize;
        let lane = UNIT_POS_X;
        let n = out.shape(1);

        if col < n {
            let g = gscale[0];
            let bytes_per_col = blocks * 8usize;
            let mut acc = Array::<f32>::new(m_max);
            let mut val = Array::<f32>::new(16usize);

            #[unroll]
            for m in 0..m_max {
                acc[m] = f32::new(0.0f32);
            }

            let mut blk = lane as usize;
            while blk < blocks {
                let byte_base = col * bytes_per_col + blk * 8usize;
                #[unroll]
                for j in 0..8usize {
                    let byte = u32::cast_from(qw[byte_base + j]);
                    val[j * 2usize] = e2m1_decode(byte & 15u32);
                    val[j * 2usize + 1usize] = e2m1_decode((byte >> 4u32) & 15u32);
                }
                let s = e4m3_decode(u32::cast_from(bs[col * blocks + blk]));
                let k0 = blk * 16usize;

                #[unroll]
                for m in 0..m_max {
                    let mut part = f32::new(0.0f32);
                    #[unroll]
                    for p in 0..16usize {
                        if (m as u32) < m_dim {
                            part += val[p] * x[m * k + k0 + p];
                        }
                    }
                    acc[m] += s * part;
                }

                blk += 32;
            }

            #[unroll]
            for m in 0..m_max {
                let full = plane_sum(acc[m]);
                if lane == 0 && (m as u32) < m_dim {
                    out[m * n + col] = g * full;
                }
            }
        }
    }

    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn fused35_gu_nvfp4_scalar(
        x: &Tensor<f32>,
        q_gu: &Tensor<u8>,
        bs_gu: &Tensor<u8>,
        gscale_gu: &Tensor<f32>,
        assign_e: &Tensor<i32>,
        gu: &mut Tensor<f32>,
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

            let mut gacc = f32::new(0.0f32);
            let mut uacc = f32::new(0.0f32);
            for hh in 0..h_dim_u {
                let h_i = i64::cast_from(hh);
                let block_i = i64::cast_from(hh / 16usize);
                let xv = x[usize::cast_from(x_base + h_i * xs1)];
                let gb = u32::cast_from(q_gu[usize::cast_from(g_q_base + h_i * qs1)]);
                let ub = u32::cast_from(q_gu[usize::cast_from(u_q_base + h_i * qs1)]);
                let gs = u32::cast_from(bs_gu[usize::cast_from(g_bs_base + block_i * bs2)]);
                let us = u32::cast_from(bs_gu[usize::cast_from(u_bs_base + block_i * bs2)]);
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
        gu: &Tensor<f32>,
        q_dn: &Tensor<u8>,
        bs_dn: &Tensor<u8>,
        gscale_dn: &Tensor<f32>,
        assign_e: &Tensor<i32>,
        sel_w: &Tensor<f32>,
        out: &mut Tensor<f32>,
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

            let mut acc = f32::new(0.0f32);
            for ci in 0..i_dim_u {
                let ci_i = i64::cast_from(ci);
                let block_i = i64::cast_from(ci / 16usize);
                let qb = u32::cast_from(q_dn[usize::cast_from(q_base + ci_i * qs1)]);
                let sb = u32::cast_from(bs_dn[usize::cast_from(bs_base + block_i * bs2)]);
                acc += gu[usize::cast_from(gu_base + ci_i * gus1)]
                    * nvfp4_dequant_nibble(qb, high, sb, gd);
            }
            out[pos] = acc * w;
        }
    }
}

fn assert_nvfp4_gemv_shapes(
    m: usize,
    k: usize,
    n: usize,
    packed_len: usize,
    scale_len: usize,
    m_max: usize,
) {
    assert!(m > 0, "nvfp4_gemv: M must be non-zero");
    assert!(n > 0, "nvfp4_gemv: N must be non-zero");
    assert_eq!(
        k % 16,
        0,
        "nvfp4_gemv requires K to be a multiple of 16, got {k}"
    );
    assert!(
        (1..=8).contains(&m_max),
        "nvfp4_gemv: m_max must be in 1..=8 for decode, got {m_max}"
    );
    assert!(
        m <= m_max,
        "nvfp4_gemv: runtime M ({m}) exceeds fixed decode m_max ({m_max})"
    );
    assert_eq!(
        packed_len,
        n * (k / 2),
        "packed_qw length {packed_len} != N*(K/2) = {}",
        n * (k / 2)
    );
    assert_eq!(
        scale_len,
        n * (k / 16),
        "block_scales length {scale_len} != N*(K/16) = {}",
        n * (k / 16)
    );
}

fn launch_nvfp4_gemv<R: CubeRuntime>(
    x: CubeTensor<R>,
    qw: CubeTensor<R>,
    bs: CubeTensor<R>,
    gscale: CubeTensor<R>,
    k: usize,
    n: usize,
    m: usize,
    m_max: usize,
) -> CubeTensor<R> {
    let x = into_contiguous(x);
    let qw = into_contiguous(qw);
    let bs = into_contiguous(bs);
    let gscale = into_contiguous(gscale);
    let blocks = k / 16;
    let out = alloc_f32(&x, &[m, n]);

    gpu::nvfp4_decode_gemv::launch::<R>(
        &x.client,
        CubeCount::Static((n as u32).div_ceil(NVFP4_COLS_PER_CTA), 1, 1),
        CubeDim {
            x: 32,
            y: NVFP4_COLS_PER_CTA,
            z: 1,
        },
        x.clone().into_tensor_arg(),
        qw.into_tensor_arg(),
        bs.into_tensor_arg(),
        gscale.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        m as u32,
        k,
        blocks,
        m_max,
    );
    out
}

fn launch_fused_nvfp4<R: CubeRuntime>(
    x: CubeTensor<R>,
    q_gu: CubeTensor<R>,
    bs_gu: CubeTensor<R>,
    gscale_gu: CubeTensor<R>,
    q_dn: CubeTensor<R>,
    bs_dn: CubeTensor<R>,
    gscale_dn: CubeTensor<R>,
    assign_e: CubeTensor<R>,
    sel_w: CubeTensor<R>,
    h: usize,
    i: usize,
    n: usize,
) -> CubeTensor<R> {
    let x = into_contiguous(x);
    let q_gu = into_contiguous(q_gu);
    let bs_gu = into_contiguous(bs_gu);
    let gscale_gu = into_contiguous(gscale_gu);
    let q_dn = into_contiguous(q_dn);
    let bs_dn = into_contiguous(bs_dn);
    let gscale_dn = into_contiguous(gscale_dn);
    let ae = into_contiguous(assign_e);
    let sw = into_contiguous(sel_w);
    let gu = alloc_f32(&x, &[n, i]);
    let out = alloc_f32(&x, &[n, h]);

    let [t, x_h] = x.meta.shape().dims::<2>();
    assert_eq!(x_h, h, "fused nvfp4: x H != h");
    assert!(
        (1..=16).contains(&t),
        "fused nvfp4 decode path requires 1 <= T <= 16, got {t}"
    );
    assert_eq!(n % t, 0, "fused nvfp4: N must be divisible by T");
    let top_k = (n / t) as u32;
    let threads = 256u32;
    let cdim = CubeDim {
        x: threads,
        y: 1,
        z: 1,
    };

    gpu::fused35_gu_nvfp4_scalar::launch::<R>(
        &x.client,
        CubeCount::Static(((n * i) as u32).div_ceil(threads), 1, 1),
        cdim,
        x.clone().into_tensor_arg(),
        q_gu.into_tensor_arg(),
        bs_gu.into_tensor_arg(),
        gscale_gu.into_tensor_arg(),
        ae.clone().into_tensor_arg(),
        gu.clone().into_tensor_arg(),
        h as u32,
        i as u32,
        top_k,
    );
    gpu::fused35_down_nvfp4_scalar::launch::<R>(
        &x.client,
        CubeCount::Static(((n * h) as u32).div_ceil(threads), 1, 1),
        cdim,
        gu.into_tensor_arg(),
        q_dn.into_tensor_arg(),
        bs_dn.into_tensor_arg(),
        gscale_dn.into_tensor_arg(),
        ae.into_tensor_arg(),
        sw.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        h as u32,
        i as u32,
    );
    out
}

macro_rules! try_backend_gemv {
    ($B:ty, $x:expr, $qw:expr, $bs:expr, $gscale:expr, $k:expr, $n:expr, $m:expr, $m_max:expr) => {{
        if let (Ok(x), Ok(qw), Ok(bs), Ok(gscale)) = (
            $x.clone().try_into_primitive::<$B>(),
            $qw.clone().try_into_primitive::<$B>(),
            $bs.clone().try_into_primitive::<$B>(),
            $gscale.clone().try_into_primitive::<$B>(),
        ) {
            let out = launch_nvfp4_gemv(x, qw, bs, gscale, $k, $n, $m, $m_max);
            return Tensor::from_primitive::<$B>(out);
        }
    }};
}

macro_rules! try_backend_fused {
    ($B:ty, $x:expr, $q_gu:expr, $bs_gu:expr, $gscale_gu:expr, $q_dn:expr, $bs_dn:expr, $gscale_dn:expr, $ae:expr, $sw:expr, $h:expr, $i:expr, $n:expr) => {{
        if let (
            Ok(x),
            Ok(q_gu),
            Ok(bs_gu),
            Ok(gscale_gu),
            Ok(q_dn),
            Ok(bs_dn),
            Ok(gscale_dn),
            Ok(ae),
            Ok(sw),
        ) = (
            $x.clone().try_into_primitive::<$B>(),
            $q_gu.clone().try_into_primitive::<$B>(),
            $bs_gu.clone().try_into_primitive::<$B>(),
            $gscale_gu.clone().try_into_primitive::<$B>(),
            $q_dn.clone().try_into_primitive::<$B>(),
            $bs_dn.clone().try_into_primitive::<$B>(),
            $gscale_dn.clone().try_into_primitive::<$B>(),
            $ae.clone().try_into_primitive::<$B>(),
            $sw.clone().try_into_primitive::<$B>(),
        ) {
            let out = launch_fused_nvfp4(
                x, q_gu, bs_gu, gscale_gu, q_dn, bs_dn, gscale_dn, ae, sw, $h, $i, $n,
            );
            return Tensor::from_primitive::<$B>(out);
        }
    }};
}

/// NVFP4 weight-only GEMV: `x:[M,K] f32` × packed `qw:[N,K/2]`, `bs:[N,K/16]` → `y:[M,N] f32`.
pub fn nvfp4_gemv(
    x: Tensor<2>,
    qw: Tensor<2, Int>,
    bs: Tensor<2, Int>,
    gscale: Tensor<1>,
    k: usize,
    n: usize,
    m_max: usize,
) -> Tensor<2> {
    let [m, xk] = x.dims();
    assert_eq!(xk, k, "nvfp4_gemv: x K ({xk}) != requested K ({k})");
    assert_eq!(x.dtype(), DType::F32, "nvfp4_gemv: x must be f32");
    assert_eq!(qw.dtype(), DType::I8, "nvfp4_gemv: qw must be I8");
    assert_eq!(bs.dtype(), DType::I8, "nvfp4_gemv: bs must be I8");
    assert_eq!(gscale.dtype(), DType::F32, "nvfp4_gemv: gscale must be f32");
    assert_eq!(qw.dims(), [n, k / 2], "nvfp4_gemv: qw must be [N,K/2]");
    assert_eq!(bs.dims(), [n, k / 16], "nvfp4_gemv: bs must be [N,K/16]");
    assert_eq!(gscale.dims(), [1], "nvfp4_gemv: gscale must be [1]");
    assert_nvfp4_gemv_shapes(m, k, n, n * (k / 2), n * (k / 16), m_max);

    #[cfg(feature = "metal")]
    try_backend_gemv!(burn::backend::Metal, x, qw, bs, gscale, k, n, m, m_max);
    #[cfg(feature = "vulkan")]
    try_backend_gemv!(burn::backend::Vulkan, x, qw, bs, gscale, k, n, m, m_max);
    #[cfg(feature = "wgpu")]
    try_backend_gemv!(burn::backend::Wgpu, x, qw, bs, gscale, k, n, m, m_max);
    #[cfg(feature = "cuda")]
    {
        type CudaRaw = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        try_backend_gemv!(CudaRaw, x, qw, bs, gscale, k, n, m, m_max);
        if let (Ok(x_p), Ok(qw_p), Ok(bs_p), Ok(gs_p)) = (
            x.clone().try_into_primitive::<burn::backend::Cuda>(),
            qw.clone().try_into_primitive::<burn::backend::Cuda>(),
            bs.clone().try_into_primitive::<burn::backend::Cuda>(),
            gscale.clone().try_into_primitive::<burn::backend::Cuda>(),
        ) {
            let outputs =
                crate::cube_custom_op::CubeCustomOp::<cubecl::cuda::CudaRuntime>::new("nvfp4_gemv")
                    .float_input(x_p)
                    .int_input(qw_p)
                    .int_input(bs_p)
                    .float_input(gs_p)
                    .float_output([m, n], DType::F32)
                    .launch(move |inputs| {
                        vec![launch_nvfp4_gemv(
                            inputs[0].clone(),
                            inputs[1].clone(),
                            inputs[2].clone(),
                            inputs[3].clone(),
                            k,
                            n,
                            m,
                            m_max,
                        )]
                    });
            return Tensor::from_primitive::<burn::backend::Cuda>(
                outputs.into_iter().next().expect("one output"),
            );
        }
    }

    panic!(
        "nvfp4_gemv: tensor is not on a CubeCL GPU backend (metal/wgpu/vulkan/cuda); device={:?}",
        x.device()
    )
}

/// Fused NVFP4 MoE gate/up SwiGLU + down GEMV for `N = T*top_k` assignments, `T <= 16`.
#[allow(clippy::too_many_arguments)]
pub fn fused_moe_gu2_down_nvfp4(
    x: Tensor<2>,
    q_gu: Tensor<3, Int>,
    bs_gu: Tensor<3, Int>,
    gscale_gu: Tensor<2>,
    q_dn: Tensor<3, Int>,
    bs_dn: Tensor<3, Int>,
    gscale_dn: Tensor<1>,
    assign_e: Tensor<1, Int>,
    sel_w: Tensor<1>,
    h: usize,
    i: usize,
    n: usize,
) -> Tensor<2> {
    let assign_e = assign_e.cast(DType::I32);
    let x = x.cast(DType::F32);
    let sel_w = sel_w.cast(DType::F32);

    #[cfg(feature = "metal")]
    try_backend_fused!(
        burn::backend::Metal,
        x,
        q_gu,
        bs_gu,
        gscale_gu,
        q_dn,
        bs_dn,
        gscale_dn,
        assign_e,
        sel_w,
        h,
        i,
        n
    );
    #[cfg(feature = "vulkan")]
    try_backend_fused!(
        burn::backend::Vulkan,
        x,
        q_gu,
        bs_gu,
        gscale_gu,
        q_dn,
        bs_dn,
        gscale_dn,
        assign_e,
        sel_w,
        h,
        i,
        n
    );
    #[cfg(feature = "wgpu")]
    try_backend_fused!(
        burn::backend::Wgpu,
        x,
        q_gu,
        bs_gu,
        gscale_gu,
        q_dn,
        bs_dn,
        gscale_dn,
        assign_e,
        sel_w,
        h,
        i,
        n
    );
    #[cfg(feature = "cuda")]
    {
        type CudaRaw = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        try_backend_fused!(
            CudaRaw, x, q_gu, bs_gu, gscale_gu, q_dn, bs_dn, gscale_dn, assign_e, sel_w, h, i, n
        );
    }

    panic!(
        "fused_moe_gu2_down_nvfp4: tensor is not on a CubeCL GPU backend; device={:?}",
        x.device()
    )
}
