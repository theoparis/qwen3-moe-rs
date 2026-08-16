//! Fused CubeCL kernels for the GDN single-token decode inner loop.
//!
//! # The problem
//! Burn generates ~45 separate GPU dispatches per GDN layer for:
//! conv + SiLU + L2-norm + delta-state update + readout + RMSNorm + z-gate.
//! On M2 Pro: ~130ms CPU command-build overhead × 30 GDN layers = **3.97s/token**.
//!
//! # Solution: 2 dispatches per GDN layer
//!
//! ## Kernel A — `gdn_conv_hist`
//! One thread per `(channel)`. Computes FIR conv + SiLU, shifts history.
//! Runtime scalars: qkv_dim, kernel_m1. Grid: 2D (ceil(QD/256), B).
//!
//! ## Kernel B — `gdn_state_gate`
//! One threadgroup per (batch, v_head). Block width = `VD` (comptime).
//! Reads conv+silu output, L2-normalizes q/k via plane_sum+Shared memory slice,
//! delta-state update + readout + RMSNorm + z-gate.
//! Comptime: VD (value_head_dim), KD (key_head_dim), nwarp (VD/32).

use burn_cubecl::CubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use cubecl::{CubeCount, CubeDim};

use crate::cubecl_rt::alloc_f32;

// ─── GdnShape ─────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
pub struct GdnShape {
    pub batch: usize,
    /// qkv_dim = 2*num_key_heads*key_head_dim + num_value_heads*value_head_dim
    pub qkv_dim: usize,
    pub num_value_heads: usize,
    pub num_key_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub kernel_size: usize,
    pub epsilon: f32,
}
impl GdnShape {
    pub fn q_dim(&self) -> usize {
        self.num_key_heads * self.key_head_dim
    }
    pub fn v_dim_total(&self) -> usize {
        self.num_value_heads * self.value_head_dim
    }
    pub fn kernel_m1(&self) -> usize {
        self.kernel_size - 1
    }
    pub fn kd_sqrt_inv(&self) -> f32 {
        1.0f32 / (self.key_head_dim as f32).sqrt()
    }
}

// ─── CubeCL kernels ───────────────────────────────────────────────────────

pub mod gpu {
    use cubecl::prelude::*;

    // ── Kernel A: FIR conv + SiLU + history shift ────────────────────────
    // Grid: CubeCount::Static(ceil(qkv_dim/threads), batch, 1).
    // Block: (threads, 1, 1).
    // ABSOLUTE_POS = channel index (usize), CUBE_POS_Y = batch.
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_conv_hist(
        qkv_unconv: &Tensor<f32>,   // [B, QD]
        conv_hist: &Tensor<f32>,    // [B, Km1, QD]
        conv_w: &Tensor<f32>,       // [QD, K]
        qkv_silu: &mut Tensor<f32>, // [B, QD]  output
        new_hist: &mut Tensor<f32>, // [B, Km1, QD]  output
        qkv_dim: u32,
        kernel_m1: u32,
    ) {
        let ch: usize = ABSOLUTE_POS;
        let b: usize = CUBE_POS_Y as usize;
        let qd: usize = qkv_dim as usize;
        let km1: usize = kernel_m1 as usize;

        if ch >= qd {
            terminate!();
        }

        let K: usize = km1 + 1;

        // FIR conv accumulate (history is oldest-first)
        let w_cur = conv_w[ch * K + km1];
        let mut acc = qkv_unconv[b * qd + ch] * w_cur;
        for i in 0..km1 {
            acc += conv_hist[(b * km1 + i) * qd + ch] * conv_w[ch * K + i];
        }
        // SiLU
        let silu_val = acc / (f32::new(1.0f32) + (-acc).exp());
        qkv_silu[b * qd + ch] = silu_val;

        // Shift history: new_hist[b, i, ch] ← old hist[b, i+1, ch] (or current input for last slot)
        for i in 0..km1 {
            let src = if i + 1 < km1 {
                conv_hist[(b * km1 + i + 1) * qd + ch]
            } else {
                qkv_unconv[b * qd + ch]
            };
            new_hist[(b * km1 + i) * qd + ch] = src;
        }
    }

    // ── Kernel B: state update + readout + RMSNorm + z-gate ──────────────
    // Grid: CubeCount::Static(num_value_heads, batch, 1).
    // Block: CubeDim::new_1d(VD).
    // CUBE_POS_X = v_head, CUBE_POS_Y = b, UNIT_POS_X = vd.
    // Comptime: VD (= value_head_dim), KD (= key_head_dim), NWARP (= VD/32).
    #[cube(launch)]
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_state_gate(
        qkv_silu: &Tensor<f32>,      // [B, QD]  from kernel A
        z_in: &Tensor<f32>,          // [B, V, VD]
        a_in: &Tensor<f32>,          // [B, V]
        b_in: &Tensor<f32>,          // [B, V]
        dt_bias: &Tensor<f32>,       // [V]
        a_log: &Tensor<f32>,         // [V]
        norm_w: &Tensor<f32>,        // [VD]
        prev_state: &Tensor<f32>,    // [B, V, KD, VD]
        out: &mut Tensor<f32>,       // [B, V, VD]
        new_state: &mut Tensor<f32>, // [B, V, KD, VD]
        qkv_dim: u32,
        q_dim: u32, // = num_key_heads * KD
        num_value_heads: u32,
        num_key_heads: u32,
        epsilon: f32,
        kd_sqrt_inv: f32,
        #[comptime] kd: usize,    // key_head_dim (= 128 for Qwen3.5-MoE)
        #[comptime] vd: usize,    // value_head_dim (= 128)
        #[comptime] nwarp: usize, // = vd / 32
    ) {
        let v_head: usize = CUBE_POS_X as usize;
        let b: usize = CUBE_POS_Y as usize;
        let d: usize = UNIT_POS_X as usize; // element index within [0, VD)

        let num_vh: usize = num_value_heads as usize;
        let num_kh: usize = num_key_heads as usize;
        let vhpk: usize = num_vh / num_kh;
        let k_head: usize = v_head / vhpk;

        if v_head >= num_vh || d >= vd {
            terminate!();
        }

        let qd: usize = qkv_dim as usize;
        let qd_u: usize = q_dim as usize;

        // ── Shared memory for warp partial reductions ────────────────────
        // Layout: [0..nwarp] = q_sq per warp, [nwarp..2*nwarp] = k_sq, [2*nwarp] = o_sq_sum.
        let mut sm = Shared::<[f32]>::new_slice(2 * nwarp + 1);

        // Warp info
        let warp_id: usize = d / 32;
        let lane: usize = d % 32;

        // ── Channel addresses for this thread ────────────────────────────
        let ch_q: usize = k_head * kd + d;
        let ch_k: usize = qd_u + k_head * kd + d;
        let ch_v: usize = 2 * qd_u + v_head * vd + d;

        let q_raw: f32 = if d < kd {
            qkv_silu[b * qd + ch_q]
        } else {
            f32::new(0.0f32)
        };
        let k_raw: f32 = if d < kd {
            qkv_silu[b * qd + ch_k]
        } else {
            f32::new(0.0f32)
        };
        let v_val: f32 = qkv_silu[b * qd + ch_v];

        // ── L2-norm for q and k (per key-head, plane_sum + shared-mem) ───
        let q_sq_partial: f32 = plane_sum(q_raw * q_raw);
        let k_sq_partial: f32 = plane_sum(k_raw * k_raw);
        if lane == 0 {
            sm[warp_id] = q_sq_partial;
            sm[nwarp + warp_id] = k_sq_partial;
        }
        sync_cube();

        // Sum warp partials on thread 0 and broadcast via shared memory
        if d == 0 {
            let mut qs = f32::new(0.0f32);
            let mut ks = f32::new(0.0f32);
            #[unroll]
            for w in 0..nwarp {
                qs += sm[w];
                ks += sm[nwarp + w];
            }
            sm[0] = qs;
            sm[nwarp] = ks;
        }
        sync_cube();
        let q_sq_total: f32 = sm[0];
        let k_sq_total: f32 = sm[nwarp];

        // ── Per-head gate scalars ────────────────────────────────────────
        let head_idx: usize = b * num_vh + v_head;
        let a_raw: f32 = a_in[head_idx] + dt_bias[v_head];
        let sp: f32 = if a_raw >= f32::new(0.0f32) {
            a_raw + ((-a_raw).exp() + f32::new(1.0f32)).ln()
        } else {
            (a_raw.exp() + f32::new(1.0f32)).ln()
        };
        let alpha: f32 = ((-a_log[v_head].exp()) * sp).exp();
        let beta_gate: f32 = f32::new(1.0f32) / (f32::new(1.0f32) + (-b_in[head_idx]).exp());

        // ── State update + readout ───────────────────────────────────────
        // state[b, v_head, kd2, d] at flat index st_bv + kd2*vd + d
        let st_bv: usize = (b * num_vh + v_head) * kd * vd;

        // Pass 1: state_k = sum_{kd2}( state[kd2, d] * k_norm[kd2] )
        let mut state_k = f32::new(0.0f32);
        #[unroll]
        for kd2 in 0..kd {
            let ch_k2: usize = qd_u + k_head * kd + kd2;
            let k_n2: f32 = qkv_silu[b * qd + ch_k2] / (k_sq_total + f32::new(1e-6f32)).sqrt();
            state_k += prev_state[st_bv + kd2 * vd + d] * k_n2;
        }
        let delta_v: f32 = v_val - alpha * state_k;

        // Pass 2: write new_state, accumulate readout o
        let mut o_acc = f32::new(0.0f32);
        #[unroll]
        for kd2 in 0..kd {
            let ch_k2: usize = qd_u + k_head * kd + kd2;
            let ch_q2: usize = k_head * kd + kd2;
            let k_n2: f32 = qkv_silu[b * qd + ch_k2] / (k_sq_total + f32::new(1e-6f32)).sqrt();
            let q_n2: f32 =
                qkv_silu[b * qd + ch_q2] / (q_sq_total + f32::new(1e-6f32)).sqrt() * kd_sqrt_inv;

            let s_idx: usize = st_bv + kd2 * vd + d;
            let new_s: f32 = alpha * prev_state[s_idx] + beta_gate * k_n2 * delta_v;
            new_state[s_idx] = new_s;
            o_acc += new_s * q_n2;
        }

        // ── RMSNorm (cross-warp reduce of o^2, same pattern) ────────────
        let o_sq_partial: f32 = plane_sum(o_acc * o_acc);
        if lane == 0 {
            sm[warp_id] = o_sq_partial;
        }
        sync_cube();

        if d == 0 {
            let mut os = f32::new(0.0f32);
            #[unroll]
            for w in 0..nwarp {
                os += sm[w];
            }
            sm[2 * nwarp] = os;
        }
        sync_cube();
        let o_sq_total: f32 = sm[2 * nwarp];

        let o_normed: f32 = o_acc / ((o_sq_total / vd as f32) + epsilon).sqrt() * norm_w[d];

        // ── z-gate ───────────────────────────────────────────────────────
        let z_val: f32 = z_in[(b * num_vh + v_head) * vd + d];
        let z_silu: f32 = z_val / (f32::new(1.0f32) + (-z_val).exp());
        out[(b * num_vh + v_head) * vd + d] = o_normed * z_silu;
    }
}

// ─── Host launcher ────────────────────────────────────────────────────────

/// Launch the 2-kernel fused GDN decode step.
///
/// Returns `(out, new_state, new_hist)`:
/// - `out`       `[B, V, VD]`     — feed to `out_proj` (reshape to `[B, 1, V*VD]`)
/// - `new_state` `[B, V, KD, VD]`
/// - `new_hist`  `[B, Km1, QD]`
pub fn launch_gdn_step_fused<R: CubeRuntime>(
    qkv_unconv: CubeTensor<R>,
    z_in: CubeTensor<R>,
    a_in: CubeTensor<R>,
    b_in: CubeTensor<R>,
    conv_hist: CubeTensor<R>,
    conv_w: CubeTensor<R>,
    dt_bias: CubeTensor<R>,
    a_log: CubeTensor<R>,
    norm_w: CubeTensor<R>,
    prev_state: CubeTensor<R>,
    sh: GdnShape,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    use burn_cubecl::kernel::into_contiguous;

    let qkv_unconv = into_contiguous(qkv_unconv);
    let z_in = into_contiguous(z_in);
    let a_in = into_contiguous(a_in);
    let b_in = into_contiguous(b_in);
    let conv_hist = into_contiguous(conv_hist);
    let conv_w = into_contiguous(conv_w);
    let dt_bias = into_contiguous(dt_bias);
    let a_log = into_contiguous(a_log);
    let norm_w = into_contiguous(norm_w);
    let prev_state = into_contiguous(prev_state);

    let client = &qkv_unconv.client;

    let qkv_silu = alloc_f32(&qkv_unconv, &[sh.batch, sh.qkv_dim]);
    let new_hist = alloc_f32(&qkv_unconv, &[sh.batch, sh.kernel_m1(), sh.qkv_dim]);
    let out = alloc_f32(
        &qkv_unconv,
        &[sh.batch, sh.num_value_heads, sh.value_head_dim],
    );
    let new_state = alloc_f32(
        &qkv_unconv,
        &[
            sh.batch,
            sh.num_value_heads,
            sh.key_head_dim,
            sh.value_head_dim,
        ],
    );

    let threads_a: u32 = 256;
    let total_ch = (sh.qkv_dim as u32).div_ceil(threads_a);

    // ── Kernel A: conv + SiLU + history shift ────────────────────────────
    gpu::gdn_conv_hist::launch::<R>(
        client,
        CubeCount::Static(total_ch, sh.batch as u32, 1),
        CubeDim::new_1d(threads_a),
        qkv_unconv.clone().into_tensor_arg(),
        conv_hist.into_tensor_arg(),
        conv_w.into_tensor_arg(),
        qkv_silu.clone().into_tensor_arg(),
        new_hist.clone().into_tensor_arg(),
        (sh.qkv_dim as u32).into(),
        (sh.kernel_m1() as u32).into(),
    );

    // ── Kernel B: state update + readout + RMSNorm + gate ────────────────
    assert!(
        sh.value_head_dim % 32 == 0,
        "gdn_state_gate: value_head_dim ({}) must be multiple of 32 (warp size)",
        sh.value_head_dim
    );
    let nwarp = sh.value_head_dim / 32;

    gpu::gdn_state_gate::launch::<R>(
        client,
        CubeCount::Static(sh.num_value_heads as u32, sh.batch as u32, 1),
        CubeDim::new_1d(sh.value_head_dim as u32),
        qkv_silu.into_tensor_arg(),
        z_in.into_tensor_arg(),
        a_in.into_tensor_arg(),
        b_in.into_tensor_arg(),
        dt_bias.into_tensor_arg(),
        a_log.into_tensor_arg(),
        norm_w.into_tensor_arg(),
        prev_state.into_tensor_arg(),
        out.clone().into_tensor_arg(),
        new_state.clone().into_tensor_arg(),
        (sh.qkv_dim as u32).into(),
        (sh.q_dim() as u32).into(),
        (sh.num_value_heads as u32).into(),
        (sh.num_key_heads as u32).into(),
        sh.epsilon.into(),
        sh.kd_sqrt_inv().into(),
        sh.key_head_dim.into(),   // comptime kd
        sh.value_head_dim.into(), // comptime vd
        nwarp.into(),             // comptime nwarp
    );

    (out, new_state, new_hist)
}

// ─── High-Level Burn Tensor Entry Point ────────────────────────────────────

use burn::tensor::Tensor;

macro_rules! try_backend_gdn {
    ($B:ty, $qkv:expr, $z:expr, $a:expr, $b:expr, $hist:expr, $w:expr, $dt:expr, $alog:expr, $norm:expr, $state:expr, $sh:expr) => {{
        if let (
            Ok(qkv),
            Ok(z),
            Ok(a),
            Ok(b),
            Ok(hist),
            Ok(w),
            Ok(dt),
            Ok(alog),
            Ok(norm),
            Ok(state),
        ) = (
            $qkv.clone().try_into_primitive::<$B>(),
            $z.clone().try_into_primitive::<$B>(),
            $a.clone().try_into_primitive::<$B>(),
            $b.clone().try_into_primitive::<$B>(),
            $hist.clone().try_into_primitive::<$B>(),
            $w.clone().try_into_primitive::<$B>(),
            $dt.clone().try_into_primitive::<$B>(),
            $alog.clone().try_into_primitive::<$B>(),
            $norm.clone().try_into_primitive::<$B>(),
            $state.clone().try_into_primitive::<$B>(),
        ) {
            let (out, new_state, new_hist) =
                launch_gdn_step_fused(qkv, z, a, b, hist, w, dt, alog, norm, state, $sh);
            return (
                Tensor::from_primitive::<$B>(out),
                Tensor::from_primitive::<$B>(new_state),
                Tensor::from_primitive::<$B>(new_hist),
            );
        }
    }};
}

/// Fused GDN decode step over standard `burn::tensor::Tensor` handles.
/// Returns `(out [B, V, VD], new_state [B, V, KD, VD], new_hist [B, Km1, QD])`.
pub fn gdn_step_fused(
    qkv_unconv: Tensor<2>,
    z_in: Tensor<3>,
    a_in: Tensor<2>,
    b_in: Tensor<2>,
    conv_hist: Tensor<3>,
    conv_w: Tensor<2>,
    dt_bias: Tensor<1>,
    a_log: Tensor<1>,
    norm_w: Tensor<1>,
    prev_state: Tensor<4>,
    sh: GdnShape,
) -> (Tensor<3>, Tensor<4>, Tensor<3>) {
    #[cfg(all(feature = "metal", not(feature = "metal-fusion-diag")))]
    try_backend_gdn!(
        burn::backend::Metal,
        qkv_unconv,
        z_in,
        a_in,
        b_in,
        conv_hist,
        conv_w,
        dt_bias,
        a_log,
        norm_w,
        prev_state,
        sh
    );
    #[cfg(feature = "vulkan")]
    try_backend_gdn!(
        burn::backend::Vulkan,
        qkv_unconv,
        z_in,
        a_in,
        b_in,
        conv_hist,
        conv_w,
        dt_bias,
        a_log,
        norm_w,
        prev_state,
        sh
    );
    #[cfg(feature = "wgpu")]
    try_backend_gdn!(
        burn::backend::Wgpu,
        qkv_unconv,
        z_in,
        a_in,
        b_in,
        conv_hist,
        conv_w,
        dt_bias,
        a_log,
        norm_w,
        prev_state,
        sh
    );
    #[cfg(feature = "cuda")]
    {
        type CudaRaw = burn_cubecl::CubeBackend<cubecl::cuda::CudaRuntime>;
        try_backend_gdn!(
            CudaRaw, qkv_unconv, z_in, a_in, b_in, conv_hist, conv_w, dt_bias, a_log, norm_w,
            prev_state, sh
        );
    }

    panic!("gdn_step_fused: backend not supported or primitive conversion failed");
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Tensor;

    #[test]
    fn test_gdn_kernel_shape_math() {
        let sh = GdnShape {
            batch: 1,
            qkv_dim: 8192,
            num_value_heads: 32,
            num_key_heads: 16,
            key_head_dim: 128,
            value_head_dim: 128,
            kernel_size: 4,
            epsilon: 1e-6,
        };
        assert_eq!(sh.q_dim(), 2048);
        assert_eq!(sh.v_dim_total(), 4096);
        assert_eq!(sh.kernel_m1(), 3);
    }
}
