//! L1.3 Gated-DeltaNet recurrent decode check.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_gdn_check
//!   ./target/release/examples/qwen35_gdn_check

use burn::{
    module::{Ignored, Param, ParamId},
    nn::{Linear, RmsNorm},
    prelude::Device,
    tensor::{DType, Tensor, TensorData},
};
use qwen3_burn::{
    GdnStateCache, Precision,
    qwen3_5::{
        ExpertNvfp4Sidecar, ExpertQuantSidecar, QuantSidecar, Qwen3_5Conv1d, Qwen3_5FusedExperts,
        Qwen3_5GdnAttention, Qwen3_5GdnLayer, Qwen3_5SharedExpert, Qwen3_5SharedMoeBlock,
    },
};

type B = Cuda;

const H: usize = 2048;
const KH: usize = 16;
const VH: usize = 32;
const D: usize = 128;
const QKV: usize = 8192;
const VDIM: usize = 4096;
const KERNEL: usize = 4;
const STEPS: usize = 8;

#[derive(Clone)]
struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_f32(&mut self, scale: f32) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = ((self.state >> 40) as u32) & 0x00ff_ffff;
        let u = bits as f32 / 16_777_216.0;
        (u * 2.0 - 1.0) * scale
    }

    fn vec(&mut self, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|_| self.next_f32(scale)).collect()
    }
}

struct HostWeights {
    w_qkv: Vec<f32>,
    w_a: Vec<f32>,
    w_b: Vec<f32>,
    w_z: Vec<f32>,
    a_log: Vec<f32>,
    dt_bias: Vec<f32>,
    conv: Vec<f32>,
    norm: Vec<f32>,
    out: Vec<f32>,
}

impl HostWeights {
    fn new() -> Self {
        let mut rng = Rng64::new(0x5eed_1234_cafe_f00d);
        let mut a_log = rng.vec(VH, 0.08);
        for x in &mut a_log {
            *x -= 1.5;
        }
        let mut norm = rng.vec(D, 0.04);
        for x in &mut norm {
            *x += 1.0;
        }
        Self {
            w_qkv: rng.vec(H * QKV, 0.018),
            w_a: rng.vec(H * VH, 0.012),
            w_b: rng.vec(H * VH, 0.012),
            w_z: rng.vec(H * VDIM, 0.018),
            a_log,
            dt_bias: rng.vec(VH, 0.03),
            conv: rng.vec(QKV * KERNEL, 0.55),
            norm,
            out: rng.vec(VDIM * H, 0.012),
        }
    }
}

struct CpuRef<'a> {
    w: &'a HostWeights,
    state: Vec<f32>,
    conv: Vec<f32>,
}

impl<'a> CpuRef<'a> {
    fn new(w: &'a HostWeights) -> Self {
        Self {
            w,
            state: vec![0.0; VH * D * D],
            conv: vec![0.0; (KERNEL - 1) * QKV],
        }
    }

    fn step(&mut self, x: &[f32]) -> Vec<f32> {
        let qkv_unconv = linear(x, &self.w.w_qkv, H, QKV);
        let in_a = linear(x, &self.w.w_a, H, VH);
        let in_b = linear(x, &self.w.w_b, H, VH);
        let z = linear(x, &self.w.w_z, H, VDIM);

        let mut qkv = vec![0.0f32; QKV];
        for c in 0..QKV {
            let mut acc = qkv_unconv[c] * self.w.conv[c * KERNEL + 3];
            for i in 0..(KERNEL - 1) {
                acc += self.conv[i * QKV + c] * self.w.conv[c * KERNEL + i];
            }
            qkv[c] = silu(acc);
        }
        self.conv.copy_within(QKV..(KERNEL - 1) * QKV, 0);
        self.conv[(KERNEL - 2) * QKV..(KERNEL - 1) * QKV].copy_from_slice(&qkv_unconv);

        let mut q = qkv[0..KH * D].to_vec();
        let mut k = qkv[KH * D..2 * KH * D].to_vec();
        let v = &qkv[2 * KH * D..QKV];
        l2_norm_heads(&mut q, KH);
        l2_norm_heads(&mut k, KH);
        for x in &mut q {
            *x *= (D as f32).sqrt().recip();
        }

        let mut o = vec![0.0f32; VDIM];
        for vh in 0..VH {
            let kh = vh / 2;
            let a = (-(self.w.a_log[vh].exp()) * softplus(in_a[vh] + self.w.dt_bias[vh])).exp();
            let b = sigmoid(in_b[vh]);
            let state_base = vh * D * D;
            let k_base = kh * D;
            let v_base = vh * D;

            let mut state_k = [0.0f32; D];
            for kk in 0..D {
                let kval = k[k_base + kk];
                for vv in 0..D {
                    state_k[vv] += self.state[state_base + kk * D + vv] * kval;
                }
            }

            let mut delta = [0.0f32; D];
            for vv in 0..D {
                delta[vv] = v[v_base + vv] - a * state_k[vv];
            }

            for kk in 0..D {
                let kval = k[k_base + kk];
                for vv in 0..D {
                    let idx = state_base + kk * D + vv;
                    self.state[idx] = a * self.state[idx] + b * kval * delta[vv];
                }
            }

            for vv in 0..D {
                let mut acc = 0.0f32;
                for kk in 0..D {
                    acc += self.state[state_base + kk * D + vv] * q[k_base + kk];
                }
                o[v_base + vv] = acc;
            }
        }

        for vh in 0..VH {
            let base = vh * D;
            let mut ss = 0.0f32;
            for d in 0..D {
                ss += o[base + d] * o[base + d];
            }
            let rms = (ss / D as f32 + 1e-6).sqrt();
            for d in 0..D {
                o[base + d] = (o[base + d] / rms) * self.w.norm[d] * silu(z[base + d]);
            }
        }

        linear(&o, &self.w.out, VDIM, H)
    }
}

fn linear(x: &[f32], w: &[f32], k: usize, n: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n];
    for kk in 0..k {
        let xv = x[kk];
        let row = &w[kk * n..(kk + 1) * n];
        for nn in 0..n {
            y[nn] += xv * row[nn];
        }
    }
    y
}

fn l2_norm_heads(x: &mut [f32], heads: usize) {
    for h in 0..heads {
        let base = h * D;
        let mut ss = 0.0f32;
        for d in 0..D {
            ss += x[base + d] * x[base + d];
        }
        let norm = (ss + 1e-6).sqrt();
        for d in 0..D {
            x[base + d] /= norm;
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64) * (x as f64);
        nb += (y as f64) * (y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn rmsnorm(x: &[f32], gamma: &[f32], eps: f32) -> Vec<f32> {
    let mut ss = 0.0f32;
    for &v in x {
        ss += v * v;
    }
    let rms = (ss / x.len() as f32 + eps).sqrt();
    x.iter()
        .zip(gamma.iter())
        .map(|(&v, &g)| (v / rms) * g)
        .collect()
}

fn linear_from(device: &CudaDevice, data: &[f32], din: usize, dout: usize, dtype: DType) -> Linear {
    Linear {
        weight: Param::initialized(
            ParamId::new(),
            Tensor::<2>::from_data(TensorData::new(data.to_vec(), [din, dout]), device).cast(dtype),
        ),
        bias: None,
    }
}

fn rms_norm_from(device: &CudaDevice, data: Vec<f32>, dtype: DType) -> RmsNorm {
    let len = data.len();
    RmsNorm {
        gamma: Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data(TensorData::new(data, [len]), device).cast(dtype),
        ),
        epsilon: 1e-6,
    }
}

fn dummy_mlp(device: &CudaDevice) -> Qwen3_5SharedMoeBlock {
    let z1 = vec![0.0f32];
    Qwen3_5SharedMoeBlock {
        gate: linear_from(device, &z1, 1, 1, DType::F32),
        experts: Qwen3_5FusedExperts {
            gate_up_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(z1.clone(), [1, 1, 1]), device),
            ),
            down_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(z1.clone(), [1, 1, 1]), device),
            ),
            fp8: ExpertQuantSidecar(None),
            nvfp4: ExpertNvfp4Sidecar(None),
        },
        shared_expert: Qwen3_5SharedExpert {
            gate_proj: linear_from(device, &z1, 1, 1, DType::F32),
            gate_proj_fp8: QuantSidecar(None),
            up_proj: linear_from(device, &z1, 1, 1, DType::F32),
            up_proj_fp8: QuantSidecar(None),
            down_proj: linear_from(device, &z1, 1, 1, DType::F32),
            down_proj_fp8: QuantSidecar(None),
        },
        shared_expert_gate: linear_from(device, &z1, 1, 1, DType::F32),
        num_experts_per_tok: (1),
        norm_topk_prob: (true),
    }
}

fn build_layer(
    device: &CudaDevice,
    w: &HostWeights,
    input_norm: &[f32],
    dtype: DType,
) -> Qwen3_5GdnLayer {
    let attn = Qwen3_5GdnAttention::<B> {
        in_proj_qkv: linear_from(device, &w.w_qkv, H, QKV, dtype),
        in_proj_qkv_fp8: QuantSidecar(None),
        in_proj_a: linear_from(device, &w.w_a, H, VH, dtype),
        in_proj_a_fp8: QuantSidecar(None),
        in_proj_b: linear_from(device, &w.w_b, H, VH, dtype),
        in_proj_b_fp8: QuantSidecar(None),
        in_proj_z: linear_from(device, &w.w_z, H, VDIM, dtype),
        in_proj_z_fp8: QuantSidecar(None),
        A_log: Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data(TensorData::new(w.a_log.clone(), [VH]), device).cast(dtype),
        ),
        dt_bias: Param::initialized(
            ParamId::new(),
            Tensor::<1>::from_data(TensorData::new(w.dt_bias.clone(), [VH]), device).cast(dtype),
        ),
        conv1d: Qwen3_5Conv1d {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(w.conv.clone(), [QKV, 1, KERNEL]), device)
                    .cast(dtype),
            ),
        },
        norm: RmsNorm {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::<1>::from_data(TensorData::new(w.norm.clone(), [D]), device).cast(dtype),
            ),
            epsilon: 1e-6,
        },
        out_proj: linear_from(device, &w.out, VDIM, H, dtype),
        out_proj_fp8: QuantSidecar(None),
    };
    Qwen3_5GdnLayer::<B> {
        input_layernorm: rms_norm_from(device, input_norm.to_vec(), dtype),
        linear_attn: attn,
        post_attention_layernorm: rms_norm_from(device, vec![1.0; H], dtype),
        mlp: dummy_mlp(device),
    }
}

fn run_case(
    label: &str,
    device: &CudaDevice,
    w: &HostWeights,
    input_norm: &[f32],
    inputs: &[Vec<f32>],
    dtype: DType,
    prec: Precision,
    min_cos: f32,
    max_mad: Option<f32>,
) {
    let layer = build_layer(device, w, input_norm, dtype);
    let mut cpu = CpuRef::new(&w);
    let mut cache = GdnStateCache::<B>::qwen3_5_default();

    let mut worst_out = 0.0f32;
    for (step, x) in inputs.iter().enumerate() {
        let x_norm = rmsnorm(x, &input_norm, 1e-6);
        let mut cpu_out = cpu.step(&x_norm);
        for i in 0..H {
            cpu_out[i] += x[i];
        }
        let xb = Tensor::<3>::from_data(TensorData::new(x.clone(), [1, 1, H]), device).cast(dtype);
        let burn_out = layer
            .forward_recurrent(xb, &mut cache, prec)
            .cast(DType::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("read Burn GDN output");
        assert!(
            burn_out.iter().all(|value| value.is_finite()),
            "{label} step {step}: output contains a non-finite value"
        );
        let mad = max_abs_diff(&cpu_out, &burn_out);
        let cos = cosine(&cpu_out, &burn_out);
        worst_out = worst_out.max(mad);
        println!("{label} step {step}: output max_abs_diff={mad:.6e} cosine={cos:.8}");
        assert!(
            cos > min_cos,
            "{label} step {step}: output cosine {cos:.8} <= {min_cos}"
        );
        if let Some(limit) = max_mad {
            assert!(
                mad < limit,
                "{label} step {step}: output max_abs_diff {mad:.6e} too large"
            );
        }
    }

    let burn_state = cache
        .state
        .expect("Burn GDN state should be populated")
        .into_data()
        .to_vec::<f32>()
        .expect("read Burn GDN state");
    let state_mad = max_abs_diff(&cpu.state, &burn_state);
    let state_cos = cosine(&cpu.state, &burn_state);
    println!("{label} final state: max_abs_diff={state_mad:.6e} cosine={state_cos:.8}");
    assert!(
        state_cos > min_cos,
        "{label} final state cosine {state_cos:.8} <= {min_cos}"
    );
    if let Some(limit) = max_mad {
        assert!(
            state_mad < limit,
            "{label} final state max_abs_diff {state_mad:.6e} too large"
        );
    }

    println!(
        "L1.3 GDN CHECK {label}: PASS output_max_abs_diff={worst_out:.6e} state_max_abs_diff={state_mad:.6e}"
    );
}

fn main() {
    let device = Device::cuda(0);
    let w = HostWeights::new();
    let input_norm = vec![1.0f32; H];
    let mut rng = Rng64::new(0x1234_5678_9abc_def0);
    let inputs: Vec<Vec<f32>> = (0..STEPS).map(|_| rng.vec(H, 1.0)).collect();

    run_case(
        "F32",
        &device,
        &w,
        &input_norm,
        &inputs,
        DType::F32,
        Precision::F32,
        0.999,
        Some(5e-3),
    );
    run_case(
        "BF16",
        &device,
        &w,
        &input_norm,
        &inputs,
        DType::BF16,
        Precision::Bf16,
        0.99,
        None,
    );
}
