//! L1.4 Qwen3.5 shared-expert MoE check.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_moe_check
//!   ./target/release/examples/qwen35_moe_check

use burn::{
    module::{Ignored, Param, ParamId},
    nn::Linear,
    prelude::Device,
    tensor::{Tensor, TensorData},
};
use qwen3_burn::{
    Precision,
    qwen3_5::{
        ExpertNvfp4Sidecar, ExpertQuantSidecar, QuantSidecar, Qwen3_5FusedExperts,
        Qwen3_5SharedExpert, Qwen3_5SharedMoeBlock,
    },
};

type B = Cuda;

const BATCH: usize = 1;
const SEQ: usize = 4;
const H: usize = 2048;
const E: usize = 256;
const K: usize = 8;
const I: usize = 512;

const SEED_INPUT: u64 = 0x1000_0000_0000_0001;
const SEED_ROUTER: u64 = 0x2000_0000_0000_0002;
const SEED_GATE_UP: u64 = 0x3000_0000_0000_0003;
const SEED_DOWN: u64 = 0x4000_0000_0000_0004;
const SEED_SHARED_GATE: u64 = 0x5000_0000_0000_0005;
const SEED_SHARED_UP: u64 = 0x6000_0000_0000_0006;
const SEED_SHARED_DOWN: u64 = 0x7000_0000_0000_0007;
const SEED_SHARED_GATE_OUT: u64 = 0x8000_0000_0000_0008;

const INPUT_SCALE: f32 = 0.2;
const ROUTER_SCALE: f32 = 0.002;
const EXPERT_SCALE: f32 = 0.0015;
const SHARED_GATE_SCALE: f32 = 0.001;

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn value(seed: u64, idx: usize, scale: f32) -> f32 {
    let bits = splitmix64(seed ^ idx as u64);
    let u = ((bits >> 40) as u32) as f32 / 16_777_216.0;
    (u * 2.0 - 1.0) * scale
}

fn tensor_data(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    for idx in 0..len {
        out.push(value(seed, idx, scale));
    }
    out
}

fn linear_from(device: &CudaDevice, seed: u64, din: usize, dout: usize, scale: f32) -> Linear {
    Linear {
        weight: Param::initialized(
            ParamId::new(),
            Tensor::<2>::from_data(
                TensorData::new(tensor_data(seed, din * dout, scale), [din, dout]),
                device,
            ),
        ),
        bias: None,
    }
}

fn block(device: &CudaDevice) -> Qwen3_5SharedMoeBlock {
    let gate_up = tensor_data(SEED_GATE_UP, E * 2 * I * H, EXPERT_SCALE);
    let down = tensor_data(SEED_DOWN, E * H * I, EXPERT_SCALE);
    Qwen3_5SharedMoeBlock {
        gate: linear_from(device, SEED_ROUTER, H, E, ROUTER_SCALE),
        experts: Qwen3_5FusedExperts {
            gate_up_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(gate_up, [E, 2 * I, H]), device),
            ),
            down_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(down, [E, H, I]), device),
            ),
            fp8: ExpertQuantSidecar(None),
            nvfp4: ExpertNvfp4Sidecar(None),
        },
        shared_expert: Qwen3_5SharedExpert {
            gate_proj: linear_from(device, SEED_SHARED_GATE, H, I, EXPERT_SCALE),
            gate_proj_fp8: QuantSidecar(None),
            up_proj: linear_from(device, SEED_SHARED_UP, H, I, EXPERT_SCALE),
            up_proj_fp8: QuantSidecar(None),
            down_proj: linear_from(device, SEED_SHARED_DOWN, I, H, EXPERT_SCALE),
            down_proj_fp8: QuantSidecar(None),
        },
        shared_expert_gate: linear_from(device, SEED_SHARED_GATE_OUT, H, 1, SHARED_GATE_SCALE),
        num_experts_per_tok: (K),
        norm_topk_prob: (true),
    }
}

fn router_w(h: usize, e: usize) -> f32 {
    value(SEED_ROUTER, h * E + e, ROUTER_SCALE)
}

fn gate_up_w(e: usize, row: usize, h: usize) -> f32 {
    value(SEED_GATE_UP, (e * 2 * I + row) * H + h, EXPERT_SCALE)
}

fn down_w(e: usize, h: usize, i: usize) -> f32 {
    value(SEED_DOWN, (e * H + h) * I + i, EXPERT_SCALE)
}

fn shared_w(seed: u64, row: usize, col: usize, out: usize, scale: f32) -> f32 {
    value(seed, row * out + col, scale)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for x in &mut exps {
        *x /= sum;
    }
    exps
}

fn topk_norm(probs: &[f32]) -> Vec<(usize, f32)> {
    let mut pairs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    pairs.truncate(K);
    let sum: f32 = pairs.iter().map(|(_, w)| *w).sum();
    for (_, w) in &mut pairs {
        *w /= sum.max(1e-20);
    }
    pairs
}

fn expert_ref(e: usize, x: &[f32]) -> Vec<f32> {
    let mut gate = vec![0.0f32; I];
    let mut up = vec![0.0f32; I];
    for i in 0..I {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for h in 0..H {
            g += x[h] * gate_up_w(e, i, h);
            u += x[h] * gate_up_w(e, I + i, h);
        }
        gate[i] = silu(g);
        up[i] = u;
    }

    let mut y = vec![0.0f32; H];
    for h in 0..H {
        let mut acc = 0.0f32;
        for i in 0..I {
            acc += gate[i] * up[i] * down_w(e, h, i);
        }
        y[h] = acc;
    }
    y
}

fn shared_ref(x: &[f32]) -> Vec<f32> {
    let mut gate = vec![0.0f32; I];
    let mut up = vec![0.0f32; I];
    for i in 0..I {
        let mut g = 0.0f32;
        let mut u = 0.0f32;
        for h in 0..H {
            g += x[h] * shared_w(SEED_SHARED_GATE, h, i, I, EXPERT_SCALE);
            u += x[h] * shared_w(SEED_SHARED_UP, h, i, I, EXPERT_SCALE);
        }
        gate[i] = silu(g);
        up[i] = u;
    }

    let mut y = vec![0.0f32; H];
    for h in 0..H {
        let mut acc = 0.0f32;
        for i in 0..I {
            acc += gate[i] * up[i] * shared_w(SEED_SHARED_DOWN, i, h, H, EXPERT_SCALE);
        }
        y[h] = acc;
    }

    let mut gate_out = 0.0f32;
    for h in 0..H {
        gate_out += x[h] * shared_w(SEED_SHARED_GATE_OUT, h, 0, 1, SHARED_GATE_SCALE);
    }
    let gate_out = sigmoid(gate_out);
    for v in &mut y {
        *v *= gate_out;
    }
    y
}

fn cpu_ref(input: &[f32]) -> Vec<f32> {
    let tokens = BATCH * SEQ;
    let mut out = vec![0.0f32; tokens * H];
    for t in 0..tokens {
        let x = &input[t * H..(t + 1) * H];
        let mut logits = vec![0.0f32; E];
        for e in 0..E {
            let mut acc = 0.0f32;
            for h in 0..H {
                acc += x[h] * router_w(h, e);
            }
            logits[e] = acc;
        }

        let probs = softmax(&logits);
        for (e, w) in topk_norm(&probs) {
            let y = expert_ref(e, x);
            for h in 0..H {
                out[t * H + h] += y[h] * w;
            }
        }

        let shared = shared_ref(x);
        for h in 0..H {
            out[t * H + h] += shared[h];
        }
    }
    out
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

fn main() {
    let device = Device::cuda(0);
    let input = tensor_data(SEED_INPUT, BATCH * SEQ * H, INPUT_SCALE);
    let moe = block(&device);
    let x = Tensor::<3>::from_data(TensorData::new(input.clone(), [BATCH, SEQ, H]), &device);

    let burn_out = moe
        .forward(x, Precision::F32)
        .into_data()
        .to_vec::<f32>()
        .expect("read Burn MoE output");
    let ref_out = cpu_ref(&input);
    let cos = cosine(&ref_out, &burn_out);
    let mad = max_abs_diff(&ref_out, &burn_out);

    println!("Qwen3.5 MoE check: cosine={cos:.8} max_abs_diff={mad:.6e}");
    assert!(cos > 0.999, "cosine {cos:.8} <= 0.999");
    assert!(mad < 2e-3, "max_abs_diff {mad:.6e} too large");
    println!("L1.4 MOE CHECK: PASS");
}
