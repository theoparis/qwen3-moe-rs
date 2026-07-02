//! fp8_probe — Phase-0 de-risk spike for docs/VLLM_PARITY_PLAN.md.
//!
//! The plan's #1 lever is fp8 weight STORAGE (read half the weight bytes) with bf16 dequant + bf16 MMA
//! — NOT fp8 tensor-core MMA (which sm_121 can't be trusted for, and which needs activation-quant for
//! zero decode gain). So the only question this probe must answer is NUMERICAL: how much error does
//! E4M3 weight storage (per-channel symmetric scale) inject vs full-precision weights, at real Qwen3
//! Linear dims? If cosine ~ 1 and the relative error is small, the bandwidth win is parity-safe.
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example fp8_probe

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{Distribution, Tensor};

type B = Cuda;

const E4M3_MAX: f32 = 448.0; // largest finite E4M3 magnitude
const LN2: f32 = std::f32::consts::LN_2;

/// Faithful E4M3 fake-quant: round to the float8-e4m3 grid (3 mantissa bits, exponent in [-6, 8]).
/// Input is assumed pre-scaled into roughly [-448, 448]. Done in f32 tensor ops (the storage error is
/// what we measure; the real path dequantizes to bf16 before a bf16 MMA).
fn fake_e4m3<const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sign = x.clone().sign();
    let absx = x.abs().clamp(1e-12, E4M3_MAX); // guard log(0); clamp to max
    let e = (absx.clone().log() / LN2).floor().clamp(-6.0, 8.0); // exponent = floor(log2|x|), e4m3 range
    let step = (e * LN2).exp() / 8.0; // 2^e / 8  -> the quantization step for 3 mantissa bits
    let q = (absx / step.clone()).round() * step; // round magnitude to the grid
    (sign * q).clamp(-E4M3_MAX, E4M3_MAX)
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?}");
    println!("E4M3 weight-storage probe: per-channel symmetric scale, round to e4m3, vs full precision\n");

    // Representative Qwen3 Linear shapes [d_in, d_out] (Burn Linear stores [in, out]).
    let shapes: &[(usize, usize, &str)] = &[
        (2048, 768, "MoE-30B expert gate/up [H=2048, I=768]"),
        (768, 2048, "MoE-30B expert down   [I=768, H=2048]"),
        (1024, 3072, "dense-0.6B MLP up     [H=1024, I=3072]"),
        (2048, 2048, "attn o_proj           [H=2048, H=2048]"),
    ];
    let tokens = 64; // a decode/rollout-ish batch

    for &(d_in, d_out, name) in shapes {
        // Weights ~ N(0, 0.02) (typical init/trained scale); activations ~ N(0,1).
        let w = Tensor::<B, 2>::random([d_in, d_out], Distribution::Normal(0.0, 0.02), &device);
        let x = Tensor::<B, 2>::random([tokens, d_in], Distribution::Normal(0.0, 1.0), &device);

        // Per-OUTPUT-channel symmetric scale (amax over the input dim) -> bring each column to ~[-448,448].
        let amax = w.clone().abs().max_dim(0); // [1, d_out]
        let scale = amax.clamp_min(1e-12) / E4M3_MAX; // [1, d_out]
        let w_q = fake_e4m3(w.clone() / scale.clone()) * scale; // quantize then dequantize

        // Weight-storage error itself.
        let werr: f32 = (w.clone() - w_q.clone()).abs().max().into_scalar();
        let wmax: f32 = w.clone().abs().max().into_scalar();

        // Matmul: full-precision oracle vs fp8-stored weight.
        let y = x.clone().matmul(w);
        let y_q = x.matmul(w_q);
        let maxabs: f32 = (y.clone() - y_q.clone()).abs().max().into_scalar();
        let refmax: f32 = y.clone().abs().max().into_scalar();
        let rel = maxabs / refmax.max(1e-9);
        // cosine(y, y_q)
        let dot: f32 = (y.clone() * y_q.clone()).sum().into_scalar();
        let ny: f32 = (y.clone() * y).sum().sqrt().into_scalar();
        let nyq: f32 = (y_q.clone() * y_q).sum().sqrt().into_scalar();
        let cos = dot / (ny * nyq).max(1e-12);

        println!("  {name}");
        println!(
            "    weight: max|w|={wmax:.4} max|w-wq|={werr:.5} ({:.2}% of max)   matmul: cosine={cos:.6} rel-max-err={rel:.4} ({:.2}%)",
            100.0 * werr / wmax.max(1e-9),
            100.0 * rel
        );
    }

    println!("\nVERDICT: cosine > 0.9999 and rel-err of a few % per layer means E4M3 weight-storage is");
    println!("parity-safe for the rollout (the small per-layer error is far below the bf16<->fp8 sampling");
    println!("margin, and sampling stays bf16). The ~2x weight-bandwidth win needs NO fp8 tensor cores.");
}
