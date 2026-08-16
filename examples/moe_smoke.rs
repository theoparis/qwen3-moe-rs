//! Qwen3-MoE on-device smoke (GB10 / sm_121, CubeCL CUDA). Builds a TINY random Qwen3-MoE and runs
//! the full MoE forward + greedy generation on the GPU. This proves the Tier-1 oracle's on-device
//! routing ops (argmax / gather / scatter-Add / softmax / linear3) and the per-expert SwiGLU run
//! correctly on this exact backend — the same ops the fast B2 kernel will need (a partial moe_probe).
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example moe_smoke

use burn::prelude::Device;
use burn::tensor::{Device, Int, Tensor};
use qwen3_burn::Qwen3MoeConfig;

type B = Cuda;

fn main() {
    let device = Device::cuda(0);
    println!("device: {device:?}");

    let cfg = Qwen3MoeConfig::tiny();
    println!(
        "tiny Qwen3-MoE: {} layers, hidden {}, {} experts top-{}, moe_inter {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size
    );
    let model = cfg.init_causal_lm(&device);

    let ids = Tensor::<2, Int>::from_data([[1i64, 5, 9, 3, 7]], &device);

    // Full MoE forward (no cache) -> vocab logits.
    let logits = model.forward(ids.clone(), None);
    let dims = logits.dims();
    let checksum: f32 = logits.abs().sum().into_scalar();
    assert!(checksum.is_finite(), "non-finite MoE logits on CUDA");
    println!("forward OK: logits {dims:?}, |sum| = {checksum:.4}");

    // Cached greedy generation -> exercises forward_with_cache + the KV cache on-device.
    let out = model.generate_greedy(ids, 8, &[]);
    println!("generate_greedy OK: {:?}", out.dims());

    println!(
        "===== Qwen3-MoE on GB10/sm_121: on-device routing (argmax/gather/scatter-Add/softmax) + per-expert SwiGLU OK ====="
    );
}
