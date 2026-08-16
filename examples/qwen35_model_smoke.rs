//! Lane L1.5a smoke for the Qwen3.6/Qwen3.5-MoE hybrid top-level model forward.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_model_smoke
//!   ./target/release/examples/qwen35_model_smoke

use burn::{
    prelude::Device,
    tensor::{Int, Tensor},
};
use qwen3_burn::{Qwen3_5LayerType, Qwen3_5MoeConfig};

type B = Cuda;

fn smoke_config() -> Qwen3_5MoeConfig {
    let mut layer_types = Vec::with_capacity(40);
    for i in 0..40 {
        layer_types.push(if i % 4 == 3 {
            Qwen3_5LayerType::FullAttention
        } else {
            Qwen3_5LayerType::LinearAttention
        });
    }

    Qwen3_5MoeConfig {
        vocab_size: 248_320,
        hidden_size: 64,
        num_hidden_layers: 40,
        layer_types,
        num_attention_heads: 4,
        num_key_value_heads: 1,
        head_dim: 16,
        partial_rotary_factor: 0.25,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        mrope_section: [11, 11, 10],
        num_experts: 4,
        num_experts_per_tok: 2,
        norm_topk_prob: true,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
        linear_key_head_dim: 16,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_value_head_dim: 16,
        linear_conv_kernel_dim: 4,
        mtp_num_hidden_layers: 1,
    }
}

fn assert_all_finite(tensor: Tensor<3>, what: &str) {
    let values = tensor
        .into_data()
        .to_vec::<f32>()
        .expect("read smoke tensor");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "{what} produced NaN/Inf"
    );
}

fn main() {
    let device = Device::cuda(0);
    let cfg = smoke_config();
    cfg.validate().expect("smoke config must be valid");

    let linear_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == Qwen3_5LayerType::LinearAttention)
        .count();
    let full_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == Qwen3_5LayerType::FullAttention)
        .count();
    assert_eq!((linear_layers, full_layers), (30, 10));

    let model = cfg.init_causal_lm(&device);
    let mut cache = model.model.new_cache();

    let input_ids = Tensor::<2, Int>::from_data(
        [[17i64, 23_001, 491, 88_000, 7, 120_337, 3_101, 248_000]],
        &device,
    );
    let positions = Tensor::<2, Int>::from_data([[0i64, 1, 2, 3, 4, 5, 6, 7]], &device);
    let logits = model.forward(input_ids, positions, &mut cache);
    assert_eq!(logits.dims(), [1, 8, 248_320]);
    assert_all_finite(logits, "prefill logits");

    for step in 0..2 {
        let tok = Tensor::<2, Int>::from_data([[42i64 + step as i64]], &device);
        let pos = Tensor::<2, Int>::from_data([[(8 + step) as i64]], &device);
        let logits = model.forward(tok, pos, &mut cache);
        assert_eq!(logits.dims(), [1, 1, 248_320]);
        assert_all_finite(logits, "decode logits");
    }

    println!("L1.5a MODEL SMOKE: PASS");
}
