//! Lane L1.2 smoke for the Qwen3.6/Qwen3.5-MoE hybrid full-attention forward.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_fullattn_smoke
//!   ./target/release/examples/qwen35_fullattn_smoke

use burn::{
    prelude::Device,
    tensor::{Distribution, Int, Tensor},
};
use qwen3_burn::{
    Qwen3_5MoeConfig,
    qwen3_5::{Qwen3_5DecoderLayer, Qwen3_5LayerType},
};

type B = Cuda;

fn main() {
    let device = Device::cuda(0);
    let cfg = Qwen3_5MoeConfig::default();

    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.num_key_value_heads, 2);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(
        (cfg.partial_rotary_factor * cfg.head_dim as f64) as usize,
        64
    );

    let model = cfg.init_causal_lm(&device);
    let full_layer_idx = cfg
        .layer_types
        .iter()
        .position(|kind| *kind == Qwen3_5LayerType::FullAttention)
        .expect("default hybrid config must include a full-attention layer");
    let layer = match &model.model.layers[full_layer_idx] {
        Qwen3_5DecoderLayer::Full(layer) => layer,
        Qwen3_5DecoderLayer::Linear(_) => panic!("selected layer is not full-attention"),
    };

    let x = Tensor::<3>::random([1, 8, 2048], Distribution::Normal(0.0, 1.0), &device);
    let positions = Tensor::<2, Int>::from_data([[0i64, 1, 2, 3, 4, 5, 6, 7]], &device);
    let out = layer.forward(x, positions);

    assert_eq!(out.dims(), [1, 8, 2048]);
    let values = out
        .into_data()
        .to_vec::<f32>()
        .expect("read full-attention output");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "full-attention smoke produced NaN/Inf"
    );

    println!("L1.2 FULLATTN SMOKE: PASS");
}
