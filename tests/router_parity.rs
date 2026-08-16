use burn::tensor::{DType, Distribution, Tensor};
use qwen3_burn::{Precision, Qwen3_5LayerType, Qwen3_5MoeConfig};

#[test]
fn test_route_topk_parity() {
    let device = Default::default();
    let cfg = Qwen3_5MoeConfig {
        vocab_size: 32,
        hidden_size: 64,
        num_hidden_layers: 1,
        layer_types: vec![Qwen3_5LayerType::FullAttention],
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 32,
        partial_rotary_factor: 0.25,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        mrope_section: [1, 1, 1],
        num_experts: 256,
        num_experts_per_tok: 8,
        norm_topk_prob: true,
        moe_intermediate_size: 32,
        shared_expert_intermediate_size: 32,
        linear_key_head_dim: 32,
        linear_num_key_heads: 1,
        linear_num_value_heads: 2,
        linear_value_head_dim: 32,
        linear_conv_kernel_dim: 4,
        mtp_num_hidden_layers: 0,
    };
    let model = cfg.init_causal_lm(&device);
    let layer = match &model.model.layers[0] {
        qwen3_burn::qwen3_5::Qwen3_5DecoderLayer::Full(l) => l,
        _ => unreachable!(),
    };
    let x: Tensor<3> = Tensor::random([1, 1, 64], Distribution::Normal(0.0, 1.0), &device);
    let (idx1, w1) = layer.mlp.route_topk(x.clone(), 8);
    let (idx2, w2) = {
        // Evaluate via batch > 1 path
        let x_dup: Tensor<3> = Tensor::cat(vec![x.clone(), x.clone()], 0);
        let (idx_b, w_b) = layer.mlp.route_topk(x_dup, 8);
        (idx_b.slice([0..1, 0..8]), w_b.slice([0..1, 0..8]))
    };

    let idx1_vec: Vec<i64> = idx1.cast(DType::I64).into_data().to_vec().unwrap();
    let idx2_vec: Vec<i64> = idx2.cast(DType::I64).into_data().to_vec().unwrap();
    assert_eq!(
        idx1_vec, idx2_vec,
        "Single-token route indices differ from multi-token reference"
    );

    let w1_vec: Vec<f32> = w1.cast(DType::F32).into_data().to_vec().unwrap();
    let w2_vec: Vec<f32> = w2.cast(DType::F32).into_data().to_vec().unwrap();
    for (a, b) in w1_vec.iter().zip(w2_vec.iter()) {
        assert!((a - b).abs() < 1e-4, "weights differ: {a} vs {b}");
    }
}
