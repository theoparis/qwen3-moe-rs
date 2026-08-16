#![cfg(feature = "cuda")]

use burn::tensor::Tensor;
use qwen3_burn::{
    GdnStateCache, Precision, Qwen3_5LayerType, Qwen3_5MoeConfig,
    capture::{CaptureBackend, float_va},
    qwen3_5::Qwen3_5DecoderLayer,
};

type B = CaptureBackend;

// GPU-run-by-orchestrator microgate. This is ignored for normal test runs but compiled by
// `cargo check --features cuda --all-targets`.
#[test]
#[ignore = "GPU VA-stability probe; run explicitly on the CUDA orchestrator"]
fn gdn_static_state_va_stays_stable() {
    let device = Device::default();
    let mut cache = GdnStateCache::<B>::new(1, 2, 2, 6, 4);
    cache.init_static(1, &device);

    let initial_state_va = float_va(cache.state.as_ref().expect("static state missing"));
    let initial_conv_va = float_va(cache.conv.as_ref().expect("static conv missing"));
    for i in 0..8 {
        let new_state = {
            let prev = cache.state.as_ref().expect("static state missing").clone();
            let new_state = prev.clone().add_scalar((i + 1) as f64);
            drop(prev);
            new_state
        };
        cache.set_state_static(new_state);
        assert_eq!(
            initial_state_va,
            float_va(cache.state.as_ref().expect("static state missing")),
            "GDN static state VA moved after set_state_static iteration {i}"
        );
        assert_eq!(
            initial_conv_va,
            float_va(cache.conv.as_ref().expect("static conv missing")),
            "GDN static conv VA moved during set_state_static iteration {i}"
        );
    }

    let cfg = Qwen3_5MoeConfig {
        vocab_size: 32,
        hidden_size: 8,
        num_hidden_layers: 1,
        layer_types: vec![Qwen3_5LayerType::LinearAttention],
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 4,
        partial_rotary_factor: 0.25,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        mrope_section: [1, 1, 1],
        num_experts: 1,
        num_experts_per_tok: 1,
        norm_topk_prob: true,
        moe_intermediate_size: 4,
        shared_expert_intermediate_size: 4,
        linear_key_head_dim: 2,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_value_head_dim: 2,
        linear_conv_kernel_dim: 4,
        mtp_num_hidden_layers: 0,
    };
    let model = cfg.init_causal_lm(&device);
    let layer = match &model.model.layers[0] {
        Qwen3_5DecoderLayer::Linear(layer) => layer,
        Qwen3_5DecoderLayer::Full(_) => unreachable!("test config has one linear layer"),
    };

    for i in 0..8 {
        let hidden = Tensor::<3>::zeros([1, 1, cfg.hidden_size], &device);
        let _out = layer
            .linear_attn
            .step_recurrent_static(hidden, &mut cache, Precision::F32);
        assert_eq!(
            initial_state_va,
            float_va(cache.state.as_ref().expect("static state missing")),
            "GDN static state VA moved after step_recurrent_static iteration {i}"
        );
        assert_eq!(
            initial_conv_va,
            float_va(cache.conv.as_ref().expect("static conv missing")),
            "GDN static conv VA moved after step_recurrent_static iteration {i}"
        );
    }

    cache.reset_for_replay();
    assert_eq!(
        initial_state_va,
        float_va(cache.state.as_ref().expect("static state missing")),
        "GDN static state VA moved after reset_for_replay"
    );
    assert_eq!(
        initial_conv_va,
        float_va(cache.conv.as_ref().expect("static conv missing")),
        "GDN static conv VA moved after reset_for_replay"
    );

    let hidden = Tensor::<3>::zeros([1, 1, cfg.hidden_size], &device);
    let _out = layer
        .linear_attn
        .step_recurrent_static(hidden, &mut cache, Precision::F32);
    assert_eq!(
        initial_state_va,
        float_va(cache.state.as_ref().expect("static state missing")),
        "GDN static state VA moved after post-reset step_recurrent_static"
    );
    assert_eq!(
        initial_conv_va,
        float_va(cache.conv.as_ref().expect("static conv missing")),
        "GDN static conv VA moved after post-reset step_recurrent_static"
    );

    let hidden = Tensor::<3>::zeros([1, 6, cfg.hidden_size], &device);
    let _out =
        layer
            .linear_attn
            .forward_prefill_recurrent_static(hidden, &mut cache, Precision::F32);
    assert_eq!(
        initial_state_va,
        float_va(cache.state.as_ref().expect("static state missing")),
        "GDN static state VA moved after forward_prefill_recurrent_static"
    );
    assert_eq!(
        initial_conv_va,
        float_va(cache.conv.as_ref().expect("static conv missing")),
        "GDN static conv VA moved after forward_prefill_recurrent_static"
    );
}
