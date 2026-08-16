#![cfg(feature = "cuda")]

use burn::{
    prelude::Device,
    tensor::{Int, Tensor},
};
use cubecl::{Runtime, cuda::CudaRuntime};
use qwen3_burn::{
    Precision, Qwen3_5LayerType, Qwen3_5MoeConfig,
    capture::{
        CaptureBackend, Qwen35DecodeState, Qwen35VaSnapshot, assert_no_new_allocs,
        memory_usage_snapshot,
    },
    rope_freqs,
};

type B = CaptureBackend;

fn tiny_config() -> Qwen3_5MoeConfig {
    Qwen3_5MoeConfig {
        vocab_size: 48,
        hidden_size: 32,
        num_hidden_layers: 2,
        layer_types: vec![
            Qwen3_5LayerType::LinearAttention,
            Qwen3_5LayerType::FullAttention,
        ],
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 8,
        partial_rotary_factor: 0.25,
        rms_norm_eps: 1e-6,
        rope_theta: 10_000_000.0,
        mrope_section: [1, 1, 1],
        num_experts: 4,
        num_experts_per_tok: 4,
        norm_topk_prob: true,
        moe_intermediate_size: 16,
        shared_expert_intermediate_size: 16,
        linear_key_head_dim: 8,
        linear_num_key_heads: 2,
        linear_num_value_heads: 4,
        linear_value_head_dim: 8,
        linear_conv_kernel_dim: 4,
        mtp_num_hidden_layers: 0,
    }
}

fn sync(device: &Device) {
    let client = CudaRuntime::client(device);
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

// GPU-run-by-orchestrator microgate. Ignored for normal test runs but compiled by
// `cargo check --features cuda --all-targets`.
#[test]
#[ignore = "GPU VA/allocator-stability probe; run explicitly on the CUDA orchestrator"]
fn qwen35_hybrid_decode_state_va_and_allocs_stay_stable() {
    let device = Device::default();
    device.seed(20_260_702);

    let cfg = tiny_config();
    let vocab = cfg.vocab_size;
    let t_max = 8usize;
    let max_new = 4usize;
    let prompt_len = 1usize;
    assert!(prompt_len + max_new <= t_max);

    let model = cfg.init_causal_lm(&device);
    let mut cache = model.model.new_cache_with_capacity(t_max);
    model.init_static_caches(&mut cache, 1);

    let prompt_ids = Tensor::<2, Int>::from_data([[3i64]], &device);
    let prompt_pos = Tensor::<2, Int>::from_data([[0i64]], &device);
    let _ = model.forward_prec(prompt_ids, prompt_pos, &mut cache, Precision::F32);
    sync(&device);

    let mut state = Qwen35DecodeState::new(1, vocab, t_max, max_new, &device, cache);
    state.pos = Some(
        state
            .pos
            .take()
            .expect("pos buffer missing")
            .add_scalar(prompt_len as i64),
    );

    let rotary_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
    let freqs = rope_freqs::<B>(rotary_dim, cfg.rope_theta, &device);
    let arange_tmax = Tensor::<1, Int>::arange(0..t_max as i64, &device);
    let client = CudaRuntime::client(&device);

    let snapshot = Qwen35VaSnapshot::from_hybrid(&state);
    let mut alloc_before = None;
    for step in 0..4 {
        let logits = model.forward_decode_static_pre(
            state.tok.as_ref().expect("tok buffer missing").clone(),
            state.pos.as_ref().expect("pos buffer missing").clone(),
            &mut state.cache,
            Precision::F32,
            &freqs,
            &arange_tmax,
        );
        state.last = Some(
            state
                .last
                .take()
                .expect("last buffer missing")
                .slice_assign([0..1, 0..vocab], logits),
        );
        state.pos = Some(
            state
                .pos
                .take()
                .expect("pos buffer missing")
                .add_scalar(1i64),
        );
        sync(&device);
        snapshot.assert_unchanged(&state, &format!("decode step {step}"));
        if step == 0 {
            alloc_before = Some(memory_usage_snapshot(&client));
        }
    }
    let alloc_after = memory_usage_snapshot(&client);
    assert_no_new_allocs(
        alloc_before.expect("allocation baseline missing after step 1"),
        alloc_after,
        "qwen35 tiny static decode steps 2..4",
    );

    state.reset_for_replay();
    sync(&device);
    snapshot.assert_unchanged(&state, "reset_for_replay");

    let logits = model.forward_decode_static_pre(
        state.tok.as_ref().expect("tok buffer missing").clone(),
        state.pos.as_ref().expect("pos buffer missing").clone(),
        &mut state.cache,
        Precision::F32,
        &freqs,
        &arange_tmax,
    );
    state.last = Some(
        state
            .last
            .take()
            .expect("last buffer missing")
            .slice_assign([0..1, 0..vocab], logits),
    );
    sync(&device);
    snapshot.assert_unchanged(&state, "post-reset decode step");
}
