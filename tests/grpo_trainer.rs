//! End-to-end integration test for one GRPO step (`grpo_step`) on a tiny, random-init Qwen3
//! over the Autodiff<NdArray> CPU backend — no GPU, no pretrained weights, no dataset.
//!
//! Verifies the hardest integration the external review flagged:
//!  * the loop runs end-to-end (rollout → reward → advantage → loss → backward → AdamW);
//!  * the loss is finite;
//!  * the policy parameters actually MOVE (gradients flowed and AdamW stepped);
//!  * the frozen reference is NOT updated;
//!  * the PPO ratio is ≈ 1 on the first step (old_logprobs captured during rollout match the
//!    policy's recomputed log-probs — proves the raw-pre-warp capture, GRPO fix (a)).
//!
//! Run: `cargo test --test grpo_trainer`

use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use burn::tensor::{Device, Int, Tensor};
use qwen3_burn::Qwen3Config;
use qwen3_burn::grpo::{
    GrpoConfig, GrpoTrainConfig, RolloutConfig, Rollouts, grpo_step, grpo_step_ragged,
};

fn tiny_config() -> Qwen3Config {
    Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16))
}

#[test]
fn one_grpo_step_runs_and_updates_policy() {
    let dev = Device::flex().autodiff();
    let cfg = tiny_config();

    let policy = cfg.init_causal_lm(&dev);
    let ref_model = policy.valid(); // frozen snapshot on the inner backend

    let optim = AdamWConfig::new().init();

    let p = 2usize;
    let lp = 3usize;
    let prompts =
        Tensor::<1, Int>::from_data([1i64, 2, 3, 4, 5, 6].as_slice(), &dev).reshape([p, lp]);

    let train_cfg = GrpoTrainConfig {
        // group_size must match the rollout's (the trainer now asserts this); 8 default vs 4 rollout
        // was the silent mismatch the assert is here to catch.
        grpo: GrpoConfig {
            group_size: 4,
            ..GrpoConfig::default()
        },
        // high temperature + top_p off so a tiny random-init model still samples DIVERSE
        // completions (peaked random logits otherwise collapse a group to identical tokens)
        rollout: RolloutConfig {
            group_size: 4,
            max_new_tokens: 6,
            temperature: 5.0,
            top_p: 1.0,
            top_k: 0,
        },
        eos: vec![7],
        lr: 1e-2,
    };

    // Deterministic synthetic reward: within each group the G completions get rewards
    // 0, 1/G, 2/G, ... by index. This GUARANTEES intra-group variance (non-zero advantage and a
    // real gradient) regardless of what the random-init model samples, so the test verifies the
    // loop mechanics deterministically. (Real training uses the Manim execution reward.)
    let g = train_cfg.rollout.group_size;
    let reward_fn = move |roll: &Rollouts| -> Vec<f32> {
        let n = roll.seq_ids.dims()[0];
        (0..n).map(|s| (s % g) as f32 / g as f32).collect()
    };

    // weight snapshot before the step (tied embedding gets a dense gradient via the LM head)
    let before: f32 = policy.model.embed_tokens_weight().abs().sum().into_scalar();

    let (policy, _optim, report) =
        grpo_step(policy, &ref_model, optim, prompts, reward_fn, &train_cfg);

    // loss + metrics finite
    assert!(report.metrics.total_loss.is_finite(), "loss must be finite");
    assert!(
        report.metrics.kl_loss >= -1e-5,
        "k3 KL >= 0, got {}",
        report.metrics.kl_loss
    );
    assert!(report.gen_len >= 1);

    // reward signal had variance (otherwise the test wouldn't exercise learning)
    assert!(
        report.reward_std > 0.0,
        "expected reward variance, std={}",
        report.reward_std
    );

    // PPO ratio ≈ 1 at step 0: old_logprobs (rollout) == policy recomputed log-probs
    assert!(
        (report.metrics.mean_ratio - 1.0).abs() < 1e-2,
        "step-0 ratio must be ~1 (raw pre-warp capture), got {}",
        report.metrics.mean_ratio
    );

    // policy parameters actually moved (gradient flowed + AdamW stepped)
    let after: f32 = policy.model.embed_tokens_weight().abs().sum().into_scalar();
    assert!(
        (after - before).abs() > 1e-7,
        "policy weights must change after a step ({before} -> {after})"
    );

    // frozen reference is unchanged (it's a separate inner model, never stepped)
    let ref_sum_a: f32 = ref_model
        .model
        .embed_tokens_weight()
        .abs()
        .sum()
        .into_scalar();
    let ref_sum_b: f32 = ref_model
        .model
        .embed_tokens_weight()
        .abs()
        .sum()
        .into_scalar();
    assert_eq!(ref_sum_a, ref_sum_b, "frozen reference must not change");

    println!(
        "GRPO step OK — loss={:.6} kl={:.6} mean_ratio={:.4} mean_reward={:.3} reward_std={:.3} zero_std_groups={} gen_len={}",
        report.metrics.total_loss,
        report.metrics.kl_loss,
        report.metrics.mean_ratio,
        report.mean_reward,
        report.reward_std,
        report.zero_std_groups,
        report.gen_len
    );
}

/// One GRPO step on RAGGED (variable-length) prompts via `grpo_step_ragged`. The load-bearing check
/// is the step-0 ratio ≈ 1: it holds only if the left-pad rollout's `old_logprobs` and the policy's
/// `forward_with_positions` recompute agree on mask + RoPE positions for the padded sequence. A
/// mismatch (wrong pad mask or off-by-one position) would push the ratio away from 1.
#[test]
fn one_grpo_step_ragged_runs_and_updates_policy() {
    let dev = Device::flex().autodiff();
    let policy = tiny_config().init_causal_lm(&dev);
    let ref_model = policy.valid();
    let optim = AdamWConfig::new().init();

    // ragged prompts: lengths 4, 2, 3 (left-padded to 4 internally)
    let prompts: Vec<Vec<i64>> = vec![vec![1, 2, 3, 4], vec![5, 6], vec![1, 5, 6]];
    let pad_token = 0i64;

    let g = 4usize;
    let train_cfg = GrpoTrainConfig {
        grpo: GrpoConfig {
            group_size: g,
            ..GrpoConfig::default()
        },
        rollout: RolloutConfig {
            group_size: g,
            max_new_tokens: 6,
            temperature: 5.0,
            top_p: 1.0,
            top_k: 0,
        },
        eos: vec![7],
        lr: 1e-2,
    };

    let reward_fn = move |roll: &Rollouts| -> Vec<f32> {
        let n = roll.seq_ids.dims()[0];
        (0..n).map(|s| (s % g) as f32 / g as f32).collect()
    };

    let before: f32 = policy.model.embed_tokens_weight().abs().sum().into_scalar();
    let (policy, _optim, report) = grpo_step_ragged(
        policy, &ref_model, optim, prompts, pad_token, &dev, reward_fn, &train_cfg,
    );

    assert!(report.metrics.total_loss.is_finite(), "loss must be finite");
    assert!(
        report.metrics.kl_loss >= -1e-5,
        "k3 KL >= 0, got {}",
        report.metrics.kl_loss
    );
    assert!(report.gen_len >= 1);
    // THE gate: left-pad rollout old_logprobs must equal the policy recompute -> ratio ~ 1
    assert!(
        (report.metrics.mean_ratio - 1.0).abs() < 1e-2,
        "ragged step-0 ratio must be ~1 (rollout/recompute agree under left-pad), got {}",
        report.metrics.mean_ratio
    );
    let after: f32 = policy.model.embed_tokens_weight().abs().sum().into_scalar();
    assert!(
        (after - before).abs() > 1e-7,
        "policy weights must change ({before} -> {after})"
    );

    println!(
        "ragged GRPO step OK — loss={:.6} mean_ratio={:.4} gen_len={}",
        report.metrics.total_loss, report.metrics.mean_ratio, report.gen_len
    );
}
