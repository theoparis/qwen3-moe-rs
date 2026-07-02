//! End-to-end GRPO training loop — the A0 convergence smoke (T8).
//!
//! This is the launch SCAFFOLD that ties the pieces together: model -> frozen reference snapshot
//! -> optimizer (AdamW + global-norm grad clipping) -> loop { grpo_step_ragged } -> metrics. It
//! runs on the NdArray CPU backend with a TINY random model and a TOY learnable reward, so it needs
//! no GPU, no weights, no Python, no corpus — `cargo run --release --example grpo_train [steps]`.
//!
//! The toy reward is DENSE: `reward = fraction of completion tokens in the upper half of the vocab`
//! (id >= vocab/2). Dense means every sampled token contributes, so a group of completions almost
//! always has intra-group variance (GRPO needs that), and it is learnable — the policy raises it by
//! emitting higher-id tokens — so `mean_reward` should trend UP over steps. That is the convergence
//! smoke. (A single sparse target token is too rare on a tiny random model: every completion scores
//! 0, every group is zero-std, and there is no gradient.)
//!
//! TO TRAIN FOR REAL, swap three things (the loop is unchanged):
//!   1. Backend `Autodiff<NdArray>` -> `Autodiff<Cuda>` (+ `with_train_precision(Bf16)` for big models).
//!   2. `Qwen3Config::new()...` -> `Qwen3Config::qwen3_0_6b()` + `model.load_weights(sft.safetensors)`.
//!   3. The toy reward -> decode ids->text + `ManimReward::new().with_script(ABS_PATH)` (run the
//!      reward subprocesses behind a worker pool; render needs OS-level sandboxing).
//! Add: a real prompt dataset, periodic eval samples, and checkpoint/resume of model+optimizer+step.
//! For large models / large `group_size * max_new_tokens`, micro-batch the policy forward/backward
//! (the single full-batch autodiff pass here will OOM at scale — a deferred perf task).

use burn::backend::{Autodiff, NdArray};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use qwen3_burn::grpo::{grpo_step_ragged, GrpoConfig, GrpoTrainConfig, RolloutConfig, Rollouts};
use qwen3_burn::Qwen3Config;

type IB = NdArray;
type B = Autodiff<IB>;

const VOCAB: usize = 32;
const HALF: i64 = (VOCAB as i64) / 2; // dense reward: pay completion tokens with id >= HALF
const PAD: i64 = 0;
const EOS: i64 = 7;

fn main() {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(30);
    let dev = Default::default();

    // ---- model: tiny random Qwen3 (swap for qwen3_0_6b() + load_weights for a real run) ----
    let cfg = Qwen3Config::new()
        .with_vocab_size(VOCAB)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let mut policy = cfg.init_causal_lm::<B>(&dev);

    // ---- frozen reference: snapshot the (SFT) weights ONCE, on the inner (no-grad) backend.
    // It is never stepped; the KL term anchors the policy to it for the whole run. ----
    let ref_model = policy.valid();
    let ref_fingerprint: f32 = ref_model.model.embed_tokens_weight().abs().sum().into_scalar();

    // ---- optimizer: AdamW + GLOBAL-NORM gradient clipping (RL is high-variance — clip or explode) ----
    let mut optim = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let g = 4usize; // group size (GRPO needs G >= 2)
    let train_cfg = GrpoTrainConfig {
        grpo: GrpoConfig { group_size: g, ..GrpoConfig::default() },
        // VERY high temperature: a tiny random-init model has peaked logits, so anything lower
        // collapses each group to identical completions (zero-std, no learning signal). A real
        // pretrained model uses a normal temperature (~0.7-1.0).
        rollout: RolloutConfig { group_size: g, max_new_tokens: 8, temperature: 5.0, top_p: 1.0, top_k: 0 },
        eos: vec![EOS],
        lr: 3e-3,
    };

    // toy learnable reward: fraction of (masked) completion tokens equal to TARGET
    let reward_fn = |roll: &Rollouts<IB>| -> Vec<f32> {
        let n = roll.seq_ids.dims()[0];
        let (lp, glen) = (roll.prompt_len, roll.gen_len);
        let width = lp + glen;
        let ids = roll.seq_ids.clone().into_data().to_vec::<i64>().unwrap();
        let mask = roll.completion_mask.clone().into_data().to_vec::<f32>().unwrap();
        (0..n)
            .map(|s| {
                let (mut hits, mut tot) = (0.0f32, 0.0f32);
                for t in 0..glen {
                    if mask[s * glen + t] > 0.5 {
                        tot += 1.0;
                        if ids[s * width + lp + t] >= HALF {
                            hits += 1.0;
                        }
                    }
                }
                if tot > 0.0 { hits / tot } else { 0.0 }
            })
            .collect()
    };

    // ragged toy prompts (variable length -> exercises grpo_step_ragged's left-pad path)
    let prompts: Vec<Vec<i64>> =
        vec![vec![1, 2, 3, 4], vec![6, 1], vec![2, 3, 6], vec![4, 1, 2, 6, 3]];

    println!(
        "GRPO smoke | {steps} steps | prompts {} x G {} = {} completions/step | reward: upper-half (id >= {HALF})",
        prompts.len(),
        g,
        prompts.len() * g
    );

    let mut first_reward = f32::NAN;
    let mut last_reward = f32::NAN;
    for step in 0..steps {
        let (p, o, report) =
            grpo_step_ragged(policy, &ref_model, optim, prompts.clone(), PAD, &dev, &reward_fn, &train_cfg);
        policy = p;
        optim = o;

        assert!(report.metrics.total_loss.is_finite(), "step {step}: non-finite loss");
        if step == 0 {
            first_reward = report.mean_reward;
        }
        last_reward = report.mean_reward;

        if step < 3 || step % 5 == 0 || step == steps - 1 {
            println!(
                "step {step:3} | loss {:8.4} | kl {:7.4} | ratio {:.3} | reward {:.3} | zero_std {} | nonfinite {}",
                report.metrics.total_loss,
                report.metrics.kl_loss,
                report.metrics.mean_ratio,
                report.mean_reward,
                report.zero_std_groups,
                report.nonfinite_rewards,
            );
        }
    }

    // the frozen reference must NOT have moved
    let ref_after: f32 = ref_model.model.embed_tokens_weight().abs().sum().into_scalar();
    assert_eq!(ref_fingerprint, ref_after, "frozen reference changed during training");

    println!("\n===== GRPO SMOKE =====");
    println!("mean reward: {first_reward:.3} (step 0) -> {last_reward:.3} (step {})", steps - 1);
    println!("frozen reference unchanged: ✓");
    println!(
        "{}",
        if last_reward > first_reward {
            "reward increased — the toy GRPO objective is being optimized ✓"
        } else {
            "reward did not increase over this window (try more steps / higher LR)"
        }
    );
    println!("======================");
}
