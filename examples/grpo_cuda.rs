//! grpo_cuda — GRPO on the REAL Qwen3-0.6B, `Autodiff<Cuda>`. On-device smoke.
//!
//! The CUDA analog of `grpo_train`: same loop, but a real pretrained model + real tokenized prompts
//! on the GPU. Proves the GRPO step works on-device — rollout, the `[N, L, vocab]` autodiff
//! forward/backward, AdamW, frozen reference — without OOM/NaN, and that the objective is optimized.
//!
//! Reward is a dense learnable toy: fraction of completion tokens with an EVEN id. ~Half the vocab
//! is even, so a real model starts near 0.5 with real intra-group variance and CAN raise it — a
//! clean signal that isolates "does the CUDA loop optimize" from "is the Manim reward good".
//! (An upper-half-of-vocab reward does NOT work on a real model: it emits coherent low-id tokens,
//! so the fraction is ~0 with no signal — the random-model `grpo_train` toy doesn't transfer.)
//! For a real Manim run,
//! swap the reward for `ManimReward::new().with_script(ABS_PATH)` (decode ids->text first) and add a
//! real prompt dataset + checkpoint/resume; micro-batch the policy pass for larger N.
//!
//! Build: RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda,train --example grpo_cuda
//! Run:   ./target/release/examples/grpo_cuda [steps]

use std::time::Instant;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::grad_clipping::GradientClippingConfig;
use burn::module::AutodiffModule;
use burn::optim::AdamWConfig;
use qwen3_burn::grpo::{grpo_step_ragged, GrpoConfig, GrpoTrainConfig, RolloutConfig, Rollouts};
use qwen3_burn::{Qwen3Config, Qwen3Tokenizer};

type IB = Cuda;
type B = burn::backend::Autodiff<Cuda>;

// Repo-relative; bring your own Qwen3-0.6B(-Base) weights (e.g. https://huggingface.co/Qwen/Qwen3-0.6B).
const WEIGHTS: &str = "models/Qwen3-0.6B-Base/model_f32.safetensors";
const TOKENIZER: &str = "models/Qwen3-0.6B-Base/tokenizer.json";
const EOS: i64 = 151643; // Qwen3 <|endoftext|>
const MAX_NEW: usize = 48;
const G: usize = 4; // group size

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let steps: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(25);
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Autodiff<Cuda>");

    // ---- tokenizer + model ----
    let tokenizer = Qwen3Tokenizer::from_file(TOKENIZER).map_err(|e| format!("tokenizer: {e}"))?;
    let cfg = Qwen3Config::qwen3_0_6b();
    let mut policy = cfg.init_causal_lm::<B>(&device);
    println!("loading f32 weights ...");
    policy.load_weights(WEIGHTS).map_err(|e| format!("load_weights: {e:?}"))?;

    // ---- real ragged prompts -> token ids ----
    let prompt_texts = ["The derivative of x squared is", "To animate a circle in Manim, we"];
    let mut prompts: Vec<Vec<i64>> = Vec::new();
    for t in &prompt_texts {
        let (ids, _) = tokenizer.encode_no_pad(t).map_err(|e| format!("encode: {e}"))?;
        prompts.push(ids.iter().map(|&u| u as i64).collect());
    }
    let pad_token: i64 = EOS; // left-pad with EOS (masked out anyway)

    // ---- frozen reference: snapshot ONCE on the inner (no-grad) backend ----
    let ref_model = policy.valid();
    let ref_fp: f32 = ref_model.model.embed_tokens_weight().abs().sum().into_scalar();

    // ---- optimizer: AdamW + global-norm grad clipping ----
    let mut optim = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let train_cfg = GrpoTrainConfig {
        grpo: GrpoConfig { group_size: G, ..GrpoConfig::default() },
        rollout: RolloutConfig { group_size: G, max_new_tokens: MAX_NEW, temperature: 1.0, top_p: 1.0, top_k: 0 },
        eos: vec![EOS],
        lr: 1e-4,
    };

    // dense learnable reward: fraction of (masked) completion tokens in the upper vocab half
    let reward_fn = move |roll: &Rollouts<IB>| -> Vec<f32> {
        let n = roll.seq_ids.dims()[0];
        let (lp, glen) = (roll.prompt_len, roll.gen_len);
        let width = lp + glen;
        let ids = roll.seq_ids.clone().into_data().to_vec::<i32>().unwrap(); // CUDA Int = I32
        let mask = roll.completion_mask.clone().into_data().to_vec::<f32>().unwrap();
        (0..n)
            .map(|s| {
                let (mut hits, mut tot) = (0.0f32, 0.0f32);
                for t in 0..glen {
                    if mask[s * glen + t] > 0.5 {
                        tot += 1.0;
                        if ids[s * width + lp + t] % 2 == 0 {
                            hits += 1.0;
                        }
                    }
                }
                if tot > 0.0 { hits / tot } else { 0.0 }
            })
            .collect()
    };

    println!(
        "prompts {} x G {} = {} completions/step | max_new {} | reward: even-id fraction | {steps} steps",
        prompts.len(),
        G,
        prompts.len() * G,
        MAX_NEW
    );

    let start = Instant::now();
    let mut first_reward = f32::NAN;
    let mut last_reward = f32::NAN;
    for step in 0..steps {
        let t0 = Instant::now();
        let (p, o, report) =
            grpo_step_ragged(policy, &ref_model, optim, prompts.clone(), pad_token, &device, &reward_fn, &train_cfg);
        policy = p;
        optim = o;
        assert!(report.metrics.total_loss.is_finite(), "step {step}: non-finite loss");
        if step == 0 {
            first_reward = report.mean_reward;
        }
        last_reward = report.mean_reward;
        println!(
            "step {step:3} | loss {:8.4} | kl {:7.4} | ratio {:.3} | reward {:.3} | zero_std {} | nonfinite {} | {:.1}s",
            report.metrics.total_loss,
            report.metrics.kl_loss,
            report.metrics.mean_ratio,
            report.mean_reward,
            report.zero_std_groups,
            report.nonfinite_rewards,
            t0.elapsed().as_secs_f64(),
        );
    }

    let ref_after: f32 = ref_model.model.embed_tokens_weight().abs().sum().into_scalar();
    assert_eq!(ref_fp, ref_after, "frozen reference changed during training");

    println!("\n===== GRPO CUDA SMOKE =====");
    println!("mean reward: {first_reward:.3} (step 0) -> {last_reward:.3} (step {})", steps - 1);
    println!("frozen reference unchanged: ✓ | wall {:.1}s", start.elapsed().as_secs_f64());
    println!("===========================");
    Ok(())
}
