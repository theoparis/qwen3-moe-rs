//! GRPO training step — the integrator that ties rollout + reward + advantage + loss + AdamW.
//!
//! The Burn-specific correctness points the external review flagged (docs/GRPO_PLAN.md §2b):
//!   * **rollout + reference are NO-GRAD on the inner backend.** Burn has no global no-grad
//!     context, so we sample with `policy.valid()` (the inner, non-autodiff module) and run the
//!     frozen reference on `B::InnerBackend`. Their log-probs are lifted into the autodiff graph
//!     as CONSTANTS (`from_data`), never tracked — so the only autodiff graph built per step is
//!     the single policy gradient pass.
//!   * **token alignment / off-by-one.** `logits[:, j]` predicts the token at `j+1`; the policy
//!     log-prob of completion token at position `Lp+i` comes from `logits[:, Lp+i-1]`. We compute
//!     shifted log-probs over the whole sequence and slice the completion region, so it lines up
//!     with the rollout's `old_logprobs` and `completion_mask` exactly.
//!   * **zero-std groups keep their KL leash** (review fix): a group whose G rewards are all equal
//!     has advantage 0 (so 0 policy gradient already), but its tokens STAY in the completion mask
//!     so the k3 KL keeps anchoring them to the reference — OpenRLHF-literal. (Dropping them from
//!     the mask, as an earlier version did, removed the KL pull-back and invited mode collapse.)
//!     We only count + report the zero-std rate.
//!
//! `grpo_step` consumes and returns `(policy, optimizer)` (the Burn manual-loop idiom) so callers
//! never have to name the optimizer's concrete type. Reward is a closure over the rollout, so the
//! trainer needs no tokenizer: production decodes ids→text and calls `ManimReward`; tests pass a
//! synthetic reward over token ids.

use burn::module::AutodiffModule;
use burn::optim::{GradientsParams, Optimizer};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Bool, Int, Tensor};

use super::{
    group_norm_advantage, group_sample_cached, group_sample_padded, grpo_loss, token_logprobs,
    AdvantageEstimator, GrpoConfig, GrpoMetrics, RolloutConfig, Rollouts,
};
use crate::Qwen3ForCausalLM;

/// Configuration for one GRPO step.
#[derive(Clone, Debug)]
pub struct GrpoTrainConfig {
    pub grpo: GrpoConfig,
    pub rollout: RolloutConfig,
    pub eos: Vec<i64>,
    pub lr: f64,
}

/// Per-step report (detached metrics + reward stats).
#[derive(Clone, Copy, Debug)]
pub struct StepReport {
    pub metrics: GrpoMetrics,
    pub mean_reward: f32,
    pub reward_std: f32,
    /// Number of zero-std prompt groups (all G rewards equal -> advantage 0). Reported only: these
    /// groups are NOT dropped — their tokens stay in the mask so the KL keeps anchoring them.
    pub zero_std_groups: usize,
    /// Completions whose reward_fn returned a non-finite value (replaced with 0.0 before the math).
    pub nonfinite_rewards: usize,
    pub gen_len: usize,
}

/// Run ONE GRPO step: rollout → reward → group-norm advantage → policy/ref log-probs → loss →
/// backward → AdamW. Returns the updated policy + optimizer and a metrics report.
///
/// * `policy`    — trainable model on the autodiff backend `B`.
/// * `ref_model` — FROZEN reference on `B::InnerBackend` (snapshot of the SFT weights; never stepped).
/// * `reward_fn` — maps the rollout to one scalar reward per completion (`N = num_prompts * G`).
///
/// Uniform prompt length. For ragged (variable-length) prompts use [`grpo_step_ragged`].
pub fn grpo_step<B, O, R>(
    policy: Qwen3ForCausalLM<B>,
    ref_model: &Qwen3ForCausalLM<B::InnerBackend>,
    optim: O,
    prompts: Tensor<B, 2, Int>,
    reward_fn: R,
    cfg: &GrpoTrainConfig,
) -> (Qwen3ForCausalLM<B>, O, StepReport)
where
    B: AutodiffBackend,
    O: Optimizer<Qwen3ForCausalLM<B>, B>,
    R: Fn(&Rollouts<B::InnerBackend>) -> Vec<f32>,
{
    grpo_step_inner(policy, ref_model, optim, prompts, None, reward_fn, cfg)
}

/// One GRPO step on a batch of VARIABLE-LENGTH prompts. The prompts are left-padded to a common
/// length with `pad_token`; the attention mask + RoPE positions make the pad inert (rollout and the
/// policy/ref forward use `forward_with_positions` / `group_sample_padded`, both left-pad-invariant
/// and parity-tested). Completions align uniformly at the padded length, so the loss is unchanged.
#[allow(clippy::too_many_arguments)]
pub fn grpo_step_ragged<B, O, R>(
    policy: Qwen3ForCausalLM<B>,
    ref_model: &Qwen3ForCausalLM<B::InnerBackend>,
    optim: O,
    prompts: Vec<Vec<i64>>,
    pad_token: i64,
    device: &B::Device,
    reward_fn: R,
    cfg: &GrpoTrainConfig,
) -> (Qwen3ForCausalLM<B>, O, StepReport)
where
    B: AutodiffBackend,
    O: Optimizer<Qwen3ForCausalLM<B>, B>,
    R: Fn(&Rollouts<B::InnerBackend>) -> Vec<f32>,
{
    let p = prompts.len();
    assert!(p > 0, "grpo_step_ragged: no prompts");
    let prompt_lens: Vec<usize> = prompts.iter().map(|q| q.len()).collect();
    // Every prompt must have at least one real token. An all-pad row attends only to its own
    // (diagonal-unmasked) pad position, so the rollout + recompute still agree (ratio ~ 1) while the
    // prompt is semantically empty — insert a BOS or drop the row rather than train on noise.
    assert!(prompt_lens.iter().all(|&l| l > 0), "grpo_step_ragged: every prompt must be non-empty");
    let lp = *prompt_lens.iter().max().unwrap();

    // left-pad each prompt to `lp` with `pad_token`
    let mut flat = Vec::with_capacity(p * lp);
    for q in &prompts {
        flat.extend(std::iter::repeat(pad_token).take(lp - q.len()));
        flat.extend_from_slice(q);
    }
    let padded = Tensor::<B, 1, Int>::from_data(flat.as_slice(), device).reshape([p, lp]);

    grpo_step_inner(policy, ref_model, optim, padded, Some(prompt_lens), reward_fn, cfg)
}

/// Shared GRPO step. `prompt_lens = None` is the uniform fast path (cached rollout, `arange`
/// positions, no attention mask). `Some(lens)` is the left-padded path (`group_sample_padded` +
/// `forward_with_positions` with a pad mask + `cumsum` positions). `prompts` is already padded to
/// `[P, lp]`; `lens[p]` is prompt p's real (unpadded) length.
#[allow(clippy::too_many_arguments)]
fn grpo_step_inner<B, O, R>(
    policy: Qwen3ForCausalLM<B>,
    ref_model: &Qwen3ForCausalLM<B::InnerBackend>,
    mut optim: O,
    prompts: Tensor<B, 2, Int>,
    prompt_lens: Option<Vec<usize>>,
    reward_fn: R,
    cfg: &GrpoTrainConfig,
) -> (Qwen3ForCausalLM<B>, O, StepReport)
where
    B: AutodiffBackend,
    O: Optimizer<Qwen3ForCausalLM<B>, B>,
    R: Fn(&Rollouts<B::InnerBackend>) -> Vec<f32>,
{
    let device = prompts.device();
    let [p, _lp] = prompts.dims();
    let g = cfg.rollout.group_size;
    assert_eq!(
        cfg.grpo.group_size, g,
        "GrpoConfig.group_size ({}) must equal RolloutConfig.group_size ({}); grouping uses the rollout value",
        cfg.grpo.group_size, g
    );
    let n = p * g;
    assert!(
        cfg.grpo.estimator != AdvantageEstimator::GroupNorm || g >= 2,
        "GroupNorm GRPO needs group_size >= 2 (G=1 gives zero centered advantage -> no policy gradient)"
    );

    // ---- 1. rollout on the inner (no-grad) policy snapshot ----
    let inner_policy = policy.valid();
    let prompts_inner = Tensor::<B::InnerBackend, 2, Int>::from_data(prompts.into_data(), &device);
    let roll: Rollouts<B::InnerBackend> = match &prompt_lens {
        // uniform: KV-cache rollout (O(T))
        None => group_sample_cached(&inner_policy, prompts_inner, &cfg.rollout, &cfg.eos),
        // ragged: left-pad-aware rollout (correct under variable prompt length)
        Some(lens) => group_sample_padded(&inner_policy, prompts_inner, lens, &cfg.rollout, &cfg.eos),
    };
    let lp = roll.prompt_len;
    let glen = roll.gen_len;

    // ---- 2. reward + zero-std group count (report only; we do NOT drop zero-std groups: their
    // advantage is already 0 via std_eps, but their tokens stay in the mask so KL keeps anchoring
    // them to the reference — OpenRLHF-literal). ----
    let raw_rewards = reward_fn(&roll);
    assert_eq!(raw_rewards.len(), n, "reward_fn must return one reward per completion");
    // Sanitize non-finite rewards: a NaN/inf (a buggy reward, or a harness printing "nan") would
    // poison mean/std, the advantage, the loss and AdamW state. Replace with 0.0 and count it.
    let nonfinite_rewards = raw_rewards.iter().filter(|r| !r.is_finite()).count();
    let rewards: Vec<f32> = raw_rewards.iter().map(|&r| if r.is_finite() { r } else { 0.0 }).collect();
    let mean_reward = rewards.iter().sum::<f32>() / n as f32;
    let reward_std = (rewards.iter().map(|r| (r - mean_reward).powi(2)).sum::<f32>() / n as f32).sqrt();
    let mut zero_std_groups = 0;
    for gi in 0..p {
        let s = &rewards[gi * g..(gi + 1) * g];
        let m = s.iter().sum::<f32>() / g as f32;
        let var = s.iter().map(|r| (r - m).powi(2)).sum::<f32>() / (g.saturating_sub(1).max(1) as f32);
        if var.sqrt() < 1e-8 {
            zero_std_groups += 1;
        }
    }

    // ---- 3. lift rollout outputs into the autodiff backend as CONSTANTS ----
    let seq_ids = Tensor::<B, 2, Int>::from_data(roll.seq_ids.clone().into_data(), &device); // [n, l]
    let old_lp = Tensor::<B, 2>::from_data(roll.old_logprobs.into_data(), &device); // [n, gen]
    let mask = Tensor::<B, 2>::from_data(roll.completion_mask.into_data(), &device); // [n, gen]
    let rewards_t = Tensor::<B, 1>::from_floats(rewards.as_slice(), &device);
    let adv = group_norm_advantage(rewards_t, p, g, &cfg.grpo); // [n] constant

    // Full-sequence attention mask + RoPE positions for the left-padded path (None for uniform).
    let attn = prompt_lens.as_ref().map(|lens| full_mask_and_positions(lens, lp, glen, g));
    let total = lp + glen;

    // ---- 4. reference log-probs (no grad, inner backend) -> lift as constant ----
    let ref_logits = match &attn {
        None => inner_ref_logits(ref_model, &roll.seq_ids),
        Some((m, pos)) => {
            let mi = Tensor::<B::InnerBackend, 1, Bool>::from_data(m.as_slice(), &device).reshape([n, total]);
            let pi = Tensor::<B::InnerBackend, 1, Int>::from_data(pos.as_slice(), &device).reshape([n, total]);
            ref_model.forward_with_positions(roll.seq_ids.clone(), Some(mi), pi)
        }
    };
    let logp_ref_inner = completion_logprobs(&ref_logits, &roll.seq_ids, lp, glen);
    let logp_ref = Tensor::<B, 2>::from_data(logp_ref_inner.into_data(), &device); // [n, gen]

    // ---- 5. policy log-probs WITH grad (autodiff) ----
    let logits = match &attn {
        None => policy.forward(seq_ids.clone(), None),
        Some((m, pos)) => {
            let mb = Tensor::<B, 1, Bool>::from_data(m.as_slice(), &device).reshape([n, total]);
            let pb = Tensor::<B, 1, Int>::from_data(pos.as_slice(), &device).reshape([n, total]);
            policy.forward_with_positions(seq_ids.clone(), Some(mb), pb)
        }
    };
    let logp_pi = completion_logprobs(&logits, &seq_ids, lp, glen); // [n, gen], grad-tracked

    // ---- 6. GRPO loss + backward + AdamW ----
    let (loss, metrics) = grpo_loss(logp_pi, old_lp, logp_ref, adv, mask, &cfg.grpo);
    let grads = GradientsParams::from_grads(loss.backward(), &policy);
    let policy = optim.step(cfg.lr, policy, grads);

    let report =
        StepReport { metrics, mean_reward, reward_std, zero_std_groups, nonfinite_rewards, gen_len: glen };
    (policy, optim, report)
}

/// Full-sequence attention mask + RoPE positions for a left-padded GRPO batch, row-major
/// `[N*(lp+glen)]` (`N = P*G`). For prompt `pi` the first `lp - lens[pi]` columns are pad
/// (`mask = false`); the real prompt tokens and all `glen` completion columns are real
/// (`mask = true`). Positions are `cumsum(mask)-1` with pad clamped to 0, so the first completion
/// token sits at position `lens[pi]` (its true prompt length).
fn full_mask_and_positions(lens: &[usize], lp: usize, glen: usize, g: usize) -> (Vec<bool>, Vec<i64>) {
    let total = lp + glen;
    let mut mask = Vec::with_capacity(lens.len() * g * total);
    let mut pos = Vec::with_capacity(lens.len() * g * total);
    for &plen in lens {
        let pad = lp - plen;
        for _ in 0..g {
            let mut c = 0i64;
            for col in 0..total {
                let real = col >= pad; // pad occupies [0, pad); real prompt [pad, lp); completions [lp, total)
                mask.push(real);
                pos.push(if real {
                    let v = c;
                    c += 1;
                    v
                } else {
                    0
                });
            }
        }
    }
    (mask, pos)
}

/// Forward the inner (no-grad) reference and return its full logits `[n, l, v]`.
fn inner_ref_logits<IB: burn::prelude::Backend>(
    ref_model: &Qwen3ForCausalLM<IB>,
    seq_ids: &Tensor<IB, 2, Int>,
) -> Tensor<IB, 3> {
    ref_model.forward(seq_ids.clone(), None)
}

/// Per-token log-probs of the COMPLETION tokens, with the off-by-one shift applied.
///
/// `logits[:, j]` predicts token `j+1`. We compute log-probs of tokens `1..l` from logits `0..l-1`,
/// then slice the completion region `[lp-1 .. lp-1+gen]`. Result `[n, gen]` aligns with the
/// rollout's `old_logprobs` / `completion_mask`.
fn completion_logprobs<BB: burn::prelude::Backend>(
    logits: &Tensor<BB, 3>,
    seq_ids: &Tensor<BB, 2, Int>,
    lp: usize,
    glen: usize,
) -> Tensor<BB, 2> {
    let [n, l, v] = logits.dims();
    let logits_shift = logits.clone().slice([0..n, 0..l - 1, 0..v]); // predicts tokens 1..l
    let targets_shift = seq_ids.clone().slice([0..n, 1..l]);
    let logp_all = token_logprobs(logits_shift, targets_shift); // [n, l-1]
    let start = lp - 1;
    logp_all.slice([0..n, start..start + glen]) // [n, glen]
}

#[cfg(test)]
mod tests {
    // The end-to-end one-step integration test lives in tests/grpo_trainer.rs (it needs the
    // NdArray dev-dependency for a concrete backend).
}
