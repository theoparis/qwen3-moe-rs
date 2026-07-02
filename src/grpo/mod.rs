//! GRPO (Group Relative Policy Optimization) for qwen3-burn — Phase A.
//!
//! Reproduces OpenRLHF v0.10.4 GRPO (see `docs/GRPO_PLAN.md`): group-normalized advantage
//! (no critic), clipped PPO surrogate, k3 KL to a frozen reference, token-level global-mean
//! reduction. The math here is verified against `tests/ref/grpo_expected.json` (emitted by the
//! A0 Python reference `a0/grpo_reference.py`) in `tests/grpo_math.rs`.
//!
//! This module currently provides the loss/advantage/log-prob math (the load-bearing,
//! parity-tested core). The rollout, reward, and trainer wiring are separate modules.

mod logprob;
mod loss;
mod reward;
mod rollout;
mod trainer;

pub use logprob::token_logprobs;
pub use loss::{grpo_loss, group_norm_advantage, GrpoMetrics};
pub use reward::{ManimReward, RewardFn};
pub use rollout::{
    group_sample, group_sample_cached, group_sample_cached_device, group_sample_cached_device_loop,
    group_sample_cached_device_static, group_sample_cached_shrink, group_sample_padded, RolloutConfig,
    Rollouts,
};
pub use trainer::{grpo_step, grpo_step_ragged, GrpoTrainConfig, StepReport};

/// How the per-token policy loss is reduced to a scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Reduction {
    /// OpenRLHF-literal: `sum(L * mask) / sum(mask)` over ALL response tokens in the batch.
    #[default]
    TokenGlobal,
    /// DeepSeekMath: average within each sequence (`1/|o_i|`) then average sequences.
    /// Exposed for parity experiments; NOT the default (see review decision).
    SeqMean,
}

/// Advantage estimator (OpenRLHF `advantage_estimator`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AdvantageEstimator {
    /// `(r - mean_G) / (std_G + eps)`, sample std (ddof=1). OpenRLHF GRPO default.
    #[default]
    GroupNorm,
    /// `r - mean_G` (no std). OpenRLHF `dr_grpo` / `reinforce_baseline`.
    DrGrpo,
}

/// GRPO hyperparameters. Defaults reproduce OpenRLHF canonical GRPO (k3 KL in the loss).
#[derive(Clone, Copy, Debug)]
pub struct GrpoConfig {
    /// Group size `G` (`n_samples_per_prompt`).
    pub group_size: usize,
    /// PPO clip lower / upper (`eps_low`, `eps_high`). OpenRLHF default 0.2 / 0.2.
    pub eps_low: f32,
    pub eps_high: f32,
    /// KL coefficient `beta` (canonical GRPO ≈ 1e-3).
    pub beta: f32,
    /// Advantage estimator.
    pub estimator: AdvantageEstimator,
    /// Loss reduction (default token-global, OpenRLHF-literal).
    pub reduction: Reduction,
    /// Clamp on `log_ratio` before `exp` (OpenRLHF: ±20).
    pub ratio_logclip: f32,
    /// Clamp on the per-token k3 KL (OpenRLHF: ±10).
    pub kl_clip: f32,
    /// Epsilon added to the group std (OpenRLHF: 1e-9).
    pub std_eps: f32,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        GrpoConfig {
            group_size: 8,
            eps_low: 0.2,
            eps_high: 0.2,
            beta: 1e-3,
            estimator: AdvantageEstimator::GroupNorm,
            reduction: Reduction::TokenGlobal,
            ratio_logclip: 20.0,
            kl_clip: 10.0,
            std_eps: 1e-9,
        }
    }
}
