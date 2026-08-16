//! GRPO loss + group-normalized advantage (OpenRLHF v0.10.4 math).
//!
//! See `docs/GRPO_PLAN.md` §2. Verified against `tests/ref/grpo_expected.json`.

use burn::tensor::{ElementConversion, Tensor};

use super::{AdvantageEstimator, GrpoConfig, Reduction};

/// Detached scalar metrics for logging (read out of the graph; do not backprop through these).
#[derive(Clone, Copy, Debug)]
pub struct GrpoMetrics {
    pub pol_loss: f32,
    pub kl_loss: f32,
    pub total_loss: f32,
    pub mean_ratio: f32,
    pub clip_frac: f32,
}

/// Group-normalized advantage (OpenRLHF `group_norm` / `dr_grpo`).
///
/// * `rewards` — `[P*G]` one scalar reward per completion, laid out prompt-major
///   (`[p0g0, p0g1, …, p1g0, …]`).
/// * `p`, `g`  — number of prompts and group size; `rewards.len() == p*g`.
///
/// `group_norm`: `A_i = (r_i − mean_G) / (std_G + eps)`, std = sample std (ddof = 1) within the
/// prompt's `G` responses. `dr_grpo`: `A_i = r_i − mean_G`. Returns `[P*G]`. No global whitening.
pub fn group_norm_advantage(rewards: Tensor<1>, p: usize, g: usize, cfg: &GrpoConfig) -> Tensor<1> {
    let r = rewards.reshape([p, g]);
    let mean = r.clone().mean_dim(1); // [p, 1]
    let centered = r - mean; // broadcast
    let adv = match cfg.estimator {
        AdvantageEstimator::DrGrpo => centered,
        AdvantageEstimator::GroupNorm => {
            // sample variance (ddof = 1): sum(centered^2) / (g - 1). Guard g==1 (degenerate group)
            // against a div-by-zero that would yield NaN/Inf advantages (review fix).
            let var = centered
                .clone()
                .powf_scalar(2.0)
                .sum_dim(1)
                .div_scalar(g.saturating_sub(1).max(1) as f32); // [p,1]
            let std = var.sqrt().add_scalar(cfg.std_eps);
            centered / std // broadcast [p,g] / [p,1]
        }
    };
    adv.reshape([p * g])
}

/// GRPO loss over a batch of `[N, T]` token-aligned log-probs.
///
/// * `logp_pi`  — policy log-probs WITH grad, `[N, T]`.
/// * `logp_old` — behavior (rollout) log-probs, detached constants, `[N, T]`.
/// * `logp_ref` — frozen-reference log-probs, detached constants, `[N, T]`.
/// * `adv_seq`  — per-sequence advantage `[N]` (broadcast to tokens).
/// * `mask`     — completion mask `[N, T]` (1.0 on response tokens, 0.0 elsewhere).
///
/// Returns the scalar loss tensor (`[1]`, in the autodiff graph) plus detached metrics.
///
/// Math (OpenRLHF):
/// ```text
/// ρ      = exp(clamp(logp_pi − logp_old, −20, 20))
/// L_pol  = − min(ρ·A, clip(ρ, 1−ε_lo, 1+ε_hi)·A)         # min via b − relu(b−a) (grad-safe)
/// δ      = logp_pi − logp_ref
/// KL     = clamp(exp(−δ) − 1 + δ, −10, 10)                # k3, ≥ 0
/// loss   = reduce(L_pol) + β · reduce(KL)                 # token-global mean by default
/// ```
pub fn grpo_loss(
    logp_pi: Tensor<2>,
    logp_old: Tensor<2>,
    logp_ref: Tensor<2>,
    adv_seq: Tensor<1>,
    mask: Tensor<2>,
    cfg: &GrpoConfig,
) -> (Tensor<1>, GrpoMetrics) {
    let [n, _t] = logp_pi.dims();
    let adv = adv_seq.reshape([n, 1]); // [N,1], broadcasts over T

    // ratio = exp(clamp(logp_pi - logp_old, -clip, clip))
    let log_ratio = (logp_pi.clone() - logp_old).clamp(-cfg.ratio_logclip, cfg.ratio_logclip);
    let ratio = log_ratio.exp(); // [N,T]

    let surr1 = ratio.clone() * adv.clone();
    let surr2 = ratio.clone().clamp(1.0 - cfg.eps_low, 1.0 + cfg.eps_high) * adv;
    // min(surr1, surr2) = surr2 - relu(surr2 - surr1)  (avoids mask_where gradient issues on CubeCL)
    let min_surr = surr2.clone() - (surr2.clone() - surr1.clone()).clamp_min(0.0);
    let l_pol = min_surr.neg(); // [N,T]

    // k3 KL: delta = logp_pi - logp_ref ; KL = clamp(exp(-delta) - 1 + delta, -clip, clip)
    let delta = logp_pi - logp_ref;
    let kl = (delta.clone().neg().exp().sub_scalar(1.0) + delta).clamp(-cfg.kl_clip, cfg.kl_clip);

    // reduction
    let (pol_loss_t, kl_loss_t) = match cfg.reduction {
        Reduction::TokenGlobal => {
            // clamp_min(1) guards the degenerate all-masked batch (e.g. every group zero-std):
            // 0-token denom would NaN; instead the masked sums are 0, so the loss is a clean 0.
            let denom = mask.clone().sum().clamp_min(1.0); // [1]
            let pol = (l_pol.clone() * mask.clone()).sum() / denom.clone();
            let kl_r = (kl.clone() * mask.clone()).sum() / denom;
            (pol, kl_r)
        }
        Reduction::SeqMean => {
            // per-sequence mean (1/|o_i|), then mean over sequences (clamp guards empty seqs)
            let tok = mask.clone().sum_dim(1).clamp_min(1.0); // [N,1] tokens per seq
            let pol_seq = (l_pol.clone() * mask.clone()).sum_dim(1) / tok.clone(); // [N,1]
            let kl_seq = (kl.clone() * mask.clone()).sum_dim(1) / tok; // [N,1]
            (pol_seq.mean(), kl_seq.mean())
        }
    };

    let loss = pol_loss_t.clone() + kl_loss_t.clone().mul_scalar(cfg.beta);

    // detached metrics
    let denom_s: f32 = mask
        .clone()
        .sum()
        .into_scalar::<f32>()
        .elem::<f32>()
        .max(1.0);
    let mean_ratio = (ratio * mask.clone())
        .sum()
        .into_scalar::<f32>()
        .elem::<f32>()
        / denom_s;
    // clip_frac: fraction of (masked) tokens where the clipped branch was active (surr2 < surr1)
    let clipped = surr2.lower(surr1).float() * mask; // [N,T]
    let clip_frac = clipped.sum().into_scalar::<f32>().elem::<f32>() / denom_s;

    let metrics = GrpoMetrics {
        pol_loss: pol_loss_t.into_scalar::<f32>().elem(),
        kl_loss: kl_loss_t.into_scalar::<f32>().elem(),
        total_loss: loss.clone().into_scalar::<f32>().elem(),
        mean_ratio,
        clip_frac,
    };
    (loss, metrics)
}
