//! Memory-safe per-token log-probabilities.
//!
//! GRPO needs `logπ(token_t | context)` for the *sampled/target* token at each position. The
//! naive `log_softmax(logits)` materializes a `[B, T, V]` tensor (V = 151,936 for Qwen3) and,
//! under autodiff, caches it for the backward pass — an instant OOM on a 128 GB GB10
//! (flagged by both external reviewers). Instead we compute the selected-token log-prob as
//!
//! ```text
//! logp[b,t] = logit[b,t, target]  −  logsumexp(logits[b,t, :])
//! ```
//!
//! which never materializes the `V`-wide softmax: the gather pulls one logit per position and
//! `logsumexp` reduces over `V` to a `[B, T, 1]`. (Adapted from burn-ppo's `log_prob_categorical`,
//! made memory-safe and extended from `[B, V]` to the sequence dimension `[B, T, V]`.)

use burn::tensor::{Int, Tensor};

/// `logsumexp` over the last axis of a 3-D tensor, keeping that axis as size 1.
/// Numerically stable: subtract the per-row max before exp.
fn logsumexp_last(x: Tensor<3>) -> Tensor<3> {
    let m = x.clone().max_dim(2); // [b, t, 1]
    let shifted = x - m.clone(); // broadcast subtract
    shifted.exp().sum_dim(2).log() + m // [b, t, 1]
}

/// Per-token log-prob of `targets` under `logits`.
///
/// * `logits`  — `[B, T, V]` raw (unnormalized) model logits.
/// * `targets` — `[B, T]` token ids whose log-prob we want.
///
/// Returns `[B, T]`. Autodiff-safe and memory-safe (no `[B,T,V]` softmax materialized).
///
/// NOTE on alignment (the off-by-one contract): `logits[:, t, :]` are the model's predictions
/// for position `t`. When scoring a *generated* token at position `t`, pass the logits from
/// position `t-1` (i.e. the caller shifts), or pass already-aligned `(logits, targets)`. This
/// function does not shift — it scores `targets[b,t]` against `logits[b,t,:]` exactly.
pub fn token_logprobs(logits: Tensor<3>, targets: Tensor<2, Int>) -> Tensor<2> {
    let [b, t, _v] = logits.dims();
    let lse = logsumexp_last(logits.clone()); // [b, t, 1]
    let idx = targets.unsqueeze_dim::<3>(2); // [b, t, 1]
    let gathered = logits.gather(2, idx); // [b, t, 1]
    (gathered - lse).reshape([b, t]) // [b, t]
}
