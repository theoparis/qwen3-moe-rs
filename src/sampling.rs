//! Token sampling: top-k + top-p (nucleus) filtering then categorical sampling.
//!
//! Extracted into one pure, unit-tested function so the generation path
//! (`decoder::sample_from_probs`) and GRPO rollouts share identical, CORRECT sampling. The
//! previous inline sampler in `decoder.rs` silently ignored `top_p` (took `_top_p`), so nucleus
//! sampling was a no-op — fatal for GRPO rollouts, which need real top-p/temperature diversity to
//! produce varied completions within a group.

/// Sample an index from a probability distribution `probs` over the vocabulary, applying top-k
/// then top-p (nucleus) filtering. `r` is a uniform sample in `[0, 1)`; the result is
/// deterministic given `r` (so callers control RNG and tests are reproducible).
///
/// * `top_k = 0` (or `>= len`) disables top-k.
/// * `top_p <= 0.0` or `>= 1.0` disables top-p.
///
/// Filtering order matches HF/vLLM: top-k truncates to the k highest-prob tokens, then top-p
/// keeps the smallest high-prob prefix whose cumulative (renormalized) mass reaches `top_p`.
pub fn sample_index(probs: &[f32], top_k: usize, top_p: f32, r: f32) -> usize {
    if probs.is_empty() {
        return 0;
    }
    // TODO(perf — RESUME, docs/VLLM_PARITY_PLAN.md §0-A): this `sort` over the FULL vocab runs on every
    // sampled token even when NO filtering is requested — which is the GRPO default (`top_k == 0` &&
    // `top_p >= 1.0`). The decode-cost measurement (examples/rollout_decode_bench.rs) put host-side
    // sampling at ~43% of the decode step, dominated by this O(V log V) sort over 151936 entries × N.
    // CHEAP WIN: when unfiltered, skip the sort and do a single O(V) inverse-CDF draw here. BIGGER WIN
    // (lever A): move argmax/logsumexp/sampling onto the GPU and copy back only [N] tokens + [N] logp
    // (Gumbel-max for temperature/top-p avoids the sort entirely). That also unblocks CUDA-graph capture.
    // sort indices by descending probability
    let mut idx: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // top-k: keep the k highest
    let k = if top_k > 0 && top_k < idx.len() { top_k } else { idx.len() };
    idx.truncate(k);

    // top-p (nucleus): keep the smallest prefix whose cumulative (renormalized) mass >= top_p
    if top_p > 0.0 && top_p < 1.0 {
        let total: f32 = idx.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            let mut cum = 0.0f32;
            let mut cutoff = idx.len();
            for (i, (_, p)) in idx.iter().enumerate() {
                cum += p / total;
                if cum >= top_p {
                    cutoff = i + 1; // inclusive of the token that crossed the threshold
                    break;
                }
            }
            idx.truncate(cutoff.max(1));
        }
    }

    // renormalize the surviving set and sample via the inverse-CDF with `r`
    let sum: f32 = idx.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return idx.first().map(|(i, _)| *i).unwrap_or(0); // degenerate -> argmax
    }
    let mut cum = 0.0f32;
    for (i, p) in &idx {
        cum += p / sum;
        if r < cum {
            return *i;
        }
    }
    idx.last().map(|(i, _)| *i).unwrap_or(0) // r rounding -> last surviving token
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_one_is_argmax() {
        let p = [0.1, 0.7, 0.2];
        for r in [0.0, 0.3, 0.99] {
            assert_eq!(sample_index(&p, 1, 1.0, r), 1, "top_k=1 must always pick argmax");
        }
    }

    #[test]
    fn top_p_excludes_tail() {
        // sorted desc: 0.6, 0.3, 0.08, 0.02. top_p=0.9 -> cum 0.6 then 0.9 (>=0.9 at idx 1)
        // => nucleus = {0, 1}; tokens 2 and 3 must NEVER be sampled for any r.
        let p = [0.6, 0.3, 0.08, 0.02];
        for i in 0..=100 {
            let r = i as f32 / 100.0;
            let s = sample_index(&p, 0, 0.9, r);
            assert!(s == 0 || s == 1, "top_p=0.9 leaked tail token {s} at r={r}");
        }
        // boundary behavior within the nucleus
        assert_eq!(sample_index(&p, 0, 0.9, 0.0), 0);
        assert_eq!(sample_index(&p, 0, 0.9, 0.99), 1);
    }

    #[test]
    fn top_p_disabled_can_reach_tail() {
        let p = [0.6, 0.3, 0.08, 0.02];
        // top_p=1.0 disabled: with r just under 1.0 we reach the last token
        assert_eq!(sample_index(&p, 0, 1.0, 0.999), 3);
    }

    #[test]
    fn top_k_and_top_p_compose() {
        // top_k=2 keeps {0.6,0.3}; top_p then can't add tail back
        let p = [0.6, 0.3, 0.08, 0.02];
        for i in 0..=50 {
            let r = i as f32 / 50.0;
            let s = sample_index(&p, 2, 0.95, r);
            assert!(s == 0 || s == 1, "top_k=2 leaked token {s}");
        }
    }

    #[test]
    fn empty_is_zero() {
        assert_eq!(sample_index(&[], 0, 1.0, 0.5), 0);
    }
}
