//! DEVICE-SIDE (GPU) sampling + raw log-prob for the GRPO rollout decode step.
//!
//! The host sampler (`crate::sampling::sample_index` + `grpo::rollout::sample_step`) syncs the WHOLE
//! `[N, V]` last-token logits to the CPU every decode step (`into_data` → `to_vec`), then per row runs
//! a full-vocab softmax + logsumexp + a FULL SORT. At the production Qwen3 vocab (151,936) the
//! decode-cost breakdown (`examples/rollout_decode_bench.rs`) measured that host read + sampling at
//! ~51% of the decode step — and a host sync can't be captured in a CUDA graph (VLLM_PARITY_PLAN §0-A).
//!
//! This module keeps the same per-token contract but does argmax / logsumexp / sampling ON the device
//! with pure Burn tensor ops (`argmax` / `max_dim` / `sum_dim` / `gather` / `Tensor::random` / `exp` /
//! `log`) — NO custom `#[cube]` kernel, NO Fusion bridge — and copies back ONLY `[N]` tokens + `[N]`
//! log-probs, never the `[N, V]` logits.
//!
//! UNFILTERED only (the GRPO default: `top_k == 0 && top_p >= 1.0`). top-k / top-p on-device filtering
//! is an explicit follow-on (it needs a partial-sort / threshold reduction, not just argmax).
//!
//! ## The raw (pre-warp) log-prob
//! GRPO's `old_logprob` MUST be the RAW model log-prob of the sampled token —
//! `logit[token] − logsumexp(RAW logits)` — NOT the temperature-warped one, or the PPO ratio breaks at
//! step 0. So `lse` is computed from the RAW logits and temperature is applied ONLY to the Gumbel-max
//! token selection, never to the log-prob. This matches `grpo::rollout::raw_token_logprob`.
//!
//! ## Sampling without a sort (Gumbel-max)
//! For `temperature > 0` we draw `u ~ Uniform(0,1)` `[N, V]` on-device, form Gumbel noise
//! `g = −ln(−ln u)`, and take `argmax(logits/temp + g)`. The Gumbel-max trick is a categorical draw
//! from `softmax(logits/temp)` — no sort, no host CDF walk.
//!
//! APPROXIMATE, not exact (3-voice review — Codex P1 / Gemini P1). The noise is drawn in **f32**, whose
//! ~6e-8 resolution near `u=1` caps the right tail of `g` at ~16; the expected max of `V=151936` Gumbels
//! is ~12.5 ± 1.3, so the rarest tokens (whose win depends on `g > ~16`) are slightly UNDER-sampled. The
//! `old_logprob` is exact regardless (it reads the true logits), so GRPO stays correct — this only mildly
//! reduces tail exploration. True exactness needs f64 noise (or an exponential-tail draw): a follow-on.
//!
//! ## NOT yet CUDA-graph-capturable (the next lever after this)
//! This removes the per-step `[N,V]` host SYNC (38.9 MB → `[N]`, a real 3.6-4.8× decode speedup), but the
//! driver still copies `[N]` tokens device→host each step for the EOS / finished check — a blocking sync
//! that shatters CUDA-graph capture. Capturing the decode loop additionally needs device-side EOS/finished
//! tracking + a device token buffer (slice-assign into a preallocated `[N, lp+max]`, not the current
//! `Tensor::cat`) + a fixed decode length (no per-step host read / host `all-finished` break). So this is a
//! STEP toward §4, not the unblock.
//!
//! ## BEFORE wiring into the trainer (`group_sample_cached_device` is NOT used by `grpo_step` yet)
//! It is referenced only by the bench + tests. The temperature test checks device-logp == a host recompute
//! from the SAME logits tensor — it does NOT compare the rollout `old_logprob` against the trainer's
//! SEPARATE full-forward recompute, which is what the PPO ratio actually divides. So the load-bearing
//! step-0 `mean_ratio ≈ 1` invariant has never run through this path on CUDA at temperature > 0. Add that
//! end-to-end `grpo_step` gate (same prerequisite the batch-shrink driver names) before flipping the
//! trainer over. Also: the host path it replaces samples in forced-f32 (`to_vec::<f32>`) while this path
//! uses the backend's native float dtype — they would diverge if the rollout is ever run below f32.

use burn::prelude::Backend;
use burn::tensor::{Distribution, Int, Tensor};

/// `logsumexp` over the vocab axis of `[N, V]`, kept as `[N, 1]`. Numerically stable (subtract the
/// per-row max before `exp`). This is the RAW-logits denominator of the pre-warp log-prob.
pub fn logsumexp_dim1<B: Backend>(logits: Tensor<B, 2>) -> Tensor<B, 2> {
    let m = logits.clone().max_dim(1); // [N, 1]
    let shifted = logits - m.clone(); // broadcast subtract
    shifted.exp().sum_dim(1).log() + m // [N, 1]
}

/// Select one token per row ON the device, returned as `[N, 1]` Int.
///
/// * GREEDY (`temperature <= 0`): `argmax(logits)`.
/// * TEMPERATURE (`temperature > 0`): GUMBEL-MAX — `argmax(logits/temp − ln(−ln u))`, `u ~ U(0,1)`
///   drawn on-device. A categorical sample from `softmax(logits/temp)` with NO sort.
///
/// The `u` draw uses the backend RNG. NOTE: Burn's `B::seed` sets a PROCESS-GLOBAL generator and
/// ignores the device, so reproducibility holds only if `B::seed` is called AND the global order of
/// every random op in the process is fixed — it is not per-device or per-rollout.
pub fn device_select_tokens<B: Backend>(logits: &Tensor<B, 2>, temperature: f32) -> Tensor<B, 2, Int> {
    if temperature <= 0.0 {
        return logits.clone().argmax(1); // [N, 1]
    }
    let [n, v] = logits.dims();
    let device = logits.device();
    // u ~ Uniform[0,1) (low inclusive, high exclusive). Clamp strictly inside (0,1) so the double log
    // never hits ln(0)=−inf at the endpoints; the clamp window is far below sampling resolution.
    let u = Tensor::<B, 2>::random([n, v], Distribution::Uniform(0.0, 1.0), &device).clamp(1e-9, 1.0 - 1e-7);
    let gumbel = u.log().neg().log().neg(); // g = −ln(−ln u)
    (logits.clone() / temperature + gumbel).argmax(1) // [N, 1]
}

/// Raw (pre-warp) log-prob `[N]` of the chosen `tokens` `[N, 1]` under `logits` `[N, V]`, given the
/// precomputed RAW `lse` `[N, 1]`: `logp = gather(logits, tokens) − lse`. Pure device gather.
pub fn device_token_logp<B: Backend>(
    logits: &Tensor<B, 2>,
    tokens: &Tensor<B, 2, Int>,
    lse: &Tensor<B, 2>,
) -> Tensor<B, 1> {
    let [n, _v] = logits.dims();
    (logits.clone().gather(1, tokens.clone()) - lse.clone()).reshape([n])
}

/// One self-contained DEVICE sampling step for the UNFILTERED GRPO case.
///
/// Input: RAW logits `[N, V]` ON the device (NOT copied to host). Returns `(tokens[N], logp[N])` as
/// device tensors — the caller copies back only these two `[N]` vectors, never the `[N, V]` logits.
///
/// `logp` is the RAW (pre-warp) log-prob of the sampled token (`logit[token] − logsumexp(RAW logits)`),
/// so it is bit-equivalent to `grpo::rollout::raw_token_logprob` for whatever token selection picked —
/// greedy argmax (`temperature == 0`) or Gumbel-max categorical (`temperature > 0`).
pub fn device_sample_step<B: Backend>(
    logits: Tensor<B, 2>,
    temperature: f32,
) -> (Tensor<B, 1, Int>, Tensor<B, 1>) {
    let [n, _v] = logits.dims();
    let lse = logsumexp_dim1(logits.clone()); // RAW lse [N, 1]
    let tokens = device_select_tokens(&logits, temperature); // [N, 1]
    let logp = device_token_logp(&logits, &tokens, &lse); // [N]
    (tokens.reshape([n]), logp)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type B = NdArray;

    /// Pure-host reference: `logit[token] − logsumexp(raw row)` (mirrors `raw_token_logprob`).
    fn host_raw_logp(row: &[f32], token: usize) -> f32 {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
        row[token] - lse
    }

    #[test]
    fn greedy_picks_argmax_with_raw_logp() {
        let dev = Default::default();
        // two rows; argmax at col 2 then col 0
        let data = [0.0f32, 1.0, 3.0, 2.0, 5.0, 1.0, -1.0, 0.5];
        let logits = Tensor::<B, 1>::from_floats(data.as_slice(), &dev).reshape([2, 4]);
        let (toks, logp) = device_sample_step(logits, 0.0);
        let tv = toks.into_data().to_vec::<i64>().unwrap();
        let lv = logp.into_data().to_vec::<f32>().unwrap();
        assert_eq!(tv, vec![2, 0], "greedy must pick the argmax per row");
        assert!((lv[0] - host_raw_logp(&data[0..4], 2)).abs() < 1e-5);
        assert!((lv[1] - host_raw_logp(&data[4..8], 0)).abs() < 1e-5);
    }

    #[test]
    fn temperature_logp_is_raw_for_sampled_token() {
        let dev = Default::default();
        <B as Backend>::seed(&dev, 42);
        let n = 8usize;
        let v = 16usize;
        let logits = Tensor::<B, 2>::random([n, v], Distribution::Normal(0.0, 1.0), &dev);
        let rows = logits.clone().into_data().to_vec::<f32>().unwrap();
        let (toks, logp) = device_sample_step(logits, 0.8);
        let tv = toks.into_data().to_vec::<i64>().unwrap();
        let lv = logp.into_data().to_vec::<f32>().unwrap();
        for i in 0..n {
            let row = &rows[i * v..(i + 1) * v];
            let want = host_raw_logp(row, tv[i] as usize);
            assert!((lv[i] - want).abs() < 1e-4, "row {i}: device logp {} vs raw {want}", lv[i]);
        }
    }
}
