//! GRPO group rollouts: sample `G` completions per prompt, with PER-SEQUENCE EOS, a completion
//! mask, and the RAW (pre-warp) old log-prob of every sampled token (GRPO fixes a + EOS bug).
//!
//! Two things the existing generation path got wrong for group sampling, fixed here:
//!  * **per-sequence EOS** — `generate_with_cache_eos` stopped the whole batch when sequence 0
//!    hit EOS; a group of G completions must each stop independently. `build_completion_mask`
//!    tracks per-sequence finished state and masks padding after each sequence's first EOS.
//!  * **raw old_logprob** — `logπ_old` is the model's log-softmax of the sampled token taken from
//!    the RAW logits, BEFORE temperature/top-p warping, or the PPO ratio is wrong at step 0.
//!
//! Two generation drivers share the per-step sampling + the finalize logic (`sample_step` /
//! `finalize_rollouts`):
//!  * [`group_sample`] — no-cache, re-forwards the growing sequence each step (`O(T^2)`). Kept as
//!    the parity reference.
//!  * [`group_sample_cached`] — prefills the prompt once then decodes through the KV cache
//!    (`O(T)`). Bit-identical to `group_sample` under greedy sampling (parity-tested); the trainer
//!    uses this one.

use burn::tensor::{Bool, DType, Device, IndexingUpdateOp, Int, Tensor};
use rand::Rng;

use crate::Qwen3ForCausalLM;
use crate::sampling::sample_index;
use crate::sampling_device::{device_select_tokens, device_token_logp, logsumexp_dim1};

/// Result of a group rollout. `N = num_prompts * group_size`, laid out prompt-major.
pub struct Rollouts {
    /// Prompt + completion token ids, right-padded. `[N, prompt_len + gen_len]`.
    pub seq_ids: Tensor<2, Int>,
    /// `1.0` for real completion tokens (up to & incl. each sequence's first EOS), else `0.0`.
    /// `[N, gen_len]`.
    pub completion_mask: Tensor<2>,
    /// Raw model log-prob of each sampled completion token (pre-warp). `[N, gen_len]`.
    pub old_logprobs: Tensor<2>,
    pub prompt_len: usize,
    pub gen_len: usize,
}

/// Sampling configuration for a rollout.
#[derive(Clone, Copy, Debug)]
pub struct RolloutConfig {
    pub group_size: usize,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
}

impl Default for RolloutConfig {
    fn default() -> Self {
        RolloutConfig {
            group_size: 8,
            max_new_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

/// Per-sequence completion mask from per-step sampled tokens.
///
/// `steps[t][s]` is the token sampled at step `t` for sequence `s`. Returns per-sequence response
/// lengths and a row-major `[N * gen_len]` mask: `1.0` up to AND INCLUDING each sequence's first
/// EOS token, `0.0` afterward (padding). Sequences with no EOS are fully unmasked.
pub(crate) fn build_completion_mask(
    steps: &[Vec<i64>],
    eos: &[i64],
    n: usize,
) -> (Vec<usize>, Vec<f32>) {
    let gen_len = steps.len();
    let mut mask = vec![0.0f32; n * gen_len];
    let mut lengths = vec![gen_len; n];
    let mut finished = vec![false; n];
    for (t, step) in steps.iter().enumerate() {
        for s in 0..n {
            if finished[s] {
                continue;
            }
            mask[s * gen_len + t] = 1.0; // this token is part of the response
            if eos.contains(&step[s]) {
                finished[s] = true;
                lengths[s] = t + 1; // inclusive of the EOS token
            }
        }
    }
    (lengths, mask)
}

/// `logπ(token | …)` from RAW logits: `logit[token] − logsumexp(logits)`. Stable.
pub(crate) fn raw_token_logprob(logits_row: &[f32], token: usize) -> f32 {
    if logits_row.is_empty() {
        return 0.0;
    }
    // A sampled/EOS token id should always be < vocab; assert in debug to surface a tokenizer/EOS
    // mismatch instead of silently clamping it to a wrong logit (review fix). The clamp remains a
    // release-mode safety net.
    debug_assert!(
        token < logits_row.len(),
        "token id {token} out of range for vocab {}",
        logits_row.len()
    );
    let m = logits_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + logits_row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    logits_row[token.min(logits_row.len() - 1)] - lse
}

/// Temperature-scaled softmax over a logit row (`temp <= 0` ⇒ one-hot argmax, i.e. greedy).
fn softmax_temp(row: &[f32], temp: f32) -> Vec<f32> {
    if temp <= 0.0 {
        let argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| {
                if x > b.1 { (i, x) } else { b }
            })
            .0;
        let mut p = vec![0.0f32; row.len()];
        p[argmax] = 1.0;
        return p;
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| ((x - m) / temp).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// One generation step over the current last-token logits `raw` (`[N*V]` row-major).
///
/// For each of the `n` sequences: sample the next token (greedy/temp/top-p/top-k) unless the
/// sequence is already `finished` (then emit the pad/EOS token), and record the RAW (pre-warp)
/// log-prob of whatever token was chosen. Returns `(next_tokens[N], logp[N])`. Shared by both
/// generation drivers so the sampling + old-logprob contract is identical.
fn sample_step(
    raw: &[f32],
    n: usize,
    v: usize,
    finished: &[bool],
    cfg: &RolloutConfig,
    _rng: &mut impl Rng,
    eos: &[i64],
) -> (Vec<i64>, Vec<f32>) {
    let mut next = Vec::with_capacity(n);
    let mut logp = Vec::with_capacity(n);
    for sidx in 0..n {
        let row = &raw[sidx * v..(sidx + 1) * v];
        let tok = if finished[sidx] {
            eos.first().copied().unwrap_or(0) // pad finished sequences
        } else {
            let probs = softmax_temp(row, cfg.temperature);
            let r: f32 = rand::random::<f32>();
            sample_index(&probs, cfg.top_k, cfg.top_p, r) as i64
        };
        logp.push(raw_token_logprob(row, tok as usize)); // RAW logp (pre-warp)
        next.push(tok);
    }
    (next, logp)
}

/// Build the [`Rollouts`] from the per-step records (shared by both drivers).
///
/// `steps_*` are step-major (`[t][s]`); we transpose `old_logprobs` to the `[N, gen_len]` row-major
/// layout that aligns with `completion_mask`.
fn finalize_rollouts(
    seq_ids: Tensor<2, Int>,
    steps_tokens: &[Vec<i64>],
    steps_logp: &[Vec<f32>],
    eos: &[i64],
    n: usize,
    prompt_len: usize,
    device: &Device,
) -> Rollouts {
    let gen_len = steps_tokens.len();
    let (_lengths, mask_flat) = build_completion_mask(steps_tokens, eos, n);
    let mut logp_flat = vec![0.0f32; n * gen_len];
    for (t, step) in steps_logp.iter().enumerate() {
        for s in 0..n {
            logp_flat[s * gen_len + t] = step[s];
        }
    }
    Rollouts {
        seq_ids,
        completion_mask: Tensor::<1>::from_floats(mask_flat.as_slice(), device)
            .reshape([n, gen_len]),
        old_logprobs: Tensor::<1>::from_floats(logp_flat.as_slice(), device).reshape([n, gen_len]),
        prompt_len,
        gen_len,
    }
}

/// Sample `cfg.group_size` completions per prompt — NO cache (re-forwards the growing sequence each
/// step, `O(T^2)`). Kept as the parity reference for [`group_sample_cached`].
///
/// Captures, per generated token: the sampled id, the RAW (pre-warp) log-prob, and per-sequence
/// EOS. Precision is the model's training precision (f32 by default for rollouts).
pub fn group_sample(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;

    let mut generated = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut steps_tokens: Vec<Vec<i64>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut steps_logp: Vec<Vec<f32>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut finished = vec![false; n];
    let mut rng = rand::rng();

    for _ in 0..cfg.max_new_tokens {
        let logits = model.forward(generated.clone(), None); // [N, S, V] RAW logits
        let [_, s, v] = logits.dims();
        let last = logits.slice([0..n, (s - 1)..s, 0..v]).reshape([n, v]); // [N, V]
        let raw: Vec<f32> = last.into_data().to_vec::<f32>().unwrap_or_default();

        let (next, logp_step) = sample_step(&raw, n, v, &finished, cfg, &mut rng, eos);
        for sidx in 0..n {
            if !finished[sidx] && eos.contains(&next[sidx]) {
                finished[sidx] = true;
            }
        }
        steps_tokens.push(next.clone());
        steps_logp.push(logp_step);

        let next_t = Tensor::<1, Int>::from_data(next.as_slice(), &device).reshape([n, 1]);
        generated = Tensor::cat(vec![generated, next_t], 1);
        if finished.iter().all(|&f| f) {
            break;
        }
    }

    finalize_rollouts(generated, &steps_tokens, &steps_logp, eos, n, lp, &device)
}

/// Sample `cfg.group_size` completions per prompt — KV-CACHE driver (`O(T)`).
///
/// Prefills the `[N, Lp]` prompt once into the cache, then decodes ONE token per step: feed the
/// just-sampled token at position `Lp + t` through `forward_with_cache`, which reuses the cached
/// prompt K/V. Same outputs + the same raw-pre-warp-logprob / per-sequence-EOS contract as
/// [`group_sample`]; bit-identical under greedy sampling (`tests/grpo_rollout.rs` parity test).
///
/// Assumes uniform prompt length `Lp` (variable-length prompts are a separate task that needs a
/// prompt padding mask + RoPE position offsets).
pub fn group_sample_cached(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;

    let mut generated = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(lp + cfg.max_new_tokens); // Phase 2: static KV, no O(T^2) cat

    // ---- prefill: prompt positions 0..lp -> last-token logits predict completion token 0 ----
    let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]); // [n, lp]
    let logits = model.forward_with_cache(generated.clone(), None, pos0, &mut cache); // [n, lp, v]
    let [_, _, v] = logits.dims();
    let mut last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n, v]

    let mut steps_tokens: Vec<Vec<i64>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut steps_logp: Vec<Vec<f32>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut finished = vec![false; n];
    let mut rng = rand::rng();

    for t in 0..cfg.max_new_tokens {
        let raw: Vec<f32> = last.clone().into_data().to_vec::<f32>().unwrap_or_default();
        let (next, logp_step) = sample_step(&raw, n, v, &finished, cfg, &mut rng, eos);
        for sidx in 0..n {
            if !finished[sidx] && eos.contains(&next[sidx]) {
                finished[sidx] = true;
            }
        }
        steps_tokens.push(next.clone());
        steps_logp.push(logp_step);

        let next_t = Tensor::<1, Int>::from_data(next.as_slice(), &device).reshape([n, 1]);
        generated = Tensor::cat(vec![generated, next_t.clone()], 1);

        // stop after recording this step if everyone is done or the budget is exhausted; no need to
        // forward the token we'd never sample from.
        if finished.iter().all(|&f| f) || t + 1 == cfg.max_new_tokens {
            break;
        }

        // decode: feed completion token `t` at position `lp + t` -> logits for completion token t+1.
        let pos = Tensor::<1, Int>::from_data([(lp + t) as i64].as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]); // [n, 1]
        let lg = model.forward_with_cache(next_t, None, pos, &mut cache); // [n, 1, v]
        last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
    }

    finalize_rollouts(generated, &steps_tokens, &steps_logp, eos, n, lp, &device)
}

/// Sample `cfg.group_size` completions per prompt — KV-CACHE driver with DEVICE-SIDE sampling (§0-A,
/// docs/VLLM_PARITY_PLAN.md — the #1 measured decode lever). Same `O(T)` decode + same [`Rollouts`]
/// contract as [`group_sample_cached`], but the per-step sampler runs ON the device.
///
/// `group_sample_cached`'s `sample_step` syncs the WHOLE `[N, V]` last-token logits to the CPU every
/// step (`into_data` → `to_vec`), then per row does a full-vocab softmax + logsumexp + sort. At the
/// production Qwen3 vocab (151,936) that host read + sampling was ~51% of the decode step
/// (`examples/rollout_decode_bench.rs`), and a host sync can't be captured in a CUDA graph. Here the
/// argmax / logsumexp / Gumbel-max categorical selection runs in pure Burn tensor ops (no custom
/// kernel); only `[N]` candidate tokens + `[N]` raw log-probs cross the host boundary — never `[N, V]`.
///
/// PARITY. Under GREEDY (`temperature == 0`) argmax is deterministic, so this is BIT-IDENTICAL to
/// [`group_sample_cached`] in `seq_ids` + `completion_mask`, with per-token RAW (pre-warp) log-prob
/// equal within fp tolerance (`tests/grpo_rollout.rs::device_sample_matches_host_greedy`). Under
/// `temperature > 0` it draws an i.i.d. Gumbel-max categorical sample from the SAME `softmax(logits/temp)`
/// policy with the correct raw log-prob — a different-but-valid trajectory than the host-RNG path (like
/// the shrink driver), NOT a wrong one. The `old_logprob` is always the RAW pre-warp value
/// (`logit[token] − logsumexp(RAW logits)`): `lse` is taken from the raw logits and temperature warps
/// ONLY the Gumbel-max token selection. UNFILTERED only (the GRPO default `top_k == 0 && top_p >= 1.0`);
/// device top-k/top-p filtering is an explicit follow-on. Uniform prompt length, like
/// [`group_sample_cached`].
pub fn group_sample_cached_device(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    // The device sampler is UNFILTERED-only (the GRPO default). Fail LOUD on a top-k/top-p config —
    // otherwise the filtering would be SILENTLY ignored and we'd sample from the full distribution
    // (wrong completions + wrong old_logprob). Device top-k/top-p is the documented follow-on.
    assert!(
        cfg.top_k == 0 && cfg.top_p >= 1.0,
        "group_sample_cached_device is unfiltered-only (got top_k={}, top_p={}); device top-k/top-p \
         filtering is not yet implemented — use group_sample_cached for filtered sampling.",
        cfg.top_k,
        cfg.top_p,
    );
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;
    let eos0 = eos.first().copied().unwrap_or(0); // pad token for finished rows

    let mut generated = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(lp + cfg.max_new_tokens);

    // ---- prefill: prompt positions 0..lp -> last-token logits predict completion token 0 ----
    let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]);
    let logits = model.forward_with_cache(generated.clone(), None, pos0, &mut cache); // [n, lp, v]
    let [_, _, v] = logits.dims();
    let mut last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n, v] RAW logits, ON device

    let mut steps_tokens: Vec<Vec<i64>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut steps_logp: Vec<Vec<f32>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut finished = vec![false; n];

    for t in 0..cfg.max_new_tokens {
        // ---- DEVICE sampling: RAW logsumexp + candidate token, all on-device (no [N,V] host sync) ----
        let lse = logsumexp_dim1(last.clone()); // [n, 1] from RAW logits (pre-warp denominator)
        let cand_t = device_select_tokens(&last, cfg.temperature); // [n, 1] Int (argmax | Gumbel-max)
        let cand: Vec<i64> = // copy back ONLY [N] candidate tokens
            cand_t.cast(burn::tensor::DType::I64).into_data().to_vec::<i64>().unwrap_or_default();

        // ---- host: per-sequence EOS / finished masking ([N], tiny). Finished rows emit the pad token
        //      (mirror sample_step's finished handling); record the chosen token id. ----
        let next: Vec<i64> = (0..n)
            .map(|s| {
                if finished[s] {
                    eos0
                } else {
                    *cand.get(s).unwrap_or(&eos0)
                }
            })
            .collect();
        for s in 0..n {
            if !finished[s] && eos.contains(&next[s]) {
                finished[s] = true;
            }
        }

        // ---- DEVICE log-prob of the FINAL token, then copy back ONLY [N]. Finished rows gather the pad
        //      token's RAW logp from the live logits — identical to sample_step's raw_token_logprob. ----
        let next_t = Tensor::<1, Int>::from_data(next.as_slice(), &device).reshape([n, 1]);
        let logp_step: Vec<f32> = device_token_logp(&last, &next_t, &lse)
            .into_data()
            .to_vec::<f32>()
            .unwrap_or_default();

        steps_tokens.push(next.clone());
        steps_logp.push(logp_step);
        generated = Tensor::cat(vec![generated, next_t.clone()], 1);

        if finished.iter().all(|&f| f) || t + 1 == cfg.max_new_tokens {
            break;
        }

        // decode: feed completion token `t` at position `lp + t` -> logits for completion token t+1.
        let pos = Tensor::<1, Int>::from_data([(lp + t) as i64].as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]); // [n, 1]
        let lg = model.forward_with_cache(next_t, None, pos, &mut cache); // [n, 1, v]
        last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
    }

    finalize_rollouts(generated, &steps_tokens, &steps_logp, eos, n, lp, &device)
}

/// Sample `cfg.group_size` completions per prompt — FULLY DEVICE-SIDE decode loop, with NO per-step
/// host sync (§4 / §0-A2, docs/VLLM_PARITY_PLAN.md). The static-shape, host-sync-free sibling of
/// [`group_sample_cached_device`], and the prerequisite for CUDA-graph capture of the decode loop.
///
/// [`group_sample_cached_device`] already samples ON the device, but it STILL pays a per-step
/// device→host round-trip: each step it copies the `[N]` candidate tokens to the host, a HOST loop
/// applies EOS/finished masking, uploads the `[N]` next token back, `Tensor::cat`s the growing
/// sequence, and a host `finished.iter().all()` decides the early break. Every one of those is a
/// stream sync that (a) stalls the GPU on the host each step — at decode the kernels are tiny, so the
/// sync latency is a real chunk — and (b) shatters CUDA-graph capture. This driver removes ALL of them:
///
///  * **Preallocated, fixed-shape device buffers (NO `Tensor::cat`).** A `[N, lp+max_new]` Int token
///    buffer (prompt prefilled ONCE, then `slice_assign` the new token at column `lp+t` each step) and
///    `[N, max_new]` f32 log-prob + completion-mask buffers (slice_assign each step). No per-step
///    reallocation, no shape growth — capturable.
///  * **Device-side EOS / finished tracking.** A `[N,1]` Bool `finished` mask lives on the device:
///    `emit = sampled.mask_where(finished, pad)`, `is_eos = OR_e(emit == e)` (pure `equal_elem` +
///    `bool_or`), `finished |= is_eos`. The completion mask is `(!finished_before).float()` written per
///    step — reproducing [`build_completion_mask`]'s "1 up to AND INCLUDING the first EOS, 0 after"
///    semantics EXACTLY, entirely on the device. NO `into_data` in the loop.
///  * **Device log-prob** via [`device_token_logp`] from the RAW `lse` (pre-warp), slice-assigned per
///    step. Already-finished rows gather the pad token's raw logp (masked out anyway), matching
///    `sample_step`.
///  * **Fixed decode length.** ALL `max_new_tokens` steps always run (no host `all-finished` break);
///    finished rows just emit `pad` via the device `mask_where` and are masked out.
///  * **Positions on-device.** A precomputed `[N, max_new]` `lp + t` arange, sliced per step (no
///    per-step host upload).
///
/// ZERO device→host transfers happen inside the driver: the token / mask / logp buffers are returned
/// directly as device tensors (the caller materializes them once, when it reads the result — that is
/// the ONLY sync). Under GREEDY (`temperature == 0`) it is BIT-IDENTICAL to [`group_sample_cached`] in
/// `seq_ids` + `completion_mask`, with per-token RAW log-prob within fp tolerance, over the reference's
/// generated region (this driver never early-breaks, so it then pads the tail; compare the common
/// prefix or a full-length reference — `tests/grpo_rollout.rs::device_loop_matches_device_greedy`).
/// UNFILTERED-only + uniform prompt length, like [`group_sample_cached_device`].
///
/// CUDA-GRAPH STATUS (3-voice review — this loop is the structural prerequisite, but item 4 is NOT a
/// spike on top of it). The loop is static-counter + host-sync-free, but capturing it is BLOCKED on
/// framework-level CubeCL work AND has a bounded payoff:
///  * CubeCL/cubecl-cuda exposes NO cudaGraph capture/replay API (cudarc has the FFI, unused); launch is
///    eager per-op. The `Cuda = Fusion<..>` layer is a LAZY/dynamic op queue, so the captured launch list
///    isn't stable (shifts with autotune/plan-store warmup).
///  * `Tensor::random` (temperature) bakes HOST seeds as frozen kernel immediates → a captured graph
///    replays IDENTICAL noise → degenerate sampling. Only greedy is capture-safe without a device-seed RNG.
///  * No graph-aware allocator (freed per-step intermediates get recycled → replay corruption), and the
///    decode ATTENTION shape grows each step (`filled = lp+t+1`) so it isn't even fixed-shape — needs
///    masked full-`T_max` attention + a device length counter.
///  * Payoff is LOW regardless: decode is BANDWIDTH-bound (the tied-head logits GEMM streams ~0.6 GB/step),
///    which graphs don't touch — they only cut launch latency, which Fusion already largely removes
///    (the measured eager ~1.0× confirms forward+vocab-GEMM dominate). Expect ~1.1-1.4× at batch-1/short
///    context at best. ⇒ The real next decode lever is the bandwidth-bound logits GEMM, not CUDA-graphs.
pub fn group_sample_cached_device_loop(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    assert!(
        cfg.top_k == 0 && cfg.top_p >= 1.0,
        "group_sample_cached_device_loop is unfiltered-only (got top_k={}, top_p={}); device \
         top-k/top-p filtering is not yet implemented — use group_sample_cached for filtered sampling.",
        cfg.top_k,
        cfg.top_p,
    );
    // `eos` must be non-empty: the device EOS test is `emit == eos[..]`, and `eos0 = eos.first()` falls
    // back to token 0 — so an empty `eos` would silently make token 0 a stop token (Codex/Gemini review).
    assert!(
        !eos.is_empty(),
        "group_sample_cached_device_loop needs a non-empty eos set"
    );
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;
    let max_new = cfg.max_new_tokens;
    let total = lp + max_new;
    let eos0 = eos.first().copied().unwrap_or(0); // pad token for finished rows

    let prompt_rep = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(total); // static KV: fixed-shape, the graph-capture prereq

    // ---- preallocated, fixed-shape device buffers (NO Tensor::cat) ----
    // token buffer [N, lp+max_new]: prompt written ONCE, completion slice-assigned at col lp+t per step.
    // Flex default int is I32; token ids / `device_select_tokens` are I64.
    let mut tok_buf = Tensor::<2, Int>::zeros([n, total], &device)
        .cast(DType::I64)
        .slice_assign([0..n, 0..lp], prompt_rep.clone().cast(DType::I64));
    let mut logp_buf = Tensor::<2>::zeros([n, max_new], &device); // RAW pre-warp logp, per step
    let mut mask_buf = Tensor::<2>::zeros([n, max_new], &device); // completion mask, per step

    // device-side EOS state: `finished` [N,1] Bool (starts all-false); constant pad token [N,1] Int.
    let mut finished = Tensor::<2, Int>::zeros([n, 1], &device).equal_elem(1i64); // 0 != 1 ⇒ all false
    let pad = Tensor::<2, Int>::full([n, 1], eos0, &device).cast(DType::I64);
    // decode RoPE positions, col t = lp + t; sliced per step (device slice, no per-step host upload).
    let pos_all = Tensor::<1, Int>::arange(lp as i64..total as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]); // [N, max_new]

    // ---- prefill: prompt positions 0..lp -> last-token logits predict completion token 0 ----
    let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]);
    let logits = model.forward_with_cache(prompt_rep, None, pos0, &mut cache); // [n, lp, v]
    let [_, _, v] = logits.dims();
    let mut last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n, v] RAW logits, ON device

    for t in 0..max_new {
        // ---- DEVICE sampling: RAW lse + candidate token, all on-device (no [N,V] host sync) ----
        let lse = logsumexp_dim1(last.clone()); // [n,1] from RAW logits (pre-warp denominator)
        let sampled = device_select_tokens(&last, cfg.temperature); // [n,1] Int (argmax | Gumbel-max)

        // ---- DEVICE EOS / finished: finished rows emit pad; is_eos = OR over the eos set ----
        let emit = sampled.mask_where(finished.clone(), pad.clone()); // pad where finished, else sampled
        let mut is_eos = emit.clone().equal_elem(eos0); // [n,1] Bool
        for &e in &eos[1..] {
            is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
        }

        // ---- DEVICE RAW (pre-warp) log-prob of emit: gather(logits, emit) - lse ----
        let logp = device_token_logp(&last, &emit, &lse).reshape([n, 1]); // [n,1]

        // ---- slice_assign into the fixed buffers (no cat, no host read) ----
        tok_buf = tok_buf.slice_assign([0..n, (lp + t)..(lp + t + 1)], emit.clone());
        logp_buf = logp_buf.slice_assign([0..n, t..t + 1], logp);
        // mask col t = 1.0 iff the row was NOT finished BEFORE this step — exactly build_completion_mask.
        let active = finished.clone().bool_not().float(); // [n,1] 1.0 active / 0.0 already-finished
        mask_buf = mask_buf.slice_assign([0..n, t..t + 1], active);

        // ---- update device finished state (finished_before OR is_eos) ----
        finished = finished.bool_or(is_eos);

        // ---- decode the NEXT token (skip after the final step); input = emit, a DEVICE tensor ----
        if t + 1 < max_new {
            let pos = pos_all.clone().slice([0..n, t..t + 1]); // [n,1], device slice of lp + t
            let lg = model.forward_with_cache(emit, None, pos, &mut cache); // [n, 1, v]
            last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
        }
    }

    // ZERO device→host transfers above. Return the device buffers directly; the caller's read is the
    // single, final sync (the per-step EOS round-trip is gone — the loop is static + sync-free).
    Rollouts {
        seq_ids: tok_buf,
        completion_mask: mask_buf,
        old_logprobs: logp_buf,
        prompt_len: lp,
        gen_len: max_new,
    }
}

/// Sample `cfg.group_size` completions per prompt — FULLY DEVICE-SIDE decode loop where EVERY per-step
/// op is **fixed-shape and indexed by a DEVICE position counter** (PHASE 2, docs/cudagraph/DESIGN.md
/// §0b P0-A + §7). The capture-READY sibling of [`group_sample_cached_device_loop`].
///
/// [`group_sample_cached_device_loop`] already removed every per-step host SYNC, but it is still NOT
/// CUDA-graph-capturable: every per-step `slice_assign` bakes the HOST loop index `t` as a frozen kernel
/// scalar — the KV write offset (`cache.update`'s `off = filled`), the token write
/// `tok_buf.slice_assign([.., (lp+t)..], …)`, the logp/mask writes `slice_assign([.., t..t+1], …)`, and
/// the growing-prefix read. A graph captured at step `t` would replay into column `t` forever. This
/// driver fixes that — the §7 "device-`pos`-indexed static cache + loop rewrite":
///
///  * **One DEVICE position counter** `pos` (`[1]` Int), starting at `lp` (completion token 0 occupies
///    absolute position `lp`) and incremented ON-DEVICE each step (`pos = pos + 1`, a constant add — never
///    a host int). It is the single source of every per-step index.
///  * **Device-`pos` KV write + masked full-`T_max` attention** ([`KVCache::update_static`] +
///    [`Qwen3Attention::forward_with_cache_static`]): the new token's K/V scatters into the static
///    `[N, T_max, ..]` buffer at the device column `pos` (`select_assign`), and decode attention runs over
///    the WHOLE constant-shape `[N, n_heads, 1, T_max]` K/V with a position mask that `-inf`s columns
///    `idx > pos`. Numerically identical to the loop driver's growing `0..=pos` prefix (the masked
///    columns contribute `exp(-inf) == 0`).
///  * **Device-`pos` token / logp / mask scatters** ([`Tensor::select_assign`]) replace the host-`t`
///    `slice_assign`s: the emitted token goes to `tok_buf[:, pos]`, and the raw logp + completion mask to
///    `logp_buf[:, pos-lp]` / `mask_buf[:, pos-lp]` (`pos - lp` is a constant-offset DEVICE index = `t`).
///  * **Device RoPE position**: the new token's RoPE position IS `pos` (device), and the decode forward
///    input is the emitted token (a device tensor) — never the host `t`.
///  * The EOS/finished tracking is the same pure-device `mask_where` + `equal_elem` + `bool_or` as the
///    loop driver. A FIXED `max_new_tokens` steps always run, with a uniform body (the final step's decode
///    is computed but its logits are unused — one extra forward, kept so every iteration is identical).
///
/// CAPTURE-READINESS (the deliverable). The loop body now contains **zero host read-backs** (no
/// device→host copy) AND **zero host-index-baked ops** (every index is the device tensor `pos` or
/// `pos - lp`; the only
/// host constants are the loop-invariant `lp`, `n`, `v`, and the fixed `0..1` last-token slice). So a
/// graph captured at one step replays correctly at every step. Wrapping this body in
/// `client.capture_arena(...)` is the remaining P-final integration step (it needs the CubeCL capture
/// FFI + a pre-reserved arena + `pos` as a pinned static buffer the host bumps per replay — none of which
/// is Burn-side and none of which P2 builds).
///
/// PARITY. Under GREEDY (`temperature == 0`) it is BIT-IDENTICAL to [`group_sample_cached_device_loop`]
/// in `seq_ids` + `completion_mask`, with per-token raw logp equal within fp tolerance
/// (`tests/grpo_rollout.rs::static_matches_device_loop_greedy`) — the device-pos-indexed path equals the
/// host-`t`-indexed path exactly. UNFILTERED-only + uniform prompt length, like the loop driver.
///
/// ⚠️ HONEST STATUS — a CAPTURE PREREQUISITE + benchmark artifact, NOT a default or a training speedup
/// (3-voice review). It is OPT-IN by call-site (the trainer uses `group_sample_cached`; this is referenced
/// only by tests + `examples/device_static_bench.rs`). Do NOT wire it into `grpo_step`:
///  * **It is a ~1.25× EAGER REGRESSION at short context** — it scans the full `T_max` K/V (+ GQA-expands)
///    every step instead of the `lp+t+1` prefix. Decode is bandwidth-bound, so this added HBM traffic is
///    the cost; it is recovered only by capture+replay, and only as context → `T_max`.
///  * **Captured replay buys ~1.0-1.1× (launch latency only)** on this bandwidth-bound step, so the
///    CAPTURED static decode is NET ≤1.0× (often negative) at short context. The CUDA-graph win for THIS
///    workload is marginal-to-negative; the framework capability (capture FFI + arena) is the deliverable.
///  * **Capture is GREEDY-ONLY** — temperature's `Tensor::random` bakes the host seed into the graph
///    (degenerate on replay) until the device-seed RNG (P3) lands. GRPO rolls out at temperature>0, so a
///    captured greedy decode is NOT the rollout GRPO uses. Even with P3, the net payoff stays ≤1.0× here.
///  * **`lp` is baked** into the captured graph (`pos` starts at `lp`, `rel = pos − lp`), so a graph is
///    valid for ONE prompt length → variable `lp` needs per-bucket graphs (P4) or padding.
///  * Remaining P-final capture blockers P2 does NOT own: the per-step RoPE freq-table `from_floats`
///    host→device upload (`rope.rs`) and the per-step `arange(T_max)` alloc (`attention.rs`) must be
///    precomputed once; and the IO/`pos` buffers must be re-zeroed per replay (else the `Add`-scatter
///    accumulates across replays).
pub fn group_sample_cached_device_static(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    assert!(
        cfg.top_k == 0 && cfg.top_p >= 1.0,
        "group_sample_cached_device_static is unfiltered-only (got top_k={}, top_p={}); device \
         top-k/top-p filtering is not yet implemented — use group_sample_cached for filtered sampling.",
        cfg.top_k,
        cfg.top_p,
    );
    assert!(
        !eos.is_empty(),
        "group_sample_cached_device_static needs a non-empty eos set"
    );
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;
    let max_new = cfg.max_new_tokens;
    let total = lp + max_new;
    let eos0 = eos.first().copied().unwrap_or(0); // pad token for finished rows

    let prompt_rep = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(total); // static KV: fixed-shape, device-pos written

    // ---- preallocated, fixed-shape device buffers (NO Tensor::cat, NO host-`t` slice_assign) ----
    // token buffer [N, lp+max_new]: prompt written ONCE; completion scattered at DEVICE column `pos`.
    // Flex default int is I32; token ids / `device_select_tokens` are I64.
    let mut tok_buf = Tensor::<2, Int>::zeros([n, total], &device)
        .cast(DType::I64)
        .slice_assign([0..n, 0..lp], prompt_rep.clone().cast(DType::I64));
    let mut logp_buf = Tensor::<2>::zeros([n, max_new], &device); // RAW pre-warp logp, at col `pos-lp`
    let mut mask_buf = Tensor::<2>::zeros([n, max_new], &device); // completion mask, at col `pos-lp`

    // device-side EOS state: `finished` [N,1] Bool (starts all-false); constant pad token [N,1] Int.
    let mut finished = Tensor::<2, Int>::zeros([n, 1], &device).equal_elem(1i64); // 0 != 1 ⇒ all false
    let pad = Tensor::<2, Int>::full([n, 1], eos0, &device).cast(DType::I64);

    // ---- the DEVICE position counter (§7). [1] Int, starts at `lp` (= absolute position of completion
    //      token 0), incremented on-device each step. The single source of every per-step index: the KV
    //      write column, the tok_buf column, the RoPE position, and (via `pos - lp`) the logp/mask column.
    let mut pos = Tensor::<1, Int>::full([1], lp as i64, &device);

    // ---- prefill: prompt positions 0..lp -> last-token logits predict completion token 0 (host path;
    //      one-shot, variable-shape — NOT part of the capture-ready decode region) ----
    let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]);
    let logits = model.forward_with_cache(prompt_rep, None, pos0, &mut cache); // [n, lp, v]
    let [_, _, v] = logits.dims();
    let mut last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n, v] RAW logits, ON device

    for _ in 0..max_new {
        // ---- DEVICE sampling: RAW lse + candidate token, all on-device (no [N,V] host sync) ----
        let lse = logsumexp_dim1(last.clone()); // [n,1] from RAW logits (pre-warp denominator)
        let sampled = device_select_tokens(&last, cfg.temperature); // [n,1] Int (argmax | Gumbel-max)

        // ---- DEVICE EOS / finished: finished rows emit pad; is_eos = OR over the eos set ----
        let emit = sampled.mask_where(finished.clone(), pad.clone()); // pad where finished, else sampled
        let mut is_eos = emit.clone().equal_elem(eos0); // [n,1] Bool
        for &e in &eos[1..] {
            is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
        }

        // ---- DEVICE RAW (pre-warp) log-prob of emit: gather(logits, emit) - lse ----
        let logp = device_token_logp(&last, &emit, &lse).reshape([n, 1]); // [n,1]

        // ---- DEVICE-pos scatters into the fixed buffers (no host index, no host read) ----
        // tok_buf column = `pos` (= lp+t); logp/mask column = `pos - lp` (= t). Both indices are DEVICE
        // tensors; `lp` is a loop-invariant constant (the prompt length), not the changing loop index.
        let rel = pos.clone().sub_scalar(lp as i64); // [1] Int = t (= pos - lp), on-device
        tok_buf = tok_buf.select_assign(1, pos.clone(), emit.clone(), IndexingUpdateOp::Add);
        logp_buf = logp_buf.select_assign(1, rel.clone(), logp, IndexingUpdateOp::Add);
        // mask col = 1.0 iff the row was NOT finished BEFORE this step — exactly build_completion_mask.
        let active = finished.clone().bool_not().float(); // [n,1] 1.0 active / 0.0 already-finished
        mask_buf = mask_buf.select_assign(1, rel, active, IndexingUpdateOp::Add);

        // ---- update device finished state (finished_before OR is_eos) ----
        finished = finished.bool_or(is_eos);

        // ---- decode the NEXT token through the STATIC masked-attention path at device `pos`. Always run
        //      (uniform body): the final step's `last` is unused. Input = emit (device); offset = pos. ----
        let lg = model.forward_with_cache_static(emit, pos.clone(), &mut cache); // [n, 1, v]
        last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);

        // ---- advance the DEVICE counter (a device add of constant 1; never a host int) ----
        pos = pos.add_scalar(1i64);
    }

    // ZERO device→host transfers above; the loop body is fixed-shape + device-pos-indexed (capture-ready).
    Rollouts {
        seq_ids: tok_buf,
        completion_mask: mask_buf,
        old_logprobs: logp_buf,
        prompt_len: lp,
        gen_len: max_new,
    }
}

/// Compact (drop finished rows from the live decode batch) once the finished FRACTION of the live
/// batch reaches this threshold. Lazy/threshold — NOT eager per-step — because each compaction is a
/// physical `Tensor::select` copy of every layer's KV buffer (`[n, T_max, kv_heads, head_dim]`), so
/// compacting on every single EOS can cost more bandwidth than the forwards it saves (Codex/Gemini
/// review, docs/VLLM_PARITY_PLAN.md Phase 3). 0.5 = "compact when ≥ half the live batch is finished".
const SHRINK_THRESHOLD: f32 = 0.5;

/// Sample `cfg.group_size` completions per prompt — KV-CACHE driver WITH DYNAMIC BATCH-SHRINK.
///
/// Same contract as [`group_sample_cached`], but it stops spending FLOPs/bandwidth on finished
/// sequences. In `group_sample_cached` a sequence that hit EOS is masked out of the loss yet still
/// forwarded every remaining step; with high length variance the late-decode batch is mostly finished
/// rows. Here we forward only the ACTIVE rows (a 2.03× rollout-decode speedup on the GB10 at high
/// length variance; the win grows with the finished fraction).
///
/// PARITY ENVELOPE (3-voice review). The GRPO requirement is **valid policy samples + the correct raw
/// `old_logprob` for each sampled token** — NOT bit-identity with `group_sample_cached`. Both hold: every
/// KEPT row is forwarded with a bit-identical cache history + the same RoPE position, so its per-row logits
/// are unchanged (the model is strictly per-row: `linear3` flattens batch, RmsNorm/RoPE/SiLU are per-token,
/// decode attention is per-row), and `sample_step` records the raw logp of whatever token it sampled.
/// * **Greedy** (`temperature == 0`): bit-IDENTICAL `seq_ids` + `completion_mask` + real-token logp vs
///   `group_sample_cached` (pinned by `shrink_matches_unshrunk_greedy`).
/// * **Temperature > 0** (the real rollout): each kept row draws one i.i.d. uniform through its OWN
///   bit-identical `softmax_temp(row)`, so it samples from the CORRECT policy distribution with a correct
///   logp — a different-but-valid trajectory than the unshrunk path, NOT a wrong one. (Exact-token
///   temperature parity is neither claimed nor needed; the two paths even share one ThreadRng from
///   different offsets.) Post-EOS PADDING logp of compacted-out rows is a placeholder (`mask == 0`, never
///   in the loss). NOTE: validated on NdArray/CPU only; an on-GB10 temperature parity gate (kept-row logp
///   match + step-0 PPO ratio ≈ 1) is the prerequisite before wiring this into `grpo_step` (sm_121 has a
///   history of batch-dependent kernel quirks). Contracts: no-grad rollout only (so `select_rows` frees
///   the dropped rows), and uniform prompt length (`lp + t` positions, like `group_sample_cached`).
///
/// LAZY/THRESHOLD compaction: once the finished fraction of the live batch reaches [`SHRINK_THRESHOLD`]
/// we `Tensor::select(0, active_local)` the running token tensor AND every layer's K/V cache down to
/// `[n_active, ..]`, keep an `active → original` index map, and decode only that subset. Each step we
/// sample the active rows and scatter the sampled token + raw log-prob back to their ORIGINAL row in
/// the full `[N, ..]` records (so `seq_ids` / `completion_mask` / `old_logprobs` stay `[N, gen_len]`).
pub fn group_sample_cached_shrink(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    let n = p * cfg.group_size;
    let eos0 = eos.first().copied().unwrap_or(0);

    let mut generated = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, cfg.group_size, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(lp + cfg.max_new_tokens);

    // ---- prefill: ALL N rows (everyone is active at prefill) ----
    let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]);
    let logits = model.forward_with_cache(generated.clone(), None, pos0, &mut cache); // [n, lp, v]
    let [_, _, v] = logits.dims();
    let mut last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n_active=N, v]

    let mut steps_tokens: Vec<Vec<i64>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut steps_logp: Vec<Vec<f32>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut finished = vec![false; n]; // by ORIGINAL row index
    let mut active_idx: Vec<usize> = (0..n).collect(); // ORIGINAL index of each live cache row, in order
    let mut rng = rand::rng();

    for t in 0..cfg.max_new_tokens {
        let n_active = active_idx.len();
        let raw: Vec<f32> = last.clone().into_data().to_vec::<f32>().unwrap_or_default(); // [n_active * v]

        // sample over the LIVE batch only; `finished` for live rows (finished-but-not-yet-compacted
        // rows are still forwarded, exactly like the no-shrink path, so their pad logp also matches).
        let finished_active: Vec<bool> = active_idx.iter().map(|&o| finished[o]).collect();
        let (next_active, logp_active) =
            sample_step(&raw, n_active, v, &finished_active, cfg, &mut rng, eos);

        // scatter the live results back to the ORIGINAL [N] step records; compacted-out rows are all
        // finished -> emit eos padding (`mask == 0`), with a placeholder pad logp that the loss ignores.
        let mut next_full = vec![eos0; n];
        let mut logp_full = vec![0.0f32; n];
        for (local, &orig) in active_idx.iter().enumerate() {
            let tok = next_active[local];
            if !finished[orig] && eos.contains(&tok) {
                finished[orig] = true;
            }
            next_full[orig] = tok;
            logp_full[orig] = logp_active[local];
        }
        steps_tokens.push(next_full.clone());
        steps_logp.push(logp_full);

        let next_t_full =
            Tensor::<1, Int>::from_data(next_full.as_slice(), &device).reshape([n, 1]);
        generated = Tensor::cat(vec![generated, next_t_full], 1);

        if finished.iter().all(|&f| f) || t + 1 == cfg.max_new_tokens {
            break;
        }

        // ---- lazy/threshold compaction: drop finished rows from the live batch ----
        let finished_live = active_idx.iter().filter(|&&o| finished[o]).count();
        if finished_live > 0 && (finished_live as f32 / n_active as f32) >= SHRINK_THRESHOLD {
            let keep_local: Vec<i64> = (0..n_active)
                .filter(|&i| !finished[active_idx[i]])
                .map(|i| i as i64)
                .collect();
            let keep_t = Tensor::<1, Int>::from_data(keep_local.as_slice(), &device);
            cache.select_rows(&keep_t); // gather kept rows of every layer's K/V buffer
            active_idx = keep_local.iter().map(|&i| active_idx[i as usize]).collect();
        }

        // decode: feed completion token `t` (active rows only) at position `lp + t` (uniform across
        // the live batch — every kept row advanced one token every step) -> logits for token t+1.
        let n_fwd = active_idx.len();
        let next_tokens: Vec<i64> = active_idx.iter().map(|&o| next_full[o]).collect();
        let next_t =
            Tensor::<1, Int>::from_data(next_tokens.as_slice(), &device).reshape([n_fwd, 1]);
        let pos = Tensor::<1, Int>::from_data([(lp + t) as i64].as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n_fwd, 1]); // [n_fwd, 1]
        let lg = model.forward_with_cache(next_t, None, pos, &mut cache); // [n_fwd, 1, v]
        last = lg.slice([0..n_fwd, 0..1, 0..v]).reshape([n_fwd, v]);
    }

    finalize_rollouts(generated, &steps_tokens, &steps_logp, eos, n, lp, &device)
}

/// RoPE positions for left-padded rows: `cumsum(mask) - 1` along each row, pad (`mask == false`)
/// clamped to 0. Real tokens are numbered from 0 regardless of how much left-pad precedes them, so
/// a left-padded prompt is position-equivalent to the unpadded prompt (see the left-pad invariance
/// gate). `mask_rows[s]` is the boolean attention mask of sequence `s` (`true` = real token).
fn positions_from_mask(mask_rows: &[Vec<bool>]) -> Vec<i64> {
    let mut pos = Vec::with_capacity(mask_rows.iter().map(|r| r.len()).sum());
    for row in mask_rows {
        let mut c = 0i64;
        for &real in row {
            if real {
                pos.push(c);
                c += 1;
            } else {
                pos.push(0); // pad position (masked out anyway)
            }
        }
    }
    pos
}

/// LEFT-PAD-AWARE group rollout (no cache). For variable-length prompts left-padded to a common
/// `lp`: `prompt_lens[p]` is the real (unpadded) length of prompt `p`, so the first `lp -
/// prompt_lens[p]` columns are pad. Each step forwards the full sequence through
/// `forward_with_positions` with the attention mask (pad masked) and `cumsum(mask)-1` RoPE
/// positions, so generation is invariant to the left-pad (parity-tested against the unpadded run).
///
/// Same `Rollouts` contract as [`group_sample`]; the completion region is uniform (starts at column
/// `lp` for every sequence), so the trainer's completion alignment is unchanged. Uniform-prompt
/// batches should use the faster [`group_sample_cached`]; this path exists for ragged prompts.
pub fn group_sample_padded(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    prompt_lens: &[usize],
    cfg: &RolloutConfig,
    eos: &[i64],
) -> Rollouts {
    let device = prompt_ids.device();
    let [p, lp] = prompt_ids.dims();
    assert_eq!(
        prompt_lens.len(),
        p,
        "prompt_lens must have one entry per prompt"
    );
    let g = cfg.group_size;
    let n = p * g;

    let mut generated = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, g, 1])
        .reshape([n, lp]);

    // Per-sequence attention mask rows (prompt left-pad), repeated G times. Grows by one `true` per
    // generated token (completions are always real).
    let mut mask_rows: Vec<Vec<bool>> = Vec::with_capacity(n);
    for pi in 0..p {
        let pad = lp - prompt_lens[pi];
        let mut row = vec![false; pad];
        row.extend(std::iter::repeat(true).take(prompt_lens[pi]));
        for _ in 0..g {
            mask_rows.push(row.clone());
        }
    }

    let mut steps_tokens: Vec<Vec<i64>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut steps_logp: Vec<Vec<f32>> = Vec::with_capacity(cfg.max_new_tokens);
    let mut finished = vec![false; n];
    let mut rng = rand::rng();

    for _ in 0..cfg.max_new_tokens {
        let cur_len = mask_rows[0].len();
        let mask_flat: Vec<bool> = mask_rows.iter().flatten().copied().collect();
        let mask_t =
            Tensor::<1, Bool>::from_data(mask_flat.as_slice(), &device).reshape([n, cur_len]);
        let pos_t =
            Tensor::<1, Int>::from_data(positions_from_mask(&mask_rows).as_slice(), &device)
                .reshape([n, cur_len]);

        let logits = model.forward_with_positions(generated.clone(), Some(mask_t), pos_t); // [n, cur_len, v]
        let [_, _, v] = logits.dims();
        let last = logits
            .slice([0..n, (cur_len - 1)..cur_len, 0..v])
            .reshape([n, v]);
        let raw: Vec<f32> = last.into_data().to_vec::<f32>().unwrap_or_default();

        let (next, logp_step) = sample_step(&raw, n, v, &finished, cfg, &mut rng, eos);
        for sidx in 0..n {
            if !finished[sidx] && eos.contains(&next[sidx]) {
                finished[sidx] = true;
            }
            mask_rows[sidx].push(true); // the new completion token is a real token
        }
        steps_tokens.push(next.clone());
        steps_logp.push(logp_step);

        let next_t = Tensor::<1, Int>::from_data(next.as_slice(), &device).reshape([n, 1]);
        generated = Tensor::cat(vec![generated, next_t], 1);
        if finished.iter().all(|&f| f) {
            break;
        }
    }

    finalize_rollouts(generated, &steps_tokens, &steps_logp, eos, n, lp, &device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completion_mask_per_sequence_eos() {
        // 3 sequences, eos = {9}, 5 steps
        let steps = vec![
            vec![5, 1, 9], // t0: seq2 hits EOS
            vec![6, 2, 1], // t1: seq2 already finished
            vec![9, 3, 1], // t2: seq0 hits EOS
            vec![2, 4, 1], // t3
            vec![3, 5, 1], // t4
        ];
        let (lengths, mask) = build_completion_mask(&steps, &[9], 3);
        assert_eq!(lengths, vec![3, 5, 1], "per-sequence response lengths");
        // seq0: 1,1,1,0,0 ; seq1: 1,1,1,1,1 ; seq2: 1,0,0,0,0
        assert_eq!(&mask[0..5], &[1.0, 1.0, 1.0, 0.0, 0.0]);
        assert_eq!(&mask[5..10], &[1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&mask[10..15], &[1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn raw_logprob_matches_logsoftmax() {
        // logits [0, ln2, ln3]; Z = 1+2+3 = 6; logp(token2) = ln3 - ln6 = ln(0.5)
        let row = [0.0f32, 2.0f32.ln(), 3.0f32.ln()];
        let lp = raw_token_logprob(&row, 2);
        assert!((lp - 0.5f32.ln()).abs() < 1e-6, "got {lp}");
        // probabilities exp(logp) sum to 1
        let total: f32 = (0..3).map(|i| raw_token_logprob(&row, i).exp()).sum();
        assert!((total - 1.0).abs() < 1e-6, "probs sum {total}");
    }

    #[test]
    fn softmax_temp_greedy_is_onehot() {
        let p = softmax_temp(&[1.0, 3.0, 2.0], 0.0);
        assert_eq!(p, vec![0.0, 1.0, 0.0]);
    }
}
