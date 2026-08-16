//! Integration test for GRPO group rollouts: runs `group_sample` through a real (tiny,
//! random-init) Qwen3 model on the NdArray CPU backend and checks the rollout mechanics —
//! shapes, mask domain, log-prob finiteness — without needing a GPU or pretrained weights.
//!
//! Run: `cargo test --test grpo_rollout`

use burn::tensor::{DType, Device, Distribution, Int, Tensor};
use qwen3_burn::Qwen3Config;
use qwen3_burn::device_sample_step;
use qwen3_burn::grpo::{
    RolloutConfig, group_sample, group_sample_cached, group_sample_cached_device,
    group_sample_cached_device_loop, group_sample_cached_device_static, group_sample_cached_shrink,
};

fn int_ids<const D: usize>(t: Tensor<D, Int>) -> Vec<i64> {
    t.cast(DType::I64).into_data().to_vec::<i64>().unwrap()
}

#[test]
fn group_sample_shapes_and_logprobs() {
    let dev = Device::flex();

    // tiny model so the test is fast and light (random weights — we test mechanics, not quality)
    let cfg = Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    let p = 2usize; // prompts
    let lp = 3usize; // prompt length
    let prompt =
        Tensor::<1, Int>::from_data([1i64, 2, 3, 4, 5, 6].as_slice(), &dev).reshape([p, lp]);

    let g = 3usize;
    let rc = RolloutConfig {
        group_size: g,
        max_new_tokens: 4,
        temperature: 1.0,
        top_p: 0.9,
        top_k: 0,
    };
    let eos = [7i64];

    let roll = group_sample(&model, prompt, &rc, &eos);
    let n = p * g;

    // shapes
    assert_eq!(roll.prompt_len, lp);
    assert!(roll.gen_len >= 1 && roll.gen_len <= rc.max_new_tokens);
    assert_eq!(roll.seq_ids.dims(), [n, lp + roll.gen_len]);
    assert_eq!(roll.completion_mask.dims(), [n, roll.gen_len]);
    assert_eq!(roll.old_logprobs.dims(), [n, roll.gen_len]);

    // mask is strictly 0/1, and the first generated token is always real (mask col 0 == 1)
    let mask = roll.completion_mask.into_data().to_vec::<f32>().unwrap();
    for &m in &mask {
        assert!(m == 0.0 || m == 1.0, "mask must be 0/1, got {m}");
    }
    for s in 0..n {
        assert_eq!(
            mask[s * roll.gen_len],
            1.0,
            "first generated token must be unmasked"
        );
    }

    // old log-probs are finite and <= 0 (they are log-probabilities)
    let lp_vals = roll.old_logprobs.into_data().to_vec::<f32>().unwrap();
    for &v in &lp_vals {
        assert!(v.is_finite(), "logprob must be finite, got {v}");
        assert!(v <= 1e-4, "logprob must be <= 0 (+eps), got {v}");
    }

    // generated ids are within vocab
    let ids = int_ids(roll.seq_ids);
    for &t in &ids {
        assert!((0..32).contains(&t), "token {t} out of vocab");
    }

    println!(
        "rollout OK — N={n} gen_len={} seq_cols={}",
        roll.gen_len,
        lp + roll.gen_len
    );
}

/// The KV-cache rollout must be IDENTICAL to the no-cache rollout under greedy sampling
/// (`temperature = 0` ⇒ deterministic argmax, no RNG). This is the load-bearing parity gate for
/// the O(T) cache path: same tokens, same mask, and (within fp tolerance) the same raw old-logprob.
#[test]
fn cached_matches_uncached_greedy() {
    let dev = Device::flex();

    let cfg = Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    let p = 2usize;
    let lp = 3usize;
    let prompt =
        Tensor::<1, Int>::from_data([1i64, 2, 3, 4, 5, 6].as_slice(), &dev).reshape([p, lp]);

    // greedy (temperature = 0) so both drivers are deterministic and must match bit-for-bit
    let rc = RolloutConfig {
        group_size: 3,
        max_new_tokens: 6,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos = [7i64];

    let a = group_sample(&model, prompt.clone(), &rc, &eos);
    let b = group_sample_cached(&model, prompt, &rc, &eos);

    assert_eq!(
        a.gen_len, b.gen_len,
        "gen_len: no-cache {} vs cache {}",
        a.gen_len, b.gen_len
    );
    assert_eq!(a.seq_ids.dims(), b.seq_ids.dims(), "seq_ids shape");

    let ai = int_ids(a.seq_ids);
    let bi = int_ids(b.seq_ids);
    assert_eq!(ai, bi, "cache and no-cache produced different token ids");

    let am = a.completion_mask.into_data().to_vec::<f32>().unwrap();
    let bm = b.completion_mask.into_data().to_vec::<f32>().unwrap();
    assert_eq!(am, bm, "completion masks differ");

    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    assert_eq!(al.len(), bl.len(), "old_logprob lengths differ");
    for (i, (x, y)) in al.iter().zip(bl.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-4,
            "old_logprob[{i}] differs: no-cache {x} vs cache {y}"
        );
    }

    println!(
        "cache==no-cache parity OK (greedy) — gen_len={} N={}",
        a.gen_len,
        p * rc.group_size
    );
}

/// Dynamic batch-shrink (`group_sample_cached_shrink`) must be IDENTICAL to `group_sample_cached`
/// (the parity reference) under greedy sampling, on a batch with HIGH length variance so the shrink
/// path actually compacts. Distinct prompts + a multi-token EOS set make the greedy completions stop
/// at staggered lengths; the 50%-finished threshold then fires the cache compaction mid-decode.
///
/// What MUST be bit-identical: `seq_ids` (incl. the EOS padding) and `completion_mask`. What must be
/// identical within fp tolerance: the raw old-logprob of every REAL completion token (`mask == 1`).
/// The post-EOS PADDING logprobs of compacted-out rows are NOT reproduced (those rows are no longer
/// forwarded, by design — that is the speedup), and they are `mask == 0` so the GRPO loss never sees
/// them; we report that drift for transparency but do not gate on it.
#[test]
fn shrink_matches_unshrunk_greedy() {
    let dev = Device::flex();
    dev.seed(7); // deterministic random init -> reproducible length spread

    let cfg = Qwen3Config::new()
        .with_vocab_size(40)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    // 8 DISTINCT prompts (distinct greedy continuations -> distinct EOS lengths), group_size 2 -> N=16.
    let (p, lp, g) = (8usize, 4usize, 2usize);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| 1 + (i * 7 + 3) % 37).collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &dev).reshape([p, lp]);

    // greedy + a BROAD EOS set (upper ~60% of the vocab) so a clear majority of the (distinct) greedy
    // continuations terminate within a few steps while the rest run long — staggered lengths that push
    // the live finished-fraction past the 50% shrink threshold and fire the cache compaction.
    let rc = RolloutConfig {
        group_size: g,
        max_new_tokens: 24,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos: Vec<i64> = (16..40).collect();
    let n = p * g;

    let a = group_sample_cached(&model, prompt.clone(), &rc, &eos);
    let b = group_sample_cached_shrink(&model, prompt, &rc, &eos);

    assert_eq!(
        a.gen_len, b.gen_len,
        "gen_len: unshrunk {} vs shrink {}",
        a.gen_len, b.gen_len
    );
    assert_eq!(a.seq_ids.dims(), b.seq_ids.dims(), "seq_ids shape");

    // (1) seq_ids bit-identical (real tokens AND the EOS padding both paths emit).
    let ai = int_ids(a.seq_ids);
    let bi = int_ids(b.seq_ids);
    assert_eq!(ai, bi, "shrink and unshrink produced different token ids");

    // (2) completion mask bit-identical.
    let am = a.completion_mask.into_data().to_vec::<f32>().unwrap();
    let bm = b.completion_mask.into_data().to_vec::<f32>().unwrap();
    assert_eq!(am, bm, "completion masks differ");

    // confirm the test actually exercises shrink: lengths must vary, and > half the batch must finish
    // before the end (so the 50% threshold compacts at least once). Lengths = sum of mask per row.
    let gen_len = a.gen_len;
    let lengths: Vec<usize> = (0..n)
        .map(|s| (0..gen_len).filter(|&t| am[s * gen_len + t] == 1.0).count())
        .collect();
    let (minl, maxl) = (
        *lengths.iter().min().unwrap(),
        *lengths.iter().max().unwrap(),
    );
    let finished_before_end = lengths.iter().filter(|&&l| l < gen_len).count();
    assert!(
        minl < maxl,
        "no length variance ({minl}..{maxl}) — shrink would never fire; retune the test"
    );
    assert!(
        finished_before_end * 2 > n,
        "only {finished_before_end}/{n} finished before the end — shrink threshold may not fire"
    );

    // (3) raw old-logprob: identical for every REAL completion token (mask == 1). Padding (mask == 0)
    // logprobs of compacted-out rows are intentionally not reproduced; report their drift separately.
    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    assert_eq!(al.len(), bl.len(), "old_logprob lengths differ");
    let (mut masked_max, mut pad_max) = (0.0f32, 0.0f32);
    for i in 0..al.len() {
        let e = (al[i] - bl[i]).abs();
        if am[i] == 1.0 {
            masked_max = masked_max.max(e);
        } else {
            pad_max = pad_max.max(e);
        }
    }
    println!(
        "shrink parity OK (greedy) — N={n} gen_len={gen_len} lengths={minl}..{maxl} \
         finished_before_end={finished_before_end} | masked logp max-err={masked_max:.2e} \
         (pad-region max-err={pad_max:.2e}, unused by loss)"
    );
    assert!(
        masked_max < 1e-4,
        "real-token logprob max-err {masked_max} exceeds 1e-4 (shrink corrupted a kept row)"
    );
}

/// DEVICE-SIDE sampling parity (§0-A, docs/VLLM_PARITY_PLAN.md). `group_sample_cached_device` runs the
/// per-step argmax/logsumexp/log-prob ON the device (pure Burn ops) and copies back only `[N]` tokens +
/// `[N]` log-probs. Under GREEDY (`temperature == 0`) argmax is deterministic, so it MUST be bit-identical
/// to the host `group_sample_cached` reference: same `seq_ids` (incl. EOS padding), same `completion_mask`,
/// and per-token RAW log-prob equal within fp tolerance. A broad EOS set under greedy makes the distinct
/// prompts terminate at staggered lengths, exercising the finished-row pad path on BOTH drivers.
#[test]
fn device_sample_matches_host_greedy() {
    let dev = Device::flex();
    dev.seed(7); // deterministic init -> reproducible length spread

    let cfg = Qwen3Config::new()
        .with_vocab_size(40)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    let (p, lp, g) = (8usize, 4usize, 2usize);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| 1 + (i * 7 + 3) % 37).collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &dev).reshape([p, lp]);

    // greedy (temperature == 0) so host argmax and device argmax must agree bit-for-bit. Broad EOS set
    // -> staggered terminations -> exercises the finished-row padding on both paths.
    let rc = RolloutConfig {
        group_size: g,
        max_new_tokens: 24,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos: Vec<i64> = (16..40).collect();
    let n = p * g;

    let a = group_sample_cached(&model, prompt.clone(), &rc, &eos); // host reference
    let b = group_sample_cached_device(&model, prompt, &rc, &eos); // device path

    assert_eq!(
        a.gen_len, b.gen_len,
        "gen_len: host {} vs device {}",
        a.gen_len, b.gen_len
    );
    assert_eq!(a.seq_ids.dims(), b.seq_ids.dims(), "seq_ids shape");

    // (1) seq_ids bit-identical (real tokens AND the EOS padding both paths emit).
    let ai = int_ids(a.seq_ids);
    let bi = int_ids(b.seq_ids);
    assert_eq!(
        ai, bi,
        "device and host produced different token ids (greedy must match)"
    );

    // (2) completion mask bit-identical.
    let am = a.completion_mask.into_data().to_vec::<f32>().unwrap();
    let bm = b.completion_mask.into_data().to_vec::<f32>().unwrap();
    assert_eq!(am, bm, "completion masks differ");

    // assert the scenario actually has length variance (so the finished-row pad path is exercised).
    let gen_len = a.gen_len;
    let lengths: Vec<usize> = (0..n)
        .map(|s| (0..gen_len).filter(|&t| am[s * gen_len + t] == 1.0).count())
        .collect();
    let (minl, maxl) = (
        *lengths.iter().min().unwrap(),
        *lengths.iter().max().unwrap(),
    );
    assert!(
        minl < maxl,
        "no length variance ({minl}..{maxl}) — finished-row pad path not exercised"
    );

    // (3) per-token RAW log-prob equal within fp tolerance over EVERY position (incl. padding: the device
    // path gathers the pad token's raw logp from the live logits, like sample_step's raw_token_logprob).
    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    assert_eq!(al.len(), bl.len(), "old_logprob lengths differ");
    let mut maxe = 0.0f32;
    for (x, y) in al.iter().zip(bl.iter()) {
        maxe = maxe.max((x - y).abs());
    }
    println!(
        "device-sample greedy parity OK — N={n} gen_len={gen_len} lengths={minl}..{maxl} | \
         raw logp max-err={maxe:.2e} (only [N] tokens+logp crossed the host boundary)"
    );
    assert!(
        maxe < 1e-4,
        "device vs host raw logp max-err {maxe} exceeds 1e-4"
    );
}

/// TEMPERATURE-logp correctness for the device sampler. Under `temperature > 0` Gumbel-max picks a
/// categorical sample (we do NOT assert WHICH token — that's a valid different draw), but the returned
/// log-prob MUST be the correct RAW (pre-warp) log-prob of whatever token it picked:
/// `logit[token] − logsumexp(RAW logits)`. We seed the backend RNG, sample once, and check each row's
/// device logp against a host recompute from the same raw logits.
#[test]
fn device_sample_temperature_logp_is_raw() {
    let dev = Device::flex();
    dev.seed(123);
    let (n, v) = (12usize, 64usize);
    let logits = Tensor::<2>::random([n, v], Distribution::Normal(0.0, 1.0), &dev);
    let rows = logits.clone().into_data().to_vec::<f32>().unwrap();

    let (toks, logp) = device_sample_step(logits, 0.7);
    let tv = int_ids(toks);
    let lv = logp.into_data().to_vec::<f32>().unwrap();

    // host reference: logit[token] − logsumexp(raw row).
    let host_raw_logp = |row: &[f32], token: usize| -> f32 {
        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
        row[token] - lse
    };
    let mut maxe = 0.0f32;
    for i in 0..n {
        let row = &rows[i * v..(i + 1) * v];
        let tok = tv[i] as usize;
        assert!(tok < v, "row {i}: sampled token {tok} out of vocab {v}");
        maxe = maxe.max((lv[i] - host_raw_logp(row, tok)).abs());
    }
    println!("device-sample temperature logp-correctness OK — N={n} raw logp max-err={maxe:.2e}");
    assert!(
        maxe < 1e-4,
        "device temperature logp max-err {maxe} exceeds 1e-4"
    );
}

/// Canonical equivalence gate (Phase 0, docs/VLLM_PARITY_PLAN.md) — KV-cache vs no-cache parity at
/// LONGER context with explicit per-token logprob max/mean error. The outside voices (Codex/Gemini)
/// flagged that a short-prompt "mean ratio ~1" check hides token-local + position/mask bugs that only
/// surface at length. This asserts bit-exact token ids AND bounded per-token logprob drift on a longer
/// prompt + longer generation — the load-bearing safety net before any rollout speed work lands.
#[test]
fn canonical_gate_long_context_parity() {
    let dev = Device::flex();
    let cfg = Qwen3Config::new()
        .with_vocab_size(64)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(3)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    // longer context than cached_matches_uncached_greedy (3+6): 12-token prompt, up to 24 generated.
    let (p, lp) = (2usize, 12usize);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| 1 + (i % 50)).collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &dev).reshape([p, lp]);
    let rc = RolloutConfig {
        group_size: 2,
        max_new_tokens: 24,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos = [63i64]; // unlikely id ⇒ generate the full length, exercising long positions

    let a = group_sample(&model, prompt.clone(), &rc, &eos);
    let b = group_sample_cached(&model, prompt, &rc, &eos);

    assert_eq!(
        a.gen_len, b.gen_len,
        "long-context gen_len: no-cache {} vs cache {}",
        a.gen_len, b.gen_len
    );
    let ai = int_ids(a.seq_ids);
    let bi = int_ids(b.seq_ids);
    assert_eq!(
        ai, bi,
        "long-context: cache vs no-cache token ids diverge (position/mask bug)"
    );

    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let (mut maxe, mut sume) = (0.0f32, 0.0f32);
    for (x, y) in al.iter().zip(bl.iter()) {
        let e = (x - y).abs();
        maxe = maxe.max(e);
        sume += e;
    }
    let meane = sume / al.len().max(1) as f32;
    println!(
        "canonical gate (long-ctx gen_len={}): per-token logprob max-err={maxe:.2e} mean-err={meane:.2e}",
        a.gen_len
    );
    assert!(
        maxe < 1e-4,
        "long-context per-token logprob max-err {maxe} exceeds 1e-4 (token-local corruption)"
    );
}

/// FULLY DEVICE-SIDE decode-loop parity (§4 / §0-A2, docs/VLLM_PARITY_PLAN.md).
/// `group_sample_cached_device_loop` removes the LAST per-step device→host sync: EOS/finished tracking,
/// the next-token buffer, and the completion mask are ALL on the device (`mask_where` / `equal_elem` /
/// `bool_or` / `slice_assign`), the decode runs a FIXED `max_new_tokens` steps (no host all-finished
/// break), and ZERO `into_data`/`to_vec` happens inside the driver. Under GREEDY (`temperature == 0`)
/// it must reproduce the host `group_sample_cached` reference EXACTLY over the reference's generated
/// region: same `seq_ids` (incl. EOS padding), same `completion_mask`, raw log-prob within fp tol. A
/// broad-but-incomplete EOS set under greedy makes the distinct prompts terminate at STAGGERED lengths
/// while ≥1 row runs the full length, exercising both the device finished-row pad path and the fixed
/// (no early-break) decode length.
#[test]
fn device_loop_matches_device_greedy() {
    let dev = Device::flex();
    dev.seed(7); // deterministic init -> reproducible length spread

    let cfg = Qwen3Config::new()
        .with_vocab_size(40)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    let (p, lp, g) = (8usize, 4usize, 2usize);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| 1 + (i * 7 + 3) % 37).collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &dev).reshape([p, lp]);

    // greedy (temp == 0). A mid-vocab EOS set: some greedy continuations stop early (staggered), at
    // least one never hits EOS within max_new (so the reference does NOT early-break and we can compare
    // the full tensor against the always-full-length loop driver).
    let max_new = 24usize;
    let rc = RolloutConfig {
        group_size: g,
        max_new_tokens: max_new,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos: Vec<i64> = vec![20, 24, 28, 31, 35];
    let n = p * g;

    let a = group_sample_cached(&model, prompt.clone(), &rc, &eos); // host reference (may early-break)
    let b = group_sample_cached_device_loop(&model, prompt, &rc, &eos); // fully device-side loop

    // the loop NEVER early-breaks: it always emits exactly max_new completion columns.
    assert_eq!(
        b.gen_len, max_new,
        "device-loop must run the full fixed decode length"
    );

    let g0 = a.gen_len; // reference generated length (<= max_new); compare the common prefix
    assert!(g0 >= 1);

    let ai = int_ids(a.seq_ids); // [n, lp+g0]
    let bi = int_ids(b.seq_ids.clone()); // [n, lp+max_new]
    let am = a.completion_mask.into_data().to_vec::<f32>().unwrap(); // [n, g0]
    let bm = b
        .completion_mask
        .clone()
        .into_data()
        .to_vec::<f32>()
        .unwrap(); // [n, max_new]
    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap(); // [n, g0]
    let bl = b.old_logprobs.clone().into_data().to_vec::<f32>().unwrap(); // [n, max_new]

    // (1) seq_ids bit-identical over [0, lp+g0) (real tokens AND the EOS padding both paths emit).
    for s in 0..n {
        for c in 0..(lp + g0) {
            assert_eq!(
                ai[s * (lp + g0) + c],
                bi[s * (lp + max_new) + c],
                "seq_ids differ at row {s} col {c} (device-loop vs host, greedy must match)"
            );
        }
    }
    // (2) completion_mask bit-identical over [0, g0); device mask_where/EOS == build_completion_mask.
    let mut maxe = 0.0f32;
    for s in 0..n {
        for t in 0..g0 {
            assert_eq!(
                am[s * g0 + t],
                bm[s * max_new + t],
                "completion mask differs at row {s} step {t}"
            );
            maxe = maxe.max((al[s * g0 + t] - bl[s * max_new + t]).abs());
        }
        // (3) the loop's TAIL beyond the reference length is pure padding: mask == 0 (the device finished
        //     state held), so it never enters the loss — exactly what an unshrunk full-length run produces.
        for t in g0..max_new {
            assert_eq!(
                bm[s * max_new + t],
                0.0,
                "device-loop tail must be masked padding (row {s} step {t})"
            );
        }
    }

    // assert the scenario actually has staggered length variance (finished-row pad path exercised on
    // both drivers) and reaches the full length somewhere.
    let lengths: Vec<usize> = (0..n)
        .map(|s| (0..g0).filter(|&t| am[s * g0 + t] == 1.0).count())
        .collect();
    let (minl, maxl) = (
        *lengths.iter().min().unwrap(),
        *lengths.iter().max().unwrap(),
    );
    assert!(
        minl < maxl,
        "no length variance ({minl}..{maxl}) — finished-row pad path not exercised; retune eos"
    );

    // (4) STRUCTURAL no-per-step-host-read check: the driver source must contain ZERO into_data/to_vec.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/grpo/rollout.rs"))
        .unwrap();
    let body = {
        let start = src
            .find("pub fn group_sample_cached_device_loop")
            .expect("driver fn present");
        let after = &src[start..];
        // up to the next top-level `pub fn ` (the following driver) — the whole function body.
        let end = after[1..]
            .find("\npub fn ")
            .map(|i| i + 1)
            .unwrap_or(after.len());
        &after[..end]
    };
    let n_host_reads = body.matches("into_data").count() + body.matches("to_vec").count();
    assert_eq!(
        n_host_reads, 0,
        "device-loop driver must have NO per-step host read (found {n_host_reads} into_data/to_vec)"
    );

    println!(
        "device-LOOP greedy parity OK — N={n} ref_gen_len={g0} loop_gen_len={max_new} lengths={minl}..{maxl} \
         | raw logp max-err={maxe:.2e} | per-step host reads in driver: {n_host_reads} (ONE final sync, the caller's read)"
    );
    assert!(
        maxe < 1e-4,
        "device-loop vs host raw logp max-err {maxe} exceeds 1e-4"
    );
}

/// PHASE 2 — the device-`pos`-indexed static cache + fixed-shape decode loop
/// (docs/cudagraph/DESIGN.md §0b P0-A + §7). `group_sample_cached_device_static` replaces every
/// host-`t`-baked per-step op of `group_sample_cached_device_loop` with DEVICE-position-counter ops:
///  * the KV write scatters into the static `[N, T_max, ..]` buffer at a `[1]` Int device index (`pos`),
///    and decode attention runs over the FULL constant-shape `[N, n_heads, 1, T_max]` K/V with a
///    position mask that `-inf`s columns `idx > pos` (the masked full-`T_max` attention — the
///    load-bearing correctness piece);
///  * the token / logp / completion-mask writes are `select_assign(1, pos|pos-lp, …)` device scatters
///    (no host-`t` `slice_assign`);
///  * the RoPE position + decode input come from device tensors, never the host loop index.
///
/// The proof: under GREEDY (`temperature == 0`, deterministic argmax) the DEVICE-pos-indexed path must
/// equal the HOST-`t`-indexed `group_sample_cached_device_loop` EXACTLY — bit-identical `seq_ids` +
/// `completion_mask`, per-token raw logp within fp tol. Both run the SAME fixed `max_new` steps with no
/// early break, so the whole `[N, ..]` tensors compare directly. Bit-exact parity here also proves the
/// masked full-`T_max` attention is numerically identical to the loop driver's growing `0..=pos` prefix
/// (the only difference between the two drivers is the cache/attention indexing). Validated on NdArray
/// CPU (deterministic sequential reduction ⇒ true bit-identity, not just argmax-stable).
#[test]
fn static_matches_device_loop_greedy() {
    let dev = Device::flex();
    dev.seed(7); // deterministic init -> reproducible length spread

    let cfg = Qwen3Config::new()
        .with_vocab_size(40)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm(&dev);

    let (p, lp, g) = (8usize, 4usize, 2usize);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| 1 + (i * 7 + 3) % 37).collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &dev).reshape([p, lp]);

    // greedy (temp == 0). Staggered EOS (some rows stop early, at least one runs the full length) so the
    // device finished-row pad path AND the masked attention at multiple positions are both exercised.
    let max_new = 24usize;
    let rc = RolloutConfig {
        group_size: g,
        max_new_tokens: max_new,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    let eos: Vec<i64> = vec![20, 24, 28, 31, 35];
    let n = p * g;

    let a = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos); // host-`t`-indexed loop
    let b = group_sample_cached_device_static(&model, prompt, &rc, &eos); // device-`pos`-indexed static

    // both run the FULL fixed decode length (no early break) -> identical shapes, compared directly.
    assert_eq!(
        a.gen_len, max_new,
        "loop driver must run the full fixed decode length"
    );
    assert_eq!(
        b.gen_len, max_new,
        "static driver must run the full fixed decode length"
    );
    assert_eq!(
        a.seq_ids.dims(),
        b.seq_ids.dims(),
        "seq_ids shape: loop {:?} vs static {:?}",
        a.seq_ids.dims(),
        b.seq_ids.dims()
    );

    // (1) seq_ids BIT-IDENTICAL (real tokens AND the device-emitted EOS padding).
    let ai = int_ids(a.seq_ids);
    let bi = int_ids(b.seq_ids);
    assert_eq!(
        ai, bi,
        "device-pos static vs host-`t` loop produced different token ids (greedy must match)"
    );

    // (2) completion_mask BIT-IDENTICAL.
    let am = a.completion_mask.into_data().to_vec::<f32>().unwrap();
    let bm = b.completion_mask.into_data().to_vec::<f32>().unwrap();
    assert_eq!(
        am, bm,
        "completion masks differ (device-pos mask scatter != host-`t` slice_assign)"
    );

    // confirm the scenario has staggered length variance (finished-row pad path exercised) and that at
    // least one row reaches the full length (so the mask boundary `pos` is exercised across all columns).
    let lengths: Vec<usize> = (0..n)
        .map(|s| (0..max_new).filter(|&t| am[s * max_new + t] == 1.0).count())
        .collect();
    let (minl, maxl) = (
        *lengths.iter().min().unwrap(),
        *lengths.iter().max().unwrap(),
    );
    assert!(
        minl < maxl,
        "no length variance ({minl}..{maxl}) — retune eos"
    );
    assert_eq!(
        maxl, max_new,
        "no row reached full length — the mask boundary isn't exercised at all positions"
    );

    // (3) per-token RAW log-prob equal within fp tol (on CPU this is bit-exact: the masked full-`T_max`
    //     softmax sums the same valid columns + exact-zero masked tails as the growing-prefix softmax).
    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    assert_eq!(al.len(), bl.len(), "old_logprob lengths differ");
    let mut maxe = 0.0f32;
    for (x, y) in al.iter().zip(bl.iter()) {
        maxe = maxe.max((x - y).abs());
    }

    // (4) STRUCTURAL capture-readiness check on the static driver body: ZERO per-step host read
    //     (into_data/to_vec) AND ZERO host-`t`-baked per-step `slice_assign` into the running buffers
    //     (those are now `select_assign` at the device index). The one-shot prompt prefill
    //     `slice_assign([0..n, 0..lp], …)` is loop-INVARIANT (constant range, outside the step loop) and
    //     is allowed; the banned forms are the per-step `slice_assign([.., (lp + t)..])` /
    //     `slice_assign([.., t..t + 1])` the loop driver used.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/grpo/rollout.rs"))
        .unwrap();
    let body = {
        let start = src
            .find("pub fn group_sample_cached_device_static")
            .expect("static driver present");
        let after = &src[start..];
        let end = after[1..]
            .find("\npub fn ")
            .map(|i| i + 1)
            .unwrap_or(after.len());
        &after[..end]
    };
    let n_host_reads = body.matches("into_data").count() + body.matches("to_vec").count();
    assert_eq!(
        n_host_reads, 0,
        "static driver must have NO per-step host read (found {n_host_reads})"
    );
    let n_per_step_slice_assign = body.matches("(lp + t)").count()
        + body.matches("t..t + 1").count()
        + body.matches("t..t+1").count();
    assert_eq!(
        n_per_step_slice_assign, 0,
        "static driver must NOT use host-`t`-baked per-step slice_assign (found {n_per_step_slice_assign}) — \
         use device-pos select_assign"
    );
    // every per-step buffer write must be a device-index select_assign (KV scatter lives in cache.rs).
    assert!(
        body.contains("select_assign"),
        "static driver must scatter per-step writes by device pos"
    );

    println!(
        "STATIC (device-pos) == LOOP (host-`t`) greedy parity OK — N={n} gen_len={max_new} lengths={minl}..{maxl} \
         | raw logp max-err={maxe:.2e} | masked full-T_max attn == growing-prefix attn (bit-exact on CPU) \
         | capture-ready: per-step host reads={n_host_reads}, host-`t` slice_assigns={n_per_step_slice_assign}"
    );
    assert!(
        maxe < 1e-4,
        "device-pos static vs host-`t` loop raw logp max-err {maxe} exceeds 1e-4"
    );
}
