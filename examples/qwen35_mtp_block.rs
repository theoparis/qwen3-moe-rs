//! Eager K=2 MTP speculative-decode correctness gate for Qwen3.6-35B-A3B.
//!
//! Build only:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_mtp_block
//!
//! Run on the 35B host only. `MODE=correct` preserves the 2d correctness-first probe.
//! `MODE=perf` keeps the T=2 verify-written target cache on accept-all and only rolls back +
//! re-forwards the single committed token on reject.

use std::{path::PathBuf, time::Instant};

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    tensor::{DType, Int, Tensor},
};
use qwen3_burn::{
    KVCache, Precision, Qwen3Tokenizer, Qwen3_5HybridCache, Qwen3_5HybridLayerCache,
    Qwen3_5LayerType, Qwen3_5MoeConfig,
};

type B = Cuda;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 128;
const K: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpecMode {
    Correct,
    Perf,
}

impl SpecMode {
    fn parse(value: &str) -> Result<Self, String> {
        if value.eq_ignore_ascii_case("correct") {
            Ok(Self::Correct)
        } else if value.eq_ignore_ascii_case("perf") {
            Ok(Self::Perf)
        } else {
            Err(format!(
                "unsupported MODE={value:?}; expected correct or perf"
            ))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuantMode {
    Bf16,
    Fp8,
}

impl QuantMode {
    fn parse(value: &str) -> Result<Self, String> {
        if value.eq_ignore_ascii_case("bf16") {
            Ok(Self::Bf16)
        } else if value.eq_ignore_ascii_case("fp8") {
            Ok(Self::Fp8)
        } else {
            Err(format!("unsupported QUANT={value:?}; expected bf16 or fp8"))
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }
}

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn positions(start: usize, len: usize, device: &CudaDevice) -> Tensor<B, 2, Int> {
    if len == 1 {
        Tensor::<B, 2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<B, 1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn last_hidden(hidden: &Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, seq, hidden_size] = hidden.dims();
    hidden
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..hidden_size])
}

fn hidden_at(hidden: &Tensor<B, 3>, idx: usize) -> Tensor<B, 3> {
    let [batch, seq, hidden_size] = hidden.dims();
    assert!(
        idx < seq,
        "hidden_at index {idx} out of range for seq_len {seq}"
    );
    hidden
        .clone()
        .slice([0..batch, idx..(idx + 1), 0..hidden_size])
}

fn assert_logits_all_finite(logits: &Tensor<B, 3>, what: &str) -> Result<(), String> {
    let [batch, seq, vocab] = logits.dims();
    let values = logits
        .clone()
        .reshape([batch * seq * vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read {what} logits: {e:?}"))?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{what} logits contain non-finite value at flat index {idx}: {value}"
        ));
    }
    Ok(())
}

fn assert_logits2_all_finite(logits: &Tensor<B, 2>, what: &str) -> Result<(), String> {
    let [batch, vocab] = logits.dims();
    let values = logits
        .clone()
        .reshape([batch * vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read {what} logits: {e:?}"))?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{what} logits contain non-finite value at flat index {idx}: {value}"
        ));
    }
    Ok(())
}

fn argmax_last(logits: &Tensor<B, 3>) -> Result<i64, String> {
    assert_logits_all_finite(logits, "decode")?;
    let [batch, seq, vocab] = logits.dims();
    let next: Tensor<B, 2, Int> = logits
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..vocab])
        .reshape([batch, vocab])
        .argmax(1)
        .cast(DType::I64);
    let ids = next
        .into_data()
        .to_vec::<i64>()
        .map_err(|e| format!("read argmax token: {e:?}"))?;
    ids.first()
        .copied()
        .ok_or_else(|| "argmax returned no token".to_string())
}

fn argmax_each_position(logits: &Tensor<B, 3>) -> Result<Vec<i64>, String> {
    assert_logits_all_finite(logits, "verify")?;
    let [batch, seq, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("verify logits batch must be 1, got {batch}"));
    }
    logits
        .clone()
        .reshape([seq, vocab])
        .argmax(1)
        .cast(DType::I64)
        .into_data()
        .to_vec::<i64>()
        .map_err(|e| format!("read verify argmax tokens: {e:?}"))
}

fn argmax_and_margin(logits: &Tensor<B, 2>) -> Result<(i64, f32), String> {
    assert_logits2_all_finite(logits, "mtp draft")?;
    let [batch, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("MTP draft expects batch 1, got {batch}"));
    }
    let values = logits
        .clone()
        .reshape([batch * vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read MTP draft logits: {e:?}"))?;
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    let mut second = (usize::MAX, f32::NEG_INFINITY);
    for (idx, value) in values.iter().copied().enumerate() {
        if value > best.1 {
            second = best;
            best = (idx, value);
        } else if value > second.1 {
            second = (idx, value);
        }
    }
    Ok((best.0 as i64, best.1 - second.1))
}

fn full_attn_filled(cache: &Qwen3_5HybridCache<B>) -> Result<usize, String> {
    let mut filled = None;
    for (idx, layer) in cache.layers.iter().enumerate() {
        if let Qwen3_5HybridLayerCache::Full(kv) = layer {
            let layer_filled = kv.filled();
            match filled {
                Some(expected) if expected != layer_filled => {
                    return Err(format!(
                        "full-attn KV filled mismatch at layer {idx}: got {layer_filled}, expected {expected}"
                    ));
                }
                None => filled = Some(layer_filled),
                _ => {}
            }
        }
    }
    filled.ok_or_else(|| "model has no full-attention KV layers".to_string())
}

fn assert_mtp_lockstep(
    mtp_cache: &KVCache<B>,
    prompt_len: usize,
    generated_len: usize,
    context: &str,
) -> Result<(), String> {
    let expected = generated_len
        .checked_sub(prompt_len)
        .ok_or_else(|| format!("generated_len {generated_len} is below prompt_len {prompt_len}"))?;
    let got = mtp_cache.filled();
    if got != expected {
        return Err(format!(
            "MTP KV lockstep invariant failed after {context}: mtp_filled={got}, expected_new_commits={expected}, prompt_len={prompt_len}, generated_len={generated_len}"
        ));
    }
    Ok(())
}

fn decode_tokens(tokenizer: &Qwen3Tokenizer, ids: &[i64]) -> Result<String, String> {
    let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    tokenizer.decode(&ids_u32)
}

fn compare_tokens(reference: &[i64], candidate: &[i64]) -> Option<usize> {
    reference
        .iter()
        .zip(candidate.iter())
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference.len() != candidate.len()).then_some(reference.len().min(candidate.len()))
        })
}

fn greedy_decode(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    prec: Precision,
    device: &CudaDevice,
) -> Result<GreedyRun, String> {
    let prompt_len = prompt_ids.len();
    let total = prompt_len + max_new_tokens;
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);

    let pos0 = positions(0, prompt_len, device);
    let mut logits = model.forward_prec(input, pos0, &mut cache, prec);
    assert_logits_all_finite(&logits, "prefill")?;

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    let decode_start = Instant::now();
    for step in 0..max_new_tokens {
        let id = argmax_last(&logits)?;
        generated.push(id);
        new_ids.push(id);

        if step + 1 < max_new_tokens {
            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
            let pos = positions(prompt_len + step, 1, device);
            logits = model.forward_prec(tok, pos, &mut cache, prec);
        }
    }
    let seconds = decode_start.elapsed().as_secs_f64();

    let text = decode_tokens(tokenizer, &generated)?;
    let new_text = decode_tokens(tokenizer, &new_ids)?;
    Ok(GreedyRun {
        new_ids,
        text,
        new_text,
        seconds,
    })
}

struct GreedyRun {
    new_ids: Vec<i64>,
    text: String,
    new_text: String,
    seconds: f64,
}

#[derive(Default)]
struct SpecStats {
    drafted: usize,
    accepted: usize,
    steps: usize,
    committed: usize,
    verify_batches: usize,
    verify_tokens: usize,
}

#[derive(Default)]
struct DraftMarginStats {
    count: usize,
    sum: f64,
    min: f32,
    max: f32,
    accepted_count: usize,
    accepted_sum: f64,
    rejected_count: usize,
    rejected_sum: f64,
}

impl DraftMarginStats {
    fn record(&mut self, margin: f32, accepted: bool) {
        if self.count == 0 {
            self.min = margin;
            self.max = margin;
        } else {
            self.min = self.min.min(margin);
            self.max = self.max.max(margin);
        }
        self.count += 1;
        self.sum += margin as f64;
        if accepted {
            self.accepted_count += 1;
            self.accepted_sum += margin as f64;
        } else {
            self.rejected_count += 1;
            self.rejected_sum += margin as f64;
        }
    }

    fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn accepted_mean(&self) -> f64 {
        if self.accepted_count == 0 {
            0.0
        } else {
            self.accepted_sum / self.accepted_count as f64
        }
    }

    fn rejected_mean(&self) -> f64 {
        if self.rejected_count == 0 {
            0.0
        } else {
            self.rejected_sum / self.rejected_count as f64
        }
    }
}

fn advance_mtp_for_committed(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    mtp_cache: &mut KVCache<B>,
    token_id: i64,
    previous_hidden: Tensor<B, 3>,
    previous_pos: usize,
    prec: Precision,
    device: &CudaDevice,
) -> Result<(i64, f32), String> {
    let tok_next = Tensor::<B, 2, Int>::from_data([[token_id]], device);
    let pos = positions(previous_pos, 1, device);
    let (draft_logits, _mtp_hidden_out) = model.mtp.forward_draft(
        tok_next,
        previous_hidden,
        pos,
        mtp_cache,
        &model.model.embed_tokens,
        &model.lm_head,
        prec,
    );
    argmax_and_margin(&draft_logits)
}

struct SpecRun {
    new_ids: Vec<i64>,
    text: String,
    new_text: String,
    stats: SpecStats,
    margin_stats: DraftMarginStats,
    seconds: f64,
}

fn spec_decode_mtp_block_correct(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    prec: Precision,
    device: &CudaDevice,
) -> Result<SpecRun, String> {
    let prompt_len = prompt_ids.len();
    let draft_limit = K.saturating_sub(1);
    let capacity = prompt_len + max_new_tokens + K.max(1);
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(capacity);
    let mut mtp_cache = model.mtp_new_cache(capacity);

    let pos0 = positions(0, prompt_len, device);
    let (hidden, mut last_logits) = model.forward_hidden_prec(input, pos0, &mut cache, prec);
    assert_logits_all_finite(&last_logits, "prefill")?;
    let mut last_committed_hidden = last_hidden(&hidden);

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    let mut stats = SpecStats::default();
    let mut margin_stats = DraftMarginStats::default();
    assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "prefill")?;

    let decode_start = Instant::now();
    while new_ids.len() < max_new_tokens {
        stats.steps += 1;
        let remaining = max_new_tokens - new_ids.len();
        let use_draft = draft_limit.min(remaining.saturating_sub(1)) > 0;

        let kv_pos = full_attn_filled(&cache)?;
        if kv_pos != generated.len() {
            return Err(format!(
                "KV filled position {kv_pos} does not match generated length {}",
                generated.len()
            ));
        }
        assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "cycle start")?;
        let mtp_pos = mtp_cache.filled();
        let gdn_snap = cache.snapshot_gdn();

        let committed_next = argmax_last(&last_logits)?;
        let prev_pos = generated
            .len()
            .checked_sub(1)
            .ok_or_else(|| "generated sequence is empty".to_string())?;
        let (draft1, draft_margin) = advance_mtp_for_committed(
            model,
            &mut mtp_cache,
            committed_next,
            last_committed_hidden.clone(),
            prev_pos,
            prec,
            device,
        )?;

        let mut accepted = 0usize;
        if use_draft {
            stats.drafted += 1;
            stats.verify_batches += 1;
            stats.verify_tokens += 1;

            let verify_input = Tensor::<B, 2, Int>::from_data([[committed_next]], device);
            let verify_pos = positions(generated.len(), 1, device);
            let verify_logits = model.forward_prec(verify_input, verify_pos, &mut cache, prec);
            let pred = argmax_each_position(&verify_logits)?;
            let target_after_committed_next = pred
                .first()
                .copied()
                .ok_or_else(|| "verify batch returned no predictions".to_string())?;
            accepted = usize::from(draft1 == target_after_committed_next);
            stats.accepted += accepted;
            margin_stats.record(draft_margin, accepted == 1);
        }

        let mut committed = vec![committed_next];
        if accepted == 1 {
            committed.push(draft1);
        }
        if committed.len() > remaining {
            committed.truncate(remaining);
        }

        let mtp_boundary = mtp_pos + committed.len();
        if use_draft && accepted == 0 {
            mtp_cache.rewind(mtp_boundary);
        } else if mtp_cache.filled() > mtp_boundary {
            mtp_cache.rewind(mtp_boundary);
        }

        cache.rewind_kv(kv_pos);
        cache.restore_gdn(gdn_snap);

        for (idx, id) in committed.into_iter().enumerate() {
            if idx > 0 {
                let previous_pos = generated
                    .len()
                    .checked_sub(1)
                    .ok_or_else(|| "generated sequence is empty".to_string())?;
                let _ = advance_mtp_for_committed(
                    model,
                    &mut mtp_cache,
                    id,
                    last_committed_hidden.clone(),
                    previous_pos,
                    prec,
                    device,
                )?;
            }

            let pos = positions(generated.len(), 1, device);
            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
            let (hidden, logits) = model.forward_hidden_prec(tok, pos, &mut cache, prec);
            last_logits = logits;
            last_committed_hidden = last_hidden(&hidden);
            generated.push(id);
            new_ids.push(id);
            stats.committed += 1;
            if new_ids.len() == max_new_tokens {
                break;
            }
        }

        let target_filled = full_attn_filled(&cache)?;
        if target_filled != generated.len() {
            return Err(format!(
                "target KV filled position {target_filled} does not match generated length {} after cycle",
                generated.len()
            ));
        }
        assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "cycle")?;
    }
    let seconds = decode_start.elapsed().as_secs_f64();

    let text = decode_tokens(tokenizer, &generated)?;
    let new_text = decode_tokens(tokenizer, &new_ids)?;
    Ok(SpecRun {
        new_ids,
        text,
        new_text,
        stats,
        margin_stats,
        seconds,
    })
}

fn spec_decode_mtp_block_perf(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    prec: Precision,
    device: &CudaDevice,
) -> Result<SpecRun, String> {
    let prompt_len = prompt_ids.len();
    let draft_limit = K.saturating_sub(1);
    let capacity = prompt_len + max_new_tokens + K.max(1);
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(capacity);
    let mut mtp_cache = model.mtp_new_cache(capacity);

    let pos0 = positions(0, prompt_len, device);
    let (hidden, mut last_logits) = model.forward_hidden_prec(input, pos0, &mut cache, prec);
    assert_logits_all_finite(&last_logits, "prefill")?;
    let mut last_committed_hidden = last_hidden(&hidden);

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    let mut stats = SpecStats::default();
    let mut margin_stats = DraftMarginStats::default();
    assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "prefill")?;

    let decode_start = Instant::now();
    while new_ids.len() < max_new_tokens {
        stats.steps += 1;
        let remaining = max_new_tokens - new_ids.len();
        let use_draft = draft_limit.min(remaining.saturating_sub(1)) > 0;

        let kv_pos = full_attn_filled(&cache)?;
        if kv_pos != generated.len() {
            return Err(format!(
                "KV filled position {kv_pos} does not match generated length {}",
                generated.len()
            ));
        }
        assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "cycle start")?;
        let mtp_pos = mtp_cache.filled();
        let gdn_snap = cache.snapshot_gdn();

        let committed_next = argmax_last(&last_logits)?;
        let prev_pos = generated
            .len()
            .checked_sub(1)
            .ok_or_else(|| "generated sequence is empty".to_string())?;
        let (draft1, draft_margin) = advance_mtp_for_committed(
            model,
            &mut mtp_cache,
            committed_next,
            last_committed_hidden.clone(),
            prev_pos,
            prec,
            device,
        )?;

        if use_draft {
            stats.drafted += 1;
            stats.verify_batches += 1;
            stats.verify_tokens += 2;

            let verify_input = Tensor::<B, 2, Int>::from_data([[committed_next, draft1]], device);
            let verify_pos = positions(generated.len(), 2, device);
            let (verify_hidden, verify_logits) =
                model.forward_hidden_prec(verify_input, verify_pos, &mut cache, prec);
            let pred = argmax_each_position(&verify_logits)?;
            let target_after_committed_next = pred
                .first()
                .copied()
                .ok_or_else(|| "verify batch returned no predictions".to_string())?;
            let accepted = usize::from(draft1 == target_after_committed_next);
            stats.accepted += accepted;
            margin_stats.record(draft_margin, accepted == 1);

            if accepted == 1 {
                let hidden_after_committed_next = hidden_at(&verify_hidden, 0);
                let _ = advance_mtp_for_committed(
                    model,
                    &mut mtp_cache,
                    draft1,
                    hidden_after_committed_next,
                    generated.len(),
                    prec,
                    device,
                )?;

                generated.push(committed_next);
                new_ids.push(committed_next);
                stats.committed += 1;
                generated.push(draft1);
                new_ids.push(draft1);
                stats.committed += 1;

                last_logits = verify_logits;
                last_committed_hidden = last_hidden(&verify_hidden);

                let mtp_boundary = mtp_pos + 2;
                if mtp_cache.filled() > mtp_boundary {
                    mtp_cache.rewind(mtp_boundary);
                }
            } else {
                let mtp_boundary = mtp_pos + 1;
                if mtp_cache.filled() > mtp_boundary {
                    mtp_cache.rewind(mtp_boundary);
                }

                cache.rewind_kv(kv_pos);
                cache.restore_gdn(gdn_snap);

                let pos = positions(generated.len(), 1, device);
                let tok = Tensor::<B, 2, Int>::from_data([[committed_next]], device);
                let (hidden, logits) = model.forward_hidden_prec(tok, pos, &mut cache, prec);
                last_logits = logits;
                last_committed_hidden = last_hidden(&hidden);
                generated.push(committed_next);
                new_ids.push(committed_next);
                stats.committed += 1;
            }
        } else {
            let mtp_boundary = mtp_pos + 1;
            if mtp_cache.filled() > mtp_boundary {
                mtp_cache.rewind(mtp_boundary);
            }

            let pos = positions(generated.len(), 1, device);
            let tok = Tensor::<B, 2, Int>::from_data([[committed_next]], device);
            let (hidden, logits) = model.forward_hidden_prec(tok, pos, &mut cache, prec);
            last_logits = logits;
            last_committed_hidden = last_hidden(&hidden);
            generated.push(committed_next);
            new_ids.push(committed_next);
            stats.committed += 1;
        }

        let target_filled = full_attn_filled(&cache)?;
        if target_filled != generated.len() {
            return Err(format!(
                "target KV filled position {target_filled} does not match generated length {} after cycle",
                generated.len()
            ));
        }
        assert_mtp_lockstep(&mtp_cache, prompt_len, generated.len(), "cycle")?;
    }
    let seconds = decode_start.elapsed().as_secs_f64();

    let text = decode_tokens(tokenizer, &generated)?;
    let new_text = decode_tokens(tokenizer, &new_ids)?;
    Ok(SpecRun {
        new_ids,
        text,
        new_text,
        stats,
        margin_stats,
        seconds,
    })
}

fn spec_decode_mtp_block(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    mode: SpecMode,
    prec: Precision,
    device: &CudaDevice,
) -> Result<SpecRun, String> {
    match mode {
        SpecMode::Correct => spec_decode_mtp_block_correct(
            model,
            tokenizer,
            prompt_ids,
            max_new_tokens,
            prec,
            device,
        ),
        SpecMode::Perf => {
            spec_decode_mtp_block_perf(model, tokenizer, prompt_ids, max_new_tokens, prec, device)
        }
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: qwen35_mtp_block failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let dir = PathBuf::from(env_string("QWEN35_DIR", MODEL_DIR));
    let prompt = env_string("PROMPT", PROMPT);
    let max_new_tokens = env_usize("MAX_NEW_TOKENS", MAX_NEW_TOKENS);
    let mode = SpecMode::parse(&env_string("MODE", "correct"))?;
    let quant_mode = QuantMode::parse(&env_string("QUANT", "bf16"))?;
    let prec = Precision::Bf16;

    let device = CudaDevice::default();
    println!("device: {device:?}");
    println!(
        "mtp block spec config: dir={dir:?} prompt={prompt:?} max_new_tokens={max_new_tokens} k={K} mode={mode:?} quant={}",
        quant_mode.as_str()
    );
    println!(
        "verify framing: MODE=correct preserves the 2d one-token verification + re-forward path; MODE=perf forwards [committed_next, draft1] as one T=2 target batch"
    );
    println!(
        "GDN checkpoint strategy: MODE=perf uses the K=2 simplification: snapshot before verify, keep accept-all GDN/KV, rollback + re-forward only committed_next on reject"
    );
    println!(
        "MTP lockstep invariant: mtp_cache.filled() == generated.len() - prompt_len because this correctness gate does not prefill the MTP block over prompt tokens"
    );

    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    let linear_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == Qwen3_5LayerType::LinearAttention)
        .count();
    let full_layers = cfg.num_hidden_layers - linear_layers;
    println!(
        "config: {} layers ({} GDN, {} full-attn), hidden {}, vocab {}, experts top-{}/{}",
        cfg.num_hidden_layers,
        linear_layers,
        full_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.num_experts_per_tok,
        cfg.num_experts
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm::<B>(&device);

    println!("loading sharded BF16 weights from {dir:?} ...");
    let load_start = Instant::now();
    let report = model
        .load_weights_sharded(&dir)
        .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!(
        "load verify: pass={}, mapped_tensors={}, params={}, time={:.1}s",
        report.pass(),
        report.mapped_tensors,
        report.param_count,
        load_start.elapsed().as_secs_f64()
    );

    if quant_mode == QuantMode::Fp8 {
        #[cfg(feature = "cuda")]
        {
            qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);
            let dense_coverage = qwen3_burn::quant_gate::quantize_dense_fp8(&mut model, &[]);
            let expert_coverage = qwen3_burn::quant_gate::quantize_experts_fp8(&mut model, &[]);
            let mtp_dense = dense_coverage
                .targets
                .iter()
                .filter(|role| role.starts_with("mtp."))
                .count();
            let mtp_experts = expert_coverage
                .targets
                .iter()
                .filter(|role| role.starts_with("mtp."))
                .count();
            println!(
                "mtp fp8 quant summary: dense_targets={mtp_dense} expert_targets={mtp_experts}"
            );
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err("QUANT=fp8 requires the cuda feature".to_string());
        }
    }

    let (prompt_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    if prompt_ids.is_empty() {
        return Err("prompt encoded to zero tokens".to_string());
    }
    println!("prompt token ids: {prompt_ids:?}");

    println!(
        "running {} greedy reference ...",
        quant_mode.as_str().to_uppercase()
    );
    let greedy = greedy_decode(
        &model,
        &tokenizer,
        &prompt_ids,
        max_new_tokens,
        prec,
        &device,
    )?;
    let greedy_tok_s = max_new_tokens as f64 / greedy.seconds.max(1e-9);
    println!("GREEDY new token ids: {:?}", greedy.new_ids);
    println!("GREEDY decoded new text: {:?}", greedy.new_text);
    println!("GREEDY decoded text: {:?}", greedy.text);
    println!(
        "GREEDY decode time: {:.3}s {:.3} tok/s",
        greedy.seconds, greedy_tok_s
    );

    println!("running trained MTP block speculative decode ({mode:?}) ...");
    let spec = spec_decode_mtp_block(
        &model,
        &tokenizer,
        &prompt_ids,
        max_new_tokens,
        mode,
        prec,
        &device,
    )?;
    let spec_tok_s = max_new_tokens as f64 / spec.seconds.max(1e-9);
    println!("SPEC new token ids: {:?}", spec.new_ids);
    println!("SPEC decoded new text: {:?}", spec.new_text);
    println!("SPEC decoded text: {:?}", spec.text);
    println!(
        "SPEC decode time: {:.3}s {:.3} tok/s",
        spec.seconds, spec_tok_s
    );

    let accept_rate = if spec.stats.drafted == 0 {
        0.0
    } else {
        spec.stats.accepted as f64 / spec.stats.drafted as f64
    };
    let avg_committed = if spec.stats.steps == 0 {
        0.0
    } else {
        spec.stats.committed as f64 / spec.stats.steps as f64
    };
    println!(
        "SPEC_STATS drafted={} accepted={} acceptance_rate={:.3} steps={} avg_committed_per_step={:.3} verify_batches={} verify_tokens={}",
        spec.stats.drafted,
        spec.stats.accepted,
        accept_rate,
        spec.stats.steps,
        avg_committed,
        spec.stats.verify_batches,
        spec.stats.verify_tokens,
    );
    println!(
        "DRAFT_MARGIN_STATS count={} mean={:.6} min={:.6} max={:.6} accepted_mean={:.6} rejected_mean={:.6}",
        spec.margin_stats.count,
        spec.margin_stats.mean(),
        if spec.margin_stats.count == 0 {
            0.0
        } else {
            spec.margin_stats.min
        },
        if spec.margin_stats.count == 0 {
            0.0
        } else {
            spec.margin_stats.max
        },
        spec.margin_stats.accepted_mean(),
        spec.margin_stats.rejected_mean(),
    );
    println!("MTP2F_GREEDY tok_s={greedy_tok_s:.3}");
    println!("MTP2F_SPEC tok_s={spec_tok_s:.3}");
    let net_speedup = spec_tok_s / greedy_tok_s.max(1e-9);
    println!("MTP2F_NET speedup={net_speedup:.3}");
    if mode == SpecMode::Perf {
        let gate = if net_speedup >= 1.05 { "GO" } else { "NO_GO" };
        println!("MTP2F_GATE {gate} threshold=1.050");
    }

    if let Some(idx) = compare_tokens(&greedy.new_ids, &spec.new_ids) {
        let greedy_id = greedy.new_ids.get(idx).copied();
        let spec_id = spec.new_ids.get(idx).copied();
        println!(
            "TOKEN_IDENTITY FAIL first_divergence={idx} greedy={greedy_id:?} spec={spec_id:?}"
        );
        return Err("speculative decode diverged from greedy reference".to_string());
    }

    println!(
        "TOKEN_IDENTITY PASS matched={}/{}",
        spec.new_ids.len(),
        greedy.new_ids.len()
    );
    println!("qwen35_mtp_block PASS");
    Ok(())
}
