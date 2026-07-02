//! N-gram speculative-decode correctness probe for Qwen3.6-35B-A3B.
//!
//! Build only:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_mtp_ngram
//!
//! Run on the 35B host only. This is a correctness probe for verify-batch + KV/GDN rollback, not a
//! performance path: accepted tokens are intentionally re-forwarded after rollback.

use std::{path::PathBuf, time::Instant};

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    tensor::{DType, Int, Tensor},
};
use qwen3_burn::{
    Precision, Qwen3_5HybridCache, Qwen3_5HybridLayerCache, Qwen3_5LayerType, Qwen3_5MoeConfig,
    Qwen3Tokenizer,
};

type B = Cuda;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 32;
const K: usize = 3;
const MAX_NGRAM: usize = 8;

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

fn ngram_draft(generated: &[i64], max_drafts: usize) -> Vec<i64> {
    if max_drafts == 0 || generated.len() < 2 {
        return Vec::new();
    }

    let max_suffix = MAX_NGRAM.min(generated.len() - 1);
    for suffix_len in (1..=max_suffix).rev() {
        let suffix_start = generated.len() - suffix_len;
        let suffix = &generated[suffix_start..];
        for start in (0..suffix_start).rev() {
            if start + suffix_len >= generated.len() {
                continue;
            }
            if &generated[start..start + suffix_len] == suffix {
                let follow_start = start + suffix_len;
                let follow_end = (follow_start + max_drafts).min(generated.len());
                return generated[follow_start..follow_end].to_vec();
            }
        }
    }

    Vec::new()
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
    device: &CudaDevice,
) -> Result<(Vec<i64>, String, String), String> {
    let prompt_len = prompt_ids.len();
    let total = prompt_len + max_new_tokens;
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);

    let pos0 = positions(0, prompt_len, device);
    let mut logits = model.forward_prec(input, pos0, &mut cache, Precision::Bf16);
    assert_logits_all_finite(&logits, "prefill")?;

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    for step in 0..max_new_tokens {
        let id = argmax_last(&logits)?;
        generated.push(id);
        new_ids.push(id);

        if step + 1 < max_new_tokens {
            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
            let pos = positions(prompt_len + step, 1, device);
            logits = model.forward_prec(tok, pos, &mut cache, Precision::Bf16);
        }
    }

    let text = decode_tokens(tokenizer, &generated)?;
    let new_text = decode_tokens(tokenizer, &new_ids)?;
    Ok((new_ids, text, new_text))
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

fn spec_decode_ngram(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    k: usize,
    device: &CudaDevice,
) -> Result<(Vec<i64>, String, String, SpecStats), String> {
    let prompt_len = prompt_ids.len();
    let draft_limit = k.saturating_sub(1);
    let capacity = prompt_len + max_new_tokens + k.max(1);
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(capacity);

    let pos0 = positions(0, prompt_len, device);
    let mut last_logits = model.forward_prec(input, pos0, &mut cache, Precision::Bf16);
    assert_logits_all_finite(&last_logits, "prefill")?;

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    let mut stats = SpecStats::default();

    while new_ids.len() < max_new_tokens {
        stats.steps += 1;
        let remaining = max_new_tokens - new_ids.len();
        let max_drafts = draft_limit.min(remaining.saturating_sub(1));
        let drafts = ngram_draft(&generated, max_drafts);
        stats.drafted += drafts.len();

        let kv_pos = full_attn_filled(&cache)?;
        if kv_pos != generated.len() {
            return Err(format!(
                "KV filled position {kv_pos} does not match generated length {}",
                generated.len()
            ));
        }
        let gdn_snap = cache.snapshot_gdn();

        let mut pred = vec![argmax_last(&last_logits)?];
        if !drafts.is_empty() {
            stats.verify_batches += 1;
            stats.verify_tokens += drafts.len();
            let verify_input =
                Tensor::<B, 1, Int>::from_data(drafts.as_slice(), device).unsqueeze();
            let verify_pos = positions(generated.len(), drafts.len(), device);
            let verify_logits =
                model.forward_prec(verify_input, verify_pos, &mut cache, Precision::Bf16);
            pred.extend(argmax_each_position(&verify_logits)?);
        }

        let mut accepted = 0usize;
        while accepted < drafts.len() && drafts[accepted] == pred[accepted] {
            accepted += 1;
        }
        stats.accepted += accepted;

        let mut committed = drafts[..accepted].to_vec();
        committed.push(pred[accepted]);
        if committed.len() > remaining {
            committed.truncate(remaining);
        }

        cache.rewind_kv(kv_pos);
        cache.restore_gdn(gdn_snap);

        for id in committed {
            let pos = positions(generated.len(), 1, device);
            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
            last_logits = model.forward_prec(tok, pos, &mut cache, Precision::Bf16);
            generated.push(id);
            new_ids.push(id);
            stats.committed += 1;
            if new_ids.len() == max_new_tokens {
                break;
            }
        }
    }

    let text = decode_tokens(tokenizer, &generated)?;
    let new_text = decode_tokens(tokenizer, &new_ids)?;
    Ok((new_ids, text, new_text, stats))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: qwen35_mtp_ngram failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let dir = PathBuf::from(env_string("QWEN35_DIR", MODEL_DIR));
    let prompt = env_string("PROMPT", PROMPT);
    let max_new_tokens = env_usize("MAX_NEW_TOKENS", MAX_NEW_TOKENS);
    let k = env_usize("K", K).max(1);

    let device = CudaDevice::default();
    println!("device: {device:?}");
    println!(
        "ngram spec config: dir={dir:?} prompt={prompt:?} max_new_tokens={max_new_tokens} k={k}"
    );
    println!(
        "verify framing: pred[0] comes from cached last_logits; verify batch forwards only draft tokens at absolute positions generated.len().."
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

    let (prompt_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    if prompt_ids.is_empty() {
        return Err("prompt encoded to zero tokens".to_string());
    }
    println!("prompt token ids: {prompt_ids:?}");

    println!("running BF16 greedy reference ...");
    let greedy_start = Instant::now();
    let (greedy_ids, greedy_text, greedy_new_text) =
        greedy_decode(&model, &tokenizer, &prompt_ids, max_new_tokens, &device)?;
    println!("GREEDY new token ids: {greedy_ids:?}");
    println!("GREEDY decoded new text: {greedy_new_text:?}");
    println!("GREEDY decoded text: {greedy_text:?}");
    println!(
        "GREEDY decode time: {:.1}s",
        greedy_start.elapsed().as_secs_f64()
    );

    println!("running n-gram speculative decode probe ...");
    let spec_start = Instant::now();
    let (spec_ids, spec_text, spec_new_text, stats) =
        spec_decode_ngram(&model, &tokenizer, &prompt_ids, max_new_tokens, k, &device)?;
    println!("SPEC new token ids: {spec_ids:?}");
    println!("SPEC decoded new text: {spec_new_text:?}");
    println!("SPEC decoded text: {spec_text:?}");
    println!(
        "SPEC decode time: {:.1}s",
        spec_start.elapsed().as_secs_f64()
    );

    let accept_rate = if stats.drafted == 0 {
        0.0
    } else {
        stats.accepted as f64 / stats.drafted as f64
    };
    let avg_committed = if stats.steps == 0 {
        0.0
    } else {
        stats.committed as f64 / stats.steps as f64
    };
    println!(
        "SPEC_STATS drafted={} accepted={} acceptance_rate={:.3} steps={} avg_committed_per_step={:.3} verify_batches={} verify_tokens={}",
        stats.drafted,
        stats.accepted,
        accept_rate,
        stats.steps,
        avg_committed,
        stats.verify_batches,
        stats.verify_tokens,
    );

    if let Some(idx) = compare_tokens(&greedy_ids, &spec_ids) {
        let greedy = greedy_ids.get(idx).copied();
        let spec = spec_ids.get(idx).copied();
        println!("TOKEN_IDENTITY FAIL first_divergence={idx} greedy={greedy:?} spec={spec:?}");
        return Err("speculative decode diverged from greedy reference".to_string());
    }

    println!(
        "TOKEN_IDENTITY PASS matched={}/{}",
        spec_ids.len(),
        greedy_ids.len()
    );
    println!("qwen35_mtp_ngram PASS");
    Ok(())
}
