//! MTP teacher-forced parity gate for Qwen3.6-35B-A3B.
//!
//! Build/run on the 35B host:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example qwen35_mtp_parity
//!
//! Diagnostic gate only: it always exits 0 and prints MTP2C_PARITY for the orchestrator.

use std::{path::PathBuf, time::Instant};

use burn::{
    prelude::Device,
    tensor::{DType, Int, Tensor},
};
use qwen3_burn::{Precision, Qwen3_5LayerType, Qwen3_5MoeConfig, Qwen3Tokenizer};

type B = Cuda;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 32;

fn env_string(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn positions(start: usize, len: usize, device: &CudaDevice) -> Tensor<2, Int> {
    if len == 1 {
        Tensor::<2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn last_hidden(hidden: &Tensor<3>) -> Tensor<3> {
    let [batch, seq, hidden_size] = hidden.dims();
    hidden
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..hidden_size])
}

fn assert_logits3_all_finite(logits: &Tensor<3>, what: &str) -> Result<(), String> {
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

fn argmax_last(logits: &Tensor<3>) -> Result<i64, String> {
    assert_logits3_all_finite(logits, "target")?;
    let [batch, seq, vocab] = logits.dims();
    let next: Tensor<2, Int> = logits
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..vocab])
        .reshape([batch, vocab])
        .argmax(1)
        .cast(DType::I64);
    let ids = next
        .into_data()
        .to_vec::<i64>()
        .map_err(|e| format!("read target argmax token: {e:?}"))?;
    ids.first()
        .copied()
        .ok_or_else(|| "target argmax returned no token".to_string())
}

fn argmax_and_margin(logits: &Tensor<2>) -> Result<(i64, f32), String> {
    let [batch, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("MTP parity expects batch 1, got {batch}"));
    }
    let values = logits
        .clone()
        .reshape([batch * vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read MTP logits: {e:?}"))?;
    let mut best = (usize::MAX, f32::NEG_INFINITY);
    let mut second = (usize::MAX, f32::NEG_INFINITY);
    for (idx, value) in values.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "MTP logits contain non-finite value at vocab index {idx}: {value}"
            ));
        }
        if value > best.1 {
            second = best;
            best = (idx, value);
        } else if value > second.1 {
            second = (idx, value);
        }
    }
    Ok((best.0 as i64, best.1 - second.1))
}

fn decode_tokens(tokenizer: &Qwen3Tokenizer, ids: &[i64]) -> Result<String, String> {
    let ids_u32: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
    tokenizer.decode(&ids_u32)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: qwen35_mtp_parity failed: {e}");
        println!("MTP2C_PARITY agree=0/0 mean_margin=0.000000");
    }
}

fn run() -> Result<(), String> {
    let dir = PathBuf::from(env_string("QWEN35_DIR", MODEL_DIR));
    let prompt = env_string("PROMPT", PROMPT);
    let max_new_tokens = env_usize("MAX_NEW_TOKENS", MAX_NEW_TOKENS);
    let quant = env_string("QUANT", "bf16");
    if !quant.eq_ignore_ascii_case("bf16") {
        return Err(format!(
            "unsupported QUANT={quant:?}; this gate expects QUANT=bf16"
        ));
    }

    let device = Device::cuda(0);
    println!(
        "MTP parity config: device={device:?} dir={dir:?} prompt={prompt:?} max_new_tokens={max_new_tokens} quant=bf16"
    );

    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    if cfg.mtp_num_hidden_layers == 0 {
        return Err("config has mtp_num_hidden_layers=0; no MTP block to test".to_string());
    }
    let linear_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == Qwen3_5LayerType::LinearAttention)
        .count();
    let full_layers = cfg.num_hidden_layers - linear_layers;
    println!(
        "config: {} layers ({} GDN, {} full-attn), MTP layers {}, hidden {}, vocab {}, experts top-{}/{}",
        cfg.num_hidden_layers,
        linear_layers,
        full_layers,
        cfg.mtp_num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.num_experts_per_tok,
        cfg.num_experts
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm(&device);

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
    println!(
        "lockstep: target cache advances on prompt/decode; MTP cache is separate and advances once per committed target token at the target hidden position"
    );

    let prompt_len = prompt_ids.len();
    let capacity = prompt_len + max_new_tokens + 2;
    let mut target_cache = model.model.new_cache_with_capacity(capacity);
    let mut mtp_cache = model.mtp_new_cache(capacity);

    let input = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).unsqueeze();
    let pos0 = positions(0, prompt_len, &device);
    let (hidden, mut logits) =
        model.forward_hidden_prec(input, pos0, &mut target_cache, Precision::Bf16);
    assert_logits3_all_finite(&logits, "prefill")?;

    let mut current_hidden = last_hidden(&hidden);
    let mut current_pos = prompt_len - 1;
    let mut pending: Option<(usize, i64, f32)> = None;
    let mut generated = prompt_ids.clone();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    let mut agree = 0usize;
    let mut total = 0usize;
    let mut margin_sum = 0.0f64;

    for step in 0..max_new_tokens {
        let target_tok = argmax_last(&logits)?;
        let committed_pos = prompt_len + step;

        if let Some((predicted_pos, draft_tok, margin)) = pending.take() {
            let matched = draft_tok == target_tok;
            agree += usize::from(matched);
            total += 1;
            margin_sum += margin as f64;
            println!(
                "MTP2C pos={predicted_pos} draft={draft_tok} target={target_tok} match={matched} margin={margin:.6}"
            );
        }

        generated.push(target_tok);
        new_ids.push(target_tok);

        let tok_next = Tensor::<2, Int>::from_data([[target_tok]], &device);
        let draft_pos = positions(current_pos, 1, &device);
        let (draft_logits, _mtp_hidden_out) = model.mtp.forward_draft(
            tok_next,
            current_hidden.clone(),
            draft_pos,
            &mut mtp_cache,
            &model.model.embed_tokens,
            &model.lm_head,
            Precision::Bf16,
        );
        let (draft_tok, margin) = argmax_and_margin(&draft_logits)?;
        pending = Some((current_pos + 2, draft_tok, margin));

        if step + 1 < max_new_tokens {
            let tok = Tensor::<2, Int>::from_data([[target_tok]], &device);
            let pos = positions(committed_pos, 1, &device);
            let (hidden, next_logits) =
                model.forward_hidden_prec(tok, pos, &mut target_cache, Precision::Bf16);
            current_hidden = last_hidden(&hidden);
            current_pos = committed_pos;
            logits = next_logits;
        }
    }

    let mean_margin = if total == 0 {
        0.0
    } else {
        margin_sum / total as f64
    };
    let new_text = decode_tokens(&tokenizer, &new_ids)?;
    println!("TARGET new token ids: {new_ids:?}");
    println!("TARGET decoded new text: {new_text:?}");
    println!("MTP2C_PARITY agree={agree}/{total} mean_margin={mean_margin:.6}");

    Ok(())
}
