//! M-D G2: eager-driven static-step parity on the real Qwen3.6-35B-A3B raw backend.
//!
//! Build/check:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo check --example qwen35_static_parity --features cuda
//!
//! Run on a CUDA host:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example qwen35_static_parity

use std::{path::PathBuf, time::Instant};

use burn::{
    backend::cuda::CudaDevice,
    prelude::Device,
    tensor::{DType, Int, Tensor},
};
use cubecl::{Runtime, cuda::CudaRuntime};
use qwen3_burn::{
    Precision, Qwen3_5HybridCache, Qwen3_5HybridLayerCache, Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM, Qwen3Tokenizer,
    capture::CaptureBackend,
    linear3,
    qwen3_5::{Qwen3_5DecoderLayer, Qwen3_5DenseQuantBackend},
    rope_freqs,
};

type B = CaptureBackend;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 64;
const T_MAX: usize = 1024;
const ROTARY_DIM: usize = 64;
const ROPE_THETA: f64 = 10_000_000.0;

fn proc_status_value(label: &str) -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix(label)
            .map(|value| value.trim().to_string())
    })
}

fn print_mem(label: &str) {
    let rss = proc_status_value("VmRSS:").unwrap_or_else(|| "unobservable".to_string());
    let hwm = proc_status_value("VmHWM:").unwrap_or_else(|| "unobservable".to_string());
    println!("{label}: VmRSS={rss}, VmHWM={hwm}");
}

fn positions(start: usize, len: usize, device: &CudaDevice) -> Tensor<2, Int> {
    if len == 1 {
        Tensor::<2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn logits_last_row3<T>(logits: &Tensor<T, 3>, what: &str) -> Result<Vec<f32>, String>
where
    T: Backend,
{
    let [batch, seq, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("{what} logits expected batch=1, got {batch}"));
    }
    let values = logits
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..vocab])
        .reshape([vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read {what} logits: {e:?}"))?;
    ensure_finite(&values, what)?;
    Ok(values)
}

fn logits_row2<T>(logits: &Tensor<T, 2>, what: &str) -> Result<Vec<f32>, String>
where
    T: Backend,
{
    let [batch, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("{what} logits expected batch=1, got {batch}"));
    }
    let values = logits
        .clone()
        .reshape([vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read {what} logits: {e:?}"))?;
    ensure_finite(&values, what)?;
    Ok(values)
}

fn ensure_finite(row: &[f32], what: &str) -> Result<(), String> {
    if let Some((idx, value)) = row.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(format!(
            "{what} logits contain non-finite value at vocab index {idx}: {value}"
        ));
    }
    Ok(())
}

fn argmax_row(row: &[f32]) -> Result<i64, String> {
    row.iter()
        .copied()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(idx, _)| idx as i64)
        .ok_or_else(|| "argmax row is empty".to_string())
}

fn logit_delta(a: &[f32], b: &[f32]) -> Result<(f64, f32), String> {
    if a.len() != b.len() {
        return Err(format!("logit lengths differ: {} != {}", a.len(), b.len()));
    }
    let mut sum = 0.0f64;
    let mut max = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        sum += d as f64;
        max = max.max(d);
    }
    Ok((sum / a.len() as f64, max))
}

fn static_prefill_logits<T>(
    model: &Qwen3_5MoeForCausalLM<T>,
    input_ids: Tensor<T, 2, Int>,
    position_ids: Tensor<T, 2, Int>,
    cache: &mut Qwen3_5HybridCache<T>,
    prec: Precision,
) -> Tensor<T, 3>
where
    T: Backend + Qwen3_5DenseQuantBackend,
{
    let mut hidden_states = model.model.embed_tokens.forward(input_ids).cast(DType::F32);
    for (idx, (layer, layer_cache)) in model
        .model
        .layers
        .iter()
        .zip(cache.layers.iter_mut())
        .enumerate()
    {
        hidden_states = match (layer, layer_cache) {
            (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(cache)) => {
                let hidden_states =
                    layer.forward_prefill_recurrent_static(hidden_states, cache, prec);
                let residual = hidden_states.clone();
                let hidden_states = layer.post_attention_layernorm.forward(hidden_states);
                let hidden_states = layer.mlp.forward(hidden_states, prec);
                residual + hidden_states
            }
            (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(cache)) => {
                layer.forward_decoder_with_cache(hidden_states, position_ids.clone(), cache, prec)
            }
            (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                panic!("Qwen3.5 hybrid cache layer {idx} is Full but model layer is Linear")
            }
            (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                panic!("Qwen3.5 hybrid cache layer {idx} is Linear but model layer is Full")
            }
        };
    }
    linear3(
        &model.lm_head,
        model.model.norm.forward(hidden_states),
        prec,
    )
}

struct DecodeRun {
    ids: Vec<i64>,
    logits: Vec<Vec<f32>>,
    seconds: f64,
}

fn eager_decode(
    model: &Qwen3_5MoeForCausalLM,
    prompt_ids: &[i64],
    prec: Precision,
    device: &CudaDevice,
) -> Result<DecodeRun, String> {
    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);

    let prompt_len = prompt_ids.len();
    let total = prompt_len + MAX_NEW_TOKENS;
    let input = Tensor::<1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);
    let start = Instant::now();
    let mut logits = model.forward_prec(input, positions(0, prompt_len, device), &mut cache, prec);

    let mut ids = Vec::with_capacity(MAX_NEW_TOKENS);
    let mut rows = Vec::with_capacity(MAX_NEW_TOKENS);
    for step in 0..MAX_NEW_TOKENS {
        let row = logits_last_row3(&logits, "eager")?;
        let id = argmax_row(&row)?;
        rows.push(row);
        ids.push(id);

        if step + 1 < MAX_NEW_TOKENS {
            let tok = Tensor::<2, Int>::from_data([[id]], device);
            let pos = positions(prompt_len + step, 1, device);
            logits = model.forward_prec(tok, pos, &mut cache, prec);
        }
    }

    Ok(DecodeRun {
        ids,
        logits: rows,
        seconds: start.elapsed().as_secs_f64(),
    })
}

fn static_decode(
    model: &Qwen3_5MoeForCausalLM,
    prompt_ids: &[i64],
    prec: Precision,
    device: &CudaDevice,
) -> Result<DecodeRun, String> {
    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);

    let prompt_len = prompt_ids.len();
    let input = Tensor::<1, Int>::from_data(prompt_ids, device).unsqueeze();
    let prompt_pos = positions(0, prompt_len, device);

    let mut cache = model.model.new_cache_with_capacity(T_MAX);
    model.init_static_caches(&mut cache, 1);

    let preflight = model.preflight_static(&cache, 1);
    println!("preflight_static: {preflight:?}");
    if let Err(e) = preflight {
        return Err(format!("preflight_static failed: {e}"));
    }

    let freqs = rope_freqs::<B>(ROTARY_DIM, ROPE_THETA, device);
    let arange_tmax = Tensor::<1, Int>::arange(0..T_MAX as i64, device);

    let start = Instant::now();
    let logits = static_prefill_logits(model, input, prompt_pos, &mut cache, prec);
    let mut current_row = logits_last_row3(&logits, "static prefill")?;
    let mut pos = Tensor::<1, Int>::full([1], prompt_len as i64, device);

    let mut ids = Vec::with_capacity(MAX_NEW_TOKENS);
    let mut rows = Vec::with_capacity(MAX_NEW_TOKENS);
    for step in 0..MAX_NEW_TOKENS {
        let id = argmax_row(&current_row)?;
        rows.push(current_row.clone());
        ids.push(id);

        if step + 1 < MAX_NEW_TOKENS {
            let tok = Tensor::<2, Int>::from_data([[id]], device);
            let logits = model.forward_decode_static_pre(
                tok,
                pos.clone(),
                &mut cache,
                prec,
                &freqs,
                &arange_tmax,
            );
            current_row = logits_row2(&logits, "static decode")?;
            pos = pos.add_scalar(1i64);
        }
    }

    Ok(DecodeRun {
        ids,
        logits: rows,
        seconds: start.elapsed().as_secs_f64(),
    })
}

fn print_first8_deltas(eager: &DecodeRun, static_run: &DecodeRun) -> Result<(), String> {
    let n = 8.min(eager.logits.len()).min(static_run.logits.len());
    println!("first {n} per-step logit deltas (static vs eager):");
    for step in 0..n {
        let (mean, max) = logit_delta(&eager.logits[step], &static_run.logits[step])?;
        println!("  step={step:02} mean_abs={mean:.9} max_abs={max:.9}");
    }
    Ok(())
}

fn print_token_diff(eager: &DecodeRun, static_run: &DecodeRun) -> Result<(), String> {
    println!("G2 FAIL: static generated ids differ from eager reference");
    println!("pos\teager_id\tstatic_id\tmarker");
    let mut first_mismatch = None;
    for pos in 0..MAX_NEW_TOKENS {
        let eager_id = eager.ids.get(pos).copied();
        let static_id = static_run.ids.get(pos).copied();
        let marker = if eager_id == static_id {
            ""
        } else {
            first_mismatch.get_or_insert(pos);
            "<--"
        };
        println!(
            "{pos}\t{}\t{}\t{marker}",
            eager_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<missing>".to_string()),
            static_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "<missing>".to_string())
        );
    }

    if let Some(pos) = first_mismatch {
        let (mean, max) = logit_delta(&eager.logits[pos], &static_run.logits[pos])?;
        println!(
            "first mismatch step={pos}: mean_abs_logit_delta={mean:.9} max_abs_logit_delta={max:.9}"
        );
    }
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: G2 qwen35_static_parity failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let backend = std::env::var("BACKEND").unwrap_or_else(|_| "raw".to_string());
    if !backend.eq_ignore_ascii_case("raw") {
        return Err(format!(
            "unsupported BACKEND={backend:?}; expected raw (CaptureBackend only)"
        ));
    }

    let dir = PathBuf::from(std::env::var("QWEN35_DIR").unwrap_or_else(|_| MODEL_DIR.to_string()));
    let device = Device::cuda(0);
    let quant = std::env::var("QUANT").unwrap_or_else(|_| "bf16".to_string());
    let quant_mode = if quant.eq_ignore_ascii_case("fp8") {
        "fp8"
    } else if quant.eq_ignore_ascii_case("bf16") {
        "bf16"
    } else {
        return Err(format!("unsupported QUANT={quant:?}; expected bf16 or fp8"));
    };

    println!("device: {device:?} | backend=raw");
    println!("quant mode: {quant_mode}");
    print_mem("memory at start");

    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    let linear_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == qwen3_burn::Qwen3_5LayerType::LinearAttention)
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
    let mut model = cfg.init_causal_lm(&device);

    println!("loading sharded BF16 weights from {dir:?} ...");
    let load_start = Instant::now();
    let report = model
        .load_weights_sharded(&dir)
        .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!(
        "load verify: pass={}, mapped_tensors={}, params={}",
        report.pass(),
        report.mapped_tensors,
        report.param_count
    );
    println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());
    print_mem("memory after load");

    let client = CudaRuntime::client(&device);
    println!(
        "QWEN35_STATIC_PARITY memory before quant: {:?}",
        client.memory_usage()
    );
    if quant_mode == "fp8" {
        #[cfg(feature = "cuda")]
        {
            qwen3_burn::quant_gate::quantize_dense_fp8(&mut model, &[]);
            qwen3_burn::quant_gate::quantize_experts_fp8(&mut model, &[]);
        }
    }
    println!(
        "QWEN35_STATIC_PARITY memory after quant: {:?}",
        client.memory_usage()
    );

    let (prompt_u32, _) = tokenizer.encode_no_pad(PROMPT)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    assert!(prompt_ids.len() + MAX_NEW_TOKENS <= T_MAX);
    println!("prompt: {PROMPT:?}");
    println!("prompt token ids: {prompt_ids:?}");
    println!("max_new_tokens={MAX_NEW_TOKENS}, T_max={T_MAX}, precision=BF16");

    let prec = Precision::Bf16;
    println!("running eager reference leg ...");
    let eager = eager_decode(&model, &prompt_ids, prec, &device)?;
    println!("eager new token ids: {:?}", eager.ids);
    println!(
        "eager throughput: {:.2} tok/s ({:.3}s)",
        MAX_NEW_TOKENS as f64 / eager.seconds,
        eager.seconds
    );

    println!("running static leg ...");
    let static_run = static_decode(&model, &prompt_ids, prec, &device)?;
    println!("static new token ids: {:?}", static_run.ids);
    println!(
        "static throughput: {:.2} tok/s ({:.3}s)",
        MAX_NEW_TOKENS as f64 / static_run.seconds,
        static_run.seconds
    );

    print_first8_deltas(&eager, &static_run)?;

    if eager.ids != static_run.ids {
        print_token_diff(&eager, &static_run)?;
        std::process::exit(1);
    }

    println!("G2 PASS (64/64) quant={quant_mode}");
    Ok(())
}
