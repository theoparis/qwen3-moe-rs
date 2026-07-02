//! D6 dense-linear fake-quantization gate for Qwen3.6-35B-A3B.
//!
//! Build only:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_nvfp4_gate
//!
//! Run on the 35B host only; this keeps one model resident and mutates weights in-place after the BF16
//! baseline token list is recorded.

use std::{path::PathBuf, time::Instant};

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    tensor::{DType, Int, Tensor},
};
use cubecl::{Runtime, cuda::CudaRuntime};
use qwen3_burn::{
    Precision, Qwen3_5LayerType, Qwen3_5MoeConfig, Qwen3Tokenizer,
    nvfp4::Nvfp4HadamardConfig,
    quant_gate::{
        DEFAULT_DENSE_SKIP, QuantPrecision, dense_linear_roles, fake_quant_all_dense,
        fake_quant_all_experts, fake_quant_one, linear_by_role_mut,
    },
};

type B = Cuda;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const NVFP4_MODEL_DIR: &str = "models/qwen3.6-35b-a3b-nvfp4";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 24;
const TEACHER_FORCE_CORPUS: &[&str] = &[
    "The capital of France is Paris, and the city is known for the Louvre and the River Seine.",
    "fn add(a: i32, b: i32) -> i32 { a + b }",
    "If all ravens are birds and this animal is a raven, then it must also be a bird.",
    "Shopping list: apples, lentils, olive oil, rice, coffee, spinach, and batteries.",
    "A transformer layer mixes token information with attention, applies a feed-forward block, and uses residual connections so the model can preserve useful context while refining each representation.",
    "In 2040, the research team compared solar output, battery prices, and grid demand before recommending a staged upgrade plan.",
    "To solve the puzzle, first separate the edge pieces, then match colors, and finally check whether any corner has been rotated incorrectly.",
    "The quick benchmark prints latency, throughput, memory use, and a short checksum so regressions are visible without reading a long trace.",
];

#[derive(Clone)]
struct TeacherForceBaseline {
    argmax: Vec<i64>,
    margins: Vec<f32>,
    targets: Vec<i64>,
    logits: Vec<f32>,
    positions: usize,
    vocab: usize,
}

#[derive(Clone, Copy, Default)]
struct BucketStats {
    count: usize,
    agree: usize,
}

#[derive(Default)]
struct TeacherForceMetrics {
    total: usize,
    agree: usize,
    disagreements: usize,
    disagreement_margin_sum: f64,
    kl_sum: f64,
    ce_delta_sum: f64,
    buckets: [BucketStats; 5],
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

fn parse_precision() -> Result<QuantPrecision, String> {
    match env_string("PREC", "mse").to_ascii_lowercase().as_str() {
        "mse" => Ok(QuantPrecision::Nvfp4Mse),
        "amax" => Ok(QuantPrecision::Nvfp4Amax),
        "hadamard" => Ok(QuantPrecision::Nvfp4Hadamard),
        "fp8" => Ok(QuantPrecision::Fp8),
        other => Err(format!(
            "PREC must be one of mse|amax|hadamard|fp8, got {other:?}"
        )),
    }
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
        .slice([0..batch, (seq - 1)..seq, 0..vocab])
        .reshape([batch, vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read {what} logits: {e:?}"))?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "{what} logits contain non-finite value at vocab index {idx}: {value}"
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

fn repeated_garbage(ids: &[i64]) -> bool {
    ids.len() >= 8 && ids.windows(2).filter(|pair| pair[0] == pair[1]).count() >= ids.len() - 2
}

fn compare_tokens(reference: &[i64], candidate: &[i64]) -> (usize, Option<usize>) {
    let matched = reference
        .iter()
        .zip(candidate.iter())
        .filter(|(a, b)| a == b)
        .count();
    let first_divergence = reference
        .iter()
        .zip(candidate.iter())
        .position(|(a, b)| a != b)
        .or_else(|| {
            (reference.len() != candidate.len()).then_some(reference.len().min(candidate.len()))
        });
    (matched, first_divergence)
}

fn argmax_top2(row: &[f32]) -> (usize, f32, f32) {
    let mut top1_id = 0usize;
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for (idx, &value) in row.iter().enumerate() {
        if value > top1 {
            top2 = top1;
            top1 = value;
            top1_id = idx;
        } else if value > top2 {
            top2 = value;
        }
    }
    (top1_id, top1, top2)
}

fn logsumexp(row: &[f32]) -> f64 {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum = row
        .iter()
        .map(|&value| ((value as f64) - max).exp())
        .sum::<f64>();
    max + sum.ln()
}

fn teacher_forced_logits(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    text: &str,
    device: &CudaDevice,
) -> Result<(Vec<f32>, Vec<i64>, usize, usize), String> {
    let (token_u32, _) = tokenizer.encode_no_pad(text)?;
    let token_ids: Vec<i64> = token_u32.iter().map(|&id| id as i64).collect();
    if token_ids.len() < 2 {
        return Err(format!("teacher-force corpus item is too short: {text:?}"));
    }

    let seq = token_ids.len();
    let input = Tensor::<B, 1, Int>::from_data(token_ids.as_slice(), device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(seq);
    let pos = positions(0, seq, device);
    let logits = model.forward_prec(input, pos, &mut cache, Precision::Bf16);
    let [batch, got_seq, vocab] = logits.dims();
    if batch != 1 || got_seq != seq {
        return Err(format!(
            "teacher-force logits shape mismatch: got [{batch}, {got_seq}, {vocab}], expected [1, {seq}, V]"
        ));
    }

    let positions = seq - 1;
    let values = logits
        .slice([0..batch, 0..positions, 0..vocab])
        .reshape([positions, vocab])
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read teacher-force logits: {e:?}"))?;
    if let Some((idx, value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "teacher-force logits contain non-finite value at flat index {idx}: {value}"
        ));
    }

    Ok((values, token_ids[1..].to_vec(), positions, vocab))
}

#[allow(dead_code)]
fn teacher_forced_argmax(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    text: &str,
    device: &CudaDevice,
) -> Result<Vec<i64>, String> {
    let (logits, _, positions, vocab) = teacher_forced_logits(model, tokenizer, text, device)?;
    Ok((0..positions)
        .map(|pos| {
            let row = &logits[pos * vocab..(pos + 1) * vocab];
            argmax_top2(row).0 as i64
        })
        .collect())
}

fn teacher_forced_baseline(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    text: &str,
    device: &CudaDevice,
) -> Result<TeacherForceBaseline, String> {
    let (logits, targets, positions, vocab) =
        teacher_forced_logits(model, tokenizer, text, device)?;
    let mut argmax = Vec::with_capacity(positions);
    let mut margins = Vec::with_capacity(positions);
    for pos in 0..positions {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        let (top1_id, top1, top2) = argmax_top2(row);
        argmax.push(top1_id as i64);
        margins.push(top1 - top2);
    }

    Ok(TeacherForceBaseline {
        argmax,
        margins,
        targets,
        logits,
        positions,
        vocab,
    })
}

fn bucket_pct(bucket: BucketStats) -> f64 {
    if bucket.count == 0 {
        0.0
    } else {
        100.0 * bucket.agree as f64 / bucket.count as f64
    }
}

/// Record the BF16 reference logits/argmax/margins for every corpus item on the host.
///
/// These are held in host memory (`TeacherForceBaseline.logits`) so the reference model can be
/// mutated in place (fake-quant) or fully dropped (real-nvfp4 two-phase) before the eval pass.
fn collect_baselines(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    device: &CudaDevice,
) -> Result<Vec<TeacherForceBaseline>, String> {
    println!(
        "running teacher-forced BF16 baseline over {} corpus items ...",
        TEACHER_FORCE_CORPUS.len()
    );
    let base_start = Instant::now();
    let mut baselines = Vec::with_capacity(TEACHER_FORCE_CORPUS.len());
    for (idx, text) in TEACHER_FORCE_CORPUS.iter().enumerate() {
        let baseline = teacher_forced_baseline(model, tokenizer, text, device)?;
        if idx == 0 {
            // Fail-fast guard on the very first forced string. A fully-loaded BF16 reference must
            // produce finite logits; if the model were left uninitialized (weights never loaded)
            // the router logits go NaN and argmax routing selects expert index -1, panicking deep
            // in Qwen3_5SharedMoeBlock::forward_impl with an opaque
            // "index out of bounds: len 256, index -1(usize)". Surface a clear message here instead.
            if let Some((flat, value)) = baseline
                .logits
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(format!(
                    "phase-1 BF16 baseline produced a non-finite logit ({value}) at flat index {flat} for {text:?}; the reference model is not fully loaded (verify load_weights_sharded ran and passed before collect_baselines)"
                ));
            }
        }
        baselines.push(baseline);
    }
    let baseline_positions: usize = baselines.iter().map(|baseline| baseline.positions).sum();
    println!(
        "TEACHER_FORCE_BASELINE positions={} time={:.1}s",
        baseline_positions,
        base_start.elapsed().as_secs_f64()
    );
    Ok(baselines)
}

/// Run the eval pass: recompute logits from `model` for each corpus item and score them against the
/// host-resident BF16 `baselines`. `prec_label`/`quant_experts` only annotate the printed metrics.
fn evaluate_teacherforce(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    baselines: &[TeacherForceBaseline],
    prec_label: &str,
    quant_experts: bool,
    device: &CudaDevice,
) -> Result<(), String> {
    println!("running teacher-forced quant eval pass ...");
    let quant_start = Instant::now();
    let mut metrics = TeacherForceMetrics::default();
    for (text, baseline) in TEACHER_FORCE_CORPUS.iter().zip(baselines.iter()) {
        let (quant_logits, _, positions, vocab) =
            teacher_forced_logits(model, tokenizer, text, device)?;
        if positions != baseline.positions || vocab != baseline.vocab {
            return Err(format!(
                "teacher-force quant shape mismatch for {text:?}: got positions={positions} vocab={vocab}, expected positions={} vocab={}",
                baseline.positions, baseline.vocab
            ));
        }

        for pos in 0..positions {
            let base_row = &baseline.logits[pos * vocab..(pos + 1) * vocab];
            let quant_row = &quant_logits[pos * vocab..(pos + 1) * vocab];
            let quant_id = argmax_top2(quant_row).0 as i64;
            let agreed = quant_id == baseline.argmax[pos];
            let margin = baseline.margins[pos];

            metrics.total += 1;
            if agreed {
                metrics.agree += 1;
            } else {
                metrics.disagreements += 1;
                metrics.disagreement_margin_sum += margin as f64;
            }

            for (idx, threshold) in [f32::NEG_INFINITY, 0.01f32, 0.05f32, 0.1f32, 0.5f32]
                .iter()
                .enumerate()
            {
                if margin > *threshold {
                    metrics.buckets[idx].count += 1;
                    if agreed {
                        metrics.buckets[idx].agree += 1;
                    }
                }
            }

            let base_lse = logsumexp(base_row);
            let quant_lse = logsumexp(quant_row);
            let mut kl = 0.0f64;
            for idx in 0..vocab {
                let base_logp = base_row[idx] as f64 - base_lse;
                let quant_logp = quant_row[idx] as f64 - quant_lse;
                kl += base_logp.exp() * (base_logp - quant_logp);
            }
            metrics.kl_sum += kl;

            let target = baseline.targets[pos] as usize;
            if target >= vocab {
                return Err(format!("target id {target} is outside vocab {vocab}"));
            }
            let base_ce = base_lse - base_row[target] as f64;
            let quant_ce = quant_lse - quant_row[target] as f64;
            metrics.ce_delta_sum += quant_ce - base_ce;
        }
    }

    let top1_pct = 100.0 * metrics.agree as f64 / metrics.total as f64;
    let mean_kl = metrics.kl_sum / metrics.total as f64;
    let mean_ce_delta = metrics.ce_delta_sum / metrics.total as f64;
    let mean_disagreement_margin = if metrics.disagreements == 0 {
        0.0
    } else {
        metrics.disagreement_margin_sum / metrics.disagreements as f64
    };
    println!(
        "TEACHER_FORCE prec={prec_label} quant_experts={quant_experts} top1={top1_pct:.3}% positions={} kl={mean_kl:.5} ce_delta={mean_ce_delta:.5} agree_by_margin: all={:.3}% >0.01={:.3}% >0.05={:.3}% >0.1={:.3}% >0.5={:.3}%",
        metrics.total,
        bucket_pct(metrics.buckets[0]),
        bucket_pct(metrics.buckets[1]),
        bucket_pct(metrics.buckets[2]),
        bucket_pct(metrics.buckets[3]),
        bucket_pct(metrics.buckets[4]),
    );
    println!(
        "TEACHER_FORCE_COUNTS all={}/{} >0.01={}/{} >0.05={}/{} >0.1={}/{} >0.5={}/{} disagreements={} mean_disagreement_margin={mean_disagreement_margin:.6}",
        metrics.buckets[0].agree,
        metrics.buckets[0].count,
        metrics.buckets[1].agree,
        metrics.buckets[1].count,
        metrics.buckets[2].agree,
        metrics.buckets[2].count,
        metrics.buckets[3].agree,
        metrics.buckets[3].count,
        metrics.buckets[4].agree,
        metrics.buckets[4].count,
        metrics.disagreements,
    );
    println!(
        "TEACHER_FORCE quant time: {:.1}s",
        quant_start.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Fake-quant teacher-force gate: BF16 baseline -> in-place fake-quant of the resident model -> eval.
fn run_teacherforce(
    model: &mut qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prec: QuantPrecision,
    quant_experts: bool,
    device: &CudaDevice,
) -> Result<(), String> {
    let baselines = collect_baselines(model, tokenizer, device)?;
    fake_quant_all_dense(model, prec, DEFAULT_DENSE_SKIP);
    if quant_experts {
        fake_quant_all_experts(model, prec);
    }
    evaluate_teacherforce(
        model,
        tokenizer,
        &baselines,
        &format!("{prec:?}"),
        quant_experts,
        device,
    )
}

/// Real-NVFP4 teacher-force gate (two-phase, memory-bounded).
///
/// The resident BF16 model (~71GB device) is *moved in* so we can drop it before loading the real
/// NVIDIA NVFP4 checkpoint (~22.5GB): both do not fit simultaneously. Phase 1 records the BF16
/// reference logits on the host; phase 2 frees the BF16 model; phase 3 loads the real NVFP4
/// checkpoint fresh (NO fake-quant) and scores it against the host-resident baselines. Peak device
/// residency is therefore max(bf16, nvfp4), never their sum.
fn run_teacherforce_real_nvfp4(
    model: qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    config: &Qwen3_5MoeConfig,
    tokenizer: &Qwen3Tokenizer,
    nvfp4_dir: &std::path::Path,
    quant_experts: bool,
    device: &CudaDevice,
) -> Result<(), String> {
    let client = CudaRuntime::client(device);

    // Phase 1: BF16 reference logits recorded to host memory.
    let baselines = collect_baselines(&model, tokenizer, device)?;

    // Phase 2: free the BF16 model before loading the NVFP4 checkpoint.
    println!(
        "device memory before dropping BF16 model: {:?}",
        client.memory_usage()
    );
    drop(model);
    println!(
        "device memory after dropping BF16 model: {:?}",
        client.memory_usage()
    );

    // Phase 3: load the real NVIDIA NVFP4 checkpoint fresh (no fake-quant) and eval.
    let raw = std::env::var("NVFP4_DEQUANT_TO_FP8").ok().as_deref() != Some("1");
    println!(
        "loading NVIDIA NVFP4 checkpoint from {nvfp4_dir:?} (mode={}) ...",
        if raw {
            "raw NVFP4 dispatch"
        } else {
            "staged fp8 fallback"
        }
    );
    let load_start = Instant::now();
    let mut model = config.init_causal_lm::<B>(device);
    model
        .load_nvidia_nvfp4(nvfp4_dir)
        .map_err(|e| format!("load_nvidia_nvfp4 failed: {e:?}"))?;
    println!(
        "NVFP4 load time: {:.1}s, device memory after nvfp4 load: {:?}",
        load_start.elapsed().as_secs_f64(),
        client.memory_usage()
    );

    evaluate_teacherforce(
        &model,
        tokenizer,
        &baselines,
        "real-nvfp4",
        quant_experts,
        device,
    )
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

    let generated_u32: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
    let new_u32: Vec<u32> = new_ids.iter().map(|&id| id as u32).collect();
    let text = tokenizer.decode(&generated_u32)?;
    let new_text = tokenizer.decode(&new_u32)?;
    Ok((new_ids, text, new_text))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: qwen35_nvfp4_gate failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let dir = PathBuf::from(env_string("QWEN35_DIR", MODEL_DIR));
    let nvfp4_dir = PathBuf::from(env_string("QWEN35_NVFP4_DIR", NVFP4_MODEL_DIR));
    let prompt = env_string("PROMPT", PROMPT);
    let max_new_tokens = env_usize("MAX_NEW_TOKENS", MAX_NEW_TOKENS);
    let mode = env_string("MODE", "all").to_ascii_lowercase();
    // PREC=real-nvfp4 (teacherforce only) skips fake-quant entirely and instead loads the real
    // NVIDIA NVFP4 checkpoint via load_nvidia_nvfp4 after the BF16 reference pass. `prec` (a
    // QuantPrecision fake-quant recipe) is unused in that path, so we do not parse it there.
    let real_nvfp4 = env_string("PREC", "mse").eq_ignore_ascii_case("real-nvfp4");
    let prec = if real_nvfp4 {
        // Placeholder recipe; never consumed on the real-nvfp4 path.
        QuantPrecision::Nvfp4Mse
    } else {
        parse_precision()?
    };
    let quant_experts = env_string("QUANT_EXPERTS", "0") == "1";

    let device = CudaDevice::default();
    println!("device: {device:?}");
    let prec_display = if real_nvfp4 {
        "real-nvfp4".to_string()
    } else {
        format!("{prec:?}")
    };
    println!(
        "gate config: dir={dir:?} prompt={prompt:?} max_new_tokens={max_new_tokens} mode={mode:?} prec={prec_display} quant_experts={quant_experts}"
    );
    if real_nvfp4 && mode != "teacherforce" {
        return Err(format!(
            "PREC=real-nvfp4 is only supported with MODE=teacherforce, got MODE={mode:?}"
        ));
    }
    if !real_nvfp4 && prec == QuantPrecision::Nvfp4Hadamard {
        let cfg = Nvfp4HadamardConfig::from_env();
        println!(
            "hadamard config: group_size={} clip_c={} base_seed=0x{:016x}",
            cfg.group_size, cfg.clip_c, cfg.base_seed
        );
    }

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

    if mode == "teacherforce" {
        if real_nvfp4 {
            return run_teacherforce_real_nvfp4(
                model,
                &cfg,
                &tokenizer,
                &nvfp4_dir,
                quant_experts,
                &device,
            );
        }
        return run_teacherforce(&mut model, &tokenizer, prec, quant_experts, &device);
    }

    let (prompt_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    println!("prompt token ids: {prompt_ids:?}");

    println!("running BF16 baseline greedy decode ...");
    let base_start = Instant::now();
    let (baseline_ids, baseline_text, baseline_new_text) =
        greedy_decode(&model, &tokenizer, &prompt_ids, max_new_tokens, &device)?;
    println!("BASELINE new token ids: {baseline_ids:?}");
    println!("BASELINE decoded new text: {baseline_new_text:?}");
    println!("BASELINE decoded text: {baseline_text:?}");
    println!(
        "BASELINE decode time: {:.1}s",
        base_start.elapsed().as_secs_f64()
    );

    match mode.as_str() {
        "all" => {
            fake_quant_all_dense(&mut model, prec, DEFAULT_DENSE_SKIP);
            println!("running fake-quant greedy decode ...");
            let quant_start = Instant::now();
            let (quant_ids, quant_text, quant_new_text) =
                greedy_decode(&model, &tokenizer, &prompt_ids, max_new_tokens, &device)?;
            let (matched, first_divergence) = compare_tokens(&baseline_ids, &quant_ids);
            let coherent = !repeated_garbage(&quant_ids);
            println!("QUANT new token ids: {quant_ids:?}");
            println!("QUANT decoded new text: {quant_new_text:?}");
            println!("QUANT decoded text: {quant_text:?}");
            println!(
                "TOKEN_IDENTITY matched={matched}/{max_new_tokens} first_divergence={first_divergence:?} coherent={coherent}"
            );
            println!(
                "QUANT decode time: {:.1}s",
                quant_start.elapsed().as_secs_f64()
            );
        }
        "sweep" => {
            let roles = dense_linear_roles(&model);
            println!("role | roundtrip_cos | tokens_matched/{max_new_tokens} | first_divergence");
            for role in roles {
                let original = {
                    let lin = linear_by_role_mut(&mut model, &role)
                        .ok_or_else(|| format!("missing role {role}"))?;
                    lin.weight.clone()
                };
                let cos = fake_quant_one(&mut model, &role, prec)
                    .ok_or_else(|| format!("failed to fake-quant role {role}"))?;
                let (ids, _, _) =
                    greedy_decode(&model, &tokenizer, &prompt_ids, max_new_tokens, &device)?;
                let (matched, first_divergence) = compare_tokens(&baseline_ids, &ids);
                {
                    let lin = linear_by_role_mut(&mut model, &role)
                        .ok_or_else(|| format!("missing role {role} during restore"))?;
                    lin.weight = original;
                }
                println!("{role} | {cos:.9} | {matched}/{max_new_tokens} | {first_divergence:?}");
            }
        }
        other => {
            return Err(format!(
                "MODE must be all, sweep, or teacherforce, got {other:?}"
            ));
        }
    }

    Ok(())
}
