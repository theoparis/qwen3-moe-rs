//! M-D T1: raw CaptureBackend vs Fusion-Cuda greedy parity gate for Qwen3.6-35B-A3B.
//!
//! Build/check:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo check --example qwen35_raw_parity --features cuda
//!
//! Run on a CUDA host:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example qwen35_raw_parity

use std::{path::PathBuf, time::Instant};

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    prelude::Backend,
    tensor::{Int, Tensor},
};
use cubecl::{Runtime, cuda::CudaRuntime};
use qwen3_burn::capture::CaptureBackend;
use qwen3_burn::{Precision, Qwen3_5MoeConfig, Qwen3Tokenizer, qwen3_5::Qwen3_5DenseQuantBackend};

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

// Fusion reference provenance: commit c039e77, GPU run 2026-07-01,
// examples/qwen35_generate QUANT=bf16 greedy.
const FUSION_REF: [i64; 16] = [
    11751, 11, 264, 3177, 34756, 364, 1141, 8807, 3712, 11, 7431, 11, 321, 25438, 57902, 13,
];

const FORCED: [i64; 21] = [
    760, 6511, 314, 9338, 369, 11751, 11, 264, 3177, 34756, 364, 1141, 8807, 3712, 11, 7431, 11,
    321, 25438, 57902, 13,
];

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

fn positions<B>(start: usize, len: usize, device: &CudaDevice) -> Tensor<B, 2, Int>
where
    B: Backend<Device = CudaDevice>,
{
    if len == 1 {
        Tensor::<B, 2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<B, 1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn logits_last_row<B>(logits: &Tensor<B, 3>, what: &str) -> Result<Vec<f32>, String>
where
    B: Backend,
{
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
    Ok(values)
}

fn assert_logits_all_finite<B>(logits: &Tensor<B, 3>, what: &str) -> Result<(), String>
where
    B: Backend,
{
    logits_last_row(logits, what).map(|_| ())
}

fn top5(row: &[f32]) -> Vec<(usize, f32)> {
    let mut idxv: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    idxv.sort_by(|a, b| b.1.total_cmp(&a.1));
    idxv.truncate(5);
    idxv
}

fn argmax_last<B>(logits: &Tensor<B, 3>) -> Result<i64, String>
where
    B: Backend,
{
    let row = logits_last_row(logits, "decode")?;
    let (id, _) = top5(&row)
        .into_iter()
        .next()
        .ok_or_else(|| "argmax returned no token".to_string())?;
    Ok(id as i64)
}

struct GreedyRun {
    generated: Vec<i64>,
    new_ids: Vec<i64>,
    text: String,
    new_text: String,
    seconds: f64,
}

fn greedy_decode<B>(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    prec: Precision,
    fused_moe: bool,
    debug_prefill: bool,
    device: &CudaDevice,
) -> Result<GreedyRun, String>
where
    B: Backend<Device = CudaDevice> + Qwen3_5DenseQuantBackend,
{
    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(fused_moe);

    let prompt_len = prompt_ids.len();
    let total = prompt_len + max_new_tokens;
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);
    let gen_start = Instant::now();
    let pos0 = positions::<B>(0, prompt_len, device);
    let mut logits = model.forward_prec(input, pos0, &mut cache, prec);
    assert_logits_all_finite(&logits, "prefill")?;
    if debug_prefill {
        let [b, s, v] = logits.dims();
        let row: Vec<f32> = logits
            .clone()
            .slice([0..b, (s - 1)..s, 0..v])
            .reshape([v])
            .into_data()
            .to_vec::<f32>()
            .expect("logits row");
        let mut idxv: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
        idxv.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap());
        let mean = row.iter().sum::<f32>() / v as f32;
        eprintln!("[dbg] prefill logits mean={mean:.4} top5={:?}", &idxv[..5]);
    }

    let mut generated = prompt_ids.to_vec();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    for step in 0..max_new_tokens {
        let id = argmax_last(&logits)?;
        generated.push(id);
        new_ids.push(id);

        if step + 1 < max_new_tokens {
            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
            let pos = positions::<B>(prompt_len + step, 1, device);
            logits = model.forward_prec(tok, pos, &mut cache, prec);
        }
    }

    let seconds = gen_start.elapsed().as_secs_f64();
    let generated_u32: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
    let new_u32: Vec<u32> = new_ids.iter().map(|&id| id as u32).collect();
    Ok(GreedyRun {
        generated,
        new_ids,
        text: tokenizer.decode(&generated_u32)?,
        new_text: tokenizer.decode(&new_u32)?,
        seconds,
    })
}

fn teacher_force_decode<B>(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    backend: &str,
    device: &CudaDevice,
) -> Result<(), String>
where
    B: Backend<Device = CudaDevice> + Qwen3_5DenseQuantBackend,
{
    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);

    let prompt_len = 5;
    if FORCED.len() != prompt_len + MAX_NEW_TOKENS {
        return Err(format!(
            "FORCED length {} does not match prompt_len {prompt_len} + MAX_NEW_TOKENS {MAX_NEW_TOKENS}",
            FORCED.len()
        ));
    }

    let input = Tensor::<B, 1, Int>::from_data(&FORCED[..prompt_len], device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(FORCED.len());
    let pos0 = positions::<B>(0, prompt_len, device);
    let mut logits = model.forward_prec(input, pos0, &mut cache, Precision::Bf16);

    let mut top1_agree = 0usize;
    let mut min_margin = f32::INFINITY;
    let mut margin_sum = 0.0f64;

    for step in 0..MAX_NEW_TOKENS {
        let pos = prompt_len + step;
        let forced = FORCED[pos];
        let row = logits_last_row(&logits, "teacher-force")?;
        let leaders = top5(&row);
        if leaders.len() < 2 {
            return Err(format!(
                "teacher-force logits have fewer than 2 entries at pos {pos}"
            ));
        }
        let top1 = leaders[0].0 as i64;
        let margin = leaders[0].1 - leaders[1].1;
        let matched = top1 == forced;
        if matched {
            top1_agree += 1;
        }
        min_margin = min_margin.min(margin);
        margin_sum += margin as f64;
        println!(
            "pos={pos} forced={forced} top1={top1} match={} margin={margin:.6} top5={:?}",
            if matched { 1 } else { 0 },
            leaders
        );

        if step + 1 < MAX_NEW_TOKENS {
            let tok = Tensor::<B, 2, Int>::from_data([[forced]], device);
            let step_pos = positions::<B>(pos, 1, device);
            logits = model.forward_prec(tok, step_pos, &mut cache, Precision::Bf16);
        }
    }

    println!(
        "TF_SUMMARY backend={backend} top1_agree={top1_agree}/{MAX_NEW_TOKENS} min_margin={min_margin:.6} mean_margin={:.6}",
        margin_sum / MAX_NEW_TOKENS as f64
    );
    Ok(())
}

fn print_diff(got: &[i64]) {
    println!("T1 PARITY FAIL: raw generated ids differ from Fusion reference");
    println!("pos\tref_id\tgot_id\tmarker");
    for (pos, &ref_id) in FUSION_REF.iter().enumerate() {
        let got_id = got.get(pos).copied();
        let marker = if got_id == Some(ref_id) { "" } else { "<--" };
        match got_id {
            Some(id) => println!("{pos}\t{ref_id}\t{id}\t{marker}"),
            None => println!("{pos}\t{ref_id}\t<missing>\t{marker}"),
        }
    }
    for (pos, id) in got.iter().enumerate().skip(FUSION_REF.len()) {
        println!("{pos}\t<extra>\t{id}\t<--");
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: T1 qwen35_raw_parity failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let backend = std::env::var("BACKEND").unwrap_or_else(|_| "raw".to_string());
    match backend.to_ascii_lowercase().as_str() {
        "fusion" => run_backend::<Cuda>("fusion"),
        "raw" => run_backend::<CaptureBackend>("raw"),
        other => Err(format!(
            "unsupported BACKEND={other:?}; expected fusion or raw"
        )),
    }
}

fn run_backend<B>(backend: &str) -> Result<(), String>
where
    B: Backend<Device = CudaDevice> + Qwen3_5DenseQuantBackend,
{
    let dir = PathBuf::from(std::env::var("QWEN35_DIR").unwrap_or_else(|_| MODEL_DIR.to_string()));
    let mode = std::env::var("MODE").unwrap_or_else(|_| "freerun".to_string());
    let device = CudaDevice::default();
    let quant = std::env::var("QUANT").unwrap_or_else(|_| "bf16".to_string());
    let quant_mode = if quant.eq_ignore_ascii_case("fp8") {
        println!("QUANT=fp8 requested; skipping T1 raw parity gate (fp8-under-capture is later)");
        return Ok(());
    } else if quant.eq_ignore_ascii_case("bf16") {
        "bf16"
    } else {
        return Err(format!("unsupported QUANT={quant:?}; expected bf16 or fp8"));
    };
    println!("device: {device:?} | backend={backend}");
    println!("mode: {mode}");
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
    let mut model = cfg.init_causal_lm::<B>(&device);

    println!("loading sharded BF16 weights from {dir:?} ...");
    let load_start = Instant::now();
    let report = model
        .load_weights_sharded(&dir)
        .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    let load_s = load_start.elapsed().as_secs_f64();
    println!(
        "load verify: pass={}, mapped_tensors={}, params={}",
        report.pass(),
        report.mapped_tensors,
        report.param_count
    );
    println!("load time: {load_s:.1}s");
    print_mem("memory after load");

    let client = CudaRuntime::client(&device);
    let before = client.memory_usage();
    println!("QWEN35_RAW_PARITY memory after load: {:?}", before);

    match mode.to_ascii_lowercase().as_str() {
        "teacherforce" => {
            println!(
                "teacher-forced decode: forced_tokens={}, prompt_tokens=5, decode_positions={MAX_NEW_TOKENS}, precision=BF16",
                FORCED.len()
            );
            teacher_force_decode::<B>(&model, backend, &device)?;
            print_mem("memory after teacher-force");
            return Ok(());
        }
        "freerun" => {}
        other => {
            return Err(format!(
                "unsupported MODE={other:?}; expected freerun or teacherforce"
            ));
        }
    }

    let (prompt_u32, _) = tokenizer.encode_no_pad(PROMPT)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    println!("prompt: {PROMPT:?}");
    println!("prompt token ids: {prompt_ids:?}");

    println!("prefill + greedy decode: max_new_tokens={MAX_NEW_TOKENS}, precision=BF16");
    let result = greedy_decode::<B>(
        &model,
        &tokenizer,
        &prompt_ids,
        MAX_NEW_TOKENS,
        Precision::Bf16,
        true,
        true,
        &device,
    )?;
    println!("generated new token ids: {:?}", result.new_ids);
    println!("generated all token ids: {:?}", result.generated);

    if result.new_ids != FUSION_REF {
        print_diff(&result.new_ids);
        std::process::exit(1);
    }

    println!("T1 PARITY PASS (16/16)");
    println!("decoded new text: {:?}", result.new_text);
    println!("decode time: {:.1}s", result.seconds);
    println!(
        "decode throughput: {:.2} tok/s",
        MAX_NEW_TOKENS as f64 / result.seconds
    );
    print_mem("memory after generation");
    println!("T1 RAW PARITY: {}", result.text);

    Ok(())
}
