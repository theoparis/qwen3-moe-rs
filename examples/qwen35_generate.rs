//! Lane L1.6: load the real Qwen3.6-35B-A3B checkpoint and greedy-decode a fixed prompt.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example qwen35_generate
//!   ./target/release/examples/qwen35_generate

use std::{path::PathBuf, time::Instant};

use cubecl::{cuda::CudaRuntime, Runtime};

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    tensor::{DType, Int, Tensor},
};
use qwen3_burn::{Precision, Qwen3_5MoeConfig, Qwen3Tokenizer};

type B = Cuda;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const NVFP4_MODEL_DIR: &str = "models/qwen3.6-35b-a3b-nvfp4";
const PROMPT: &str = "The capital of France is";
const MAX_NEW_TOKENS: usize = 16;

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

struct GreedyRun {
    generated: Vec<i64>,
    new_ids: Vec<i64>,
    text: String,
    new_text: String,
    seconds: f64,
}

fn greedy_decode(
    model: &qwen3_burn::Qwen3_5MoeForCausalLM<B>,
    tokenizer: &Qwen3Tokenizer,
    prompt_ids: &[i64],
    max_new_tokens: usize,
    prec: Precision,
    fused_moe: bool,
    debug_prefill: bool,
    device: &CudaDevice,
) -> Result<GreedyRun, String> {
    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(fused_moe);

    let prompt_len = prompt_ids.len();
    let total = prompt_len + max_new_tokens;
    let input = Tensor::<B, 1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);
    let gen_start = Instant::now();
    let pos0 = positions(0, prompt_len, device);
    // Greedy decode only needs the LAST position's logits. `forward_last_logits` slices the last
    // hidden BEFORE the head, keeping a quantized (NVFP4/fp8) lm_head at M=1 within its m_max on
    // prefill (T>1) exactly as on decode; identical to slicing full-T logits on the bf16/fp8 heads.
    let mut logits: Tensor<B, 3> = model
        .forward_last_logits(input, pos0, &mut cache, prec)
        .unsqueeze_dim(1);
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
            let pos = positions(prompt_len + step, 1, device);
            logits = model
                .forward_last_logits(tok, pos, &mut cache, prec)
                .unsqueeze_dim(1);
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

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: L1.6 qwen35_generate failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let device = CudaDevice::default();
    let quant = std::env::var("QUANT").unwrap_or_else(|_| "bf16".to_string());
    // QUANT=nvfp4 loads the official NVIDIA NVFP4 checkpoint via load_nvidia_nvfp4 (raw dispatch);
    // NVFP4_DEQUANT_TO_FP8=1 selects the staged B5.0c fp8 fallback inside the loader.
    let quant_mode = if quant.eq_ignore_ascii_case("fp8") {
        "fp8"
    } else if quant.eq_ignore_ascii_case("bf16") {
        "bf16"
    } else if quant.eq_ignore_ascii_case("nvfp4") {
        "nvfp4"
    } else {
        return Err(format!(
            "unsupported QUANT={quant:?}; expected bf16, fp8, or nvfp4"
        ));
    };
    let default_dir = if quant_mode == "nvfp4" {
        NVFP4_MODEL_DIR
    } else {
        MODEL_DIR
    };
    let dir =
        PathBuf::from(std::env::var("QWEN35_DIR").unwrap_or_else(|_| default_dir.to_string()));
    println!("device: {device:?}");
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

    let client = CudaRuntime::client(&device);
    if quant_mode == "nvfp4" {
        // Official NVIDIA NVFP4 checkpoint: quantized bytes are ingested straight into the sidecars
        // (no separate bf16 load + quantize step). load_nvidia_nvfp4 prints its own per-stage host/
        // device memory instrumentation ([load_nvidia_nvfp4] stage=...).
        let raw = std::env::var("NVFP4_DEQUANT_TO_FP8").ok().as_deref() != Some("1");
        println!(
            "loading NVIDIA NVFP4 checkpoint from {dir:?} (mode={}) ...",
            if raw {
                "raw NVFP4 dispatch"
            } else {
                "staged fp8 fallback"
            }
        );
        let load_start = Instant::now();
        #[cfg(feature = "cuda")]
        model
            .load_nvidia_nvfp4(&dir)
            .map_err(|e| format!("load_nvidia_nvfp4 failed: {e:?}"))?;
        println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());
        print_mem("memory after load");
        println!(
            "QWEN35_GENERATE device memory after nvfp4 load: {:?}",
            client.memory_usage()
        );
    } else {
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

        let before = client.memory_usage();
        println!("QWEN35_GENERATE memory before quant: {:?}", before);
        if quant_mode == "fp8" {
            #[cfg(feature = "cuda")]
            {
                qwen3_burn::quant_gate::quantize_dense_fp8(&mut model, &[]);
                qwen3_burn::quant_gate::quantize_experts_fp8(&mut model, &[]);
            }
        }
        let after = client.memory_usage();
        println!("QWEN35_GENERATE memory after quant: {:?}", after);
    }

    let (prompt_u32, _) = tokenizer.encode_no_pad(PROMPT)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    println!("prompt: {PROMPT:?}");
    println!("prompt token ids: {prompt_ids:?}");

    // Precision toggle: QWEN35_PREC=f32 forces f32 matmuls (weights cast on-the-fly) to test whether
    // the incoherent output is bf16 matmul-accumulation error vs a real semantic bug. Default bf16.
    let prec = match std::env::var("QWEN35_PREC").as_deref() {
        Ok("f32") | Ok("F32") => Precision::F32,
        _ => Precision::Bf16,
    };
    println!("prefill + greedy decode: max_new_tokens={MAX_NEW_TOKENS}, precision={prec:?}");

    if quant_mode == "fp8" && std::env::var("COMPARE_FP8_FUSED_HOST").as_deref() == Ok("1") {
        println!("FP8 fused-vs-host gate: crossing note T<=16 uses fused path, T>16 remains host-loop");
        let host = greedy_decode(
            &model,
            &tokenizer,
            &prompt_ids,
            MAX_NEW_TOKENS,
            prec,
            false,
            true,
            &device,
        )?;
        let fused = greedy_decode(
            &model,
            &tokenizer,
            &prompt_ids,
            MAX_NEW_TOKENS,
            prec,
            true,
            false,
            &device,
        )?;
        #[cfg(feature = "cuda")]
        qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);
        if host.new_ids != fused.new_ids {
            return Err(format!(
                "FP8 fused-vs-host greedy token mismatch: host={:?} fused={:?}",
                host.new_ids, fused.new_ids
            ));
        }
        println!("FP8_FUSED_VS_HOST PASS token-identical");
        println!("host-loop new token ids: {:?}", host.new_ids);
        println!("fused new token ids: {:?}", fused.new_ids);
        println!(
            "host-loop decode: {:.1}s {:.2} tok/s",
            host.seconds,
            MAX_NEW_TOKENS as f64 / host.seconds
        );
        println!(
            "fused decode: {:.1}s {:.2} tok/s",
            fused.seconds,
            MAX_NEW_TOKENS as f64 / fused.seconds
        );
        println!("decoded new text: {:?}", fused.new_text);
        println!("L1.6 GENERATE: {}", fused.text);
        print_mem("memory after fused-vs-host gate");
        return Ok(());
    }

    let fused_moe = std::env::var("QWEN35_FUSED_MOE").as_deref() != Ok("0");
    let result = greedy_decode(
        &model,
        &tokenizer,
        &prompt_ids,
        MAX_NEW_TOKENS,
        prec,
        fused_moe,
        true,
        &device,
    )?;
    println!("generated new token ids: {:?}", result.new_ids);
    println!("generated all token ids: {:?}", result.generated);
    println!("decoded new text: {:?}", result.new_text);
    println!("decode time: {:.1}s", result.seconds);
    println!(
        "decode throughput: {:.2} tok/s",
        MAX_NEW_TOKENS as f64 / result.seconds
    );
    print_mem("memory after generation");
    if repeated_garbage(&result.new_ids) {
        println!("CRITICAL: repeated-token symptom in generated ids: {:?}", result.new_ids);
    }
    println!("L1.6 GENERATE: {}", result.text);

    Ok(())
}
