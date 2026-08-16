//! Portable variant of `qwen35_generate` for non-CUDA backends (flex/wgpu/vulkan/metal).
//!
//! `qwen35_generate` hardcodes `Device::cuda(0)` plus CUDA-only memory diagnostics. This example
//! drives the same real Qwen3.5/3.6-MoE checkpoint load + greedy-decode path through the
//! backend-generic `Device`/`Tensor` API. On `--device metal|wgpu|vulkan`, streamed experts
//! (QWEN35_STREAM_EXPERTS=1) pack to NVFP4 and run the CubeCL decode-GEMV kernel; set
//! QWEN35_STREAM_NVFP4=0 to keep the previous f16/bf16 matmul path.
//!
//! Build/run:
//!   cargo run --release --example qwen35_generate_portable --features metal -- --device metal
//!   cargo run --release --example qwen35_generate_portable --features wgpu  -- --device wgpu
//!
//! Env:
//!   QWEN35_DIR=models          (default "models", pointed at a dir with config.json + tokenizer.json
//!                                + model-*-of-*.safetensors + model.safetensors.index.json)
//!   QWEN35_PREC=f32|bf16       (default bf16)
//!   QWEN35_MAX_NEW_TOKENS=16   (default 16)
//!   QWEN35_PROMPT="..."        (default "The capital of France is")

use std::{collections::BTreeMap, path::PathBuf, time::Instant};

use burn::prelude::Device;
#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
use burn::tensor::DeviceKind;
use burn::tensor::{DType, Int, Tensor};
use qwen3_burn::expert_stream::ExpertSlotPool;
use qwen3_burn::qwen3_5::parse_weight_map;
use qwen3_burn::{Precision, Qwen3_5MoeConfig, Qwen3Tokenizer};

fn positions(start: usize, len: usize, device: &Device) -> Tensor<2, Int> {
    if len == 1 {
        Tensor::<2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn assert_logits_all_finite(logits: &Tensor<2>, what: &str) -> Result<(), String> {
    let values = logits
        .clone()
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

fn argmax_last(logits: &Tensor<2>) -> Result<i64, String> {
    assert_logits_all_finite(logits, "decode")?;
    let ids = logits
        .clone()
        .argmax(1)
        .cast(DType::I64)
        .into_data()
        .to_vec::<i64>()
        .map_err(|e| format!("read argmax token: {e:?}"))?;
    ids.first()
        .copied()
        .ok_or_else(|| "argmax returned no token".to_string())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: qwen35_generate_portable failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let device_name = args
        .iter()
        .position(|x| x == "--device")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "flex".to_string());
    let device = match device_name.as_str() {
        "flex" => Device::flex(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "metal")]
        "metal" => Device::metal(DeviceKind::DefaultDevice),
        other => {
            return Err(format!(
                "unknown or unbuilt --device {other:?} (build with --features wgpu/vulkan/metal to enable it)"
            ));
        }
    };
    println!("device: {device:?}");

    let dir = PathBuf::from(std::env::var("QWEN35_DIR").unwrap_or_else(|_| "models".to_string()));
    let prec = match std::env::var("QWEN35_PREC").as_deref() {
        Ok("f32") | Ok("F32") => Precision::F32,
        // f16: confirmed correct + working on Metal (unlike bf16, which Metal's matmul autotune
        // has no kernel for). Roughly half the compute/cast memory of f32 for streamed experts.
        Ok("f16") | Ok("F16") => Precision::F16,
        Ok("bf16") | Ok("Bf16") | Ok("BF16") => Precision::Bf16,
        // Default to f16, not bf16: bf16 is CUDA-validated only and crashes on Metal (no autotune
        // kernel for BF16 lhs/rhs). f16 is confirmed correct + faster + lower-memory than f32 on
        // Metal (examples/f16_matmul_probe.rs, streamed-decode measurements in
        // docs/MEMORY_STREAMING_PLAN.md) and should be fine on CPU/wgpu too; opt into bf16
        // explicitly only on backends where it's known to work (CUDA).
        _ => Precision::F16,
    };
    let max_new_tokens: usize = std::env::var("QWEN35_MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let prompt =
        std::env::var("QWEN35_PROMPT").unwrap_or_else(|_| "The capital of France is".to_string());

    // QWEN35_PROFILE=1 attributes decode wall-clock across embed / attention / router / MoE /
    // lm_head, syncing at each boundary (so a profiled run is slower, but the split is real).
    qwen3_burn::qwen3_5::profile::init_from_env();

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

    // QWEN35_STREAM_EXPERTS=1 loads only the resident core (skips the 68GB-scale routed-expert
    // stacks) and fetches each step's top-k experts on demand through an ExpertSlotPool instead —
    // see docs/MEMORY_STREAMING_PLAN.md. Falls back to the original fully-resident eager load.
    let stream_experts = std::env::var("QWEN35_STREAM_EXPERTS").as_deref() == Ok("1");
    let load_start = Instant::now();
    let report = if stream_experts {
        println!("loading RESIDENT CORE ONLY from {dir:?} (routed experts streamed on demand) ...");
        model
            .load_weights_sharded_resident_core(&dir)
            .map_err(|e| format!("load_weights_sharded_resident_core failed: {e:?}"))?
    } else {
        println!("loading sharded BF16 weights from {dir:?} (eager, backend-portable path) ...");
        model
            .load_weights_sharded(&dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?
    };
    println!(
        "load verify: pass={}, mapped_tensors={}, params={}",
        report.pass(),
        report.mapped_tensors,
        report.param_count
    );
    println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());

    let (prompt_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    println!("prompt: {prompt:?}");
    println!("prompt token ids: {prompt_ids:?}");
    println!("prefill + greedy decode: max_new_tokens={max_new_tokens}, precision={prec:?}");

    let prompt_len = prompt_ids.len();
    let total = prompt_len + max_new_tokens;
    let input = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).unsqueeze();
    let mut cache = model.model.new_cache_with_capacity(total);

    // Pool capacity, in (layer, projection, expert) slots.
    //
    // The previous default of 64 came from an era when (a) the LRU touch was O(capacity), and (b)
    // slots held dense bf16 tensors, so a big pool cost a lot of memory and bought nothing. Both
    // premises are now false: the LRU is O(1), and NVFP4 slots are ~4x smaller (~1 MB per gate_up,
    // ~0.5 MB per down on this model). The old "hits=0 regardless of capacity" note is also no
    // longer what the counters show.
    //
    // Measured on the real 68 GB checkpoint, 16 tokens, Metal / M2 Pro, identical output text:
    //   capacity=64   -> hits=11643 misses=12029, upload=104.0s, decode=194.5s (0.08 tok/s)
    //   capacity=4096 -> hits=19054 misses= 4618, upload= 38.4s, decode=143.4s (0.11 tok/s)
    // i.e. 64 slots (~32 experts across ALL 40 layers) self-evicts constantly and re-pays the
    // expensive expert repack ~2.6x more often than it needs to. 4096 slots costs roughly 3 GB of
    // NVFP4 expert residency; lower it on a memory-tight machine, raise it if `misses` stays high.
    let pool_capacity: usize = std::env::var("QWEN35_STREAM_POOL_CAPACITY")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4096);
    let index_map: BTreeMap<String, String>;
    let mut pool_holder: Option<ExpertSlotPool> = None;
    if stream_experts {
        let index_path = dir.join("model.safetensors.index.json");
        let text = std::fs::read_to_string(&index_path)
            .map_err(|e| format!("read {}: {e}", index_path.display()))?;
        let pairs = parse_weight_map(&text).map_err(|e| format!("parse weight_map: {e}"))?;
        index_map = pairs.into_iter().collect();
        pool_holder = Some(ExpertSlotPool::new(&dir, &index_map, pool_capacity));
        println!("streamed expert pool: capacity={pool_capacity} slots");
        #[cfg(feature = "cubecl-gpu")]
        if std::env::var("QWEN35_STREAM_NVFP4").ok().as_deref() != Some("0") {
            println!("streamed experts: NVFP4 CubeCL GEMV (set QWEN35_STREAM_NVFP4=0 to disable)");
        }
    }

    let gen_start = Instant::now();
    let pos0 = positions(0, prompt_len, &device);
    let mut logits = if let Some(pool) = pool_holder.as_mut() {
        model.forward_last_logits_streamed(input, pos0, &mut cache, prec, pool)
    } else {
        model.forward_last_logits(input, pos0, &mut cache, prec)
    };
    assert_logits_all_finite(&logits, "prefill")?;

    let mut generated = prompt_ids.clone();
    let mut new_ids = Vec::with_capacity(max_new_tokens);
    for step in 0..max_new_tokens {
        let id = argmax_last(&logits)?;
        generated.push(id);
        new_ids.push(id);

        if step + 1 < max_new_tokens {
            let tok = Tensor::<2, Int>::from_data([[id]], &device);
            let pos = positions(prompt_len + step, 1, &device);
            logits = if let Some(pool) = pool_holder.as_mut() {
                model.forward_last_logits_streamed(tok, pos, &mut cache, prec, pool)
            } else {
                model.forward_last_logits(tok, pos, &mut cache, prec)
            };
        }
    }
    let seconds = gen_start.elapsed().as_secs_f64();
    if let Some(pool) = pool_holder.as_ref() {
        println!(
            "streamed pool final stats: hits={} misses={} resident_slots={}",
            pool.hits,
            pool.misses,
            pool.resident_slots()
        );
        println!("streamed pool timing: {}", pool.timing_report());
    }
    if qwen3_burn::qwen3_5::profile::enabled() {
        print!("{}", qwen3_burn::qwen3_5::profile::report());
    }

    let generated_u32: Vec<u32> = generated.iter().map(|&id| id as u32).collect();
    let new_u32: Vec<u32> = new_ids.iter().map(|&id| id as u32).collect();
    println!("generated new token ids: {new_ids:?}");
    println!("generated all token ids: {generated:?}");
    println!("decoded new text: {:?}", tokenizer.decode(&new_u32)?);
    println!("decode time: {seconds:.1}s");
    println!(
        "decode throughput: {:.2} tok/s",
        max_new_tokens as f64 / seconds
    );
    println!("GENERATE: {}", tokenizer.decode(&generated_u32)?);

    Ok(())
}
