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
    // Host argmax with lowest-index ties. Metal GPU reduce is not run-to-run stable
    // (autotuned kernels + ties); one flipped token diverges greedy decode.
    let values = logits
        .clone()
        .cast(DType::F32)
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read decode logits: {e:?}"))?;
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (idx, &v) in values.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = idx;
        }
    }
    if !best_v.is_finite() {
        return Err(format!(
            "decode logits non-finite at selected vocab index {best_i}: {best_v}"
        ));
    }
    Ok(best_i as i64)
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
    let stream_experts = std::env::var("QWEN35_STREAM_EXPERTS").as_deref() != Ok("0");

    // Prefer the offline-quantized resident core (`<blob>/resident_core.nvfp4`, ~1.3 GB) over the
    // BF16 checkpoint (~3.9 GB at F16). Decode is memory-pressure bound on a 16 GB machine, and
    // this path never materializes the dense weights at all -- quantizing after a dense load does
    // not help, because freed tensors go back to CubeCL's memory pool rather than to the OS.
    // QWEN35_RESIDENT_NVFP4=0 forces the old dense path.
    #[cfg(feature = "cubecl-gpu")]
    let resident_blob_dir: Option<PathBuf> =
        if std::env::var("QWEN35_RESIDENT_NVFP4").as_deref() == Ok("0") {
            None
        } else {
            [
                std::env::var("QWEN35_NVFP4_BLOB_DIR")
                    .ok()
                    .map(PathBuf::from),
                Some(PathBuf::from("models-nvfp4")),
                Some(dir.join("models-nvfp4")),
                Some(dir.clone()),
            ]
            .into_iter()
            .flatten()
            .find(|p| {
                p.join("resident_core.nvfp4").exists() && p.join("resident_manifest.txt").exists()
            })
        };
    #[cfg(not(feature = "cubecl-gpu"))]
    let resident_blob_dir: Option<PathBuf> = None;

    let load_start = Instant::now();
    let quantized_resident = resident_blob_dir.is_some();

    #[cfg(feature = "cubecl-gpu")]
    if let Some(blob_dir) = resident_blob_dir.as_ref() {
        println!(
            "loading QUANTIZED RESIDENT CORE from {} (NVFP4 sidecars, dense weights never allocated) ...",
            blob_dir.display()
        );
        model
            .load_weights_resident_nvfp4(blob_dir, &dir)
            .map_err(|e| format!("load_weights_resident_nvfp4 failed: {e:?}"))?;
    }

    if !quantized_resident {
        let report = if stream_experts {
            println!(
                "loading RESIDENT CORE ONLY from {dir:?} (routed experts streamed on demand) ..."
            );
            model
                .load_weights_sharded_resident_core(&dir)
                .map_err(|e| format!("load_weights_sharded_resident_core failed: {e:?}"))?
        } else {
            println!(
                "loading sharded BF16 weights from {dir:?} (eager, backend-portable path) ..."
            );
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
    }
    println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());

    // Pre-cast resident-core Linear weights to the compute dtype ONCE. `linear3` otherwise
    // re-casts every weight matrix on every forward call (bf16 -> f16), which at decode meant
    // re-casting ~1.0B params per token across the GDN layers alone. Set QWEN35_PRECAST=0 to
    // disable and compare.
    // Skipped entirely on the quantized-resident path: those Linears are `[1,n]` placeholders whose
    // real weights live in NVFP4 sidecars, so casting them would be pure waste.
    if !quantized_resident && std::env::var("QWEN35_PRECAST").as_deref() != Ok("0") {
        let want = match prec {
            Precision::F16 => Some(DType::F16),
            Precision::Bf16 => Some(DType::BF16),
            Precision::F32 => None,
        };
        if let Some(dt) = want {
            let t = Instant::now();
            let n = model.cast_resident_core_linears(dt);
            println!(
                "pre-cast resident-core Linears to {:?}: {} params in {:.2}s",
                dt,
                n,
                t.elapsed().as_secs_f64()
            );
        }
    }

    // Cast the 1 GB token-embedding table off BF16. burn's Metal/wgpu embedding gather has no BF16
    // fast path: an isolated row lookup costs 9.15 ms at BF16 vs 0.35 ms at F16 and 0.011 ms at F32
    // (`examples/embed_dtype_probe.rs`).
    //
    // Default is F16, NOT the microbenchmark-fastest F32: decode here is memory-pressure bound, and
    // F32 doubles this table to ~2 GB, which measured 3.5x SLOWER end-to-end (0.22 vs 0.84 tok/s)
    // with every profile bucket inflating. F16 keeps BF16's footprint. QWEN35_EMBED_DTYPE=f32|f16|bf16.
    {
        let embed_dt = match std::env::var("QWEN35_EMBED_DTYPE").as_deref() {
            Ok("bf16") => None,
            Ok("f32") => Some(DType::F32),
            _ => Some(DType::F16),
        };
        if let Some(dt) = embed_dt {
            let t = Instant::now();
            let n = model.cast_embedding(dt);
            if n > 0 {
                println!(
                    "cast embedding to {:?}: {} params in {:.2}s",
                    dt,
                    n,
                    t.elapsed().as_secs_f64()
                );
            }
        }
    }

    // Quantize the resident core (q/k/v/o_proj, GDN projections, shared-expert MLP) to NVFP4 via
    // sidecars, freeing the F16/bf16 weight afterward. The routed experts stream from
    // models-nvfp4/ already; this is the resident-core half of the same idea (~6.1 GiB F16 ->
    // ~1.5 GiB), which was previously CUDA-only (`quantize_dense_fp8`).
    //
    // OFF BY DEFAULT: naive round-to-nearest NVFP4 (`Nvfp4Linear::from_linear` -> plain
    // `quantize_nvfp4`, no Hadamard rotation / MSE calibration) measurably degrades output quality
    // on the attention/GDN projections -- this file's `fake_quant_*` helpers already use
    // `quantize_nvfp4_mse`/`quantize_nvfp4_clip` + Hadamard rotation for exactly this reason, but
    // that calibrated path isn't wired into the real sidecar quantizer yet. Set
    // QWEN35_QUANTIZE_DENSE_NVFP4=1 to try it anyway (quality is currently visibly worse).
    #[cfg(feature = "cubecl-gpu")]
    if !quantized_resident && std::env::var("QWEN35_QUANTIZE_DENSE_NVFP4").as_deref() == Ok("1") {
        let t = Instant::now();
        let coverage = qwen3_burn::quant_gate::quantize_dense_nvfp4(&mut model, &[]);
        println!(
            "quantize_dense_nvfp4: {}/{} dense linears in {:.2}s",
            coverage.quantized,
            coverage.intended,
            t.elapsed().as_secs_f64()
        );
    }

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
    // 40 layers x 8 experts x 2 projections = 640 slots per token working set.
    // 4096 measures better than 1024 (1.83 vs 1.62 tok/s, hit rate 78% vs 67%) now that the
    // resident core loads pre-quantized: the extra ~3 GB of slot headroom used to be cancelled out
    // by the swap pressure a 3.9 GB F16 resident core created, which is why earlier capacity
    // sweeps came out flat.
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
    if qwen3_burn::qwen3_5::profile::cpu_time_enabled() {
        print!("{}", qwen3_burn::qwen3_5::profile::cpu_report());
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
