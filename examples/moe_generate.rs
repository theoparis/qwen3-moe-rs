//! Load a REAL Qwen3-MoE checkpoint (per-expert HF safetensors format) and generate text on the GB10.
//! Validates the full pipeline on pretrained weights: the loader key-remap, sharded load, the MoE
//! forward (Tier-1 oracle), and greedy generation. The config is read from the model's `config.json`,
//! so it works for any per-expert Qwen3-MoE (e.g. Qwen3-15B-A2B, Qwen3-30B-A3B).
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example moe_generate -- \
//!     --dir models/qwen3-15b-a2b --prompt "The capital of France is" --max-tokens 48

use std::path::PathBuf;

use burn::prelude::Device;
use burn::tensor::{Device, Int, Tensor};
use qwen3_burn::{Qwen3MoeConfig, Qwen3Tokenizer};

type B = Cuda;

fn arg<'a>(a: &'a [String], f: &str) -> Option<&'a String> {
    a.iter().position(|x| x == f).and_then(|i| a.get(i + 1))
}

/// Build a `Qwen3MoeConfig` from a HuggingFace `config.json`.
fn config_from_hf(dir: &PathBuf) -> Result<Qwen3MoeConfig, String> {
    let txt = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&txt).map_err(|e| format!("parse config.json: {e}"))?;
    let u = |k: &str, d: u64| -> usize { v.get(k).and_then(|x| x.as_u64()).unwrap_or(d) as usize };
    let f = |k: &str, d: f64| -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(d) };
    let mut cfg = Qwen3MoeConfig::new()
        .with_vocab_size(u("vocab_size", 151936))
        .with_hidden_size(u("hidden_size", 2048))
        .with_num_hidden_layers(u("num_hidden_layers", 24))
        .with_num_attention_heads(u("num_attention_heads", 32))
        .with_num_key_value_heads(u("num_key_value_heads", 4))
        .with_num_experts(u("num_experts", 128))
        .with_num_experts_per_tok(u("num_experts_per_tok", 8))
        .with_moe_intermediate_size(u("moe_intermediate_size", 768))
        .with_rms_norm_eps(f("rms_norm_eps", 1e-6))
        .with_rope_theta(f("rope_theta", 1_000_000.0))
        .with_max_position_embeddings(u("max_position_embeddings", 40960));
    if let Some(hd) = v.get("head_dim").and_then(|x| x.as_u64()) {
        cfg = cfg.with_head_dim(Some(hd as usize));
    }
    if let Some(n) = v.get("norm_topk_prob").and_then(|x| x.as_bool()) {
        cfg = cfg.with_norm_topk_prob(n);
    }
    Ok(cfg)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        arg(&args, "--dir")
            .cloned()
            .unwrap_or_else(|| "models/qwen3-15b-a2b".into()),
    );
    let prompt = arg(&args, "--prompt")
        .cloned()
        .unwrap_or_else(|| "The capital of France is".into());
    let max_tokens: usize = arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);

    let device = Device::cuda(0);
    println!("device: {device:?}");

    let cfg = config_from_hf(&dir)?;
    println!(
        "config: {} layers, hidden {}, {} experts top-{}, moe_inter {}, untied head",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm(&device);

    println!("loading sharded weights from {dir:?} ...");
    let t0 = std::time::Instant::now();
    model
        .load_weights_sharded(&dir)
        .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!(
        "loaded {} layers in {:.1}s",
        model.num_layers(),
        t0.elapsed().as_secs_f64()
    );

    let (ids_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let input: Tensor<1, Int> = Tensor::from_data(ids.as_slice(), &device);
    let input: Tensor<2, Int> = input.unsqueeze();

    println!("\n--- prompt ---\n{prompt}");
    // Report the ACTUAL MoE dispatch (mirrors `Qwen3MoeSparseBlock::forward` env-toggle order in
    // src/moe.rs): QWEN3_MOE_ONDEVICE wins, then QWEN3_MOE_ROUTED, else the dense oracle.
    let moe_path = if std::env::var("QWEN3_MOE_ONDEVICE").is_ok() {
        "on-device routed (no host sync)"
    } else if std::env::var("QWEN3_MOE_ROUTED").is_ok() {
        "host token-routing"
    } else {
        "dense oracle (all experts)"
    };
    println!(
        "generating {max_tokens} tokens (greedy, MoE path: {moe_path}; {} experts top-{}/layer)...",
        cfg.num_experts, cfg.num_experts_per_tok
    );
    let start = std::time::Instant::now();
    let out = model.generate_greedy(input, max_tokens, &[151643, 151645]);
    let out_ids: Vec<i64> = out
        .cast(burn::tensor::DType::I64)
        .to_data()
        .to_vec()
        .map_err(|e| format!("read output: {e:?}"))?;
    let out_u32: Vec<u32> = out_ids.iter().map(|&x| x as u32).collect();
    let text = tokenizer.decode(&out_u32)?;

    println!(
        "\n===== GENERATION ({:.1}s) =====\n{text}\n==============================",
        start.elapsed().as_secs_f64()
    );
    Ok(())
}
