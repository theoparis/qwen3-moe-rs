//! End-to-end plumbing check for `docs/MEMORY_STREAMING_PLAN.md`'s streamed path, one layer deep:
//! loads the REAL checkpoint's resident core (everything except routed-expert stacks, via
//! `load_weights_sharded_resident_core`), embeds a real prompt with the real tokenizer + real
//! embedding table, runs that layer's REAL router (`route_topk`, real gate weights) to get real
//! top-k routed expert indices/weights, then fetches exactly those experts on demand through
//! `ExpertSlotPool` and combines them with the router's weights.
//!
//! Caveat: this feeds raw embeddings (not the true post-attention hidden state) into the chosen
//! layer's router, since a full 40-layer forward would need every layer's experts streamed (the next
//! integration step, not this one). This is a plumbing/memory validation — router selection, on-demand
//! multi-expert fetch, and weighted combine all running against real weights with bounded RAM — not a
//! full-model output correctness check.
//!
//! Usage:
//!   cargo run --release --example moe_block_stream_probe -- [dir] [layer] [prompt]

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::prelude::Device;
use burn::tensor::{Int, Tensor};
use qwen3_burn::expert_stream::ExpertSlotPool;
use qwen3_burn::qwen3_5::{Qwen3_5DecoderLayer, Qwen3_5MoeConfig, parse_weight_map};
use qwen3_burn::{Precision, Qwen3Tokenizer};

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    let layer_idx: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let prompt = args
        .get(3)
        .cloned()
        .unwrap_or_else(|| "The capital of France is".to_string());

    let device = Device::flex();
    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    println!(
        "config: {} layers, hidden {}, experts top-{}/{}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.num_experts_per_tok, cfg.num_experts
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm(&device);

    println!("loading resident core from {dir:?} (routed experts skipped) ...");
    let t0 = std::time::Instant::now();
    let report = model
        .load_weights_sharded_resident_core(&dir)
        .map_err(|e| format!("load_weights_sharded_resident_core failed: {e:?}"))?;
    println!(
        "resident-core load: pass={}, mapped_tensors={}, params={}, took {:.1}s",
        report.pass(),
        report.mapped_tensors,
        report.param_count,
        t0.elapsed().as_secs_f64()
    );

    let (prompt_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = prompt_u32.iter().map(|&id| id as i64).collect();
    println!("prompt: {prompt:?} -> {} tokens", prompt_ids.len());

    let input: Tensor<2, Int> =
        Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).unsqueeze();
    let hidden_states: Tensor<3> = model.model.embed_tokens.forward(input);
    let [batch, seq_len, hidden] = hidden_states.dims();
    println!("embeddings: [{batch}, {seq_len}, {hidden}]");

    let layer = model.model.layers.get(layer_idx).ok_or_else(|| {
        format!(
            "layer {layer_idx} out of range ({} layers)",
            model.model.layers.len()
        )
    })?;
    let block = match layer {
        Qwen3_5DecoderLayer::Linear(l) => &l.mlp,
        Qwen3_5DecoderLayer::Full(l) => &l.mlp,
    };

    let top_k = cfg.num_experts_per_tok.min(cfg.num_experts);
    let (sel_idx, sel_w) = block.route_topk(hidden_states.clone(), top_k);
    let tokens = batch * seq_len;
    let idx_host: Vec<i32> = sel_idx
        .into_data()
        .to_vec::<i32>()
        .map_err(|e| format!("read sel_idx: {e:?}"))?;
    let w_host: Vec<f32> = sel_w
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read sel_w: {e:?}"))?;
    println!("router picked top-{top_k} experts per token (layer {layer_idx}):");
    for t in 0..tokens {
        let picks: Vec<(i32, f32)> = (0..top_k)
            .map(|k| (idx_host[t * top_k + k], w_host[t * top_k + k]))
            .collect();
        println!("  token {t}: {picks:?}");
    }

    let index_path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("read {}: {e}", index_path.display()))?;
    let pairs = parse_weight_map(&text).map_err(|e| format!("parse weight_map: {e}"))?;
    let index: BTreeMap<String, String> = pairs.into_iter().collect();

    // Capacity = distinct experts actually selected across this batch (bounded, not `num_experts`).
    let mut distinct: Vec<usize> = idx_host.iter().map(|&i| i as usize).collect();
    distinct.sort_unstable();
    distinct.dedup();
    let mut pool = ExpertSlotPool::new(&dir, &index, distinct.len().max(1));
    println!(
        "streaming {} distinct expert(s) for this batch ...",
        distinct.len()
    );

    let x2 = hidden_states.reshape([tokens, hidden]);
    let mut combined = Tensor::<2>::zeros([tokens, hidden], &device);
    for t in 0..tokens {
        for k in 0..top_k {
            let expert = idx_host[t * top_k + k] as usize;
            let w = w_host[t * top_k + k];
            let x_tok = x2.clone().slice([t..t + 1, 0..hidden]);
            let out = pool.expert_forward(layer_idx, expert, x_tok, Precision::Bf16, &device)?;
            let updated = combined.clone().slice([t..t + 1, 0..hidden]) + out * w;
            combined = combined.slice_assign([t..t + 1, 0..hidden], updated);
        }
    }

    println!(
        "streamed pool: hits={} misses={} resident_slots={} (capacity={})",
        pool.hits,
        pool.misses,
        pool.resident_slots(),
        distinct.len()
    );

    let finite = combined
        .clone()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read combined: {e:?}"))?
        .iter()
        .all(|v| v.is_finite());
    if !finite {
        return Err("combined routed output contains non-finite values".to_string());
    }
    println!(
        "PASS: streamed router + multi-expert fetch + weighted combine produced finite output"
    );
    Ok(())
}
