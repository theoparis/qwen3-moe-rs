//! Lane L1.1 gate for Qwen3.6-35B-A3B: config + module tree + strict weight-map shape coverage.
//!
//! Default mode is header-only verification so it can run on machines that cannot resident-load the
//! 71 GB bf16 checkpoint. Pass `--materialize` to also copy non-visual tensors into the lazy module
//! tree.

use std::path::PathBuf;

use burn::prelude::Device;
use qwen3_burn::{Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM};

type B = Cuda;

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter()
        .position(|x| x == flag)
        .and_then(|i| args.get(i + 1))
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|x| x == flag)
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        arg(&args, "--dir")
            .cloned()
            .unwrap_or_else(|| "models/qwen3.6-35b-a3b".to_string()),
    );
    let materialize = has_flag(&args, "--materialize");

    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    let linear_layers = cfg
        .layer_types
        .iter()
        .filter(|&&kind| kind == qwen3_burn::Qwen3_5LayerType::LinearAttention)
        .count();
    let full_layers = cfg.num_hidden_layers - linear_layers;
    println!(
        "config: {} layers ({} GDN, {} full), hidden {}, vocab {}, experts {} top-{}, MTP {}",
        cfg.num_hidden_layers,
        linear_layers,
        full_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.mtp_num_hidden_layers
    );
    println!(
        "GDN dims: key_heads {} key_dim {}, value_heads {} value_dim {}, conv {}, partial_rotary {}, rope_theta {}, mrope {:?}",
        cfg.linear_num_key_heads,
        cfg.linear_key_head_dim,
        cfg.linear_num_value_heads,
        cfg.linear_value_head_dim,
        cfg.linear_conv_kernel_dim,
        cfg.partial_rotary_factor,
        cfg.rope_theta,
        cfg.mrope_section
    );

    let device = Device::cuda(0);
    let mut model = cfg.init_causal_lm(&device);
    println!(
        "module tree: embed_tokens + {} dispatched layers + final norm + untied lm_head + MTP block",
        model.model.layers.len()
    );

    let report = if materialize {
        println!("mode: materializing non-visual tensors into the lazy module tree");
        model
            .load_weights_sharded(&dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?
    } else {
        println!("mode: safetensors header shape verification (no tensor data allocation)");
        Qwen3_5MoeForCausalLM::<B>::verify_weights_sharded(&cfg, &dir)
            .map_err(|e| format!("verify_weights_sharded failed: {e:?}"))?
    };

    println!(
        "weight_map: {} tensors; mapped text/MTP/lm_head: {}; skipped visual: {}",
        report.weight_map_tensors, report.mapped_tensors, report.skipped_visual_tensors
    );
    println!(
        "param_count: {} ({:.6}B); skipped visual params: {} ({:.6}B)",
        report.param_count,
        report.param_count as f64 / 1e9,
        report.skipped_visual_param_count,
        report.skipped_visual_param_count as f64 / 1e9
    );

    if !report.missing.is_empty() {
        eprintln!("missing tensors ({}):", report.missing.len());
        for item in &report.missing {
            eprintln!("  {item}");
        }
    }
    if !report.orphan.is_empty() {
        eprintln!("orphan tensors ({}):", report.orphan.len());
        for item in &report.orphan {
            eprintln!("  {item}");
        }
    }
    if !report.shape_mismatches.is_empty() {
        eprintln!("shape mismatches ({}):", report.shape_mismatches.len());
        for item in &report.shape_mismatches {
            eprintln!("  {item}");
        }
    }

    if !report.pass() {
        return Err("L1.1 load verification failed".to_string());
    }
    let params_b = report.param_count as f64 / 1e9;
    if !(35.0..=36.0).contains(&params_b) {
        return Err(format!(
            "unexpected non-visual parameter count: {params_b:.6}B"
        ));
    }

    println!("L1.1 LOAD-VERIFY: PASS");
    Ok(())
}
