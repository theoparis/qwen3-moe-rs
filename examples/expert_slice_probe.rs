//! Validates `ShardReader::read_expert_slice` (Phase 1 of `docs/MEMORY_STREAMING_PLAN.md`) against
//! a real checkpoint: for a handful of (layer, expert) picks, slices just that expert's bytes via
//! `read_expert_slice` and compares them byte-for-byte against the corresponding sub-range of the
//! full tensor read by `read_raw_tensor`. Deliberately reads only a few MB total — never loads the
//! full model — so this is safe to run on machines that can't fit the whole checkpoint in RAM.
//!
//! Usage:
//!   cargo run --release --example expert_slice_probe -- [dir]   (default "models")

use std::collections::BTreeMap;
use std::path::PathBuf;

use qwen3_burn::nvidia_ckpt::ShardReader;
use qwen3_burn::qwen3_5::parse_weight_map;

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));

    let index_path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("read {}: {e}", index_path.display()))?;
    let pairs = parse_weight_map(&text).map_err(|e| format!("parse weight_map: {e}"))?;
    let index: BTreeMap<String, String> = pairs.into_iter().collect();

    let keys_to_probe: Vec<&String> = index
        .keys()
        .filter(|k| k.ends_with("experts.gate_up_proj") || k.ends_with("experts.down_proj"))
        .take(4)
        .collect();
    if keys_to_probe.is_empty() {
        return Err("no experts.gate_up_proj/down_proj keys found in weight_map".to_string());
    }

    let mut reader = ShardReader::new(&dir, &index);
    let expert_ids = [0usize, 1, 5, 255];

    for key in &keys_to_probe {
        let full = reader.read_raw_tensor(key)?;
        let num_experts = full.shape[0];
        let stride = full.data.len() / num_experts;
        println!(
            "{key}: shape={:?} dtype={:?} num_experts={num_experts} bytes/expert={stride}",
            full.shape, full.dtype
        );

        for &e in expert_ids.iter().filter(|&&e| e < num_experts) {
            let slice = reader.read_expert_slice(key, e)?;
            let expected = &full.data[e * stride..(e + 1) * stride];
            if slice.data.as_slice() != expected {
                return Err(format!(
                    "{key} expert {e}: MISMATCH ({} bytes vs {} bytes, or content differs)",
                    slice.data.len(),
                    expected.len()
                ));
            }
            if slice.shape[0] != 1 || slice.shape[1..] != full.shape[1..] {
                return Err(format!(
                    "{key} expert {e}: unexpected slice shape {:?} (full shape {:?})",
                    slice.shape, full.shape
                ));
            }
        }
        println!(
            "  {} expert slice(s) byte-identical to full-tensor sub-range \u{2713}",
            expert_ids.iter().filter(|&&e| e < num_experts).count()
        );
    }

    println!(
        "PASS: read_expert_slice matches read_raw_tensor for all probed (layer, expert) pairs"
    );
    Ok(())
}
