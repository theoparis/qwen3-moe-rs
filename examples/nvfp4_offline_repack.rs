//! Pre-quantize every routed MoE expert to NVFP4 once, into a fixed-stride mmap-ready store.
//!
//! Decode currently re-derives each expert from the bf16 checkpoint on every cache miss: ~5.0 ms of
//! cold SSD read (the 68 GiB source can never be page-cache resident in 16 GiB) plus ~8.6 ms of CPU
//! transpose+quantize. That dominates the run; the expert math is only ~0.4 ms. Quantization is a
//! pure function of the checkpoint, so it belongs offline. See `src/nvfp4_blob.rs` for the layout
//! and `docs/metal-streamed-decode-findings.md` for the measurements.
//!
//! Output is ~17.6 GiB for Qwen3.6-35B-A3B (vs 60 GiB of bf16 experts), which should stay resident
//! in the page cache and remove the per-miss SSD latency as well as the CPU work.
//!
//! Usage:
//!   cargo run --release --example nvfp4_offline_repack -- --src models --out models-nvfp4
//!   # resumable: re-running skips layers already written with a matching size
//!
//! Then point decode at it with `QWEN35_NVFP4_BLOB_DIR=models-nvfp4`.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use qwen3_burn::nvfp4::quantize_nvfp4_from_nk_bf16;
use qwen3_burn::nvfp4_blob::{BlobManifest, BlobProj, ProjLayout, write_record};
use qwen3_burn::nvidia_ckpt::ShardReader;

fn arg(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .map(|i| args[i + 1].clone())
}

/// Read `model.safetensors.index.json`'s `weight_map`.
fn load_index(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let path = dir.join("model.safetensors.index.json");
    let text =
        std::fs::read_to_string(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let v: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
    let map = v
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| format!("{} has no weight_map object", path.display()))?;
    Ok(map
        .iter()
        .filter_map(|(k, val)| val.as_str().map(|s| (k.clone(), s.to_string())))
        .collect())
}

fn gate_up_key(layer: usize) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj")
}

fn down_key(layer: usize) -> String {
    format!("model.language_model.layers.{layer}.mlp.experts.down_proj")
}

fn main() -> Result<(), String> {
    let src = PathBuf::from(arg("--src").unwrap_or_else(|| "models".into()));
    let out = PathBuf::from(arg("--out").unwrap_or_else(|| "models-nvfp4".into()));
    std::fs::create_dir_all(&out).map_err(|e| format!("create {}: {e}", out.display()))?;

    let cfg_text = std::fs::read_to_string(src.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let cfg: serde_json::Value =
        serde_json::from_str(&cfg_text).map_err(|e| format!("parse config.json: {e}"))?;
    let text_cfg = cfg.get("text_config").unwrap_or(&cfg);
    let usize_of = |k: &str| -> Result<usize, String> {
        text_cfg
            .get(k)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("config.json missing {k}"))
    };
    let num_layers = usize_of("num_hidden_layers")?;
    let num_experts = usize_of("num_experts")?;
    let hidden = usize_of("hidden_size")?;
    let inner = usize_of("moe_intermediate_size")?;

    // gate_up is [2*inner, hidden]; down is [hidden, inner]. Both stored [N,K] row-major.
    let manifest = BlobManifest::new(num_layers, num_experts, 2 * inner, hidden, hidden, inner);
    let layer_len = manifest.layer_file_len();
    println!(
        "source {}\ntarget {}\n{num_layers} layers x {num_experts} experts, gate_up [{},{}] down [{},{}]",
        src.display(),
        out.display(),
        manifest.gate_up.n,
        manifest.gate_up.k,
        manifest.down.n,
        manifest.down.k
    );
    println!(
        "per-layer {:.1} MiB, total {:.1} GiB",
        layer_len as f64 / (1 << 20) as f64,
        (layer_len * num_layers) as f64 / (1u64 << 30) as f64
    );
    manifest.save(&out)?;

    let index = load_index(&src)?;
    let mut reader = ShardReader::new(&src, &index);

    // Repack Resident Core into resident_core.nvfp4
    let resident_blob_path = qwen3_burn::nvfp4_blob::ResidentCoreManifest::blob_path(&out);
    let resident_manifest_path = qwen3_burn::nvfp4_blob::ResidentCoreManifest::manifest_path(&out);
    if !resident_blob_path.exists() || !resident_manifest_path.exists() {
        println!(
            "packing resident core into {}...",
            resident_blob_path.display()
        );
        let t_res = Instant::now();
        let mut res_manifest = qwen3_burn::nvfp4_blob::ResidentCoreManifest::new();
        let mut res_bytes = Vec::new();

        for (key, _shard) in &index {
            if key.starts_with("model.visual.")
                || key.ends_with("mlp.experts.gate_up_proj")
                || key.ends_with("mlp.experts.down_proj")
            {
                continue;
            }
            let raw = reader.read_raw_tensor(key)?;
            let is_2d_linear = raw.shape.len() == 2
                && raw.shape[1] % 16 == 0
                && raw.dtype == safetensors::Dtype::BF16;
            if is_2d_linear {
                let [n, k] = [raw.shape[0], raw.shape[1]];
                let (qw, bs, gscale) = quantize_nvfp4_from_nk_bf16(&raw.data, k, n);
                let layout = ProjLayout::new(n, k, 0);
                let mut rec_buf = vec![0u8; layout.stride];
                write_record(&mut rec_buf, &layout, &qw, &bs, gscale)?;
                let offset = res_manifest.add_nvfp4_linear(key.clone(), k, n, gscale);
                if res_bytes.len() < offset {
                    res_bytes.resize(offset, 0u8);
                }
                res_bytes.extend_from_slice(&rec_buf);
            } else {
                let is_f32 = raw.dtype == safetensors::Dtype::F32;
                let offset = res_manifest.add_raw(key.clone(), raw.shape.clone(), is_f32);
                if res_bytes.len() < offset {
                    res_bytes.resize(offset, 0u8);
                }
                let payload_len = raw.data.len();
                let aligned_len =
                    qwen3_burn::nvfp4_blob::align_up(payload_len, qwen3_burn::nvfp4_blob::ALIGN);
                res_bytes.extend_from_slice(&raw.data);
                if aligned_len > payload_len {
                    res_bytes.resize(res_bytes.len() + (aligned_len - payload_len), 0u8);
                }
            }
        }
        res_manifest.total_size = res_bytes.len();
        res_manifest.save(&out)?;
        std::fs::write(&resident_blob_path, &res_bytes)
            .map_err(|e| format!("write {}: {e}", resident_blob_path.display()))?;
        println!(
            "resident core packed: {:.2} MiB in {:.1}s",
            res_bytes.len() as f64 / (1024.0 * 1024.0),
            t_res.elapsed().as_secs_f64()
        );
    } else {
        println!("resident core already present, skipping");
    }

    let all_experts: Vec<usize> = (0..num_experts).collect();

    let t_start = Instant::now();
    let mut layers_written = 0usize;
    for layer in 0..num_layers {
        let path = BlobManifest::layer_path(&out, layer);
        // Resumable: a layer file of exactly the right size is treated as already done, so an
        // interrupted multi-hour repack can be restarted without redoing work.
        if std::fs::metadata(&path).is_ok_and(|m| m.len() as usize == layer_len) {
            println!("layer {layer:>3}: already present, skipping");
            continue;
        }

        let t_layer = Instant::now();
        let mut buf = vec![0u8; layer_len];

        for (proj, key) in [
            (BlobProj::GateUp, gate_up_key(layer)),
            (BlobProj::Down, down_key(layer)),
        ] {
            let layout: ProjLayout = *manifest.layout(proj);
            let raws = reader.read_expert_slices(&key, &all_experts)?;
            if raws.len() != num_experts {
                return Err(format!(
                    "{key}: got {} expert slices, expected {num_experts}",
                    raws.len()
                ));
            }
            for (expert, raw) in raws {
                let dims = &raw.shape[1..];
                if dims != [layout.n, layout.k] {
                    return Err(format!(
                        "{key} expert {expert}: shape {dims:?} != expected [{}, {}]",
                        layout.n, layout.k
                    ));
                }
                if raw.data.len() != layout.n * layout.k * 2 {
                    return Err(format!(
                        "{key} expert {expert}: {} bytes, expected {} (bf16)",
                        raw.data.len(),
                        layout.n * layout.k * 2
                    ));
                }
                // Exactly the transform the streamed path does per miss today -- just done once.
                let (qw, bs, gscale) = quantize_nvfp4_from_nk_bf16(&raw.data, layout.k, layout.n);
                let off = layout.record_offset(expert);
                write_record(
                    &mut buf[off..off + layout.stride],
                    &layout,
                    &qw,
                    &bs,
                    gscale,
                )?;
            }
        }

        // Write to a temp file then rename, so an interrupted run never leaves a
        // correctly-sized-but-incomplete layer that the resume check would skip.
        let tmp = path.with_extension("nvfp4.tmp");
        {
            let mut f = std::fs::File::create(&tmp)
                .map_err(|e| format!("create {}: {e}", tmp.display()))?;
            f.write_all(&buf)
                .map_err(|e| format!("write {}: {e}", tmp.display()))?;
            f.sync_all()
                .map_err(|e| format!("sync {}: {e}", tmp.display()))?;
        }
        std::fs::rename(&tmp, &path)
            .map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))?;

        layers_written += 1;
        let elapsed = t_start.elapsed().as_secs_f64();
        let remaining = (num_layers - layer - 1) as f64 * elapsed / layers_written as f64;
        println!(
            "layer {layer:>3}: {:.1}s  ({:.0}s elapsed, ~{:.0}s left)",
            t_layer.elapsed().as_secs_f64(),
            elapsed,
            remaining
        );
    }

    println!(
        "done in {:.0}s; {} layers written to {}",
        t_start.elapsed().as_secs_f64(),
        layers_written,
        out.display()
    );
    Ok(())
}
