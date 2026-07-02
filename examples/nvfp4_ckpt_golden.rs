//! Golden gate for NVIDIA ModelOpt NVFP4/FP8 checkpoint ingestion.
//!
//! This compares our Rust ingest path against `scripts/nvfp4_reference_dequant.py`, a named external
//! reference that implements the prior-art formulas directly from safetensors bytes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use qwen3_burn::nvfp4::{dequant_nvfp4_outmajor, e4m3_to_f32};
use qwen3_burn::nvidia_ckpt::{
    dense_fp8_parts, dequant_nvfp4_linear_to_kn, expert_projection_parts, fuse_expert_nvfp4_parts,
    nvfp4_linear_parts, parse_expert_base, shard_index,
};

fn main() -> Result<(), String> {
    let model_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/qwen3.6-35b-a3b-nvfp4"));
    let out_dir = PathBuf::from("target/nvfp4_ckpt_golden_ref");
    fs::create_dir_all(&out_dir).map_err(|e| format!("create {}: {e}", out_dir.display()))?;

    let status = Command::new("python3")
        .arg("scripts/nvfp4_reference_dequant.py")
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("--out-dir")
        .arg(&out_dir)
        .status()
        .map_err(|e| format!("run python reference: {e}"))?;
    if !status.success() {
        return Err(format!("python reference exited with {status}"));
    }

    let index = shard_index(&model_dir)?;
    let manifest = fs::read_to_string(out_dir.join("manifest.tsv"))
        .map_err(|e| format!("read manifest: {e}"))?;
    let mut failed = false;
    for line in manifest.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() != 5 && cols.len() != 6 {
            return Err(format!("bad manifest line: {line}"));
        }
        let kind = cols[0];
        let k: usize = cols[1].parse().map_err(|e| format!("parse K: {e}"))?;
        let n: usize = cols[2].parse().map_err(|e| format!("parse N: {e}"))?;
        let reference = read_f32_bin(&out_dir.join(cols[3]))?;
        let base = cols[4];
        let n_start: usize = if cols.len() == 6 {
            cols[5].parse().map_err(|e| format!("parse N start: {e}"))?
        } else {
            0
        };
        let ours = match kind {
            "fp8" => our_dense_fp8(&model_dir, &index, base)?,
            "nvfp4" => our_nvfp4(&model_dir, &index, base, n_start, n)?,
            other => return Err(format!("unknown manifest kind {other}")),
        };
        if ours.len() != k * n || reference.len() != k * n {
            return Err(format!(
                "{base}: bad lengths ours={} reference={} expected={}",
                ours.len(),
                reference.len(),
                k * n
            ));
        }
        let (max_abs, cos) = compare(&ours, &reference).map_err(|msg| format!("{base}: {msg}"))?;
        println!("{base}: K={k} N={n} N0={n_start} max_abs={max_abs:.9e} cos={cos:.9}");
        if max_abs != 0.0 || (1.0 - cos).abs() > 1e-7 {
            failed = true;
        }
    }
    if failed {
        return Err("nvfp4 checkpoint golden mismatch".to_string());
    }
    Ok(())
}

fn our_dense_fp8(
    model_dir: &Path,
    index: &std::collections::BTreeMap<String, String>,
    base: &str,
) -> Result<Vec<f32>, String> {
    let parts = dense_fp8_parts(model_dir, index, base)?;
    let mut out = vec![0.0f32; parts.k * parts.n];
    for kk in 0..parts.k {
        for nn in 0..parts.n {
            out[kk * parts.n + nn] =
                e4m3_to_f32(parts.q_bytes_kn[kk * parts.n + nn]) * parts.scale_n[nn];
        }
    }
    Ok(out)
}

fn our_nvfp4(
    model_dir: &Path,
    index: &std::collections::BTreeMap<String, String>,
    base: &str,
    n_start: usize,
    expected_n: usize,
) -> Result<Vec<f32>, String> {
    if let Some((layer, expert, proj)) = parse_expert_base(base) {
        if n_start != 0 {
            return Err(format!(
                "{base}: expert samples do not support N offset {n_start}"
            ));
        }
        let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
        let gate = expert_projection_parts(model_dir, index, &format!("{prefix}.gate_proj"))?;
        let up = expert_projection_parts(model_dir, index, &format!("{prefix}.up_proj"))?;
        let down = expert_projection_parts(model_dir, index, &format!("{prefix}.down_proj"))?;
        let h = gate.k;
        let i = gate.n;
        let fused = fuse_expert_nvfp4_parts(gate, up, down)?;
        return match proj {
            "gate_proj" | "up_proj" => {
                let mut gscale = vec![fused.gscale_gu[0]; i];
                gscale.extend(std::iter::repeat_n(fused.gscale_gu[1], i));
                let gu =
                    dequant_nvfp4_outmajor(&fused.qw_gu_outmajor, &fused.bs_gu, &gscale, h, i * 2);
                let offset = if proj == "gate_proj" { 0 } else { i };
                let mut out = vec![0.0f32; h * i];
                for kk in 0..h {
                    out[kk * i..(kk + 1) * i]
                        .copy_from_slice(&gu[kk * i * 2 + offset..kk * i * 2 + offset + i]);
                }
                Ok(out)
            }
            "down_proj" => Ok(dequant_nvfp4_outmajor(
                &fused.qw_dn_outmajor,
                &fused.bs_dn,
                &[fused.gscale_dn],
                i,
                h,
            )),
            _ => Err(format!("unknown expert projection {proj}")),
        };
    }
    let parts = nvfp4_linear_parts(model_dir, index, base)?;
    if n_start == 0 && expected_n == parts.n {
        Ok(dequant_nvfp4_linear_to_kn(&parts))
    } else {
        let end = n_start
            .checked_add(expected_n)
            .ok_or_else(|| format!("{base}: N window overflow"))?;
        if end > parts.n {
            return Err(format!(
                "{base}: N window start={n_start} count={expected_n} exceeds N={}",
                parts.n
            ));
        }
        Ok(dequant_nvfp4_linear_cols(&parts, n_start, expected_n))
    }
}

fn dequant_nvfp4_linear_cols(
    parts: &qwen3_burn::nvidia_ckpt::Nvfp4LinearParts,
    n_start: usize,
    n_out: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; parts.k * n_out];
    let packed_per_col = parts.k / 2;
    let blocks_per_col = parts.k / 16;
    for nn in 0..n_out {
        let src_n = n_start + nn;
        for block in 0..blocks_per_col {
            let scale = qwen3_burn::nvfp4::e4m3_to_f32(parts.bs[src_n * blocks_per_col + block])
                * parts.gscale;
            for pair in 0..8 {
                let byte = parts.qw[src_n * packed_per_col + block * 8 + pair];
                let kk = block * 16 + pair * 2;
                out[kk * n_out + nn] = qwen3_burn::nvfp4::e2m1_bits_to_f32(byte & 0x0f) * scale;
                out[(kk + 1) * n_out + nn] =
                    qwen3_burn::nvfp4::e2m1_bits_to_f32((byte >> 4) & 0x0f) * scale;
            }
        }
    }
    out
}

fn read_f32_bin(path: &Path) -> Result<Vec<f32>, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} length is not f32-aligned", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn compare(a: &[f32], b: &[f32]) -> Result<(f32, f32), String> {
    let mut max_abs = 0.0f32;
    let mut dot = 0.0f64;
    let mut aa = 0.0f64;
    let mut bb = 0.0f64;
    for (idx, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        if x.is_nan() || y.is_nan() {
            return Err(format!(
                "NaN at element {idx}: ours_nan={} reference_nan={}",
                x.is_nan(),
                y.is_nan()
            ));
        }
        max_abs = max_abs.max((x - y).abs());
        dot += (x as f64) * (y as f64);
        aa += (x as f64) * (x as f64);
        bb += (y as f64) * (y as f64);
    }
    let cos = if aa == 0.0 || bb == 0.0 {
        1.0
    } else {
        (dot / (aa.sqrt() * bb.sqrt())) as f32
    };
    Ok((max_abs, cos))
}
