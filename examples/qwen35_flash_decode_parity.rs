//! CUDA parity gate for the Qwen3.6/Qwen3.5 full-attention decode flash-decode path.
//!
//! Run on a GPU host:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example qwen35_flash_decode_parity -- --dir models/qwen3.6-35b-a3b --load
//!
//! Without `--load`, this runs the same path on the random/lazy initialized default layer.

use std::path::PathBuf;

use burn::{
    backend::cuda::{Cuda, CudaDevice},
    tensor::{Distribution, Int, Tensor, TensorData},
};
use qwen3_burn::{
    KVCache, Precision, Qwen3_5MoeConfig,
    qwen3_5::{Qwen3_5DecoderLayer, Qwen3_5LayerType},
};

type B = Cuda;

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|x| x == flag)
}

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter()
        .position(|x| x == flag)
        .and_then(|i| args.get(i + 1))
}

fn positions(start: usize, len: usize, device: &CudaDevice) -> Tensor<B, 2, Int> {
    let vals: Vec<i64> = (start..start + len).map(|x| x as i64).collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(vals, [1, len]), device)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += (x as f64) * (y as f64);
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt()).max(1.0e-30)
}

fn argmax(xs: &[f32]) -> usize {
    xs.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn topk(xs: &[f32], k: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..xs.len()).collect();
    ids.sort_by(|&a, &b| xs[b].total_cmp(&xs[a]));
    ids.truncate(k);
    ids
}

fn topk_overlap(a: &[f32], b: &[f32], k: usize) -> usize {
    let ak = topk(a, k);
    let bk = topk(b, k);
    ak.iter().filter(|id| bk.contains(id)).count()
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
    let load = has_flag(&args, "--load");
    let device = CudaDevice::default();
    let cfg = if dir.join("config.json").exists() {
        Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?
    } else {
        Qwen3_5MoeConfig::default()
    };

    let mut model = cfg.init_causal_lm::<B>(&device);
    if load {
        model
            .load_weights_sharded(&dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    }

    let full_idx = cfg
        .layer_types
        .iter()
        .position(|kind| *kind == Qwen3_5LayerType::FullAttention)
        .ok_or("config has no full-attention layer")?;
    let layer = match &model.model.layers[full_idx] {
        Qwen3_5DecoderLayer::Full(layer) => layer,
        Qwen3_5DecoderLayer::Linear(_) => return Err("selected layer is not full-attention".into()),
    };

    for &total_seq in &[800usize, 1100, 4096] {
        let prefix_len = total_seq - 1;
        let mut flash_cache = KVCache::<B>::new();
        let mut sdpa_cache = KVCache::<B>::new();

        if prefix_len > 0 {
            let prefix =
                Tensor::<B, 3>::random([1, prefix_len, cfg.hidden_size], Distribution::Normal(0.0, 1.0), &device);
            let pos = positions(0, prefix_len, &device);
            let _ = layer.self_attn.forward_with_cache_sdpa_reference(
                prefix.clone(),
                pos.clone(),
                &mut flash_cache,
                Precision::F32,
            );
            let _ = layer.self_attn.forward_with_cache_sdpa_reference(
                prefix,
                pos,
                &mut sdpa_cache,
                Precision::F32,
            );
        }

        let x = Tensor::<B, 3>::random([1, 1, cfg.hidden_size], Distribution::Normal(0.0, 1.0), &device);
        let pos = positions(total_seq - 1, 1, &device);
        let flash = layer.self_attn.forward_with_cache(
            x.clone(),
            pos.clone(),
            &mut flash_cache,
            Precision::F32,
        );
        let sdpa =
            layer
                .self_attn
                .forward_with_cache_sdpa_reference(x, pos, &mut sdpa_cache, Precision::F32);

        let flash = flash.into_data().to_vec::<f32>().unwrap();
        let sdpa = sdpa.into_data().to_vec::<f32>().unwrap();
        let cos = cosine(&flash, &sdpa);
        let a_flash = argmax(&flash);
        let a_sdpa = argmax(&sdpa);
        let overlap = topk_overlap(&flash, &sdpa, 10);
        println!(
            "total_seq={total_seq}: cosine={cos:.8} argmax_flash={a_flash} argmax_sdpa={a_sdpa} top10_overlap={overlap}/10"
        );
        assert!(cos > 0.9999, "cosine {cos:.8} <= 0.9999 at total_seq={total_seq}");
        assert_eq!(a_flash, a_sdpa, "argmax mismatch at total_seq={total_seq}");
        assert!(
            overlap >= 9,
            "top-10 overlap {overlap}/10 below 9/10 at total_seq={total_seq}"
        );
    }

    println!("Qwen3.6 full-attn flash-decode SDPA parity: PASS");
    Ok(())
}
