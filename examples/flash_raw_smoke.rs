//! L2A.1 verification: flash_attention_raw (raw CubeBackend, below Fusion) vs an independent CPU
//! oracle (stable-softmax causal GQA attention). Confirms the A3 raw-launch port preserves the
//! proven FA-2 numerics. Run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example flash_raw_smoke
use burn::tensor::{Tensor, TensorData};
use qwen3_burn::capture::CaptureBackend;
use qwen3_burn::flash_attn::flash_attention_raw;
use qwen3_burn::flash_decode::flash_decode_raw;

type B = CaptureBackend;

/// Independent CPU oracle: causal GQA attention with a numerically-stable softmax (f32).
/// q:[H,Sq,D], k,v:[Hkv,Sk,D], queries occupy the LAST Sq of Sk (q_global = sk-sq+qi).
fn cpu_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    hq: usize,
    hkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let n_rep = hq / hkv;
    let q_off = sk - sq;
    let mut out = vec![0.0f32; hq * sq * d];
    for h in 0..hq {
        let kvh = h / n_rep;
        for qi in 0..sq {
            let qg = q_off + qi;
            let qb = (h * sq + qi) * d;
            // scores over visible keys [0..=qg]
            let mut sc = vec![0.0f32; qg + 1];
            let mut m = f32::NEG_INFINITY;
            for (kj, s) in sc.iter_mut().enumerate() {
                let kb = (kvh * sk + kj) * d;
                let mut dot = 0.0f32;
                for e in 0..d {
                    dot += q[qb + e] * k[kb + e];
                }
                *s = dot * scale;
                if *s > m {
                    m = *s;
                }
            }
            let mut l = 0.0f32;
            for s in sc.iter_mut() {
                *s = (*s - m).exp();
                l += *s;
            }
            let ob = (h * sq + qi) * d;
            for e in 0..d {
                let mut acc = 0.0f32;
                for (kj, &p) in sc.iter().enumerate() {
                    let vb = (kvh * sk + kj) * d;
                    acc += p * v[vb + e];
                }
                out[ob + e] = acc / l;
            }
        }
    }
    out
}

fn pseudo(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 2654435761 + seed * 40503) % 2003) as f32 / 2003.0 - 0.5) * 1.6)
        .collect()
}

fn check(hq: usize, hkv: usize, sq: usize, sk: usize, d: usize) {
    let dev = Default::default();
    let scale = 1.0 / (d as f32).sqrt();
    let qd = pseudo(hq * sq * d, 1);
    let kd = pseudo(hkv * sk * d, 2);
    let vd = pseudo(hkv * sk * d, 3);

    let q = Tensor::<B, 4>::from_data(TensorData::new(qd.clone(), [1, hq, sq, d]), &dev);
    let k = Tensor::<B, 4>::from_data(TensorData::new(kd.clone(), [1, hkv, sk, d]), &dev);
    let v = Tensor::<B, 4>::from_data(TensorData::new(vd.clone(), [1, hkv, sk, d]), &dev);
    let got = flash_attention_raw(q, k, v, scale)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let want = cpu_oracle(&qd, &kd, &vd, hq, hkv, sq, sk, d, scale);
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut maxabs = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        dot += (*g as f64) * (*w as f64);
        ng += (*g as f64).powi(2);
        nw += (*w as f64).powi(2);
        maxabs = maxabs.max((g - w).abs());
    }
    let cos = dot / (ng.sqrt() * nw.sqrt() + 1e-12);
    let ok = cos > 0.99999 && maxabs < 1e-3;
    println!(
        "  hq={hq} hkv={hkv} sq={sq} sk={sk} d={d}: cosine={cos:.7} max_abs={maxabs:.2e}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "flash_attention_raw diverged from CPU oracle");
}

/// L2A.2: split-K flash_decode_raw (decode, sq=1) vs the same CPU oracle, across n_splits.
fn check_decode(hq: usize, hkv: usize, sk: usize, d: usize, n_splits: usize) {
    let dev = Default::default();
    let scale = 1.0 / (d as f32).sqrt();
    let qd = pseudo(hq * d, 1); // sq=1
    let kd = pseudo(hkv * sk * d, 2);
    let vd = pseudo(hkv * sk * d, 3);

    let q = Tensor::<B, 4>::from_data(TensorData::new(qd.clone(), [1, hq, 1, d]), &dev);
    let k = Tensor::<B, 4>::from_data(TensorData::new(kd.clone(), [1, hkv, sk, d]), &dev);
    let v = Tensor::<B, 4>::from_data(TensorData::new(vd.clone(), [1, hkv, sk, d]), &dev);
    let got = flash_decode_raw(q, k, v, scale, n_splits)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let want = cpu_oracle(&qd, &kd, &vd, hq, hkv, 1, sk, d, scale); // decode: query is last pos, sees all keys
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut maxabs = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        dot += (*g as f64) * (*w as f64);
        ng += (*g as f64).powi(2);
        nw += (*w as f64).powi(2);
        maxabs = maxabs.max((g - w).abs());
    }
    let cos = dot / (ng.sqrt() * nw.sqrt() + 1e-12);
    let ok = cos > 0.99999 && maxabs < 1e-3;
    println!(
        "  split-K hq={hq} hkv={hkv} sk={sk} d={d} n_splits={n_splits}: cosine={cos:.7} max_abs={maxabs:.2e}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "flash_decode_raw diverged from CPU oracle");
}

/// L2A.2 bf16 KV path: q stays f32, k/v are stored as bf16, accumulation/output stay f32.
fn check_decode_bf16(hq: usize, hkv: usize, sk: usize, d: usize, n_splits: usize) {
    let dev = Default::default();
    let scale = 1.0 / (d as f32).sqrt();
    let qd = pseudo(hq * d, 1); // sq=1
    let kd = pseudo(hkv * sk * d, 2);
    let vd = pseudo(hkv * sk * d, 3);

    let q = Tensor::<B, 4>::from_data(TensorData::new(qd.clone(), [1, hq, 1, d]), &dev);
    let k = Tensor::<B, 4>::from_data(TensorData::new(kd.clone(), [1, hkv, sk, d]), &dev)
        .cast(burn::tensor::DType::BF16);
    let v = Tensor::<B, 4>::from_data(TensorData::new(vd.clone(), [1, hkv, sk, d]), &dev)
        .cast(burn::tensor::DType::BF16);
    let got = flash_decode_raw(q, k, v, scale, n_splits)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let want = cpu_oracle(&qd, &kd, &vd, hq, hkv, 1, sk, d, scale);
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut maxabs = 0.0f32;
    for (g, w) in got.iter().zip(want.iter()) {
        dot += (*g as f64) * (*w as f64);
        ng += (*g as f64).powi(2);
        nw += (*w as f64).powi(2);
        maxabs = maxabs.max((g - w).abs());
    }
    let cos = dot / (ng.sqrt() * nw.sqrt() + 1e-12);
    let ok = cos > 0.999 && maxabs < 3e-2;
    println!(
        "  split-K bf16-KV hq={hq} hkv={hkv} sk={sk} d={d} n_splits={n_splits}: cosine={cos:.7} max_abs={maxabs:.2e}  {}",
        if ok { "PASS" } else { "FAIL" }
    );
    assert!(ok, "flash_decode_raw bf16-KV diverged from CPU oracle");
}

fn main() {
    println!(
        "device: {:?} | flash_attention_raw (raw CubeBackend) vs CPU oracle",
        <B as burn::tensor::backend::Backend>::Device::default()
    );
    check(4, 2, 1, 8, 16); // decode, GQA 2, head_dim 16
    check(4, 2, 1, 800, 64); // decode, long-ish ctx, head_dim 64
    check(8, 2, 6, 6, 32); // prefill (sq=sk), GQA 4
    check(2, 2, 1, 64, 128); // decode, MHA, head_dim 128 (35B shape)
    println!("L2A.1 flash_attention_raw: ALL PASS (raw-launch port preserves FA-2 numerics)");

    println!("\n=== L2A.2 split-K flash_decode_raw vs CPU oracle ===");
    check_decode(4, 2, 8, 32, 1); // 1 split == plain FA-2 decode (baseline)
    check_decode(4, 2, 8, 32, 4); // 4 splits over 8 keys
    check_decode(4, 2, 800, 64, 8); // long ctx, 8 splits
    check_decode(4, 2, 800, 64, 16); // 16 splits (some may be full, uneven)
    check_decode(16, 2, 1000, 128, 32); // 35B-ish: 16 q-heads GQA 8, head_dim 128, 32 splits
    check_decode(2, 2, 37, 64, 8); // sk not divisible by n_splits (last-split clamp + empty tail)
    check_decode(4, 2, 1, 64, 4); // sk=1, n_splits>sk (3-voice review: mostly-empty splits + 1 key; f32::MIN sentinel)
    check_decode(4, 2, 3, 32, 8); // sk<n_splits (5 empty splits, 3 one-key splits)
    for &sk in &[1usize, 800, 4096] {
        for &n_splits in &[1usize, 16, 32] {
            check_decode(16, 2, sk, 256, n_splits); // 35B full-attn decode shape
        }
    }
    println!("L2A.2 flash_decode_raw: ALL PASS (split-K + plane_sum + LSE merge == FA-2)");

    println!("\n=== L2A.2 split-K flash_decode_raw bf16 KV vs CPU oracle ===");
    check_decode_bf16(4, 2, 800, 128, 8);
    check_decode_bf16(16, 2, 256, 128, 16);
    for &sk in &[1usize, 800, 4096] {
        for &n_splits in &[1usize, 16, 32] {
            check_decode_bf16(16, 2, sk, 256, n_splits); // 35B full-attn decode shape
        }
    }
    println!("L2A.2 flash_decode_raw bf16 KV: ALL PASS");
}
