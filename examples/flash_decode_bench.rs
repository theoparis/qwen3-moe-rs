//! L2A.2 perf: does split-K flash-decode beat the single-thread flash_attention_raw at decode
//! (q_len=1) across context lengths? Also probes the P0.4 idle-CTA question (does a large static
//! split grid hurt at SHORT context?). Latency-oriented: per-call GPU sync (decode is latency-bound).
//! Run: RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example flash_decode_bench
use std::time::Instant;

use burn::tensor::{Tensor, TensorData};
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use qwen3_burn::capture::CaptureBackend;
use qwen3_burn::flash_attn::flash_attention_raw;
use qwen3_burn::flash_decode::flash_decode_raw;

type B = CaptureBackend;

fn block_sync() {
    let device = <B as burn::tensor::backend::Backend>::Device::default();
    let client = CudaRuntime::client(&device);
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

fn pseudo(n: usize, seed: usize) -> Vec<f32> {
    (0..n).map(|i| (((i * 2654435761 + seed * 40503) % 2003) as f32 / 2003.0 - 0.5) * 1.4).collect()
}

fn time_ms<F: Fn()>(warmup: usize, reps: usize, f: F) -> f64 {
    for _ in 0..warmup { f(); }
    block_sync();
    let t = Instant::now();
    for _ in 0..reps { f(); }
    block_sync();
    t.elapsed().as_secs_f64() * 1e3 / reps as f64
}

fn main() {
    let dev = Default::default();
    let (hq, hkv, d) = (16usize, 2usize, 128usize); // 30B-ish decode shape (GQA 8, head_dim 128)
    let scale = 1.0 / (d as f32).sqrt();
    println!("device: Cuda | decode latency (ms/call), hq={hq} hkv={hkv} d={d}, per-call sync");
    println!("{:>7} | {:>12} | {:>10} {:>10} {:>10} | {}", "sk", "1thread(FA)", "splitK=8", "splitK=32", "splitK=64", "best speedup");
    for &sk in &[128usize, 512, 2048, 8192] {
        let q = Tensor::<B, 4>::from_data(TensorData::new(pseudo(hq * d, 1), [1, hq, 1, d]), &dev);
        let k = Tensor::<B, 4>::from_data(TensorData::new(pseudo(hkv * sk * d, 2), [1, hkv, sk, d]), &dev);
        let v = Tensor::<B, 4>::from_data(TensorData::new(pseudo(hkv * sk * d, 3), [1, hkv, sk, d]), &dev);
        let reps = if sk >= 4096 { 30 } else { 80 };

        let single = time_ms(5, reps, || {
            let _ = flash_attention_raw(q.clone(), k.clone(), v.clone(), scale);
        });
        let mut best = f64::INFINITY;
        let mut sk_ms = [0.0f64; 3];
        for (i, &ns) in [8usize, 32, 64].iter().enumerate() {
            let ns = ns.min(sk);
            sk_ms[i] = time_ms(5, reps, || {
                let _ = flash_decode_raw(q.clone(), k.clone(), v.clone(), scale, ns);
            });
            best = best.min(sk_ms[i]);
        }
        println!("{sk:>7} | {single:>12.4} | {:>10.4} {:>10.4} {:>10.4} | {:.2}x",
            sk_ms[0], sk_ms[1], sk_ms[2], single / best);
    }
    println!("\n(single-thread FA scales O(sk); split-K parallelizes the KV scan across CTAs.)");
    println!("(P0.4 read: at sk=128 the 32/64-split rows should NOT be much slower than split=8 — idle CTAs cheap.)");
}
