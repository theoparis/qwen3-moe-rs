//! kv_cache_bench — Phase 2 (docs/VLLM_PARITY_PLAN.md) perf check: legacy `cat` KV cache (O(T^2)
//! realloc+copy each step) vs the static pre-allocated buffer (`with_capacity`, slice_assign in place).
//! Times T decode-step cache updates at a rollout-ish shape on the GB10.
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example kv_cache_bench

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{Distribution, Tensor};
use qwen3_burn::KVCache;
use std::time::Instant;

type B = Cuda;

fn sync(d: &CudaDevice) {
    let _ = Tensor::<B, 1>::zeros([1], d).sum().into_scalar();
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?}");
    let (n, kvh, hd, lp) = (32usize, 8usize, 128usize, 16usize); // rollout batch 32, GQA kv-heads 8, head_dim 128
    println!("KV cache update: N={n} kv_heads={kvh} head_dim={hd}, prefill={lp}\n");

    for &t in &[128usize, 512] {
        let cap = lp + t;
        let mut results = vec![];
        for &(name, capacity) in &[("cat   (legacy O(T^2))", None::<usize>), ("static (Phase 2)", Some(cap))] {
            let mut cache = match capacity {
                Some(c) => KVCache::<B>::with_capacity(c),
                None => KVCache::<B>::new(),
            };
            let k0 = Tensor::<B, 4>::random([n, lp, kvh, hd], Distribution::Normal(0.0, 1.0), &device);
            let v0 = Tensor::<B, 4>::random([n, lp, kvh, hd], Distribution::Normal(0.0, 1.0), &device);
            let _ = cache.update(k0, v0); // prefill
            sync(&device);

            let t0 = Instant::now();
            for _ in 0..t {
                let k = Tensor::<B, 4>::random([n, 1, kvh, hd], Distribution::Normal(0.0, 1.0), &device);
                let v = Tensor::<B, 4>::random([n, 1, kvh, hd], Distribution::Normal(0.0, 1.0), &device);
                let (kk, vv) = cache.update(k, v);
                core::hint::black_box((kk, vv));
            }
            sync(&device);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            println!("  T={t:4} {name}: {ms:7.1} ms total  ({:.3} ms/step)", ms / t as f64);
            results.push(ms);
        }
        println!("  T={t:4} -> static is {:.2}x the cat time\n", results[1] / results[0]);
    }
}
