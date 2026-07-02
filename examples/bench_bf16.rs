//! T6 perf: bf16 vs f32 forward+backward throughput for Qwen3-0.6B (random init + random tokens).
//!
//! The honest perf question from the plan: at small scale the per-forward weight cast + tiny
//! kernels can erase the bf16 win. Configurable via env so we can push the batch/seq up:
//!   BATCH (default 8)  SEQ (default 512)  STEPS (default 10)
//!
//! Build:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda,train --example bench_bf16
//! Run:
//!   BATCH=16 SEQ=1024 STEPS=10 ./target/release/examples/bench_bf16

use std::time::Instant;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::{Int, Tensor};
use qwen3_burn::{Precision, Qwen3Config, Qwen3ForCausalLM};

type Backend = burn::backend::Autodiff<Cuda>;

fn env(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Run `steps` forward+backward+step iterations and return tokens/sec (after 2 warmup steps).
fn bench(prec: Precision, batch: usize, seq: usize, steps: usize, device: &CudaDevice) -> f64 {
    let config = Qwen3Config::qwen3_0_6b();
    let mut model: Qwen3ForCausalLM<Backend> =
        config.init_causal_lm(device).with_train_precision(prec);
    let mut optim = AdamWConfig::new().init();
    let vocab = config.vocab_size;

    // Deterministic hashed token ids in [0, vocab) — scattered gather addresses (not a linear
    // sweep) for more realistic embedding-lookup behavior (review #4).
    let data: Vec<i64> = (0..batch * seq)
        .map(|i| ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) % vocab as u64) as i64)
        .collect();
    let tokens: Tensor<Backend, 2, Int> =
        Tensor::<Backend, 1, Int>::from_data(data.as_slice(), device).reshape([batch, seq]);
    let inputs = tokens.clone().slice([0..batch, 0..seq - 1]);
    let targets = tokens.slice([0..batch, 1..seq]);
    let s = seq - 1;

    let mut run = |model: Qwen3ForCausalLM<Backend>| {
        let logits = model.forward(inputs.clone(), None);
        let loss = CrossEntropyLoss::new(None, device)
            .forward(logits.reshape([batch * s, vocab]), targets.clone().reshape([batch * s]));
        let grads = GradientsParams::from_grads(loss.backward(), &model);
        optim.step(1e-9, model, grads)
    };

    for _ in 0..2 {
        model = run(model);
    }
    // CubeCL CUDA is async: drain the warmup queue so the timer starts clean (review #3).
    let _ = model.model.embed_tokens_weight().sum().into_data();
    let t = Instant::now();
    for _ in 0..steps {
        model = run(model);
    }
    // Force the GPU to finish ALL queued work before stopping the timer — otherwise
    // Instant::elapsed() measures only CPU kernel-dispatch and inflates tok/s (review #3).
    let _ = model.model.embed_tokens_weight().sum().into_data();
    let el = t.elapsed().as_secs_f64().max(1e-9);
    (steps * batch * s) as f64 / el
}

fn main() {
    let device = CudaDevice::default();
    let batch = env("BATCH", 8);
    let seq = env("SEQ", 512);
    let steps = env("STEPS", 10);
    println!("perf bench: Qwen3-0.6B  batch={batch} seq={seq} steps={steps} (random init/tokens)");
    let f32_tps = bench(Precision::F32, batch, seq, steps, &device);
    let bf16_tps = bench(Precision::Bf16, batch, seq, steps, &device);
    println!("f32  : {f32_tps:9.0} tok/s");
    println!("bf16 : {bf16_tps:9.0} tok/s   ({:.2}x f32)", bf16_tps / f32_tps);
}
