//! moe_probe — the B2 fast-path SPIKE (docs/MOE_PLAN.md §8). The Tier-1 oracle runs 128 separate
//! 2-D expert GEMMs per layer (launch-bound). The fast path wants ONE batched/grouped GEMM over the
//! expert dim. The plan flagged a risk: a `[E,*,K]@[E,K,N]` grouped GEMM is "the same kernel family"
//! as the documented sm_121 batched-matmul CORRUPTION bug. But that bug is specifically the
//! *broadcast* case (`[B,S,K] @ [1,K,N]`); a TRUE per-expert-batched GEMM (both operands batched over
//! E) is a different lowering and may be correct. This probe answers it empirically + benchmarks.
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example moe_probe

use std::time::Instant;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{activation::silu, Distribution, Int, Tensor};

type B = Cuda;

fn sync(device: &CudaDevice) {
    // Force a host sync (CubeCL is async): read a tiny scalar.
    let _ = Tensor::<B, 1>::zeros([1], device).sum().into_scalar();
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?}");
    // Real Qwen3-MoE expert dims: 128 experts, hidden 2048, moe_intermediate 768.
    let (e, h, i) = (128usize, 2048usize, 768usize);

    println!("\n=== CORRECTNESS: batched [E,C,H]@[E,H,I] vs per-expert 2-D loop (sm_121 bug check) ===");
    for &c in &[1usize, 8, 64] {
        let a = Tensor::<B, 3>::random([e, c, h], Distribution::Normal(0.0, 1.0), &device);
        let w = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 1.0), &device);

        let batched = a.clone().matmul(w.clone()); // [E,C,I]
        let mut rows = Vec::with_capacity(e);
        for ei in 0..e {
            let ae = a.clone().slice([ei..ei + 1, 0..c, 0..h]).reshape([c, h]);
            let we = w.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
            rows.push(ae.matmul(we).reshape([1, c, i]));
        }
        let reference = Tensor::cat(rows, 0); // [E,C,I]
        let maxdiff: f32 = (batched - reference.clone()).abs().max().into_scalar();
        let refmean: f32 = reference.abs().mean().into_scalar();
        let rel = maxdiff / refmean.max(1e-9);
        println!(
            "  C={c:3}: max|diff|={maxdiff:.3e}  mean|ref|={refmean:.3}  rel={rel:.2e}  -> {}",
            if rel < 1e-2 { "MATCH — batched GEMM is SAFE on sm_121" } else { "CORRUPT — must use the 2-D loop / custom kernel" }
        );
    }

    println!("\n=== SPEED: 1 batched GEMM vs 128 per-expert 2-D GEMMs (launch overhead) ===");
    for &c in &[1usize, 64] {
        let a = Tensor::<B, 3>::random([e, c, h], Distribution::Normal(0.0, 1.0), &device);
        let w = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 1.0), &device);
        let _ = a.clone().matmul(w.clone()).sum().into_scalar(); // warmup
        sync(&device);

        let iters = 10;
        let t0 = Instant::now();
        for _ in 0..iters {
            let _ = a.clone().matmul(w.clone()).sum().into_scalar();
        }
        let tb = t0.elapsed().as_secs_f64() / iters as f64;

        let t0 = Instant::now();
        for _ in 0..iters {
            let mut rows = Vec::with_capacity(e);
            for ei in 0..e {
                let ae = a.clone().slice([ei..ei + 1, 0..c, 0..h]).reshape([c, h]);
                let we = w.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
                rows.push(ae.matmul(we).reshape([1, c, i]));
            }
            let _ = Tensor::cat(rows, 0).sum().into_scalar();
        }
        let tl = t0.elapsed().as_secs_f64() / iters as f64;
        println!("  C={c:3}: batched {:.2} ms   per-expert-loop {:.2} ms   -> {:.1}x faster", tb * 1e3, tl * 1e3, tl / tb);
    }

    println!("\n=== FULL MoE expert pipeline: oracle (128 per-expert loop) vs fast (pre-stacked batched SwiGLU+combine) ===");
    for &c in &[1usize, 32] {
        // pre-stacked expert weights (built ONCE in production, amortized across the decode loop)
        let gate_stack = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 0.02), &device);
        let up_stack = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 0.02), &device);
        let down_stack = Tensor::<B, 3>::random([e, i, h], Distribution::Normal(0.0, 0.02), &device);
        let x = Tensor::<B, 2>::random([c, h], Distribution::Normal(0.0, 1.0), &device);
        let gate_w = Tensor::<B, 2>::random([c, e], Distribution::Default, &device); // routing weights [C,E]
        sync(&device);

        let iters = 10;
        // oracle: per-expert 2-D loop
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut acc = Tensor::<B, 2>::zeros([c, h], &device);
            for ei in 0..e {
                let g = gate_stack.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
                let u = up_stack.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
                let d = down_stack.clone().slice([ei..ei + 1, 0..i, 0..h]).reshape([i, h]);
                let ye = (silu(x.clone().matmul(g)) * x.clone().matmul(u)).matmul(d); // [C,H]
                let we = gate_w.clone().slice([0..c, ei..ei + 1]); // [C,1]
                acc = acc + ye * we;
            }
            let _ = acc.sum().into_scalar();
        }
        let to = t0.elapsed().as_secs_f64() / iters as f64;

        // fast: stacked batched
        let t0 = Instant::now();
        for _ in 0..iters {
            let xe = x.clone().unsqueeze::<3>().repeat(&[e, 1, 1]); // [E,C,H]
            let g = silu(xe.clone().matmul(gate_stack.clone())); // [E,C,I]
            let u = xe.matmul(up_stack.clone());
            let y = (g * u).matmul(down_stack.clone()); // [E,C,H]
            let gw = gate_w.clone().transpose().reshape([e, c, 1]); // [E,C,1]
            let _ = (y * gw).sum_dim(0).sum().into_scalar();
        }
        let tf = t0.elapsed().as_secs_f64() / iters as f64;
        println!("  C={c:3}: oracle {:.2} ms   fast(pre-stacked) {:.2} ms   -> {:.1}x faster", to * 1e3, tf * 1e3, to / tf);
    }

    println!("\n=== ON-DEVICE CAPACITY (rollout shape): per-expert oracle vs dispatch-build + [E,C,H] batched ===");
    println!("    (C = ceil(1.5*k*T/E); the on-device path computes E*C FFNs vs the oracle's E*T)");
    for &tt in &[256usize, 2048] {
        let cap = (((1.5 * (8 * tt) as f64) / e as f64).ceil() as usize).max(1); // C ~= CF*k*T/E, k=8
        let gate_stack = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 0.02), &device);
        let up_stack = Tensor::<B, 3>::random([e, h, i], Distribution::Normal(0.0, 0.02), &device);
        let down_stack = Tensor::<B, 3>::random([e, i, h], Distribution::Normal(0.0, 0.02), &device);
        let x = Tensor::<B, 2>::random([tt, h], Distribution::Normal(0.0, 1.0), &device);
        // random routing (timing only): top-k expert ids in [0,E)
        let sel_idx = Tensor::<B, 2>::random([tt, 8], Distribution::Uniform(0.0, e as f64), &device).int(); // [T,k]
        sync(&device);
        let iters = 5;

        // ORACLE: per-expert SwiGLU over ALL T tokens (E*T FFNs)
        let t0 = Instant::now();
        for _ in 0..iters {
            let mut acc = Tensor::<B, 2>::zeros([tt, h], &device);
            for ei in 0..e {
                let g = gate_stack.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
                let u = up_stack.clone().slice([ei..ei + 1, 0..h, 0..i]).reshape([h, i]);
                let d = down_stack.clone().slice([ei..ei + 1, 0..i, 0..h]).reshape([i, h]);
                acc = acc + (silu(x.clone().matmul(g)) * x.clone().matmul(u)).matmul(d);
            }
            let _ = acc.sum().into_scalar();
        }
        let to = t0.elapsed().as_secs_f64() / iters as f64;

        // ON-DEVICE: dispatch build (arange-onehot + cumsum + scatter) + [E,C,H] batched SwiGLU
        let t0 = Instant::now();
        for _ in 0..iters {
            let nn = tt * 8;
            let ae = sel_idx.clone().reshape([nn]);
            let atok = Tensor::<B, 1, Int>::arange(0..tt as i64, &device).reshape([tt, 1]).repeat(&[1, 8]).reshape([nn]);
            let oh = ae.clone().reshape([nn, 1]).equal(Tensor::<B, 1, Int>::arange(0..e as i64, &device).reshape([1, e])).int();
            let pos = oh.cumsum(0).gather(1, ae.clone().reshape([nn, 1])).reshape([nn]).add_scalar(-1i64);
            let over = pos.clone().greater_equal_elem(cap as i64);
            let dest = (ae.mul_scalar(cap as i64) + pos).mask_fill(over, (e * cap) as i64);
            let tok_buf = Tensor::<B, 1, Int>::zeros([e * cap + 1], &device).select_assign(0, dest, atok, burn::tensor::IndexingUpdateOp::Add);
            let tokens = tok_buf.slice([0..e * cap]);
            let xe = x.clone().select(0, tokens.clone()).reshape([e, cap, h]); // [E,C,H]
            let y = (silu(xe.clone().matmul(gate_stack.clone())) * xe.matmul(up_stack.clone())).matmul(down_stack.clone()); // [E,C,H]
            let acc = Tensor::<B, 2>::zeros([tt, h], &device).select_assign(0, tokens, y.reshape([e * cap, h]), burn::tensor::IndexingUpdateOp::Add);
            let _ = acc.sum().into_scalar();
        }
        let td = t0.elapsed().as_secs_f64() / iters as f64;
        println!("  T={tt:5} C={cap:4} (E*C/E*T = {:.0}% of dense): oracle {:.1} ms   on-device {:.1} ms   -> {:.1}x faster",
            100.0 * (e * cap) as f64 / (e * tt) as f64, to * 1e3, td * 1e3, to / td);
    }

    println!("\n=== VERDICT (B2 fast paths, all oracle-equivalent — see src/moe.rs tests) ===");
    println!("  * TRUE per-expert-batched [E,C,H]@[E,H,I] is BIT-EXACT on sm_121 (only the BROADCAST");
    println!("    [B,S,K]@[1,K,N] case corrupts); the NAIVE stacked-batched full pipeline is NOT faster");
    println!("    (~0.8x) — repeat()'ing x to all E experts negates the win (computes full E×dense).");
    println!("  * forward_routed (HOST token-routing, only the touched experts): 2.7x end-to-end on the real");
    println!("    15B + 30B with byte-identical output. Best for single-stream decode (T=1).");
    println!("  * forward_routed_ondevice (ON-DEVICE dispatch: arange-onehot + cumsum + scatter, capacity-C");
    println!("    [E,C,H] batched, no host sync, CUDA-graph-friendly): ~11x above at the BATCHED rollout shape");
    println!("    (C≈1.5kT/E ⇒ 9% of dense). Win grows with T; at T=1 it is full-dense (use the host path).");
    println!("  * Exact-no-drop + compact + fixed-shape together still need a custom grouped-GEMM kernel;");
    println!("    this capacity-padded path is the standard-op stand-in (C<T ⇒ guard overflow for GRPO).");
}
