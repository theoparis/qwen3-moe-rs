//! Host-side enqueue cost of burn ops at GDN-decode tensor sizes.
//!
//! The streamed decode spends ~450 ms/token on the CPU just ENQUEUING the 30 GDN layers.
//! A prior probe measured ~9 us/op for [1,128] tensors. This probe measures pure host enqueue
//! time (no device sync until the very end) to see whether per-op host cost scales with tensor
//! size / allocator behavior.
//!
//! Run: cargo run --release --features metal --example host_enqueue_probe -- --device metal

use std::time::Instant;

use burn::prelude::Device;
use burn::tensor::{DType, Tensor};

macro_rules! bench {
    ($name:expr, $reps:expr, $body:expr) => {{
        // Warm once (shader compile, autotune, allocator prime), then time pure enqueue.
        let warm = $body;
        let _ = warm.into_data();
        let start = Instant::now();
        let mut last = $body;
        for _ in 1..$reps {
            last = $body;
        }
        let host_ns = start.elapsed().as_nanos();
        let _ = last.into_data();
        println!(
            "{:44} host enqueue {:8.1} us/op  ({} ops)",
            $name,
            host_ns as f64 / 1e3 / $reps as f64,
            $reps
        );
    }};
}

fn main() {
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);

    let tiny = Tensor::<2>::zeros([1, 128], &device);
    let mid = Tensor::<2>::zeros([1, 8192], &device);
    let state = Tensor::<4>::zeros([1, 32, 128, 128], &device);
    let kv = Tensor::<3>::zeros([1, 32, 128], &device);
    let q = Tensor::<3>::zeros([1, 16, 128], &device);

    bench!("add [1,128]", 500, tiny.clone() + 1.0);
    bench!("add [1,8192]", 500, mid.clone() + 1.0);
    bench!("mul+sum_dim2 [1,32,128,128]", 100, {
        (state.clone() * kv.clone().unsqueeze_dim::<4>(3)).sum_dim(2)
    });
    bench!("state-update (3 muls + add)", 100, {
        state.clone() * 0.9 + kv.clone().unsqueeze_dim::<4>(3) * kv.clone().unsqueeze_dim::<4>(2)
    });
    bench!("slice+reshape [1,8192]->[1,2048]", 500, {
        mid.clone().slice([0..1, 0..2048]).reshape([1, 2048])
    });
    bench!("cast f32->f32 [1,8192]", 500, mid.clone().cast(DType::F32));
    bench!("silu-chain [1,8192]", 200, {
        mid.clone() / (mid.clone().mul_scalar(-1.0).exp() + 1.0)
    });
    bench!("repeat+reshape [1,16,128]->[1,32,128]", 200, {
        q.clone()
            .unsqueeze_dim::<4>(2)
            .repeat(&[1, 1, 2, 1])
            .reshape([1, 32, 128])
    });

    // Full GDN-recurrence-shaped chain, host enqueue only:
    let start = Instant::now();
    let reps = 50;
    let mut sink = Tensor::<2>::zeros([1, 1], &device);
    for _ in 0..reps {
        let s = state.clone();
        let k = kv.clone();
        let state_k = (s.clone() * k.clone().unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .reshape([1, 32, 128]);
        let new_state =
            s.clone() * 0.9 + k.clone().unsqueeze_dim::<4>(3) * state_k.unsqueeze_dim::<4>(2);
        let o = (new_state * k.unsqueeze_dim::<4>(3)).sum_dim(2);
        sink = sink + o.slice([0..1, 0..1, 0..1, 0..1]).reshape([1, 1]);
    }
    let host_ns = start.elapsed().as_nanos();
    let _ = sink.into_data();
    println!(
        "{:44} host enqueue {:8.1} us/chain ({} chains)",
        "GDN recurrence chain (7 ops)",
        host_ns as f64 / 1e3 / reps as f64,
        reps
    );
}
