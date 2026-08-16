//! What does ONE trivial GPU op actually cost on this backend?
//!
//! The streamed-decode profile attributes ~725 ms/token to GDN_ATTN across 30 layers, i.e. ~24 ms
//! per layer for a recurrent step whose real arithmetic is a few MB of elementwise traffic. That
//! only makes sense if per-op *dispatch* cost — not compute — dominates. This probe measures the
//! floor directly: a trivial elementwise op on a tiny tensor, pipelined vs synced, so we can price
//! one dispatch and multiply by the op count the model actually issues.
//!
//! Run:
//!   cargo run --release --features metal --example dispatch_overhead_probe -- --device metal

use std::time::Instant;

use burn::prelude::Device;
use burn::tensor::{DeviceKind, Tensor};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let device_name = args
        .iter()
        .position(|x| x == "--device")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .unwrap_or_else(|| "flex".to_string());

    let device = match device_name.as_str() {
        "flex" => Device::flex(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "metal")]
        "metal" => Device::metal(DeviceKind::DefaultDevice),
        other => panic!("unknown or unbuilt --device {other:?}"),
    };
    println!("device: {device:?}");

    // A tiny tensor: any cost here is pure overhead, not bandwidth.
    let x = Tensor::<2>::zeros([1, 128], &device);

    // Warm up (shader compile, autotune, allocator).
    let mut w = x.clone();
    for _ in 0..50 {
        w = w + 1.0;
    }
    let _ = w.into_data();

    for &n in &[100usize, 1000] {
        // Pipelined: N ops, ONE sync at the end. Prices the amortized per-dispatch cost.
        let start = Instant::now();
        let mut acc = x.clone();
        for _ in 0..n {
            acc = acc + 1.0;
        }
        let _ = acc.into_data();
        let pipelined = start.elapsed().as_secs_f64();

        // Synced: N ops, forcing a host round-trip each. Prices a full submit+wait.
        let start = Instant::now();
        let mut acc = x.clone();
        for _ in 0..n {
            acc = acc + 1.0;
            let _ = acc.clone().into_data();
        }
        let synced = start.elapsed().as_secs_f64();

        println!(
            "n={n:5}  pipelined {:8.3} ms total = {:7.1} us/op   |   synced {:8.3} ms total = {:7.1} us/op",
            pipelined * 1e3,
            pipelined * 1e6 / n as f64,
            synced * 1e3,
            synced * 1e6 / n as f64,
        );
    }

    // How many ops does one GDN recurrent step issue? Roughly: 4 projections + conv loop
    // (~4 iters x 5 ops) + norms/slices/reshapes/casts + the state recurrence. Price a
    // representative chain at the real GDN state shape to see if it explains ~24 ms/layer.
    let state = Tensor::<4>::zeros([1, 32, 128, 128], &device);
    let kv = Tensor::<3>::zeros([1, 32, 128], &device);
    let start = Instant::now();
    let reps = 20;
    for _ in 0..reps {
        // The two heaviest recurrence lines: state*k -> sum_dim, and the new_state update.
        let state_k = (state.clone() * kv.clone().unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .reshape([1, 32, 128]);
        let new_state =
            state.clone() * 0.9 + kv.clone().unsqueeze_dim::<4>(3) * state_k.unsqueeze_dim::<4>(2);
        let o = (new_state * kv.clone().unsqueeze_dim::<4>(3)).sum_dim(2);
        let _ = o.into_data();
    }
    let recur = start.elapsed().as_secs_f64() / reps as f64;
    println!(
        "\nGDN-shaped recurrence core (state [1,32,128,128]): {:.3} ms/step  -> x30 layers = {:.1} ms/token",
        recur * 1e3,
        recur * 1e3 * 30.0
    );
}
