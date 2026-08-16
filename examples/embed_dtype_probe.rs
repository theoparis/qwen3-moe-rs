//! Prices a single decode-shaped embedding gather from a 248320x2048 table, per weight dtype.
//!
//! The resident-core pre-cast in `qwen35_generate_portable` only touches `Linear` weights, so
//! `embed_tokens` stays at the checkpoint's BF16. This probe checks whether that dtype choice is
//! what makes the decode profile's EMBED bucket ~200ms/token for one row lookup.

use std::time::Instant;

use burn::prelude::Device;
use burn::tensor::{DType, DeviceKind, Int, Tensor};

fn bench(label: &str, dtype: Option<DType>, device: &Device) {
    let weight = Tensor::<2>::zeros([248320, 2048], device);
    let weight = match dtype {
        Some(dt) => weight.cast(dt),
        None => weight,
    };
    // Force the table to be fully materialized before timing.
    let _ = weight.clone().slice([0..1, 0..1]).into_data();

    // Warm up (kernel compile / first-use binding).
    let warm = burn::tensor::module::embedding(
        weight.clone(),
        Tensor::<2, Int>::from_data([[7i32]], device),
    );
    let _ = warm.cast(DType::F32).into_data();

    let n = 16;
    let start = Instant::now();
    for i in 0..n {
        let idx = Tensor::<2, Int>::from_data([[i as i32]], device);
        let out = burn::tensor::module::embedding(weight.clone(), idx).cast(DType::F32);
        let _ = out.into_data();
    }
    let ms = start.elapsed().as_secs_f64() * 1e3;
    println!(
        "{label:>22}: {ms:8.3} ms / {n} gathers = {:7.3} ms per gather",
        ms / n as f64
    );
}

fn main() {
    let device = Device::metal(DeviceKind::DefaultDevice);
    println!("device: {device:?}");
    bench("F32 table", None, &device);
    bench("F16 table", Some(DType::F16), &device);
    bench("BF16 table (as loaded)", Some(DType::BF16), &device);
}
