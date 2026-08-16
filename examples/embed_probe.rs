use burn::prelude::Device;
use burn::tensor::{Int, Tensor};
use std::time::Instant;

fn main() {
    let device = Device::metal(burn::tensor::DeviceKind::DefaultDevice);

    // 248320 x 2048 table in BF16 (~1.0 GB)
    println!("allocating 1GB embedding table...");
    let t0 = Instant::now();
    let weight = Tensor::<2>::zeros([248320, 2048], &device);
    let _ = weight.clone().slice([0..1, 0..1]).into_data();
    println!("allocated in {:.2}s", t0.elapsed().as_secs_f64());

    let indices = Tensor::<2, Int>::from_data([[42i32]], &device);

    // Warm up
    let out = burn::tensor::module::embedding(weight.clone(), indices.clone());
    let _ = out.into_data();

    // Time 16 individual lookups, syncing each time
    let start = Instant::now();
    for i in 0..16 {
        let indices = Tensor::<2, Int>::from_data([[i as i32]], &device);
        let out = burn::tensor::module::embedding(weight.clone(), indices);
        let _ = out.into_data();
    }
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "16 embedding selects with sync: {:.3}ms total ({:.3}ms/select)",
        elapsed * 1e3,
        elapsed * 1e3 / 16.0
    );

    // Time 16 lookups pipelined (no sync between)
    let start = Instant::now();
    let mut last = weight.clone().slice([0..1, 0..1]);
    for i in 0..16 {
        let indices = Tensor::<2, Int>::from_data([[i as i32]], &device);
        last = burn::tensor::module::embedding(weight.clone(), indices)
            .reshape([1, 2048])
            .slice([0..1, 0..1]);
    }
    let _ = last.into_data();
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "16 embedding selects pipelined: {:.3}ms total ({:.3}ms/select)",
        elapsed * 1e3,
        elapsed * 1e3 / 16.0
    );
}
