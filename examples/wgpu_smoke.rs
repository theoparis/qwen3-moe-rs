//! Minimal smoke test: does the wgpu/vulkan/metal backend actually execute a kernel?
//! Not model-specific -- just proves `Device::{wgpu,vulkan,metal}` compute works end to end.
//!
//! Usage: cargo run --example wgpu_smoke --features metal -- metal
//!        cargo run --example wgpu_smoke --features wgpu   -- wgpu
//!        cargo run --example wgpu_smoke --features vulkan -- vulkan

use burn::prelude::Device;
use burn::tensor::{DeviceKind, Tensor};

fn main() {
    let which = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "metal".to_string());
    let device = match which.as_str() {
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "metal")]
        "metal" => Device::metal(DeviceKind::DefaultDevice),
        other => {
            eprintln!("unknown or unbuilt backend {other:?}");
            std::process::exit(1);
        }
    };
    println!("device = {device:?}");

    let a: Tensor<2> = Tensor::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let b: Tensor<2> = Tensor::from_floats([[5.0, 6.0], [7.0, 8.0]], &device);
    let c = a.matmul(b);
    device.sync().expect("sync failed");
    println!("a @ b = {}", c.to_data());
}
