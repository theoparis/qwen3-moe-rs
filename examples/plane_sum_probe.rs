//! L2A.2 risk #1 probe: does CubeCL `plane_sum` (warp-shuffle reduction) work on sm_121 at this pin?
//! The split-K flash-decode kernel reduces the per-key QK dot across the 32 D-partitioned lanes with
//! `plane_sum` — which has ZERO in-repo precedent (moe_grouped.rs:322 deliberately avoided plane ops).
//! Verify in isolation before building the kernel: plane_sum(lane_id) must == 496 (=0+..+31) on ALL
//! 32 lanes, and a weighted variant must match. If this FAILS, fall back to a plane_shuffle_xor
//! butterfly (offsets 1,2,4,8,16).
//! Run: RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example plane_sum_probe
use burn::tensor::{Tensor, TensorPrimitive};
use burn_cubecl::tensor::CubeTensor;
use cubecl::cuda::CudaRuntime;
use cubecl::{CubeCount, CubeDim};
use qwen3_burn::capture::CaptureBackend;

type B = CaptureBackend;

mod gpu {
    use cubecl::prelude::*;
    /// out[lane] = plane_sum(lane_id * w). One warp (32 lanes), one cube.
    #[cube(launch)]
    pub fn plane_sum_probe(out: &mut Tensor<f32>, w: f32) {
        let lane = UNIT_POS_X;
        let val = f32::cast_from(lane) * w;
        let s = plane_sum(val);
        out[lane as usize] = s; // every lane should hold the SAME full-warp sum
    }
}

fn run(w: f32) -> Vec<f32> {
    let dev = Default::default();
    let seed = Tensor::<B, 1>::zeros([32], &dev);
    let seed_ct = seed.into_primitive().tensor();
    let buffer = seed_ct.client.empty(32 * core::mem::size_of::<f32>());
    let out = CubeTensor::new_contiguous(
        seed_ct.client.clone(),
        seed_ct.device.clone(),
        [32].into(),
        buffer,
        burn::tensor::DType::F32,
    );
    gpu::plane_sum_probe::launch::<CudaRuntime>(
        &seed_ct.client,
        CubeCount::Static(1, 1, 1),
        CubeDim { x: 32, y: 1, z: 1 },
        out.as_tensor_arg(1),
        cubecl::prelude::ScalarArg::new(w),
    )
    .expect("plane_sum_probe launch failed");
    Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(out)).into_data().to_vec::<f32>().unwrap()
}

fn main() {
    println!("=== L2A.2 plane_sum probe (sm_121) ===");
    let r1 = run(1.0);
    let expect1 = 496.0f32; // 0+1+..+31
    let all_ok1 = r1.iter().all(|&x| (x - expect1).abs() < 1e-3);
    println!("  plane_sum(lane*1.0): out[0]={} out[31]={} (expect {expect1} on all lanes) -> {}",
        r1[0], r1[31], if all_ok1 { "PASS" } else { "FAIL" });

    let r2 = run(1.5);
    let expect2 = 744.0f32; // 1.5 * 496
    let all_ok2 = r2.iter().all(|&x| (x - expect2).abs() < 1e-2);
    println!("  plane_sum(lane*1.5): out[0]={} out[17]={} (expect {expect2} on all lanes) -> {}",
        r2[0], r2[17], if all_ok2 { "PASS" } else { "FAIL" });

    if all_ok1 && all_ok2 {
        println!("PLANE_SUM WORKS on sm_121 — the split-K QK-dot reduction primitive is validated.");
    } else {
        println!("PLANE_SUM BROKEN/mis-lowered — use the plane_shuffle_xor butterfly fallback (offsets 1,2,4,8,16).");
        std::process::exit(1);
    }
}
