//! GO/NO-GO spike: can a hand-written `#[cube(launch)]` CubeCL kernel be invoked on a tensor that
//! lives on the default Fusion-wrapped `Cuda` backend (`Fusion<CubeBackend<CudaRuntime>>`)?
//!
//! Every planned custom kernel (fp8 dequant-GEMM, MoE grouped-GEMM, CUDA graphs) is blocked on this
//! one question. This example answers it end-to-end on the real GPU:
//!
//!   1. Build a `burn::tensor::Tensor` on the DEFAULT `Cuda` (Fusion) backend.
//!   2. Apply a trivial custom kernel `y = x*2 + 1` to it via the Fusion custom-op bridge.
//!   3. Assert the kernel output matches the Burn-ops reference within 1e-5.
//!   4. Print `FUSION BRIDGE: GO|NO-GO — <pattern|blocker>`.
//!
//! It also independently runs the same kernel on the RAW `CubeBackend` (no Fusion) so we can tell
//! "the kernel itself works, only the bridge is broken" from "the kernel is broken".
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example fusion_bridge_spike 2>&1 | tail -30

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{Tensor, TensorPrimitive};

use cubecl::cuda::CudaRuntime;
use cubecl::{CubeCount, CubeDim};

use burn_cubecl::CubeBackend;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;

use burn_cubecl_fusion::CubeFusionHandle;

use burn_fusion::stream::{Operation, OperationStreams};
use burn_fusion::FusionTensor;
use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};

/// The RAW (non-Fusion) compute backend that `Cuda` wraps. Its `FloatTensorPrimitive` is a
/// `CubeTensor<CudaRuntime>`, which is what a `#[cube(launch)]` kernel can be launched against.
type Inner = CubeBackend<CudaRuntime, f32, i32, u8>;

/// The Fusion runtime used by `Cuda = Fusion<CubeBackend<CudaRuntime, f32, i32, u8>>`.
type Fr = FusionCubeRuntime<CudaRuntime, u8>;

// ---------------------------------------------------------------------------------------------
// The custom kernel. Kept in its own module so `cubecl::prelude::Tensor` (the GPU-side tensor)
// does not clash with `burn::tensor::Tensor` (the host-side tensor) used everywhere else.
// ---------------------------------------------------------------------------------------------
mod gpu_kernel {
    use cubecl::prelude::*;

    /// Trivial elementwise kernel: `y[i] = x[i] * 2 + 1`.
    #[cube(launch)]
    pub fn mul2_add1<F: Float>(input: &Tensor<F>, output: &mut Tensor<F>) {
        if ABSOLUTE_POS < input.len() {
            output[ABSOLUTE_POS] = input[ABSOLUTE_POS] * F::new(2.0) + F::new(1.0);
        }
    }
}

/// Launch the custom kernel on a RAW `CubeTensor` (the inner primitive). This is the actual
/// GPU work; both the raw-backend path and the Fusion bridge funnel through here.
fn mul2_add1_cube(input: CubeTensor<CudaRuntime>) -> CubeTensor<CudaRuntime> {
    let input = into_contiguous(input);
    let shape = input.meta.shape().clone();
    let n = shape.num_elements();

    // Allocate the output buffer on the same device/client and wrap it as a CubeTensor.
    let buffer = input.client.empty(n * core::mem::size_of::<f32>());
    let output = CubeTensor::new_contiguous(
        input.client.clone(),
        input.device.clone(),
        shape,
        buffer,
        input.dtype,
    );

    // 1-D launch: 256 threads/block, enough blocks to cover all elements.
    let threads: u32 = 256;
    let blocks = (n as u32).div_ceil(threads);

    gpu_kernel::mul2_add1::launch::<f32, CudaRuntime>(
        &input.client,
        CubeCount::Static(blocks, 1, 1),
        CubeDim { x: threads, y: 1, z: 1 },
        input.as_tensor_arg(1),
        output.as_tensor_arg(1),
    )
    .expect("kernel launch failed");

    output
}

// ---------------------------------------------------------------------------------------------
// The Fusion custom-op bridge.
//
// A Fusion stream executes opaque operations through the `Operation` trait. We register an
// `OperationIr::Custom` describing the op's input/output tensors, plus an `Operation` impl whose
// `execute` pulls the inner `CubeTensor` handles out of the fusion `HandleContainer`, runs the
// real kernel, and registers the output handle. This keeps everything on-device (no host copy).
// ---------------------------------------------------------------------------------------------
#[derive(Debug)]
struct Mul2Add1Op {
    desc: CustomOpIr,
}

impl Operation<Fr> for Mul2Add1Op {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<CudaRuntime>>) {
        let ([input_ir], [output_ir]) = self.desc.as_fixed::<1, 1>();

        // Drop from the fusion handle to the inner CubeBackend primitive...
        let input: CubeTensor<CudaRuntime> = handles.get_float_tensor::<Inner>(input_ir);
        // ...run the hand-written kernel...
        let output = mul2_add1_cube(input);
        // ...and hand the resulting handle back to the fusion stream.
        handles.register_float_tensor::<Inner>(&output_ir.id, output);
    }
}

/// Apply the custom kernel to a tensor living on the default `Cuda` (Fusion) backend.
fn mul2_add1_fusion(x: Tensor<Cuda, 1>) -> Tensor<Cuda, 1> {
    // Get the Fusion float primitive (a lazy `FusionTensor`) out of the host tensor.
    let prim: FusionTensor<Fr> = x.into_primitive().tensor();

    let client = prim.client.clone();
    let shape = prim.shape.clone();
    let dtype = prim.dtype;

    // Record the input stream BEFORE consuming the tensor into its IR.
    let streams = OperationStreams::with_inputs([&prim]);
    let input_ir = prim.into_ir();

    // Mint a fresh, uninitialized output tensor id (its handle is filled in by `execute`).
    let out_id = client.create_empty_handle();
    let output_ir = TensorIr {
        id: out_id,
        shape,
        dtype,
        status: TensorStatus::NotInit,
    };

    let desc = CustomOpIr::new("mul2_add1_custom", &[input_ir], &[output_ir]);

    // Register the opaque op on the fusion stream; this returns the (lazy) output tensor(s).
    let outputs = client.register(
        streams,
        OperationIr::Custom(desc.clone()),
        Mul2Add1Op { desc },
    );
    let out = outputs.into_iter().next().expect("custom op should yield one output");

    Tensor::from_primitive(TensorPrimitive::Float(out))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Cuda = Fusion<CubeBackend<CudaRuntime>>");

    let values: [f32; 8] = [-3.0, -1.0, 0.0, 0.5, 1.0, 2.5, 7.0, 100.0];
    let n = values.len();

    // ---- Reference (pure Burn ops on the Fusion backend) --------------------------------------
    let x_ref = Tensor::<Cuda, 1>::from_floats(values, &device);
    let reference = (x_ref * 2.0 + 1.0).into_data().to_vec::<f32>().unwrap();

    // ---- RAW CubeBackend path (no Fusion): does the kernel itself work? ------------------------
    let raw_ok = {
        let x_raw = Tensor::<Inner, 1>::from_floats(values, &device);
        let prim: CubeTensor<CudaRuntime> = x_raw.into_primitive().tensor();
        let out = mul2_add1_cube(prim);
        let got = Tensor::<Inner, 1>::from_primitive(TensorPrimitive::Float(out))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let diff = max_abs_diff(&got, &reference);
        println!("RAW CubeBackend  : got={got:?} max_abs_diff={diff:.3e}");
        diff < 1e-5
    };

    // ---- Fusion bridge path -------------------------------------------------------------------
    let x = Tensor::<Cuda, 1>::from_floats(values, &device);
    let got = mul2_add1_fusion(x).into_data().to_vec::<f32>().unwrap();
    let diff = max_abs_diff(&got, &reference);
    println!("FUSION bridge    : got={got:?} max_abs_diff={diff:.3e}");
    let fusion_ok = got.len() == n && diff < 1e-5;

    println!("reference (x*2+1): {reference:?}");
    println!();

    if fusion_ok {
        println!(
            "FUSION BRIDGE: GO — register OperationIr::Custom + an Operation impl whose execute() \
             calls HandleContainer::get_float_tensor::<CubeBackend>() to drop a FusionTensor to the \
             inner CubeTensor, launches the #[cube(launch)] kernel, then register_float_tensor() to \
             hand the output handle back to the fusion stream (on-device, no host copy)."
        );
    } else if raw_ok {
        println!(
            "FUSION BRIDGE: NO-GO — the #[cube(launch)] kernel runs correctly on the RAW \
             CubeBackend, but the Fusion custom-op registration path did not reproduce x*2+1. \
             Fix lives in the Fusion bridge, not the kernel."
        );
    } else {
        println!(
            "FUSION BRIDGE: NO-GO — the #[cube(launch)] kernel did not even run correctly on the \
             RAW CubeBackend; the kernel/launch itself is the blocker."
        );
    }

    assert!(
        raw_ok,
        "kernel failed on the raw CubeBackend (got {got:?} vs ref {reference:?})"
    );
    assert!(
        fusion_ok,
        "Fusion bridge output mismatch (got {got:?} vs ref {reference:?})"
    );
}
