//! Validates the typed Burn-Fusion custom-op wrapper (`qwen3_burn::cube_custom_op`) on the real
//! GB10 GPU, on cases the original `fusion_bridge_spike` does NOT cover:
//!
//!   A. **Two inputs of DIFFERENT shapes producing a DIFFERENT-shaped output** — a rank-1 outer
//!      product with bias: `a:[M]`, `b:[N]`, scalar `c` → `out:[M,N]`, `out[i,j] = a[i]*b[j] + c`.
//!      This is the shape pattern real GEMMs need (`[M,K]·[K,N]→[M,N]`) and the spike (1-in/1-out,
//!      same-shape) never exercises. The scalar `c` is captured by the kernel closure (rule 6).
//!
//!   B. **A mixed Int + Float input op** — a fused int dequant-scale: `q:[K]` (Int handle, i32) and
//!      `s:[K]` (Float handle, f32) → `out:[K]` (f32), `out[i] = (q[i] as f32) * s[i]`. This drives
//!      the `get_int_tensor` path (rule 4) alongside `get_float_tensor` in the SAME op — the fp8
//!      W8A16 shape (packed int weights + a float scale).
//!
//!   C. **Negative path** — declare a deliberately-wrong output shape and assert the wrapper's
//!      cross-validation (rule 2) PANICS instead of silently corrupting downstream tensors.
//!
//! Each positive case is checked against a pure-Burn-ops reference within 1e-5.
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example custom_op_test 2>&1 | tail -30

use std::panic::{self, AssertUnwindSafe};

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{DType, Int, Tensor, TensorPrimitive};

use cubecl::cuda::CudaRuntime;
use cubecl::{CubeCount, CubeDim};

use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::tensor::CubeTensor;

use qwen3_burn::cube_custom_op::CubeCustomOp;

// -------------------------------------------------------------------------------------------------
// The hand-written GPU kernels. In their own module so `cubecl::prelude::Tensor` (the GPU-side
// tensor) does not clash with `burn::tensor::Tensor` (the host-side tensor) used everywhere else.
// -------------------------------------------------------------------------------------------------
mod gpu_kernels {
    use cubecl::prelude::*;

    /// Rank-1 outer product with bias: `out[i,j] = a[i] * b[j] + c`.
    /// `a:[M]`, `b:[N]`, `out:[M,N]` row-major; one thread per output element.
    /// Concrete `f32` (a generic scalar `c: F` would need an `F: ScalarArgSettings` bound).
    #[cube(launch)]
    pub fn outer_product(a: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>, c: f32) {
        let n = b.len();
        if ABSOLUTE_POS < out.len() {
            let i = ABSOLUTE_POS / n;
            let j = ABSOLUTE_POS % n;
            out[ABSOLUTE_POS] = a[i] * b[j] + c;
        }
    }

    /// Fused int dequant-scale: `out[i] = (q[i] as f32) * s[i]`.
    /// `q:[K]` int, `s:[K]` float, `out:[K]` float; one thread per element.
    #[cube(launch)]
    pub fn dequant_scale<I: Int, F: Float>(q: &Tensor<I>, s: &Tensor<F>, out: &mut Tensor<F>) {
        if ABSOLUTE_POS < out.len() {
            out[ABSOLUTE_POS] = F::cast_from(q[ABSOLUTE_POS]) * s[ABSOLUTE_POS];
        }
    }
}

/// Allocate a fresh contiguous f32 output `CubeTensor` of `shape` on the same client as `like`.
fn alloc_like_f32(like: &CubeTensor<CudaRuntime>, shape: [usize; 2]) -> CubeTensor<CudaRuntime> {
    let n = shape[0] * shape[1];
    // Byte-size derived from the dtype, not a hardcoded constant (rule 3, even on the caller side).
    let buffer = like.client.empty(n * DType::F32.size());
    CubeTensor::new_contiguous(
        like.client.clone(),
        like.device.clone(),
        shape.into(),
        buffer,
        DType::F32,
    )
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// -------------------------------------------------------------------------------------------------
// Case A — float outer product with bias, different-shaped output.
// -------------------------------------------------------------------------------------------------
fn case_outer_product(device: &CudaDevice) -> bool {
    let m = 3usize;
    let n = 4usize;
    let a_vals: [f32; 3] = [1.0, 2.0, 3.0];
    let b_vals: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
    let c: f32 = 0.5;

    let a = Tensor::<Cuda, 1>::from_floats(a_vals, device);
    let b = Tensor::<Cuda, 1>::from_floats(b_vals, device);

    // Reference: broadcast [M,1] * [1,N] + c via pure Burn ops.
    let reference = (a.clone().reshape([m, 1]) * b.clone().reshape([1, n]) + c)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // Through the wrapper: two float inputs of different shapes → a [M,N] float output.
    let a_prim = a.into_primitive().tensor();
    let b_prim = b.into_primitive().tensor();

    let outputs = CubeCustomOp::<CudaRuntime>::new("outer_product")
        .float_input(a_prim) // registered in BOTH the stream AND the op IR (rule 1)
        .float_input(b_prim)
        .float_output([m, n], DType::F32) // cross-validated against the kernel alloc (rule 2)
        .launch(move |inputs| {
            let a = into_contiguous(inputs[0].clone());
            let b = into_contiguous(inputs[1].clone());
            let out = alloc_like_f32(&a, [m, n]);

            let total = (m * n) as u32;
            let threads = 256u32;
            let blocks = total.div_ceil(threads);
            gpu_kernels::outer_product::launch::<CudaRuntime>(
                &a.client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim { x: threads, y: 1, z: 1 },
                a.as_tensor_arg(1),
                b.as_tensor_arg(1),
                out.as_tensor_arg(1),
                cubecl::prelude::ScalarArg::new(c), // scalar `c` captured by this closure (rule 6)
            )
            .expect("outer_product launch failed");
            vec![out]
        });

    let out_fusion = outputs.into_iter().next().expect("one output");
    let got = Tensor::<Cuda, 2>::from_primitive(TensorPrimitive::Float(out_fusion))
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let diff = max_abs_diff(&got, &reference);
    println!("A. outer product  a:[{m}] ⊗ b:[{n}] + c → out:[{m},{n}]  max_abs_diff={diff:.3e}");
    println!("   got      = {got:?}");
    println!("   reference= {reference:?}");
    got.len() == m * n && diff < 1e-5
}

// -------------------------------------------------------------------------------------------------
// Case B — mixed Int + Float input op (drives the get_int_tensor path, rule 4).
// -------------------------------------------------------------------------------------------------
fn case_dequant_scale(device: &CudaDevice) -> bool {
    let k = 5usize;
    let q_vals: [i32; 5] = [2, -3, 4, 0, 7];
    let s_vals: [f32; 5] = [0.5, 1.0, 2.0, -1.0, 0.25];

    let q = Tensor::<Cuda, 1, Int>::from_ints(q_vals, device);
    let s = Tensor::<Cuda, 1>::from_floats(s_vals, device);

    // Reference: q.float() * s via pure Burn ops.
    let reference = (q.clone().float() * s.clone())
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let q_prim = q.into_primitive(); // Int kind → the IntTensorPrimitive (a FusionTensor) directly
    let s_prim = s.into_primitive().tensor();

    let outputs = CubeCustomOp::<CudaRuntime>::new("dequant_scale")
        .int_input(q_prim) // pulled via get_int_tensor in execute (rule 4)
        .float_input(s_prim) // pulled via get_float_tensor
        .float_output([k, 1], DType::F32)
        .launch(move |inputs| {
            let q = into_contiguous(inputs[0].clone());
            let s = into_contiguous(inputs[1].clone());
            let out = alloc_like_f32(&s, [k, 1]);

            let total = k as u32;
            let threads = 256u32;
            let blocks = total.div_ceil(threads);
            gpu_kernels::dequant_scale::launch::<i32, f32, CudaRuntime>(
                &s.client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim { x: threads, y: 1, z: 1 },
                q.as_tensor_arg(1),
                s.as_tensor_arg(1),
                out.as_tensor_arg(1),
            )
            .expect("dequant_scale launch failed");
            vec![out]
        });

    let out_fusion = outputs.into_iter().next().expect("one output");
    let got = Tensor::<Cuda, 2>::from_primitive(TensorPrimitive::Float(out_fusion))
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let diff = max_abs_diff(&got, &reference);
    println!("B. dequant-scale  q:[{k}](int) * s:[{k}](f32) → out:[{k}]  max_abs_diff={diff:.3e}");
    println!("   got      = {got:?}");
    println!("   reference= {reference:?}");
    got.len() == k && diff < 1e-5
}

// -------------------------------------------------------------------------------------------------
// Case C — negative path: a deliberately-wrong declared output shape must be CAUGHT (rule 2).
// -------------------------------------------------------------------------------------------------
fn case_negative_wrong_output_shape(device: &CudaDevice) -> bool {
    let m = 3usize;
    let n = 4usize;
    let a = Tensor::<Cuda, 1>::from_floats([1.0f32, 2.0, 3.0], device);
    let b = Tensor::<Cuda, 1>::from_floats([10.0f32, 20.0, 30.0, 40.0], device);

    // Silence the default panic hook so the EXPECTED panic doesn't print a scary backtrace.
    let prev_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let a_prim = a.into_primitive().tensor();
        let b_prim = b.into_primitive().tensor();

        // DECLARE the output as [M, N+1] while the kernel actually produces [M, N].
        let outputs = CubeCustomOp::<CudaRuntime>::new("outer_product_bad_shape")
            .float_input(a_prim)
            .float_input(b_prim)
            .float_output([m, n + 1], DType::F32) // <-- wrong on purpose
            .launch(move |inputs| {
                let a = into_contiguous(inputs[0].clone());
                let b = into_contiguous(inputs[1].clone());
                let out = alloc_like_f32(&a, [m, n]); // real shape [M, N]
                let total = (m * n) as u32;
                let threads = 256u32;
                let blocks = total.div_ceil(threads);
                gpu_kernels::outer_product::launch::<CudaRuntime>(
                    &a.client,
                    CubeCount::Static(blocks, 1, 1),
                    CubeDim { x: threads, y: 1, z: 1 },
                    a.as_tensor_arg(1),
                    b.as_tensor_arg(1),
                    out.as_tensor_arg(1),
                    cubecl::prelude::ScalarArg::new(0.0f32),
                )
                .expect("launch failed");
                vec![out]
            });

        // Force the stream to drain → execute() runs → cross-validation should panic here.
        let out_fusion = outputs.into_iter().next().expect("one output");
        let _ = Tensor::<Cuda, 2>::from_primitive(TensorPrimitive::Float(out_fusion))
            .into_data()
            .to_vec::<f32>()
            .unwrap();
    }));

    panic::set_hook(prev_hook);

    // Confirm the panic came from OUR rule-2 cross-validation (a "SHAPE mismatch" in the named op),
    // not some unrelated failure — otherwise the negative test could pass for the wrong reason.
    let msg = match &result {
        Err(payload) => payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
        Ok(()) => String::new(),
    };
    let caught = result.is_err()
        && msg.contains("SHAPE mismatch")
        && msg.contains("outer_product_bad_shape");

    if caught {
        println!(
            "C. negative path  declared [{m},{}] but kernel made [{m},{n}]  → CAUGHT by rule-2 \
             cross-validation (panicked as expected)",
            n + 1,
        );
        println!("   panic msg: {}", msg.replace('\n', " "));
    } else if result.is_err() {
        println!("C. negative path  FAILED — panicked, but NOT from rule-2 validation: {msg}");
    } else {
        println!("C. negative path  FAILED — wrong declared output shape was NOT caught!");
    }
    caught
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Cuda = Fusion<CubeBackend<CudaRuntime>>");
    println!("--- typed custom-op wrapper validation (cube_custom_op) ---");

    let a_ok = case_outer_product(&device);
    println!();
    let b_ok = case_dequant_scale(&device);
    println!();
    // Negative path runs LAST: a panic under the fusion server lock can poison it, which is
    // irrelevant once everything else has passed.
    let c_ok = case_negative_wrong_output_shape(&device);
    println!();

    println!(
        "RESULTS: A(outer-product)={} B(int-dequant)={} C(negative-catch)={}",
        pass(a_ok),
        pass(b_ok),
        pass(c_ok),
    );

    assert!(a_ok, "Case A (outer product) mismatch vs Burn reference");
    assert!(b_ok, "Case B (int dequant-scale) mismatch vs Burn reference");
    assert!(c_ok, "Case C (negative path) — wrong declared output shape was not caught (rule 2)");

    println!(
        "CUSTOM-OP WRAPPER: GO — N-input/different-shaped-output + mixed Float/Int handles validated \
         on GB10, and rule-2 cross-validation catches a declared/actual output-shape drift."
    );
}

fn pass(b: bool) -> &'static str {
    if b { "PASS" } else { "FAIL" }
}
