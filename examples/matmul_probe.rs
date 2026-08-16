//! Is plain batched matmul batch-correct at seq=255, batch=2 on CUDA?
//! [2, seq, K] @ [K, N] with identical rows => row0 must equal row1.
use burn::prelude::Device;
use burn::tensor::{DType, Device, Distribution, Tensor};

type B = Cuda;

fn rowdiff3(t: &Tensor<3>) -> f32 {
    let [b, s, n] = t.dims();
    assert_eq!(b, 2);
    let r0 = t.clone().slice([0..1, 0..s, 0..n]);
    let r1 = t.clone().slice([1..2, 0..s, 0..n]);
    (r0 - r1)
        .abs()
        .max()
        .cast(DType::F32)
        .into_data()
        .as_slice::<f32>()
        .map(|x| x[0])
        .unwrap_or(f32::NAN)
}

fn case(seq: usize, k: usize, n: usize) {
    let device = Device::cuda(0);
    let x1 = Tensor::<3>::random([1, seq, k], Distribution::Normal(0.0, 1.0), &device);
    let x = x1.repeat(&[2, 1, 1]); // identical rows
    let w = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.02), &device);
    let wb = w.clone().unsqueeze::<3>(); // [1, k, n] -> broadcast
    let out = x.matmul(wb);
    println!(
        "matmul [2,{seq},{k}] @ [{k},{n}]  |r0-r1| = {:.6}  (must be 0)",
        rowdiff3(&out)
    );
}

fn case_2d(seq: usize, k: usize, n: usize) {
    // Flatten [2, seq, k] -> [2*seq, k], single 2D GEMM, reshape back.
    let device = Device::cuda(0);
    let x1 = Tensor::<3>::random([1, seq, k], Distribution::Normal(0.0, 1.0), &device);
    let x = x1.repeat(&[2, 1, 1]);
    let w = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.02), &device);
    let x2 = x.reshape([2 * seq, k]);
    let out2 = x2.matmul(w); // [2*seq, n]
    let out = out2.reshape([2, seq, n]);
    println!(
        "2D-flat [2,{seq},{k}] @ [{k},{n}]  |r0-r1| = {:.6}  (must be 0)",
        rowdiff3(&out)
    );
}

fn scalar(t: Tensor<1>) -> f32 {
    t.cast(DType::F32)
        .into_data()
        .as_slice::<f32>()
        .map(|x| x[0])
        .unwrap_or(f32::NAN)
}

/// bf16 2-D GEMM probe: the mixed-precision path used by `linear3(Precision::Bf16)`.
/// Reports G2 (batch-safety: identical input rows -> identical output rows) and
/// G1 (parity vs the f32 reference: relative Frobenius error + cosine similarity).
fn case_2d_bf16(seq: usize, k: usize, n: usize) {
    let device = Device::cuda(0);
    let x1 = Tensor::<3>::random([1, seq, k], Distribution::Normal(0.0, 1.0), &device);
    let x = x1.repeat(&[2, 1, 1]); // identical rows
    let w = Tensor::<2>::random([k, n], Distribution::Normal(0.0, 0.02), &device);
    let x2 = x.reshape([2 * seq, k]);
    // f32 reference (2-D flatten path).
    let ref2 = x2.clone().matmul(w.clone()); // [2*seq, n] f32
    // bf16 compute: cast inputs to bf16, GEMM (f32 accumulation on CUDA), widen back to f32.
    let bf2 = x2
        .cast(DType::BF16)
        .matmul(w.cast(DType::BF16))
        .cast(DType::F32);
    // G2: batch-safety.
    let rdiff = rowdiff3(&bf2.clone().reshape([2, seq, n]));
    // G1: parity vs f32 reference.
    let rel = scalar(((bf2.clone() - ref2.clone()) * (bf2.clone() - ref2.clone())).sum()).sqrt()
        / scalar((ref2.clone() * ref2.clone()).sum()).sqrt();
    let dot = scalar((bf2.clone() * ref2.clone()).sum());
    let nb = scalar((bf2.clone() * bf2.clone()).sum()).sqrt();
    let nr = scalar((ref2.clone() * ref2.clone()).sum()).sqrt();
    let cos = dot / (nb * nr);
    println!(
        "bf16-2D [2,{seq},{k}]@[{k},{n}]  |r0-r1|={rdiff:.6}  rel_err={rel:.4}  cos={cos:.6}   (G2: |r0-r1|=0 ; G1: rel small, cos~1)"
    );
}

/// [G6] Residual stays f32: a tiny increment survives in f32 but is absorbed in bf16, and a
/// bf16 matmul output cast back to f32 is F32-typed (so the downstream residual add is f32).
fn case_g6_residual() {
    let device = Device::cuda(0);
    let one = Tensor::<1>::from_floats([1.0f32], &device);
    let inc = Tensor::<1>::from_floats([1e-5f32], &device);
    let v_f32 = scalar(one.clone() + inc.clone());
    let v_bf16 = scalar((one.cast(DType::BF16) + inc.cast(DType::BF16)).cast(DType::F32));
    let out = Tensor::<2>::from_floats([[1.0f32, 2.0]], &device)
        .cast(DType::BF16)
        .matmul(Tensor::<2>::from_floats([[1.0f32], [1.0]], &device).cast(DType::BF16))
        .cast(DType::F32);
    println!(
        "[G6] residual f32(1.0+1e-5)={v_f32:.6} (want 1.00001)  bf16={v_bf16:.6} (absorbs->1.0)  bf16-matmul-out dtype={:?} (want F32)",
        out.dtype()
    );
}

fn main() {
    println!("device: {:?}", Device::cuda(0));
    println!("--- 3D batched matmul (burn Linear path) ---");
    case(255, 1024, 2048);
    case(255, 1024, 1024);
    case(255, 1024, 3072);
    case(255, 3072, 1024);
    case(8, 1024, 2048);
    case(128, 1024, 2048);
    println!("--- 2D flattened matmul (workaround) ---");
    case_2d(255, 1024, 2048);
    case_2d(255, 1024, 1024);
    case_2d(255, 1024, 3072);
    case_2d(255, 3072, 1024);
    case_2d(8, 1024, 2048);
    case_2d(128, 1024, 2048);
    println!("--- bf16 2D matmul: G1 parity (vs f32) + G2 batch-safety ---");
    case_2d_bf16(255, 1024, 2048);
    case_2d_bf16(255, 1024, 1024);
    case_2d_bf16(255, 1024, 3072);
    case_2d_bf16(255, 3072, 1024);
    case_2d_bf16(8, 1024, 2048);
    case_2d_bf16(128, 1024, 2048);
    println!("--- G6 residual / dtype probe ---");
    case_g6_residual();
}
