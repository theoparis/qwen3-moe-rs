//! Probe: does CubeCL 2-D GEMM corrupt mixed operand dtypes on sm_121?
//! Uniform rows are the regression gate. Mixed rows document the known CubeCL GB10 mixed-dtype
//! corruption; do not rely on them staying broken.
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{DType, Tensor, TensorData};
use half::bf16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OperandDType {
    F32,
    BF16,
}

impl OperandDType {
    fn label(self) -> &'static str {
        match self {
            OperandDType::F32 => "f32",
            OperandDType::BF16 => "bf16",
        }
    }

    fn quantize(self, v: f32) -> f32 {
        match self {
            OperandDType::F32 => v,
            OperandDType::BF16 => bf16::from_f32(v).to_f32(),
        }
    }
}

struct CaseResult {
    m: usize,
    k: usize,
    n: usize,
    lhs: OperandDType,
    rhs: OperandDType,
    out_dtype: String,
    max_abs_diff: f32,
    cosine: f64,
    threshold: f32,
    uniform: bool,
    pass: bool,
    status: &'static str,
}

fn data(len: usize, mul: usize, add: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i.wrapping_mul(mul).wrapping_add(add) % 1000) as f32 / 1000.0) - 0.5)
        .collect()
}

fn quantized(data: &[f32], dtype: OperandDType) -> Vec<f32> {
    data.iter().map(|&v| dtype.quantize(v)).collect()
}

fn cast_operand(t: Tensor<Cuda, 2>, dtype: OperandDType) -> Tensor<Cuda, 2> {
    match dtype {
        OperandDType::F32 => t,
        OperandDType::BF16 => t.cast(DType::BF16),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let x = x as f64;
        let y = y as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn run_case(
    dev: &CudaDevice,
    cdev: &NdArrayDevice,
    m: usize,
    k: usize,
    n: usize,
    lhs: OperandDType,
    rhs: OperandDType,
) -> CaseResult {
    let xdata = data(m * k, 1103515245, 12345);
    let wdata = data(k * n, 1664525, 1013904223);

    let xref_data = quantized(&xdata, lhs);
    let wref_data = quantized(&wdata, rhs);
    let xc = Tensor::<NdArray, 2>::from_data(TensorData::new(xref_data, [m, k]), cdev);
    let wc = Tensor::<NdArray, 2>::from_data(TensorData::new(wref_data, [k, n]), cdev);
    let y_cpu = xc.matmul(wc);
    let cpu_vec = y_cpu.into_data().to_vec::<f32>().unwrap();

    let x = Tensor::<Cuda, 2>::from_data(TensorData::new(xdata, [m, k]), dev);
    let w = Tensor::<Cuda, 2>::from_data(TensorData::new(wdata, [k, n]), dev);
    let y_gpu = cast_operand(x, lhs).matmul(cast_operand(w, rhs));
    let out_dtype = format!("{:?}", y_gpu.dtype());
    let gpu_vec = y_gpu.cast(DType::F32).into_data().to_vec::<f32>().unwrap();

    let max_abs_diff = max_abs_diff(&cpu_vec, &gpu_vec);
    let cosine = cosine(&cpu_vec, &gpu_vec);
    let threshold = if lhs == OperandDType::F32 && rhs == OperandDType::F32 {
        0.02
    } else {
        0.06
    };
    let uniform = lhs == rhs;
    let pass = cosine >= 0.9999 && max_abs_diff <= threshold;
    let status = match (uniform, pass) {
        (true, true) => "PASS",
        (true, false) => "FAIL",
        (false, true) => "MIXED-OK",
        (false, false) => "MIXED-BROKEN",
    };

    println!(
        "M={m:<2} K={k:<4} N={n:<4} {:>4}x{:<4} out={out_dtype:<4} max_abs_diff={max_abs_diff:.6} cosine={cosine:.8} {}",
        lhs.label(),
        rhs.label(),
        status
    );

    CaseResult {
        m,
        k,
        n,
        lhs,
        rhs,
        out_dtype,
        max_abs_diff,
        cosine,
        threshold,
        uniform,
        pass,
        status,
    }
}

fn main() {
    let dev = CudaDevice::default();
    let cdev = NdArrayDevice::default();
    let shapes = [
        (1usize, 2048usize, 512usize),
        (5, 2048, 512),
        (7, 2048, 512),
        (1, 2048, 4096),
        (7, 2048, 4096),
        (1, 2048, 128),
        (7, 2048, 128),
        (1, 768, 2048),
        (7, 768, 2048),
    ];
    let pairings = [
        (OperandDType::F32, OperandDType::F32),
        (OperandDType::BF16, OperandDType::BF16),
        (OperandDType::BF16, OperandDType::F32),
        (OperandDType::F32, OperandDType::BF16),
    ];

    println!("device: {:?}", dev);
    println!("--- mixed-dtype 2D GEMM canary vs NdArray f32 reference ---");

    let mut results = Vec::new();
    for (m, k, n) in shapes {
        for (lhs, rhs) in pairings {
            results.push(run_case(&dev, &cdev, m, k, n, lhs, rhs));
        }
    }

    println!();
    println!("summary:");
    println!("M  K     N     lhs x rhs   out   max_abs_diff  cosine      threshold  status");
    for r in &results {
        println!(
            "{:<2} {:<5} {:<5} {:>4}x{:<4} {:<5} {:>12.6}  {:.8}  {:>9.6}  {}",
            r.m,
            r.k,
            r.n,
            r.lhs.label(),
            r.rhs.label(),
            r.out_dtype,
            r.max_abs_diff,
            r.cosine,
            r.threshold,
            r.status
        );
    }

    let failed = results.iter().filter(|r| r.uniform && !r.pass).count();
    if failed > 0 {
        eprintln!("matmul_mixed_probe: {failed} uniform case(s) FAILED");
        std::process::exit(1);
    }
    println!("matmul_mixed_probe: all uniform case(s) passed");
}
