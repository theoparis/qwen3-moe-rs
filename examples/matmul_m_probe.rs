//! Probe: is the CubeCL 2-D GEMM [M, K] @ [K, N] correct for M>1 on sm_121?
//! linear3 flattens [b,s,K]->[b*s,K] to dodge the batched-matmul bug; this checks whether the
//! flattened 2-D path is itself correct for M=5 (prefill) vs M=1 (decode) by comparing the batched
//! result to a per-row loop AND to an ndarray CPU reference.
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use burn::tensor::{Tensor, TensorData};

fn main() {
    let dev = CudaDevice::default();
    let cdev = NdArrayDevice::default();
    let (m, k, n) = (5usize, 2048usize, 512usize);
    // deterministic pseudo-random data
    let xdata: Vec<f32> = (0..m * k).map(|i| (((i * 1103515245 + 12345) % 1000) as f32 / 1000.0 - 0.5) * 1.1).collect();
    let wdata: Vec<f32> = (0..k * n).map(|i| (((i * 1664525 + 1013904223) % 1000) as f32 / 1000.0 - 0.5) * 0.024).collect();

    let x = Tensor::<Cuda, 2>::from_data(TensorData::new(xdata.clone(), [m, k]), &dev);
    let w = Tensor::<Cuda, 2>::from_data(TensorData::new(wdata.clone(), [k, n]), &dev);

    // (A) batched 2-D GEMM (what linear3 does for M=5)
    let y_batched = x.clone().matmul(w.clone());
    // (B) per-row loop (M=1 each, the path that works for the GDN in prefill)
    let mut rows = Vec::new();
    for i in 0..m {
        let xi = x.clone().slice([i..i + 1, 0..k]);
        rows.push(xi.matmul(w.clone()));
    }
    let y_loop = Tensor::cat(rows, 0);
    // (C) CPU ndarray reference
    let xc = Tensor::<NdArray, 2>::from_data(TensorData::new(xdata, [m, k]), &cdev);
    let wc = Tensor::<NdArray, 2>::from_data(TensorData::new(wdata, [k, n]), &cdev);
    let y_cpu = xc.matmul(wc);

    let nb = y_batched.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    let nl = y_loop.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    let ncpu = y_cpu.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
    let diff_bl = (y_batched.clone() - y_loop.clone()).abs().max().into_scalar();
    let cpu_vec = y_cpu.into_data().to_vec::<f32>().unwrap();
    let batched_vec = y_batched.into_data().to_vec::<f32>().unwrap();
    let max_bc = cpu_vec.iter().zip(batched_vec.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);

    println!("M={m} K={k} N={n}");
    println!("norm batched(GPU 2D)={nb:.5}  loop(GPU M=1)={nl:.5}  cpu(ndarray)={ncpu:.5}");
    println!("max|batched - loop| = {diff_bl:.6}");
    println!("max|batched - cpu|  = {max_bc:.6}");
    if max_bc > 0.01 {
        println!("=> CUBECL 2-D GEMM IS WRONG for M={m} (batched != cpu reference)");
    } else {
        println!("=> CubeCL 2-D GEMM matches CPU for M={m} (matmul is NOT the bug)");
    }
}
