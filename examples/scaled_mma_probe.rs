//! GO/NO-GO probe for CubeCL's block-scaled FP4 tensor-core MMA on GB10 / sm_121.
//!
//! This intentionally bypasses Burn/Fusion and launches a raw CubeCL CUDA kernel. Cargo build only
//! checks Rust-side API compatibility; CubeCL JIT-compiles the scaled-MMA PTX at runtime.
//!
//! Build only:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example scaled_mma_probe

use std::{any::Any, panic};

use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::ir::MatrixIdent;
use cubecl::ir::features::ScaledMmaConfig;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime, e2m1, e2m1x2, e4m3};

type R = CudaRuntime;
type Client = ComputeClient<R>;

const M: usize = 16;
const N: usize = 8;
const K: usize = 64;
const SCALES_FACTOR: usize = 4;
const PASS_EPS: f32 = 1.0e-3;

#[cube(launch)]
fn kernel_scaled_fp4_e4m3(
    a: &Tensor<Line<e2m1x2>>,
    b: &Tensor<Line<e2m1x2>>,
    c: &Tensor<f32>,
    scales_a: &Tensor<e4m3>,
    scales_b: &Tensor<e4m3>,
    out: &mut Tensor<f32>,
) {
    let def =
        cmma::MmaDefinition::<e2m1x2, e2m1x2, f32>::new_scaled::<e4m3>(M, N, K, SCALES_FACTOR);
    let lane_id = UNIT_POS_PLANE;

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let line_size_a = def.line_size(MatrixIdent::A);
    let line_count_a = comptime!(elem_count_a / line_size_a);
    let mut registers_a = Array::<Line<e2m1x2>>::lined(line_count_a, line_size_a);

    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let line_size_b = def.line_size(MatrixIdent::B);
    let line_count_b = comptime!(elem_count_b / line_size_b);
    let mut registers_b = Array::<Line<e2m1x2>>::lined(line_count_b, line_size_b);

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = comptime!(elem_count_c / line_size_c);
    let mut registers_c = Array::<Line<f32>>::lined(line_count_c, line_size_c);

    let scales_count = def.scales_count();
    let mut scales_register_a = Line::<e4m3>::empty(def.scales_line_size());
    let mut scales_register_b = Line::<e4m3>::empty(def.scales_line_size());

    // A is row-major logical [M,K], physically packed as e2m1x2.
    #[unroll]
    for i in 0..line_count_a {
        let n_elem = i * line_size_a * e2m1x2::packing_factor();
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        let idx = row as usize * K + col as usize;
        registers_a[i] = a[idx / (a.line_size() * e2m1x2::packing_factor())];
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a as usize * SCALES_FACTOR + i];
    }

    // B is passed in the col-major layout expected by CubeCL's manual MMA loader:
    // host index is B[col, k], representing logical B[k, col].
    #[unroll]
    for i in 0..line_count_b {
        let n_elem = i * line_size_b * e2m1x2::packing_factor();
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
        let idx = col as usize * K + row as usize;
        registers_b[i] = b[idx / (b.line_size() * e2m1x2::packing_factor())];
    }

    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b as usize * SCALES_FACTOR + i];
    }

    // Start from a zero f32 accumulator tile supplied by global memory.
    #[unroll]
    for i in 0..line_count_c {
        let mut reg = Line::<f32>::empty(line_size_c);
        #[unroll]
        for j in 0..line_size_c {
            let n_elem = i * line_size_c + j;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
            let idx = row as usize * N + col as usize;
            reg[j] = c[idx];
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute_scaled(
        &registers_a,
        &registers_b,
        &registers_c,
        scales_register_a,
        scales_register_b,
    );

    #[unroll]
    for i in 0..line_count_c {
        let reg = registers_d[i];
        #[unroll]
        for j in 0..line_size_c {
            let n_elem = i * line_size_c + j;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
            let idx = row as usize * N + col as usize;
            out[idx] = reg[j];
        }
    }
}

fn chosen_config() -> ScaledMmaConfig {
    ScaledMmaConfig {
        a_type: e2m1x2::cube_type(),
        b_type: e2m1x2::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: e4m3::cube_type(),
        m: M as u32,
        n: N as u32,
        k: K as u32,
        scales_factor: SCALES_FACTOR as u32,
    }
}

fn make_inputs() -> (
    Vec<f32>,
    Vec<e2m1x2>,
    Vec<e4m3>,
    Vec<f32>,
    Vec<e2m1x2>,
    Vec<e4m3>,
) {
    // Values are deliberately small and exactly representable in E2M1.
    let vals = [
        -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    ];

    let a_f32: Vec<f32> = (0..M)
        .flat_map(|row| {
            (0..K).map(move |k| {
                let idx = (row * 11 + k * 5 + (row * k) % 7) % vals.len();
                e2m1::from_f32(vals[idx]).to_f32()
            })
        })
        .collect();
    let a = e2m1x2::from_f32_slice(&a_f32);

    // B is stored col-major: B[col*K + k] is logical B[k,col].
    let b_f32: Vec<f32> = (0..N)
        .flat_map(|col| {
            (0..K).map(move |k| {
                let idx = (col * 13 + k * 3 + (col * k) % 5 + 1) % vals.len();
                e2m1::from_f32(vals[idx]).to_f32()
            })
        })
        .collect();
    let b = e2m1x2::from_f32_slice(&b_f32);

    // Four scale blocks over K. Keep the scales simple and exactly representable in E4M3.
    let scale_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let scales_a: Vec<e4m3> = (0..M)
        .flat_map(|row| {
            (0..SCALES_FACTOR).map(move |block| {
                e4m3::from_f32(scale_vals[(row * 3 + block * 2) % scale_vals.len()])
            })
        })
        .collect();
    let scales_b: Vec<e4m3> = (0..N)
        .flat_map(|col| {
            (0..SCALES_FACTOR)
                .map(move |block| e4m3::from_f32(scale_vals[(col * 5 + block) % scale_vals.len()]))
        })
        .collect();

    (a_f32, a, scales_a, b_f32, b, scales_b)
}

fn host_reference(a: &[f32], scales_a: &[e4m3], b: &[f32], scales_b: &[e4m3]) -> Vec<f32> {
    let mut out = vec![0.0; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for kk in 0..K {
                let scale_block = kk / (K / SCALES_FACTOR);
                let av = a[row * K + kk] * scales_a[row * SCALES_FACTOR + scale_block].to_f32();
                let bv = b[col * K + kk] * scales_b[col * SCALES_FACTOR + scale_block].to_f32();
                acc += av * bv;
            }
            out[row * N + col] = acc;
        }
    }
    out
}

fn run_probe(client: &Client) -> Result<(Vec<f32>, Vec<f32>, f32), String> {
    let (a_f32, a, scales_a, b_f32, b, scales_b) = make_inputs();
    let expected = host_reference(&a_f32, &scales_a, &b_f32, &scales_b);
    let zeros = vec![0.0f32; M * N];

    let a_handle = client.create_from_slice(e2m1x2::as_bytes(&a));
    let b_handle = client.create_from_slice(e2m1x2::as_bytes(&b));
    let c_handle = client.create_from_slice(f32::as_bytes(&zeros));
    let scales_a_handle = client.create_from_slice(e4m3::as_bytes(&scales_a));
    let scales_b_handle = client.create_from_slice(e4m3::as_bytes(&scales_b));
    let out_handle = client.create_from_slice(f32::as_bytes(&zeros));

    let ab_line_size = 32 / e2m1x2::cube_type().size_bits();

    unsafe {
        kernel_scaled_fp4_e4m3::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            TensorArg::from_raw_parts::<e2m1x2>(&a_handle, &[K / 2, 1], &[M, K / 2], ab_line_size),
            TensorArg::from_raw_parts::<e2m1x2>(&b_handle, &[K / 2, 1], &[N, K / 2], ab_line_size),
            TensorArg::from_raw_parts::<f32>(&c_handle, &[N, 1], &[M, N], 1),
            TensorArg::from_raw_parts::<e4m3>(
                &scales_a_handle,
                &[SCALES_FACTOR, 1],
                &[M, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<e4m3>(
                &scales_b_handle,
                &[SCALES_FACTOR, 1],
                &[N, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out_handle, &[N, 1], &[M, N], 1),
        )
        .map_err(|err| format!("{err:?}"))?;
    }

    cubecl::future::block_on(client.sync()).map_err(|err| format!("{err:?}"))?;
    let bytes = client.read_one(out_handle);
    let actual = f32::from_bytes(&bytes).to_vec();
    let max_abs_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);

    Ok((actual, expected, max_abs_diff))
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn print_matrix(label: &str, data: &[f32]) {
    println!("{label}");
    for row in 0..M {
        print!("  ");
        for col in 0..N {
            print!("{:9.3}", data[row * N + col]);
        }
        println!();
    }
}

fn main() {
    let device = Device::cuda(0);
    let client = R::client(&device);

    let config = chosen_config();
    let reports_support = client.properties().features.scaled_mma.contains(&config);
    println!("DEVICE_REPORTS_SCALED_MMA: {reports_support}");

    let launched = panic::catch_unwind(panic::AssertUnwindSafe(|| run_probe(&client)));
    match launched {
        Ok(Ok((actual, expected, max_abs_diff))) => {
            println!("LAUNCH: OK");
            print_matrix("OUTPUT:", &actual);
            print_matrix("REFERENCE:", &expected);
            let pass = max_abs_diff <= PASS_EPS;
            println!(
                "NUMERICS: max_abs_diff={max_abs_diff:.6} {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
        Ok(Err(err)) => {
            println!("LAUNCH: FAILED — {err}");
        }
        Err(payload) => {
            println!("LAUNCH: FAILED — {}", panic_payload_to_string(payload));
        }
    }
}
