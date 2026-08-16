//! P0.3b tiled NVFP4 GEMM probe.
//!
//! This is intentionally a raw CubeCL CUDA example, not a Burn/Fusion path. It answers the
//! Phase-0 question: can the canonical scaled-MMA path do a real K-loop tiled GEMM with the natural
//! full accumulator, staged through shared memory, on the sm_121 e2m1/e4m3 route?
//!
//! Build and run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example nvfp4_gemm_probe
//!   ./target/release/examples/nvfp4_gemm_probe

use std::{any::Any, panic, time::Instant};

use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::ir::MatrixIdent;
use cubecl::ir::features::ScaledMmaConfig;
use cubecl::prelude::*;
use cubecl::{CubeCount, CubeDim, Runtime, e2m1, e2m1x2, e4m3};

type R = CudaRuntime;
type Client = ComputeClient<R>;

const M: usize = 128;
const N: usize = 128;
const K: usize = 2048;

const MMA_M: usize = 16;
const MMA_N: usize = 8;
const MMA_K: usize = 64;
const SCALES_FACTOR: usize = 4;
const K_TILES: usize = K / MMA_K;

const WARPS_M: usize = 4;
const WARPS_N: usize = 2;
const WARPS_PER_CTA: usize = WARPS_M * WARPS_N;
const CTA_M: usize = WARPS_M * MMA_M;
const CTA_N: usize = WARPS_N * MMA_N;

const FP4_PACK: usize = 2;
const AB_LINE_SIZE: usize = 4;
const A_LINES_PER_ROW: usize = MMA_K / (FP4_PACK * AB_LINE_SIZE);
const B_LINES_PER_COL: usize = MMA_K / (FP4_PACK * AB_LINE_SIZE);
const A_STAGE_LINES: usize = CTA_M * A_LINES_PER_ROW;
const B_STAGE_LINES: usize = CTA_N * B_LINES_PER_COL;
const A_SCALE_ELEMS: usize = CTA_M * SCALES_FACTOR;
const B_SCALE_ELEMS: usize = CTA_N * SCALES_FACTOR;
const SHARED_BYTES: usize =
    (A_STAGE_LINES + B_STAGE_LINES) * AB_LINE_SIZE + A_SCALE_ELEMS + B_SCALE_ELEMS;
const SHARED_LIMIT_99KIB: usize = 99 * 1024;
const NVIDIA_MAX_WARPS_PER_SM: usize = 64;
const DOUBLE_BUFFERED: bool = false;
const SECOND_LEVEL_SCALE: f32 = -0.125;
const PASS_EPS: f32 = 2.5e-2;
const WARMUP_ITERS: usize = 3;
const BENCH_ITERS: usize = 20;

#[cube(launch)]
fn kernel_single_canonical_nvfp4(
    a: &Tensor<Line<e2m1x2>>,
    b: &Tensor<Line<e2m1x2>>,
    scales_a: &Tensor<e4m3>,
    scales_b: &Tensor<e4m3>,
    out: &mut Tensor<f32>,
) {
    let def = cmma::MmaDefinition::<e2m1x2, e2m1x2, f32>::new_scaled::<e4m3>(
        MMA_M,
        MMA_N,
        MMA_K,
        SCALES_FACTOR,
    );
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

    #[unroll]
    for i in 0..line_count_a {
        let n_elem = i * line_size_a * FP4_PACK;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
        let idx = row as usize * MMA_K + col as usize;
        registers_a[i] = a[idx / (a.line_size() * FP4_PACK)];
    }

    #[unroll]
    for i in 0..line_count_b {
        let n_elem = i * line_size_b * FP4_PACK;
        let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
        let idx = col as usize * MMA_K + row as usize;
        registers_b[i] = b[idx / (b.line_size() * FP4_PACK)];
    }

    #[unroll]
    for i in 0..line_count_c {
        let mut zero = Line::<f32>::empty(line_size_c);
        #[unroll]
        for j in 0..line_size_c {
            zero[j] = 0.0;
        }
        registers_c[i] = zero;
    }

    let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
    #[unroll]
    for i in 0..scales_count {
        scales_register_a[i] = scales_a[scales_idx_a as usize * SCALES_FACTOR + i];
    }
    let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
    #[unroll]
    for i in 0..scales_count {
        scales_register_b[i] = scales_b[scales_idx_b as usize * SCALES_FACTOR + i];
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
            out[row as usize * MMA_N + col as usize] = reg[j] * SECOND_LEVEL_SCALE;
        }
    }
}

#[cube(launch)]
fn kernel_tiled_nvfp4(
    a: &Tensor<Line<e2m1x2>>,
    b: &Tensor<Line<e2m1x2>>,
    scales_a: &Tensor<e4m3>,
    scales_b: &Tensor<e4m3>,
    out: &mut Tensor<f32>,
) {
    let def = cmma::MmaDefinition::<e2m1x2, e2m1x2, f32>::new_scaled::<e4m3>(
        MMA_M,
        MMA_N,
        MMA_K,
        SCALES_FACTOR,
    );

    let mut stage_a = SharedMemory::<e2m1x2>::new_lined(A_STAGE_LINES, AB_LINE_SIZE);
    let mut stage_b = SharedMemory::<e2m1x2>::new_lined(B_STAGE_LINES, AB_LINE_SIZE);
    let mut stage_scales_a = SharedMemory::<e4m3>::new(A_SCALE_ELEMS);
    let mut stage_scales_b = SharedMemory::<e4m3>::new(B_SCALE_ELEMS);

    let lane_id = UNIT_POS_PLANE;
    let warp_id = PLANE_POS as usize;
    let warp_m = warp_id / WARPS_N;
    let warp_n = warp_id % WARPS_N;
    let cta_m = CUBE_POS_Y as usize * CTA_M;
    let cta_n = CUBE_POS_X as usize * CTA_N;
    let tile_m = cta_m + warp_m * MMA_M;
    let tile_n = cta_n + warp_n * MMA_N;

    let elem_count_c = def.elems_per_lane(MatrixIdent::Accumulator);
    let line_size_c = def.line_size(MatrixIdent::Accumulator);
    let line_count_c = comptime!(elem_count_c / line_size_c);
    let mut registers_c = Array::<Line<f32>>::lined(line_count_c, line_size_c);

    #[unroll]
    for i in 0..line_count_c {
        let mut zero = Line::<f32>::empty(line_size_c);
        #[unroll]
        for j in 0..line_size_c {
            zero[j] = 0.0;
        }
        registers_c[i] = zero;
    }

    let elem_count_a = def.elems_per_lane(MatrixIdent::A);
    let line_size_a = def.line_size(MatrixIdent::A);
    let line_count_a = comptime!(elem_count_a / line_size_a);
    let elem_count_b = def.elems_per_lane(MatrixIdent::B);
    let line_size_b = def.line_size(MatrixIdent::B);
    let line_count_b = comptime!(elem_count_b / line_size_b);
    let scales_count = def.scales_count();

    #[unroll]
    for kt in 0..K_TILES {
        let unit = UNIT_POS as usize;

        if unit < A_STAGE_LINES {
            let row = unit / A_LINES_PER_ROW;
            let col_line = unit % A_LINES_PER_ROW;
            let global =
                (cta_m + row) * (K / (FP4_PACK * AB_LINE_SIZE)) + kt * A_LINES_PER_ROW + col_line;
            stage_a[unit] = a[global];
        }
        let unit_a2 = unit + WARPS_PER_CTA * 32;
        if unit_a2 < A_STAGE_LINES {
            let row = unit_a2 / A_LINES_PER_ROW;
            let col_line = unit_a2 % A_LINES_PER_ROW;
            let global =
                (cta_m + row) * (K / (FP4_PACK * AB_LINE_SIZE)) + kt * A_LINES_PER_ROW + col_line;
            stage_a[unit_a2] = a[global];
        }

        if unit < B_STAGE_LINES {
            let col = unit / B_LINES_PER_COL;
            let row_line = unit % B_LINES_PER_COL;
            let global =
                (cta_n + col) * (K / (FP4_PACK * AB_LINE_SIZE)) + kt * B_LINES_PER_COL + row_line;
            stage_b[unit] = b[global];
        }

        if unit < A_SCALE_ELEMS {
            let row = unit / SCALES_FACTOR;
            let scale = unit % SCALES_FACTOR;
            stage_scales_a[unit] = scales_a[((cta_m + row) * K_TILES + kt) * SCALES_FACTOR + scale];
        }
        if unit < B_SCALE_ELEMS {
            let col = unit / SCALES_FACTOR;
            let scale = unit % SCALES_FACTOR;
            stage_scales_b[unit] = scales_b[((cta_n + col) * K_TILES + kt) * SCALES_FACTOR + scale];
        }

        sync_cube();

        let mut registers_a = Array::<Line<e2m1x2>>::lined(line_count_a, line_size_a);
        let mut registers_b = Array::<Line<e2m1x2>>::lined(line_count_b, line_size_b);
        let mut scales_register_a = Line::<e4m3>::empty(def.scales_line_size());
        let mut scales_register_b = Line::<e4m3>::empty(def.scales_line_size());

        #[unroll]
        for i in 0..line_count_a {
            let n_elem = i * line_size_a * FP4_PACK;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::A);
            let local_row = warp_m * MMA_M + row as usize;
            let local_line = local_row * A_LINES_PER_ROW + col as usize / (line_size_a * FP4_PACK);
            registers_a[i] = stage_a[local_line];
        }

        #[unroll]
        for i in 0..line_count_b {
            let n_elem = i * line_size_b * FP4_PACK;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::B);
            let local_col = warp_n * MMA_N + col as usize;
            let local_line = local_col * B_LINES_PER_COL + row as usize / (line_size_b * FP4_PACK);
            registers_b[i] = stage_b[local_line];
        }

        let scales_idx_a = def.scales_index(lane_id, MatrixIdent::A);
        #[unroll]
        for i in 0..scales_count {
            scales_register_a[i] =
                stage_scales_a[(warp_m * MMA_M + scales_idx_a as usize) * SCALES_FACTOR + i];
        }
        let scales_idx_b = def.scales_index(lane_id, MatrixIdent::B);
        #[unroll]
        for i in 0..scales_count {
            scales_register_b[i] =
                stage_scales_b[(warp_n * MMA_N + scales_idx_b as usize) * SCALES_FACTOR + i];
        }

        // The decisive path: one canonical full-tile scaled MMA updates the natural accumulator.
        // There is deliberately no two-call rows-0..7 / rows-8..15 workaround here.
        let registers_d = def.execute_scaled(
            &registers_a,
            &registers_b,
            &registers_c,
            scales_register_a,
            scales_register_b,
        );
        #[unroll]
        for i in 0..line_count_c {
            registers_c[i] = registers_d[i];
        }

        sync_cube();
    }

    #[unroll]
    for i in 0..line_count_c {
        let reg = registers_c[i];
        #[unroll]
        for j in 0..line_size_c {
            let value = reg[j] * SECOND_LEVEL_SCALE;
            let n_elem = i * line_size_c + j;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, MatrixIdent::Accumulator);
            let idx = (tile_m + row as usize) * N + tile_n + col as usize;
            out[idx] = value;
        }
    }
}

fn chosen_config() -> ScaledMmaConfig {
    ScaledMmaConfig {
        a_type: e2m1x2::cube_type(),
        b_type: e2m1x2::cube_type(),
        cd_type: f32::cube_type(),
        scales_type: e4m3::cube_type(),
        m: MMA_M as u32,
        n: MMA_N as u32,
        k: MMA_K as u32,
        scales_factor: SCALES_FACTOR as u32,
    }
}

fn quantized_e2m1(seed: usize) -> f32 {
    let raw = [
        -6.7, -4.4, -3.1, -2.3, -1.7, -1.2, -0.7, -0.2, 0.2, 0.7, 1.2, 1.7, 2.3, 3.1, 4.4, 6.7,
    ];
    e2m1::from_f32(raw[seed % raw.len()]).to_f32()
}

fn scale_a(row: usize, kt: usize, block: usize) -> e4m3 {
    let vals = [-1.5, -0.75, -0.5, 0.5, 0.75, 1.0, 1.5, 2.0];
    e4m3::from_f32(vals[(row * 17 + kt * 5 + block * 3) % vals.len()])
}

fn scale_b(col: usize, kt: usize, block: usize) -> e4m3 {
    let vals = [-1.25, -0.625, 0.375, 0.5, 0.875, 1.25, 1.75, 2.5];
    e4m3::from_f32(vals[(col * 13 + kt * 7 + block * 5 + 1) % vals.len()])
}

fn make_inputs() -> (
    Vec<f32>,
    Vec<e2m1x2>,
    Vec<e4m3>,
    Vec<f32>,
    Vec<e2m1x2>,
    Vec<e4m3>,
) {
    let a_f32: Vec<f32> = (0..M)
        .flat_map(|row| (0..K).map(move |k| quantized_e2m1(row * 31 + k * 7 + (row * k) % 19)))
        .collect();
    let a = e2m1x2::from_f32_slice(&a_f32);

    // B is stored col-major: B[col*K + k] is logical B[k,col].
    let b_f32: Vec<f32> = (0..N)
        .flat_map(|col| (0..K).map(move |k| quantized_e2m1(col * 29 + k * 11 + (col * k) % 23)))
        .collect();
    let b = e2m1x2::from_f32_slice(&b_f32);

    let scales_a: Vec<e4m3> = (0..M)
        .flat_map(|row| {
            (0..K_TILES)
                .flat_map(move |kt| (0..SCALES_FACTOR).map(move |block| scale_a(row, kt, block)))
        })
        .collect();
    let scales_b: Vec<e4m3> = (0..N)
        .flat_map(|col| {
            (0..K_TILES)
                .flat_map(move |kt| (0..SCALES_FACTOR).map(move |block| scale_b(col, kt, block)))
        })
        .collect();

    (a_f32, a, scales_a, b_f32, b, scales_b)
}

fn host_reference(
    a: &[f32],
    scales_a: &[e4m3],
    b: &[f32],
    scales_b: &[e4m3],
    abs_scales: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; M * N];
    let k_block = MMA_K / SCALES_FACTOR;
    for row in 0..M {
        for col in 0..N {
            let mut acc = 0.0f32;
            for kk in 0..K {
                let kt = kk / MMA_K;
                let block = (kk % MMA_K) / k_block;
                let mut sa = scales_a[(row * K_TILES + kt) * SCALES_FACTOR + block].to_f32();
                let mut sb = scales_b[(col * K_TILES + kt) * SCALES_FACTOR + block].to_f32();
                if abs_scales {
                    sa = sa.abs();
                    sb = sb.abs();
                }
                let av = a[row * K + kk] * sa;
                let bv = b[col * K + kk] * sb;
                acc += av * bv;
            }
            out[row * N + col] = acc * SECOND_LEVEL_SCALE;
        }
    }
    out
}

fn run_single_canonical_probe(
    client: &Client,
    positive_scales_only: bool,
) -> Result<(f32, f32), String> {
    let a_f32: Vec<f32> = (0..MMA_M)
        .flat_map(|row| (0..MMA_K).map(move |k| quantized_e2m1(row * 31 + k * 7 + (row * k) % 19)))
        .collect();
    let a = e2m1x2::from_f32_slice(&a_f32);
    let b_f32: Vec<f32> = (0..MMA_N)
        .flat_map(|col| (0..MMA_K).map(move |k| quantized_e2m1(col * 29 + k * 11 + (col * k) % 23)))
        .collect();
    let b = e2m1x2::from_f32_slice(&b_f32);
    let scales_a: Vec<e4m3> = (0..MMA_M)
        .flat_map(|row| {
            (0..SCALES_FACTOR).map(move |block| {
                let value = scale_a(row, 0, block).to_f32();
                e4m3::from_f32(if positive_scales_only {
                    value.abs()
                } else {
                    value
                })
            })
        })
        .collect();
    let scales_b: Vec<e4m3> = (0..MMA_N)
        .flat_map(|col| {
            (0..SCALES_FACTOR).map(move |block| {
                let value = scale_b(col, 0, block).to_f32();
                e4m3::from_f32(if positive_scales_only {
                    value.abs()
                } else {
                    value
                })
            })
        })
        .collect();
    let zeros = vec![0.0f32; MMA_M * MMA_N];

    let mut expected = vec![0.0f32; MMA_M * MMA_N];
    let k_block = MMA_K / SCALES_FACTOR;
    for row in 0..MMA_M {
        for col in 0..MMA_N {
            let mut acc = 0.0f32;
            for kk in 0..MMA_K {
                let block = kk / k_block;
                let av = a_f32[row * MMA_K + kk] * scales_a[row * SCALES_FACTOR + block].to_f32();
                let bv = b_f32[col * MMA_K + kk] * scales_b[col * SCALES_FACTOR + block].to_f32();
                acc += av * bv;
            }
            expected[row * MMA_N + col] = acc * SECOND_LEVEL_SCALE;
        }
    }

    let a_handle = client.create_from_slice(e2m1x2::as_bytes(&a));
    let b_handle = client.create_from_slice(e2m1x2::as_bytes(&b));
    let scales_a_handle = client.create_from_slice(e4m3::as_bytes(&scales_a));
    let scales_b_handle = client.create_from_slice(e4m3::as_bytes(&scales_b));
    let out_handle = client.create_from_slice(f32::as_bytes(&zeros));

    unsafe {
        kernel_single_canonical_nvfp4::launch::<R>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim { x: 32, y: 1, z: 1 },
            TensorArg::from_raw_parts::<e2m1x2>(
                &a_handle,
                &[MMA_K / FP4_PACK, 1],
                &[MMA_M, MMA_K / FP4_PACK],
                AB_LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<e2m1x2>(
                &b_handle,
                &[MMA_K / FP4_PACK, 1],
                &[MMA_N, MMA_K / FP4_PACK],
                AB_LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<e4m3>(
                &scales_a_handle,
                &[SCALES_FACTOR, 1],
                &[MMA_M, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<e4m3>(
                &scales_b_handle,
                &[SCALES_FACTOR, 1],
                &[MMA_N, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(&out_handle, &[MMA_N, 1], &[MMA_M, MMA_N], 1),
        )
        .map_err(|err| format!("{err:?}"))?;
    }
    sync_client(client)?;

    let bytes = client.read_one(out_handle);
    let actual = f32::from_bytes(&bytes).to_vec();
    let mut top = 0.0f32;
    let mut bottom = 0.0f32;
    for row in 0..8 {
        for col in 0..MMA_N {
            let idx = row * MMA_N + col;
            top = top.max((actual[idx] - expected[idx]).abs());
        }
    }
    for row in 8..16 {
        for col in 0..MMA_N {
            let idx = row * MMA_N + col;
            bottom = bottom.max((actual[idx] - expected[idx]).abs());
        }
    }
    Ok((top, bottom))
}

fn sync_client(client: &Client) -> Result<(), String> {
    cubecl::future::block_on(client.sync()).map_err(|err| format!("{err:?}"))
}

fn launch_kernel(
    client: &Client,
    a_handle: &cubecl::server::Handle,
    b_handle: &cubecl::server::Handle,
    scales_a_handle: &cubecl::server::Handle,
    scales_b_handle: &cubecl::server::Handle,
    out_handle: &cubecl::server::Handle,
) -> Result<(), String> {
    unsafe {
        kernel_tiled_nvfp4::launch::<R>(
            client,
            CubeCount::Static((N / CTA_N) as u32, (M / CTA_M) as u32, 1),
            CubeDim {
                x: 32,
                y: WARPS_PER_CTA as u32,
                z: 1,
            },
            TensorArg::from_raw_parts::<e2m1x2>(
                a_handle,
                &[K / FP4_PACK, 1],
                &[M, K / FP4_PACK],
                AB_LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<e2m1x2>(
                b_handle,
                &[K / FP4_PACK, 1],
                &[N, K / FP4_PACK],
                AB_LINE_SIZE,
            ),
            TensorArg::from_raw_parts::<e4m3>(
                scales_a_handle,
                &[K_TILES * SCALES_FACTOR, SCALES_FACTOR, 1],
                &[M, K_TILES, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<e4m3>(
                scales_b_handle,
                &[K_TILES * SCALES_FACTOR, SCALES_FACTOR, 1],
                &[N, K_TILES, SCALES_FACTOR],
                1,
            ),
            TensorArg::from_raw_parts::<f32>(out_handle, &[N, 1], &[M, N], 1),
        )
        .map_err(|err| format!("{err:?}"))?;
    }
    Ok(())
}

fn run_probe(client: &Client) -> Result<(Vec<f32>, Vec<f32>, f32, f32, f64, f64), String> {
    let (a_f32, a, scales_a, b_f32, b, scales_b) = make_inputs();
    let expected = host_reference(&a_f32, &scales_a, &b_f32, &scales_b, false);
    let expected_abs_scales = host_reference(&a_f32, &scales_a, &b_f32, &scales_b, true);
    let zeros = vec![0.0f32; M * N];

    let a_handle = client.create_from_slice(e2m1x2::as_bytes(&a));
    let b_handle = client.create_from_slice(e2m1x2::as_bytes(&b));
    let scales_a_handle = client.create_from_slice(e4m3::as_bytes(&scales_a));
    let scales_b_handle = client.create_from_slice(e4m3::as_bytes(&scales_b));
    let out_handle = client.create_from_slice(f32::as_bytes(&zeros));

    for _ in 0..WARMUP_ITERS {
        launch_kernel(
            client,
            &a_handle,
            &b_handle,
            &scales_a_handle,
            &scales_b_handle,
            &out_handle,
        )?;
    }
    sync_client(client)?;

    let start = Instant::now();
    for _ in 0..BENCH_ITERS {
        launch_kernel(
            client,
            &a_handle,
            &b_handle,
            &scales_a_handle,
            &scales_b_handle,
            &out_handle,
        )?;
    }
    sync_client(client)?;
    let ms = start.elapsed().as_secs_f64() * 1.0e3 / BENCH_ITERS as f64;
    let tflops = (2.0 * M as f64 * N as f64 * K as f64) / (ms * 1.0e-3) / 1.0e12;

    let bytes = client.read_one(out_handle);
    let actual = f32::from_bytes(&bytes).to_vec();
    let max_abs_diff = actual
        .iter()
        .zip(expected.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_abs_diff_abs_scales = actual
        .iter()
        .zip(expected_abs_scales.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);

    Ok((
        actual,
        expected,
        max_abs_diff,
        max_abs_diff_abs_scales,
        ms,
        tflops,
    ))
}

fn max_abs_diff_range(actual: &[f32], expected: &[f32], rows: std::ops::Range<usize>) -> f32 {
    rows.flat_map(|row| {
        (0..N).map(move |col| {
            let idx = row * N + col;
            (actual[idx] - expected[idx]).abs()
        })
    })
    .fold(0.0f32, f32::max)
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

fn main() {
    let device = Device::cuda(0);
    let client = R::client(&device);
    let props = client.properties();
    let config = chosen_config();
    let reports_support = props.features.scaled_mma.contains(&config);
    let max_smem = props.hardware.max_shared_memory_size;
    let smem_bound_ctas = if SHARED_BYTES == 0 {
        0
    } else {
        max_smem / SHARED_BYTES
    };
    let smem_bound_warps = (smem_bound_ctas * WARPS_PER_CTA).min(NVIDIA_MAX_WARPS_PER_SM);

    println!(
        "P0.3b NVFP4 tiled GEMM probe: M={M} N={N} K={K}, CTA={CTA_M}x{CTA_N}, warps/CTA={WARPS_PER_CTA}"
    );
    println!("DEVICE_REPORTS_SCALED_MMA_E2M1_E4M3: {reports_support}");
    println!(
        "SHARED_MEMORY: used={SHARED_BYTES} B, 99KiB_limit={SHARED_LIMIT_99KIB} B, device_limit={max_smem} B, fits_99KiB={}, smem_bound_ctas/SM={smem_bound_ctas}, smem_bound_occupancy<={} warps/SM, double_buffered={DOUBLE_BUFFERED}",
        SHARED_BYTES <= SHARED_LIMIT_99KIB,
        smem_bound_warps
    );
    println!(
        "OCCUPANCY_NOTE: CubeCL exposes SM count ({:?}) but not post-JIT achieved occupancy here; printed occupancy is the shared-memory upper bound capped at {NVIDIA_MAX_WARPS_PER_SM} warps/SM.",
        props.hardware.num_streaming_multiprocessors
    );
    println!(
        "PIPELINE_NOTE: A, B, and per-block scales are staged through shared memory; this probe uses synchronous barriers, not an async double-buffered K-loop."
    );
    println!(
        "NUMERICS: second_level_scale={SECOND_LEVEL_SCALE}, signed_e4m3_scales=true, rounded_e2m1_inputs=true"
    );

    match panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_single_canonical_probe(&client, false)
    })) {
        Ok(Ok((top, bottom))) => {
            println!(
                "CANONICAL_SINGLE_MMA_SINGLE_TILE_SIGNED_SCALES: rows0_7_max_abs_diff={top:.6}, rows8_15_max_abs_diff={bottom:.6} {}",
                if top <= PASS_EPS && bottom <= PASS_EPS {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
        }
        Ok(Err(err)) => {
            println!("CANONICAL_SINGLE_MMA_SINGLE_TILE_SIGNED_SCALES: FAILED_TO_RUN - {err}")
        }
        Err(payload) => println!(
            "CANONICAL_SINGLE_MMA_SINGLE_TILE_SIGNED_SCALES: PANIC - {}",
            panic_payload_to_string(payload)
        ),
    }
    match panic::catch_unwind(panic::AssertUnwindSafe(|| {
        run_single_canonical_probe(&client, true)
    })) {
        Ok(Ok((top, bottom))) => {
            println!(
                "CANONICAL_SINGLE_MMA_SINGLE_TILE_POSITIVE_SCALES: rows0_7_max_abs_diff={top:.6}, rows8_15_max_abs_diff={bottom:.6} {}",
                if top <= PASS_EPS && bottom <= PASS_EPS {
                    "PASS"
                } else {
                    "FAIL"
                }
            );
            if top <= PASS_EPS && bottom > PASS_EPS {
                println!(
                    "BUG_FINDING: positive-scale canonical single-MMA full tile passes rows 0-7 but fails rows 8-15 on e4m3/scales_factor=4."
                );
            }
        }
        Ok(Err(err)) => {
            println!("CANONICAL_SINGLE_MMA_SINGLE_TILE_POSITIVE_SCALES: FAILED_TO_RUN - {err}")
        }
        Err(payload) => println!(
            "CANONICAL_SINGLE_MMA_SINGLE_TILE_POSITIVE_SCALES: PANIC - {}",
            panic_payload_to_string(payload)
        ),
    }

    let launched = panic::catch_unwind(panic::AssertUnwindSafe(|| run_probe(&client)));
    match launched {
        Ok(Ok((actual, expected, max_abs_diff, max_abs_diff_abs_scales, ms, tflops))) => {
            let pass = max_abs_diff <= PASS_EPS;
            let top = max_abs_diff_range(&actual, &expected, 0..8);
            let bottom = max_abs_diff_range(&actual, &expected, 8..16);
            let canonical_pass = top <= PASS_EPS && bottom <= PASS_EPS;
            println!(
                "CANONICAL_SINGLE_MMA_FULL_TILE: rows0_7_max_abs_diff={top:.6}, rows8_15_max_abs_diff={bottom:.6} {}",
                if canonical_pass { "PASS" } else { "FAIL" }
            );
            if top <= PASS_EPS && bottom > PASS_EPS {
                println!(
                    "BUG_FINDING: rows 8-15 fail on the natural one execute_scaled e4m3/scales_factor=4 path; no workaround was used."
                );
            }
            println!(
                "NVFP4_GEMM: max_abs_diff={max_abs_diff:.6} {}, time={ms:.4} ms, throughput={tflops:.3} TFLOP/s",
                if pass { "PASS" } else { "FAIL" }
            );
            println!(
                "SIGNED_SCALE_DIAGNOSTIC: max_abs_diff_if_scale_signs_ignored={max_abs_diff_abs_scales:.6}"
            );
            println!(
                "BF16_BASELINE: not measured in this one-file probe; NVFP4/bf16 ratio unavailable."
            );
        }
        Ok(Err(err)) => {
            println!("LAUNCH: FAILED - {err}");
            println!(
                "BLOCKER: tiled canonical NVFP4 kernel did not complete, so correctness/perf are unavailable."
            );
        }
        Err(payload) => {
            println!("LAUNCH: FAILED - {}", panic_payload_to_string(payload));
            println!("BLOCKER: tiled canonical NVFP4 kernel panicked before completion.");
        }
    }
}
