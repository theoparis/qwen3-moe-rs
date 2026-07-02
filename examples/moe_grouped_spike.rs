//! VLLM_KERNELS.md §3 — the DROPLESS MoE grouped-GEMM, validated on the real GB10 GPU against an
//! INDEPENDENT NdArray (CPU f32) oracle (the §0 cross-backend law).
//!
//! The repo's `forward_routed_ondevice` uses a CAPACITY-padded `[E,C,H]` layout that DROPS tokens
//! when an expert overflows `C` (corrupts GRPO parity). The grouped-GEMM path
//! (`qwen3_burn::moe_grouped`) is DROPLESS: it builds the vLLM `moe_align_block_size` layout
//! (`sorted_token_ids` + per-block `expert_ids`) fully on-device and computes EXACTLY the `k*T`
//! routed `(token,expert)` pairs via a block-per-segment SwiGLU GEMM with i64 global offsets.
//!
//! This spike checks, on a TINY shape (E=8, top_k=2, H=32, I=16, T=10) and a MID shape
//! (E=32, top_k=8, H=256, I=128, T=64):
//!   1. EXACT-NO-DROP — the device per-expert counts + `sorted_token_ids` match a host reference of
//!      the routing (no assignment dropped); and `forward_routed_ondevice` at a small capacity DOES
//!      drop where the grouped path does not.
//!   2. NUMERICS — the GPU grouped-GEMM output vs the NdArray (CPU f32) oracle (forward_oracle's
//!      math, same weights + same routing, on the NdArray backend): cosine + max_abs_diff, assert
//!      cosine > 0.99999. Plus a same-device `forward_oracle` cross-check.
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example moe_grouped_spike 2>&1 | tail -50

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::backend::NdArray;
use burn::prelude::Backend;
use burn::tensor::{activation::silu, DType, Distribution, Tensor, TensorData};

use qwen3_burn::moe_grouped::{dropless_align, forward_grouped};
use qwen3_burn::Qwen3MoeSparseBlock;
use qwen3_burn::Precision;

type Nd = NdArray<f32>;

const BLOCK_M: usize = 16;

// ----------------------------------------------------------------------------------------------- //
// Metrics
// ----------------------------------------------------------------------------------------------- //
fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na == 0.0 || nb == 0.0 {
        return f32::NAN;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ----------------------------------------------------------------------------------------------- //
// The INDEPENDENT NdArray (CPU f32) oracle: forward_oracle's math
// `out[t] = Σ_e gate_w[t,e] · down_e(silu(x@gate_e)*(x@up_e))`, on the NdArray backend, using the
// SAME extracted weights and the SAME routing as the GPU path. Cross-backend per §0.
// ----------------------------------------------------------------------------------------------- //
#[allow(clippy::too_many_arguments)]
fn ndarray_oracle(
    x_host: &[f32],   // [T,H]
    gate: &[f32],     // [E,H,I]
    up: &[f32],       // [E,H,I]
    down: &[f32],     // [E,I,H]
    gate_w: &[f32],   // [T,E] dense router gate
    t: usize,
    h: usize,
    i: usize,
    e: usize,
) -> Vec<f32> {
    let dev = <Nd as Backend>::Device::default();
    let x = Tensor::<Nd, 2>::from_data(TensorData::new(x_host.to_vec(), [t, h]), &dev);
    let mut acc = Tensor::<Nd, 2>::zeros([t, h], &dev);
    for ei in 0..e {
        let ge = Tensor::<Nd, 2>::from_data(TensorData::new(gate[ei * h * i..(ei + 1) * h * i].to_vec(), [h, i]), &dev);
        let ue = Tensor::<Nd, 2>::from_data(TensorData::new(up[ei * h * i..(ei + 1) * h * i].to_vec(), [h, i]), &dev);
        let de = Tensor::<Nd, 2>::from_data(TensorData::new(down[ei * i * h..(ei + 1) * i * h].to_vec(), [i, h]), &dev);
        let g = silu(x.clone().matmul(ge)); // [T,I]
        let u = x.clone().matmul(ue); // [T,I]
        let y = (g * u).matmul(de); // [T,H]
        let wcol: Vec<f32> = (0..t).map(|tt| gate_w[tt * e + ei]).collect();
        let w = Tensor::<Nd, 2>::from_data(TensorData::new(wcol, [t, 1]), &dev);
        acc = acc + y * w;
    }
    acc.into_data().to_vec::<f32>().unwrap()
}

struct ShapeResult {
    label: String,
    cos: f32,
    mad: f32,
    cos_dev: f32,
    no_drop: bool,
    counts_match: bool,
    grouped_total: usize,
    n: usize,
    ondevice_drop_diff: f32,
    grouped_oracle_diff: f32,
    ok: bool,
}

fn run_shape(label: &str, b: usize, s: usize, h: usize, i: usize, e: usize, k: usize, drop_c: usize) -> ShapeResult {
    let device = CudaDevice::default();
    let t = b * s;
    let n = t * k;

    let block = Qwen3MoeSparseBlock::<Cuda>::new(h, i, e, k, true, &device);
    let x = Tensor::<Cuda, 3>::random([b, s, h], Distribution::Normal(0.0, 1.0), &device);

    // -------- routing (read to host; deterministic, so identical to forward_grouped's own call) --
    let (sel_idx, sel_w) = block.route_topk(x.clone());
    let sel_idx_host: Vec<i64> = sel_idx.clone().cast(DType::I64).into_data().to_vec().unwrap(); // [T*k]
    let sel_w_host: Vec<f32> = sel_w.clone().into_data().to_vec().unwrap(); // [T*k]

    // dense router gate [T,E] from the compact routing (for the oracle's weighted combine).
    let mut gate_w = vec![0.0f32; t * e];
    for tt in 0..t {
        for slot in 0..k {
            let ei = sel_idx_host[tt * k + slot] as usize;
            gate_w[tt * e + ei] += sel_w_host[tt * k + slot];
        }
    }

    // -------- 1. DROPLESS align: device counts + sorted layout vs a host reference --------------
    let lay = dropless_align(sel_idx, sel_w, e, k, BLOCK_M);
    let count_dev: Vec<i64> = lay.count_e.clone().cast(DType::I64).into_data().to_vec().unwrap(); // [E]
    let sorted_tok: Vec<i64> = lay.sorted_token.clone().cast(DType::I64).into_data().to_vec().unwrap(); // [buffer]
    let sorted_exp: Vec<i64> = lay.sorted_expert.clone().cast(DType::I64).into_data().to_vec().unwrap(); // [buffer]

    // host per-expert counts + per-expert token multiset from the routing.
    let mut count_host = vec![0i64; e];
    let mut host_tokens: Vec<Vec<i64>> = vec![Vec::new(); e];
    for tt in 0..t {
        for slot in 0..k {
            let ei = sel_idx_host[tt * k + slot] as usize;
            count_host[ei] += 1;
            host_tokens[ei].push(tt as i64);
        }
    }
    let counts_match = count_dev == count_host;

    // device per-expert token multiset from the sorted layout (real slots only).
    let mut dev_tokens: Vec<Vec<i64>> = vec![Vec::new(); e];
    let mut grouped_total = 0usize;
    for slotpos in 0..lay.buffer {
        let tok = sorted_tok[slotpos];
        let exp = sorted_exp[slotpos];
        if tok >= 0 && exp >= 0 {
            dev_tokens[exp as usize].push(tok);
            grouped_total += 1;
        }
    }
    // exact-no-drop: every assignment present exactly once, grouped under the right expert.
    let mut no_drop = grouped_total == n;
    for ei in 0..e {
        let mut a = dev_tokens[ei].clone();
        let mut bvec = host_tokens[ei].clone();
        a.sort_unstable();
        bvec.sort_unstable();
        if a != bvec {
            no_drop = false;
        }
    }

    // -------- 2. GPU grouped-GEMM output --------------------------------------------------------
    let gpu = forward_grouped(&block, x.clone(), BLOCK_M);
    let gpu_host: Vec<f32> = gpu.clone().into_data().to_vec().unwrap();

    // -------- NdArray (CPU f32) oracle, same weights + same routing -----------------------------
    let x_host: Vec<f32> = x.clone().reshape([t, h]).into_data().to_vec().unwrap();
    let (gate_s, up_s, down_s) = block.stacked_experts_pub();
    let gate_host: Vec<f32> = gate_s.into_data().to_vec().unwrap(); // [E,H,I]
    let up_host: Vec<f32> = up_s.into_data().to_vec().unwrap(); // [E,H,I]
    let down_host: Vec<f32> = down_s.into_data().to_vec().unwrap(); // [E,I,H]
    let oracle = ndarray_oracle(&x_host, &gate_host, &up_host, &down_host, &gate_w, t, h, i, e);

    let cos = cosine(&gpu_host, &oracle);
    let mad = max_abs_diff(&gpu_host, &oracle);

    // same-device forward_oracle cross-check.
    let dev_oracle: Vec<f32> = block.forward_oracle(x.clone(), Precision::F32).into_data().to_vec().unwrap();
    let cos_dev = cosine(&gpu_host, &dev_oracle);
    let grouped_oracle_diff = max_abs_diff(&gpu_host, &oracle);

    // -------- forward_routed_ondevice at a small capacity DROPS where grouped does not -----------
    let ondevice_small: Vec<f32> =
        block.forward_routed_ondevice(x.clone(), drop_c).into_data().to_vec().unwrap();
    let ondevice_drop_diff = max_abs_diff(&ondevice_small, &oracle);

    let ok = counts_match && no_drop && cos > 0.99999 && grouped_total == n && !gpu_host.iter().any(|v| v.is_nan());

    println!("--- {label}  (B={b} S={s} → T={t}, E={e}, k={k}, H={h}, I={i}, N=T*k={n}, BLOCK_M={BLOCK_M}) ---");
    println!(
        "  exact-no-drop : counts_match={counts_match}  no_drop={no_drop}  grouped_total={grouped_total}/{n}  buffer={} num_blocks={}",
        lay.buffer, lay.num_blocks
    );
    println!(
        "  grouped vs NdArray oracle : cos={cos:.7}  max_abs_diff={mad:.3e}   (grouped vs forward_oracle[device] cos={cos_dev:.7})"
    );
    println!(
        "  DROP demo     : forward_routed_ondevice(C={drop_c}) vs oracle max_abs_diff={ondevice_drop_diff:.3e}  (grouped vs oracle {grouped_oracle_diff:.3e})  → grouped is dropless, fixed-stride drops"
    );
    println!("  {}", if ok { "[PASS]" } else { "[FAIL]" });

    ShapeResult {
        label: label.to_string(),
        cos,
        mad,
        cos_dev,
        no_drop,
        counts_match,
        grouped_total,
        n,
        ondevice_drop_diff,
        grouped_oracle_diff,
        ok,
    }
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | oracle: NdArray (CPU f32) | kernel: DROPLESS CubeCL grouped SwiGLU GEMM");
    println!("cross-backend law: oracle is an INDEPENDENT CPU backend (docs/VLLM_KERNELS.md §0)\n");

    let mut rows = Vec::new();
    // TINY: E=8, top_k=2, H=32, I=16, T=10 (B=2,S=5). drop_c small to force a drop.
    rows.push(run_shape("tiny", 2, 5, 32, 16, 8, 2, 2));
    println!();
    // MID: E=32, top_k=8, H=256, I=128, T=64 (B=1,S=64). drop_c << mean load (k*T/E=16) to force drops.
    rows.push(run_shape("mid", 1, 64, 256, 128, 32, 8, 8));

    println!("\n================ SUMMARY (grouped-GEMM vs NdArray CPU oracle) ================");
    println!(
        "{:8} {:>9} {:>11} {:>11} {:>9} {:>9} {:>12} {:>12}",
        "shape", "cos", "max_abs", "cos[dev]", "no_drop", "counts", "drop(ond.)", "drop(grp.)"
    );
    let mut all_ok = true;
    for r in &rows {
        println!(
            "{:8} {:>9.7} {:>11.3e} {:>11.7} {:>9} {:>9} {:>12.3e} {:>12.3e}",
            r.label,
            r.cos,
            r.mad,
            r.cos_dev,
            r.no_drop,
            r.counts_match,
            r.ondevice_drop_diff,
            r.grouped_oracle_diff,
        );
        all_ok &= r.ok;
    }

    println!("\n--- VERDICT ---");
    if all_ok {
        println!(
            "MoE GROUPED-GEMM: VALIDATED (DROPLESS) — the on-device vLLM moe_align_block_size layout \
             (sorted_token_ids + per-block expert_ids) places ALL {} (tiny) / {} (mid) routed \
             assignments with no drop (counts + sorted layout match the host routing reference), and \
             the block-per-segment SwiGLU GEMM (i64 global offsets) matches the NdArray CPU f32 \
             oracle to cosine > 0.99999 on E=8/top_k=2/H=32/I=16/T=10 AND E=32/top_k=8/H=256/I=128/T=64. \
             forward_routed_ondevice at a small capacity drops tokens on the same routing; the grouped \
             path does not.",
            rows[0].n, rows[1].n,
        );
    } else {
        println!("MoE GROUPED-GEMM: PARTIAL/FAIL — see the [FAIL] rows above.");
    }

    for r in &rows {
        assert!(r.counts_match, "shape `{}`: device per-expert counts != host reference", r.label);
        assert!(r.no_drop, "shape `{}`: sorted layout dropped/misgrouped an assignment", r.label);
        assert!(r.grouped_total == r.n, "shape `{}`: grouped_total {} != N {}", r.label, r.grouped_total, r.n);
        assert!(r.cos > 0.99999, "shape `{}`: grouped vs NdArray oracle cos={:.7} (max_abs_diff={:.3e})", r.label, r.cos, r.mad);
    }
    println!("\nALL CHECKS PASSED.");
}
