//! Validates `ExpertSlotPool::expert_forward` (Phase 1/2 of `docs/MEMORY_STREAMING_PLAN.md`) against
//! the real checkpoint: computes one routed expert's MLP output via the on-demand streamed pool
//! (only ever holding that one expert's ~6.3MB of weights resident) and compares it against the same
//! math computed by fully materializing the layer's `[256, ...]` fused expert tensors and slicing —
//! i.e. today's eager `expert_forward` path — for numerical agreement.
//!
//! Usage:
//!   cargo run --release --example expert_stream_probe -- [dir] [layer] [expert]
//!   (defaults: dir="models", layer=0, expert=7)

use std::collections::BTreeMap;
use std::path::PathBuf;

use burn::prelude::Device;
use burn::tensor::activation::silu;
use burn::tensor::{DType, Tensor, TensorData};
use qwen3_burn::Precision;
use qwen3_burn::expert_stream::ExpertSlotPool;
use qwen3_burn::nvidia_ckpt::ShardReader;
use qwen3_burn::qwen3_5::parse_weight_map;

fn matmul_out_in(x: Tensor<2>, weight_out_in: Tensor<2>, prec: Precision) -> Tensor<2> {
    let weight_in_out = weight_out_in.transpose();
    let xdt = x.dtype();
    match prec {
        Precision::F32 => x.matmul(weight_in_out.cast(xdt)),
        Precision::Bf16 => x
            .cast(DType::BF16)
            .matmul(weight_in_out.cast(DType::BF16))
            .cast(DType::F32),
        Precision::F16 => x
            .cast(DType::F16)
            .matmul(weight_in_out.cast(DType::F16))
            .cast(DType::F32),
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("FAIL: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = args
        .get(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models"));
    let layer: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
    let expert: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(7);
    let device = Device::flex();

    let index_path = dir.join("model.safetensors.index.json");
    let text = std::fs::read_to_string(&index_path)
        .map_err(|e| format!("read {}: {e}", index_path.display()))?;
    let pairs = parse_weight_map(&text).map_err(|e| format!("parse weight_map: {e}"))?;
    let index: BTreeMap<String, String> = pairs.into_iter().collect();

    let tokens = 4usize;

    // Deterministic, non-trivial input activations (no RNG dependency).
    let hidden_probe = 2048usize; // sized below from the real tensor once we know it; placeholder unused.
    let _ = hidden_probe;

    println!("probing layer={layer} expert={expert} dir={dir:?}");

    // Streamed path: pool capacity 1 means only ONE expert's weights are ever resident at a time.
    let mut pool = ExpertSlotPool::new(&dir, &index, 1);
    let gate_up_key = format!("model.language_model.layers.{layer}.mlp.experts.gate_up_proj");
    let down_key = format!("model.language_model.layers.{layer}.mlp.experts.down_proj");

    // Peek shape via a single-expert slice read (cheap) to build the input tensor with correct hidden dim.
    let mut probe_reader = ShardReader::new(&dir, &index);
    let sample = probe_reader.read_expert_slice(&gate_up_key, 0)?;
    let hidden = sample.shape[2];
    let two_inner = sample.shape[1];
    let inner = two_inner / 2;
    println!("hidden={hidden} inner={inner}");

    let x_vals: Vec<f32> = (0..tokens * hidden)
        .map(|i| ((i as f32) * 0.001).sin())
        .collect();
    let x2 = Tensor::<2>::from_data(TensorData::new(x_vals, [tokens, hidden]), &device);

    let streamed = pool.expert_forward(layer, expert, x2.clone(), Precision::Bf16, &device)?;
    println!(
        "streamed: hits={} misses={} resident_slots={}",
        pool.hits,
        pool.misses,
        pool.resident_slots()
    );

    // Reference: fully materialize the layer's fused tensors and slice, exactly like today's eager
    // `Qwen3_5SparseMoeBlock::expert_forward`.
    let mut full_reader = ShardReader::new(&dir, &index);
    let gate_up_full = full_reader.read_raw_tensor(&gate_up_key)?;
    let down_full = full_reader.read_raw_tensor(&down_key)?;
    let num_experts = gate_up_full.shape[0];
    let gate_up_tensor = Tensor::<3>::from_data(
        TensorData::from_bytes_vec(gate_up_full.data, gate_up_full.shape.clone(), DType::BF16),
        (&device, DType::BF16),
    );
    let down_tensor = Tensor::<3>::from_data(
        TensorData::from_bytes_vec(down_full.data, down_full.shape.clone(), DType::BF16),
        (&device, DType::BF16),
    );
    assert!(
        expert < num_experts,
        "expert {expert} out of range {num_experts}"
    );

    let gate_w = gate_up_tensor
        .clone()
        .slice([expert..expert + 1, 0..inner, 0..hidden])
        .reshape([inner, hidden]);
    let up_w = gate_up_tensor
        .slice([expert..expert + 1, inner..two_inner, 0..hidden])
        .reshape([inner, hidden]);
    let down_w = down_tensor
        .slice([expert..expert + 1, 0..hidden, 0..inner])
        .reshape([hidden, inner]);

    let gate = silu(matmul_out_in(x2.clone(), gate_w, Precision::Bf16));
    let up = matmul_out_in(x2, up_w, Precision::Bf16);
    let reference = matmul_out_in(gate * up, down_w, Precision::Bf16);

    let diff = (streamed - reference)
        .abs()
        .max()
        .into_data()
        .to_vec::<f32>()
        .map_err(|e| format!("read max diff: {e:?}"))?;
    let max_abs_diff = diff[0];
    println!("max_abs_diff streamed vs eager-reference: {max_abs_diff}");
    if max_abs_diff > 1e-4 {
        return Err(format!(
            "mismatch: max_abs_diff={max_abs_diff} exceeds tolerance"
        ));
    }
    println!("PASS: streamed expert_forward matches eager reference within tolerance");
    Ok(())
}
