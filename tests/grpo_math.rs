//! GRPO math parity test: the Rust implementation must reproduce the A0 Python reference
//! (`a0/grpo_reference.py` → `tests/ref/grpo_expected.json`, OpenRLHF v0.10.4 math) within
//! f32 tolerance. This is the load-bearing correctness gate for the port.
//!
//! Run: `cargo test --test grpo_math`

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor, TensorData};
use qwen3_burn::grpo::{grpo_loss, group_norm_advantage, token_logprobs, GrpoConfig};
use serde_json::Value;

type B = NdArray;
const TOL: f32 = 2e-4; // f32 vs f64 reference

fn load() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/ref/grpo_expected.json");
    let txt = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("missing {path}: {e}. Run `python3 a0/grpo_reference.py` first."));
    serde_json::from_str(&txt).expect("valid json")
}

/// Recursively flatten a nested JSON array of numbers into a row-major Vec<f32>.
fn flat_f32(v: &Value) -> Vec<f32> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat_f32).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap() as f32],
        other => panic!("not a number: {other:?}"),
    }
}

fn flat_i64(v: &Value) -> Vec<i64> {
    match v {
        Value::Array(a) => a.iter().flat_map(flat_i64).collect(),
        Value::Number(n) => vec![n.as_i64().unwrap()],
        other => panic!("not an int: {other:?}"),
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn as_vec<const D: usize>(t: Tensor<B, D>) -> Vec<f32> {
    t.into_data().to_vec::<f32>().unwrap()
}

#[test]
fn grpo_math_matches_openrlhf_reference() {
    let dev = Default::default();
    let j = load();
    let cfg_j = &j["config"];
    let p = cfg_j["P"].as_u64().unwrap() as usize;
    let g = cfg_j["G"].as_u64().unwrap() as usize;
    let t = cfg_j["T"].as_u64().unwrap() as usize;
    let v = cfg_j["V"].as_u64().unwrap() as usize;
    let n = p * g;
    let inp = &j["inputs"];
    let exp = &j["expected"];

    let mk3 = |val: &Value| Tensor::<B, 1>::from_floats(flat_f32(val).as_slice(), &dev).reshape([n, t, v]);
    let mk2 = |val: &Value| Tensor::<B, 1>::from_floats(flat_f32(val).as_slice(), &dev).reshape([n, t]);

    let logits_pi = mk3(&inp["logits_pi"]);
    let logits_old = mk3(&inp["logits_old"]);
    let logits_ref = mk3(&inp["logits_ref"]);
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(flat_i64(&inp["target_ids"]), [n, t]),
        &dev,
    );
    let rewards = Tensor::<B, 1>::from_floats(flat_f32(&inp["rewards"]).as_slice(), &dev);
    let mask = mk2(&inp["mask"]);

    let cfg = GrpoConfig::default(); // group_norm + token_global + k3, matches the reference

    // (1) per-token log-probs (gather - logsumexp)
    let logp_pi = token_logprobs(logits_pi, targets.clone());
    let logp_old = token_logprobs(logits_old, targets.clone());
    let logp_ref = token_logprobs(logits_ref, targets);
    let d = max_abs_diff(&as_vec(logp_pi.clone()), &flat_f32(&exp["logp_pi"]));
    assert!(d < TOL, "logp_pi max|diff| = {d} (tol {TOL})");

    // (2) group-normalized advantage
    let adv = group_norm_advantage(rewards, p, g, &cfg);
    let d = max_abs_diff(&as_vec(adv.clone()), &flat_f32(&exp["advantages"]));
    assert!(d < TOL, "advantages max|diff| = {d} (tol {TOL})");

    // (3) full GRPO loss + metrics
    let (loss, m) = grpo_loss(logp_pi, logp_old, logp_ref, adv, mask, &cfg);
    let want_pol = exp["pol_loss"].as_f64().unwrap() as f32;
    let want_kl = exp["kl_loss"].as_f64().unwrap() as f32;
    let want_total = exp["total_loss"].as_f64().unwrap() as f32;
    assert!((m.pol_loss - want_pol).abs() < TOL, "pol_loss {} vs {want_pol}", m.pol_loss);
    assert!((m.kl_loss - want_kl).abs() < TOL, "kl_loss {} vs {want_kl}", m.kl_loss);
    assert!((m.total_loss - want_total).abs() < TOL, "total_loss {} vs {want_total}", m.total_loss);
    assert!(
        (loss.into_scalar() - want_total).abs() < TOL,
        "loss tensor must equal total_loss"
    );

    // sanity: KL >= 0 (k3), loss finite
    assert!(m.kl_loss >= -1e-6, "k3 KL must be >= 0, got {}", m.kl_loss);
    assert!(m.total_loss.is_finite());
    println!(
        "PARITY OK — pol={:.8} kl={:.8} total={:.8} clip_frac={:.4} mean_ratio={:.6}",
        m.pol_loss, m.kl_loss, m.total_loss, m.clip_frac, m.mean_ratio
    );
}
