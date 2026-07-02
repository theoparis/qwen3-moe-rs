//! WAVE-2 STEP 1 validation + measurement: the FIXED-SHAPE, HOST-SYNC-FREE static MoE decode on the
//! REAL Qwen3-30B-A3B (GB10, bf16 weights). Two deliverables in one load:
//!
//!  (1) **bf16 PARITY GATE** (the gate the Block-A review flagged): the static decode
//!      (`generate_greedy_static` = Block A `decode_topk` + the reused dense static attention, every
//!      per-step op device-`pos`-indexed) must produce GREEDY-TOKEN-IDENTICAL output to the EAGER
//!      `generate_greedy` (the oracle MoE + growing-prefix cache — the numerical reference `moe_generate`
//!      uses). This is the first time `decode_topk` runs the bf16 expert weights on the real model.
//!      Both run with eos=[] (no early stop) so the comparison is total-length.
//!
//!  (2) **PERF**: steady-state per-token decode speed of the static path, measured TWO ways:
//!        * per-step (host-reads the argmax id each step — the SAME methodology as
//!          `examples/decode_perf_bench.rs`, so it is directly comparable to the §5 table:
//!          oracle 0.673 / routed 5.72 / ondevice 1.03 tok/s), and
//!        * host-sync-free amortized (one end sync) — the realistic deployment number the static path
//!          unlocks (no 48 host-syncs/layer like routed, no 128-expert re-stack like ondevice).
//!      Reported with the top-8 weight-byte model (≈6.06 GB/tok) → eff GB/s and % of the 273 GB/s peak.
//!
//! Build/run:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example moe_static_decode -- \
//!     --dir models/qwen3-30b-a3b --prompt "The capital of France is" --max-tokens 24 --decode-tokens 16

use std::path::PathBuf;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{DType, Int, Tensor};
use qwen3_burn::{MoeStaticDecode, Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3Tokenizer};

type B = Cuda;

/// GB10 / DGX-Spark LPDDR5X peak (GB/s, decimal 1e9). Verified in docs/PERF_80TOKS_PLAN.md §0.
const PEAK_GBPS: f64 = 273.0;

fn arg<'a>(a: &'a [String], f: &str) -> Option<&'a String> {
    a.iter().position(|x| x == f).and_then(|i| a.get(i + 1))
}

/// Build a `Qwen3MoeConfig` from a HuggingFace `config.json` (same as examples/decode_perf_bench.rs).
fn config_from_hf(dir: &PathBuf) -> Result<Qwen3MoeConfig, String> {
    let txt = std::fs::read_to_string(dir.join("config.json")).map_err(|e| format!("read config.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_str(&txt).map_err(|e| format!("parse config.json: {e}"))?;
    let u = |k: &str, d: u64| -> usize { v.get(k).and_then(|x| x.as_u64()).unwrap_or(d) as usize };
    let f = |k: &str, d: f64| -> f64 { v.get(k).and_then(|x| x.as_f64()).unwrap_or(d) };
    let mut cfg = Qwen3MoeConfig::new()
        .with_vocab_size(u("vocab_size", 151936))
        .with_hidden_size(u("hidden_size", 2048))
        .with_num_hidden_layers(u("num_hidden_layers", 48))
        .with_num_attention_heads(u("num_attention_heads", 32))
        .with_num_key_value_heads(u("num_key_value_heads", 4))
        .with_num_experts(u("num_experts", 128))
        .with_num_experts_per_tok(u("num_experts_per_tok", 8))
        .with_moe_intermediate_size(u("moe_intermediate_size", 768))
        .with_rms_norm_eps(f("rms_norm_eps", 1e-6))
        .with_rope_theta(f("rope_theta", 1_000_000.0))
        .with_max_position_embeddings(u("max_position_embeddings", 40960));
    if let Some(hd) = v.get("head_dim").and_then(|x| x.as_u64()) {
        cfg = cfg.with_head_dim(Some(hd as usize));
    }
    if let Some(n) = v.get("norm_topk_prob").and_then(|x| x.as_bool()) {
        cfg = cfg.with_norm_topk_prob(n);
    }
    Ok(cfg)
}

/// Top-8 per-token weight bytes the static path reads (experts read only k of E), broken out.
fn path_bytes_topk(cfg: &Qwen3MoeConfig, dtype_bytes: usize) -> (f64, f64, f64) {
    let l = cfg.num_hidden_layers;
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    let k = cfg.num_experts_per_tok;
    let hd = cfg.get_head_dim();
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let db = dtype_bytes as f64;
    let experts = (l * k * 3 * h * i) as f64 * db; // gate+up+down per ROUTED expert (top-k)
    let attn = (l * 2 * h * hd * (nq + nkv)) as f64 * db; // q,o + k,v projections
    let head = (cfg.vocab_size * h) as f64 * db; // untied lm_head
    (experts, attn, head)
}

/// Read the single argmax id to host — forces a full device sync (the queue must drain to produce data).
fn read_id(t: Tensor<B, 1, Int>) -> i64 {
    t.cast(DType::I64).into_data().as_slice::<i64>().map(|s| s.first().copied().unwrap_or(0)).unwrap_or(0)
}

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}
fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { f64::NAN } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}

/// Per-step steady-state decode bench for one static-decode path (`sd` selects oracle vs fused). Each
/// step is wall-clocked around the argmax host read (same methodology as `decode_perf_bench.rs`).
/// Returns `(per_step_ms, gen_ids)`. Prefill is run (excluded from the returned step times).
fn bench_per_step(
    model: &Qwen3MoeForCausalLM<B>,
    input: &Tensor<B, 2, Int>,
    sd: &MoeStaticDecode<B>,
    lp: usize,
    decode_tokens: usize,
    device: &CudaDevice,
) -> (Vec<f64>, Vec<i64>) {
    // The static attention masks over the FULL `sd.capacity()` (= `arange_tmax` width), so the KV
    // cache MUST be that wide regardless of how many steps this call runs (e.g. warmup runs fewer).
    let total = sd.capacity();
    assert!(lp + decode_tokens <= total, "lp+decode_tokens {} > sd.capacity {total}", lp + decode_tokens);
    let mut cache = model.model.new_cache_with_capacity(total);
    let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, device).unsqueeze_dim::<2>(0);
    let logits = model.forward_with_cache(input.clone(), None, pos0, &mut cache);
    let vocab = logits.dims()[2];
    let mut next: Tensor<B, 2, Int> =
        logits.slice([0..1, (lp - 1)..lp, 0..vocab]).reshape([1, vocab]).argmax(1).reshape([1, 1]);
    let _ = read_id(next.clone().reshape([1])); // sync — completes prefill

    let mut steps_ms = Vec::with_capacity(decode_tokens);
    let mut gen_ids: Vec<i64> = Vec::new();
    let mut pos = Tensor::<B, 1, Int>::full([1], lp as i64, device);
    for _ in 0..decode_tokens {
        let t = std::time::Instant::now();
        let lg = model.forward_with_cache_static_pre(next.clone(), pos.clone(), &mut cache, sd); // [1,1,v]
        next = lg.slice([0..1, 0..1, 0..vocab]).reshape([1, vocab]).argmax(1).reshape([1, 1]);
        let id = read_id(next.clone().reshape([1])); // forces device sync → true per-token latency
        steps_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        gen_ids.push(id);
        pos = pos.add_scalar(1i64);
    }
    (steps_ms, gen_ids)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(arg(&args, "--dir").cloned().unwrap_or_else(|| "models/qwen3-30b-a3b".into()));
    let prompt = arg(&args, "--prompt").cloned().unwrap_or_else(|| "The capital of France is".into());
    let max_tokens: usize = arg(&args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(24); // parity gate
    let decode_tokens: usize = arg(&args, "--decode-tokens").and_then(|s| s.parse().ok()).unwrap_or(16); // bench
    let warmup: usize = arg(&args, "--warmup").and_then(|s| s.parse().ok()).unwrap_or(2);
    let dtype_bytes: usize = arg(&args, "--dtype-bytes").and_then(|s| s.parse().ok()).unwrap_or(2); // bf16

    let device = CudaDevice::default();
    println!("device: {device:?}");
    let cfg = config_from_hf(&dir)?;
    println!(
        "config: {} layers, hidden {}, {} experts top-{}, moe_inter {}, head_dim {}, vocab {}",
        cfg.num_hidden_layers, cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok,
        cfg.moe_intermediate_size, cfg.get_head_dim(), cfg.vocab_size
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm::<B>(&device);
    println!("loading sharded weights from {dir:?} ...");
    let t0 = std::time::Instant::now();
    model.load_weights_sharded(&dir).map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!("loaded {} layers in {:.1}s (load EXCLUDED from timing)", model.num_layers(), t0.elapsed().as_secs_f64());

    let (ids_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let lp = prompt_ids.len();
    let input: Tensor<B, 2, Int> = Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), &device).unsqueeze();
    println!("\nprompt ({lp} tok): {prompt:?}");

    // ========================================================================================
    // (1) bf16 PARITY GATE — static greedy must equal eager greedy tokens (eos=[] ⇒ full length).
    // ========================================================================================
    println!("\n===== (1) bf16 PARITY GATE: static decode vs eager generate_greedy ({max_tokens} tok, eos=[]) =====");
    let t_e = std::time::Instant::now();
    let eager = model.generate_greedy(input.clone(), max_tokens, &[]); // oracle MoE + growing-prefix cache
    let eager_ids: Vec<i64> = eager.cast(DType::I64).to_data().to_vec().map_err(|e| format!("read eager: {e:?}"))?;
    let eager_s = t_e.elapsed().as_secs_f64();

    let sd_parity = model.build_static_decode(lp + max_tokens); // built ONCE post-load
    let t_s = std::time::Instant::now();
    let stat = model.generate_greedy_static(input.clone(), max_tokens, &[], &sd_parity);
    let stat_ids: Vec<i64> = stat.cast(DType::I64).to_data().to_vec().map_err(|e| format!("read static: {e:?}"))?;
    let stat_s = t_s.elapsed().as_secs_f64();

    let identical = eager_ids == stat_ids;
    let first_div = eager_ids.iter().zip(stat_ids.iter()).position(|(a, b)| a != b);
    let eager_txt = tokenizer.decode(&eager_ids.iter().map(|&x| x as u32).collect::<Vec<_>>()).unwrap_or_default();
    let stat_txt = tokenizer.decode(&stat_ids.iter().map(|&x| x as u32).collect::<Vec<_>>()).unwrap_or_default();
    println!("  eager  ({eager_s:.1}s): {eager_txt:?}");
    println!("  static ({stat_s:.1}s): {stat_txt:?}");
    println!("  eager  ids: {:?}", &eager_ids[lp..]);
    println!("  static ids: {:?}", &stat_ids[lp..]);
    if identical {
        println!("  ==> GATE PASS: static decode is GREEDY-TOKEN-IDENTICAL to eager (bf16 parity holds).");
    } else {
        println!("  ==> GATE FAIL: first divergence at absolute pos {first_div:?} (completion idx {:?}).",
            first_div.map(|i| i.saturating_sub(lp)));
    }

    // ========================================================================================
    // (2) PERF — steady-state per-step decode (§5 methodology) for BOTH MoE-decode kernels in ONE load:
    //     `decode_topk_pre` (materializing oracle, the ~6.45 baseline) vs `decode_topk_fused` (lever (c),
    //     the fused gather-GEMV that reads each routed expert's weights ONCE from the stacks). Same
    //     routing/combine/attention, so this isolates the no-materialization (bandwidth) win.
    // ========================================================================================
    println!("\n===== (2) PERF: static MoE decode — oracle (materializing) vs FUSED gather-GEMV (lever c) =====");
    let total = lp + decode_tokens;
    let (be, ba, bh) = path_bytes_topk(&cfg, dtype_bytes);
    let gb_tok = (be + ba + bh) / 1e9;

    let sd_oracle = model.build_static_decode(total); // default = decode_topk_pre
    let sd_fused = model.build_static_decode(total).with_fused(true); // lever (c)

    // Warm both kernels (CubeCL JIT/autotune of the [1,1] shapes) before timing.
    let _ = bench_per_step(&model, &input, &sd_oracle, lp, warmup.max(1), &device);
    let _ = bench_per_step(&model, &input, &sd_fused, lp, warmup.max(1), &device);

    let mut rows: Vec<(&str, f64, f64, Vec<i64>)> = Vec::new();
    for (label, sd) in [("oracle (decode_topk_pre, materializing)", &sd_oracle), ("FUSED  (decode_topk_fused, lever c)", &sd_fused)] {
        let (steps_ms, gen_ids) = bench_per_step(&model, &input, sd, lp, decode_tokens, &device);
        let steady = &steps_ms[warmup.min(steps_ms.len())..];
        let ms = median(steady);
        let tok_s = 1000.0 / ms;
        let eff = gb_tok * tok_s;
        let pct = eff / PEAK_GBPS * 100.0;
        println!("  {label}:");
        println!("    steps(ms): {}", steps_ms.iter().map(|m| format!("{m:.0}")).collect::<Vec<_>>().join(","));
        println!("    median {ms:.1} ms/tok | mean {:.1} | min {:.1}", mean(steady), steady.iter().cloned().fold(f64::INFINITY, f64::min));
        println!("    => {tok_s:.3} tok/s | {gb_tok:.2} GB/tok (experts {:.2}+attn {:.2}+head {:.2}) | {eff:.1} GB/s eff | {pct:.0}% peak", be/1e9, ba/1e9, bh/1e9);
        rows.push((label, ms, tok_s, gen_ids));
    }

    // ---- token parity on the REAL bf16 30B: fused must generate the SAME ids as the oracle decode. ----
    let oracle_ids = &rows[0].3;
    let fused_ids = &rows[1].3;
    let tok_identical = oracle_ids == fused_ids;
    let first_div = oracle_ids.iter().zip(fused_ids.iter()).position(|(a, b)| a != b);
    println!("\n  ----- FUSED vs ORACLE token parity (real 30B bf16) -----");
    println!("    oracle ids: {oracle_ids:?}");
    println!("    fused  ids: {fused_ids:?}");
    if tok_identical {
        println!("    ==> TOKEN-IDENTICAL: fused gather-GEMV == materializing oracle on the real bf16 model.");
    } else {
        println!("    ==> MISMATCH at decode idx {first_div:?} (fused diverged from the oracle decode).");
    }

    // ---- speedup + roofline summary. ----
    let (o_ms, o_toks) = (rows[0].1, rows[0].2);
    let (f_ms, f_toks) = (rows[1].1, rows[1].2);
    let speedup = o_ms / f_ms;
    let o_pct = gb_tok * o_toks / PEAK_GBPS * 100.0;
    let f_pct = gb_tok * f_toks / PEAK_GBPS * 100.0;
    println!("\n  ----- LEVER (c) RESULT (per-step, §5-comparable) -----");
    println!("    oracle (materializing select) : {o_toks:6.3} tok/s   {:6.1} GB/s   {o_pct:3.0}% peak", gb_tok * o_toks);
    println!("    FUSED  (gather-GEMV, lever c) : {f_toks:6.3} tok/s   {:6.1} GB/s   {f_pct:3.0}% peak", gb_tok * f_toks);
    println!("    => fused speedup over materializing oracle: {speedup:.2}x  ({} the no-materialization win)",
        if speedup > 1.03 { "REALIZED" } else { "did NOT realize" });
    println!("\n  ----- vs PERF_80TOKS_PLAN §5 (real 30B, GB10) -----");
    println!("    oracle  (dense 128 experts)      : 0.673 tok/s   40.7 GB/s   15% peak");
    println!("    routed  (host top-8, 48 syncs/L) : 5.720 tok/s   34.7 GB/s   13% peak");
    Ok(())
}
