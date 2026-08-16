//! Decode-performance bench for the REAL Qwen3-30B-A3B on the GB10 — the MEASUREMENT GATE for
//! docs/PERF_80TOKS_PLAN.md. It answers ONE question with numbers: at the measured single-stream
//! decode speed, is the model **launch-bound** (overhead dominates, far below the memory roofline) or
//! **bandwidth-bound** (saturating LPDDR5X)? Method (textbook roofline back-calc):
//!
//!   effective DRAM bandwidth  =  (weight bytes a path reads per token) × (tok/s)
//!   % of peak                 =  effective / 273 GB/s            (GB10 LPDDR5X, verified)
//!
//! If a path runs at ≪ peak it is NOT memory-saturated → the time is spent on kernel launches /
//! host-syncs, i.e. LAUNCH-BOUND. If it runs near peak (~60-93% — batch-1 MoE sustains ~60-75% of the
//! byte roofline; llama.cpp hits 80-93%) it is BANDWIDTH-BOUND and only fewer bytes (fp8/top-k) help.
//!
//! Per-token weight-byte model (derived from config.json, bf16 = 2 B/weight; matches the plan's table):
//!   experts  = L · (E or K) · 3·H·I · 2      (oracle/ondevice read all E=128; routed reads K=8)
//!   attn     = L · 2·H·hd·(n_q+n_kv) · 2      (q/k/v/o projections)
//!   lm_head  = vocab · H · 2                   (untied; can't be chunked for exact greedy argmax)
//!   → oracle/ondevice ≈ 60.4 GB/token (full-dense),  routed ≈ 6.05 GB/token (top-8).
//! KV-cache reads are excluded (negligible at short context; ~4 GB/token only at the 40,960 ctx max).
//! Router gate (E·H/layer) and embedding row are negligible and omitted, matching the plan.
//!
//! It times STEADY-STATE per-token decode only: model load and the prefill step are excluded, and the
//! first `--warmup` decode steps (CubeCL JIT/autotune of the [1,1] shapes) are discarded. Each step is
//! wall-clocked around a forced device→host sync (reading the argmax id, exactly as `generate_greedy`
//! does), so the timer captures the true per-token latency including the host sync.
//!
//! MoE path selection mirrors `Qwen3MoeSparseBlock::forward` (src/moe.rs): QWEN3_MOE_ONDEVICE wins,
//! then QWEN3_MOE_ROUTED, else the dense oracle. Pass `--all-paths` to bench oracle/routed/ondevice in
//! ONE process (one 60 GB load) by setting those env vars internally per run.
//!
//! Build/run:
//!   cargo build --release --features cuda --example decode_perf_bench
//!   RUSTFLAGS="-C target-feature=+fp16" QWEN3_MOE_ONDEVICE=1 \
//!     ./target/release/examples/decode_perf_bench --dir models/qwen3-30b-a3b --decode-tokens 8
//!   # or all three paths in one load:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     ./target/release/examples/decode_perf_bench --dir models/qwen3-30b-a3b --all-paths --decode-tokens 8

use std::path::PathBuf;

use burn::prelude::Device;
use burn::tensor::{DType, Device, Int, Tensor};
use qwen3_burn::{Qwen3MoeConfig, Qwen3Tokenizer};

type B = Cuda;

/// GB10 / DGX-Spark LPDDR5X peak (GB/s, decimal 1e9). Verified in docs/PERF_80TOKS_PLAN.md §0.
const PEAK_GBPS: f64 = 273.0;
/// Below this fraction of peak the path is NOT memory-saturated → overhead (launch/host-sync) bound.
/// Batch-1 MoE that IS bandwidth-bound sustains ~60-75% of the byte roofline (plan §0; llama.cpp
/// 80-93%), so 60% is a conservative LAUNCH-vs-BANDWIDTH cutoff. The actual % is always printed.
const BANDWIDTH_BOUND_PCT: f64 = 60.0;

#[derive(Clone, Copy, PartialEq)]
enum Path {
    Oracle,
    Routed,
    Ondevice,
}

impl Path {
    fn label(self) -> &'static str {
        match self {
            Path::Oracle => "oracle (dense, all 128 experts)",
            Path::Routed => "routed (host top-8, 48 syncs/layer)",
            Path::Ondevice => "ondevice (on-device route; re-stacks 128 @ T=1)",
        }
    }
    /// Experts read PER LAYER per token. oracle + ondevice both touch all E (ondevice re-stacks the
    /// full 128 and at decode T=1 capacity C=1 makes it dense); routed reads only K.
    fn experts_read(self, e: usize, k: usize) -> usize {
        match self {
            Path::Routed => k,
            _ => e,
        }
    }
    /// Set the env toggles read inside `Qwen3MoeSparseBlock::forward` so this path is exercised.
    fn apply_env(self) {
        // SAFETY: single-threaded example; we mutate process env before each (synchronous) bench run.
        unsafe {
            std::env::remove_var("QWEN3_MOE_ONDEVICE");
            std::env::remove_var("QWEN3_MOE_ROUTED");
            match self {
                Path::Oracle => {}
                Path::Routed => std::env::set_var("QWEN3_MOE_ROUTED", "1"),
                Path::Ondevice => std::env::set_var("QWEN3_MOE_ONDEVICE", "1"),
            }
        }
    }
}

fn arg<'a>(a: &'a [String], f: &str) -> Option<&'a String> {
    a.iter().position(|x| x == f).and_then(|i| a.get(i + 1))
}

/// Build a `Qwen3MoeConfig` from a HuggingFace `config.json` (same as examples/moe_generate.rs).
fn config_from_hf(dir: &PathBuf) -> Result<Qwen3MoeConfig, String> {
    let txt = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| format!("read config.json: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(&txt).map_err(|e| format!("parse config.json: {e}"))?;
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

/// Per-token weight bytes a path reads, broken out (experts / attn / head), in bytes.
fn path_bytes(cfg: &Qwen3MoeConfig, dtype_bytes: usize, path: Path) -> (f64, f64, f64) {
    let l = cfg.num_hidden_layers;
    let h = cfg.hidden_size;
    let i = cfg.moe_intermediate_size;
    let e = cfg.num_experts;
    let k = cfg.num_experts_per_tok;
    let hd = cfg.get_head_dim();
    let nq = cfg.num_attention_heads;
    let nkv = cfg.num_key_value_heads;
    let db = dtype_bytes as f64;
    let experts = (l * path.experts_read(e, k) * 3 * h * i) as f64 * db; // gate+up+down per expert
    let attn = (l * 2 * h * hd * (nq + nkv)) as f64 * db; // q,o (n_q·hd) + k,v (n_kv·hd)
    let head = (cfg.vocab_size * h) as f64 * db; // untied lm_head
    (experts, attn, head)
}

/// Read the single argmax id to host — forces a full device sync (queue must drain to produce data).
fn read_id(t: Tensor<1, Int>) -> i64 {
    t.cast(DType::I64)
        .into_data()
        .as_slice::<i64>()
        .map(|s| s.first().copied().unwrap_or(0))
        .unwrap_or(0)
}

struct Stats {
    prefill_ms: f64,
    steps_ms: Vec<f64>,
    gen_ids: Vec<i64>,
}

/// One steady-state decode bench for `path`: prefill (excluded) then `decode_tokens` timed steps.
fn bench_path(
    model: &qwen3_burn::Qwen3MoeForCausalLM,
    prompt_ids: &[i64],
    device: &CudaDevice,
    decode_tokens: usize,
    path: Path,
    eos: &[i64],
) -> Stats {
    path.apply_env();
    let init_len = prompt_ids.len();
    let input: Tensor<2, Int> = Tensor::<1, Int>::from_data(prompt_ids, device).unsqueeze();
    let mut cache = model.model.new_cache();

    // --- Prefill (EXCLUDED from steady-state): full-prompt forward → first next token. ---
    let t_pf = std::time::Instant::now();
    let pos = Tensor::<1, Int>::arange(0..init_len as i64, device).unsqueeze_dim::<2>(0);
    let logits = model.forward_with_cache(input, None, pos, &mut cache);
    let vocab = logits.dims()[2];
    let mut next: Tensor<1, Int> = logits
        .slice([0..1, (init_len - 1)..init_len, 0..vocab])
        .reshape([1, vocab])
        .argmax(1)
        .flatten(0, 1);
    let first_id = read_id(next.clone()); // sync — completes prefill
    let prefill_ms = t_pf.elapsed().as_secs_f64() * 1000.0;

    // --- Steady-state decode: one token at a time, each wall-clocked around a host sync. ---
    let mut steps_ms = Vec::with_capacity(decode_tokens);
    let mut gen_ids = vec![first_id];
    let mut cur = init_len;
    for _ in 0..decode_tokens {
        cur += 1;
        let t = std::time::Instant::now();
        let pos = Tensor::<1, Int>::from_data([cur as i64 - 1], device).unsqueeze_dim::<2>(0);
        let logits = model.forward_with_cache(next.clone().unsqueeze_dim(1), None, pos, &mut cache);
        next = logits
            .slice([0..1, 0..1, 0..vocab])
            .reshape([1, vocab])
            .argmax(1)
            .flatten(0, 1);
        let id = read_id(next.clone()); // forces device sync → timer captures true per-token latency
        steps_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        gen_ids.push(id);
        // NOTE: we do NOT break on EOS — a fixed step count keeps the steady-state sample stable.
        let _ = eos;
    }
    Stats {
        prefill_ms,
        steps_ms,
        gen_ids,
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}
fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        arg(&args, "--dir")
            .cloned()
            .unwrap_or_else(|| "models/qwen3-30b-a3b".into()),
    );
    let prompt = arg(&args, "--prompt")
        .cloned()
        .unwrap_or_else(|| "The capital of France is".into());
    let decode_tokens: usize = arg(&args, "--decode-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let warmup: usize = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let dtype_bytes: usize = arg(&args, "--dtype-bytes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2); // bf16
    let all_paths = args.iter().any(|x| x == "--all-paths");
    if decode_tokens <= warmup {
        return Err(format!(
            "--decode-tokens ({decode_tokens}) must exceed --warmup ({warmup})"
        ));
    }

    let device = Device::cuda(0);
    println!("device: {device:?}");
    let cfg = config_from_hf(&dir)?;
    println!(
        "config: {} layers, hidden {}, {} experts top-{}, moe_inter {}, head_dim {}, vocab {}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.num_experts_per_tok,
        cfg.moe_intermediate_size,
        cfg.get_head_dim(),
        cfg.vocab_size
    );
    println!(
        "byte model: bf16 weights ({dtype_bytes} B/weight); peak = {PEAK_GBPS} GB/s; KV-read excluded (short ctx)"
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm(&device);
    println!("loading sharded weights from {dir:?} ...");
    let t0 = std::time::Instant::now();
    model
        .load_weights_sharded(&dir)
        .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!(
        "loaded {} layers in {:.1}s (load EXCLUDED from timing)",
        model.num_layers(),
        t0.elapsed().as_secs_f64()
    );

    let (ids_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let prompt_ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let eos = [151643i64, 151645];
    println!("\nprompt ({} tok): {prompt:?}", prompt_ids.len());
    println!(
        "decode-tokens: {decode_tokens} (steady-state excludes prefill + first {warmup} step(s))\n"
    );

    // Which paths to run: --all-paths loops all three (one load); else the env-selected single path.
    let paths: Vec<Path> = if all_paths {
        vec![Path::Oracle, Path::Routed, Path::Ondevice]
    } else if std::env::var("QWEN3_MOE_ONDEVICE").is_ok() {
        vec![Path::Ondevice]
    } else if std::env::var("QWEN3_MOE_ROUTED").is_ok() {
        vec![Path::Routed]
    } else {
        vec![Path::Oracle]
    };

    println!("{:-<108}", "");
    println!(
        "{:<46} {:>9} {:>10} {:>9} {:>11} {:>9}  {}",
        "path", "ms/tok", "tok/s", "GB/tok", "eff GB/s", "% peak", "verdict"
    );
    println!("{:-<108}", "");

    let mut summary: Vec<(Path, f64, f64, f64, f64, bool)> = Vec::new();
    for path in paths {
        let stats = bench_path(&model, &prompt_ids, &device, decode_tokens, path, &eos);
        let steady = &stats.steps_ms[warmup.min(stats.steps_ms.len())..];
        let ms = median(steady); // median = robust steady-state per-token latency
        let tok_s = 1000.0 / ms;
        let (be, ba, bh) = path_bytes(&cfg, dtype_bytes, path);
        let gb_tok = (be + ba + bh) / 1e9;
        let eff_gbps = gb_tok * tok_s;
        let pct = eff_gbps / PEAK_GBPS * 100.0;
        let launch_bound = pct < BANDWIDTH_BOUND_PCT;
        let verdict = if launch_bound {
            format!("LAUNCH-BOUND ({pct:.0}% of peak)")
        } else {
            format!("BANDWIDTH-BOUND ({pct:.0}% of peak)")
        };

        // Per-path detail (per-step times expose the warmup spike + steady-state stability).
        println!(
            "{:<46} {:>9.1} {:>10.3} {:>9.2} {:>11.1} {:>8.0}%  {}",
            path.label(),
            ms,
            tok_s,
            gb_tok,
            eff_gbps,
            pct,
            verdict
        );
        println!(
            "    prefill {:.0} ms | steps(ms): {} | mean {:.1} median {:.1} min {:.1}",
            stats.prefill_ms,
            stats
                .steps_ms
                .iter()
                .map(|m| format!("{m:.0}"))
                .collect::<Vec<_>>()
                .join(","),
            mean(steady),
            ms,
            steady.iter().cloned().fold(f64::INFINITY, f64::min)
        );
        println!(
            "    bytes/tok: experts {:.2} + attn {:.2} + head {:.2} GB = {:.2} GB",
            be / 1e9,
            ba / 1e9,
            bh / 1e9,
            gb_tok
        );
        if let Ok(txt) =
            tokenizer.decode(&stats.gen_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())
        {
            println!("    sample: {txt:?}");
        }
        summary.push((path, ms, tok_s, eff_gbps, pct, launch_bound));
    }
    println!("{:-<108}", "");

    // --- Verdict on the plan's central claim (the ORACLE / full-dense path = the measured 0.73). ---
    println!(
        "\n===== VERDICT vs PERF_80TOKS_PLAN §1 (\"0.73 tok/s is LAUNCH-bound, ~16% of peak\") ====="
    );
    for (path, ms, tok_s, eff, pct, lb) in &summary {
        let line = if *lb {
            format!("LAUNCH-BOUND ({pct:.0}% of peak)")
        } else {
            format!("BANDWIDTH-BOUND ({pct:.0}% of peak)")
        };
        println!(
            "  {:<46} {:>7.3} tok/s  {:>6.1} ms/tok  {:>6.1} GB/s eff  →  {line}",
            path.label(),
            tok_s,
            ms,
            eff
        );
    }
    if let Some((_, _, _, _, pct, lb)) = summary.iter().find(|(p, ..)| *p == Path::Oracle) {
        println!(
            "\n  ORACLE (dense) verdict: {}  →  the plan's '~16% of peak, launch-bound' is {}.",
            if *lb {
                format!("LAUNCH-BOUND ({pct:.0}% of peak)")
            } else {
                format!("BANDWIDTH-BOUND ({pct:.0}% of peak)")
            },
            if *lb {
                "CONFIRMED (well below the memory roofline ⇒ overhead-bound)"
            } else {
                "REFUTED (near the memory roofline)"
            }
        );
    }
    Ok(())
}
