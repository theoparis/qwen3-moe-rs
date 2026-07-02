//! WAVE-2 STEP 2: CUDA-graph CAPTURE the Qwen3-30B-A3B MoE static decode step + replay per token.
//!
//! This mirrors `examples/cudagraph_pfinal_bench.rs` (the dense GRPO capture template) for the MoE: it
//! captures ONE `Qwen3MoeForCausalLM::forward_with_cache_static_pre` step (48 layers of reused static
//! attention + Block-A `MoeExpertCache::decode_topk`) once, then REPLAYS it per token for a greedy
//! generation, with the device-`pos` counter + static KV append + device argmax — NO per-step host sync
//! inside the captured/replayed region.
//!
//! ## Why this can win (unlike the dense case)
//! P-final measured the DENSE GRPO decode at ~1.0× under capture: it is BANDWIDTH-bound (the tied-head
//! logits GEMM streams ~0.6 GB/step), so removing launch latency buys almost nothing. The Wave-1 bench
//! found the 30B MoE decode is LAUNCH-bound (13-23% of peak), because at T=1 each of the 48 layers issues
//! many tiny kernels (route iterated-argmax, 3 expert-slab gathers, a batched M=1 GEMV, a scatter-add)
//! that are individually too small to saturate HBM — the launch tax dominates. CUDA-graph replay collapses
//! all those launches into ONE host call, so capture should push past the eager 6.45 tok/s toward the
//! bandwidth regime. This bench MEASURES whether that holds on the real 30B.
//!
//! ## The capture prerequisites (Step 1, already landed)
//! `forward_with_cache_static_pre` is fixed-shape (`[B,1,*]` every step), host-sync-free (no
//! `into_data`/`to_vec`/`into_scalar` anywhere — argmax/route/EOS are all on-device) and writes the KV at
//! a `[1]` Int DEVICE `pos` (stable-base-pointer `select_assign`). Those are exactly C1's capture
//! requirements. The framework pieces (C1 capture/replay, C2 capture arena + METADATA INTERNING, the
//! device-`pos` static decode) are the same ones P-final assembled; this bench reuses them verbatim.
//!
//! ## Backend: RAW CubeBackend (below Fusion) — NOT `Cuda`
//! Step 1's `moe_static_decode.rs` runs on `Cuda = Fusion<CubeBackend>`. Capture MUST run BELOW Fusion
//! (docs/cudagraph/DESIGN.md §0b P0-B) so the captured launch list is determined by CODE, not by the lazy
//! Fusion queue. So this bench builds the model on `CubeBackend<CudaRuntime, f32, i32, u8>` and loads the
//! bf16 weights there. The eager reference (`generate_greedy_static`) runs on the SAME raw backend, so
//! captured-vs-eager is apples-to-apples (both below Fusion); the Fusion 6.45 tok/s is reported as context.
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_moe_decode_bench -- \
//!       --dir models/qwen3-30b-a3b --prompt "The capital of France is" --max-tokens 16 --warmup 8 2>&1 | tail -60

use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::Backend;
use burn::tensor::{DType, Int, Tensor};
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use qwen3_burn::capture::{
    CaptureBackend, CapturedDecoder, DecodeState, scatter_emit_to_tok, write_last_in_place,
};
use qwen3_burn::{MoeStaticDecode, Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3Tokenizer};

type B = CaptureBackend;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

/// GB10 / DGX-Spark LPDDR5X peak (GB/s, decimal 1e9). Verified in docs/PERF_80TOKS_PLAN.md §0.
const PEAK_GBPS: f64 = 273.0;

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

fn arg<'a>(a: &'a [String], f: &str) -> Option<&'a String> {
    a.iter().position(|x| x == f).and_then(|i| a.get(i + 1))
}

/// Build a `Qwen3MoeConfig` from a HuggingFace `config.json` (same as moe_static_decode.rs).
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

/// Top-8 per-token weight bytes the static path reads (the decode_perf_bench / moe_static_decode model).
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

/// Result of one capture+replay run.
struct Captured {
    ids: Vec<i64>,
    arena_bytes: u64,
    replay_ms_per_tok: f64,
}

/// CAPTURED greedy decode of the MoE static step: capture ONE `forward_with_cache_static_pre` step,
/// reset to clean post-prefill state, replay `max_new` times. Mirrors P-final's `captured_greedy_decode`,
/// batch=1, greedy (no RNG => no C3 needed). Returns the token ids + arena high-water + replay ms/token.
fn captured_greedy_decode(
    model: &Qwen3MoeForCausalLM<B>,
    input_ids: Tensor<B, 2, Int>,
    sd: &MoeStaticDecode<B>,
    max_new: usize,
    eos: &[i64],
    vocab: usize,
    warmup: usize,
    timing_reps: usize,
) -> Captured {
    let device = input_ids.device();
    let [b, lp] = input_ids.dims();
    let total = lp + max_new;
    assert_eq!(
        sd.capacity(),
        total,
        "sd capacity {} != lp+max_new {total}",
        sd.capacity()
    );
    let cache = model.model.new_cache_with_capacity(total);

    // ---- prefill (eager, variable-shape — NOT captured): KV cols 0..lp + first logits -> last_buf ----
    let prefill = |state: &mut DecodeState<B>| {
        let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[b, 1]);
        let logits =
            model.forward_with_cache(state.input_ids.clone(), None, pos0, &mut state.cache); // [b, lp, v]
        // ROOT-CAUSE FIX (replay garbage): the lm_head runs in bf16 on the real model, so `logits` is
        // BF16, but `last_buf` is the persistent F32 logits buffer. `slice_assign` of a BF16 source into
        // an F32 destination does NOT value-cast — it corrupts the stored logits, so the very first
        // greedy `argmax` picks a wrong token and the whole decode cascades into garbage. (The eager
        // `generate_greedy_static` oracle never hit this: it argmaxes the bf16 logits directly, with no
        // persistent f32 buffer.) Cast logits to the buffer's dtype BEFORE the in-place write. The cast
        // is value-preserving widening, so the argmax — hence every token — is bit-identical to eager.
        let prefill_last = logits
            .slice([0..b, (lp - 1)..lp, 0..vocab])
            .reshape([b, vocab])
            .cast(DType::F32);
        let lb = state.last.take().unwrap();
        state.last = Some(lb.slice_assign([0..b, 0..vocab], prefill_last));
    };

    // ---- the captured ONE-STEP closure: structurally identical to generate_greedy_static's loop body,
    //      but with in-place writeback into the persistent buffers (take()+single-op, NEVER clone). ----
    let step = |state: &mut DecodeState<B>| {
        let last = state.last.take().unwrap(); // storage L (unique)
        let sampled = last.clone().argmax(1); // [b,1] Int (greedy argmax, device)

        // EOS / finished (Int 0/1): pre-step state drives the emit, then update.
        let fin = state.finished.take().unwrap(); // [b,1] Int (unique)
        let emit = sampled.mask_where(fin.clone().equal_elem(1i64), state.pad.clone()); // pad finished rows
        let mut is_eos = Tensor::<B, 2, Int>::zeros([b, 1], &device).equal_elem(1i64); // all false
        for &e in eos {
            is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
        }

        // device-`pos` scatter into the fixed token buffer (Add over zero == assign; one write per col).
        let pos_idx = state.pos.as_ref().unwrap().clone();
        state.tok = Some(scatter_emit_to_tok(
            state.tok.take().unwrap(),
            pos_idx,
            emit.clone(),
        ));
        state.finished = Some(fin.add(is_eos.int()).clamp(0i64, 1i64)); // Int OR, clamped to {0,1}

        // decode the NEXT logits from `emit` at device `pos` through the STATIC MoE step (48 layers).
        let lg = model.forward_with_cache_static_pre(
            emit,
            state.pos.as_ref().unwrap().clone(),
            &mut state.cache,
            sd,
        );
        // Cast bf16 logits to the F32 `last_buf` dtype BEFORE the in-place slice_assign (see the prefill
        // comment): a bf16->f32 slice_assign corrupts the stored logits and garbages the next argmax.
        state.last = Some(write_last_in_place(last, lg, b, vocab)); // `last` unique -> in place at L

        // advance the device counter IN-GRAPH (a captured device add of constant 1).
        state.pos = Some(state.pos.take().unwrap().add_scalar(1i64));
    };

    // ---- CAPTURE one step through the reusable harness. It owns persistent buffers, asserts VA
    //      stability across capture, resets cache/buffers in place, and performs one D2H after replay. ----
    let mut decoder = CapturedDecoder::build(
        input_ids,
        cache,
        max_new,
        vocab,
        eos.to_vec(),
        warmup,
        prefill,
        step,
    );
    let arena_bytes = decoder.arena_bytes();

    // ---- correctness replay: replay max_new times, read tok_buf ONCE (the only D2H) ----
    let ids = decoder.decode_n(max_new, eos);

    // ---- timing: reset (untimed) + time max_new pure replays, median over reps ----
    let replay_ms_per_tok = decoder.replay_ms_per_token(max_new, timing_reps);
    Captured {
        ids,
        arena_bytes,
        replay_ms_per_tok,
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
    let max_new: usize = arg(&args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let warmup: usize = arg(&args, "--warmup")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let timing_reps: usize = arg(&args, "--reps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);
    let dtype_bytes: usize = arg(&args, "--dtype-bytes")
        .and_then(|s| s.parse().ok())
        .unwrap_or(2); // bf16
    // `--fused`: capture the FUSED gather-GEMV MoE decode (lever (c)) instead of the materializing
    // `decode_topk_pre`. The raw `CubeBackend` path launches the two Static kernels DIRECTLY below
    // Fusion, so the capture records them — same hazard surface (no `CubeCount::Dynamic`).
    let fused = args.iter().any(|x| x == "--fused");

    let device: <B as Backend>::Device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (below Fusion)");
    println!(
        "=== WAVE-2 STEP 2: CUDA-graph CAPTURE the 30B MoE static decode step + replay/token ===\n"
    );

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

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm::<B>(&device);
    println!("loading sharded weights from {dir:?} (RAW backend, below Fusion) ...");
    let t0 = Instant::now();
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
    let lp = prompt_ids.len();
    let input: Tensor<B, 2, Int> =
        Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), &device).unsqueeze();
    println!("\nprompt ({lp} tok): {prompt:?}");

    let total = lp + max_new;
    let vocab = cfg.vocab_size;
    let eos: Vec<i64> = vec![]; // full length (no early stop), like the Step-1 parity gate
    // built ONCE post-load (shared by eager + captured). `--fused` selects lever (c)'s gather-GEMV.
    let sd = model.build_static_decode(total).with_fused(fused);
    println!(
        "MoE decode kernel: {}",
        if fused {
            "FUSED gather-GEMV (lever c)"
        } else {
            "materializing decode_topk_pre (oracle)"
        }
    );

    // ========================================================================================
    // (A) EAGER reference: generate_greedy_static on the SAME raw backend (the numerical oracle + the
    //     apples-to-apples eager timing). e2e timed; prefill measured + subtracted to isolate decode.
    // ========================================================================================
    println!(
        "\n===== (A) EAGER static decode (raw backend, below Fusion) — oracle + baseline ====="
    );
    let _ = model.generate_greedy_static(input.clone(), max_new, &eos, &sd); // warm
    block_sync(&client);
    // prefill-only (subtracted from e2e): reuse one cache, reset in place between reps.
    let mut pcache = model.model.new_cache_with_capacity(total);
    let pos0 = Tensor::<B, 1, Int>::arange(0..lp as i64, &device).unsqueeze_dim::<2>(0);
    let _ = model.forward_with_cache(input.clone(), None, pos0.clone(), &mut pcache); // warm+alloc
    block_sync(&client);
    let mut prefill_ms = Vec::new();
    for _ in 0..timing_reps {
        pcache.reset_for_replay();
        block_sync(&client);
        let t = Instant::now();
        let _ = model.forward_with_cache(input.clone(), None, pos0.clone(), &mut pcache);
        block_sync(&client);
        prefill_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let prefill_ms = median(&prefill_ms);

    let mut eager_e2e = Vec::new();
    let mut eager_ids: Vec<i64> = Vec::new();
    for _ in 0..timing_reps {
        let t = Instant::now();
        let out = model.generate_greedy_static(input.clone(), max_new, &eos, &sd);
        block_sync(&client);
        eager_e2e.push(t.elapsed().as_secs_f64() * 1e3);
        eager_ids = out
            .into_data()
            .to_vec::<i32>()
            .unwrap()
            .into_iter()
            .map(|x| x as i64)
            .collect();
    }
    let eager_e2e = median(&eager_e2e);
    let eager_decode_ms = (eager_e2e - prefill_ms).max(0.0);
    let eager_ms_tok = eager_decode_ms / max_new as f64;
    let eager_tok_s = 1000.0 / eager_ms_tok;
    println!(
        "  prefill {prefill_ms:.1} ms | e2e {eager_e2e:.1} ms | decode loop {eager_decode_ms:.1} ms ({eager_ms_tok:.2} ms/tok)"
    );
    println!("  => EAGER (raw, below Fusion): {eager_tok_s:.3} tok/s");

    // ========================================================================================
    // (B) CAPTURED: capture ONE static step, replay max_new/token. Catch a hard capture failure
    //     (metadata miss / Dynamic grid / OOM / illegal address) and report WHICH hazard bit.
    // ========================================================================================
    println!(
        "\n===== (B) CAPTURED static decode (capture 1 step, replay {max_new}/token), warmup={warmup} ====="
    );
    let cap_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        captured_greedy_decode(
            &model,
            input.clone(),
            &sd,
            max_new,
            &eos,
            vocab,
            warmup,
            timing_reps,
        )
    }));

    let captured = match cap_res {
        Ok(c) => c,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic payload>".into());
            println!("\n  ==> CAPTURE FAILED (hard error during capture/replay):");
            println!("      {msg}");
            let hazard = classify_hazard(&msg);
            println!("      HAZARD: {hazard}");
            println!("\n===== SUMMARY =====");
            println!("  (1) capture+replay on 30B: NO — capture aborted (see error above)");
            println!("  (2) token-identical to eager: N/A (no captured output)");
            println!(
                "  (3) captured tok/s: N/A | eager (raw) {eager_tok_s:.3} tok/s | Fusion ref 6.45 tok/s"
            );
            println!("  (4) hazard that bit: {hazard}");
            return Ok(());
        }
    };

    // ---- (3) correctness: captured ids vs eager ids ----
    let captured_ids = &captured.ids;
    let identical = eager_ids == *captured_ids;
    let first_div = eager_ids
        .iter()
        .zip(captured_ids.iter())
        .position(|(a, c)| a != c);
    let eager_txt = tokenizer
        .decode(&eager_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())
        .unwrap_or_default();
    let cap_txt = tokenizer
        .decode(&captured_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())
        .unwrap_or_default();
    println!(
        "  arena high-water = {} MB | captured {} ids",
        captured.arena_bytes / (1024 * 1024),
        captured_ids.len()
    );
    println!("  eager   ids: {:?}", &eager_ids[lp.min(eager_ids.len())..]);
    println!(
        "  capture ids: {:?}",
        &captured_ids[lp.min(captured_ids.len())..]
    );
    println!("  eager   txt: {eager_txt:?}");
    println!("  capture txt: {cap_txt:?}");
    if identical {
        println!(
            "  ==> TOKEN-IDENTICAL: captured replay == eager static decode (capture correctness gate PASS)"
        );
    } else {
        println!(
            "  ==> MISMATCH: first divergence at absolute pos {first_div:?} (completion idx {:?})",
            first_div.map(|i| i.saturating_sub(lp))
        );
    }

    // ---- (4) timing: captured replay tok/s vs eager + GB/s + % peak ----
    let cap_ms_tok = captured.replay_ms_per_tok;
    let cap_tok_s = 1000.0 / cap_ms_tok;
    let (be, ba, bh) = path_bytes_topk(&cfg, dtype_bytes);
    let gb_tok = (be + ba + bh) / 1e9;
    let cap_eff = gb_tok * cap_tok_s;
    let cap_pct = cap_eff / PEAK_GBPS * 100.0;
    let eager_eff = gb_tok * eager_tok_s;
    let eager_pct = eager_eff / PEAK_GBPS * 100.0;
    let speedup = eager_ms_tok / cap_ms_tok;
    println!(
        "\n  ----- PERF (top-8 byte model: {gb_tok:.2} GB/tok = experts {:.2}+attn {:.2}+head {:.2}) -----",
        be / 1e9,
        ba / 1e9,
        bh / 1e9
    );
    println!(
        "    eager  (raw, below Fusion): {eager_ms_tok:6.2} ms/tok  {eager_tok_s:6.3} tok/s  {eager_eff:6.1} GB/s  {eager_pct:3.0}% peak"
    );
    println!(
        "    capture(replay/token)     : {cap_ms_tok:6.2} ms/tok  {cap_tok_s:6.3} tok/s  {cap_eff:6.1} GB/s  {cap_pct:3.0}% peak"
    );
    println!("    capture speedup over eager-raw: {speedup:.2}x");
    println!(
        "    (reference: Fusion eager static was 6.45 tok/s — moe_static_decode.rs amortized)"
    );

    println!("\n===== SUMMARY =====");
    println!(
        "  (1) capture+replay on 30B: YES (arena {} MB)",
        captured.arena_bytes / (1024 * 1024)
    );
    println!(
        "  (2) token-identical to eager static: {}",
        if identical { "YES" } else { "NO" }
    );
    println!(
        "  (3) captured {cap_tok_s:.3} tok/s ({cap_pct:.0}% peak) vs eager-raw {eager_tok_s:.3} tok/s ({eager_pct:.0}% peak), {speedup:.2}x | Fusion ref 6.45 tok/s"
    );
    println!(
        "      => capture {} the eager-raw baseline",
        if speedup > 1.05 {
            "BEAT (launch-bound confirmed)"
        } else {
            "did NOT beat"
        }
    );
    Ok(())
}

/// Map a capture panic message to the most likely Step-2 hazard (best-effort, for the report).
fn classify_hazard(msg: &str) -> &'static str {
    let m = msg.to_lowercase();
    if m.contains("dynamic") {
        "CubeCount::Dynamic — a data-dependent grid was emitted during capture (HARD REJECT)"
    } else if m.contains("intern")
        || m.contains("metadata")
        || m.contains("miss")
        || m.contains("stage")
    {
        "metadata-interning MISS — a locked-pass op's metadata was not staged in warmup (bump --warmup)"
    } else if m.contains("out of memory") || m.contains("oom") || m.contains("alloc") {
        "OOM / arena overflow — the capture arena could not size the 30B step's intermediates"
    } else if m.contains("illegal") || m.contains("address") {
        "illegal address on replay — a baked VA went stale (VA-stability) or pos overran T_max"
    } else if m.contains("va-stability") {
        "VA-STABILITY — a persistent buffer relocated across capture (a stray clone broke the chain)"
    } else {
        "unclassified — see the panic message above"
    }
}
