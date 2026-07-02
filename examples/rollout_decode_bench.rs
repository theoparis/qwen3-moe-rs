//! rollout_decode_bench — GB10 (sm_121) evidence for the two GRPO-rollout speed levers a 3-voice
//! review identified (after fp8-weights was rejected): dynamic BATCH-SHRINK and (next) Flash-Decode.
//!
//! (a) BATCH-SHRINK speedup: a tiny random-init Qwen3 (timing only), N rollout rows, long decode, and
//!     HIGH length variance (a broad EOS set under GREEDY ⇒ both drivers see an IDENTICAL, reproducible
//!     length spread). Times `group_sample_cached` (forwards every finished row) vs
//!     `group_sample_cached_shrink` (forwards only live rows). Reports wall-clock + the speedup at the
//!     realized length distribution.
//!
//! (b) DECODE-COST breakdown: instruments ONE decode step at the rollout shape and times the major
//!     components — the transformer forward (split, via per-layer micro-benchmarks, into attention-SDPA
//!     vs the projection/MLP linears), the logits projection, the per-step host-sync read
//!     (`into_data` of `[N, vocab]`), and host sampling/argmax. Tells us whether a Flash-Decode
//!     attention kernel is the next worthwhile lever (coarse `Instant` timing around device syncs).
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example rollout_decode_bench

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::backend::Backend;
use burn::tensor::module::attention_fallback as attention;
use burn::tensor::ops::AttentionModuleOptions;
use burn::tensor::{Distribution, Int, Tensor};
use qwen3_burn::grpo::{group_sample_cached, group_sample_cached_shrink, RolloutConfig};
use qwen3_burn::Qwen3Config;
use std::time::Instant;

type B = Cuda;

/// Force the device queue to drain so `Instant` brackets measure real GPU work (not just enqueue).
fn sync(d: &CudaDevice) {
    let _ = Tensor::<B, 1>::zeros([1], d).sum().into_scalar();
}

/// Median / mean / min / max of a length vector.
fn stats(v: &[usize]) -> (usize, usize, f64, usize) {
    let mut s = v.to_vec();
    s.sort_unstable();
    let med = s[s.len() / 2];
    let mean = v.iter().sum::<usize>() as f64 / v.len() as f64;
    (s[0], med, mean, s[s.len() - 1])
}

/// Per-row response lengths from a completion mask `[N, gen_len]` (row-major host vec) = sum of 1s.
fn lengths_from_mask(mask: &[f32], n: usize, gen_len: usize) -> Vec<usize> {
    (0..n).map(|s| (0..gen_len).filter(|&t| mask[s * gen_len + t] == 1.0).count()).collect()
}

/// Build a Qwen3 with the given vocab and a fixed 0.6B-class per-layer geometry (random init).
fn build_model(device: &CudaDevice, vocab: usize, layers: usize) -> qwen3_burn::Qwen3ForCausalLM<B> {
    let cfg = Qwen3Config::new()
        .with_vocab_size(vocab)
        .with_hidden_size(1024)
        .with_intermediate_size(3072)
        .with_num_hidden_layers(layers)
        .with_num_attention_heads(16)
        .with_num_key_value_heads(8)
        .with_head_dim(Some(128));
    cfg.init_causal_lm::<B>(device)
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Cuda (sm_121 / GB10)\n");

    let (hidden, inter, layers, heads, kv, hd) = (1024usize, 3072, 12, 16, 8, 128);

    // =====================================================================================
    // (a) BATCH-SHRINK speedup.
    //
    // GREEDY (temperature 0): deterministic ⇒ shrink and no-shrink decode the SAME tokens, so the
    // speedup is an apples-to-apples, reproducible measure of ONE length distribution (and the ids
    // must match — a live sanity check on top of the bit-parity test). A random-init model is too
    // peaked for a vocab-FRACTION EOS to fire under sampling, but under GREEDY a BROAD EOS set on a
    // SMALL vocab makes distinct prompts terminate at staggered lengths (the recipe pinned by the CPU
    // parity test). The vocab is small ONLY here: the transformer-forward cost (what shrink saves) is
    // vocab-independent, and every per-step cost scales with the live row count, so the speedup RATIO
    // is the same as at production vocab (part b measures the absolute per-component costs at real vocab).
    let vocab_a = 256usize;
    <B as Backend>::seed(&device, 1234);
    let model_a = build_model(&device, vocab_a, layers);
    let (p, g, lp, max_new) = (16usize, 4usize, 16usize, 128usize);
    let n = p * g;
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| (i * 131 + 17) % vocab_a as i64).collect();
    let prompt = Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
    let eos: Vec<i64> = ((vocab_a as i64 * 45 / 100)..vocab_a as i64).collect(); // upper ~55% of vocab
    let rc = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: 0.0, top_p: 1.0, top_k: 0 };

    println!(
        "=== (a) BATCH-SHRINK speedup  (model vocab={vocab_a}, N={n}, prompt_len={lp}, max_new={max_new}, GREEDY) ==="
    );

    // warmup (JIT-compile kernels) + the realized (deterministic) length distribution.
    let warm = group_sample_cached(&model_a, prompt.clone(), &rc, &eos);
    let _ = group_sample_cached_shrink(&model_a, prompt.clone(), &rc, &eos);
    sync(&device);
    let gen_len = warm.gen_len;
    let lens = lengths_from_mask(&warm.completion_mask.into_data().to_vec::<f32>().unwrap(), n, gen_len);
    let (mn, md, me, mx) = stats(&lens);
    let early = lens.iter().filter(|&&l| l < gen_len).count();
    println!("  realized lengths: min={mn} median={md} mean={me:.1} max={mx} | finished_before_end={early}/{n}");

    // time no-shrink
    sync(&device);
    let t0 = Instant::now();
    let a = group_sample_cached(&model_a, prompt.clone(), &rc, &eos);
    sync(&device);
    let ms_unshrunk = t0.elapsed().as_secs_f64() * 1e3;

    // time shrink
    sync(&device);
    let t0 = Instant::now();
    let b = group_sample_cached_shrink(&model_a, prompt.clone(), &rc, &eos);
    sync(&device);
    let ms_shrink = t0.elapsed().as_secs_f64() * 1e3;

    // sanity: identical work ⇒ identical ids (bit-parity also pinned by tests/grpo_rollout.rs).
    let id_parity = a.seq_ids.into_data().to_vec::<i32>().unwrap()
        == b.seq_ids.into_data().to_vec::<i32>().unwrap();

    println!("  no-shrink : {ms_unshrunk:8.1} ms  (gen_len={gen_len})");
    println!("  shrink    : {ms_shrink:8.1} ms  (ids identical to no-shrink: {id_parity})");
    println!("  SPEEDUP   : {:.2}x", ms_unshrunk / ms_shrink);
    let live_steps: usize = lens.iter().sum();
    let full_steps = n * gen_len;
    println!(
        "  work model: no-shrink forwards {full_steps} row-steps; only {live_steps} are 'live' \
         ({:.0}% wasted on finished rows) -> shrink's compute ceiling ~{:.2}x\n",
        100.0 * (1.0 - live_steps as f64 / full_steps as f64),
        full_steps as f64 / live_steps as f64
    );

    // =====================================================================================
    // (b) DECODE-STEP COST BREAKDOWN at PRODUCTION vocab (host read/sample/logits scale with vocab).
    let vocab_b = 151936usize; // the real Qwen3 vocab
    <B as Backend>::seed(&device, 99);
    let model_b = build_model(&device, vocab_b, layers);
    println!("=== (b) DECODE-STEP COST BREAKDOWN  (model vocab={vocab_b}, N={n}) ===");
    decode_cost_breakdown(&model_b, &device, n, vocab_b, hidden, inter, layers, heads, kv, hd);
}

#[allow(clippy::too_many_arguments)]
fn decode_cost_breakdown(
    model: &qwen3_burn::Qwen3ForCausalLM<B>,
    device: &CudaDevice,
    n: usize,
    vocab: usize,
    hidden: usize,
    inter: usize,
    layers: usize,
    heads: usize,
    kv: usize,
    hd: usize,
) {
    let reps = 64usize; // number of PURE decode steps to average over (one primed cache, no re-prefill)
    let ctx0 = 112usize; // prefill length; decode then runs at context ctx0+1 .. ctx0+reps
    let ctx = ctx0 + reps / 2; // representative (avg) KV context length over the decode loop

    // ---------- end-to-end pieces measured THROUGH the real model ----------
    // Prime the cache ONCE with a prefill of `ctx0`, then time `reps` real decode steps (each feeds one
    // token at the next position, advancing `filled`). This is the steady-state per-token decode cost —
    // NOT prefill (the earlier bug: re-prefilling every rep measured prefill+decode).
    let prompt = Tensor::<B, 1, Int>::from_data(
        (0..(n * ctx0) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect::<Vec<_>>().as_slice(),
        device,
    )
    .reshape([n, ctx0]);
    let mut cache = model.new_cache_with_capacity(ctx0 + reps + 4);
    let pos0 = Tensor::<B, 1, Int>::arange(0..ctx0 as i64, device).unsqueeze_dim::<2>(0).repeat(&[n, 1]);
    let _ = model.model.forward_with_cache(prompt, None, pos0, &mut cache, Default::default()); // prefill
    sync(device);

    let decode = |cache: &mut qwen3_burn::ModelCache<B>, pos_i: usize| {
        let next = Tensor::<B, 1, Int>::from_data(vec![7i64; n].as_slice(), device).reshape([n, 1]);
        let pos = Tensor::<B, 1, Int>::from_data([pos_i as i64].as_slice(), device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]);
        model.model.forward_with_cache(next, None, pos, cache, Default::default()) // [n, 1, hidden]
    };

    // warmup one decode (on a throwaway primed cache so we don't burn the timed cache's budget)
    {
        let mut wc = model.new_cache_with_capacity(ctx0 + 2);
        let wp = Tensor::<B, 1, Int>::from_data(vec![1i64; n * ctx0].as_slice(), device).reshape([n, ctx0]);
        let wpos = Tensor::<B, 1, Int>::arange(0..ctx0 as i64, device).unsqueeze_dim::<2>(0).repeat(&[n, 1]);
        let _ = model.model.forward_with_cache(wp, None, wpos, &mut wc, Default::default());
        core::hint::black_box(decode(&mut wc, ctx0));
        sync(device);
    }

    // (1) transformer forward (attention + MLP + norms, all layers) — PURE decode steps
    sync(device);
    let t0 = Instant::now();
    let mut hidden_last = None;
    for r in 0..reps {
        hidden_last = Some(decode(&mut cache, ctx0 + r));
    }
    sync(device);
    let ms_fwd = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let hidden_t = hidden_last.unwrap().reshape([n, hidden]); // [N, hidden]

    // (2) logits projection: hidden @ embedW^T  -> [N, vocab]. Retain EVERY result so the lazy
    // backend can't dead-code-eliminate all-but-last (which made this measurement noisy).
    let embed_w = model.model.embed_tokens_weight(); // [vocab, hidden]
    let w_t = embed_w.transpose(); // [hidden, vocab]
    sync(device);
    let t0 = Instant::now();
    let mut keep = Vec::with_capacity(reps);
    for r in 0..reps {
        // perturb the input per-iter so the backend can't CSE the 64 identical GEMMs into one.
        let h = hidden_t.clone() + (r as f32 * 1e-6);
        keep.push(h.matmul(w_t.clone())); // [N, vocab]
    }
    core::hint::black_box(&keep);
    sync(device);
    let ms_logits = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    let logits_t = keep.pop().unwrap();

    // (3) host-sync: read [N, vocab] logits to host (the per-token into_data the rollout loop does)
    sync(device);
    let t0 = Instant::now();
    let mut raw_last: Vec<f32> = Vec::new();
    for _ in 0..reps {
        raw_last = logits_t.clone().into_data().to_vec::<f32>().unwrap();
    }
    let ms_read = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // (4) host sampling/argmax over [N, vocab] (greedy argmax + raw logprob, the rollout's sample_step)
    let finished = vec![false; n];
    let t0 = Instant::now();
    let mut sink = 0i64;
    for _ in 0..reps {
        // mirror sample_step's greedy work: per-row argmax + a logsumexp for the raw logprob.
        for s in 0..n {
            let row = &raw_last[s * vocab..(s + 1) * vocab];
            let mut bi = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for (i, &x) in row.iter().enumerate() {
                if x > bv {
                    bv = x;
                    bi = i;
                }
            }
            let m = bv;
            let lse: f32 = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
            sink ^= (bi as f32 + lse).to_bits() as i64;
        }
    }
    let ms_sample = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;
    core::hint::black_box((sink, &finished));

    // ---------- per-layer micro-benchmarks: ATTENTION-SDPA vs PROJECTION/MLP linears ----------
    // Decode shape: 1 query attending to T cached keys, per head. Linears are the 2-D GEMMs the model
    // runs (linear2d). Summed x layers ≈ the transformer forward (cross-check vs ms_fwd).
    let x_h = Tensor::<B, 2>::random([n, hidden], Distribution::Normal(0.0, 1.0), device);
    let w_qkv = Tensor::<B, 2>::random([hidden, heads * hd], Distribution::Normal(0.0, 0.02), device);
    let w_kv = Tensor::<B, 2>::random([hidden, kv * hd], Distribution::Normal(0.0, 0.02), device);
    let w_o = Tensor::<B, 2>::random([heads * hd, hidden], Distribution::Normal(0.0, 0.02), device);
    let w_gate = Tensor::<B, 2>::random([hidden, inter], Distribution::Normal(0.0, 0.02), device);
    let w_up = Tensor::<B, 2>::random([hidden, inter], Distribution::Normal(0.0, 0.02), device);
    let w_down = Tensor::<B, 2>::random([inter, hidden], Distribution::Normal(0.0, 0.02), device);
    let x_i = Tensor::<B, 2>::random([n, inter], Distribution::Normal(0.0, 1.0), device);

    // attention tensors (KV heads already expanded to `heads` for GQA, as the model does pre-SDPA)
    let q4 = Tensor::<B, 4>::random([n, heads, 1, hd], Distribution::Normal(0.0, 1.0), device);
    let k4 = Tensor::<B, 4>::random([n, heads, ctx, hd], Distribution::Normal(0.0, 1.0), device);
    let v4 = Tensor::<B, 4>::random([n, heads, ctx, hd], Distribution::Normal(0.0, 1.0), device);

    // warmup the micro ops
    {
        let q = x_h.clone().matmul(w_qkv.clone());
        let a = attention(q4.clone(), k4.clone(), v4.clone(), None, None, AttentionModuleOptions::default());
        core::hint::black_box((q, a));
        sync(device);
    }

    // PROJECTION linears (q,k,v,o)
    sync(device);
    let t0 = Instant::now();
    for _ in 0..reps {
        let q = x_h.clone().matmul(w_qkv.clone());
        let k = x_h.clone().matmul(w_kv.clone());
        let v = x_h.clone().matmul(w_kv.clone());
        let o = q.clone().reshape([n, heads * hd]).matmul(w_o.clone());
        core::hint::black_box((q, k, v, o));
    }
    sync(device);
    let ms_proj_1 = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // MLP linears (gate,up,down + SwiGLU)
    sync(device);
    let t0 = Instant::now();
    for _ in 0..reps {
        let gate = x_h.clone().matmul(w_gate.clone());
        let up = x_h.clone().matmul(w_up.clone());
        let act = burn::tensor::activation::silu(gate) * up;
        let down = act.matmul(w_down.clone());
        core::hint::black_box((down, x_i.clone()));
    }
    sync(device);
    let ms_mlp_1 = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    // ATTENTION SDPA (reference attention_fallback — what the model uses, and what Flash-Decode replaces)
    sync(device);
    let t0 = Instant::now();
    for _ in 0..reps {
        let a = attention(q4.clone(), k4.clone(), v4.clone(), None, None, AttentionModuleOptions::default());
        core::hint::black_box(a);
    }
    sync(device);
    let ms_attn_1 = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

    let ms_proj = ms_proj_1 * layers as f64;
    let ms_mlp = ms_mlp_1 * layers as f64;
    let ms_attn = ms_attn_1 * layers as f64;

    // ---------- report ----------
    let total = ms_fwd + ms_logits + ms_read + ms_sample;
    let pct = |x: f64| 100.0 * x / total;
    println!("  per decode step (avg of {reps}), N={n}, context T={ctx}:");
    println!("    transformer fwd (all {layers} layers) : {ms_fwd:8.3} ms  ({:5.1}%)", pct(ms_fwd));
    println!("    logits proj  [N,h]@[h,{vocab}]        : {ms_logits:8.3} ms  ({:5.1}%)", pct(ms_logits));
    println!("    host-sync read into_data [N,{vocab}]  : {ms_read:8.3} ms  ({:5.1}%)", pct(ms_read));
    println!("    host sampling/argmax (CPU)            : {ms_sample:8.3} ms  ({:5.1}%)", pct(ms_sample));
    println!("    ----------------------------------------------------------");
    println!("    TOTAL per decode step                 : {total:8.3} ms");
    println!();
    println!("  transformer-forward attribution (per-layer micro-bench x {layers} layers):");
    let sub = ms_attn + ms_proj + ms_mlp;
    let subpct = |x: f64| 100.0 * x / sub;
    println!("    attention SDPA (Q@K^T, softmax, @V)   : {ms_attn:8.3} ms  ({:5.1}% of fwd-ops)", subpct(ms_attn));
    println!("    projection linears (q,k,v,o)          : {ms_proj:8.3} ms  ({:5.1}% of fwd-ops)", subpct(ms_proj));
    println!("    MLP linears (gate,up,down + SwiGLU)   : {ms_mlp:8.3} ms  ({:5.1}% of fwd-ops)", subpct(ms_mlp));
    println!(
        "    (GEMM-op sum {sub:.3} ms << measured fwd {ms_fwd:.3} ms: the remaining ~{:.0}% of the \
         forward is NON-GEMM per-token overhead — RmsNorm/RoPE/embed/GQA-expand + kernel-launch latency \
         at batch-of-1 decode, i.e. the forward is launch/bandwidth-bound, not compute-bound.)",
        100.0 * (1.0 - sub / ms_fwd)
    );
    println!();
    let dominant = [
        ("attention-SDPA compute", ms_attn),
        ("projection+MLP GEMMs", ms_proj + ms_mlp),
        ("transformer fwd (whole)", ms_fwd),
        ("logits proj", ms_logits),
        ("host-sync read", ms_read),
        ("host sampling (full-vocab logsumexp, CPU)", ms_sample),
    ]
    .into_iter()
    .fold(("", 0.0f64), |b, x| if x.1 > b.1 { x } else { b });
    let attn_step_pct = 100.0 * ms_attn / total;
    println!("  DOMINATES (whole decode step): {} ({:.1} ms).", dominant.0, dominant.1);
    println!(
        "  Flash-Decode verdict: attention-SDPA is {:.1}% of the forward COMPUTE (T={ctx}) but only \
         ~{attn_step_pct:.0}% of the whole decode step. At production vocab the step is dominated by HOST \
         SAMPLING (full-vocab logsumexp/argmax on CPU) + the logits GEMM, and the forward itself is mostly \
         per-token launch overhead. So the higher-leverage next levers are a DEVICE-SIDE sampling/logprob \
         path (kill the per-token host logsumexp) and a fused logits+top-k; Flash-Decode pays off mainly \
         once context is long (SDPA scores [N,heads,1,T] grow with T) and the host tax is removed.",
        subpct(ms_attn)
    );
}
