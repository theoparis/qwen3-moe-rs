//! device_sample_bench — GB10 (sm_121) evidence for the #1 measured GRPO-rollout decode lever (§0-A,
//! docs/VLLM_PARITY_PLAN.md): move per-step sampling + log-prob ONTO the device.
//!
//! `group_sample_cached`'s host sampler syncs the WHOLE `[N, V]` last-token logits to the CPU EVERY
//! decode step (`into_data` → `to_vec`), then per row does a full-vocab softmax + logsumexp + (for
//! temperature/top-p) a FULL SORT (`sample_index`). At the production Qwen3 vocab (151,936) the
//! decode-cost breakdown put that host read + sampling at ~51% of the step.
//! `group_sample_cached_device` instead runs argmax / logsumexp / Gumbel-max categorical selection in
//! pure Burn tensor ops ON the device and copies back only `[N]` tokens + `[N]` log-probs.
//!
//! This bench measures, at production vocab on the real GB10:
//!  (1) per-decode-step SAMPLING cost — host (`into_data[N,V]` + CPU softmax/sort) vs device
//!      (argmax/logsumexp/Gumbel-max + `[N]` copy-back), for BOTH greedy and temperature (the real
//!      GRPO config, where the host pays the O(V log V) sort);
//!  (2) the bytes crossing the host boundary (host: `N*V` floats/step; device: `2*N`);
//!  (3) END-TO-END decode wall-clock for `group_sample_cached` vs `group_sample_cached_device`
//!      (greedy, with an id-parity sanity check; and temperature, the real GRPO config).
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example device_sample_bench

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use qwen3_burn::device_sample_step;
use qwen3_burn::grpo::{group_sample_cached, group_sample_cached_device, RolloutConfig};
use qwen3_burn::Qwen3Config;
use std::time::Instant;

type B = Cuda;

/// Force the device queue to drain so `Instant` brackets measure real GPU work (not just enqueue).
fn sync(d: &CudaDevice) {
    let _ = Tensor::<B, 1>::zeros([1], d).sum().into_scalar();
}

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

// ---- host sampler mirrors (copied from src/sampling.rs + src/grpo/rollout.rs so the example can
//      reproduce the EXACT host per-step cost without crate-private access) ----

/// `logit[token] − logsumexp(raw row)` — the RAW (pre-warp) old log-prob (`raw_token_logprob`).
fn raw_token_logprob(row: &[f32], token: usize) -> f32 {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = m + row.iter().map(|x| (x - m).exp()).sum::<f32>().ln();
    row[token.min(row.len() - 1)] - lse
}

/// Temperature softmax (`temp <= 0` ⇒ one-hot argmax). Mirrors `softmax_temp`.
fn softmax_temp(row: &[f32], temp: f32) -> Vec<f32> {
    if temp <= 0.0 {
        let argmax = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b })
            .0;
        let mut p = vec![0.0f32; row.len()];
        p[argmax] = 1.0;
        return p;
    }
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| ((x - m) / temp).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

/// UNFILTERED categorical draw via the host's full-vocab sort + inverse-CDF (mirrors `sample_index`
/// with `top_k == 0 && top_p >= 1.0` — the GRPO default: still sorts the entire vocab every token).
fn sample_index_unfiltered(probs: &[f32], r: f32) -> usize {
    let mut idx: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)); // FULL O(V log V) sort
    let sum: f32 = idx.iter().map(|(_, p)| *p).sum();
    if sum <= 0.0 {
        return idx.first().map(|(i, _)| *i).unwrap_or(0);
    }
    let mut cum = 0.0f32;
    for (i, p) in &idx {
        cum += p / sum;
        if r < cum {
            return *i;
        }
    }
    idx.last().map(|(i, _)| *i).unwrap_or(0)
}

/// Prime a KV cache with a `ctx0` prefill, then return the steady-state last-token logits `[N, V]` (on
/// device) — the per-step tensor the rollout sampler consumes.
fn primed_last(model: &qwen3_burn::Qwen3ForCausalLM<B>, device: &CudaDevice, n: usize, vocab: usize, ctx0: usize) -> Tensor<B, 2> {
    let prompt = Tensor::<B, 1, Int>::from_data(
        (0..(n * ctx0) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect::<Vec<_>>().as_slice(),
        device,
    )
    .reshape([n, ctx0]);
    let mut cache = model.new_cache_with_capacity(ctx0 + 4);
    let pos0 = Tensor::<B, 1, Int>::arange(0..ctx0 as i64, device).unsqueeze_dim::<2>(0).repeat(&[n, 1]);
    let logits = model.forward_with_cache(prompt, None, pos0, &mut cache); // [n, ctx0, v]
    let [_, _, v] = logits.dims();
    logits.slice([0..n, (ctx0 - 1)..ctx0, 0..v]).reshape([n, v])
}

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Cuda (sm_121 / GB10)\n");

    let vocab = 151936usize; // the real Qwen3 vocab
    let layers = 12usize;
    let n = 64usize; // rollout rows (e.g. 16 prompts x group 4)
    let reps = 64usize;

    <B as Backend>::seed(&device, 7);
    let model = build_model(&device, vocab, layers);

    // ===================================================================================
    // (1) PER-DECODE-STEP SAMPLING COST at production vocab.
    println!("=== (1) PER-STEP SAMPLING COST  (vocab={vocab}, N={n}, avg of {reps}) ===");
    let last = primed_last(&model, &device, n, vocab, 112);
    sync(&device);

    // --- HOST sampling: into_data[N,V] sync + per-row CPU work, EXACTLY as the rollout's `sample_step`
    //     does it: softmax_temp -> sample_index (which sorts the FULL vocab even when unfiltered, the GRPO
    //     default) -> raw_token_logprob. NOTE greedy (temp=0) ALSO pays the sort, because `sample_step`
    //     routes greedy through one-hot softmax_temp + sample_index too — the device argmax skips it. ---
    let host_cost = |temp: f32| -> f64 {
        let t0 = Instant::now();
        let mut sink = 0i64;
        for rep in 0..reps {
            let raw: Vec<f32> = last.clone().into_data().to_vec::<f32>().unwrap(); // [N*V] HOST SYNC
            for s in 0..n {
                let row = &raw[s * vocab..(s + 1) * vocab];
                let probs = softmax_temp(row, temp); // one-hot for temp=0
                let r = (rep as f32 * 0.6180339 + s as f32 * 0.31831) % 1.0;
                let tok = sample_index_unfiltered(&probs, r); // FULL O(V log V) sort (unfiltered default)
                let logp = raw_token_logprob(row, tok); // RAW (pre-warp) old_logprob
                sink ^= (tok as f32 + logp).to_bits() as i64;
            }
        }
        core::hint::black_box(sink);
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let ms_host_greedy = host_cost(0.0);
    let ms_host_temp = host_cost(1.0);

    // --- DEVICE sampling: argmax/logsumexp/Gumbel-max on-device + copy back [N] tokens + [N] logp. ---
    let device_cost = |temp: f32| -> f64 {
        sync(&device);
        let t0 = Instant::now();
        let mut sink = 0i64;
        for _ in 0..reps {
            let (toks, logp) = device_sample_step(last.clone(), temp); // ON device
            let tv = toks.cast(burn::tensor::DType::I64).into_data().to_vec::<i64>().unwrap(); // copy back [N] only
            let lv = logp.into_data().to_vec::<f32>().unwrap(); // copy back [N] only
            sink ^= tv[0] ^ (lv[0].to_bits() as i64);
        }
        core::hint::black_box(sink);
        sync(&device);
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };
    let ms_dev_greedy = device_cost(0.0);
    let ms_dev_temp = device_cost(1.0);

    let host_bytes = n * vocab * 4;
    let dev_bytes = 2 * n * 8; // [N] i64 tokens + [N] f32 logp (~[N]*4); use 8 as an upper bound
    println!("  host boundary: into_data[N,V] = {} floats/step ({:.2} MB)", n * vocab, host_bytes as f64 / 1e6);
    println!("  dev  boundary: [N] tokens + [N] logp = {} values/step ({:.4} MB)\n", 2 * n, dev_bytes as f64 / 1e6);
    println!("  GREEDY  (temp=0):");
    println!("    host sampling (into_data[N,V] + softmax + SORT, as sample_step) : {ms_host_greedy:8.3} ms");
    println!("    device sampling (argmax/logsumexp + [N] copy-back)             : {ms_dev_greedy:8.3} ms");
    println!("    host time removed: {:+.3} ms  ({:.2}x)\n", ms_host_greedy - ms_dev_greedy, ms_host_greedy / ms_dev_greedy);
    println!("  TEMPERATURE  (temp=1.0, unfiltered — the GRPO default, host pays the FULL SORT):");
    println!("    host sampling (into_data[N,V] + CPU softmax + SORT)   : {ms_host_temp:8.3} ms");
    println!("    device sampling (Gumbel-max + [N] copy-back)          : {ms_dev_temp:8.3} ms");
    println!("    host time removed: {:+.3} ms  ({:.2}x)\n", ms_host_temp - ms_dev_temp, ms_host_temp / ms_dev_temp);

    // ===================================================================================
    // (2) END-TO-END decode wall-clock: group_sample_cached vs group_sample_cached_device.
    println!("=== (2) END-TO-END DECODE WALL-CLOCK  (vocab={vocab}, N={n}) ===");
    let (p, g, lp, max_new) = (16usize, 4usize, 16usize, 64usize);
    assert_eq!(p * g, n);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect();
    let prompt = Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
    let eos = [vocab as i64 - 1]; // unlikely id -> generate the full length (worst case for sampling cost)

    let bench_e2e = |temp: f32, parity: bool| {
        let rc = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: temp, top_p: 1.0, top_k: 0 };
        // warmup (JIT-compile kernels) both paths
        let _ = group_sample_cached(&model, prompt.clone(), &rc, &eos);
        let _ = group_sample_cached_device(&model, prompt.clone(), &rc, &eos);
        sync(&device);

        let t0 = Instant::now();
        let a = group_sample_cached(&model, prompt.clone(), &rc, &eos);
        sync(&device);
        let ms_host = t0.elapsed().as_secs_f64() * 1e3;

        let t0 = Instant::now();
        let b = group_sample_cached_device(&model, prompt.clone(), &rc, &eos);
        sync(&device);
        let ms_dev = t0.elapsed().as_secs_f64() * 1e3;

        let tag = if temp <= 0.0 { "GREEDY     " } else { "TEMPERATURE" };
        print!("  {tag} (temp={temp}): host {ms_host:8.1} ms | device {ms_dev:8.1} ms | speedup {:.2}x", ms_host / ms_dev);
        if parity {
            let id_ok = a.seq_ids.into_data().to_vec::<i32>().unwrap() == b.seq_ids.into_data().to_vec::<i32>().unwrap();
            print!(" | ids identical: {id_ok}");
        }
        println!(" (gen_len={})", a.gen_len);
    };
    bench_e2e(0.0, true); // greedy: deterministic -> assert id-parity as a live sanity check
    bench_e2e(1.0, false); // temperature: the real GRPO config (host pays the sort; ids diverge by design)

    println!(
        "\nNOTE: end-to-end includes the transformer forward + logits GEMM (vocab-heavy), so the net \
         decode speedup is smaller than the per-step SAMPLING speedup in (1) — the device path removes \
         the host sampling tax, not the forward. Only [N] tokens + [N] logp cross the host boundary."
    );
}
