//! device_loop_bench — GB10 (sm_121) evidence that removing the LAST per-step device→host sync makes
//! the GRPO decode loop static + host-sync-free (§4 / §0-A2, docs/VLLM_PARITY_PLAN.md — the CUDA-graph
//! prerequisite).
//!
//! `group_sample_cached_device` ALREADY samples on the device, but every decode step it still pays a
//! residual `[N]` round-trip: it copies the `[N]` candidate tokens to the host (`into_data`/`to_vec`),
//! a HOST loop applies EOS/finished masking, uploads the `[N]` next token back (`from_data`), copies the
//! `[N]` log-prob back, `Tensor::cat`s the growing sequence, and a host `finished.iter().all()` decides
//! the early break. Each of those is a stream sync that stalls the GPU on the host — and at DECODE the
//! per-step kernels are tiny, so that sync latency is a real chunk of the step.
//!
//! `group_sample_cached_device_loop` removes ALL of it: EOS/finished tracking, the next-token buffer,
//! and the completion mask are on the device (`mask_where`/`equal_elem`/`bool_or`/`slice_assign`), the
//! decode runs a FIXED `max_new_tokens` steps, and ZERO `into_data`/`to_vec` happens inside the driver.
//!
//! This bench measures, at production vocab on the real GB10:
//!  (1) the ISOLATED per-step EOS/buffer cost — the residual `[N]` host round-trip vs the pure-device
//!      EOS path (the forward is identical in both, so this is exactly the sync we removed);
//!  (2) END-TO-END decode wall-clock — `group_sample_cached_device` (per-step `[N]` sync) vs
//!      `group_sample_cached_device_loop` (one sync at the end), greedy (id-parity sanity) + temperature.
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example device_loop_bench

use burn::prelude::Device;

use burn::tensor::{Device, Int, Tensor};
use qwen3_burn::Qwen3Config;
use qwen3_burn::device_sample_step;
use qwen3_burn::grpo::{
    RolloutConfig, group_sample_cached_device, group_sample_cached_device_loop,
};
use std::time::Instant;

type B = Cuda;

/// Force the device queue to drain so `Instant` brackets measure real GPU work (not just enqueue).
fn sync(d: &CudaDevice) {
    let _ = Tensor::<1>::zeros([1], d).sum().into_scalar();
}

fn build_model(device: &CudaDevice, vocab: usize, layers: usize) -> qwen3_burn::Qwen3ForCausalLM {
    let cfg = Qwen3Config::new()
        .with_vocab_size(vocab)
        .with_hidden_size(1024)
        .with_intermediate_size(3072)
        .with_num_hidden_layers(layers)
        .with_num_attention_heads(16)
        .with_num_key_value_heads(8)
        .with_head_dim(Some(128));
    cfg.init_causal_lm(device)
}

/// Prime a KV cache with a `ctx0` prefill, then return the steady-state last-token logits `[N, V]` (on
/// device) — the per-step tensor the rollout sampler consumes.
fn primed_last(
    model: &qwen3_burn::Qwen3ForCausalLM,
    device: &CudaDevice,
    n: usize,
    vocab: usize,
    ctx0: usize,
) -> Tensor<2> {
    let prompt = Tensor::<1, Int>::from_data(
        (0..(n * ctx0) as i64)
            .map(|i| (i * 131 + 17) % vocab as i64)
            .collect::<Vec<_>>()
            .as_slice(),
        device,
    )
    .reshape([n, ctx0]);
    let mut cache = model.new_cache_with_capacity(ctx0 + 4);
    let pos0 = Tensor::<1, Int>::arange(0..ctx0 as i64, device)
        .unsqueeze_dim::<2>(0)
        .repeat(&[n, 1]);
    let logits = model.forward_with_cache(prompt, None, pos0, &mut cache); // [n, ctx0, v]
    let [_, _, v] = logits.dims();
    logits.slice([0..n, (ctx0 - 1)..ctx0, 0..v]).reshape([n, v])
}

fn main() {
    let device = Device::cuda(0);
    println!("device: {device:?} | backend: Cuda (sm_121 / GB10)\n");

    let vocab = 151936usize; // the real Qwen3 vocab
    let layers = 12usize;
    let n = 64usize; // rollout rows (e.g. 16 prompts x group 4)
    let reps = 64usize;
    let eos_set: Vec<i64> = vec![vocab as i64 - 1]; // unlikely id -> both drivers run the FULL length

    device.seed(7);
    let model = build_model(&device, vocab, layers);

    // ===================================================================================
    // (1) ISOLATED per-step EOS / buffer cost — the residual `[N]` host round-trip the device-SAMPLING
    //     driver still pays, vs the pure-device EOS path of the device-LOOP driver. The forward is
    //     identical in both and is EXCLUDED here, so this is exactly the per-step sync we removed.
    println!("=== (1) PER-STEP EOS/BUFFER COST  (vocab={vocab}, N={n}, avg of {reps}) ===");
    let last = primed_last(&model, &device, n, vocab, 112);
    let eos0 = eos_set[0];
    sync(&device);

    // --- group_sample_cached_device's residual per-step host round-trip: device_sample_step -> copy
    //     BOTH [N] token + [N] logp to the host (`into_data`/`to_vec`) -> host EOS/finished loop ->
    //     [N] from_data upload of the next token. Each into_data is a blocking device->host sync. ---
    let host_sync_cost = || -> f64 {
        let mut finished = vec![false; n];
        let t0 = Instant::now();
        let mut sink = 0i64;
        for _ in 0..reps {
            let (toks, logp_t) = device_sample_step(last.clone(), 0.0); // on device
            let cand: Vec<i64> = toks
                .cast(burn::tensor::DType::I64)
                .into_data()
                .to_vec::<i64>()
                .unwrap(); // [N] SYNC
            let lv: Vec<f32> = logp_t.into_data().to_vec::<f32>().unwrap(); // [N] SYNC
            let next: Vec<i64> = (0..n)
                .map(|s| if finished[s] { eos0 } else { cand[s] })
                .collect(); // host EOS loop
            for s in 0..n {
                if !finished[s] && next[s] == eos0 {
                    finished[s] = true;
                }
            }
            let _next_t = Tensor::<1, Int>::from_data(next.as_slice(), &device).reshape([n, 1]); // [N] upload
            sink ^= next[0] ^ (lv[0].to_bits() as i64);
        }
        core::hint::black_box(sink);
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };

    // --- group_sample_cached_device_loop's pure-device EOS path: device_sample_step ->
    //     mask_where(finished,pad) -> equal_elem -> slice_assign into fixed buffers -> finished bool_or.
    //     NO into_data. We drain ONCE at the end (the single sync), not per step. ---
    let device_path_cost = || -> f64 {
        let total = 128usize;
        let mut tok_buf = Tensor::<2, Int>::zeros([n, total], &device);
        let mut logp_buf = Tensor::<2>::zeros([n, reps], &device);
        let mut mask_buf = Tensor::<2>::zeros([n, reps], &device);
        let mut finished = Tensor::<2, Int>::zeros([n, 1], &device).equal_elem(1i64);
        let pad = Tensor::<2, Int>::full([n, 1], eos0, &device);
        sync(&device);
        let t0 = Instant::now();
        for t in 0..reps {
            let (toks, logp_t) = device_sample_step(last.clone(), 0.0); // on device, no host read
            let sampled = toks.reshape([n, 1]);
            let emit = sampled.mask_where(finished.clone(), pad.clone());
            let is_eos = emit.clone().equal_elem(eos0);
            tok_buf = tok_buf.slice_assign([0..n, t..t + 1], emit);
            logp_buf = logp_buf.slice_assign([0..n, t..t + 1], logp_t.reshape([n, 1]));
            let active = finished.clone().bool_not().float();
            mask_buf = mask_buf.slice_assign([0..n, t..t + 1], active);
            finished = finished.bool_or(is_eos);
        }
        sync(&device); // the SINGLE end-of-loop sync
        // touch the buffers so nothing is dead-code-eliminated
        core::hint::black_box((
            tok_buf.sum().into_scalar(),
            logp_buf.sum().into_scalar(),
            mask_buf.sum().into_scalar(),
        ));
        t0.elapsed().as_secs_f64() * 1e3 / reps as f64
    };

    // warmup
    let _ = host_sync_cost();
    let _ = device_path_cost();
    let ms_host_sync = host_sync_cost();
    let ms_dev_path = device_path_cost();
    println!(
        "  per-step host round-trip ([N] into_data + host EOS + [N] upload + [N] logp sync) : {ms_host_sync:8.4} ms"
    );
    println!(
        "  per-step pure-device EOS path (mask_where/equal_elem/bool_or + slice_assign)      : {ms_dev_path:8.4} ms"
    );
    println!(
        "  net per-step delta (host_round_trip - device_path): {:+.4} ms",
        ms_host_sync - ms_dev_path
    );
    println!(
        "  NOTE: device_sample_step (full-vocab logsumexp/argmax/gather over [N,{vocab}]) dominates BOTH,\n        \
         so the residual [N] sync is a small fraction of the per-step cost at production vocab — the\n        \
         device path trades 2 tiny [N] host copies for a few small device kernels (near break-even).\n"
    );

    // ===================================================================================
    // (2) END-TO-END decode wall-clock: group_sample_cached_device (per-step [N] sync) vs
    //     group_sample_cached_device_loop (one sync at the end). Both forward the same model; with an
    //     unlikely EOS both run the FULL max_new steps, so the delta is the per-step sync we removed.
    println!("=== (2) END-TO-END DECODE WALL-CLOCK  (vocab={vocab}, N={n}) ===");
    let (p, g, lp, max_new) = (16usize, 4usize, 16usize, 64usize);
    assert_eq!(p * g, n);
    let prompt_ids: Vec<i64> = (0..(p * lp) as i64)
        .map(|i| (i * 131 + 17) % vocab as i64)
        .collect();
    let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);

    let e2e_reps = 5usize;
    let bench_e2e = |temp: f32, parity: bool| {
        let rc = RolloutConfig {
            group_size: g,
            max_new_tokens: max_new,
            temperature: temp,
            top_p: 1.0,
            top_k: 0,
        };
        // warmup (JIT-compile kernels) both paths
        let _ = group_sample_cached_device(&model, prompt.clone(), &rc, &eos_set);
        let _ = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos_set);
        sync(&device);

        // average over a few reps (single decode is ~2 s; a few reps tames run-to-run noise).
        let (mut ms_sync, mut ms_loop) = (0.0f64, 0.0f64);
        let (mut last_a_gen, mut last_b_gen) = (0usize, 0usize);
        let mut id_ok = true;
        for _ in 0..e2e_reps {
            let t0 = Instant::now();
            let a = group_sample_cached_device(&model, prompt.clone(), &rc, &eos_set); // per-step [N] sync
            sync(&device);
            ms_sync += t0.elapsed().as_secs_f64() * 1e3;

            let t0 = Instant::now();
            let b = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos_set); // one sync at end
            sync(&device);
            ms_loop += t0.elapsed().as_secs_f64() * 1e3;

            last_a_gen = a.gen_len;
            last_b_gen = b.gen_len;
            if parity {
                // compare the common [0, lp+a.gen_len) prefix (the loop never early-breaks; here EOS is
                // unlikely so a.gen_len == max_new and the whole completion is compared). Cuda Int = i32.
                let g0 = a.gen_len;
                let ai = a.seq_ids.into_data().to_vec::<i32>().unwrap();
                let bi = b.seq_ids.into_data().to_vec::<i32>().unwrap();
                'outer: for s in 0..n {
                    for c in 0..(lp + g0) {
                        if ai[s * (lp + g0) + c] != bi[s * (lp + max_new) + c] {
                            id_ok = false;
                            break 'outer;
                        }
                    }
                }
            }
        }
        ms_sync /= e2e_reps as f64;
        ms_loop /= e2e_reps as f64;

        let tag = if temp <= 0.0 {
            "GREEDY     "
        } else {
            "TEMPERATURE"
        };
        print!(
            "  {tag} (temp={temp}): device-sampling {ms_sync:8.1} ms | device-LOOP {ms_loop:8.1} ms | speedup {:.3}x",
            ms_sync / ms_loop
        );
        if parity {
            print!(" | ids identical: {id_ok}");
        }
        println!(" (gen_len: sampling={last_a_gen} loop={last_b_gen}, avg of {e2e_reps})");
    };
    bench_e2e(0.0, true); // greedy: deterministic -> assert id-parity as a live sanity check
    bench_e2e(1.0, false); // temperature: the real GRPO config (valid different draw; ids diverge by design)

    println!(
        "\nNOTE (honest): end-to-end is dominated by the transformer forward + vocab-heavy logits GEMM \
         (identical in both paths), and device_sample_step's full-vocab work dwarfs the residual [N] sync, \
         so the EAGER speedup is small (~1.0x). What the device-LOOP removes: the per-step [N] EOS \
         device->host round-trip AND the per-step `Tensor::cat` sequence growth (O(T) realloc/copy, not in \
         microbench (1)) — both replaced by slice_assign into FIXED buffers. The payoff is structural: the \
         decode loop is now STATIC + host-sync-free, the CUDA-graph-capture prerequisite. EXACTLY ONE \
         device->host transfer remains: the caller's read of the final buffers."
    );
}
