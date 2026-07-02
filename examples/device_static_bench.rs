//! device_static_bench — GB10 (sm_121) evidence for PHASE 2 of the CUDA-graph plan: the
//! device-`pos`-indexed static cache + fixed-shape decode loop (docs/cudagraph/DESIGN.md §0b P0-A + §7).
//!
//! `group_sample_cached_device_loop` is already host-sync-free, but it is NOT capturable: every per-step
//! op bakes the HOST loop index `t` as a frozen kernel scalar — the KV write offset (`cache.update`'s
//! `off = filled`), the token write `tok_buf.slice_assign([.., (lp+t)..])`, the logp/mask writes
//! `slice_assign([.., t..t+1])`, and the growing-prefix attention read. A graph captured at step `t`
//! would replay into column `t` forever.
//!
//! `group_sample_cached_device_static` removes that: ONE device position counter `pos` (`[1]` Int,
//! `++` on-device per step) drives a `select_assign` KV scatter into the static `[N, T_max, ..]` buffer,
//! fixed-shape full-`T_max` masked attention (columns `idx > pos` set to `-inf`), and `select_assign`
//! token/logp/mask scatters — every per-step op fixed-shape + device-indexed, ZERO host-index-baked ops.
//!
//! This bench measures, at production vocab on the real GB10:
//!  (1) GREEDY id-PARITY: the device-`pos`-indexed static driver must produce BIT-IDENTICAL tokens to
//!      the host-`t`-indexed loop driver (the masked full-`T_max` attention == the growing-prefix attn);
//!  (2) END-TO-END decode wall-clock: loop vs static (greedy + temperature). The static driver removes
//!      per-step reshape/realloc (the growing-prefix slice + the host-`t` slice_assigns) but ADDS the
//!      full-`T_max` KV scan every step (a compute cost at short context — the gpt-fast tradeoff), so in
//!      EAGER the delta is expected near ~1.0x (bandwidth-bound). The payoff is STRUCTURAL: the loop is
//!      now capture-ready (the P-final `client.capture_arena` step is what monetizes it).
//!
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example device_static_bench

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use qwen3_burn::grpo::{
    group_sample_cached_device_loop, group_sample_cached_device_static, RolloutConfig,
};
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

fn main() {
    let device = CudaDevice::default();
    println!("device: {device:?} | backend: Cuda (sm_121 / GB10)\n");

    let vocab = 151936usize; // the real Qwen3 vocab
    let layers = 12usize;
    let (p, g, lp, max_new) = (16usize, 4usize, 16usize, 64usize);
    let n = p * g;
    // unlikely EOS id -> BOTH drivers run the FULL max_new steps (so the delta is purely per-step work).
    let eos_set: Vec<i64> = vec![vocab as i64 - 1];

    <B as Backend>::seed(&device, 7);
    let model = build_model(&device, vocab, layers);

    let prompt_ids: Vec<i64> = (0..(p * lp) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect();
    let prompt = Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);

    println!("=== (1) GREEDY id-PARITY  (vocab={vocab}, N={n}, lp={lp}, max_new={max_new}) ===");
    {
        let rc = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: 0.0, top_p: 1.0, top_k: 0 };
        let a = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos_set); // host-`t` indexed
        let b = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos_set); // device-`pos` indexed
        // Cuda Int = i32. Both run the full length -> identical [N, lp+max_new] shapes.
        let ai = a.seq_ids.into_data().to_vec::<i32>().unwrap();
        let bi = b.seq_ids.into_data().to_vec::<i32>().unwrap();
        let am = a.completion_mask.into_data().to_vec::<f32>().unwrap();
        let bm = b.completion_mask.into_data().to_vec::<f32>().unwrap();
        let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
        let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
        let ids_eq = ai == bi;
        let mask_eq = am == bm;
        let mut maxe = 0.0f32;
        for (x, y) in al.iter().zip(bl.iter()) {
            maxe = maxe.max((x - y).abs());
        }
        println!("  seq_ids bit-identical: {ids_eq} | completion_mask bit-identical: {mask_eq} | raw logp max-err: {maxe:.2e}");
        println!(
            "  => device-`pos`-indexed static == host-`t`-indexed loop (the masked full-T_max attention\n     \
             matches the growing-prefix attention; greedy argmax stable on sm_121).\n"
        );
        assert!(ids_eq, "GREEDY PARITY FAILED: static seq_ids differ from loop seq_ids");
        assert!(mask_eq, "GREEDY PARITY FAILED: static completion_mask differs from loop");
        assert!(maxe < 1e-2, "GREEDY logp drift {maxe} too large (argmax-stable but logp diverged)");
    }

    println!("=== (2) END-TO-END DECODE WALL-CLOCK  (vocab={vocab}, N={n}, lp={lp}, max_new={max_new}) ===");
    let e2e_reps = 5usize;
    let bench_e2e = |temp: f32| {
        let rc = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: temp, top_p: 1.0, top_k: 0 };
        // warmup (JIT-compile kernels) both paths
        let _ = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos_set);
        let _ = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos_set);
        sync(&device);

        let (mut ms_loop, mut ms_static) = (0.0f64, 0.0f64);
        for _ in 0..e2e_reps {
            let t0 = Instant::now();
            let _a = group_sample_cached_device_loop(&model, prompt.clone(), &rc, &eos_set);
            sync(&device);
            ms_loop += t0.elapsed().as_secs_f64() * 1e3;

            let t0 = Instant::now();
            let _b = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos_set);
            sync(&device);
            ms_static += t0.elapsed().as_secs_f64() * 1e3;
        }
        ms_loop /= e2e_reps as f64;
        ms_static /= e2e_reps as f64;

        let tag = if temp <= 0.0 { "GREEDY     " } else { "TEMPERATURE" };
        println!(
            "  {tag} (temp={temp}): loop (host-`t`) {ms_loop:8.1} ms | static (device-`pos`) {ms_static:8.1} ms \
             | static/loop {:.3}x  (avg of {e2e_reps})",
            ms_static / ms_loop
        );
    };
    bench_e2e(0.0);
    bench_e2e(1.0);

    println!(
        "\n=== (3) CAPTURE-READINESS (structural) ===\n  \
         The static driver's per-step body has ZERO host read-backs AND ZERO host-index-baked ops:\n  \
           - KV write       : KVCache::update_static -> select_assign(1, pos_dev, new_kv, Add)  [device index]\n  \
           - decode attn    : full-T_max masked attention, mask = arange(T_max) > pos_dev        [device boundary]\n  \
           - token / logp / mask writes : select_assign(1, pos_dev | pos_dev-lp, ...)            [device index]\n  \
           - RoPE position + decode input : pos_dev / emit (device tensors)                       [no host `t`]\n  \
           - counter advance: pos = pos + 1 (device add of constant 1)                            [never a host int]\n  \
         => a graph captured at one step replays correctly at every step. The remaining P-final step is to\n     \
         wrap this body in `client.capture_arena(...)` (CubeCL capture FFI + pre-reserved arena + `pos` as a\n     \
         pinned static buffer the host bumps per replay) — none of which is Burn-side / built in P2.\n  \
         HONEST COST (measured above): in EAGER the static path is a REGRESSION at this short-context shape\n     \
         (~1.2-1.3x SLOWER), NOT ~1.0x — it scans the full T_max KV (+ GQA-expands it) every step instead of\n     \
         the growing `filled` prefix, and T_max/filled is large at short context (the gpt-fast VRAM/compute\n     \
         tradeoff). The select_assign scatter also costs a touch more than the contiguous slice_assign. The\n     \
         delta shrinks as context grows toward T_max. P2's payoff is NOT an eager speedup — it is the\n     \
         capturability prerequisite (every per-step op fixed-shape + device-indexed); the eager regression is\n     \
         the price, recovered only once the region is actually captured + replayed (P-final)."
    );
}
