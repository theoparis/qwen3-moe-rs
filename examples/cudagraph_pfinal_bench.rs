//! P-FINAL of the CUDA-graph plan (docs/cudagraph/DESIGN.md): an actually-CAPTURED GRPO decode that
//! assembles the four built components into a capture-once / replay-per-token loop on the GB10.
//!
//!   C1 capture/replay  (cubecl-runtime `ComputeClient::capture_arena` + `CapturedGraph`)
//!   C2 capture arena   (graph-private, stable-VA, recycled — now with METADATA INTERNING, below)
//!   C3 device-seed RNG (cubek `random_uniform_with_seeds`; greedy needs no RNG)
//!   P2 static decode   (`forward_with_cache_static` — device-`pos`-indexed, fixed-shape)
//!
//! THE BLOCKER WE HAD TO CLEAR (and the honest story). The four pieces did NOT compose out of the box.
//! The decode runs as BURN tensor ops (matmul / reductions / gather / argmax / select_assign) below
//! Fusion, and EVERY such op stages NON-EMPTY dynamic metadata (`Sequence<FastDivmod>` shapes/strides
//! beyond the by-value grid-constant portion) into a transient device buffer per launch via
//! `create_with_data`. Doing that H2D inside the locked capture window is uncapturable, so the C2 arena
//! hard-errored on all of them (a probe showed matmul-alone captured, but max_dim/sum_dim/argmax/gather/
//! select_assign/slice_assign-at-decode-shapes all BLOCKED). The arena handled ALLOCATION but not the
//! per-op metadata staging (the pinned-keepalive TODO it explicitly deferred).
//!
//! THE FIX (cubecl, local patch): METADATA INTERNING. For a fixed-shape captured region the metadata is
//! IDENTICAL across replays (shape-derived; only device-buffer CONTENTS — `pos`, the token, the logits,
//! the KV — change), so each distinct blob is staged ONCE during warmup into a RETAINED arena block
//! (stable VA) and reused BY CONTENT on the locked capture pass with ZERO H2D. The captured kernels just
//! read the stable-VA metadata; replay re-reads the unchanged bytes. (See `CaptureArena::intern_metadata`.)
//!
//! THE CHAINING (capture ONE step, replay max_new). State that must carry across replays — the device
//! `pos` counter, the logits `last`, `finished`, the token/logp/mask buffers, the KV cache — lives in
//! PERSISTENT buffers allocated OUTSIDE the captured region, written IN PLACE each step (Burn reuses the
//! storage when a tensor is uniquely owned, so the device ADDRESS the graph baked stays stable). `pos`
//! advances IN-GRAPH (`pos = pos + 1`, a captured device add); replay k writes column `lp+k`, so each
//! column is written EXACTLY ONCE across replays and the `select_assign(Add)` over a zero-init buffer is
//! a bit-exact assign. Host staging is HOISTED OUT (RoPE freq table + arange(T_max) precomputed once).
//! GREEDY = zero host work per replay: capture once, `graph.replay()` max_new times, ONE read at the end.
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_pfinal_bench 2>&1 | tail -40

use burn::tensor::{DType, Device, IndexingUpdateOp, Int, Shape, Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use burn_cubecl::tensor::CubeTensor;
use cubecl::Runtime;
use cubecl::cuda::CudaRuntime;
use cubecl::prelude::*;
use cubek_random::{N_SEEDS, random_uniform_with_seeds};
use qwen3_burn::grpo::{RolloutConfig, Rollouts, group_sample_cached_device_static};
use qwen3_burn::sampling_device::{device_select_tokens, device_token_logp, logsumexp_dim1};
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, rope_freqs};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::time::Instant;

type B = CubeBackend<CudaRuntime, f32, i32, u8>;
type Client = cubecl::client::ComputeClient<CudaRuntime>;
type Handle = cubecl::server::Handle;

const HEAD_DIM: usize = 128;
const THETA: f64 = 1_000_000.0;

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

// ---- FIX 2b (3-voice hardening): in-place-VA stability guard ---------------------------------------
// The autoregressive chain RELIES on Burn reusing the SAME device storage for the in-place writeback
// ops (`slice_assign`/`select_assign`/`add_scalar`) on uniquely-owned (`Option::take()`) tensors, so
// the device address the captured graph baked stays valid across replays. That is a heuristic with no
// compiler guarantee: a stray `.clone()` would relocate the write to a fresh VA and silently break
// replay. These read each persistent buffer's TRUE device pointer (`get_resource().resource().ptr`)
// so the harness can assert VA-stability across the capture pass — a real runtime detector for the
// footgun, complementing greedy bit-parity. A transient `.clone()` here shares the SAME storage (same
// VA) and is dropped before the closure runs, so inspection performs no in-place op and never trips
// the `can_mut` heuristic itself.
fn float_va<const D: usize>(t: &Tensor<D>) -> u64 {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(ct) => {
            ct.client
                .get_resource(ct.handle.clone().binding())
                .resource()
                .ptr
        }
        TensorPrimitive::QFloat(_) => unreachable!("persistent buffers are never quantized"),
    }
}
fn int_va<const D: usize>(t: &Tensor<D, Int>) -> u64 {
    let ct = t.clone().into_primitive();
    ct.client
        .get_resource(ct.handle.clone().binding())
        .resource()
        .ptr
}

fn build_model(device: &Device, vocab: usize, layers: usize) -> Qwen3ForCausalLM {
    Qwen3Config::new()
        .with_vocab_size(vocab)
        .with_hidden_size(1024)
        .with_intermediate_size(3072)
        .with_num_hidden_layers(layers)
        .with_num_attention_heads(16)
        .with_num_key_value_heads(8)
        .with_head_dim(Some(HEAD_DIM))
        .init_causal_lm(device)
}

/// CAPTURED greedy decode: assemble C1+C2+C3+P2 into a capture-once / replay-per-token loop.
/// Returns the rollout buffers (device tensors) plus the captured arena high-water (bytes).
fn captured_greedy_decode(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
    v: usize,
    warmup: usize,
) -> (Rollouts, u64, f64) {
    assert_eq!(
        cfg.temperature, 0.0,
        "captured_greedy_decode is greedy-only"
    );
    let device = prompt_ids.device();
    let client = CudaRuntime::client(&device);
    let g = cfg.group_size;
    let [p, lp] = prompt_ids.dims();
    let n = p * g;
    let max_new = cfg.max_new_tokens;
    let total = lp + max_new;
    let eos0 = eos[0];

    // FIX 1 (3-voice hardening): the warmup/capture pass runs at pos = lp+warmup and scatters into
    // column lp+warmup of tok_buf ([n, total=lp+max_new]); the per-replay device-pos advance then
    // writes columns lp..lp+max_new-1. BOTH stay in [0, total) ONLY if warmup < max_new. Raising
    // warmup to settle autotune on a short max_new would scatter OUT OF BOUNDS during warmup/capture
    // -> silent corruption (or a CUDA_ERROR_ILLEGAL_ADDRESS on replay if pos ever exceeds T_max-1).
    // This had no guard before; make it explicit and loud.
    assert!(
        warmup < max_new,
        "capture warmup pass writes column lp+warmup = {lp}+{warmup}; warmup must be < max_new ({max_new}) to stay in [0, T_max={total})"
    );

    // ---- hoisted, position-INDEPENDENT constants (precomputed ONCE, outside the captured region) ----
    let freqs = rope_freqs::<B>(HEAD_DIM, THETA, &device); // [head_dim/2]
    let arange_tmax = Tensor::<1, Int>::arange(0..total as i64, &device); // [T_max]

    let prompt_rep = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, g, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(total);

    // ---- PERSISTENT buffers (allocated ONCE; their device addresses are baked into the graph) ----
    let mut tok_buf = Some(
        Tensor::<2, Int>::zeros([n, total], &device)
            .slice_assign([0..n, 0..lp], prompt_rep.clone()),
    );
    let mut logp_buf = Some(Tensor::<2>::zeros([n, max_new], &device));
    let mut mask_buf = Some(Tensor::<2>::zeros([n, max_new], &device));
    let mut pos = Some(Tensor::<1, Int>::full([1], lp as i64, &device)); // device counter, starts at lp
    let mut finished = Some(Tensor::<2, Int>::zeros([n, 1], &device)); // 0/1 (Int -> resets in place)
    let pad = Tensor::<2, Int>::full([n, 1], eos0, &device); // constant
    let mut last_buf = Some(Tensor::<2>::zeros([n, v], &device)); // persistent logits

    // ---- prefill (eager, variable-shape — NOT captured): KV cols 0..lp + initial logits -> last_buf ----
    let prefill = |cache: &mut _, last_buf: &mut Option<Tensor<2>>| {
        let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]);
        let logits = model.forward_with_cache(prompt_rep.clone(), None, pos0, cache); // [n, lp, v]
        let prefill_last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]); // [n, v]
        let lb = last_buf.take().unwrap();
        *last_buf = Some(lb.slice_assign([0..n, 0..v], prefill_last)); // in place into last_buf
    };
    prefill(&mut cache, &mut last_buf);
    block_sync(&client);

    // ===== FIX 2a (3-voice hardening) — THE IN-PLACE-VA INVARIANT (read this before touching `step`) =====
    // INVARIANT: every persistent buffer (last/emit/pos/tok/logp/mask/KV/seed) MUST be updated IN PLACE
    // via take()+a single op; NEVER `.clone()` then assign. A copy relocates the VA the captured graph
    // baked and silently breaks replay (frozen logits or UB) — Burn's storage reuse is a uniquely-owned
    // heuristic with NO compile/runtime guard. Greedy bit-parity is the ONLY end-to-end detector; keep
    // it in CI. The VA-stability assert below is a cheap second detector for the relocation case.
    //
    // FIX 2b: snapshot each persistent buffer's TRUE device VA BEFORE the closure (and capture), so we
    // can assert it is UNCHANGED after warmup+capture — a stray .clone() relocating any of them is caught
    // here instead of only surfacing as a wrong/frozen replay. Read BEFORE `step` is defined: the closure
    // mutably borrows these buffers for its lifetime, so the snapshot cannot share that borrow window.
    let va_labels = ["tok", "logp", "mask", "pos", "finished", "last"];
    let va_before = [
        int_va(tok_buf.as_ref().unwrap()),
        float_va(logp_buf.as_ref().unwrap()),
        float_va(mask_buf.as_ref().unwrap()),
        int_va(pos.as_ref().unwrap()),
        int_va(finished.as_ref().unwrap()),
        float_va(last_buf.as_ref().unwrap()),
    ];

    // ---- the captured ONE-STEP closure (in-place writeback into the persistent buffers) ----
    let mut step = || {
        // read the CURRENT logits (persistent), sample greedily.
        let last = last_buf.take().unwrap(); // storage L (unique)
        let lse = logsumexp_dim1(last.clone()); // [n,1]
        let sampled = device_select_tokens(&last, 0.0); // [n,1] Int argmax (greedy: no RNG)

        // EOS / finished (Int 0/1): pre-step state drives the emit + mask, then update.
        let fin = finished.take().unwrap(); // [n,1] Int, storage F (unique)
        let fin_mask = fin.clone().equal_elem(1i64); // Bool: true where already finished
        let active = fin.clone().equal_elem(0i64).float(); // [n,1] 1.0 where NOT yet finished
        let emit = sampled.mask_where(fin_mask, pad.clone()); // pad finished rows
        let mut is_eos = emit.clone().equal_elem(eos0);
        for &e in &eos[1..] {
            is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
        }
        let logp = device_token_logp(&last, &emit, &lse).reshape([n, 1]); // [n,1] (borrows last)

        // device-`pos` scatters into the fixed buffers (Add over zero == assign; one write per column).
        let pos_idx = pos.as_ref().unwrap().clone();
        let rel = pos.as_ref().unwrap().clone().sub_scalar(lp as i64); // [1] = t
        tok_buf = Some(tok_buf.take().unwrap().select_assign(
            1,
            pos_idx,
            emit.clone(),
            IndexingUpdateOp::Add,
        ));
        logp_buf = Some(logp_buf.take().unwrap().select_assign(
            1,
            rel.clone(),
            logp,
            IndexingUpdateOp::Add,
        ));
        mask_buf = Some(mask_buf.take().unwrap().select_assign(
            1,
            rel,
            active,
            IndexingUpdateOp::Add,
        ));

        // update finished (Int OR, clamped to {0,1}) — in place at storage F.
        finished = Some(fin.add(is_eos.int()).clamp(0i64, 1i64));

        // decode the NEXT logits from `emit` at device `pos`; write into last_buf IN PLACE (storage L).
        let lg = model.forward_with_cache_static_pre(
            emit,
            pos.as_ref().unwrap().clone(),
            &mut cache,
            &freqs,
            &arange_tmax,
        );
        let new_last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
        last_buf = Some(last.slice_assign([0..n, 0..v], new_last)); // `last` unique -> in place at L

        // advance the device counter IN-GRAPH (a captured device add of constant 1).
        pos = Some(pos.take().unwrap().add_scalar(1i64));
    };

    // ---- CAPTURE one step through the arena (warmup pre-sizes + interns metadata; capture issues 0 H2D) ----
    let graph = unsafe { client.capture_arena(warmup, &mut step) };
    block_sync(&client);
    let arena_bytes = graph.arena_bytes();

    // FIX 2b: re-read VAs after warmup+capture — each persistent buffer MUST still live at the SAME
    // device address the graph baked, or replay would read a stale VA. Hard `assert!` (not debug_assert:
    // the bench runs in --release, where debug_assert is compiled out) — the check is ~6 pointer reads.
    let va_after = [
        int_va(tok_buf.as_ref().unwrap()),
        float_va(logp_buf.as_ref().unwrap()),
        float_va(mask_buf.as_ref().unwrap()),
        int_va(pos.as_ref().unwrap()),
        int_va(finished.as_ref().unwrap()),
        float_va(last_buf.as_ref().unwrap()),
    ];
    for (i, (b, a)) in va_before.iter().zip(va_after.iter()).enumerate() {
        assert_eq!(
            b, a,
            "VA-STABILITY VIOLATION: persistent buffer '{}' relocated across capture ({b:#x} -> {a:#x}); \
             a non-in-place update (likely a stray .clone()) broke the chain — replay would read the stale \
             baked VA (frozen logits / UB)",
            va_labels[i]
        );
    }

    // ---- RESET the persistent buffers to the clean post-prefill state (warmup+capture advanced them),
    //      WITHOUT reallocating (addresses must stay what the graph baked) ----
    cache.reset_for_replay(); // zero KV in place, filled = 0
    prefill(&mut cache, &mut last_buf); // re-write KV cols 0..lp + restore last_buf (in place)
    tok_buf = Some(tok_buf.take().unwrap().mul_scalar(0));
    tok_buf = Some(
        tok_buf
            .take()
            .unwrap()
            .slice_assign([0..n, 0..lp], prompt_rep.clone()),
    );
    logp_buf = Some(logp_buf.take().unwrap().mul_scalar(0.0));
    mask_buf = Some(mask_buf.take().unwrap().mul_scalar(0.0));
    finished = Some(finished.take().unwrap().mul_scalar(0));
    pos = Some(pos.take().unwrap().mul_scalar(0).add_scalar(lp as i64));
    block_sync(&client);

    // ---- the IDEAL captured decode: replay max_new times, ZERO host work per replay (timed) ----
    let t0 = Instant::now();
    for _ in 0..max_new {
        graph.replay();
    }
    block_sync(&client);
    let replay_ms = t0.elapsed().as_secs_f64() * 1e3;

    let rollouts = Rollouts {
        seq_ids: tok_buf.take().unwrap(),
        completion_mask: mask_buf.take().unwrap(),
        old_logprobs: logp_buf.take().unwrap(),
        prompt_len: lp,
        gen_len: max_new,
    };
    // `pos`/`finished` carry no Rust-visible read after their reset, but their device buffers ARE read
    // by every `graph.replay()` above (the graph baked their addresses) — keep them alive until here.
    let _keep_alive = (pos, finished, cache, last_buf);
    (rollouts, arena_bytes, replay_ms)
}

fn seed_bytes(seeds: &[u32; N_SEEDS]) -> Vec<u8> {
    seeds.iter().flat_map(|s| s.to_le_bytes()).collect()
}
fn draw_seeds(rng: &mut StdRng) -> [u32; N_SEEDS] {
    let mut s = [0u32; N_SEEDS];
    for x in s.iter_mut() {
        *x = rng.random::<u32>();
    }
    s
}

/// CAPTURABLE Gumbel-max sampler (C3): draws its uniform from `random_uniform_with_seeds` into a
/// PERSISTENT device handle (a stable VA the captured kernel reads) instead of burn's `Tensor::random`
/// (which allocates a fresh internal seed each call -> frozen under capture). The host rewrites fresh
/// seeds into `seed_handle` before each replay, so the captured stochastic step DECORRELATES. Wraps the
/// filled handle back into a Burn tensor for the gumbel/argmax (all metadata-interned, so capturable).
fn seeded_gumbel_select(
    client: &Client,
    device: &Device,
    logits: &Tensor<2>,
    temp: f32,
    u_handle: &Handle,
    seed_handle: &Handle,
    n: usize,
    v: usize,
) -> Tensor<2, Int> {
    // u ~ Uniform[0,1) into the persistent u_handle (captured kernel; reads seed_handle's stable VA).
    let shape = [n * v];
    let strides = [1usize];
    let out_ref =
        unsafe { TensorHandleRef::<CudaRuntime>::from_raw_parts(u_handle, &strides, &shape, 4) };
    random_uniform_with_seeds::<CudaRuntime>(
        client,
        0.0,
        1.0,
        out_ref,
        f32::cube_type(),
        seed_handle,
    )
    .expect("random_uniform_with_seeds launch failed");
    // bridge the filled handle into a Burn [n,v] f32 tensor (stable VA; refcount bump is fine).
    let ct = CubeTensor::<CudaRuntime>::new_contiguous(
        client.clone(),
        device.clone(),
        Shape::from([n, v]),
        u_handle.clone(),
        DType::F32,
    );
    let u = Tensor::<2>::from_primitive(TensorPrimitive::Float(ct)).clamp(1e-9, 1.0 - 1e-7);
    let gumbel = u.log().neg().log().neg(); // g = -ln(-ln u)
    (logits.clone() / temp + gumbel).argmax(1) // [n,1] categorical sample from softmax(logits/temp)
}

/// TEMPERATURE decode through the seeded Gumbel-max sampler, in one of two modes (FIX 3, 3-voice
/// hardening — gives temperature a REAL autoregressive correctness detector, not just a plumbing check):
///   * `captured = true`  — capture one step, then replay max_new times, writing FRESH seeds into the
///     persistent `seed_handle` before EACH replay (C3 option (c)) so each step decorrelates.
///   * `captured = false` — EAGER reference: run the SAME step closure directly max_new times (no
///     capture/replay), writing the IDENTICAL per-step seed stream. With the same `seed_base` the two
///     modes draw the same noise on a bit-identical forward, so captured == eager token-for-token IFF
///     the captured autoregressive chain is correct. `seed_base` seeds the per-step seed STREAM.
fn temperature_decode(
    model: &Qwen3ForCausalLM,
    prompt_ids: Tensor<2, Int>,
    cfg: &RolloutConfig,
    eos: &[i64],
    v: usize,
    warmup: usize,
    seed_base: u64,
    captured: bool,
) -> Rollouts {
    assert!(
        cfg.temperature > 0.0,
        "temperature_decode needs temperature > 0"
    );
    let device = prompt_ids.device();
    let client = CudaRuntime::client(&device);
    let g = cfg.group_size;
    let [p, lp] = prompt_ids.dims();
    let n = p * g;
    let max_new = cfg.max_new_tokens;
    let total = lp + max_new;
    let eos0 = eos[0];
    let temp = cfg.temperature;

    // FIX 1: same OOB guard as the greedy harness — the capture pass writes column lp+warmup, in
    // bounds only if warmup < max_new. (Vacuous for the eager reference, which never captures.)
    if captured {
        assert!(
            warmup < max_new,
            "capture warmup pass writes column lp+warmup = {lp}+{warmup}; warmup must be < max_new ({max_new}) to stay in [0, T_max={total})"
        );
    }

    let freqs = rope_freqs::<B>(HEAD_DIM, THETA, &device);
    let arange_tmax = Tensor::<1, Int>::arange(0..total as i64, &device);
    let prompt_rep = prompt_ids
        .unsqueeze_dim::<3>(1)
        .repeat(&[1, g, 1])
        .reshape([n, lp]);
    let mut cache = model.new_cache_with_capacity(total);

    let mut tok_buf = Some(
        Tensor::<2, Int>::zeros([n, total], &device)
            .slice_assign([0..n, 0..lp], prompt_rep.clone()),
    );
    let mut logp_buf = Some(Tensor::<2>::zeros([n, max_new], &device));
    let mut mask_buf = Some(Tensor::<2>::zeros([n, max_new], &device));
    let mut pos = Some(Tensor::<1, Int>::full([1], lp as i64, &device));
    let mut finished = Some(Tensor::<2, Int>::zeros([n, 1], &device));
    let pad = Tensor::<2, Int>::full([n, 1], eos0, &device);
    let mut last_buf = Some(Tensor::<2>::zeros([n, v], &device));

    // PERSISTENT C3 buffers (allocated OUTSIDE capture): u (the gumbel uniform) + the 4 seeds.
    let u_handle = client.empty(n * v * 4);
    let seed_handle = client.empty(N_SEEDS * 4);
    let mut seed_rng = StdRng::seed_from_u64(seed_base);
    client.write_to_handle(&seed_handle, &seed_bytes(&draw_seeds(&mut seed_rng))); // seeds for warmup/capture

    let prefill = |cache: &mut _, last_buf: &mut Option<Tensor<2>>| {
        let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]);
        let logits = model.forward_with_cache(prompt_rep.clone(), None, pos0, cache);
        let prefill_last = logits.slice([0..n, (lp - 1)..lp, 0..v]).reshape([n, v]);
        let lb = last_buf.take().unwrap();
        *last_buf = Some(lb.slice_assign([0..n, 0..v], prefill_last));
    };
    prefill(&mut cache, &mut last_buf);
    block_sync(&client);

    // FIX 2a INVARIANT (see captured_greedy_decode): persistent buffers update IN PLACE via take()+
    // single-op; NEVER clone-then-assign (relocates the baked VA -> frozen/UB). FIX 2b: snapshot device
    // VAs BEFORE the closure (the closure mutably borrows these buffers for its lifetime, so the snapshot
    // must precede its definition) to assert them unchanged after capture. (u_handle/seed_handle are raw
    // Handles never reassigned -> VA stable by construction, no check needed.) Unused in the eager path.
    let va_labels = ["tok", "logp", "mask", "pos", "finished", "last"];
    let va_before = [
        int_va(tok_buf.as_ref().unwrap()),
        float_va(logp_buf.as_ref().unwrap()),
        float_va(mask_buf.as_ref().unwrap()),
        int_va(pos.as_ref().unwrap()),
        int_va(finished.as_ref().unwrap()),
        float_va(last_buf.as_ref().unwrap()),
    ];

    let mut step = || {
        let last = last_buf.take().unwrap();
        let lse = logsumexp_dim1(last.clone());
        let sampled =
            seeded_gumbel_select(&client, &device, &last, temp, &u_handle, &seed_handle, n, v);
        let fin = finished.take().unwrap();
        let fin_mask = fin.clone().equal_elem(1i64);
        let active = fin.clone().equal_elem(0i64).float();
        let emit = sampled.mask_where(fin_mask, pad.clone());
        let mut is_eos = emit.clone().equal_elem(eos0);
        for &e in &eos[1..] {
            is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
        }
        let logp = device_token_logp(&last, &emit, &lse).reshape([n, 1]);
        let pos_idx = pos.as_ref().unwrap().clone();
        let rel = pos.as_ref().unwrap().clone().sub_scalar(lp as i64);
        tok_buf = Some(tok_buf.take().unwrap().select_assign(
            1,
            pos_idx,
            emit.clone(),
            IndexingUpdateOp::Add,
        ));
        logp_buf = Some(logp_buf.take().unwrap().select_assign(
            1,
            rel.clone(),
            logp,
            IndexingUpdateOp::Add,
        ));
        mask_buf = Some(mask_buf.take().unwrap().select_assign(
            1,
            rel,
            active,
            IndexingUpdateOp::Add,
        ));
        finished = Some(fin.add(is_eos.int()).clamp(0i64, 1i64));
        let lg = model.forward_with_cache_static_pre(
            emit,
            pos.as_ref().unwrap().clone(),
            &mut cache,
            &freqs,
            &arange_tmax,
        );
        let new_last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
        last_buf = Some(last.slice_assign([0..n, 0..v], new_last));
        pos = Some(pos.take().unwrap().add_scalar(1i64));
    };
    if captured {
        let graph = unsafe { client.capture_arena(warmup, &mut step) };
        block_sync(&client);
        let va_after = [
            int_va(tok_buf.as_ref().unwrap()),
            float_va(logp_buf.as_ref().unwrap()),
            float_va(mask_buf.as_ref().unwrap()),
            int_va(pos.as_ref().unwrap()),
            int_va(finished.as_ref().unwrap()),
            float_va(last_buf.as_ref().unwrap()),
        ];
        for (i, (b, a)) in va_before.iter().zip(va_after.iter()).enumerate() {
            assert_eq!(
                b, a,
                "VA-STABILITY VIOLATION (temp): persistent buffer '{}' relocated across capture \
                 ({b:#x} -> {a:#x}) — a non-in-place update broke the chain",
                va_labels[i]
            );
        }

        // reset to clean post-prefill state (addresses preserved).
        cache.reset_for_replay();
        prefill(&mut cache, &mut last_buf);
        tok_buf = Some(tok_buf.take().unwrap().mul_scalar(0));
        tok_buf = Some(
            tok_buf
                .take()
                .unwrap()
                .slice_assign([0..n, 0..lp], prompt_rep.clone()),
        );
        logp_buf = Some(logp_buf.take().unwrap().mul_scalar(0.0));
        mask_buf = Some(mask_buf.take().unwrap().mul_scalar(0.0));
        finished = Some(finished.take().unwrap().mul_scalar(0));
        pos = Some(pos.take().unwrap().mul_scalar(0).add_scalar(lp as i64));
        block_sync(&client);

        // TEMPERATURE replay: rewrite FRESH seeds into the persistent buffer BEFORE each replay (on-
        // stream, ordered before launch) so each captured step draws independent noise -> decorrelates.
        let mut replay_rng = StdRng::seed_from_u64(seed_base);
        for _ in 0..max_new {
            client.write_to_handle(&seed_handle, &seed_bytes(&draw_seeds(&mut replay_rng)));
            graph.replay();
        }
        block_sync(&client);
    } else {
        // FIX 3 — EAGER temperature reference: run the SAME step closure DIRECTLY (no capture/replay)
        // max_new times from the clean post-prefill state, writing the IDENTICAL per-step seed stream
        // (StdRng::seed_from_u64(seed_base), the same sequence the captured replay draws). Bit-identical
        // forward + deterministic seed => this reproduces the captured decode token-for-token IFF the
        // captured autoregressive chain is correct. This is the temperature-parity reference.
        let mut eager_rng = StdRng::seed_from_u64(seed_base);
        for _ in 0..max_new {
            client.write_to_handle(&seed_handle, &seed_bytes(&draw_seeds(&mut eager_rng)));
            step();
        }
        block_sync(&client);
    }

    let rollouts = Rollouts {
        seq_ids: tok_buf.take().unwrap(),
        completion_mask: mask_buf.take().unwrap(),
        old_logprobs: logp_buf.take().unwrap(),
        prompt_len: lp,
        gen_len: max_new,
    };
    let _keep_alive = (pos, finished, cache, last_buf, u_handle, seed_handle);
    rollouts
}

fn main() {
    let device: Device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (below Fusion)\n");
    println!(
        "=== P-FINAL: actually-CAPTURED GRPO greedy decode (capture 1 step, replay max_new) ===\n"
    );

    // ---------------------------------------------------------------------------------------------
    // (1) GREEDY BIT-IDENTITY: captured (capture+replay) == eager static (device-`pos` loop).
    // ---------------------------------------------------------------------------------------------
    {
        let (vocab, layers) = (151936usize, 6usize);
        let (p, g, lp, max_new) = (4usize, 2usize, 8usize, 24usize);
        let n = p * g;
        let eos: Vec<i64> = vec![vocab as i64 - 1]; // unlikely -> both run the full length
        device.seed(7);
        let model = build_model(&device, vocab, layers);
        let prompt_ids: Vec<i64> = (0..(p * lp) as i64)
            .map(|i| (i * 131 + 17) % vocab as i64)
            .collect();
        let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
        let rc = RolloutConfig {
            group_size: g,
            max_new_tokens: max_new,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        };

        let eager = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos);
        let (captured, arena, _) =
            captured_greedy_decode(&model, prompt.clone(), &rc, &eos, vocab, 3);

        let ei = eager.seq_ids.into_data().to_vec::<i32>().unwrap();
        let ci = captured.seq_ids.into_data().to_vec::<i32>().unwrap();
        let em = eager.completion_mask.into_data().to_vec::<f32>().unwrap();
        let cm = captured
            .completion_mask
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let el = eager.old_logprobs.into_data().to_vec::<f32>().unwrap();
        let cl = captured.old_logprobs.into_data().to_vec::<f32>().unwrap();
        let ids_eq = ei == ci;
        let mask_eq = em == cm;
        let logp_max_err = el
            .iter()
            .zip(cl.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let logp_eq = logp_max_err == 0.0;

        println!(
            "  vocab={vocab} layers={layers} N={n} lp={lp} max_new={max_new} | arena high-water = {} KB",
            arena / 1024
        );
        println!(
            "  [guard] FIX-1 OOB: warmup(3) < max_new({max_new}) asserted | FIX-2b VA-stability: all 6 persistent buffers (tok/logp/mask/pos/finished/last) UNCHANGED across capture (asserted in-harness)"
        );
        println!("  seq_ids  bit-identical (captured == eager static): {ids_eq}");
        println!("  comp_mask bit-identical:                            {mask_eq}");
        println!(
            "  logp     bit-identical:                            {logp_eq}  (max_abs_err = {logp_max_err:.2e})"
        );
        let pass = ids_eq && mask_eq && logp_eq;
        println!(
            "  => {}\n",
            if pass {
                "PASS: the assembled C1+C2+C3(none)+P2 capture path is CORRECT"
            } else {
                "FAIL"
            }
        );
        assert!(ids_eq, "captured seq_ids differ from eager static");
        assert!(mask_eq, "captured completion_mask differ from eager static");
        assert!(
            logp_eq,
            "captured logp differ from eager static (max_err {logp_max_err:.2e})"
        );
    }

    // ---------------------------------------------------------------------------------------------
    // (2) EOS / pad path: captured handles per-row EOS + post-EOS padding bit-identically.
    // ---------------------------------------------------------------------------------------------
    {
        let (vocab, layers) = (64usize, 4usize);
        let (p, g, lp, max_new) = (3usize, 2usize, 6usize, 20usize);
        let n = p * g;
        device.seed(123);
        let model = build_model(&device, vocab, layers);
        let prompt_ids: Vec<i64> = (0..(p * lp) as i64)
            .map(|i| (i * 17 + 3) % vocab as i64)
            .collect();
        let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
        let rc = RolloutConfig {
            group_size: g,
            max_new_tokens: max_new,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        };

        // PROBE: decode once with an unreachable eos, then pick a REAL generated token (mid-completion)
        // as the eos so the finished/pad transition is actually exercised in BOTH paths.
        let probe =
            group_sample_cached_device_static(&model, prompt.clone(), &rc, &[vocab as i64 - 1]);
        let pi = probe.seq_ids.into_data().to_vec::<i32>().unwrap();
        let mid = pi[(lp + max_new / 2) as usize] as i64; // row 0's token at completion step max_new/2
        let other = pi[(lp + max_new / 2 + 3) as usize] as i64;
        let eos: Vec<i64> = vec![mid, other]; // 2-element eos set (also exercises the bool_or fold)

        let eager = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos);
        let (captured, _, _) = captured_greedy_decode(&model, prompt.clone(), &rc, &eos, vocab, 3);
        let ei = eager.seq_ids.into_data().to_vec::<i32>().unwrap();
        let ci = captured.seq_ids.into_data().to_vec::<i32>().unwrap();
        let em = eager.completion_mask.into_data().to_vec::<f32>().unwrap();
        let cm = captured
            .completion_mask
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let finished_rows = em
            .chunks(max_new)
            .filter(|r| r.iter().any(|&x| x == 0.0))
            .count();
        let ids_eq = ei == ci;
        let mask_eq = em == cm;
        println!(
            "  [EOS] vocab={vocab} N={n} eos={eos:?}: rows that hit EOS = {finished_rows}/{n}"
        );
        println!("  [EOS] seq_ids bit-identical: {ids_eq} | comp_mask bit-identical: {mask_eq}");
        println!(
            "  => {}\n",
            if ids_eq && mask_eq {
                "PASS: device-pos EOS/pad path captures correctly"
            } else {
                "FAIL"
            }
        );
        assert!(
            ids_eq && mask_eq,
            "captured EOS/pad path differs from eager static"
        );
    }

    // ---------------------------------------------------------------------------------------------
    // (3) HONEST TIMING at a GRPO-ish shape: eager static decode vs N captured replays.
    // ---------------------------------------------------------------------------------------------
    {
        let (vocab, layers) = (151936usize, 12usize);
        let (p, g, lp, max_new) = (16usize, 4usize, 16usize, 64usize); // the real GRPO shape: N=64
        let n = p * g;
        let eos: Vec<i64> = vec![vocab as i64 - 1];
        device.seed(7);
        let model = build_model(&device, vocab, layers);
        let prompt_ids: Vec<i64> = (0..(p * lp) as i64)
            .map(|i| (i * 131 + 17) % vocab as i64)
            .collect();
        let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
        let rc = RolloutConfig {
            group_size: g,
            max_new_tokens: max_new,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        };

        let reps = 5usize;
        let prompt_rep = prompt
            .clone()
            .unsqueeze_dim::<3>(1)
            .repeat(&[1, g, 1])
            .reshape([n, lp]);
        let pos0 = Tensor::<1, Int>::arange(0..lp as i64, &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[n, 1]);

        // PREFILL-ONLY timing — BOTH paths pay this once per rollout; subtract it to isolate the decode
        // loop (the part capture actually replaces). Reuse ONE cache (zero in place between reps, NOT
        // realloc — the 629 MB KV alloc would otherwise dwarf the prefill compute and skew the ratio).
        let mut pcache = model.new_cache_with_capacity(lp + max_new);
        let _ = model.forward_with_cache(prompt_rep.clone(), None, pos0.clone(), &mut pcache); // warm + alloc
        block_sync(&client);
        let mut prefill_ms = 0.0f64;
        for _ in 0..reps {
            pcache.reset_for_replay(); // zero KV in place (cheap), filled = 0
            block_sync(&client);
            let t0 = Instant::now();
            let _ = model.forward_with_cache(prompt_rep.clone(), None, pos0.clone(), &mut pcache);
            block_sync(&client);
            prefill_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        prefill_ms /= reps as f64;

        // eager: warm + time end-to-end static decode.
        let _ = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos);
        block_sync(&client);
        let mut eager_ms = 0.0f64;
        for _ in 0..reps {
            let t0 = Instant::now();
            let _ = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos);
            block_sync(&client);
            eager_ms += t0.elapsed().as_secs_f64() * 1e3;
        }
        eager_ms /= reps as f64;
        let eager_decode = eager_ms - prefill_ms; // ~the N-step decode loop alone

        // captured: build the harness (capture once), then time its N-replay hot loop. Average over reps.
        let mut replay_ms = 0.0f64;
        let mut arena = 0u64;
        // warmup=8 so autotune settles BEFORE the capture pass (else the graph bakes a slower untuned
        // matmul and replay is artificially slow — the design's R2 "freeze autotune before capture").
        for _ in 0..reps {
            let (_r, a, ms) = captured_greedy_decode(&model, prompt.clone(), &rc, &eos, vocab, 8);
            replay_ms += ms;
            arena = a;
        }
        replay_ms /= reps as f64;

        println!(
            "  shape: vocab={vocab} layers={layers} N={n} lp={lp} max_new={max_new}  | arena = {} KB",
            arena / 1024
        );
        println!("  prefill only (shared by both):                     {prefill_ms:8.2} ms");
        println!(
            "  eager static decode e2e (prefill + {max_new} steps + finalize): {eager_ms:8.2} ms"
        );
        println!(
            "  eager DECODE LOOP (e2e - prefill):                 {eager_decode:8.2} ms ({:.2} ms/step)",
            eager_decode / max_new as f64
        );
        println!(
            "  captured {max_new} replays (decode hot-loop):                 {replay_ms:8.2} ms ({:.2} ms/replay)",
            replay_ms / max_new as f64
        );
        let speedup = eager_decode / replay_ms;
        println!(
            "  net decode-loop (replays / eager-decode-loop):     {:.3}x  ({:.2}x speedup)",
            replay_ms / eager_decode,
            speedup
        );
        println!(
            "  net end-to-end  (replays / eager-e2e):             {:.3}x",
            replay_ms / eager_ms
        );
        println!(
            "\n  HONEST READ: capture removes the per-step host LAUNCH LATENCY (~{:.1} ms/step here, the cost\n  \
             of issuing the ~70 kernels of a decode step), giving a measured {:.2}x decode-loop speedup. But\n  \
             the step is BANDWIDTH-bound — the tied-head logits GEMM streams ~0.6 GB/step at production vocab,\n  \
             which graphs do NOT touch — so the win is bounded and SHRINKS toward 1.0x as N / context / model\n  \
             grow (more GEMM bandwidth per launch). This sits in the design's predicted ~1.1-1.4x@small ->\n  \
             ~1.0x@large band. The DELIVERABLE is a CORRECT, working captured decode + the framework capability\n  \
             (metadata interning + in-graph chaining), not a large speedup for this workload.",
            (eager_decode - replay_ms) / max_new as f64,
            speedup
        );
    }

    // ---------------------------------------------------------------------------------------------
    // (4) TEMPERATURE DECORRELATION: a CAPTURED temperature decode, with a fresh per-replay seed write
    //     (C3 option (c)) through the real Gumbel-max sampler, produces VARIED + DECORRELATED samples.
    // ---------------------------------------------------------------------------------------------
    {
        let (vocab, layers) = (151936usize, 6usize);
        let (p, g, lp, max_new) = (4usize, 2usize, 8usize, 24usize);
        let n = p * g;
        let eos: Vec<i64> = vec![vocab as i64 - 1];
        device.seed(7);
        let model = build_model(&device, vocab, layers);
        let prompt_ids: Vec<i64> = (0..(p * lp) as i64)
            .map(|i| (i * 131 + 17) % vocab as i64)
            .collect();
        let prompt = Tensor::<1, Int>::from_data(prompt_ids.as_slice(), &device).reshape([p, lp]);
        // temp high enough that the Gumbel noise dominates the (uncalibrated random-weight) logits, so
        // the per-replay SEED visibly controls the draw — the sharpest probe of "does the seed write
        // reach the captured kernel". (At temp=1.0 these random-init logits dominate, hiding the noise;
        // a trained model is calibrated so temp=1.0 noise matters — that's a weights property, not the
        // mechanism's.)
        let rc = RolloutConfig {
            group_size: g,
            max_new_tokens: max_new,
            temperature: 64.0,
            top_p: 1.0,
            top_k: 0,
        };

        // DIAGNOSTIC (eager, NOT captured): does the seeded Gumbel sampler itself decorrelate when the
        // seed handle changes? Isolates "sampler broken" from "capture froze the seed".
        {
            let u_h = client.empty(n * vocab * 4);
            let s_h = client.empty(N_SEEDS * 4);
            let logits = Tensor::<2>::zeros([n, vocab], &device); // flat logits -> pure-noise argmax
            client.write_to_handle(&s_h, &seed_bytes(&[1, 2, 3, 4]));
            let t1 = seeded_gumbel_select(&client, &device, &logits, 8.0, &u_h, &s_h, n, vocab);
            let v1 = t1.into_data().to_vec::<i32>().unwrap();
            client.write_to_handle(&s_h, &seed_bytes(&[9, 8, 7, 6]));
            let t2 = seeded_gumbel_select(&client, &device, &logits, 8.0, &u_h, &s_h, n, vocab);
            let v2 = t2.into_data().to_vec::<i32>().unwrap();
            let eager_differ = v1.iter().zip(v2.iter()).filter(|(a, b)| a != b).count();
            println!(
                "  [TEMP-diag] eager seeded sampler, 2 seeds: argmax differs in {eager_differ}/{n} rows (expect >0)"
            );
        }
        // DIAGNOSTIC 2: capture JUST the seeded sampler, replay with 2 seeds. Isolates seeded-sampler-
        // under-capture from the full-decode integration.
        {
            let u_h = client.empty(n * vocab * 4);
            let s_h = client.empty(N_SEEDS * 4);
            let logits = Tensor::<2>::zeros([n, vocab], &device);
            let mut out = Some(Tensor::<2, Int>::zeros([n, 1], &device));
            client.write_to_handle(&s_h, &seed_bytes(&[1, 2, 3, 4]));
            let mut sstep = || {
                let sel =
                    seeded_gumbel_select(&client, &device, &logits, 8.0, &u_h, &s_h, n, vocab);
                out = Some(out.take().unwrap().mul_scalar(0).add(sel));
            };
            let cg = unsafe { client.capture_arena(2, &mut sstep) };
            block_sync(&client);
            client.write_to_handle(&s_h, &seed_bytes(&[1, 2, 3, 4]));
            cg.replay();
            block_sync(&client);
            let vp = out
                .as_ref()
                .unwrap()
                .clone()
                .into_data()
                .to_vec::<i32>()
                .unwrap();
            client.write_to_handle(&s_h, &seed_bytes(&[99, 98, 97, 96]));
            cg.replay();
            block_sync(&client);
            let vq = out.take().unwrap().into_data().to_vec::<i32>().unwrap();
            let cap_differ = vp.iter().zip(vq.iter()).filter(|(a, b)| a != b).count();
            println!(
                "  [TEMP-diag] CAPTURED seeded sampler, 2 replay-seeds: argmax differs in {cap_differ}/{n} rows (expect >0)"
            );
            drop(cg);
        }

        // two captured temperature runs with DIFFERENT per-replay seed streams + one repeat of stream A.
        let a1 = temperature_decode(&model, prompt.clone(), &rc, &eos, vocab, 3, 0xA, true);
        let a2 = temperature_decode(&model, prompt.clone(), &rc, &eos, vocab, 3, 0xA, true);
        let b1 = temperature_decode(&model, prompt.clone(), &rc, &eos, vocab, 3, 0xB, true);
        let a1i = a1.seq_ids.into_data().to_vec::<i32>().unwrap();
        let a2i = a2.seq_ids.into_data().to_vec::<i32>().unwrap();
        let b1i = b1.seq_ids.into_data().to_vec::<i32>().unwrap();
        // completion region only (cols lp..lp+max_new).
        let comp = |all: &[i32]| -> Vec<i32> {
            (0..n)
                .flat_map(|r| (lp..lp + max_new).map(move |c| all[r * (lp + max_new) + c]))
                .collect()
        };
        let (ca1, ca2, cb1) = (comp(&a1i), comp(&a2i), comp(&b1i));
        let valid = ca1.iter().all(|&t| t >= 0 && (t as usize) < vocab);
        // VARIED: the sampled completions are not a single frozen token id (Gumbel noise spreads them).
        let distinct: std::collections::HashSet<i32> = ca1.iter().copied().collect();
        let varied = distinct.len() > 3;
        // DECORRELATED: stream A != stream B (the per-replay seed write controls the noise end-to-end).
        let differ_ab = ca1.iter().zip(cb1.iter()).filter(|(x, y)| x != y).count();
        let frac_ab = differ_ab as f64 / ca1.len() as f64;
        let decorrelated = frac_ab > 0.3;
        // DETERMINISTIC: same seed stream A -> identical (proves the seed buffer fully drives the draw).
        let same_aa = ca1 == ca2;

        // FIX 3 (3-voice hardening) — TEMPERATURE PARITY: a REAL autoregressive correctness detector,
        // not just a plumbing check. Drive an EAGER temperature decode with the IDENTICAL per-step seed
        // stream (seed_base 0xA — the same StdRng the captured replay draws) through the same seeded
        // Gumbel sampler + forward_with_cache_static_pre. The captured forward is bit-identical to eager
        // (greedy section (1) proves it) and the Gumbel noise is deterministic given the seed, so the
        // captured temperature decode MUST reproduce the eager one token-for-token. A bug UNIQUE to the
        // captured autoregressive chain (frozen `last`, mis-chained emit->next forward, stale pos) would
        // diverge here even though decorrelation/determinism above pass.
        let eager_a = temperature_decode(&model, prompt.clone(), &rc, &eos, vocab, 3, 0xA, false);
        let eai = eager_a.seq_ids.into_data().to_vec::<i32>().unwrap();
        let cea = comp(&eai);
        let parity_match = ca1.iter().zip(cea.iter()).filter(|(x, y)| x == y).count();
        let parity_frac = parity_match as f64 / ca1.len() as f64;
        let parity_bit_identical = ca1 == cea;
        // AUTOREGRESSIVE-CONSISTENCY complement: the captured completion is a genuine CHAIN, not one
        // token stamped at advancing columns — count rows whose completion has >1 distinct token (so
        // token t demonstrably fed a different token at t+1 somewhere in the row).
        let chained_rows = (0..n)
            .filter(|&r| {
                let row: std::collections::HashSet<i32> = (lp..lp + max_new)
                    .map(|c| a1i[r * (lp + max_new) + c])
                    .collect();
                row.len() > 1
            })
            .count();

        println!(
            "  [TEMP] vocab={vocab} N={n} max_new={max_new} temp={} (captured Gumbel-max, fresh seed per replay)",
            rc.temperature
        );
        println!(
            "  [TEMP] (high temp so the Gumbel noise dominates the UNCALIBRATED random-weight logits and the"
        );
        println!(
            "  [TEMP]  seed visibly controls the captured decode — the diagnostics above prove the captured"
        );
        println!(
            "  [TEMP]  seeded sampler decorrelates even at flat logits; a trained model needs no such boost.)"
        );
        println!(
            "  [TEMP] valid token ids: {valid} | distinct completion ids: {} (varied={varied})",
            distinct.len()
        );
        println!(
            "  [TEMP] stream-A vs stream-B differing fraction: {frac_ab:.3} (decorrelated={decorrelated})"
        );
        println!("  [TEMP] stream-A reproducible (A==A): {same_aa}");
        println!(
            "  [TEMP-parity] captured(0xA) vs EAGER(0xA), identical per-step seed stream: {parity_match}/{} tokens match (frac={parity_frac:.3}, bit_identical={parity_bit_identical})",
            ca1.len()
        );
        println!(
            "  [TEMP-parity] autoregressive chain: {chained_rows}/{n} completion rows have >1 distinct token (token t feeds t+1)"
        );
        let pass =
            valid && varied && decorrelated && same_aa && parity_bit_identical && chained_rows > 0;
        println!(
            "  => {}\n",
            if pass {
                "PASS: captured temperature decode DECORRELATES + is AUTOREGRESSIVELY CORRECT (== eager under a fixed seed)"
            } else {
                "FAIL"
            }
        );
        assert!(
            valid,
            "captured temperature produced out-of-range token ids"
        );
        assert!(
            varied,
            "captured temperature froze to a single token (noise not applied)"
        );
        assert!(
            decorrelated,
            "captured temperature did NOT decorrelate across seed streams (frozen noise)"
        );
        assert!(
            same_aa,
            "captured temperature non-reproducible for a fixed seed stream"
        );
        assert!(
            chained_rows > 0,
            "captured temperature is not autoregressive (no row varies across steps — a single token stamped at advancing columns)"
        );
        assert!(
            parity_bit_identical,
            "captured temperature decode DIVERGES from the eager temperature decode under an identical seed stream ({parity_match}/{} match) — the captured autoregressive chain is wrong",
            ca1.len()
        );
    }

    println!("\n=== P-FINAL SUMMARY: CAPTURED greedy decode BIT-IDENTICAL to eager static; ===");
    println!(
        "=== captured TEMPERATURE decode DECORRELATES + matches eager token-for-token under a fixed seed (C3); ==="
    );
    println!(
        "=== guards: FIX-1 OOB-warmup assert + FIX-2b in-place-VA stability assert active in both harnesses. ==="
    );
}
