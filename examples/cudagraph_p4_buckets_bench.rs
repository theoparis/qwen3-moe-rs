//! P4 of the CUDA-graph plan (docs/cudagraph/DESIGN.md §0c gate 5 + §11 R-buckets): prompt-length
//! BUCKETS over a SHARED graph-pool. P-final captures ONE static decode whose `lp` is BAKED into the
//! graph (`pos` starts at `lp`), so a captured graph is valid for exactly ONE prompt length. P4 makes
//! VARIABLE prompt lengths work the way vLLM does:
//!
//!   * **lp BUCKETS** (Part B, this repo). Pick a bucket set (here `{16, 32, 64}`). For a real prompt
//!     of length `L`, LEFT-PAD it to the next bucket `B`, and decode through the captured static graph
//!     for prompt-length `B`. The left-pad columns `[0, B-L)` are masked out of attention by a DEVICE
//!     `lo` counter (`= B - L`), combined with the existing `idx > pos` future mask
//!     (`Qwen3Attention::forward_with_cache_static_pre_lp`). Because RoPE is RELATIVE, shifting every
//!     real/decode position uniformly by `lo` and masking the pad makes the bucketized decode match
//!     the prompt run at its TRUE length `L`. One graph per bucket bakes that bucket's `lp`; `lo`
//!     (rewritten per dispatch, a stable-VA device buffer the graph reads) routes any `L <= B`.
//!
//!   * **SHARED graph-pool** (Part A, cubecl). All buckets capture into ONE `CapturePoolHandle`
//!     (the vLLM `graph_pool_handle` model): serially-replayed graphs SHARE one arena, so the blocks
//!     of the largest bucket (captured FIRST) back the smaller buckets too. K buckets then cost
//!     ~1 bucket's arena high-water, not K× (`ComputeClient::capture_arena_in_pool`). SOUND ONLY for
//!     serial single-stream replay (a block is baked into several graphs).
//!
//! Validates, on the GB10:
//!   (1) PER-BUCKET CORRECTNESS — for buckets {16,32,64}, a CAPTURED greedy decode at the bucket
//!       (pad_len=0) is BIT-IDENTICAL to an eager `group_sample_cached_device_static` at the same
//!       length (seq_ids + completion_mask + logp). Each bucket reuses the P-final capture mechanism.
//!   (2) LEFT-PAD CORRECTNESS — a real prompt of length L < B, left-padded to B and dispatched through
//!       the bucket-B graph, produces the same greedy completion as decoding it at its TRUE length L
//!       UP TO FP/ARGMAX ROBUSTNESS (RoPE is relative, so the uniform `lo`-shift is invariant only in
//!       real arithmetic — in FP the absolute rotation angles differ and a near-tie argmax can flip;
//!       genuinely bit-identical only at pad_len==0). The `lo` mask is load-bearing: with it ON the
//!       completions match; with it OFF (pad keys leak into attention) they diverge.
//!   (3) SHARED-POOL MEMORY — capturing K buckets into ONE shared pool uses ~1 bucket's arena
//!       high-water; capturing them into K separate (pool-of-one) arenas uses ~K×. Reports both.
//!
//! Run (GB10 / aarch64):
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_p4_buckets_bench 2>&1 | tail -30

use burn::tensor::backend::Backend;
use burn::tensor::{Bool, Int, IndexingUpdateOp, Tensor, TensorPrimitive};
use burn_cubecl::CubeBackend;
use cubecl::client::CapturePoolHandle;
use cubecl::cuda::CudaRuntime;
use cubecl::Runtime;
use qwen3_burn::grpo::{group_sample_cached_device_static, RolloutConfig, Rollouts};
use qwen3_burn::ModelCache;
use qwen3_burn::sampling_device::{device_select_tokens, device_token_logp, logsumexp_dim1};
use qwen3_burn::{rope_freqs, Qwen3Config, Qwen3ForCausalLM};

type B = CubeBackend<CudaRuntime, f32, i32, u8>;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

const HEAD_DIM: usize = 128;
const THETA: f64 = 1_000_000.0;

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
}

// VA-stability helpers (see cudagraph_pfinal_bench.rs FIX 2b): read a persistent buffer's TRUE device
// pointer so the harness can assert the captured graph's baked addresses never relocate.
fn float_va<const D: usize>(t: &Tensor<B, D>) -> u64 {
    match t.clone().into_primitive() {
        TensorPrimitive::Float(ct) => ct.client.get_resource(ct.handle.clone().binding()).resource().ptr,
        TensorPrimitive::QFloat(_) => unreachable!("persistent buffers are never quantized"),
    }
}
fn int_va<const D: usize>(t: &Tensor<B, D, Int>) -> u64 {
    let ct = t.clone().into_primitive();
    ct.client.get_resource(ct.handle.clone().binding()).resource().ptr
}

fn build_model(device: &<B as Backend>::Device, vocab: usize, layers: usize) -> Qwen3ForCausalLM<B> {
    Qwen3Config::new()
        .with_vocab_size(vocab)
        .with_hidden_size(1024)
        .with_intermediate_size(3072)
        .with_num_hidden_layers(layers)
        .with_num_attention_heads(16)
        .with_num_key_value_heads(8)
        .with_head_dim(Some(HEAD_DIM))
        .init_causal_lm::<B>(device)
}

/// A captured per-bucket greedy decode graph + its persistent device state, RE-DISPATCHABLE for any
/// real prompt length `L <= bucket`. The captured graph baked the addresses of every buffer below, so
/// each dispatch RESETS them IN PLACE (never reallocates) and rewrites the `lo` left-pad counter,
/// then replays — the P-final capture-once / replay-per-token loop, now reusable across prompts.
///
/// `pad`/`freqs`/`arange_tmax` are KEEPALIVES: the captured graph baked their device addresses, so
/// they must outlive every replay even though Rust sees no further host read of them.
#[allow(dead_code)]
struct BucketGraph {
    graph: cubecl::client::CapturedGraph<CudaRuntime>,
    bucket: usize, // = lp baked into this graph
    p: usize,
    g: usize,
    n: usize,
    max_new: usize,
    v: usize,
    eos0: i64,
    // persistent buffers (device addresses baked into `graph`)
    tok_buf: Option<Tensor<B, 2, Int>>,
    logp_buf: Option<Tensor<B, 2>>,
    mask_buf: Option<Tensor<B, 2>>,
    pos: Option<Tensor<B, 1, Int>>,
    finished: Option<Tensor<B, 2, Int>>,
    last_buf: Option<Tensor<B, 2>>,
    lo: Option<Tensor<B, 1, Int>>, // left-pad column count (= bucket - L); read by the captured mask
    pad: Tensor<B, 2, Int>,
    cache: ModelCache<B>,
    freqs: Tensor<B, 1>,
    arange_tmax: Tensor<B, 1, Int>,
    device: <B as Backend>::Device,
}

impl BucketGraph {
    /// Build the persistent buffers, prefill a CANONICAL full-length-`bucket` prompt (pad_len=0), and
    /// capture ONE greedy decode step into `pool`. `arena_bytes` is the pool's high-water AFTER this
    /// graph (the shared pool grows monotonically as buckets are added).
    #[allow(clippy::too_many_arguments)]
    fn capture(
        model: &Qwen3ForCausalLM<B>,
        pool: &CapturePoolHandle<CudaRuntime>,
        p: usize,
        g: usize,
        bucket: usize,
        max_new: usize,
        v: usize,
        eos: &[i64],
        warmup: usize,
    ) -> (BucketGraph, u64) {
        let device = <B as Backend>::Device::default();
        let client = CudaRuntime::client(&device);
        let n = p * g;
        let total = bucket + max_new;
        let eos0 = eos[0];
        assert!(warmup < max_new, "warmup {warmup} must be < max_new {max_new} (OOB guard)");

        let freqs = rope_freqs::<B>(HEAD_DIM, THETA, &device);
        let arange_tmax = Tensor::<B, 1, Int>::arange(0..total as i64, &device);

        // canonical capture prompt: a deterministic length-`bucket` prompt (pad_len=0). Its decode
        // output is discarded — every dispatch re-prefills and resets — it only has to make the warmup
        // + capture passes run a valid step.
        let cap_ids: Vec<i64> = (0..(p * bucket) as i64).map(|i| (i * 131 + 17) % v as i64).collect();
        let prompt = Tensor::<B, 1, Int>::from_data(cap_ids.as_slice(), &device).reshape([p, bucket]);
        let prompt_rep = prompt.unsqueeze_dim::<3>(1).repeat(&[1, g, 1]).reshape([n, bucket]);

        let mut cache = model.new_cache_with_capacity(total);
        let mut tok_buf = Some(
            Tensor::<B, 2, Int>::zeros([n, total], &device).slice_assign([0..n, 0..bucket], prompt_rep.clone()),
        );
        let mut logp_buf = Some(Tensor::<B, 2>::zeros([n, max_new], &device));
        let mut mask_buf = Some(Tensor::<B, 2>::zeros([n, max_new], &device));
        let mut pos = Some(Tensor::<B, 1, Int>::full([1], bucket as i64, &device));
        let mut finished = Some(Tensor::<B, 2, Int>::zeros([n, 1], &device));
        let pad = Tensor::<B, 2, Int>::full([n, 1], eos0, &device);
        let mut last_buf = Some(Tensor::<B, 2>::zeros([n, v], &device));
        // lo = left-pad column count. Captured at pad_len=0; rewritten per dispatch (stable VA).
        let lo = Some(Tensor::<B, 1, Int>::zeros([1], &device));

        // canonical prefill (pad_len=0, no mask): fill KV cols 0..bucket + initial logits -> last_buf.
        let prefill = |cache: &mut ModelCache<B>, last_buf: &mut Option<Tensor<B, 2>>| {
            let pos0 = Tensor::<B, 1, Int>::arange(0..bucket as i64, &device).unsqueeze_dim::<2>(0).repeat(&[n, 1]);
            let logits = model.forward_with_cache(prompt_rep.clone(), None, pos0, cache);
            let prefill_last = logits.slice([0..n, (bucket - 1)..bucket, 0..v]).reshape([n, v]);
            let lb = last_buf.take().unwrap();
            *last_buf = Some(lb.slice_assign([0..n, 0..v], prefill_last));
        };
        prefill(&mut cache, &mut last_buf);
        block_sync(&client);

        let va_labels = ["tok", "logp", "mask", "pos", "finished", "last", "lo"];
        let va_before = [
            int_va(tok_buf.as_ref().unwrap()),
            float_va(logp_buf.as_ref().unwrap()),
            float_va(mask_buf.as_ref().unwrap()),
            int_va(pos.as_ref().unwrap()),
            int_va(finished.as_ref().unwrap()),
            float_va(last_buf.as_ref().unwrap()),
            int_va(lo.as_ref().unwrap()),
        ];

        // The captured ONE-STEP greedy closure (identical to P-final, plus the `lo` left-pad mask).
        let eos_vec = eos.to_vec();
        let mut step = || {
            let last = last_buf.take().unwrap();
            let lse = logsumexp_dim1(last.clone());
            let sampled = device_select_tokens(&last, 0.0); // greedy argmax
            let fin = finished.take().unwrap();
            let fin_mask = fin.clone().equal_elem(1i64);
            let active = fin.clone().equal_elem(0i64).float();
            let emit = sampled.mask_where(fin_mask, pad.clone());
            let mut is_eos = emit.clone().equal_elem(eos0);
            for &e in &eos_vec[1..] {
                is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
            }
            let logp = device_token_logp(&last, &emit, &lse).reshape([n, 1]);
            let pos_idx = pos.as_ref().unwrap().clone();
            let rel = pos.as_ref().unwrap().clone().sub_scalar(bucket as i64);
            tok_buf = Some(tok_buf.take().unwrap().select_assign(1, pos_idx, emit.clone(), IndexingUpdateOp::Add));
            logp_buf = Some(logp_buf.take().unwrap().select_assign(1, rel.clone(), logp, IndexingUpdateOp::Add));
            mask_buf = Some(mask_buf.take().unwrap().select_assign(1, rel, active, IndexingUpdateOp::Add));
            finished = Some(fin.add(is_eos.int()).clamp(0i64, 1i64));
            // decode next logits at device `pos`, masking the left-pad columns `< lo`.
            let lg = model.forward_with_cache_static_pre_lp(
                emit,
                pos.as_ref().unwrap().clone(),
                &mut cache,
                &freqs,
                &arange_tmax,
                lo.as_ref(),
            );
            let new_last = lg.slice([0..n, 0..1, 0..v]).reshape([n, v]);
            last_buf = Some(last.slice_assign([0..n, 0..v], new_last));
            pos = Some(pos.take().unwrap().add_scalar(1i64));
        };

        let graph = unsafe { client.capture_arena_in_pool(pool, warmup, &mut step) };
        block_sync(&client);
        let arena_bytes = graph.arena_bytes();
        drop(step); // release the closure's mutable borrows so the buffers can move into the struct

        let va_after = [
            int_va(tok_buf.as_ref().unwrap()),
            float_va(logp_buf.as_ref().unwrap()),
            float_va(mask_buf.as_ref().unwrap()),
            int_va(pos.as_ref().unwrap()),
            int_va(finished.as_ref().unwrap()),
            float_va(last_buf.as_ref().unwrap()),
            int_va(lo.as_ref().unwrap()),
        ];
        for (i, (b, a)) in va_before.iter().zip(va_after.iter()).enumerate() {
            assert_eq!(b, a, "VA-STABILITY VIOLATION: persistent buffer '{}' relocated across capture (bucket {bucket})", va_labels[i]);
        }

        let bg = BucketGraph {
            graph, bucket, p, g, n, max_new, v, eos0,
            tok_buf, logp_buf, mask_buf, pos, finished, last_buf, lo,
            pad, cache, freqs, arange_tmax, device,
        };
        (bg, arena_bytes)
    }

    /// Dispatch a real prompt `[p, L]` (L <= bucket) through this bucket's captured graph: left-pad to
    /// `bucket`, set `lo = bucket - L`, re-prefill (eager, with the left-pad attention mask), reset the
    /// persistent buffers IN PLACE, then replay `max_new` times. `lo_on = false` SKIPS the left-pad
    /// mask (sets lo=0) — used to demonstrate the mask is load-bearing.
    fn dispatch(&mut self, model: &Qwen3ForCausalLM<B>, prompt: Tensor<B, 2, Int>, lo_on: bool) -> Rollouts<B> {
        let device = &self.device;
        let client = CudaRuntime::client(device);
        let (p, l) = (prompt.dims()[0], prompt.dims()[1]);
        assert_eq!(p, self.p, "dispatch prompt batch must match the bucket graph");
        assert!(l <= self.bucket, "prompt length {l} exceeds bucket {}", self.bucket);
        let (n, bucket, v, max_new) = (self.n, self.bucket, self.v, self.max_new);
        let pad_len = bucket - l;

        // LEFT-PAD the prompt to `bucket`: [pad x pad_len, real x L]. Pad id = eos0 (masked anyway).
        let prompt_rep_real = prompt.unsqueeze_dim::<3>(1).repeat(&[1, self.g, 1]).reshape([n, l]);
        let prompt_rep = if pad_len > 0 {
            let pad_block = Tensor::<B, 2, Int>::full([n, pad_len], self.eos0, device);
            Tensor::cat(vec![pad_block, prompt_rep_real], 1)
        } else {
            prompt_rep_real
        };

        // left-pad attention mask: false (pad) for cols 0..pad_len, true (real) for pad_len..bucket.
        let mask_row: Vec<bool> = (0..bucket).map(|c| c >= pad_len).collect();
        let mask_flat: Vec<bool> = (0..n).flat_map(|_| mask_row.clone()).collect();
        let attn_mask = Tensor::<B, 1, Bool>::from_data(mask_flat.as_slice(), device).reshape([n, bucket]);

        // set lo = pad_len (or 0 if the mask is disabled), IN PLACE (preserve baked VA).
        let lo_val = if lo_on { pad_len as i64 } else { 0 };
        self.lo = Some(self.lo.take().unwrap().mul_scalar(0).add_scalar(lo_val));

        // re-prefill (eager): fixed-shape forward_with_cache over the padded prompt + left-pad mask +
        // arange(0..bucket) positions. The last real token (col bucket-1) seeds last_buf.
        self.cache.reset_for_replay();
        {
            let pos0 = Tensor::<B, 1, Int>::arange(0..bucket as i64, device).unsqueeze_dim::<2>(0).repeat(&[n, 1]);
            let mask_in = if lo_on && pad_len > 0 { Some(attn_mask) } else { None };
            let logits = model.forward_with_cache(prompt_rep.clone(), mask_in, pos0, &mut self.cache);
            let prefill_last = logits.slice([0..n, (bucket - 1)..bucket, 0..v]).reshape([n, v]);
            let lb = self.last_buf.take().unwrap();
            self.last_buf = Some(lb.slice_assign([0..n, 0..v], prefill_last));
        }

        // reset the decode buffers IN PLACE to post-prefill state (addresses preserved).
        self.tok_buf = Some(self.tok_buf.take().unwrap().mul_scalar(0));
        self.tok_buf = Some(self.tok_buf.take().unwrap().slice_assign([0..n, 0..bucket], prompt_rep));
        self.logp_buf = Some(self.logp_buf.take().unwrap().mul_scalar(0.0));
        self.mask_buf = Some(self.mask_buf.take().unwrap().mul_scalar(0.0));
        self.finished = Some(self.finished.take().unwrap().mul_scalar(0));
        self.pos = Some(self.pos.take().unwrap().mul_scalar(0).add_scalar(bucket as i64));
        block_sync(&client);

        for _ in 0..max_new {
            self.graph.replay();
        }
        block_sync(&client);

        Rollouts {
            seq_ids: self.tok_buf.clone().unwrap(),
            completion_mask: self.mask_buf.clone().unwrap(),
            old_logprobs: self.logp_buf.clone().unwrap(),
            prompt_len: bucket,
            gen_len: max_new,
        }
    }
}

/// completion region [n, max_new] of a Rollouts whose seq_ids is [n, prompt_len + max_new].
fn completion_ids(r: &Rollouts<B>) -> Vec<i32> {
    let n = r.seq_ids.dims()[0];
    let lp = r.prompt_len;
    let mn = r.gen_len;
    let all = r.seq_ids.clone().into_data().to_vec::<i32>().unwrap();
    let mut out = Vec::with_capacity(n * mn);
    for row in 0..n {
        for c in lp..lp + mn {
            out.push(all[row * (lp + mn) + c]);
        }
    }
    out
}

/// FIX 1 regression test (P4 abort-path USE-AFTER-FREE). Capture bucket A into a SHARED pool, seal it,
/// then FORCE a SECOND bucket's capture into the SAME pool to PANIC mid-capture (caught). With the
/// pre-fix abort (`capture_arena_abort`, free-everything) that teardown freed the shared arena — which
/// already held bucket A's baked device blocks — so replaying bucket A afterwards read FREED device
/// memory (`CUDA_ERROR_ILLEGAL_ADDRESS`), and the next pooled capture recycled those StorageIds (silent
/// cross-graph clobber). With the fix (`capture_pool_abort` returns the arena UNMODIFIED whenever a
/// sealed graph still depends on it), bucket A must still replay BIT-IDENTICALLY.
///
/// Run under `compute-sanitizer --tool memcheck <binary>` (env `P4_ABORT_UAF=1`) to prove no UAF.
fn run_abort_uaf() {
    let device: <B as Backend>::Device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (below Fusion)\n");
    println!("=== P4 ABORT-UAF (FIX 1): bucket A survives an aborted later-bucket capture ===\n");

    let (vocab, layers) = (151936usize, 4usize);
    let (p, g) = (4usize, 2usize);
    let max_new = 16usize;
    let warmup = 4usize;
    let eos: Vec<i64> = vec![vocab as i64 - 1];

    <B as Backend>::seed(&device, 7);
    let model = build_model(&device, vocab, layers);

    let pool = client.capture_pool();

    // Bucket A: captured + SEALED into the shared pool (its blocks are now baked into a live graph).
    let bucket_a = 64usize;
    let (mut bg_a, a_bytes) =
        BucketGraph::capture(&model, &pool, p, g, bucket_a, max_new, vocab, &eos, warmup);
    println!("  bucket A (lp={bucket_a}) captured + sealed into shared pool ({} KB)", a_bytes / 1024);

    // Reference completion from bucket A BEFORE the aborted capture.
    let a_ids: Vec<i64> = (0..(p * bucket_a) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect();
    let prompt_a = Tensor::<B, 1, Int>::from_data(a_ids.as_slice(), &device).reshape([p, bucket_a]);
    let ref_ids = bg_a
        .dispatch(&model, prompt_a.clone(), true)
        .seq_ids.clone().into_data().to_vec::<i32>().unwrap();
    block_sync(&client);
    println!("  bucket A reference completion recorded ({} ids)", ref_ids.len());

    // Force a SECOND bucket's capture into the SAME pool to abort: panic on the capture pass (mimics
    // an intern_metadata locked-miss / launch failure inside the capture window). Catch the unwind.
    let bucket_b = 32usize;
    let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // Persistent buffer allocated OUTSIDE the closure (capture-arena contract).
        let mut buf = Some(Tensor::<B, 1>::zeros([bucket_b], &device));
        let mut counter = 0usize;
        let mut step_b = || {
            // Touch the shared arena with an intermediate so the capture is a real pooled capture.
            let cur = buf.take().unwrap();
            buf = Some(cur.add_scalar(1.0f32));
            counter += 1;
            if counter > warmup {
                panic!("DELIBERATE bucket B capture abort (FIX 1 injection)");
            }
        };
        let _g = unsafe { client.capture_arena_in_pool(&pool, warmup, &mut step_b) };
    }));
    assert!(panicked.is_err(), "bucket B capture was expected to PANIC/abort but did not");
    block_sync(&client);
    println!("  bucket B (lp={bucket_b}) capture into the shared pool deliberately ABORTED (panic caught)");

    // CRUCIAL: replay bucket A again. Pre-fix -> illegal address / clobbered ids; with the fix -> identical.
    let after_ids = bg_a
        .dispatch(&model, prompt_a.clone(), true)
        .seq_ids.clone().into_data().to_vec::<i32>().unwrap();
    block_sync(&client);

    let identical = ref_ids == after_ids;
    println!("  bucket A replay AFTER aborted bucket B: bit-identical={identical}");
    assert!(
        identical,
        "bucket A replay DIVERGED after an aborted bucket B capture -> use-after-free / cross-graph clobber"
    );

    drop(bg_a);
    drop(pool);
    block_sync(&client);
    println!("\n  => PASS: bucket A replays bit-identical after an aborted bucket B capture (no UAF)");
}

fn main() {
    // FIX 1 regression mode: prove an earlier sealed bucket survives a later bucket's aborted capture.
    if std::env::var("P4_ABORT_UAF").is_ok() {
        run_abort_uaf();
        return;
    }

    let device: <B as Backend>::Device = Default::default();
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW CubeBackend<CudaRuntime> (below Fusion)\n");
    println!("=== P4: lp BUCKETS over a SHARED graph-pool (vLLM graph_pool_handle) ===\n");

    let (vocab, layers) = (151936usize, 4usize);
    let (p, g) = (4usize, 2usize);
    let n = p * g;
    let max_new = 16usize;
    let buckets = [64usize, 32, 16]; // captured LARGEST-FIRST so the pool is sized to the worst case
    let eos: Vec<i64> = vec![vocab as i64 - 1]; // unreachable -> both run the full length
    let warmup = 4usize;

    <B as Backend>::seed(&device, 7);
    let model = build_model(&device, vocab, layers);
    let rc = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: 0.0, top_p: 1.0, top_k: 0 };

    // -------------------------------------------------------------------------------------------------
    // (1) PER-BUCKET CORRECTNESS + (3) SHARED-POOL: capture all buckets into ONE shared pool.
    // -------------------------------------------------------------------------------------------------
    println!("--- (1) per-bucket: CAPTURED greedy (bucket graph, pad_len=0) == eager static decode ---");
    let pool = client.capture_pool();
    let mut graphs: std::collections::HashMap<usize, BucketGraph> = std::collections::HashMap::new();
    let mut shared_pool_bytes = 0u64;
    let mut all_pass = true;
    for &b in &buckets {
        let (bg, bytes) = BucketGraph::capture(&model, &pool, p, g, b, max_new, vocab, &eos, warmup);
        shared_pool_bytes = bytes; // monotone; final value = whole pool's high-water
        graphs.insert(b, bg);
    }
    // dispatch each bucket at its FULL length (pad_len=0) and compare to the eager static decode.
    for &b in &buckets {
        let ids: Vec<i64> = (0..(p * b) as i64).map(|i| (i * 131 + 17) % vocab as i64).collect();
        let prompt = Tensor::<B, 1, Int>::from_data(ids.as_slice(), &device).reshape([p, b]);

        let eager = group_sample_cached_device_static(&model, prompt.clone(), &rc, &eos);
        let bg = graphs.get_mut(&b).unwrap();
        let captured = bg.dispatch(&model, prompt.clone(), true);

        let ei = eager.seq_ids.clone().into_data().to_vec::<i32>().unwrap();
        let ci = captured.seq_ids.clone().into_data().to_vec::<i32>().unwrap();
        let em = eager.completion_mask.clone().into_data().to_vec::<f32>().unwrap();
        let cm = captured.completion_mask.clone().into_data().to_vec::<f32>().unwrap();
        let el = eager.old_logprobs.clone().into_data().to_vec::<f32>().unwrap();
        let cl = captured.old_logprobs.clone().into_data().to_vec::<f32>().unwrap();
        let ids_eq = ei == ci;
        let mask_eq = em == cm;
        let logp_err = el.iter().zip(cl.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let pass = ids_eq && mask_eq && logp_err == 0.0;
        all_pass &= pass;
        println!(
            "  bucket lp={b:>2} (N={n}, T_max={}): seq_ids={ids_eq} mask={mask_eq} logp_bit_identical={} (max_err={logp_err:.1e}) => {}",
            b + max_new, logp_err == 0.0, if pass { "PASS" } else { "FAIL" }
        );
        assert!(pass, "bucket {b} captured decode diverged from eager static");
    }
    println!("  => {}\n", if all_pass { "PASS: every bucket's captured decode is bit-identical to eager static" } else { "FAIL" });

    // -------------------------------------------------------------------------------------------------
    // (2) LEFT-PAD CORRECTNESS: a real length-L prompt, left-padded to bucket B, must MATCH the decode
    //     at its TRUE length L *up to FP/argmax robustness* (NOT bit-identical when pad_len>0). RoPE is
    //     relative, so the uniform `lo`-shift is invariant only in REAL arithmetic; in FP, q/k are
    //     rotated by the ABSOLUTE angles `(lo+a)` vs `a`, so a left-padded decode (pad_len>0) can differ
    //     in the low bits and, at a near-tie logit, flip the greedy argmax and then autoregressively
    //     diverge. Only pad_len==0 (L==B) is genuinely bit-identical. The gate is therefore robustness-
    //     based (`on_frac>0.98 && on>off+0.05`), not equality; the `lo` mask is load-bearing (lo ON vs OFF).
    // -------------------------------------------------------------------------------------------------
    println!("--- (2) left-pad: real length L -> bucket B (lo mask) ~= decode at true length L (FP/argmax-robust) ---");
    let cases: [(usize, usize); 3] = [(20, 32), (10, 16), (50, 64)]; // (L, bucket B) with L != B
    let mut leftpad_pass = true;
    for (l, b) in cases {
        assert!(buckets.contains(&b));
        let ids: Vec<i64> = (0..(p * l) as i64).map(|i| (i * 97 + 5) % vocab as i64).collect();
        let prompt = Tensor::<B, 1, Int>::from_data(ids.as_slice(), &device).reshape([p, l]);

        // reference: decode at the TRUE length L (eager static, no padding).
        let rc_l = RolloutConfig { group_size: g, max_new_tokens: max_new, temperature: 0.0, top_p: 1.0, top_k: 0 };
        let truth = group_sample_cached_device_static(&model, prompt.clone(), &rc_l, &eos);
        let truth_c = completion_ids(&truth);

        let bg = graphs.get_mut(&b).unwrap();
        // lo ON: left-pad masked -> should match.
        let padded_on = bg.dispatch(&model, prompt.clone(), true);
        let on_c = completion_ids(&padded_on);
        // lo OFF: pad keys leak into attention -> should diverge.
        let padded_off = bg.dispatch(&model, prompt.clone(), false);
        let off_c = completion_ids(&padded_off);

        let on_match = on_c.iter().zip(truth_c.iter()).filter(|(a, b)| a == b).count();
        let off_match = off_c.iter().zip(truth_c.iter()).filter(|(a, b)| a == b).count();
        let total = truth_c.len();
        let on_frac = on_match as f64 / total as f64;
        let off_frac = off_match as f64 / total as f64;
        // `bit_identical` here is only EXPECTED at pad_len==0; with pad_len>0 the FP absolute-angle RoPE
        // difference can flip a near-tie argmax (see the section comment), so a `false` is not a failure
        // — the gate below is robustness-based, NOT equality. Printed only as an informational signal.
        let bit_identical = on_c == truth_c;
        let pass = on_frac > 0.98 && on_frac > off_frac + 0.05;
        leftpad_pass &= pass;
        println!(
            "  L={l:>2} -> B={b:>2} (pad_len={:>2}): lo-ON match {on_match}/{total} ({on_frac:.3}; bit-identical only at pad_len=0, here={bit_identical}) | lo-OFF match {off_frac:.3} => {}",
            b - l, if pass { "PASS" } else { "FAIL" }
        );
        assert!(pass, "left-pad to bucket {b} did not match the true-length-{l} completion up to FP/argmax robustness (on={on_frac:.3} off={off_frac:.3})");
    }
    println!("  => {}\n", if leftpad_pass { "PASS: left-pad to a bucket matches the true-length decode up to FP/argmax robustness (bit-identical only at pad_len=0); the lo mask is load-bearing" } else { "FAIL" });

    // -------------------------------------------------------------------------------------------------
    // (3) SHARED-POOL MEMORY: K-in-one-pool vs K-separate (pool-of-one). Report the arena high-water.
    // -------------------------------------------------------------------------------------------------
    println!("--- (3) shared-pool memory: K buckets in 1 pool vs K separate arenas ---");
    // K separate: capture each bucket into its OWN (single-graph) arena, sum the high-water.
    let mut separate_total = 0u64;
    let mut separate_each = Vec::new();
    {
        // a throwaway pool-of-one per bucket via capture_arena (single-graph path).
        for &b in &buckets {
            let sep_pool = client.capture_pool(); // fresh pool, ONE graph -> pool-of-one
            let (bg, bytes) = BucketGraph::capture(&model, &sep_pool, p, g, b, max_new, vocab, &eos, warmup);
            separate_each.push((b, bytes));
            separate_total += bytes;
            drop(bg);
            drop(sep_pool);
        }
    }
    println!("  K separate pools (one graph each):");
    for (b, bytes) in &separate_each {
        println!("    bucket lp={b:>2}: {:>8} KB", bytes / 1024);
    }
    println!("    sum (K separate):            {:>8} KB", separate_total / 1024);
    println!("  K-in-1 shared pool high-water: {:>8} KB", shared_pool_bytes / 1024);
    // The shared pool is sized by the LARGEST bucket (captured first); smaller buckets reuse its
    // blocks, adding only their distinct (tiny) metadata. So "~1 bucket" means ~the largest bucket.
    let largest = separate_each.iter().map(|(_, b)| *b).max().unwrap_or(1).max(1);
    let vs_largest = shared_pool_bytes as f64 / largest as f64;
    let saving = separate_total as f64 / shared_pool_bytes.max(1) as f64;
    println!(
        "  shared / largest-bucket = {vs_largest:.3}x (target ~1x: smaller buckets reuse the largest's blocks)   |   separate / shared = {saving:.2}x saving (graph_pool_handle; ideal-cap {}x)",
        buckets.len()
    );
    let mem_pass = vs_largest < 1.10 && saving > 1.8;
    println!("  => {}\n", if mem_pass { "PASS: K buckets in 1 pool ~= 1 (largest) bucket's arena, not K x" } else { "FAIL (see numbers)" });

    // keep the pool + its graphs alive until here (they back the dispatches above).
    drop(graphs);
    drop(pool);
    block_sync(&client);

    println!("=== P4 SUMMARY ===");
    println!("  (1) per-bucket capture bit-identical to eager static: {}", if all_pass { "PASS" } else { "FAIL" });
    println!("  (2) left-pad to a bucket preserves the completion:    {}", if leftpad_pass { "PASS" } else { "FAIL" });
    println!("  (3) K buckets share ~1 bucket's arena (shared pool):  {}", if mem_pass { "PASS" } else { "FAIL" });
    assert!(all_pass && leftpad_pass && mem_pass, "P4 validation failed");
}
