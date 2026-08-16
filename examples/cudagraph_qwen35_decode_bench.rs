//! M-D T6: CUDA-graph captured Qwen3.6-35B-A3B static decode driver.
//!
//! Build/check:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo check --example cudagraph_qwen35_decode_bench --features cuda
//!
//! Run on a CUDA host:
//!   QUANT=fp8 RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cudagraph_qwen35_decode_bench

use std::path::PathBuf;
use std::time::Instant;

use burn::{
    backend::cuda::CudaDevice,
    prelude::Device,
    tensor::{Bool, DType, Int, Tensor},
};
use cubecl::{Runtime, cuda::CudaRuntime};
use qwen3_burn::{
    Precision, Qwen3_5HybridCache, Qwen3_5HybridLayerCache, Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM, Qwen3Tokenizer,
    capture::{
        CaptureBackend, Qwen35DecodeState, Qwen35VaSnapshot, assert_no_new_allocs, int_va,
        memory_usage_snapshot, scatter_emit_to_tok, write_last_in_place,
    },
    linear3,
    qwen3_5::{Qwen3_5DecoderLayer, Qwen3_5DenseQuantBackend},
    rope_freqs,
};

type B = CaptureBackend;
type Client = cubecl::client::ComputeClient<CudaRuntime>;

const MODEL_DIR: &str = "models/qwen3.6-35b-a3b";
const NVFP4_MODEL_DIR: &str = "models/qwen3.6-35b-a3b-nvfp4";
const PROMPT1: &str = "The capital of France is";
const PROMPT2: &str = "The largest planet in the solar system is";
const EOS: [i64; 2] = [151643, 151645];
const BATCH: usize = 1;
const MAX_NEW: usize = 64;
const T_MAX: usize = 1024;
const ROTARY_DIM: usize = 64;
const ROPE_THETA: f64 = 10_000_000.0;
const DEFAULT_WARMUP: usize = 8;
const DEFAULT_REPS: usize = 3;

struct DecodeRun {
    ids: Vec<i64>,
    seconds: f64,
}

struct CapturedRun {
    ids: Vec<i64>,
    seconds: f64,
}

fn block_sync(client: &Client) {
    cubecl::future::block_on(client.sync()).expect("sync failed");
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

fn env_usize(name: &str, default: usize) -> Result<usize, String> {
    match std::env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map_err(|e| format!("invalid {name}={value:?}: {e}")),
        Err(_) => Ok(default),
    }
}

fn proc_status_value(label: &str) -> Option<String> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix(label)
            .map(|value| value.trim().to_string())
    })
}

fn print_mem(label: &str) {
    let rss = proc_status_value("VmRSS:").unwrap_or_else(|| "unobservable".to_string());
    let hwm = proc_status_value("VmHWM:").unwrap_or_else(|| "unobservable".to_string());
    println!("{label}: VmRSS={rss}, VmHWM={hwm}");
}

fn positions(start: usize, len: usize, device: &CudaDevice) -> Tensor<2, Int> {
    if len == 1 {
        Tensor::<2, Int>::from_data([[start as i64]], device)
    } else {
        Tensor::<1, Int>::arange(start as i64..(start + len) as i64, device).unsqueeze()
    }
}

fn logits_last<T>(logits: &Tensor<T, 3>, what: &str) -> Result<Tensor<T, 2>, String>
where
    T: Backend,
{
    let [batch, seq, vocab] = logits.dims();
    if batch != 1 {
        return Err(format!("{what} logits expected batch=1, got {batch}"));
    }
    Ok(logits
        .clone()
        .slice([0..batch, (seq - 1)..seq, 0..vocab])
        .reshape([batch, vocab])
        .cast(DType::F32))
}

fn sample_emit_from_last(
    last: &Tensor<2>,
    finished: &Tensor<1, Bool>,
    eos_pad: &Tensor<2, Int>,
    device: &CudaDevice,
) -> (Tensor<2, Int>, Tensor<1, Bool>) {
    let sampled = last.clone().argmax(1);
    let emit = sampled.mask_where(finished.clone().reshape([BATCH, 1]), eos_pad.clone());
    let mut is_eos = Tensor::<2, Int>::zeros([BATCH, 1], device).equal_elem(1i64);
    for &e in &EOS {
        is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
    }
    let new_finished = finished.clone().bool_or(is_eos.reshape([BATCH]));
    (emit, new_finished)
}

fn static_prefill_logits<T>(
    model: &Qwen3_5MoeForCausalLM<T>,
    input_ids: Tensor<T, 2, Int>,
    position_ids: Tensor<T, 2, Int>,
    cache: &mut Qwen3_5HybridCache<T>,
    prec: Precision,
) -> Tensor<T, 3>
where
    T: Backend + Qwen3_5DenseQuantBackend,
{
    let mut hidden_states = model.model.embed_tokens.forward(input_ids).cast(DType::F32);
    for (idx, (layer, layer_cache)) in model
        .model
        .layers
        .iter()
        .zip(cache.layers.iter_mut())
        .enumerate()
    {
        hidden_states = match (layer, layer_cache) {
            (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(cache)) => {
                let hidden_states =
                    layer.forward_prefill_recurrent_static(hidden_states, cache, prec);
                let residual = hidden_states.clone();
                let hidden_states = layer.post_attention_layernorm.forward(hidden_states);
                let hidden_states = layer.mlp.forward(hidden_states, prec);
                residual + hidden_states
            }
            (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(cache)) => {
                layer.forward_decoder_with_cache(hidden_states, position_ids.clone(), cache, prec)
            }
            (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                panic!("Qwen3.5 hybrid cache layer {idx} is Full but model layer is Linear")
            }
            (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                panic!("Qwen3.5 hybrid cache layer {idx} is Linear but model layer is Full")
            }
        };
    }
    linear3(
        &model.lm_head,
        model.model.norm.forward(hidden_states),
        prec,
    )
}

#[allow(clippy::too_many_arguments)]
fn reset_and_prefill(
    model: &Qwen3_5MoeForCausalLM,
    state: &mut Qwen35DecodeState,
    prompt_base: &mut Option<Tensor<1, Int>>,
    prompt_ids: &[i64],
    prec: Precision,
    device: &CudaDevice,
) {
    let lp = prompt_ids.len();
    assert!(
        lp + state.max_new <= state.t_max,
        "prompt_len ({lp}) + max_new ({}) exceeds T_max ({})",
        state.max_new,
        state.t_max
    );

    state.reset_for_replay();
    let input = Tensor::<1, Int>::from_data(prompt_ids, device).unsqueeze();
    let prompt_pos = positions(0, lp, device);
    let logits = static_prefill_logits(model, input, prompt_pos, &mut state.cache, prec);
    let last = logits
        .slice([0..state.batch, (lp - 1)..lp, 0..state.vocab])
        .reshape([state.batch, state.vocab])
        .cast(DType::F32);
    let last_buf = state.last.take().expect("last buffer missing");
    state.last = Some(last_buf.slice_assign([0..state.batch, 0..state.vocab], last));
    state.pos = Some(
        state
            .pos
            .take()
            .expect("pos buffer missing")
            .mul_scalar(0)
            .add_scalar(lp as i64),
    );
    *prompt_base = Some(
        prompt_base
            .take()
            .expect("prompt_base buffer missing")
            .slice_assign([0..1], Tensor::<1, Int>::from_data([lp as i64], device)),
    );
}

#[allow(clippy::too_many_arguments)]
fn eager_static_decode(
    model: &Qwen3_5MoeForCausalLM,
    prompt_ids: &[i64],
    prec: Precision,
    device: &CudaDevice,
    freqs: &Tensor<1>,
    arange_tmax: &Tensor<1, Int>,
    eos_pad: &Tensor<2, Int>,
) -> Result<DecodeRun, String> {
    let prompt_len = prompt_ids.len();
    assert!(prompt_len + MAX_NEW <= T_MAX);

    let input = Tensor::<1, Int>::from_data(prompt_ids, device).unsqueeze();
    let prompt_pos = positions(0, prompt_len, device);
    let mut cache = model.model.new_cache_with_capacity(T_MAX);
    model.init_static_caches(&mut cache, BATCH);
    model
        .preflight_static(&cache, BATCH)
        .map_err(|e| format!("preflight_static failed in eager-static reference: {e}"))?;

    let logits = static_prefill_logits(model, input, prompt_pos, &mut cache, prec);
    let mut last = logits_last(&logits, "static prefill")?;
    let mut pos = Tensor::<1, Int>::full([1], prompt_len as i64, device);
    let mut finished = Tensor::<1, Int>::zeros([BATCH], device).equal_elem(1i64);
    let mut ids = Vec::with_capacity(MAX_NEW);

    let start = Instant::now();
    for _ in 0..MAX_NEW {
        let (emit, new_finished) = sample_emit_from_last(&last, &finished, eos_pad, device);
        finished = new_finished;
        // The backend's Int rep is I32; convert() widens so the readback is rep-agnostic.
        let emit_id = emit
            .clone()
            .into_data()
            .convert::<i64>()
            .to_vec::<i64>()
            .map_err(|e| format!("read reference emit scalar: {e:?}"))?[0];
        ids.push(emit_id);

        let logits = model.forward_decode_static_pre(
            emit,
            pos.clone(),
            &mut cache,
            prec,
            freqs,
            arange_tmax,
        );
        last = logits;
        pos = pos.add_scalar(1i64);
    }

    Ok(DecodeRun {
        ids,
        seconds: start.elapsed().as_secs_f64(),
    })
}

fn emit_ids(state: &Qwen35DecodeState) -> Result<Vec<i64>, String> {
    state
        .emit
        .as_ref()
        .expect("emit buffer missing")
        .clone()
        .into_data()
        .to_vec::<i32>()
        .map_err(|e| format!("read emit buffer: {e:?}"))
        .map(|xs| xs.into_iter().map(|x| x as i64).collect())
}

fn print_token_diff(label: &str, expected: &[i64], got: &[i64]) {
    println!("{label} token mismatch");
    println!("pos\texpected\tcaptured\tmarker");
    for pos in 0..MAX_NEW {
        let e = expected.get(pos).copied();
        let g = got.get(pos).copied();
        let marker = if e == g { "" } else { "<--" };
        println!(
            "{pos}\t{}\t{}\t{marker}",
            e.map(|id| id.to_string())
                .unwrap_or_else(|| "<missing>".to_string()),
            g.map(|id| id.to_string())
                .unwrap_or_else(|| "<missing>".to_string())
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn replay_captured(
    graph: &cubecl::client::CapturedGraph<CudaRuntime>,
    state: &mut Qwen35DecodeState,
    client: &Client,
    n: usize,
    va_check_each: bool,
    alloc_context: &str,
) -> Result<CapturedRun, String> {
    let va = Qwen35VaSnapshot::from_hybrid(state);
    let alloc_before = memory_usage_snapshot(client);
    let start = Instant::now();
    for step in 0..n {
        graph.replay();
        if va_check_each {
            va.assert_unchanged(state, &format!("{alloc_context} step {step}"));
        }
    }
    block_sync(client);
    let seconds = start.elapsed().as_secs_f64();
    va.assert_unchanged(state, alloc_context);
    let alloc_after = memory_usage_snapshot(client);
    assert_no_new_allocs(alloc_before, alloc_after, alloc_context);
    let ids = emit_ids(state)?;
    Ok(CapturedRun { ids, seconds })
}

#[allow(clippy::too_many_arguments)]
fn time_captured_replay(
    graph: &cubecl::client::CapturedGraph<CudaRuntime>,
    model: &Qwen3_5MoeForCausalLM,
    state: &mut Qwen35DecodeState,
    prompt_base: &mut Option<Tensor<1, Int>>,
    prompt_ids: &[i64],
    prec: Precision,
    device: &CudaDevice,
    client: &Client,
    reps: usize,
) -> f64 {
    let mut xs = Vec::with_capacity(reps);
    for _ in 0..reps {
        reset_and_prefill(model, state, prompt_base, prompt_ids, prec, device);
        block_sync(client);
        let start = Instant::now();
        for _ in 0..MAX_NEW {
            graph.replay();
        }
        block_sync(client);
        xs.push(start.elapsed().as_secs_f64() / MAX_NEW as f64);
    }
    median(&xs)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("CRITICAL: G3 qwen35 captured decode failed: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let quant = std::env::var("QUANT").unwrap_or_else(|_| "fp8".to_string());
    // QUANT=nvfp4 loads the official NVIDIA NVFP4 checkpoint via load_nvidia_nvfp4 (raw dispatch);
    // NVFP4_DEQUANT_TO_FP8=1 selects the staged fp8 fallback inside the loader. The captured step is
    // format-agnostic (forward_decode_static_pre dispatches by sidecar; preflight_static accepts nvfp4).
    let quant_mode = if quant.eq_ignore_ascii_case("fp8") {
        "fp8"
    } else if quant.eq_ignore_ascii_case("bf16") {
        "bf16"
    } else if quant.eq_ignore_ascii_case("nvfp4") {
        "nvfp4"
    } else {
        return Err(format!(
            "unsupported QUANT={quant:?}; expected bf16, fp8, or nvfp4"
        ));
    };
    let default_dir = if quant_mode == "nvfp4" {
        NVFP4_MODEL_DIR
    } else {
        MODEL_DIR
    };
    let dir =
        PathBuf::from(std::env::var("QWEN35_DIR").unwrap_or_else(|_| default_dir.to_string()));
    let warmup = env_usize("WARMUP", DEFAULT_WARMUP)?;
    if warmup < 3 || warmup >= MAX_NEW {
        return Err(format!(
            "WARMUP must be in 3..{MAX_NEW} for capture, got {warmup}"
        ));
    }
    let reps = env_usize("REPS", DEFAULT_REPS)?;
    let va_check_each = std::env::var("VA_CHECK")
        .map(|value| value == "1")
        .unwrap_or(false);

    let device = Device::cuda(0);
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | backend=raw CaptureBackend");
    println!("quant mode: {quant_mode}, warmup={warmup}, reps={reps}, VA_CHECK={va_check_each}");
    print_mem("memory at start");

    #[cfg(feature = "cuda")]
    qwen3_burn::qwen3_5::set_qwen35_fused_moe_enabled(true);

    let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))?;
    println!(
        "config: {} layers, hidden {}, vocab {}, experts top-{}/{}",
        cfg.num_hidden_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.num_experts_per_tok,
        cfg.num_experts
    );

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;
    let mut model = cfg.init_causal_lm(&device);

    if quant_mode == "nvfp4" {
        // Official NVIDIA NVFP4 checkpoint: quantized bytes are ingested straight into the sidecars
        // (no separate bf16 load + quantize step). Dense GDN/full-attn projections load as fp8/bf16;
        // experts land in the nvfp4 fused path (covered at T<=16 for the prompt), shared expert chunks.
        let raw = std::env::var("NVFP4_DEQUANT_TO_FP8").ok().as_deref() != Some("1");
        println!(
            "loading NVIDIA NVFP4 checkpoint from {dir:?} (mode={}) ...",
            if raw {
                "raw NVFP4 dispatch"
            } else {
                "staged fp8 fallback"
            }
        );
        let load_start = Instant::now();
        #[cfg(feature = "cuda")]
        model
            .load_nvidia_nvfp4(&dir)
            .map_err(|e| format!("load_nvidia_nvfp4 failed: {e:?}"))?;
        println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());
        print_mem("memory after load");
        println!("memory after nvfp4 load: {:?}", client.memory_usage());
    } else {
        println!("loading sharded BF16 weights from {dir:?} ...");
        let load_start = Instant::now();
        let report = model
            .load_weights_sharded(&dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
        println!(
            "load verify: pass={}, mapped_tensors={}, params={}",
            report.pass(),
            report.mapped_tensors,
            report.param_count
        );
        println!("load time: {:.1}s", load_start.elapsed().as_secs_f64());
        print_mem("memory after load");

        println!("memory before quant: {:?}", client.memory_usage());
        if quant_mode == "fp8" {
            #[cfg(feature = "cuda")]
            {
                qwen3_burn::quant_gate::quantize_dense_fp8(&mut model, &[]);
                qwen3_burn::quant_gate::quantize_experts_fp8(&mut model, &[]);
            }
        }
        println!("memory after quant: {:?}", client.memory_usage());
    }

    let (prompt1_u32, _) = tokenizer.encode_no_pad(PROMPT1)?;
    let prompt1_ids: Vec<i64> = prompt1_u32.iter().map(|&id| id as i64).collect();
    let (prompt2_u32, _) = tokenizer.encode_no_pad(PROMPT2)?;
    let prompt2_ids: Vec<i64> = prompt2_u32.iter().map(|&id| id as i64).collect();
    assert!(prompt1_ids.len() + MAX_NEW <= T_MAX);
    assert!(prompt2_ids.len() + MAX_NEW <= T_MAX);
    println!("prompt1: {PROMPT1:?} ids={prompt1_ids:?}");
    println!("prompt2: {PROMPT2:?} ids={prompt2_ids:?}");
    println!("max_new={MAX_NEW}, T_max={T_MAX}, precision=BF16");

    let prec = Precision::Bf16;
    let mut preflight_cache = model.model.new_cache_with_capacity(T_MAX);
    model.init_static_caches(&mut preflight_cache, BATCH);
    let preflight = model.preflight_static(&preflight_cache, BATCH);
    println!("preflight_static: {preflight:?}");
    if let Err(e) = preflight {
        return Err(format!(
            "preflight_static failed before decode/capture: {e}"
        ));
    }

    let freqs = rope_freqs::<B>(ROTARY_DIM, ROPE_THETA, &device);
    let arange_tmax = Tensor::<1, Int>::arange(0..T_MAX as i64, &device);
    let eos_pad = Tensor::<2, Int>::full([BATCH, 1], EOS[0], &device);

    println!("running eager-static reference for prompt1 ...");
    let eager1 = eager_static_decode(
        &model,
        &prompt1_ids,
        prec,
        &device,
        &freqs,
        &arange_tmax,
        &eos_pad,
    )?;
    let eager_static_tok_s = MAX_NEW as f64 / eager1.seconds;
    println!("eager-static prompt1 ids: {:?}", eager1.ids);
    println!(
        "eager-static prompt1 throughput: {:.3} tok/s ({:.3}s)",
        eager_static_tok_s, eager1.seconds
    );

    println!("building fresh captured state ...");
    let mut cache = model.model.new_cache_with_capacity(T_MAX);
    model.init_static_caches(&mut cache, BATCH);
    model
        .preflight_static(&cache, BATCH)
        .map_err(|e| format!("preflight_static failed for captured state: {e}"))?;
    let mut state = Qwen35DecodeState::new(BATCH, cfg.vocab_size, T_MAX, MAX_NEW, &device, cache);
    let mut prompt_base = Some(Tensor::<1, Int>::zeros([1], &device));

    reset_and_prefill(
        &model,
        &mut state,
        &mut prompt_base,
        &prompt1_ids,
        prec,
        &device,
    );
    block_sync(&client);
    let va_before_capture = Qwen35VaSnapshot::from_hybrid(&state);
    let prompt_base_va = int_va(prompt_base.as_ref().expect("prompt_base missing"));

    println!("capturing one Qwen3.5 static decode step ...");
    let graph = {
        let step = |state: &mut Qwen35DecodeState| {
            let last = state.last.take().expect("last buffer missing");
            let finished = state.finished.take().expect("finished buffer missing");
            let (emit, new_finished) = sample_emit_from_last(&last, &finished, &eos_pad, &device);
            state.finished = Some(finished.slice_assign([0..BATCH], new_finished));

            let pos_abs = state.pos.as_ref().expect("pos buffer missing").clone();
            let emit_pos =
                pos_abs.clone() - prompt_base.as_ref().expect("prompt_base missing").clone();
            state.emit = Some(scatter_emit_to_tok(
                state.emit.take().expect("emit buffer missing"),
                emit_pos,
                emit.clone(),
            ));
            state.tok = Some(
                state
                    .tok
                    .take()
                    .expect("tok buffer missing")
                    .slice_assign([0..BATCH, 0..1], emit),
            );

            let logits = model.forward_decode_static_pre(
                state.tok.as_ref().expect("tok buffer missing").clone(),
                pos_abs,
                &mut state.cache,
                prec,
                &freqs,
                &arange_tmax,
            );
            state.last = Some(write_last_in_place(
                last,
                logits.reshape([BATCH, 1, cfg.vocab_size]),
                BATCH,
                cfg.vocab_size,
            ));
            state.pos = Some(
                state
                    .pos
                    .take()
                    .expect("pos buffer missing")
                    .add_scalar(1i64),
            );
        };
        unsafe { client.capture_arena(warmup, || step(&mut state)) }
    };
    block_sync(&client);
    va_before_capture.assert_unchanged(&state, "after capture build");
    assert_eq!(
        prompt_base_va,
        int_va(prompt_base.as_ref().expect("prompt_base missing")),
        "VA-STABILITY VIOLATION (after capture build): prompt_base relocated"
    );
    println!("capture arena_bytes={}", graph.arena_bytes());

    reset_and_prefill(
        &model,
        &mut state,
        &mut prompt_base,
        &prompt1_ids,
        prec,
        &device,
    );
    block_sync(&client);
    assert_eq!(
        prompt_base_va,
        int_va(prompt_base.as_ref().expect("prompt_base missing")),
        "VA-STABILITY VIOLATION (prompt1 reset): prompt_base relocated"
    );
    println!("replaying captured prompt1 ...");
    let captured1 = replay_captured(
        &graph,
        &mut state,
        &client,
        MAX_NEW,
        va_check_each,
        "prompt1 replay",
    )?;
    println!("captured prompt1 ids: {:?}", captured1.ids);
    if captured1.ids != eager1.ids {
        print_token_diff("G3 FAIL prompt1", &eager1.ids, &captured1.ids);
        std::process::exit(1);
    }

    let cap_sec_per_tok = time_captured_replay(
        &graph,
        &model,
        &mut state,
        &mut prompt_base,
        &prompt1_ids,
        prec,
        &device,
        &client,
        reps,
    );
    let captured_tok_s = 1.0 / cap_sec_per_tok;

    println!("running fresh eager-static reference for prompt2 ...");
    let eager2 = eager_static_decode(
        &model,
        &prompt2_ids,
        prec,
        &device,
        &freqs,
        &arange_tmax,
        &eos_pad,
    )?;

    reset_and_prefill(
        &model,
        &mut state,
        &mut prompt_base,
        &prompt2_ids,
        prec,
        &device,
    );
    block_sync(&client);
    assert_eq!(
        prompt_base_va,
        int_va(prompt_base.as_ref().expect("prompt_base missing")),
        "VA-STABILITY VIOLATION (prompt2 reset): prompt_base relocated"
    );
    println!("replaying captured prompt2 on reset state ...");
    let captured2 = replay_captured(
        &graph,
        &mut state,
        &client,
        MAX_NEW,
        va_check_each,
        "prompt2 replay",
    )?;
    println!("eager-static prompt2 ids: {:?}", eager2.ids);
    println!("captured prompt2 ids: {:?}", captured2.ids);
    if captured2.ids != eager2.ids {
        print_token_diff("G3 FAIL prompt2", &eager2.ids, &captured2.ids);
        std::process::exit(1);
    }

    let speedup = captured_tok_s / eager_static_tok_s;
    println!("===== G3 SUMMARY =====");
    println!("eager-static tok/s: {eager_static_tok_s:.3}");
    println!(
        "captured tok/s: {captured_tok_s:.3} (prompt1 correctness replay {:.3}s, arena_bytes={})",
        captured1.seconds,
        graph.arena_bytes()
    );
    println!("speedup: {speedup:.3}x");
    println!(
        "G3 PASS quant={quant_mode} captured={captured_tok_s:.3} eager_static={eager_static_tok_s:.3} eager=known 4.85 fp8 baseline note"
    );
    Ok(())
}
