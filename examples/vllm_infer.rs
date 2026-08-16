//! vLLM-like inference EXAMPLE for Qwen3-30B-A3B on the FAST FUSED static-decode path (lever (c)).
//!
//! This mirrors vLLM's `LLM(model).generate(prompts, SamplingParams)`: the 30B is loaded ONCE
//! ([`Llm::from_dir`]) and then driven over many prompts with PER-REQUEST sampling
//! (temperature / top_p / top_k / max_tokens). Unlike `examples/moe_generate.rs` (the slow greedy
//! ORACLE that re-runs all 128 experts and grows the KV prefix), every decode step here runs the
//! fixed-shape, host-sync-free **fused gather-GEMV** decode — `MoeStaticDecode::with_fused(true)`
//! (the lever (c) kernel that reads each routed expert's weights ONCE from the persistent stacks,
//! validated token-identical to the oracle in `examples/moe_static_decode.rs`).
//!
//! Per token: PREFILL is eager (the variable-shape prompt), then the DECODE loop is the static path
//!   `forward_with_cache_static_pre(emit[1,1], pos[1], &cache, &sd)`
//! over a device `pos` counter and a STATIC KV cache — exactly `generate_greedy_static`
//! (src/moe.rs:1014) but with real SAMPLING (temperature/top-p/top-k) instead of argmax.
//!
//! Sampling uses the crate's canonical `qwen3_burn::sampling::sample_index` (now `pub mod sampling`);
//! only the trivial `softmax_temp` helper (4 lines, mirrors `grpo/rollout.rs`) is kept local.
//!
//! Build/run (greedy — should match the known static-decode output "... Paris. Which of the ..."):
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example vllm_infer -- \
//!     --dir models/qwen3-30b-a3b --prompt "The capital of France is" --max-tokens 48 --temperature 0
//!
//! Build/run (sampling — coherent, different, on-topic):
//!   RUSTFLAGS="-C target-feature=+fp16" cargo run --release --features cuda --example vllm_infer -- \
//!     --dir models/qwen3-30b-a3b --prompt "The capital of France is" --max-tokens 48 \
//!     --temperature 0.7 --top-p 0.95 --seed 0

use std::path::PathBuf;

use burn::prelude::Device;

use burn::tensor::{DType, Device, Int, Tensor};
use qwen3_burn::capture::{
    CaptureBackend, CapturedDecoder, DecodeState, scatter_emit_to_tok, write_last_in_place,
};
use qwen3_burn::{Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3Tokenizer, sampling::sample_index};

type B = Cuda;
type CapB = CaptureBackend;

/// Qwen3 end-of-text / im_end. Stop decoding (vLLM-style) when one is sampled.
const EOS: [i64; 2] = [151643, 151645];

// ============================================================================================
// SamplingParams — the per-request knobs (mirrors vLLM's `SamplingParams`).
// ============================================================================================
#[derive(Clone, Debug)]
struct SamplingParams {
    max_tokens: usize,
    /// `<= 0` ⇒ greedy (device argmax). `> 0` ⇒ temperature-scaled categorical sampling.
    temperature: f32,
    /// Nucleus mass kept. `>= 1.0` (or `<= 0`) disables.
    top_p: f32,
    /// Keep the `k` highest-prob tokens. `0` disables.
    top_k: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 64,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

// ============================================================================================
// Deterministic RNG — xorshift64* seeded by `--seed`, so sampling is reproducible WITHOUT a
// `rand` dep in the example (we deliberately do not pull one in).
// ============================================================================================
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // SplitMix-style mix so even seed 0 gives a well-distributed nonzero state.
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform in `[0, 1)` from the top 24 bits (exactly representable in f32).
    fn uniform(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

// ============================================================================================
// Host sampler — a faithful copy of src/sampling.rs (private module) + grpo/rollout.rs::softmax_temp.
// ============================================================================================

/// Temperature-scaled softmax over a logit row (`temp <= 0` ⇒ one-hot argmax, i.e. greedy).
fn softmax_temp(row: &[f32], temp: f32) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| ((x - m) / temp).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

// Token selection uses the crate's canonical `qwen3_burn::sampling::sample_index` (top-k → top-p →
// inverse-CDF, HF/vLLM order) — imported above, no longer copied.

// ============================================================================================
// Llm — load the model ONCE, then `generate` per prompt (vLLM's `LLM`).
// ============================================================================================
struct Llm {
    model: Qwen3MoeForCausalLM,
    tokenizer: Qwen3Tokenizer,
    device: CudaDevice,
    vocab: usize,
    eos: Vec<i64>,
}

impl Llm {
    /// Load config + tokenizer + sharded bf16 weights ONCE (the expensive step vLLM amortizes).
    fn from_dir(dir: &PathBuf) -> Result<Self, String> {
        let device = Device::cuda(0);
        let cfg = config_from_hf(dir)?;
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
        let mut model = cfg.init_causal_lm(&device);
        println!("loading sharded weights from {dir:?} ...");
        let t0 = std::time::Instant::now();
        model
            .load_weights_sharded(dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
        println!(
            "loaded {} layers in {:.1}s (load done ONCE, reused for every prompt)",
            model.num_layers(),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            vocab: cfg.vocab_size,
            eos: EOS.to_vec(),
        })
    }

    /// Sample one next-token id from the last-token logits `[1, vocab]`.
    /// Greedy (`temperature <= 0`) stays on-device (argmax → read 1 int). Otherwise pull the row to
    /// host, softmax at temperature, and draw via top-k/top-p inverse-CDF.
    fn sample_token(&self, last: &Tensor<2>, p: &SamplingParams, rng: &mut Rng) -> i64 {
        if p.temperature <= 0.0 {
            last.clone()
                .argmax(1)
                .reshape([1])
                .cast(DType::I64)
                .into_data()
                .as_slice::<i64>()
                .ok()
                .and_then(|s| s.first().copied())
                .unwrap_or(0)
        } else {
            // logits are bf16 on-device — cast to f32 before the host copy (else TypeMismatch).
            let logits: Vec<f32> = last
                .clone()
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("logits to host");
            let probs = softmax_temp(&logits, p.temperature);
            let r = rng.uniform();
            sample_index(&probs, p.top_k, p.top_p, r) as i64
        }
    }

    /// Generate a completion for `prompt`. Returns `(generated_text, n_tokens, tok_per_sec)` where
    /// `tok_per_sec` times ONLY the fused static-decode loop (load + prefill excluded). The decode
    /// loop is exactly `generate_greedy_static`'s body (src/moe.rs:1051-1071) with SAMPLING.
    fn generate(
        &self,
        prompt: &str,
        p: &SamplingParams,
        rng: &mut Rng,
    ) -> Result<(String, usize, f64), String> {
        let device = &self.device;
        let v = self.vocab;

        // ---- tokenize → [1, lp] ----
        let (ids_u32, _) = self.tokenizer.encode_no_pad(prompt)?;
        let prompt_ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
        let lp = prompt_ids.len();
        let input: Tensor<2, Int> =
            Tensor::<1, Int>::from_data(prompt_ids.as_slice(), device).unsqueeze();

        // ---- build the FUSED static-decode path (lever c) + a STATIC KV cache sized to the run ----
        let total = lp + p.max_tokens;
        let sd = self.model.build_static_decode(total).with_fused(true); // cheap: shares the persistent expert stacks
        let mut cache = self.model.model.new_cache_with_capacity(total);

        // ---- PREFILL (eager, the whole prompt) → last-token logits [1, v] ----
        let pos0 = Tensor::<1, Int>::arange(0..lp as i64, device).unsqueeze_dim::<2>(0); // [1, lp]
        let t_prefill = std::time::Instant::now();
        let logits = self
            .model
            .forward_with_cache(input.clone(), None, pos0, &mut cache);
        let mut last = logits.slice([0..1, (lp - 1)..lp, 0..v]).reshape([1, v]);
        let prefill_s = t_prefill.elapsed().as_secs_f64();

        // ---- DECODE loop (the FAST fused static path) — timed in isolation ----
        let mut out_ids: Vec<i64> = Vec::with_capacity(p.max_tokens);
        let mut pos_val = lp as i64; // absolute position of the token we are about to emit
        let t_decode = std::time::Instant::now();
        for i in 0..p.max_tokens {
            let id = self.sample_token(&last, p, rng); // next token, sampled from `last`
            if self.eos.contains(&id) {
                break; // vLLM stops at EOS and does not include it in the text
            }
            out_ids.push(id);
            if i + 1 == p.max_tokens {
                break; // last token wanted — no need to forward once more
            }
            // feed `id` at device `pos` through the FUSED static decode → logits for the next token.
            let emit: Tensor<2, Int> =
                Tensor::<1, Int>::from_data([id].as_slice(), device).reshape([1, 1]);
            let pos = Tensor::<1, Int>::full([1], pos_val, device);
            let lg = self
                .model
                .forward_with_cache_static_pre(emit, pos, &mut cache, &sd); // [1,1,v]
            last = lg.reshape([1, v]);
            pos_val += 1;
        }
        let decode_s = t_decode.elapsed().as_secs_f64();

        let n = out_ids.len();
        let tok_s = if decode_s > 0.0 {
            n as f64 / decode_s
        } else {
            f64::NAN
        };
        let text = self
            .tokenizer
            .decode(&out_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())?;
        eprintln!("    (prefill {prefill_s:.2}s for {lp} prompt tok; decode {decode_s:.2}s)");
        Ok((text, n, tok_s))
    }
}

// ============================================================================================
// CapturedLlm — raw-backend CUDA graph decode for greedy static generation.
// ============================================================================================
struct CapturedLlm {
    model: Qwen3MoeForCausalLM<CapB>,
    tokenizer: Qwen3Tokenizer,
    device: <CapB as Backend>::Device,
    vocab: usize,
    eos: Vec<i64>,
}

impl CapturedLlm {
    /// Load config + tokenizer + sharded bf16 weights on the RAW capture backend (below Fusion).
    fn from_dir(dir: &PathBuf) -> Result<Self, String> {
        let device: <CapB as Backend>::Device = Default::default();
        let cfg = config_from_hf(dir)?;
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
        let mut model = cfg.init_causal_lm::<CapB>(&device);
        println!("loading sharded weights from {dir:?} (RAW backend, below Fusion) ...");
        let t0 = std::time::Instant::now();
        model
            .load_weights_sharded(dir)
            .map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
        println!(
            "loaded {} layers in {:.1}s (load done ONCE, reused for every prompt)",
            model.num_layers(),
            t0.elapsed().as_secs_f64()
        );
        Ok(Self {
            model,
            tokenizer,
            device,
            vocab: cfg.vocab_size,
            eos: EOS.to_vec(),
        })
    }

    /// Greedy generation with the decode step captured/replayed as one CUDA graph.
    fn generate_capture(
        &self,
        prompt: &str,
        p: &SamplingParams,
        warmup: usize,
    ) -> Result<(String, usize, f64), String> {
        if p.max_tokens == 0 {
            return Ok((String::new(), 0, f64::NAN));
        }
        if warmup < 8 {
            return Err(format!(
                "--capture-warmup must be >= 8 for this integration path (got {warmup})"
            ));
        }
        if warmup >= p.max_tokens {
            return Err(format!(
                "--capture-warmup ({warmup}) must be < --max-tokens ({})",
                p.max_tokens
            ));
        }

        let device = &self.device;
        let vocab = self.vocab;

        // ---- tokenize -> [1, lp] ----
        let (ids_u32, _) = self.tokenizer.encode_no_pad(prompt)?;
        let prompt_ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
        let input: Tensor<CapB, 2, Int> =
            Tensor::<CapB, 1, Int>::from_data(prompt_ids.as_slice(), device).unsqueeze();
        let [batch, lp] = input.dims();

        // ---- build FUSED static decode + static KV cache on CaptureBackend ----
        let total = lp + p.max_tokens;
        let sd = self.model.build_static_decode(total).with_fused(true);
        let cache = self.model.model.new_cache_with_capacity(total);

        // ---- prefill (eager, variable-shape): refill KV + restore persistent last buffer in place ----
        let prefill = |state: &mut DecodeState<CapB>| {
            let pos0 = Tensor::<CapB, 1, Int>::arange(0..lp as i64, device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[batch, 1]);
            let logits = self.model.forward_with_cache(
                state.input_ids.clone(),
                None,
                pos0,
                &mut state.cache,
            );
            // The real model's lm_head emits bf16 logits, while DecodeState::last is f32. Widen before
            // slice_assign or captured argmax can read corrupted f32 storage.
            let prefill_last = logits
                .slice([0..batch, (lp - 1)..lp, 0..vocab])
                .reshape([batch, vocab])
                .cast(DType::F32);
            let lb = state.last.take().unwrap();
            state.last = Some(lb.slice_assign([0..batch, 0..vocab], prefill_last));
        };

        // ---- captured one-step closure: device greedy argmax + in-place persistent updates only ----
        let step = |state: &mut DecodeState<CapB>| {
            let last = state.last.take().unwrap();
            let sampled = last.clone().argmax(1);

            let fin = state.finished.take().unwrap();
            let emit = sampled.mask_where(fin.clone().equal_elem(1i64), state.pad.clone());
            let mut is_eos = Tensor::<CapB, 2, Int>::zeros([batch, 1], device).equal_elem(1i64);
            for &e in &self.eos {
                is_eos = is_eos.bool_or(emit.clone().equal_elem(e));
            }

            let pos_idx = state.pos.as_ref().unwrap().clone();
            state.tok = Some(scatter_emit_to_tok(
                state.tok.take().unwrap(),
                pos_idx,
                emit.clone(),
            ));
            state.finished = Some(fin.add(is_eos.int()).clamp(0i64, 1i64));

            let lg = self.model.forward_with_cache_static_pre(
                emit,
                state.pos.as_ref().unwrap().clone(),
                &mut state.cache,
                &sd,
            );
            state.last = Some(write_last_in_place(last, lg, batch, vocab));
            state.pos = Some(state.pos.take().unwrap().add_scalar(1i64));
        };

        let t_build = std::time::Instant::now();
        let mut decoder = CapturedDecoder::build(
            input,
            cache,
            p.max_tokens,
            vocab,
            self.eos.clone(),
            warmup,
            prefill,
            step,
        );
        let build_s = t_build.elapsed().as_secs_f64();
        let arena_mb = decoder.arena_bytes() / (1024 * 1024);

        let t_decode = std::time::Instant::now();
        let ids = decoder.decode_n(p.max_tokens, &self.eos);
        let decode_s = t_decode.elapsed().as_secs_f64();

        let mut out_ids = ids[lp..(lp + p.max_tokens).min(ids.len())].to_vec();
        if let Some(i) = out_ids.iter().position(|id| self.eos.contains(id)) {
            out_ids.truncate(i);
        }
        let n = out_ids.len();
        let tok_s = if decode_s > 0.0 {
            n as f64 / decode_s
        } else {
            f64::NAN
        };
        let text = self
            .tokenizer
            .decode(&out_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())?;
        eprintln!(
            "    (capture build+prefill {build_s:.2}s, arena {arena_mb} MB; replay+read {decode_s:.2}s for {n} tok)"
        );
        Ok((text, n, tok_s))
    }
}

// ============================================================================================
// helpers
// ============================================================================================
fn arg<'a>(a: &'a [String], f: &str) -> Option<&'a String> {
    a.iter().position(|x| x == f).and_then(|i| a.get(i + 1))
}

/// Build a `Qwen3MoeConfig` from a HuggingFace `config.json` (same as examples/moe_static_decode.rs).
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

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

/// Qwen ChatML single-turn wrap for an INSTRUCT model. The `<|im_start|>`/`<|im_end|>` specials are
/// already in the Qwen3 vocab (151644/151645), so the tokenizer encodes them as single tokens.
fn chatml(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = PathBuf::from(
        arg(&args, "--dir")
            .cloned()
            .unwrap_or_else(|| "models/qwen3-30b-a3b".into()),
    );
    let seed: u64 = arg(&args, "--seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    // `--chat`: wrap each prompt in the Qwen ChatML template so an INSTRUCT model (e.g.
    // Qwen3-30B-A3B-Instruct-2507) actually follows it. The <|im_start|>/<|im_end|> specials
    // (151644/151645) are already in the vocab; <|im_end|>=151645 is in EOS so the turn stops cleanly.
    let chat = args.iter().any(|x| x == "--chat");
    let capture = args.iter().any(|x| x == "--capture");
    let capture_warmup: usize = arg(&args, "--capture-warmup")
        .or_else(|| arg(&args, "--warmup"))
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    // `--chat` defaults to the instruct generation_config (temp 0.7 / top_p 0.8 / top_k 20) UNLESS the
    // user overrides — greedy at long --max-tokens tends to loop and never emit EOS (the "won't stop"
    // instruct-test failure flagged by the 3-voice review). Base/raw mode keeps greedy defaults.
    let params = SamplingParams {
        // Default max_tokens: 64 is fine for a base completion, far too short for a chat reply — so
        // --chat defaults to 512 (still bounded; raise with --max-tokens up to the context length).
        // NOT a hard cap: generation runs `0..max_tokens` with an early EOS break.
        max_tokens: arg(&args, "--max-tokens")
            .and_then(|s| s.parse().ok())
            .unwrap_or(if chat { 512 } else { 64 }),
        temperature: arg(&args, "--temperature")
            .and_then(|s| s.parse().ok())
            .unwrap_or(if chat { 0.7 } else { 0.0 }),
        top_p: arg(&args, "--top-p")
            .and_then(|s| s.parse().ok())
            .unwrap_or(if chat { 0.8 } else { 1.0 }),
        top_k: arg(&args, "--top-k")
            .and_then(|s| s.parse().ok())
            .unwrap_or(if chat { 20 } else { 0 }),
    };

    // `--interactive`: a REPL — load once, then read a prompt per line from stdin, generate, repeat
    // (the natural multi-prompt chat UX; blank line or EOF/Ctrl-D quits).
    let interactive = args.iter().any(|x| x == "--interactive" || x == "-i");
    // No --prompt ⇒ a small built-in batch, vLLM-style. Instruction-style under --chat, completion-style else.
    let prompts: Vec<String> = match arg(&args, "--prompt") {
        Some(p) => vec![p.clone()],
        None if chat => vec![
            "In one sentence, explain why the sky is blue.".to_string(),
            "Write a haiku about autumn.".to_string(),
            "List 3 tips for writing clean code.".to_string(),
            "Write a Python function that returns the nth Fibonacci number.".to_string(),
        ],
        None => vec![
            "The capital of France is".to_string(),
            "Here is a haiku about the ocean:".to_string(),
            "In one sentence, explain why the sky is blue.".to_string(),
            "def fibonacci(n):".to_string(),
        ],
    };

    if capture && params.temperature > 0.0 {
        println!(
            "--capture currently supports greedy decode only; falling back to eager sampling path for temperature={}",
            params.temperature
        );
    }
    let use_capture = capture && params.temperature <= 0.0;

    if use_capture {
        let device: <CapB as Backend>::Device = Default::default();
        println!("device: {device:?} | RAW CaptureBackend (below Fusion)");
    } else {
        println!("device: {:?}", Device::cuda(0));
    }
    let mode = if params.temperature <= 0.0 {
        "GREEDY (argmax)".to_string()
    } else {
        format!(
            "SAMPLING temp={} top_p={} top_k={}",
            params.temperature, params.top_p, params.top_k
        )
    };
    println!(
        "sampling: {mode} | max_tokens={} | seed={seed} | decode={}FUSED static (lever c)\n",
        params.max_tokens,
        if use_capture { "CAPTURED " } else { "" }
    );

    // ---- load ONCE, then generate for every prompt ----
    if use_capture {
        let llm = CapturedLlm::from_dir(&dir)?;

        if interactive {
            use std::io::Write;
            println!(
                "\n==================== INTERACTIVE ({}, captured greedy) — blank line / Ctrl-D quits ====================",
                if chat { "chat" } else { "completion" }
            );
            let stdin = std::io::stdin();
            loop {
                print!("\nprompt> ");
                std::io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let prompt = line.trim();
                if prompt.is_empty() {
                    break;
                }
                let encoded = if chat {
                    chatml(prompt)
                } else {
                    prompt.to_string()
                };
                let (text, n, tok_s) = llm.generate_capture(&encoded, &params, capture_warmup)?;
                println!("{text}");
                println!("  ({n} tokens, {tok_s:.2} tok/s, captured greedy)");
            }
            println!("\n(bye)");
            return Ok(());
        }

        println!(
            "\n==================== GENERATIONS ({} prompt{}, captured greedy) ====================",
            prompts.len(),
            if prompts.len() == 1 { "" } else { "s" }
        );
        for (i, prompt) in prompts.iter().enumerate() {
            let encoded = if chat { chatml(prompt) } else { prompt.clone() };
            let (text, n, tok_s) = llm.generate_capture(&encoded, &params, capture_warmup)?;
            println!("\n[{}/{}]", i + 1, prompts.len());
            println!("PROMPT:     {prompt}");
            println!("GENERATION: {text}");
            println!("            ({n} tokens, {tok_s:.2} tok/s, captured greedy)");
        }
        return Ok(());
    }

    let llm = Llm::from_dir(&dir)?;

    // ---- INTERACTIVE REPL: load once, read a prompt per stdin line, generate, repeat ----
    if interactive {
        use std::io::Write;
        println!(
            "\n==================== INTERACTIVE ({}) — blank line / Ctrl-D quits ====================",
            if chat { "chat" } else { "completion" }
        );
        let stdin = std::io::stdin();
        let mut turn = 0u64;
        loop {
            print!("\nprompt> ");
            std::io::stdout().flush().ok();
            let mut line = String::new();
            if stdin.read_line(&mut line).unwrap_or(0) == 0 {
                break; // EOF / Ctrl-D
            }
            let prompt = line.trim();
            if prompt.is_empty() {
                break;
            }
            let mut rng = Rng::new(seed.wrapping_add(turn));
            let encoded = if chat {
                chatml(prompt)
            } else {
                prompt.to_string()
            };
            let (text, n, tok_s) = llm.generate(&encoded, &params, &mut rng)?;
            println!("{text}");
            println!("  ({n} tokens, {tok_s:.2} tok/s)");
            turn += 1;
        }
        println!("\n(bye)");
        return Ok(());
    }

    println!(
        "\n==================== GENERATIONS ({} prompt{}) ====================",
        prompts.len(),
        if prompts.len() == 1 { "" } else { "s" }
    );

    for (i, prompt) in prompts.iter().enumerate() {
        // Independent, reproducible RNG per request (seed + index), like a per-request vLLM seed.
        let mut rng = Rng::new(seed.wrapping_add(i as u64));
        // `--chat` ⇒ encode the ChatML-wrapped prompt (instruct model); else the raw prompt (base).
        let encoded = if chat { chatml(prompt) } else { prompt.clone() };
        let (text, n, tok_s) = llm.generate(&encoded, &params, &mut rng)?;
        println!("\n[{}/{}]", i + 1, prompts.len());
        println!("PROMPT:     {prompt}");
        println!("GENERATION: {text}");
        println!("            ({n} tokens, {tok_s:.2} tok/s)");
    }
    Ok(())
}
