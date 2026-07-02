//! S.3: the engine thread — owns the (!Sync) model, loads once at startup,
//! serves requests FIFO with per-token generators (eager-static v1).
//!
//! Two concrete arms, no generic model abstraction:
//! 30B = build_static_decode fused pattern; 35B = forward_last_logits loop.
//! See docs/SERVE_PLAN.md S.3.
//!
//! # Design invariants (why the shape is what it is)
//!
//! * **One engine thread, model loaded once.** The Burn model is `!Sync` and its
//!   weights are the expensive resource, so it is loaded exactly once per
//!   `MODEL+QUANT` INSIDE a dedicated `std::thread` and never leaves it. All the
//!   axum handlers touch is the [`EngineHandle`] (a cheap `Send`+`Sync` mpsc
//!   sender) — never the model.
//!
//! * **Off-runtime `blocking_send` is legal.** The engine thread is a plain OS
//!   thread, NOT a tokio task, so calling [`tokio::sync::mpsc::Sender::blocking_send`]
//!   / [`tokio::sync::mpsc::Receiver::blocking_recv`] here cannot deadlock the
//!   async runtime (there is no runtime on this thread to block). The bounded
//!   return channel gives us backpressure: a slow SSE client makes the engine
//!   block on `blocking_send` instead of buffering unbounded logits into RAM.
//!
//! * **Fresh cache per request is the cross-request state-bleed guard.** Every
//!   request allocates its own KV cache sized to `prompt_len + max_tokens_effective`
//!   and drops it when done, so no token from request N can ever leak into
//!   request N+1's attention. (Cache-pool reuse is a later efficiency milestone;
//!   correctness first.)
//!
//! * **`max_tokens_effective` is never unbounded.** When the client OMITS
//!   `max_tokens` it is clamped to `T_MAX - prompt_len`; when the client sets it
//!   EXPLICITLY and `prompt_len + max_tokens > T_MAX` the request is a per-request
//!   user error (400), not a silent clamp. Either way a request can neither run
//!   past the process `T_MAX` nor loop forever.
//!
//! * **Cancel == channel closure.** When the SSE stream (or the non-stream
//!   oneshot receiver) is dropped by the handler, the next `blocking_send`
//!   errors (stream) or `is_closed()` reports true (oneshot); either way we stop
//!   decoding immediately and move to the next request. We ALSO check closure
//!   BEFORE tokenize/prefill so dead queued requests are skipped without work.
//!
//! * **Panic policy (best-effort bounded grace).** The per-request body runs
//!   under [`std::panic::catch_unwind`]. The [`EngineRequest`] (and therefore its
//!   reply channel) is MOVED into the caught closure, so a panic drops that reply
//!   channel as the stack unwinds; the handler can then observe the closed channel
//!   and flush a 500. Crucially this is NOT synchronized with the flush:
//!   `std::process::exit(1)` tears down the tokio runtime immediately. So on a
//!   caught panic we log to stderr and sleep a bounded ~500ms grace to give the
//!   runtime a best-effort chance to observe the dropped channel and flush the 500
//!   BEFORE we exit. Best-effort, NOT a guarantee — a slow flush may still be cut
//!   off by the exit.

use std::path::PathBuf;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{DType, Int, Tensor};

use crate::sampling::sample_index;
use crate::{
    Precision, Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3Tokenizer, Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
};

/// The one backend the engine runs on (Fusion-over-CUDA), matching the proven
/// examples (`vllm_infer` for 30B, `qwen35_generate` for 35B). The CUDA-graph
/// `CaptureBackend` path is explicitly OUT of scope for v1 (eager-static only).
type B = Cuda;

// ============================================================================
// Public engine-owned config / request / event types (self-contained: this
// module deliberately imports NOTHING from `serve::api` — S.0 owns that layer
// and may still be a stub).
// ============================================================================

/// Which of the two proven models this process serves. One model per process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WhichModel {
    /// Qwen3-30B-A3B (bf16 only), fused static-decode path.
    Qwen3Moe30b,
    /// Qwen3.6-35B-A3B, `forward_last_logits` path (bf16/fp8/nvfp4).
    Qwen35Moe,
}

/// Weight precision to load/run. 30B supports only [`Quant::Bf16`]; 35B supports all three.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quant {
    Bf16,
    Fp8,
    Nvfp4,
}

/// Startup configuration for the engine thread.
#[derive(Clone, Debug)]
pub struct EngineConfig {
    pub model: WhichModel,
    pub quant: Quant,
    /// Directory holding `config.json`, `tokenizer.json`, weight shards and the
    /// `generation_config.json` we read the EOS LIST from.
    pub model_dir: PathBuf,
    /// Process-wide context limit. A request with `prompt_len >= t_max` (or an
    /// empty prompt, or explicit `max_tokens` that overflows `t_max`) is a
    /// per-request user error; when `max_tokens` is omitted, `max_tokens_effective`
    /// is clamped to `t_max - prompt_len`.
    pub t_max: usize,
    /// Bounded submit-queue depth (single-stream backpressure: a deep queue = dead
    /// connections). Full ⇒ `try_submit` returns `TrySendError::Full` for the 429 path.
    pub queue_depth: usize,
}

impl EngineConfig {
    /// Default process context limit (docs/SERVE_PLAN.md: single T_MAX, default 4096).
    pub const DEFAULT_T_MAX: usize = 4096;
    /// Default bounded submit-queue depth (plan: 2–4; a deep queue = dead connections).
    pub const DEFAULT_QUEUE_DEPTH: usize = 2;

    /// Build a config with the plan's default `t_max` / `queue_depth`.
    pub fn new(model: WhichModel, quant: Quant, model_dir: impl Into<PathBuf>) -> Self {
        Self {
            model,
            quant,
            model_dir: model_dir.into(),
            t_max: Self::DEFAULT_T_MAX,
            queue_depth: Self::DEFAULT_QUEUE_DEPTH,
        }
    }
}

/// Per-request sampling knobs (mirrors vLLM's `SamplingParams`). `temperature <= 0.0`
/// ⇒ greedy device argmax; otherwise host top-k/top-p categorical sampling.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    /// Fully-specified RNG seed — the engine treats this as authoritative and
    /// seed `0` is a legitimate value, NOT a sentinel. Injecting entropy for
    /// requests that OMIT a seed is the S.4 HTTP handler's job (done before
    /// submit); by the time a request reaches the engine, `seed` is exactly what
    /// sampling will use.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        // Greedy defaults; the HTTP layer overrides from the request / model
        // generation_config.json.
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            seed: 0,
        }
    }
}

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// An EOS-list token was sampled (not emitted to the client, vLLM-style).
    Stop,
    /// `max_tokens_effective` reached.
    Length,
}

/// Token accounting for the OpenAI `usage` object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Streaming event sent back to the axum handler (one per generated token, then `Done`).
///
/// `Error` carries a PER-REQUEST user error (e.g. prompt too long) on the streaming
/// path; the non-stream path reports the same class of error via the oneshot
/// [`EngineResult`]'s `Err` arm instead. Either way it is NOT a process-fatal error.
#[derive(Clone, Debug)]
pub enum EngineEvent {
    /// Emitted EXACTLY ONCE, right after tokenize + length-check succeed and
    /// before the first [`EngineEvent::Token`] (S.4 item 3). It hands the HTTP
    /// handler the prompt length up-front so it can build the OpenAI `usage`
    /// object even when it terminates generation EARLY on a stop-string hit — in
    /// that case the handler drops the receiver (engine cancels by design) and
    /// the terminal [`EngineEvent::Done`] never arrives, so `Done`'s `usage` is
    /// unavailable. Streaming path only; the non-stream oneshot path already
    /// carries `usage` inside [`EngineOutput`], so `Start` is a no-op there.
    Start {
        prompt_tokens: usize,
    },
    Token(u32),
    Done {
        finish_reason: FinishReason,
        usage: TokenUsage,
    },
    Error {
        message: String,
    },
}

/// Non-stream success payload (accumulated token ids + finish + usage), sent ONCE.
#[derive(Clone, Debug)]
pub struct EngineOutput {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub usage: TokenUsage,
}

/// A per-request user error (documented: reported to the streaming path as
/// [`EngineEvent::Error`], and to the non-stream path as the `Err` arm here).
#[derive(Clone, Debug)]
pub struct EngineUserError {
    pub message: String,
}

/// The value delivered on the non-stream oneshot channel: `Ok` on success,
/// `Err` for a per-request user error (prompt too long / tokenize failure).
pub type EngineResult = Result<EngineOutput, EngineUserError>;

/// How a request wants its output delivered.
///
/// * `Stream` — a bounded (~4) mpsc; the engine `blocking_send`s each token then a
///   final `Done`. Receiver drop == cancel.
/// * `Oneshot` — the engine accumulates token ids and sends one [`EngineResult`].
///   Receiver drop == cancel.
pub enum EngineReply {
    Stream(tokio::sync::mpsc::Sender<EngineEvent>),
    Oneshot(tokio::sync::oneshot::Sender<EngineResult>),
}

/// A unit of work handed to the engine thread over the bounded submit queue.
pub struct EngineRequest {
    /// Fully-rendered prompt text (the HTTP layer applies the chat template).
    pub prompt: String,
    pub params: SamplingParams,
    /// `None` ⇒ bounded only by `T_MAX - prompt_len`.
    pub max_tokens: Option<usize>,
    pub reply: EngineReply,
}

/// Startup/load failure (fatal to `spawn`). Per-request user errors are NOT here —
/// they flow back through the reply channel (see [`EngineUserError`]).
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("failed to spawn engine thread: {0}")]
    Thread(#[source] std::io::Error),
    #[error("engine startup channel closed before the model reported ready")]
    StartupChannelClosed,
    #[error("model config error: {0}")]
    Config(String),
    #[error("tokenizer load error: {0}")]
    Tokenizer(String),
    #[error("EOS parsing error: {0}")]
    Eos(String),
    #[error("weight load error: {0}")]
    Load(String),
    #[error("unsupported MODEL+QUANT combination: {0}")]
    UnsupportedQuant(String),
}

/// Cheap, `Send`+`Sync` handle the axum handlers keep. Submitting is non-blocking
/// (`try_submit`); the model itself never leaves the engine thread.
pub struct EngineHandle {
    tx: tokio::sync::mpsc::Sender<EngineRequest>,
    model_id: String,
    report: String,
}

impl EngineHandle {
    /// Non-blocking submit for the 429 backpressure path. `Err(TrySendError::Full)`
    /// ⇒ the queue is at `queue_depth`; `Err(TrySendError::Closed)` ⇒ the engine
    /// thread is gone (it `exit(1)`s on panic, so this generally means the process
    /// is on its way down).
    pub fn try_submit(
        &self,
        req: EngineRequest,
    ) -> Result<(), tokio::sync::mpsc::error::TrySendError<EngineRequest>> {
        self.tx.try_send(req)
    }

    /// Model id string for the `/v1/models` response and the startup banner.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// One-line memory/report string captured right after load (banner only).
    pub fn report(&self) -> &str {
        &self.report
    }
}

// ============================================================================
// spawn — build the queue, start the engine thread, block until it reports ready.
// ============================================================================

/// Load the model ONCE on a dedicated thread and return a handle. Blocks until the
/// model is loaded + warmed (or a load error is reported). All heavy work — and the
/// `!Sync` model — stay on the spawned thread.
pub fn spawn(config: EngineConfig) -> Result<EngineHandle, EngineError> {
    let (tx, rx) = tokio::sync::mpsc::channel::<EngineRequest>(config.queue_depth.max(1));
    // Host-only, `Send` startup signal (model stays on the thread; only a small
    // info struct or the error crosses back).
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<StartupInfo, EngineError>>();

    std::thread::Builder::new()
        .name("qwen-engine".into())
        .spawn(move || engine_main(config, rx, ready_tx))
        .map_err(EngineError::Thread)?;

    match ready_rx.recv() {
        Ok(Ok(info)) => Ok(EngineHandle {
            tx,
            model_id: info.model_id,
            report: info.report,
        }),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(EngineError::StartupChannelClosed),
    }
}

/// Small `Send` payload the engine thread returns once the model is ready.
struct StartupInfo {
    model_id: String,
    report: String,
}

// ============================================================================
// engine_main — load, warm, then the FIFO request loop with the panic catch point.
// ============================================================================

fn engine_main(
    config: EngineConfig,
    mut rx: tokio::sync::mpsc::Receiver<EngineRequest>,
    ready_tx: std::sync::mpsc::Sender<Result<StartupInfo, EngineError>>,
) {
    let engine = match Engine::load(&config) {
        Ok(e) => e,
        Err(e) => {
            // Load failed: report to `spawn` and end the thread cleanly.
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    engine.warmup();
    let info = StartupInfo {
        model_id: engine.model_id.clone(),
        report: engine.report.clone(),
    };
    if ready_tx.send(Ok(info)).is_err() {
        // `spawn`'s caller went away before we finished loading — nothing to serve.
        return;
    }
    drop(ready_tx);

    // FIFO loop. Each request body runs under catch_unwind so a per-request panic
    // drops that request's reply channel (→ handler 500) BEFORE we exit(1).
    while let Some(req) = rx.blocking_recv() {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            engine.run_request(req);
        }));
        if let Err(payload) = outcome {
            // By now `req` (moved into the closure) has been dropped during unwind,
            // so its reply channel is closed and the handler CAN flush a 500.
            let msg = payload
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "non-string panic payload".to_string());
            eprintln!(
                "[qwen-engine] FATAL: request handler panicked: {msg}; exiting(1) after grace"
            );
            // exit(1) kills the tokio runtime immediately and is NOT synchronized
            // with the handler's 500 flush. Sleep a bounded, best-effort grace so
            // the runtime can observe the dropped reply channel and flush before we
            // tear the process down. (Best-effort only — see the module panic-policy
            // note; a slow flush may still be cut off.)
            std::thread::sleep(std::time::Duration::from_millis(500));
            std::process::exit(1);
        }
    }
    // All submit senders dropped (server shutting down) — end the thread.
}

// ============================================================================
// Engine — the loaded model + shared per-process state, and the per-request loop.
// ============================================================================

/// Two concrete arms, wrapping the two proven `ForCausalLM` types directly (NO generic
/// model trait — the plan pins this). Boxed to keep the enum small despite the large
/// 35B variant.
enum LoadedModel {
    /// Qwen3-30B-A3B via the fused static-decode path (vllm_infer's non-captured arm).
    Moe30b(Box<Qwen3MoeForCausalLM<B>>),
    /// Qwen3.6-35B-A3B via `forward_last_logits` (qwen35_generate's arm).
    Moe35b {
        model: Box<Qwen3_5MoeForCausalLM<B>>,
        /// Activation precision for the 35B forward (bf16, matching qwen35_generate's default).
        prec: Precision,
    },
}

struct Engine {
    model: LoadedModel,
    tokenizer: Qwen3Tokenizer,
    /// EOS token LIST, read dynamically from generation_config.json (NEVER hardcoded).
    eos: Vec<i64>,
    device: CudaDevice,
    vocab: usize,
    t_max: usize,
    model_id: String,
    report: String,
}

impl Engine {
    /// Load config + tokenizer + EOS list + weights ONCE (reusing the repo's load fns:
    /// `Qwen3MoeForCausalLM::load_weights_sharded` for 30B; for 35B
    /// `Qwen3_5MoeForCausalLM::{load_weights_sharded,load_nvidia_nvfp4}` plus
    /// `quant_gate::{quantize_dense_fp8,quantize_experts_fp8}` for the fp8 arm).
    fn load(config: &EngineConfig) -> Result<Self, EngineError> {
        let device = CudaDevice::default();
        let dir = &config.model_dir;

        let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(EngineError::Tokenizer)?;
        let eos = load_eos_list(dir, &tokenizer)?;

        let (model, vocab, model_id) = match config.model {
            WhichModel::Qwen3Moe30b => {
                if config.quant != Quant::Bf16 {
                    return Err(EngineError::UnsupportedQuant(format!(
                        "Qwen3-30B-A3B supports only bf16 (got {:?})",
                        config.quant
                    )));
                }
                let cfg = load_moe30b_config(dir)?;
                let mut model = cfg.init_causal_lm::<B>(&device);
                model
                    .load_weights_sharded(dir.clone())
                    .map_err(|e| EngineError::Load(format!("30B sharded load: {e:?}")))?;
                (
                    LoadedModel::Moe30b(Box::new(model)),
                    cfg.vocab_size,
                    "qwen3-30b-a3b".to_string(),
                )
            }
            WhichModel::Qwen35Moe => {
                let cfg = Qwen3_5MoeConfig::from_hf_config_file(dir.join("config.json"))
                    .map_err(EngineError::Config)?;
                let mut model = cfg.init_causal_lm::<B>(&device);
                match config.quant {
                    Quant::Bf16 => {
                        model
                            .load_weights_sharded(dir)
                            .map_err(|e| EngineError::Load(format!("35B sharded load: {e:?}")))?;
                    }
                    Quant::Fp8 => {
                        model
                            .load_weights_sharded(dir)
                            .map_err(|e| EngineError::Load(format!("35B sharded load: {e:?}")))?;
                        crate::quant_gate::quantize_dense_fp8(&mut model, &[]);
                        crate::quant_gate::quantize_experts_fp8(&mut model, &[]);
                    }
                    Quant::Nvfp4 => {
                        model
                            .load_nvidia_nvfp4(dir)
                            .map_err(|e| EngineError::Load(format!("35B nvfp4 load: {e:?}")))?;
                    }
                }
                // Prefer the fused MoE decode kernel (qwen35_generate's default).
                crate::qwen3_5::set_qwen35_fused_moe_enabled(true);
                let quant_tag = match config.quant {
                    Quant::Bf16 => "bf16",
                    Quant::Fp8 => "fp8",
                    Quant::Nvfp4 => "nvfp4",
                };
                (
                    LoadedModel::Moe35b {
                        model: Box::new(model),
                        // 35B runs bf16 activations regardless of weight quant (qwen35_generate default).
                        prec: Precision::Bf16,
                    },
                    cfg.vocab_size,
                    format!("qwen3.6-35b-a3b-{quant_tag}"),
                )
            }
        };

        let report = build_report(&device, &model_id, config.t_max);

        Ok(Self {
            model,
            tokenizer,
            eos,
            device,
            vocab,
            t_max: config.t_max,
            model_id,
            report,
        })
    }

    /// Run one small forward after load so the first real request doesn't pay the
    /// kernel-compile/allocation warm cost. Output discarded.
    fn warmup(&self) {
        let device = &self.device;
        let vocab = self.vocab;
        match &self.model {
            LoadedModel::Moe30b(m) => {
                let sd = m.build_static_decode(2).with_fused(true);
                let mut cache = m.model.new_cache_with_capacity(2);
                let input = Tensor::<B, 1, Int>::from_data([0i64].as_slice(), device).unsqueeze();
                let pos0 = Tensor::<B, 1, Int>::arange(0..1, device).unsqueeze_dim::<2>(0);
                let _ = m.forward_with_cache(input, None, pos0, &mut cache);
                let emit = Tensor::<B, 1, Int>::from_data([0i64].as_slice(), device).reshape([1, 1]);
                let pos = Tensor::<B, 1, Int>::full([1], 1i64, device);
                let lg = m.forward_with_cache_static_pre(emit, pos, &mut cache, &sd);
                // Force a host sync so the warm kernels actually run before we report ready.
                let _ = lg.reshape([1, vocab]).into_data();
            }
            LoadedModel::Moe35b { model, prec } => {
                let mut cache = model.model.new_cache_with_capacity(2);
                let input = Tensor::<B, 2, Int>::from_data([[0i64]], device);
                let pos0 = Tensor::<B, 2, Int>::from_data([[0i64]], device);
                let _ = model.forward_last_logits(input, pos0, &mut cache, *prec);
                let tok = Tensor::<B, 2, Int>::from_data([[0i64]], device);
                let pos = Tensor::<B, 2, Int>::from_data([[1i64]], device);
                let lg = model.forward_last_logits(tok, pos, &mut cache, *prec);
                let _ = lg.into_data();
            }
        }
    }

    /// Serve ONE request end-to-end (tokenize → length check → fresh cache → eager
    /// prefill → per-token decode loop). Runs under the engine loop's `catch_unwind`.
    fn run_request(&self, req: EngineRequest) {
        let EngineRequest {
            prompt,
            params,
            max_tokens,
            reply,
        } = req;
        let mut sink = ReplySink::new(reply);

        // Skip dead queued requests BEFORE doing any tokenize/prefill work.
        if sink.is_closed() {
            return;
        }

        // ---- tokenize ----
        let ids = match self.tokenizer.encode_no_pad(&prompt) {
            Ok((ids, _)) => ids,
            Err(e) => {
                sink.user_error(format!("tokenize failed: {e}"));
                return;
            }
        };
        let prompt_len = ids.len();

        // ---- length check ----
        if prompt_len >= self.t_max {
            sink.user_error(format!(
                "prompt length {prompt_len} >= context limit {} (t_max)",
                self.t_max
            ));
            return;
        }
        // Empty prompt guard: an empty /v1/completions prompt tokenizes to zero
        // ids, and the 30B arm's `(prompt_len - 1)` slice would underflow → panic →
        // the panic policy exits the process. Reject as a per-request user error.
        if prompt_len == 0 {
            sink.user_error("prompt tokenized to zero tokens (empty prompt)".to_string());
            return;
        }

        // Length policy (plan S.3), two cases:
        //   * max_tokens EXPLICITLY set: `prompt_len + max_tokens` must fit T_MAX,
        //     else it is a per-request user error (the "else 400" path). We do NOT
        //     silently clamp a value the client explicitly asked for.
        //   * max_tokens OMITTED: clamp to the remaining context
        //     (`max_eff = t_max - prompt_len`); NEVER unbounded.
        let max_eff = match max_tokens {
            Some(n) => {
                // checked_add guards a pathological huge max_tokens from overflowing.
                let overflows = prompt_len.checked_add(n).map_or(true, |sum| sum > self.t_max);
                if overflows {
                    sink.user_error(format!(
                        "prompt length {prompt_len} + max_tokens {n} exceeds context limit {} (t_max)",
                        self.t_max
                    ));
                    return;
                }
                n
            }
            None => self.t_max - prompt_len,
        };

        // S.4 item 3: hand the streaming handler the prompt length ONCE, before any
        // token, so it can build `usage` if it stops early on a stop-string hit
        // (the terminal `Done` never arrives in that case). No-op for oneshot.
        sink.start(prompt_len);

        let prompt_ids: Vec<i64> = ids.iter().map(|&x| x as i64).collect();
        // Fresh per-request cache capacity = prompt + generated (state-bleed guard).
        let total = (prompt_len + max_eff).max(1);

        let mut rng = Rng::new(params.seed);
        let mut completion: usize = 0;

        // Dispatch to the concrete arm. The sampling / EOS / max / sink logic lives in
        // `step_token` (shared); each arm supplies only its prefill + one-token forward.
        let finish: Option<FinishReason> = match &self.model {
            LoadedModel::Moe30b(m) => {
                let vocab = self.vocab;
                let device = &self.device;
                let sd = m.build_static_decode(total).with_fused(true);
                let mut cache = m.model.new_cache_with_capacity(total);

                // ---- eager prefill (variable-shape prompt) → last-token logits [1, v] ----
                let input =
                    Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), device).unsqueeze();
                let pos0 =
                    Tensor::<B, 1, Int>::arange(0..prompt_len as i64, device).unsqueeze_dim::<2>(0);
                let logits = m.forward_with_cache(input, None, pos0, &mut cache);
                let mut last = logits
                    .slice([0..1, (prompt_len - 1)..prompt_len, 0..vocab])
                    .reshape([1, vocab]);

                let mut pos_val = prompt_len as i64;
                loop {
                    match self.step_token(&last, &params, &mut rng, &mut completion, max_eff, &mut sink)
                    {
                        StepOutcome::Feed(id) => {
                            // FUSED static decode of `id` at device `pos` → next logits [1,1,v].
                            let emit = Tensor::<B, 1, Int>::from_data([id].as_slice(), device)
                                .reshape([1, 1]);
                            let pos = Tensor::<B, 1, Int>::full([1], pos_val, device);
                            let lg = m.forward_with_cache_static_pre(emit, pos, &mut cache, &sd);
                            last = lg.reshape([1, vocab]);
                            pos_val += 1;
                        }
                        StepOutcome::Finish(r) => break Some(r),
                        StepOutcome::Cancelled => break None,
                    }
                }
            }
            LoadedModel::Moe35b { model, prec } => {
                let device = &self.device;
                let prec = *prec;
                let mut cache = model.model.new_cache_with_capacity(total);

                // ---- eager prefill → last-token logits [1, v] ----
                let input =
                    Tensor::<B, 1, Int>::from_data(prompt_ids.as_slice(), device).unsqueeze();
                let pos0 =
                    Tensor::<B, 1, Int>::arange(0..prompt_len as i64, device).unsqueeze();
                let mut last = model.forward_last_logits(input, pos0, &mut cache, prec);

                let mut pos_val = prompt_len as i64;
                loop {
                    match self.step_token(&last, &params, &mut rng, &mut completion, max_eff, &mut sink)
                    {
                        StepOutcome::Feed(id) => {
                            let tok = Tensor::<B, 2, Int>::from_data([[id]], device);
                            let pos = Tensor::<B, 2, Int>::from_data([[pos_val]], device);
                            last = model.forward_last_logits(tok, pos, &mut cache, prec);
                            pos_val += 1;
                        }
                        StepOutcome::Finish(r) => break Some(r),
                        StepOutcome::Cancelled => break None,
                    }
                }
            }
        };

        // Cancelled (client gone) ⇒ drop the sink silently; otherwise emit the terminal
        // Done / oneshot result.
        if let Some(reason) = finish {
            let usage = TokenUsage {
                prompt_tokens: prompt_len,
                completion_tokens: completion,
            };
            sink.finish(reason, usage);
        }
    }

    /// Sample the next token from `last` logits `[1, vocab]`, apply the EOS-list and
    /// `max_eff` stops, and push an emitted token to the sink. Shared by both arms so
    /// the greedy and sampled paths run IDENTICAL prompt/stop handling.
    fn step_token(
        &self,
        last: &Tensor<B, 2>,
        params: &SamplingParams,
        rng: &mut Rng,
        completion: &mut usize,
        max_eff: usize,
        sink: &mut ReplySink,
    ) -> StepOutcome {
        // Guards the max_tokens==0 request and is a cheap safety net.
        if *completion >= max_eff {
            return StepOutcome::Finish(FinishReason::Length);
        }
        let id = self.sample(last, params, rng);
        // EOS is a Stop and (vLLM-style) is NOT pushed/emitted to the client — but
        // OpenAI COUNTS the stop token in usage.completion_tokens, so we increment
        // the completion counter here before finishing. Counted, not emitted: the
        // token never reaches `push_token`, yet `completion` reflects it.
        if self.eos.contains(&id) {
            *completion += 1;
            return StepOutcome::Finish(FinishReason::Stop);
        }
        // Emit — a send error means the receiver was dropped ⇒ CANCEL.
        if sink.push_token(id as u32).is_err() {
            return StepOutcome::Cancelled;
        }
        *completion += 1;
        // Reached the clamp exactly: stop WITHOUT a wasted extra forward.
        if *completion >= max_eff {
            return StepOutcome::Finish(FinishReason::Length);
        }
        StepOutcome::Feed(id)
    }

    /// Greedy (device argmax) when `temperature <= 0`, else host top-k/top-p categorical
    /// sampling via [`crate::sampling::sample_index`] (identical to vllm_infer's sampler).
    fn sample(&self, last: &Tensor<B, 2>, p: &SamplingParams, rng: &mut Rng) -> i64 {
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
            // logits may be bf16 on-device — cast to f32 before the host copy.
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
}

/// The result of one decode step, driving the per-arm loop.
enum StepOutcome {
    /// Emitted `id`; feed it back through the model to get the next logits.
    Feed(i64),
    /// Generation is complete for this reason.
    Finish(FinishReason),
    /// The client's reply channel is gone — stop and move on (no terminal message).
    Cancelled,
}

// ============================================================================
// ReplySink — unifies the streaming (per-token) and non-stream (accumulate) paths.
// ============================================================================

/// Internal working wrapper over [`EngineReply`]. For streaming it forwards each token
/// via `blocking_send` (bounded backpressure); for non-stream it accumulates ids and
/// sends one [`EngineResult`] at the end.
enum ReplySink {
    Stream(tokio::sync::mpsc::Sender<EngineEvent>),
    Oneshot {
        tx: Option<tokio::sync::oneshot::Sender<EngineResult>>,
        acc: Vec<u32>,
    },
}

impl ReplySink {
    fn new(reply: EngineReply) -> Self {
        match reply {
            EngineReply::Stream(tx) => ReplySink::Stream(tx),
            EngineReply::Oneshot(tx) => ReplySink::Oneshot {
                tx: Some(tx),
                acc: Vec::new(),
            },
        }
    }

    /// Emit the one-shot [`EngineEvent::Start`] on the streaming path (S.4 item 3).
    /// A send error just means the client already went away — harmless here; the
    /// subsequent decode loop's `push_token` will observe the closed channel and
    /// cancel. The oneshot path is a deliberate no-op: its [`EngineOutput`] already
    /// carries `usage`, so `Start` would be redundant.
    fn start(&self, prompt_tokens: usize) {
        if let ReplySink::Stream(tx) = self {
            let _ = tx.blocking_send(EngineEvent::Start { prompt_tokens });
        }
    }

    /// Cheap pre-run / periodic liveness check (skip dead queued requests).
    fn is_closed(&self) -> bool {
        match self {
            ReplySink::Stream(tx) => tx.is_closed(),
            ReplySink::Oneshot { tx, .. } => tx.as_ref().map_or(true, |t| t.is_closed()),
        }
    }

    /// Deliver one token. `Err(())` ⇒ the receiver was dropped (cancel).
    fn push_token(&mut self, id: u32) -> Result<(), ()> {
        match self {
            // Legal off-runtime: this is a plain OS thread; blocking here just applies
            // backpressure to a slow client (bounded channel) — it cannot stall tokio.
            ReplySink::Stream(tx) => tx
                .blocking_send(EngineEvent::Token(id))
                .map_err(|_| ()),
            ReplySink::Oneshot { tx, acc } => {
                if tx.as_ref().map_or(true, |t| t.is_closed()) {
                    return Err(());
                }
                acc.push(id);
                Ok(())
            }
        }
    }

    /// Per-request user error (prompt too long / tokenize failure). NOT process-fatal.
    fn user_error(self, message: String) {
        match self {
            ReplySink::Stream(tx) => {
                let _ = tx.blocking_send(EngineEvent::Error { message });
            }
            ReplySink::Oneshot { tx, .. } => {
                if let Some(tx) = tx {
                    let _ = tx.send(Err(EngineUserError { message }));
                }
            }
        }
    }

    /// Terminal success: `Done` on the stream, or the accumulated `EngineOutput` oneshot.
    fn finish(self, finish_reason: FinishReason, usage: TokenUsage) {
        match self {
            ReplySink::Stream(tx) => {
                let _ = tx.blocking_send(EngineEvent::Done {
                    finish_reason,
                    usage,
                });
            }
            ReplySink::Oneshot { tx, acc } => {
                if let Some(tx) = tx {
                    let _ = tx.send(Ok(EngineOutput {
                        tokens: acc,
                        finish_reason,
                        usage,
                    }));
                }
            }
        }
    }
}

// ============================================================================
// EOS list + 30B config parsing (dynamic — never hardcode ids).
// ============================================================================

/// Read the EOS token LIST for the model dir. Order of truth:
///   1. `generation_config.json` `eos_token_id` — accepted as an int OR a list of ints
///      (35B ships `[248046, 248044]`; 30B `[151645, 151643]`).
///   2. Fallback ONLY if `generation_config.json` is absent: `tokenizer_config.json`'s
///      `eos_token` STRING, mapped to an id via the tokenizer.
///
/// We assert nothing about specific ids — they are always READ from the files.
fn load_eos_list(dir: &PathBuf, tokenizer: &Qwen3Tokenizer) -> Result<Vec<i64>, EngineError> {
    let gen_path = dir.join("generation_config.json");
    if gen_path.exists() {
        let text = std::fs::read_to_string(&gen_path)
            .map_err(|e| EngineError::Eos(format!("read generation_config.json: {e}")))?;
        let v: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| EngineError::Eos(format!("parse generation_config.json: {e}")))?;
        // A JSON `null` is treated EXACTLY like an absent key: `eos_token_id: null`
        // must fall through to the tokenizer_config eos_token fallback rather than
        // hard-error. `.filter(|n| !n.is_null())` collapses the null case to `None`.
        if let Some(node) = v.get("eos_token_id").filter(|n| !n.is_null()) {
            let mut ids = Vec::new();
            match node {
                serde_json::Value::Number(_) => {
                    if let Some(n) = node.as_i64() {
                        ids.push(n);
                    }
                }
                serde_json::Value::Array(arr) => {
                    for e in arr {
                        if let Some(n) = e.as_i64() {
                            ids.push(n);
                        }
                    }
                }
                _ => {}
            }
            if ids.is_empty() {
                return Err(EngineError::Eos(
                    "generation_config.json eos_token_id present but not an int or int list"
                        .to_string(),
                ));
            }
            return Ok(ids);
        }
        // generation_config.json present but eos_token_id absent OR null ⇒ fallback.
    }

    // Fallback: tokenizer_config.json eos_token (string) → id via tokenizer.
    let tok_cfg = dir.join("tokenizer_config.json");
    let text = std::fs::read_to_string(&tok_cfg).map_err(|e| {
        EngineError::Eos(format!(
            "no generation_config.json eos_token_id and cannot read tokenizer_config.json: {e}"
        ))
    })?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| EngineError::Eos(format!("parse tokenizer_config.json: {e}")))?;
    let eos_str = v
        .get("eos_token")
        .and_then(|n| match n {
            serde_json::Value::String(s) => Some(s.clone()),
            // Some configs store {"content": "<|im_end|>"}.
            serde_json::Value::Object(o) => {
                o.get("content").and_then(|c| c.as_str()).map(str::to_string)
            }
            _ => None,
        })
        .ok_or_else(|| EngineError::Eos("tokenizer_config.json has no eos_token".to_string()))?;
    let id = tokenizer.token_to_id(&eos_str).ok_or_else(|| {
        EngineError::Eos(format!("eos_token {eos_str:?} not found in tokenizer vocab"))
    })?;
    Ok(vec![id as i64])
}

/// Build a [`Qwen3MoeConfig`] from a HuggingFace `config.json` (30B has no library
/// `from_hf` constructor; this parses the same fields as examples/moe_static_decode.rs).
/// This is config METADATA parsing, not weight loading — the weights still go through
/// the repo's `load_weights_sharded`.
fn load_moe30b_config(dir: &PathBuf) -> Result<Qwen3MoeConfig, EngineError> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|e| EngineError::Config(format!("read config.json: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| EngineError::Config(format!("parse config.json: {e}")))?;
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

/// One-line device-memory report captured right after load for the startup banner.
fn build_report(device: &CudaDevice, model_id: &str, t_max: usize) -> String {
    use cubecl::Runtime;
    let mem = cubecl::cuda::CudaRuntime::client(device).memory_usage();
    format!("model={model_id} t_max={t_max} device_mem={mem:?}")
}

// ============================================================================
// Host sampler helpers — the xorshift RNG pattern (matches the examples' reproducible
// per-request RNG) + temperature softmax (mirrors grpo/rollout.rs).
// ============================================================================

/// Deterministic xorshift64* RNG seeded per request (same pattern as the examples), so
/// sampling is reproducible from `SamplingParams::seed` without pulling in a `rand` dep.
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

/// Temperature-scaled softmax over a logit row (`temp <= 0` ⇒ handled upstream as greedy).
fn softmax_temp(row: &[f32], temp: f32) -> Vec<f32> {
    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|x| ((x - m) / temp).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}
