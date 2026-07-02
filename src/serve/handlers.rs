//! S.4: HTTP handlers + router assembly composing api/template/detok/engine.
//!
//! Endpoints: `POST /v1/chat/completions`, `POST /v1/completions`,
//! `GET /v1/models`, `GET /health`. See docs/SERVE_PLAN.md S.4.
//!
//! # Module split (testability)
//!
//! Everything that is PURE host logic — param/seed/max_tokens resolution, request
//! validation, the think-boundary state machine, SSE frame/chunk construction,
//! sampling-default loading — lives OUTSIDE any `cuda` gate and is unit-tested
//! under `cargo test --features serve` (no GPU). Only the pieces that touch the
//! [`crate::serve::engine`] (which is itself `cuda`-gated) — [`ServeState`], the
//! async handler bodies, and [`build_router`] — are `#[cfg(feature = "cuda")]`.
//!
//! # Raw-body double parse (key order is load-bearing)
//!
//! The chat body is extracted as raw [`axum::body::Bytes`] and parsed TWICE from
//! the same bytes: once into the typed [`ChatCompletionRequest`] (validation +
//! params) and once into an [`OrderedJson`] to pull `messages`/`tools` for
//! template rendering. Going through a `serde_json::Value` would re-sort object
//! keys (this build's `Value` is `BTreeMap`-backed) and silently change the
//! rendered prompt — proven in review — so we never do that hop.
//!
//! # Streaming transport (true incremental SSE) + the cancel drop-chain
//!
//! Streaming responses are INCREMENTAL, exactly as the plan pins: a spawned tokio
//! task drives `engine channel → detok → think-splitter` and forwards each
//! finished frame into a BOUNDED frame channel; the handler returns
//! `Sse::new(ReceiverStream::new(frame_rx))` (+ `KeepAlive`) immediately. Every
//! frame the client sees has passed BOTH holdbacks — detok/stop-scan GATES
//! emission, never trails it.
//!
//! Backpressure is two-staged and bounded end to end: a slow client fills the
//! frame channel (cap 8) → the forwarding task blocks on `send().await` → it
//! stops receiving from the engine channel (cap 4) → the engine blocks on
//! `blocking_send`. Nothing buffers unboundedly.
//!
//! CANCEL = channel closure, propagated by drop alone (no separate token):
//! client disconnects → axum drops the `Sse` body → `ReceiverStream`/`frame_rx`
//! drop → the task's next `frame_tx.send` fails → `drain` returns with
//! `cancelled` → the task returns, dropping the ENGINE receiver it owns → the
//! engine's next `blocking_send` fails → the engine stops decoding (bounded by
//! ≤ 1 token, the plan's latency budget) and moves to the next request.

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::http::StatusCode;
use serde_json::{Map, Value};

use super::api::{
    ApiError, ChatCompletionChunk, ChatCompletionRequest, ChunkChoice, CompletionRequest, Delta,
    MessageContent, Prompt, StringOrArray, Usage,
};
use super::template::OrderedJson;

/// A ready-to-serialize error paired with the HTTP status it should carry. Pure
/// (`axum::http` is available without `cuda`), so validation is fully unit-tested.
pub type ApiFailure = (StatusCode, ApiError);

// ===========================================================================
// Which of the two served models — a `cuda`-free mirror of engine::WhichModel,
// so pure validation (30B array-content rule) needs no engine types.
// ===========================================================================

/// The two served models, for pure host logic. The 30B chat template renders
/// array (`Parts`) message content as EMPTY (proven in review), so array content
/// is rejected for it; the 35B template handles parts natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServedModel {
    /// Qwen3-30B-A3B-Instruct — string-only message content.
    Qwen30b,
    /// Qwen3.6-35B-A3B — multimodal parts pass through.
    Qwen35b,
}

impl ServedModel {
    /// Whether array (`Parts`) message content is accepted (35B only).
    fn allows_array_content(self) -> bool {
        matches!(self, ServedModel::Qwen35b)
    }
}

// ===========================================================================
// Sampling defaults (loaded at STARTUP from the model dir's generation_config.json —
// never hardcoded: 35B ships 1.0/0.95/20, 30B 0.7/0.8/20).
// ===========================================================================

/// Per-model sampling defaults read from `generation_config.json`. Request fields
/// override these; a field absent from the file falls back to a neutral value
/// (temperature 1.0 / top_p 1.0 / top_k 0 = disabled).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingDefaults {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

impl Default for SamplingDefaults {
    fn default() -> Self {
        SamplingDefaults {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
        }
    }
}

/// Read `temperature`/`top_p`/`top_k` from `<dir>/generation_config.json`. Missing
/// file or missing keys → the neutral [`SamplingDefaults::default`] value for that
/// field (never a panic; the server still boots). Values are READ, never assumed.
pub fn load_sampling_defaults(dir: &Path) -> SamplingDefaults {
    let mut d = SamplingDefaults::default();
    let path = dir.join("generation_config.json");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return d;
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return d;
    };
    if let Some(t) = v.get("temperature").and_then(Value::as_f64) {
        d.temperature = t as f32;
    }
    if let Some(p) = v.get("top_p").and_then(Value::as_f64) {
        d.top_p = p as f32;
    }
    if let Some(k) = v.get("top_k").and_then(Value::as_u64) {
        d.top_k = k as u32;
    }
    d
}

/// Resolve `(temperature, top_p, top_k)`: request field when present, else the
/// per-model default. `temperature == 0.0` flows through unchanged — the engine
/// treats `<= 0` as greedy argmax.
pub fn resolve_sampling(
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_top_k: Option<u32>,
    defaults: &SamplingDefaults,
) -> (f32, f32, u32) {
    (
        req_temperature.unwrap_or(defaults.temperature),
        req_top_p.unwrap_or(defaults.top_p),
        req_top_k.unwrap_or(defaults.top_k),
    )
}

/// Resolve the RNG seed. A request `seed` is AUTHORITATIVE — the signed i64 is
/// reinterpreted to `u64` bit-for-bit (`-1` → `u64::MAX`, matching vLLM's negative
/// seeds). When absent, entropy is injected by hashing `(time_nanos, counter)` so
/// two otherwise-identical concurrent requests still sample differently; the
/// engine then treats the result as authoritative (review finding: seed 0 is a
/// legitimate value, not a sentinel, so we never pass a "no seed" marker).
pub fn resolve_seed(req_seed: Option<i64>, counter: u64, time_nanos: u128) -> u64 {
    match req_seed {
        Some(s) => s as u64,
        None => {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (time_nanos, counter).hash(&mut h);
            h.finish()
        }
    }
}

/// `max_completion_tokens` wins over `max_tokens` when both are present (newer
/// OpenAI spelling), else whichever is set, else `None` (engine clamps to the
/// remaining context). Passed through as `Option` so an explicit overflow 400s in
/// the engine while an omitted value is clamped, never unbounded.
pub fn resolve_max_tokens(
    max_completion_tokens: Option<u32>,
    max_tokens: Option<u32>,
) -> Option<u32> {
    max_completion_tokens.or(max_tokens)
}

/// `enable_thinking` from the request's flattened `extra` map: `Some(bool)` when
/// present as a JSON bool, else `None` (leaves the template variable UNDEFINED,
/// which is the template default — ON for the 35B).
pub fn enable_thinking_from_extra(extra: &Map<String, Value>) -> Option<bool> {
    extra.get("enable_thinking").and_then(Value::as_bool)
}

/// The rendered prompt opens thinking mode iff it ends with `"<think>\n"` — that
/// is exactly how the Qwen templates emit the generation-prompt when thinking is
/// on, so the model's first output token is already INSIDE `<think>`.
pub fn initial_think_state(rendered_prompt: &str) -> bool {
    rendered_prompt.ends_with("<think>\n")
}

/// Flatten the request `stop` (string OR array) into the detok stop list.
pub fn stops_from(stop: &Option<StringOrArray>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(StringOrArray::Single(s)) => vec![s.clone()],
        Some(StringOrArray::Multiple(v)) => v.clone(),
    }
}

/// Look up a top-level object field of an [`OrderedJson`] (insertion order kept).
pub fn ordered_field<'a>(v: &'a OrderedJson, key: &str) -> Option<&'a OrderedJson> {
    match v {
        OrderedJson::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, val)| val),
        _ => None,
    }
}

// ===========================================================================
// Validation — all 400/404 paths render the OpenAI error envelope.
// ===========================================================================

/// Validate the shape of `tools` (the OrderedJson the template will consume).
/// The Qwen templates access `tool.function.name` etc. with LENIENT undefined
/// semantics — a malformed tool renders as silent empty holes in the prompt
/// instead of erroring (caught by the S.5(h) gate). Mirror OpenAI: each tool
/// must be an object with `type == "function"` and a `function` object whose
/// `name` is a non-empty string; anything else is a 400.
pub fn validate_tools(tools: &OrderedJson) -> Result<(), ApiFailure> {
    let bad = |msg: &str| {
        Err((
            StatusCode::BAD_REQUEST,
            ApiError::invalid_request(msg, Some("tools".to_string())),
        ))
    };
    let OrderedJson::Array(items) = tools else {
        return bad("`tools` must be an array");
    };
    for (i, t) in items.iter().enumerate() {
        if !matches!(t, OrderedJson::Object(_)) {
            return bad(&format!("tools[{i}] must be an object"));
        }
        match ordered_field(t, "type") {
            Some(OrderedJson::String(k)) if k == "function" => {}
            _ => return bad(&format!("tools[{i}].type must be the string \"function\"")),
        }
        let Some(f) = ordered_field(t, "function") else {
            return bad(&format!("tools[{i}].function is required"));
        };
        if !matches!(f, OrderedJson::Object(_)) {
            return bad(&format!("tools[{i}].function must be an object"));
        }
        match ordered_field(f, "name") {
            Some(OrderedJson::String(n)) if !n.is_empty() => {}
            _ => return bad(&format!("tools[{i}].function.name must be a non-empty string")),
        }
    }
    Ok(())
}

/// Validate a chat request against the plan's unsupported-param matrix and the
/// model-specific content rule. Returns the OpenAI error envelope + status on the
/// FIRST violation (model identity checked first → 404, then params, then body).
pub fn validate_chat(
    req: &ChatCompletionRequest,
    loaded_model_id: &str,
    model: ServedModel,
) -> Result<(), ApiFailure> {
    if req.model != loaded_model_id {
        return Err(model_not_found(&req.model, loaded_model_id));
    }
    if req.n.is_some_and(|n| n > 1) {
        return Err(bad_request(
            "n > 1 is not supported (this server returns a single completion)",
            Some("n"),
        ));
    }
    if req.logprobs == Some(true) {
        return Err(bad_request("logprobs is not supported", Some("logprobs")));
    }
    if req.presence_penalty.is_some_and(|p| p != 0.0) {
        return Err(bad_request(
            "presence_penalty is not supported (only 0.0)",
            Some("presence_penalty"),
        ));
    }
    if req.frequency_penalty.is_some_and(|p| p != 0.0) {
        return Err(bad_request(
            "frequency_penalty is not supported (only 0.0)",
            Some("frequency_penalty"),
        ));
    }
    if req.messages.is_empty() {
        return Err(bad_request("messages must not be empty", Some("messages")));
    }
    if !model.allows_array_content() {
        for m in &req.messages {
            if matches!(m.content, Some(MessageContent::Parts(_))) {
                return Err(bad_request(
                    "30B chat template requires string content \
                     (array/multimodal content is not supported for this model)",
                    Some("messages"),
                ));
            }
        }
    }
    Ok(())
}

/// Validate a legacy completions request: model identity + reject pre-tokenized
/// (token-id) prompt forms, which the engine cannot ingest.
pub fn validate_completion(
    req: &CompletionRequest,
    loaded_model_id: &str,
) -> Result<(), ApiFailure> {
    if req.model != loaded_model_id {
        return Err(model_not_found(&req.model, loaded_model_id));
    }
    match &req.prompt {
        Prompt::Tokens(_) | Prompt::TokenBatches(_) => Err(bad_request(
            "token-id prompts unsupported (provide a string or array of strings)",
            Some("prompt"),
        )),
        _ => Ok(()),
    }
}

/// The single prompt string for `/v1/completions`. A one-element string array is
/// accepted (unwrapped); a multi-element array is rejected (no batching in v1).
pub fn completion_prompt_text(prompt: &Prompt) -> Result<String, ApiFailure> {
    match prompt {
        Prompt::Text(s) => Ok(s.clone()),
        Prompt::Texts(v) if v.len() == 1 => Ok(v[0].clone()),
        Prompt::Texts(_) => Err(bad_request(
            "batched string prompts are not supported (send a single prompt)",
            Some("prompt"),
        )),
        // Token forms are rejected earlier by `validate_completion`.
        Prompt::Tokens(_) | Prompt::TokenBatches(_) => Err(bad_request(
            "token-id prompts unsupported (provide a string or array of strings)",
            Some("prompt"),
        )),
    }
}

/// 400 `invalid_request_error` envelope.
fn bad_request(message: &str, param: Option<&str>) -> ApiFailure {
    (
        StatusCode::BAD_REQUEST,
        ApiError::invalid_request(message, param.map(str::to_string)),
    )
}

/// 404 `model_not_found` envelope (only the loaded model id is accepted).
fn model_not_found(requested: &str, loaded: &str) -> ApiFailure {
    (
        StatusCode::NOT_FOUND,
        ApiError::new(
            format!("model '{requested}' not found (this server loaded only '{loaded}')"),
            "invalid_request_error",
            Some("model".to_string()),
            Some("model_not_found".to_string()),
        ),
    )
}

/// Wrap a serde/JSON parse error as a 400 envelope (NOT axum's plain-text default).
pub fn json_parse_failure(err: &serde_json::Error) -> ApiFailure {
    bad_request(&format!("invalid request body: {err}"), None)
}

// ===========================================================================
// Think-boundary state machine (pure, unit-tested).
//
// Routes model output into reasoning_content vs content. The `<think>`/`</think>`
// tags go to NEITHER field. `</think>` may be split across chunks, so a trailing
// prefix of the boundary is held back exactly like detok's stop holdback. After
// the close tag a single leading "\n\n" is stripped from content (mirrors vLLM's
// qwen3 reasoning parser's split-only behavior).
// ===========================================================================

const OPEN_TAG: &str = "<think>";
const CLOSE_TAG: &str = "</think>";

/// One routed slice of model output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Segment {
    /// Goes to `delta.reasoning_content` (inside `<think>…</think>`).
    Reasoning(String),
    /// Goes to `delta.content` (after `</think>`, or when not thinking).
    Content(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ThinkState {
    /// Inside reasoning; everything until `</think>` is reasoning.
    InThink,
    /// At the very start, not yet known to be thinking: a leading `<think>`
    /// (possibly split) enters `InThink`, otherwise we fall through to content.
    MaybeOpen,
    /// Emitting content. `strip` = still owe a one-time leading "\n\n" strip
    /// (set right after the close tag).
    Content { strip: bool },
}

/// Incremental `<think>` splitter. One instance per generation; feed detok output
/// via [`push`], call [`finish`] once at end to flush held-back text.
///
/// [`push`]: ThinkSplitter::push
/// [`finish`]: ThinkSplitter::finish
pub struct ThinkSplitter {
    state: ThinkState,
    /// Held-back text (a possible partial tag boundary, or a possible leading
    /// "\n\n" prefix) not yet safe to route.
    buf: String,
}

impl ThinkSplitter {
    /// `initial_in_think` should be [`initial_think_state`] of the rendered prompt.
    pub fn new(initial_in_think: bool) -> Self {
        ThinkSplitter {
            state: if initial_in_think {
                ThinkState::InThink
            } else {
                ThinkState::MaybeOpen
            },
            buf: String::new(),
        }
    }

    /// Feed a chunk of detokenized text; return zero or more routed segments.
    pub fn push(&mut self, text: &str) -> Vec<Segment> {
        self.buf.push_str(text);
        let mut out = Vec::new();
        // Drive the machine until it can make no further progress without more input.
        loop {
            match self.state {
                ThinkState::InThink => match find_or_holdback(&self.buf, CLOSE_TAG) {
                    Boundary::Found(pos) => {
                        let reasoning: String = self.buf.drain(..pos).collect();
                        self.buf.drain(..CLOSE_TAG.len());
                        if !reasoning.is_empty() {
                            out.push(Segment::Reasoning(reasoning));
                        }
                        self.state = ThinkState::Content { strip: true };
                        // Continue: process any content already in buf.
                    }
                    Boundary::Safe(len) => {
                        if len > 0 {
                            let reasoning: String = self.buf.drain(..len).collect();
                            out.push(Segment::Reasoning(reasoning));
                        }
                        break; // holdback (partial `</think>`) stays in buf.
                    }
                },
                ThinkState::MaybeOpen => {
                    if self.buf.is_empty() {
                        break;
                    }
                    if self.buf.starts_with(OPEN_TAG) {
                        self.buf.drain(..OPEN_TAG.len());
                        self.state = ThinkState::InThink;
                    } else if is_proper_prefix(&self.buf, OPEN_TAG) {
                        break; // could still grow into `<think>` — wait.
                    } else {
                        // Not a think opener; from here on it is content.
                        self.state = ThinkState::Content { strip: false };
                    }
                }
                ThinkState::Content { strip } => {
                    if strip {
                        if self.buf.is_empty() {
                            break; // wait to see whether content opens with "\n\n".
                        }
                        if self.buf.starts_with("\n\n") {
                            self.buf.drain(..2);
                            self.state = ThinkState::Content { strip: false };
                        } else if self.buf == "\n" {
                            break; // could still grow into "\n\n".
                        } else {
                            self.state = ThinkState::Content { strip: false };
                        }
                    } else {
                        if !self.buf.is_empty() {
                            out.push(Segment::Content(std::mem::take(&mut self.buf)));
                        }
                        break;
                    }
                }
            }
        }
        out
    }

    /// Flush any held-back text at end-of-generation.
    pub fn finish(&mut self) -> Vec<Segment> {
        let rest = std::mem::take(&mut self.buf);
        let mut out = Vec::new();
        match self.state {
            // An unterminated `</think>` prefix is genuine reasoning output.
            ThinkState::InThink => {
                if !rest.is_empty() {
                    out.push(Segment::Reasoning(rest));
                }
            }
            // A partial `<think>` prefix that never completed is content.
            ThinkState::MaybeOpen => {
                if !rest.is_empty() {
                    out.push(Segment::Content(rest));
                }
            }
            ThinkState::Content { strip } => {
                let mut r = rest;
                if strip && r.starts_with("\n\n") {
                    r.drain(..2);
                }
                if !r.is_empty() {
                    out.push(Segment::Content(r));
                }
            }
        }
        out
    }
}

/// Whether `buf` is a NONEMPTY proper prefix of `tag` (so it could still grow into
/// `tag`).
fn is_proper_prefix(buf: &str, tag: &str) -> bool {
    !buf.is_empty() && buf.len() < tag.len() && tag.starts_with(buf)
}

/// Outcome of scanning `buf` for a tag boundary.
enum Boundary {
    /// `tag` fully occurs at this byte offset.
    Found(usize),
    /// No full occurrence; `usize` bytes are safe to release, the remaining
    /// trailing suffix is the longest proper prefix of `tag` (held back).
    Safe(usize),
}

/// Find `tag` in `buf`, else hold back the longest trailing suffix of `buf` that
/// is a proper prefix of `tag` (cross-chunk boundary handling, mirrors detok).
fn find_or_holdback(buf: &str, tag: &str) -> Boundary {
    if let Some(pos) = buf.find(tag) {
        return Boundary::Found(pos);
    }
    let len = buf.len();
    for start in 0..len {
        if !buf.is_char_boundary(start) {
            continue;
        }
        let suffix = &buf[start..];
        if tag.len() > suffix.len() && tag.starts_with(suffix) {
            return Boundary::Safe(start);
        }
    }
    Boundary::Safe(len)
}

// ===========================================================================
// SSE frame + chunk construction (pure — built from api structs, serialized to
// `data: …\n\n`; see the module streaming note for why not axum::sse::Event).
// ===========================================================================

/// Stable per-request identity for every chunk of one response.
#[derive(Clone)]
pub struct ChunkCtx {
    pub id: String,
    pub created: u64,
    pub model: String,
    /// `stream_options.include_usage`: when true every chunk carries `"usage":null`
    /// and a trailing usage-only chunk is emitted.
    pub include_usage: bool,
}

impl ChunkCtx {
    fn usage_field(&self) -> Option<Option<Usage>> {
        if self.include_usage { Some(None) } else { None }
    }

    fn base_chunk(&self, choices: Vec<ChunkChoice>, usage: Option<Option<Usage>>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: self.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: self.created,
            model: self.model.clone(),
            choices,
            usage,
        }
    }

    /// First chunk: `delta { role:"assistant", content:"" }`, no finish_reason.
    pub fn role_chunk(&self) -> ChatCompletionChunk {
        self.base_chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    role: Some("assistant".to_string()),
                    content: Some(String::new()),
                    reasoning_content: None,
                },
                finish_reason: None,
            }],
            self.usage_field(),
        )
    }

    /// A content or reasoning delta chunk (one field per chunk, per the splitter).
    pub fn segment_chunk(&self, seg: &Segment) -> ChatCompletionChunk {
        let delta = match seg {
            Segment::Content(t) => Delta {
                content: Some(t.clone()),
                ..Default::default()
            },
            Segment::Reasoning(t) => Delta {
                reasoning_content: Some(t.clone()),
                ..Default::default()
            },
        };
        self.base_chunk(
            vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: None,
            }],
            self.usage_field(),
        )
    }

    /// Penultimate chunk: empty delta carrying `finish_reason`.
    pub fn finish_chunk(&self, finish_reason: &str) -> ChatCompletionChunk {
        self.base_chunk(
            vec![ChunkChoice {
                index: 0,
                delta: Delta::default(),
                finish_reason: Some(finish_reason.to_string()),
            }],
            self.usage_field(),
        )
    }

    /// Final usage-only chunk: EMPTY choices, `usage` set. Only emitted when
    /// `include_usage`.
    pub fn usage_chunk(&self, usage: Usage) -> ChatCompletionChunk {
        self.base_chunk(vec![], Some(Some(usage)))
    }
}

/// Serialize any chunk/response value into one SSE `data:` frame.
pub fn sse_frame<T: serde::Serialize>(value: &T) -> String {
    // Serialization of these plain structs cannot fail; fall back to an empty
    // object defensively rather than panicking inside a live response.
    let json = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
    format!("data: {json}\n\n")
}

/// The terminating SSE frame.
pub const SSE_DONE: &str = "data: [DONE]\n\n";

/// Build a [`Usage`] from prompt + completion counts.
pub fn usage(prompt_tokens: usize, completion_tokens: usize) -> Usage {
    Usage {
        prompt_tokens: prompt_tokens as u32,
        completion_tokens: completion_tokens as u32,
        total_tokens: (prompt_tokens + completion_tokens) as u32,
    }
}

/// Collapse a stream of [`Segment`]s into `(content, reasoning)` for the non-stream
/// response. `reasoning` is `None` when empty; `content` is `None` only when the
/// whole answer was reasoning (mirrors OpenAI's shape).
pub fn collapse_segments(segments: &[Segment]) -> (Option<String>, Option<String>) {
    let mut content = String::new();
    let mut reasoning = String::new();
    for s in segments {
        match s {
            Segment::Content(t) => content.push_str(t),
            Segment::Reasoning(t) => reasoning.push_str(t),
        }
    }
    let reasoning = if reasoning.is_empty() {
        None
    } else {
        Some(reasoning)
    };
    let content = if content.is_empty() && reasoning.is_some() {
        None
    } else {
        Some(content)
    };
    (content, reasoning)
}

// ===========================================================================
// Small time / id helpers (pure).
// ===========================================================================

/// Unix seconds now (0 if the clock is before the epoch — never panics).
pub fn now_unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Nanoseconds since the epoch (entropy source for seed/id; 0 on clock error).
pub fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0)
}

/// Stable request id `"<prefix>-<16 hex>"` (e.g. `chatcmpl-…`) from a per-request
/// counter mixed with the clock.
pub fn gen_id(prefix: &str, counter: u64, nanos: u128) -> String {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (counter, nanos).hash(&mut h);
    format!("{prefix}-{:016x}", h.finish())
}

// ===========================================================================
// cuda-gated: engine-touching state, handler bodies, router assembly.
// ===========================================================================

#[cfg(feature = "cuda")]
pub use engine_wiring::{build_router, ServeState};

#[cfg(feature = "cuda")]
mod engine_wiring {
    use std::convert::Infallible;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    use axum::body::Bytes;
    use axum::extract::State;
    use axum::http::StatusCode;
    use axum::response::sse::{Event, KeepAlive, Sse};
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};
    use axum::{Json, Router};
    use tokio_stream::wrappers::ReceiverStream;

    use super::*;
    use crate::serve::api::{
        self, ChatChoice, ChatCompletionRequest, ChatCompletionResponse, CompletionChoice,
        CompletionRequest, CompletionResponse, Model, ModelList, ResponseMessage,
    };
    use crate::serve::engine::{
        EngineEvent, EngineHandle, EngineReply, EngineRequest, FinishReason, SamplingParams,
    };
    use crate::serve::template::{ChatTemplate, OrderedJson};
    use tokenizers::Tokenizer;

    /// Shared, `Send`+`Sync` server state behind every handler. Holds only cheap
    /// handles: the engine sender, the loaded chat template, a clonable tokenizer
    /// (for per-request incremental detok), sampling defaults, the loaded model
    /// id, and a monotonic request counter (seed entropy + request ids).
    pub struct ServeState {
        pub engine: EngineHandle,
        pub template: ChatTemplate,
        pub tokenizer: Tokenizer,
        pub defaults: SamplingDefaults,
        pub model_id: String,
        pub model: ServedModel,
        counter: AtomicU64,
    }

    impl ServeState {
        /// Build server state. `model_id` MUST equal `engine.model_id()`.
        pub fn new(
            engine: EngineHandle,
            template: ChatTemplate,
            tokenizer: Tokenizer,
            defaults: SamplingDefaults,
            model: ServedModel,
        ) -> Self {
            let model_id = engine.model_id().to_string();
            ServeState {
                engine,
                template,
                tokenizer,
                defaults,
                model_id,
                model,
                counter: AtomicU64::new(0),
            }
        }

        fn next_counter(&self) -> u64 {
            self.counter.fetch_add(1, Ordering::Relaxed)
        }
    }

    /// Assemble the full router: extend api::base_router (which owns `/health`)
    /// with the `/v1` endpoints, finalized with the shared state.
    pub fn build_router(state: Arc<ServeState>) -> Router {
        let v1 = Router::new()
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .with_state(state);
        api::base_router().merge(v1)
    }

    // ---- GET /v1/models — reports ONLY the loaded model id. ----
    async fn models(State(state): State<Arc<ServeState>>) -> Json<ModelList> {
        Json(ModelList {
            object: "list".to_string(),
            data: vec![Model {
                id: state.model_id.clone(),
                object: "model".to_string(),
                created: now_unix_secs(),
                owned_by: "local".to_string(),
            }],
        })
    }

    /// Turn an [`ApiFailure`] into a response with the OpenAI error envelope.
    fn fail(f: ApiFailure) -> Response {
        (f.0, Json(f.1)).into_response()
    }

    /// One SSE event whose `data:` payload is the JSON of `value` — the SAME
    /// payload bytes [`sse_frame`] produces (single-line JSON, so axum's Event
    /// serialization `data: <json>\n\n` is byte-identical to the tested frames).
    fn sse_event<T: serde::Serialize>(value: &T) -> Event {
        let json = serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string());
        Event::default().data(json)
    }

    /// The terminating `data: [DONE]` event (mirrors [`SSE_DONE`]).
    fn sse_done() -> Event {
        Event::default().data("[DONE]")
    }

    /// The bounded frame channel feeding an in-flight SSE response. Cap 8: small
    /// enough that a stalled client back-pressures the engine within a few
    /// tokens, big enough that frame bursts (segment + finish + usage + DONE)
    /// don't block a healthy stream.
    type FrameTx = tokio::sync::mpsc::Sender<Result<Event, Infallible>>;

    /// Wrap a frame receiver into the SSE response (KeepAlive per the plan).
    fn sse_response(frx: tokio::sync::mpsc::Receiver<Result<Event, Infallible>>) -> Response {
        Sse::new(ReceiverStream::new(frx))
            .keep_alive(KeepAlive::default())
            .into_response()
    }

    /// Streaming sink handed to [`drain`]: maps each produced [`Segment`] to an
    /// SSE event and awaits it into the bounded frame channel. A failed send
    /// means the HTTP client is gone — drain reports `cancelled` and its caller
    /// drops the engine receiver (the cancel drop-chain in the module docs).
    struct SegSink {
        tx: FrameTx,
        map: Box<dyn Fn(&Segment) -> Event + Send + Sync>,
    }

    // ---- POST /v1/chat/completions ----
    async fn chat_completions(
        State(state): State<Arc<ServeState>>,
        body: Bytes,
    ) -> Response {
        // Parse ONCE into the typed request (validation / params).
        let req: ChatCompletionRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => return fail(json_parse_failure(&e)),
        };
        if let Err(f) = validate_chat(&req, &state.model_id, state.model) {
            return fail(f);
        }
        // Parse ONCE into OrderedJson for messages/tools (key order load-bearing).
        let ordered: OrderedJson = match serde_json::from_slice(&body) {
            Ok(o) => o,
            Err(e) => return fail(json_parse_failure(&e)),
        };
        let messages = ordered_field(&ordered, "messages")
            .cloned()
            .unwrap_or(OrderedJson::Array(Vec::new()));
        let tools = ordered_field(&ordered, "tools")
            .cloned()
            .filter(|t| !matches!(t, OrderedJson::Null));
        if let Some(t) = &tools {
            if let Err(f) = validate_tools(t) {
                return fail(f);
            }
        }

        let enable_thinking = enable_thinking_from_extra(&req.extra);
        // add_generation_prompt is ALWAYS true.
        let prompt = match state
            .template
            .render(&messages, tools.as_ref(), true, enable_thinking)
        {
            Ok(p) => p,
            Err(e) => return fail(bad_request(&format!("template render failed: {e}"), None)),
        };

        let (temperature, top_p, top_k) =
            resolve_sampling(req.temperature, req.top_p, req.top_k, &state.defaults);
        let seed = resolve_seed(req.seed, state.next_counter(), now_nanos());
        let max_tokens = resolve_max_tokens(req.max_completion_tokens, req.max_tokens);
        let stops = stops_from(&req.stop);
        let stream = req.stream.unwrap_or(false);
        let include_usage = stream
            && req
                .stream_options
                .as_ref()
                .and_then(|s| s.include_usage)
                .unwrap_or(false);
        let initial_think = initial_think_state(&prompt);

        let params = SamplingParams {
            temperature,
            top_p,
            top_k: top_k as usize,
            seed,
        };
        run_chat(
            state,
            prompt,
            params,
            max_tokens.map(|n| n as usize),
            stops,
            stream,
            include_usage,
            initial_think,
        )
        .await
    }

    // ---- POST /v1/completions (legacy, raw prompt, no think split) ----
    async fn completions(State(state): State<Arc<ServeState>>, body: Bytes) -> Response {
        let req: CompletionRequest = match serde_json::from_slice(&body) {
            Ok(r) => r,
            Err(e) => return fail(json_parse_failure(&e)),
        };
        if let Err(f) = validate_completion(&req, &state.model_id) {
            return fail(f);
        }
        let prompt = match completion_prompt_text(&req.prompt) {
            Ok(p) => p,
            Err(f) => return fail(f),
        };
        let (temperature, top_p, top_k) =
            resolve_sampling(req.temperature, req.top_p, None, &state.defaults);
        let seed = resolve_seed(None, state.next_counter(), now_nanos());
        let stops = stops_from(&req.stop);
        let stream = req.stream.unwrap_or(false);
        let params = SamplingParams {
            temperature,
            top_p,
            top_k: top_k as usize,
            seed,
        };
        run_completion(
            state,
            prompt,
            params,
            req.max_tokens.map(|n| n as usize),
            stops,
            stream,
        )
        .await
    }

    /// Submit a request on a fresh bounded return channel; `Full` → 429 envelope.
    /// Returns the receiver on success.
    fn submit(
        state: &ServeState,
        prompt: String,
        params: SamplingParams,
        max_tokens: Option<usize>,
    ) -> Result<tokio::sync::mpsc::Receiver<EngineEvent>, Response> {
        // Bounded return channel (~4): a slow client makes the engine block on
        // blocking_send instead of buffering logits into RAM.
        let (tx, rx) = tokio::sync::mpsc::channel::<EngineEvent>(4);
        let request = EngineRequest {
            prompt,
            params,
            max_tokens,
            reply: EngineReply::Stream(tx),
        };
        match state.engine.try_submit(request) {
            Ok(()) => Ok(rx),
            Err(tokio::sync::mpsc::error::TrySendError::Full(_)) => Err(fail((
                StatusCode::TOO_MANY_REQUESTS,
                ApiError::new(
                    "engine busy, queue full",
                    "rate_limit_error",
                    None,
                    Some("engine_overloaded".to_string()),
                ),
            ))),
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => Err(fail((
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiError::new(
                    "engine unavailable",
                    "internal_error",
                    None,
                    Some("engine_gone".to_string()),
                ),
            ))),
        }
    }

    /// Outcome of draining the engine channel (shared by stream + non-stream).
    struct DrainResult {
        segments: Vec<Segment>,
        prompt_tokens: usize,
        completion_tokens: usize,
        finish_reason: &'static str,
        error: Option<String>,
        /// The streaming client disconnected mid-generation (frame send failed).
        /// The caller must simply return, dropping the engine receiver — that IS
        /// the cancel signal (module docs: the drop-chain).
        cancelled: bool,
    }

    /// Deliver produced segments: forward each into the streaming sink (awaiting
    /// the bounded frame channel — this is where a slow client back-pressures the
    /// engine) and record it for the non-stream collapse. Returns `false` when
    /// the client is gone (send failed) — the caller stops draining.
    async fn deliver(
        segs: Vec<Segment>,
        out: &mut Vec<Segment>,
        sink: &Option<SegSink>,
    ) -> bool {
        for s in segs {
            if let Some(k) = sink {
                if k.tx.send(Ok((k.map)(&s))).await.is_err() {
                    return false;
                }
            }
            out.push(s);
        }
        true
    }

    /// ONE shared drain: engine events → detok → think-splitter, accumulating
    /// routed segments. Used by BOTH stream and non-stream so stop-scan +
    /// think-split + usage are byte-for-byte identical between the two paths (the
    /// engine's oneshot arm stays unused by HTTP). For streaming, `sink` forwards
    /// each segment into the frame channel AS IT IS PRODUCED; for non-stream it
    /// is `None`.
    async fn drain(
        rx: &mut tokio::sync::mpsc::Receiver<EngineEvent>,
        tokenizer: Tokenizer,
        stops: Vec<String>,
        initial_think: bool,
        sink: Option<SegSink>,
    ) -> DrainResult {
        use crate::serve::detok::IncrementalDetok;
        let mut detok = IncrementalDetok::new(tokenizer, stops);
        let mut splitter = ThinkSplitter::new(initial_think);
        let mut segments: Vec<Segment> = Vec::new();
        let mut prompt_tokens = 0usize;
        let mut completion_tokens = 0usize;
        let mut finish_reason: &'static str = "stop";
        let mut error: Option<String> = None;
        let mut cancelled = false;

        'outer: while let Some(ev) = rx.recv().await {
            match ev {
                EngineEvent::Start { prompt_tokens: pt } => {
                    prompt_tokens = pt;
                    // Client may have vanished during a long prefill; without a
                    // frame send we would only notice on the FIRST token — check
                    // the channel now so the engine cancels before decoding
                    // (review finding).
                    if let Some(k) = &sink {
                        if k.tx.is_closed() {
                            cancelled = true;
                            break 'outer;
                        }
                    }
                }
                EngineEvent::Token(id) => {
                    completion_tokens += 1;
                    let pr = detok.push(id);
                    if !pr.text.is_empty() {
                        let segs = splitter.push(&pr.text);
                        if !deliver(segs, &mut segments, &sink).await {
                            cancelled = true;
                            break 'outer;
                        }
                    }
                    if pr.stop.is_some() {
                        // Stop-string hit: drop the receiver so the engine cancels
                        // (Done never arrives). usage comes from Start + our count.
                        finish_reason = "stop";
                        break;
                    }
                }
                EngineEvent::Done {
                    finish_reason: fr,
                    usage,
                } => {
                    prompt_tokens = usage.prompt_tokens;
                    completion_tokens = usage.completion_tokens;
                    finish_reason = match fr {
                        FinishReason::Stop => "stop",
                        FinishReason::Length => "length",
                    };
                    break;
                }
                EngineEvent::Error { message } => {
                    error = Some(message);
                    break;
                }
            }
        }

        // Flush detok + splitter tails (unless the engine errored or the client left).
        if error.is_none() && !cancelled {
            let tail = detok.finish();
            let mut segs = if tail.is_empty() {
                Vec::new()
            } else {
                splitter.push(&tail)
            };
            segs.extend(splitter.finish());
            if !deliver(segs, &mut segments, &sink).await {
                cancelled = true;
            }
        }

        DrainResult {
            segments,
            prompt_tokens,
            completion_tokens,
            finish_reason,
            error,
            cancelled,
        }
    }

    /// Log one request line to stderr (queue depth omitted — not cheaply exposed
    /// by EngineHandle without an extra engine API, which is out of S.4 scope).
    fn log_request(
        endpoint: &str,
        id: &str,
        stream: bool,
        prompt_tokens: usize,
        completion_tokens: usize,
        started: Instant,
        finish_reason: &str,
    ) {
        let elapsed = started.elapsed();
        let secs = elapsed.as_secs_f64();
        let toks = if secs > 0.0 {
            completion_tokens as f64 / secs
        } else {
            0.0
        };
        eprintln!(
            "[qwen-serve] {endpoint} id={id} stream={stream} prompt_tokens={prompt_tokens} \
             completion_tokens={completion_tokens} elapsed={:.3}s tok/s={:.2} finish={finish_reason}",
            secs, toks,
        );
    }

    /// Chat generation (ONE path for stream + non-stream; see `drain`).
    #[allow(clippy::too_many_arguments)]
    async fn run_chat(
        state: Arc<ServeState>,
        prompt: String,
        params: SamplingParams,
        max_tokens: Option<usize>,
        stops: Vec<String>,
        stream: bool,
        include_usage: bool,
        initial_think: bool,
    ) -> Response {
        let started = Instant::now();
        let id = gen_id("chatcmpl", state.next_counter(), now_nanos());
        let created = now_unix_secs();
        let tokenizer = state.tokenizer.clone();

        let mut rx = match submit(&state, prompt, params, max_tokens) {
            Ok(rx) => rx,
            Err(resp) => return resp,
        };

        let ctx = ChunkCtx {
            id: id.clone(),
            created,
            model: state.model_id.clone(),
            include_usage,
        };

        if stream {
            // TRUE incremental SSE (module docs): the spawned task owns the
            // engine receiver; frames flow through the bounded channel as they
            // are produced. Client disconnect propagates by drop alone.
            let (ftx, frx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(8);
            let map_ctx = ctx.clone();
            tokio::spawn(async move {
                let mut rx = rx; // dropped when this task returns ⇒ engine cancels
                if ftx.send(Ok(sse_event(&ctx.role_chunk()))).await.is_err() {
                    log_request("chat", &id, true, 0, 0, started, "cancelled");
                    return;
                }
                let sink = SegSink {
                    tx: ftx.clone(),
                    map: Box::new(move |seg| sse_event(&map_ctx.segment_chunk(seg))),
                };
                let res = drain(&mut rx, tokenizer, stops, initial_think, Some(sink)).await;

                if res.cancelled {
                    log_request(
                        "chat", &id, true, res.prompt_tokens, res.completion_tokens, started,
                        "cancelled",
                    );
                    return;
                }
                if let Some(message) = res.error {
                    // Stream error: emit an error data-frame then close (documented
                    // stream-error shape: one `data: {error:{…}}` frame + `[DONE]`).
                    let _ = ftx
                        .send(Ok(sse_event(&ApiError::invalid_request(message, None))))
                        .await;
                    let _ = ftx.send(Ok(sse_done())).await;
                    log_request(
                        "chat", &id, true, res.prompt_tokens, res.completion_tokens, started,
                        "error",
                    );
                    return;
                }

                let _ = ftx.send(Ok(sse_event(&ctx.finish_chunk(res.finish_reason)))).await;
                if include_usage {
                    let _ = ftx
                        .send(Ok(sse_event(
                            &ctx.usage_chunk(usage(res.prompt_tokens, res.completion_tokens)),
                        )))
                        .await;
                }
                let _ = ftx.send(Ok(sse_done())).await;
                log_request(
                    "chat", &id, true, res.prompt_tokens, res.completion_tokens, started,
                    res.finish_reason,
                );
            });
            sse_response(frx)
        } else {
            let res = drain(&mut rx, tokenizer, stops, initial_think, None).await;
            if let Some(message) = res.error {
                return fail(bad_request(&message, None));
            }
            let (content, reasoning) = collapse_segments(&res.segments);
            log_request(
                "chat", &id, false, res.prompt_tokens, res.completion_tokens, started,
                res.finish_reason,
            );
            Json(ChatCompletionResponse {
                id,
                object: "chat.completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ResponseMessage {
                        role: "assistant".to_string(),
                        content,
                        reasoning_content: reasoning,
                    },
                    finish_reason: Some(res.finish_reason.to_string()),
                }],
                usage: usage(res.prompt_tokens, res.completion_tokens),
            })
            .into_response()
        }
    }

    /// Legacy completion generation (raw text, no think split). Same shared drain;
    /// `initial_think=false` so ALL output routes to content.
    async fn run_completion(
        state: Arc<ServeState>,
        prompt: String,
        params: SamplingParams,
        max_tokens: Option<usize>,
        stops: Vec<String>,
        stream: bool,
    ) -> Response {
        let started = Instant::now();
        let id = gen_id("cmpl", state.next_counter(), now_nanos());
        let created = now_unix_secs();
        let tokenizer = state.tokenizer.clone();

        let mut rx = match submit(&state, prompt, params, max_tokens) {
            Ok(rx) => rx,
            Err(resp) => return resp,
        };

        // Completions treat everything as content (no reasoning field on the wire).
        let text_of = |segs: &[Segment]| -> String {
            segs.iter()
                .map(|s| match s {
                    Segment::Content(t) | Segment::Reasoning(t) => t.as_str(),
                })
                .collect()
        };

        if stream {
            // TRUE incremental SSE — same drop-chain as run_chat (module docs).
            let (ftx, frx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(8);
            let model = state.model_id.clone();
            let mk_chunk = {
                let id = id.clone();
                let model = model.clone();
                move |text: String, finish_reason: Option<String>, u: Option<Usage>| CompletionResponse {
                    id: id.clone(),
                    object: "text_completion".to_string(),
                    created,
                    model: model.clone(),
                    choices: vec![CompletionChoice {
                        text,
                        index: 0,
                        finish_reason,
                    }],
                    usage: u,
                }
            };
            tokio::spawn(async move {
                let mut rx = rx; // dropped when this task returns ⇒ engine cancels
                let mk = mk_chunk.clone();
                let sink = SegSink {
                    tx: ftx.clone(),
                    map: Box::new(move |seg| {
                        let delta = match seg {
                            Segment::Content(t) | Segment::Reasoning(t) => t.clone(),
                        };
                        sse_event(&mk(delta, None, None))
                    }),
                };
                let res = drain(&mut rx, tokenizer, stops, false, Some(sink)).await;

                if res.cancelled {
                    log_request(
                        "completion", &id, true, res.prompt_tokens, res.completion_tokens,
                        started, "cancelled",
                    );
                    return;
                }
                if let Some(message) = res.error {
                    let _ = ftx
                        .send(Ok(sse_event(&ApiError::invalid_request(message, None))))
                        .await;
                    let _ = ftx.send(Ok(sse_done())).await;
                    log_request(
                        "completion", &id, true, res.prompt_tokens, res.completion_tokens,
                        started, "error",
                    );
                    return;
                }
                // Final chunk carries finish_reason + usage, then [DONE].
                let fin = mk_chunk(
                    String::new(),
                    Some(res.finish_reason.to_string()),
                    Some(usage(res.prompt_tokens, res.completion_tokens)),
                );
                let _ = ftx.send(Ok(sse_event(&fin))).await;
                let _ = ftx.send(Ok(sse_done())).await;
                log_request(
                    "completion", &id, true, res.prompt_tokens, res.completion_tokens, started,
                    res.finish_reason,
                );
            });
            sse_response(frx)
        } else {
            let res = drain(&mut rx, tokenizer, stops, false, None).await;
            if let Some(message) = res.error {
                return fail(bad_request(&message, None));
            }
            let text = text_of(&res.segments);
            log_request(
                "completion", &id, false, res.prompt_tokens, res.completion_tokens, started,
                res.finish_reason,
            );
            Json(CompletionResponse {
                id,
                object: "text_completion".to_string(),
                created,
                model: state.model_id.clone(),
                choices: vec![CompletionChoice {
                    text,
                    index: 0,
                    finish_reason: Some(res.finish_reason.to_string()),
                }],
                usage: Some(usage(res.prompt_tokens, res.completion_tokens)),
            })
            .into_response()
        }
    }
}

// ===========================================================================
// Pure-logic unit tests (run under `cargo test --features serve`, no GPU).
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tools validation ----

    #[test]
    fn tools_missing_function_object_rejected() {
        let t: OrderedJson = serde_json::from_str(r#"[{"type": "function"}]"#).unwrap();
        assert!(validate_tools(&t).is_err());
        let t: OrderedJson =
            serde_json::from_str(r#"[{"type": "function", "function": {"name": ""}}]"#).unwrap();
        assert!(validate_tools(&t).is_err());
        let t: OrderedJson = serde_json::from_str(r#"[{"type": "web_search"}]"#).unwrap();
        assert!(validate_tools(&t).is_err());
        let t: OrderedJson = serde_json::from_str(r#"{"not": "an array"}"#).unwrap();
        assert!(validate_tools(&t).is_err());
    }

    #[test]
    fn tools_wellformed_accepted() {
        let t: OrderedJson = serde_json::from_str(
            r#"[{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]"#,
        )
        .unwrap();
        assert!(validate_tools(&t).is_ok());
    }

    // ---- think-splitter ----

    fn run_splitter(initial: bool, chunks: &[&str]) -> (String, String) {
        let mut s = ThinkSplitter::new(initial);
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut push = |segs: Vec<Segment>| {
            for seg in segs {
                match seg {
                    Segment::Content(t) => content.push_str(&t),
                    Segment::Reasoning(t) => reasoning.push_str(&t),
                }
            }
        };
        for c in chunks {
            push(s.push(c));
        }
        push(s.finish());
        (content, reasoning)
    }

    #[test]
    fn think_initial_in_think_routes_reasoning_then_content() {
        // Template opened `<think>\n` ⇒ start InThink. Reasoning, close, content.
        let (content, reasoning) =
            run_splitter(true, &["let me think", "</think>", "the answer"]);
        assert_eq!(reasoning, "let me think");
        assert_eq!(content, "the answer");
    }

    #[test]
    fn think_close_tag_split_across_chunks_held_back() {
        // `</think>` arrives in pieces; no fragment may leak into either field.
        let (content, reasoning) =
            run_splitter(true, &["reason", "</", "thi", "nk>", "answer"]);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "answer");
    }

    #[test]
    fn think_strips_single_leading_double_newline_after_close() {
        let (content, reasoning) = run_splitter(true, &["r</think>\n\nhello"]);
        assert_eq!(reasoning, "r");
        assert_eq!(content, "hello", "one leading \\n\\n stripped");
    }

    #[test]
    fn think_strips_only_one_double_newline() {
        // Three newlines: strip exactly the first "\n\n", keep the remaining "\n".
        let (content, _r) = run_splitter(true, &["r</think>\n\n\nx"]);
        assert_eq!(content, "\nx");
    }

    #[test]
    fn think_leading_double_newline_split_across_chunks() {
        // "\n" then "\nhello": the strip must span the chunk boundary.
        let (content, _r) = run_splitter(true, &["r</think>", "\n", "\nhello"]);
        assert_eq!(content, "hello");
    }

    #[test]
    fn think_single_leading_newline_is_not_stripped() {
        // A lone leading "\n" (not "\n\n") after close is genuine content.
        let (content, _r) = run_splitter(true, &["r</think>", "\nhi"]);
        assert_eq!(content, "\nhi");
    }

    #[test]
    fn think_not_thinking_passes_through_as_content() {
        let (content, reasoning) = run_splitter(false, &["plain answer"]);
        assert_eq!(content, "plain answer");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn think_literal_open_tag_at_start_enters_think() {
        // Not initially thinking, but the model emits `<think>` first.
        let (content, reasoning) =
            run_splitter(false, &["<think>", "reasoning", "</think>", "answer"]);
        assert_eq!(reasoning, "reasoning");
        assert_eq!(content, "answer");
    }

    #[test]
    fn think_open_tag_split_at_start() {
        let (content, reasoning) =
            run_splitter(false, &["<thi", "nk>reason</think>ans"]);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "ans");
    }

    #[test]
    fn think_leading_lt_that_is_not_open_tag_is_content() {
        // A leading '<' that never becomes `<think>` must be released as content.
        let (content, reasoning) = run_splitter(false, &["<b>bold"]);
        assert_eq!(content, "<b>bold");
        assert_eq!(reasoning, "");
    }

    #[test]
    fn think_unterminated_think_flushes_as_reasoning() {
        // No close tag ever arrives ⇒ everything is reasoning, flushed at finish.
        let (content, reasoning) = run_splitter(true, &["still thinking", " more"]);
        assert_eq!(reasoning, "still thinking more");
        assert_eq!(content, "");
    }

    #[test]
    fn think_partial_close_prefix_at_finish_is_reasoning() {
        // A dangling `</thi` never completes ⇒ it is reasoning output, not dropped.
        let (content, reasoning) = run_splitter(true, &["done</thi"]);
        assert_eq!(reasoning, "done</thi");
        assert_eq!(content, "");
    }

    #[test]
    fn think_close_at_very_end_no_content() {
        let (content, reasoning) = run_splitter(true, &["reason</think>"]);
        assert_eq!(reasoning, "reason");
        assert_eq!(content, "");
    }

    #[test]
    fn collapse_segments_reasoning_none_when_empty() {
        let (c, r) = collapse_segments(&[Segment::Content("hi".into())]);
        assert_eq!(c, Some("hi".to_string()));
        assert_eq!(r, None);
    }

    #[test]
    fn collapse_segments_content_none_when_all_reasoning() {
        let (c, r) = collapse_segments(&[Segment::Reasoning("why".into())]);
        assert_eq!(c, None);
        assert_eq!(r, Some("why".to_string()));
    }

    // ---- param / seed / max_tokens resolution ----

    #[test]
    fn sampling_request_overrides_defaults() {
        let d = SamplingDefaults {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
        };
        assert_eq!(resolve_sampling(None, None, None, &d), (1.0, 0.95, 20));
        assert_eq!(
            resolve_sampling(Some(0.0), Some(0.5), Some(5), &d),
            (0.0, 0.5, 5),
            "temperature 0 passes through (greedy)"
        );
    }

    #[test]
    fn seed_request_is_authoritative_and_bitcast() {
        assert_eq!(resolve_seed(Some(42), 0, 0), 42u64);
        assert_eq!(resolve_seed(Some(-1), 0, 0), u64::MAX, "-1 → all-ones bits");
        assert_eq!(resolve_seed(Some(0), 7, 9), 0u64, "seed 0 is a real value");
    }

    #[test]
    fn seed_absent_injects_entropy_varying_by_counter() {
        let a = resolve_seed(None, 1, 1000);
        let b = resolve_seed(None, 2, 1000);
        assert_ne!(a, b, "different counters ⇒ different seeds");
    }

    #[test]
    fn max_tokens_prefers_completion_spelling() {
        assert_eq!(resolve_max_tokens(Some(10), Some(20)), Some(10));
        assert_eq!(resolve_max_tokens(None, Some(20)), Some(20));
        assert_eq!(resolve_max_tokens(Some(5), None), Some(5));
        assert_eq!(resolve_max_tokens(None, None), None);
    }

    #[test]
    fn enable_thinking_reads_extra_bool() {
        let mut extra = Map::new();
        assert_eq!(enable_thinking_from_extra(&extra), None);
        extra.insert("enable_thinking".to_string(), Value::Bool(false));
        assert_eq!(enable_thinking_from_extra(&extra), Some(false));
        extra.insert("enable_thinking".to_string(), Value::Bool(true));
        assert_eq!(enable_thinking_from_extra(&extra), Some(true));
    }

    #[test]
    fn initial_think_only_on_open_suffix() {
        assert!(initial_think_state("<|im_start|>assistant\n<think>\n"));
        assert!(!initial_think_state("<|im_start|>assistant\n"));
        assert!(!initial_think_state("<think> "));
    }

    #[test]
    fn stops_from_string_and_array() {
        assert_eq!(stops_from(&None), Vec::<String>::new());
        assert_eq!(
            stops_from(&Some(StringOrArray::Single("x".into()))),
            vec!["x".to_string()]
        );
        assert_eq!(
            stops_from(&Some(StringOrArray::Multiple(vec!["a".into(), "b".into()]))),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    // ---- validation ----

    fn chat_req(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn validate_chat_rejects_unsupported_params() {
        let base = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]"#;
        let cases = [
            (r#","n":2}"#, "n"),
            (r#","logprobs":true}"#, "logprobs"),
            (r#","presence_penalty":0.5}"#, "presence_penalty"),
            (r#","frequency_penalty":-0.5}"#, "frequency_penalty"),
        ];
        for (tail, param) in cases {
            let req = chat_req(&format!("{base}{tail}"));
            let (code, err) = validate_chat(&req, "m", ServedModel::Qwen35b).unwrap_err();
            assert_eq!(code, StatusCode::BAD_REQUEST, "param {param}");
            assert_eq!(err.error.param.as_deref(), Some(param));
        }
    }

    #[test]
    fn validate_chat_allows_zero_penalties() {
        let req = chat_req(
            r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
                "presence_penalty":0.0,"frequency_penalty":0.0,"n":1}"#,
        );
        assert!(validate_chat(&req, "m", ServedModel::Qwen35b).is_ok());
    }

    #[test]
    fn validate_chat_unknown_model_is_404() {
        let req = chat_req(r#"{"model":"other","messages":[{"role":"user","content":"hi"}]}"#);
        let (code, err) = validate_chat(&req, "m", ServedModel::Qwen35b).unwrap_err();
        assert_eq!(code, StatusCode::NOT_FOUND);
        assert_eq!(err.error.code.as_deref(), Some("model_not_found"));
    }

    #[test]
    fn validate_chat_empty_messages_is_400() {
        let req = chat_req(r#"{"model":"m","messages":[]}"#);
        let (code, err) = validate_chat(&req, "m", ServedModel::Qwen35b).unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(err.error.param.as_deref(), Some("messages"));
    }

    #[test]
    fn validate_chat_30b_rejects_array_content_but_35b_allows() {
        let req = chat_req(
            r#"{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        );
        // 30B: array content → 400.
        let (code, err) = validate_chat(&req, "m", ServedModel::Qwen30b).unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(err.error.param.as_deref(), Some("messages"));
        // 35B: same body passes.
        assert!(validate_chat(&req, "m", ServedModel::Qwen35b).is_ok());
    }

    #[test]
    fn validate_completion_rejects_token_prompts() {
        let req: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":[1,2,3]}"#).unwrap();
        let (code, err) = validate_completion(&req, "m").unwrap_err();
        assert_eq!(code, StatusCode::BAD_REQUEST);
        assert_eq!(err.error.param.as_deref(), Some("prompt"));

        let batch: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":[[1,2],[3,4]]}"#).unwrap();
        assert!(validate_completion(&batch, "m").is_err());
    }

    #[test]
    fn completion_prompt_text_variants() {
        assert_eq!(
            completion_prompt_text(&Prompt::Text("hi".into())).unwrap(),
            "hi"
        );
        assert_eq!(
            completion_prompt_text(&Prompt::Texts(vec!["only".into()])).unwrap(),
            "only"
        );
        assert!(completion_prompt_text(&Prompt::Texts(vec!["a".into(), "b".into()])).is_err());
    }

    // ---- SSE frame construction ----

    #[test]
    fn sse_frame_shape_and_done() {
        let ctx = ChunkCtx {
            id: "chatcmpl-x".into(),
            created: 1,
            model: "m".into(),
            include_usage: false,
        };
        let frame = sse_frame(&ctx.role_chunk());
        assert!(frame.starts_with("data: "));
        assert!(frame.ends_with("\n\n"));
        let json = frame.trim_start_matches("data: ").trim_end();
        let v: Value = serde_json::from_str(json).unwrap();
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "");
        assert!(v.get("usage").is_none(), "include_usage off ⇒ no usage key");
        assert_eq!(SSE_DONE, "data: [DONE]\n\n");
    }

    #[test]
    fn sse_include_usage_puts_null_on_content_and_object_on_final() {
        let ctx = ChunkCtx {
            id: "chatcmpl-x".into(),
            created: 1,
            model: "m".into(),
            include_usage: true,
        };
        // Content chunk carries "usage": null.
        let v: Value = serde_json::from_str(
            sse_frame(&ctx.segment_chunk(&Segment::Content("hi".into())))
                .trim_start_matches("data: ")
                .trim_end(),
        )
        .unwrap();
        assert!(v.as_object().unwrap().contains_key("usage"));
        assert_eq!(v["usage"], Value::Null);
        assert_eq!(v["choices"][0]["delta"]["content"], "hi");
        // Final usage chunk: empty choices, usage object.
        let f: Value = serde_json::from_str(
            sse_frame(&ctx.usage_chunk(usage(3, 4)))
                .trim_start_matches("data: ")
                .trim_end(),
        )
        .unwrap();
        assert_eq!(f["choices"].as_array().unwrap().len(), 0);
        assert_eq!(f["usage"]["total_tokens"], 7);
    }

    #[test]
    fn sse_finish_chunk_has_empty_delta_and_reason() {
        let ctx = ChunkCtx {
            id: "chatcmpl-x".into(),
            created: 1,
            model: "m".into(),
            include_usage: false,
        };
        let v: Value = serde_json::from_str(
            sse_frame(&ctx.finish_chunk("stop"))
                .trim_start_matches("data: ")
                .trim_end(),
        )
        .unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert!(v["choices"][0]["delta"].as_object().unwrap().is_empty());
    }

    #[test]
    fn segment_chunk_reasoning_field() {
        let ctx = ChunkCtx {
            id: "x".into(),
            created: 1,
            model: "m".into(),
            include_usage: false,
        };
        let v: Value = serde_json::from_str(
            sse_frame(&ctx.segment_chunk(&Segment::Reasoning("why".into())))
                .trim_start_matches("data: ")
                .trim_end(),
        )
        .unwrap();
        assert_eq!(v["choices"][0]["delta"]["reasoning_content"], "why");
        assert!(v["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn gen_id_is_prefixed_16_hex() {
        let id = gen_id("chatcmpl", 1, 2);
        assert!(id.starts_with("chatcmpl-"));
        let hex = id.trim_start_matches("chatcmpl-");
        assert_eq!(hex.len(), 16);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn load_sampling_defaults_reads_generation_config() {
        // The in-repo 30B instruct dir ships temp 0.7 / top_p 0.8 / top_k 20.
        let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("models/qwen3-30b-a3b-instruct-2507");
        if dir.join("generation_config.json").exists() {
            let d = load_sampling_defaults(&dir);
            assert!((d.temperature - 0.7).abs() < 1e-6, "temp {}", d.temperature);
            assert!((d.top_p - 0.8).abs() < 1e-6, "top_p {}", d.top_p);
            assert_eq!(d.top_k, 20);
        }
    }

    #[test]
    fn load_sampling_defaults_missing_dir_falls_back() {
        let d = load_sampling_defaults(Path::new("/nonexistent/model/dir"));
        assert_eq!(d, SamplingDefaults::default());
    }
}
