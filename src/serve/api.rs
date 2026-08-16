//! S.0: OpenAI-compatible request/response types (serde) + axum app skeleton.
//!
//! Field names follow the OpenAI spec; unknown fields are tolerated via a
//! flattened `extra` map (e.g. Qwen's `enable_thinking` arrives this way and
//! MUST be preserved, not dropped). See docs/SERVE_PLAN.md S.0.
//!
//! Re-serialization is STRUCTURALLY faithful, not byte-faithful: unknown fields
//! are preserved (via the flattened `extra` maps), but the wire form is
//! normalized — explicit JSON `null`s on modeled optional fields are dropped
//! (they deserialize to `None`, which `skip_serializing_if` omits), and key
//! order is not preserved (flattened `extra` fields re-emit AFTER the modeled
//! fields). Round-tripping a request yields an equivalent, not identical, body.
//!
//! INVARIANTS:
//! - Every optional request field is `Option<_>` with `skip_serializing_if` so
//!   re-serialized requests stay minimal (absent/null optionals never re-emit).
//! - `stop` and `prompt`/`content` polymorphism (string OR array) is modeled
//!   with `#[serde(untagged)]` enums — order matters (string variant first so a
//!   bare string never mis-parses as a one-element array).
//! - Fixed discriminator strings (`object`: "chat.completion", "list", ...) are
//!   plain `String` fields set at construction, not enums — real OpenAI JSON
//!   round-trips through them unchanged.
//! - This module is ENGINE-FREE: `base_router()` wires only `/health`; the `/v1`
//!   endpoints are added in S.4. No `cuda`/model deps here (host-testable).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

// ---------------------------------------------------------------------------
// Shared polymorphic helpers
// ---------------------------------------------------------------------------

/// A field that is EITHER a single string OR an array of strings (OpenAI models
/// `stop` and legacy-completions `prompt` this way). String variant is listed
/// first so untagged deserialization prefers it for a bare JSON string.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    Single(String),
    Multiple(Vec<String>),
}

/// Legacy `/v1/completions` `prompt`: the OpenAI spec allows FOUR shapes — a
/// string, an array of strings, an array of token ids, or an array of token-id
/// arrays (batched pre-tokenized prompts). String variants are listed FIRST so
/// untagged deserialization prefers them (a bare string / array-of-strings never
/// mis-parses as tokens). S.4 accepts the string forms and returns a clean 400
/// for the token forms; modeling them here just moves that rejection out of the
/// serde layer (a raw `{"prompt":[123,456]}` no longer fails to deserialize).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Prompt {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenBatches(Vec<Vec<u32>>),
}

/// Chat message `content`: a plain string OR an array of typed content parts
/// (text/image/etc.). String variant first (see `StringOrArray`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

/// One element of a multimodal `content` array. We keep only the discriminator
/// (`type`) and `text` explicit; everything else (e.g. `image_url`) is retained
/// in `extra` so unknown part shapes round-trip losslessly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

/// Request-side chat message. `content` may be absent (e.g. an assistant turn
/// carrying only tool_calls), so it is `Option`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Unknown message fields (tool_calls, tool_call_id, reasoning_content on an
    /// echoed history turn, ...) preserved verbatim.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — request
// ---------------------------------------------------------------------------

/// `stream_options` object; only `include_usage` is honored in v1 (emit the
/// trailing usage-only chunk).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_usage: Option<bool>,
}

/// OpenAI `POST /v1/chat/completions` request body. Unknown fields (notably
/// Qwen's `enable_thinking`) land in `extra` and MUST NOT be dropped — the chat
/// template reads them in S.4.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Non-standard OpenAI, but accepted (Qwen sampling parity).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Newer spelling of `max_tokens`; either may be sent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Signed: OpenAI/vLLM accept negative seeds (e.g. `-1`), so this is `i64`
    /// not `u64` (a `u64` field rejects `{"seed":-1}` at the serde layer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    /// Everything the OpenAI spec (or a client extension like `enable_thinking`)
    /// sends that we do not model explicitly. Preserved for round-trip + the
    /// template layer.
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — non-stream response
// ---------------------------------------------------------------------------

/// Token accounting echoed on every non-stream response and on the trailing
/// usage chunk when `stream_options.include_usage` is set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Assistant message on a completed (non-stream) choice. `reasoning_content`
/// carries the `<think>...</think>` split (mirrors vLLM's qwen3 reasoning
/// parser); it is omitted when empty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResponseMessage {
    pub role: String,
    /// Present but may be JSON `null` if the whole answer was reasoning; kept
    /// `Option` and always serialized (no skip) to match OpenAI's shape.
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    /// Always "chat.completion".
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
}

// ---------------------------------------------------------------------------
// /v1/chat/completions — streaming chunks (object = "chat.completion.chunk")
// ---------------------------------------------------------------------------

/// Incremental delta on a streaming choice. `role` appears only on the FIRST
/// chunk; `content`/`reasoning_content` carry the incremental text (never both
/// non-empty in the same chunk — the think-boundary state machine routes one or
/// the other). All three are `Option` + skipped when absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Delta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: Delta,
    pub finish_reason: Option<String>,
}

/// One SSE `data:` payload. The final chunk (when usage is requested) has an
/// EMPTY `choices` array and `usage` set; all prior chunks have empty usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    /// Always "chat.completion.chunk".
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    /// THREE states, per the `stream_options.include_usage` spec:
    /// - `None` → key ABSENT (include_usage off: no usage on any chunk).
    /// - `Some(None)` → serializes `"usage": null` (include_usage ON: every
    ///   intermediate content chunk carries an explicit null usage).
    /// - `Some(Some(u))` → serializes `"usage": {...}` (the final usage-only
    ///   chunk, which also has an empty `choices` array).
    ///
    /// The outer `Option` is `skip_serializing_if`-gated so `None` drops the key;
    /// `Some(None)` intentionally emits JSON `null` (the inner `Option<Usage>`
    /// serializes as null).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Option<Usage>>,
}

// ---------------------------------------------------------------------------
// /v1/completions (legacy)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    /// A raw prompt string, an array of prompt strings, or (rejected in S.4)
    /// pre-tokenized ids. See [`Prompt`].
    pub prompt: Prompt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop: Option<StringOrArray>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    /// Always "text_completion".
    pub object: String,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    /// Present on the non-stream response and the FINAL stream chunk; omitted on
    /// intermediate stream chunks (OpenAI legacy streaming does not carry usage
    /// per delta — review finding).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

// ---------------------------------------------------------------------------
// /v1/models
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    /// Always "model".
    pub object: String,
    pub created: u64,
    pub owned_by: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    /// Always "list".
    pub object: String,
    pub data: Vec<Model>,
}

// ---------------------------------------------------------------------------
// Error body: {"error": {message, type, param, code}}
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
    /// The offending parameter name, or null.
    pub param: Option<String>,
    /// A short machine code, or null.
    pub code: Option<String>,
}

/// OpenAI error envelope. Serializes as `{"error": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApiError {
    pub error: ApiErrorBody,
}

impl ApiError {
    /// Build an error envelope. `param`/`code` are optional (pass `None` for the
    /// OpenAI-conventional JSON `null`).
    pub fn new(
        message: impl Into<String>,
        kind: impl Into<String>,
        param: Option<String>,
        code: Option<String>,
    ) -> Self {
        ApiError {
            error: ApiErrorBody {
                message: message.into(),
                kind: kind.into(),
                param,
                code,
            },
        }
    }

    /// A 400-style `invalid_request_error` — the common case for unsupported
    /// params (n>1, logprobs) and length overflows.
    pub fn invalid_request(message: impl Into<String>, param: Option<String>) -> Self {
        ApiError::new(message, "invalid_request_error", param, None)
    }
}

// ---------------------------------------------------------------------------
// axum app skeleton — /health only (the /v1 routes are wired in S.4)
// ---------------------------------------------------------------------------

/// Engine-free router with `GET /health` → `200 {"status":"ok"}`. The `/v1`
/// endpoints are layered on in S.4 once the engine handle exists; keeping this
/// skeleton dependency-free lets the HTTP surface be smoke-tested standalone.
pub fn base_router() -> axum::Router {
    axum::Router::new().route("/health", axum::routing::get(health))
}

async fn health() -> axum::Json<Value> {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

// ---------------------------------------------------------------------------
// Tests: serde round-trip fixtures cribbed from real OpenAI request/response
// JSON. These pin the wire shape (S.5 gate a) and the unknown-field tolerance.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Unknown fields (here Qwen's `enable_thinking` plus an OpenAI param we do
    /// not model) MUST survive into `extra`, not be silently dropped.
    #[test]
    fn chat_request_unknown_fields_preserved_in_extra() {
        let raw = r#"{
            "model": "qwen3-30b",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "enable_thinking": false,
            "response_format": {"type": "text"}
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.model, "qwen3-30b");
        assert_eq!(req.temperature, Some(0.7));
        // Unknown fields tolerated AND preserved.
        assert_eq!(req.extra.get("enable_thinking"), Some(&Value::Bool(false)));
        assert!(req.extra.contains_key("response_format"));
        // And they re-serialize (flatten round-trips them back out).
        let out = serde_json::to_value(&req).unwrap();
        assert_eq!(out["enable_thinking"], Value::Bool(false));
        assert!(out.get("temperature").is_some());
    }

    /// `seed` is signed: OpenAI/vLLM accept negatives (e.g. `-1`), which a
    /// `u64` field would reject at deserialization.
    #[test]
    fn chat_request_negative_seed() {
        let raw = r#"{"model":"m","messages":[],"seed":-1}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.seed, Some(-1));
        // Positive seeds still work and round-trip.
        let raw2 = r#"{"model":"m","messages":[],"seed":42}"#;
        let req2: ChatCompletionRequest = serde_json::from_str(raw2).unwrap();
        assert_eq!(req2.seed, Some(42));
        assert_eq!(serde_json::to_value(&req2).unwrap()["seed"], 42);
    }

    #[test]
    fn chat_request_stop_as_string() {
        let raw = r#"{"model":"m","messages":[],"stop":"\n\n"}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.stop, Some(StringOrArray::Single("\n\n".to_string())));
    }

    #[test]
    fn chat_request_stop_as_array() {
        let raw = r#"{"model":"m","messages":[],"stop":["\n","<|im_end|>"]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(
            req.stop,
            Some(StringOrArray::Multiple(vec![
                "\n".to_string(),
                "<|im_end|>".to_string()
            ]))
        );
    }

    #[test]
    fn chat_request_content_as_string() {
        let raw = r#"{"model":"m","messages":[{"role":"user","content":"hello"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(
            req.messages[0].content,
            Some(MessageContent::Text("hello".to_string()))
        );
    }

    #[test]
    fn chat_request_content_as_parts() {
        let raw = r#"{
            "model":"m",
            "messages":[{"role":"user","content":[
                {"type":"text","text":"describe"},
                {"type":"image_url","image_url":{"url":"http://x/y.png"}}
            ]}]
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(raw).unwrap();
        match req.messages[0].content.as_ref().unwrap() {
            MessageContent::Parts(parts) => {
                assert_eq!(parts.len(), 2);
                assert_eq!(parts[0].kind, "text");
                assert_eq!(parts[0].text.as_deref(), Some("describe"));
                // Unknown part payload preserved.
                assert_eq!(parts[1].kind, "image_url");
                assert!(parts[1].extra.contains_key("image_url"));
            }
            other => panic!("expected parts, got {other:?}"),
        }
        // Parts survive re-serialization.
        let out = serde_json::to_value(&req).unwrap();
        assert!(out["messages"][0]["content"].is_array());
    }

    #[test]
    fn chat_response_non_stream_roundtrip() {
        let raw = r#"{
            "id":"chatcmpl-abc",
            "object":"chat.completion",
            "created":1720000000,
            "model":"qwen3-30b",
            "choices":[{
                "index":0,
                "message":{"role":"assistant","content":"The answer is 4.","reasoning_content":"2+2"},
                "finish_reason":"stop"
            }],
            "usage":{"prompt_tokens":10,"completion_tokens":6,"total_tokens":16}
        }"#;
        let resp: ChatCompletionResponse = serde_json::from_str(raw).unwrap();
        assert_eq!(resp.object, "chat.completion");
        assert_eq!(resp.choices[0].finish_reason.as_deref(), Some("stop"));
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("The answer is 4.")
        );
        assert_eq!(
            resp.choices[0].message.reasoning_content.as_deref(),
            Some("2+2")
        );
        assert_eq!(resp.usage.total_tokens, 16);
        // Re-serialize and compare structurally (field order-independent).
        let back = serde_json::to_value(&resp).unwrap();
        let orig: Value = serde_json::from_str(raw).unwrap();
        assert_eq!(back, orig);
    }

    /// First streaming chunk carries `delta.role` and no finish_reason.
    #[test]
    fn chat_chunk_first_delta_role() {
        let raw = r#"{
            "id":"chatcmpl-abc",
            "object":"chat.completion.chunk",
            "created":1720000000,
            "model":"qwen3-30b",
            "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]
        }"#;
        let chunk: ChatCompletionChunk = serde_json::from_str(raw).unwrap();
        assert_eq!(chunk.object, "chat.completion.chunk");
        assert_eq!(chunk.choices[0].delta.role.as_deref(), Some("assistant"));
        assert!(chunk.choices[0].delta.content.is_none());
        assert!(chunk.choices[0].finish_reason.is_none());
        assert!(chunk.usage.is_none());
        // A content-only delta serializes WITHOUT a role key.
        let content_chunk = ChatCompletionChunk {
            id: "x".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("Hello".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage: None,
        };
        let v = serde_json::to_value(&content_chunk).unwrap();
        assert!(v["choices"][0]["delta"].get("role").is_none());
        assert_eq!(v["choices"][0]["delta"]["content"], "Hello");
        // include_usage OFF (usage: None) ⇒ the key is ABSENT entirely.
        assert!(v.get("usage").is_none());
    }

    /// The THREE usage states of a streaming chunk (stream_options.include_usage):
    /// absent key, explicit `null` on intermediate chunks, object on the final.
    #[test]
    fn chat_chunk_usage_three_states() {
        let base = |usage: Option<Option<Usage>>| ChatCompletionChunk {
            id: "x".into(),
            object: "chat.completion.chunk".into(),
            created: 1,
            model: "m".into(),
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta {
                    content: Some("Hi".into()),
                    ..Default::default()
                },
                finish_reason: None,
            }],
            usage,
        };

        // (1) include_usage OFF ⇒ no usage key at all.
        let off = serde_json::to_value(base(None)).unwrap();
        assert!(off.get("usage").is_none(), "None must drop the key");

        // (2) include_usage ON, intermediate content chunk ⇒ "usage": null.
        let mid = serde_json::to_value(base(Some(None))).unwrap();
        assert!(mid.get("usage").is_some(), "Some(None) must emit the key");
        assert_eq!(mid["usage"], Value::Null, "Some(None) serializes as null");

        // NOTE: this type is SERVER-serialized (the wire contract is emission,
        // not client-round-trip). On DESERIALIZE serde collapses both absent and
        // explicit-null into the outer `None` — the classic double-`Option`
        // limitation — so `null` reads back as `None`, not `Some(None)`. That is
        // fine here: nothing consumes a chunk we produced back into this struct.
        let back: ChatCompletionChunk = serde_json::from_value(mid).unwrap();
        assert_eq!(
            back.usage, None,
            "serde collapses null → outer None on read"
        );

        // (3) final usage-only chunk ⇒ "usage": {...}.
        let fin = serde_json::to_value(base(Some(Some(Usage {
            prompt_tokens: 1,
            completion_tokens: 2,
            total_tokens: 3,
        }))))
        .unwrap();
        assert_eq!(fin["usage"]["total_tokens"], 3);
    }

    /// The trailing usage chunk has an EMPTY choices array and usage set.
    #[test]
    fn chat_chunk_final_usage_empty_choices() {
        let chunk = ChatCompletionChunk {
            id: "chatcmpl-abc".into(),
            object: "chat.completion.chunk".into(),
            created: 1720000000,
            model: "qwen3-30b".into(),
            choices: vec![],
            usage: Some(Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 6,
                total_tokens: 16,
            })),
        };
        let v = serde_json::to_value(&chunk).unwrap();
        assert_eq!(v["choices"].as_array().unwrap().len(), 0);
        assert_eq!(v["usage"]["total_tokens"], 16);
        // Round-trips back.
        let back: ChatCompletionChunk = serde_json::from_value(v).unwrap();
        assert_eq!(back, chunk);
    }

    #[test]
    fn completion_roundtrip() {
        let raw = r#"{
            "model":"qwen3-30b",
            "prompt":"Once upon a time",
            "max_tokens":16,
            "temperature":0.0,
            "stop":["\n"]
        }"#;
        let req: CompletionRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.prompt, Prompt::Text("Once upon a time".into()));
        assert_eq!(req.max_tokens, Some(16));

        let resp = CompletionResponse {
            id: "cmpl-1".into(),
            object: "text_completion".into(),
            created: 1720000000,
            model: "qwen3-30b".into(),
            choices: vec![CompletionChoice {
                text: ", there was".into(),
                index: 0,
                finish_reason: Some("length".into()),
            }],
            usage: Some(Usage {
                prompt_tokens: 4,
                completion_tokens: 3,
                total_tokens: 7,
            }),
        };
        let v = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["object"], "text_completion");
        assert_eq!(v["choices"][0]["text"], ", there was");
        let back: CompletionResponse = serde_json::from_value(v).unwrap();
        assert_eq!(back, resp);
    }

    /// All four `prompt` shapes deserialize (untagged, string variants first).
    /// The token forms are rejected later in S.4, not at the serde layer.
    #[test]
    fn completion_prompt_variants() {
        let text: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":"hi"}"#).unwrap();
        assert_eq!(text.prompt, Prompt::Text("hi".into()));

        let texts: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":["a","b"]}"#).unwrap();
        assert_eq!(texts.prompt, Prompt::Texts(vec!["a".into(), "b".into()]));

        // A bare id array parses as Tokens (this previously failed to deserialize).
        let tokens: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":[123,456]}"#).unwrap();
        assert_eq!(tokens.prompt, Prompt::Tokens(vec![123, 456]));

        let batches: CompletionRequest =
            serde_json::from_str(r#"{"model":"m","prompt":[[1,2],[3,4]]}"#).unwrap();
        assert_eq!(
            batches.prompt,
            Prompt::TokenBatches(vec![vec![1, 2], vec![3, 4]])
        );
    }

    #[test]
    fn error_body_shape() {
        let err = ApiError::invalid_request("n > 1 is not supported", Some("n".into()));
        let v = serde_json::to_value(&err).unwrap();
        assert_eq!(v["error"]["message"], "n > 1 is not supported");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert_eq!(v["error"]["param"], "n");
        assert_eq!(v["error"]["code"], Value::Null);
        // Deserializes from the canonical OpenAI error JSON too.
        let raw = r#"{"error":{"message":"bad","type":"invalid_request_error","param":null,"code":null}}"#;
        let back: ApiError = serde_json::from_str(raw).unwrap();
        assert_eq!(back.error.kind, "invalid_request_error");
        assert!(back.error.param.is_none());
    }

    #[test]
    fn models_list_shape() {
        let list = ModelList {
            object: "list".into(),
            data: vec![Model {
                id: "qwen3-30b".into(),
                object: "model".into(),
                created: 1720000000,
                owned_by: "local".into(),
            }],
        };
        let v = serde_json::to_value(&list).unwrap();
        assert_eq!(v["object"], "list");
        assert_eq!(v["data"][0]["id"], "qwen3-30b");
        assert_eq!(v["data"][0]["object"], "model");
        let raw = r#"{"object":"list","data":[{"id":"m","object":"model","created":1,"owned_by":"local"}]}"#;
        let back: ModelList = serde_json::from_str(raw).unwrap();
        let reser: ModelList =
            serde_json::from_value(serde_json::to_value(&back).unwrap()).unwrap();
        assert_eq!(back, reser);
    }

    /// The skeleton router builds without an engine (S.4 wires /v1 later).
    #[test]
    fn base_router_builds() {
        let _router = base_router();
    }
}
