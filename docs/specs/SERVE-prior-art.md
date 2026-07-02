# SERVE prior art

> Prior-art survey for an OpenAI-compatible, single-stream LLM server in Rust
> (HTTP + SSE front-end over a Burn inference engine). Section A (our design /
> integration notes) is appended separately. This file covers **section B: the
> Rust ecosystem choices**, with what the two most-cribbed reference servers
> (`mistral.rs`, `candle-vllm`) actually use.

<!-- Section A appended separately -->

## B. Ecosystem

### B1. HTTP + SSE stack recommendation

**Recommendation: `axum` + `tower-http` (CORS) + `tokio`, with SSE via `axum::response::sse::{Sse, Event, KeepAlive}`.**

This is the de-facto standard for LLM serving in Rust and is exactly what both
reference engines use:

- **mistral.rs** (`mistralrs-server-core`): depends on `axum` (features
  `tokio`, `multipart`), `tower-http` (feature `cors`), `tokio`, `serde` /
  `serde_json`, and `utoipa` + `utoipa-swagger-ui` for OpenAPI/Swagger docs.
  Streaming lives in a dedicated `streaming.rs` module built around a generic
  `BaseStreamer<R, C, D>` that implements `futures::Stream` and a `DoneState`
  enum (`Running → SendingDone → Done`). Keep-alive is configurable via a
  `KEEP_ALIVE_INTERVAL` env var (default 10 s).
- **candle-vllm**: `axum = 0.8.9`, `tower-http = 0.6.6` (cors), `hyper = 0.14`
  (full), `tokio = 1.38` (sync), `utoipa = 4.2` (axum_extras). Actix-web is
  present but **commented out** in `Cargo.toml` — they chose axum. Its
  `Streamer` implements `Stream<Item = Result<Event, axum::Error>>` directly and
  yields `axum::response::sse::Event`.

Why axum over actix-web / raw hyper for a single-stream server:

- `axum::response::sse` gives you `Sse<S>` (an `IntoResponse` wrapping any
  `Stream<Item = Result<Event, E>>`), `Event::default().json_data(x)` /
  `.data("[DONE]")`, and `KeepAlive::new().interval(..)` out of the box — no
  hand-rolled `text/event-stream` framing, correct `data:`/`\n\n` handling for
  free.
- Tower middleware ecosystem (CORS, timeout, concurrency-limit, trace) composes
  cleanly; a single-stream server especially wants
  `tower::limit::ConcurrencyLimit` (or a semaphore) since the engine serves one
  decode stream at a time.
- Backed by `hyper` 1.x / `tokio`; async-native, cancel-safe (client disconnect
  drops the `Stream`, which you can observe to stop generation).

Versions/licenses: `axum` 0.8.x (MIT), `tower-http` 0.6.x (MIT), `tokio` 1.x
(MIT), `hyper` 1.x (MIT).

### B2. minijinja fitness for Qwen chat templates

**`minijinja` is the right choice and is what the ecosystem uses — but you must
wire up the Python-compatibility shims or Qwen templates will crash.** candle-vllm
depends on `minijinja = 2.10.2` (features `builtins`, `json`) **plus**
`minijinja-contrib = 2.10.2` (feature `pycompat`) and `either` (serde). That
combination is the tested recipe.

Required wiring:

- **`minijinja-contrib` `pycompat` → `env.set_unknown_method_callback(
  minijinja_contrib::pycompat::unknown_method_callback)`.** HF chat templates
  call Python *string/list methods* (`.strip()`, `.split()`, `.startswith()`,
  `.items()`, `.rstrip()`, etc.) that minijinja does not implement natively. The
  pycompat callback intercepts unknown method calls on primitives and provides
  the Python semantics. Without it, Qwen/Llama/Gemma templates raise
  "unknown method" errors.
- **`tojson` filter** — needed for tool-calling templates (serialising the
  `tools` array / function arguments). Provided by the `json` feature; already
  enabled above.
- **`strftime_now(fmt)`** — newer Qwen/Llama templates embed a current-date
  string via this global. It is **not** built in; register it yourself with
  `env.add_function("strftime_now", |fmt: String| { chrono/time now → fmt })`.
- **`raise_exception(msg)`** — templates call this to reject malformed message
  sequences. Not built in; register `env.add_function("raise_exception", |m|
  Err(minijinja::Error::new(InvalidOperation, m)))`. (This is the exact class of
  bug tracked in llama.cpp #11402 / #11866 for the C++ `minja` runtime.)
- Set `undefined_behavior(UndefinedBehavior::Chainable)` and keep whitespace
  trimming (`trim_blocks`/`lstrip_blocks` semantics) matching Jinja2, or the
  emitted prompt string will drift from HF's reference by stray newlines —
  which silently changes tokenisation.

Known Qwen-family gotchas (from HF/llama.cpp issue trackers):

- The **official Qwen3.5 tool-calling template is buggy** (upstream discussion
  "tool calling chat template is broken"); community "fixed" drop-in templates
  exist (`froggeric/Qwen-Fixed-Chat-Templates`). Plan to ship a
  vetted/patched template rather than blindly trusting `tokenizer_config.json`.
- **Thinking mode**: Qwen3.x templates gate `<think>` blocks on template logic;
  substituting a generic `chatml` template silently disables thinking mode.
  Preserve the model's own template.
- **`loop.previtem`/`loop.nextitem`** and other loop-state accessors historically
  crashed older minijinja/minja; on `minijinja` 2.10 the loop object is
  supported, but pin ≥ 2.10 to be safe.
- The `replace` filter had a payload-dropping bug in the C++ `minja` (not Rust
  minijinja) — a reminder to **parity-test the rendered prompt against HF
  `apply_chat_template`** on a fixture set, byte-for-byte, in CI.

Versions/licenses: `minijinja` / `minijinja-contrib` 2.10.x (Apache-2.0).

### B3. OpenAI API schema: crates vs hand-rolled serde

**Recommendation: hand-roll the request/response structs with `serde` (crib
field-for-field from `async-openai`'s types), rather than depending on a client
crate.**

Findings:

- **`async-openai`** (v0.41.x, MIT) is the dominant crate, but it is
  **client-focused** — builder pattern, retk with exponential backoff, reqwest
  transport. Its *types* are good and feature-gated (`types`,
  `chat-completion-types`, etc.), and it supports `byot` ("bring your own
  types") with `serde_json::Value`. You can depend on it purely for the type
  definitions, but you inherit a large client surface (reqwest, backoff) you
  don't need on the server side.
- Neither reference engine depends on `async-openai` for its server types. **mistral.rs
  uses its own `openai-protocol` crate (1.6.x)**; **candle-vllm hand-rolls**
  request/response structs in `src/openai/requests.rs` and `responses.rs`.
- Rationale for hand-rolling: an OpenAI-*compatible* server only needs a small,
  stable subset (`/v1/chat/completions`, `/v1/completions`, `/v1/models`), and
  you frequently need **extra non-standard fields** (`top_k`, `repetition_penalty`,
  `min_p`, engine-specific `stream_options`) that a strict client type rejects.
  `#[serde(default)]` + `#[serde(flatten)] extra: Map<String,Value>` on your own
  structs is cleaner than fighting a client crate's shape. Use
  `#[serde(skip_serializing_if = "Option::is_none")]` so absent fields (e.g.
  `usage: null`, `system_fingerprint`) match OpenAI's wire format.
- If you want a typed contract + generated OpenAPI docs, add `utoipa`
  (`ToSchema` derives) as both engines do — this gives `/docs` Swagger UI cheaply.

Type crates surveyed: `async-openai` (MIT), `async-openai-wasm` (MIT), and the
smaller `openai-types` / `openai-core` — none is a clear server-side standard,
which is why hand-rolling wins for a compatibility layer.

### B4. Reference server implementations worth cribbing

Both are by Eric Buehler; both are axum + serde + minijinja and MIT-licensed.

**mistral.rs** — `mistralrs-server-core` (MIT):
- Chat completions handler: `mistralrs-server-core/src/chat_completion.rs`
  <https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-server-core/src/chat_completion.rs>
- SSE streaming (BaseStreamer, DoneState, keep-alive):
  `mistralrs-server-core/src/streaming.rs`
  <https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-server-core/src/streaming.rs>
- Shared completion core: `.../src/completion_core.rs`; routing:
  `.../src/handlers.rs`; response shaping: `.../src/responses.rs`
- Notable: also exposes Anthropic-compatible `/v1/messages` — useful reference
  if you want dual compat later.

**candle-vllm** — `src/openai/` (MIT):
- Server / route wiring + request handling: `src/openai/openai_server.rs`
  <https://github.com/EricLBuehler/candle-vllm/blob/master/src/openai/openai_server.rs>
- SSE streaming (`Streamer: Stream<Item=Result<Event, axum::Error>>`, `[DONE]`
  via `Event::default().data("[DONE]")`, chunks via `.json_data(response)`):
  `src/openai/streaming.rs`
  <https://github.com/EricLBuehler/candle-vllm/blob/master/src/openai/streaming.rs>
- Request/response schema: `src/openai/requests.rs`, `src/openai/responses.rs`
- Chat-template / conversation formatting: `src/openai/conversation/` (minijinja
  + pycompat lives here)

Both are the closest analogues to our task (Candle/Burn-family tensor engine +
axum OpenAI front-end); candle-vllm is the leaner, more directly cribbable of the
two for a single-stream server.

Other references (lower priority): `vllm-project/vllm`'s Rust frontend
(`vllm-frontend-rs`, axum) and `litellm-rs` (gateway, not an engine).

### B5. `/v1/chat/completions` SSE chunk format

Content-Type `text/event-stream`; each SSE message is `data: <json>\n\n`. Each
JSON object is a **`chat.completion.chunk`**:

```jsonc
{
  "id": "chatcmpl-...",              // stable across all chunks of one response
  "object": "chat.completion.chunk", // constant
  "created": 1751000000,             // unix seconds
  "model": "qwen3.5-35b",
  "system_fingerprint": "fp_...",    // optional; emit a stable engine build id
  "choices": [
    {
      "index": 0,
      "delta": {                     // incremental; NOT "message"
        "role": "assistant",         // only on the FIRST chunk
        "content": "Paris"           // token text; may be "" or absent
        // "tool_calls": [...]       // for function calling, streamed by index
      },
      "logprobs": null,              // or a logprobs object if requested
      "finish_reason": null          // null until the terminating chunk
    }
  ],
  "usage": null                      // null on every chunk unless include_usage
}
```

Sequence and framing rules:

1. **First chunk** carries `delta.role = "assistant"` (usually empty content).
2. **Middle chunks** carry `delta.content` token fragments; `finish_reason: null`.
3. **Final content chunk** sets `delta: {}` and `finish_reason` to one of:
   `"stop"` (natural end / stop sequence), `"length"` (hit `max_tokens`),
   `"tool_calls"` (function invocation; legacy alias `"function_call"`),
   `"content_filter"`.
4. **Usage**: only emitted when the request sets
   `stream_options: {"include_usage": true}`. Then send one extra chunk **after**
   the finish-reason chunk whose `choices` is `[]` and whose `usage` is
   `{prompt_tokens, completion_tokens, total_tokens}`. On all earlier chunks
   `usage` is `null` (omit it otherwise).
5. **Terminate** the stream with a literal sentinel line: `data: [DONE]\n\n`
   (this is **not** JSON — clients must string-match it). candle-vllm does
   exactly `Event::default().data("[DONE]")`.

Implementation notes for axum: build each event as
`Event::default().json_data(&chunk)?` and the terminator as
`Event::default().data("[DONE]")`; wrap the whole `Stream` in
`Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))`.
Watch client disconnects (the `Stream` is dropped) to cancel the decode loop.

Sources / versions / licenses:
- OpenAI streaming reference (chat.completion.chunk fields, `[DONE]`,
  finish_reason values):
  <https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events>,
  <https://platform.openai.com/docs/api-reference/chat-streaming/streaming>
- mistral.rs `mistralrs-server-core` (MIT), axum + tower-http + minijinja + utoipa:
  <https://github.com/EricLBuehler/mistral.rs>
- candle-vllm (MIT), axum 0.8.9 + minijinja 2.10.2 (+pycompat) + hyper 0.14:
  <https://github.com/EricLBuehler/candle-vllm>
- async-openai v0.41.x (MIT), client-focused OpenAI types:
  <https://github.com/64bit/async-openai>
- minijinja / minijinja-contrib pycompat (Apache-2.0):
  <https://docs.rs/minijinja/latest/minijinja/>,
  <https://docs.rs/minijinja-contrib/latest/minijinja_contrib/pycompat/index.html>
- axum SSE: <https://docs.rs/axum/latest/axum/response/sse/index.html>
- Qwen template gotchas: <https://huggingface.co/Qwen/Qwen3.5-35B-A3B/discussions/4>,
  <https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates>,
  llama.cpp jinja crash issues #11402 / #11866

### Recommended stack

- **HTTP framework** — `axum` 0.8.x (MIT): ecosystem standard; both reference engines use it; native SSE.
- **SSE** — `axum::response::sse::{Sse, Event, KeepAlive}`: correct `text/event-stream` framing + keep-alive for free.
- **Middleware/runtime** — `tower-http` (CORS) + `tokio` + `hyper` 1.x: cancel-safe streaming, composable limits.
- **Chat templates** — `minijinja` 2.10+ with `minijinja-contrib` `pycompat` (Apache-2.0): only Jinja engine that runs real HF/Qwen templates once the callback + `strftime_now`/`raise_exception` shims are registered.
- **OpenAI schema** — hand-rolled `serde` structs (crib field names from `async-openai`), `#[serde(flatten)] extra` for engine params: avoids client-crate bloat, allows non-standard fields.
- **API docs (optional)** — `utoipa` + `utoipa-swagger-ui`: typed schema + `/docs` for free, as both engines do.
- **Crib target** — `candle-vllm` `src/openai/` (MIT): leanest axum single-stream OpenAI server closest to a Candle/Burn engine.

## A. Pieces already in this repo (Explore inventory, 2026-07-02)
- **Engine objects**: `examples/vllm_infer.rs:112-118` Llm (load-once: model+tokenizer+device+eos) and :259 CapturedLlm. `generate()` :187-253 is the per-request template (tokenize -> prefill -> step loop -> EOS -> text/tok/s). **WARNING: its generate_capture rebuilds the CUDA graph per prompt** — do NOT lift; the efficient pattern is `examples/cudagraph_qwen35_decode_bench.rs:183-223` reset_and_prefill + capture-ONCE (:584) + graph.replay() per request (proven across prompt1/prompt2).
- **SamplingParams** :47-67 (max_tokens/temperature/top_p/top_k + defaults; chat defaults :497 temp 0.7/top_p 0.8/top_k 20). Missing for OpenAI compat: seed, stop, penalties, n, logprobs. Self-contained xorshift Rng :73-92.
- **Bucket constraint**: one captured graph per (BATCH=1, T_MAX); `prompt_len+max_new<=T_MAX` asserted (:192-197). Repo has a single T_MAX=1024 today; no multi-bucket.
- **Tokenizer** (`src/tokenizer.rs`): encode_no_pad/decode/decode_i64/vocab_size/token_to_id. **No incremental streaming decode** (needs decode-and-diff with UTF-8 holdback). `apply_chat_template` :46 is a fake single-turn stub. `eos_token_id` falls back to 151643 (30B).
- **35B specials (both 35B dirs byte-identical)**: eos=<|im_end|>(248046), pad=<|endoftext|>(248044), im_start=248045, think=248068/248069, add_bos=false. **vllm_infer's hardcoded EOS [151643,151645] is WRONG for the 35B** — read from tokenizer_config.json.
- **chat_template.jinja** (35B): macros, namespace state, messages[::-1], tojson/trim/items filters, think-tag extraction, enable_thinking/preserve_thinking, tool-call XML — requires a real Jinja engine (minijinja+pycompat), unreproducible by string formatting.
- **Sampling**: host `sample_index` (top-k->top-p->CDF; O(V log V)); device greedy argmax + UNFILTERED Gumbel (`sampling_device.rs`, not capture-safe for sampling). Captured decode is greedy-only; temp>0 => eager-static path.
- **No async/HTTP/serde deps in [dependencies]; no server code anywhere.** Model types are not Sync — single engine thread + mpsc is the natural fit.
