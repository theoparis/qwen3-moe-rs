# M-S: OpenAI-compatible single-stream server (`qwen-serve`) — plan v1

Scope locked with the user (D1 → option A; refined 2026-07-02): an OpenAI-compatible HTTP
server for EXACTLY the two proven models — **Qwen3-30B-A3B** (bf16, captured 20.9 tok/s) and
**Qwen3.6-35B-A3B** (bf16 / fp8 / nvfp4) — selected at launch, one model per process.
HONEST PERF FRAMING: v1 serves the eager-static path only (D2), so expect ~7 tok/s (35B) —
the 11.78/20.9 captured numbers are greedy-only and land with the capture follow-up; the
DEFAULT chat path (temp>0) is host-sampled (~1MB logits D2H + O(V log V) sort per token). **Qwen-specific by design, not a generic inference engine**; efficiency
improves gradually on this base. One request decodes at a time (FIFO); no batching (separate
future milestone). Grounding: `docs/specs/SERVE-prior-art.md`.

## Architecture

```
 client (openai sdk / curl)
   │  POST /v1/chat/completions | /v1/completions     GET /v1/models /health
   ▼
 axum 0.8 (tokio) ── serde structs (OpenAI field names, cribbed from async-openai;
   │                  #[serde(flatten)] extra map; utoipa optional)
   │  validate → render chat template (minijinja 2.10 + minijinja-contrib pycompat,
   │  template + specials loaded from the MODEL DIR's tokenizer_config.json/chat_template.jinja)
   ▼
 submit: bounded queue, DEPTH 2-4 default (single-stream backpressure: a deep queue = dead
 connections; full ⇒ immediate 429 via try_send). Return path: tokio::sync::mpsc (bounded ~4;
 engine uses blocking_send — legal off-runtime; ReceiverStream feeds Sse); non-stream:
 tokio oneshot. CANCEL = channel closure: SSE stream drop closes the receiver; the engine's
 next blocking_send errs ⇒ stop (no separate token); engine ALSO checks closure BEFORE
 tokenize/prefill (skip dead queued requests). max_tokens OPTIONAL per spec ⇒
 max_tokens_effective = min(requested?, T_MAX - prompt_len) — never unbounded.
   ▼
 ENGINE THREAD (std::thread; owns the !Sync model; loads ONCE at startup per MODEL+QUANT)
   startup: load → warm one small forward
   per request (ONE PATH, eager-static — D2 decision B: simplest correct v1; capture wiring =
   the first follow-up efficiency milestone):
     tokenize → length check (prompt_len + max_tokens ≤ CTX_MAX, else 400) →
     fresh per-request cache (capacity = lp + max_tokens_effective) → eager prefill →
     per-token loop: sample (device argmax | host top-k/top-p) → cancel check →
     incremental detok (bounded-tail decode-and-diff, UTF-8 holdback) →
     think-boundary state machine (in_think: route text to delta.reasoning_content until
     </think>, then delta.content) → stop-string scan + holdback (longest-stop-prefix) →
     ONLY THEN emit the SSE chunk  ← detok/scan GATES emission, never after it →
     finish_reason {stop|length} → usage chunk (choices: []) → [DONE]
   ▼ per-token channel back to the axum handler (SSE) or accumulate (non-stream)
```

- Streaming: `axum::response::sse` (Event + KeepAlive). Chunk schema per the OpenAI spec
  (delta.role first chunk, delta.content, finish_reason, `data: [DONE]`; usage in the final
  chunk when `stream_options.include_usage`).
- EOS/specials read DYNAMICALLY from the model dir (35B: eos 248046, pad 248044 — the repo's
  hardcoded 30B ids are a known trap). `<think>` handling: template renders per
  `enable_thinking` (request `extra` flag, default on per template); response returns
  `reasoning_content` split on think-tags (mirrors vLLM's qwen3 reasoning parser) — v1: split
  only, no parser plugins.
- One model per process: `MODEL={qwen3-30b|qwen3.6-35b}` + `QUANT` + dir override. Two engine
  variants behind one trait-thin seam (Qwen3MoeForCausalLM w/ its CapturedLlm machinery vs
  Qwen3_5MoeForCausalLM w/ the bench pattern) — NO generic model abstraction; two concrete arms.
  `/v1/models` reports the loaded one. Template loading order: chat_template.jinja file, ELSE
  the `chat_template` STRING embedded in tokenizer_config.json (the 30B instruct dir ships a
  full 2630-char Qwen template this way — NO hand-rolled ChatML fallback, ever). EOS is a LIST
  from generation_config.json (35B [248046,248044]; 30B [151645,151643]) with tokenizer_config
  eos_token as fallback. Server sampling defaults follow each model's generation_config.json
  (35B: temp 1.0/top_p 0.95/top_k 20) — auto-decided, user may veto.
- Cancellation: client disconnect flips the request's cancel token; engine checks between
  tokens (bounded latency ≤ 1 token).
- NOT thread-safe by design: exactly one engine thread; axum handlers are thin.

## Deliverables
- **S1 `src/serve/` module**: `api.rs` (serde types), `template.rs` (minijinja env + pycompat
  shims + specials loading), `detok.rs` (incremental decoder w/ UTF-8 holdback + stop-string
  scan), `engine.rs` (engine thread: load/capture/request loop; greedy-captured + sampled-eager
  paths), `mod.rs`.
- **S2 `[[bin]] qwen-serve`** (`src/bin/qwen_serve.rs`): clap-less env/args (HOST/PORT/QUANT/
  MODEL_DIR/T_MAX/QUEUE_DEPTH), startup banner w/ memory + capture status. Distribution: it's a
  cargo binary in this repo (`cargo run --release --features cuda,serve --bin qwen-serve`);
  no CI/packaging in v1 (flagged NOT-in-scope).
- **S3 feature flag `serve`** gating the new deps (tokio/axum/minijinja/serde) so the core
  library builds unchanged without them.
- **S4 gates** (below).

## GATE RESULTS (2026-07-02 — milestone COMPLETE, branch serve-m-s)

All S.5 gates green on BOTH models, run against the live `qwen-serve` binary on the GB10:

| Gate | 35B nvfp4 | 30B bf16 (instruct-2507) |
|---|---|---|
| (a) serde fixtures | PASS (72 lib tests total) | same suite |
| (b) template parity | PASS — 12/12 byte-identical vs HF transformers 5.12.1 | same gate (both models in one battery) |
| (c) detok battery | PASS (incl. adversarial U+FFFD force-commit) | same suite |
| (d) E2E greedy parity | PASS — byte-identical to `qwen35_generate` (16 tok); non-stream == streamed-concat | PASS — byte-identical to `vllm_infer` (16 tok); non-stream == streamed-concat |
| (e) SSE raw-socket shape | PASS (role-first chunk, one finish chunk, usage:null matrix, [DONE]) | PASS |
| (f) python openai SDK | PASS (chat/stream/completions; stream==nonstream incl reasoning_content) | PASS |
| (g) 20-request sustained | PASS — per-class tok/s stable; RSS plateaus (no per-request leak) | PASS — RSS flat across repeated shapes after one-time per-shape JIT/autotune warmup |
| (h) error paths + FIFO | PASS — 400 length/tools/n>1/logprobs, 404 model, 429 saturation, FIFO under 2 clients, cancel-mid-stream | PASS |

Measured eager throughput (per-token, steady, sequential requests):
- 35B nvfp4: **sampled ~7.5 tok/s** (matches the plan's ~7 expectation); **greedy ~1.07 tok/s** — the greedy/device-argmax
  path is ~7x slower than host-sampled, an UNEXPECTED inversion (root-cause candidate: per-token device argmax sync).
- 30B bf16: greedy-short ~6.0, greedy-long ~9.9 tok/s (fused static-decode); sampled ~1.4 tok/s on terse replies (prefill-dominated).
- Follow-up efficiency ladder (in order): CUDA-graph capture wiring (designated first, 7.3→11.78 expected on 35B greedy),
  the 35B greedy-eager gap above, cache-pool reuse.

Post-plan hardening from the 3-voice reviews (all folded, see git log serve-m-s): empty-prompt guard (was: one request
could kill the process), bounded exit grace, EOS `null` fallback, EOS counted in usage, explicit-max_tokens 400,
OrderedJson end-to-end for messages+tools key order, Python-repr float tojson, tool-shape validation 400 (gate (h)
caught minijinja's lenient-undefined rendering malformed tools as silent empty prompt holes).

## Tasks
- [x] **S.0 deps + skeleton** — Cargo features, api.rs types (chat/completions/models/errors,
  SSE chunk structs), axum app w/ /health; unit tests: serde round-trip fixtures cribbed from
  real OpenAI request/response JSON (incl. unknown-field tolerance).
- [x] **S.1 template.rs — FIRST, as the de-risk spike** (the minijinja-vs-Qwen-jinja gamble is
  the plan's highest-uncertainty item; note: [::-1] slicing + negative indexing are NATIVE
  minijinja core features, pycompat only adds str/list methods — believed fine, PROVEN only by
  this gate): minijinja + pycompat + shims; loader reads chat_template.jinja OR the embedded
  tokenizer_config `chat_template` string (30B). **GATE: byte-identical rendering vs HF
  transformers apply_chat_template dumps** for BOTH models: single user; system+multi-turn;
  enable_thinking on/off; assistant-with-think history (exercises the [::-1] path); tool defs.
  If the gate fails on a minijinja limitation: fallback decision escalates (PyO3 vs template
  rewrite) — do NOT silently hand-roll.
- [x] **S.2 detok.rs** — incremental decode-and-diff w/ UTF-8 holdback (decode a bounded TAIL
  window, not the full growing buffer — avoids O(len²) over long generations); stop-string scan
  w/ cross-chunk boundary handling (hold back longest-stop-prefix); unit tests incl. multi-byte
  chars split across tokens, stop string spanning two chunks, think-tag passthrough.
- [x] **S.3 engine.rs** — engine thread; startup load per MODEL+QUANT (load arms reused from
  qwen35_generate / vllm_infer). The generate LOOPS are re-authored as per-token GENERATORS
  (vllm_infer::generate has no per-token hook — its loop STRUCTURE is the reference, the fn is
  not reusable verbatim): 30B arm = build_static_decode fused pattern; 35B arm =
  forward_last_logits loop; each yields token ids to the return channel w/ per-token
  closed-channel cancel check, distinct finish_reason {stop|length}, dynamic EOS LIST,
  max_tokens_effective clamp, usage counting, fresh cache per request; pre-run closed check.
- [x] **S.4 handlers + wiring** — /v1/chat/completions (template→engine), /v1/completions
  (raw prompt), stream & non-stream, stream_options.include_usage, error mapping (400 length /
  429 queue / 500 engine w/ id), request logging (id, tokens, tok/s, queue depth).
- [x] **S.5 gate battery** —
  (a) serde/spec fixtures green (S.0);
  (b) template parity byte-identical (S.1);
  (c) detok unit battery (S.2);
  (d) E2E-greedy: server(35B nvfp4) response tokens == the qwen35_generate greedy fixture AND
      server(30B) == the vllm_infer greedy fixture; non-stream AND streamed-concatenated
      identical;
  (e) SSE framing: raw-socket capture asserts chunk shape, [DONE], usage;
  (f) python `openai` client smoke: chat + completions + streaming against the live server;
  (g) sustained smoke: 20 sequential mixed requests (greedy/sampled/cancel-mid-stream,
      both short+long prompts), stable tok/s, device memory flat across requests (fresh-cache
      leak check);
  (h) error paths exercised: 400 (length; template render failure on malformed tool schema;
      unsupported params n>1/logprobs), 429 (queue full), and 2 concurrent HTTP clients
      (FIFO order preserved, both complete).

## NOT in scope (explicit)
- Concurrent/continuous batching, paged KV, prefix caching (future milestone; D1 option B/C).
- Multi-model routing, model hot-swap, auth, TLS, rate limiting beyond the bounded queue.
- Tool-call PARSING of model output into structured tool_calls (template renders tool defs;
  output returned as text; parser = follow-up).
- logprobs, n>1, presence/frequency penalties (400 w/ clear message if requested).
- CUDA-graph captured decode in the server — **the designated FIRST follow-up efficiency
  milestone** (D2: user chose simplest-correct-first; the capture-once + reset_and_prefill
  pattern is proven in the bench and slots into engine.rs without API changes; expected
  uplift 7.3→11.78 tok/s nvfp4 greedy).
- MTP speculative decoding in the server (engine supports it eagerly; wiring = follow-up).
- Multi-T_MAX buckets (single T_MAX per process, default 4096; overflow ⇒ 400).
- CI/packaging/container for the binary.

## What already exists (reused, not rebuilt)
Capture-once serving pattern (cudagraph bench), load arms for all three quants, host sampling,
xorshift RNG, tokenizer wrapper, EOS/finished device machinery, memory instrumentation. The
vllm_infer example stays as-is (its per-prompt re-capture noted as a known inefficiency).

## Failure modes (silent-first)
| Risk | Guard |
|---|---|
| Wrong/missing EOS ids → over-generation | EOS LIST from generation_config.json (NOT tokenizer_config — 248044 lives only there); unit test asserts both lists from the real files |
| Stop-string/think-tag leaks to client | detok+scan GATE emission (holdback); streamed-vs-nonstream consistency test |
| Slow client → engine dumps chunks into RAM | bounded return channel (engine blocks on backpressure) |
| Template drift vs HF → silently different prompts | S.1 byte-identical parity gate vs transformers dumps |
| UTF-8 split / stop-string split across chunks → mojibake or missed stop | S.2 unit battery |
| Cross-request state bleed | fresh cache per request (by construction) + gate (g) memory-flat smoke |
| Cancel mid-stream leaks engine loop | cancel token checked per token; gate (g) cancel case |
| Queue starvation/unbounded memory | bounded mpsc + 429; depth metric in logs |
| Engine thread panic mid-request | let the panic DROP the channels (handler detects closure → 500 flushed), log, THEN exit(1) — never exit before the response flushes |
| Fresh-cache alloc churn per request | acceptable at single-stream; cache-pool reuse listed in the efficiency ladder |
| Sampled path ≠ greedy path prompt handling | both paths share tokenize/prefill code (one fn) |

## Parallelization
Lane A: S.0 + S.1 + S.2 (pure host, independent). Lane B: S.3 (engine, touches capture reuse).
Converge: S.4 → S.5. Conflicts: none with other milestones (new module + examples untouched).

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|--------|---------|-----|------|--------|----------|
| Eng Review | `/plan-eng-review` | Architecture & tests (required) | 1 | issues_found→folded | Step-0: scope user-locked (D1 single-stream, D2 eager-only, 2 Qwen models only); 4 sections: 5 findings folded (panic policy, O(len²) detok, template-fail/concurrency tests, cache-churn note); complexity trigger pre-empted by D1 |
| Outside Voice — Gemini | 3.1 Pro high (agy-direct) | Independent 2nd opinion | 1 | issues_found→folded | 10 findings, all convergent-corrective: emission gating order, streaming think state machine, queue depth 2-4, optional max_tokens clamp, bounded return channel, pre-run cancel, minijinja PoC-first, usage-chunk shape, panic flush order, T_MAX naming |
| Outside Voice — Opus | 4.8 (repo-grounded agent) | Independent 2nd opinion | 1 | issues_found→folded | 9 findings: 30B embedded chat_template (ChatML fallback killed), EOS LIST in generation_config.json, honest tok/s framing, generator inversion (not "verbatim"), streaming reasoning boundary, channel types pinned, 2 feared traps DEFUSED (512-truncation; minijinja [::-1] native) |
| Codex Review | `codex exec` | Independent 2nd opinion | 0 | unavailable | weekly quota exhausted until Jul 7 — Claude-subagent fallback used per skill preflight |

- **CROSS-MODEL:** Gemini and Opus fully convergent (no tension); Gemini's minijinja-gamble
  concern is bounded by Opus's grounded slicing check — resolved as "S.1 parity gate first,
  escalate on failure, never hand-roll".
- **VERDICT:** ENG CLEARED (scope user-locked D1/D2; all review + outside-voice findings
  folded) — ready to implement, S.1 template spike first.

**UNRESOLVED DECISIONS:**
- Server sampling defaults follow each model's generation_config.json (35B: temp 1.0 / top_p 0.95 / top_k 20) rather than the repo examples' 0.7/0.8/20 — auto-decided for HF parity; veto if you prefer the repo defaults.
