//! M-S: OpenAI-compatible single-stream HTTP server (docs/SERVE_PLAN.md).
//!
//! Serves exactly the two proven models — Qwen3-30B-A3B (bf16) and
//! Qwen3.6-35B-A3B (bf16/fp8/nvfp4) — one model per process, one request
//! decoding at a time (FIFO). Host-side parts (api/template/detok) build and
//! test without `cuda`; the engine needs `cuda`.

pub mod api;
pub mod detok;
pub mod template;

#[cfg(feature = "cuda")]
pub mod engine;

/// S.4: HTTP handlers + router. Pure host logic (validation, param/seed
/// resolution, think-splitter, SSE frame building) is host-testable without
/// `cuda`; only the engine-touching handler bodies + router assembly are
/// `#[cfg(feature = "cuda")]` inside this module.
pub mod handlers;
