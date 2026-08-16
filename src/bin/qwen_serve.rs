//! `qwen-serve`: OpenAI-compatible single-stream server binary (M-S, S.4).
//!
//! One model per process, one request decoding at a time (FIFO). Config is env-var
//! only (no clap dep):
//!
//! | var          | default                                   | meaning                          |
//! |--------------|-------------------------------------------|----------------------------------|
//! | `HOST`       | `0.0.0.0`                                  | bind host                        |
//! | `PORT`       | `8000`                                     | bind port                        |
//! | `MODEL`      | `qwen3.6-35b`                              | `qwen3-30b` \| `qwen3.6-35b`     |
//! | `QUANT`      | `bf16`                                     | `bf16` \| `fp8` \| `nvfp4`       |
//! | `MODEL_DIR`  | per-model default under `models/`          | checkpoint dir                   |
//! | `T_MAX`      | `4096`                                     | process context limit            |
//! | `QUEUE_DEPTH`| `2`                                        | bounded submit-queue depth       |
//!
//! The 30B default dir is the **instruct-2507** checkpoint: the base 30B dir has no
//! `generation_config.json` and cannot boot (review finding).
//!
//! Run: `cargo run --release --features cuda,serve --bin qwen-serve`.

use std::path::PathBuf;
use std::sync::Arc;

use qwen3_burn::serve::engine::{self, EngineConfig, Quant, WhichModel};
use qwen3_burn::serve::handlers::{ServeState, ServedModel, build_router, load_sampling_defaults};
use qwen3_burn::serve::template::ChatTemplate;
use tokenizers::Tokenizer;

/// Read an env var or fall back to `default`.
fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

/// Fatal startup error: print to stderr and exit nonzero (never leave a
/// half-initialized process).
fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("[qwen-serve] FATAL: {}", msg.as_ref());
    std::process::exit(1);
}

#[tokio::main]
async fn main() {
    // ---- parse config from the environment ----
    let host = env_or("HOST", "0.0.0.0");
    let port: u16 = env_or("PORT", "8000")
        .parse()
        .unwrap_or_else(|e| die(format!("invalid PORT: {e}")));

    let (which, served, default_dir_bf16, default_dir_nvfp4) =
        match env_or("MODEL", "qwen3.6-35b").as_str() {
            "qwen3-30b" => (
                WhichModel::Qwen3Moe30b,
                ServedModel::Qwen30b,
                "models/qwen3-30b-a3b-instruct-2507",
                "models/qwen3-30b-a3b-instruct-2507",
            ),
            "qwen3.6-35b" => (
                WhichModel::Qwen35Moe,
                ServedModel::Qwen35b,
                "models/qwen3.6-35b-a3b",
                "models/qwen3.6-35b-a3b-nvfp4",
            ),
            other => die(format!(
                "unknown MODEL '{other}' (expected 'qwen3-30b' or 'qwen3.6-35b')"
            )),
        };

    let quant = match env_or("QUANT", "bf16").as_str() {
        "bf16" => Quant::Bf16,
        "fp8" => Quant::Fp8,
        "nvfp4" => Quant::Nvfp4,
        other => die(format!(
            "unknown QUANT '{other}' (expected 'bf16', 'fp8', or 'nvfp4')"
        )),
    };

    // MODEL_DIR override, else the per-model default (nvfp4 35B has its own dir).
    let model_dir: PathBuf = match std::env::var("MODEL_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let d = if quant == Quant::Nvfp4 {
                default_dir_nvfp4
            } else {
                default_dir_bf16
            };
            PathBuf::from(d)
        }
    };

    let t_max: usize = env_or("T_MAX", "4096")
        .parse()
        .unwrap_or_else(|e| die(format!("invalid T_MAX: {e}")));
    let queue_depth: usize = env_or("QUEUE_DEPTH", "2")
        .parse()
        .unwrap_or_else(|e| die(format!("invalid QUEUE_DEPTH: {e}")));

    if !model_dir.exists() {
        die(format!(
            "MODEL_DIR {} does not exist (set MODEL_DIR or place the checkpoint there)",
            model_dir.display()
        ));
    }

    // ---- load host-side pieces (template + tokenizer + sampling defaults) ----
    // These must come from the SAME dir the engine loads weights from, so the
    // rendered prompt and detok match the served model exactly.
    let template = ChatTemplate::from_model_dir(&model_dir)
        .unwrap_or_else(|e| die(format!("loading chat template: {e}")));
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .unwrap_or_else(|e| die(format!("loading tokenizer.json: {e}")));
    let defaults = load_sampling_defaults(&model_dir);

    // ---- spawn the engine (loads the model ONCE; blocks until ready) ----
    let config = EngineConfig {
        model: which,
        quant,
        model_dir: model_dir.clone(),
        t_max,
        queue_depth,
    };
    let handle = engine::spawn(config).unwrap_or_else(|e| die(format!("engine load: {e}")));

    // ---- startup banner ----
    let quant_str = match quant {
        Quant::Bf16 => "bf16",
        Quant::Fp8 => "fp8",
        Quant::Nvfp4 => "nvfp4",
    };
    eprintln!("[qwen-serve] ready");
    eprintln!("  model id    : {}", handle.model_id());
    eprintln!("  quant       : {quant_str}");
    eprintln!("  model dir   : {}", model_dir.display());
    eprintln!("  t_max       : {t_max}");
    eprintln!("  queue depth : {queue_depth}");
    eprintln!(
        "  sampling def: temp {} / top_p {} / top_k {}",
        defaults.temperature, defaults.top_p, defaults.top_k
    );
    eprintln!("  decode path : eager-static (CUDA-graph capture = first follow-up milestone)");
    eprintln!("  {}", handle.report());

    // ---- build router + serve ----
    let state = Arc::new(ServeState::new(
        handle, template, tokenizer, defaults, served,
    ));
    let app = build_router(state);

    let addr = format!("{host}:{port}");
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|e| die(format!("binding {addr}: {e}")));
    eprintln!("[qwen-serve] listening on http://{addr}");

    if let Err(e) = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
    {
        die(format!("server error: {e}"));
    }
    eprintln!("[qwen-serve] shut down cleanly");
}

/// Resolve when Ctrl-C is received (graceful shutdown).
async fn shutdown_signal() {
    if let Err(e) = tokio::signal::ctrl_c().await {
        eprintln!("[qwen-serve] failed to install Ctrl-C handler: {e}");
    }
    eprintln!("[qwen-serve] Ctrl-C received, shutting down");
}
