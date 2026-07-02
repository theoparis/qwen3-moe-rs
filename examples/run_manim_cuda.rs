//! Run a Qwen3-14B Manim finetune (haidangung/qwen3-manim-16bit) on the CubeCL CUDA backend and
//! generate Manim code. This exercises the new UNTIED-embedding path (Qwen3-14B has a separate
//! `lm_head.weight`) and the sharded safetensors loader.
//!
//! Build:
//!   RUSTFLAGS="-C target-feature=+fp16" cargo build --release --features cuda --example run_manim_cuda
//! Run:
//!   ./target/release/examples/run_manim_cuda --prompt "Draw a blue circle." --max-tokens 160

use std::path::PathBuf;

use burn::backend::cuda::{Cuda, CudaDevice};
use burn::tensor::{Int, Tensor};
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};

type Backend = Cuda;

fn arg<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter().position(|x| x == flag).and_then(|i| args.get(i + 1))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let dir = arg(&args, "--dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/qwen3-manim-14b"));
    let user = arg(&args, "--prompt")
        .cloned()
        .unwrap_or_else(|| "Write a Manim scene that draws a blue circle and transforms it into a square.".to_string());
    let max_tokens: usize = arg(&args, "--max-tokens").and_then(|s| s.parse().ok()).unwrap_or(160);

    let device = CudaDevice::default();
    println!("device: {device:?}");

    let tokenizer = Qwen3Tokenizer::from_file(dir.join("tokenizer.json"))?;

    let config = Qwen3Config::qwen3_14b();
    println!(
        "config: {} layers, {} hidden, {} heads, tied={}",
        config.num_hidden_layers, config.hidden_size, config.num_attention_heads, config.tie_word_embeddings
    );
    let mut model: Qwen3ForCausalLM<Backend> = config.init_causal_lm(&device);

    println!("loading sharded weights from {dir:?} ...");
    let t0 = std::time::Instant::now();
    model.load_weights_sharded(&dir).map_err(|e| format!("load_weights_sharded failed: {e:?}"))?;
    println!("loaded untied 14B in {:.1}s", t0.elapsed().as_secs_f64());

    // Qwen chat template — this is an instruct finetune.
    let prompt = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    println!("\n--- prompt ---\n{user}\n");

    let (ids_u32, _) = tokenizer.encode_no_pad(&prompt)?;
    let ids: Vec<i64> = ids_u32.iter().map(|&x| x as i64).collect();
    let input: Tensor<Backend, 1, Int> = Tensor::from_data(ids.as_slice(), &device);
    let input: Tensor<Backend, 2, Int> = input.unsqueeze();

    println!("generating {max_tokens} tokens (greedy)...");
    let start = std::time::Instant::now();
    let output = model.generate_with_cache(input, max_tokens, 0.0, 1.0, 0);
    let out_ids: Vec<i64> = output
        .cast(burn::tensor::DType::I64)
        .to_data()
        .to_vec()
        .map_err(|e| format!("read output: {e:?}"))?;
    let out_u32: Vec<u32> = out_ids.iter().map(|&x| x as u32).collect();
    let text = tokenizer.decode(&out_u32)?;

    println!("\n===== GENERATION ({:.1}s) =====\n{text}\n==============================", start.elapsed().as_secs_f64());
    Ok(())
}
