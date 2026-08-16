//! Example: Text generation with Qwen3-0.6B
//!
//! Usage:
//!   cargo run --example generate --release -- --model models/model.safetensors --tokenizer models/tokenizer.json
//!
//! `--device flex|wgpu|vulkan|metal` selects the compute backend. `wgpu`/`vulkan`/`metal`
//! require building with the matching feature (`--features wgpu`, etc).

use std::path::PathBuf;

use burn::prelude::Device;
#[cfg(any(feature = "wgpu", feature = "vulkan", feature = "metal"))]
use burn::tensor::DeviceKind;
use burn::tensor::{Int, Tensor};
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM, Qwen3Tokenizer};

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .iter()
        .position(|x| x == "--model")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("models/model.safetensors"));

    let tokenizer_path = args
        .iter()
        .position(|x| x == "--tokenizer")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("models/tokenizer.json"));

    let prompt = args
        .iter()
        .position(|x| x == "--prompt")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "Hello, I am a language model and".to_string());

    let max_tokens: usize = args
        .iter()
        .position(|x| x == "--max-tokens")
        .map(|i| args[i + 1].parse().unwrap_or(50))
        .unwrap_or(50);

    println!("Loading tokenizer from {:?}...", tokenizer_path);
    let tokenizer = Qwen3Tokenizer::from_file(&tokenizer_path)?;

    let device_name = args
        .iter()
        .position(|x| x == "--device")
        .map(|i| args[i + 1].clone())
        .unwrap_or_else(|| "flex".to_string());

    println!("Initializing Qwen3-0.6B model on {device_name}...");
    let device = match device_name.as_str() {
        "flex" => Device::flex(),
        #[cfg(feature = "wgpu")]
        "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        #[cfg(feature = "vulkan")]
        "vulkan" => Device::vulkan(DeviceKind::DefaultDevice),
        #[cfg(feature = "metal")]
        "metal" => Device::metal(DeviceKind::DefaultDevice),
        other => {
            return Err(format!(
                "unknown or unbuilt --device {other:?} (build with --features wgpu/vulkan/metal to enable it)"
            ));
        }
    };

    // Use the 0.6B config preset
    let config = Qwen3Config::qwen3_0_6b();
    println!(
        "Config: {} layers, {} hidden, {} heads",
        config.num_hidden_layers, config.hidden_size, config.num_attention_heads
    );

    let mut model: Qwen3ForCausalLM = config.init_causal_lm(&device);

    println!("Loading weights from {:?}...", model_path);
    model
        .load_weights(&model_path)
        .map_err(|e| format!("Failed to load weights: {e:?}"))?;
    println!("Model loaded successfully!");

    // Tokenize prompt
    println!("\nPrompt: {}", prompt);
    // Use the no-pad encoder for generation (the default `encode` pads to 512 with
    // <|endoftext|>, a Z-Image text-encoder behavior that breaks autoregressive gen).
    let (input_ids_u32, _attention_mask) = tokenizer.encode_no_pad(&prompt)?;
    let input_ids: Vec<i64> = input_ids_u32.iter().map(|&x| x as i64).collect();
    println!("Input tokens: {:?}", input_ids);

    // Create input tensor [1, seq_len]
    let input_tensor: Tensor<1, Int> = Tensor::from_data(input_ids.as_slice(), &device);
    let input_tensor: Tensor<2, Int> = input_tensor.unsqueeze();

    println!("\nGenerating {} tokens...", max_tokens);
    let start = std::time::Instant::now();

    // Generate with KV cache for efficiency
    let output = model.generate_with_cache(
        input_tensor,
        max_tokens,
        0.7, // temperature
        0.9, // top_p
        50,  // top_k
    );

    let elapsed = start.elapsed();

    // Get output tokens
    let output_data: Vec<i64> = output
        .to_data()
        .to_vec::<i64>()
        .map_err(|e| format!("Failed to convert tensor: {e:?}"))?;
    let output_tokens: Vec<u32> = output_data.iter().map(|&x| x as u32).collect();

    println!(
        "Output tokens ({} total): {:?}",
        output_tokens.len(),
        &output_tokens[..output_tokens.len().min(20)]
    );

    // Decode
    let generated_text = tokenizer.decode(&output_tokens)?;

    println!("\n=== Generated Text ===");
    println!("{}", generated_text);
    println!("======================");

    let tokens_per_sec = max_tokens as f64 / elapsed.as_secs_f64();
    println!(
        "\nGenerated {} tokens in {:.2}s ({:.2} tokens/sec)",
        max_tokens,
        elapsed.as_secs_f64(),
        tokens_per_sec
    );

    Ok(())
}
