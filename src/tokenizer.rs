//! Tokenizer for Qwen3 text encoder.
//!
//! Uses the HuggingFace tokenizers crate to load and run the Qwen2 BPE tokenizer.

use std::path::Path;

use tokenizers::Tokenizer;

/// Special tokens for Qwen3.
pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
pub const ENDOFTEXT: &str = "<|endoftext|>";

/// Qwen3 tokenizer wrapper.
pub struct Qwen3Tokenizer {
    tokenizer: Tokenizer,
    max_length: usize,
}

impl Qwen3Tokenizer {
    /// Load the tokenizer from a tokenizer.json file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let tokenizer =
            Tokenizer::from_file(path).map_err(|e| format!("Failed to load tokenizer: {e}"))?;

        Ok(Qwen3Tokenizer {
            tokenizer,
            max_length: 512,
        })
    }

    /// Set the maximum sequence length for tokenization.
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Apply the chat template to a user prompt for Z-Image.
    ///
    /// This formats the prompt as:
    /// <|im_start|>user
    /// {prompt}<|im_end|>
    /// <|im_start|>assistant
    ///
    /// Note: Does NOT include <think> token - that's for text generation, not image generation.
    pub fn apply_chat_template(&self, prompt: &str) -> String {
        format!("{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n")
    }

    /// Tokenize a prompt and return input IDs and attention mask.
    /// This version pads to max_length (used for Z-Image text encoding).
    ///
    /// # Returns
    /// A tuple of (input_ids, attention_mask) where both are Vec<i64>.
    pub fn encode(&self, text: &str) -> Result<(Vec<i64>, Vec<bool>), String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Failed to encode text: {e}"))?;

        let mut input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        let mut attention_mask: Vec<bool> = vec![true; input_ids.len()];

        // Truncate if too long
        if input_ids.len() > self.max_length {
            input_ids.truncate(self.max_length);
            attention_mask.truncate(self.max_length);
        }

        // Pad if too short
        let pad_token_id = self.tokenizer.token_to_id(ENDOFTEXT).unwrap_or(0) as i64;
        while input_ids.len() < self.max_length {
            input_ids.push(pad_token_id);
            attention_mask.push(false);
        }

        Ok((input_ids, attention_mask))
    }

    /// Tokenize a prompt without padding (for text generation).
    ///
    /// # Returns
    /// A tuple of (input_ids, attention_mask) where both are Vec<u32>.
    pub fn encode_no_pad(&self, text: &str) -> Result<(Vec<u32>, Vec<bool>), String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("Failed to encode text: {e}"))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let attention_mask: Vec<bool> = vec![true; input_ids.len()];

        Ok((input_ids, attention_mask))
    }

    /// Tokenize a user prompt with chat template applied.
    ///
    /// This is the main entry point for encoding prompts for Z-Image.
    pub fn encode_prompt(&self, prompt: &str) -> Result<(Vec<i64>, Vec<bool>), String> {
        let formatted = self.apply_chat_template(prompt);
        self.encode(&formatted)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| format!("Failed to decode tokens: {e}"))
    }

    /// Decode token IDs (as i64) back to text.
    pub fn decode_i64(&self, ids: &[i64]) -> Result<String, String> {
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        self.decode(&ids_u32)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get the token ID for a special token.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get the end-of-text token ID.
    pub fn eos_token_id(&self) -> u32 {
        self.tokenizer.token_to_id(ENDOFTEXT).unwrap_or(151643)
    }
}
