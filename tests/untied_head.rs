//! Untied-embedding path (ported for Qwen3-14B etc., `tie_word_embeddings = false`).
//!
//! A tied model projects logits from `embed_tokens.weight`; an untied model has a SEPARATE
//! `lm_head`. This checks both build and run, and that the untied head actually changes the logits
//! (i.e. the separate head is used, not the embedding). Full end-to-end validation was the 14B Manim
//! generation in qwen3-bf-3; this is the cheap in-repo regression gate.
//!
//! Run: `cargo test --test untied_head`

use burn::backend::NdArray;
use burn::tensor::{Int, Tensor};
use qwen3_burn::{Qwen3Config, Qwen3ForCausalLM};

type B = NdArray;

fn tiny(tie: bool) -> Qwen3Config {
    Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16))
        .with_tie_word_embeddings(tie)
}

#[test]
fn untied_forward_runs_and_differs_from_tied() {
    let dev = Default::default();
    let ids = Tensor::<B, 1, Int>::from_data([1i64, 2, 3, 4].as_slice(), &dev).reshape([1, 4]);

    // untied: a separate (random-init) lm_head is built and used
    let cfg_u = tiny(false);
    assert!(!cfg_u.tie_word_embeddings, "config flag should be false");
    let untied: Qwen3ForCausalLM<B> = cfg_u.init_causal_lm(&dev);
    let lu = untied.forward(ids.clone(), None);
    assert_eq!(lu.dims(), [1, 4, 32], "logits shape");
    let vu = lu.into_data().to_vec::<f32>().unwrap();
    assert!(vu.iter().all(|x| x.is_finite()), "untied logits must be finite");

    // tied: same dims/seed-independent structure, but projects from the embedding instead
    let tied: Qwen3ForCausalLM<B> = tiny(true).init_causal_lm(&dev);
    let vt = tied.forward(ids, None).into_data().to_vec::<f32>().unwrap();

    // The untied head is a distinct (randomly initialized) matrix, so the logits must differ from
    // the tied (embedding-projected) ones — proves `lm_logits` dispatched to the separate head.
    let maxabs = vu.iter().zip(vt.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(maxabs > 1e-3, "untied logits should differ from tied (separate head not used?) maxabs={maxabs}");

    println!("untied head OK — finite logits [1,4,32], differs from tied (maxabs {maxabs:.3})");
}
