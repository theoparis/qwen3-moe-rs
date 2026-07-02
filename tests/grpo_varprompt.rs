//! Padding-invariance for variable-length prompt batches (GRPO Phase-B B2).
//!
//! Left-padding a prompt — with the right attention mask + RoPE positions — must produce the SAME
//! logits on the real tokens as running the prompt unpadded. This is the model foundation for
//! ragged prompt batches: if it holds, the trainer can left-pad a batch of variable-length prompts
//! and the GRPO completion log-probs are unaffected by the padding.
//!
//! Run: `cargo test --test grpo_varprompt`

use burn::backend::NdArray;
use burn::tensor::{Bool, Int, Tensor};
use qwen3_burn::grpo::{group_sample, group_sample_padded, RolloutConfig};
use qwen3_burn::Qwen3Config;

type B = NdArray;

fn tiny_model(dev: &<B as burn::tensor::backend::Backend>::Device) -> qwen3_burn::Qwen3ForCausalLM<B> {
    Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16))
        .init_causal_lm::<B>(dev)
}

#[test]
fn left_pad_is_logit_invariant() {
    let dev = Default::default();

    let cfg = Qwen3Config::new()
        .with_vocab_size(32)
        .with_hidden_size(64)
        .with_intermediate_size(128)
        .with_num_hidden_layers(2)
        .with_num_attention_heads(4)
        .with_num_key_value_heads(2)
        .with_head_dim(Some(16));
    let model = cfg.init_causal_lm::<B>(&dev);

    // unpadded prompt: 4 real tokens at positions 0..3, full causal attention
    let real = [3i64, 7, 1, 9];
    let lp = real.len();
    let unpadded = Tensor::<B, 1, Int>::from_data(real.as_slice(), &dev).reshape([1, lp]);
    let logits_u = model.forward(unpadded, None); // [1, lp, v]

    // left-pad with `pad` pad tokens (id 0): [0, 0, 3, 7, 1, 9]
    let pad = 2usize;
    let l = pad + lp;
    let mut padded_ids = vec![0i64; pad];
    padded_ids.extend_from_slice(&real);
    let padded = Tensor::<B, 1, Int>::from_data(padded_ids.as_slice(), &dev).reshape([1, l]);

    // attention mask: false for the left-pad, true for the real tokens
    let mut mask_v = vec![false; pad];
    mask_v.extend(std::iter::repeat(true).take(lp));
    let mask = Tensor::<B, 1, Bool>::from_data(mask_v.as_slice(), &dev).reshape([1, l]);

    // RoPE positions = cumsum(mask) - 1, pad clamped to 0: [0, 0, 0, 1, 2, 3]
    let mut pos_v = vec![0i64; pad];
    pos_v.extend(0..lp as i64);
    let pos = Tensor::<B, 1, Int>::from_data(pos_v.as_slice(), &dev).reshape([1, l]);

    let logits_p = model.forward_with_positions(padded, Some(mask), pos); // [1, l, v]

    // the real-token logits must match: unpadded[0..lp] vs padded[pad..pad+lp]
    let v = logits_u.dims()[2];
    let u = logits_u.into_data().to_vec::<f32>().unwrap(); // lp * v
    let p = logits_p.into_data().to_vec::<f32>().unwrap(); // l * v
    // NaN-aware: a fully-masked pad query row softmaxes to NaN and can poison real positions; a
    // plain f32::max over diffs would SILENTLY drop those NaNs, so assert finiteness explicitly.
    assert!(p.iter().all(|x| x.is_finite()), "padded forward produced non-finite logits (NaN poisoning)");
    let mut maxabs = 0f32;
    for t in 0..lp {
        for c in 0..v {
            let (a, b) = (u[t * v + c], p[(pad + t) * v + c]);
            assert!(a.is_finite() && b.is_finite(), "non-finite logit at t={t}, c={c}");
            maxabs = maxabs.max((a - b).abs());
        }
    }
    assert!(maxabs < 1e-3, "left-pad changed the real-token logits: max|diff| = {maxabs}");
    println!("left-pad logit-invariance OK — max|diff| = {maxabs:.2e}");
}

/// End-to-end rollout invariance: a left-padded prompt run through `group_sample_padded` must
/// produce the SAME greedy completion (and old-logprobs) as the unpadded prompt through
/// `group_sample`. This is the gate for ragged-prompt rollouts: the left-pad must not change what
/// gets generated or its old log-probs (which feed the GRPO ratio).
#[test]
fn padded_rollout_matches_unpadded_greedy() {
    let dev = Default::default();
    let model = tiny_model(&dev);

    let real = [3i64, 7, 1, 9];
    let lp_real = real.len();
    // greedy so both runs are deterministic; one prompt, one completion
    let rc = RolloutConfig { group_size: 1, max_new_tokens: 5, temperature: 0.0, top_p: 1.0, top_k: 0 };
    let eos = [7i64];

    // unpadded reference
    let unpadded = Tensor::<B, 1, Int>::from_data(real.as_slice(), &dev).reshape([1, lp_real]);
    let a = group_sample(&model, unpadded, &rc, &eos);

    // left-pad with 3 pad tokens -> lp = 7; real length is 4
    let pad = 3usize;
    let mut padded_ids = vec![0i64; pad];
    padded_ids.extend_from_slice(&real);
    let lp = pad + lp_real;
    let padded = Tensor::<B, 1, Int>::from_data(padded_ids.as_slice(), &dev).reshape([1, lp]);
    let b = group_sample_padded(&model, padded, &[lp_real], &rc, &eos);

    assert_eq!(a.gen_len, b.gen_len, "gen_len differs: unpadded {} vs padded {}", a.gen_len, b.gen_len);

    // completion tokens: unpadded seq_ids[:, lp_real..] vs padded seq_ids[:, lp..]
    let ai = a.seq_ids.into_data().to_vec::<i64>().unwrap();
    let bi = b.seq_ids.into_data().to_vec::<i64>().unwrap();
    let a_comp = &ai[lp_real..lp_real + a.gen_len];
    let b_comp = &bi[lp..lp + b.gen_len];
    assert_eq!(a_comp, b_comp, "left-pad changed the greedy completion: {a_comp:?} vs {b_comp:?}");

    // old log-probs must match (they feed the GRPO ratio)
    let al = a.old_logprobs.into_data().to_vec::<f32>().unwrap();
    let bl = b.old_logprobs.into_data().to_vec::<f32>().unwrap();
    assert_eq!(al.len(), bl.len(), "old_logprob length");
    for (i, (x, y)) in al.iter().zip(bl.iter()).enumerate() {
        assert!((x - y).abs() < 1e-3, "old_logprob[{i}] differs: unpadded {x} vs padded {y}");
    }
    println!("padded-rollout parity OK (greedy) — gen_len={} completion={:?}", a.gen_len, a_comp);
}

/// The cache PREFILL path must be left-pad-safe too (it builds its own combined mask, so it needs
/// the same diagonal-unmask): a left-padded prompt through `forward_with_cache` must produce FINITE
/// logits matching the no-cache `forward_with_positions`. Guards the cached path before it is used
/// for ragged prompts (Codex review).
#[test]
fn cached_prefill_is_left_pad_safe() {
    let dev = Default::default();
    let model = tiny_model(&dev);

    let real = [3i64, 7, 1, 9];
    let pad = 2usize;
    let lp = pad + real.len();
    let mut ids = vec![0i64; pad];
    ids.extend_from_slice(&real);
    let padded = Tensor::<B, 1, Int>::from_data(ids.as_slice(), &dev).reshape([1, lp]);
    let mut mv = vec![false; pad];
    mv.extend(std::iter::repeat(true).take(real.len()));
    let mask = Tensor::<B, 1, Bool>::from_data(mv.as_slice(), &dev).reshape([1, lp]);
    let mut pv = vec![0i64; pad];
    pv.extend(0..real.len() as i64);
    let pos = Tensor::<B, 1, Int>::from_data(pv.as_slice(), &dev).reshape([1, lp]);

    // reference: the proven no-cache path
    let ref_logits = model.forward_with_positions(padded.clone(), Some(mask.clone()), pos.clone());
    // cached prefill (builds its own mask internally)
    let mut cache = model.new_cache();
    let cached = model.forward_with_cache(padded, Some(mask), pos, &mut cache);

    let v = ref_logits.dims()[2];
    let r = ref_logits.into_data().to_vec::<f32>().unwrap();
    let c = cached.into_data().to_vec::<f32>().unwrap();
    assert!(c.iter().all(|x| x.is_finite()), "cached prefill produced non-finite logits (NaN poisoning)");
    let mut maxabs = 0f32;
    for t in pad..lp {
        for k in 0..v {
            let (a, b) = (r[t * v + k], c[t * v + k]);
            assert!(a.is_finite() && b.is_finite(), "non-finite at t={t}, k={k}");
            maxabs = maxabs.max((a - b).abs());
        }
    }
    assert!(maxabs < 1e-3, "cached prefill != no-cache on real tokens: max|diff| = {maxabs}");
    println!("cached-prefill left-pad-safe OK — max|diff| = {maxabs:.2e}");
}
