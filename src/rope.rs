//! Rotary Position Embeddings for Qwen3.
//!
//! Qwen3 uses RoPE with theta=1000000 and applies it differently than Z-Image's
//! image transformer.

use burn::{Tensor, prelude::Device, tensor::Int};

/// Float activations may be Autodiff while Int-derived floats stay on the inner backend.
fn float_on_device<const D: usize>(t: Tensor<D>, device: &Device) -> Tensor<D> {
    match (device.is_autodiff(), t.device().is_autodiff()) {
        (true, false) => Tensor::from_inner(t),
        (false, true) => t.inner(),
        _ => t,
    }
}

/// Compute rotary position embeddings for Qwen3.
///
/// # Arguments
/// * `positions` - Position indices [batch_size, seq_len]
/// * `head_dim` - Dimension per attention head
/// * `theta` - RoPE theta parameter (default 1000000 for Qwen3)
/// * `device` - Device to create tensors on
pub fn compute_rope_embeddings(
    positions: Tensor<2, Int>,
    head_dim: usize,
    theta: f64,
    device: &Device,
) -> (Tensor<4>, Tensor<4>) {
    let half_dim = head_dim / 2;

    // Compute frequency bands: 1 / (theta^(2i/d)) for i in 0..half_dim
    let freq_seq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    let freqs = float_on_device(
        Tensor::<1>::from_floats(freq_seq.as_slice(), device),
        device,
    );
    let positions_float = float_on_device(positions.float(), device);
    let angles = positions_float.unsqueeze_dim::<3>(2) * freqs.unsqueeze::<3>();

    // cos and sin: [batch, seq, half_dim] -> [batch, seq, 1, half_dim]
    let cos = angles.clone().cos().unsqueeze_dim::<4>(2);
    let sin = angles.sin().unsqueeze_dim::<4>(2);

    (cos, sin)
}

/// Precompute the RoPE inverse-frequency table `1 / theta^(2i/d)` for `i in 0..head_dim/2`, ONCE.
///
/// CUDA-graph capture (P-final): [`compute_rope_embeddings`] builds this table every call via
/// `Tensor::from_floats` — a host->device staging that is uncapturable inside a graph (it bakes a
/// host source pointer freed after launch; the P-final `write_to_gpu` guard hard-errors on it). The
/// table is position-INDEPENDENT, so the captured decode hoists it out of the step and feeds it to
/// [`compute_rope_embeddings_pre`]; only the position-dependent `angles = pos * freqs` (a kernel, no
/// host staging) stays in the captured region.
pub fn rope_freqs(head_dim: usize, theta: f64, device: &Device) -> Tensor<1> {
    let half_dim = head_dim / 2;
    let freq_seq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / (theta as f32).powf(2.0 * i as f32 / head_dim as f32))
        .collect();
    Tensor::<1>::from_floats(freq_seq.as_slice(), device)
}

/// [`compute_rope_embeddings`] with a PRECOMPUTED frequency table (see [`rope_freqs`]) instead of a
/// per-call `from_floats`. Capture-safe: contains no host->device staging, only the device kernels
/// `pos.float() * freqs`, `cos`, `sin`. Numerically identical to [`compute_rope_embeddings`].
pub fn compute_rope_embeddings_pre(
    positions: Tensor<2, Int>,
    freqs: Tensor<1>,
) -> (Tensor<4>, Tensor<4>) {
    let positions_float = float_on_device(positions.float(), &freqs.device());
    let angles = positions_float.unsqueeze_dim::<3>(2) * freqs.unsqueeze::<3>();
    let cos = angles.clone().cos().unsqueeze_dim::<4>(2);
    let sin = angles.sin().unsqueeze_dim::<4>(2);
    (cos, sin)
}

/// Apply rotary embeddings to query and key tensors.
///
/// # Arguments
/// * `q` - Query tensor [batch, seq, n_heads, head_dim]
/// * `k` - Key tensor [batch, seq, n_kv_heads, head_dim]
/// * `cos` - Cosine embeddings [batch, seq, 1, half_dim]
/// * `sin` - Sine embeddings [batch, seq, 1, half_dim]
pub fn apply_rope(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: Tensor<4>,
    sin: Tensor<4>,
) -> (Tensor<4>, Tensor<4>) {
    let q_rotated = rotate_half(q, cos.clone(), sin.clone());
    let k_rotated = rotate_half(k, cos, sin);
    (q_rotated, k_rotated)
}

/// Apply rotary embeddings to only the first `rotary_dim` dimensions of each query/key head.
/// The remaining dimensions are passed through unchanged.
pub fn apply_rope_partial(
    q: Tensor<4>,
    k: Tensor<4>,
    cos: Tensor<4>,
    sin: Tensor<4>,
    rotary_dim: usize,
) -> (Tensor<4>, Tensor<4>) {
    let [batch, seq, q_heads, q_head_dim] = q.dims();
    let [_, _, k_heads, k_head_dim] = k.dims();

    // cos/sin are computed in f32 (from_floats); HF casts them to the activation dtype before the
    // RoPE multiply (`cos.to(dtype=x.dtype)`). Match that so bf16 q/k * cos/sin doesn't DTypeMismatch
    // (no-op on the f32 path). Done once here, before the rotate_half multiplies + the apply_rope delegate.
    let cos = cos.cast(q.dtype());
    let sin = sin.cast(q.dtype());

    if rotary_dim == q_head_dim && rotary_dim == k_head_dim {
        return apply_rope(q, k, cos, sin);
    }

    let q_rot = q
        .clone()
        .slice([0..batch, 0..seq, 0..q_heads, 0..rotary_dim]);
    let q_pass = q.slice([0..batch, 0..seq, 0..q_heads, rotary_dim..q_head_dim]);
    let k_rot = k
        .clone()
        .slice([0..batch, 0..seq, 0..k_heads, 0..rotary_dim]);
    let k_pass = k.slice([0..batch, 0..seq, 0..k_heads, rotary_dim..k_head_dim]);

    let q_rotated = rotate_half(q_rot, cos.clone(), sin.clone());
    let k_rotated = rotate_half(k_rot, cos, sin);

    (
        Tensor::cat(vec![q_rotated, q_pass], 3),
        Tensor::cat(vec![k_rotated, k_pass], 3),
    )
}

/// Rotate half of the tensor dimensions using RoPE.
fn rotate_half(x: Tensor<4>, cos: Tensor<4>, sin: Tensor<4>) -> Tensor<4> {
    let [batch, seq, n_heads, head_dim] = x.dims();
    let half_dim = head_dim / 2;

    // RoPE's cos/sin are computed in f32 (via `from_floats`), but the residual stream
    // can be a lower-precision float (e.g. bf16 when bf16 weights are loaded on CUDA).
    // Mixing dtypes in `*` triggers a DTypeMismatch, so align cos/sin to x's dtype.
    let x_dtype = x.dtype();
    let cos = cos.cast(x_dtype);
    let sin = sin.cast(x_dtype);

    // Split into first and second halves
    let x1 = x.clone().slice([0..batch, 0..seq, 0..n_heads, 0..half_dim]);
    let x2 = x.slice([0..batch, 0..seq, 0..n_heads, half_dim..head_dim]);

    // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    let rotated_x1 = x1.clone() * cos.clone() - x2.clone() * sin.clone();
    let rotated_x2 = x1 * sin + x2 * cos;

    // Concatenate back
    Tensor::cat(vec![rotated_x1, rotated_x2], 3)
}
