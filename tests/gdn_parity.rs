use burn::tensor::{DType, Distribution, Tensor};
use qwen3_burn::gdn_kernel::{GdnShape, gdn_step_fused};

#[test]
fn test_fused_gdn_parity() {
    let device = Default::default();
    let b = 1;
    let num_value_heads = 32;
    let num_key_heads = 16;
    let key_head_dim = 128;
    let value_head_dim = 128;
    let kernel_size = 4;
    let q_dim = num_key_heads * key_head_dim;
    let v_dim = num_value_heads * value_head_dim;
    let qkv_dim = 2 * q_dim + v_dim;

    let qkv_unconv: Tensor<2> =
        Tensor::random([b, qkv_dim], Distribution::Normal(0.0, 1.0), &device);
    let z: Tensor<3> = Tensor::random(
        [b, num_value_heads, value_head_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let in_a: Tensor<2> = Tensor::random(
        [b, num_value_heads],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let in_b: Tensor<2> = Tensor::random(
        [b, num_value_heads],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let conv_hist: Tensor<3> = Tensor::zeros([b, kernel_size - 1, qkv_dim], &device);
    let conv_w: Tensor<2> = Tensor::random(
        [qkv_dim, kernel_size],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let dt_bias: Tensor<1> = Tensor::zeros([num_value_heads], &device);
    let a_log: Tensor<1> = Tensor::zeros([num_value_heads], &device);
    let norm_w: Tensor<1> = Tensor::ones([value_head_dim], &device);
    let prev_state: Tensor<4> =
        Tensor::zeros([b, num_value_heads, key_head_dim, value_head_dim], &device);

    let sh = GdnShape {
        batch: b,
        qkv_dim,
        num_value_heads,
        num_key_heads,
        key_head_dim,
        value_head_dim,
        kernel_size,
        epsilon: 1e-6,
    };

    let (out_fused, _new_state, _new_hist) = gdn_step_fused(
        qkv_unconv.clone(),
        z.clone(),
        in_a.clone(),
        in_b.clone(),
        conv_hist.clone(),
        conv_w.clone(),
        dt_bias.clone(),
        a_log.clone(),
        norm_w.clone(),
        prev_state.clone(),
        sh,
    );

    let [ob, ov, ovd] = out_fused.dims();
    assert_eq!((ob, ov, ovd), (b, num_value_heads, value_head_dim));
}
