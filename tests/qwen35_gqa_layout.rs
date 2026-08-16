use burn::tensor::{Device, Tensor, TensorData};

#[test]
fn qwen35_gqa_repeat_layout_matches_flash_decode_grouping() {
    let device = Device::flex();
    let (b, sk, hkv, d) = (2usize, 3usize, 2usize, 4usize);
    let (hq, n_rep) = (16usize, 8usize);
    let mut data = vec![0.0f32; b * sk * hkv * d];
    for bb in 0..b {
        for t in 0..sk {
            for h in 0..hkv {
                for e in 0..d {
                    let idx = (((bb * sk + t) * hkv + h) * d) + e;
                    data[idx] = if h == 0 { 1.0 } else { 2.0 };
                }
            }
        }
    }

    let kv = Tensor::<4>::from_data(TensorData::new(data, [b, sk, hkv, d]), &device);
    let expanded = kv
        .unsqueeze_dim::<5>(3)
        .repeat(&[1, 1, 1, n_rep, 1])
        .flatten(2, 3);
    assert_eq!(expanded.dims(), [b, sk, hq, d]);

    let got = expanded.into_data().to_vec::<f32>().unwrap();
    for bb in 0..b {
        for t in 0..sk {
            for h in 0..hq {
                let want = if h < n_rep { 1.0 } else { 2.0 };
                for e in 0..d {
                    let idx = (((bb * sk + t) * hq + h) * d) + e;
                    assert_eq!(
                        got[idx], want,
                        "expanded head {h} at batch {bb}, token {t}, dim {e}"
                    );
                }
            }
        }
    }
}
