//! NVIDIA ModelOpt NVFP4/FP8 checkpoint ingestion helpers.
//!
//! The on-disk NVFP4 tensors are ModelOpt W4A16: `weight:[out,in/2]` low-nibble-first along K,
//! `weight_scale:[out,in/16]` raw E4M3 bytes, `weight_scale_2:f32` scalar. `input_scale` is
//! intentionally ignored for W4A16 because vLLM deletes it unread in the W4A16 path; keeping bf16/f32
//! activations is accuracy-positive relative to calibrated fp8 activations.
//!
//! B5.0c fallback note: fallback dequantizes expert sidecars to the existing fp8 expert path and
//! dequantizes `lm_head` to bf16, but SHARED expert projections still use the NVFP4 dense GEMV
//! sidecar. It is not a pure "no new kernels" artifact.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::MmapOptions;
use safetensors::{tensor::Metadata, Dtype, SafeTensors};

use crate::nvfp4::{dequant_nvfp4, repack_kmajor_to_outmajor};
#[cfg(feature = "cuda")]
use crate::qwen3_5::ExpertFp8;
use crate::qwen3_5::{ExpertNvfp4Parts, Qwen3_5LayerType, Qwen3_5MoeConfig};

#[derive(Clone, Debug)]
pub struct RawTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

#[derive(Debug)]
struct ShardData {
    path: PathBuf,
    mmap: memmap2::Mmap,
    data_start: usize,
    metadata: Metadata,
}

/// Reuses mmap and parsed safetensors metadata across many tensor reads from the same shard.
///
/// The pure helper APIs below intentionally keep their original signatures for golden/reference
/// callers; the checkpoint loader uses this reader directly so the multi-MB shard header is parsed
/// once per shard instead of once per tensor.
#[derive(Debug)]
pub struct ShardReader<'a> {
    dir: &'a Path,
    index: &'a BTreeMap<String, String>,
    shards: HashMap<String, ShardData>,
}

impl<'a> ShardReader<'a> {
    pub fn new(dir: &'a Path, index: &'a BTreeMap<String, String>) -> Self {
        Self {
            dir,
            index,
            shards: HashMap::new(),
        }
    }

    pub fn read_raw_tensor(&mut self, key: &str) -> Result<RawTensor, String> {
        let shard = self
            .index
            .get(key)
            .ok_or_else(|| format!("{key} missing from weight_map"))?
            .clone();
        if !self.shards.contains_key(&shard) {
            let data = self.load_shard(&shard)?;
            self.shards.insert(shard.clone(), data);
        }
        let shard_data = self.shards.get(&shard).unwrap();
        let info = shard_data.metadata.info(key).ok_or_else(|| {
            format!(
                "read tensor {key} from {}: tensor not found",
                shard_data.path.display()
            )
        })?;
        let start = shard_data
            .data_start
            .checked_add(info.data_offsets.0)
            .ok_or_else(|| {
                format!(
                    "{key}: data offset overflow in {}",
                    shard_data.path.display()
                )
            })?;
        let end = shard_data
            .data_start
            .checked_add(info.data_offsets.1)
            .ok_or_else(|| {
                format!(
                    "{key}: data offset overflow in {}",
                    shard_data.path.display()
                )
            })?;
        let data = shard_data
            .mmap
            .get(start..end)
            .ok_or_else(|| {
                format!(
                    "{key}: data offsets out of bounds in {}",
                    shard_data.path.display()
                )
            })?
            .to_vec();
        Ok(RawTensor {
            dtype: info.dtype,
            shape: info.shape.clone(),
            data,
        })
    }

    fn load_shard(&self, shard: &str) -> Result<ShardData, String> {
        let path = self.dir.join(shard);
        let file = File::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let mmap = unsafe { MmapOptions::new().map(&file) }
            .map_err(|e| format!("mmap {}: {e}", path.display()))?;
        let (header_len, metadata) = SafeTensors::read_metadata(&mmap)
            .map_err(|e| format!("parse {}: {e}", path.display()))?;
        Ok(ShardData {
            path,
            mmap,
            data_start: 8 + header_len,
            metadata,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantGroupKind {
    DenseFp8,
    Nvfp4,
}

#[derive(Clone, Debug)]
pub struct QuantGroup {
    pub base: String,
    pub kind: QuantGroupKind,
}

#[derive(Clone, Debug, Default)]
pub struct QuantizedNameAccounting {
    pub dense_fp8_groups: usize,
    pub nvfp4_expert_groups: usize,
    pub nvfp4_shared_groups: usize,
    pub nvfp4_lm_head_groups: usize,
    pub quantized_keys: usize,
    pub consumed_quantized: usize,
    pub unconsumed_quantized: Vec<String>,
}

impl QuantizedNameAccounting {
    pub fn pass(&self) -> bool {
        self.unconsumed_quantized.is_empty()
    }
}

#[derive(Clone, Debug)]
pub struct DenseFp8Parts {
    pub q_bytes_kn: Vec<u8>,
    pub scale_n: Vec<f32>,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct Nvfp4LinearParts {
    pub qw: Vec<u8>,
    pub bs: Vec<u8>,
    pub gscale: f32,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone, Debug)]
pub struct ExpertProjectionParts {
    pub qw: Vec<u8>,
    pub bs: Vec<u8>,
    pub gscale: f32,
    pub k: usize,
    pub n: usize,
}

pub fn read_raw_tensor(
    dir: &Path,
    index: &BTreeMap<String, String>,
    key: &str,
) -> Result<RawTensor, String> {
    let mut reader = ShardReader::new(dir, index);
    reader.read_raw_tensor(key)
}

pub fn f32_scalar(t: &RawTensor, key: &str) -> Result<f32, String> {
    if t.dtype != Dtype::F32 || !t.shape.is_empty() || t.data.len() != 4 {
        return Err(format!(
            "{key}: expected F32 scalar, got {:?} {:?} ({} bytes)",
            t.dtype,
            t.shape,
            t.data.len()
        ));
    }
    Ok(f32::from_le_bytes(t.data.as_slice().try_into().unwrap()))
}

pub fn collect_quant_groups(index_keys: impl IntoIterator<Item = String>) -> Vec<QuantGroup> {
    let mut by_base: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    for key in index_keys {
        if let Some(base) = key.strip_suffix(".input_scale") {
            by_base
                .entry(base.to_string())
                .or_default()
                .insert("input_scale");
        } else if let Some(base) = key.strip_suffix(".weight_scale_2") {
            by_base
                .entry(base.to_string())
                .or_default()
                .insert("weight_scale_2");
        } else if let Some(base) = key.strip_suffix(".weight_scale") {
            by_base
                .entry(base.to_string())
                .or_default()
                .insert("weight_scale");
        } else if let Some(base) = key.strip_suffix(".weight") {
            by_base
                .entry(base.to_string())
                .or_default()
                .insert("weight");
        }
    }
    by_base
        .into_iter()
        .filter_map(|(base, tags)| {
            let has_quant = tags.contains("input_scale") && tags.contains("weight_scale");
            if !has_quant || !tags.contains("weight") {
                return None;
            }
            let kind = if tags.contains("weight_scale_2") {
                QuantGroupKind::Nvfp4
            } else {
                QuantGroupKind::DenseFp8
            };
            Some(QuantGroup { base, kind })
        })
        .collect()
}

pub fn quant_group_keys(base: &str, kind: QuantGroupKind) -> Vec<String> {
    let mut keys = vec![
        format!("{base}.weight"),
        format!("{base}.weight_scale"),
        format!("{base}.input_scale"),
    ];
    if kind == QuantGroupKind::Nvfp4 {
        keys.push(format!("{base}.weight_scale_2"));
    }
    keys
}

pub fn dense_fp8_role(base: &str) -> Option<String> {
    let rest = base.strip_prefix("model.language_model.layers.")?;
    let (layer_s, tail) = rest.split_once('.')?;
    let layer: usize = layer_s.parse().ok()?;
    let tail = match tail {
        "linear_attn.in_proj_qkv" => "gdn.in_proj_qkv",
        "linear_attn.in_proj_z" => "gdn.in_proj_z",
        "linear_attn.out_proj" => "gdn.out_proj",
        "self_attn.q_proj" => "attn.q_proj",
        "self_attn.k_proj" => "attn.k_proj",
        "self_attn.v_proj" => "attn.v_proj",
        "self_attn.o_proj" => "attn.o_proj",
        _ => return None,
    };
    Some(format!("L{layer}.{tail}"))
}

pub fn shared_nvfp4_role(base: &str) -> Option<String> {
    let rest = base.strip_prefix("model.language_model.layers.")?;
    let (layer_s, tail) = rest.split_once('.')?;
    let layer: usize = layer_s.parse().ok()?;
    let tail = match tail {
        "mlp.shared_expert.gate_proj" => "moe.shared.gate_proj",
        "mlp.shared_expert.up_proj" => "moe.shared.up_proj",
        "mlp.shared_expert.down_proj" => "moe.shared.down_proj",
        _ => return None,
    };
    Some(format!("L{layer}.{tail}"))
}

pub fn parse_expert_base(base: &str) -> Option<(usize, usize, &'static str)> {
    let rest = base.strip_prefix("model.language_model.layers.")?;
    let (layer_s, rest) = rest.split_once(".mlp.experts.")?;
    let layer: usize = layer_s.parse().ok()?;
    let (expert_s, proj) = rest.split_once('.')?;
    let expert: usize = expert_s.parse().ok()?;
    let proj = match proj {
        "gate_proj" => "gate_proj",
        "up_proj" => "up_proj",
        "down_proj" => "down_proj",
        _ => return None,
    };
    Some((layer, expert, proj))
}

pub fn dense_fp8_parts(
    dir: &Path,
    index: &BTreeMap<String, String>,
    base: &str,
) -> Result<DenseFp8Parts, String> {
    let mut reader = ShardReader::new(dir, index);
    dense_fp8_parts_from_reader(&mut reader, base)
}

pub fn dense_fp8_parts_from_reader(
    reader: &mut ShardReader<'_>,
    base: &str,
) -> Result<DenseFp8Parts, String> {
    let weight_key = format!("{base}.weight");
    let scale_key = format!("{base}.weight_scale");
    let weight = reader.read_raw_tensor(&weight_key)?;
    let scale = reader.read_raw_tensor(&scale_key)?;
    if weight.dtype != Dtype::F8_E4M3 || weight.shape.len() != 2 {
        return Err(format!(
            "{weight_key}: expected F8_E4M3 [N,K], got {:?} {:?}",
            weight.dtype, weight.shape
        ));
    }
    let n = weight.shape[0];
    let k = weight.shape[1];
    if weight.data.len() != n * k {
        return Err(format!(
            "{weight_key}: byte length {} != N*K {}",
            weight.data.len(),
            n * k
        ));
    }
    let scalar = f32_scalar(&scale, &scale_key)?;
    let mut q_bytes_kn = vec![0u8; k * n];
    for nn in 0..n {
        for kk in 0..k {
            q_bytes_kn[kk * n + nn] = weight.data[nn * k + kk];
        }
    }
    Ok(DenseFp8Parts {
        q_bytes_kn,
        scale_n: vec![scalar; n],
        k,
        n,
    })
}

pub fn nvfp4_linear_parts(
    dir: &Path,
    index: &BTreeMap<String, String>,
    base: &str,
) -> Result<Nvfp4LinearParts, String> {
    let mut reader = ShardReader::new(dir, index);
    nvfp4_linear_parts_from_reader(&mut reader, base)
}

pub fn nvfp4_linear_parts_from_reader(
    reader: &mut ShardReader<'_>,
    base: &str,
) -> Result<Nvfp4LinearParts, String> {
    let weight_key = format!("{base}.weight");
    let scale_key = format!("{base}.weight_scale");
    let gscale_key = format!("{base}.weight_scale_2");
    let weight = reader.read_raw_tensor(&weight_key)?;
    let scale = reader.read_raw_tensor(&scale_key)?;
    let gscale = f32_scalar(&reader.read_raw_tensor(&gscale_key)?, &gscale_key)?;
    if weight.dtype != Dtype::U8 || weight.shape.len() != 2 {
        return Err(format!(
            "{weight_key}: expected U8 [N,K/2], got {:?} {:?}",
            weight.dtype, weight.shape
        ));
    }
    let n = weight.shape[0];
    let k = weight.shape[1] * 2;
    if k % 16 != 0 {
        return Err(format!("{weight_key}: K={k} is not divisible by 16"));
    }
    if scale.dtype != Dtype::F8_E4M3 || scale.shape != vec![n, k / 16] {
        return Err(format!(
            "{scale_key}: expected F8_E4M3 [{n},{}], got {:?} {:?}",
            k / 16,
            scale.dtype,
            scale.shape
        ));
    }
    Ok(Nvfp4LinearParts {
        qw: weight.data,
        bs: scale.data,
        gscale,
        k,
        n,
    })
}

pub fn expert_projection_parts(
    dir: &Path,
    index: &BTreeMap<String, String>,
    base: &str,
) -> Result<ExpertProjectionParts, String> {
    let mut reader = ShardReader::new(dir, index);
    expert_projection_parts_from_reader(&mut reader, base)
}

pub fn expert_projection_parts_from_reader(
    reader: &mut ShardReader<'_>,
    base: &str,
) -> Result<ExpertProjectionParts, String> {
    let p = nvfp4_linear_parts_from_reader(reader, base)?;
    Ok(ExpertProjectionParts {
        qw: p.qw,
        bs: p.bs,
        gscale: p.gscale,
        k: p.k,
        n: p.n,
    })
}

pub fn fuse_expert_nvfp4_parts(
    gate: ExpertProjectionParts,
    up: ExpertProjectionParts,
    down: ExpertProjectionParts,
) -> Result<ExpertNvfp4Parts, String> {
    if gate.k != up.k || gate.n != up.n {
        return Err(format!(
            "gate/up shape mismatch: gate K,N=({},{}), up K,N=({},{})",
            gate.k, gate.n, up.k, up.n
        ));
    }
    let h = gate.k;
    let i = gate.n;
    if down.k != i || down.n != h {
        return Err(format!(
            "down shape mismatch: got K,N=({},{}), expected ({i},{h})",
            down.k, down.n
        ));
    }
    let mut q_gu_kmajor = Vec::with_capacity((i * 2) * (h / 2));
    q_gu_kmajor.extend_from_slice(&gate.qw);
    q_gu_kmajor.extend_from_slice(&up.qw);
    let mut bs_gu = Vec::with_capacity((i * 2) * (h / 16));
    bs_gu.extend_from_slice(&gate.bs);
    bs_gu.extend_from_slice(&up.bs);
    Ok(ExpertNvfp4Parts {
        qw_gu_outmajor: repack_kmajor_to_outmajor(&q_gu_kmajor, h, i * 2),
        bs_gu,
        gscale_gu: [gate.gscale, up.gscale],
        qw_dn_outmajor: repack_kmajor_to_outmajor(&down.qw, i, h),
        bs_dn: down.bs,
        gscale_dn: down.gscale,
    })
}

#[cfg(feature = "cuda")]
pub fn quantize_dequantized_expert_to_fp8<B: burn::prelude::Backend>(
    parts: &[ExpertNvfp4Parts],
    h: usize,
    i: usize,
    device: &B::Device,
) -> ExpertFp8<B> {
    let e = parts.len();
    let mut q_gu = Vec::with_capacity(e * h * i * 2);
    let mut s_gu = Vec::with_capacity(e * i * 2);
    let mut q_dn = Vec::with_capacity(e * i * h);
    let mut s_dn = Vec::with_capacity(e * h);
    for part in parts {
        let mut gu_gscale = vec![part.gscale_gu[0]; i];
        gu_gscale.extend(std::iter::repeat_n(part.gscale_gu[1], i));
        let gu = crate::nvfp4::dequant_nvfp4_outmajor(
            &part.qw_gu_outmajor,
            &part.bs_gu,
            &gu_gscale,
            h,
            i * 2,
        );
        let (q, s) = crate::w8a16::quantize_e4m3_per_channel(&gu, h, i * 2);
        q_gu.extend(q.into_iter().map(|b| b as i8));
        s_gu.extend(s);

        let dn = crate::nvfp4::dequant_nvfp4_outmajor(
            &part.qw_dn_outmajor,
            &part.bs_dn,
            &[part.gscale_dn],
            i,
            h,
        );
        let (q, s) = crate::w8a16::quantize_e4m3_per_channel(&dn, i, h);
        q_dn.extend(q.into_iter().map(|b| b as i8));
        s_dn.extend(s);
    }
    ExpertFp8 {
        q_gu: burn::tensor::Tensor::<B, 3, burn::tensor::Int>::from_data_dtype(
            burn::tensor::TensorData::new(q_gu, [e, h, i * 2]),
            device,
            burn::tensor::DType::I8,
        ),
        s_gu: burn::tensor::Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(s_gu, [e, i * 2]),
            device,
        ),
        q_dn: burn::tensor::Tensor::<B, 3, burn::tensor::Int>::from_data_dtype(
            burn::tensor::TensorData::new(q_dn, [e, i, h]),
            device,
            burn::tensor::DType::I8,
        ),
        s_dn: burn::tensor::Tensor::<B, 2>::from_data(
            burn::tensor::TensorData::new(s_dn, [e, h]),
            device,
        ),
        e,
        h,
        i,
    }
}

pub fn dequant_nvfp4_linear_to_kn(parts: &Nvfp4LinearParts) -> Vec<f32> {
    dequant_nvfp4(&parts.qw, &parts.bs, parts.gscale, parts.k, parts.n)
}

pub fn expected_quantized_counts(config: &Qwen3_5MoeConfig) -> (usize, usize, usize, usize) {
    let dense = config
        .layer_types
        .iter()
        .map(|kind| match kind {
            Qwen3_5LayerType::LinearAttention => 3usize,
            Qwen3_5LayerType::FullAttention => 4usize,
        })
        .sum::<usize>();
    let experts = config.num_hidden_layers * config.num_experts * 3;
    let shared = config.num_hidden_layers * 3;
    let lm_head = 1;
    (dense, experts, shared, lm_head)
}

pub fn shard_index(dir: &Path) -> Result<BTreeMap<String, String>, String> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .map_err(|e| format!("read model.safetensors.index.json: {e}"))?;
    let pairs = crate::qwen3_5::parse_weight_map(&text)
        .map_err(|e| format!("parse model.safetensors.index.json: {e}"))?;
    Ok(pairs.into_iter().collect())
}

pub fn checkpoint_dir(path: impl AsRef<Path>) -> PathBuf {
    path.as_ref().to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nvfp4::repack_outmajor_to_kmajor;

    #[test]
    fn name_remap_table_covers_quantized_projection_roles() {
        assert_eq!(
            dense_fp8_role("model.language_model.layers.3.self_attn.q_proj").as_deref(),
            Some("L3.attn.q_proj")
        );
        assert_eq!(
            dense_fp8_role("model.language_model.layers.4.linear_attn.out_proj").as_deref(),
            Some("L4.gdn.out_proj")
        );
        assert_eq!(
            shared_nvfp4_role("model.language_model.layers.2.mlp.shared_expert.down_proj")
                .as_deref(),
            Some("L2.moe.shared.down_proj")
        );
        assert_eq!(
            parse_expert_base("model.language_model.layers.7.mlp.experts.12.up_proj"),
            Some((7, 12, "up_proj"))
        );
        assert!(dense_fp8_role("model.language_model.layers.0.linear_attn.in_proj_a").is_none());
    }

    #[test]
    fn synthetic_kmajor_to_outmajor_fixture_preserves_nibble_order() {
        let k = 16;
        let n = 4;
        let mut kmajor = vec![0u8; n * (k / 2)];
        for nn in 0..n {
            for kk_pair in 0..(k / 2) {
                let even = ((nn * k + kk_pair * 2) & 0x0f) as u8;
                let odd = ((nn * k + kk_pair * 2 + 1) & 0x0f) as u8;
                kmajor[nn * (k / 2) + kk_pair] = even | (odd << 4);
            }
        }
        let outmajor = repack_kmajor_to_outmajor(&kmajor, k, n);
        assert_eq!(outmajor[0], 0x00);
        assert_eq!(outmajor[1], 0x00);
        assert_eq!(outmajor[n / 2], 0x11);
        assert_eq!(outmajor[(n / 2) + 1], 0x11);
        assert_eq!(repack_outmajor_to_kmajor(&outmajor, k, n), kmajor);
    }

    #[test]
    fn fuse_gate_up_assigns_gscale_halves() {
        let h = 16;
        let i = 16;
        let mk = |gscale| ExpertProjectionParts {
            qw: vec![0x10; i * (h / 2)],
            bs: vec![0x38; i * (h / 16)],
            gscale,
            k: h,
            n: i,
        };
        let down = ExpertProjectionParts {
            qw: vec![0x32; h * (i / 2)],
            bs: vec![0x40; h * (i / 16).max(1)],
            gscale: 3.0,
            k: i,
            n: h,
        };
        let fused = fuse_expert_nvfp4_parts(mk(1.0), mk(2.0), down).unwrap();
        assert_eq!(fused.gscale_gu, [1.0, 2.0]);
        assert_eq!(fused.gscale_dn, 3.0);
        assert_eq!(fused.bs_gu.len(), (i * 2) * (h / 16));
    }

    #[test]
    fn quant_group_collection_classifies_fp8_and_nvfp4() {
        let keys = [
            "x.weight",
            "x.weight_scale",
            "x.input_scale",
            "y.weight",
            "y.weight_scale",
            "y.weight_scale_2",
            "y.input_scale",
            "z.weight",
        ]
        .into_iter()
        .map(str::to_string);
        let groups = collect_quant_groups(keys);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].kind, QuantGroupKind::DenseFp8);
        assert_eq!(groups[1].kind, QuantGroupKind::Nvfp4);
    }
}
