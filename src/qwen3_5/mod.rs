//! Qwen3.6 / Qwen3.5-MoE hybrid text tower skeleton.
//!
//! Lane L1.1 only needs the correctly-shaped module tree and weight-map verification/loading.
//! The forward paths are intentionally left unimplemented for the later full-attention, GDN,
//! shared-MoE, and MTP increments.

#![allow(non_camel_case_types, non_snake_case)]

use std::collections::BTreeMap;

use burn::{
    module::{
        AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault,
        ModuleMapper, ModuleVisitor, Param, ParamId,
    },
    nn::{Embedding, Linear, RmsNorm},
    prelude::Device,
    tensor::{
        DType, Distribution, IndexingUpdateOp, Int, Shape, Tensor, TensorData, activation::sigmoid,
        activation::silu, activation::softmax, module::attention_fallback as attention,
        ops::AttentionModuleOptions,
    },
};

use crate::{
    cache::{GdnStateCache, KVCache, Qwen3_5HybridCache, Qwen3_5HybridLayerCache},
    expert_stream::ExpertSlotPool,
    linear2d::{Precision, linear3},
    nvfp4_linear::QuantLinear,
    rope::{apply_rope_partial, compute_rope_embeddings, compute_rope_embeddings_pre},
};

#[cfg(feature = "cuda")]
pub trait Qwen3_5DenseQuantBackend {}

#[cfg(feature = "cuda")]
impl Qwen3_5DenseQuantBackend for () {}

#[cfg(not(feature = "cuda"))]
pub trait Qwen3_5DenseQuantBackend {}

#[cfg(not(feature = "cuda"))]
impl Qwen3_5DenseQuantBackend for () {}

#[cfg(feature = "cubecl-gpu")]
const QWEN35_FUSED_MOE_MAX_T: usize = 16;

#[cfg(feature = "cubecl-gpu")]
static QWEN35_FUSED_MOE_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "cubecl-gpu")]
pub fn set_qwen35_fused_moe_enabled(enabled: bool) {
    QWEN35_FUSED_MOE_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl-gpu")]
fn qwen35_fused_moe_enabled() -> bool {
    QWEN35_FUSED_MOE_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl-gpu")]
static QWEN35_FUSED_GDN_ENABLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(true);

#[cfg(feature = "cubecl-gpu")]
pub fn set_qwen35_fused_gdn_enabled(enabled: bool) {
    QWEN35_FUSED_GDN_ENABLED.store(enabled, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cubecl-gpu")]
fn qwen35_fused_gdn_enabled() -> bool {
    if let Ok(val) = std::env::var("QWEN35_FUSED_GDN") {
        return val != "0" && val != "false";
    }
    QWEN35_FUSED_GDN_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

#[cfg(feature = "cubecl-gpu")]
fn flash_decode_n_splits(sk: usize) -> usize {
    if sk < 1024 {
        1
    } else {
        sk.div_ceil(512).clamp(1, 32)
    }
}

/// Numerically stable softmax + top-k on the host, lowest expert id on ties.
fn host_softmax_topk(
    logits: &[f32],
    tokens: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
) -> (Vec<i64>, Vec<f32>) {
    assert_eq!(logits.len(), tokens * num_experts);
    let mut sel_idx = Vec::with_capacity(tokens * top_k);
    let mut sel_w = Vec::with_capacity(tokens * top_k);
    for tok in 0..tokens {
        let row = &logits[tok * num_experts..(tok + 1) * num_experts];
        let mut max_l = f32::NEG_INFINITY;
        for &l in row {
            if l > max_l {
                max_l = l;
            }
        }
        let mut exp_sum = 0.0f32;
        let mut probs: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .map(|(i, &l)| {
                let e = (l - max_l).exp();
                exp_sum += e;
                (i, e)
            })
            .collect();
        for p in &mut probs {
            p.1 /= exp_sum;
        }
        probs.sort_by(|a, b| match b.1.partial_cmp(&a.1) {
            Some(std::cmp::Ordering::Equal) | None => a.0.cmp(&b.0),
            Some(ord) => ord,
        });
        probs.truncate(top_k);
        let mut wsum = 0.0f32;
        for (_, w) in &probs {
            wsum += *w;
        }
        let wsum_inv = if norm_topk_prob && wsum > 1e-20 {
            1.0 / wsum
        } else {
            1.0
        };
        for (i, w) in probs {
            sel_idx.push(i as i64);
            sel_w.push(w * wsum_inv);
        }
    }
    (sel_idx, sel_w)
}

#[derive(Clone, Debug)]
pub struct QuantSidecar(pub Option<QuantLinear>);

#[derive(Clone, Debug)]
pub struct ExpertFp8 {
    pub q_gu: Tensor<3, Int>,
    pub s_gu: Tensor<2>,
    pub q_dn: Tensor<3, Int>,
    pub s_dn: Tensor<2>,
    pub e: usize,
    pub h: usize,
    pub i: usize,
}

/// Per-expert host NVFP4 parts consumed by [`ExpertNvfp4::from_expert_parts`].
///
/// `qw_gu_outmajor` is `[H, I]` because fused gate/up has `K=H, N=2I`, and the output-major NVFP4
/// byte layout stores `N/2` bytes per reduction row. `bs_gu` is `[2I, H/16]`; `gscale_gu[0]`
/// applies to the gate half and `gscale_gu[1]` applies to the up half. Down uses `qw_dn_outmajor:
/// [I, H/2]`, `bs_dn: [H, I/16]`, and one per-expert global scale.
#[derive(Clone, Debug)]
pub struct ExpertNvfp4Parts {
    pub qw_gu_outmajor: Vec<u8>,
    pub bs_gu: Vec<u8>,
    pub gscale_gu: [f32; 2],
    pub qw_dn_outmajor: Vec<u8>,
    pub bs_dn: Vec<u8>,
    pub gscale_dn: f32,
}

/// Stacked NVFP4 expert bytes, built after checkpoint tensors have been repacked.
///
/// This sidecar is deliberately independent of the bf16 expert params: B5.0 loaders build it last
/// from per-expert checkpoint bytes, then may replace the bf16 params with tiny placeholders. Forward
/// wiring is intentionally left for B5.3.
#[derive(Clone, Debug)]
pub struct ExpertNvfp4 {
    /// Output-major fused gate/up E2M1 bytes `[E, H, I]` for logical `K=H, N=2I`.
    pub qw_gu: Tensor<3, Int>,
    /// Fused gate/up E4M3 block-scale bytes `[E, 2I, H/16]`.
    pub bs_gu: Tensor<3, Int>,
    /// Per-expert, per-half ModelOpt `weight_scale_2` values `[E,2]`: gate then up.
    pub gscale_gu: Tensor<2>,
    /// Output-major down E2M1 bytes `[E, I, H/2]` for logical `K=I, N=H`.
    pub qw_dn: Tensor<3, Int>,
    /// Down E4M3 block-scale bytes `[E, H, I/16]`.
    pub bs_dn: Tensor<3, Int>,
    /// Per-expert down ModelOpt `weight_scale_2` values `[E]`.
    pub gscale_dn: Tensor<1>,
    pub e: usize,
    pub h: usize,
    pub i: usize,
}

#[derive(Clone, Debug)]
pub struct ExpertNvfp4Sidecar(pub Option<ExpertNvfp4>);

#[derive(Clone, Debug)]
pub struct ExpertQuantSidecar(pub Option<ExpertFp8>);

impl ExpertNvfp4 {
    /// Stack already-repacked per-expert NVFP4 host parts onto `device`.
    pub fn from_expert_parts(
        parts: Vec<ExpertNvfp4Parts>,
        h: usize,
        i: usize,
        device: &Device,
    ) -> Self {
        let e = parts.len();
        assert!(e > 0, "ExpertNvfp4::from_expert_parts: no experts");
        assert_eq!(
            h % 16,
            0,
            "ExpertNvfp4::from_expert_parts: hidden H must be divisible by 16, got {h}"
        );
        assert_eq!(
            i % 16,
            0,
            "ExpertNvfp4::from_expert_parts: inner I must be divisible by 16, got {i}"
        );
        assert_eq!(
            h % 2,
            0,
            "ExpertNvfp4::from_expert_parts: hidden H must be even for down output-major bytes, got {h}"
        );

        let gu_q_len = h * i;
        let gu_bs_len = (i * 2) * (h / 16);
        let dn_q_len = i * (h / 2);
        let dn_bs_len = h * (i / 16);
        let mut qw_gu = Vec::with_capacity(e * gu_q_len);
        let mut bs_gu = Vec::with_capacity(e * gu_bs_len);
        let mut gscale_gu = Vec::with_capacity(e * 2);
        let mut qw_dn = Vec::with_capacity(e * dn_q_len);
        let mut bs_dn = Vec::with_capacity(e * dn_bs_len);
        let mut gscale_dn = Vec::with_capacity(e);

        for (expert, part) in parts.into_iter().enumerate() {
            assert_eq!(
                part.qw_gu_outmajor.len(),
                gu_q_len,
                "ExpertNvfp4 expert {expert}: qw_gu_outmajor length {} != H*I = {gu_q_len}",
                part.qw_gu_outmajor.len()
            );
            assert_eq!(
                part.bs_gu.len(),
                gu_bs_len,
                "ExpertNvfp4 expert {expert}: bs_gu length {} != 2I*(H/16) = {gu_bs_len}",
                part.bs_gu.len()
            );
            assert_eq!(
                part.qw_dn_outmajor.len(),
                dn_q_len,
                "ExpertNvfp4 expert {expert}: qw_dn_outmajor length {} != I*(H/2) = {dn_q_len}",
                part.qw_dn_outmajor.len()
            );
            assert_eq!(
                part.bs_dn.len(),
                dn_bs_len,
                "ExpertNvfp4 expert {expert}: bs_dn length {} != H*(I/16) = {dn_bs_len}",
                part.bs_dn.len()
            );
            assert!(
                part.gscale_gu.iter().all(|v| v.is_finite() && *v > 0.0)
                    && part.gscale_dn.is_finite()
                    && part.gscale_dn > 0.0,
                "ExpertNvfp4 expert {expert}: gscale values must be finite and positive"
            );
            qw_gu.extend(part.qw_gu_outmajor.into_iter().map(|b| b as i8));
            bs_gu.extend(part.bs_gu.into_iter().map(|b| b as i8));
            gscale_gu.extend(part.gscale_gu);
            qw_dn.extend(part.qw_dn_outmajor.into_iter().map(|b| b as i8));
            bs_dn.extend(part.bs_dn.into_iter().map(|b| b as i8));
            gscale_dn.push(part.gscale_dn);
        }

        Self {
            qw_gu: Tensor::<3, Int>::from_data(
                TensorData::new(qw_gu, [e, h, i]),
                (device, DType::I8),
            ),
            bs_gu: Tensor::<3, Int>::from_data(
                TensorData::new(bs_gu, [e, i * 2, h / 16]),
                (device, DType::I8),
            ),
            gscale_gu: Tensor::<2>::from_data(TensorData::new(gscale_gu, [e, 2]), device),
            qw_dn: Tensor::<3, Int>::from_data(
                TensorData::new(qw_dn, [e, i, h / 2]),
                (device, DType::I8),
            ),
            bs_dn: Tensor::<3, Int>::from_data(
                TensorData::new(bs_dn, [e, h, i / 16]),
                (device, DType::I8),
            ),
            gscale_dn: Tensor::<1>::from_data(TensorData::new(gscale_dn, [e]), device),
            e,
            h,
            i,
        }
    }
}

impl Module for QuantSidecar {
    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn to_device(self, _: &Device) -> Self {
        self
    }

    fn fork(self, _: &Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        devices
    }
}

impl AutodiffModule for QuantSidecar {
    fn valid(&self) -> Self {
        QuantSidecar(None)
    }

    fn from_inner(_module: Self) -> Self {
        QuantSidecar(None)
    }
}

impl ModuleDisplayDefault for QuantSidecar {
    fn content(&self, content: Content) -> Option<Content> {
        let state = if self.0.is_some() { "Some" } else { "None" };
        content.add_formatted(&state).optional()
    }
}

impl ModuleDisplay for QuantSidecar {}

impl Module for ExpertQuantSidecar {
    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn to_device(self, _: &Device) -> Self {
        self
    }

    fn fork(self, _: &Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        devices
    }
}

impl AutodiffModule for ExpertQuantSidecar {
    fn valid(&self) -> Self {
        ExpertQuantSidecar(None)
    }

    fn from_inner(_module: Self) -> Self {
        ExpertQuantSidecar(None)
    }
}

impl ModuleDisplayDefault for ExpertQuantSidecar {
    fn content(&self, content: Content) -> Option<Content> {
        let state = if self.0.is_some() { "Some" } else { "None" };
        content.add_formatted(&state).optional()
    }
}

impl ModuleDisplay for ExpertQuantSidecar {}

impl Module for ExpertNvfp4Sidecar {
    fn visit<V: ModuleVisitor>(&self, _visitor: &mut V) {}

    fn map<M: ModuleMapper>(self, _mapper: &mut M) -> Self {
        self
    }

    fn to_device(self, _: &Device) -> Self {
        self
    }

    fn fork(self, _: &Device) -> Self {
        self
    }

    fn collect_devices(&self, devices: Devices) -> Devices {
        devices
    }
}

impl AutodiffModule for ExpertNvfp4Sidecar {
    fn valid(&self) -> Self {
        ExpertNvfp4Sidecar(None)
    }

    fn from_inner(_module: Self) -> Self {
        ExpertNvfp4Sidecar(None)
    }
}

impl ModuleDisplayDefault for ExpertNvfp4Sidecar {
    fn content(&self, content: Content) -> Option<Content> {
        let state = if self.0.is_some() { "Some" } else { "None" };
        content.add_formatted(&state).optional()
    }
}

impl ModuleDisplay for ExpertNvfp4Sidecar {}

#[cfg(feature = "cubecl-gpu")]
fn ql3(q: &QuantSidecar, lin: &Linear, x: Tensor<3>, prec: Precision) -> Tensor<3> {
    match &q.0 {
        Some(ql) => ql.forward3(x.cast(DType::F32)),
        None => linear3(lin, x, prec),
    }
}

#[cfg(not(feature = "cubecl-gpu"))]
fn ql3(q: &QuantSidecar, lin: &Linear, x: Tensor<3>, prec: Precision) -> Tensor<3> {
    let _ = q;
    linear3(lin, x, prec)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen3_5LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug)]
pub struct Qwen3_5MoeConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub layer_types: Vec<Qwen3_5LayerType>,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub partial_rotary_factor: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub mtp_num_hidden_layers: usize,
}

impl Default for Qwen3_5MoeConfig {
    fn default() -> Self {
        let mut layer_types = Vec::with_capacity(40);
        for i in 0..40 {
            layer_types.push(if i % 4 == 3 {
                Qwen3_5LayerType::FullAttention
            } else {
                Qwen3_5LayerType::LinearAttention
            });
        }

        Self {
            vocab_size: 248_320,
            hidden_size: 2048,
            num_hidden_layers: 40,
            layer_types,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            head_dim: 256,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            mrope_section: [11, 11, 10],
            num_experts: 256,
            num_experts_per_tok: 8,
            norm_topk_prob: true,
            moe_intermediate_size: 512,
            shared_expert_intermediate_size: 512,
            linear_key_head_dim: 128,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            mtp_num_hidden_layers: 1,
        }
    }
}

impl Qwen3_5MoeConfig {
    pub fn from_hf_config_str(json: &str) -> Result<Self, String> {
        let text =
            extract_object(json, "text_config").ok_or("config.json is missing text_config")?;
        let mut cfg = Self::default();

        cfg.vocab_size = get_usize(text, "vocab_size").unwrap_or(cfg.vocab_size);
        cfg.hidden_size = get_usize(text, "hidden_size").unwrap_or(cfg.hidden_size);
        cfg.num_hidden_layers =
            get_usize(text, "num_hidden_layers").unwrap_or(cfg.num_hidden_layers);
        cfg.num_attention_heads =
            get_usize(text, "num_attention_heads").unwrap_or(cfg.num_attention_heads);
        cfg.num_key_value_heads =
            get_usize(text, "num_key_value_heads").unwrap_or(cfg.num_key_value_heads);
        cfg.head_dim = get_usize(text, "head_dim").unwrap_or(cfg.head_dim);
        cfg.partial_rotary_factor =
            get_f64(text, "partial_rotary_factor").unwrap_or(cfg.partial_rotary_factor);
        cfg.rms_norm_eps = get_f64(text, "rms_norm_eps").unwrap_or(cfg.rms_norm_eps);
        cfg.rope_theta = get_f64(text, "rope_theta").unwrap_or(cfg.rope_theta);
        cfg.num_experts = get_usize(text, "num_experts").unwrap_or(cfg.num_experts);
        cfg.num_experts_per_tok =
            get_usize(text, "num_experts_per_tok").unwrap_or(cfg.num_experts_per_tok);
        cfg.norm_topk_prob = get_bool(text, "norm_topk_prob").unwrap_or(cfg.norm_topk_prob);
        cfg.moe_intermediate_size =
            get_usize(text, "moe_intermediate_size").unwrap_or(cfg.moe_intermediate_size);
        cfg.shared_expert_intermediate_size = get_usize(text, "shared_expert_intermediate_size")
            .unwrap_or(cfg.shared_expert_intermediate_size);
        cfg.linear_key_head_dim =
            get_usize(text, "linear_key_head_dim").unwrap_or(cfg.linear_key_head_dim);
        cfg.linear_num_key_heads =
            get_usize(text, "linear_num_key_heads").unwrap_or(cfg.linear_num_key_heads);
        cfg.linear_num_value_heads =
            get_usize(text, "linear_num_value_heads").unwrap_or(cfg.linear_num_value_heads);
        cfg.linear_value_head_dim =
            get_usize(text, "linear_value_head_dim").unwrap_or(cfg.linear_value_head_dim);
        cfg.linear_conv_kernel_dim =
            get_usize(text, "linear_conv_kernel_dim").unwrap_or(cfg.linear_conv_kernel_dim);
        cfg.mtp_num_hidden_layers =
            get_usize(text, "mtp_num_hidden_layers").unwrap_or(cfg.mtp_num_hidden_layers);
        if let Some(section) = get_usize_array(text, "mrope_section") {
            if section.len() == 3 {
                cfg.mrope_section = [section[0], section[1], section[2]];
            }
        }
        if let Some(types) = get_string_array(text, "layer_types") {
            cfg.layer_types = types
                .iter()
                .map(|kind| match kind.as_str() {
                    "linear_attention" => Ok(Qwen3_5LayerType::LinearAttention),
                    "full_attention" => Ok(Qwen3_5LayerType::FullAttention),
                    other => Err(format!("unsupported layer type {other:?}")),
                })
                .collect::<Result<Vec<_>, _>>()?;
        }

        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_hf_config_file(path: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read config.json: {e}"))?;
        Self::from_hf_config_str(&text)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(format!(
                "layer_types has {} entries but num_hidden_layers is {}",
                self.layer_types.len(),
                self.num_hidden_layers
            ));
        }
        if self.num_experts_per_tok == 0 || self.num_experts_per_tok > self.num_experts {
            return Err(format!(
                "num_experts_per_tok={} must be in 1..={}",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        Ok(())
    }

    pub fn init_causal_lm(&self, device: &Device) -> Qwen3_5MoeForCausalLM {
        let model = Qwen3_5Model {
            config: (self.clone()),
            embed_tokens: lazy_embedding(self.vocab_size, self.hidden_size, device),
            layers: self
                .layer_types
                .iter()
                .map(|kind| match kind {
                    Qwen3_5LayerType::LinearAttention => {
                        Qwen3_5DecoderLayer::Linear(self.init_gdn_layer(device))
                    }
                    Qwen3_5LayerType::FullAttention => {
                        Qwen3_5DecoderLayer::Full(self.init_full_layer(device))
                    }
                })
                .collect(),
            norm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
        };
        let mtp = Qwen3_5MtpBlock {
            pre_fc_norm_embedding: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            pre_fc_norm_hidden: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            fc: lazy_linear(self.hidden_size * 2, self.hidden_size, device),
            fc_fp8: QuantSidecar(None),
            layers: (0..self.mtp_num_hidden_layers)
                .map(|_| self.init_full_layer(device))
                .collect(),
            norm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
        };
        Qwen3_5MoeForCausalLM {
            model,
            lm_head: lazy_linear(self.hidden_size, self.vocab_size, device),
            lm_head_quant: QuantSidecar(None),
            mtp,
        }
    }

    fn init_gdn_layer(&self, device: &Device) -> Qwen3_5GdnLayer {
        let qkv = self.linear_qkv_dim();
        let v = self.linear_v_dim();
        Qwen3_5GdnLayer {
            input_layernorm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            linear_attn: Qwen3_5GdnAttention {
                in_proj_qkv: lazy_linear(self.hidden_size, qkv, device),
                in_proj_qkv_fp8: QuantSidecar(None),
                in_proj_a: lazy_linear(self.hidden_size, self.linear_num_value_heads, device),
                in_proj_a_fp8: QuantSidecar(None),
                in_proj_b: lazy_linear(self.hidden_size, self.linear_num_value_heads, device),
                in_proj_b_fp8: QuantSidecar(None),
                in_proj_z: lazy_linear(self.hidden_size, v, device),
                in_proj_z_fp8: QuantSidecar(None),
                A_log: lazy_param1([self.linear_num_value_heads], device),
                dt_bias: lazy_param1([self.linear_num_value_heads], device),
                conv1d: Qwen3_5Conv1d {
                    weight: lazy_param3([qkv, 1, self.linear_conv_kernel_dim], device),
                },
                norm: lazy_rms_norm(self.linear_value_head_dim, self.rms_norm_eps, device),
                out_proj: lazy_linear(v, self.hidden_size, device),
                out_proj_fp8: QuantSidecar(None),
            },
            post_attention_layernorm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            mlp: self.init_mlp(device),
        }
    }

    fn init_full_layer(&self, device: &Device) -> Qwen3_5FullAttnLayer {
        Qwen3_5FullAttnLayer {
            input_layernorm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            self_attn: Qwen3_5FullAttention {
                q_proj: lazy_linear(self.hidden_size, self.full_q_proj_dim(), device),
                q_proj_fp8: QuantSidecar(None),
                k_proj: lazy_linear(self.hidden_size, self.full_kv_dim(), device),
                k_proj_fp8: QuantSidecar(None),
                v_proj: lazy_linear(self.hidden_size, self.full_kv_dim(), device),
                v_proj_fp8: QuantSidecar(None),
                o_proj: lazy_linear(self.full_o_in_dim(), self.hidden_size, device),
                o_proj_fp8: QuantSidecar(None),
                q_norm: lazy_rms_norm(self.head_dim, self.rms_norm_eps, device),
                k_norm: lazy_rms_norm(self.head_dim, self.rms_norm_eps, device),
            },
            post_attention_layernorm: lazy_rms_norm(self.hidden_size, self.rms_norm_eps, device),
            mlp: self.init_mlp(device),
        }
    }

    fn init_mlp(&self, device: &Device) -> Qwen3_5SharedMoeBlock {
        Qwen3_5SharedMoeBlock {
            gate: lazy_linear(self.hidden_size, self.num_experts, device),
            experts: Qwen3_5FusedExperts {
                gate_up_proj: lazy_param3(
                    [
                        self.num_experts,
                        self.moe_intermediate_size * 2,
                        self.hidden_size,
                    ],
                    device,
                ),
                down_proj: lazy_param3(
                    [
                        self.num_experts,
                        self.hidden_size,
                        self.moe_intermediate_size,
                    ],
                    device,
                ),
                fp8: ExpertQuantSidecar(None),
                nvfp4: ExpertNvfp4Sidecar(None),
            },
            shared_expert: Qwen3_5SharedExpert {
                gate_proj: lazy_linear(
                    self.hidden_size,
                    self.shared_expert_intermediate_size,
                    device,
                ),
                gate_proj_fp8: QuantSidecar(None),
                up_proj: lazy_linear(
                    self.hidden_size,
                    self.shared_expert_intermediate_size,
                    device,
                ),
                up_proj_fp8: QuantSidecar(None),
                down_proj: lazy_linear(
                    self.shared_expert_intermediate_size,
                    self.hidden_size,
                    device,
                ),
                down_proj_fp8: QuantSidecar(None),
            },
            shared_expert_gate: lazy_linear(self.hidden_size, 1, device),
            num_experts_per_tok: (self.num_experts_per_tok),
            norm_topk_prob: (self.norm_topk_prob),
        }
    }

    fn linear_qkv_dim(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    fn linear_v_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    fn full_q_proj_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim * 2
    }

    fn full_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    fn full_o_in_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    pub fn expected_weight_shapes(&self) -> BTreeMap<String, Vec<usize>> {
        let mut out = BTreeMap::new();
        out.insert(
            "lm_head.weight".to_string(),
            vec![self.vocab_size, self.hidden_size],
        );
        out.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            vec![self.vocab_size, self.hidden_size],
        );
        for layer in 0..self.num_hidden_layers {
            let prefix = format!("model.language_model.layers.{layer}");
            self.insert_layer_shapes(&mut out, &prefix, self.layer_types[layer]);
        }
        out.insert(
            "model.language_model.norm.weight".to_string(),
            vec![self.hidden_size],
        );
        out.insert(
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            vec![self.hidden_size],
        );
        out.insert(
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            vec![self.hidden_size],
        );
        out.insert(
            "mtp.fc.weight".to_string(),
            vec![self.hidden_size, self.hidden_size * 2],
        );
        for layer in 0..self.mtp_num_hidden_layers {
            let prefix = format!("mtp.layers.{layer}");
            self.insert_layer_shapes(&mut out, &prefix, Qwen3_5LayerType::FullAttention);
        }
        out.insert("mtp.norm.weight".to_string(), vec![self.hidden_size]);
        out
    }

    fn insert_layer_shapes(
        &self,
        out: &mut BTreeMap<String, Vec<usize>>,
        prefix: &str,
        kind: Qwen3_5LayerType,
    ) {
        out.insert(
            format!("{prefix}.input_layernorm.weight"),
            vec![self.hidden_size],
        );
        match kind {
            Qwen3_5LayerType::LinearAttention => {
                let qkv = self.linear_qkv_dim();
                let v = self.linear_v_dim();
                out.insert(
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    vec![qkv, self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.linear_attn.in_proj_a.weight"),
                    vec![self.linear_num_value_heads, self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.linear_attn.in_proj_b.weight"),
                    vec![self.linear_num_value_heads, self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    vec![v, self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.linear_attn.A_log"),
                    vec![self.linear_num_value_heads],
                );
                out.insert(
                    format!("{prefix}.linear_attn.dt_bias"),
                    vec![self.linear_num_value_heads],
                );
                out.insert(
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    vec![qkv, 1, self.linear_conv_kernel_dim],
                );
                out.insert(
                    format!("{prefix}.linear_attn.norm.weight"),
                    vec![self.linear_value_head_dim],
                );
                out.insert(
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    vec![self.hidden_size, v],
                );
            }
            Qwen3_5LayerType::FullAttention => {
                out.insert(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    vec![self.full_q_proj_dim(), self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    vec![self.full_kv_dim(), self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![self.full_kv_dim(), self.hidden_size],
                );
                out.insert(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    vec![self.hidden_size, self.full_o_in_dim()],
                );
                out.insert(
                    format!("{prefix}.self_attn.q_norm.weight"),
                    vec![self.head_dim],
                );
                out.insert(
                    format!("{prefix}.self_attn.k_norm.weight"),
                    vec![self.head_dim],
                );
            }
        }
        out.insert(
            format!("{prefix}.post_attention_layernorm.weight"),
            vec![self.hidden_size],
        );
        self.insert_mlp_shapes(out, &format!("{prefix}.mlp"));
    }

    fn insert_mlp_shapes(&self, out: &mut BTreeMap<String, Vec<usize>>, prefix: &str) {
        out.insert(
            format!("{prefix}.gate.weight"),
            vec![self.num_experts, self.hidden_size],
        );
        out.insert(
            format!("{prefix}.experts.gate_up_proj"),
            vec![
                self.num_experts,
                self.moe_intermediate_size * 2,
                self.hidden_size,
            ],
        );
        out.insert(
            format!("{prefix}.experts.down_proj"),
            vec![
                self.num_experts,
                self.hidden_size,
                self.moe_intermediate_size,
            ],
        );
        out.insert(
            format!("{prefix}.shared_expert.gate_proj.weight"),
            vec![self.shared_expert_intermediate_size, self.hidden_size],
        );
        out.insert(
            format!("{prefix}.shared_expert.up_proj.weight"),
            vec![self.shared_expert_intermediate_size, self.hidden_size],
        );
        out.insert(
            format!("{prefix}.shared_expert.down_proj.weight"),
            vec![self.hidden_size, self.shared_expert_intermediate_size],
        );
        out.insert(
            format!("{prefix}.shared_expert_gate.weight"),
            vec![1, self.hidden_size],
        );
    }
}

#[derive(Module, Debug)]
pub struct Qwen3_5MoeForCausalLM {
    pub model: Qwen3_5Model,
    pub lm_head: Linear,
    pub lm_head_quant: QuantSidecar,
    pub mtp: Qwen3_5MtpBlock,
}

#[derive(Module, Debug)]
pub struct Qwen3_5Model {
    #[module(skip)]
    pub config: Qwen3_5MoeConfig,
    pub embed_tokens: Embedding,
    pub layers: Vec<Qwen3_5DecoderLayer>,
    pub norm: RmsNorm,
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
pub enum Qwen3_5DecoderLayer {
    Linear(Qwen3_5GdnLayer),
    Full(Qwen3_5FullAttnLayer),
}

#[derive(Module, Debug)]
pub struct Qwen3_5GdnLayer {
    pub input_layernorm: RmsNorm,
    pub linear_attn: Qwen3_5GdnAttention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Qwen3_5SharedMoeBlock,
}

#[derive(Module, Debug)]
#[allow(non_snake_case)]
pub struct Qwen3_5GdnAttention {
    pub in_proj_qkv: Linear,
    pub in_proj_qkv_fp8: QuantSidecar,
    pub in_proj_a: Linear,
    pub in_proj_a_fp8: QuantSidecar,
    pub in_proj_b: Linear,
    pub in_proj_b_fp8: QuantSidecar,
    pub in_proj_z: Linear,
    pub in_proj_z_fp8: QuantSidecar,
    pub A_log: Param<Tensor<1>>,
    pub dt_bias: Param<Tensor<1>>,
    pub conv1d: Qwen3_5Conv1d,
    pub norm: RmsNorm,
    pub out_proj: Linear,
    pub out_proj_fp8: QuantSidecar,
}

#[derive(Module, Debug)]
pub struct Qwen3_5Conv1d {
    pub weight: Param<Tensor<3>>,
}

#[derive(Module, Debug)]
pub struct Qwen3_5FullAttnLayer {
    pub input_layernorm: RmsNorm,
    pub self_attn: Qwen3_5FullAttention,
    pub post_attention_layernorm: RmsNorm,
    pub mlp: Qwen3_5SharedMoeBlock,
}

#[derive(Module, Debug)]
pub struct Qwen3_5FullAttention {
    pub q_proj: Linear,
    pub q_proj_fp8: QuantSidecar,
    pub k_proj: Linear,
    pub k_proj_fp8: QuantSidecar,
    pub v_proj: Linear,
    pub v_proj_fp8: QuantSidecar,
    pub o_proj: Linear,
    pub o_proj_fp8: QuantSidecar,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

#[derive(Module, Debug)]
pub struct Qwen3_5SharedMoeBlock {
    pub gate: Linear,
    pub experts: Qwen3_5FusedExperts,
    pub shared_expert: Qwen3_5SharedExpert,
    pub shared_expert_gate: Linear,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
}

#[derive(Module, Debug)]
pub struct Qwen3_5FusedExperts {
    pub gate_up_proj: Param<Tensor<3>>,
    pub down_proj: Param<Tensor<3>>,
    pub fp8: ExpertQuantSidecar,
    pub nvfp4: ExpertNvfp4Sidecar,
}

#[derive(Module, Debug)]
pub struct Qwen3_5SharedExpert {
    pub gate_proj: Linear,
    pub gate_proj_fp8: QuantSidecar,
    pub up_proj: Linear,
    pub up_proj_fp8: QuantSidecar,
    pub down_proj: Linear,
    pub down_proj_fp8: QuantSidecar,
}

#[derive(Module, Debug)]
pub struct Qwen3_5MtpBlock {
    pub pre_fc_norm_embedding: RmsNorm,
    pub pre_fc_norm_hidden: RmsNorm,
    pub fc: Linear,
    pub fc_fp8: QuantSidecar,
    pub layers: Vec<Qwen3_5FullAttnLayer>,
    pub norm: RmsNorm,
}

impl Qwen3_5MoeForCausalLM {
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
    ) -> Tensor<3> {
        self.forward_prec(input_ids, position_ids, cache, Precision::F32)
    }

    pub fn forward_prec(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
    ) -> Tensor<3> {
        self.forward_hidden_prec(input_ids, position_ids, cache, prec)
            .1
    }

    pub fn forward_hidden_prec(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
    ) -> (Tensor<3>, Tensor<3>) {
        let hidden_states = self
            .model
            .forward_prec(input_ids, position_ids, cache, prec);
        // Route the head through the quant sidecar when present (NVFP4/fp8), else the bf16 Linear.
        // NOTE: the quantized head has a fixed comptime `m_max` (the NVFP4 head loads at m_max=1 for
        // decode perf), so full-T (S>1) logits through the quant head are only valid when S<=m_max.
        // The greedy driver funnels the LAST position (M=1) via `forward_last_logits`; a caller that
        // genuinely needs full-T logits through the quant head must chunk S by the head's m_max.
        let logits = ql3(
            &self.lm_head_quant,
            &self.lm_head,
            hidden_states.clone(),
            prec,
        );
        (hidden_states, logits)
    }

    /// Next-token logits from the LAST position of a (possibly multi-token) forward pass: `[B, vocab]`.
    ///
    /// Greedy generation only needs the final position's distribution. The quantized lm_head kernels
    /// cap runtime M at their comptime `m_max` (the NVFP4 head loads at m_max=1 for decode perf), so
    /// this slices the last hidden position BEFORE the head, keeping the quant head at M=1 on prefill
    /// (T>1) exactly as on decode (T=1). Numerically identical to slicing the argmax off full-T logits
    /// (the head is applied per position), but valid for the raw-NVFP4 head where full-T would exceed
    /// m_max. The bf16/fp8 paths are unchanged (still the same per-position linear).
    pub fn forward_last_logits(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
    ) -> Tensor<2> {
        let hidden_states = self
            .model
            .forward_prec(input_ids, position_ids, cache, prec);
        let [batch, seq_len, hidden] = hidden_states.dims();
        let last = hidden_states.slice([0..batch, (seq_len - 1)..seq_len, 0..hidden]);
        let logits = ql3(&self.lm_head_quant, &self.lm_head, last, prec);
        let vocab = logits.dims()[2];
        logits.reshape([batch, vocab])
    }

    /// `docs/MEMORY_STREAMING_PLAN.md` streamed sibling of [`Self::forward_last_logits`]: routed
    /// experts are fetched on demand through `pool` (an [`ExpertSlotPool`] opened on the same
    /// checkpoint dir) instead of requiring `self.model.layers[*].mlp.experts.*` to be fully resident.
    /// Pair with `load_weights_sharded_resident_core` at load time.
    pub fn forward_last_logits_streamed(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
        pool: &mut ExpertSlotPool,
    ) -> Tensor<2> {
        let hidden_states =
            self.model
                .forward_prec_streamed(input_ids, position_ids, cache, prec, pool);
        let [batch, seq_len, hidden] = hidden_states.dims();
        let last = hidden_states.slice([0..batch, (seq_len - 1)..seq_len, 0..hidden]);
        let prof_device = last.device();
        let logits = timed(&profile::LM_HEAD, &prof_device, || {
            ql3(&self.lm_head_quant, &self.lm_head, last, prec)
        });
        let vocab = logits.dims()[2];
        logits.reshape([batch, vocab])
    }

    /// CUDA-graph-capturable single-token decode step (spec §4): one `[B,1]` token in, `[B, vocab]`
    /// f32 logits out. Drives [`Qwen3_5Model::forward_decode_static_pre`] then the `lm_head` (linear3,
    /// F32). All static-path constraints (no D2H/H2D staging, no env reads, no `set_state`, device
    /// `pos`) hold; call [`Self::preflight_static`] + [`Self::init_static_caches`] before capture.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_static_pre(
        &self,
        input_ids: Tensor<2, Int>,
        pos: Tensor<1, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
    ) -> Tensor<2> {
        let hidden_states =
            self.model
                .forward_decode_static_pre(input_ids, pos, cache, prec, freqs, arange_tmax);
        let [batch, seq_len, _] = hidden_states.dims();
        debug_assert_eq!(
            seq_len, 1,
            "forward_decode_static_pre expects a single decode token"
        );
        // Static/captured decode: S=1 -> M=1 through the head, within the quant head's m_max=1. The
        // quant sidecar (NVFP4/fp8) GEMV stages no D2H/H2D and reads no env, so it is capture-stable
        // exactly like the dense fp8 projections already used on this path.
        let logits = ql3(&self.lm_head_quant, &self.lm_head, hidden_states, prec);
        let vocab = logits.dims()[2];
        logits.reshape([batch, vocab]).cast(DType::F32)
    }

    /// Model-level preflight aggregating every layer's MoE preconditions and asserting every GDN state
    /// cache is `init_static`'d and every full-attn KV cache is static-capacity. See
    /// [`Qwen3_5Model::preflight_static`]. `tokens` is the per-step token count (1 at B=1 decode).
    pub fn preflight_static(
        &self,
        cache: &Qwen3_5HybridCache,
        tokens: usize,
    ) -> Result<(), String> {
        self.model.preflight_static(cache, tokens)
    }

    /// Allocate the capture-stable GDN state buffers for every linear-attention layer in `cache`. See
    /// [`Qwen3_5Model::init_static_caches`].
    pub fn init_static_caches(&self, cache: &mut Qwen3_5HybridCache, batch: usize) {
        self.model.init_static_caches(cache, batch);
    }

    pub fn mtp_new_cache(&self, t_max: usize) -> KVCache {
        KVCache::with_capacity(t_max)
    }
}

impl Qwen3_5MtpBlock {
    pub fn mtp_new_cache(t_max: usize) -> KVCache {
        KVCache::with_capacity(t_max)
    }

    pub fn forward_draft(
        &self,
        tok_next: Tensor<2, Int>,
        hidden: Tensor<3>,
        position_ids: Tensor<2, Int>,
        mtp_cache: &mut KVCache,
        embed: &Embedding,
        lm_head: &Linear,
        prec: Precision,
    ) -> (Tensor<2>, Tensor<3>) {
        assert!(
            !self.layers.is_empty(),
            "Qwen3_5MtpBlock::forward_draft requires at least one MTP layer"
        );
        let [batch, seq_len] = tok_next.dims();
        debug_assert_eq!(
            seq_len, 1,
            "Qwen3_5MtpBlock::forward_draft expects tok_next [B,1]"
        );
        debug_assert_eq!(
            hidden.dims()[1],
            1,
            "Qwen3_5MtpBlock::forward_draft expects hidden [B,1,H]"
        );

        let e = embed.forward(tok_next).cast(DType::F32);
        let e_n = self.pre_fc_norm_embedding.forward(e);
        let h_n = self.pre_fc_norm_hidden.forward(hidden);
        let x = Tensor::cat(vec![e_n, h_n], 2);
        let x = ql3(&self.fc_fp8, &self.fc, x, prec);
        let x = self.layers[0].forward_decoder_with_cache(x, position_ids, mtp_cache, prec);
        let mtp_hidden_out = self.norm.forward(x);
        let logits = linear3(lm_head, mtp_hidden_out.clone(), prec);
        let vocab = logits.dims()[2];
        (
            logits.reshape([batch, vocab]).cast(DType::F32),
            mtp_hidden_out,
        )
    }
}

impl Qwen3_5SharedMoeBlock {
    pub fn forward(&self, hidden_states: Tensor<3>, prec: Precision) -> Tensor<3> {
        self.forward_impl(hidden_states, prec, true)
    }

    #[cfg(feature = "cubecl-gpu")]
    fn combine_token_major_assignments(
        y: Tensor<2>,
        tokens: usize,
        top_k: usize,
        hidden: usize,
    ) -> Tensor<2> {
        y.reshape([tokens, top_k, hidden])
            .sum_dim(1)
            .reshape([tokens, hidden])
    }

    fn forward_impl(
        &self,
        hidden_states: Tensor<3>,
        prec: Precision,
        fused_experts: bool,
    ) -> Tensor<3> {
        let [batch, seq_len, hidden] = hidden_states.dims();
        let tokens = batch * seq_len;
        let num_experts = self.gate.weight.val().dims()[1];
        let top_k = (self.num_experts_per_tok).min(num_experts);
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();
        #[cfg(not(feature = "cubecl-gpu"))]
        let _ = fused_experts;

        let (sel_idx, sel_w) = self.route_topk(hidden_states.clone(), top_k);
        #[cfg(feature = "cubecl-gpu")]
        if fused_experts && qwen35_fused_moe_enabled() && tokens <= QWEN35_FUSED_MOE_MAX_T {
            // Dispatch precedence: NVFP4 experts (the official nvidia checkpoint) take priority over
            // the fp8 sidecar, which takes priority over the bf16 stacks. Mirrors the fp8 arm exactly
            // (deterministic reshape+sum combine, hard-panic diagnostics, no silent fallback).
            if let Some(nvfp4) = &self.experts.nvfp4.0 {
                assert_eq!(
                    nvfp4.e, num_experts,
                    "Qwen3.5 fused nvfp4 MoE sidecar expert count {} != gate experts {num_experts}",
                    nvfp4.e
                );
                assert_eq!(
                    nvfp4.h, hidden,
                    "Qwen3.5 fused nvfp4 MoE sidecar hidden {} != activation hidden {hidden}",
                    nvfp4.h
                );
                assert_eq!(
                    nvfp4.qw_gu.dims(),
                    [num_experts, hidden, nvfp4.i],
                    "nvfp4 qw_gu must be [E,H,I]"
                );
                assert_eq!(
                    nvfp4.bs_gu.dims(),
                    [num_experts, nvfp4.i * 2, hidden / 16],
                    "nvfp4 bs_gu must be [E,2I,H/16]"
                );
                assert_eq!(
                    nvfp4.gscale_gu.dims(),
                    [num_experts, 2],
                    "nvfp4 gscale_gu must be [E,2]"
                );
                assert_eq!(
                    nvfp4.qw_dn.dims(),
                    [num_experts, nvfp4.i, hidden / 2],
                    "nvfp4 qw_dn must be [E,I,H/2]"
                );
                assert_eq!(
                    nvfp4.bs_dn.dims(),
                    [num_experts, hidden, nvfp4.i / 16],
                    "nvfp4 bs_dn must be [E,H,I/16]"
                );
                assert_eq!(
                    nvfp4.gscale_dn.dims(),
                    [num_experts],
                    "nvfp4 gscale_dn must be [E]"
                );
                assert_eq!(
                    nvfp4.qw_gu.dtype(),
                    DType::I8,
                    "nvfp4 qw_gu must be DType::I8"
                );
                assert_eq!(
                    nvfp4.qw_dn.dtype(),
                    DType::I8,
                    "nvfp4 qw_dn must be DType::I8"
                );
                assert_eq!(
                    nvfp4.bs_gu.dtype(),
                    DType::I8,
                    "nvfp4 bs_gu must be DType::I8"
                );
                assert_eq!(
                    nvfp4.bs_dn.dtype(),
                    DType::I8,
                    "nvfp4 bs_dn must be DType::I8"
                );

                let n = tokens * top_k;
                let assign_e = sel_idx.reshape([n]);
                let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
                let x2 = hidden_states
                    .clone()
                    .reshape([tokens, hidden])
                    .cast(DType::F32);
                let y = crate::nvfp4::fused_moe_gu2_down_nvfp4(
                    x2,
                    nvfp4.qw_gu.clone(),
                    nvfp4.bs_gu.clone(),
                    nvfp4.gscale_gu.clone(),
                    nvfp4.qw_dn.clone(),
                    nvfp4.bs_dn.clone(),
                    nvfp4.gscale_dn.clone(),
                    assign_e,
                    sel_w_flat,
                    hidden,
                    nvfp4.i,
                    n,
                )
                .cast(dtype);
                let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
                let shared = self
                    .shared_expert_forward(hidden_states, prec)
                    .reshape([tokens, hidden]);
                return (routed + shared)
                    .reshape([batch, seq_len, hidden])
                    .cast(dtype);
            }
            #[cfg(feature = "cuda")]
            if let Some(fp8) = &self.experts.fp8.0 {
                assert_eq!(
                    fp8.e, num_experts,
                    "Qwen3.5 fused fp8 MoE sidecar expert count {} != gate experts {num_experts}",
                    fp8.e
                );
                assert_eq!(
                    fp8.h, hidden,
                    "Qwen3.5 fused fp8 MoE sidecar hidden {} != activation hidden {hidden}",
                    fp8.h
                );
                assert_eq!(
                    fp8.q_gu.dims(),
                    [num_experts, hidden, fp8.i * 2],
                    "fp8 q_gu must be [E,H,2I]"
                );
                assert_eq!(
                    fp8.s_gu.dims(),
                    [num_experts, fp8.i * 2],
                    "fp8 s_gu must be [E,2I]"
                );
                assert_eq!(
                    fp8.q_dn.dims(),
                    [num_experts, fp8.i, hidden],
                    "fp8 q_dn must be [E,I,H]"
                );
                assert_eq!(
                    fp8.s_dn.dims(),
                    [num_experts, hidden],
                    "fp8 s_dn must be [E,H]"
                );
                assert_eq!(fp8.q_gu.dtype(), DType::I8, "fp8 q_gu must be DType::I8");
                assert_eq!(fp8.q_dn.dtype(), DType::I8, "fp8 q_dn must be DType::I8");
                assert_eq!(fp8.s_gu.dtype(), DType::F32, "fp8 s_gu must be DType::F32");
                assert_eq!(fp8.s_dn.dtype(), DType::F32, "fp8 s_dn must be DType::F32");

                let n = tokens * top_k;
                let assign_e = sel_idx.reshape([n]);
                let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
                let x2 = hidden_states
                    .clone()
                    .reshape([tokens, hidden])
                    .cast(DType::F32);
                let y = crate::moe_grouped::fused_moe_gu2_down_fp8(
                    x2,
                    fp8.q_gu.clone(),
                    fp8.s_gu.clone(),
                    fp8.q_dn.clone(),
                    fp8.s_dn.clone(),
                    assign_e,
                    sel_w_flat,
                    hidden,
                    fp8.i,
                    n,
                )
                .cast(dtype);
                let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
                let shared = self
                    .shared_expert_forward(hidden_states, prec)
                    .reshape([tokens, hidden]);
                return (routed + shared)
                    .reshape([batch, seq_len, hidden])
                    .cast(dtype);
            }

            #[cfg(feature = "cuda")]
            {
                let gate_up_dims = self.experts.gate_up_proj.val().dims();
                let down_dims = self.experts.down_proj.val().dims();
                let placeholder = gate_up_dims == [1, 1, 1] || down_dims == [1, 1, 1];
                if !placeholder && down_dims[0] == num_experts && down_dims[1] == hidden {
                    let inner = down_dims[2];
                    assert_eq!(
                        gate_up_dims,
                        [num_experts, inner * 2, hidden],
                        "Qwen3.5 fused bf16 MoE requires non-placeholder gate_up [E,2I,H]"
                    );
                    assert_eq!(
                        down_dims,
                        [num_experts, hidden, inner],
                        "Qwen3.5 fused bf16 MoE requires non-placeholder down [E,H,I]"
                    );
                    let n = tokens * top_k;
                    let assign_e = sel_idx.reshape([n]);
                    let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
                    let x2 = hidden_states
                        .clone()
                        .reshape([tokens, hidden])
                        .cast(DType::F32);
                    let y = crate::moe_grouped::fused_moe_gu2_down_bf16(
                        x2,
                        self.experts.gate_up_proj.val(),
                        self.experts.down_proj.val(),
                        assign_e,
                        sel_w_flat,
                        hidden,
                        inner,
                        n,
                    )
                    .cast(dtype);
                    let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
                    let shared = self
                        .shared_expert_forward(hidden_states, prec)
                        .reshape([tokens, hidden]);
                    return (routed + shared)
                        .reshape([batch, seq_len, hidden])
                        .cast(dtype);
                }
            }
        }

        let idx_host: Vec<i64> = sel_idx
            .cast(DType::I64)
            .into_data()
            .to_vec()
            .expect("read Qwen3.5 MoE route ids");
        let w_host: Vec<f32> = sel_w
            .into_data()
            .to_vec()
            .expect("read Qwen3.5 MoE route weights");
        let mut by_expert: Vec<(Vec<i64>, Vec<f32>)> = vec![(Vec::new(), Vec::new()); num_experts];
        for tok in 0..tokens {
            for slot in 0..top_k {
                let expert = idx_host[tok * top_k + slot] as usize;
                by_expert[expert].0.push(tok as i64);
                by_expert[expert].1.push(w_host[tok * top_k + slot]);
            }
        }

        let x2 = hidden_states.clone().reshape([tokens, hidden]);
        // Activations run in F32 (linear3/matmul_out_in always output F32 — the repo's "F32 activations,
        // bf16 matmul-compute" convention). expert_forward + shared_expert_forward both return F32, so the
        // whole combine stays F32; cast the RESULT back to the residual-stream dtype at the end so the
        // decoder `residual + moe_out` matches (bf16 on the real model). Do NOT cast routed/w to `dtype`
        // (bf16) here — that mismatches the F32 expert outputs (the `F32 * bf16` panic).
        let mut routed = Tensor::<2>::zeros([tokens, hidden], &device);
        for expert in 0..num_experts {
            let (tok_ids, weights) = &by_expert[expert];
            if tok_ids.is_empty() {
                continue;
            }
            let n = tok_ids.len();
            let tok_idx = Tensor::<1, Int>::from_data(tok_ids.as_slice(), &device);
            let w = Tensor::<1>::from_data(weights.as_slice(), &device).reshape([n, 1]);
            let x_e = x2
                .clone()
                .select(0, tok_idx.clone())
                .reshape([n, 1, hidden]);
            let y_e = self.expert_forward(expert, x_e, prec).reshape([n, hidden]);
            routed = routed.select_assign(0, tok_idx, y_e * w, IndexingUpdateOp::Add);
        }

        let shared = self
            .shared_expert_forward(hidden_states, prec)
            .reshape([tokens, hidden]);
        (routed + shared)
            .reshape([batch, seq_len, hidden])
            .cast(dtype)
    }

    /// `docs/MEMORY_STREAMING_PLAN.md` streamed sibling of the bf16 host-loop branch of
    /// [`Self::forward_impl`] (the default, non-`cuda`-fused path): identical routing/combine math,
    /// but each selected expert's weights come from `pool` (on-demand `read_expert_slice` + LRU
    /// cache) instead of a fully-resident `self.experts.gate_up_proj`/`down_proj`. Requires the model
    /// to have been loaded via `load_weights_sharded_resident_core` (so `self.experts.*` stays at its
    /// tiny placeholder and is never read here) plus a `pool` opened on the same checkpoint dir.
    /// `layer_idx` must match this block's position in `Qwen3_5Model::layers` (the same index used to
    /// build the checkpoint's `model.language_model.layers.{layer_idx}.mlp.experts.*` keys).
    pub fn forward_streamed(
        &self,
        hidden_states: Tensor<3>,
        prec: Precision,
        pool: &mut ExpertSlotPool,
        layer_idx: usize,
    ) -> Tensor<3> {
        let [batch, seq_len, hidden] = hidden_states.dims();
        let tokens = batch * seq_len;
        let num_experts = self.gate.weight.val().dims()[1];
        let top_k = (self.num_experts_per_tok).min(num_experts);
        let device = hidden_states.device();
        let dtype = hidden_states.dtype();

        let (sel_idx, sel_w) = timed(&profile::ROUTER, &device, || {
            self.route_topk(hidden_states.clone(), top_k)
        });
        let idx_host: Vec<i64> = timed(&profile::ROUTER, &device, || {
            if profile::cpu_time_enabled() {
                let _tpre = std::time::Instant::now();
                let _ = device.sync();
                let drain_ns = _tpre.elapsed().as_nanos() as u64;
                let _trb = std::time::Instant::now();
                let v = sel_idx
                    .cast(DType::I64)
                    .into_data()
                    .to_vec()
                    .expect("read Qwen3.5 MoE route ids (streamed)");
                profile::add(&profile::CPU_ROUTE_RB, _trb.elapsed().as_nanos() as u64);
                profile::add(&profile::CPU_ROUTE_DRAIN, drain_ns);
                v
            } else {
                sel_idx
                    .cast(DType::I64)
                    .into_data()
                    .to_vec()
                    .expect("read Qwen3.5 MoE route ids (streamed)")
            }
        });
        // `sel_idx` must come to the host -- the CPU issues the disk reads, so it has to know which
        // experts were routed. `sel_w` must NOT: it was previously downloaded only to be sliced per
        // expert and re-uploaded, costing a second full device->host round trip per layer (~680 forced
        // GPU syncs per 16-token run). Instead keep it resident and gather from it on-device with the
        // flat `tok * top_k + slot` offsets, which we already know host-side from `idx_host`.
        let sel_w_flat = sel_w.clone().reshape([tokens * top_k]);
        let mut by_expert: Vec<(Vec<i64>, Vec<i64>)> = vec![(Vec::new(), Vec::new()); num_experts];
        for tok in 0..tokens {
            for slot in 0..top_k {
                let expert = idx_host[tok * top_k + slot] as usize;
                by_expert[expert].0.push(tok as i64);
                by_expert[expert].1.push((tok * top_k + slot) as i64);
            }
        }

        // Batch all of this layer's distinct routed experts into two reader calls (one for
        // gate_up, one for down) instead of one `read_expert_slice` call per expert per
        // projection -- see `ExpertSlotPool::prefetch_layer` for why (removes redundant
        // shard/header lookups, ~640 tiny fetches/token -> ~80).
        let distinct_experts: Vec<usize> = (0..num_experts)
            .filter(|&e| !by_expert[e].0.is_empty())
            .collect();
        // Hit-first execution (ported from turbo-fieldfare's DEC-18, ~14% win there): classify hits
        // vs misses and, for the offline blob store, start reading misses on a background thread
        // pool without blocking. Cache hits dispatch on the GPU immediately below while those reads
        // are still in flight, instead of the whole layer waiting on I/O before any GEMV runs.
        let hits: Vec<usize> = timed(&profile::MOE_PREFETCH, &device, || {
            pool.prefetch_layer_begin(layer_idx, &distinct_experts, &device)
                .expect("streamed prefetch_layer_begin")
        });

        let x2 = hidden_states.clone().reshape([tokens, hidden]);
        let mut routed = Tensor::<2>::zeros([tokens, hidden], &device);
        if tokens == 1 {
            // OFF BY DEFAULT (opt in with QWEN35_STREAM_2KERNEL=1): despite being the "fused fast
            // path", this measures SLOWER than the sequential fallback below (1.41 vs 1.78 tok/s on
            // M2 Pro / Qwen3.6-35B). Two reasons, both structural:
            //   1. It `Tensor::cat`s the top_k experts' packed weights into a fresh [k,N,K/2] stack
            //      on every layer of every token -- ~13.5 MB per layer per token, ~11 GB of pure
            //      device memcpy plus ~1.7k transient allocations over a 16-token run. The kernel
            //      already accepts an `assign_e` index tensor, so this copy only exists because the
            //      pool stores each slot as an independent tensor rather than as slices of one
            //      contiguous slab.
            //   2. It must `prefetch_layer_finish` up front (all experts resident before the single
            //      fused launch), which forfeits the hit-first I/O overlap the fallback gets.
            // Making this genuinely fast needs the pool to own a pre-stacked slab; until then the
            // fallback wins.
            #[cfg(feature = "cubecl-gpu")]
            if crate::expert_stream::stream_nvfp4()
                && std::env::var("QWEN35_STREAM_2KERNEL").ok().as_deref() == Some("1")
            {
                // Fused 2-Kernel grouped MoE path (matching turbo-fieldfare):
                // Finish background prefetch so all 8 routed experts are resident in the pool.
                timed(&profile::MOE_PREFETCH, &device, || {
                    pool.prefetch_layer_finish(&device)
                        .expect("streamed prefetch_layer_finish")
                });

                let mut q_gu_vec = Vec::with_capacity(top_k);
                let mut bs_gu_vec = Vec::with_capacity(top_k);
                let mut gs_gu_f32 = Vec::with_capacity(top_k * 2);
                let mut q_dn_vec = Vec::with_capacity(top_k);
                let mut bs_dn_vec = Vec::with_capacity(top_k);
                let mut gs_dn_f32 = Vec::with_capacity(top_k);

                for slot in 0..top_k {
                    let expert = idx_host[slot] as usize;
                    let gu_lin = pool
                        .gate_up_nvfp4(layer_idx, expert, &device)
                        .expect("fetch gate_up_nvfp4 for 2-kernel MoE");
                    let dn_lin = pool
                        .down_nvfp4(layer_idx, expert, &device)
                        .expect("fetch down_nvfp4 for 2-kernel MoE");

                    q_gu_vec.push(gu_lin.qw().clone().unsqueeze::<3>());
                    bs_gu_vec.push(gu_lin.bs().clone().unsqueeze::<3>());
                    let gs_val = gu_lin.gscale_f32();
                    gs_gu_f32.push(gs_val);
                    gs_gu_f32.push(gs_val); // gate + up share the same gscale

                    q_dn_vec.push(dn_lin.qw().clone().unsqueeze::<3>());
                    bs_dn_vec.push(dn_lin.bs().clone().unsqueeze::<3>());
                    let gs_dn_val = dn_lin.gscale_f32();
                    gs_dn_f32.push(gs_dn_val);
                }

                let q_gu_3d = Tensor::cat(q_gu_vec, 0);
                let bs_gu_3d = Tensor::cat(bs_gu_vec, 0);
                let gscale_gu_2d =
                    Tensor::<2>::from_data(TensorData::new(gs_gu_f32, [top_k, 2]), &device);
                let q_dn_3d = Tensor::cat(q_dn_vec, 0);
                let bs_dn_3d = Tensor::cat(bs_dn_vec, 0);
                let gscale_dn_1d =
                    Tensor::<1>::from_data(TensorData::new(gs_dn_f32, [top_k]), &device);
                let assign_e = Tensor::<1, Int>::from_data(
                    TensorData::new((0..top_k as i64).collect::<Vec<_>>(), [top_k]),
                    &device,
                );
                let sel_w_1d = sel_w.clone().reshape([top_k]);

                let inner = q_gu_3d.dims()[1] / 2; // N of gate_up is 2 * inner
                let out = timed(&profile::MOE_EXPERTS, &device, || {
                    crate::nvfp4_kernels::fused_moe_2kernel_nvfp4(
                        x2.clone().cast(DType::F32),
                        q_gu_3d,
                        bs_gu_3d,
                        gscale_gu_2d,
                        q_dn_3d,
                        bs_dn_3d,
                        gscale_dn_1d,
                        assign_e,
                        sel_w_1d,
                        hidden,
                        inner,
                        top_k,
                        1,
                    )
                });
                routed = out;
            } else {
                // Fallback sequential path
                let x2_f32 = x2;
                // Dispatch hits first so GEMVs overlap miss I/O, but fold in expert-id
                // order. Hit/miss order is cache-dependent; adding in that order made
                // greedy decode non-deterministic across runs.
                let mut contribs: Vec<Option<Tensor<2>>> = vec![None; num_experts];
                for &expert in &hits {
                    let slot = by_expert[expert].1[0] as usize;
                    let w = sel_w.clone().slice([0..1, slot..(slot + 1)]);
                    let y_e = timed(&profile::MOE_EXPERTS, &device, || {
                        pool.expert_forward(layer_idx, expert, x2_f32.clone(), prec, &device)
                            .expect("streamed expert_forward")
                    });
                    contribs[expert] = Some(y_e * w);
                }
                timed(&profile::MOE_PREFETCH, &device, || {
                    pool.prefetch_layer_finish(&device)
                        .expect("streamed prefetch_layer_finish")
                });
                for &expert in &distinct_experts {
                    if contribs[expert].is_none() {
                        let slot = by_expert[expert].1[0] as usize;
                        let w = sel_w.clone().slice([0..1, slot..(slot + 1)]);
                        let y_e = timed(&profile::MOE_EXPERTS, &device, || {
                            pool.expert_forward(layer_idx, expert, x2_f32.clone(), prec, &device)
                                .expect("streamed expert_forward")
                        });
                        contribs[expert] = Some(y_e * w);
                    }
                }
                for &expert in &distinct_experts {
                    let contrib = contribs[expert].take().expect("routed expert contrib");
                    routed = timed(&profile::MOE_SCATTER, &device, || routed + contrib);
                }
            }
            #[cfg(not(feature = "cubecl-gpu"))]
            {
                let x2_f32 = x2;
                // Dispatch hits first so GEMVs overlap miss I/O, but fold in expert-id
                // order. Hit/miss order is cache-dependent; adding in that order made
                // greedy decode non-deterministic across runs.
                let mut contribs: Vec<Option<Tensor<2>>> = vec![None; num_experts];
                for &expert in &hits {
                    let slot = by_expert[expert].1[0] as usize;
                    let w = sel_w.clone().slice([0..1, slot..(slot + 1)]);
                    let y_e = timed(&profile::MOE_EXPERTS, &device, || {
                        pool.expert_forward(layer_idx, expert, x2_f32.clone(), prec, &device)
                            .expect("streamed expert_forward")
                    });
                    contribs[expert] = Some(y_e * w);
                }
                timed(&profile::MOE_PREFETCH, &device, || {
                    pool.prefetch_layer_finish(&device)
                        .expect("streamed prefetch_layer_finish")
                });
                for &expert in &distinct_experts {
                    if contribs[expert].is_none() {
                        let slot = by_expert[expert].1[0] as usize;
                        let w = sel_w.clone().slice([0..1, slot..(slot + 1)]);
                        let y_e = timed(&profile::MOE_EXPERTS, &device, || {
                            pool.expert_forward(layer_idx, expert, x2_f32.clone(), prec, &device)
                                .expect("streamed expert_forward")
                        });
                        contribs[expert] = Some(y_e * w);
                    }
                }
                for &expert in &distinct_experts {
                    let contrib = contribs[expert].take().expect("routed expert contrib");
                    routed = timed(&profile::MOE_SCATTER, &device, || routed + contrib);
                }
            }
        } else {
            let run_expert =
                |routed: Tensor<2>, expert: usize, pool: &mut ExpertSlotPool<'_>| -> Tensor<2> {
                    let (tok_ids, weight_offsets) = &by_expert[expert];
                    if tok_ids.is_empty() {
                        return routed;
                    }
                    let n = tok_ids.len();
                    let tok_idx = Tensor::<1, Int>::from_data(tok_ids.as_slice(), &device);
                    let w = sel_w_flat
                        .clone()
                        .select(
                            0,
                            Tensor::<1, Int>::from_data(weight_offsets.as_slice(), &device),
                        )
                        .reshape([n, 1]);
                    let x_e = x2.clone().select(0, tok_idx.clone());
                    let y_e = timed(&profile::MOE_EXPERTS, &device, || {
                        pool.expert_forward(layer_idx, expert, x_e, prec, &device)
                            .expect("streamed expert_forward")
                    });
                    timed(&profile::MOE_SCATTER, &device, || {
                        routed.select_assign(0, tok_idx, y_e * w, IndexingUpdateOp::Add)
                    })
                };
            // Hits first for I/O overlap, then fold select_assign in expert-id order so
            // FP add order does not depend on the LRU hit set.
            let mut pending: Vec<Option<Tensor<2>>> = vec![None; num_experts];
            for &expert in &hits {
                pending[expert] = Some(run_expert(
                    Tensor::<2>::zeros([tokens, hidden], &device),
                    expert,
                    pool,
                ));
            }
            timed(&profile::MOE_PREFETCH, &device, || {
                pool.prefetch_layer_finish(&device)
                    .expect("streamed prefetch_layer_finish")
            });
            for &expert in &distinct_experts {
                if pending[expert].is_none() {
                    pending[expert] = Some(run_expert(
                        Tensor::<2>::zeros([tokens, hidden], &device),
                        expert,
                        pool,
                    ));
                }
            }
            for &expert in &distinct_experts {
                let part = pending[expert].take().expect("routed expert contrib");
                routed = timed(&profile::MOE_SCATTER, &device, || routed + part);
            }
        }

        // When the per-expert sync is disabled (QWEN35_STREAM_SYNC=0), still bound the in-flight
        // dispatch backlog -- but once per layer instead of once per routed expert, so this
        // layer's tiny GEMVs pipeline instead of round-tripping to the host ~8 times.
        if std::env::var("QWEN35_STREAM_SYNC").ok().as_deref() == Some("0") {
            let _ = device.sync();
        }

        let shared = timed(&profile::MOE_SHARED, &device, || {
            self.shared_expert_forward(hidden_states, prec)
                .reshape([tokens, hidden])
        });
        (routed + shared)
            .reshape([batch, seq_len, hidden])
            .cast(dtype)
    }

    pub fn route_topk(
        &self,
        hidden_states: Tensor<3>,
        top_k: usize,
    ) -> (Tensor<2, Int>, Tensor<2>) {
        let [batch, seq_len, _hidden] = hidden_states.dims();
        let tokens = batch * seq_len;
        let num_experts = self.gate.weight.val().dims()[1];
        let device = hidden_states.device();

        // Host softmax + top-k for every token count. GPU argmax over the 256-way gate is not
        // run-to-run stable on Metal (autotuned reduce / ties); one expert flip in prefill
        // diverges the greedy continuation. The gate is tiny (tokens×256), so the download
        // is cheaper than 8 GPU argmax/scatter launches anyway.
        let logits = linear3(&self.gate, hidden_states, Precision::F32)
            .reshape([tokens, num_experts])
            .cast(DType::F32);
        let logits_vec: Vec<f32> = logits.into_data().to_vec().expect("read router logits");
        let (sel_idx_vec, sel_w_vec) =
            host_softmax_topk(&logits_vec, tokens, num_experts, top_k, self.norm_topk_prob);
        let sel_idx =
            Tensor::<2, Int>::from_data(TensorData::new(sel_idx_vec, [tokens, top_k]), &device);
        let sel_w = Tensor::<2>::from_data(TensorData::new(sel_w_vec, [tokens, top_k]), &device);
        (sel_idx, sel_w)
    }

    fn expert_forward(&self, expert: usize, x: Tensor<3>, prec: Precision) -> Tensor<3> {
        let [batch, seq_len, hidden] = x.dims();
        let tokens = batch * seq_len;
        // NVFP4 experts take precedence over the fp8 sidecar and the bf16 stacks. This is the
        // per-expert host (T>16 prefill) path: the fused device gather-GEMV is T<=16-capped, and on
        // the official nvidia checkpoint the bf16 stacks are `[1,1,1]` placeholders, so there is no
        // bf16 fallback. Per plan pin 7 (correctness first, TTFT measured at the gate) this transiently
        // dequantizes the routed expert's NVFP4 bytes to f32 and runs the ordinary dense expert math.
        // Dequant-then-matmul has no `m_max` cap, so it needs no M-by-8 chunking (the M<=8 chunking pin
        // applies only to a raw NVFP4-GEMV path); it accepts any prefill M directly. Not on the
        // captured decode path (that is T=1 -> the fused arm), so the host read here is acceptable.
        if let Some(nvfp4) = &self.experts.nvfp4.0 {
            assert!(
                expert < nvfp4.e,
                "expert_forward nvfp4: expert index {expert} out of range {}",
                nvfp4.e
            );
            assert_eq!(
                hidden, nvfp4.h,
                "expert_forward nvfp4: activation hidden {hidden} != sidecar hidden {}",
                nvfp4.h
            );
            let inner = nvfp4.i;
            let device = x.device();

            let read_bytes = |t: Tensor<2, Int>| -> Vec<u8> {
                t.into_data()
                    .to_vec::<i8>()
                    .expect("read nvfp4 expert bytes")
                    .into_iter()
                    .map(|b| b as u8)
                    .collect()
            };
            let qw_gu_e = read_bytes(
                nvfp4
                    .qw_gu
                    .clone()
                    .slice([expert..expert + 1, 0..hidden, 0..inner])
                    .reshape([hidden, inner]),
            );
            let bs_gu_e = read_bytes(
                nvfp4
                    .bs_gu
                    .clone()
                    .slice([expert..expert + 1, 0..inner * 2, 0..hidden / 16])
                    .reshape([inner * 2, hidden / 16]),
            );
            let gscale_gu_e: Vec<f32> = nvfp4
                .gscale_gu
                .clone()
                .slice([expert..expert + 1, 0..2])
                .reshape([2])
                .into_data()
                .to_vec::<f32>()
                .expect("read nvfp4 gscale_gu");
            let qw_dn_e = read_bytes(
                nvfp4
                    .qw_dn
                    .clone()
                    .slice([expert..expert + 1, 0..inner, 0..hidden / 2])
                    .reshape([inner, hidden / 2]),
            );
            let bs_dn_e = read_bytes(
                nvfp4
                    .bs_dn
                    .clone()
                    .slice([expert..expert + 1, 0..hidden, 0..inner / 16])
                    .reshape([hidden, inner / 16]),
            );
            let gscale_dn_e: f32 = nvfp4
                .gscale_dn
                .clone()
                .slice([expert..expert + 1])
                .into_data()
                .to_vec::<f32>()
                .expect("read nvfp4 gscale_dn")[0];

            // Fused gate/up: gate half uses gscale_gu[0], up half uses gscale_gu[1] (per-output-channel
            // second-level scale), matching the loader's fusion order.
            let mut gu_gscale = vec![gscale_gu_e[0]; inner];
            gu_gscale.extend(std::iter::repeat_n(gscale_gu_e[1], inner));
            let gu = crate::nvfp4::dequant_nvfp4_outmajor(
                &qw_gu_e,
                &bs_gu_e,
                &gu_gscale,
                hidden,
                inner * 2,
            );
            let dn = crate::nvfp4::dequant_nvfp4_outmajor(
                &qw_dn_e,
                &bs_dn_e,
                &[gscale_dn_e],
                inner,
                hidden,
            );

            // dequant returns row-major [K,N]; transpose to the [out,in] layout `matmul_out_in` expects
            // (identical to how the bf16 branch slices `gate_up_proj`/`down_proj`).
            let gu_t = Tensor::<2>::from_data(TensorData::new(gu, [hidden, inner * 2]), &device)
                .transpose();
            let gate_w = gu_t.clone().slice([0..inner, 0..hidden]);
            let up_w = gu_t.slice([inner..inner * 2, 0..hidden]);
            let down_w =
                Tensor::<2>::from_data(TensorData::new(dn, [inner, hidden]), &device).transpose();

            let x2 = x.reshape([tokens, hidden]);
            let gate = silu(matmul_out_in(x2.clone(), gate_w, prec));
            let up = matmul_out_in(x2, up_w, prec);
            return matmul_out_in(gate * up, down_w, prec).reshape([batch, seq_len, hidden]);
        }
        #[cfg(feature = "cuda")]
        if let Some(fp8) = &self.experts.fp8.0 {
            assert!(
                expert < fp8.e,
                "expert_forward fp8: expert index {expert} out of range {}",
                fp8.e
            );
            assert_eq!(
                hidden, fp8.h,
                "expert_forward fp8: activation hidden {hidden} != sidecar hidden {}",
                fp8.h
            );
            let inner = fp8.i;
            let q_gu_e = fp8
                .q_gu
                .clone()
                .slice([expert..expert + 1, 0..hidden, 0..inner * 2])
                .reshape([hidden, inner * 2]);
            let s_gu_e = fp8
                .s_gu
                .clone()
                .slice([expert..expert + 1, 0..inner * 2])
                .reshape([inner * 2]);
            assert_eq!(
                q_gu_e.dims(),
                [hidden, inner * 2],
                "expert_forward fp8: q_gu_e layout mismatch"
            );
            assert_eq!(
                s_gu_e.dims(),
                [inner * 2],
                "expert_forward fp8: s_gu_e layout mismatch"
            );

            let x2 = x.reshape([tokens, hidden]).cast(DType::F32);
            let gu = crate::w8a16::w8a16_gemv(x2, q_gu_e, s_gu_e);
            let gate = gu.clone().slice([0..tokens, 0..inner]);
            let up = gu.slice([0..tokens, inner..inner * 2]);
            let h = silu(gate) * up;

            let q_dn_e = fp8
                .q_dn
                .clone()
                .slice([expert..expert + 1, 0..inner, 0..hidden])
                .reshape([inner, hidden]);
            let s_dn_e = fp8
                .s_dn
                .clone()
                .slice([expert..expert + 1, 0..hidden])
                .reshape([hidden]);
            assert_eq!(
                q_dn_e.dims(),
                [inner, hidden],
                "expert_forward fp8: q_dn_e layout mismatch"
            );
            assert_eq!(
                s_dn_e.dims(),
                [hidden],
                "expert_forward fp8: s_dn_e layout mismatch"
            );

            return crate::w8a16::w8a16_gemv(h, q_dn_e, s_dn_e).reshape([batch, seq_len, hidden]);
        }

        let gate_up_dims = self.experts.gate_up_proj.val().dims();
        let inner = gate_up_dims[1] / 2;
        let gate_w = self
            .experts
            .gate_up_proj
            .val()
            .slice([expert..expert + 1, 0..inner, 0..hidden])
            .reshape([inner, hidden]);
        let up_w = self
            .experts
            .gate_up_proj
            .val()
            .slice([expert..expert + 1, inner..(inner * 2), 0..hidden])
            .reshape([inner, hidden]);
        let down_w = self
            .experts
            .down_proj
            .val()
            .slice([expert..expert + 1, 0..hidden, 0..inner])
            .reshape([hidden, inner]);
        let x2 = x.reshape([tokens, hidden]);
        let gate = silu(matmul_out_in(x2.clone(), gate_w, prec));
        let up = matmul_out_in(x2, up_w, prec);
        matmul_out_in(gate * up, down_w, prec).reshape([batch, seq_len, hidden])
    }

    fn shared_expert_forward(&self, hidden_states: Tensor<3>, prec: Precision) -> Tensor<3> {
        // HF Qwen3_5MoeSparseMoeBlock: shared = sigmoid(shared_expert_gate(x)) · MLP(x), where
        // MLP(x) = down_proj(silu(gate_proj(x)) · up_proj(x)). The gate is a strongly-negative-logit
        // sigmoid (≈0.076) that keeps the shared expert nearly off — so the F32 matmul path in linear3
        // MUST cast the bf16 weight to f32 (else CubeCL's f32×bf16 matmul silently corrupts the gate
        // logit, over-activating the shared expert and wrecking the residual stream).
        let gate = silu(ql3(
            &self.shared_expert.gate_proj_fp8,
            &self.shared_expert.gate_proj,
            hidden_states.clone(),
            prec,
        ));
        let up = ql3(
            &self.shared_expert.up_proj_fp8,
            &self.shared_expert.up_proj,
            hidden_states.clone(),
            prec,
        );
        let shared = ql3(
            &self.shared_expert.down_proj_fp8,
            &self.shared_expert.down_proj,
            gate * up,
            prec,
        );
        let shared_gate = sigmoid(linear3(
            &self.shared_expert_gate,
            hidden_states,
            Precision::F32,
        ))
        .cast(shared.dtype());
        shared * shared_gate
    }

    /// CUDA-graph-capturable static MoE step: the fused device-routed branch of [`Self::forward_impl`].
    /// The fused per-assignment rows are combined by a fixed-order token-major sum, so the captured path
    /// does not stage any per-step routing helper tensors or atomic scatter-adds.
    ///
    /// Fused preconditions are HARD: an unmet one `panic!`s (release-visible, dumping which precondition
    /// plus the routing shape/dtype and `T`) rather than silently taking the host-loop fallback (that
    /// fallback D2H-syncs the router — capture poison). Run [`Self::preflight_static`] before capture to
    /// fail at build time instead. The math is identical to the fused branch of [`Self::forward_impl`].
    #[cfg(feature = "cuda")]
    pub fn forward_static(&self, hidden_states: Tensor<3>, prec: Precision) -> Tensor<3> {
        let [batch, seq_len, hidden] = hidden_states.dims();
        let tokens = batch * seq_len;
        let num_experts = self.gate.weight.val().dims()[1];
        let top_k = (self.num_experts_per_tok).min(num_experts);
        let dtype = hidden_states.dtype();
        let n = tokens * top_k;

        let (sel_idx, sel_w) = self.route_topk(hidden_states.clone(), top_k);
        let route_shape = sel_idx.dims();
        let route_dtype = sel_idx.dtype();
        assert!(
            qwen35_fused_moe_enabled(),
            "Qwen3_5SharedMoeBlock::forward_static: fused MoE is DISABLED \
             (set_qwen35_fused_moe_enabled(true) before capture) — static decode cannot fall back to \
             the host loop. route[{route_shape:?} {route_dtype:?}] tokens={tokens} top_k={top_k}"
        );
        assert!(
            tokens <= QWEN35_FUSED_MOE_MAX_T,
            "Qwen3_5SharedMoeBlock::forward_static: tokens {tokens} > QWEN35_FUSED_MOE_MAX_T \
             {QWEN35_FUSED_MOE_MAX_T} (fused MoE kernel token bound). \
             route[{route_shape:?} {route_dtype:?}] top_k={top_k}"
        );
        // Dispatch precedence (static/captured path): NVFP4 experts -> fp8 sidecar -> bf16 stacks.
        // The nvfp4 arm mirrors the fp8 arm exactly (same deterministic reshape+sum combine, same
        // hard-panic diagnostics) and stages no per-step routing helpers, so it is capture-stable.
        if let Some(nvfp4) = &self.experts.nvfp4.0 {
            assert_eq!(
                nvfp4.e, num_experts,
                "Qwen3.5 fused nvfp4 MoE sidecar expert count {} != gate experts {num_experts}",
                nvfp4.e
            );
            assert_eq!(
                nvfp4.h, hidden,
                "Qwen3.5 fused nvfp4 MoE sidecar hidden {} != activation hidden {hidden}",
                nvfp4.h
            );
            assert_eq!(
                nvfp4.qw_gu.dims(),
                [num_experts, hidden, nvfp4.i],
                "nvfp4 qw_gu must be [E,H,I]"
            );
            assert_eq!(
                nvfp4.bs_gu.dims(),
                [num_experts, nvfp4.i * 2, hidden / 16],
                "nvfp4 bs_gu must be [E,2I,H/16]"
            );
            assert_eq!(
                nvfp4.gscale_gu.dims(),
                [num_experts, 2],
                "nvfp4 gscale_gu must be [E,2]"
            );
            assert_eq!(
                nvfp4.qw_dn.dims(),
                [num_experts, nvfp4.i, hidden / 2],
                "nvfp4 qw_dn must be [E,I,H/2]"
            );
            assert_eq!(
                nvfp4.bs_dn.dims(),
                [num_experts, hidden, nvfp4.i / 16],
                "nvfp4 bs_dn must be [E,H,I/16]"
            );
            assert_eq!(
                nvfp4.gscale_dn.dims(),
                [num_experts],
                "nvfp4 gscale_dn must be [E]"
            );
            assert_eq!(
                nvfp4.qw_gu.dtype(),
                DType::I8,
                "nvfp4 qw_gu must be DType::I8"
            );
            assert_eq!(
                nvfp4.qw_dn.dtype(),
                DType::I8,
                "nvfp4 qw_dn must be DType::I8"
            );
            assert_eq!(
                nvfp4.bs_gu.dtype(),
                DType::I8,
                "nvfp4 bs_gu must be DType::I8"
            );
            assert_eq!(
                nvfp4.bs_dn.dtype(),
                DType::I8,
                "nvfp4 bs_dn must be DType::I8"
            );

            let assign_e = sel_idx.reshape([n]);
            let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
            let x2 = hidden_states
                .clone()
                .reshape([tokens, hidden])
                .cast(DType::F32);
            let y = crate::moe_grouped::fused_moe_gu2_down_nvfp4(
                x2,
                nvfp4.qw_gu.clone(),
                nvfp4.bs_gu.clone(),
                nvfp4.gscale_gu.clone(),
                nvfp4.qw_dn.clone(),
                nvfp4.bs_dn.clone(),
                nvfp4.gscale_dn.clone(),
                assign_e,
                sel_w_flat,
                hidden,
                nvfp4.i,
                n,
            )
            .cast(dtype);
            let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
            let shared = self
                .shared_expert_forward(hidden_states, prec)
                .reshape([tokens, hidden]);
            return (routed + shared)
                .reshape([batch, seq_len, hidden])
                .cast(dtype);
        }
        if let Some(fp8) = &self.experts.fp8.0 {
            assert_eq!(
                fp8.e, num_experts,
                "Qwen3.5 fused fp8 MoE sidecar expert count {} != gate experts {num_experts}",
                fp8.e
            );
            assert_eq!(
                fp8.h, hidden,
                "Qwen3.5 fused fp8 MoE sidecar hidden {} != activation hidden {hidden}",
                fp8.h
            );
            assert_eq!(
                fp8.q_gu.dims(),
                [num_experts, hidden, fp8.i * 2],
                "fp8 q_gu must be [E,H,2I]"
            );
            assert_eq!(
                fp8.s_gu.dims(),
                [num_experts, fp8.i * 2],
                "fp8 s_gu must be [E,2I]"
            );
            assert_eq!(
                fp8.q_dn.dims(),
                [num_experts, fp8.i, hidden],
                "fp8 q_dn must be [E,I,H]"
            );
            assert_eq!(
                fp8.s_dn.dims(),
                [num_experts, hidden],
                "fp8 s_dn must be [E,H]"
            );
            assert_eq!(fp8.q_gu.dtype(), DType::I8, "fp8 q_gu must be DType::I8");
            assert_eq!(fp8.q_dn.dtype(), DType::I8, "fp8 q_dn must be DType::I8");
            assert_eq!(fp8.s_gu.dtype(), DType::F32, "fp8 s_gu must be DType::F32");
            assert_eq!(fp8.s_dn.dtype(), DType::F32, "fp8 s_dn must be DType::F32");

            let assign_e = sel_idx.reshape([n]);
            let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
            let x2 = hidden_states
                .clone()
                .reshape([tokens, hidden])
                .cast(DType::F32);
            let y = crate::moe_grouped::fused_moe_gu2_down_fp8(
                x2,
                fp8.q_gu.clone(),
                fp8.s_gu.clone(),
                fp8.q_dn.clone(),
                fp8.s_dn.clone(),
                assign_e,
                sel_w_flat,
                hidden,
                fp8.i,
                n,
            )
            .cast(dtype);
            let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
            let shared = self
                .shared_expert_forward(hidden_states, prec)
                .reshape([tokens, hidden]);
            return (routed + shared)
                .reshape([batch, seq_len, hidden])
                .cast(dtype);
        }

        let gate_up_dims = self.experts.gate_up_proj.val().dims();
        let down_dims = self.experts.down_proj.val().dims();
        let placeholder = gate_up_dims == [1, 1, 1] || down_dims == [1, 1, 1];
        assert!(
            !placeholder && down_dims[0] == num_experts && down_dims[1] == hidden,
            "Qwen3_5SharedMoeBlock::forward_static: no fused MoE weights available (fp8 sidecar absent \
             AND bf16 expert stacks are placeholder/mis-shaped: gate_up={gate_up_dims:?} \
             down={down_dims:?}, expected down=[{num_experts},{hidden},inner]). Quantize the fp8 \
             sidecar or load real bf16 stacks before capture. route[{route_shape:?} {route_dtype:?}] \
             tokens={tokens} top_k={top_k}"
        );
        let inner = down_dims[2];
        assert_eq!(
            gate_up_dims,
            [num_experts, inner * 2, hidden],
            "Qwen3.5 fused bf16 MoE requires non-placeholder gate_up [E,2I,H]"
        );
        assert_eq!(
            down_dims,
            [num_experts, hidden, inner],
            "Qwen3.5 fused bf16 MoE requires non-placeholder down [E,H,I]"
        );
        let assign_e = sel_idx.reshape([n]);
        let sel_w_flat = sel_w.reshape([n]).cast(DType::F32);
        let x2 = hidden_states
            .clone()
            .reshape([tokens, hidden])
            .cast(DType::F32);
        let y = crate::moe_grouped::fused_moe_gu2_down_bf16(
            x2,
            self.experts.gate_up_proj.val(),
            self.experts.down_proj.val(),
            assign_e,
            sel_w_flat,
            hidden,
            inner,
            n,
        )
        .cast(dtype);
        let routed = Self::combine_token_major_assignments(y, tokens, top_k, hidden);
        let shared = self
            .shared_expert_forward(hidden_states, prec)
            .reshape([tokens, hidden]);
        (routed + shared)
            .reshape([batch, seq_len, hidden])
            .cast(dtype)
    }

    /// NdArray (non-CUDA) static MoE step. There is no `Fused35MoeBackend` off CUDA, so the fused
    /// device-routed branch does not exist here; the static step runs the eager oracle (host-loop)
    /// expert path, which is mathematically identical to the fused branch (the CUDA G2 gate covers the
    /// fused dispatch end-to-end).
    #[cfg(not(feature = "cuda"))]
    pub fn forward_static(&self, hidden_states: Tensor<3>, prec: Precision) -> Tensor<3> {
        self.forward_impl(hidden_states, prec, false)
    }

    /// Build-time preflight for [`Self::forward_static`]: verify the fused MoE preconditions for a step
    /// of `tokens` tokens WITHOUT running the router, so a capture driver can fail before the expensive
    /// eager prefill/warmup instead of `panic!`ing mid-capture. Same predicate as `forward_static`'s
    /// hard asserts; returns `Err(reason)` instead of panicking.
    #[cfg(feature = "cuda")]
    pub fn preflight_static(&self, tokens: usize) -> Result<(), String> {
        let [hidden, num_experts] = self.gate.weight.val().dims();
        if !qwen35_fused_moe_enabled() {
            return Err("fused MoE disabled (call set_qwen35_fused_moe_enabled(true))".to_string());
        }
        if tokens > QWEN35_FUSED_MOE_MAX_T {
            return Err(format!(
                "tokens {tokens} > QWEN35_FUSED_MOE_MAX_T {QWEN35_FUSED_MOE_MAX_T}"
            ));
        }
        // NVFP4 experts are fused-capable and take dispatch precedence over fp8/bf16 (see
        // forward_static). A present, well-shaped sidecar preflights OK.
        if let Some(nvfp4) = &self.experts.nvfp4.0 {
            if nvfp4.e != num_experts {
                return Err(format!(
                    "nvfp4 sidecar expert count {} != gate experts {num_experts}",
                    nvfp4.e
                ));
            }
            if nvfp4.h != hidden {
                return Err(format!(
                    "nvfp4 sidecar hidden {} != gate hidden {hidden}",
                    nvfp4.h
                ));
            }
            if nvfp4.qw_gu.dtype() != DType::I8
                || nvfp4.qw_dn.dtype() != DType::I8
                || nvfp4.bs_gu.dtype() != DType::I8
                || nvfp4.bs_dn.dtype() != DType::I8
            {
                return Err("nvfp4 quantized/scale stacks must be DType::I8".to_string());
            }
            return Ok(());
        }
        if let Some(fp8) = &self.experts.fp8.0 {
            if fp8.e != num_experts {
                return Err(format!(
                    "fp8 sidecar expert count {} != gate experts {num_experts}",
                    fp8.e
                ));
            }
            if fp8.h != hidden {
                return Err(format!(
                    "fp8 sidecar hidden {} != gate hidden {hidden}",
                    fp8.h
                ));
            }
            if fp8.q_gu.dtype() != DType::I8 || fp8.q_dn.dtype() != DType::I8 {
                return Err("fp8 quantized stacks must be DType::I8".to_string());
            }
            return Ok(());
        }
        let gate_up_dims = self.experts.gate_up_proj.val().dims();
        let down_dims = self.experts.down_proj.val().dims();
        if gate_up_dims == [1, 1, 1] || down_dims == [1, 1, 1] {
            return Err(format!(
                "no fp8 sidecar AND bf16 expert stacks are placeholder: gate_up={gate_up_dims:?} \
                 down={down_dims:?}"
            ));
        }
        if down_dims[0] != num_experts || down_dims[1] != hidden {
            return Err(format!(
                "bf16 down stack {down_dims:?} does not match [num_experts={num_experts}, \
                 hidden={hidden}, inner]"
            ));
        }
        Ok(())
    }

    /// NdArray (non-CUDA) preflight: the oracle host-loop path always runs, so there is nothing to
    /// gate. Present so the model-level preflight aggregates uniformly across backends.
    #[cfg(not(feature = "cuda"))]
    pub fn preflight_static(&self, tokens: usize) -> Result<(), String> {
        let _ = tokens;
        Ok(())
    }
}

/// Env-gated (`QWEN35_PROFILE=1`) wall-clock attribution for the streamed decode path.
///
/// Exists because the streamed-expert pool's own io/upload/compute counters accounted for only
/// ~64s of a measured 132s decode -- the rest was outside the pool and unattributed. Each section
/// syncs the device before stopping its timer, so the numbers are real GPU time rather than
/// enqueue time; that makes a profiled run somewhat slower than an unprofiled one, but it is the
/// only way to attribute async dispatch correctly.
pub mod profile {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    pub static ENABLED: AtomicBool = AtomicBool::new(false);

    macro_rules! counters {
        ($($name:ident),* $(,)?) => {
            $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
            /// `(label, nanoseconds)` for every counter, in declaration order.
            pub fn snapshot() -> Vec<(&'static str, u64)> {
                vec![$((stringify!($name), $name.load(Ordering::Relaxed))),*]
            }
        };
    }

    counters!(
        EMBED,
        GDN_ATTN,
        FULL_ATTN,
        ROUTER,
        MOE_PREFETCH,
        MOE_EXPERTS,
        MOE_SHARED,
        MOE_SCATTER,
        FINAL_NORM,
        LM_HEAD
    );

    // Host-side ENQUEUE-time counters (no device sync), gated by QWEN35_CPU_TIME=1. They show
    // how long the CPU spends building/dispatching work -- when this rivals wall-clock decode
    // time, the host, not the GPU, is the bottleneck.
    pub static CPU_GDN_PROJ: AtomicU64 = AtomicU64::new(0);
    pub static CPU_GDN_CONV: AtomicU64 = AtomicU64::new(0);
    pub static CPU_GDN_SPLIT: AtomicU64 = AtomicU64::new(0);
    pub static CPU_GDN_GATE: AtomicU64 = AtomicU64::new(0);
    pub static CPU_GDN_STATE: AtomicU64 = AtomicU64::new(0);
    pub static CPU_GDN_OUT: AtomicU64 = AtomicU64::new(0);
    pub static CPU_ROUTE_RB: AtomicU64 = AtomicU64::new(0);
    pub static CPU_ROUTE_DRAIN: AtomicU64 = AtomicU64::new(0);

    pub fn cpu_time_enabled() -> bool {
        std::env::var("QWEN35_CPU_TIME").ok().as_deref() == Some("1")
    }

    pub fn cpu_report() -> String {
        let rows = vec![
            ("CPU_GDN_PROJ", CPU_GDN_PROJ.load(Ordering::Relaxed)),
            ("CPU_GDN_CONV", CPU_GDN_CONV.load(Ordering::Relaxed)),
            ("CPU_GDN_SPLIT", CPU_GDN_SPLIT.load(Ordering::Relaxed)),
            ("CPU_GDN_GATE", CPU_GDN_GATE.load(Ordering::Relaxed)),
            ("CPU_GDN_STATE", CPU_GDN_STATE.load(Ordering::Relaxed)),
            ("CPU_GDN_OUT", CPU_GDN_OUT.load(Ordering::Relaxed)),
            ("CPU_ROUTE_RB", CPU_ROUTE_RB.load(Ordering::Relaxed)),
            ("CPU_ROUTE_DRAIN", CPU_ROUTE_DRAIN.load(Ordering::Relaxed)),
        ];
        let mut out = String::from(
            "host enqueue profile (no device sync):
",
        );
        for (label, ns) in rows {
            out.push_str(&format!(
                "    {label:14} {:8.2}s
",
                ns as f64 / 1e9
            ));
        }
        out
    }

    pub fn init_from_env() {
        ENABLED.store(
            std::env::var("QWEN35_PROFILE").ok().as_deref() == Some("1"),
            Ordering::Relaxed,
        );
    }

    pub fn enabled() -> bool {
        ENABLED.load(Ordering::Relaxed)
    }

    pub fn add(counter: &AtomicU64, ns: u64) {
        counter.fetch_add(ns, Ordering::Relaxed);
    }

    pub fn report() -> String {
        let rows = snapshot();
        let total: u64 = rows.iter().map(|(_, ns)| *ns).sum();
        let mut out = String::from("model profile (device-synced):\n");
        for (label, ns) in rows {
            out.push_str(&format!(
                "    {label:14} {:8.2}s  ({:5.1}%)\n",
                ns as f64 / 1e9,
                if total > 0 {
                    ns as f64 * 100.0 / total as f64
                } else {
                    0.0
                }
            ));
        }
        out.push_str(&format!("    {:14} {:8.2}s\n", "TOTAL", total as f64 / 1e9));
        out
    }
}

/// Time `body`, syncing `device` first so async GPU work lands inside the measured window.
#[inline]
fn timed<T>(
    counter: &std::sync::atomic::AtomicU64,
    device: &Device,
    body: impl FnOnce() -> T,
) -> T {
    if !profile::enabled() {
        return body();
    }
    let t = std::time::Instant::now();
    let out = body();
    let _ = device.sync();
    profile::add(counter, t.elapsed().as_nanos() as u64);
    out
}

fn matmul_out_in(x: Tensor<2>, weight_out_in: Tensor<2>, prec: Precision) -> Tensor<2> {
    let weight_in_out = weight_out_in.transpose();
    let xdt = x.dtype();
    match prec {
        // Keep the GEMM uniform-dtype; see the linear3 F32 invariant.
        Precision::F32 => x.matmul(weight_in_out.cast(xdt)),
        Precision::Bf16 => x
            .cast(DType::BF16)
            .matmul(weight_in_out.cast(DType::BF16))
            .cast(DType::F32),
        Precision::F16 => x
            .cast(DType::F16)
            .matmul(weight_in_out.cast(DType::F16))
            .cast(DType::F32),
    }
}

impl Qwen3_5FullAttnLayer {
    /// Full-attention sublayer forward for the Qwen3.6/Qwen3.5-MoE hybrid text tower.
    ///
    /// This lane intentionally stops after the attention residual; the shared-MoE forward remains
    /// stubbed for a later increment.
    pub fn forward(&self, hidden_states: Tensor<3>, position_ids: Tensor<2, Int>) -> Tensor<3> {
        self.forward_prec(hidden_states, position_ids, Precision::F32)
    }

    pub fn forward_prec(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        prec: Precision,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        residual + self.self_attn.forward(hidden_states, position_ids, prec)
    }

    pub fn forward_decoder_with_cache(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states =
            self.self_attn
                .forward_with_cache(hidden_states, position_ids, cache, prec);
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    /// `docs/MEMORY_STREAMING_PLAN.md` streamed sibling of [`Self::forward_decoder_with_cache`]:
    /// identical attention/residual computation, but the MoE sublayer streams its routed experts
    /// through `pool` (see [`Qwen3_5SharedMoeBlock::forward_streamed`]) instead of reading a
    /// fully-resident expert stack.
    pub fn forward_decoder_with_cache_streamed(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
        pool: &mut ExpertSlotPool,
        layer_idx: usize,
    ) -> Tensor<3> {
        let prof_device = hidden_states.device();
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = timed(&profile::FULL_ATTN, &prof_device, || {
            self.self_attn
                .forward_with_cache(hidden_states, position_ids, cache, prec)
        });
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self
            .mlp
            .forward_streamed(hidden_states, prec, pool, layer_idx);
        residual + hidden_states
    }

    /// CUDA-graph-capturable full-attention decode layer: one `[B,1,H]` token, device-`pos`
    /// static KV write, pre-hoisted partial-RoPE frequency table, and fixed-`T_max` masked SDPA.
    /// The MoE goes through the static [`Qwen3_5SharedMoeBlock::forward_static`], NOT the eager
    /// `mlp.forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decoder_with_cache_static_pre(
        &self,
        hidden_states: Tensor<3>,
        pos: Tensor<1, Int>,
        cache: &mut KVCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache_static_pre(
            hidden_states,
            pos,
            cache,
            prec,
            freqs,
            arange_tmax,
        );
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward_static(hidden_states, prec);
        residual + hidden_states
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decoder_with_cache_sdpa_reference(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        let hidden_states = self.self_attn.forward_with_cache_sdpa_reference(
            hidden_states,
            position_ids,
            cache,
            prec,
        );
        let hidden_states = residual + hidden_states;

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }
}

impl Qwen3_5GdnLayer {
    /// Recurrent single-token decode for a linear-attention layer, stopping after the attention
    /// residual just like the full-attention lane.
    pub fn forward_recurrent(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        residual
            + self
                .linear_attn
                .forward_recurrent(hidden_states, cache, prec)
    }

    /// Static-cache recurrent single-token decode for a linear-attention layer.
    pub fn step_recurrent_static(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let residual = hidden_states.clone();
        let hidden_states = self.input_layernorm.forward(hidden_states);
        residual
            + self
                .linear_attn
                .step_recurrent_static(hidden_states, cache, prec)
    }

    /// CUDA-graph-capturable full GDN decode layer (attention residual + MoE) for one `[B,1,H]` token:
    /// the static sibling of [`Self::forward_decoder_recurrent`]. The GDN recurrence uses the static
    /// state cache ([`Self::step_recurrent_static`]) and the MoE goes through the static
    /// [`Qwen3_5SharedMoeBlock::forward_static`], NOT the eager `mlp.forward`. Norms/residuals are
    /// identical to the eager path.
    pub fn forward_decoder_recurrent_static(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let hidden_states = self.step_recurrent_static(hidden_states, cache, prec);

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward_static(hidden_states, prec);
        residual + hidden_states
    }

    /// O(S) sequential prefill helper: applies the recurrent decode step token-by-token.
    pub fn forward_prefill_recurrent(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let [_batch, seq_len, hidden] = hidden_states.dims();
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let step =
                hidden_states
                    .clone()
                    .slice([0..hidden_states.dims()[0], t..(t + 1), 0..hidden]);
            outs.push(self.forward_recurrent(step, cache, prec));
        }
        Tensor::cat(outs, 1)
    }

    /// O(S) static-cache prefill helper: applies the recurrent static step token-by-token.
    pub fn forward_prefill_recurrent_static(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        assert!(
            cache.is_static(),
            "Qwen3_5GdnLayer::forward_prefill_recurrent_static requires \
             GdnStateCache::init_static(batch, device) before prefill"
        );
        let [_batch, seq_len, hidden] = hidden_states.dims();
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let step =
                hidden_states
                    .clone()
                    .slice([0..hidden_states.dims()[0], t..(t + 1), 0..hidden]);
            outs.push(self.step_recurrent_static(step, cache, prec));
        }
        Tensor::cat(outs, 1)
    }

    pub fn forward_decoder_recurrent(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let [_, seq_len, _] = hidden_states.dims();
        let hidden_states = if seq_len == 1 {
            self.forward_recurrent(hidden_states, cache, prec)
        } else {
            self.forward_prefill_recurrent(hidden_states, cache, prec)
        };

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self.mlp.forward(hidden_states, prec);
        residual + hidden_states
    }

    /// `docs/MEMORY_STREAMING_PLAN.md` streamed sibling of [`Self::forward_decoder_recurrent`]:
    /// identical GDN recurrence/residual computation, but the MoE sublayer streams its routed
    /// experts through `pool` instead of reading a fully-resident expert stack.
    pub fn forward_decoder_recurrent_streamed(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
        pool: &mut ExpertSlotPool,
        layer_idx: usize,
    ) -> Tensor<3> {
        let [_, seq_len, _] = hidden_states.dims();
        let prof_device = hidden_states.device();
        let hidden_states = timed(&profile::GDN_ATTN, &prof_device, || {
            if seq_len == 1 {
                self.forward_recurrent(hidden_states, cache, prec)
            } else {
                self.forward_prefill_recurrent(hidden_states, cache, prec)
            }
        });

        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states);
        let hidden_states = self
            .mlp
            .forward_streamed(hidden_states, prec, pool, layer_idx);
        residual + hidden_states
    }
}

#[derive(Clone, Copy)]
enum GdnStateWriteMode {
    Functional,
    Static,
}

impl Qwen3_5GdnAttention {
    /// Recurrent Qwen3.6/Qwen3.5-MoE Gated-DeltaNet decode for `hidden_states: [B, 1, H]`.
    ///
    /// Implements `docs/specs/L1.3-gdn-math.md`: qkv projection, depthwise causal conv over the
    /// concatenated qkv vector, SiLU, q/k L2 normalization, per-value-head gated delta update in f32,
    /// readout from the updated state, RMSNorm before the SiLU z gate, then output projection.
    pub fn forward_recurrent(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        self.forward_recurrent_impl(hidden_states, cache, prec, GdnStateWriteMode::Functional)
    }

    /// Static-cache recurrent single-token decode for `hidden_states: [B, 1, H]`.
    pub fn step_recurrent_static(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        assert!(
            cache.is_static(),
            "Qwen3_5GdnAttention::step_recurrent_static requires \
             GdnStateCache::init_static(batch, device) before decode"
        );
        self.forward_recurrent_impl(hidden_states, cache, prec, GdnStateWriteMode::Static)
    }

    fn forward_recurrent_impl(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
        write_mode: GdnStateWriteMode,
    ) -> Tensor<3> {
        let [batch_size, seq_len, hidden_size] = hidden_states.dims();
        debug_assert_eq!(seq_len, 1, "GDN recurrent decode expects [B, 1, H]");
        let ct = profile::cpu_time_enabled();
        macro_rules! cpu_t {
            ($ctr:expr, $t0:expr) => {
                if ct {
                    profile::add(&$ctr, $t0.elapsed().as_nanos() as u64);
                }
            };
        }
        // Output the residual-stream dtype (F32 on the real model), NOT `prec` — `residual + gdn_out`
        // must match. The recurrence is f32 internally; the out_proj (linear3) is F32; cast to this.
        let out_dtype = hidden_states.dtype();

        let qkv_dim = self.in_proj_qkv.weight.val().dims()[1];
        let value_dim_total = self.in_proj_z.weight.val().dims()[1];
        let num_value_heads = self.A_log.val().dims()[0];
        let value_head_dim = self.norm.gamma.val().dims()[0];
        let num_key_heads = (qkv_dim - value_dim_total) / (2 * value_head_dim);
        let key_head_dim = value_head_dim;
        debug_assert_eq!(num_value_heads * value_head_dim, value_dim_total);
        debug_assert_eq!(num_value_heads, cache.num_value_heads);
        debug_assert_eq!(key_head_dim, cache.key_dim);
        debug_assert_eq!(value_head_dim, cache.value_dim);
        debug_assert_eq!(qkv_dim, cache.qkv_dim);

        let _tp = std::time::Instant::now();
        let qkv_unconv = ql3(
            &self.in_proj_qkv_fp8,
            &self.in_proj_qkv,
            hidden_states.clone(),
            prec,
        )
        .reshape([batch_size, qkv_dim])
        .cast(DType::F32);
        let z = ql3(
            &self.in_proj_z_fp8,
            &self.in_proj_z,
            hidden_states.clone(),
            prec,
        )
        .reshape([batch_size, num_value_heads, value_head_dim])
        .cast(DType::F32);
        let in_a = ql3(
            &self.in_proj_a_fp8,
            &self.in_proj_a,
            hidden_states.clone(),
            prec,
        )
        .reshape([batch_size, num_value_heads])
        .cast(DType::F32);
        let in_b = ql3(&self.in_proj_b_fp8, &self.in_proj_b, hidden_states, prec)
            .reshape([batch_size, num_value_heads])
            .cast(DType::F32);
        cpu_t!(profile::CPU_GDN_PROJ, _tp);

        let _tc = std::time::Instant::now();
        let device = qkv_unconv.device();
        let dtype = qkv_unconv.dtype();
        cache.ensure_allocated(batch_size, &device, dtype);

        // `value_head_dim % 32 == 0` is a hard precondition of `gdn_state_gate` (its threadgroup is
        // one warp wide per value head), asserted host-side inside the launch. Check it here instead
        // of panicking: toy/test configs (value_head_dim 8) must fall through to the unfused path.
        #[cfg(feature = "cubecl-gpu")]
        if qwen35_fused_gdn_enabled()
            && batch_size == 1
            && value_head_dim % 32 == 0
            && matches!(write_mode, GdnStateWriteMode::Functional)
        {
            let conv_weight = self
                .conv1d
                .weight
                .val()
                .cast(DType::F32)
                .reshape([qkv_dim, cache.kernel_dim]);
            let conv_hist = cache
                .conv
                .as_ref()
                .expect("GDN conv cache must be allocated")
                .clone();
            let prev_state = cache
                .state
                .take()
                .unwrap_or_else(|| {
                    Tensor::<4>::zeros(
                        [batch_size, num_value_heads, key_head_dim, value_head_dim],
                        &device,
                    )
                })
                .cast(DType::F32);
            let dt_bias = self
                .dt_bias
                .val()
                .cast(DType::F32)
                .reshape([num_value_heads]);
            let a_log = self.A_log.val().cast(DType::F32).reshape([num_value_heads]);
            let norm_w = self
                .norm
                .gamma
                .val()
                .cast(DType::F32)
                .reshape([value_head_dim]);

            let sh = crate::gdn_kernel::GdnShape {
                batch: batch_size,
                qkv_dim,
                num_value_heads,
                num_key_heads,
                key_head_dim,
                value_head_dim,
                kernel_size: cache.kernel_dim,
                epsilon: self.norm.epsilon as f32,
            };

            let (o_gated, new_state, new_hist) = crate::gdn_kernel::gdn_step_fused(
                qkv_unconv,
                z,
                in_a,
                in_b,
                conv_hist,
                conv_weight,
                dt_bias,
                a_log,
                norm_w,
                prev_state,
                sh,
            );

            cache.conv = Some(new_hist);
            cache.set_state(new_state);
            cpu_t!(profile::CPU_GDN_CONV, _tc);

            let out = o_gated.reshape([batch_size, 1, value_dim_total]);
            let out = ql3(&self.out_proj_fp8, &self.out_proj, out, prec).reshape([
                batch_size,
                1,
                hidden_size,
            ]);
            return out.cast(out_dtype);
        }

        let history = cache
            .conv
            .as_ref()
            .expect("GDN conv cache must be allocated")
            .clone();
        let conv_weight = self
            .conv1d
            .weight
            .val()
            .cast(DType::F32)
            .reshape([qkv_dim, cache.kernel_dim]);
        let mut qkv_conv = qkv_unconv.clone()
            * conv_weight
                .clone()
                .slice([0..qkv_dim, (cache.kernel_dim - 1)..cache.kernel_dim])
                .reshape([1, qkv_dim]);
        for i in 0..(cache.kernel_dim - 1) {
            let x_i = history
                .clone()
                .slice([0..batch_size, i..(i + 1), 0..qkv_dim])
                .reshape([batch_size, qkv_dim])
                .cast(DType::F32);
            let w_i = conv_weight
                .clone()
                .slice([0..qkv_dim, i..(i + 1)])
                .reshape([1, qkv_dim]);
            qkv_conv = qkv_conv + x_i * w_i;
        }
        // second strong ref would COW-move the conv VA under capture — same read-before-write discipline as the state copy-back
        core::mem::drop(history);
        match write_mode {
            GdnStateWriteMode::Functional => {
                cache.push_conv(qkv_unconv);
            }
            GdnStateWriteMode::Static => {
                cache.push_conv_static(qkv_unconv);
            }
        }
        let qkv_silu = silu(qkv_conv).cast(DType::F32);
        cpu_t!(profile::CPU_GDN_CONV, _tc);
        let _ts = std::time::Instant::now();

        // qwen3_5_moe Qwen3_5MoeGatedDeltaNet splits the post-conv mixed_qkv with a FLAT block split
        // `torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)` then reshapes each block to
        // heads — NOT the per-k-head interleave that qwen3_next uses (verified against the authoritative
        // modeling_qwen3_5_moe.py). q=[0:2048]→[16,128], k=[2048:4096]→[16,128], v=[4096:8192]→[32,128].
        let q_dim = num_key_heads * key_head_dim; // key_dim = 2048
        let q = qkv_silu.clone().slice([0..batch_size, 0..q_dim]).reshape([
            batch_size,
            num_key_heads,
            key_head_dim,
        ]);
        let k = qkv_silu
            .clone()
            .slice([0..batch_size, q_dim..(2 * q_dim)])
            .reshape([batch_size, num_key_heads, key_head_dim]);
        let v = qkv_silu
            .slice([0..batch_size, (2 * q_dim)..qkv_dim])
            .reshape([batch_size, num_value_heads, value_head_dim]);

        let q_norm = ((q.clone() * q.clone()).sum_dim(2) + 1e-6).sqrt().reshape([
            batch_size,
            num_key_heads,
            1,
        ]);
        let k_norm = ((k.clone() * k.clone()).sum_dim(2) + 1e-6).sqrt().reshape([
            batch_size,
            num_key_heads,
            1,
        ]);
        let q = (q / q_norm).mul_scalar((key_head_dim as f64).sqrt().recip());
        let k = k / k_norm;

        let q = q
            .unsqueeze_dim::<4>(2)
            .repeat(&[1, 1, num_value_heads / num_key_heads, 1])
            .reshape([batch_size, num_value_heads, key_head_dim]);
        let k = k
            .unsqueeze_dim::<4>(2)
            .repeat(&[1, 1, num_value_heads / num_key_heads, 1])
            .reshape([batch_size, num_value_heads, key_head_dim]);
        cpu_t!(profile::CPU_GDN_SPLIT, _ts);
        let _tg = std::time::Instant::now();

        let dt = in_a
            + self
                .dt_bias
                .val()
                .cast(DType::F32)
                .reshape([1, num_value_heads]);
        let softplus = dt.clone().clamp_min(0.0) + ((dt.abs().mul_scalar(-1.0)).exp() + 1.0).log();
        let a = (self
            .A_log
            .val()
            .cast(DType::F32)
            .exp()
            .reshape([1, num_value_heads])
            * softplus)
            .mul_scalar(-1.0)
            .exp();
        let b = sigmoid(in_b).cast(DType::F32);
        cpu_t!(profile::CPU_GDN_GATE, _tg);
        let _tr = std::time::Instant::now();

        let prev_state = match write_mode {
            GdnStateWriteMode::Functional => cache
                .state
                .take()
                .unwrap_or_else(|| {
                    let device = q.device();
                    Tensor::<4>::zeros(
                        [batch_size, num_value_heads, key_head_dim, value_head_dim],
                        &device,
                    )
                })
                .cast(DType::F32),
            GdnStateWriteMode::Static => cache
                .state
                .as_ref()
                .unwrap_or_else(|| {
                    panic!(
                        "Qwen3_5GdnAttention static recurrent step requires \
                         GdnStateCache::init_static(batch, device) to allocate state"
                    )
                })
                .clone()
                .cast(DType::F32),
        };
        let k_f32 = k.cast(DType::F32);
        let q_f32 = q.cast(DType::F32);
        let v_f32 = v.cast(DType::F32);
        let a_f32 = a.cast(DType::F32);
        let b_f32 = b.cast(DType::F32);

        let state_k = (prev_state.clone() * k_f32.clone().unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .reshape([batch_size, num_value_heads, value_head_dim]);
        let delta_v = v_f32 - state_k * a_f32.clone().unsqueeze_dim::<3>(2);
        let new_state = prev_state.clone()
            * a_f32.clone().unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3)
            + k_f32.clone().unsqueeze_dim::<4>(3)
                * delta_v.unsqueeze_dim::<4>(2)
                * b_f32.unsqueeze_dim::<3>(2).unsqueeze_dim::<4>(3);

        let o = (new_state.clone() * q_f32.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .reshape([batch_size, num_value_heads, value_head_dim]);
        cpu_t!(profile::CPU_GDN_STATE, _tr);
        let _to = std::time::Instant::now();
        match write_mode {
            GdnStateWriteMode::Functional => cache.set_state(new_state),
            GdnStateWriteMode::Static => {
                drop(prev_state);
                cache.set_state_static(new_state);
            }
        }

        // qwen3_5_moe Qwen3_5MoeRMSNormGated: normalize FIRST, then PLAIN weight, THEN gate — the source
        // literally comments "# Norm before gate": out = normalize(o) · weight · silu(z). This is the
        // OPPOSITE of qwen3_next's gate-before-norm. weight is a plain absolute gamma (set_norm_plain at
        // load — NOT the (1+weight) regular RMSNorm). z is a separate in_proj (not conv'd).
        let norm_weight = self
            .norm
            .gamma
            .val()
            .cast(DType::F32)
            .reshape([1, 1, value_head_dim]);
        let o_var = (o.clone() * o.clone()).mean_dim(2);
        let o_norm = (o / (o_var + self.norm.epsilon).sqrt()) * norm_weight;
        let o_gated = o_norm * silu(z);
        let out = o_gated.reshape([batch_size, 1, value_dim_total]);
        let out = ql3(&self.out_proj_fp8, &self.out_proj, out, prec).reshape([
            batch_size,
            1,
            hidden_size,
        ]);
        cpu_t!(profile::CPU_GDN_OUT, _to);
        out.cast(out_dtype)
    }

    /// O(S) sequential prefill helper for the recurrent GDN state.
    pub fn forward_prefill_recurrent(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        let [batch, seq_len, hidden] = hidden_states.dims();
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let step = hidden_states
                .clone()
                .slice([0..batch, t..(t + 1), 0..hidden]);
            outs.push(self.forward_recurrent(step, cache, prec));
        }
        Tensor::cat(outs, 1)
    }

    /// O(S) sequential prefill helper for static recurrent GDN state.
    pub fn forward_prefill_recurrent_static(
        &self,
        hidden_states: Tensor<3>,
        cache: &mut GdnStateCache,
        prec: Precision,
    ) -> Tensor<3> {
        assert!(
            cache.is_static(),
            "Qwen3_5GdnAttention::forward_prefill_recurrent_static requires \
             GdnStateCache::init_static(batch, device) before prefill"
        );
        let [batch, seq_len, hidden] = hidden_states.dims();
        let mut outs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let step = hidden_states
                .clone()
                .slice([0..batch, t..(t + 1), 0..hidden]);
            outs.push(self.step_recurrent_static(step, cache, prec));
        }
        Tensor::cat(outs, 1)
    }
}

impl Qwen3_5FullAttention {
    pub fn forward(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        prec: Precision,
    ) -> Tensor<3> {
        const PARTIAL_ROTARY_FACTOR: f64 = 0.25;
        const ROPE_THETA: f64 = 10_000_000.0;

        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Residual-stream dtype (bf16 on the real model): linear3 outputs F32, so cast the attn output
        // back to this at the end so `residual + attn_out` matches (same pattern as GDN + MoE outputs).
        let out_dtype = hidden_states.dtype();
        let query_and_gate = ql3(&self.q_proj_fp8, &self.q_proj, hidden_states.clone(), prec);
        let key = ql3(&self.k_proj_fp8, &self.k_proj, hidden_states.clone(), prec);
        let value = ql3(&self.v_proj_fp8, &self.v_proj, hidden_states, prec);

        let q_proj_dim = query_and_gate.dims()[2];
        let query_dim = q_proj_dim / 2;
        let head_dim = self.q_norm.gamma.val().dims()[0];
        let num_heads = query_dim / head_dim;
        let num_kv_heads = key.dims()[2] / head_dim;
        let rotary_dim = (head_dim as f64 * PARTIAL_ROTARY_FACTOR) as usize;

        // attn_output_gate (HF-verified, L1.2-fix): q_proj output is viewed as
        // [B, S, num_heads, 2*head_dim] and chunked on the LAST dim into [query; gate] PER HEAD
        // (interleaved), NOT a flat block split. A block split scrambles query/gate across heads
        // (shape-correct, value-wrong). Reshape FIRST, then slice the last dim.
        let qg = query_and_gate.reshape([batch_size, seq_len, num_heads, 2 * head_dim]);
        let query = qg
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..num_heads, 0..head_dim]);
        let output_gate = qg.slice([
            0..batch_size,
            0..seq_len,
            0..num_heads,
            head_dim..(2 * head_dim),
        ]);
        let key = key.reshape([batch_size, seq_len, num_kv_heads, head_dim]);
        let value = value.reshape([batch_size, seq_len, num_kv_heads, head_dim]);

        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        let (cos, sin) = compute_rope_embeddings(position_ids, rotary_dim, ROPE_THETA, &device);
        let (query, key) = apply_rope_partial(query, key, cos, sin, rotary_dim);

        let n_rep = num_heads / num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        let row_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let col_idx: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let rows = Tensor::<1>::from_floats(row_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(1)
            .repeat(&[1, seq_len]);
        let cols = Tensor::<1>::from_floats(col_idx.as_slice(), &device)
            .unsqueeze_dim::<2>(0)
            .repeat(&[seq_len, 1]);
        let causal_mask = rows.lower(cols).unsqueeze_dims::<4>(&[0, 1]);

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            Some(causal_mask),
            None,
            AttentionModuleOptions::default(),
        );

        let attn_output = attn_output.movedim(1, 2) * sigmoid(output_gate);
        let attn_output = attn_output.reshape([
            batch_size as i64,
            seq_len as i64,
            (num_heads * head_dim) as i64,
        ]);

        ql3(&self.o_proj_fp8, &self.o_proj, attn_output, prec).cast(out_dtype)
    }

    pub fn forward_with_cache(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        self.forward_with_cache_impl(hidden_states, position_ids, cache, prec, true)
    }

    /// CUDA-graph-capturable Qwen3.5 full-attention decode step.
    ///
    /// This is the static sibling of [`Self::forward_with_cache`] for a single decode token:
    /// K/V are written by device `pos` into the fixed-capacity cache, attention reads the full
    /// `[B,T_max,Hkv,D]` buffer with `idx > pos` masked, and RoPE uses a pre-hoisted partial-RoPE
    /// frequency table (`rope_freqs(rotary_dim, 1e7)`) instead of per-step host staging.
    pub fn forward_with_cache_static_pre(
        &self,
        hidden_states: Tensor<3>,
        pos: Tensor<1, Int>,
        cache: &mut KVCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
    ) -> Tensor<3> {
        const PARTIAL_ROTARY_FACTOR: f64 = 0.25;

        let [batch_size, seq_len, _] = hidden_states.dims();
        debug_assert_eq!(
            seq_len, 1,
            "Qwen3_5FullAttention static decode expects [B, 1, H]"
        );

        let out_dtype = hidden_states.dtype();
        let query_and_gate = ql3(&self.q_proj_fp8, &self.q_proj, hidden_states.clone(), prec);
        let key = ql3(&self.k_proj_fp8, &self.k_proj, hidden_states.clone(), prec);
        let value = ql3(&self.v_proj_fp8, &self.v_proj, hidden_states, prec);

        let q_proj_dim = query_and_gate.dims()[2];
        let query_dim = q_proj_dim / 2;
        let head_dim = self.q_norm.gamma.val().dims()[0];
        let num_heads = query_dim / head_dim;
        let num_kv_heads = key.dims()[2] / head_dim;
        let rotary_dim = (head_dim as f64 * PARTIAL_ROTARY_FACTOR) as usize;
        // The pre-hoisted RoPE table must be rope_freqs(rotary_dim, ..) (len == rotary_dim/2), NOT
        // rope_freqs(head_dim, ..): a head_dim-sized table silently mis-rotates the partial slice.
        debug_assert_eq!(
            freqs.dims()[0],
            rotary_dim / 2,
            "forward_with_cache_static_pre: freqs table length {} != rotary_dim/2 {} — pass \
             rope_freqs(rotary_dim, theta), not rope_freqs(head_dim, theta)",
            freqs.dims()[0],
            rotary_dim / 2
        );

        let qg = query_and_gate.reshape([batch_size, seq_len, num_heads, 2 * head_dim]);
        let query = qg
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..num_heads, 0..head_dim]);
        let output_gate = qg.slice([
            0..batch_size,
            0..seq_len,
            0..num_heads,
            head_dim..(2 * head_dim),
        ]);
        let key = key.reshape([batch_size, seq_len, num_kv_heads, head_dim]);
        let value = value.reshape([batch_size, seq_len, num_kv_heads, head_dim]);

        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        let position_ids = pos.clone().reshape([1, 1]).repeat(&[batch_size, 1]);
        let (cos, sin) = compute_rope_embeddings_pre(position_ids, freqs.clone());
        let (query, key) = apply_rope_partial(query, key, cos, sin, rotary_dim);

        let (key, value) = cache.update_static(&pos, key, value);
        let t_max = key.dims()[1];

        let n_rep = num_heads / num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        let idx = arange_tmax.clone().reshape([1, 1, 1, t_max]);
        let pos_mask = idx.greater(pos.reshape([1, 1, 1, 1]));

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            Some(pos_mask),
            None,
            AttentionModuleOptions::default(),
        );

        let attn_output = attn_output.movedim(1, 2) * sigmoid(output_gate);
        let attn_output = attn_output.reshape([
            batch_size as i64,
            seq_len as i64,
            (num_heads * head_dim) as i64,
        ]);

        ql3(&self.o_proj_fp8, &self.o_proj, attn_output, prec).cast(out_dtype)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_with_cache_sdpa_reference(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
    ) -> Tensor<3> {
        self.forward_with_cache_impl(hidden_states, position_ids, cache, prec, false)
    }

    fn forward_with_cache_impl(
        &self,
        hidden_states: Tensor<3>,
        position_ids: Tensor<2, Int>,
        cache: &mut KVCache,
        prec: Precision,
        use_flash_decode: bool,
    ) -> Tensor<3> {
        const PARTIAL_ROTARY_FACTOR: f64 = 0.25;
        const ROPE_THETA: f64 = 10_000_000.0;

        let [batch_size, seq_len, _] = hidden_states.dims();
        let device = hidden_states.device();

        // Residual-stream dtype (bf16 on the real model): linear3 outputs F32, so cast the attn output
        // back to this at the end so `residual + attn_out` matches (same pattern as GDN + MoE outputs).
        let out_dtype = hidden_states.dtype();
        let query_and_gate = ql3(&self.q_proj_fp8, &self.q_proj, hidden_states.clone(), prec);
        let key = ql3(&self.k_proj_fp8, &self.k_proj, hidden_states.clone(), prec);
        let value = ql3(&self.v_proj_fp8, &self.v_proj, hidden_states, prec);

        let q_proj_dim = query_and_gate.dims()[2];
        let query_dim = q_proj_dim / 2;
        let head_dim = self.q_norm.gamma.val().dims()[0];
        let num_heads = query_dim / head_dim;
        let num_kv_heads = key.dims()[2] / head_dim;
        let rotary_dim = (head_dim as f64 * PARTIAL_ROTARY_FACTOR) as usize;

        let qg = query_and_gate.reshape([batch_size, seq_len, num_heads, 2 * head_dim]);
        let query = qg
            .clone()
            .slice([0..batch_size, 0..seq_len, 0..num_heads, 0..head_dim]);
        let output_gate = qg.slice([
            0..batch_size,
            0..seq_len,
            0..num_heads,
            head_dim..(2 * head_dim),
        ]);
        let key = key.reshape([batch_size, seq_len, num_kv_heads, head_dim]);
        let value = value.reshape([batch_size, seq_len, num_kv_heads, head_dim]);

        let query = self.q_norm.forward(query);
        let key = self.k_norm.forward(key);

        let (cos, sin) = compute_rope_embeddings(position_ids, rotary_dim, ROPE_THETA, &device);
        let (query, key) = apply_rope_partial(query, key, cos, sin, rotary_dim);

        let (key, value) = cache.update(key, value);
        let [_, total_seq, _, _] = key.dims();

        #[cfg(feature = "cubecl-gpu")]
        {
            // No minimum-context gate: flash_decode with n_splits=1 (see `flash_decode_n_splits`)
            // is 2 dispatches total and skips the GQA-repeat materialization, which beats the
            // unfused path's ~5 dispatches (repeat + QK matmul + softmax + PV matmul + mask) at
            // every context length, not just long ones.
            //
            // `head_dim % 32 == 0` is a hard kernel precondition, not a heuristic: the split kernel
            // partitions D strictly across the 32 lanes of a warp. Small test/toy configs (head_dim
            // 8) must fall through to the generic path below.
            if use_flash_decode && seq_len == 1 && head_dim % 32 == 0 {
                let q4 = query.movedim(1, 2).cast(DType::F32);
                let k4 = key.movedim(1, 2);
                let v4 = value.movedim(1, 2);
                let scale = (head_dim as f32).sqrt().recip();
                let attn = crate::flash_decode::flash_decode(
                    q4,
                    k4,
                    v4,
                    scale,
                    flash_decode_n_splits(total_seq),
                );
                let gated = attn.movedim(1, 2) * sigmoid(output_gate);
                return ql3(
                    &self.o_proj_fp8,
                    &self.o_proj,
                    gated.reshape([batch_size as i64, 1, (num_heads * head_dim) as i64]),
                    prec,
                )
                .cast(out_dtype);
            }
        }
        #[cfg(not(feature = "cubecl-gpu"))]
        let _ = use_flash_decode;

        let n_rep = num_heads / num_kv_heads;
        let (key, value) = if n_rep > 1 {
            (
                key.unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
                value
                    .unsqueeze_dim::<5>(3)
                    .repeat(&[1, 1, 1, n_rep, 1])
                    .flatten(2, 3),
            )
        } else {
            (key, value)
        };

        let mask = if seq_len > 1 {
            let q_offset = total_seq - seq_len;
            let row_idx: Vec<f32> = (0..seq_len).map(|i| (q_offset + i) as f32).collect();
            let col_idx: Vec<f32> = (0..total_seq).map(|i| i as f32).collect();
            let rows = Tensor::<1>::from_floats(row_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(1)
                .repeat(&[1, total_seq]);
            let cols = Tensor::<1>::from_floats(col_idx.as_slice(), &device)
                .unsqueeze_dim::<2>(0)
                .repeat(&[seq_len, 1]);
            Some(rows.lower(cols).unsqueeze_dims::<4>(&[0, 1]))
        } else {
            None
        };

        let attn_output = attention(
            query.movedim(1, 2),
            key.movedim(1, 2),
            value.movedim(1, 2),
            mask,
            None,
            AttentionModuleOptions::default(),
        );

        let attn_output = attn_output.movedim(1, 2) * sigmoid(output_gate);
        let attn_output = attn_output.reshape([
            batch_size as i64,
            seq_len as i64,
            (num_heads * head_dim) as i64,
        ]);

        ql3(&self.o_proj_fp8, &self.o_proj, attn_output, prec).cast(out_dtype)
    }
}

impl Qwen3_5Model {
    pub fn forward(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
    ) -> Tensor<3> {
        self.forward_prec(input_ids, position_ids, cache, Precision::F32)
    }

    pub fn forward_prec(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
    ) -> Tensor<3> {
        assert_eq!(
            self.layers.len(),
            cache.layers.len(),
            "Qwen3.5 hybrid cache layer count mismatch"
        );

        // Residual stream runs in F32 (the "F32 activations, bf16 matmul-compute" convention): the embed
        // weight is bf16, so cast its output to F32 here. With rank-1 params (norm gammas) also loaded F32,
        // every norm/residual/elementwise op is F32×F32 — no bf16/f32 DTypeMismatch anywhere in the stack.
        let mut hidden_states = self.embed_tokens.forward(input_ids).cast(DType::F32);
        if std::env::var("QWEN35_DEBUG_LAYERS").is_ok() {
            let [b, s, hsz] = hidden_states.dims();
            let h = hidden_states
                .clone()
                .slice([0..b, (s - 1)..s, 0..hsz])
                .reshape([hsz]);
            let amax = h
                .clone()
                .abs()
                .max()
                .into_data()
                .to_vec::<f32>()
                .expect("amax")[0];
            let norm = (h.clone() * h)
                .sum()
                .sqrt()
                .into_data()
                .to_vec::<f32>()
                .expect("norm")[0];
            eprintln!("[dbg] embed         abs_max={amax:.5} norm={norm:.4}");
        }
        for (idx, (layer, layer_cache)) in
            self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            hidden_states = match (layer, layer_cache) {
                (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(cache)) => {
                    layer.forward_decoder_recurrent(hidden_states, cache, prec)
                }
                (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(cache)) => layer
                    .forward_decoder_with_cache(hidden_states, position_ids.clone(), cache, prec),
                (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Full but model layer is Linear")
                }
                (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Linear but model layer is Full")
                }
            };
            if std::env::var("QWEN35_DEBUG_LAYERS").is_ok() {
                let ltype = match layer {
                    Qwen3_5DecoderLayer::Linear(_) => "GDN",
                    Qwen3_5DecoderLayer::Full(_) => "FULL",
                };
                // LAST-position stats to match the HF reference (hidden_states[0, -1]).
                let [b, s, hsz] = hidden_states.dims();
                let h = hidden_states
                    .clone()
                    .slice([0..b, (s - 1)..s, 0..hsz])
                    .reshape([hsz])
                    .cast(DType::F32);
                let amax = h
                    .clone()
                    .abs()
                    .max()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("amax")[0];
                let amean = h
                    .clone()
                    .abs()
                    .mean()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("amean")[0];
                let norm = (h.clone() * h)
                    .sum()
                    .sqrt()
                    .into_data()
                    .to_vec::<f32>()
                    .expect("norm")[0];
                eprintln!(
                    "[dbg] layer {idx:2} [{ltype:>4}] abs_mean={amean:.5} abs_max={amax:.5} norm={norm:.4} finite={}",
                    amax.is_finite()
                );
            }
        }
        self.norm.forward(hidden_states)
    }

    /// `docs/MEMORY_STREAMING_PLAN.md` streamed sibling of [`Self::forward_prec`]: identical
    /// embed/attention/norm computation, but every layer's MoE sublayer streams its routed experts
    /// through `pool` instead of reading fully-resident expert stacks. Requires the model to have been
    /// loaded via `Qwen3_5MoeForCausalLM::load_weights_sharded_resident_core`.
    pub fn forward_prec_streamed(
        &self,
        input_ids: Tensor<2, Int>,
        position_ids: Tensor<2, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
        pool: &mut ExpertSlotPool,
    ) -> Tensor<3> {
        assert_eq!(
            self.layers.len(),
            cache.layers.len(),
            "Qwen3.5 hybrid cache layer count mismatch"
        );
        let prof_device = input_ids.device();
        let mut hidden_states = timed(&profile::EMBED, &prof_device, || {
            self.embed_tokens.forward(input_ids).cast(DType::F32)
        });
        let cpu_time = std::env::var("QWEN35_CPU_TIME").ok().as_deref() == Some("1");
        let mut layer_ns: Vec<u64> = if cpu_time {
            vec![0; self.layers.len()]
        } else {
            Vec::new()
        };
        for (idx, (layer, layer_cache)) in
            self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            let t0 = if cpu_time {
                Some(std::time::Instant::now())
            } else {
                None
            };
            hidden_states = match (layer, layer_cache) {
                (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(cache)) => {
                    layer.forward_decoder_recurrent_streamed(hidden_states, cache, prec, pool, idx)
                }
                (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(cache)) => layer
                    .forward_decoder_with_cache_streamed(
                        hidden_states,
                        position_ids.clone(),
                        cache,
                        prec,
                        pool,
                        idx,
                    ),
                (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Full but model layer is Linear")
                }
                (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Linear but model layer is Full")
                }
            };
            if let Some(t0) = t0 {
                layer_ns[idx] += t0.elapsed().as_nanos() as u64;
            }
        }
        if cpu_time {
            // NOTE: these are CPU-side enqueue times (no device.sync), so they measure how long
            // the HOST spends building/dispatching work -- not GPU execution. Compare against
            // wall-clock decode time to see whether the host is the bottleneck.
            let total: u64 = layer_ns.iter().sum();
            let gdn: u64 = layer_ns
                .iter()
                .enumerate()
                .filter(|(i, _)| matches!(self.layers[*i], Qwen3_5DecoderLayer::Linear(_)))
                .map(|(_, ns)| *ns)
                .sum();
            let full: u64 = layer_ns
                .iter()
                .enumerate()
                .filter(|(i, _)| matches!(self.layers[*i], Qwen3_5DecoderLayer::Full(_)))
                .map(|(_, ns)| *ns)
                .sum();
            eprintln!(
                "[cpu-time] layers host total: {:.1} ms (GDN {:.1} ms, full-attn {:.1} ms)",
                total as f64 / 1e6,
                gdn as f64 / 1e6,
                full as f64 / 1e6
            );
        }
        timed(&profile::FINAL_NORM, &prof_device, || {
            self.norm.forward(hidden_states)
        })
    }

    /// CUDA-graph-capturable single-token decode tower (spec §4): one `[B,1]` token in, final-norm
    /// hidden `[B,1,H]` out. GDN layers use the T2 static recurrent step, full-attn layers use the T3
    /// static KV step, and every layer's MoE goes through the static
    /// [`Qwen3_5SharedMoeBlock::forward_static`]. Norms/residuals are identical to
    /// [`Self::forward_prec`]. PROHIBITED here (vs the eager path): D2H/H2D staging, env reads (no
    /// `QWEN35_DEBUG_LAYERS` dumps), `set_state`, host `pos`, shape changes.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_decode_static_pre(
        &self,
        input_ids: Tensor<2, Int>,
        pos: Tensor<1, Int>,
        cache: &mut Qwen3_5HybridCache,
        prec: Precision,
        freqs: &Tensor<1>,
        arange_tmax: &Tensor<1, Int>,
    ) -> Tensor<3> {
        assert_eq!(
            self.layers.len(),
            cache.layers.len(),
            "Qwen3.5 hybrid cache layer count mismatch"
        );
        // Residual stream in F32 (the "F32 activations, bf16 matmul-compute" convention): cast the bf16
        // embed output up front so every norm/residual/elementwise op stays F32×F32.
        let mut hidden_states = self.embed_tokens.forward(input_ids).cast(DType::F32);
        for (idx, (layer, layer_cache)) in
            self.layers.iter().zip(cache.layers.iter_mut()).enumerate()
        {
            hidden_states = match (layer, layer_cache) {
                (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(cache)) => {
                    layer.forward_decoder_recurrent_static(hidden_states, cache, prec)
                }
                (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(cache)) => layer
                    .forward_decoder_with_cache_static_pre(
                        hidden_states,
                        pos.clone(),
                        cache,
                        prec,
                        freqs,
                        arange_tmax,
                    ),
                (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Full but model layer is Linear")
                }
                (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                    panic!("Qwen3.5 hybrid cache layer {idx} is Linear but model layer is Full")
                }
            };
        }
        self.norm.forward(hidden_states)
    }

    /// Build-time preflight for [`Self::forward_decode_static_pre`]: assert every GDN layer's state
    /// cache was `init_static`'d, every full-attn KV cache is static-capacity, and every layer's MoE
    /// passes [`Qwen3_5SharedMoeBlock::preflight_static`] — so the driver fails at build time instead
    /// of mid-capture. `Err(reason)` names the offending layer. `tokens` is the per-step token count
    /// (1 at B=1 decode).
    pub fn preflight_static(
        &self,
        cache: &Qwen3_5HybridCache,
        tokens: usize,
    ) -> Result<(), String> {
        if self.layers.len() != cache.layers.len() {
            return Err(format!(
                "hybrid cache layer count {} != model layers {}",
                cache.layers.len(),
                self.layers.len()
            ));
        }
        for (idx, (layer, layer_cache)) in self.layers.iter().zip(cache.layers.iter()).enumerate() {
            match (layer, layer_cache) {
                (Qwen3_5DecoderLayer::Linear(layer), Qwen3_5HybridLayerCache::Linear(c)) => {
                    if !c.is_static() {
                        return Err(format!(
                            "layer {idx} (GDN) state cache is not init_static'd; call \
                             init_static_caches before capture"
                        ));
                    }
                    layer
                        .mlp
                        .preflight_static(tokens)
                        .map_err(|e| format!("layer {idx} (GDN) MoE preflight failed: {e}"))?;
                }
                (Qwen3_5DecoderLayer::Full(layer), Qwen3_5HybridLayerCache::Full(c)) => {
                    if !c.is_static() {
                        return Err(format!(
                            "layer {idx} (full-attn) KV cache is not static-capacity; build the cache \
                             with new_cache_with_capacity"
                        ));
                    }
                    layer.mlp.preflight_static(tokens).map_err(|e| {
                        format!("layer {idx} (full-attn) MoE preflight failed: {e}")
                    })?;
                }
                (Qwen3_5DecoderLayer::Linear(_), Qwen3_5HybridLayerCache::Full(_)) => {
                    return Err(format!("layer {idx} model type Linear but cache is Full"));
                }
                (Qwen3_5DecoderLayer::Full(_), Qwen3_5HybridLayerCache::Linear(_)) => {
                    return Err(format!("layer {idx} model type Full but cache is Linear"));
                }
            }
        }
        Ok(())
    }

    /// Allocate the capture-stable GDN state buffers (`init_static`) for every linear-attention layer
    /// in `cache` that is not already static. Full-attn KV caches are made static by
    /// [`Self::new_cache_with_capacity`]; this fills the GDN gap T2 left at the model level. Idempotent.
    pub fn init_static_caches(&self, cache: &mut Qwen3_5HybridCache, batch: usize) {
        let device = self.device();
        for layer in cache.layers.iter_mut() {
            if let Qwen3_5HybridLayerCache::Linear(gdn) = layer {
                if !gdn.is_static() {
                    gdn.init_static(batch, &device);
                }
            }
        }
    }

    pub fn new_cache(&self) -> Qwen3_5HybridCache {
        let cfg = &self.config;
        Qwen3_5HybridCache::new(
            &cfg.layer_types,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.linear_qkv_dim(),
            cfg.linear_conv_kernel_dim,
        )
    }

    pub fn new_cache_with_capacity(&self, capacity: usize) -> Qwen3_5HybridCache {
        let cfg = &self.config;
        Qwen3_5HybridCache::with_capacity(
            &cfg.layer_types,
            cfg.linear_num_value_heads,
            cfg.linear_key_head_dim,
            cfg.linear_value_head_dim,
            cfg.linear_qkv_dim(),
            cfg.linear_conv_kernel_dim,
            capacity,
        )
    }

    pub(crate) fn device(&self) -> Device {
        self.embed_tokens.weight.lazy_device()
    }
}

fn lazy_linear(d_input: usize, d_output: usize, device: &Device) -> Linear {
    Linear {
        weight: lazy_param2([d_input, d_output], device),
        bias: None,
    }
}

fn lazy_embedding(n_embedding: usize, d_model: usize, device: &Device) -> Embedding {
    Embedding {
        weight: lazy_param2([n_embedding, d_model], device),
    }
}

fn lazy_rms_norm(d_model: usize, epsilon: f64, device: &Device) -> RmsNorm {
    RmsNorm {
        gamma: lazy_param1([d_model], device),
        epsilon,
    }
}

fn lazy_param1(shape: [usize; 1], device: &Device) -> Param<Tensor<1>> {
    Param::uninitialized(
        ParamId::new(),
        move |dev: &Device, _req_grad: bool| {
            Tensor::<1>::random(shape, Distribution::Normal(0.0, 0.02), dev)
        },
        device.clone(),
        false,
        Shape::from(shape),
    )
}

fn lazy_param2(shape: [usize; 2], device: &Device) -> Param<Tensor<2>> {
    Param::uninitialized(
        ParamId::new(),
        move |dev: &Device, _req_grad: bool| {
            Tensor::<2>::random(shape, Distribution::Normal(0.0, 0.02), dev)
        },
        device.clone(),
        false,
        Shape::from(shape),
    )
}

fn lazy_param3(shape: [usize; 3], device: &Device) -> Param<Tensor<3>> {
    Param::uninitialized(
        ParamId::new(),
        move |dev: &Device, _req_grad: bool| {
            Tensor::<3>::random(shape, Distribution::Normal(0.0, 0.02), dev)
        },
        device.clone(),
        false,
        Shape::from(shape),
    )
}

fn extract_object<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let start = json.find(&marker)?;
    let after_key = &json[start + marker.len()..];
    let colon = after_key.find(':')?;
    let after_colon = &after_key[colon + 1..];
    let brace_rel = after_colon.find('{')?;
    let brace_abs = start + marker.len() + colon + 1 + brace_rel;
    let bytes = json.as_bytes();
    let mut depth = 0usize;
    let mut in_str = false;
    let mut escaped = false;
    for i in brace_abs..bytes.len() {
        let b = bytes[i];
        if in_str {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_str = false;
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&json[brace_abs..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

fn get_value_slice<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let marker = format!("\"{key}\"");
    let start = json.find(&marker)?;
    let after_key = &json[start + marker.len()..];
    let colon = after_key.find(':')?;
    let value = after_key[colon + 1..].trim_start();
    let bytes = value.as_bytes();
    if bytes.first() == Some(&b'[') {
        let mut depth = 0usize;
        let mut in_str = false;
        let mut escaped = false;
        for i in 0..bytes.len() {
            let b = bytes[i];
            if in_str {
                if escaped {
                    escaped = false;
                } else if b == b'\\' {
                    escaped = true;
                } else if b == b'"' {
                    in_str = false;
                }
                continue;
            }
            match b {
                b'"' => in_str = true,
                b'[' => depth += 1,
                b']' => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(&value[..=i]);
                    }
                }
                _ => {}
            }
        }
        return None;
    }
    let end = value.find([',', '}']).unwrap_or(value.len());
    Some(value[..end].trim())
}

fn get_usize(json: &str, key: &str) -> Option<usize> {
    get_value_slice(json, key)?.parse::<usize>().ok()
}

fn get_f64(json: &str, key: &str) -> Option<f64> {
    get_value_slice(json, key)?.parse::<f64>().ok()
}

fn get_bool(json: &str, key: &str) -> Option<bool> {
    get_value_slice(json, key)?.parse::<bool>().ok()
}

fn get_usize_array(json: &str, key: &str) -> Option<Vec<usize>> {
    let slice = get_value_slice(json, key)?;
    if slice == "null" {
        return None;
    }
    let inner = slice.strip_prefix('[')?.strip_suffix(']')?;
    Some(
        inner
            .split(',')
            .filter_map(|x| {
                let t = x.trim();
                if t.is_empty() {
                    None
                } else {
                    t.parse::<usize>().ok()
                }
            })
            .collect(),
    )
}

fn get_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let slice = get_value_slice(json, key)?;
    if slice == "null" {
        return None;
    }
    let inner = slice.strip_prefix('[')?.strip_suffix(']')?;
    let mut out = Vec::new();
    for part in inner.split(',') {
        let item = part.trim().strip_prefix('"')?.strip_suffix('"')?;
        out.push(item.to_string());
    }
    Some(out)
}

pub fn parse_weight_map(index_json: &str) -> Result<Vec<(String, String)>, String> {
    let weight_map =
        extract_object(index_json, "weight_map").ok_or("index JSON is missing weight_map")?;
    let bytes = weight_map.as_bytes();
    let mut i = 1usize;
    let mut out = Vec::new();
    loop {
        skip_ws_and_commas(bytes, &mut i);
        if i >= bytes.len() || bytes[i] == b'}' {
            break;
        }
        let key = parse_json_string(bytes, &mut i)?;
        skip_ws(bytes, &mut i);
        if bytes.get(i) != Some(&b':') {
            return Err("malformed weight_map: expected ':'".to_string());
        }
        i += 1;
        skip_ws(bytes, &mut i);
        let value = parse_json_string(bytes, &mut i)?;
        out.push((key, value));
    }
    Ok(out)
}

fn skip_ws_and_commas(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && (bytes[*i].is_ascii_whitespace() || bytes[*i] == b',') {
        *i += 1;
    }
}

fn skip_ws(bytes: &[u8], i: &mut usize) {
    while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
        *i += 1;
    }
}

fn parse_json_string(bytes: &[u8], i: &mut usize) -> Result<String, String> {
    if bytes.get(*i) != Some(&b'"') {
        return Err("malformed JSON: expected string".to_string());
    }
    *i += 1;
    let mut out = String::new();
    while *i < bytes.len() {
        let b = bytes[*i];
        *i += 1;
        match b {
            b'"' => return Ok(out),
            b'\\' => {
                let escaped = *bytes.get(*i).ok_or("malformed JSON escape")?;
                *i += 1;
                out.push(escaped as char);
            }
            _ => out.push(b as char),
        }
    }
    Err("unterminated JSON string".to_string())
}

#[cfg(test)]
mod nvfp4_sidecar_tests {
    use super::*;
    use burn::prelude::Device;

    use crate::nvfp4::{
        dequant_nvfp4, dequant_nvfp4_outmajor, quantize_nvfp4, repack_kmajor_to_outmajor,
    };

    fn synth(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let mut z = seed ^ (idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn concat_gate_up_rows(gate: &[f32], up: &[f32], h: usize, i: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(h * i * 2);
        for kk in 0..h {
            out.extend_from_slice(&gate[kk * i..(kk + 1) * i]);
            out.extend_from_slice(&up[kk * i..(kk + 1) * i]);
        }
        out
    }

    #[test]
    fn expert_nvfp4_from_expert_parts_shapes_and_gate_up_half_scales() {
        let device = Device::default();
        let (e, h, i) = (2usize, 32usize, 16usize);
        let mut parts = Vec::new();
        let mut expected_e1 = Vec::new();
        let mut swapped_e1 = Vec::new();

        for expert in 0..e {
            let gate = synth(0xA001 + expert as u64, h * i, 0.015 * (expert + 1) as f32);
            let up = synth(0xB001 + expert as u64, h * i, 0.075 * (expert + 1) as f32);
            let down = synth(0xD001 + expert as u64, i * h, 0.025 * (expert + 1) as f32);

            let (q_gate, bs_gate, g_gate) = quantize_nvfp4(&gate, h, i);
            let (q_up, bs_up, g_up) = quantize_nvfp4(&up, h, i);
            assert_ne!(
                g_gate.to_bits(),
                g_up.to_bits(),
                "test setup needs different gate/up global scales"
            );

            let mut q_gu_kmajor = q_gate.clone();
            q_gu_kmajor.extend_from_slice(&q_up);
            let mut bs_gu = bs_gate.clone();
            bs_gu.extend_from_slice(&bs_up);
            let qw_gu_outmajor = repack_kmajor_to_outmajor(&q_gu_kmajor, h, i * 2);

            let (q_dn, bs_dn, g_dn) = quantize_nvfp4(&down, i, h);
            let qw_dn_outmajor = repack_kmajor_to_outmajor(&q_dn, i, h);

            if expert == 1 {
                let gate_ref = dequant_nvfp4(&q_gate, &bs_gate, g_gate, h, i);
                let up_ref = dequant_nvfp4(&q_up, &bs_up, g_up, h, i);
                expected_e1 = concat_gate_up_rows(&gate_ref, &up_ref, h, i);
                let mut expanded = vec![g_gate; i];
                expanded.extend(std::iter::repeat_n(g_up, i));
                let mut swapped = vec![g_up; i];
                swapped.extend(std::iter::repeat_n(g_gate, i));
                let correct = dequant_nvfp4_outmajor(&qw_gu_outmajor, &bs_gu, &expanded, h, i * 2);
                swapped_e1 = dequant_nvfp4_outmajor(&qw_gu_outmajor, &bs_gu, &swapped, h, i * 2);
                assert!(
                    correct
                        .iter()
                        .zip(expected_e1.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "correct gate/up half scales must reconstruct the fused reference"
                );
            }

            parts.push(ExpertNvfp4Parts {
                qw_gu_outmajor,
                bs_gu,
                gscale_gu: [g_gate, g_up],
                qw_dn_outmajor,
                bs_dn,
                gscale_dn: g_dn,
            });
        }

        let sidecar = ExpertNvfp4::from_expert_parts(parts.clone(), h, i, &device);
        assert_eq!((sidecar.e, sidecar.h, sidecar.i), (e, h, i));
        assert_eq!(sidecar.qw_gu.dims(), [e, h, i]);
        assert_eq!(sidecar.bs_gu.dims(), [e, i * 2, h / 16]);
        assert_eq!(sidecar.gscale_gu.dims(), [e, 2]);
        assert_eq!(sidecar.qw_dn.dims(), [e, i, h / 2]);
        assert_eq!(sidecar.bs_dn.dims(), [e, h, i / 16]);
        assert_eq!(sidecar.gscale_dn.dims(), [e]);
        assert_eq!(sidecar.qw_gu.dtype(), DType::I8);
        assert_eq!(sidecar.bs_dn.dtype(), DType::I8);

        let gscale_gu = sidecar
            .gscale_gu
            .clone()
            .into_data()
            .to_vec::<f32>()
            .expect("read gscale_gu");
        assert_eq!(gscale_gu[2].to_bits(), parts[1].gscale_gu[0].to_bits());
        assert_eq!(gscale_gu[3].to_bits(), parts[1].gscale_gu[1].to_bits());
        assert!(
            swapped_e1
                .iter()
                .zip(expected_e1.iter())
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "swapping gate/up gscale halves must change the dequantized expert"
        );
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod full_attention_static_tests {
    use super::*;
    use burn::{
        module::{Param, ParamId},
        nn::{Linear, RmsNorm},
        prelude::Device,
        tensor::TensorData,
    };

    use crate::rope::rope_freqs;

    fn synth_value(seed: u64, idx: usize, scale: f32) -> f32 {
        let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
        (u * 2.0 - 1.0) * scale
    }

    fn synth_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|idx| synth_value(seed, idx, scale)).collect()
    }

    fn linear_from(
        device: &Device,
        seed: u64,
        in_dim: usize,
        out_dim: usize,
        scale: f32,
    ) -> Linear {
        Linear {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<2>::from_data(
                    TensorData::new(synth_vec(seed, in_dim * out_dim, scale), [in_dim, out_dim]),
                    device,
                ),
            ),
            bias: None,
        }
    }

    fn rms_ones(dim: usize, device: &Device) -> RmsNorm {
        RmsNorm {
            gamma: Param::initialized(
                ParamId::new(),
                Tensor::<1>::from_data(TensorData::new(vec![1.0f32; dim], [dim]), device),
            ),
            epsilon: 1e-6,
        }
    }

    fn test_attention(device: &Device) -> Qwen3_5FullAttention {
        let hidden = 32;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = 8;
        Qwen3_5FullAttention {
            q_proj: linear_from(device, 0xA001, hidden, num_heads * head_dim * 2, 0.045),
            q_proj_fp8: QuantSidecar(None),
            k_proj: linear_from(device, 0xA002, hidden, num_kv_heads * head_dim, 0.040),
            k_proj_fp8: QuantSidecar(None),
            v_proj: linear_from(device, 0xA003, hidden, num_kv_heads * head_dim, 0.040),
            v_proj_fp8: QuantSidecar(None),
            o_proj: linear_from(device, 0xA004, num_heads * head_dim, hidden, 0.035),
            o_proj_fp8: QuantSidecar(None),
            q_norm: rms_ones(head_dim, device),
            k_norm: rms_ones(head_dim, device),
        }
    }

    fn pos1(pos: usize, device: &Device) -> Tensor<1, Int> {
        Tensor::<1, Int>::from_data([pos as i64], device)
    }

    fn pos2(pos: usize, device: &Device) -> Tensor<2, Int> {
        pos1(pos, device).reshape([1, 1])
    }

    fn step(x: Tensor<3>, t: usize, hidden: usize) -> Tensor<3> {
        x.slice([0..1, t..(t + 1), 0..hidden])
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn argmax(xs: &[f32]) -> usize {
        xs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    #[test]
    fn qwen35_full_attention_static_pre_matches_eager_after_prefill() {
        let device = Default::default();
        let attn = test_attention(&device);
        let hidden = 32;
        let total_tokens = 5;
        let t_max = 16;
        let rotary_dim = 2;
        let x = Tensor::<3>::from_data(
            TensorData::new(
                synth_vec(0xBEEF, total_tokens * hidden, 0.30),
                [1, total_tokens, hidden],
            ),
            &device,
        );
        let freqs = rope_freqs(rotary_dim, 10_000_000.0, &device);
        let arange_tmax = Tensor::<1, Int>::arange(0..t_max as i64, &device);

        let mut eager_cache = KVCache::new();
        let mut static_cache = KVCache::with_capacity(t_max);

        for t in 0..total_tokens {
            let eager = vec3(attn.forward_with_cache(
                step(x.clone(), t, hidden),
                pos2(t, &device),
                &mut eager_cache,
                Precision::F32,
            ));
            let got = if t < 3 {
                vec3(attn.forward_with_cache(
                    step(x.clone(), t, hidden),
                    pos2(t, &device),
                    &mut static_cache,
                    Precision::F32,
                ))
            } else {
                vec3(attn.forward_with_cache_static_pre(
                    step(x.clone(), t, hidden),
                    pos1(t, &device),
                    &mut static_cache,
                    Precision::F32,
                    &freqs,
                    &arange_tmax,
                ))
            };
            let diff = max_abs_diff(&eager, &got);
            assert!(
                diff <= 1.0e-5,
                "token {t}: max_abs_diff {diff:.9} exceeds 1e-5"
            );
            assert_eq!(argmax(&eager), argmax(&got), "token {t}: argmax mismatch");
        }
    }

    #[test]
    fn qwen35_full_attention_static_pre_masks_poisoned_future_column() {
        let device = Default::default();
        let attn = test_attention(&device);
        let hidden = 32;
        let num_kv_heads = 2;
        let head_dim = 8;
        let t_max = 16;
        let rotary_dim = 2;
        let x = Tensor::<3>::from_data(
            TensorData::new(synth_vec(0xCAFE, 4 * hidden, 0.30), [1, 4, hidden]),
            &device,
        );
        let freqs = rope_freqs(rotary_dim, 10_000_000.0, &device);
        let arange_tmax = Tensor::<1, Int>::arange(0..t_max as i64, &device);

        let mut clean_cache = KVCache::with_capacity(t_max);
        let mut poisoned_cache = KVCache::with_capacity(t_max);
        for t in 0..3 {
            let _ = attn.forward_with_cache(
                step(x.clone(), t, hidden),
                pos2(t, &device),
                &mut clean_cache,
                Precision::F32,
            );
            let _ = attn.forward_with_cache(
                step(x.clone(), t, hidden),
                pos2(t, &device),
                &mut poisoned_cache,
                Precision::F32,
            );
        }

        let future = pos1(4, &device);
        let poison_k = Tensor::<4>::from_data(
            TensorData::new(
                vec![1_000.0f32; num_kv_heads * head_dim],
                [1, 1, num_kv_heads, head_dim],
            ),
            &device,
        );
        let poison_v = Tensor::<4>::from_data(
            TensorData::new(
                vec![-777.0f32; num_kv_heads * head_dim],
                [1, 1, num_kv_heads, head_dim],
            ),
            &device,
        );
        poisoned_cache.key = Some(poisoned_cache.key.take().unwrap().select_assign(
            1,
            future.clone(),
            poison_k,
            IndexingUpdateOp::Add,
        ));
        poisoned_cache.value = Some(poisoned_cache.value.take().unwrap().select_assign(
            1,
            future,
            poison_v,
            IndexingUpdateOp::Add,
        ));

        let clean = vec3(attn.forward_with_cache_static_pre(
            step(x.clone(), 3, hidden),
            pos1(3, &device),
            &mut clean_cache,
            Precision::F32,
            &freqs,
            &arange_tmax,
        ));
        let poisoned = vec3(attn.forward_with_cache_static_pre(
            step(x, 3, hidden),
            pos1(3, &device),
            &mut poisoned_cache,
            Precision::F32,
            &freqs,
            &arange_tmax,
        ));
        let diff = max_abs_diff(&clean, &poisoned);
        assert!(
            diff <= 1.0e-6,
            "poisoned future KV column changed output: max_abs_diff {diff:.9}"
        );
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod static_decode_tests {
    //! Model-level `forward_decode_static_pre` parity + preflight tests on NdArray.
    //!
    //! NdArray has no `Fused35MoeBackend`, so `Qwen3_5SharedMoeBlock::forward_static` runs the eager
    //! oracle (host-loop) expert path here; this exercises the full static assembly (embed → per-layer
    //! GDN T2 static step / full-attn T3 static step → final norm → lm_head) end-to-end against the
    //! eager decode. The fused device-routed MoE branch is covered by the CUDA G2 gate. The tiny config
    //! sets `top_k == num_experts` so tiny (≤1e-5) attention differences cannot flip expert selection
    //! (all experts are always selected; their per-expert weights vary continuously), keeping parity
    //! robust regardless of the lazily-initialized random weights.
    use super::*;
    use crate::rope::rope_freqs;
    use burn::prelude::Device;

    fn tiny_config() -> Qwen3_5MoeConfig {
        Qwen3_5MoeConfig {
            vocab_size: 48,
            hidden_size: 32,
            num_hidden_layers: 2,
            layer_types: vec![
                Qwen3_5LayerType::LinearAttention,
                Qwen3_5LayerType::FullAttention,
            ],
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            partial_rotary_factor: 0.25,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            mrope_section: [1, 1, 1],
            num_experts: 4,
            num_experts_per_tok: 4,
            norm_topk_prob: true,
            moe_intermediate_size: 16,
            shared_expert_intermediate_size: 16,
            linear_key_head_dim: 8,
            linear_num_key_heads: 2,
            linear_num_value_heads: 4,
            linear_value_head_dim: 8,
            linear_conv_kernel_dim: 4,
            mtp_num_hidden_layers: 0,
        }
    }

    fn int1(vals: &[i64], device: &Device) -> Tensor<1, Int> {
        Tensor::<1, Int>::from_data(vals, device)
    }

    fn vecf(t: Tensor<2>) -> Vec<f32> {
        t.cast(DType::F32).into_data().to_vec::<f32>().unwrap()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max)
    }

    fn argmax(xs: &[f32]) -> usize {
        xs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap()
    }

    #[test]
    fn qwen35_mtp_forward_draft_shape_finite_and_chains() {
        let device = Device::default();
        device.seed(20_260_702);
        let mut cfg = tiny_config();
        cfg.mtp_num_hidden_layers = 1;
        let vocab = cfg.vocab_size;
        let hidden_size = cfg.hidden_size;
        let model = cfg.init_causal_lm(&device);

        let mut mtp_cache = model.mtp_new_cache(8);
        let hidden =
            Tensor::<3>::random([1, 1, hidden_size], Distribution::Normal(0.0, 1.0), &device);
        let tok_next = int1(&[5], &device).reshape([1, 1]);
        let pos = int1(&[2], &device).reshape([1, 1]);
        let (logits, mtp_hidden) = model.mtp.forward_draft(
            tok_next,
            hidden,
            pos,
            &mut mtp_cache,
            &model.model.embed_tokens,
            &model.lm_head,
            Precision::F32,
        );
        assert_eq!(logits.dims(), [1, vocab]);
        assert_eq!(mtp_hidden.dims(), [1, 1, hidden_size]);
        let finite_logits: f32 = logits.clone().mul_scalar(0.0f32).sum().into_scalar();
        let finite_hidden: f32 = mtp_hidden.clone().mul_scalar(0.0f32).sum().into_scalar();
        assert_eq!(finite_logits, 0.0, "MTP logits contain non-finite values");
        assert_eq!(finite_hidden, 0.0, "MTP hidden contains non-finite values");

        let draft_tok = logits.argmax(1);
        let (logits_2, mtp_hidden_2) = model.mtp.forward_draft(
            draft_tok,
            mtp_hidden,
            int1(&[3], &device).reshape([1, 1]),
            &mut mtp_cache,
            &model.model.embed_tokens,
            &model.lm_head,
            Precision::F32,
        );
        assert_eq!(logits_2.dims(), [1, vocab]);
        assert_eq!(mtp_hidden_2.dims(), [1, 1, hidden_size]);
        let finite_logits_2: f32 = logits_2.mul_scalar(0.0f32).sum().into_scalar();
        let finite_hidden_2: f32 = mtp_hidden_2.mul_scalar(0.0f32).sum().into_scalar();
        assert_eq!(
            finite_logits_2, 0.0,
            "chained MTP logits contain non-finite values"
        );
        assert_eq!(
            finite_hidden_2, 0.0,
            "chained MTP hidden contains non-finite values"
        );
    }

    #[test]
    fn qwen35_static_decode_matches_eager_after_prefill() {
        let device = Device::default();
        device.seed(20_260_701);
        let cfg = tiny_config();
        let vocab = cfg.vocab_size;
        let model = cfg.init_causal_lm(&device);

        let t_max = 16;
        let prompt_len = 3usize;
        let new_tokens = 4usize;
        let ids: Vec<i64> = (0..(prompt_len + new_tokens))
            .map(|i| ((i * 7 + 3) % vocab) as i64)
            .collect();

        // Prefill both caches identically via the eager path. The static cache's GDN layers are
        // init_static'd so the later static decode writes their state in place.
        let mut eager_cache = model.model.new_cache_with_capacity(t_max);
        let mut static_cache = model.model.new_cache_with_capacity(t_max);
        model.init_static_caches(&mut static_cache, 1);

        let prompt_ids = int1(&ids[0..prompt_len], &device).reshape([1, prompt_len]);
        let prompt_pos =
            int1(&(0..prompt_len as i64).collect::<Vec<_>>(), &device).reshape([1, prompt_len]);
        let _ = model.forward_prec(
            prompt_ids.clone(),
            prompt_pos.clone(),
            &mut eager_cache,
            Precision::F32,
        );
        let _ = model.forward_prec(prompt_ids, prompt_pos, &mut static_cache, Precision::F32);

        // Static-decode hoisted tables.
        let rotary_dim = (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize;
        let freqs = rope_freqs(rotary_dim, cfg.rope_theta, &device);
        let arange_tmax = Tensor::<1, Int>::arange(0..t_max as i64, &device);

        for step in 0..new_tokens {
            let t = prompt_len + step;
            let tok = int1(&ids[t..t + 1], &device).reshape([1, 1]);

            let eager = vecf(
                model
                    .forward_prec(
                        tok.clone(),
                        int1(&[t as i64], &device).reshape([1, 1]),
                        &mut eager_cache,
                        Precision::F32,
                    )
                    .reshape([1, vocab]),
            );
            let got = vecf(model.forward_decode_static_pre(
                tok,
                int1(&[t as i64], &device),
                &mut static_cache,
                Precision::F32,
                &freqs,
                &arange_tmax,
            ));

            let diff = max_abs_diff(&eager, &got);
            assert!(
                diff <= 1.0e-5,
                "decode step {step} (pos {t}): logit max_abs_diff {diff:.9} exceeds 1e-5"
            );
            assert_eq!(
                argmax(&eager),
                argmax(&got),
                "decode step {step} (pos {t}): argmax mismatch"
            );
        }
    }

    #[test]
    fn qwen35_preflight_static_missing_init_errors_then_ok() {
        let device = Device::default();
        let model = tiny_config().init_causal_lm(&device);

        // A static-capacity cache whose GDN layers have NOT been init_static'd must fail preflight.
        let mut cache = model.model.new_cache_with_capacity(16);
        let err = model
            .preflight_static(&cache, 1)
            .expect_err("preflight must reject a GDN cache without init_static");
        assert!(
            err.contains("init_static"),
            "preflight error should name the missing init_static step, got: {err}"
        );

        // After init_static_caches the same cache passes (bf16 expert stacks are non-placeholder on
        // NdArray, and the oracle MoE preflight always accepts).
        model.init_static_caches(&mut cache, 1);
        model
            .preflight_static(&cache, 1)
            .expect("preflight must accept an init_static'd static-capacity cache");
    }

    #[test]
    fn qwen35_preflight_static_rejects_non_static_kv() {
        let device = Device::default();
        let model = tiny_config().init_causal_lm(&device);

        // A legacy (non-capacity) cache: GDN not static AND full-attn KV not static-capacity.
        let mut cache = model.model.new_cache();
        // init_static the GDN layers so the first failure surfaced is the full-attn KV one.
        model.init_static_caches(&mut cache, 1);
        let err = model
            .preflight_static(&cache, 1)
            .expect_err("preflight must reject a non-static KV cache");
        assert!(
            err.contains("static-capacity"),
            "preflight error should name the non-static KV cache, got: {err}"
        );
    }
}

/// NdArray coverage for the NVFP4 expert dispatch arm (`expert_forward`), the oracle host path that
/// runs on the real model for T>16 prefill (the fused device gather-GEMV is CUDA-only + T<=16-capped,
/// covered by the CUDA parity ladder). Mirrors how the fp8 arm is unit-covered: a block carrying an
/// NVFP4 sidecar + placeholder bf16 stacks must reproduce the bf16-stack oracle built from the SAME
/// dequantized weights (this catches gate/up swap + transpose/layout bugs), and preflight must accept
/// the NVFP4 sidecar as fused-capable.
#[cfg(all(test, not(feature = "cuda")))]
mod nvfp4_expert_oracle_tests {
    use super::*;
    use burn::{
        module::{Param, ParamId},
        nn::Linear,
        prelude::Device,
        tensor::TensorData,
    };

    use crate::nvfp4::{dequant_nvfp4_outmajor, quantize_nvfp4, repack_kmajor_to_outmajor};

    fn synth_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len)
            .map(|idx| {
                let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
                (u * 2.0 - 1.0) * scale
            })
            .collect()
    }

    fn linf(vals: Vec<f32>, in_dim: usize, out_dim: usize, device: &Device) -> Linear {
        Linear {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<2>::from_data(TensorData::new(vals, [in_dim, out_dim]), device),
            ),
            bias: None,
        }
    }

    fn dummy_shared(hidden: usize, inner: usize, device: &Device) -> Qwen3_5SharedExpert {
        // Unused by `expert_forward`; present only to satisfy the block type.
        Qwen3_5SharedExpert {
            gate_proj: linf(vec![0.0; hidden * inner], hidden, inner, device),
            gate_proj_fp8: QuantSidecar(None),
            up_proj: linf(vec![0.0; hidden * inner], hidden, inner, device),
            up_proj_fp8: QuantSidecar(None),
            down_proj: linf(vec![0.0; inner * hidden], inner, hidden, device),
            down_proj_fp8: QuantSidecar(None),
        }
    }

    fn block(
        experts: Qwen3_5FusedExperts,
        hidden: usize,
        inner: usize,
        num_experts: usize,
        device: &Device,
    ) -> Qwen3_5SharedMoeBlock {
        Qwen3_5SharedMoeBlock {
            gate: linf(vec![0.0; hidden * num_experts], hidden, num_experts, device),
            experts,
            shared_expert: dummy_shared(hidden, inner, device),
            shared_expert_gate: linf(vec![0.0; hidden], hidden, 1, device),
            num_experts_per_tok: (num_experts),
            norm_topk_prob: (true),
        }
    }

    fn placeholder_stack(device: &Device) -> Param<Tensor<3>> {
        Param::initialized(
            ParamId::new(),
            Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), device),
        )
    }

    fn transpose_kn_to_nk(w_kn: &[f32], k: usize, n: usize) -> Vec<f32> {
        // [K,N] row-major -> [N,K] row-major (the bf16 stack's [out,in] layout).
        let mut out = vec![0.0f32; n * k];
        for kk in 0..k {
            for nn in 0..n {
                out[nn * k + kk] = w_kn[kk * n + nn];
            }
        }
        out
    }

    #[test]
    fn nvfp4_expert_forward_matches_dequant_bf16_oracle() {
        let device = Device::default();
        let (num_experts, hidden, inner, m) = (2usize, 32usize, 16usize, 3usize);

        let mut parts = Vec::with_capacity(num_experts);
        let mut gate_up_stack = Vec::with_capacity(num_experts * inner * 2 * hidden);
        let mut down_stack = Vec::with_capacity(num_experts * hidden * inner);

        for e in 0..num_experts {
            // "True" weights in [K,N] row-major: gate/up has K=H, N=2I; down has K=I, N=H.
            // Quantize the gate [H,I] and up [H,I] halves SEPARATELY (distinct amax -> distinct
            // ModelOpt weight_scale_2), so the two tensor-wide gscales are bit-distinct. This is what
            // gives the test teeth against a gu_gscale half-order swap in `expert_forward`: if both
            // halves shared one gscale (the old joint quantize), swapping gscale_gu[0]<->[1] would be a
            // numeric no-op and slip through. The up half uses a larger amplitude to force a different
            // amax (the assert_ne below hard-guarantees the two scales never coincide).
            let gate_kn = synth_vec(0x6A00 + e as u64, hidden * inner, 0.30);
            let up_kn = synth_vec(0x7B00 + e as u64, hidden * inner, 1.10);
            let down_kn = synth_vec(0xD000 + e as u64, inner * hidden, 0.03);

            // Quantize each half -> codec [N,K/2]; the fused [2I,K/2] codec is the two halves stacked
            // along N (rows 0..I = gate @ g_gate, rows I..2I = up @ g_up), since NVFP4 quantizes every
            // output column independently. Same for the block scales [2I,K/16].
            let (qw_gate_km, bs_gate, g_gate) = quantize_nvfp4(&gate_kn, hidden, inner);
            let (qw_up_km, bs_up, g_up) = quantize_nvfp4(&up_kn, hidden, inner);
            assert_ne!(
                g_gate.to_bits(),
                g_up.to_bits(),
                "test setup: gate/up gscales must be bit-distinct so a gu_gscale half swap is observable"
            );

            let mut qw_gu_km = qw_gate_km;
            qw_gu_km.extend_from_slice(&qw_up_km);
            let mut bs_gu = bs_gate;
            bs_gu.extend_from_slice(&bs_up);
            let qw_gu_out = repack_kmajor_to_outmajor(&qw_gu_km, hidden, inner * 2);
            let (qw_dn_km, bs_dn, g_dn) = quantize_nvfp4(&down_kn, inner, hidden);
            let qw_dn_out = repack_kmajor_to_outmajor(&qw_dn_km, inner, hidden);

            // Oracle bf16 stacks store the SAME dequantized weights (transposed to [out,in]), so the
            // only thing under test is the arm's dequant/transpose/slice layout, not 4-bit accuracy.
            // The per-output-channel gscale vector is the ground truth: g_gate for the gate half, g_up
            // for the up half. `expert_forward` must rebuild exactly this from gscale_gu=[g_gate,g_up];
            // if it swapped the two halves the oracle here would disagree and max_diff would blow past
            // the 1e-4 tolerance.
            let mut gu_gscale = vec![g_gate; inner];
            gu_gscale.extend(std::iter::repeat_n(g_up, inner));
            let gate_up_deq_kn =
                dequant_nvfp4_outmajor(&qw_gu_out, &bs_gu, &gu_gscale, hidden, inner * 2);
            let down_deq_kn = dequant_nvfp4_outmajor(&qw_dn_out, &bs_dn, &[g_dn], inner, hidden);
            gate_up_stack.extend(transpose_kn_to_nk(&gate_up_deq_kn, hidden, inner * 2));
            down_stack.extend(transpose_kn_to_nk(&down_deq_kn, inner, hidden));

            parts.push(ExpertNvfp4Parts {
                qw_gu_outmajor: qw_gu_out,
                bs_gu,
                gscale_gu: [g_gate, g_up],
                qw_dn_outmajor: qw_dn_out,
                bs_dn,
                gscale_dn: g_dn,
            });
        }

        let sidecar = ExpertNvfp4::from_expert_parts(parts, hidden, inner, &device);
        let nvfp4_block = block(
            Qwen3_5FusedExperts {
                gate_up_proj: placeholder_stack(&device),
                down_proj: placeholder_stack(&device),
                fp8: ExpertQuantSidecar(None),
                nvfp4: ExpertNvfp4Sidecar(Some(sidecar)),
            },
            hidden,
            inner,
            num_experts,
            &device,
        );
        let bf16_block = block(
            Qwen3_5FusedExperts {
                gate_up_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(gate_up_stack, [num_experts, inner * 2, hidden]),
                        &device,
                    ),
                ),
                down_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(down_stack, [num_experts, hidden, inner]),
                        &device,
                    ),
                ),
                fp8: ExpertQuantSidecar(None),
                nvfp4: ExpertNvfp4Sidecar(None),
            },
            hidden,
            inner,
            num_experts,
            &device,
        );

        // preflight_static must accept the NVFP4-sidecar block as fused-capable (no-op-OK on NdArray;
        // the CUDA preflight's shape/dtype acceptance is type-checked under --features cuda).
        assert!(
            nvfp4_block.preflight_static(1).is_ok(),
            "NVFP4-sidecar block must preflight OK"
        );

        for e in 0..num_experts {
            // Large activations (and the 0.30/1.10 gate/up weight amplitudes above) are deliberate: the
            // expert computes silu(x@gate) * (x@up), and silu is ~homogeneous (linear) for small inputs,
            // so silu(a*g)*(u/a) ~= silu(g)*u — i.e. a gate<->up gscale swap (inflate gate by a, deflate
            // up by a) would CANCEL in the product and hide. Driving x@gate into silu's nonlinear region
            // breaks that cancellation, so the swap moves max|diff| far past 1e-4 (verified: ~0.38).
            let x = Tensor::<3>::from_data(
                TensorData::new(
                    synth_vec(0xAB00 + e as u64, m * hidden, 4.0),
                    [1, m, hidden],
                ),
                &device,
            );
            let got = nvfp4_block
                .expert_forward(e, x.clone(), Precision::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("nvfp4 expert output");
            let reference = bf16_block
                .expert_forward(e, x, Precision::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("bf16 oracle expert output");
            assert!(
                got.iter().all(|v| v.is_finite()),
                "expert {e}: NVFP4 arm produced non-finite output"
            );
            let max_diff = got
                .iter()
                .zip(reference.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff <= 1.0e-4,
                "expert {e}: NVFP4 arm vs dequant-bf16 oracle max|diff| {max_diff:.9} > 1e-4 \
                 (gate/up swap or transpose/layout bug)"
            );
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod fp8_expert_tests {
    use super::*;
    use burn::{
        module::{Ignored, Param, ParamId},
        nn::LinearConfig,
        prelude::Device,
        tensor::TensorData,
    };

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "cosine input lengths differ");
        let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += (x as f64) * (y as f64);
            na += (x as f64) * (x as f64);
            nb += (y as f64) * (y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "diff input lengths differ");
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "diff input lengths differ");
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).abs() / x.abs().max(1.0e-6))
            .fold(0.0f32, f32::max)
    }

    fn synth_value(seed: u64, idx: usize, scale: f32) -> f32 {
        let mut z = seed.wrapping_add((idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        let u = ((z ^ (z >> 31)) >> 40) as f32 / 16_777_216.0;
        (u * 2.0 - 1.0) * scale
    }

    fn synth_vec(seed: u64, len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|idx| synth_value(seed, idx, scale)).collect()
    }

    fn linear_from(
        device: &CudaDevice,
        seed: u64,
        in_dim: usize,
        out_dim: usize,
        scale: f32,
    ) -> Linear {
        Linear {
            weight: Param::initialized(
                ParamId::new(),
                Tensor::<2>::from_data(
                    TensorData::new(synth_vec(seed, in_dim * out_dim, scale), [in_dim, out_dim]),
                    device,
                ),
            ),
            bias: None,
        }
    }

    fn synthetic_fused35_block(device: &CudaDevice) -> Qwen3_5SharedMoeBlock {
        let (experts, hidden, inner, top_k) = (8usize, 32usize, 16usize, 2usize);
        let gate_up_host = synth_vec(0xA501, experts * inner * 2 * hidden, 0.015);
        let down_host = synth_vec(0xD052, experts * hidden * inner, 0.015);
        let zeros_hi = vec![0.0f32; hidden * inner];
        let zeros_ih = vec![0.0f32; inner * hidden];
        Qwen3_5SharedMoeBlock::<Cuda> {
            gate: linear_from(device, 0x6017, hidden, experts, 0.05),
            experts: Qwen3_5FusedExperts {
                gate_up_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(gate_up_host, [experts, inner * 2, hidden]),
                        device,
                    )
                    .cast(DType::BF16),
                ),
                down_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(down_host, [experts, hidden, inner]),
                        device,
                    )
                    .cast(DType::BF16),
                ),
                fp8: ExpertQuantSidecar(None),
                nvfp4: ExpertNvfp4Sidecar(None),
            },
            shared_expert: Qwen3_5SharedExpert {
                gate_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(
                            TensorData::new(zeros_hi.clone(), [hidden, inner]),
                            device,
                        ),
                    ),
                    bias: None,
                },
                gate_proj_fp8: QuantSidecar(None),
                up_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(TensorData::new(zeros_hi, [hidden, inner]), device),
                    ),
                    bias: None,
                },
                up_proj_fp8: QuantSidecar(None),
                down_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(TensorData::new(zeros_ih, [inner, hidden]), device),
                    ),
                    bias: None,
                },
                down_proj_fp8: QuantSidecar(None),
            },
            shared_expert_gate: linear_from(device, 0x6600, hidden, 1, 0.0),
            num_experts_per_tok: (top_k),
            norm_topk_prob: (true),
        }
    }

    #[test]
    fn fused35_bf16_forward_matches_host_loop_cuda() {
        let device = Device::cuda(0);
        let block = synthetic_fused35_block(&device);
        for tokens in [1usize, 4, 40] {
            let x = Tensor::<3>::from_data(
                TensorData::new(
                    synth_vec(0x1A2B + tokens as u64, tokens * 32, 0.25),
                    [1, tokens, 32],
                ),
                &device,
            );
            let reference = block
                .forward_impl(x.clone(), Precision::F32, false)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("host-loop output");
            let got = block
                .forward_impl(x, Precision::F32, true)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("fused output");
            assert!(
                got.iter().all(|v| v.is_finite()),
                "fused output contains NaN/Inf at T={tokens}"
            );
            let max_diff = max_abs_diff(&reference, &got);
            let cos = cosine(&reference, &got);
            assert!(
                max_diff < 1.0e-4,
                "T={tokens}: max|diff| {max_diff:.9} >= 1e-4"
            );
            assert!(cos > 0.9999, "T={tokens}: cosine {cos:.9} <= 0.9999");
        }
    }

    fn quantize_stack(
        src: &[f32],
        experts: usize,
        out_dim: usize,
        in_dim: usize,
        device: &CudaDevice,
    ) -> (Tensor<3, Int>, Tensor<2>) {
        let mut q_all = vec![0i8; experts * in_dim * out_dim];
        let mut s_all = vec![0.0f32; experts * out_dim];
        let mut transposed = vec![0.0f32; in_dim * out_dim];

        for expert in 0..experts {
            let src_base = expert * out_dim * in_dim;
            for out_idx in 0..out_dim {
                for in_idx in 0..in_dim {
                    transposed[in_idx * out_dim + out_idx] =
                        src[src_base + out_idx * in_dim + in_idx];
                }
            }
            let (q, s) = crate::w8a16::quantize_e4m3_per_channel(&transposed, in_dim, out_dim);
            let q_base = expert * in_dim * out_dim;
            for (dst, byte) in q_all[q_base..q_base + q.len()].iter_mut().zip(q.iter()) {
                *dst = *byte as i8;
            }
            let s_base = expert * out_dim;
            s_all[s_base..s_base + out_dim].copy_from_slice(&s);
        }

        let q = Tensor::<3, Int>::from_data(
            TensorData::new(q_all, ([experts, in_dim, out_dim])),
            (device, DType::I8),
        );
        let s = Tensor::<2>::from_data(TensorData::new(s_all, [experts, out_dim]), device);
        (q, s)
    }

    #[test]
    fn fp8_expert_forward_matches_bf16_path() {
        let device = Device::cuda(0);
        let (experts, hidden, inner) = (2usize, 32usize, 64usize);
        let gate_up_host: Vec<f32> = (0..experts * inner * 2 * hidden)
            .map(|idx| ((idx % 97) as f32 - 48.0) * 0.0005)
            .collect();
        let down_host: Vec<f32> = (0..experts * hidden * inner)
            .map(|idx| ((idx % 89) as f32 - 44.0) * 0.0005)
            .collect();

        let mut block = Qwen3_5SharedMoeBlock::<Cuda> {
            gate: LinearConfig::new(hidden, experts)
                .with_bias(false)
                .init(&device),
            experts: Qwen3_5FusedExperts {
                gate_up_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(gate_up_host.clone(), [experts, inner * 2, hidden]),
                        &device,
                    ),
                ),
                down_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(
                        TensorData::new(down_host.clone(), [experts, hidden, inner]),
                        &device,
                    ),
                ),
                fp8: ExpertQuantSidecar(None),
                nvfp4: ExpertNvfp4Sidecar(None),
            },
            shared_expert: Qwen3_5SharedExpert {
                gate_proj: LinearConfig::new(hidden, inner)
                    .with_bias(false)
                    .init(&device),
                gate_proj_fp8: QuantSidecar(None),
                up_proj: LinearConfig::new(hidden, inner)
                    .with_bias(false)
                    .init(&device),
                up_proj_fp8: QuantSidecar(None),
                down_proj: LinearConfig::new(inner, hidden)
                    .with_bias(false)
                    .init(&device),
                down_proj_fp8: QuantSidecar(None),
            },
            shared_expert_gate: LinearConfig::new(hidden, 1).with_bias(false).init(&device),
            num_experts_per_tok: (1),
            norm_topk_prob: (false),
        };

        let x = Tensor::<3>::random([1, 3, hidden], Distribution::Normal(0.0, 1.0), &device);
        let reference = block
            .expert_forward(0, x.clone(), Precision::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("reference expert output");

        let (q_gu, s_gu) = quantize_stack(&gate_up_host, experts, inner * 2, hidden, &device);
        let (q_dn, s_dn) = quantize_stack(&down_host, experts, hidden, inner, &device);
        block.experts.fp8 = ExpertQuantSidecar(Some(ExpertFp8 {
            q_gu,
            s_gu,
            q_dn,
            s_dn,
            e: experts,
            h: hidden,
            i: inner,
        }));

        let got = block
            .expert_forward(0, x, Precision::F32)
            .into_data()
            .to_vec::<f32>()
            .expect("fp8 expert output");
        assert!(
            got.iter().all(|v| v.is_finite()),
            "fp8 expert output contains NaN/Inf"
        );
        let cos = cosine(&reference, &got);
        // Gross-bug TRIPWIRE, not the accuracy gate. A layout/scale/transpose bug gives cosine <<0.9;
        // benign e4m3 error for a COMPOSED expert (two fp8 GEMMs + a silu nonlinearity) lands ~0.997
        // (vs the single-GEMM dense W8A16Linear's ~0.999). The real accuracy gate is the end-to-end
        // teacher-forced top1/KL (examples/qwen35_experts_fp8_gate.rs), and D6 already proved e4m3
        // experts near-lossless end-to-end (top1 97.9%, KL 0.0055). 0.99 catches gross bugs with margin.
        assert!(
            cos > 0.99,
            "fp8 expert cosine {cos:.9} <= 0.99 (gross layout/scale bug)"
        );
    }

    #[test]
    fn fused35_fp8_forward_matches_host_fp8_loop_cuda() {
        let device = Device::cuda(0);
        let (experts, hidden, inner, top_k) = (8usize, 32usize, 64usize, 2usize);
        let gate_up_host = synth_vec(0xF805, experts * inner * 2 * hidden, 0.010);
        let down_host = synth_vec(0xD8F7, experts * hidden * inner, 0.010);
        let zeros_hi = vec![0.0f32; hidden * inner];
        let zeros_ih = vec![0.0f32; inner * hidden];
        let (q_gu, s_gu) = quantize_stack(&gate_up_host, experts, inner * 2, hidden, &device);
        let (q_dn, s_dn) = quantize_stack(&down_host, experts, hidden, inner, &device);
        let block = Qwen3_5SharedMoeBlock::<Cuda> {
            gate: linear_from(&device, 0x91E7, hidden, experts, 0.05),
            experts: Qwen3_5FusedExperts {
                gate_up_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), &device),
                ),
                down_proj: Param::initialized(
                    ParamId::new(),
                    Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), &device),
                ),
                fp8: ExpertQuantSidecar(Some(ExpertFp8 {
                    q_gu,
                    s_gu,
                    q_dn,
                    s_dn,
                    e: experts,
                    h: hidden,
                    i: inner,
                })),
                nvfp4: ExpertNvfp4Sidecar(None),
            },
            shared_expert: Qwen3_5SharedExpert {
                gate_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(
                            TensorData::new(zeros_hi.clone(), [hidden, inner]),
                            &device,
                        ),
                    ),
                    bias: None,
                },
                gate_proj_fp8: QuantSidecar(None),
                up_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(TensorData::new(zeros_hi, [hidden, inner]), &device),
                    ),
                    bias: None,
                },
                up_proj_fp8: QuantSidecar(None),
                down_proj: Linear {
                    weight: Param::initialized(
                        ParamId::new(),
                        Tensor::<2>::from_data(TensorData::new(zeros_ih, [inner, hidden]), &device),
                    ),
                    bias: None,
                },
                down_proj_fp8: QuantSidecar(None),
            },
            shared_expert_gate: linear_from(&device, 0x6600, hidden, 1, 0.0),
            num_experts_per_tok: (top_k),
            norm_topk_prob: (true),
        };

        for tokens in [1usize, 4] {
            let x = Tensor::<3>::from_data(
                TensorData::new(
                    synth_vec(0xA8A8 + tokens as u64, tokens * hidden, 0.25),
                    [1, tokens, hidden],
                ),
                &device,
            );
            let reference = block
                .forward_impl(x.clone(), Precision::F32, false)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("host fp8 loop output");
            let got = block
                .forward_impl(x, Precision::F32, true)
                .cast(DType::F32)
                .into_data()
                .to_vec::<f32>()
                .expect("fused fp8 output");
            assert!(
                got.iter().all(|v| v.is_finite()),
                "fused fp8 output contains NaN/Inf at T={tokens}"
            );
            let max_abs = max_abs_diff(&reference, &got);
            let max_rel = max_rel_diff(&reference, &got);
            let cos = cosine(&reference, &got);
            assert!(cos > 0.9999, "T={tokens}: cosine {cos:.9} <= 0.9999");
            assert!(
                max_abs < 1.0e-3,
                "T={tokens}: max|diff| {max_abs:.9} >= 1e-3"
            );
            assert!(
                max_rel < 2.0e-2,
                "T={tokens}: max relative diff {max_rel:.9} >= 2e-2"
            );
        }
    }

    #[test]
    fn preflight_static_gates_fused_moe_preconditions() {
        let device = Device::cuda(0);

        // Non-placeholder bf16 expert stacks, no fp8 sidecar: a decode-sized step preflights OK.
        let ok_block = synthetic_fused35_block(&device);
        assert!(
            ok_block.preflight_static(1).is_ok(),
            "non-placeholder bf16 stacks should preflight-OK"
        );
        // Token count over the fused kernel bound is rejected.
        assert!(
            ok_block
                .preflight_static(QWEN35_FUSED_MOE_MAX_T + 1)
                .is_err(),
            "tokens over QWEN35_FUSED_MOE_MAX_T must be rejected"
        );

        // Placeholder [1,1,1] stacks with no fp8 sidecar: no fused path exists, so preflight must Err
        // (and forward_static would panic) rather than silently allow the capture-poison host fallback.
        let mut placeholder = synthetic_fused35_block(&device);
        placeholder.experts = Qwen3_5FusedExperts {
            gate_up_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), &device),
            ),
            down_proj: Param::initialized(
                ParamId::new(),
                Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), &device),
            ),
            fp8: ExpertQuantSidecar(None),
            nvfp4: ExpertNvfp4Sidecar(None),
        };
        let err = placeholder
            .preflight_static(1)
            .expect_err("placeholder stacks with no fp8 sidecar must be rejected");
        assert!(
            err.contains("placeholder"),
            "preflight error should name the placeholder stacks, got: {err}"
        );
    }
}
