//! Weight loading utilities for Qwen3 models.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use burn::{
    module::{Param, ParamId},
    nn::{Embedding, Linear, RmsNorm},
    prelude::Device,
    store::{BurnpackStore, ModuleStore, PyTorchToBurnAdapter, SafetensorsStore, TensorSnapshot},
    tensor::{DType, Tensor},
};
use rootcause::{Report, prelude::ResultExt};
use thiserror::Error;

#[cfg(feature = "cuda")]
use crate::nvfp4_linear::{Nvfp4Linear, QuantLinear};
#[cfg(feature = "cuda")]
use crate::nvidia_ckpt::quantize_dequantized_expert_to_fp8;
#[cfg(feature = "cuda")]
use crate::nvidia_ckpt::{
    QuantGroupKind, RawTensor, ShardReader, collect_quant_groups, dense_fp8_parts_from_reader,
    dense_fp8_role, expected_quantized_counts, expert_projection_parts_from_reader,
    fuse_expert_nvfp4_parts, nvfp4_linear_parts_from_reader, parse_expert_base, quant_group_keys,
    shard_index, shared_nvfp4_role,
};
#[cfg(feature = "cuda")]
use crate::qwen3_5::{ExpertNvfp4, ExpertNvfp4Sidecar, ExpertQuantSidecar, QuantSidecar};
use crate::qwen3_5::{
    Qwen3_5DecoderLayer, Qwen3_5FullAttention, Qwen3_5FullAttnLayer, Qwen3_5GdnAttention,
    Qwen3_5GdnLayer, Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM, Qwen3_5SharedMoeBlock,
    parse_weight_map,
};
#[cfg(feature = "cuda")]
use crate::w8a16_linear::W8A16Linear;
use crate::{Qwen3ForCausalLM, Qwen3Model, Qwen3MoeForCausalLM};
#[cfg(feature = "cuda")]
use burn::tensor::TensorData;

#[derive(Error, Debug)]
pub enum ModelLoadError {
    #[error("Error while loading weights")]
    LoadError,
    #[error("Unrecognised file extension")]
    UnknownExtension,
    #[error("Checkpoint incomplete: some model weights were not present in any shard")]
    IncompleteLoad,
}

#[derive(Clone, Debug)]
pub struct Qwen35LoadVerifyReport {
    pub weight_map_tensors: usize,
    pub mapped_tensors: usize,
    pub skipped_visual_tensors: usize,
    pub param_count: u128,
    pub skipped_visual_param_count: u128,
    pub missing: Vec<String>,
    pub orphan: Vec<String>,
    pub shape_mismatches: Vec<String>,
}

impl Qwen35LoadVerifyReport {
    pub fn pass(&self) -> bool {
        self.missing.is_empty() && self.orphan.is_empty() && self.shape_mismatches.is_empty()
    }
}

/// Create a SafetensorsStore with HuggingFace-to-Burn key remapping for Qwen3Model.
/// This removes the "model." prefix since Qwen3Model doesn't have that wrapper.
fn create_safetensors_store_base(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // Remove "model." prefix (for base model weights)
        .with_key_remapping(r"^model\.", "")
        // RmsNorm uses gamma in burn
        .with_key_remapping(r"\.weight$", ".gamma")
        // But Linear layers use weight, not gamma
        .with_key_remapping(r"_proj\.gamma$", "_proj.weight")
        .with_key_remapping(r"embed_tokens\.gamma$", "embed_tokens.weight")
}

/// Create a SafetensorsStore with HuggingFace-to-Burn key remapping for Qwen3ForCausalLM.
/// This keeps the "model." prefix since Qwen3ForCausalLM wraps Qwen3Model in a `model` field.
///
/// Qwen3-0.6B-Base ships TIED embeddings: the checkpoint contains `model.embed_tokens.weight`
/// and has NO `lm_head.weight`. We therefore do NOT remap anything onto `lm_head` (there is no
/// source tensor) and enable `allow_partial(true)` so the absent `lm_head.weight` is tolerated.
/// The caller MUST tie the head afterwards via `Qwen3ForCausalLM::tie_lm_head_to_embeddings`.
fn create_safetensors_store_causal_lm(path: PathBuf) -> SafetensorsStore {
    SafetensorsStore::from_file(path)
        .with_from_adapter(PyTorchToBurnAdapter::default())
        // RmsNorm uses gamma in burn
        .with_key_remapping(r"\.weight$", ".gamma")
        // But Linear layers use weight, not gamma
        .with_key_remapping(r"_proj\.gamma$", "_proj.weight")
        .with_key_remapping(r"embed_tokens\.gamma$", "embed_tokens.weight")
        // UNTIED models (Qwen3-14B etc.) ship a separate `lm_head.weight`; the `.weight$ -> .gamma`
        // rule above turned it into `lm_head.gamma`, so map it back to the Linear's `lm_head.weight`.
        .with_key_remapping(r"^lm_head\.gamma$", "lm_head.weight")
        // TIED models have NO `lm_head` source tensor; `allow_partial` tolerates the absent target.
        .allow_partial(true)
}

impl Qwen3Model {
    /// Load weights and return self (builder pattern).
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    /// Load weights from a file.
    ///
    /// Supports `.safetensors` (HuggingFace format) and `.bpk` (Burn format).
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = create_safetensors_store_base(path);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path).auto_extension(false);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}

impl Qwen3ForCausalLM {
    /// Load weights and return self (builder pattern).
    pub fn with_weights(
        mut self,
        path: impl Into<PathBuf>,
    ) -> Result<Self, Report<ModelLoadError>> {
        self.load_weights(path)?;
        Ok(self)
    }

    /// Load SHARDED safetensors (e.g. Qwen3-14B: `model-00001-of-00006.safetensors` ...).
    ///
    /// Applies every `model*.safetensors` in `dir` to the model in turn. `allow_partial` (set in
    /// the store) tolerates each shard covering only a subset of the weights, so sequential applies
    /// accumulate into a fully-loaded model. Use this for models too big for a single file.
    pub fn load_weights_sharded(
        &mut self,
        dir: impl Into<PathBuf>,
    ) -> Result<(), Report<ModelLoadError>> {
        let dir = dir.into();
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|_| Report::new(ModelLoadError::LoadError))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|x| x.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(Report::new(ModelLoadError::LoadError));
        }
        // Accumulate coverage across shards. allow_partial tolerates each shard covering only a
        // SUBSET, but a param present in NO shard would silently stay at random init and the model
        // would still "run" with garbage layers. So verify the UNION of applied weights and fail
        // loudly on any gap (corrupt/missing shard, remap typo, or an absent lm_head.weight on an
        // untied config). Cross-model review (Codex + Gemini): this was the P1 finding.
        let mut applied: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut all_params: std::collections::HashSet<String> = std::collections::HashSet::new();
        for shard in &shards {
            let mut weights = create_safetensors_store_causal_lm(shard.clone());
            let result = weights.apply_to(self).context(ModelLoadError::LoadError)?;
            for a in &result.applied {
                applied.insert(a.clone());
                all_params.insert(a.clone());
            }
            // `missing` lists module params absent from THIS shard; union(applied, missing) over a
            // shard is every model param, so accumulating both gives the full param set.
            for (m, _) in &result.missing {
                all_params.insert(m.clone());
            }
        }
        let mut gap: Vec<&String> = all_params.difference(&applied).collect();
        gap.sort();
        if !gap.is_empty() {
            eprintln!(
                "[load_weights_sharded] {} weight(s) present in NO shard (random-init): {gap:?}",
                gap.len()
            );
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }
        Ok(())
    }

    /// Load weights from a file.
    ///
    /// Supports `.safetensors` (HuggingFace format) and `.bpk` (Burn format).
    /// For HuggingFace format, expects the standard Qwen3ForCausalLM weight structure.
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());

        match extension.as_deref() {
            Some("safetensors") => {
                let mut weights = create_safetensors_store_causal_lm(path);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path).auto_extension(false);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
            }
            _ => {
                return Err(Report::new(ModelLoadError::UnknownExtension));
            }
        }

        Ok(())
    }
}

/// MoE store = the causal-LM store + a router-key restore. After the generic `\.weight$ -> .gamma`
/// rule, HF's router key `mlp.gate.weight` becomes `mlp.gate.gamma`, which the `_proj\.gamma$ ->
/// _proj.weight` restore does NOT catch (it is `gate`, not `gate_proj`). Map it back. ANCHORED to
/// `.mlp.gate.gamma$` so it can never touch the expert `gate_proj` keys (which the `_proj` rule has
/// already restored to `.weight` by the time this runs).
fn create_safetensors_store_moe(path: PathBuf) -> SafetensorsStore {
    create_safetensors_store_causal_lm(path)
        .with_key_remapping(r"\.mlp\.gate\.gamma$", ".mlp.gate.weight")
}

/// Parse a Burn-side (post-remap) per-expert weight key into `(layer, proj, expert)`, where `proj` is
/// `0=gate_proj, 1=up_proj, 2=down_proj`. Returns `None` for any non-expert key (router, attention,
/// norms, embed, lm_head), which the declarative store loads. Matches
/// `model.layers.{L}.mlp.experts.{E}.{gate|up|down}_proj.weight`.
fn parse_expert_key(key: &str) -> Option<(usize, usize, usize)> {
    let rest = key.strip_prefix("model.layers.")?;
    let (layer_s, rest) = rest.split_once(".mlp.experts.")?;
    let layer: usize = layer_s.parse().ok()?;
    let (expert_s, proj_s) = rest.split_once('.')?;
    let expert: usize = expert_s.parse().ok()?;
    let proj = match proj_s {
        "gate_proj.weight" => 0,
        "up_proj.weight" => 1,
        "down_proj.weight" => 2,
        _ => return None,
    };
    Some((layer, proj, expert))
}

impl Qwen3MoeForCausalLM {
    /// Load a single-file `.safetensors` (or `.bpk`) Qwen3-MoE checkpoint. (Qwen3-MoE checkpoints are
    /// normally sharded — see [`load_weights_sharded`](Self::load_weights_sharded).) The safetensors
    /// path uses the single-owner contiguous expert loader (see [`load_moe_safetensors`](Self::load_moe_safetensors)).
    pub fn load_weights(&mut self, path: impl Into<PathBuf>) -> Result<(), Report<ModelLoadError>> {
        let path = path.into();
        let extension = path.extension().map(|s| s.to_string_lossy().to_lowercase());
        match extension.as_deref() {
            Some("safetensors") => self.load_moe_safetensors(&[path]),
            Some("bpk") | None => {
                let mut weights = BurnpackStore::from_file(path).auto_extension(false);
                weights.apply_to(self).context(ModelLoadError::LoadError)?;
                Ok(())
            }
            _ => Err(Report::new(ModelLoadError::UnknownExtension)),
        }
    }

    /// Load SHARDED safetensors (Qwen3-30B-A3B ships many `model-*-of-*.safetensors` shards) through the
    /// single-owner contiguous expert loader.
    pub fn load_weights_sharded(
        &mut self,
        dir: impl Into<PathBuf>,
    ) -> Result<(), Report<ModelLoadError>> {
        let dir = dir.into();
        let mut shards: Vec<PathBuf> = std::fs::read_dir(&dir)
            .map_err(|_| Report::new(ModelLoadError::LoadError))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension().and_then(|x| x.to_str()) == Some("safetensors")
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("model"))
                        .unwrap_or(false)
            })
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(Report::new(ModelLoadError::LoadError));
        }
        self.load_moe_safetensors(&shards)
    }

    /// SINGLE-OWNER contiguous MoE load (fix (b), `docs/WAVE2_STATIC_DECODE.md`; the vLLM `FusedMoE`
    /// pattern). Per-expert weights are NEVER materialized into separate per-expert `Linear`s and never
    /// `cat`-duplicated — instead each `mlp.experts.{j}.{gate,up,down}_proj.weight` shard is read lazily
    /// (mmap-backed `TensorSnapshot`) and slot-written into the layer's contiguous `[E,..]` stack (the
    /// Burn analogue of vLLM's `param.data[expert_id].copy_(shard)`, with the PyTorch→Burn `[out,in]→
    /// [in,out]` Linear transpose). The non-expert params (router `mlp.gate`, attention, norms, embed,
    /// lm_head) load through the declarative [`SafetensorsStore`] exactly as before. So a 30B load holds
    /// ONE resident copy of the experts (~58 GB), not two — what previously OOM'd.
    ///
    /// COVERAGE (strict, like the dense loader): fails loud (`IncompleteLoad`) if any NON-expert model
    /// param is present in NO shard, OR if any `(layer, proj, expert)` slot is missing — so a corrupt
    /// shard, a remap regression, or a short expert set can never silently leave weights at random init.
    fn load_moe_safetensors(&mut self, shards: &[PathBuf]) -> Result<(), Report<ModelLoadError>> {
        let (n_layers, n_experts, _h, _i) = self.model.expert_layout();
        let device = self.model.device();

        // Pass over shards: (1) collect lazy per-expert snapshot handles (mmap-backed clones keep their
        // file mapped without materializing data), and (2) apply the declarative store for every
        // NON-expert param. The expert source tensors have no module target now → they land in
        // `result.unused` (NOT errors), so the apply succeeds; we fill the stacks ourselves below.
        let mut expert_snaps: HashMap<(usize, usize, usize), TensorSnapshot> = HashMap::new();
        let mut applied: HashSet<String> = HashSet::new();
        let mut all_params: HashSet<String> = HashSet::new();
        let mut weight_dtype: Option<DType> = None;

        for shard in shards {
            let mut store = create_safetensors_store_moe(shard.clone());
            {
                let snaps = store
                    .get_all_snapshots()
                    .context(ModelLoadError::LoadError)?;
                for (key, snap) in snaps.iter() {
                    if let Some(idx) = parse_expert_key(key) {
                        weight_dtype.get_or_insert(snap.dtype);
                        expert_snaps.insert(idx, snap.clone());
                    }
                }
            }
            let result = store.apply_to(self).context(ModelLoadError::LoadError)?;
            for a in &result.applied {
                applied.insert(a.clone());
                all_params.insert(a.clone());
            }
            for (m, _) in &result.missing {
                all_params.insert(m.clone());
            }
        }

        // NON-expert coverage. The `*_stack` params are visited+missing on EVERY shard (no snapshot
        // ever targets them — we fill them custom), so exclude them; any OTHER gap is a real hole.
        let mut gap: Vec<&String> = all_params
            .difference(&applied)
            .filter(|p| !p.ends_with("_stack"))
            .collect();
        gap.sort();
        if !gap.is_empty() {
            eprintln!(
                "[load_moe_safetensors] {} non-expert weight(s) present in NO shard (random-init): {gap:?}",
                gap.len()
            );
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }

        // EXPERT coverage: every (layer, proj, expert) slot must have a source tensor.
        let mut missing_experts: Vec<(usize, usize, usize)> = Vec::new();
        for l in 0..n_layers {
            for p in 0..3 {
                for e in 0..n_experts {
                    if !expert_snaps.contains_key(&(l, p, e)) {
                        missing_experts.push((l, p, e));
                    }
                }
            }
        }
        if !missing_experts.is_empty() {
            eprintln!(
                "[load_moe_safetensors] {} expert weight slot(s) present in NO shard (random-init), first: {:?}",
                missing_experts.len(),
                &missing_experts[..missing_experts.len().min(8)]
            );
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }
        let weight_dtype = match weight_dtype {
            Some(d) => d,
            None => {
                eprintln!("[load_moe_safetensors] no per-expert weights found in any shard");
                return Err(Report::new(ModelLoadError::IncompleteLoad));
            }
        };

        // Build each layer's three contiguous stacks slot-by-slot, ONE (layer, proj) at a time, so the
        // transient working set is one layer's experts (~hundreds of MB), never a full second copy.
        // PyTorch→Burn transpose: HF `[d_out, d_in]` → `transpose()` → Burn `[d_in, d_out]`
        // (gate/up `[I,H]→[H,I]`, down `[H,I]→[I,H]`), then `cat` over the new leading expert axis.
        for l in 0..n_layers {
            let mut built: Vec<Tensor<3>> = Vec::with_capacity(3);
            for p in 0..3 {
                let mut slabs: Vec<Tensor<3>> = Vec::with_capacity(n_experts);
                for e in 0..n_experts {
                    let snap = &expert_snaps[&(l, p, e)];
                    let data = snap
                        .to_data()
                        .map_err(|_| Report::new(ModelLoadError::LoadError))?;
                    let raw = Tensor::<2>::from_data(data, (&device, weight_dtype)); // [out,in]
                    let slab = raw.transpose(); // [in,out]
                    let [d0, d1] = slab.dims();
                    slabs.push(slab.reshape([1, d0, d1]));
                }
                built.push(Tensor::cat(slabs, 0)); // [E, d0, d1]
            }
            let down = built.pop().unwrap();
            let up = built.pop().unwrap();
            let gate = built.pop().unwrap();
            self.model.set_layer_expert_stacks(l, gate, up, down);
        }
        Ok(())
    }
}

impl Qwen3_5MoeForCausalLM {
    pub fn verify_weights_sharded(
        config: &Qwen3_5MoeConfig,
        dir: impl AsRef<Path>,
    ) -> Result<Qwen35LoadVerifyReport, Report<ModelLoadError>> {
        verify_qwen35_weight_map(config, dir)
    }

    pub fn load_weights_sharded(
        &mut self,
        dir: impl AsRef<Path>,
    ) -> Result<Qwen35LoadVerifyReport, Report<ModelLoadError>> {
        let dir = dir.as_ref();
        let config = self.model.config.clone();
        let report = verify_qwen35_weight_map(&config, dir)?;
        if !report.pass() {
            eprintln!("[qwen3_5 load] verify failed before materializing weights: {report:?}");
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }

        let index = read_qwen35_weight_map(dir)?;
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (key, shard) in index {
            if key.starts_with("model.visual.") {
                continue;
            }
            by_shard.entry(shard).or_default().push(key);
        }

        let device = self.model.device();
        for (shard, keys) in by_shard {
            let path = dir.join(&shard);
            let mut store = SafetensorsStore::from_file(path);
            let snaps = store
                .get_all_snapshots()
                .context(ModelLoadError::LoadError)?;
            for key in keys {
                let snap = snaps
                    .get(&key)
                    .ok_or_else(|| Report::new(ModelLoadError::IncompleteLoad))?;
                self.load_qwen35_tensor(&key, snap, &device)
                    .map_err(|msg| {
                        eprintln!("[qwen3_5 load] {msg}");
                        Report::new(ModelLoadError::LoadError)
                    })?;
            }
        }
        Ok(report)
    }

    /// Load everything EXCEPT the routed-expert weight stacks (`mlp.experts.gate_up_proj` /
    /// `mlp.experts.down_proj`), leaving those `Param<Tensor<3>>`s at whatever `init_causal_lm` set
    /// them to (a tiny placeholder — see `moe.rs`/`qwen3_5/mod.rs` init). This is
    /// `docs/MEMORY_STREAMING_PLAN.md`'s "resident core" load path: attention, GDN, router, norms,
    /// embedding/head, and shared-expert weights (small relative to routed experts) are fully
    /// materialized as usual; routed experts are meant to be fetched on demand later via
    /// [`crate::expert_stream::ExpertSlotPool`] instead of being resident. Structural verification
    /// (`verify_qwen35_weight_map`) still runs against the FULL expected weight set, since we still
    /// want a loud failure on a genuinely corrupt/incomplete checkpoint — only the two big per-layer
    /// tensors are skipped during materialization.
    pub fn load_weights_sharded_resident_core(
        &mut self,
        dir: impl AsRef<Path>,
    ) -> Result<Qwen35LoadVerifyReport, Report<ModelLoadError>> {
        let dir = dir.as_ref();
        let config = self.model.config.clone();
        let report = verify_qwen35_weight_map(&config, dir)?;
        if !report.pass() {
            eprintln!("[qwen3_5 load] verify failed before materializing weights: {report:?}");
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }

        let index = read_qwen35_weight_map(dir)?;
        let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
        let mut skipped = 0usize;
        for (key, shard) in index {
            if key.starts_with("model.visual.") {
                continue;
            }
            if key.ends_with("mlp.experts.gate_up_proj") || key.ends_with("mlp.experts.down_proj") {
                skipped += 1;
                continue;
            }
            by_shard.entry(shard).or_default().push(key);
        }

        let device = self.model.device();
        for (shard, keys) in by_shard {
            let path = dir.join(&shard);
            let mut store = SafetensorsStore::from_file(path);
            let snaps = store
                .get_all_snapshots()
                .context(ModelLoadError::LoadError)?;
            for key in keys {
                let snap = snaps
                    .get(&key)
                    .ok_or_else(|| Report::new(ModelLoadError::IncompleteLoad))?;
                self.load_qwen35_tensor(&key, snap, &device)
                    .map_err(|msg| {
                        eprintln!("[qwen3_5 load] {msg}");
                        Report::new(ModelLoadError::LoadError)
                    })?;
            }
        }
        eprintln!(
            "[qwen3_5 load] resident-core load: skipped {skipped} routed-expert tensor(s), fetch them via ExpertSlotPool instead"
        );
        Ok(report)
    }

    /// Load NVIDIA's ModelOpt mixed-precision Qwen3.6 NVFP4 checkpoint.
    ///
    /// Dense GDN/full-attention projections are loaded as fp8 sidecars from raw E4M3 bytes
    /// (`[N,K] -> [K,N]`, scalar scale expanded to `[N]`). W4A16 NVFP4 `input_scale` tensors are
    /// consumed but deliberately dropped: the serving math keeps bf16/f32 activations, matching the
    /// vLLM W4A16 path that deletes `input_scale` unread. Expert bf16 tensors are never materialized;
    /// expert params are replaced with `[1,1,1]` placeholders after the sidecar is built.
    ///
    /// Set `NVFP4_DEQUANT_TO_FP8=1` to use the staged B5.0c fallback: experts are dequantized on host
    /// and requantized through the existing fp8 expert sidecar path, while dense fp8 stays unchanged.
    /// In that mode `lm_head` is dequantized into the ordinary linear weight so the model remains
    /// runnable before the NVFP4 lm_head forward path is wired.
    #[cfg(feature = "cuda")]
    pub fn load_nvidia_nvfp4(
        &mut self,
        dir: impl AsRef<Path>,
    ) -> Result<(), Report<ModelLoadError>> {
        let dir = dir.as_ref();
        let config = self.model.config.clone();
        let index = shard_index(dir).map_err(|msg| {
            eprintln!("[load_nvidia_nvfp4] {msg}");
            Report::new(ModelLoadError::LoadError)
        })?;
        let groups = collect_quant_groups(index.keys().cloned());
        let fallback_fp8 = std::env::var("NVFP4_DEQUANT_TO_FP8").ok().as_deref() == Some("1");
        let device = self.model.device();

        print_load_mem("start");

        let mut consumed_quantized: HashSet<String> = HashSet::new();
        let mut consumed_all: HashSet<String> = HashSet::new();
        let mut dense_groups = 0usize;
        let mut expert_groups = 0usize;
        let mut shared_groups = 0usize;
        let mut lm_head_groups = 0usize;

        for group in &groups {
            for key in quant_group_keys(&group.base, group.kind) {
                consumed_all.insert(key);
            }
            match group.kind {
                QuantGroupKind::DenseFp8 => dense_groups += 1,
                QuantGroupKind::Nvfp4 => {
                    if parse_expert_base(&group.base).is_some() {
                        expert_groups += 1;
                    } else if group.base == "lm_head" {
                        lm_head_groups += 1;
                    } else if shared_nvfp4_role(&group.base).is_some() {
                        shared_groups += 1;
                    }
                }
            }
        }

        let (expected_dense, expected_experts, expected_shared, expected_lm_head) =
            expected_quantized_counts(&config);
        if dense_groups != expected_dense
            || expert_groups != expected_experts
            || shared_groups != expected_shared
            || lm_head_groups != expected_lm_head
        {
            eprintln!(
                "[load_nvidia_nvfp4] quant group count mismatch: dense {dense_groups}/{expected_dense}, experts {expert_groups}/{expected_experts}, shared {shared_groups}/{expected_shared}, lm_head {lm_head_groups}/{expected_lm_head}"
            );
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }

        let mut shard_reader = ShardReader::new(dir, &index);

        // bf16-path tensors (embeddings, norms, router gates, MTP, GDN 1-D params, …) are read RAW via
        // the parsed-once-mmap ShardReader instead of burn-store's `get_all_snapshots`. The NVIDIA
        // shards interleave F8_E4M3/U8 quant tensors with the bf16 ones, and burn-store eagerly runs
        // `safetensor_dtype_to_burn` over EVERY tensor while building its snapshot cache (before any
        // key filter applies) — F8_E4M3 is unsupported there, so `get_all_snapshots` errored on the
        // quant tensors even though this loop only consumes the bf16 keys. Reading the exact keys we
        // need never touches the quant dtypes.
        let bf16_keys: Vec<(String, String)> = index
            .iter()
            .filter(|(key, _)| !key.starts_with("model.visual.") && !consumed_all.contains(*key))
            .map(|(key, shard)| (key.clone(), shard.clone()))
            .collect();
        for (key, shard) in bf16_keys {
            let raw = shard_reader.read_raw_tensor(&key).map_err(|msg| {
                eprintln!("[load_nvidia_nvfp4] read bf16 tensor {key} (shard {shard}): {msg}");
                Report::new(ModelLoadError::LoadError)
            })?;
            let snap = raw_tensor_to_snapshot(&key, raw).map_err(|msg| {
                eprintln!("[load_nvidia_nvfp4] {msg} (shard {shard})");
                Report::new(ModelLoadError::LoadError)
            })?;
            self.load_qwen35_tensor(&key, &snap, &device)
                .map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] load bf16 tensor {key} (shard {shard}): {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
            consumed_all.insert(key);
        }
        print_load_mem("bf16 paths loaded");

        for group in groups.iter().filter(|g| g.kind == QuantGroupKind::DenseFp8) {
            let role = dense_fp8_role(&group.base).ok_or_else(|| {
                eprintln!(
                    "[load_nvidia_nvfp4] unhandled dense fp8 group {}",
                    group.base
                );
                Report::new(ModelLoadError::LoadError)
            })?;
            let parts =
                dense_fp8_parts_from_reader(&mut shard_reader, &group.base).map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
            set_quant_sidecar(
                self,
                &role,
                QuantLinear::Fp8(W8A16Linear::from_packed_parts(
                    parts.q_bytes_kn,
                    parts.scale_n,
                    parts.k,
                    parts.n,
                    &device,
                )),
            )
            .map_err(|msg| {
                eprintln!("[load_nvidia_nvfp4] {msg}");
                Report::new(ModelLoadError::LoadError)
            })?;
            mark_quant_group_consumed(&mut consumed_quantized, &group.base, group.kind);
        }
        print_load_mem("dense fp8 sidecars loaded");

        for group in groups
            .iter()
            .filter(|g| g.kind == QuantGroupKind::Nvfp4 && shared_nvfp4_role(&g.base).is_some())
        {
            let role = shared_nvfp4_role(&group.base).unwrap();
            let parts =
                nvfp4_linear_parts_from_reader(&mut shard_reader, &group.base).map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
            if parts.n % 16 != 0 {
                eprintln!(
                    "[load_nvidia_nvfp4] {}: N={} is not divisible by 16",
                    group.base, parts.n
                );
                return Err(Report::new(ModelLoadError::LoadError));
            }
            set_quant_sidecar(
                self,
                &role,
                QuantLinear::Nvfp4(Nvfp4Linear::from_packed_parts(
                    parts.qw,
                    parts.bs,
                    parts.gscale,
                    parts.k,
                    parts.n,
                    &device,
                )),
            )
            .map_err(|msg| {
                eprintln!("[load_nvidia_nvfp4] {msg}");
                Report::new(ModelLoadError::LoadError)
            })?;
            mark_quant_group_consumed(&mut consumed_quantized, &group.base, group.kind);
        }
        print_load_mem("shared expert nvfp4 sidecars loaded");

        let lm_parts =
            nvfp4_linear_parts_from_reader(&mut shard_reader, "lm_head").map_err(|msg| {
                eprintln!("[load_nvidia_nvfp4] {msg}");
                Report::new(ModelLoadError::LoadError)
            })?;
        if lm_parts.n % 16 != 0 || lm_parts.k % 16 != 0 {
            eprintln!(
                "[load_nvidia_nvfp4] lm_head requires vocab%16 and K%16, got N={} K={}",
                lm_parts.n, lm_parts.k
            );
            return Err(Report::new(ModelLoadError::LoadError));
        }
        if fallback_fp8 {
            let w = crate::nvidia_ckpt::dequant_nvfp4_linear_to_kn(&lm_parts);
            self.lm_head.weight = Param::initialized(
                ParamId::new(),
                Tensor::<2>::from_data(TensorData::new(w, [lm_parts.k, lm_parts.n]), &device)
                    .cast(DType::BF16),
            );
            self.lm_head_quant = QuantSidecar(None);
        } else {
            self.lm_head_quant =
                QuantSidecar(Some(QuantLinear::Nvfp4(Nvfp4Linear::from_packed_parts(
                    lm_parts.qw,
                    lm_parts.bs,
                    lm_parts.gscale,
                    lm_parts.k,
                    lm_parts.n,
                    &device,
                ))));
        }
        mark_quant_group_consumed(&mut consumed_quantized, "lm_head", QuantGroupKind::Nvfp4);
        print_load_mem("lm_head loaded");

        for layer in 0..config.num_hidden_layers {
            let mut expert_parts = Vec::with_capacity(config.num_experts);
            for expert in 0..config.num_experts {
                let prefix = format!("model.language_model.layers.{layer}.mlp.experts.{expert}");
                let gate = expert_projection_parts_from_reader(
                    &mut shard_reader,
                    &format!("{prefix}.gate_proj"),
                )
                .map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
                let up = expert_projection_parts_from_reader(
                    &mut shard_reader,
                    &format!("{prefix}.up_proj"),
                )
                .map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
                let down = expert_projection_parts_from_reader(
                    &mut shard_reader,
                    &format!("{prefix}.down_proj"),
                )
                .map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?;
                expert_parts.push(fuse_expert_nvfp4_parts(gate, up, down).map_err(|msg| {
                    eprintln!("[load_nvidia_nvfp4] L{layer} expert {expert}: {msg}");
                    Report::new(ModelLoadError::LoadError)
                })?);
                mark_quant_group_consumed(
                    &mut consumed_quantized,
                    &format!("{prefix}.gate_proj"),
                    QuantGroupKind::Nvfp4,
                );
                mark_quant_group_consumed(
                    &mut consumed_quantized,
                    &format!("{prefix}.up_proj"),
                    QuantGroupKind::Nvfp4,
                );
                mark_quant_group_consumed(
                    &mut consumed_quantized,
                    &format!("{prefix}.down_proj"),
                    QuantGroupKind::Nvfp4,
                );
            }
            let h = config.hidden_size;
            let i = config.moe_intermediate_size;
            let gu_q_len = h * i;
            let gu_bs_len = (i * 2) * (h / 16);
            let dn_q_len = i * (h / 2);
            let dn_bs_len = h * (i / 16);
            if expert_parts.iter().any(|p| {
                p.qw_gu_outmajor.len() != gu_q_len
                    || p.bs_gu.len() != gu_bs_len
                    || p.qw_dn_outmajor.len() != dn_q_len
                    || p.bs_dn.len() != dn_bs_len
            }) {
                eprintln!(
                    "[load_nvidia_nvfp4] L{layer}: expert part length does not match config H={h} I={i}"
                );
                return Err(Report::new(ModelLoadError::LoadError));
            }
            let experts = experts_by_layer_mut(self, layer).ok_or_else(|| {
                eprintln!("[load_nvidia_nvfp4] expert layer {layer} out of range");
                Report::new(ModelLoadError::LoadError)
            })?;
            if fallback_fp8 {
                let fp8 = quantize_dequantized_expert_to_fp8(&expert_parts, h, i, &device);
                experts.fp8 = ExpertQuantSidecar(Some(fp8));
                experts.nvfp4 = ExpertNvfp4Sidecar(None);
            } else {
                experts.nvfp4 = ExpertNvfp4Sidecar(Some(ExpertNvfp4::from_expert_parts(
                    expert_parts,
                    h,
                    i,
                    &device,
                )));
                experts.fp8 = ExpertQuantSidecar(None);
            }
            set_expert_placeholders(experts, &device);
            print_load_mem(&format!("experts layer {layer} loaded"));
        }

        let quantized_keys: Vec<String> = groups
            .iter()
            .flat_map(|g| quant_group_keys(&g.base, g.kind))
            .collect();
        let mut unconsumed: Vec<String> = quantized_keys
            .into_iter()
            .filter(|k| !consumed_quantized.contains(k))
            .collect();
        unconsumed.sort();
        if !unconsumed.is_empty() {
            eprintln!(
                "[load_nvidia_nvfp4] unconsumed quantized tensor names ({}): {:?}",
                unconsumed.len(),
                unconsumed.iter().take(32).collect::<Vec<_>>()
            );
            return Err(Report::new(ModelLoadError::IncompleteLoad));
        }

        print_load_mem("done");
        println!(
            "[load_nvidia_nvfp4] loaded dense_fp8={dense_groups}, nvfp4_expert={expert_groups}, nvfp4_shared={shared_groups}, lm_head={lm_head_groups}, fallback_fp8={fallback_fp8}"
        );
        // B5.3 wired the NVFP4 dispatch (forward_static/forward_impl expert arm, expert_forward host
        // prefill arm, and the lm_head/shared dense NVFP4 path), so raw mode is runnable — the former
        // NVFP4_RAW_ACK panic guard is removed.
        Ok(())
    }

    fn load_qwen35_tensor(
        &mut self,
        key: &str,
        snap: &TensorSnapshot,
        device: &Device,
    ) -> Result<(), String> {
        if key == "lm_head.weight" {
            return set_linear(&mut self.lm_head, snap, device);
        }
        if let Some(rest) = key.strip_prefix("model.language_model.") {
            return self.load_language_tensor(rest, snap, device);
        }
        if let Some(rest) = key.strip_prefix("mtp.") {
            return self.load_mtp_tensor(rest, snap, device);
        }
        Err(format!("unhandled Qwen3.6 tensor key {key}"))
    }

    fn load_language_tensor(
        &mut self,
        rest: &str,
        snap: &TensorSnapshot,
        device: &Device,
    ) -> Result<(), String> {
        if rest == "embed_tokens.weight" {
            return set_embedding(&mut self.model.embed_tokens, snap, device);
        }
        if rest == "norm.weight" {
            return set_norm(&mut self.model.norm, snap, device);
        }
        let rest = rest
            .strip_prefix("layers.")
            .ok_or_else(|| format!("unhandled language tensor suffix {rest}"))?;
        let (layer, suffix) = split_index(rest, "language layer")?;
        let layer = self
            .model
            .layers
            .get_mut(layer)
            .ok_or_else(|| format!("language layer index {layer} out of range"))?;
        load_decoder_layer(layer, suffix, snap, device)
    }

    fn load_mtp_tensor(
        &mut self,
        rest: &str,
        snap: &TensorSnapshot,
        device: &Device,
    ) -> Result<(), String> {
        match rest {
            "pre_fc_norm_embedding.weight" => {
                return set_norm(&mut self.mtp.pre_fc_norm_embedding, snap, device);
            }
            "pre_fc_norm_hidden.weight" => {
                return set_norm(&mut self.mtp.pre_fc_norm_hidden, snap, device);
            }
            "fc.weight" => return set_linear(&mut self.mtp.fc, snap, device),
            "norm.weight" => return set_norm(&mut self.mtp.norm, snap, device),
            _ => {}
        }
        let rest = rest
            .strip_prefix("layers.")
            .ok_or_else(|| format!("unhandled mtp tensor suffix {rest}"))?;
        let (layer, suffix) = split_index(rest, "mtp layer")?;
        let layer = self
            .mtp
            .layers
            .get_mut(layer)
            .ok_or_else(|| format!("mtp layer index {layer} out of range"))?;
        load_full_layer(layer, suffix, snap, device)
    }
}

pub fn verify_qwen35_weight_map(
    config: &Qwen3_5MoeConfig,
    dir: impl AsRef<Path>,
) -> Result<Qwen35LoadVerifyReport, Report<ModelLoadError>> {
    let dir = dir.as_ref();
    let expected = config.expected_weight_shapes();
    let index = read_qwen35_weight_map(dir)?;
    let mut by_shard: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut index_keys = HashSet::new();
    for (key, shard) in &index {
        index_keys.insert(key.clone());
        by_shard.entry(shard.clone()).or_default().push(key.clone());
    }

    let mut seen = HashSet::new();
    let mut mapped_tensors = 0usize;
    let mut skipped_visual_tensors = 0usize;
    let mut param_count = 0u128;
    let mut skipped_visual_param_count = 0u128;
    let mut orphan = Vec::new();
    let mut missing = Vec::new();
    let mut shape_mismatches = Vec::new();

    for (shard, keys) in by_shard {
        let path = dir.join(&shard);
        let mut store = SafetensorsStore::from_file(path);
        let snaps = store
            .get_all_snapshots()
            .context(ModelLoadError::LoadError)?;
        for extra in snaps.keys() {
            if !index_keys.contains(extra) {
                orphan.push(format!(
                    "{extra} present in shard {shard} but absent from weight_map"
                ));
            }
        }
        for key in keys {
            let Some(snap) = snaps.get(&key) else {
                missing.push(format!(
                    "{key} listed in {shard} but absent from safetensors header"
                ));
                continue;
            };
            let elems = elem_count(&snap.shape);
            if key.starts_with("model.visual.") {
                skipped_visual_tensors += 1;
                skipped_visual_param_count += elems;
                continue;
            }
            let Some(expect) = expected.get(&key) else {
                orphan.push(format!("{key} has no text/MTP module target"));
                continue;
            };
            seen.insert(key.clone());
            mapped_tensors += 1;
            param_count += elems;
            if snap.shape.as_slice() != expect.as_slice() {
                shape_mismatches.push(format!(
                    "{key}: checkpoint {:?}, expected {:?}",
                    snap.shape, expect
                ));
            }
        }
    }

    for key in expected.keys() {
        if !seen.contains(key) {
            missing.push(key.clone());
        }
    }
    missing.sort();
    orphan.sort();
    shape_mismatches.sort();

    Ok(Qwen35LoadVerifyReport {
        weight_map_tensors: index_keys.len(),
        mapped_tensors,
        skipped_visual_tensors,
        param_count,
        skipped_visual_param_count,
        missing,
        orphan,
        shape_mismatches,
    })
}

fn read_qwen35_weight_map(dir: &Path) -> Result<Vec<(String, String)>, Report<ModelLoadError>> {
    let text = std::fs::read_to_string(dir.join("model.safetensors.index.json"))
        .map_err(|_| Report::new(ModelLoadError::LoadError))?;
    parse_weight_map(&text).map_err(|e| {
        eprintln!("[qwen3_5 load] failed to parse model.safetensors.index.json: {e}");
        Report::new(ModelLoadError::LoadError)
    })
}

#[cfg(feature = "cuda")]
fn mark_quant_group_consumed(consumed: &mut HashSet<String>, base: &str, kind: QuantGroupKind) {
    for key in quant_group_keys(base, kind) {
        consumed.insert(key);
    }
}

#[cfg(feature = "cuda")]
fn print_load_mem(stage: &str) {
    let rss_kb = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|text| {
            text.lines()
                .find_map(|line| {
                    line.strip_prefix("VmHWM:")
                        .or_else(|| line.strip_prefix("VmRSS:"))
                })
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|v| v.parse::<u64>().ok())
        });
    match rss_kb {
        Some(kb) => println!(
            "[load_nvidia_nvfp4] stage={stage} host_hwm_mib={:.1} device_mem=N/A",
            kb as f64 / 1024.0
        ),
        None => println!("[load_nvidia_nvfp4] stage={stage} host_hwm_mib=N/A device_mem=N/A"),
    }
}

#[cfg(feature = "cuda")]
fn experts_by_layer_mut(
    m: &mut Qwen3_5MoeForCausalLM,
    layer: usize,
) -> Option<&mut crate::qwen3_5::Qwen3_5FusedExperts> {
    match m.model.layers.get_mut(layer)? {
        Qwen3_5DecoderLayer::Linear(layer) => Some(&mut layer.mlp.experts),
        Qwen3_5DecoderLayer::Full(layer) => Some(&mut layer.mlp.experts),
    }
}

#[cfg(feature = "cuda")]
fn set_expert_placeholders(experts: &mut crate::qwen3_5::Qwen3_5FusedExperts, device: &Device) {
    let tiny = Tensor::<3>::from_data(TensorData::new(vec![0.0f32], [1, 1, 1]), device);
    experts.gate_up_proj = Param::initialized(ParamId::new(), tiny.clone());
    experts.down_proj = Param::initialized(ParamId::new(), tiny);
}

#[cfg(feature = "cuda")]
fn set_quant_sidecar(
    m: &mut Qwen3_5MoeForCausalLM,
    role: &str,
    q: QuantLinear,
) -> Result<(), String> {
    let slot = quant_sidecar_slot_mut(m, role)
        .ok_or_else(|| format!("no quant sidecar slot for role {role}"))?;
    *slot = Some(q);
    Ok(())
}

#[cfg(feature = "cuda")]
fn quant_sidecar_slot_mut<'a>(
    m: &'a mut Qwen3_5MoeForCausalLM,
    role: &str,
) -> Option<&'a mut Option<QuantLinear>> {
    if role == "lm_head" {
        return Some(&mut m.lm_head_quant.0);
    }
    if role == "mtp.fc" {
        return Some(&mut m.mtp.fc_fp8.0);
    }
    let rest = role.strip_prefix('L')?;
    let (idx, tail) = rest.split_once('.')?;
    let idx = idx.parse::<usize>().ok()?;
    match m.model.layers.get_mut(idx)? {
        Qwen3_5DecoderLayer::Linear(layer) => match tail {
            "gdn.in_proj_qkv" => Some(&mut layer.linear_attn.in_proj_qkv_fp8.0),
            "gdn.in_proj_a" => Some(&mut layer.linear_attn.in_proj_a_fp8.0),
            "gdn.in_proj_b" => Some(&mut layer.linear_attn.in_proj_b_fp8.0),
            "gdn.in_proj_z" => Some(&mut layer.linear_attn.in_proj_z_fp8.0),
            "gdn.out_proj" => Some(&mut layer.linear_attn.out_proj_fp8.0),
            _ => quant_mlp_sidecar_slot_mut(&mut layer.mlp, tail),
        },
        Qwen3_5DecoderLayer::Full(layer) => match tail {
            "attn.q_proj" => Some(&mut layer.self_attn.q_proj_fp8.0),
            "attn.k_proj" => Some(&mut layer.self_attn.k_proj_fp8.0),
            "attn.v_proj" => Some(&mut layer.self_attn.v_proj_fp8.0),
            "attn.o_proj" => Some(&mut layer.self_attn.o_proj_fp8.0),
            _ => quant_mlp_sidecar_slot_mut(&mut layer.mlp, tail),
        },
    }
}

#[cfg(feature = "cuda")]
fn quant_mlp_sidecar_slot_mut<'a>(
    mlp: &'a mut Qwen3_5SharedMoeBlock,
    tail: &str,
) -> Option<&'a mut Option<QuantLinear>> {
    match tail {
        "moe.shared.gate_proj" => Some(&mut mlp.shared_expert.gate_proj_fp8.0),
        "moe.shared.up_proj" => Some(&mut mlp.shared_expert.up_proj_fp8.0),
        "moe.shared.down_proj" => Some(&mut mlp.shared_expert.down_proj_fp8.0),
        _ => None,
    }
}

fn elem_count(shape: &[usize]) -> u128 {
    shape.iter().fold(1u128, |acc, &dim| acc * dim as u128)
}

fn split_index<'a>(rest: &'a str, label: &str) -> Result<(usize, &'a str), String> {
    let (idx, suffix) = rest
        .split_once('.')
        .ok_or_else(|| format!("malformed {label} key suffix {rest}"))?;
    let idx = idx
        .parse::<usize>()
        .map_err(|_| format!("malformed {label} index in key suffix {rest}"))?;
    Ok((idx, suffix))
}

fn load_decoder_layer(
    layer: &mut Qwen3_5DecoderLayer,
    suffix: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match layer {
        Qwen3_5DecoderLayer::Linear(layer) => load_gdn_layer(layer, suffix, snap, device),
        Qwen3_5DecoderLayer::Full(layer) => load_full_layer(layer, suffix, snap, device),
    }
}

fn load_gdn_layer(
    layer: &mut Qwen3_5GdnLayer,
    suffix: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match suffix {
        "input_layernorm.weight" => return set_norm(&mut layer.input_layernorm, snap, device),
        "post_attention_layernorm.weight" => {
            return set_norm(&mut layer.post_attention_layernorm, snap, device);
        }
        _ => {}
    }
    if let Some(rest) = suffix.strip_prefix("linear_attn.") {
        return load_gdn_attention(&mut layer.linear_attn, rest, snap, device);
    }
    if let Some(rest) = suffix.strip_prefix("mlp.") {
        return load_mlp(&mut layer.mlp, rest, snap, device);
    }
    Err(format!("unhandled GDN layer tensor suffix {suffix}"))
}

fn load_full_layer(
    layer: &mut Qwen3_5FullAttnLayer,
    suffix: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match suffix {
        "input_layernorm.weight" => return set_norm(&mut layer.input_layernorm, snap, device),
        "post_attention_layernorm.weight" => {
            return set_norm(&mut layer.post_attention_layernorm, snap, device);
        }
        _ => {}
    }
    if let Some(rest) = suffix.strip_prefix("self_attn.") {
        return load_full_attention(&mut layer.self_attn, rest, snap, device);
    }
    if let Some(rest) = suffix.strip_prefix("mlp.") {
        return load_mlp(&mut layer.mlp, rest, snap, device);
    }
    Err(format!(
        "unhandled full-attention layer tensor suffix {suffix}"
    ))
}

fn load_gdn_attention(
    attn: &mut Qwen3_5GdnAttention,
    rest: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match rest {
        "in_proj_qkv.weight" => set_linear(&mut attn.in_proj_qkv, snap, device),
        "in_proj_a.weight" => set_linear(&mut attn.in_proj_a, snap, device),
        "in_proj_b.weight" => set_linear(&mut attn.in_proj_b, snap, device),
        "in_proj_z.weight" => set_linear(&mut attn.in_proj_z, snap, device),
        "A_log" => set_param1(&mut attn.A_log, snap, device),
        "dt_bias" => set_param1(&mut attn.dt_bias, snap, device),
        "conv1d.weight" => set_param3(&mut attn.conv1d.weight, snap, device),
        // GDN output norm is the Mamba2-style RMSNormGated: it stores an ABSOLUTE gamma (~0.88 across
        // every GDN layer, all-positive) — PLAIN, not the model's (1+weight) regular RMSNorm. Load
        // without the +1.0 offset (set_norm_plain) so the gated norm reads `gamma`, not `1+gamma`.
        "norm.weight" => set_norm_plain(&mut attn.norm, snap, device),
        "out_proj.weight" => set_linear(&mut attn.out_proj, snap, device),
        _ => Err(format!("unhandled GDN attention tensor suffix {rest}")),
    }
}

fn load_full_attention(
    attn: &mut Qwen3_5FullAttention,
    rest: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match rest {
        "q_proj.weight" => set_linear(&mut attn.q_proj, snap, device),
        "k_proj.weight" => set_linear(&mut attn.k_proj, snap, device),
        "v_proj.weight" => set_linear(&mut attn.v_proj, snap, device),
        "o_proj.weight" => set_linear(&mut attn.o_proj, snap, device),
        "q_norm.weight" => set_norm(&mut attn.q_norm, snap, device),
        "k_norm.weight" => set_norm(&mut attn.k_norm, snap, device),
        _ => Err(format!("unhandled full-attention tensor suffix {rest}")),
    }
}

fn load_mlp(
    mlp: &mut Qwen3_5SharedMoeBlock,
    rest: &str,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    match rest {
        "gate.weight" => set_linear(&mut mlp.gate, snap, device),
        "experts.gate_up_proj" => set_param3(&mut mlp.experts.gate_up_proj, snap, device),
        "experts.down_proj" => set_param3(&mut mlp.experts.down_proj, snap, device),
        "shared_expert.gate_proj.weight" => {
            set_linear(&mut mlp.shared_expert.gate_proj, snap, device)
        }
        "shared_expert.up_proj.weight" => set_linear(&mut mlp.shared_expert.up_proj, snap, device),
        "shared_expert.down_proj.weight" => {
            set_linear(&mut mlp.shared_expert.down_proj, snap, device)
        }
        "shared_expert_gate.weight" => set_linear(&mut mlp.shared_expert_gate, snap, device),
        _ => Err(format!("unhandled MLP tensor suffix {rest}")),
    }
}

fn set_embedding(
    embedding: &mut Embedding,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    let data = snap
        .to_data()
        .map_err(|e| format!("load embedding data: {e:?}"))?;
    let tensor = Tensor::<2>::from_data(data, (device, snap.dtype));
    embedding.weight = Param::initialized(ParamId::new(), tensor);
    Ok(())
}

fn set_linear(linear: &mut Linear, snap: &TensorSnapshot, device: &Device) -> Result<(), String> {
    let data = snap
        .to_data()
        .map_err(|e| format!("load linear data: {e:?}"))?;
    let tensor = Tensor::<2>::from_data(data, (device, snap.dtype)).transpose();
    linear.weight = Param::initialized(ParamId::new(), tensor);
    Ok(())
}

fn set_norm(norm: &mut RmsNorm, snap: &TensorSnapshot, device: &Device) -> Result<(), String> {
    // qwen3_5_moe uses Gemma-style (1 + weight) RMSNorm — the stored gamma is the DELTA from 1.0.
    // The checkpoint's input/post-attention layernorm weights are centered at ~0 (mean 0.03 / -0.10,
    // with negative values), so a plain `x_normed * gamma` multiplies each sublayer's input by ~0,
    // turning all 40 layers into near-identity no-ops: the residual stream collapses to the bare
    // embedding and lm_head emits the unconditional base-language prior (the L1.6 incoherent-token bug,
    // present in both bf16 AND f32). Fold the +1.0 in at load so the plain RmsNorm.forward (and the GDN
    // gated norm, which reads this same gamma) compute `x_normed * (1 + gamma)`. A_log/dt_bias are NOT
    // norms — they call set_param1 directly and correctly skip this offset.
    let data = snap
        .to_data()
        .map_err(|e| format!("load norm gamma: {e:?}"))?;
    let tensor = Tensor::<1>::from_data(data, (device, snap.dtype))
        .cast(DType::F32)
        .add_scalar(1.0);
    norm.gamma = Param::initialized(ParamId::new(), tensor);
    Ok(())
}

/// PLAIN RMSNorm gamma load (no +1.0 offset). Used ONLY for the GDN GatedDeltaNet output norm
/// (`linear_attn.norm`, the Mamba2-style RMSNormGated `self.weight * x`), which stores an absolute
/// gamma (~0.88 across all GDN layers, all-positive) — unlike the model's regular (1+weight) RMSNorm.
fn set_norm_plain(
    norm: &mut RmsNorm,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    set_param1(&mut norm.gamma, snap, device)
}

fn set_param1(
    param: &mut Param<Tensor<1>>,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    let data = snap
        .to_data()
        .map_err(|e| format!("load rank-1 data: {e:?}"))?;
    // Rank-1 params (RmsNorm gammas, GDN A_log/dt_bias) are ELEMENTWISE operands, never matmul inputs.
    // The activation stream is F32 (linear3/matmul_out_in always output F32 — "F32 activations, bf16
    // matmul-compute"), so load these in F32 to avoid F32-activation × bf16-gamma DTypeMismatch in the
    // norms. Negligible memory (1-D). Large 2-D/3-D weights stay bf16 (cast to bf16 inside linear3).
    let tensor = Tensor::<1>::from_data(data, (device, snap.dtype)).cast(DType::F32);
    *param = Param::initialized(ParamId::new(), tensor);
    Ok(())
}

fn set_param3(
    param: &mut Param<Tensor<3>>,
    snap: &TensorSnapshot,
    device: &Device,
) -> Result<(), String> {
    let data = snap
        .to_data()
        .map_err(|e| format!("load rank-3 data: {e:?}"))?;
    let tensor = Tensor::<3>::from_data(data, (device, snap.dtype));
    *param = Param::initialized(ParamId::new(), tensor);
    Ok(())
}

/// Wrap a raw safetensors tensor (read via [`ShardReader`]) as a [`TensorSnapshot`] so the existing
/// `load_qwen35_tensor` setters (`set_linear`/`set_norm`/`set_embedding`/`set_param{1,3}`) consume it
/// unchanged. Used only by the NVFP4 bf16-path read, where burn-store's `get_all_snapshots` cannot be
/// used because it eagerly rejects the shards' interleaved F8_E4M3 quant tensors.
#[cfg(feature = "cuda")]
fn raw_tensor_to_snapshot(key: &str, raw: RawTensor) -> Result<TensorSnapshot, String> {
    use safetensors::Dtype;
    let dtype = match raw.dtype {
        Dtype::F64 => DType::F64,
        Dtype::F32 => DType::F32,
        Dtype::F16 => DType::F16,
        Dtype::BF16 => DType::BF16,
        Dtype::I64 => DType::I64,
        Dtype::I32 => DType::I32,
        Dtype::I16 => DType::I16,
        Dtype::I8 => DType::I8,
        Dtype::U8 => DType::U8,
        Dtype::BOOL => DType::Bool,
        other => {
            return Err(format!(
                "bf16-path tensor {key}: unsupported dtype {other:?} (shape {:?})",
                raw.shape
            ));
        }
    };
    let expected = raw.shape.iter().product::<usize>() * dtype.size();
    if raw.data.len() != expected {
        return Err(format!(
            "bf16-path tensor {key}: byte length {} != shape {:?} * {dtype:?} ({expected})",
            raw.data.len(),
            raw.shape
        ));
    }
    let data = TensorData::from_bytes_vec(raw.data, raw.shape, dtype);
    Ok(TensorSnapshot::from_data(
        data,
        key.split('.').map(str::to_string).collect(),
        Vec::new(),
        ParamId::new(),
    ))
}
