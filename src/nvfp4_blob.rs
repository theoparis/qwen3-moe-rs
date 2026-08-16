//! Offline-repacked NVFP4 expert store: fixed-stride, page-aligned, mmap-ready.
//!
//! # Why
//!
//! The streamed decode path re-derives NVFP4 experts from the original bf16 checkpoint on every
//! cache miss: read `[N,K]` bf16 out of a 68 GiB safetensors shard, then transpose+quantize it on
//! the CPU. Measured on an M2 Pro (`docs/metal-streamed-decode-findings.md`) that is ~5.0 ms of SSD
//! latency plus ~8.6 ms of CPU quantization *per miss*, and it dominates decode -- the expert math
//! itself is only ~0.4 ms.
//!
//! Both costs are avoidable, because the quantization result is a pure function of the checkpoint.
//! This module defines a store that is written once, offline, and thereafter consumed by
//! `offset = record_index * stride` into an mmap. A cache miss becomes a byte-slice plus an H2D
//! upload: no decode, no quantization, no safetensors header walk.
//!
//! Shrinking the bytes is the other half. The bf16 source is 60 GiB of routed experts, which can
//! never be page-cache resident in 16 GiB, so every miss is a cold SSD read. The NVFP4 form is
//! ~4.3x smaller (~17.6 GiB), which plausibly *does* stay cached.
//!
//! # Layout
//!
//! One blob file per layer, `layer_{L}.nvfp4`, so a layer's working set is contiguous and the
//! files can be built/copied incrementally. Within a file, records run gate_up experts `0..E` then
//! down experts `0..E`. Each record is
//!
//! ```text
//! [ qw: N*(K/2) bytes ][ block_scales: N*(K/16) bytes ][ gscale: 4 bytes f32-le ][ zero pad ]
//! ```
//!
//! padded up to [`ALIGN`] so every record starts on a page boundary -- that keeps the mmap slices
//! page-aligned for the driver and avoids a record straddling a page it doesn't need.
//!
//! A sidecar `manifest.txt` records the shapes and strides. It is validated against the live model
//! config at load time, so a stale or mismatched store is a clean error rather than silent garbage.
//! The manifest is a dependency-free `key=value` text format on purpose: `serde`/`serde_json` are
//! behind an optional cargo feature here, and the streamed decode path must work without it.

use std::path::{Path, PathBuf};

/// Record alignment. 16 KiB is the Apple-silicon page size and a multiple of the 4 KiB used
/// elsewhere, so records are page-aligned on both.
pub const ALIGN: usize = 16 * 1024;

/// Bumped whenever the on-disk layout or the quantizer's output changes; a mismatch forces a rebuild
/// rather than risking a silently wrong decode.
pub const FORMAT_VERSION: u32 = 1;

#[inline]
pub const fn align_up(n: usize, align: usize) -> usize {
    n.div_ceil(align) * align
}

/// Geometry of one projection's records: `[N,K]` NVFP4, one record per expert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjLayout {
    /// Output features (rows of the `[N,K]` weight).
    pub n: usize,
    /// Input features; must be a multiple of 16 (the NVFP4 block size).
    pub k: usize,
    /// Byte offset of expert 0's record within the layer file.
    pub base: usize,
    /// Distance in bytes between consecutive experts' records (already aligned).
    pub stride: usize,
}

impl ProjLayout {
    pub fn new(n: usize, k: usize, base: usize) -> Self {
        Self {
            n,
            k,
            base,
            stride: align_up(Self::payload_len(n, k), ALIGN),
        }
    }

    /// `qw + block_scales + gscale`, before alignment padding.
    pub const fn payload_len(n: usize, k: usize) -> usize {
        n * (k / 2) + n * (k / 16) + 4
    }

    pub const fn qw_len(&self) -> usize {
        self.n * (self.k / 2)
    }

    pub const fn bs_len(&self) -> usize {
        self.n * (self.k / 16)
    }

    /// Byte offset of `expert`'s record.
    pub fn record_offset(&self, expert: usize) -> usize {
        self.base + expert * self.stride
    }
}

/// Sidecar describing every layer file in the store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlobManifest {
    pub format_version: u32,
    pub num_layers: usize,
    pub num_experts: usize,
    pub gate_up: ProjLayout,
    pub down: ProjLayout,
}

impl BlobManifest {
    /// Build the canonical layout: gate_up records first, then down records.
    pub fn new(
        num_layers: usize,
        num_experts: usize,
        gate_up_n: usize,
        gate_up_k: usize,
        down_n: usize,
        down_k: usize,
    ) -> Self {
        let gate_up = ProjLayout::new(gate_up_n, gate_up_k, 0);
        let down = ProjLayout::new(down_n, down_k, gate_up.stride * num_experts);
        Self {
            format_version: FORMAT_VERSION,
            num_layers,
            num_experts,
            gate_up,
            down,
        }
    }

    /// Total size of one layer file.
    pub fn layer_file_len(&self) -> usize {
        self.down.base + self.down.stride * self.num_experts
    }

    pub fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("manifest.txt")
    }

    pub fn layer_path(dir: &Path, layer: usize) -> PathBuf {
        dir.join(format!("layer_{layer}.nvfp4"))
    }

    /// Render the manifest as `key=value` lines.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("format_version={}\n", self.format_version));
        s.push_str(&format!("num_layers={}\n", self.num_layers));
        s.push_str(&format!("num_experts={}\n", self.num_experts));
        for (name, l) in [("gate_up", &self.gate_up), ("down", &self.down)] {
            s.push_str(&format!("{name}.n={}\n", l.n));
            s.push_str(&format!("{name}.k={}\n", l.k));
            s.push_str(&format!("{name}.base={}\n", l.base));
            s.push_str(&format!("{name}.stride={}\n", l.stride));
        }
        s
    }

    pub fn from_text(text: &str) -> Result<Self, String> {
        let mut kv = std::collections::BTreeMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (k, v) = line
                .split_once('=')
                .ok_or_else(|| format!("malformed NVFP4 manifest line: {line:?}"))?;
            kv.insert(k.trim().to_string(), v.trim().to_string());
        }
        let get = |key: &str| -> Result<usize, String> {
            kv.get(key)
                .ok_or_else(|| format!("NVFP4 manifest missing key {key:?}"))?
                .parse::<usize>()
                .map_err(|e| format!("NVFP4 manifest key {key:?} is not a number: {e}"))
        };
        let proj = |name: &str| -> Result<ProjLayout, String> {
            Ok(ProjLayout {
                n: get(&format!("{name}.n"))?,
                k: get(&format!("{name}.k"))?,
                base: get(&format!("{name}.base"))?,
                stride: get(&format!("{name}.stride"))?,
            })
        };
        Ok(Self {
            format_version: get("format_version")? as u32,
            num_layers: get("num_layers")?,
            num_experts: get("num_experts")?,
            gate_up: proj("gate_up")?,
            down: proj("down")?,
        })
    }

    pub fn load(dir: &Path) -> Result<Self, String> {
        let path = Self::manifest_path(dir);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("read NVFP4 blob manifest {}: {e}", path.display()))?;
        let m = Self::from_text(&text)
            .map_err(|e| format!("parse NVFP4 blob manifest {}: {e}", path.display()))?;
        if m.format_version != FORMAT_VERSION {
            return Err(format!(
                "NVFP4 blob store at {} is format v{} but this build expects v{FORMAT_VERSION}; rebuild it",
                dir.display(),
                m.format_version
            ));
        }
        Ok(m)
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = Self::manifest_path(dir);
        std::fs::write(&path, self.to_text())
            .map_err(|e| format!("write NVFP4 blob manifest {}: {e}", path.display()))
    }

    /// Fail loudly if the store does not match the model actually being run.
    pub fn check_matches(
        &self,
        num_layers: usize,
        num_experts: usize,
        gate_up_n: usize,
        gate_up_k: usize,
        down_n: usize,
        down_k: usize,
    ) -> Result<(), String> {
        let want = Self::new(
            num_layers,
            num_experts,
            gate_up_n,
            gate_up_k,
            down_n,
            down_k,
        );
        if *self == want {
            return Ok(());
        }
        Err(format!(
            "NVFP4 blob store does not match this model.\n  store: {self:?}\n  model: {want:?}"
        ))
    }
}

/// Which of the two expert projections a record belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobProj {
    GateUp,
    Down,
}

impl BlobManifest {
    pub fn layout(&self, proj: BlobProj) -> &ProjLayout {
        match proj {
            BlobProj::GateUp => &self.gate_up,
            BlobProj::Down => &self.down,
        }
    }
}

/// Serialize one expert's packed NVFP4 parts into its aligned record.
///
/// `out` must be exactly `layout.stride` bytes; padding is left as-is (the writer zeroes it).
pub fn write_record(
    out: &mut [u8],
    layout: &ProjLayout,
    qw: &[u8],
    block_scales: &[u8],
    gscale: f32,
) -> Result<(), String> {
    if out.len() != layout.stride {
        return Err(format!(
            "NVFP4 record buffer is {} bytes, expected stride {}",
            out.len(),
            layout.stride
        ));
    }
    if qw.len() != layout.qw_len() {
        return Err(format!(
            "NVFP4 qw is {} bytes, expected {}",
            qw.len(),
            layout.qw_len()
        ));
    }
    if block_scales.len() != layout.bs_len() {
        return Err(format!(
            "NVFP4 block_scales is {} bytes, expected {}",
            block_scales.len(),
            layout.bs_len()
        ));
    }
    let (qw_dst, rest) = out.split_at_mut(layout.qw_len());
    qw_dst.copy_from_slice(qw);
    let (bs_dst, rest) = rest.split_at_mut(layout.bs_len());
    bs_dst.copy_from_slice(block_scales);
    rest[..4].copy_from_slice(&gscale.to_le_bytes());
    Ok(())
}

/// Borrowed view of one expert's record inside an mmap. No copying, no decode.
#[derive(Debug, Clone, Copy)]
pub struct RecordRef<'a> {
    pub qw: &'a [u8],
    pub block_scales: &'a [u8],
    pub gscale: f32,
}

/// Slice `expert`'s record out of a whole-layer byte range.
pub fn read_record<'a>(
    layer_bytes: &'a [u8],
    layout: &ProjLayout,
    expert: usize,
) -> Result<RecordRef<'a>, String> {
    let start = layout.record_offset(expert);
    let payload = ProjLayout::payload_len(layout.n, layout.k);
    let end = start + payload;
    if end > layer_bytes.len() {
        return Err(format!(
            "NVFP4 record for expert {expert} ends at {end} but layer blob is {} bytes",
            layer_bytes.len()
        ));
    }
    let rec = &layer_bytes[start..end];
    let (qw, rest) = rec.split_at(layout.qw_len());
    let (block_scales, gs) = rest.split_at(layout.bs_len());
    let gscale = f32::from_le_bytes([gs[0], gs[1], gs[2], gs[3]]);
    Ok(RecordRef {
        qw,
        block_scales,
        gscale,
    })
}

/// Kind of resident core tensor entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentDType {
    Nvfp4Linear { k: usize, n: usize },
    Bf16,
    F32,
}

/// Metadata entry for one resident tensor stored in the contiguous blob.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: ResidentDType,
    pub offset: usize,
    pub byte_len: usize,
    pub gscale: Option<f32>,
}

/// Manifest describing the entire contiguous offline-quantized resident core (~1.35 GB).
#[derive(Debug, Clone, PartialEq)]
pub struct ResidentCoreManifest {
    pub format_version: u32,
    pub total_size: usize,
    pub entries: Vec<ResidentEntry>,
}

impl ResidentCoreManifest {
    pub fn new() -> Self {
        Self {
            format_version: FORMAT_VERSION,
            total_size: 0,
            entries: Vec::new(),
        }
    }

    /// Add a 2D NVFP4 quantized linear entry.
    pub fn add_nvfp4_linear(&mut self, name: String, k: usize, n: usize, gscale: f32) -> usize {
        let payload_len = ProjLayout::payload_len(n, k);
        let offset = align_up(self.total_size, ALIGN);
        let byte_len = align_up(payload_len, ALIGN);
        self.total_size = offset + byte_len;
        self.entries.push(ResidentEntry {
            name,
            shape: vec![k, n],
            dtype: ResidentDType::Nvfp4Linear { k, n },
            offset,
            byte_len,
            gscale: Some(gscale),
        });
        offset
    }

    /// Add a raw BF16/F32 entry (e.g. embeddings or layernorms).
    pub fn add_raw(&mut self, name: String, shape: Vec<usize>, is_f32: bool) -> usize {
        let elem_count: usize = shape.iter().product();
        let bytes_per_elem = if is_f32 { 4 } else { 2 };
        let payload_len = elem_count * bytes_per_elem;
        let offset = align_up(self.total_size, ALIGN);
        let byte_len = align_up(payload_len, ALIGN);
        self.total_size = offset + byte_len;
        self.entries.push(ResidentEntry {
            name,
            shape,
            dtype: if is_f32 {
                ResidentDType::F32
            } else {
                ResidentDType::Bf16
            },
            offset,
            byte_len,
            gscale: None,
        });
        offset
    }

    pub fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("resident_manifest.txt")
    }

    pub fn blob_path(dir: &Path) -> PathBuf {
        dir.join("resident_core.nvfp4")
    }

    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("format_version={}\n", self.format_version));
        s.push_str(&format!("total_size={}\n", self.total_size));
        s.push_str(&format!("entry_count={}\n", self.entries.len()));
        for (i, e) in self.entries.iter().enumerate() {
            let shape_str: Vec<String> = e.shape.iter().map(|d| d.to_string()).collect();
            s.push_str(&format!(
                "entry.{i}={};{};{:?};{};{};{}\n",
                e.name,
                shape_str.join(","),
                e.dtype,
                e.offset,
                e.byte_len,
                e.gscale.unwrap_or(1.0)
            ));
        }
        s
    }

    pub fn save(&self, dir: &Path) -> Result<(), String> {
        let path = Self::manifest_path(dir);
        std::fs::write(&path, self.to_text())
            .map_err(|e| format!("write resident manifest {}: {e}", path.display()))
    }
}

/// Parse one `entry.N=` value written by [`ResidentCoreManifest::to_text`], whose layout is
/// `name;shape_csv;dtype;offset;byte_len;gscale` with `dtype` in its `Debug` spelling
/// (`Bf16`, `F32`, or `Nvfp4Linear { k: K, n: N }`).
fn parse_resident_entry(v: &str) -> Result<ResidentEntry, String> {
    let f: Vec<&str> = v.split(';').collect();
    if f.len() != 6 {
        return Err(format!(
            "resident entry needs 6 fields, got {}: {v:?}",
            f.len()
        ));
    }
    let name = f[0].to_string();
    let shape = f[1]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("shape in {v:?}: {e}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let dtype = parse_resident_dtype(f[2])?;
    let offset = f[3]
        .trim()
        .parse()
        .map_err(|e| format!("offset in {v:?}: {e}"))?;
    let byte_len = f[4]
        .trim()
        .parse()
        .map_err(|e| format!("byte_len in {v:?}: {e}"))?;
    let gscale: f32 = f[5]
        .trim()
        .parse()
        .map_err(|e| format!("gscale in {v:?}: {e}"))?;
    Ok(ResidentEntry {
        name,
        shape,
        dtype,
        offset,
        byte_len,
        // Only NVFP4 entries carry a meaningful global scale; raw ones are written as 1.0.
        gscale: matches!(dtype, ResidentDType::Nvfp4Linear { .. }).then_some(gscale),
    })
}

fn parse_resident_dtype(s: &str) -> Result<ResidentDType, String> {
    let s = s.trim();
    if s == "Bf16" {
        return Ok(ResidentDType::Bf16);
    }
    if s == "F32" {
        return Ok(ResidentDType::F32);
    }
    let inner = s
        .strip_prefix("Nvfp4Linear")
        .ok_or_else(|| format!("unknown resident dtype {s:?}"))?
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}');
    let (mut k, mut n) = (None, None);
    for part in inner.split(',') {
        let Some((key, val)) = part.split_once(':') else {
            continue;
        };
        let val: usize = val
            .trim()
            .parse()
            .map_err(|e| format!("dtype field in {s:?}: {e}"))?;
        match key.trim() {
            "k" => k = Some(val),
            "n" => n = Some(val),
            _ => {}
        }
    }
    match (k, n) {
        (Some(k), Some(n)) => Ok(ResidentDType::Nvfp4Linear { k, n }),
        _ => Err(format!("Nvfp4Linear dtype missing k/n: {s:?}")),
    }
}

/// Contiguous reader for the offline-quantized resident core (~1.35 GB).
pub struct ResidentCoreBlob {
    pub manifest: ResidentCoreManifest,
    pub mmap: memmap2::Mmap,
}

impl ResidentCoreBlob {
    pub fn open(dir: &Path) -> Result<Self, String> {
        let manifest_path = ResidentCoreManifest::manifest_path(dir);
        let blob_path = ResidentCoreManifest::blob_path(dir);
        if !manifest_path.exists() || !blob_path.exists() {
            return Err(format!("resident core blob not found at {}", dir.display()));
        }
        let text = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("read {}: {e}", manifest_path.display()))?;
        let mut manifest = ResidentCoreManifest::new();
        let mut declared_count: Option<usize> = None;
        let mut indexed: Vec<(usize, ResidentEntry)> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            match k {
                "format_version" => {
                    manifest.format_version = v.parse().map_err(|e| format!("{e}"))?
                }
                "total_size" => manifest.total_size = v.parse().map_err(|e| format!("{e}"))?,
                "entry_count" => declared_count = Some(v.parse().map_err(|e| format!("{e}"))?),
                _ => {
                    if let Some(idx) = k.strip_prefix("entry.") {
                        let idx: usize = idx
                            .parse()
                            .map_err(|e| format!("bad entry index in {k:?}: {e}"))?;
                        indexed.push((idx, parse_resident_entry(v)?));
                    }
                }
            }
        }
        indexed.sort_by_key(|(i, _)| *i);
        manifest.entries = indexed.into_iter().map(|(_, e)| e).collect();

        // Fail loudly on a manifest that parses to nothing. The previous version of this reader
        // silently ignored every `entry.N=` line, so the blob loaded as an empty model whose lazily
        // initialized params were then RANDOMLY initialized at full size on first use -- slow,
        // memory-hungry, and producing garbage, with no error anywhere.
        if manifest.entries.is_empty() {
            return Err(format!(
                "resident manifest {} declared no entries (parsed 0)",
                manifest_path.display()
            ));
        }
        if let Some(want) = declared_count {
            if want != manifest.entries.len() {
                return Err(format!(
                    "resident manifest {}: entry_count={} but parsed {}",
                    manifest_path.display(),
                    want,
                    manifest.entries.len()
                ));
            }
        }

        let file = std::fs::File::open(&blob_path)
            .map_err(|e| format!("open {}: {e}", blob_path.display()))?;
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| format!("mmap resident core {}: {e}", blob_path.display()))?
        };
        Ok(Self { manifest, mmap })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_manifest() -> BlobManifest {
        // Small but shape-legal: K multiples of 16.
        BlobManifest::new(2, 3, 8, 32, 4, 16)
    }

    #[test]
    fn records_are_page_aligned_and_non_overlapping() {
        let m = tiny_manifest();
        for proj in [BlobProj::GateUp, BlobProj::Down] {
            let l = m.layout(proj);
            assert_eq!(l.stride % ALIGN, 0, "stride must be page-aligned");
            assert!(
                l.stride >= ProjLayout::payload_len(l.n, l.k),
                "stride must fit the payload"
            );
            for e in 0..m.num_experts {
                assert_eq!(l.record_offset(e) % ALIGN, 0, "record must be page-aligned");
            }
            // Consecutive records cannot overlap.
            for e in 1..m.num_experts {
                assert!(
                    l.record_offset(e)
                        >= l.record_offset(e - 1) + ProjLayout::payload_len(l.n, l.k)
                );
            }
        }
        // gate_up and down sections are disjoint, and everything fits the declared file length.
        assert!(m.down.base >= m.gate_up.record_offset(m.num_experts - 1) + m.gate_up.stride - 1);
        assert!(m.layer_file_len() >= m.down.record_offset(m.num_experts - 1));
    }

    #[test]
    fn write_then_read_round_trips_every_expert() {
        let m = tiny_manifest();
        let mut file = vec![0u8; m.layer_file_len()];
        let mut expected = Vec::new();

        for proj in [BlobProj::GateUp, BlobProj::Down] {
            let l = *m.layout(proj);
            for e in 0..m.num_experts {
                // Distinct, order-sensitive contents so a mis-slice cannot pass.
                let tag = if matches!(proj, BlobProj::GateUp) {
                    0x10
                } else {
                    0x90
                };
                let qw: Vec<u8> = (0..l.qw_len())
                    .map(|i| (i as u8) ^ (tag + e as u8))
                    .collect();
                let bs: Vec<u8> = (0..l.bs_len())
                    .map(|i| (i as u8).wrapping_add(tag + e as u8))
                    .collect();
                let gscale = 0.5 + e as f32;
                let off = l.record_offset(e);
                write_record(&mut file[off..off + l.stride], &l, &qw, &bs, gscale).unwrap();
                expected.push((proj, e, qw, bs, gscale));
            }
        }

        for (proj, e, qw, bs, gscale) in expected {
            let l = m.layout(proj);
            let r = read_record(&file, l, e).unwrap();
            assert_eq!(r.qw, &qw[..], "{proj:?} expert {e} qw mismatch");
            assert_eq!(
                r.block_scales,
                &bs[..],
                "{proj:?} expert {e} scales mismatch"
            );
            assert_eq!(r.gscale, gscale, "{proj:?} expert {e} gscale mismatch");
        }
    }

    #[test]
    fn manifest_round_trips_and_rejects_mismatch() {
        let dir = std::env::temp_dir().join(format!("nvfp4blob-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let m = tiny_manifest();
        assert_eq!(BlobManifest::from_text(&m.to_text()).unwrap(), m);
        m.save(&dir).unwrap();
        assert_eq!(BlobManifest::load(&dir).unwrap(), m);
        assert!(m.check_matches(2, 4, 8, 32, 4, 16).is_err());
        assert!(m.check_matches(2, 3, 8, 32, 4, 32).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Round-trips through `ResidentCoreBlob::open`, not just `save`. The previous version of this
    /// test only asserted the manifest file existed, which is how a reader that silently dropped
    /// every `entry.N=` line went unnoticed -- the blob then loaded as an *empty* model.
    #[test]
    fn resident_core_manifest_round_trips_through_open() {
        let dir = std::env::temp_dir().join(format!("residentcore-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut m = ResidentCoreManifest::new();
        m.add_nvfp4_linear(
            "model.layers.0.self_attn.q_proj".to_string(),
            2048,
            4096,
            1.25,
        );
        m.add_raw("model.norm.weight".to_string(), vec![2048], false);
        m.add_raw("model.some_f32".to_string(), vec![16, 4], true);
        m.save(&dir).unwrap();
        std::fs::write(
            ResidentCoreManifest::blob_path(&dir),
            vec![0u8; m.total_size],
        )
        .unwrap();

        let opened = ResidentCoreBlob::open(&dir).unwrap();
        assert_eq!(opened.manifest.entries.len(), 3);
        assert_eq!(opened.manifest.total_size, m.total_size);
        assert_eq!(opened.manifest.entries, m.entries);

        // Field-level checks on the NVFP4 entry, since its Debug-spelled dtype is the fiddly one.
        let e0 = &opened.manifest.entries[0];
        assert_eq!(e0.name, "model.layers.0.self_attn.q_proj");
        assert_eq!(e0.dtype, ResidentDType::Nvfp4Linear { k: 2048, n: 4096 });
        assert_eq!(e0.gscale, Some(1.25));
        assert_eq!(opened.manifest.entries[1].dtype, ResidentDType::Bf16);
        assert_eq!(opened.manifest.entries[1].gscale, None);
        assert_eq!(opened.manifest.entries[2].dtype, ResidentDType::F32);

        // Every entry must actually address bytes inside the blob.
        for e in &opened.manifest.entries {
            assert!(
                e.offset + e.byte_len <= opened.mmap.len(),
                "entry {} runs past the blob",
                e.name
            );
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resident_core_open_rejects_manifest_with_no_entries() {
        let dir = std::env::temp_dir().join(format!("residentcore-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            ResidentCoreManifest::manifest_path(&dir),
            "format_version=1\ntotal_size=64\n",
        )
        .unwrap();
        std::fs::write(ResidentCoreManifest::blob_path(&dir), vec![0u8; 64]).unwrap();
        assert!(ResidentCoreBlob::open(&dir).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }
}
