//! Minimal host-side repro for the NVFP4 bf16-path load failure at src/load.rs.
//!
//! Confirms:
//!   1. `SafetensorsStore::get_all_snapshots()` ERRORS on an NVIDIA NVFP4 shard because the shard
//!      interleaves F8_E4M3/U8 quant tensors that burn-store cannot snapshot (the old bf16-path read).
//!   2. The `ShardReader` raw read of the SAME bf16 keys SUCCEEDS (the fix path).
//!
//! Run (no GPU needed):
//!   cargo run --release --example nvfp4_bf16_read_repro -- models/qwen3.6-35b-a3b-nvfp4

use std::collections::BTreeSet;
use std::path::Path;

use burn::store::{ModuleStore, SafetensorsStore};
use qwen3_burn::nvidia_ckpt::{ShardReader, shard_index};

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "models/qwen3.6-35b-a3b-nvfp4".to_string());
    let dir = Path::new(&dir);

    let index = shard_index(dir).expect("read index");
    let shard1 = index
        .values()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .next()
        .expect("at least one shard");
    println!("[repro] testing shard: {shard1}");

    // (1) Old path: get_all_snapshots over the whole shard.
    let mut store = SafetensorsStore::from_file(dir.join(&shard1));
    match store.get_all_snapshots() {
        Ok(snaps) => println!(
            "[repro] get_all_snapshots OK ({} tensors) — no dtype in this shard is unsupported",
            snaps.len()
        ),
        Err(e) => println!("[repro] get_all_snapshots ERRORED (the reported bug): {e:?}"),
    }

    // (2) Fix path: read the bf16 keys of shard1 raw via ShardReader.
    let mut reader = ShardReader::new(dir, &index);
    let bf16_keys: Vec<String> = index
        .iter()
        .filter(|(k, s)| **s == shard1 && !k.starts_with("model.visual."))
        .map(|(k, _)| k.clone())
        .collect();
    let mut ok = 0usize;
    let mut sample_dtypes = BTreeSet::new();
    for key in &bf16_keys {
        let raw = reader
            .read_raw_tensor(key)
            .unwrap_or_else(|e| panic!("read {key}: {e}"));
        sample_dtypes.insert(format!("{:?}", raw.dtype));
        ok += 1;
    }
    println!(
        "[repro] ShardReader read {ok}/{} shard-1 keys raw; dtypes present: {sample_dtypes:?}",
        bf16_keys.len()
    );
    println!("[repro] DONE");
}
