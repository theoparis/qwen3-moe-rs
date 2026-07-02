use super::{
    MemoryConfiguration, MemoryPoolOptions, MemoryUsage, PoolType,
    memory_pool::{ExclusiveMemoryPool, MemoryPool, PersistentPool, SlicedPool},
};
use crate::{
    config::{
        GlobalConfig,
        memory::{MemoryLogLevel, PersistentMemory},
    },
    logging::ServerLogger,
    memory_management::BytesFormat,
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

use super::CaptureArena;
use alloc::format;
use alloc::string::{String, ToString};
#[cfg(not(exclusive_memory_only))]
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::{backtrace::BackTrace, stub::Arc};
use cubecl_ir::MemoryDeviceProperties;
use hashbrown::HashMap;

pub use super::memory_pool::{SliceBinding, handle::*};

// These are 288 bytes vs 64 bytes. Adding boxing isn't really worth
// saving the 200 bytes.
#[allow(clippy::large_enum_variant)]
enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
}

impl MemoryPool for DynamicPool {
    fn accept(&self, size: u64) -> bool {
        match self {
            DynamicPool::Sliced(pool) => pool.accept(size),
            DynamicPool::Exclusive(pool) => pool.accept(size),
        }
    }

    fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        match self {
            DynamicPool::Sliced(m) => m.get(binding),
            DynamicPool::Exclusive(m) => m.get(binding),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn try_reserve(&mut self, size: u64) -> Option<SliceHandle> {
        match self {
            DynamicPool::Sliced(m) => m.try_reserve(size),
            DynamicPool::Exclusive(m) => m.try_reserve(size),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Exclusive(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr, explicit),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr, explicit),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a persistent memory management strategy, meaning that all allocations are for data that is
    /// likely never going to be freed.
    Persistent,
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    name: String,
    persistent: PersistentPool,
    pools: Vec<DynamicPool>,
    storage: Storage,
    alloc_reserve_count: u64,
    mode: MemoryAllocationMode,
    config: PersistentMemory,
    logger: Arc<ServerLogger>,
    /// Storage alignment, used to pad capture-arena blocks (component C2).
    alignment: u64,
    /// The currently-recording capture arena (component C2). While `Some`, every [`Self::reserve`]
    /// is served from this isolated arena instead of the general pools, and freed blocks recycle
    /// within it. Created by [`Self::capture_arena_begin`].
    capture: Option<CaptureArena>,
    /// Capture arenas sealed to a captured graph id (kept alive at fixed device addresses for the
    /// graph's whole lifetime; freed by [`Self::capture_arena_free`] on graph destroy).
    capture_sealed: HashMap<u64, CaptureArena>,
    /// SHARED capture pools (P4 — vLLM `graph_pool_handle`), keyed by pool id. ONE arena is shared by
    /// several captured graphs (e.g. prompt-length buckets) that replay SERIALLY, so K graphs cost
    /// ~1 graph's high-water instead of K×. A pool stays installed here between captures; the active
    /// [`Self::capture`] slot borrows it (via [`Self::capture_pool_begin`]) and returns it (via
    /// [`Self::capture_pool_seal`]). Freed when its refcount hits 0 (see [`Self::pool_refcount`]).
    capture_pools: HashMap<u64, CaptureArena>,
    /// Refcount per shared pool: 1 for the live [`crate::client::CapturePoolHandle`] + 1 per live
    /// captured graph sealed into it. The pool's device blocks are freed when this reaches 0 (handle
    /// dropped AND every graph destroyed). Decremented by [`Self::capture_arena_free`] (per graph) and
    /// [`Self::capture_pool_release`] (the handle).
    pool_refcount: HashMap<u64, usize>,
    /// Maps a sealed graph id to the shared pool it belongs to (if any), so [`Self::capture_arena_free`]
    /// knows to decrement the pool refcount rather than free a per-graph arena.
    graph_to_pool: HashMap<u64, u64>,
}

fn generate_bucket_sizes(
    start_size: u64,
    end_size: u64,
    max_buckets: usize,
    alignment: u64,
) -> Vec<u64> {
    let mut buckets = Vec::with_capacity(max_buckets);
    let log_min = (start_size as f64).ln();
    let log_max = (end_size as f64).ln();
    let log_range = log_max - log_min;

    // Pure exponential performed best, but let's try slightly denser in lower-mid range
    for i in 0..max_buckets {
        let p = i as f64 / (max_buckets - 1) as f64;
        // Slight bias toward lower-mid range with less aggressive curve than sigmoid
        let log_size = log_min + log_range * p;
        let size = log_size.exp() as u64;
        let aligned_size = size.next_multiple_of(alignment);
        buckets.push(aligned_size);
    }

    buckets.dedup();
    buckets
}

const DEALLOC_SCALE_MB: u64 = 1024 * 1024 * 1024;
const BASE_DEALLOC_PERIOD: u64 = 5000;

/// The options for creating a new [`MemoryManagement`] instance.
#[derive(Debug)]
pub struct MemoryManagementOptions {
    /// The name of the memory management.
    name: String,
    /// The [`MemoryAllocationOption`] used by this instance.
    memory: MemoryAllocationOption,
}

impl MemoryManagementOptions {
    /// Creates a new [`MemoryManagementOptions`].
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    /// Forces the [`MemoryAllocationMode`] during execution to always be the provided one.
    pub fn mode(mut self, mode: MemoryAllocationMode) -> Self {
        self.memory = MemoryAllocationOption::Provided(mode);
        self
    }
}

#[derive(Default, Debug)]
/// Determines which [`MemoryAllocationMode`] is used during allocations.
enum MemoryAllocationOption {
    #[default]
    /// Uses the [`GlobalConfig`] to determine the mode of allocation.
    FromConfig,
    /// Use the provided [`MemoryAllocationMode`].
    Provided(MemoryAllocationMode),
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        let pool_options = match config {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::SubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.alignment;
                let max_page = properties.max_page_size;
                let mut pools = Vec::new();

                const MB: u64 = 1024 * 1024;

                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
                    dealloc_period: None,
                });

                let mut current = max_page;
                let mut max_sizes = vec![];
                let mut page_sizes = vec![];
                let mut base = pools.len() as u32;

                while current >= 32 * MB {
                    current /= 4;

                    // Make sure every pool has an aligned size.
                    current = current.next_multiple_of(memory_alignment);

                    max_sizes.push(current / 2u64.pow(base));
                    page_sizes.push(current);
                    base += 1;
                }

                max_sizes.reverse();
                page_sizes.reverse();

                for i in 0..max_sizes.len() {
                    let max = max_sizes[i];
                    let page_size = page_sizes[i];

                    pools.push(MemoryPoolOptions {
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::SlicedPages {
                            page_size,
                            max_slice_size: max,
                        },
                        dealloc_period: None,
                    });
                }

                // Add pools from big to small.
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: max_page / memory_alignment * memory_alignment,
                        max_slice_size: max_page / memory_alignment * memory_alignment,
                    },
                    dealloc_period: None,
                });
                pools
            }
            MemoryConfiguration::ExclusivePages => {
                // Add all bin sizes. Nb: because of alignment some buckets
                // end up as the same size, so only want unique ones,
                // but also keep the order, so a BTree will do.
                const MIN_BUCKET_SIZE: u64 = 1024 * 32;
                const NUM_POOLS: usize = 24;

                let sizes = generate_bucket_sizes(
                    MIN_BUCKET_SIZE,
                    properties.max_page_size,
                    NUM_POOLS,
                    properties.alignment,
                );

                sizes
                    .iter()
                    .map(|&size| {
                        let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                            * (1.0 + size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                            as u64;

                        MemoryPoolOptions {
                            pool_type: PoolType::ExclusivePages {
                                max_alloc_size: size,
                            },
                            dealloc_period: Some(dealloc_period),
                        }
                    })
                    .collect()
            }
            MemoryConfiguration::Custom { pool_options } => pool_options,
        };

        logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                let mut msg = String::new();
                for pool in pool_options.iter() {
                    msg += &format!("[{}] Using memory pool: \n {pool:?}", options.name);
                }
                msg
            },
        );

        let pools: Vec<_> = pool_options
            .iter()
            .map(|options| match options.pool_type {
                PoolType::SlicedPages {
                    page_size,
                    max_slice_size,
                } => DynamicPool::Sliced(SlicedPool::new(
                    page_size,
                    max_slice_size,
                    properties.alignment,
                )),
                PoolType::ExclusivePages { max_alloc_size } => {
                    DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                        max_alloc_size,
                        properties.alignment,
                        options.dealloc_period.unwrap_or(u64::MAX),
                    ))
                }
            })
            .collect();

        let config = GlobalConfig::get().memory.persistent_memory.clone();

        let mode = match options.memory {
            MemoryAllocationOption::Provided(mode) => mode,
            MemoryAllocationOption::FromConfig => match config {
                PersistentMemory::Enabled => MemoryAllocationMode::Auto,
                PersistentMemory::Disabled => MemoryAllocationMode::Auto,
                PersistentMemory::Enforced => MemoryAllocationMode::Persistent,
            },
        };

        Self {
            name: options.name,
            persistent: PersistentPool::new(properties.max_page_size, properties.alignment),
            pools,
            storage,
            alloc_reserve_count: 0,
            mode,
            config,
            logger,
            alignment: properties.alignment,
            capture: None,
            capture_sealed: HashMap::new(),
            capture_pools: HashMap::new(),
            pool_refcount: HashMap::new(),
            graph_to_pool: HashMap::new(),
        }
    }

    /// Change the mode of allocation.
    pub fn mode(&mut self, mode: MemoryAllocationMode) {
        // We override the mode based on the cubecl config.
        let mode = match self.config {
            PersistentMemory::Enabled => mode,
            PersistentMemory::Disabled | PersistentMemory::Enforced => return,
        };

        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                format!(
                    "[{}] Setting memory allocation mode: from {:?} => {mode:?}",
                    self.name, self.mode
                )
            },
        );
        self.mode = mode;
    }

    /// Begin a graph-capture arena session (component C2). A fresh, growable [`CaptureArena`] is
    /// installed; from now until [`Self::capture_arena_seal`]/[`Self::capture_arena_abort`] every
    /// [`Self::reserve`] is served from it.
    ///
    /// Unlike [`Self::mode`], this is NOT gated by the `persistent_memory` config — the arena is a
    /// separate, opt-in mechanism, so capture works regardless of that setting (a silent no-op here
    /// would corrupt replays). The expected flow is: `begin` -> run the closure eagerly to grow the
    /// arena to the peak-live working set -> [`Self::capture_arena_lock`] -> `cuStreamBeginCapture`
    /// -> run the closure again (recycles, zero new allocation) -> `cuStreamEndCapture` ->
    /// [`Self::capture_arena_seal`].
    pub fn capture_arena_begin(&mut self) {
        self.capture = Some(CaptureArena::new(self.alignment));
    }

    /// Lock the active capture arena: no further growth (called right before the CUDA capture window
    /// opens, so the window issues zero `malloc_async`).
    pub fn capture_arena_lock(&mut self) {
        if let Some(arena) = self.capture.as_mut() {
            arena.lock();
        }
    }

    /// Whether a capture arena is currently recording.
    pub fn capture_arena_active(&self) -> bool {
        self.capture.is_some()
    }

    /// Whether the active capture arena is locked (inside the real CUDA capture window).
    pub fn capture_arena_locked(&self) -> bool {
        self.capture.as_ref().map(|a| a.is_locked()).unwrap_or(false)
    }

    /// TODO(P3): copy `data` into the active arena's persistent host keepalive and return a stable
    /// source pointer for a captured H2D memcpy node (the source must outlive the graph). Returns
    /// `None` if no arena is active.
    ///
    /// DEAD until P3. Routes to [`CaptureArena::keepalive`], whose backing store is pageable and so
    /// unsafe to record during capture (it can invalidate the capture). The CUDA backend hard-errors
    /// on non-empty device staging during capture instead of calling this; it is kept (behind
    /// `#[allow(dead_code)]`) as the seam the pinned P3 implementation will fill.
    #[allow(dead_code)]
    pub fn capture_arena_keepalive(&mut self, data: &[u8]) -> Option<*const u8> {
        self.capture.as_mut().map(|arena| arena.keepalive(data))
    }

    /// Seal the active arena to `graph_id` (keep its device blocks alive at fixed addresses for the
    /// graph's lifetime) and stop serving reserves from it.
    pub fn capture_arena_seal(&mut self, graph_id: u64) {
        if let Some(arena) = self.capture.take() {
            self.capture_sealed.insert(graph_id, arena);
        }
    }

    /// Register a fresh SHARED capture pool (P4) with refcount 1 (held by the
    /// [`crate::client::CapturePoolHandle`]). The arena is created lazily on the first
    /// [`Self::capture_pool_begin`].
    pub fn capture_pool_create(&mut self, pool_id: u64) {
        self.pool_refcount.insert(pool_id, 1);
    }

    /// Install the shared pool `pool_id` as the active capture arena for the next warmup/capture pass.
    /// Takes the pool's arena out of [`Self::capture_pools`] (creating a fresh shared arena the first
    /// time) and re-opens it for growth — a new bucket may need more blocks than the pool holds. The
    /// pass ends with [`Self::capture_pool_seal`], which returns the (grown) arena to the pool.
    pub fn capture_pool_begin(&mut self, pool_id: u64) {
        let mut arena = self
            .capture_pools
            .remove(&pool_id)
            .unwrap_or_else(|| CaptureArena::new_shared(self.alignment));
        arena.unlock();
        self.capture = Some(arena);
    }

    /// Return the active arena to the shared pool `pool_id` and attach `graph_id` to it (refcount += 1).
    /// The arena's blocks stay alive (shared by every graph in the pool) until the pool refcount hits 0.
    pub fn capture_pool_seal(&mut self, pool_id: u64, graph_id: u64) {
        if let Some(arena) = self.capture.take() {
            self.capture_pools.insert(pool_id, arena);
            self.graph_to_pool.insert(graph_id, pool_id);
            *self.pool_refcount.entry(pool_id).or_insert(0) += 1;
        }
    }

    /// Drop the pool's handle ref (the [`crate::client::CapturePoolHandle`] was dropped): refcount -= 1,
    /// freeing the pool's device blocks if it reaches 0 (i.e. every graph was already destroyed too).
    pub fn capture_pool_release(&mut self, pool_id: u64) {
        self.pool_decref(pool_id);
    }

    /// Decrement a shared pool's refcount and free its arena (all device blocks) once it hits 0.
    fn pool_decref(&mut self, pool_id: u64) {
        let remaining = match self.pool_refcount.get_mut(&pool_id) {
            Some(rc) => {
                *rc = rc.saturating_sub(1);
                *rc
            }
            None => return,
        };
        if remaining == 0 {
            self.pool_refcount.remove(&pool_id);
            if let Some(mut arena) = self.capture_pools.remove(&pool_id) {
                arena.free(&mut self.storage);
                self.storage.flush();
            }
        }
    }

    /// Intern a per-launch dynamic-metadata blob in the active capture arena by CONTENT (the P-final
    /// capture unblock — see [`CaptureArena::intern_metadata`]). Returns `None` if no capture arena is
    /// active (caller uses the normal staging path), else `Some((slice_handle, needs_h2d))`.
    pub fn capture_arena_intern_metadata(
        &mut self,
        data: &[u8],
    ) -> Option<Result<(SliceHandle, bool), IoError>> {
        // Disjoint field borrows: `&mut self.capture` and `&mut self.storage` are distinct fields.
        let arena = self.capture.as_mut()?;
        Some(arena.intern_metadata(&mut self.storage, data))
    }

    /// Abort the active arena (error/unwind path): free its device blocks and stop serving reserves
    /// from it. No-op if none is active.
    ///
    /// NON-POOLED only. The active arena owns ITS OWN blocks (no sealed graph depends on them), so
    /// freeing them here is correct. A POOLED capture must use [`Self::capture_pool_abort`] instead —
    /// its active arena is the SHARED pool arena, which may already hold earlier sealed graphs' baked
    /// blocks (freeing them would be a use-after-free on the next replay).
    pub fn capture_arena_abort(&mut self) {
        if let Some(mut arena) = self.capture.take() {
            arena.free(&mut self.storage);
            self.storage.flush();
        }
    }

    /// Abort an in-progress POOLED (P4 shared-pool) capture (error/unwind path). Unlike
    /// [`Self::capture_arena_abort`], this must NOT blindly `free()` the active arena: when
    /// [`Self::capture_pool_begin`] installs a shared pool, the active arena ALREADY holds every
    /// EARLIER sealed bucket's device blocks (baked WRITABLE/READ into those still-live graphs). If a
    /// later bucket's warmup/capture aborts, `free()`-ing the arena would dealloc those baked VAs ->
    /// replaying an earlier bucket reads/writes FREED device memory (`CUDA_ERROR_ILLEGAL_ADDRESS`),
    /// and the next [`Self::capture_pool_begin`] would recycle the freed `StorageId`s into a new
    /// bucket -> silent cross-graph clobber.
    ///
    /// So:
    /// * If the pool already has a SEALED graph (refcount > 1: the handle's `1` plus >=1 sealed
    ///   graph), DO NOT free — return the arena to [`Self::capture_pools`] UNMODIFIED. The aborted
    ///   bucket's just-added blocks stay in the (shared) arena and are released only when the whole
    ///   pool is freed (handle dropped AND every graph destroyed). This is a bounded, temporary
    ///   over-retention, never a UAF; those blocks were never baked into any live graph, so a later
    ///   bucket may even reuse them. The partial CUgraph is discarded by the backend.
    /// * If this is the FIRST bucket (refcount == 1, no sealed graph depends on the arena yet), free
    ///   it — matches [`Self::capture_arena_abort`]'s behavior (nothing baked into a live graph). The
    ///   next [`Self::capture_pool_begin`] lazily re-creates a fresh shared arena.
    pub fn capture_pool_abort(&mut self, pool_id: u64) {
        if let Some(mut arena) = self.capture.take() {
            let has_sealed_graph = self.pool_refcount.get(&pool_id).copied().unwrap_or(0) > 1;
            if has_sealed_graph {
                // Earlier sealed graphs' blocks live in this arena: returning it UNMODIFIED keeps
                // their baked VAs valid. (Defensive: never overwrite an arena already parked under
                // this id — `capture_pool_begin` always `remove`d it, so the slot is empty here.)
                self.capture_pools.entry(pool_id).or_insert(arena);
            } else {
                arena.free(&mut self.storage);
                self.storage.flush();
            }
        }
    }

    /// Free a sealed arena (its captured graph was destroyed), releasing all its device blocks. For a
    /// graph that belongs to a SHARED pool (P4) this only drops the graph's pool ref — the pool's
    /// blocks are shared with its other graphs and freed only when the last ref goes (see
    /// [`Self::pool_decref`]).
    pub fn capture_arena_free(&mut self, graph_id: u64) {
        if let Some(pool_id) = self.graph_to_pool.remove(&graph_id) {
            self.pool_decref(pool_id);
            return;
        }
        if let Some(mut arena) = self.capture_sealed.remove(&graph_id) {
            arena.free(&mut self.storage);
            self.storage.flush();
        }
    }

    /// Device bytes reserved by the arena backing `graph_id` (its peak-live high-water mark) — the
    /// SHARED pool's reservation for a pooled graph, the per-graph arena otherwise — or the active
    /// arena's current reservation if `graph_id` is unknown and a capture is in progress.
    pub fn capture_arena_bytes(&self, graph_id: u64) -> u64 {
        if let Some(pool_id) = self.graph_to_pool.get(&graph_id) {
            if let Some(arena) = self.capture_pools.get(pool_id) {
                return arena.reserved_bytes();
            }
        }
        if let Some(arena) = self.capture_sealed.get(&graph_id) {
            return arena.reserved_bytes();
        }
        self.capture.as_ref().map(|a| a.reserved_bytes()).unwrap_or(0)
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self, explicit: bool) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled) && explicit,
            || "Manual memory cleanup ...".to_string(),
        );

        self.persistent
            .cleanup(&mut self.storage, self.alloc_reserve_count, explicit);

        for pool in self.pools.iter_mut() {
            pool.cleanup(&mut self.storage, self.alloc_reserve_count, explicit);
        }
    }

    /// Returns the storage from the specified binding
    pub fn get(&mut self, binding: SliceBinding) -> Option<StorageHandle> {
        if let Some(val) = self.persistent.get(&binding) {
            return Some(val.clone());
        }

        // Capture-arena resources (component C2). Only consulted when a capture is active or sealed
        // arenas exist (both empty in the common path -> no overhead). SliceIds are globally unique,
        // so there is never a collision with the general pools.
        if let Some(arena) = &self.capture {
            if let Some(val) = arena.get(&binding) {
                return Some(val.clone());
            }
        }
        if !self.capture_sealed.is_empty() {
            for arena in self.capture_sealed.values() {
                if let Some(val) = arena.get(&binding) {
                    return Some(val.clone());
                }
            }
        }
        if !self.capture_pools.is_empty() {
            for arena in self.capture_pools.values() {
                if let Some(val) = arena.get(&binding) {
                    return Some(val.clone());
                }
            }
        }

        self.pools.iter().find_map(|p| p.get(&binding)).cloned()
    }

    /// Returns the resource from the storage at the specified handle
    pub fn get_resource(
        &mut self,
        binding: SliceBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Option<Storage::Resource> {
        let handle = self.get(binding);

        handle.map(|handle| {
            let handle = match offset_start {
                Some(offset) => handle.offset_start(offset),
                None => handle,
            };
            let handle = match offset_end {
                Some(offset) => handle.offset_end(offset),
                None => handle,
            };
            self.storage().get(&handle)
        })
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<SliceHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        // Capture-arena interception (component C2): while a CUDA graph is being captured, ALL
        // allocations are served from the isolated, graph-private arena (recycled within the graph,
        // never returned to the general pool). This is what makes replay memory-safe: the device VAs
        // the graph nodes bake stay valid and exclusively owned for the graph's whole lifetime.
        // Disjoint field borrows: `&mut self.capture` and `&mut self.storage` are distinct fields.
        if let Some(arena) = self.capture.as_mut() {
            return arena.reserve(&mut self.storage, size);
        }

        if let Some(val) = self.persistent.try_reserve(size) {
            self.logger.log_memory(
                |level| matches!(level, MemoryLogLevel::Full),
                || {
                    format!(
                        "[{}] Reserved memory {size} using persistent memory",
                        self.name
                    )
                },
            );
            return Ok(val);
        }

        if matches!(self.mode, MemoryAllocationMode::Persistent) || self.persistent.has_size(size) {
            let allocated = self.persistent.alloc(&mut self.storage, size);

            self.logger.log_memory(
                |level| !matches!(level, MemoryLogLevel::Disabled),
                || {
                    format!(
                        "[{}] Allocated a new memory page using persistent memory, \n{}",
                        self.name, self,
                    )
                },
            );
            return allocated;
        }

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}] Reserved memory {} using dynamic pool",
                    self.name,
                    BytesFormat::new(size)
                )
            },
        );

        // Find first pool that fits this allocation
        let pool = self
            .pools
            .iter_mut()
            .find(|p| p.accept(size))
            .ok_or(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            })?;

        if let Some(slice) = pool.try_reserve(size) {
            return Ok(slice);
        }

        let allocated = pool.alloc(&mut self.storage, size);

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}], Allocated a new memory page, current usage: \n{}",
                    self.name, self
                )
            },
        );

        allocated
    }

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](ComputeStorage::alloc) and [dealloc](ComputeStorage::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    pub fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get the current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        let memory_usage = self.pools.iter().map(|x| x.get_memory_usage()).fold(
            MemoryUsage {
                number_allocs: 0,
                bytes_in_use: 0,
                bytes_padding: 0,
                bytes_reserved: 0,
            },
            |m1, m2| m1.combine(m2),
        );
        let mut memory_usage = memory_usage.combine(self.persistent.get_memory_usage());

        // Include capture-arena device reservations (component C2) so they are visible to
        // `memory_usage()` — e.g. a leak test that captures + destroys graphs and watches the
        // reserved bytes return to baseline. Arenas allocate directly from storage (bypassing the
        // pools), so they would otherwise be invisible here.
        let arena_bytes = self.capture.as_ref().map(|a| a.reserved_bytes()).unwrap_or(0)
            + self
                .capture_sealed
                .values()
                .map(|a| a.reserved_bytes())
                .sum::<u64>()
            + self
                .capture_pools
                .values()
                .map(|a| a.reserved_bytes())
                .sum::<u64>();
        if arena_bytes > 0 {
            memory_usage = memory_usage.combine(MemoryUsage {
                number_allocs: 0,
                bytes_in_use: arena_bytes,
                bytes_padding: 0,
                bytes_reserved: arena_bytes,
            });
        }
        memory_usage
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_usage());
    }
}
impl<Storage: ComputeStorage> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_str("\n## Dynamic\n\n")?;

        for pool in self.pools.iter() {
            match pool {
                DynamicPool::Sliced(pool) => f.write_fmt(format_args!("{pool}\n"))?,
                DynamicPool::Exclusive(pool) => f.write_fmt(format_args!("{pool}\n"))?,
            }
        }
        let memory_usage = self.memory_usage();
        f.write_fmt(format_args!("\n## Summary\n\n{memory_usage}"))?;

        Ok(())
    }
}

impl<Storage> core::fmt::Debug for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?}",
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};
    use alloc::vec;

    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties {
        max_page_size: 128 * 1024 * 1024,
        alignment: 32,
    };

    fn options() -> MemoryManagementOptions {
        MemoryManagementOptions {
            name: "test".into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let max_page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: max_page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(100);
        let usage = memory_management.memory_usage();

        assert_eq!(usage.bytes_in_use, 100);
        assert!(usage.bytes_reserved >= 100 && usage.bytes_reserved <= max_page_size);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100);
        let usage_new = memory_management.memory_usage();
        assert_eq!(usage, usage_new);
    }

    #[test_log::test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_allocs_new_storage() {
        let page_size = 1024;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test_log::test]
    fn alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn allocs_on_correct_page() {
        let sizes = [100, 200, 300, 400];

        let pools = sizes
            .iter()
            .map(|size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: *size,
                    max_slice_size: *size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));

        let usage = memory_management.memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert!(usage.bytes_reserved >= sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        // Usage after can actually be _less_ because of defragging.
        assert!(usage_before.bytes_reserved >= usage_after.bytes_reserved);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_fragmentation_resistance() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size).unwrap())
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size).unwrap();
        }
        let usage_after = memory_management.memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test_log::test]
    fn noslice_test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &(MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            }),
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test_log::test]
    fn noslice_alloc_two_chunk() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert!(usage.bytes_reserved >= alloc_size);
    }

    #[test_log::test]
    fn noslice_alloc_allocs_new_storage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 50 * 20,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 60 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn noslice_allocs_on_correct_page() {
        let pools = [100, 200, 300, 400]
            .iter()
            .map(|&size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: size,
                    max_slice_size: size,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));
        let usage = memory_management.memory_usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }

    fn pool_mm() -> MemoryManagement<BytesStorage> {
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        )
    }

    // FIX 1 (P4 shared-pool abort-path USE-AFTER-FREE): aborting a LATER bucket's capture must NOT
    // free the shared arena, because it already holds an EARLIER sealed graph's baked device blocks.
    // If it did, replaying the earlier bucket would read/write freed device memory (illegal address).
    #[test_log::test]
    fn capture_pool_abort_preserves_sealed_graph_blocks() {
        let mut mm = pool_mm();
        let pool = 0u64;
        mm.capture_pool_create(pool);

        // Bucket A: reserve a block in the shared pool, then seal it to graph 100.
        mm.capture_pool_begin(pool);
        let a = mm.reserve(4096).unwrap();
        let a_binding = a.clone().binding();
        drop(a); // the captured graph "owns" the block; the warmup handle is dropped (block now free)
        mm.capture_pool_seal(pool, 100);
        assert_eq!(*mm.pool_refcount.get(&pool).unwrap(), 2); // handle (1) + sealed graph 100 (1)
        assert!(mm.get(a_binding.clone()).is_some());
        let bytes_after_a = mm.capture_arena_bytes(100);
        assert!(bytes_after_a >= 4096);

        // Bucket B: re-open the shared pool, reserve a fresh block, then ABORT (simulating a panic
        // mid-capture: OOM / locked-metadata miss / launch failure).
        mm.capture_pool_begin(pool);
        let b = mm.reserve(2048).unwrap();
        drop(b);
        mm.capture_pool_abort(pool);

        // The shared arena must be back in the pool with bucket A's block INTACT (not freed -> no UAF),
        // and the refcount untouched (handle + graph 100 still live).
        assert!(
            mm.get(a_binding).is_some(),
            "sealed bucket A's block was freed on a later bucket's abort -> use-after-free"
        );
        assert!(mm.capture_pools.get(&pool).is_some(), "shared arena was not returned to the pool");
        assert!(mm.capture_arena_bytes(100) >= bytes_after_a);
        assert_eq!(*mm.pool_refcount.get(&pool).unwrap(), 2);

        // And releasing the whole pool (handle + graph) DOES free it (the over-retained bucket-B block
        // included) — no permanent leak.
        mm.capture_arena_free(100); // graph destroyed: refcount 2 -> 1
        mm.capture_pool_release(pool); // handle dropped: refcount 1 -> 0, arena freed
        assert!(mm.capture_pools.get(&pool).is_none());
        assert!(mm.pool_refcount.get(&pool).is_none());
    }

    // FIX 1, first-bucket case: when NO graph is sealed yet (refcount == 1, just the handle), aborting
    // the capture is safe to free the arena — nothing baked into a live graph depends on it. (This is
    // the only case the old free-everything abort was actually correct for.)
    #[test_log::test]
    fn capture_pool_abort_first_bucket_frees() {
        let mut mm = pool_mm();
        let pool = 0u64;
        mm.capture_pool_create(pool);

        mm.capture_pool_begin(pool);
        let b = mm.reserve(4096).unwrap();
        let binding = b.clone().binding();
        assert!(mm.get(binding.clone()).is_some());
        drop(b);
        mm.capture_pool_abort(pool);

        // First-bucket abort: the (unsealed) block is freed, no arena parked, pool handle still alive.
        assert!(
            mm.get(binding).is_none(),
            "first-bucket abort should free its unsealed blocks"
        );
        assert!(mm.capture_pools.get(&pool).is_none());
        assert_eq!(*mm.pool_refcount.get(&pool).unwrap(), 1);

        // The pool is still usable: a fresh bucket can be captured into it afterwards.
        mm.capture_pool_begin(pool);
        let _c = mm.reserve(4096).unwrap();
        mm.capture_pool_seal(pool, 200);
        assert_eq!(*mm.pool_refcount.get(&pool).unwrap(), 2);
    }
}
