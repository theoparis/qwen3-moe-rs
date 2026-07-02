use super::memory_pool::{Slice, SliceBinding, SliceHandle, SliceId, calculate_padding};
use crate::{
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};
use alloc::vec::Vec;
use cubecl_common::backtrace::BackTrace;
use hashbrown::HashMap;

/// A graph-private capture ARENA (component C2 of the CUDA-graph plan).
///
/// PyTorch `CUDACachingAllocator::capture_begin` private-pool model: while a CUDA graph is being
/// captured, every device allocation is served from THIS isolated pool instead of the general
/// allocator, and freed blocks are RECYCLED within the pool (so peak memory == peak-LIVE working
/// set, NOT the sum of all allocations). The pool's blocks are never returned to the general pool;
/// they stay locked at fixed device addresses for the captured graph's whole lifetime, so the
/// virtual addresses baked into the graph nodes stay valid across unlimited replays.
///
/// # Sizing (no `malloc_async` inside the capture window)
///
/// During a WARMUP / measure pass (`allow_grow == true`, NO CUDA capture active) the capture
/// closure is run once eagerly; the arena grows by `storage.alloc` to its peak-live working set.
/// The arena is then LOCKED (`allow_grow == false`) and the real capture pass runs: every
/// allocation finds a recycled block (the warmup pre-sized them), so ZERO `malloc_async` happens
/// inside the capture window -> no graph mem-alloc nodes -> a `flags = 0` `cuGraphInstantiate`
/// stays valid (no `AUTO_FREE_ON_LAUNCH` needed). An allocation that overflows a locked arena is a
/// hard error rather than a silent corruption.
///
/// # Recycling
///
/// A size-bucketed free-list (one device buffer per block, keyed by padded size), mirroring
/// [`crate::memory_management::memory_pool`]'s `PersistentPool`. A freed block (its [`SliceHandle`]
/// fully dropped, `is_free()`) is re-handed to the next same-size request. This is the
/// "size-bucketed free-list" the design calls for; it is simpler and more VA-stable than a single
/// contiguous bump buffer, because each block keeps its own fixed `StorageId` (hence fixed device
/// pointer) for the graph lifetime.
///
/// # Allocation model & tradeoff (FIX 3 — P2+ optimization opportunity)
///
/// This is one DRIVER allocation (`storage.alloc`) per block, bucketed by padded size. It is chosen
/// for being VA-stable and dead simple — each block's `StorageId` (device pointer) is fixed for the
/// graph's whole life, exactly what replay needs. The costs, fine for a decode step's handful of
/// intermediates but NOT for model-scale capture, are:
///
/// - [`Self::reserve`] does an O(N) linear scan of the same-size bucket for a free block, so a bucket
///   holding thousands of identically-sized blocks makes a capture pass that reserves them O(N²).
/// - One driver alloc per block means many small device allocations and the fragmentation that comes
///   with them at scale.
///
/// A single contiguous slab / page arena (sub-allocated by offset, VA still stable because the slab
/// base is fixed) is the P2+ optimization that removes both. It is deliberately deferred: the current
/// model is correct and adequate for the targeted workload (a decode step), and the slab variant is a
/// pure performance change behind the same interface.
pub struct CaptureArena {
    /// Every block ever minted in this arena, keyed by its slice id.
    slices: HashMap<SliceId, Slice>,
    /// Free-list buckets: padded block size -> slice ids of that size.
    sizes: HashMap<u64, Vec<SliceId>>,
    /// TODO(P3): persistent host sources for H2D staging recorded as graph memcpy nodes. The captured
    /// node bakes the *host source pointer*; it must outlive the graph, so the staged bytes would be
    /// copied here (kept alive until the arena is freed on graph destroy).
    ///
    /// DEAD until P3. This `Vec<Vec<u8>>` is PAGEABLE; a pageable `cuMemcpyHtoDAsync` recorded inside
    /// the capture window can synchronize internally and INVALIDATE the capture (null graph -> wedge).
    /// PyTorch stages from PINNED host memory specifically to avoid this. Until [`Self::keepalive`] is
    /// reimplemented over a pinned staging buffer AND validated on non-grid-constant HW (where scalars
    /// are NOT baked by value and so actually hit this device-staging route), the CUDA backend
    /// HARD-ERRORS on any non-empty device staging during capture instead of using this. See
    /// `cubecl-cuda` `command.rs::create_with_data`.
    #[allow(dead_code)]
    host_keepalive: Vec<Vec<u8>>,
    /// Content-addressed cache of staged dynamic per-launch METADATA (`Sequence<FastDivmod>` shapes/
    /// strides etc.) — the P-final unblock for capturing real Burn ops below Fusion. Every Burn op
    /// whose metadata exceeds the by-value grid-constant static portion stages a small device buffer
    /// per launch via `create_with_data`; doing that H2D inside the locked capture window is
    /// uncapturable (bakes a stale host source). BUT for a fixed-shape captured region the metadata is
    /// IDENTICAL across replays (it is shape-derived; only device-buffer CONTENTS change), so we stage
    /// each distinct blob exactly ONCE during warmup into a RETAINED arena block (its `SliceHandle`
    /// held here -> never recycled -> stable VA) and, on the locked capture pass, return that same
    /// buffer BY CONTENT with zero H2D. The captured kernel just reads the stable-VA metadata buffer;
    /// replay re-reads the unchanged bytes. No pinned source, no memcpy-during-capture (cf. the
    /// `host_keepalive` pinned-staging route, still deferred). A locked-pass content MISS is a hard
    /// error (the warmup did not pre-stage it => non-deterministic metadata).
    meta_cache: HashMap<Vec<u8>, SliceHandle>,
    alignment: u64,
    /// `true` during the warmup/measure pass (may grow via `storage.alloc`); `false` inside the
    /// actual CUDA capture window (overflow is a hard error instead of a `malloc_async` graph node).
    allow_grow: bool,
    /// Total device bytes reserved by this arena (== peak-live high-water once warmed up).
    reserved: u64,
    /// SHARED-POOL mode (P4, vLLM `graph_pool_handle`). When `true`, one arena backs several captured
    /// graphs of DIFFERENT shapes (e.g. prompt-length buckets) that replay SERIALLY, costing ~1 graph's
    /// high-water instead of N×. It changes two recycling rules vs the single-graph [`Self::new`] arena:
    ///
    /// 1. **Oversize block reuse** ([`Self::reserve`]): a request with no exact-size free block may
    ///    recycle the smallest free block that is at least as large (best-fit), so the largest bucket
    ///    (captured FIRST) sizes the pool and smaller buckets reuse its blocks instead of growing it.
    /// 2. **Fresh retained-metadata blocks** ([`Self::intern_metadata`]): a per-graph RETAINED metadata
    ///    block is always GROWN fresh, never recycled from the free list. A recycled block is still
    ///    baked WRITABLE into the earlier graph's nodes; reusing it for a later graph's READ-ONLY
    ///    metadata would let the earlier graph's replay CLOBBER that metadata (-> garbage strides ->
    ///    out-of-bounds). Regular intermediates are safe to share because every graph WRITES them
    ///    before reading on each replay; metadata is the one read-only-retained exception.
    ///
    /// SOUND ONLY for serially-replayed graphs (a block is baked into many graphs; two concurrent
    /// replays clobber it).
    shared: bool,
}

impl CaptureArena {
    /// Create a fresh, empty arena that may grow (warmup phase). Single-graph (pool-of-one): strict
    /// exact-size recycling, sealed to ONE captured graph id.
    pub fn new(alignment: u64) -> Self {
        Self {
            slices: HashMap::new(),
            sizes: HashMap::new(),
            host_keepalive: Vec::new(),
            meta_cache: HashMap::new(),
            alignment: alignment.max(1),
            allow_grow: true,
            reserved: 0,
            shared: false,
        }
    }

    /// Create a fresh SHARED-POOL arena (P4 — vLLM `graph_pool_handle`). Identical to [`Self::new`]
    /// but with the shared-pool recycling rules, so several DIFFERENT-shape captured graphs that replay
    /// SERIALLY can share it at ~1 graph's high-water. See [`Self::shared`].
    pub fn new_shared(alignment: u64) -> Self {
        let mut arena = Self::new(alignment);
        arena.shared = true;
        arena
    }

    /// Re-open a (previously locked) arena for growth. Called when a shared pool is re-installed for
    /// the NEXT bucket's warmup/capture pass: the prior pass locked it, but the new bucket may need
    /// more (or differently-sized) blocks than the pool currently holds.
    pub fn unlock(&mut self) {
        self.allow_grow = true;
    }

    /// Intern a per-launch dynamic-METADATA blob by CONTENT (the P-final capture unblock). Returns the
    /// arena handle backing `data` plus whether the caller must perform the H2D into it:
    /// * HIT  -> `(handle, false)`: this exact blob was already staged (a prior warmup pass); the
    ///   device bytes are already correct, so the caller writes NOTHING (crucial: on the locked capture
    ///   pass this is the only allowed outcome — zero H2D inside the capture window).
    /// * MISS during warmup (`allow_grow`) -> `(handle, true)`: a fresh RETAINED block is reserved and
    ///   remembered by content; the caller does the eager H2D once (not capturing).
    /// * MISS while locked -> hard error: the capture pass needs metadata the warmup never staged, so
    ///   the recorded region would depend on an un-staged buffer (non-deterministic shapes). This is a
    ///   SAFE, LOUD ABORT (no silent corruption): a common trigger is AUTOTUNE resolving a *different*
    ///   kernel between warmup and the capture pass (which changes the staged metadata blob) — the fix is
    ///   MORE warmup iterations to settle autotune before capture, NOT a logic change. Interning is
    ///   FIXED-TRACE / FIXED-SHAPE only by construction.
    pub fn intern_metadata<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        data: &[u8],
    ) -> Result<(SliceHandle, bool), IoError> {
        if let Some(handle) = self.meta_cache.get(data) {
            return Ok((handle.clone(), false));
        }
        if !self.allow_grow {
            return Err(IoError::Unknown {
                description: alloc::format!(
                    "CUDA-graph capture metadata miss: a launch inside the capture window staged a \
                     {}-byte dynamic-metadata blob the warmup/measure pass never produced. The \
                     captured region cannot depend on an un-staged buffer (its content is baked at \
                     warmup). Make the closure's shapes deterministic across warmup + capture, or add \
                     more warmup iterations.",
                    data.len()
                ),
                backtrace: BackTrace::capture(),
            });
        }
        // Warmup miss: reserve a dedicated block and RETAIN its handle here so it is never recycled
        // (its content must stay valid + stable-VA for the graph's whole life). The caller H2Ds once.
        //
        // In a SHARED pool the metadata block must be GROWN FRESH, never recycled from the free list:
        // a recycled free block is still baked WRITABLE into an earlier graph captured into this pool,
        // and retaining it here as READ-ONLY metadata would let that earlier graph's replay CLOBBER it
        // (garbage strides -> out-of-bounds). Regular intermediates are safe to share (every graph
        // rewrites them before reading on each replay); read-only retained metadata is the exception.
        let handle = if self.shared {
            let padded = self.padded(data.len() as u64);
            self.grow(storage, data.len() as u64, padded)?
        } else {
            self.reserve(storage, data.len() as u64)?
        };
        self.meta_cache.insert(data.to_vec(), handle.clone());
        Ok((handle, true))
    }

    /// Lock the arena: subsequent allocations must be served by recycling (no growth). Called right
    /// before `cuStreamBeginCapture` so the capture window issues zero `malloc_async`.
    pub fn lock(&mut self) {
        self.allow_grow = false;
    }

    /// Whether the arena is locked (i.e. inside the real CUDA capture window).
    pub fn is_locked(&self) -> bool {
        !self.allow_grow
    }

    /// Total device bytes reserved (the peak-live high-water mark).
    pub fn reserved_bytes(&self) -> u64 {
        self.reserved
    }

    /// Padded size used for a request, never zero (a zero-size staging buffer still needs a stable
    /// address, so it is rounded up to one alignment unit).
    fn padded(&self, size: u64) -> u64 {
        (size + calculate_padding(size, self.alignment)).max(self.alignment)
    }

    /// Serve an allocation from the arena: recycle a free same-size block if one exists, otherwise
    /// grow (warmup only). Mirrors `MemoryManagement::reserve` for the capture path.
    pub fn reserve<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<SliceHandle, IoError> {
        let padded = self.padded(size);

        // Recycle: any free block of exactly this padded size keeps the same StorageId -> same VA.
        if let Some(ids) = self.sizes.get(&padded) {
            for id in ids {
                let slice = self.slices.get(id).expect("arena slice to exist");
                if slice.is_free() {
                    return Ok(slice.handle.clone());
                }
            }
        }

        // SHARED-POOL (P4): no exact-size block is free, so recycle the SMALLEST free block that is
        // at least as large as the request (best-fit). A smaller bucket then runs on a prefix of a
        // larger bucket's block — VA-stable (the block keeps its StorageId) and zero-growth, which is
        // exactly how K serially-replayed buckets share one pool at ~1 bucket's high-water. The block
        // still reports its OWN (larger) storage size; the tensor only ever touches its logical bytes
        // (same as ordinary alignment padding), so this is sound.
        if self.shared {
            let mut best: Option<(u64, SliceId)> = None;
            for slice in self.slices.values() {
                if slice.is_free() {
                    // Use the block's TRUE capacity — the padded driver allocation, i.e.
                    // `storage.size()` — NOT `effective_size()`. `grow` sets `storage.size() == padded`
                    // AND `padding == padded - size`, so `effective_size() == 2*padded - size`
                    // OVERSTATES the real capacity. For nonzero requests the overstatement is
                    // `< alignment` (harmless), but a ZERO-size reserve makes `padded == align` and
                    // `padding == align`, so `effective_size() == 2*align`; without this fix a free
                    // 0-byte block could be best-fit-selected to back a request in `(align, 2*align]`
                    // -> up to `align` bytes of out-of-bounds device write. `storage.size()` is the
                    // exact number of device bytes the block actually owns, so it is always safe.
                    let cap = slice.storage.size();
                    if cap >= padded && best.map(|(b, _)| cap < b).unwrap_or(true) {
                        best = Some((cap, slice.id()));
                    }
                }
            }
            if let Some((_, id)) = best {
                return Ok(self.slices.get(&id).expect("arena slice to exist").handle.clone());
            }
        }

        self.grow(storage, size, padded)
    }

    /// Allocate a BRAND-NEW block of `padded` bytes (one driver alloc, its own fixed StorageId -> fixed
    /// VA for the graph life) and register it. Errors if the arena is locked (growth inside the capture
    /// window would inject an uncapturable `cuMemAllocAsync` node). `size` is the caller's request (for
    /// the error message + padding bookkeeping).
    fn grow<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
        padded: u64,
    ) -> Result<SliceHandle, IoError> {
        if !self.allow_grow {
            return Err(IoError::Unknown {
                description: alloc::format!(
                    "CUDA-graph capture arena overflow: an allocation of {size} bytes (padded \
                     {padded}) appeared inside the capture window that the warmup/measure pass did \
                     not pre-size (no free block of that size). Growing here would inject a \
                     `cuMemAllocAsync` graph node whose matching free never fires. Run the capture \
                     closure with more warmup iterations, or make its allocation pattern \
                     deterministic so the warmup pass reserves the same blocks."
                ),
                backtrace: BackTrace::capture(),
            });
        }

        // Grow: one device buffer per block (its own fixed StorageId -> fixed VA for graph life).
        let storage_handle = storage.alloc(padded)?;
        let slice_handle = SliceHandle::new();
        let padding = padded - size;
        let slice = Slice::new(storage_handle, slice_handle.clone(), padding);
        let sid = slice.id();
        self.sizes.entry(padded).or_default().push(sid);
        self.slices.insert(sid, slice);
        self.reserved += padded;
        Ok(slice_handle)
    }

    /// Resolve a binding to its (stable) storage handle. Consulted by `MemoryManagement::get` while
    /// a capture is in progress so the captured kernels can find their arena-backed resources.
    pub fn get(&self, binding: &SliceBinding) -> Option<&StorageHandle> {
        self.slices.get(binding.id()).map(|slice| &slice.storage)
    }

    /// TODO(P3): copy `data` into a persistent host buffer owned by the arena and return a stable
    /// pointer to it, to make a captured H2D memcpy node's *source* outlive the graph (PyTorch's
    /// static pinned-staging model). The returned pointer would be valid until [`Self::free`].
    ///
    /// DEAD until P3, and NOT safe as written: the backing store is PAGEABLE (`data.to_vec()`), but a
    /// pageable async H2D recorded during capture can synchronize internally and invalidate the
    /// capture. A correct implementation must stage into PINNED host memory and be validated on
    /// non-grid-constant HW. Until then the CUDA backend hard-errors rather than calling this; see
    /// `cubecl-cuda` `command.rs::create_with_data`.
    #[allow(dead_code)]
    pub fn keepalive(&mut self, data: &[u8]) -> *const u8 {
        self.host_keepalive.push(data.to_vec());
        // The inner Vec's heap buffer is stable across outer-Vec reallocations.
        self.host_keepalive.last().unwrap().as_ptr()
    }

    /// Free every device block (and host keepalive) owned by the arena. Called on graph destroy /
    /// capture abort. The caller is responsible for `storage.flush()`.
    pub fn free<Storage: ComputeStorage>(&mut self, storage: &mut Storage) {
        for slice in self.slices.values() {
            storage.dealloc(slice.storage.id);
        }
        self.slices.clear();
        self.sizes.clear();
        self.host_keepalive.clear();
        self.meta_cache.clear();
        self.reserved = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::BytesStorage;

    /// The padded reservation `grow` makes for `size` at the given alignment (== `storage.size()`).
    fn padded_of(size: u64, align: u64) -> u64 {
        (size + calculate_padding(size, align)).max(align)
    }

    // FIX 2 (zero-size best-fit OOB): a free ZERO-size block reserves `padded == align` device bytes
    // but reports `effective_size() == 2*align` (its `padding == align` is added on top of the
    // `storage.size() == align`). Best-fit MUST compare against the block's TRUE capacity
    // (`storage.size()`), not `effective_size()`, or a request in `(align, 2*align]` could be
    // best-fit-served from the 0-byte block -> up to `align` bytes of out-of-bounds device write.
    #[test_log::test]
    fn shared_zero_size_block_not_oversize_reused() {
        let align = 64u64;
        let mut storage = BytesStorage::default();
        let mut arena = CaptureArena::new_shared(align);

        // Reserve a zero-size block (padded to `align`) and free it.
        let h0 = arena.reserve(&mut storage, 0).unwrap();
        drop(h0);
        assert_eq!(arena.reserved, align);
        assert!(arena.slices.values().any(|s| s.is_free()));

        // Request `align + 1` bytes -> padded `2*align`. The 0-byte block's real capacity is `align`,
        // which is < the request, so best-fit must REJECT it and grow a fresh `2*align` block.
        let before = arena.reserved;
        let h1 = arena.reserve(&mut storage, align + 1).unwrap();
        assert_eq!(
            arena.reserved,
            before + 2 * align,
            "the 0-byte block was wrongly oversize-reused (no growth) -> OOB device write"
        );
        // The block actually handed out is large enough for the request.
        let id = *h1.id();
        assert!(arena.slices.get(&id).unwrap().storage.size() >= align + 1);
    }

    // FIX 2 must NOT break the normal (nonzero) oversize best-fit: a smaller bucket still reuses a
    // larger free block with ZERO growth (the whole point of the shared pool).
    #[test_log::test]
    fn shared_oversize_best_fit_still_reuses_larger_block() {
        let align = 64u64;
        let mut storage = BytesStorage::default();
        let mut arena = CaptureArena::new_shared(align);

        // Bucket A: a large block, then freed.
        let big = arena.reserve(&mut storage, 1000).unwrap(); // padded -> 1024
        let big_id = *big.id();
        drop(big);
        let reserved_after_big = arena.reserved;
        assert_eq!(reserved_after_big, padded_of(1000, align));

        // Bucket B: a smaller request reuses the larger free block (best-fit) -> no growth, same VA.
        let small = arena.reserve(&mut storage, 500).unwrap(); // padded -> 512
        assert_eq!(
            arena.reserved, reserved_after_big,
            "smaller bucket should reuse the larger free block instead of growing"
        );
        assert_eq!(*small.id(), big_id, "best-fit should hand back the larger block's stable id");
    }
}
