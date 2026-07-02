use super::storage::gpu::{GpuResource, GpuStorage};
use crate::{
    CudaCompiler,
    compute::{
        command::{Command, write_to_cpu},
        context::CudaContext,
        stream::CudaStreamBackend,
        sync::Fence,
    },
};
use cubecl_common::{
    backtrace::BackTrace, bytes::Bytes, profile::ProfileDuration, stream_id::StreamId,
};
use cubecl_core::{
    MemoryConfiguration,
    future::{self, DynFut},
    ir::{ElemType, FloatKind, IntKind, MemoryDeviceProperties, StorageType, UIntKind},
    prelude::*,
    server::{
        Allocation, AllocationDescriptor, AllocationKind, Binding, Bindings, CopyDescriptor,
        ExecutionError, IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication,
        ServerUtilities, TensorMapBinding, TensorMapMeta,
    },
    zspace::{Shape, Strides, strides},
};
use cubecl_runtime::{
    compiler::CubeTask,
    config::GlobalConfig,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryUsage, offset_handles, optimal_align},
    server::{self, ComputeServer},
    storage::BindingResource,
    stream::MultiStream,
};
use cudarc::driver::sys::{
    CUcontext, CUresult, CUtensorMapDataType, CUtensorMapFloatOOBfill, CUtensorMapInterleave,
    CUtensorMapL2promotion, CUtensorMapSwizzle, cuCtxEnablePeerAccess, cuTensorMapEncodeIm2col,
    cuTensorMapEncodeTiled,
};
use std::{collections::HashMap, ffi::c_void, mem::MaybeUninit, sync::Arc};

pub(crate) const MB: usize = 1024 * 1024;

/// Hard-stop (FIX 2) for launch paths that cannot be captured into a CUDA graph in P0: staging a
/// scalar / dynamic-metadata / any `cuMemAlloc`-backed buffer during capture allocates a transient
/// device buffer (its VA is returned to the pool and baked into the graph node) and does a pageable
/// async H2D from a host pointer freed once `launch` returns -> use-after-free / VA corruption at
/// replay. Panic loudly instead of silently corrupting the graph; the P1 capture arena lifts this.
#[cold]
#[inline(never)]
#[track_caller]
fn panic_alloc_during_capture(what: &str) -> ! {
    panic!(
        "{what} during CUDA-graph capture is not supported until the P1 capture arena: it \
         allocates a transient device buffer (its VA is freed back to the pool yet baked into the \
         graph node) and does a pageable async H2D from a host pointer freed after `launch` \
         (use-after-free at replay). Pre-stage scalars/metadata in persistent buffers, or resolve \
         them at comptime, before capturing."
    );
}

/// A captured + instantiated CUDA graph owned by the server (component C1 of the CUDA-graph plan).
///
/// The raw handles are `!Send` `*mut`s, but [`CudaServer`] is `unsafe impl Send` and all access is
/// serialized under the device lock, mirroring how the rest of the server treats its raw CUDA state.
#[derive(Debug)]
struct CapturedCudaGraph {
    graph: cudarc::driver::sys::CUgraph,
    exec: cudarc::driver::sys::CUgraphExec,
    /// The stream the graph was captured on. Stored so [`ComputeServer::graph_destroy`] (which takes
    /// no `stream_id`) can reach the same per-stream memory management to free the graph's sealed
    /// capture arena (component C2).
    stream_id: StreamId,
}

#[derive(Debug)]
pub struct CudaServer {
    ctx: CudaContext,
    streams: MultiStream<CudaStreamBackend>,
    peer_activated: bool,
    mem_alignment: usize,
    utilities: Arc<ServerUtilities<Self>>,
    /// Instantiated graphs captured via [`ComputeServer::capture_begin`]/`capture_end`, keyed by the
    /// opaque id handed back to the client.
    graphs: HashMap<u64, CapturedCudaGraph>,
    next_graph_id: u64,
    /// Set between `capture_begin` and `capture_end` so the launch path can hard-reject
    /// `CubeCount::Dynamic` (which needs a host read that poisons capture).
    capturing: bool,
    /// Monotonic id source for SHARED capture pools (P4).
    next_pool_id: u64,
    /// The shared pool the in-progress capture (if any) is being recorded INTO. When `capture_end`
    /// finds it `Some`, it seals the arena into that pool (P4) instead of into a per-graph arena.
    active_pool: Option<u64>,
}

unsafe impl Send for CudaServer {}

impl ComputeServer for CudaServer {
    type Kernel = Box<dyn CubeTask<CudaCompiler>>;
    type Storage = GpuStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        let mut command = self.command_no_inputs(stream_id);

        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size, true, None))
            .collect())
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let mut command = self.command(stream_id, descriptors.iter().map(|d| &d.binding));

        Box::pin(command.read_async(descriptors))
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut strides = Vec::new();
        let mut sizes = Vec::new();
        let mut total_size = 0;

        for descriptor in descriptors {
            let last_dim = descriptor.shape.last().copied().unwrap_or(1);
            let pitch_align = match descriptor.kind {
                AllocationKind::Contiguous => 1,
                AllocationKind::Optimized => {
                    optimal_align(last_dim, descriptor.elem_size, self.mem_alignment)
                }
            };

            let rank = descriptor.shape.len();
            let width = *descriptor.shape.last().unwrap_or(&1);
            let height: usize = descriptor.shape.iter().rev().skip(1).product();
            let height = Ord::max(height, 1);
            let width_bytes = width * descriptor.elem_size;
            let pitch = width_bytes.next_multiple_of(pitch_align);
            let size = height * pitch;
            total_size += size.next_multiple_of(self.mem_alignment);
            let mut stride = strides![1; rank];
            if rank > 1 {
                stride[rank - 2] = pitch / descriptor.elem_size;
            }
            if rank > 2 {
                for i in (0..rank - 2).rev() {
                    stride[i] = stride[i + 1] * descriptor.shape[i + 1];
                }
            }

            strides.push(stride);
            sizes.push(size);
        }

        let mem_alignment = self.mem_alignment;
        let mut command = self.command_no_inputs(stream_id);

        let handle = command.reserve(total_size as u64)?;
        let handles = offset_handles(handle, &sizes, mem_alignment);

        Ok(handles
            .into_iter()
            .zip(strides)
            .map(|(handle, strides)| Allocation::new(handle, strides))
            .collect())
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        let mut command = self.command(stream_id, descriptors.iter().map(|desc| &desc.0.binding));

        for (descriptor, data) in descriptors {
            command.write_to_gpu(descriptor, data)?;
        }

        Ok(())
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let mut kernel_id = kernel.id();
        let logger = self.streams.logger.clone();
        kernel_id.mode(mode);
        let grid_constants = self
            .ctx
            .compilation_options
            .supports_features
            .grid_constants;
        // Captured-region guard (C1): a `Dynamic` count does a host `block_on` read (below) to fetch
        // the grid dims, which poisons an in-progress capture. Hard-reject it instead.
        let capturing = self.capturing;
        let mut command = self.command(stream_id, bindings.buffers.iter());
        // Whether the graph-private capture arena (component C2) is active. When it is, the
        // DESTINATION of a staged buffer is arena-served (stable VA for the graph lifetime, recycled
        // within the graph) — that is the real C2 win and lifts the coarse P0 panic guards below
        // (which only fire when capturing WITHOUT the arena). But the H2D SOURCE staging is a separate
        // concern: persisting a non-empty host source across the graph lifetime needs PINNED memory
        // and is a TODO(P3) (FIX 2), so `create_with_data` HARD-ERRORS on any NON-EMPTY device staging
        // during capture. Only the harmless empty/0-byte metadata case (the common grid-constant path
        // the C2 bench hits) proceeds. Net: the panic guards gate "arena vs no arena"; the finer
        // "non-empty source not yet supported" check lives in `create_with_data`.
        let arena_active = command.capture_arena_active();

        let count = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            // TODO: CUDA doesn't have an exact equivalent of dynamic dispatch. Instead, kernels are free to launch other kernels.
            // One option is to create a dummy kernel with 1 thread that launches the real kernel with the dynamic dispatch settings.
            // For now, just read the dispatch settings from the buffer.
            CubeCount::Dynamic(binding) => {
                if capturing {
                    return Err(LaunchError::Unknown {
                        reason: "CubeCount::Dynamic cannot be captured into a CUDA graph: it \
                                 requires a host read of the dispatch dims, which poisons capture. \
                                 Resolve the count to CubeCount::Static before capturing."
                            .into(),
                        backtrace: BackTrace::capture(),
                    });
                }
                let data = future::block_on(command.read_async(vec![CopyDescriptor::new(
                    binding,
                    &[3],
                    &[1],
                    4,
                )]))?;
                let data = bytemuck::cast_slice(&data[0]);
                assert!(
                    data.len() == 3,
                    "Dynamic cube count should contain 3 values"
                );
                (data[0], data[1], data[2])
            }
        };

        let (scalars, scalar_bindings) = if grid_constants {
            let mut scalars = Vec::with_capacity(bindings.scalars.len() + 1);
            // We need to sort by largest first to have proper packed alignment. Assumes device
            // pointers are 64-bit aligned, which I believe is true on all cards that support grid
            // constants regardless. Metadata is inserted after the 8-aligned scalars to ensure proper
            // packing
            //
            // Capture note (FIX 2): the `scalars`/`metadata` host pointers pushed below are passed as
            // `__grid_constant__` kernel params. During capture the driver copies the *pointee value*
            // into the kernel node at record time (the host pointer itself is NOT stored), so a host
            // pointer freed after `launch` is fine — this is capture-safe and is the path the P0 bench
            // exercises (its bit-identical replay empirically confirms the values are baked at
            // capture). The genuinely uncapturable path is the *dynamic-metadata device staging*
            // below, which is guarded.
            for binding in bindings.scalars.values().filter(|it| it.ty.size() == 8) {
                scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
            }
            if bindings.metadata.static_len > 0 {
                scalars.push(bindings.metadata.data.as_ptr() as *const _ as *mut c_void);
            }
            for size in [4, 2, 1] {
                for binding in bindings.scalars.values().filter(|it| it.ty.size() == size) {
                    scalars.push(binding.data.as_ptr() as *const _ as *mut c_void);
                }
            }

            let mut handles = Vec::new();
            if bindings.metadata.static_len > 0 {
                let bytes_offs = bindings.metadata.static_len * kernel.address_type().size();
                let dyn_meta = &bytemuck::cast_slice(&bindings.metadata.data)[bytes_offs..];
                // FIX 2: staging dynamic metadata allocates a transient device buffer and does a
                // pageable async H2D from a host pointer freed after launch -> corrupts the graph at
                // replay. Hard-reject during capture UNLESS the C2 capture arena is active (then the
                // dest is arena-backed with a stable VA and the H2D source is persisted). Empty
                // `dyn_meta` is a no-op (no real staging); that's the case the scalar-free P0 bench
                // hits.
                if capturing && !arena_active && !dyn_meta.is_empty() {
                    panic_alloc_during_capture("dynamic tensor-metadata staging");
                }
                handles.push(command.create_with_data(dyn_meta)?);
            }

            (scalars, handles)
        } else {
            // Non-grid-constant hardware: ALL metadata and scalars are staged into transient device
            // buffers via `create_with_data` (alloc + async H2D from a soon-freed host pointer), so
            // none of it is capturable until the P1 arena. Hard-reject any such staging during
            // capture (FIX 2).
            let mut handles = Vec::new();
            if !bindings.metadata.data.is_empty() {
                if capturing && !arena_active {
                    panic_alloc_during_capture("tensor-metadata staging");
                }
                handles
                    .push(command.create_with_data(bytemuck::cast_slice(&bindings.metadata.data))?)
            }
            for binding in bindings.scalars.values() {
                if capturing && !arena_active {
                    panic_alloc_during_capture("scalar-argument staging");
                }
                handles.push(command.create_with_data(binding.data())?);
            }
            (Vec::new(), handles)
        };

        let mut resources = bindings
            .tensor_maps
            .iter()
            .map(|it| it.binding.clone())
            .chain(bindings.buffers)
            .map(|binding| command.resource(binding).expect("Resource to exist."))
            .collect::<Vec<_>>();

        let mut tensor_maps = Vec::with_capacity(bindings.tensor_maps.len());

        for TensorMapBinding { map, binding } in bindings.tensor_maps.into_iter() {
            let resource = command
                .resource(binding)
                .expect("Tensor map resource exists.");
            let device_ptr = resource.ptr as *mut c_void;

            let mut map_ptr = MaybeUninit::zeroed();

            let shape: Vec<_> = map
                .metadata
                .shape()
                .iter()
                .rev()
                .map(|s| *s as u64)
                .collect();
            let strides: Vec<_> = map
                .metadata
                .strides()
                .iter()
                .rev()
                .skip(1)
                .map(|s| *s as u64 * map.storage_ty.size() as u64)
                .collect();
            let elem_stride: Vec<_> = map.elem_stride.iter().rev().map(|s| *s as u32).collect();

            match &map.format {
                TensorMapFormat::Tiled(TiledArgs { tile_size }) => unsafe {
                    let tile_size: Vec<_> =
                        tile_size.iter().rev().copied().map(|s| s as u32).collect();

                    cuTensorMapEncodeTiled(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        tile_size.as_ptr(),
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_tiled(&map, &tile_size).err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                TensorMapFormat::Im2col(args) => unsafe {
                    let lower_corner: Vec<_> =
                        args.pixel_box_lower_corner.iter().rev().copied().collect();
                    let upper_corner: Vec<_> =
                        args.pixel_box_upper_corner.iter().rev().copied().collect();

                    cuTensorMapEncodeIm2col(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        lower_corner.as_ptr(),
                        upper_corner.as_ptr(),
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        let tiled_err = check_tma_im2col(
                            &map,
                            &lower_corner,
                            &upper_corner,
                            args.channels_per_pixel,
                            args.pixels_per_column,
                        )
                        .err();
                        generic_err
                            .or(tiled_err)
                            .unwrap_or_else(|| LaunchError::Unknown {
                                reason: format!("{err}"),
                                backtrace: BackTrace::capture(),
                            })
                    })?;
                },
                #[cfg(cuda_12080)]
                TensorMapFormat::Im2colWide(args) => unsafe {
                    use cudarc::driver::sys::{
                        CUtensorMapIm2ColWideMode, cuTensorMapEncodeIm2colWide,
                    };
                    cuTensorMapEncodeIm2colWide(
                        map_ptr.as_mut_ptr(),
                        elem_to_tensor_map_type(map.storage_ty),
                        map.metadata.rank() as u32,
                        device_ptr,
                        shape.as_ptr(),
                        strides.as_ptr(),
                        args.pixel_box_lower_corner_width,
                        args.pixel_box_upper_corner_width,
                        args.channels_per_pixel,
                        args.pixels_per_column,
                        elem_stride.as_ptr(),
                        interleave_to_cuda(map.interleave),
                        CUtensorMapIm2ColWideMode::CU_TENSOR_MAP_IM2COL_WIDE_MODE_W,
                        swizzle_to_cuda(map.swizzle),
                        prefetch_to_cuda(map.prefetch),
                        oob_to_cuda(map.oob_fill),
                    )
                    .result()
                    .map_err(|err| {
                        let generic_err =
                            check_tma_generic(&map, device_ptr, &shape, &strides, &elem_stride)
                                .err();
                        generic_err.unwrap_or_else(|| LaunchError::Unknown {
                            reason: format!("{err}"),
                            backtrace: BackTrace::capture(),
                        })
                    })?;
                },
                #[cfg(not(cuda_12080))]
                TensorMapFormat::Im2colWide(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "CUDA version 12.8 required for tensor map format Im2colWide"
                            .into(),
                        backtrace: BackTrace::capture(),
                    });
                }
            };
            let binding = unsafe { map_ptr.assume_init() };
            tensor_maps.push(binding);
        }

        resources.extend(
            scalar_bindings
                .into_iter()
                .map(|s| command.resource(s.binding()).expect("Resource to exist")),
        );

        command.kernel(
            kernel_id,
            kernel,
            mode,
            count,
            &tensor_maps,
            &resources,
            &scalars,
            logger,
        )?;

        Ok(())
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        let mut command = self.command_no_inputs(stream_id);
        command.sync()
    }

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            log::warn!("{err}");
        }

        self.ctx.timestamps.start()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.ctx.timestamps.error(err.into());
        }
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: server::Binding,
        stream_id: StreamId,
    ) -> BindingResource<GpuResource> {
        let mut command = self.command(stream_id, [&binding].into_iter());

        BindingResource::new(
            binding.clone(),
            command.resource(binding).expect("Failed to find resource"),
        )
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> MemoryUsage {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.memory_cleanup()
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.allocation_mode(mode)
    }

    fn capture_arena_begin(&mut self, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.capture_arena_begin();
    }

    fn capture_arena_lock(&mut self, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.capture_arena_lock();
    }

    fn capture_pool_create(&mut self, stream_id: StreamId) -> u64 {
        let id = self.next_pool_id;
        self.next_pool_id += 1;
        let mut command = self.command_no_inputs(stream_id);
        command.capture_pool_create(id);
        id
    }

    fn capture_pool_begin(&mut self, pool_id: u64, stream_id: StreamId) {
        // Record which pool this capture is going into; `capture_end` seals the arena back here.
        self.active_pool = Some(pool_id);
        let mut command = self.command_no_inputs(stream_id);
        command.capture_pool_begin(pool_id);
    }

    fn capture_pool_release(&mut self, pool_id: u64, stream_id: StreamId) {
        let mut command = self.command_no_inputs(stream_id);
        command.capture_pool_release(pool_id);
    }

    fn graph_arena_bytes(&mut self, graph_id: u64, stream_id: StreamId) -> u64 {
        let mut command = self.command_no_inputs(stream_id);
        command.capture_arena_bytes(graph_id)
    }

    fn capture_begin(&mut self, stream_id: StreamId) {
        let sys = self.stream_sys(stream_id);
        unsafe {
            cudarc::driver::result::stream::begin_capture(
                sys,
                cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            )
            .expect("cuStreamBeginCapture failed");
        }
        self.capturing = true;
    }

    fn capture_end(&mut self, stream_id: StreamId) -> u64 {
        let sys = self.stream_sys(stream_id);
        self.capturing = false;

        let graph = unsafe {
            cudarc::driver::result::stream::end_capture(sys).expect("cuStreamEndCapture failed")
        };
        assert!(
            !graph.is_null(),
            "Captured CUDA graph is null/empty: no kernel nodes were recorded. The captured work \
             must target the same stream that begin/end_capture run on, with no host sync inside."
        );

        // Instantiate with flags = 0 (the cudarc `result::graph::instantiate` enum has no zero
        // variant, so call the raw FFI to pass 0). The P0 captured region is allocation-free, so
        // there are no mem-alloc nodes that would need AUTO_FREE_ON_LAUNCH.
        let exec = unsafe {
            let mut exec = MaybeUninit::uninit();
            cudarc::driver::sys::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), graph, 0)
                .result()
                .expect("cuGraphInstantiateWithFlags failed");
            exec.assume_init()
        };

        let id = self.next_graph_id;
        self.next_graph_id += 1;

        // C2/P4: seal the capture arena. Its device blocks stay alive at fixed addresses (so the VAs
        // the graph nodes baked stay valid). For a SHARED pool capture (`active_pool` set by
        // `capture_pool_begin`) the arena returns to the pool, shared with its other graphs (freed
        // when the pool's last ref goes); otherwise it is a per-graph arena freed on `graph_destroy`.
        {
            let active_pool = self.active_pool.take();
            let mut command = self.command_no_inputs(stream_id);
            match active_pool {
                Some(pool_id) => command.capture_pool_seal(pool_id, id),
                None => command.capture_arena_seal(id),
            }
        }

        self.graphs.insert(
            id,
            CapturedCudaGraph {
                graph,
                exec,
                stream_id,
            },
        );
        id
    }

    fn capture_abort(&mut self, stream_id: StreamId) {
        // Unwind/error path (FIX 1): the closure passed to `ComputeClient::capture` panicked or
        // errored between `capture_begin` and `capture_end`. If we did nothing the stream would stay
        // stuck in capture mode (every later op on it would error) and `capturing` would stay `true`
        // forever. End the capture to pull the stream back out of capture mode and DISCARD the
        // partial graph; never instantiate it.
        let sys = self.stream_sys(stream_id);
        self.capturing = false;
        // P4 (FIX 1): take the shared-pool intent so a later capture does not mis-seal, AND route the
        // arena teardown by whether this was a pooled capture. The active arena IS the shared pool's
        // arena, which already holds every EARLIER sealed bucket's baked device blocks — so a blind
        // `capture_arena_abort` (free-all) here would dealloc those still-live VAs: replaying an
        // earlier bucket would then hit `CUDA_ERROR_ILLEGAL_ADDRESS`, and the next `capture_pool_begin`
        // would recycle the freed StorageIds into a new bucket (silent cross-graph clobber). The old
        // "not a leak, just restart the pool" reasoning is only valid for the FIRST bucket (refcount 1,
        // nothing sealed yet); `capture_pool_abort` distinguishes the two cases (free iff first bucket,
        // else return the arena to the pool UNMODIFIED so the sealed graphs stay intact).
        let active_pool = self.active_pool.take();

        // C2/P4: discard any active capture arena. Safe no-op if none active.
        {
            let mut command = self.command_no_inputs(stream_id);
            match active_pool {
                Some(pool_id) => command.capture_pool_abort(pool_id),
                None => command.capture_arena_abort(),
            }
        }

        unsafe {
            // `end_capture` returns the partial graph AND takes the stream out of capture mode even
            // when the capture was invalidated (it then yields a null graph). Either way the stream
            // is usable again afterwards; we just throw the graph away.
            match cudarc::driver::result::stream::end_capture(sys) {
                Ok(graph) if !graph.is_null() => {
                    if let Err(err) = cudarc::driver::result::graph::destroy(graph) {
                        log::warn!("capture_abort: discarding partial CUDA graph failed: {err}");
                    }
                }
                Ok(_) => {}
                Err(err) => {
                    // Capture was already invalidated, or the stream wasn't capturing (e.g. the
                    // panic happened inside `capture_begin`). The stream is left non-capturing
                    // regardless; nothing else to clean up.
                    log::warn!(
                        "capture_abort: cuStreamEndCapture during abort returned an error \
                         (stream is left non-capturing): {err}"
                    );
                }
            }
        }
    }

    fn graph_replay(&mut self, graph_id: u64, stream_id: StreamId) {
        let sys = self.stream_sys(stream_id);
        let exec = self
            .graphs
            .get(&graph_id)
            .expect("Replay of an unknown/destroyed graph id")
            .exec;
        unsafe {
            cudarc::driver::result::graph::launch(exec, sys).expect("cuGraphLaunch failed");
        }
    }

    fn graph_destroy(&mut self, graph_id: u64) {
        // FIX 3: bind this server's CUDA context current before touching the graph objects.
        // `graph_replay` gets this for free via `stream_sys()` (which calls `unsafe_set_current`),
        // but `graph_destroy` takes no `stream_id` and `CapturedGraph` is `Send`, so its `Drop` can
        // run on a different thread where no context is current. Without binding the context first,
        // `cuGraphExecDestroy`/`cuGraphDestroy` could run under the wrong/no context.
        self.unsafe_set_current();
        if let Some(g) = self.graphs.remove(&graph_id) {
            unsafe {
                cudarc::driver::result::graph::exec_destroy(g.exec)
                    .expect("cuGraphExecDestroy failed");
                cudarc::driver::result::graph::destroy(g.graph).expect("cuGraphDestroy failed");
            }
            // C2: free the graph's sealed capture arena (its device blocks were held alive at fixed
            // addresses for the graph's lifetime). Uses the stream the graph was captured on.
            let mut command = self.command_no_inputs(g.stream_id);
            command.capture_arena_free(graph_id);
        }
    }
}

impl ServerCommunication for CudaServer {
    const SERVER_COMM_ENABLED: bool = true;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(server_src, server_dst, src))
    )]
    fn copy(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        if server_src.peer_activated {
            Self::change_server_peer(server_src, server_dst, src, stream_id_src, stream_id_dst)
        } else {
            Self::change_server_serialized(
                server_src,
                server_dst,
                src,
                stream_id_src,
                stream_id_dst,
            )
        }
    }
}

impl CudaServer {
    /// Create a new cuda server.
    pub(crate) fn new(
        ctx: CudaContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        mem_alignment: usize,
        device_id: i32,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        ctx.unsafe_set_current().unwrap();

        let peer_activated = enable_one_way_peer_access(ctx.context).is_ok();
        if peer_activated {
            log::info!("Peer data transfer activated for device {device_id}");
        } else {
            log::info!("Peer data transfer not available for device {device_id}");
        }

        Self {
            mem_alignment,
            ctx,
            peer_activated,
            streams: MultiStream::new(
                utilities.logger.clone(),
                CudaStreamBackend::new(
                    mem_props,
                    mem_config,
                    mem_alignment,
                    utilities.logger.clone(),
                ),
                max_streams,
            ),
            utilities: Arc::new(utilities),
            graphs: HashMap::new(),
            next_graph_id: 0,
            capturing: false,
            next_pool_id: 0,
            active_pool: None,
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(server_src, server_dst, src))
    )]
    fn change_server_peer(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let strides = src.strides.into();
        let binding = src.binding.clone();

        let context_src = server_src.ctx.context;
        let context_dst = server_dst.ctx.context;

        // We create a command on the source server to retrieve the correct resource from the
        // source memory pools. We also make sure the current stream is aligned with the stream of
        // the binding, where the data was first allocated.
        let mut command_src = server_src.command(stream_id_src, [&src.binding].into_iter());
        let resource_src = command_src.resource(binding.clone())?;
        let stream_src = command_src.streams.current().sys;
        let fence_src = Fence::new(stream_src);

        // We need to free the command before creating another one.
        core::mem::drop(command_src);

        // We create a new command on the destination server to reserve the necessary GPU memory
        // and wait on the source server, making sure the execution is updated. Then, we perform
        // the peer memcpy on the destination server.
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let stream_dst = command_dst.streams.current().sys;

        let handle = command_dst.reserve(binding.size())?;
        let resource_dst = command_dst.resource(handle.clone().binding())?;
        fence_src.wait_async(stream_dst);

        unsafe {
            cudarc::driver::sys::cuMemcpyPeerAsync(
                resource_dst.ptr,
                context_dst,
                resource_src.ptr,
                context_src,
                binding.size() as usize,
                stream_dst,
            )
            .result()
            .expect("Peer communication should be activated");
        }

        // We drop the last command.
        core::mem::drop(command_dst);

        Ok(Allocation { handle, strides })
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(server_src, server_dst, src))
    )]
    #[allow(unused)]
    fn change_server_serialized(
        server_src: &mut Self,
        server_dst: &mut Self,
        src: CopyDescriptor<'_>,
        stream_id_src: StreamId,
        stream_id_dst: StreamId,
    ) -> Result<Allocation, IoError> {
        let shape: Shape = src.shape.into();
        let strides: Strides = src.strides.into();
        let elem_size = src.elem_size;
        let binding = src.binding.clone();
        let num_bytes = shape.iter().product::<usize>() * elem_size;

        // ACTIVE: command_src
        let mut command_src = server_src.command(stream_id_src, [&src.binding].into_iter());
        let stream_src = command_src.streams.current().sys;
        let resource_src = command_src.resource(binding.clone())?;

        // ACTIVE: command_dst
        let mut command_dst = server_dst.command_no_inputs(stream_id_dst);
        let stream_dst = command_dst.streams.current().sys;

        // the cpu buffer sequences before the destination resource,
        // be cause we don't need the destination resource to exist
        // until after the post src>cpu write fence.
        let mut cpu_buffer = command_dst.reserve_cpu(num_bytes, true, None);

        let handle_dst = command_dst.reserve(binding.size())?;
        let resource_dst = command_dst.resource(handle_dst.clone().binding())?;

        // Since the `cpu_buffer` lives in the destination stream timeline,
        // we ensure that the allocation of that copy buffer is complete
        // in both timelines before we permit the source stream to proceed.
        Fence::new(stream_dst).wait_async(stream_src);

        // TODO: Interleave
        // The entire src>cpu completes before cpu>dst starts,
        // meaning that half of our hardware comms busses are unused
        // at any given moment.
        //
        // By splitting this into k chunked writes, we could sequence:
        // - src[0] > cpu[0]
        // - src[1] > cpu[1]; cpu[0] > dst[0]
        // - src[2] > cpu[2]; cpu[1] > dst[1]
        // - ...
        // - src[k-1] > cpu[k-1]; cpu[k-2] > dst[k-2]
        // - cpu(k-1) > dst(k-1)
        //
        // By selecting a buffer with j slots, j<<k; we can restrict
        // to an active stream with no more than j active at a time;
        // reducing cpu buffer usage.
        //
        // The entire rotation can be scheduled via fence events,
        // and delegated to the stream management after this method exits.
        //
        // # Challenge 1: Chunk Size Selection
        // On a given machine, there is a "good" chunk size. It depends
        // upon the host plane and memory, as well as the GPUs. We can
        // probably tune to a good size, but some form of active global
        // active policy lookup to get the size could be useful here.
        //
        // # Challenge 2: Sharding
        // To leverage the existing write machinery, we don't want to
        // change the shape of tensors; so shard selection should be
        // taking contiguous slices of one dimension.
        //
        // The dimension to slice, and how to slice it, should be
        // selected based upon the target chunk size.

        command_src.unsafe_set_current();
        unsafe {
            write_to_cpu(
                &shape,
                &strides,
                elem_size,
                &mut cpu_buffer,
                resource_src.ptr,
                stream_src,
            )?;
        }

        // stream_dst waits until the stream_src write is sequenced.
        Fence::new(stream_src).wait_async(stream_dst);
        core::mem::drop(command_src);

        // ACTIVE: command_dst
        command_dst.unsafe_set_current();
        unsafe {
            write_to_gpu(
                &shape,
                &strides,
                elem_size,
                &cpu_buffer,
                resource_dst.ptr,
                stream_dst,
            )
        }?;

        core::mem::drop(cpu_buffer);
        core::mem::drop(command_dst);

        Ok(Allocation {
            handle: handle_dst,
            strides,
        })
    }

    fn command_no_inputs(&mut self, stream_id: StreamId) -> Command<'_> {
        self.command(stream_id, [].into_iter())
    }

    /// The raw `CUstream` for `stream_id` (the same stream the eager `launch` path targets), used by
    /// the capture/replay FFI. Binds the CUDA context to the thread as a side effect.
    fn stream_sys(&mut self, stream_id: StreamId) -> cudarc::driver::sys::CUstream {
        let mut command = self.command_no_inputs(stream_id);
        command.streams.current().sys
    }

    fn unsafe_set_current(&self) {
        self.ctx.unsafe_set_current().unwrap();
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        bindings: impl Iterator<Item = &'a Binding>,
    ) -> Command<'_> {
        self.unsafe_set_current();
        let streams = self.streams.resolve(stream_id, bindings);

        Command::new(&mut self.ctx, streams)
    }
}

fn elem_to_tensor_map_type(ty: StorageType) -> CUtensorMapDataType {
    use cudarc::driver::sys::CUtensorMapDataType::*;
    match ty {
        // packed fp4 should be treated as single 4-bit values to simplify indexing/shape handling
        // So a tile of width 16 with fp4 elements is 8 x fp4x2 elements wide.
        #[cfg(cuda_12080)]
        StorageType::Packed(ty, 2) if ty.size_bits() == 4 => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
        StorageType::Scalar(ElemType::Float(kind)) => match kind {
            // There's no special handling for FP8, so load as u8. `0u8 == 0.0` when reinterpreting.
            FloatKind::E2M1 // single fp4s are padded to a full byte
            | FloatKind::E4M3
            | FloatKind::E5M2
            | FloatKind::UE8M0
            | FloatKind::E2M3
            | FloatKind::E3M2 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            FloatKind::F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            FloatKind::BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            FloatKind::Flex32 | FloatKind::F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            FloatKind::TF32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            FloatKind::F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
        },
        StorageType::Scalar(ElemType::Int(kind)) => match kind {
            // UInt is fine because zero bits and size is the same between both
            IntKind::I8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            IntKind::I16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            IntKind::I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            IntKind::I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
        },
        StorageType::Scalar(ElemType::UInt(kind)) => match kind {
            UIntKind::U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            UIntKind::U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            UIntKind::U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            UIntKind::U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
        },
        _ => unimplemented!("Not supported for tensor map type"),
    }
}

fn interleave_to_cuda(interleave: TensorMapInterleave) -> CUtensorMapInterleave {
    use cudarc::driver::sys::CUtensorMapInterleave::*;
    match interleave {
        TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
        TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
        TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
    }
}

fn swizzle_to_cuda(swizzle: TensorMapSwizzle) -> CUtensorMapSwizzle {
    use cudarc::driver::sys::CUtensorMapSwizzle::*;
    match swizzle {
        TensorMapSwizzle::None => CU_TENSOR_MAP_SWIZZLE_NONE,
        TensorMapSwizzle::B32 => CU_TENSOR_MAP_SWIZZLE_32B,
        TensorMapSwizzle::B64 => CU_TENSOR_MAP_SWIZZLE_64B,
        TensorMapSwizzle::B128 => CU_TENSOR_MAP_SWIZZLE_128B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom32BFlip8B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
        #[cfg(cuda_12080)]
        TensorMapSwizzle::B128Atom64B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
        #[cfg(not(cuda_12080))]
        _ => unimplemented!("Swizzle atomicity requires CUDA 12.8 or higher"),
    }
}

fn prefetch_to_cuda(prefetch: TensorMapPrefetch) -> CUtensorMapL2promotion {
    use cudarc::driver::sys::CUtensorMapL2promotion::*;
    match prefetch {
        TensorMapPrefetch::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
        TensorMapPrefetch::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
        TensorMapPrefetch::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        TensorMapPrefetch::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
    }
}

fn oob_to_cuda(fill: OobFill) -> CUtensorMapFloatOOBfill {
    use cudarc::driver::sys::CUtensorMapFloatOOBfill::*;
    match fill {
        OobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        OobFill::NaN => CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA,
    }
}

macro_rules! launch_check {
    ($assertion: expr, $($arg:tt)+) => {
        if $assertion {
            Ok(())
        } else {
            Err(LaunchError::Unknown {
                reason: format!($($arg)*),
                backtrace: BackTrace::capture(),
            })
        }
    };
}

fn check_tma_generic(
    map: &TensorMapMeta,
    device_ptr: *mut c_void,
    shape: &[u64],
    strides: &[u64],
    elem_strides: &[u32],
) -> Result<(), LaunchError> {
    // globalAddress invariants
    launch_check!(
        (device_ptr as usize).is_multiple_of(16),
        "Tensor pointer must be 16 byte aligned"
    )?;
    if !matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            (device_ptr as usize).is_multiple_of(32),
            "Tensor pointer must be 32 byte aligned"
        )?;
    }

    // tensorRank invariants
    launch_check!(
        (1..=5).contains(&map.metadata.rank()),
        "Rank must be between 1 and 5"
    )?;
    launch_check!(
        matches!(map.interleave, TensorMapInterleave::None) || map.metadata.rank() >= 3,
        "When interleave is enabled, rank must be >= 3"
    )?;

    // globalDim invariants
    launch_check!(
        shape.iter().all(|it| *it <= u32::MAX as u64),
        "Shape must be <= u32::MAX"
    )?;
    #[cfg(cuda_12080)]
    if matches!(map.storage_ty, StorageType::Packed(ty, 2) if ty.size_bits() == 4) {
        launch_check!(
            shape[0].is_multiple_of(2),
            "Packed tensor map must have multiple of 2 for the innermost dimension"
        )?;
    }

    // globalStrides invariants
    launch_check!(
        strides.iter().all(|it| it.is_multiple_of(16)),
        "Strides must be 16 byte aligned"
    )?;
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            strides.iter().all(|it| it.is_multiple_of(32)),
            "Strides must be 32 byte aligned when interleave is B32"
        )?;
    }

    // elementStrides invariants
    launch_check!(
        elem_strides.iter().all(|it| *it > 0 && *it <= 8),
        "Element strides must be non-zero and <= 8"
    )?;
    if matches!(map.interleave, TensorMapInterleave::None) {
        launch_check!(
            elem_strides[0] == 1,
            "Innermost element stride is ignored without interleaving"
        )?;
    }

    // oobFill invariants
    if matches!(map.oob_fill, OobFill::NaN) {
        launch_check!(
            map.storage_ty.is_float(),
            "NaN fill is only supported for float types"
        )?;
    }

    Ok(())
}

fn check_tma_tiled(map: &TensorMapMeta, tile_size: &[u32]) -> Result<(), LaunchError> {
    launch_check!(
        tile_size.len() == map.metadata.rank(),
        "Tile shape should match rank"
    )?;
    launch_check!(
        tile_size.iter().all(|it| *it > 0 && *it <= 256),
        "Tile shape must be non-zero and <= 256"
    )?;
    let tile_size_0_bytes = tile_size[0] as usize * map.storage_ty.size();
    if matches!(map.interleave, TensorMapInterleave::None) {
        let max_tile_bytes = match map.swizzle {
            TensorMapSwizzle::None => usize::MAX,
            TensorMapSwizzle::B32 => 32,
            TensorMapSwizzle::B64 => 64,
            TensorMapSwizzle::B128
            | TensorMapSwizzle::B128Atom32B
            | TensorMapSwizzle::B128Atom32BFlip8B
            | TensorMapSwizzle::B128Atom64B => 128,
        };
        launch_check!(
            tile_size_0_bytes <= max_tile_bytes,
            "Innermost tile dim must be <= swizzle size"
        )?;
    }
    if matches!(map.interleave, TensorMapInterleave::B32) {
        launch_check!(
            map.swizzle == TensorMapSwizzle::B32,
            "If interleave is B32, swizzle must be B32"
        )?;
    }

    Ok(())
}

fn check_tma_im2col(
    map: &TensorMapMeta,
    lower_corner: &[i32],
    upper_corner: &[i32],
    channels_per_pixel: u32,
    pixels_per_column: u32,
) -> Result<(), LaunchError> {
    launch_check!(
        lower_corner.len() == map.metadata.rank() - 2,
        "Lower corner must be rank - 2 elements"
    )?;
    launch_check!(
        upper_corner.len() == map.metadata.rank() - 2,
        "Upper corner must be rank - 2 elements"
    )?;

    launch_check!(
        map.metadata.rank() >= 3 && map.metadata.rank() <= 5,
        "im2col requires rank to be between 3 and 5"
    )?;

    let (range_lower, range_upper) = match map.metadata.rank() {
        3 => (-32768, 32767),
        4 => (-128, 127),
        5 => (-16, 15),
        _ => unreachable!(),
    };
    launch_check!(
        lower_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Lower corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;
    launch_check!(
        upper_corner
            .iter()
            .all(|it| *it >= range_lower && *it <= range_upper),
        "Upper corner must be in range [{range_lower}, {range_upper}] for {}D im2col",
        map.metadata.rank()
    )?;

    launch_check!(
        channels_per_pixel <= 256,
        "Channels per pixel must be <= 256"
    )?;
    launch_check!(
        pixels_per_column <= 1024,
        "Pixels per column must be <= 1024"
    )?;

    Ok(())
}

use crate::compute::command::write_to_gpu;
use cudarc::driver::sys::cudaError_enum::CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
use cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS;

fn enable_one_way_peer_access(ctx_src: CUcontext) -> Result<(), CUresult> {
    unsafe {
        match cuCtxEnablePeerAccess(ctx_src, 0) {
            CUDA_SUCCESS | CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED => Ok(()),
            err => Err(err),
        }
    }
}
