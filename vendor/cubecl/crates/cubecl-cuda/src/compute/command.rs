use crate::{
    CudaCompiler,
    compute::{
        MB, context::CudaContext, io::controller::PinnedMemoryManagedAllocController,
        storage::gpu::GpuResource, stream::CudaStreamBackend, sync::Fence,
    },
};
use cubecl_common::{
    backtrace::BackTrace,
    bytes::{AllocationProperty, Bytes},
    stream_id::StreamId,
};
#[cfg(debug_assertions)]
use cubecl_core::zspace::striding::try_check_pitched_row_major_strides;
use cubecl_core::{
    MemoryUsage,
    future::DynFut,
    server::{
        Binding, CopyDescriptor, ExecutionError, ExecutionMode, Handle, IoError, LaunchError,
        ProfileError,
    },
    zspace::striding::has_pitched_row_major_strides,
};
use cubecl_runtime::{
    compiler::CubeTask,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{MemoryAllocationMode, MemoryHandle},
    stream::{GcTask, ResolvedStreams},
};
use cudarc::driver::sys::{
    CUDA_MEMCPY2D_st, CUmemorytype, CUstream_st, CUtensorMap, cuMemcpy2DAsync_v2,
};
use std::{ffi::c_void, ops::DerefMut, sync::Arc};

#[derive(new)]
/// The `Command` struct encapsulates a CUDA context and a set of resolved CUDA streams, providing an
/// interface for executing GPU-related operations such as memory allocation, data transfers, kernel
/// registration, and task execution.
pub struct Command<'a> {
    ctx: &'a mut CudaContext,
    pub(crate) streams: ResolvedStreams<'a, CudaStreamBackend>,
}

impl<'a> Command<'a> {
    /// Retrieves a GPU resource associated with the provided binding.
    ///
    /// # Parameters
    ///
    /// * `binding` - The binding specifying the stream, memory, and offsets for the resource.
    ///
    /// # Returns
    ///
    /// * `Ok(GpuResource)` - The GPU resource associated with the binding.
    /// * `Err(IoError::InvalidHandle)` - If the binding does not correspond to a valid resource.
    pub fn resource(&mut self, binding: Binding) -> Result<GpuResource, IoError> {
        self.streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .ok_or(IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })
    }

    /// Switches the current CUDA context to the one associated with this command.
    ///
    /// Users should not make calls to other [`Command`]s while the context is switched.
    pub fn unsafe_set_current(&self) {
        self.ctx.unsafe_set_current().unwrap();
    }

    /// Retrieves the gpu memory usage of the current stream.
    ///
    /// # Returns
    ///
    /// * The [`MemoryUsage`] struct.
    pub fn memory_usage(&mut self) -> MemoryUsage {
        self.streams.current().memory_management_gpu.memory_usage()
    }

    /// Explicitly cleanup gpu memory on the current stream.
    pub fn memory_cleanup(&mut self) {
        self.streams.current().memory_management_gpu.cleanup(true)
    }

    /// Set the [`MemoryAllocationMode`] for the current stream.
    ///
    /// # Parameters
    ///
    /// * `mode` - The allocation mode to be used.
    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.streams.current().memory_management_gpu.mode(mode)
    }

    /// Begin a graph-capture arena session on the current stream (component C2). Allocations made
    /// while it is active are served from the isolated, graph-private arena.
    pub fn capture_arena_begin(&mut self) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_begin()
    }

    /// Lock the active capture arena (no further growth before the CUDA capture window opens).
    pub fn capture_arena_lock(&mut self) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_lock()
    }

    /// Register a fresh SHARED capture pool (P4) with the given id (held by a `CapturePoolHandle`).
    pub fn capture_pool_create(&mut self, pool_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_pool_create(pool_id)
    }

    /// Install the shared pool `pool_id` as the active capture arena for the next warmup/capture pass.
    pub fn capture_pool_begin(&mut self, pool_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_pool_begin(pool_id)
    }

    /// Return the active arena to shared pool `pool_id` and attach `graph_id` to it.
    pub fn capture_pool_seal(&mut self, pool_id: u64, graph_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_pool_seal(pool_id, graph_id)
    }

    /// Drop the pool handle's ref to `pool_id` (frees the pool once its last graph is destroyed).
    pub fn capture_pool_release(&mut self, pool_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_pool_release(pool_id)
    }

    /// Whether a capture arena is currently recording on the current stream.
    pub fn capture_arena_active(&mut self) -> bool {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_active()
    }

    /// Seal the active arena to `graph_id` (kept alive for the graph's lifetime).
    pub fn capture_arena_seal(&mut self, graph_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_seal(graph_id)
    }

    /// Abort the active arena, freeing its device blocks (error/unwind path). NON-POOLED only.
    pub fn capture_arena_abort(&mut self) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_abort()
    }

    /// Abort an in-progress POOLED (P4 shared-pool) capture: keep the shared arena (with earlier
    /// sealed graphs' baked blocks) alive instead of freeing it, so an earlier bucket's replay does
    /// not hit freed device memory. See `MemoryManagement::capture_pool_abort`.
    pub fn capture_pool_abort(&mut self, pool_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_pool_abort(pool_id)
    }

    /// Free the arena sealed to `graph_id` (its captured graph was destroyed).
    pub fn capture_arena_free(&mut self, graph_id: u64) {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_free(graph_id)
    }

    /// Device bytes reserved by the arena sealed to `graph_id` (its peak-live high-water mark).
    pub fn capture_arena_bytes(&mut self, graph_id: u64) -> u64 {
        self.streams
            .current()
            .memory_management_gpu
            .capture_arena_bytes(graph_id)
    }

    /// Allocates a new GPU memory buffer of the specified size.
    ///
    /// # Parameters
    ///
    /// * `size` - The size of the memory to allocate (in bytes).
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated GPU memory.
    /// * `Err(IoError)` - If the allocation fails.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<Handle, IoError> {
        let handle = self.streams.current().memory_management_gpu.reserve(size)?;

        Ok(Handle::new(
            handle,
            None,
            None,
            self.streams.current,
            self.streams.cursor,
            size,
        ))
    }

    /// Creates a [Bytes] instance from pinned memory, if suitable for the given size.
    ///
    /// For small data transfers (<= 100 MB) or when explicitly marked as pinned, this function
    /// uses pinned memory to optimize performance. For larger transfers, it falls back to regular memory.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes to allocate.
    /// * `marked_pinned` - Whether to force the use of pinned memory.
    ///
    /// # Returns
    ///
    /// A [Bytes] instance of the correct size.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve_cpu(
        &mut self,
        size: usize,
        marked_pinned: bool,
        origin: Option<StreamId>,
    ) -> Bytes {
        // Use pinned memory for small transfers (<= 100 MB) or when explicitly marked.
        if !marked_pinned && size > 100 * MB {
            return Bytes::from_bytes_vec(vec![0; size]);
        }

        self.reserve_pinned(size, origin)
            .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn reserve_pinned(&mut self, size: usize, origin: Option<StreamId>) -> Option<Bytes> {
        let stream = match origin {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };
        let handle = stream.memory_management_cpu.reserve(size as u64).ok()?;

        let binding = MemoryHandle::binding(handle);
        let resource = stream
            .memory_management_cpu
            .get_resource(binding.clone(), None, None)
            .ok_or(IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })
            .ok()?;

        let controller = Box::new(PinnedMemoryManagedAllocController::init(binding, resource));
        // SAFETY: The binding has initialized memory for at least `size` bytes.
        Some(unsafe { Bytes::from_controller(controller, size) })
    }

    /// Asynchronously reads data from GPU memory to host memory based on the provided copy descriptors.
    ///
    /// # Parameters
    ///
    /// * `descriptors` - A vector of descriptors specifying the source GPU memory and its layout.
    ///
    /// # Returns
    ///
    /// * A `Future` resolving to:
    ///   * `Ok(Vec<Bytes>)` - The data read from the GPU as a vector of byte arrays.
    ///   * `Err(IoError)` - If the read operation fails.
    pub fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let descriptors_moved = descriptors
            .iter()
            .map(|b| b.binding.clone())
            .collect::<Vec<_>>();
        let result = self.copies_to_bytes(descriptors, true);
        let fence = Fence::new(self.streams.current().sys);

        async move {
            let sync = fence.wait_sync();
            // Release memory handle.
            core::mem::drop(descriptors_moved);

            sync?;

            result
        }
    }

    #[allow(unused)]
    /// TODO: Read data using the origin stream where the data was allocated.
    pub fn read_async_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let results = self.copies_to_bytes_origin(descriptors, true);

        async move {
            let (bytes, fences) = results?;

            for fence in fences {
                fence.wait_sync();
            }
            Ok(bytes)
        }
    }

    fn copies_to_bytes(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        pinned: bool,
    ) -> Result<Vec<Bytes>, IoError> {
        let mut result = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            result.push(self.copy_to_bytes(descriptor, pinned, None)?);
        }

        Ok(result)
    }

    fn copies_to_bytes_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        pinned: bool,
    ) -> Result<(Vec<Bytes>, Vec<Fence>), IoError> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut fences = Vec::with_capacity(descriptors.len());
        let mut fenced = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            let stream = descriptor.binding.stream;
            let bytes = self.copy_to_bytes(descriptor, pinned, Some(stream))?;

            if !fenced.contains(&stream) {
                let fence = Fence::new(self.streams.get(&stream).sys);
                fenced.push(stream);
                fences.push(fence);
            }

            data.push(bytes);
        }

        Ok((data, fences))
    }

    pub fn copy_to_bytes(
        &mut self,
        descriptor: CopyDescriptor<'_>,
        pinned: bool,
        stream_id: Option<StreamId>,
    ) -> Result<Bytes, IoError> {
        let num_bytes = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut bytes = self.reserve_cpu(num_bytes, pinned, stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Writes data to the host from the GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the source GPU memory, its shape, strides, and element size.
    /// * `bytes` - The host bytes to write from the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
    pub fn write_to_cpu(
        &mut self,
        descriptor: CopyDescriptor,
        bytes: &mut Bytes,
        stream_id: Option<StreamId>,
    ) -> Result<(), IoError> {
        let CopyDescriptor {
            binding,
            shape,
            strides,
            elem_size,
        } = descriptor;

        if !has_pitched_row_major_strides(shape, strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(binding)?;
        let stream = match stream_id {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };

        unsafe {
            write_to_cpu(shape, strides, elem_size, bytes, resource.ptr, stream.sys)?;
        }

        Ok(())
    }

    /// Writes data from the host to GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the destination GPU memory, its shape, strides, and element size.
    /// * `data` - The host data to write to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, descriptor, data))
    )]
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let CopyDescriptor {
            binding,
            shape,
            strides,
            elem_size,
        } = descriptor;
        if !has_pitched_row_major_strides(shape, strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        // Capture guard (Opus P0): a NON-EMPTY host->device write recorded inside the locked capture
        // window bakes a host SOURCE pointer that is freed after `launch` (and a pageable staged copy
        // can invalidate the capture). `write_to_gpu` is the funnel for `from_data` / `from_floats` /
        // `arange` / `create_from_slice` / `write_to_handle`, so an ACCIDENTAL default `Tensor::random`
        // / host-stage inside a captured region would otherwise silently bake stale bytes. Fail LOUD
        // instead. (Legitimate per-replay seed writes via `write_to_handle` happen BETWEEN replays,
        // when the arena is sealed — not locked — so they are unaffected; dynamic-metadata staging is
        // interned in `create_with_data` and never reaches here under a locked arena.)
        if !data.is_empty()
            && self
                .streams
                .current()
                .memory_management_gpu
                .capture_arena_locked()
        {
            return Err(IoError::Unknown {
                description: format!(
                    "host->device staging of {} bytes inside a CUDA-graph capture is not supported \
                     (it would bake a host source pointer freed after launch / a pageable copy can \
                     invalidate the capture). This is the funnel for `from_data`/`from_floats`/ \
                     `arange`/`create_from_slice` — pre-compute such tensors OUTSIDE the captured \
                     region (persistent buffers) before capturing.",
                    data.len()
                ),
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(binding)?;

        let size = data.len();
        let data = match data.property() {
            AllocationProperty::File => {
                let mut buffer = self.reserve_pinned(size, None).unwrap();
                data.copy_into(&mut buffer);
                buffer
            }
            _ => data,
        };
        let current = self.streams.current();

        unsafe { write_to_gpu(shape, strides, elem_size, &data, resource.ptr, current.sys) }?;

        // Make sure we don't reuse the pinned memory until the write to gpu is completed.
        let event = Fence::new(current.sys);
        self.streams.gc(GcTask::new(data, event));

        Ok(())
    }

    /// Allocates a new GPU memory buffer and immediately copies contiguous host data into it.
    ///
    /// # Parameters
    ///
    /// * `data` - The host data to copy to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated and populated GPU memory.
    /// * `Err(IoError)` - If the allocation or data copy fails.
    pub fn create_with_data(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        // Capture-arena dynamic-METADATA interning (the P-final unblock for capturing real Burn ops
        // below Fusion). Every Burn op whose per-launch metadata (`Sequence<FastDivmod>` shapes/strides)
        // exceeds the by-value grid-constant static portion stages it here as a small device buffer; an
        // H2D inside the locked capture window is uncapturable (it bakes a host source freed after
        // `launch`). BUT for a fixed-shape captured region the metadata is IDENTICAL across replays
        // (shape-derived; only device-buffer CONTENTS change), so when a capture arena is ACTIVE we
        // intern each distinct blob BY CONTENT: staged exactly ONCE during warmup into a RETAINED
        // arena block (stable VA), then reused with ZERO H2D on the locked capture pass — the captured
        // kernel just reads the stable-VA buffer; replay re-reads the unchanged bytes. A locked-pass
        // content MISS hard-errors inside `intern_metadata` (warmup never staged it). Empty `data`
        // (the common grid-constant path) falls through to the normal 0-byte handling below.
        if !data.is_empty() {
            if let Some(result) = self
                .streams
                .current()
                .memory_management_gpu
                .capture_arena_intern_metadata(data)
            {
                let (slice_handle, needs_write) = result?;
                let handle = Handle::new(
                    slice_handle,
                    None,
                    None,
                    self.streams.current,
                    self.streams.cursor,
                    data.len() as u64,
                );
                if needs_write {
                    // Warmup miss (not capturing): eager H2D into the freshly-interned RETAINED block.
                    let shape = [data.len()];
                    let desc = CopyDescriptor::new(handle.clone().binding(), &shape, &[1], 1);
                    if !has_pitched_row_major_strides(desc.shape, desc.strides) {
                        return Err(IoError::UnsupportedStrides {
                            backtrace: BackTrace::capture(),
                        });
                    }
                    let resource = self.resource(desc.binding)?;
                    let current = self.streams.current();
                    unsafe {
                        write_to_gpu(
                            desc.shape,
                            desc.strides,
                            desc.elem_size,
                            data,
                            resource.ptr,
                            current.sys,
                        )?;
                    };
                }
                return Ok(handle);
            }
        }

        let handle = self.reserve(data.len() as u64)?;
        let shape = [data.len()];
        let desc = CopyDescriptor::new(handle.clone().binding(), &shape, &[1], 1);
        if !has_pitched_row_major_strides(desc.shape, desc.strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(desc.binding)?;

        let current = self.streams.current();
        let src: &[u8] = data;

        unsafe {
            write_to_gpu(
                desc.shape,
                desc.strides,
                desc.elem_size,
                src,
                resource.ptr,
                current.sys,
            )?;
        };

        Ok(handle)
    }

    /// Synchronizes the current stream, ensuring all pending operations are complete.
    ///
    /// # Returns
    ///
    /// * A `DynFut<()>` future that resolves when the stream is synchronized.
    pub fn sync(&mut self) -> DynFut<Result<(), ExecutionError>> {
        let fence = Fence::new(self.streams.current().sys);

        Box::pin(async { fence.wait_sync() })
    }

    /// Executes a registered CUDA kernel with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `kernel_id` - The identifier of the kernel to execute.
    /// * `kernel` - The cube task to compile if not cached.
    /// * `mode` - The execution mode for the current kernel.
    /// * `dispatch_count` - The number of thread blocks in the x, y, and z dimensions.
    /// * `tensor_maps` - Tensor maps for structured memory access.
    /// * `resources` - GPU resources (e.g., buffers) used by the kernel.
    /// * `scalars` - Scalar arguments passed to the kernel.
    /// * `logger` - The logger to use to write compilation & runtime info.
    ///
    /// # Panics
    ///
    /// * If the execution fails, with an error message or profiling error.
    #[allow(clippy::too_many_arguments)]
    pub fn kernel(
        &mut self,
        kernel_id: KernelId,
        kernel: Box<dyn CubeTask<CudaCompiler>>,
        mode: ExecutionMode,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        scalars: &[*mut c_void],
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if !self.ctx.module_names.contains_key(&kernel_id) {
            self.ctx.compile_kernel(&kernel_id, kernel, mode, logger)?;
        }

        let stream = self.streams.current();

        let result = self.ctx.execute_task(
            stream,
            kernel_id,
            dispatch_count,
            tensor_maps,
            resources,
            scalars,
        );

        if let Err(err) = result {
            match self.ctx.timestamps.is_empty() {
                true => return Err(err),
                false => self.ctx.timestamps.error(ProfileError::Launch(err)),
            }
        };
        Ok(())
    }
}

/// Internal write to GPU command.
///
/// Writes data from a CPU buffer to a CUDA resource.
///
/// Requires that `shape`/`stride` satisfy contiguous row-major order.
/// - the caller is responsible for guaranteeing this.
/// - this is checked locally only under debug.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(strides, data, dst_ptr, stream))
)]
pub(crate) unsafe fn write_to_gpu(
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
    data: &[u8],
    dst_ptr: u64,
    stream: *mut CUstream_st,
) -> Result<(), IoError> {
    #[cfg(debug_assertions)]
    try_check_pitched_row_major_strides(shape, strides).map_err(|e| IoError::Unknown {
        description: format!("write_to_gpu: invalid strides: {e}"),
        backtrace: BackTrace::capture(),
    })?;

    let rank = shape.len();
    if rank <= 1 {
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dst_ptr, data, stream).map_err(|e| {
                IoError::Unknown {
                    description: format!("CUDA memcpy_htod failed: {e}"),
                    backtrace: BackTrace::capture(),
                }
            })
        }
    } else {
        // As we've enforced that the strides are contiguous row-major,
        // and we know that the rank >= 2, we can construct a 2D view
        // for the aligned GPU pitched memcpy.

        let dim_x_shape = shape[rank - 1];
        let width_bytes = dim_x_shape * elem_size;

        // the second "dim"'s shape is the product of the rest of the space.
        let dim_y_shape: usize = shape[..rank - 1].iter().product();
        let pitch = strides[rank - 2] * elem_size;

        let cpy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: data.as_ptr() as *const c_void,
            srcPitch: width_bytes,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstDevice: dst_ptr,
            dstPitch: pitch,
            WidthInBytes: width_bytes,
            Height: dim_y_shape,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcDevice: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstHost: Default::default(),
            dstArray: Default::default(),
        };

        unsafe {
            cuMemcpy2DAsync_v2(&cpy, stream)
                .result()
                .map_err(|e| IoError::Unknown {
                    description: format!("CUDA memcpy failed: {e}"),
                    backtrace: BackTrace::capture(),
                })
        }
    }
}

/// Internal write to CPU command.
///
/// Writes data from a CUDA resource to a CPU buffer.
///
/// Requires that `shape`/`stride` satisfy contiguous row-major order.
/// - the caller is responsible for guaranteeing this.
/// - this is checked locally only under debug.
#[cfg_attr(
    feature = "tracing",
    tracing::instrument(level = "trace", skip(strides, bytes, resource_ptr, stream))
)]
pub(crate) unsafe fn write_to_cpu(
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
    bytes: &mut Bytes,
    resource_ptr: u64,
    stream: *mut CUstream_st,
) -> Result<(), IoError> {
    #[cfg(debug_assertions)]
    try_check_pitched_row_major_strides(shape, strides).map_err(|e| IoError::Unknown {
        description: format!("write_to_cpu: invalid strides: {e}"),
        backtrace: BackTrace::capture(),
    })?;

    let rank = shape.len();
    let bytes = bytes.deref_mut();
    if rank <= 1 {
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(bytes, resource_ptr, stream).map_err(|e| {
                IoError::Unknown {
                    description: format!("CUDA memcpy_dtoh failed: {e}"),
                    backtrace: BackTrace::capture(),
                }
            })
        }
    } else {
        // As we've enforced that the strides are contiguous row-major,
        // and we know that the rank >= 2, we can construct a 2D view
        // for the aligned GPU pitched memcpy.

        let dim_x_shape = shape[rank - 1];
        let width_bytes = dim_x_shape * elem_size;

        // the second "dim"'s shape is the product of the rest of the space.
        let dim_y_shape: usize = shape[..rank - 1].iter().product();
        let pitch = strides[rank - 2] * elem_size;

        let cpy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcDevice: resource_ptr,
            srcPitch: pitch,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            dstHost: bytes.as_mut_ptr() as *mut c_void,
            dstPitch: width_bytes,
            WidthInBytes: width_bytes,
            Height: dim_y_shape,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstArray: Default::default(),
            srcHost: Default::default(),
            dstDevice: Default::default(),
        };

        unsafe {
            cuMemcpy2DAsync_v2(&cpy, stream)
                .result()
                .map_err(|e| IoError::Unknown {
                    description: format!("CUDA 2D memcpy failed: {e}"),
                    backtrace: BackTrace::capture(),
                })
        }
    }
}
