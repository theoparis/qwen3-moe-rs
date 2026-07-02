use crate::{
    config::{TypeNameFormatLevel, type_name_format},
    kernel::KernelMetadata,
    logging::ProfileLevel,
    memory_management::{MemoryAllocationMode, MemoryUsage},
    runtime::Runtime,
    server::{
        Allocation, AllocationDescriptor, AllocationKind, Binding, Bindings, ComputeServer,
        CopyDescriptor, CubeCount, ExecutionError, ExecutionMode, Handle, IoError, LaunchError,
        ProfileError, ServerCommunication, ServerUtilities,
    },
    storage::{BindingResource, ComputeStorage},
};
use alloc::format;
use alloc::sync::Arc;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::DerefMut;
use cubecl_common::{
    bytes::{AllocationProperty, Bytes},
    device::{Device, DeviceContext},
    future::DynFut,
    profile::ProfileDuration,
};
use cubecl_ir::{DeviceProperties, LineSize};

#[cfg(feature = "profile-tracy")]
use alloc::boxed::Box;

#[allow(unused)]
use cubecl_common::profile::TimingMethod;
use cubecl_common::stream_id::StreamId;

/// The `ComputeClient` is the entry point to require tasks from the `ComputeServer`.
/// It should be obtained for a specific device via the Compute struct.
pub struct ComputeClient<R: Runtime> {
    context: DeviceContext<R::Server>,
    utilities: Arc<ServerUtilities<R::Server>>,
    stream_id: Option<StreamId>,
}

impl<R: Runtime> Clone for ComputeClient<R> {
    fn clone(&self) -> Self {
        Self {
            context: self.context.clone(),
            utilities: self.utilities.clone(),
            stream_id: self.stream_id,
        }
    }
}

impl<R: Runtime> ComputeClient<R> {
    /// Get the info of the current backend.
    pub fn info(&self) -> &<R::Server as ComputeServer>::Info {
        &self.utilities.info
    }

    /// Create a new client with a new server.
    pub fn init<D: Device>(device: &D, server: R::Server) -> Self {
        let utilities = server.utilities();

        let context = DeviceContext::<R::Server>::insert(device, server)
            .expect("Can't create a new client on an already registered server");

        Self {
            context,
            utilities,
            stream_id: None,
        }
    }

    /// Load the client for the given device.
    pub fn load<D: Device>(device: &D) -> Self {
        let context = DeviceContext::<R::Server>::locate(device);
        let utilities = context.lock().utilities();

        Self {
            context,
            utilities,
            stream_id: None,
        }
    }

    fn stream_id(&self) -> StreamId {
        match self.stream_id {
            Some(val) => val,
            None => StreamId::current(),
        }
    }

    /// Set the stream in which the current client is operating on.
    ///
    /// # Safety
    ///
    /// This is highly unsafe and should probably only be used by the CubeCL/Burn projects for now.
    pub unsafe fn set_stream(&mut self, stream_id: StreamId) {
        self.stream_id = Some(stream_id);
    }

    fn do_read(&self, descriptors: Vec<CopyDescriptor<'_>>) -> DynFut<Result<Vec<Bytes>, IoError>> {
        let stream_id = self.stream_id();
        let mut state = self.context.lock();
        let fut = state.read(descriptors, stream_id);
        core::mem::drop(state);
        fut
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_async(
        &self,
        handles: Vec<Handle>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send {
        let strides = [1];
        let shapes = handles
            .iter()
            .map(|it| [it.size() as usize])
            .collect::<Vec<_>>();
        let bindings = handles
            .into_iter()
            .map(|it| it.binding())
            .collect::<Vec<_>>();
        let descriptors = bindings
            .into_iter()
            .zip(shapes.iter())
            .map(|(binding, shape)| CopyDescriptor::new(binding, shape, &strides, 1))
            .collect();

        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    pub fn read(&self, handles: Vec<Handle>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_async(handles)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    pub fn read_one(&self, handle: Handle) -> Bytes {
        cubecl_common::reader::read_sync(self.read_async(vec![handle]))
            .expect("TODO")
            .remove(0)
    }

    /// Given bindings, returns owned resources as bytes.
    pub fn read_tensor_async(
        &self,
        descriptors: Vec<CopyDescriptor<'_>>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send {
        self.do_read(descriptors)
    }

    /// Given bindings, returns owned resources as bytes.
    ///
    /// # Remarks
    ///
    /// Panics if the read operation fails.
    ///
    /// The tensor must be in the same layout as created by the runtime, or more strict.
    /// Contiguous tensors are always fine, strided tensors are only ok if the stride is similar to
    /// the one created by the runtime (i.e. padded on only the last dimension). A way to check
    /// stride compatibility on the runtime will be added in the future.
    ///
    /// Also see [`ComputeClient::create_tensor`].
    pub fn read_tensor(&self, descriptors: Vec<CopyDescriptor<'_>>) -> Vec<Bytes> {
        cubecl_common::reader::read_sync(self.read_tensor_async(descriptors)).expect("TODO")
    }

    /// Given a binding, returns owned resource as bytes.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_tensor_async(
        &self,
        descriptor: CopyDescriptor<'_>,
    ) -> impl Future<Output = Result<Bytes, IoError>> + Send {
        let fut = self.read_tensor_async(vec![descriptor]);

        async { Ok(fut.await?.remove(0)) }
    }

    /// Given a binding, returns owned resource as bytes.
    ///
    /// # Remarks
    /// Panics if the read operation fails.
    /// See [`ComputeClient::read_tensor`]
    pub fn read_one_tensor(&self, descriptor: CopyDescriptor) -> Bytes {
        self.read_tensor(vec![descriptor]).remove(0)
    }

    /// Given a resource handle, returns the storage resource.
    pub fn get_resource(
        &self,
        binding: Binding,
    ) -> BindingResource<<<R::Server as ComputeServer>::Storage as ComputeStorage>::Resource> {
        let stream_id = self.stream_id();
        self.context.lock().get_resource(binding, stream_id)
    }

    fn do_create_from_slices(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        slices: Vec<&[u8]>,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut state = self.context.lock();
        let allocations = state.create(descriptors.clone(), self.stream_id())?;
        let descriptors = descriptors
            .into_iter()
            .zip(allocations.iter())
            .zip(slices)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.handle.clone().binding(),
                        desc.shape,
                        &alloc.strides,
                        desc.elem_size,
                    ),
                    Bytes::from_bytes_vec(data.to_vec()),
                )
            })
            .collect();
        let stream_id = self.stream_id();
        state.write(descriptors, stream_id)?;
        Ok(allocations)
    }

    fn do_create(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        mut data: Vec<Bytes>,
    ) -> Result<Vec<Allocation>, IoError> {
        self.staging(data.iter_mut(), true);

        let mut state = self.context.lock();
        let allocations = state.create(descriptors.clone(), self.stream_id())?;
        let descriptors = descriptors
            .into_iter()
            .zip(allocations.iter())
            .zip(data)
            .map(|((desc, alloc), data)| {
                (
                    CopyDescriptor::new(
                        alloc.handle.clone().binding(),
                        desc.shape,
                        &alloc.strides,
                        desc.elem_size,
                    ),
                    data,
                )
            })
            .collect();
        let stream_id = self.stream_id();
        state.write(descriptors, stream_id)?;
        Ok(allocations)
    }

    /// Returns a resource handle containing the given data.
    ///
    /// # Notes
    ///
    /// Prefer using the more efficient [`Self::create`] function.
    pub fn create_from_slice(&self, slice: &[u8]) -> Handle {
        let shape = [slice.len()];

        self.do_create_from_slices(
            vec![AllocationDescriptor::new(
                AllocationKind::Contiguous,
                &shape,
                1,
            )],
            vec![slice],
        )
        .unwrap()
        .remove(0)
        .handle
    }

    /// Returns a resource handle containing the given [Bytes].
    pub fn create(&self, data: Bytes) -> Handle {
        let shape = [data.len()];

        self.do_create(
            vec![AllocationDescriptor::new(
                AllocationKind::Contiguous,
                &shape,
                1,
            )],
            vec![data],
        )
        .unwrap()
        .remove(0)
        .handle
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensor`] for better performance.
    pub fn create_tensor_from_slice(
        &self,
        slice: &[u8],
        shape: &[usize],
        elem_size: usize,
    ) -> Allocation {
        self.do_create_from_slices(
            vec![AllocationDescriptor::new(
                AllocationKind::Optimized,
                shape,
                elem_size,
            )],
            vec![slice],
        )
        .unwrap()
        .remove(0)
    }

    /// Given a resource and shape, stores it and returns the tensor handle and strides.
    /// This may or may not return contiguous strides. The layout is up to the runtime, and care
    /// should be taken when indexing.
    ///
    /// Currently the tensor may either be contiguous (most runtimes), or "pitched", to use the CUDA
    /// terminology. This means the last (contiguous) dimension is padded to fit a certain alignment,
    /// and the strides are adjusted accordingly. This can make memory accesses significantly faster
    /// since all rows are aligned to at least 16 bytes (the maximum load width), meaning the GPU
    /// can load as much data as possible in a single instruction. It may be aligned even more to
    /// also take cache lines into account.
    ///
    /// However, the stride must be taken into account when indexing and reading the tensor
    /// (also see [`ComputeClient::read_tensor`]).
    pub fn create_tensor(&self, bytes: Bytes, shape: &[usize], elem_size: usize) -> Allocation {
        self.do_create(
            vec![AllocationDescriptor::new(
                AllocationKind::Optimized,
                shape,
                elem_size,
            )],
            vec![bytes],
        )
        .unwrap()
        .remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    ///
    /// # Notes
    ///
    /// Prefer using [`Self::create_tensors`] for better performance.
    pub fn create_tensors_from_slices(
        &self,
        descriptors: Vec<(AllocationDescriptor<'_>, &[u8])>,
    ) -> Vec<Allocation> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create_from_slices(descriptors, data).unwrap()
    }

    /// Reserves all `shapes` in a single storage buffer, copies the corresponding `data` into each
    /// handle, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn create_tensors(
        &self,
        descriptors: Vec<(AllocationDescriptor<'_>, Bytes)>,
    ) -> Vec<Allocation> {
        let (descriptors, data) = descriptors.into_iter().unzip();

        self.do_create(descriptors, data).unwrap()
    }

    fn do_empty(
        &self,
        descriptors: Vec<AllocationDescriptor<'_>>,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut state = self.context.lock();
        state.create(descriptors, self.stream_id())
    }

    /// Reserves `size` bytes in the storage, and returns a handle over them.
    pub fn empty(&self, size: usize) -> Handle {
        let shape = [size];
        let descriptor = AllocationDescriptor::new(AllocationKind::Contiguous, &shape, 1);
        self.do_empty(vec![descriptor]).unwrap().remove(0).handle
    }

    /// Reserves `shape` in the storage, and returns a tensor handle for it.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensor(&self, shape: &[usize], elem_size: usize) -> Allocation {
        let descriptor = AllocationDescriptor::new(AllocationKind::Optimized, shape, elem_size);
        self.do_empty(vec![descriptor]).unwrap().remove(0)
    }

    /// Reserves all `shapes` in a single storage buffer, and returns the handles for them.
    /// See [`ComputeClient::create_tensor`]
    pub fn empty_tensors(&self, descriptors: Vec<AllocationDescriptor<'_>>) -> Vec<Allocation> {
        self.do_empty(descriptors).unwrap()
    }

    /// Write `data` into the storage of an EXISTING `handle` on this client's stream — an on-stream
    /// H2D copy into the handle's CURRENT device address. Unlike [`Self::create`] it does NOT allocate
    /// a new device buffer, so the device pointer is unchanged.
    ///
    /// This is the host side of the CUDA-graph device-seed RNG (component C3, option (c)): a captured
    /// kernel bakes the POINTER of a persistent seed buffer, and before each [`CapturedGraph::replay`]
    /// the host writes fresh seeds into that same buffer with this call, ordered before the replay on
    /// the same stream, so the replayed kernel re-reads new seeds and decorrelates.
    ///
    /// # Safety contract (caller-upheld; this is a raw on-stream write with NO synchronization)
    ///
    /// - **Existing-allocation write only.** It writes into `handle`'s current allocation; it never
    ///   reallocates, so the device VA is stable (this is exactly why a captured graph can bake it).
    /// - **Same-stream ordering is the caller's job.** The write is enqueued on THIS client's stream
    ///   (`stream_id()`) with NO implicit cross-stream sync. To make a replay observe the new bytes,
    ///   issue this write on the SAME stream as `replay()` and BEFORE it. A write on a different stream
    ///   (or unsynchronized concurrent device access to the buffer) races — undefined contents.
    /// - **No concurrent kernel access.** The buffer must not be read/written by an in-flight kernel
    ///   while this write is in flight; order it between replays, not during one.
    /// - **Bounds.** `data.len()` must not exceed the handle's allocation size (`handle.size()`); a
    ///   debug build asserts this. An over-long write would clobber neighbouring storage.
    pub fn write_to_handle(&self, handle: &Handle, data: &[u8]) {
        debug_assert!(
            data.len() as u64 <= handle.size(),
            "write_to_handle: data ({} bytes) exceeds handle allocation ({} bytes)",
            data.len(),
            handle.size(),
        );
        let shape = [data.len()];
        let strides = [1usize];
        let descriptor = handle.copy_descriptor(&shape, &strides, 1);
        let stream_id = self.stream_id();
        // One owned host copy (`to_vec`) is UNAVOIDABLE here, not wasteful churn: the backend issues an
        // ASYNC H2D whose host SOURCE must outlive the call (the CUDA backend keeps it alive past return
        // via a stream GC fence — see `command.rs::write_to_gpu`). `ComputeServer::write` is therefore
        // defined over an OWNED, `'static` `Bytes`; the type-erased `Box<dyn AllocationController>`
        // cannot borrow the caller's `&[u8]`. The only way to drop this copy would be a SYNCHRONOUS H2D
        // from the borrowed slice, which injects a host sync per replay — strictly worse for the graph's
        // CPU-overhead goal. So we keep exactly one small copy (no extra realign: a fresh `Vec<u8>` is
        // already `MAX_ALIGN`-aligned for these allocations).
        self.context
            .lock()
            .write(
                vec![(descriptor, Bytes::from_bytes_vec(data.to_vec()))],
                stream_id,
            )
            .expect("write_to_handle: on-stream H2D into existing handle failed");
    }

    /// Marks the given [Bytes] as being a staging buffer, maybe transferring it to pinned memory
    /// for faster data transfer with compute device.
    pub fn staging<'a, I>(&self, bytes: I, file_only: bool)
    where
        I: Iterator<Item = &'a mut Bytes>,
    {
        let has_staging = |b: &Bytes| match b.property() {
            AllocationProperty::Pinned => false,
            AllocationProperty::File => true,
            AllocationProperty::Native | AllocationProperty::Other => !file_only,
        };

        let mut to_be_updated = Vec::new();
        let sizes = bytes
            .filter_map(|b| match has_staging(b) {
                true => {
                    let len = b.len();
                    to_be_updated.push(b);
                    Some(len)
                }
                false => None,
            })
            .collect::<Vec<usize>>();

        if sizes.is_empty() {
            return;
        }

        let stream_id = self.stream_id();
        let mut context = self.context.lock();
        let stagings = match context.staging(&sizes, stream_id) {
            Ok(val) => val,
            Err(_) => return,
        };
        core::mem::drop(context);

        to_be_updated
            .into_iter()
            .zip(stagings)
            .for_each(|(b, mut staging)| {
                b.copy_into(&mut staging);
                core::mem::swap(b, &mut staging);
            });
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src, dst_server))
    )]
    pub fn to_client(&self, src: Handle, dst_server: &Self) -> Allocation {
        let shape = [src.size() as usize];
        let src_descriptor = src.copy_descriptor(&shape, &[1], 1);

        if R::Server::SERVER_COMM_ENABLED {
            self.to_client_tensor(src_descriptor, dst_server)
        } else {
            let alloc_desc = AllocationDescriptor::new(
                AllocationKind::Contiguous,
                src_descriptor.shape,
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
        }
    }

    /// Transfer data from one client to another
    ///
    /// Make sure the source description can be read in a contiguous manner.
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, src_descriptor, dst_server))
    )]
    pub fn to_client_tensor(
        &self,
        src_descriptor: CopyDescriptor<'_>,
        dst_server: &Self,
    ) -> Allocation {
        if R::Server::SERVER_COMM_ENABLED {
            let guard = self.context.lock_device_kind();
            let mut server_src = self.context.lock();
            let mut server_dst = dst_server.context.lock();

            let copied = R::Server::copy(
                server_src.deref_mut(),
                server_dst.deref_mut(),
                src_descriptor,
                self.stream_id(),
                dst_server.stream_id(),
            )
            .unwrap();
            core::mem::drop(server_src);
            core::mem::drop(server_dst);
            core::mem::drop(guard);
            copied
        } else {
            let alloc_desc = AllocationDescriptor::new(
                AllocationKind::Optimized,
                src_descriptor.shape,
                src_descriptor.elem_size,
            );
            self.change_client_sync(src_descriptor, alloc_desc, dst_server)
        }
    }

    #[track_caller]
    #[cfg_attr(feature = "tracing", tracing::instrument(level="trace",
        skip(self, kernel, bindings),
        fields(
            kernel.name = %kernel.name(),
            kernel.id = %kernel.id(),
        )
    ))]
    unsafe fn launch_inner(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        let level = self.utilities.logger.profile_level();

        match level {
            None | Some(ProfileLevel::ExecutionOnly) => {
                let mut state = self.context.lock();
                let name = kernel.name();

                let result = unsafe { state.launch(kernel, count, bindings, mode, stream_id) };

                if matches!(level, Some(ProfileLevel::ExecutionOnly)) {
                    let info = type_name_format(name, TypeNameFormatLevel::Balanced);
                    self.utilities.logger.register_execution(info);
                }
                result
            }
            Some(level) => {
                let name = kernel.name();
                let kernel_id = kernel.id();
                let (result, profile) = self
                    .profile(
                        || unsafe {
                            let mut state = self.context.lock();
                            state.launch(kernel, count.clone(), bindings, mode, stream_id)
                        },
                        name,
                    )
                    .unwrap();
                let info = match level {
                    ProfileLevel::Full => {
                        format!("{name}: {kernel_id} CubeCount {count:?}")
                    }
                    _ => type_name_format(name, TypeNameFormatLevel::Balanced),
                };
                self.utilities.logger.register_profiled(info, profile);
                result
            }
        }
    }

    /// Launches the `kernel` with the given `bindings`.
    #[track_caller]
    pub fn launch(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
    ) -> Result<(), LaunchError> {
        // SAFETY: Using checked execution mode.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Checked,
                self.stream_id(),
            )
        }
    }

    /// Launches the `kernel` with the given `bindings` without performing any bound checks.
    ///
    /// # Safety
    ///
    /// To ensure this is safe, you must verify your kernel:
    /// - Has no out-of-bound reads and writes that can happen.
    /// - Has no infinite loops that might never terminate.
    #[track_caller]
    pub unsafe fn launch_unchecked(
        &self,
        kernel: <R::Server as ComputeServer>::Kernel,
        count: CubeCount,
        bindings: Bindings,
    ) -> Result<(), LaunchError> {
        // SAFETY: Caller has to uphold kernel being safe.
        unsafe {
            self.launch_inner(
                kernel,
                count,
                bindings,
                ExecutionMode::Unchecked,
                self.stream_id(),
            )
        }
    }

    /// Flush all outstanding commands.
    pub fn flush(&self) {
        let stream_id = self.stream_id();
        self.context.lock().flush(stream_id)
    }

    /// Wait for the completion of every task in the server.
    pub fn sync(&self) -> DynFut<Result<(), ExecutionError>> {
        let stream_id = self.stream_id();
        let mut state = self.context.lock();
        let fut = state.sync(stream_id);
        core::mem::drop(state);
        self.utilities.logger.profile_summary();

        fut
    }

    /// Get the features supported by the compute server.
    pub fn properties(&self) -> &DeviceProperties {
        &self.utilities.properties
    }

    /// # Warning
    ///
    /// For private use only.
    pub fn properties_mut(&mut self) -> Option<&mut DeviceProperties> {
        Arc::get_mut(&mut self.utilities).map(|state| &mut state.properties)
    }

    /// Get the current memory usage of this client.
    pub fn memory_usage(&self) -> MemoryUsage {
        self.context.lock().memory_usage(self.stream_id())
    }

    /// Change the memory allocation mode.
    ///
    /// # Safety
    ///
    /// This function isn't thread safe and might create memory leaks.
    pub unsafe fn allocation_mode(&self, mode: MemoryAllocationMode) {
        self.context.lock().allocation_mode(mode, self.stream_id())
    }

    /// Use a persistent memory strategy to execute the provided function.
    ///
    /// # Notes
    ///
    /// - Using that memory strategy is beneficial for stating model parameters and similar workflows.
    /// - You can call [`Self::memory_cleanup()`] if you want to free persistent memory.
    pub fn memory_persistent_allocation<Input, Output, Func: Fn(Input) -> Output>(
        &self,
        input: Input,
        func: Func,
    ) -> Output {
        let device_guard = self.context.lock_device();

        self.context
            .lock()
            .allocation_mode(MemoryAllocationMode::Persistent, self.stream_id());

        let output = func(input);

        self.context
            .lock()
            .allocation_mode(MemoryAllocationMode::Auto, self.stream_id());

        core::mem::drop(device_guard);

        output
    }

    /// Ask the client to release memory that it can release.
    ///
    /// Nb: Results will vary on what the memory allocator deems beneficial,
    /// so it's not guaranteed any memory is freed.
    pub fn memory_cleanup(&self) {
        self.context.lock().memory_cleanup(self.stream_id())
    }

    /// Capture all device work issued by `f` into a replayable graph (CUDA-graph capture, component
    /// C1 of the CUDA-graph plan). Returns a [`CapturedGraph`] handle whose [`CapturedGraph::replay`]
    /// re-issues the whole recorded launch list with a single host call.
    ///
    /// The capture brackets [`ComputeServer::capture_begin`] / [`ComputeServer::capture_end`] around
    /// `f`, holding the device lock so no other thread interleaves work on the device (the
    /// capturing stream itself uses thread-local capture mode). The per-launch server lock is the
    /// same reentrant lock, so `f` may launch kernels normally.
    ///
    /// # Safety
    ///
    /// Captured kernels bake the *device virtual addresses* of every buffer they touch directly into
    /// the graph nodes. Replay re-issues those exact addresses with no rebinding, so the caller must
    /// uphold ALL of the following or replay will silently corrupt memory:
    ///
    /// - **Buffers must outlive the handle and never be reallocated.** Every buffer touched by a
    ///   captured kernel must stay alive at a fixed address for the lifetime of the returned
    ///   [`CapturedGraph`]. If the allocator hands a captured VA back out and reuses it, replay
    ///   writes through a stale pointer — silent corruption. P0 requires pre-allocated fixed buffers;
    ///   the graph-aware capture arena is P1.
    /// - **`f` must issue ONLY static-shape, host-sync-free work onto this client's stream:** no
    ///   `sync`, no reads/`read_async`, no profiling, and no `CubeCount::Dynamic` launches (they need
    ///   a host read of the dispatch dims and are hard-rejected during capture).
    /// - **No allocation / scalar / dynamic-metadata staging inside `f`** whose transient buffer is
    ///   freed before replay (hard-rejected during capture on the CUDA backend; see FIX 2).
    /// - **A WARMUP eager run of the exact kernels is REQUIRED before capturing.** First use of a
    ///   kernel triggers JIT / `cuModuleLoad`, and that host work inside the capture region poisons
    ///   the capture. Run every kernel once eagerly (and `sync`) first.
    /// - **Autotune must be warmed before capture.** Autotuning synchronizes and can change the
    ///   recorded node list; trigger and finish all tuning eagerly before capturing.
    /// - **Capture is SINGLE-STREAM only.** Tensors shared across streams record cross-stream
    ///   event-waits that poison or mis-capture the graph; keep all work in `f` on this one stream.
    ///
    /// Not all runtimes support capture; calling this on one that doesn't will panic.
    pub unsafe fn capture<F: FnOnce()>(&self, f: F) -> CapturedGraph<R> {
        let stream_id = self.stream_id();

        // FIX 4(b): profiling inside the capture region calls `block_on(sync())` (host sync) on every
        // launch, which poisons the capture. Refuse to capture while a *syncing* profile level is
        // active. `ExecutionOnly` only logs a name (no sync) and is fine.
        assert!(
            !matches!(
                self.utilities.logger.profile_level(),
                Some(ProfileLevel::Basic) | Some(ProfileLevel::Medium) | Some(ProfileLevel::Full)
            ),
            "CUDA-graph capture cannot run with a syncing profile level active (it forces a host \
             sync per launch, which poisons capture). Disable profiling before capturing."
        );

        // Serialize the whole capture under the device lock (the cudarc graph objects are not
        // thread-safe). The per-op server lock is the same reentrant lock, so `f`'s launches and the
        // begin/end calls below all re-enter it fine on this thread.
        let device_guard = self.context.lock_device();

        self.context.lock().capture_begin(stream_id);

        // FIX 1 (unwind safety): if `f` panics (reachable via `.expect` in the launch path, the
        // `CubeCount::Dynamic` reject `Err`/`?`, OOM, or the FIX 2 alloc-during-capture guard),
        // skipping `capture_end` would leave `capturing == true` forever AND the CUDA stream stuck in
        // capture mode -> every later op on that stream errors (parking_lot doesn't poison, so the
        // server just wedges). Catch the unwind, abort the capture (pull the stream out of capture +
        // discard the partial graph), drop the device lock, then re-raise the original panic.
        #[cfg(feature = "std")]
        {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
                self.context.lock().capture_abort(stream_id);
                core::mem::drop(device_guard);
                std::panic::resume_unwind(payload);
            }
        }
        // `no_std`: `catch_unwind` is unavailable. Capture is a CUDA-only / `std` feature in practice,
        // so this path is just to keep the crate compiling; a panic in `f` here cannot be recovered.
        #[cfg(not(feature = "std"))]
        f();

        let id = self.context.lock().capture_end(stream_id);

        core::mem::drop(device_guard);

        CapturedGraph {
            client: self.clone(),
            id,
            stream_id,
        }
    }

    /// Capture `f` into a replayable graph backed by a graph-private capture ARENA (component C2 of
    /// the CUDA-graph plan), so `f` may ALLOCATE intermediates and stage scalars/metadata inside the
    /// captured region — unlike [`Self::capture`], which hard-rejects any allocation during capture.
    ///
    /// `f` is run `warmup.max(1)` times EAGERLY first: this both JIT-compiles the kernels and grows
    /// the arena to the closure's peak-LIVE working set (freed intermediates recycle within the
    /// arena, so the arena is sized to the simultaneous high-water, NOT the sum of all allocations).
    /// The arena is then LOCKED and `f` is run once more under capture; every allocation now hits a
    /// recycled arena block, so the capture window issues ZERO device `malloc`s (no graph mem-alloc
    /// nodes -> a `flags = 0` instantiate stays valid). The arena's device blocks are held at fixed
    /// addresses for the returned [`CapturedGraph`]'s whole lifetime and freed when it is dropped.
    ///
    /// # Safety
    ///
    /// All the safety requirements of [`Self::capture`] apply, EXCEPT that allocation / scalar /
    /// metadata staging inside `f` is now permitted (it is routed to the arena). Additionally:
    ///
    /// - **`f` must be safe to run `warmup + 1` times** and must allocate DETERMINISTICALLY (the
    ///   same sequence of sizes every call) — the warmup passes pre-size the arena, and an
    ///   allocation during the capture pass that the warmup did not pre-size is a hard error (growing
    ///   it would inject an uncapturable `cuMemAllocAsync` graph node).
    /// - **Fixed input/output buffers must be allocated OUTSIDE `f`** (before this call). Buffers
    ///   reserved inside `f` live in the arena and are recycled; do not hold the graph's outputs in
    ///   arena buffers across replays — `copy_` them into externally-owned buffers, or read them
    ///   between replays.
    ///
    /// # Determinism contract (FIX 3 — read this)
    ///
    /// The arena only catches ONE class of divergence: the capture pass needing MORE memory than the
    /// warmup pre-sized (that overflows the locked arena -> hard error, never silent corruption). It
    /// does NOT make a non-deterministic closure safe. Replay re-issues the program that the CAPTURE
    /// pass recorded — NOT whatever `f` would do on a later, divergent call. So `f` must be
    /// deterministic across the warmup and capture passes in:
    ///
    /// - **kernel sequence** (same kernels, same order, same launch dims),
    /// - **allocation sizes** (same sequence of `reserve` sizes — a different size that the warmup
    ///   did not pre-size overflows the locked arena), AND
    /// - **live-ranges** (the same buffers live SIMULTANEOUSLY). Two same-size blocks needed at once
    ///   require two arena blocks; if the warmup only ever had one live at a time, the capture pass
    ///   asking for a second concurrent one overflows.
    ///
    /// A closure that, say, takes a *different control-flow branch* on the capture pass than on the
    /// warmup pass — even one that allocates the SAME total bytes — records that branch's program into
    /// the graph; every replay then runs it, regardless of what the inputs "should" select. The arena
    /// cannot detect this (the sizes matched); it is purely the caller's contract.
    pub unsafe fn capture_arena<F: FnMut()>(&self, warmup: usize, mut f: F) -> CapturedGraph<R> {
        let stream_id = self.stream_id();

        assert!(
            !matches!(
                self.utilities.logger.profile_level(),
                Some(ProfileLevel::Basic) | Some(ProfileLevel::Medium) | Some(ProfileLevel::Full)
            ),
            "CUDA-graph capture cannot run with a syncing profile level active (it forces a host \
             sync per launch, which poisons capture). Disable profiling before capturing."
        );

        let device_guard = self.context.lock_device();

        // Install a fresh, growable graph-private arena. From here, allocations route to it.
        self.context.lock().capture_arena_begin(stream_id);

        // Warmup / measure passes (eager): JIT-compile + grow the arena to the peak-live working set.
        #[cfg(feature = "std")]
        {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for _ in 0..warmup.max(1) {
                    f();
                }
            })) {
                // No CUDA capture is open yet; `capture_abort` just discards the arena (and harmlessly
                // no-ops the stream end-capture).
                self.context.lock().capture_abort(stream_id);
                core::mem::drop(device_guard);
                std::panic::resume_unwind(payload);
            }
        }
        #[cfg(not(feature = "std"))]
        for _ in 0..warmup.max(1) {
            f();
        }

        // Flush warmup work and drop its intermediate handles before opening the capture window, so
        // the arena's blocks are all free and ready to be recycled by the capture pass.
        let fut = self.context.lock().sync(stream_id);
        let _ = cubecl_common::future::block_on(fut);

        // Lock the arena (no growth). Infallible (just flips a bool), so it stays outside the guard.
        self.context.lock().capture_arena_lock(stream_id);

        // FIX 1 (unwind safety): open the CUDA capture window, run the capture pass, AND instantiate
        // the graph under ONE catch_unwind. A panic ANYWHERE in this window must not wedge the
        // allocator. The panic points are real: `capture_begin`'s `cuStreamBeginCapture.expect`, any
        // `.expect`/`?` inside `f` (launch failures, the FIX 2 staging hard-error), and — the gap the
        // earlier code missed — `capture_end`'s null-graph `assert!` and `cuGraphInstantiateWithFlags
        // .expect` in the CUDA backend. Without the guard, skipping `capture_end` leaves the LOCKED
        // `CaptureArena` installed in the server's `capture` slot, so every subsequent `reserve` on
        // this stream overflows (hard error) -> allocator wedged, AND the arena's device blocks leak
        // (a `CaptureArena` has no `Drop` that can dealloc — it does not own the storage). On any
        // panic, `capture_abort` pulls the stream back out of capture mode, FREES the active arena's
        // device blocks (`storage.dealloc`) and clears the `capture` slot, leaving the allocator fully
        // usable; we then re-raise the original panic.
        #[cfg(feature = "std")]
        let id = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.context.lock().capture_begin(stream_id);
            f();
            self.context.lock().capture_end(stream_id)
        })) {
            Ok(id) => id,
            Err(payload) => {
                self.context.lock().capture_abort(stream_id);
                core::mem::drop(device_guard);
                std::panic::resume_unwind(payload);
            }
        };
        // `no_std`: `catch_unwind` is unavailable; capture is a CUDA-only / `std` feature in practice.
        #[cfg(not(feature = "std"))]
        let id = {
            self.context.lock().capture_begin(stream_id);
            f();
            self.context.lock().capture_end(stream_id)
        };

        core::mem::drop(device_guard);

        CapturedGraph {
            client: self.clone(),
            id,
            stream_id,
        }
    }

    /// Create a SHARED capture pool (P4 of the CUDA-graph plan — the vLLM `graph_pool_handle` model).
    /// Pass the returned [`CapturePoolHandle`] to [`Self::capture_arena_in_pool`] to capture several
    /// graphs into ONE arena. Graphs sharing a pool reuse each other's device blocks, so N graphs of
    /// (possibly different) static shapes cost ~1 graph's arena high-water instead of N× — exactly how
    /// vLLM holds ~50 batch-size graphs at roughly one graph's memory.
    ///
    /// SOUNDNESS: a pool's blocks are baked into EVERY graph captured into it, so the graphs must
    /// replay SERIALLY on a single stream (this client's stream). Replaying two pooled graphs
    /// concurrently (or on two streams) would have them write through the same device addresses —
    /// undefined behavior. The pool is bound to this client's stream; keep all of its captures and
    /// replays here. The pool (and its device blocks) is freed when the handle AND every graph
    /// captured from it have been dropped.
    pub fn capture_pool(&self) -> CapturePoolHandle<R> {
        let stream_id = self.stream_id();
        let id = self.context.lock().capture_pool_create(stream_id);
        CapturePoolHandle {
            client: self.clone(),
            id,
            stream_id,
        }
    }

    /// Capture `f` into a replayable graph backed by the SHARED `pool` (P4), instead of a fresh
    /// per-graph arena. Identical to [`Self::capture_arena`] in every other respect (warmup pre-sizes,
    /// the locked capture pass issues zero device `malloc`s, metadata is interned), EXCEPT that the
    /// arena is the pool's: this graph reuses the blocks of earlier graphs captured into `pool`, and
    /// contributes its own only when it needs MORE (or differently-sized) blocks than the pool holds.
    ///
    /// To get the ~1-graph-memory win, capture the LARGEST static shape FIRST (it sizes the pool); each
    /// later, smaller graph then reuses those blocks (the shared arena recycles a free block at least as
    /// large as a request). The pool's blocks live until the handle + all its graphs are dropped.
    ///
    /// # Safety
    ///
    /// All of [`Self::capture_arena`]'s requirements, PLUS the pool soundness contract from
    /// [`Self::capture_pool`]: every graph captured into `pool` must replay SERIALLY on this one
    /// stream. The captures themselves are serialized (this call holds the device lock); the REPLAYS
    /// are the caller's responsibility — never replay two of `pool`'s graphs concurrently.
    pub unsafe fn capture_arena_in_pool<F: FnMut()>(
        &self,
        pool: &CapturePoolHandle<R>,
        warmup: usize,
        mut f: F,
    ) -> CapturedGraph<R> {
        let stream_id = self.stream_id();
        assert_eq!(
            pool.stream_id, stream_id,
            "capture_arena_in_pool must run on the SAME stream the pool was created on (shared-pool \
             graphs replay serially on one stream); pool stream {:?} != client stream {:?}",
            pool.stream_id, stream_id
        );

        assert!(
            !matches!(
                self.utilities.logger.profile_level(),
                Some(ProfileLevel::Basic) | Some(ProfileLevel::Medium) | Some(ProfileLevel::Full)
            ),
            "CUDA-graph capture cannot run with a syncing profile level active (it forces a host \
             sync per launch, which poisons capture). Disable profiling before capturing."
        );

        let device_guard = self.context.lock_device();

        // Install the SHARED pool's arena as the active capture arena (P4) — instead of a fresh
        // per-graph one. Earlier graphs' blocks are recycled; growth only for what this graph adds.
        self.context.lock().capture_pool_begin(pool.id, stream_id);

        // Warmup / measure passes (eager): JIT-compile + grow the (shared) arena to the peak-live set.
        #[cfg(feature = "std")]
        {
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                for _ in 0..warmup.max(1) {
                    f();
                }
            })) {
                self.context.lock().capture_abort(stream_id);
                core::mem::drop(device_guard);
                std::panic::resume_unwind(payload);
            }
        }
        #[cfg(not(feature = "std"))]
        for _ in 0..warmup.max(1) {
            f();
        }

        // Flush warmup work + drop its intermediate handles so the shared arena's blocks are all free
        // and ready to be recycled by the capture pass.
        let fut = self.context.lock().sync(stream_id);
        let _ = cubecl_common::future::block_on(fut);

        self.context.lock().capture_arena_lock(stream_id);

        // Same unwind-safe capture window as `capture_arena` (see its FIX 1): a panic anywhere here
        // must abort the capture (pull the stream out of capture + restart the pool) rather than wedge.
        #[cfg(feature = "std")]
        let id = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.context.lock().capture_begin(stream_id);
            f();
            self.context.lock().capture_end(stream_id)
        })) {
            Ok(id) => id,
            Err(payload) => {
                self.context.lock().capture_abort(stream_id);
                core::mem::drop(device_guard);
                std::panic::resume_unwind(payload);
            }
        };
        #[cfg(not(feature = "std"))]
        let id = {
            self.context.lock().capture_begin(stream_id);
            f();
            self.context.lock().capture_end(stream_id)
        };

        core::mem::drop(device_guard);

        CapturedGraph {
            client: self.clone(),
            id,
            stream_id,
        }
    }

    /// Measure the execution time of some inner operations.
    #[track_caller]
    pub fn profile<O>(
        &self,
        func: impl FnOnce() -> O,
        #[allow(unused)] func_name: &str,
    ) -> Result<(O, ProfileDuration), ProfileError> {
        // Get the outer caller. For execute() this points straight to the
        // cube kernel. For general profiling it points to whoever calls profile.
        #[cfg(feature = "profile-tracy")]
        let location = std::panic::Location::caller();

        // Make a CPU span. If the server has system profiling this is all you need.
        #[cfg(feature = "profile-tracy")]
        let _span = tracy_client::Client::running().unwrap().span_alloc(
            None,
            func_name,
            location.file(),
            location.line(),
            0,
        );

        let device_guard = self.context.lock_device();

        #[cfg(feature = "profile-tracy")]
        let gpu_span = if self.utilities.properties.timing_method == TimingMethod::Device {
            let gpu_span = self
                .utilities
                .gpu_client
                .span_alloc(func_name, "profile", location.file(), location.line())
                .unwrap();
            Some(gpu_span)
        } else {
            None
        };

        let token = self.context.lock().start_profile(self.stream_id());

        let out = func();

        #[allow(unused_mut, reason = "Used in profile-tracy")]
        let mut result = self.context.lock().end_profile(self.stream_id(), token);

        #[cfg(feature = "profile-tracy")]
        if let Some(mut gpu_span) = gpu_span {
            gpu_span.end_zone();
            let epoch = self.utilities.epoch_time;
            // Add in the work to upload the timestamp data.
            result = result.map(|result| {
                ProfileDuration::new(
                    Box::pin(async move {
                        let ticks = result.resolve().await;
                        let start_duration = ticks.start_duration_since(epoch).as_nanos() as i64;
                        let end_duration = ticks.end_duration_since(epoch).as_nanos() as i64;
                        gpu_span.upload_timestamp_start(start_duration);
                        gpu_span.upload_timestamp_end(end_duration);
                        ticks
                    }),
                    TimingMethod::Device,
                )
            });
        }
        core::mem::drop(device_guard);

        match result {
            Ok(result) => Ok((out, result)),
            Err(err) => Err(err),
        }
    }

    /// Transfer data from one client to another
    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(
            level = "trace",
            skip(self, src_descriptor, alloc_descriptor, dst_server)
        )
    )]
    fn change_client_sync(
        &self,
        src_descriptor: CopyDescriptor<'_>,
        alloc_descriptor: AllocationDescriptor<'_>,
        dst_server: &Self,
    ) -> Allocation {
        let shape = src_descriptor.shape;
        let elem_size = src_descriptor.elem_size;
        let stream_id = self.stream_id();

        // Allocate destination
        let alloc = dst_server
            .context
            .lock()
            .create(vec![alloc_descriptor], self.stream_id())
            .unwrap()
            .remove(0);

        let read = self.context.lock().read(vec![src_descriptor], stream_id);
        let mut data = cubecl_common::future::block_on(read).unwrap();

        let desc_descriptor = CopyDescriptor {
            binding: alloc.handle.clone().binding(),
            shape,
            strides: &alloc.strides,
            elem_size,
        };

        dst_server
            .context
            .lock()
            .write(vec![(desc_descriptor, data.remove(0))], stream_id)
            .unwrap();

        alloc
    }

    /// Returns all line sizes that are useful to perform optimal IO operation on the given element.
    pub fn io_optimized_line_sizes(&self, size: usize) -> impl Iterator<Item = LineSize> + Clone {
        let load_width = self.properties().hardware.load_width as usize;
        let size_bits = size * 8;
        let max = load_width / size_bits;
        let max = usize::min(self.properties().hardware.max_line_size, max);

        // If the max is 8, we want to test 1, 2, 4, 8 which is log2(8) + 1.
        let num_candidates = max.trailing_zeros() + 1;

        (0..num_candidates).map(|i| 2usize.pow(i)).rev()
    }
}

/// A replayable handle over a graph captured with [`ComputeClient::capture`]. Replaying re-issues
/// the whole recorded launch list with a single host call. The graph's device resources are freed
/// when this handle is dropped.
pub struct CapturedGraph<R: Runtime> {
    client: ComputeClient<R>,
    id: u64,
    stream_id: StreamId,
}

impl<R: Runtime> CapturedGraph<R> {
    /// Replay the captured graph on the originating stream (a single host call).
    pub fn replay(&self) {
        self.client
            .context
            .lock()
            .graph_replay(self.id, self.stream_id);
    }

    /// Device bytes reserved by this graph's capture arena (component C2): the peak-LIVE working-set
    /// high-water mark of the captured closure. 0 for graphs captured with [`ComputeClient::capture`]
    /// (no arena) or on runtimes without arena support.
    pub fn arena_bytes(&self) -> u64 {
        self.client
            .context
            .lock()
            .graph_arena_bytes(self.id, self.stream_id)
    }
}

impl<R: Runtime> Drop for CapturedGraph<R> {
    fn drop(&mut self) {
        self.client.context.lock().graph_destroy(self.id);
    }
}

/// A SHARED capture pool (P4 of the CUDA-graph plan — the vLLM `graph_pool_handle`). Create one with
/// [`ComputeClient::capture_pool`] and capture several graphs into it with
/// [`ComputeClient::capture_arena_in_pool`]; they share ONE device arena (serially-replayed graphs
/// reuse each other's blocks), so N graphs cost ~1 graph's high-water instead of N×.
///
/// The pool's device blocks live until BOTH this handle and every [`CapturedGraph`] captured from it
/// are dropped (the pool is refcounted by the handle + its graphs). Dropping the handle while graphs
/// are still alive is fine — the blocks survive until the last graph goes.
pub struct CapturePoolHandle<R: Runtime> {
    client: ComputeClient<R>,
    id: u64,
    stream_id: StreamId,
}

impl<R: Runtime> Drop for CapturePoolHandle<R> {
    fn drop(&mut self) {
        self.client
            .context
            .lock()
            .capture_pool_release(self.id, self.stream_id);
    }
}
