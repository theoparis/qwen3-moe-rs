//! Typed, safe wrapper around the Burn-Fusion custom-op bridge.
//!
//! Every planned custom CubeCL kernel (fused fp8 W8A16 GEMM, MoE grouped-GEMM, CUDA-graph
//! capture) reaches the model through the *same* mechanism proven in
//! [`examples/fusion_bridge_spike.rs`]: register an [`OperationIr::Custom`] on the default
//! `Cuda = Fusion<CubeBackend<CudaRuntime>>` stream, plus an [`Operation::execute`] that pulls the
//! inner [`CubeTensor`] handles out of the fusion [`HandleContainer`], launches a hand-written
//! `#[cube(launch)]` kernel, and hands the output handle back to the stream.
//!
//! That raw pattern is a "manual tri-contract" — three independent reviews (Codex gpt-5.5,
//! Opus 4.8, Gemini 3.1 Pro) all flagged it as the single highest-risk surface, because the fusion
//! engine cross-validates **none** of it. See `docs/VLLM_KERNELS.md` §0b for the 8 production rules.
//! This module turns the rules the type system *can* enforce into compile-/run-time guarantees so a
//! kernel author cannot silently violate them:
//!
//! * **Rule 1 (declare every input in BOTH places).** The caller hands each input to the builder
//!   exactly **once** (via [`CubeCustomOp::float_input`] / [`CubeCustomOp::int_input`]). The builder
//!   threads that single list into *both* [`OperationStreams::with_inputs`] *and*
//!   [`CustomOpIr::new`], so the two can never diverge. There is no API to register one without the
//!   other.
//! * **Rule 2 (declared output must match what `execute` allocates).** Each declared output
//!   `TensorIr{shape,dtype}` is cross-validated inside `execute` against the actual [`CubeTensor`]
//!   the kernel closure produced — logical shape, dtype, **and** buffer byte-size — and panics with
//!   a precise message on any drift. See [`validate_output`].
//! * **Rule 3 (thread dtype → byte-size dynamically).** Byte-sizes are always
//!   `shape.num_elements() * dtype.size()` ([`DType::size`]), never a hardcoded `size_of::<f32>()`,
//!   so a `bf16`/`i8` tensor is sized correctly.
//! * **Rule 4 (route Float vs Int handles).** Float inputs/outputs go through
//!   `get_float_tensor`/`register_float_tensor`; int (e.g. packed `i8`/`u8` fp8 weights) go through
//!   `get_int_tensor`/`register_int_tensor`. The handle kind is captured by the builder method name,
//!   so the wrong getter (which would panic) is unreachable.
//! * **Rule 6 (scalars live in the `Operation`, not `CustomOpIr`).** The kernel is an
//!   `Fn(&[CubeTensor]) -> Vec<CubeTensor>` closure; scalars are captured by value by the caller.
//!   `CustomOpIr` is tensors-only by construction, so a scalar *cannot* be smuggled in as a tensor.
//! * **Rule 7 (`execute` touches only the raw client + `HandleContainer`).** `execute` uses only the
//!   passed `HandleContainer` and the raw CubeCL `ComputeClient` carried by each `CubeTensor`; it
//!   never re-enters the fusion client (which would deadlock under the server lock).
//!
//! Rules that are **caller contracts** (the wrapper cannot see inside the kernel closure to enforce
//! them — documented, not mechanical):
//!
//! * **Rule 5** — never `into_contiguous` a pre-packed/swizzled weight (it triggers a layout-fixing
//!   copy that destroys the packing). The closure owns its launch; honor this when handling packed
//!   inputs.
//! * **Rule 8** — the 5 cuda-gated deps must move in lockstep with burn's pinned rev; skew fails at
//!   build time (a TypeId/downcast mismatch), not here.
//!
//! Additional caller contracts surfaced by a 3-voice review (Codex gpt-5.5 / Opus 4.8 / Gemini 3.1
//! Pro) of this wrapper — the type system cannot enforce these, so honor them:
//!
//! * **Never capture a `CubeTensor` (or model weight) INTO the kernel closure** — the BIGGEST hole
//!   (both Codex and Gemini, P1). `'static` forbids borrows but allows *moving* an owned tensor in;
//!   such a tensor is invisible to the fusion dependency graph (no read edge, no device check), so
//!   the engine may free/mutate it before the lazy closure runs → use-after-free / stale data. EVERY
//!   tensor the kernel reads MUST arrive via `float_input`/`int_input`; the closure may only capture
//!   plain scalars/config by value.
//! * **The kernel must return outputs in the SAME ORDER they were declared** — `launch` zips them
//!   positionally; two outputs of identical shape/dtype/kind returned swapped pass validation but are
//!   bound to the wrong downstream handles (silent corruption). Same for indexing inputs in the closure.
//! * **Single-device only (today):** the fusion client is taken from `inputs[0]`; all inputs must be
//!   on that same device/client. On this single-GB10 setup that always holds; a multi-GPU caller would
//!   need an explicit same-client assert (Codex+Gemini P0/P1).
//! * **Sub-byte packed weights (int4/W4A16):** the byte-size check uses `num_elements * dtype.size()`
//!   with Burn's smallest `U8` (1 B). fp8/e4m3-as-`u8` is exact; for true sub-byte packing, declare the
//!   PHYSICAL packed-byte shape as `U8` and carry the logical dims/packing as separate metadata.
//! * **Keep `execute` light** — it runs on the fusion server thread; a heavy host sync inside the
//!   closure starves the global scheduler. Launch the kernel and return.
//!
//! The wrapper is generic over the CubeCL runtime `R: CubeRuntime` but fixes the element carriers to
//! the ones the default `Cuda` backend uses (`f32` float / `i32` int / `u8` bool), so the inner
//! `BackendIr` is `CubeBackend<R, f32, i32, u8>` and the fusion handle is `CubeFusionHandle<R>` —
//! exactly the types the model already flows through.

use burn::tensor::{DType, Shape};

use burn_cubecl::CubeBackend;
use burn_cubecl::CubeRuntime;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;

use burn_cubecl_fusion::CubeFusionHandle;

use burn_fusion::FusionTensor;
use burn_fusion::stream::{Operation, OperationStreams};

use burn_ir::{CustomOpIr, HandleContainer, OperationIr, TensorIr, TensorStatus};

/// The inner (non-fusion) compute backend that the default `Cuda` wraps. Its float/int tensor
/// primitive is a [`CubeTensor`], which is what a `#[cube(launch)]` kernel runs against.
type InnerBackend<R> = CubeBackend<R, f32, i32, u8>;

/// The fusion runtime used by `Cuda = Fusion<CubeBackend<CudaRuntime, f32, i32, u8>>`.
type Fr<R> = FusionCubeRuntime<R, u8>;

/// Which kind of handle a tensor is carried by in the fusion [`HandleContainer`].
///
/// This decides `get_float_tensor` vs `get_int_tensor` (and the matching `register_*`). Picking the
/// wrong one panics at runtime (`get_float_tensor` rejects Int handles), so the kind is fixed by the
/// builder method the caller chose and is never inferred from a guess.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandleKind {
    Float,
    Int,
}

/// A declared output tensor: its logical shape, dtype, and handle kind. The wrapper mints the
/// fusion id and, inside `execute`, cross-validates the kernel's real [`CubeTensor`] against this.
struct OutputDecl {
    shape: Shape,
    dtype: DType,
    kind: HandleKind,
}

/// A typed builder for a Burn-Fusion custom op.
///
/// Construct it with [`CubeCustomOp::new`], declare inputs and outputs with the typed methods, then
/// call [`CubeCustomOp::launch`] with the kernel closure. The builder guarantees rules 1–4/6–7 by
/// construction (see the module docs).
///
/// # Example
///
/// ```ignore
/// // a:[M] (f32) ⊗ b:[N] (f32) → out:[M,N] (f32) rank-1 outer product, fused on-device.
/// let out = CubeCustomOp::<CudaRuntime>::new("outer_product")
///     .float_input(a_prim)             // registered in BOTH streams + CustomOpIr (rule 1)
///     .float_input(b_prim)
///     .float_output([m, n], DType::F32) // cross-validated in execute (rule 2)
///     .launch(move |inputs| {
///         let out = outer_product_kernel(&inputs[0], &inputs[1]); // your #[cube(launch)] kernel
///         vec![out]
///     });
/// let out: FusionTensor<_> = out.into_iter().next().unwrap();
/// ```
pub struct CubeCustomOp<R: CubeRuntime> {
    name: &'static str,
    /// The ONE inputs list. Threaded into both `OperationStreams::with_inputs` and `CustomOpIr`
    /// inputs in [`Self::launch`] — the structural guarantee behind rule 1.
    inputs: Vec<(FusionTensor<Fr<R>>, HandleKind)>,
    outputs: Vec<OutputDecl>,
}

impl<R: CubeRuntime> CubeCustomOp<R> {
    /// Start building a custom op. `name` is the opaque `CustomOpIr` id (used in panic messages).
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Declare a **float** input. Pass the fusion primitive of a float tensor
    /// (`tensor.into_primitive().tensor()`). Registered in both the stream and the op IR (rule 1),
    /// and pulled with `get_float_tensor` in `execute` (rule 4).
    pub fn float_input(mut self, tensor: FusionTensor<Fr<R>>) -> Self {
        self.inputs.push((tensor, HandleKind::Float));
        self
    }

    /// Declare an **int** input (e.g. packed `i8`/`u8` fp8 weights). Pass the fusion primitive of an
    /// int tensor. Pulled with `get_int_tensor` in `execute` (rule 4) — `get_float_tensor` would
    /// panic on this handle.
    pub fn int_input(mut self, tensor: FusionTensor<Fr<R>>) -> Self {
        self.inputs.push((tensor, HandleKind::Int));
        self
    }

    /// Declare a **float** output of the given shape and dtype. The kernel closure must produce a
    /// matching [`CubeTensor`] — shape, dtype, and buffer byte-size are cross-validated in `execute`
    /// (rule 2). The shape may differ from any input shape (the GEMM case).
    pub fn float_output(mut self, shape: impl Into<Shape>, dtype: DType) -> Self {
        self.outputs.push(OutputDecl {
            shape: shape.into(),
            dtype,
            kind: HandleKind::Float,
        });
        self
    }

    /// Declare an **int** output of the given shape and dtype. Registered with `register_int_tensor`
    /// in `execute` (rule 4).
    pub fn int_output(mut self, shape: impl Into<Shape>, dtype: DType) -> Self {
        self.outputs.push(OutputDecl {
            shape: shape.into(),
            dtype,
            kind: HandleKind::Int,
        });
        self
    }

    /// Register the op on the fusion stream and return the (lazy) output [`FusionTensor`]s, in the
    /// order they were declared.
    ///
    /// `kernel` receives the input [`CubeTensor`]s (in declared order) and must return one
    /// [`CubeTensor`] per declared output (in declared order). Scalars are captured by the closure
    /// (rule 6). The closure runs inside `execute`, on the raw CubeCL client only (rule 7).
    ///
    /// # Panics
    ///
    /// * at registration if no inputs were declared (the fusion client is located from input 0), or
    ///   if no outputs were declared;
    /// * later, inside `execute` (when the stream drains), if the kernel returns the wrong number of
    ///   outputs, or any output fails cross-validation (rule 2).
    pub fn launch<F>(self, kernel: F) -> Vec<FusionTensor<Fr<R>>>
    where
        F: Fn(&[CubeTensor<R>]) -> Vec<CubeTensor<R>> + Send + Sync + 'static,
    {
        assert!(
            !self.inputs.is_empty(),
            "custom op `{}`: at least one input is required (the fusion client is located from it)",
            self.name,
        );
        assert!(
            !self.outputs.is_empty(),
            "custom op `{}`: at least one output must be declared",
            self.name,
        );

        // The fusion client is shared by all tensors on the device; take it from the first input.
        let client = self.inputs[0].0.client.clone();

        // --- Rule 1: ONE inputs list → BOTH the stream registration AND the op IR. ---------------
        // (a) Record the input streams by *borrowing* the same list (must happen before we consume
        //     the tensors into their IR below).
        let streams = OperationStreams::with_inputs(self.inputs.iter().map(|(tensor, _)| tensor));

        // (b) Consume the same list into input IRs (and remember each input's handle kind so
        //     `execute` can route it). Because both (a) and (b) iterate the identical `self.inputs`,
        //     a caller cannot register an input in one place but forget the other.
        let mut input_irs: Vec<TensorIr> = Vec::with_capacity(self.inputs.len());
        let mut input_kinds: Vec<HandleKind> = Vec::with_capacity(self.inputs.len());
        for (tensor, kind) in self.inputs {
            input_kinds.push(kind);
            input_irs.push(tensor.into_ir());
        }

        // --- Declared outputs: mint a fresh uninitialized id + IR for each. ----------------------
        // The handle is filled in by `execute`; the shape/dtype here is the contract `execute`
        // validates against (rule 2).
        let mut output_irs: Vec<TensorIr> = Vec::with_capacity(self.outputs.len());
        let mut output_kinds: Vec<HandleKind> = Vec::with_capacity(self.outputs.len());
        for decl in self.outputs {
            let id = client.create_empty_handle();
            output_irs.push(TensorIr {
                id,
                shape: decl.shape,
                dtype: decl.dtype,
                status: TensorStatus::NotInit,
            });
            output_kinds.push(decl.kind);
        }

        // The op IR uses the SAME input_irs and output_irs (rule 1 closed).
        let desc = CustomOpIr::new(self.name, &input_irs, &output_irs);

        let op = CubeCustomOpExec::<R> {
            name: self.name,
            inputs: input_irs.into_iter().zip(input_kinds).collect(),
            outputs: output_irs.into_iter().zip(output_kinds).collect(),
            kernel: Box::new(kernel),
        };

        client.register(streams, OperationIr::Custom(desc), op)
    }
}

/// The opaque [`Operation`] registered on the fusion stream. Holds the cross-validation contract
/// (input/output IRs + their handle kinds) and the kernel closure. `execute` runs INLINE on the
/// thread that drains the stream, **while the fusion server mutex is held** (not a separate server
/// thread) — which is exactly why it must touch ONLY the passed `HandleContainer` + the raw CubeCL
/// client carried by each `CubeTensor`, never the fusion client (re-locking the same non-reentrant
/// mutex → deadlock). Rule 7.
struct CubeCustomOpExec<R: CubeRuntime> {
    name: &'static str,
    inputs: Vec<(TensorIr, HandleKind)>,
    outputs: Vec<(TensorIr, HandleKind)>,
    #[allow(clippy::type_complexity)]
    kernel: Box<dyn Fn(&[CubeTensor<R>]) -> Vec<CubeTensor<R>> + Send + Sync>,
}

impl<R: CubeRuntime> core::fmt::Debug for CubeCustomOpExec<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CubeCustomOpExec")
            .field("name", &self.name)
            .field("inputs", &self.inputs.len())
            .field("outputs", &self.outputs.len())
            .finish()
    }
}

impl<R: CubeRuntime> Operation<Fr<R>> for CubeCustomOpExec<R> {
    fn execute(&self, handles: &mut HandleContainer<CubeFusionHandle<R>>) {
        // 1. Pull each input from the fusion HandleContainer, routed by its declared kind (rule 4).
        //    `get_float_tensor` panics on an Int handle, so the kind tag (set at build time) is what
        //    keeps this sound.
        let mut inputs: Vec<CubeTensor<R>> = Vec::with_capacity(self.inputs.len());
        for (ir, kind) in &self.inputs {
            let tensor = match kind {
                HandleKind::Float => handles.get_float_tensor::<InnerBackend<R>>(ir),
                HandleKind::Int => handles.get_int_tensor::<InnerBackend<R>>(ir),
            };
            inputs.push(tensor);
        }

        // 2. Run the hand-written kernel on the raw CubeCL client (rule 7: no fusion client here).
        let outputs = (self.kernel)(&inputs);

        // 3. Arity check: the kernel must produce exactly the declared outputs.
        assert_eq!(
            outputs.len(),
            self.outputs.len(),
            "custom op `{}`: kernel produced {} output(s) but {} were declared",
            self.name,
            outputs.len(),
            self.outputs.len(),
        );

        // 4. Cross-validate each output against its declaration (rule 2), then register it back into
        //    the stream routed by kind (rule 4).
        for (idx, (actual, (ir, kind))) in outputs.into_iter().zip(self.outputs.iter()).enumerate()
        {
            validate_output(self.name, idx, &actual, ir);
            match kind {
                HandleKind::Float => {
                    handles.register_float_tensor::<InnerBackend<R>>(&ir.id, actual)
                }
                HandleKind::Int => handles.register_int_tensor::<InnerBackend<R>>(&ir.id, actual),
            }
        }
    }
}

/// Cross-validate a kernel-produced output against the tensor that was **declared** to the fusion
/// stream (rule 2). The lazy `FusionTensor` the engine handed downstream consumers was built from
/// the *declared* `shape`/`dtype`; if the real allocation drifts from that, every downstream op
/// reads corrupt metadata. The engine checks none of this, so we do — and panic loudly on drift.
///
/// Three independent checks:
/// * logical **shape** (catches a wrong declared shape, e.g. `[M,N]` vs `[M,N+1]`);
/// * **dtype** (catches a bf16 buffer declared as f32, etc.);
/// * **buffer byte-size**, computed dynamically as `num_elements * dtype.size()` (rule 3) — a
///   backstop that catches an under-/over-allocated buffer even when shape+dtype look right.
fn validate_output<R: CubeRuntime>(
    name: &str,
    idx: usize,
    actual: &CubeTensor<R>,
    declared: &TensorIr,
) {
    let got_shape = actual.meta.shape();
    assert!(
        *got_shape == declared.shape,
        "custom op `{name}` output #{idx}: SHAPE mismatch — kernel produced {got_shape:?} but the \
         op declared {declared_shape:?}. The declared shape is what downstream ops see; they must \
         match exactly (rule 2).",
        declared_shape = declared.shape,
    );

    assert!(
        actual.dtype == declared.dtype,
        "custom op `{name}` output #{idx}: DTYPE mismatch — kernel produced {:?} but the op \
         declared {:?} (rule 2/3).",
        actual.dtype,
        declared.dtype,
    );

    // NOTE on strides/contiguity: we do NOT require the output to be contiguous. The idiomatic GPU
    // allocator (`MemoryLayoutStrategy::Optimized`) pitch-pads the last dim and adjusts strides, so a
    // legitimate CMMA/CUTLASS GEMM output is strided + over-allocated; the real strides propagate to
    // downstream ops via the handle, so a strided output is correct. (Opus review corrected an earlier
    // contiguity assert that would falsely panic on the first real pitched-allocator kernel.) We pin
    // logical shape + dtype (what downstream reads) and a lower-bound buffer size; strides are the
    // kernel's business.

    // BYTE-SIZE: byte count derived dynamically from the dtype, never a hardcoded element size
    // (rule 3). Use `>=`, not `==` (all three reviewers): the pitched allocator + memory-pool bucket
    // rounding legitimately make `handle.size()` EXCEED the logical size (and `size()` ignores view
    // offsets), so `==` would falsely panic on a correct kernel. The real danger is an UNDER-allocated
    // buffer (out-of-bounds read downstream), which `>=` still catches.
    let declared_bytes = declared.shape.num_elements() * declared.dtype.size();
    let got_bytes = actual.handle.size() as usize;
    assert!(
        got_bytes >= declared_bytes,
        "custom op `{name}` output #{idx}: BUFFER UNDER-allocated — kernel buffer is {got_bytes} B \
         but the declaration needs {declared_bytes} B \
         ({num} elems × {elem} B/elem for {dtype:?}). A short buffer is an out-of-bounds read \
         downstream (rule 2/3).",
        num = declared.shape.num_elements(),
        elem = declared.dtype.size(),
        dtype = declared.dtype,
    );
}
