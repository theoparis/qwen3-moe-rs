use cubecl::prelude::*;
use cubecl::server::Handle;
use cubecl::std::tensor::{
    View,
    layout::{
        Coords1d,
        linear::{LinearView, linear_view},
    },
};
use cubecl_common::{rand::get_seeded_rng, stub::Mutex};
use rand::{RngExt, SeedableRng, rngs::StdRng};

pub(crate) const N_VALUES_PER_THREAD: usize = 128;

/// Number of `u32` seed words the kernel reads (the TAUS88+LCG key). The device seed buffer that the
/// kernel dereferences holds exactly this many `u32`.
pub const N_SEEDS: usize = 4;

static SEED: Mutex<Option<StdRng>> = Mutex::new(None);

pub fn seed(seed: u64) {
    let rng = StdRng::seed_from_u64(seed);
    let mut seed = SEED.lock().unwrap();
    *seed = Some(rng);
}

/// Pseudo-random generator (DEFAULT eager path).
///
/// This is the original, zero-overhead path: the 4 fresh host seeds are passed as `ScalarArg`
/// IMMEDIATES (no device alloc, no H2D). On grid-constant HW they lower into the launch params
/// by-value — identical to the pre-C3 behaviour. This path is NOT CUDA-graph-capturable (the
/// immediates would freeze at capture, making every replay reuse identical noise); the opt-in
/// [`random_with_seed_handle`] entry exists for that. Keeping eager on immediates means every
/// `Tensor::random` call (incl. the GRPO rollout's eager Gumbel sampling) pays no per-call
/// seed-buffer allocation or upload.
pub(crate) fn random<F: RandomFamily, R: Runtime>(
    client: &ComputeClient<R>,
    prng: F::Runtime,
    output: TensorHandleRef<'_, R>,
    dtype: StorageType,
) -> Result<(), LaunchError> {
    let seeds = get_seeds();
    let args = prng.args();

    let cube_dim = CubeDim::new(client, output.size().div_ceil(N_VALUES_PER_THREAD));
    let cube_count = prng_cube_count(output.size(), cube_dim, N_VALUES_PER_THREAD);

    let output_line_size = 1;
    // TODO: Higher vectorization can add some correlation locally.
    //
    // let output_line_size = tensor_line_size_parallel(
    //     R::line_size_elem(&E::as_elem_native_unchecked()),
    //     output.shape,
    //     output.strides,
    //     output.strides.len() - 1,
    // );

    let address_type = output.required_address_type();
    let output = linear_view(client, &output, output_line_size);

    prng_kernel::launch::<F, R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        output,
        ScalarArg::new(seeds[0]),
        ScalarArg::new(seeds[1]),
        ScalarArg::new(seeds[2]),
        ScalarArg::new(seeds[3]),
        args,
        N_VALUES_PER_THREAD,
        output_line_size,
        dtype,
    )
}

/// Upload `N_SEEDS` host seeds into a FRESH device buffer (one alloc + H2D). A host-side helper for
/// the opt-in capturable path: it produces a persistent seed buffer the caller can later rewrite with
/// [`cubecl::client::ComputeClient::write_to_handle`] before each replay. For CUDA-graph capture this
/// buffer MUST be allocated OUTSIDE the captured region and outlive the graph (option (c)). The
/// DEFAULT eager [`random`] does NOT call this — it uses immediate seeds and never allocates.
pub fn create_seed_buffer<R: Runtime>(client: &ComputeClient<R>, seeds: &[u32; N_SEEDS]) -> Handle {
    let mut bytes = [0u8; N_SEEDS * 4];
    for (i, s) in seeds.iter().enumerate() {
        bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
    }
    client.create_from_slice(&bytes)
}

/// Launch the capturable `prng_kernel_seeded`, reading its `N_SEEDS` seeds from `seed_handle`, a
/// device buffer of `N_SEEDS` `u32`. This is the OPT-IN entry — the default eager [`random`] uses the
/// immediate-seed `prng_kernel` instead and never touches a seed buffer.
///
/// The seeds are bound as an `Array<u32>` (a device POINTER), not `ScalarArg` immediates, so a
/// captured kernel node bakes the buffer pointer; writing fresh seeds into that same buffer before
/// each replay therefore decorrelates the draws (PyTorch's capturable model, option (c)). The caller
/// owns `seed_handle`; for capture it must be allocated OUTSIDE the captured region and stay alive for
/// the graph's lifetime.
///
/// # P-final plumbing gap (NOT yet built)
///
/// This is the low-level seam. burn's `Tensor::random` does NOT reach here: it allocates a FRESH
/// internal seed buffer per call (the default [`random`] above draws host seeds and passes them as
/// immediates), so a region containing `Tensor::random` — the GRPO Gumbel sampler — is NOT capturable
/// as-is. To capture such a region, P-final must thread a PERSISTENT, externally-owned "generator
/// handle" (a `N_SEEDS`-u32 buffer allocated OUTSIDE the capture region) through burn-cubecl's prng
/// path down to THIS function, so the host can `write_to_handle` fresh seeds into it before each
/// replay. The `cudagraph_p3_rng_bench` external-handle pattern is the template. That plumbing is the
/// remaining P-final work; until then only direct callers of this entry (or `random_uniform_with_seeds`)
/// are capturable.
pub(crate) fn random_with_seed_handle<F: RandomFamily, R: Runtime>(
    client: &ComputeClient<R>,
    prng: F::Runtime,
    output: TensorHandleRef<'_, R>,
    dtype: StorageType,
    seed_handle: &Handle,
) -> Result<(), LaunchError> {
    let args = prng.args();

    let cube_dim = CubeDim::new(client, output.size().div_ceil(N_VALUES_PER_THREAD));
    let cube_count = prng_cube_count(output.size(), cube_dim, N_VALUES_PER_THREAD);

    let output_line_size = 1;
    // TODO: Higher vectorization can add some correlation locally.
    //
    // let output_line_size = tensor_line_size_parallel(
    //     R::line_size_elem(&E::as_elem_native_unchecked()),
    //     output.shape,
    //     output.strides,
    //     output.strides.len() - 1,
    // );

    let address_type = output.required_address_type();
    let output = linear_view(client, &output, output_line_size);

    // SAFETY: `seed_handle` holds at least `N_SEEDS` contiguous `u32` (from `create_seed_buffer`, or a
    // caller pre-allocated `N_SEEDS`-u32 buffer); the kernel reads exactly `N_SEEDS`, line size 1.
    let seeds = unsafe { ArrayArg::from_raw_parts::<u32>(seed_handle, N_SEEDS, 1) };

    prng_kernel_seeded::launch::<F, R>(
        client,
        cube_count,
        cube_dim,
        address_type,
        output,
        seeds,
        args,
        N_VALUES_PER_THREAD,
        output_line_size,
        dtype,
    )
}

fn prng_cube_count(num_elems: usize, cube_dim: CubeDim, n_values_per_thread: usize) -> CubeCount {
    let num_threads = f32::ceil(num_elems as f32 / n_values_per_thread as f32);
    let num_invocations = f32::ceil(num_threads / cube_dim.num_elems() as f32);
    let cubes_x = f32::ceil(f32::sqrt(num_invocations));
    let cubes_y = f32::ceil(num_invocations / cubes_x);

    CubeCount::Static(cubes_x as u32, cubes_y as u32, 1)
}

pub(crate) fn get_seeds() -> [u32; 4] {
    let mut seed = SEED.lock().unwrap();
    let mut rng: StdRng = match seed.take() {
        Some(rng_seeded) => rng_seeded,
        None => get_seeded_rng(),
    };
    let mut seeds: Vec<u32> = Vec::with_capacity(4);
    for _ in 0..4 {
        seeds.push(rng.random());
    }
    *seed = Some(rng);

    seeds.try_into().unwrap()
}

pub(crate) trait PrngArgs: Send + Sync + 'static {
    type Args: LaunchArg;

    fn args<'a, R: Runtime>(self) -> <Self::Args as LaunchArg>::RuntimeArg<'a, R>;
}

pub(crate) trait RandomFamily: Send + Sync + 'static + std::fmt::Debug {
    type Runtime: PrngRuntime;
}

#[cube]
pub(crate) trait PrngRuntime: Send + Sync + 'static + PrngArgs {
    #[allow(clippy::too_many_arguments)]
    fn inner_loop<E: Numeric>(
        args: Self::Args,
        write_index_base: usize,
        n_invocations: u32,
        #[comptime] n_values_per_thread: usize,
        #[comptime] line_size: usize,
        state_0: &mut u32,
        state_1: &mut u32,
        state_2: &mut u32,
        state_3: &mut u32,
        output: &mut View<Line<E>, Coords1d, ReadWrite>,
    );
}

type Args<F> = <<F as RandomFamily>::Runtime as PrngArgs>::Args;

/// DEFAULT eager kernel: the 4 seeds arrive as `ScalarArg` IMMEDIATES (zero device transfer). Used by
/// [`random`]. NOT CUDA-graph-capturable — the immediates freeze at capture.
#[cube(launch, address_type = "dynamic")]
fn prng_kernel<F: RandomFamily, E: Numeric>(
    output: &mut LinearView<Line<E>, ReadWrite>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Args<F>,
    #[comptime] n_values_per_thread: usize,
    #[comptime] line_size: usize,
    #[define(E)] _dtype: StorageType,
) {
    prng_body::<F, E>(
        output,
        seed_0,
        seed_1,
        seed_2,
        seed_3,
        args,
        n_values_per_thread,
        line_size,
    );
}

/// OPT-IN capturable kernel (component C3, option (c)): the 4 seeds live in a DEVICE buffer (a stable
/// captured pointer) instead of immediates, so a captured node bakes the buffer POINTER and re-reads
/// fresh seeds the host writes before each replay. Used only by [`random_with_seed_handle`]. The
/// mixing is identical to [`prng_kernel`]; only WHERE the seeds live differs.
#[cube(launch, address_type = "dynamic")]
fn prng_kernel_seeded<F: RandomFamily, E: Numeric>(
    output: &mut LinearView<Line<E>, ReadWrite>,
    seeds: &Array<u32>,
    args: Args<F>,
    #[comptime] n_values_per_thread: usize,
    #[comptime] line_size: usize,
    #[define(E)] _dtype: StorageType,
) {
    prng_body::<F, E>(
        output,
        seeds[0],
        seeds[1],
        seeds[2],
        seeds[3],
        args,
        n_values_per_thread,
        line_size,
    );
}

/// Shared prng body for both the immediate-seed ([`prng_kernel`]) and device-buffer-seed
/// ([`prng_kernel_seeded`]) launch variants — the RNG mixing is identical; only how the 4 seeds are
/// delivered to the kernel differs.
#[cube]
fn prng_body<F: RandomFamily, E: Numeric>(
    output: &mut LinearView<Line<E>, ReadWrite>,
    seed_0: u32,
    seed_1: u32,
    seed_2: u32,
    seed_3: u32,
    args: Args<F>,
    #[comptime] n_values_per_thread: usize,
    #[comptime] line_size: usize,
) {
    let cube_offset = CUBE_POS * CUBE_DIM as usize;

    let write_index_base = cube_offset * n_values_per_thread / line_size + UNIT_POS as usize;

    // Truncating position should be fine here, it's no issue if the seed repeats
    #[allow(arithmetic_overflow)]
    let thread_seed = 1000000007u32 * ABSOLUTE_POS as u32;

    let mut state_0 = thread_seed + seed_0;
    let mut state_1 = thread_seed + seed_1;
    let mut state_2 = thread_seed + seed_2;
    let mut state_3 = thread_seed + seed_3;

    // Creation of n_values_per_thread values, specific to the distribution
    F::Runtime::inner_loop(
        args,
        write_index_base,
        CUBE_DIM,
        n_values_per_thread,
        line_size,
        &mut state_0,
        &mut state_1,
        &mut state_2,
        &mut state_3,
        output,
    );
}

#[cube]
pub(crate) fn taus_step_0(z: u32) -> u32 {
    taus_step(z, 13u32, 19u32, 12u32, 4294967294u32)
}

#[cube]
pub(crate) fn taus_step_1(z: u32) -> u32 {
    taus_step(z, 2u32, 25u32, 4u32, 4294967288u32)
}

#[cube]
pub(crate) fn taus_step_2(z: u32) -> u32 {
    taus_step(z, 3u32, 11u32, 17u32, 4294967280u32)
}

#[cube]
fn taus_step(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = z << s1;
    let b = b ^ z;
    let b = b >> s2;
    let z = (z & m) << s3;
    z ^ b
}

#[cube]
pub(crate) fn lcg_step(z: u32) -> u32 {
    let a = 1664525u32;
    let b = 1013904223u32;

    z * a + b
}

/// Converts a `u32` into a `f32` in the unit interval `[0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_closed_open(int_random: u32) -> f32 {
    // Use upper 24 bits for f32 precision
    // https://lemire.me/blog/2017/02/28/how-many-floating-point-numbers-are-in-the-interval-01/
    let shifted = int_random >> 8;
    f32::cast_from(shifted) / 16777216.0 // 2^24
}

/// Converts a `u32` into a `f32` in the unit interval `(0.0, 1.0)`.
/// Used for generating random floats.
#[cube]
pub fn to_unit_interval_open(int_random: u32) -> f32 {
    // Use upper 23 bits to leave room for the offset
    let shifted = int_random >> 9;
    (f32::cast_from(shifted) + 1.0) / 8388609.0 // 2^23 + 1
}
