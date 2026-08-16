//! SPIKE — can we ADOPT `cubek-attention` (Blackwell flash-attention, already in our dep tree at
//! /workspace/cubek/crates/cubek-attention) to replace our slow hand-rolled decode attention?
//!
//! This mirrors the crate's own working test harness
//! (cubek-attention/tests/attention/{launcher,reference}.rs): it builds raw cubecl
//! `TensorHandle<CudaRuntime>` inputs, calls `cubek_attention::launch::<CudaRuntime>(..)` on the real
//! GB10 (sm_121), and compares the device output to a CPU f32 softmax(QKᵀ/√d + mask)·V reference.
//!
//! It answers, on hardware:
//!   1. sm_121 SMOKE   — does it COMPILE + RUN without the broadcast-mask corruption that breaks
//!                       Burn's fused `module::attention`? (We test BlackboxAccelerated tensor-core +
//!                       Unit, in f32 AND bf16.)
//!   2. GQA            — does it accept num_kv_heads < num_heads (K/V with fewer heads, Q with more),
//!                       broadcasting internally, or does it need a physical repeat?
//!   3. DECODE shape   — query seq=1 over K/V seq=T (the real target), and is the KV length a free
//!                       runtime value (toward O(pos))?
//!   4. NUMERICS       — eps-equal to the reference?
//!   5. PERF (rough)   — cubek-attention vs Burn's `attention_fallback` on the same decode shape.
//!
//! Run:
//!   RUSTFLAGS="-C target-feature=+fp16" \
//!     cargo run --release --features cuda --example cubek_attn_spike 2>&1 | tail -80

use cubecl::client::ComputeClient;
use cubecl::cuda::{CudaDevice, CudaRuntime};
use cubecl::prelude::*; // CubeElement (as_bytes/from_bytes), CubePrimitive (as_type_native_unchecked), StorageType
use cubecl::std::tensor::TensorHandle;
use half::{bf16, f16};

use cubek_attention::definition::{AttentionGlobalTypes, AttentionOptions};
use cubek_attention::launch::{BlueprintStrategy, Strategy, launch};

type R = CudaRuntime;
type Client = ComputeClient<R>;

// ------------------------------------------------------------------------------------------------
// Raw-handle helpers (mirror cubek-test-utils: create_from_slice -> contiguous TensorHandle)
// ------------------------------------------------------------------------------------------------

fn f32_handle(client: &Client, data: &[f32], shape: &[usize]) -> TensorHandle<R> {
    let handle = client.create_from_slice(f32::as_bytes(data));
    TensorHandle::new_contiguous(shape.to_vec(), handle, f32::as_type_native_unchecked())
}

fn bf16_handle(client: &Client, data: &[f32], shape: &[usize]) -> TensorHandle<R> {
    let bf: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    let handle = client.create_from_slice(bf16::as_bytes(&bf));
    TensorHandle::new_contiguous(shape.to_vec(), handle, bf16::as_type_native_unchecked())
}

fn f16_handle(client: &Client, data: &[f32], shape: &[usize]) -> TensorHandle<R> {
    let hf: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    let handle = client.create_from_slice(f16::as_bytes(&hf));
    TensorHandle::new_contiguous(shape.to_vec(), handle, f16::as_type_native_unchecked())
}

fn u8_mask_handle(client: &Client, mask: &[u8], shape: &[usize]) -> TensorHandle<R> {
    let handle = client.create_from_slice(u8::as_bytes(mask));
    TensorHandle::new_contiguous(shape.to_vec(), handle, u8::as_type_native_unchecked())
}

fn empty_out(client: &Client, shape: &[usize], storage: StorageType) -> TensorHandle<R> {
    let n: usize = shape.iter().product();
    let handle = client.empty(n * storage.size());
    TensorHandle::new_contiguous(shape.to_vec(), handle, storage)
}

fn read_f32(client: &Client, out: &TensorHandle<R>) -> Vec<f32> {
    let bytes = client.read_one_tensor(out.as_copy_descriptor());
    f32::from_bytes(&bytes).to_vec()
}

fn read_bf16(client: &Client, out: &TensorHandle<R>) -> Vec<f32> {
    let bytes = client.read_one_tensor(out.as_copy_descriptor());
    bf16::from_bytes(&bytes)
        .iter()
        .map(|x| x.to_f32())
        .collect()
}
fn read_f16(client: &Client, out: &TensorHandle<R>) -> Vec<f32> {
    let bytes = client.read_one_tensor(out.as_copy_descriptor());
    f16::from_bytes(&bytes).iter().map(|x| x.to_f32()).collect()
}

fn block_sync(client: &Client) -> Result<(), String> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("{e:?}"))
}

// ------------------------------------------------------------------------------------------------
// Host data + CPU f32 reference (the trusted oracle)
// ------------------------------------------------------------------------------------------------

struct Lcg(u64);
impl Lcg {
    fn new(s: u64) -> Self {
        Lcg(s)
    }
    fn next(&mut self, amp: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = ((self.0 >> 33) as u32) as f32 / (u32::MAX as f32);
        (u * 2.0 - 1.0) * amp
    }
}
fn make_data(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.next(1.0)).collect()
}

/// CPU f32 reference. q:[b,h,sq,d]; k,v:[b,hkv,skv,d] (GQA: head h reads kv head h/(h_q/h_kv)).
/// `mask` (if Some) is [b,h,sq,skv] row-major, true => masked out. `causal` masks j>i.
#[allow(clippy::too_many_arguments)]
fn reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    hq: usize,
    hkv: usize,
    sq: usize,
    skv: usize,
    d: usize,
    causal: bool,
    mask: Option<&[bool]>,
) -> Vec<f32> {
    let n_rep = hq / hkv;
    let scale = (d as f32).sqrt().recip();
    let mut out = vec![0.0f32; b * hq * sq * d];
    for bi in 0..b {
        for h in 0..hq {
            let hk = h / n_rep;
            for i in 0..sq {
                let mut scores = vec![f32::NEG_INFINITY; skv];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..skv {
                    let masked = (causal && j > i)
                        || mask
                            .map(|m| m[((bi * hq + h) * sq + i) * skv + j])
                            .unwrap_or(false);
                    if masked {
                        continue;
                    }
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += q[((bi * hq + h) * sq + i) * d + dd]
                            * k[((bi * hkv + hk) * skv + j) * d + dd];
                    }
                    dot *= scale;
                    scores[j] = dot;
                    if dot > mx {
                        mx = dot;
                    }
                }
                let mut denom = 0.0f32;
                for j in 0..skv {
                    if scores[j] > f32::NEG_INFINITY {
                        scores[j] = (scores[j] - mx).exp();
                        denom += scores[j];
                    } else {
                        scores[j] = 0.0;
                    }
                }
                let denom = if denom > 1e-20 { denom } else { 1e-20 };
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..skv {
                        acc += scores[j] * v[((bi * hkv + hk) * skv + j) * d + dd];
                    }
                    out[((bi * hq + h) * sq + i) * d + dd] = acc / denom;
                }
            }
        }
    }
    out
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return f32::NAN;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}
fn has_nan(a: &[f32]) -> bool {
    a.iter().any(|x| x.is_nan())
}

// ------------------------------------------------------------------------------------------------
// One launch + readback. Returns Ok(device output as f32) or Err(setup/exec error string).
// ------------------------------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum Dtype {
    F32,
    Bf16,
    F16,
}
#[derive(Clone, Copy)]
enum Strat {
    Accel,
    Unit,
}

#[allow(clippy::too_many_arguments)]
fn run(
    client: &Client,
    strat: Strat,
    dtype: Dtype,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_shape: &[usize],  // [b,hq,sq,d]
    kv_shape: &[usize], // [b,hkv,skv,d]
    mask: Option<(&[u8], &[usize])>,
    causal: bool,
) -> Result<Vec<f32>, String> {
    let strategy = match strat {
        Strat::Accel => Strategy::BlackboxAccelerated(BlueprintStrategy::Inferred(())),
        Strat::Unit => Strategy::Unit(BlueprintStrategy::Inferred(())),
    };
    let float_st = match dtype {
        Dtype::F32 => f32::as_type_native_unchecked(),
        Dtype::Bf16 => bf16::as_type_native_unchecked(),
        Dtype::F16 => f16::as_type_native_unchecked(),
    };
    let mask_st = AttentionGlobalTypes::mask_dtype(client);
    let gtypes = AttentionGlobalTypes::from_single_float_dtype(float_st, mask_st);

    let (qh, kh, vh) = match dtype {
        Dtype::F32 => (
            f32_handle(client, q, q_shape),
            f32_handle(client, k, kv_shape),
            f32_handle(client, v, kv_shape),
        ),
        Dtype::Bf16 => (
            bf16_handle(client, q, q_shape),
            bf16_handle(client, k, kv_shape),
            bf16_handle(client, v, kv_shape),
        ),
        Dtype::F16 => (
            f16_handle(client, q, q_shape),
            f16_handle(client, k, kv_shape),
            f16_handle(client, v, kv_shape),
        ),
    };
    let mh = mask.map(|(m, sh)| u8_mask_handle(client, m, sh));

    // output shape = [b, hq, sq, val_dim==d]
    let out_shape = [q_shape[0], q_shape[1], q_shape[2], kv_shape[3]];
    let out = empty_out(client, &out_shape, float_st);

    let opts = AttentionOptions {
        causal,
        ..Default::default()
    };
    launch::<R>(strategy, client, qh, kh, vh, mh, out.clone(), &gtypes, opts)
        .map_err(|e| format!("setup: {e:?}"))?;
    block_sync(client).map_err(|e| format!("exec: {e}"))?;

    Ok(match dtype {
        Dtype::F32 => read_f32(client, &out),
        Dtype::Bf16 => read_bf16(client, &out),
        Dtype::F16 => read_f16(client, &out),
    })
}

fn report(label: &str, dev: &Result<Vec<f32>, String>, oracle: &[f32], thresh: f32) -> bool {
    match dev {
        Err(e) => {
            println!("  {label:30} -> ERROR: {e}");
            false
        }
        Ok(out) => {
            if out.len() != oracle.len() {
                println!(
                    "  {label:30} -> LEN MISMATCH dev={} oracle={}",
                    out.len(),
                    oracle.len()
                );
                return false;
            }
            let cos = cosine(out, oracle);
            let mad = max_abs_diff(out, oracle);
            let nan = has_nan(out);
            let ok = !nan && cos > thresh;
            println!(
                "  {label:30} -> cos={cos:.6} max_abs_diff={mad:.3e}{} {}",
                if nan { " NAN!" } else { "" },
                if ok { "[PASS]" } else { "[FAIL]" }
            );
            ok
        }
    }
}

fn main() {
    let device = Device::cuda(0);
    let client = CudaRuntime::client(&device);
    println!("device: {device:?} | RAW cubek_attention::launch on CudaRuntime (sm_121 / GB10)");
    println!("oracle: CPU f32 softmax(QKᵀ/√d + mask)·V  (independent reference)\n");

    let d = 128usize; // Qwen3 head_dim

    // =========================================================================================
    // TEST 1 — sm_121 SMOKE + numerics: small prefill, EQUAL heads, causal=true, NO mask tensor.
    //          (cubek uses a POSITIONAL `causal` flag, not a broadcast [1,1,s,s] mask — so the
    //           specific corruption that breaks Burn's fused module::attention cannot arise here.)
    // =========================================================================================
    println!("== TEST 1: prefill smoke  b1 h4 sq8 skv8 d128  causal=true (no mask tensor) ==");
    {
        let (b, h, sq, skv) = (1, 4, 8, 8);
        let q = make_data(b * h * sq * d, 1);
        let k = make_data(b * h * skv * d, 2);
        let v = make_data(b * h * skv * d, 3);
        let oracle = reference(&q, &k, &v, b, h, h, sq, skv, d, true, None);
        let qs = [b, h, sq, d];
        let kvs = [b, h, skv, d];
        let r_acc_f32 = run(
            &client,
            Strat::Accel,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            true,
        );
        report("Accelerated f32", &r_acc_f32, &oracle, 0.99);
        let r_unit_f32 = run(
            &client,
            Strat::Unit,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            true,
        );
        report("Unit f32", &r_unit_f32, &oracle, 0.9999);
        let r_acc_bf16 = run(
            &client,
            Strat::Accel,
            Dtype::Bf16,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            true,
        );
        report("Accelerated bf16", &r_acc_bf16, &oracle, 0.98);
        let r_unit_bf16 = run(
            &client,
            Strat::Unit,
            Dtype::Bf16,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            true,
        );
        report("Unit bf16", &r_unit_bf16, &oracle, 0.98);
        // Does ANY tensor-core dtype compile on sm_121? Probe f16 (sometimes registered when bf16 isn't).
        let r_acc_f16 = run(
            &client,
            Strat::Accel,
            Dtype::F16,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            true,
        );
        report("Accelerated f16", &r_acc_f16, &oracle, 0.98);
    }

    // =========================================================================================
    // TEST 3a — DECODE shape, VARIABLE KV length, no causal/mask (single query attends ALL keys).
    //           This is the O(pos) decode path: KV length is a free runtime dim (re-launch grows it).
    // =========================================================================================
    println!(
        "\n== TEST 3a: DECODE  b1 h8 sq1 skv={{64,1024}} d128  causal=false, no mask (attend all) =="
    );
    for &skv in &[64usize, 1024usize] {
        let (b, h, sq) = (1, 8, 1);
        let q = make_data(b * h * sq * d, 10 + skv as u64);
        let k = make_data(b * h * skv * d, 20 + skv as u64);
        let v = make_data(b * h * skv * d, 30 + skv as u64);
        let oracle = reference(&q, &k, &v, b, h, h, sq, skv, d, false, None);
        let qs = [b, h, sq, d];
        let kvs = [b, h, skv, d];
        let r_acc = run(
            &client,
            Strat::Accel,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            false,
        );
        report(&format!("Accelerated f32 skv={skv}"), &r_acc, &oracle, 0.99);
        let r_unit = run(
            &client,
            Strat::Unit,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            false,
        );
        report(&format!("Unit f32 skv={skv}"), &r_unit, &oracle, 0.9999);
        let r_bf16 = run(
            &client,
            Strat::Accel,
            Dtype::Bf16,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            false,
        );
        report(
            &format!("Accelerated bf16 skv={skv}"),
            &r_bf16,
            &oracle,
            0.98,
        );
    }

    // =========================================================================================
    // TEST 3b — DECODE, FIXED T_max with a MATERIALIZED mask masking future cols (the CUDA-graph,
    //           fixed-shape path: shape is constant, only the mask's boundary moves with `pos`).
    //           mask[.. , j] = 1 (mask out) for j > pos. reference attends keys 0..=pos.
    // =========================================================================================
    println!("\n== TEST 3b: DECODE fixed T_max=64, materialized mask (cols>pos masked), pos=20 ==");
    {
        let (b, h, sq, skv, pos) = (1, 8, 1, 64, 20usize);
        let q = make_data(b * h * sq * d, 100);
        let k = make_data(b * h * skv * d, 200);
        let v = make_data(b * h * skv * d, 300);
        // bool mask + u8 mask, shape [b,h,sq,skv], true/1 => mask out (future/unwritten cols).
        let mut mb = vec![false; b * h * sq * skv];
        let mut mu = vec![0u8; b * h * sq * skv];
        for idx in 0..(b * h * sq) {
            for j in 0..skv {
                if j > pos {
                    mb[idx * skv + j] = true;
                    mu[idx * skv + j] = 1;
                }
            }
        }
        let oracle = reference(&q, &k, &v, b, h, h, sq, skv, d, false, Some(&mb));
        let qs = [b, h, sq, d];
        let kvs = [b, h, skv, d];
        let ms = [b, h, sq, skv];
        let r_acc = run(
            &client,
            Strat::Accel,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            Some((&mu, &ms)),
            false,
        );
        report("Accelerated f32 (masked)", &r_acc, &oracle, 0.99);
        let r_unit = run(
            &client,
            Strat::Unit,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            Some((&mu, &ms)),
            false,
        );
        report("Unit f32 (masked)", &r_unit, &oracle, 0.9999);
    }

    // =========================================================================================
    // TEST 5 — PERF (rough): cubek-attention vs Burn attention_fallback on the SAME decode shape.
    // =========================================================================================
    println!("\n== TEST 5: PERF (rough)  decode b1 h8 sq1 skv1024 d128 f32 (attend all) ==");
    perf(&client, d);

    // =========================================================================================
    // TEST 2 — GQA probe, run LAST: K/V with FEWER heads than Q. The launch derives num_heads from
    //          query.shape[1] ONLY and applies it to K/V (no kv-heads field) — so this is expected
    //          to mismatch the GQA reference (or fault). A device fault here can't taint earlier
    //          results, which are already printed.  Compared against a GQA (expand) reference.
    // =========================================================================================
    println!(
        "\n== TEST 2: GQA probe (LAST)  q heads=8, K/V heads=2, sq1 skv64 d128  causal=false =="
    );
    {
        let (b, hq, hkv, sq, skv) = (1, 8, 2, 1, 64);
        let q = make_data(b * hq * sq * d, 7);
        let k = make_data(b * hkv * skv * d, 8);
        let v = make_data(b * hkv * skv * d, 9);
        let oracle = reference(&q, &k, &v, b, hq, hkv, sq, skv, d, false, None); // expands 2->8
        let qs = [b, hq, sq, d];
        let kvs = [b, hkv, skv, d];
        let r = run(
            &client,
            Strat::Unit,
            Dtype::F32,
            &q,
            &k,
            &v,
            &qs,
            &kvs,
            None,
            false,
        );
        let ok = report("Unit f32 (mismatched heads)", &r, &oracle, 0.99);
        println!(
            "  -> GQA verdict: {}",
            if ok {
                "matched the expand reference => possibly native broadcast (unexpected)"
            } else {
                "did NOT match (or errored) => NO native GQA; physical repeat (4->32) still required"
            }
        );
    }

    println!("\n(see the spike's returned analysis for the GO/NO-GO verdict)");
}

// ------------------------------------------------------------------------------------------------
// Rough perf: cubek-attention launch loop vs Burn attention_fallback loop, same f32 decode shape.
// ------------------------------------------------------------------------------------------------
fn perf(client: &Client, d: usize) {
    use std::time::Instant;
    let (b, h, sq, skv) = (1usize, 8usize, 1usize, 1024usize);
    let iters = 200u32;

    let q = make_data(b * h * sq * d, 11);
    let k = make_data(b * h * skv * d, 12);
    let v = make_data(b * h * skv * d, 13);
    let qs = [b, h, sq, d];
    let kvs = [b, h, skv, d];

    // --- cubek-attention (Unit f32 — the ONLY routine that compiles on sm_121; Accelerated/CMMA
    //     is unavailable here). Inputs uploaded ONCE and reused, so this is kernel-vs-kernel. ---
    let float_st = f32::as_type_native_unchecked();
    let mask_st = AttentionGlobalTypes::mask_dtype(client);
    let gtypes = AttentionGlobalTypes::from_single_float_dtype(float_st, mask_st);
    let out_shape = [b, h, sq, d];
    let qh = f32_handle(client, &q, &qs);
    let kh = f32_handle(client, &k, &kvs);
    let vh = f32_handle(client, &v, &kvs);

    let do_launch = |client: &Client| {
        let out = empty_out(client, &out_shape, float_st);
        launch::<R>(
            Strategy::Unit(BlueprintStrategy::Inferred(())),
            client,
            qh.clone(),
            kh.clone(),
            vh.clone(),
            None,
            out,
            &gtypes,
            AttentionOptions {
                causal: false,
                ..Default::default()
            },
        )
    };
    // warmup
    for _ in 0..5 {
        let _ = do_launch(client);
    }
    let _ = block_sync(client);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = do_launch(client);
    }
    let _ = block_sync(client);
    let cubek_us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "  cubek-attention (Unit f32, SIMT):  {cubek_us:8.2} us/call  (Accelerated/CMMA unavailable on sm_121)"
    );

    // --- Burn attention_fallback (the path we use today) ---
    use burn::backend::cuda::Cuda;
    use burn::tensor::{Device, TensorData, ops::AttentionModuleOptions};
    type T4 = burn::tensor::Tensor<4>;
    let bdev = <Cuda as burn::prelude::Device>::Device::default();
    let qb = T4::from_data(TensorData::new(q.clone(), qs), &bdev);
    let kb = T4::from_data(TensorData::new(k.clone(), kvs), &bdev);
    let vb = T4::from_data(TensorData::new(v.clone(), kvs), &bdev);
    // warmup
    let mut last = burn::tensor::module::attention_fallback(
        qb.clone(),
        kb.clone(),
        vb.clone(),
        None,
        None,
        AttentionModuleOptions::default(),
    );
    let _ = last.clone().into_data();
    let t1 = Instant::now();
    for _ in 0..iters {
        last = burn::tensor::module::attention_fallback(
            qb.clone(),
            kb.clone(),
            vb.clone(),
            None,
            None,
            AttentionModuleOptions::default(),
        );
    }
    let _ = last.into_data(); // force the whole queue
    let fb_us = t1.elapsed().as_secs_f64() * 1e6 / iters as f64;
    println!(
        "  Burn attention_fallback (f32):     {fb_us:8.2} us/call  (reference SDPA, what we use now)"
    );
    println!(
        "  -> cubek is ~{:.2}x {} than the fallback on this shape (rough; input-upload included for cubek)",
        (fb_us / cubek_us).max(cubek_us / fb_us),
        if cubek_us < fb_us { "FASTER" } else { "SLOWER" }
    );
}
