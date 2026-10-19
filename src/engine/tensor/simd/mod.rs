//! SIMD dispatch infrastructure and shared kernel utilities.
//!
//! Automatically selects the highest-performance SIMD tier supported by the
//! host CPU at runtime, falling back to portable scalar kernels when required.

pub mod scalar;

#[cfg(target_arch = "x86_64")]
pub mod sse41;

#[cfg(target_arch = "x86_64")]
pub mod avx2;

#[cfg(target_arch = "x86_64")]
pub mod vnni;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub mod neon;

use std::sync::{LazyLock, OnceLock};

// Re-exports
pub use scalar::quantize_row_scalar;

#[cfg(target_arch = "x86_64")]
pub use avx2::quantize_row_avx2;

#[cfg(target_arch = "x86_64")]
pub use sse41::{quantize_row_sse2, quantize_row_sse41};

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
pub use neon::quantize_row_neon;

/// Single-row (m == 1) matmul with bias: `out[j] = bias[j] + Σ_t h[t]·b[t·n+j]`.
///
/// `b` is a `[k, n]` row-major matrix (the shortlist's transposed gather of
/// embedding rows). Dispatches to the same runtime-selected SIMD kernel as
/// the full matmul with `m = 1`, writing into a caller-owned buffer so the
/// autoregressive loop allocates nothing per step.
#[inline]
pub fn gemv_biased(h: &[f32], b: &[f32], k: usize, n: usize, bias: &[f32], out: &mut [f32]) {
    debug_assert_eq!(h.len(), k, "gemv_biased: h length");
    debug_assert_eq!(bias.len(), n, "gemv_biased: bias length");
    debug_assert_eq!(out.len(), n, "gemv_biased: out length");
    debug_assert_eq!(b.len(), k * n, "gemv_biased: b length");
    // SAFETY: dispatch fn from CPU feature flags; slices are valid.
    unsafe { matmul_row_f32_dispatch()(h, b, k, n, Some(bias), out, false) };
}

/// i8 `[1, K] @ [K, N]` single-row kernel with fused bias + dequantization
/// (+ optional fused `relu` epilogue), selected at runtime.
pub type I8Row =
    unsafe fn(&[i8], &[i8], usize, usize, Option<&[f32]>, Option<&[i32]>, &mut [f32], f32, bool);

#[inline]
pub fn matmul_row_i8_dispatch() -> I8Row {
    use crate::utils::arch::SimdLevel;
    static FN: LazyLock<I8Row> = LazyLock::new(|| match crate::utils::arch::simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => {
            if crate::utils::arch::has_avxvnni() {
                vnni::matmul_row_i8_avxvnni
            } else {
                avx2::matmul_row_i8_avx2
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 | SimdLevel::Avx => sse41::matmul_row_i8_sse41,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon if std::arch::is_aarch64_feature_detected!("dotprod") => {
            crate::engine::arm_dotprod::matmul_row_i8_dotprod
        }
        _ => scalar::matmul_row_i8_scalar,
    });
    *FN
}

// ---------------------------------------------------------------------------
// SIMD dispatch
//
// Kernels are selected once per process from the running CPU's feature flags
// and cached, so the hot inference loops pay a single indirect call instead
// of a feature check per matmul. The caller remains platform-agnostic and
// only sees a `unsafe fn` pointer.
//
// All kernels are *N-blocked*: the output columns are walked in tiles of
// `NR` (the vector width), and inside each tile the whole K dot product is
// accumulated into `NR` in-register accumulators per output row. Because the
// B-panel `[k, NR]` for one tile is ~LR_KB = (k·NR·4) bytes, it stays resident
// in L1 and is reused across all M rows — cutting DRAM traffic from
// `m × |B|` down to `|B|` (plus one L1 hit per row). This is the dominant
// memory-traffic fix for the FFN (k=384, n=1536) and attention projections.
// ---------------------------------------------------------------------------

/// f32 `M×K @ K×N` kernel, selected at runtime. `bias` (optional) is added to
/// every output row; `relu` applies `max(0, x)` fused in the store step.
pub type F32Matmul =
    unsafe fn(&[f32], &[f32], usize, usize, usize, Option<&[f32]>, &mut [f32], bool);

/// i8 `M×K @ K×N → i32` (dequantized by `denom`), optionally + bias, with a
/// fused `relu` epilogue on the dequantized value. `row_sums` carries the
/// per-column `128·Σw` correction precomputed at load time (`None` falls
/// back to an in-kernel recomputation).
pub type I8Matmul = unsafe fn(
    &[i8],
    &[i8],
    usize,
    usize,
    usize,
    Option<&[f32]>,
    Option<&[i32]>,
    &mut [f32],
    f32,
    bool,
);

#[inline]
pub fn matmul_f32_dispatch() -> F32Matmul {
    use crate::utils::arch::SimdLevel;
    static FN: OnceLock<F32Matmul> = OnceLock::new();
    *FN.get_or_init(|| match crate::utils::arch::simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::matmul_avx2_fma,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx => avx2::matmul_avx,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => sse41::matmul_sse41,
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        SimdLevel::Neon => neon::matmul_neon,
        _ => scalar::matmul_scalar,
    })
}

#[inline]
pub fn matmul_i8_dispatch() -> I8Matmul {
    use crate::utils::arch::SimdLevel;
    static FN: LazyLock<I8Matmul> = LazyLock::new(|| match crate::utils::arch::simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => {
            if crate::utils::arch::has_avxvnni() {
                vnni::matmul_i8_avxvnni
            } else {
                avx2::matmul_i8_avx2
            }
        }
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 | SimdLevel::Avx => sse41::matmul_i8_sse41,
        #[cfg(target_arch = "aarch64")]
        SimdLevel::Neon if std::arch::is_aarch64_feature_detected!("dotprod") => {
            crate::engine::arm_dotprod::matmul_i8_dotprod
        }
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        SimdLevel::Neon => neon::matmul_i8_neon,
        _ => scalar::matmul_i8_scalar,
    });
    *FN
}

// ---------------------------------------------------------------------------
// M-parallel GEMM
//
// The encoder path runs non-autoregressive `m = seq_len` matmuls whose output
// rows are independent. Splitting M across a shared rayon pool lets one
// translation use idle cores while keeping bit-identical results: every row is
// written by the same runtime-selected kernel with the same accumulation order
// (exact i32 for intgemm), so outputs match the sequential path byte for byte.
// Only M is split — the N-blocked kernels keep their L1-resident B-panel for
// the rows a worker owns, so per-worker weight traffic stays one panel per
// tile rather than one per row.
//
// Gating: a minimum work threshold amortises the scope setup, and each chunk
// keeps at least `GEMM_MIN_ROWS` rows so a worker never runs the N-blocked
// kernel on a sliver that loses B-panel reuse. The rayon global pool is
// bounded by the machine's thread count, so concurrent requests sharing it
// cannot oversubscribe the CPU.
// ---------------------------------------------------------------------------

/// Minimum matmul FMAs (m·k·n) before the M dimension is worth splitting.
const GEMM_PARALLEL_WORK: u64 = 4_000_000;

/// FMAs required to split when other matmuls are already running in
/// parallel (concurrent requests share this pool; splitting each one many
/// ways multiplies ready tasks and thrashes per-core L2 with B panels).
const GEMM_PARALLEL_WORK_BUSY: u64 = 16_000_000;

/// Smallest chunk a split is allowed to produce. Must keep enough rows to
/// amortise the panel layout.
const GEMM_MIN_ROWS: usize = 16;

/// Count of concurrent `m`-splits currently in flight across all threads.
/// When non-zero, subsequent matmuls use a much higher work threshold before
/// splitting so one multi-threaded encode does not cascade into thrashing
/// the rayon pool.
static ACTIVE_SPLITS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[inline]
fn gemm_should_split(m: usize, k: usize, n: usize) -> bool {
    if m < 2 * GEMM_MIN_ROWS {
        return false;
    }
    let work = (m as u64) * (k as u64) * (n as u64);
    let active = ACTIVE_SPLITS.load(std::sync::atomic::Ordering::Relaxed);
    let threshold = if active == 0 {
        GEMM_PARALLEL_WORK
    } else {
        GEMM_PARALLEL_WORK_BUSY
    };
    work >= threshold && rayon::current_num_threads() > 1
}

struct SplitGuard;
impl Drop for SplitGuard {
    fn drop(&mut self) {
        ACTIVE_SPLITS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

unsafe fn matmul_f32_parallel_recursive(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    if m < 2 * GEMM_MIN_ROWS {
        matmul_f32_dispatch()(a, b, m, k, n, bias, out, relu);
        return;
    }
    let m_mid = m / 2;
    let a_mid = m_mid * k;
    let o_mid = m_mid * n;
    let (a_top, a_bot) = a.split_at(a_mid);
    let (o_top, o_bot) = out.split_at_mut(o_mid);
    rayon::join(
        || matmul_f32_parallel_recursive(a_top, b, m_mid, k, n, bias, o_top, relu),
        || matmul_f32_parallel_recursive(a_bot, b, m - m_mid, k, n, bias, o_bot, relu),
    );
}

#[inline]
pub unsafe fn matmul_f32_split(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    if gemm_should_split(m, k, n) {
        ACTIVE_SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _guard = SplitGuard;
        matmul_f32_parallel_recursive(a, b, m, k, n, bias, out, relu);
    } else {
        matmul_f32_dispatch()(a, b, m, k, n, bias, out, relu);
    }
}

unsafe fn matmul_i8_parallel_recursive(
    x: &[i8],
    w: &[i8],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    if m < 2 * GEMM_MIN_ROWS {
        matmul_i8_dispatch()(x, w, m, k, n, bias, row_sums, out, denom, relu);
        return;
    }
    let m_mid = m / 2;
    let x_mid = m_mid * k;
    let o_mid = m_mid * n;
    let (x_top, x_bot) = x.split_at(x_mid);
    let (o_top, o_bot) = out.split_at_mut(o_mid);
    rayon::join(
        || matmul_i8_parallel_recursive(x_top, w, m_mid, k, n, bias, row_sums, o_top, denom, relu),
        || {
            matmul_i8_parallel_recursive(
                x_bot,
                w,
                m - m_mid,
                k,
                n,
                bias,
                row_sums,
                o_bot,
                denom,
                relu,
            )
        },
    );
}

#[inline]
pub unsafe fn matmul_i8_split(
    x: &[i8],
    w: &[i8],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    if gemm_should_split(m, k, n) {
        ACTIVE_SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _guard = SplitGuard;
        matmul_i8_parallel_recursive(x, w, m, k, n, bias, row_sums, out, denom, relu);
    } else {
        matmul_i8_dispatch()(x, w, m, k, n, bias, row_sums, out, denom, relu);
    }
}

// ---------------------------------------------------------------------------
// Single-row (m == 1) matmul
//
// The decoder path is all m == 1, and a one-row gemm has no B-panel reuse to
// amortise the panel layout across rows: each activation `a[t]` is loaded
// once and broadcast to every output column, so streaming weight rows
// `b[t * n + j]` linearly (the layout on disk) gives maximum memory-system
// throughput with zero setup overhead.
// ---------------------------------------------------------------------------

pub type F32Row = unsafe fn(&[f32], &[f32], usize, usize, Option<&[f32]>, &mut [f32], bool);

#[inline]
pub fn matmul_row_f32_dispatch() -> F32Row {
    use crate::utils::arch::SimdLevel;
    static FN: OnceLock<F32Row> = OnceLock::new();
    *FN.get_or_init(|| match crate::utils::arch::simd_level() {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2Fma => avx2::matmul_row_avx2,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx => avx2::matmul_row_avx,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse41 => sse41::matmul_row_sse41,
        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        SimdLevel::Neon => neon::matmul_row_neon,
        _ => scalar::matmul_row_scalar,
    })
}

pub type QuantizeFn = unsafe fn(&[f32], f32, &mut [i8]);

#[inline]
pub fn resolve_quantize_fn() -> QuantizeFn {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::utils::arch::has_avx2() {
            return avx2::quantize_row_avx2;
        }
        if is_x86_feature_detected!("sse4.1") {
            return sse41::quantize_row_sse41;
        }
        if is_x86_feature_detected!("sse2") {
            return sse41::quantize_row_sse2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return neon::quantize_row_neon;
    }
    #[allow(unreachable_code)]
    scalar::quantize_row_scalar
}

#[inline]
pub fn quantize_row(data: &[f32], qa: f32, out: &mut [i8]) {
    debug_assert!(
        out.len() >= data.len(),
        "quantize_row: out buffer length {} < data length {}",
        out.len(),
        data.len()
    );
    static DISPATCH: LazyLock<QuantizeFn> = LazyLock::new(resolve_quantize_fn);
    unsafe { DISPATCH(data, qa, out) };
}

/// Compute `128·Σw` for every row in `w[n, k]`.
///
/// Hoisted to model load time (see [`crate::engine::tensor::Quant8Tensor::row_sums`]) so the
/// kernels never rescan the weights at call time.
pub(crate) fn compute_row_sums(w: &[i8], n: usize, k: usize) -> std::borrow::Cow<'_, [i32]> {
    let rows: Vec<i32> = (0..n)
        .map(|j| {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc: i32 = 0;
            for &b in wr {
                acc += b as i32;
            }
            acc * 128
        })
        .collect();
    std::borrow::Cow::Owned(rows)
}
