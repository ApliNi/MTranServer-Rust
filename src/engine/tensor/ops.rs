//! High-level tensor operations: LayerNorm, activations, softmax, bias addition,
//! and attention auxiliary functions.

use crate::engine::tensor::Tensor2;
use std::sync::LazyLock;

// ===========================================================================
// LayerNorm
// ===========================================================================

#[inline]
pub fn layer_norm_mean_inv_scalar(slice: &[f32], eps: f32) -> (f32, f32) {
    let d = slice.len();
    let d_f = d as f32;
    let sum: f32 = slice.iter().sum();
    let mean = sum / d_f;
    let mut var_sum = 0.0f32;
    for &x in slice {
        let diff = x - mean;
        var_sum += diff * diff;
    }
    let inv = 1.0 / (var_sum / d_f + eps).sqrt();
    (mean, inv)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn layer_norm_mean_inv_avx2(slice: &[f32], eps: f32) -> (f32, f32) {
    use std::arch::x86_64::*;
    let d = slice.len();
    let ptr = slice.as_ptr();
    let chunks32 = d / 32;
    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    for i in 0..chunks32 {
        let p = ptr.add(i * 32);
        sum0 = _mm256_add_ps(sum0, _mm256_loadu_ps(p));
        sum1 = _mm256_add_ps(sum1, _mm256_loadu_ps(p.add(8)));
        sum2 = _mm256_add_ps(sum2, _mm256_loadu_ps(p.add(16)));
        sum3 = _mm256_add_ps(sum3, _mm256_loadu_ps(p.add(24)));
    }

    let mut sum_v = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
    let chunks8 = d / 8;
    for i in (chunks32 * 4)..chunks8 {
        sum_v = _mm256_add_ps(sum_v, _mm256_loadu_ps(ptr.add(i * 8)));
    }

    let sum128 = _mm_add_ps(
        _mm256_castps256_ps128(sum_v),
        _mm256_extractf128_ps(sum_v, 1),
    );
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut sum = _mm_cvtss_f32(sum32);

    let tail_start = chunks8 * 8;
    for i in tail_start..d {
        sum += *ptr.add(i);
    }

    let d_f = d as f32;
    let mean = sum / d_f;
    let mean_v = _mm256_set1_ps(mean);

    // Pass 2: compute sum((x - mean)^2)
    let mut var0 = _mm256_setzero_ps();
    let mut var1 = _mm256_setzero_ps();
    let mut var2 = _mm256_setzero_ps();
    let mut var3 = _mm256_setzero_ps();
    for i in 0..chunks32 {
        let p = ptr.add(i * 32);
        let diff0 = _mm256_sub_ps(_mm256_loadu_ps(p), mean_v);
        let diff1 = _mm256_sub_ps(_mm256_loadu_ps(p.add(8)), mean_v);
        let diff2 = _mm256_sub_ps(_mm256_loadu_ps(p.add(16)), mean_v);
        let diff3 = _mm256_sub_ps(_mm256_loadu_ps(p.add(24)), mean_v);
        var0 = _mm256_fmadd_ps(diff0, diff0, var0);
        var1 = _mm256_fmadd_ps(diff1, diff1, var1);
        var2 = _mm256_fmadd_ps(diff2, diff2, var2);
        var3 = _mm256_fmadd_ps(diff3, diff3, var3);
    }

    let mut var_v = _mm256_add_ps(_mm256_add_ps(var0, var1), _mm256_add_ps(var2, var3));
    for i in (chunks32 * 4)..chunks8 {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(ptr.add(i * 8)), mean_v);
        var_v = _mm256_fmadd_ps(diff, diff, var_v);
    }

    let var128 = _mm_add_ps(
        _mm256_castps256_ps128(var_v),
        _mm256_extractf128_ps(var_v, 1),
    );
    let var64 = _mm_add_ps(var128, _mm_movehl_ps(var128, var128));
    let var32 = _mm_add_ss(var64, _mm_shuffle_ps(var64, var64, 1));
    let mut var_sum = _mm_cvtss_f32(var32);

    for i in tail_start..d {
        let diff = *ptr.add(i) - mean;
        var_sum += diff * diff;
    }

    let inv = 1.0 / (var_sum / d_f + eps).sqrt();
    (mean, inv)
}

// ---------------------------------------------------------------------------
// Layer-norm store step
// ---------------------------------------------------------------------------
pub type LayerNormStore = unsafe fn(&[f32], &[f32], &[f32], f32, f32, &mut [f32]);

#[inline]
pub fn layer_norm_store_dispatch() -> LayerNormStore {
    use crate::utils::arch::SimdLevel;
    static FN: LazyLock<LayerNormStore> =
        LazyLock::new(|| match crate::utils::arch::simd_level() {
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx2Fma => layer_norm_store_avx,
            #[cfg(target_arch = "x86_64")]
            SimdLevel::Avx | SimdLevel::Sse41 => layer_norm_store_sse41,
            #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
            SimdLevel::Neon => layer_norm_store_neon,
            _ => layer_norm_store_scalar,
        });
    *FN
}

/// Portable scalar reference kernel for layer-norm store.
pub fn layer_norm_store_scalar(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: f32,
    inv: f32,
    out: &mut [f32],
) {
    for (i, &v) in x.iter().enumerate() {
        out[i] = (v - mean) * inv * scale[i] + bias[i];
    }
}

/// SSE4.1 layer-norm store: the scalar expression tree on 4-wide lanes, with
/// a scalar tail for `d % 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn layer_norm_store_sse41(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: f32,
    inv: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::{
        _mm_add_ps, _mm_loadu_ps, _mm_mul_ps, _mm_set1_ps, _mm_storeu_ps, _mm_sub_ps,
    };
    const W: usize = 4;
    let mean_v = _mm_set1_ps(mean);
    let inv_v = _mm_set1_ps(inv);
    let groups = x.len() / W;
    for g in 0..groups {
        let o = g * W;
        let v = _mm_sub_ps(_mm_loadu_ps(x[o..].as_ptr()), mean_v);
        let v = _mm_mul_ps(v, inv_v);
        let v = _mm_mul_ps(v, _mm_loadu_ps(scale[o..].as_ptr()));
        let v = _mm_add_ps(v, _mm_loadu_ps(bias[o..].as_ptr()));
        _mm_storeu_ps(out[o..].as_mut_ptr(), v);
    }
    for i in groups * W..x.len() {
        out[i] = (x[i] - mean) * inv * scale[i] + bias[i];
    }
}

/// AVX layer-norm store: the scalar expression tree on 8-wide lanes, with a
/// scalar tail for `d % 8`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,fma")]
pub unsafe fn layer_norm_store_avx(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: f32,
    inv: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::{
        _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_mul_ps, _mm256_set1_ps, _mm256_storeu_ps,
        _mm256_sub_ps,
    };
    const W: usize = 32;
    let mean_v = _mm256_set1_ps(mean);
    let inv_v = _mm256_set1_ps(inv);
    let groups = x.len() / W;
    let x_ptr = x.as_ptr();
    let s_ptr = scale.as_ptr();
    let b_ptr = bias.as_ptr();
    let o_ptr = out.as_mut_ptr();

    for g in 0..groups {
        let o = g * W;
        let v0 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(o)), mean_v), inv_v);
        let v1 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(o + 8)), mean_v), inv_v);
        let v2 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(o + 16)), mean_v), inv_v);
        let v3 = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(o + 24)), mean_v), inv_v);

        let out0 = _mm256_fmadd_ps(v0, _mm256_loadu_ps(s_ptr.add(o)), _mm256_loadu_ps(b_ptr.add(o)));
        let out1 = _mm256_fmadd_ps(v1, _mm256_loadu_ps(s_ptr.add(o + 8)), _mm256_loadu_ps(b_ptr.add(o + 8)));
        let out2 = _mm256_fmadd_ps(v2, _mm256_loadu_ps(s_ptr.add(o + 16)), _mm256_loadu_ps(b_ptr.add(o + 16)));
        let out3 = _mm256_fmadd_ps(v3, _mm256_loadu_ps(s_ptr.add(o + 24)), _mm256_loadu_ps(b_ptr.add(o + 24)));

        _mm256_storeu_ps(o_ptr.add(o), out0);
        _mm256_storeu_ps(o_ptr.add(o + 8), out1);
        _mm256_storeu_ps(o_ptr.add(o + 16), out2);
        _mm256_storeu_ps(o_ptr.add(o + 24), out3);
    }

    let mut tail_start = groups * W;
    while x.len() - tail_start >= 8 {
        let o = tail_start;
        let v = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(o)), mean_v), inv_v);
        let out_v = _mm256_fmadd_ps(v, _mm256_loadu_ps(s_ptr.add(o)), _mm256_loadu_ps(b_ptr.add(o)));
        _mm256_storeu_ps(o_ptr.add(o), out_v);
        tail_start += 8;
    }
    for i in tail_start..x.len() {
        *o_ptr.add(i) = (*x_ptr.add(i) - mean) * inv * *s_ptr.add(i) + *b_ptr.add(i);
    }
}

/// ARM NEON layer-norm store: the scalar expression tree on 4-wide lanes,
/// with a scalar tail for `d % 4`.
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[target_feature(enable = "neon")]
pub unsafe fn layer_norm_store_neon(
    x: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: f32,
    inv: f32,
    out: &mut [f32],
) {
    use std::arch::aarch64::{vaddq_f32, vdupq_n_f32, vld1q_f32, vmulq_f32, vst1q_f32, vsubq_f32};
    const W: usize = 4;
    let mean_v = vdupq_n_f32(mean);
    let inv_v = vdupq_n_f32(inv);
    let groups = x.len() / W;
    for g in 0..groups {
        let o = g * W;
        let v = vsubq_f32(vld1q_f32(x[o..].as_ptr()), mean_v);
        let v = vmulq_f32(v, inv_v);
        let v = vaddq_f32(
            vmulq_f32(v, vld1q_f32(scale[o..].as_ptr())),
            vld1q_f32(bias[o..].as_ptr()),
        );
        vst1q_f32(out[o..].as_mut_ptr(), v);
    }
    for i in groups * W..x.len() {
        out[i] = (x[i] - mean) * inv * scale[i] + bias[i];
    }
}

#[inline]
pub fn layer_norm(x: &Tensor2, scale: &Tensor2, bias: &Tensor2, eps: f32) -> Tensor2 {
    let mut out = Tensor2 {
        shape: [0, 0],
        data: Vec::new(),
    };
    layer_norm_into(x, scale, bias, eps, &mut out);
    out
}

#[inline]
pub fn layer_norm_into(x: &Tensor2, scale: &Tensor2, bias: &Tensor2, eps: f32, out: &mut Tensor2) {
    let d = x.shape[1];
    assert!(d > 0, "layer_norm: last dim must be > 0");
    debug_assert_eq!(
        scale.data.len(),
        d,
        "layer_norm: scale length {} != last dim {}",
        scale.data.len(),
        d
    );
    debug_assert_eq!(
        bias.data.len(),
        d,
        "layer_norm: bias length {} != last dim {}",
        bias.data.len(),
        d
    );
    let rows = x.data.len() / d;
    unsafe {
        out.prepare_uninit(x.data.len());
    }
    let store = layer_norm_store_dispatch();
    let use_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::utils::arch::has_avx2()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    for r in 0..rows {
        let start = r * d;
        let slice = &x.data[start..start + d];
        let (mean, inv) = {
            #[cfg(target_arch = "x86_64")]
            if use_avx2 {
                unsafe { layer_norm_mean_inv_avx2(slice, eps) }
            } else {
                layer_norm_mean_inv_scalar(slice, eps)
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                layer_norm_mean_inv_scalar(slice, eps)
            }
        };
        unsafe {
            store(
                slice,
                &scale.data,
                &bias.data,
                mean,
                inv,
                &mut out.data[start..start + d],
            );
        }
    }
    out.shape = x.shape;
}
/// Fused add + layer norm AVX2+FMA implementation on raw pointers.
///
/// 2-Pass Zero-STLF (Store-To-Load Forwarding) architecture:
/// - Pass 1: Streaming loads `a_ptr` and `b_ptr`, computes `a + b` in AVX2 registers,
///   unpacks with `_mm256_cvtps_pd` to double-precision `f64` accumulators for `sum` and `sum_sq`
///   (guaranteeing numerical stability without cancellation). Out pointer is never written to.
///   Computes exact `mean` and `inv = 1.0 / sqrt(var + eps)`.
/// - Pass 2: Re-reads `a_ptr`, `b_ptr`, `scale`, and `bias` from hot L1 cache, computes
///   `(a + b - mean) * inv * scale + bias` in SIMD registers and writes once to `out_ptr`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_add_layer_norm_avx2_raw(
    a_ptr: *const f32,
    b_ptr: *const f32,
    s_ptr: *const f32,
    bias_ptr: *const f32,
    eps: f32,
    out_ptr: *mut f32,
    d: usize,
) {
    use std::arch::x86_64::*;

    let chunks16 = d / 16;
    let mut sum0 = _mm256_setzero_pd();
    let mut sum1 = _mm256_setzero_pd();
    let mut sum_sq0 = _mm256_setzero_pd();
    let mut sum_sq1 = _mm256_setzero_pd();

    for i in 0..chunks16 {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a_ptr.add(o));
        let b0 = _mm256_loadu_ps(b_ptr.add(o));
        let a1 = _mm256_loadu_ps(a_ptr.add(o + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(o + 8));

        let x0 = _mm256_add_ps(a0, b0);
        let x1 = _mm256_add_ps(a1, b1);

        let d0_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x0));
        let d0_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x0, 1));
        let d1_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x1));
        let d1_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x1, 1));

        sum0 = _mm256_add_pd(sum0, d0_lo);
        sum1 = _mm256_add_pd(sum1, d0_hi);
        sum_sq0 = _mm256_fmadd_pd(d0_lo, d0_lo, sum_sq0);
        sum_sq1 = _mm256_fmadd_pd(d0_hi, d0_hi, sum_sq1);

        sum0 = _mm256_add_pd(sum0, d1_lo);
        sum1 = _mm256_add_pd(sum1, d1_hi);
        sum_sq0 = _mm256_fmadd_pd(d1_lo, d1_lo, sum_sq0);
        sum_sq1 = _mm256_fmadd_pd(d1_hi, d1_hi, sum_sq1);
    }

    let tail16_start = chunks16 * 16;
    let mut tail_start = tail16_start;
    if d - tail_start >= 8 {
        let o = tail_start;
        let a = _mm256_loadu_ps(a_ptr.add(o));
        let b = _mm256_loadu_ps(b_ptr.add(o));
        let x = _mm256_add_ps(a, b);
        let d_lo = _mm256_cvtps_pd(_mm256_castps256_ps128(x));
        let d_hi = _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1));

        sum0 = _mm256_add_pd(sum0, d_lo);
        sum1 = _mm256_add_pd(sum1, d_hi);
        sum_sq0 = _mm256_fmadd_pd(d_lo, d_lo, sum_sq0);
        sum_sq1 = _mm256_fmadd_pd(d_hi, d_hi, sum_sq1);
        tail_start += 8;
    }

    let sum_v = _mm256_add_pd(sum0, sum1);
    let sum_sq_v = _mm256_add_pd(sum_sq0, sum_sq1);

    let sum128 = _mm_add_pd(
        _mm256_castpd256_pd128(sum_v),
        _mm256_extractf128_pd(sum_v, 1),
    );
    let sum_sq128 = _mm_add_pd(
        _mm256_castpd256_pd128(sum_sq_v),
        _mm256_extractf128_pd(sum_sq_v, 1),
    );

    let sum_high = _mm_unpackhi_pd(sum128, sum128);
    let sum_sq_high = _mm_unpackhi_pd(sum_sq128, sum_sq128);

    let mut sum_scalar = _mm_cvtsd_f64(_mm_add_sd(sum128, sum_high));
    let mut sum_sq_scalar = _mm_cvtsd_f64(_mm_add_sd(sum_sq128, sum_sq_high));

    for i in tail_start..d {
        let x = (*a_ptr.add(i) + *b_ptr.add(i)) as f64;
        sum_scalar += x;
        sum_sq_scalar += x * x;
    }

    let d_f64 = d as f64;
    let mean_f64 = sum_scalar / d_f64;
    let var_f64 = (sum_sq_scalar / d_f64) - mean_f64 * mean_f64;
    let var_f64 = if var_f64 > 0.0 { var_f64 } else { 0.0 };
    let inv_f64 = 1.0 / (var_f64 + eps as f64).sqrt();

    let mean = mean_f64 as f32;
    let inv = inv_f64 as f32;
    let mean_v = _mm256_set1_ps(mean);
    let inv_v = _mm256_set1_ps(inv);

    // Pass 2: compute (a + b - mean) * inv * scale + bias and write once to out_ptr
    for i in 0..chunks16 {
        let o = i * 16;
        let a0 = _mm256_loadu_ps(a_ptr.add(o));
        let b0 = _mm256_loadu_ps(b_ptr.add(o));
        let a1 = _mm256_loadu_ps(a_ptr.add(o + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(o + 8));

        let x0 = _mm256_add_ps(a0, b0);
        let x1 = _mm256_add_ps(a1, b1);

        let diff0 = _mm256_sub_ps(x0, mean_v);
        let diff1 = _mm256_sub_ps(x1, mean_v);

        let norm0 = _mm256_mul_ps(diff0, inv_v);
        let norm1 = _mm256_mul_ps(diff1, inv_v);

        let s0 = _mm256_loadu_ps(s_ptr.add(o));
        let s1 = _mm256_loadu_ps(s_ptr.add(o + 8));

        let bias0 = _mm256_loadu_ps(bias_ptr.add(o));
        let bias1 = _mm256_loadu_ps(bias_ptr.add(o + 8));

        let res0 = _mm256_fmadd_ps(norm0, s0, bias0);
        let res1 = _mm256_fmadd_ps(norm1, s1, bias1);

        _mm256_storeu_ps(out_ptr.add(o), res0);
        _mm256_storeu_ps(out_ptr.add(o + 8), res1);
    }

    let mut tail_start2 = tail16_start;
    if d - tail_start2 >= 8 {
        let o = tail_start2;
        let a = _mm256_loadu_ps(a_ptr.add(o));
        let b = _mm256_loadu_ps(b_ptr.add(o));
        let x = _mm256_add_ps(a, b);
        let diff = _mm256_sub_ps(x, mean_v);
        let norm = _mm256_mul_ps(diff, inv_v);
        let s = _mm256_loadu_ps(s_ptr.add(o));
        let bias = _mm256_loadu_ps(bias_ptr.add(o));
        let res = _mm256_fmadd_ps(norm, s, bias);
        _mm256_storeu_ps(out_ptr.add(o), res);
        tail_start2 += 8;
    }

    for i in tail_start2..d {
        let x = *a_ptr.add(i) + *b_ptr.add(i);
        *out_ptr.add(i) = (x - mean) * inv * *s_ptr.add(i) + *bias_ptr.add(i);
    }
}

/// Fused add + layer norm AVX2+FMA kernel for separate slices.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn fused_add_layer_norm_avx2(
    a: &[f32],
    b: &[f32],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let d = a.len();
    debug_assert_eq!(b.len(), d);
    debug_assert_eq!(scale.len(), d);
    debug_assert_eq!(bias.len(), d);
    debug_assert_eq!(out.len(), d);
    fused_add_layer_norm_avx2_raw(
        a.as_ptr(),
        b.as_ptr(),
        scale.as_ptr(),
        bias.as_ptr(),
        eps,
        out.as_mut_ptr(),
        d,
    );
}

/// Raw pointer scalar implementation for fused add + layer norm.
pub unsafe fn fused_add_layer_norm_scalar_raw(
    a_ptr: *const f32,
    b_ptr: *const f32,
    scale_ptr: *const f32,
    bias_ptr: *const f32,
    eps: f32,
    out_ptr: *mut f32,
    d: usize,
) {
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for i in 0..d {
        let x = (*a_ptr.add(i) + *b_ptr.add(i)) as f64;
        sum += x;
        sum_sq += x * x;
    }
    let d_f64 = d as f64;
    let mean_f64 = sum / d_f64;
    let var_f64 = (sum_sq / d_f64) - mean_f64 * mean_f64;
    let var_f64 = if var_f64 > 0.0 { var_f64 } else { 0.0 };
    let inv_f64 = 1.0 / (var_f64 + eps as f64).sqrt();

    let mean = mean_f64 as f32;
    let inv = inv_f64 as f32;

    for i in 0..d {
        let x = *a_ptr.add(i) + *b_ptr.add(i);
        let norm = (x - mean) * inv;
        *out_ptr.add(i) = norm.mul_add(*scale_ptr.add(i), *bias_ptr.add(i));
    }
}

/// Scalar reference and fallback for fused add + layer norm.
pub fn fused_add_layer_norm_scalar(
    a: &[f32],
    b: &[f32],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let d = a.len();
    debug_assert_eq!(b.len(), d, "fused_add_layer_norm: b len != a len");
    debug_assert_eq!(scale.len(), d, "fused_add_layer_norm: scale len != a len");
    debug_assert_eq!(bias.len(), d, "fused_add_layer_norm: bias len != a len");
    debug_assert_eq!(out.len(), d, "fused_add_layer_norm: out len != a len");

    unsafe {
        fused_add_layer_norm_scalar_raw(
            a.as_ptr(),
            b.as_ptr(),
            scale.as_ptr(),
            bias.as_ptr(),
            eps,
            out.as_mut_ptr(),
            d,
        );
    }
}

/// In-place scalar reference for fused add + layer norm.
pub fn fused_add_layer_norm_in_place_scalar(
    a_and_out: &mut [f32],
    b: &[f32],
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) {
    let d = a_and_out.len();
    debug_assert_eq!(b.len(), d, "fused_add_layer_norm: b len != a_and_out len");
    debug_assert_eq!(
        scale.len(),
        d,
        "fused_add_layer_norm: scale len != a_and_out len"
    );
    debug_assert_eq!(
        bias.len(),
        d,
        "fused_add_layer_norm: bias len != a_and_out len"
    );

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    for i in 0..d {
        let x = (a_and_out[i] + b[i]) as f64;
        sum += x;
        sum_sq += x * x;
    }
    let d_f64 = d as f64;
    let mean_f64 = sum / d_f64;
    let var_f64 = (sum_sq / d_f64) - mean_f64 * mean_f64;
    let var_f64 = if var_f64 > 0.0 { var_f64 } else { 0.0 };
    let inv_f64 = 1.0 / (var_f64 + eps as f64).sqrt();

    let mean = mean_f64 as f32;
    let inv = inv_f64 as f32;

    for i in 0..d {
        let x = a_and_out[i] + b[i];
        let norm = (x - mean) * inv;
        a_and_out[i] = norm.mul_add(scale[i], bias[i]);
    }
}

/// Computes `out = LayerNorm(a + b)` into a separate tensor `out`.
pub fn fused_add_layer_norm_into(
    a: &Tensor2,
    b: &Tensor2,
    scale: &Tensor2,
    bias: &Tensor2,
    eps: f32,
    out: &mut Tensor2,
) {
    let d = a.shape[1];
    assert!(d > 0, "fused_add_layer_norm: last dim must be > 0");
    debug_assert_eq!(
        scale.data.len(),
        d,
        "fused_add_layer_norm: scale length != d"
    );
    debug_assert_eq!(bias.data.len(), d, "fused_add_layer_norm: bias length != d");
    let is_b_broadcast =
        (b.shape[0] == 1 || b.shape[0] == 0) && b.shape[1] == d && b.data.len() == d;
    debug_assert!(
        a.data.len() == b.data.len() || is_b_broadcast,
        "fused_add_layer_norm: shapes must match or b be [last_dim], got {:?} vs {:?}",
        a.shape,
        b.shape
    );

    let rows = a.data.len() / d;
    if out.data.len() != a.data.len() {
        unsafe {
            out.prepare_uninit(a.data.len());
        }
    }
    out.shape = a.shape;

    let use_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::utils::arch::has_avx2()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    let a_ptr = a.data.as_ptr();
    let b_ptr = b.data.as_ptr();
    let s_ptr = scale.data.as_ptr();
    let bias_ptr = bias.data.as_ptr();
    let out_ptr = out.data.as_mut_ptr();

    for r in 0..rows {
        let start = r * d;
        let ar_ptr = unsafe { a_ptr.add(start) };
        let br_ptr = unsafe {
            if is_b_broadcast {
                b_ptr
            } else {
                b_ptr.add(start)
            }
        };
        let outr_ptr = unsafe { out_ptr.add(start) };

        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe {
                fused_add_layer_norm_avx2_raw(ar_ptr, br_ptr, s_ptr, bias_ptr, eps, outr_ptr, d);
            }
            continue;
        }

        unsafe {
            fused_add_layer_norm_scalar_raw(ar_ptr, br_ptr, s_ptr, bias_ptr, eps, outr_ptr, d);
        }
    }
}

/// Computes `a_and_out = LayerNorm(a_and_out + b)` in-place.
pub fn fused_add_layer_norm_in_place(
    a_and_out: &mut Tensor2,
    b: &Tensor2,
    scale: &Tensor2,
    bias: &Tensor2,
    eps: f32,
) {
    let d = a_and_out.shape[1];
    assert!(d > 0, "fused_add_layer_norm: last dim must be > 0");
    debug_assert_eq!(
        scale.data.len(),
        d,
        "fused_add_layer_norm: scale length != d"
    );
    debug_assert_eq!(bias.data.len(), d, "fused_add_layer_norm: bias length != d");
    let is_b_broadcast =
        (b.shape[0] == 1 || b.shape[0] == 0) && b.shape[1] == d && b.data.len() == d;
    debug_assert!(
        a_and_out.data.len() == b.data.len() || is_b_broadcast,
        "fused_add_layer_norm: shapes must match or b be [last_dim], got {:?} vs {:?}",
        a_and_out.shape,
        b.shape
    );

    let rows = a_and_out.data.len() / d;
    let use_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::utils::arch::has_avx2()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    for r in 0..rows {
        let start = r * d;
        let b_slice = if is_b_broadcast {
            &b.data[..d]
        } else {
            &b.data[start..start + d]
        };
        let out_slice = &mut a_and_out.data[start..start + d];

        #[cfg(target_arch = "x86_64")]
        if use_avx2 {
            unsafe {
                fused_add_layer_norm_avx2_raw(
                    out_slice.as_ptr(),
                    b_slice.as_ptr(),
                    scale.data.as_ptr(),
                    bias.data.as_ptr(),
                    eps,
                    out_slice.as_mut_ptr(),
                    d,
                );
            }
            continue;
        }

        fused_add_layer_norm_in_place_scalar(out_slice, b_slice, &scale.data, &bias.data, eps);
    }
}

// ===========================================================================
// Bias addition & elementwise add
// ===========================================================================

#[inline]
pub fn add(a: &Tensor2, b: &Tensor2) -> Tensor2 {
    let mut out = Tensor2 {
        shape: [0, 0],
        data: Vec::new(),
    };
    add_into(a, b, &mut out);
    out
}

#[inline]
pub fn add_into(a: &Tensor2, b: &Tensor2, out: &mut Tensor2) {
    let d = a.shape[1];
    if (b.shape[0] == 1 || b.shape[0] == 0) && b.shape[1] == d && b.data.len() == d {
        if out.data.len() != a.data.len() {
            unsafe {
                out.prepare_uninit(a.data.len());
            }
        }
        for (row_o, row) in out.data.chunks_exact_mut(d).zip(a.data.chunks_exact(d)) {
            for ((o, &x_val), &y_val) in row_o.iter_mut().zip(row.iter()).zip(b.data.iter()) {
                *o = x_val + y_val;
            }
        }
    } else {
        debug_assert_eq!(
            a.shape, b.shape,
            "add: shapes must match or other be [last_dim], got {:?} vs {:?}",
            a.shape, b.shape
        );
        if out.data.len() != a.data.len() {
            unsafe {
                out.prepare_uninit(a.data.len());
            }
        }
        for ((o, &x_val), &y_val) in out.data.iter_mut().zip(a.data.iter()).zip(b.data.iter()) {
            *o = x_val + y_val;
        }
    }
    out.shape = a.shape;
}

#[inline]
pub fn add_bias_into(out: &mut [f32], bias: &[f32], rows: usize, d: usize) {
    debug_assert_eq!(out.len(), rows * d);
    debug_assert_eq!(bias.len(), d);
    for row in out.chunks_exact_mut(d) {
        for (o, &b) in row.iter_mut().zip(bias.iter()) {
            *o += b;
        }
    }
}

// ===========================================================================
// Softmax
// ===========================================================================

/// Vectorized single-precision exponential calculation for 8 floats using
/// Cody-Waite range reduction and degree-6 Taylor polynomial evaluation.
///
/// # Safety
/// Caller must ensure the host CPU supports AVX2 and FMA feature sets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn exp256_ps(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let min_x = _mm256_set1_ps(-87.336544);
    let max_x = _mm256_set1_ps(88.376262);

    // Any input strictly below min_x underflows to 0.0 in f32
    let underflow_mask = _mm256_cmp_ps(x, min_x, _CMP_GE_OQ);
    let xc = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let k = _mm256_round_ps(
        _mm256_mul_ps(xc, log2e),
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
    );

    // Cody-Waite range reduction: r = xc - k * C1 - k * C2
    let c1 = _mm256_set1_ps(0.693145751953125);
    let c2 = _mm256_set1_ps(1.4286068203094172e-6);
    let r = _mm256_fnmadd_ps(k, c1, xc);
    let r = _mm256_fnmadd_ps(k, c2, r);

    // Degree 6 Taylor polynomial with Horner's scheme
    let p6 = _mm256_set1_ps(1.0 / 720.0);
    let p5 = _mm256_set1_ps(1.0 / 120.0);
    let p4 = _mm256_set1_ps(1.0 / 24.0);
    let p3 = _mm256_set1_ps(1.0 / 6.0);
    let p2 = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);

    let mut poly = _mm256_fmadd_ps(p6, r, p5);
    poly = _mm256_fmadd_ps(poly, r, p4);
    poly = _mm256_fmadd_ps(poly, r, p3);
    poly = _mm256_fmadd_ps(poly, r, p2);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    // Scale by 2^k
    let k_i = _mm256_cvtps_epi32(k);
    let bias_int = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, bias_int), 23);
    let pow2 = _mm256_castsi256_ps(exp_bits);

    let res = _mm256_mul_ps(pow2, poly);
    _mm256_and_ps(res, underflow_mask)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn exp256_ps_4x(
    x0: std::arch::x86_64::__m256,
    x1: std::arch::x86_64::__m256,
    x2: std::arch::x86_64::__m256,
    x3: std::arch::x86_64::__m256,
    min_x: std::arch::x86_64::__m256,
    log2e: std::arch::x86_64::__m256,
    c1: std::arch::x86_64::__m256,
    c2: std::arch::x86_64::__m256,
    c_p6: std::arch::x86_64::__m256,
    c_p5: std::arch::x86_64::__m256,
    c_p4: std::arch::x86_64::__m256,
    c_p3: std::arch::x86_64::__m256,
    c_p2: std::arch::x86_64::__m256,
    c_one: std::arch::x86_64::__m256,
    bias_int: std::arch::x86_64::__m256i,
) -> (
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
    std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;

    let mask0 = _mm256_cmp_ps(x0, min_x, _CMP_GE_OQ);
    let mask1 = _mm256_cmp_ps(x1, min_x, _CMP_GE_OQ);
    let mask2 = _mm256_cmp_ps(x2, min_x, _CMP_GE_OQ);
    let mask3 = _mm256_cmp_ps(x3, min_x, _CMP_GE_OQ);

    let xc0 = _mm256_max_ps(x0, min_x);
    let xc1 = _mm256_max_ps(x1, min_x);
    let xc2 = _mm256_max_ps(x2, min_x);
    let xc3 = _mm256_max_ps(x3, min_x);

    let k0 = _mm256_round_ps(_mm256_mul_ps(xc0, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let k1 = _mm256_round_ps(_mm256_mul_ps(xc1, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let k2 = _mm256_round_ps(_mm256_mul_ps(xc2, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let k3 = _mm256_round_ps(_mm256_mul_ps(xc3, log2e), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    let r0 = _mm256_fnmadd_ps(k0, c1, xc0);
    let r1 = _mm256_fnmadd_ps(k1, c1, xc1);
    let r2 = _mm256_fnmadd_ps(k2, c1, xc2);
    let r3 = _mm256_fnmadd_ps(k3, c1, xc3);

    let r0 = _mm256_fnmadd_ps(k0, c2, r0);
    let r1 = _mm256_fnmadd_ps(k1, c2, r1);
    let r2 = _mm256_fnmadd_ps(k2, c2, r2);
    let r3 = _mm256_fnmadd_ps(k3, c2, r3);

    let mut poly0 = _mm256_fmadd_ps(c_p6, r0, c_p5);
    let mut poly1 = _mm256_fmadd_ps(c_p6, r1, c_p5);
    let mut poly2 = _mm256_fmadd_ps(c_p6, r2, c_p5);
    let mut poly3 = _mm256_fmadd_ps(c_p6, r3, c_p5);

    poly0 = _mm256_fmadd_ps(poly0, r0, c_p4);
    poly1 = _mm256_fmadd_ps(poly1, r1, c_p4);
    poly2 = _mm256_fmadd_ps(poly2, r2, c_p4);
    poly3 = _mm256_fmadd_ps(poly3, r3, c_p4);

    poly0 = _mm256_fmadd_ps(poly0, r0, c_p3);
    poly1 = _mm256_fmadd_ps(poly1, r1, c_p3);
    poly2 = _mm256_fmadd_ps(poly2, r2, c_p3);
    poly3 = _mm256_fmadd_ps(poly3, r3, c_p3);

    poly0 = _mm256_fmadd_ps(poly0, r0, c_p2);
    poly1 = _mm256_fmadd_ps(poly1, r1, c_p2);
    poly2 = _mm256_fmadd_ps(poly2, r2, c_p2);
    poly3 = _mm256_fmadd_ps(poly3, r3, c_p2);

    poly0 = _mm256_fmadd_ps(poly0, r0, c_one);
    poly1 = _mm256_fmadd_ps(poly1, r1, c_one);
    poly2 = _mm256_fmadd_ps(poly2, r2, c_one);
    poly3 = _mm256_fmadd_ps(poly3, r3, c_one);

    poly0 = _mm256_fmadd_ps(poly0, r0, c_one);
    poly1 = _mm256_fmadd_ps(poly1, r1, c_one);
    poly2 = _mm256_fmadd_ps(poly2, r2, c_one);
    poly3 = _mm256_fmadd_ps(poly3, r3, c_one);

    let k_i0 = _mm256_cvtps_epi32(k0);
    let k_i1 = _mm256_cvtps_epi32(k1);
    let k_i2 = _mm256_cvtps_epi32(k2);
    let k_i3 = _mm256_cvtps_epi32(k3);

    let pow2_0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_i0, bias_int), 23));
    let pow2_1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_i1, bias_int), 23));
    let pow2_2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_i2, bias_int), 23));
    let pow2_3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(k_i3, bias_int), 23));

    (
        _mm256_and_ps(_mm256_mul_ps(pow2_0, poly0), mask0),
        _mm256_and_ps(_mm256_mul_ps(pow2_1, poly1), mask1),
        _mm256_and_ps(_mm256_mul_ps(pow2_2, poly2), mask2),
        _mm256_and_ps(_mm256_mul_ps(pow2_3, poly3), mask3),
    )
}

/// AVX2-accelerated in-place softmax over query-key attention scores.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_scores_inplace_avx2(scores: &mut [f32], q_len: usize, kv_len: usize) {
    use std::arch::x86_64::*;

    let d = kv_len;
    if d < 8 {
        softmax_scores_inplace_scalar(scores, q_len, kv_len);
        return;
    }

    let chunks32 = d / 32;
    let chunks8 = d / 8;
    let tail_start = chunks8 * 8;

    let min_x = _mm256_set1_ps(-87.336544);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693145751953125);
    let c2 = _mm256_set1_ps(1.4286068203094172e-6);
    let c_p6 = _mm256_set1_ps(1.0 / 720.0);
    let c_p5 = _mm256_set1_ps(1.0 / 120.0);
    let c_p4 = _mm256_set1_ps(1.0 / 24.0);
    let c_p3 = _mm256_set1_ps(1.0 / 6.0);
    let c_p2 = _mm256_set1_ps(0.5);
    let c_one = _mm256_set1_ps(1.0);
    let bias_int = _mm256_set1_epi32(127);

    for row in scores[..q_len * d].chunks_exact_mut(d) {
        let ptr = row.as_mut_ptr();

        // Pass 1: compute row maximum
        let mut max0 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max1 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max2 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max3 = _mm256_set1_ps(f32::NEG_INFINITY);

        for i in 0..chunks32 {
            let p = ptr.add(i * 32);
            max0 = _mm256_max_ps(max0, _mm256_loadu_ps(p));
            max1 = _mm256_max_ps(max1, _mm256_loadu_ps(p.add(8)));
            max2 = _mm256_max_ps(max2, _mm256_loadu_ps(p.add(16)));
            max3 = _mm256_max_ps(max3, _mm256_loadu_ps(p.add(24)));
        }

        let mut max_v = _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));
        for i in (chunks32 * 4)..chunks8 {
            max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(ptr.add(i * 8)));
        }

        let max128 = _mm_max_ps(
            _mm256_castps256_ps128(max_v),
            _mm256_extractf128_ps(max_v, 1),
        );
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
        let mut max_val = _mm_cvtss_f32(max32);

        for i in tail_start..d {
            let v = *ptr.add(i);
            if v > max_val {
                max_val = v;
            }
        }

        // Pass 2: compute exp(x - max), accumulate sum, write back exp
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        for i in 0..chunks32 {
            let p0 = ptr.add(i * 32);
            let p1 = p0.add(8);
            let p2 = p0.add(16);
            let p3 = p0.add(24);

            let x0 = _mm256_sub_ps(_mm256_loadu_ps(p0), max_broadcast);
            let x1 = _mm256_sub_ps(_mm256_loadu_ps(p1), max_broadcast);
            let x2 = _mm256_sub_ps(_mm256_loadu_ps(p2), max_broadcast);
            let x3 = _mm256_sub_ps(_mm256_loadu_ps(p3), max_broadcast);

            let (e0, e1, e2, e3) = exp256_ps_4x(
                x0, x1, x2, x3, min_x, log2e, c1, c2, c_p6, c_p5, c_p4, c_p3, c_p2, c_one, bias_int,
            );
            sum0 = _mm256_add_ps(sum0, e0);
            sum1 = _mm256_add_ps(sum1, e1);
            sum2 = _mm256_add_ps(sum2, e2);
            sum3 = _mm256_add_ps(sum3, e3);

            _mm256_storeu_ps(p0, e0);
            _mm256_storeu_ps(p1, e1);
            _mm256_storeu_ps(p2, e2);
            _mm256_storeu_ps(p3, e3);
        }

        let mut sum_v = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        for i in (chunks32 * 4)..chunks8 {
            let p = ptr.add(i * 8);
            let diff = _mm256_sub_ps(_mm256_loadu_ps(p), max_broadcast);
            let e = exp256_ps(diff);
            sum_v = _mm256_add_ps(sum_v, e);
            _mm256_storeu_ps(p, e);
        }

        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(sum_v),
            _mm256_extractf128_ps(sum_v, 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut sum = _mm_cvtss_f32(sum32);

        for i in tail_start..d {
            let p = ptr.add(i);
            let e = (*p - max_val).exp();
            *p = e;
            sum += e;
        }

        // Pass 3: normalize by multiplying inverse sum
        let inv = 1.0 / sum;
        let inv_broadcast = _mm256_set1_ps(inv);

        for i in 0..chunks32 {
            let p0 = ptr.add(i * 32);
            let p1 = p0.add(8);
            let p2 = p0.add(16);
            let p3 = p0.add(24);

            let v0 = _mm256_mul_ps(_mm256_loadu_ps(p0), inv_broadcast);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(p1), inv_broadcast);
            let v2 = _mm256_mul_ps(_mm256_loadu_ps(p2), inv_broadcast);
            let v3 = _mm256_mul_ps(_mm256_loadu_ps(p3), inv_broadcast);

            _mm256_storeu_ps(p0, v0);
            _mm256_storeu_ps(p1, v1);
            _mm256_storeu_ps(p2, v2);
            _mm256_storeu_ps(p3, v3);
        }

        for i in (chunks32 * 4)..chunks8 {
            let p = ptr.add(i * 8);
            let v = _mm256_mul_ps(_mm256_loadu_ps(p), inv_broadcast);
            _mm256_storeu_ps(p, v);
        }

        for i in tail_start..d {
            *ptr.add(i) *= inv;
        }
    }
}

/// AVX2-accelerated out-of-place softmax reading from `src` and writing to `dst`.
/// Eliminates redundant memcpy buffer copies.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn softmax_scores_into_avx2(src: &[f32], dst: &mut [f32], q_len: usize, kv_len: usize) {
    use std::arch::x86_64::*;

    let d = kv_len;
    if d < 8 {
        softmax_scores_into_scalar(src, dst, q_len, kv_len);
        return;
    }

    let chunks32 = d / 32;
    let chunks8 = d / 8;
    let tail_start = chunks8 * 8;

    let min_x = _mm256_set1_ps(-87.336544);
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693145751953125);
    let c2 = _mm256_set1_ps(1.4286068203094172e-6);
    let c_p6 = _mm256_set1_ps(1.0 / 720.0);
    let c_p5 = _mm256_set1_ps(1.0 / 120.0);
    let c_p4 = _mm256_set1_ps(1.0 / 24.0);
    let c_p3 = _mm256_set1_ps(1.0 / 6.0);
    let c_p2 = _mm256_set1_ps(0.5);
    let c_one = _mm256_set1_ps(1.0);
    let bias_int = _mm256_set1_epi32(127);

    for (src_row, dst_row) in src[..q_len * d].chunks_exact(d).zip(dst[..q_len * d].chunks_exact_mut(d)) {
        let s_ptr = src_row.as_ptr();
        let d_ptr = dst_row.as_mut_ptr();

        // Pass 1: compute row maximum from src
        let mut max0 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max1 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max2 = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut max3 = _mm256_set1_ps(f32::NEG_INFINITY);

        for i in 0..chunks32 {
            let p = s_ptr.add(i * 32);
            max0 = _mm256_max_ps(max0, _mm256_loadu_ps(p));
            max1 = _mm256_max_ps(max1, _mm256_loadu_ps(p.add(8)));
            max2 = _mm256_max_ps(max2, _mm256_loadu_ps(p.add(16)));
            max3 = _mm256_max_ps(max3, _mm256_loadu_ps(p.add(24)));
        }

        let mut max_v = _mm256_max_ps(_mm256_max_ps(max0, max1), _mm256_max_ps(max2, max3));
        for i in (chunks32 * 4)..chunks8 {
            max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(s_ptr.add(i * 8)));
        }

        let max128 = _mm_max_ps(
            _mm256_castps256_ps128(max_v),
            _mm256_extractf128_ps(max_v, 1),
        );
        let max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
        let max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, 1));
        let mut max_val = _mm_cvtss_f32(max32);

        for i in tail_start..d {
            let v = *s_ptr.add(i);
            if v > max_val {
                max_val = v;
            }
        }

        // Pass 2: compute exp(x - max) from src, store directly to dst
        let max_broadcast = _mm256_set1_ps(max_val);
        let mut sum0 = _mm256_setzero_ps();
        let mut sum1 = _mm256_setzero_ps();
        let mut sum2 = _mm256_setzero_ps();
        let mut sum3 = _mm256_setzero_ps();

        for i in 0..chunks32 {
            let sp0 = s_ptr.add(i * 32);
            let sp1 = sp0.add(8);
            let sp2 = sp0.add(16);
            let sp3 = sp0.add(24);

            let dp0 = d_ptr.add(i * 32);
            let dp1 = dp0.add(8);
            let dp2 = dp0.add(16);
            let dp3 = dp0.add(24);

            let x0 = _mm256_sub_ps(_mm256_loadu_ps(sp0), max_broadcast);
            let x1 = _mm256_sub_ps(_mm256_loadu_ps(sp1), max_broadcast);
            let x2 = _mm256_sub_ps(_mm256_loadu_ps(sp2), max_broadcast);
            let x3 = _mm256_sub_ps(_mm256_loadu_ps(sp3), max_broadcast);

            let (e0, e1, e2, e3) = exp256_ps_4x(
                x0, x1, x2, x3, min_x, log2e, c1, c2, c_p6, c_p5, c_p4, c_p3, c_p2, c_one, bias_int,
            );
            sum0 = _mm256_add_ps(sum0, e0);
            sum1 = _mm256_add_ps(sum1, e1);
            sum2 = _mm256_add_ps(sum2, e2);
            sum3 = _mm256_add_ps(sum3, e3);

            _mm256_storeu_ps(dp0, e0);
            _mm256_storeu_ps(dp1, e1);
            _mm256_storeu_ps(dp2, e2);
            _mm256_storeu_ps(dp3, e3);
        }

        let mut sum_v = _mm256_add_ps(_mm256_add_ps(sum0, sum1), _mm256_add_ps(sum2, sum3));
        for i in (chunks32 * 4)..chunks8 {
            let sp = s_ptr.add(i * 8);
            let dp = d_ptr.add(i * 8);
            let diff = _mm256_sub_ps(_mm256_loadu_ps(sp), max_broadcast);
            let e = exp256_ps(diff);
            sum_v = _mm256_add_ps(sum_v, e);
            _mm256_storeu_ps(dp, e);
        }

        let sum128 = _mm_add_ps(
            _mm256_castps256_ps128(sum_v),
            _mm256_extractf128_ps(sum_v, 1),
        );
        let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
        let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
        let mut sum = _mm_cvtss_f32(sum32);

        for i in tail_start..d {
            let e = (*s_ptr.add(i) - max_val).exp();
            *d_ptr.add(i) = e;
            sum += e;
        }

        // Pass 3: normalize dst
        let inv = 1.0 / sum;
        let inv_broadcast = _mm256_set1_ps(inv);

        for i in 0..chunks32 {
            let dp0 = d_ptr.add(i * 32);
            let dp1 = dp0.add(8);
            let dp2 = dp0.add(16);
            let dp3 = dp0.add(24);

            let v0 = _mm256_mul_ps(_mm256_loadu_ps(dp0), inv_broadcast);
            let v1 = _mm256_mul_ps(_mm256_loadu_ps(dp1), inv_broadcast);
            let v2 = _mm256_mul_ps(_mm256_loadu_ps(dp2), inv_broadcast);
            let v3 = _mm256_mul_ps(_mm256_loadu_ps(dp3), inv_broadcast);

            _mm256_storeu_ps(dp0, v0);
            _mm256_storeu_ps(dp1, v1);
            _mm256_storeu_ps(dp2, v2);
            _mm256_storeu_ps(dp3, v3);
        }

        for i in (chunks32 * 4)..chunks8 {
            let dp = d_ptr.add(i * 8);
            let v = _mm256_mul_ps(_mm256_loadu_ps(dp), inv_broadcast);
            _mm256_storeu_ps(dp, v);
        }

        for i in tail_start..d {
            *d_ptr.add(i) *= inv;
        }
    }
}

#[inline]
pub fn softmax_scores_into_scalar(src: &[f32], dst: &mut [f32], q_len: usize, kv_len: usize) {
    if q_len == 0 || kv_len == 0 {
        return;
    }
    for (src_row, dst_row) in src[..q_len * kv_len]
        .chunks_exact(kv_len)
        .zip(dst[..q_len * kv_len].chunks_exact_mut(kv_len))
    {
        let mut max = f32::NEG_INFINITY;
        for &s in src_row.iter() {
            if s > max {
                max = s;
            }
        }

        let mut sum = 0.0f32;
        for (s, d) in src_row.iter().zip(dst_row.iter_mut()) {
            let e = (*s - max).exp();
            *d = e;
            sum += e;
        }

        let inv = 1.0 / sum;
        for d in dst_row.iter_mut() {
            *d *= inv;
        }
    }
}

#[inline]
pub fn softmax_scores_into(src: &[f32], dst: &mut [f32], q_len: usize, kv_len: usize) {
    let total = match q_len.checked_mul(kv_len) {
        Some(t) if t > 0 => t,
        _ => return,
    };
    if src.len() < total || dst.len() < total {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::utils::arch::has_avx2() {
            unsafe {
                softmax_scores_into_avx2(src, dst, q_len, kv_len);
                return;
            }
        }
    }
    softmax_scores_into_scalar(src, dst, q_len, kv_len);
}

#[inline]
pub fn softmax_scores_inplace_scalar(scores: &mut [f32], q_len: usize, kv_len: usize) {
    if q_len == 0 || kv_len == 0 {
        return;
    }
    for row in scores[..q_len * kv_len].chunks_exact_mut(kv_len) {
        let mut max = f32::NEG_INFINITY;
        for &s in row.iter() {
            if s > max {
                max = s;
            }
        }

        let mut sum = 0.0f32;
        for s in row.iter_mut() {
            let e = (*s - max).exp();
            *s = e;
            sum += e;
        }

        let inv = 1.0 / sum;
        for s in row.iter_mut() {
            *s *= inv;
        }
    }
}

#[inline]
pub fn softmax_scores_inplace(scores: &mut [f32], q_len: usize, kv_len: usize) {
    let total = match q_len.checked_mul(kv_len) {
        Some(t) if t > 0 => t,
        _ => return,
    };
    if scores.len() < total {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if crate::utils::arch::has_avx2() {
            unsafe {
                softmax_scores_inplace_avx2(scores, q_len, kv_len);
                return;
            }
        }
    }
    softmax_scores_inplace_scalar(scores, q_len, kv_len);
}

#[inline]
pub fn softmax(x: &Tensor2) -> Tensor2 {
    let mut out = Tensor2::zeros(x.shape);
    softmax_into(x, &mut out);
    out
}

#[inline]
pub fn softmax_into(x: &Tensor2, out: &mut Tensor2) {
    let d = x.shape[1];
    if d == 0 || x.data.is_empty() {
        out.shape = x.shape;
        out.data.clear();
        return;
    }
    let total_len = x.shape[0] * d;
    if out.data.len() != total_len {
        out.data.resize(total_len, 0.0);
    }
    out.shape = x.shape;
    softmax_scores_into(&x.data, &mut out.data, x.shape[0], d);
}

#[inline]
pub fn softmax_inplace(x: &mut Tensor2) {
    let d = x.shape[1];
    if d == 0 || x.data.is_empty() {
        return;
    }
    let rows = x.data.len() / d;
    softmax_scores_inplace(&mut x.data, rows, d);
}

// ===========================================================================
// Activation functions (ReLU, GELU, Sigmoid)
// ===========================================================================

#[inline]
pub fn relu_f32(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

#[inline]
pub fn relu(x: &Tensor2) -> Tensor2 {
    let mut out = x.clone();
    relu_inplace(&mut out);
    out
}

#[inline]
pub fn relu_into(x: &Tensor2, out: &mut Tensor2) {
    unsafe {
        out.prepare_uninit(x.data.len());
    }
    for (o, &v) in out.data.iter_mut().zip(x.data.iter()) {
        *o = relu_f32(v);
    }
    out.shape = x.shape;
}

#[inline]
pub fn relu_inplace(x: &mut Tensor2) {
    for v in x.data.iter_mut() {
        *v = relu_f32(*v);
    }
}

#[inline]
pub fn gelu_f32(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.7978845608;
    0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh())
}

#[inline]
pub fn gelu(x: &Tensor2) -> Tensor2 {
    let mut out = x.clone();
    gelu_inplace(&mut out);
    out
}

#[inline]
pub fn gelu_into(x: &Tensor2, out: &mut Tensor2) {
    unsafe {
        out.prepare_uninit(x.data.len());
    }
    #[cfg(target_arch = "x86_64")]
    if crate::utils::arch::has_avx2() {
        unsafe {
            gelu_into_avx2(&x.data, &mut out.data);
        }
        out.shape = x.shape;
        return;
    }
    for (o, &v) in out.data.iter_mut().zip(x.data.iter()) {
        *o = gelu_f32(v);
    }
    out.shape = x.shape;
}

#[inline]
pub fn gelu_inplace(x: &mut Tensor2) {
    #[cfg(target_arch = "x86_64")]
    if crate::utils::arch::has_avx2() {
        unsafe {
            gelu_inplace_avx2(&mut x.data);
        }
        return;
    }
    for v in x.data.iter_mut() {
        *v = gelu_f32(*v);
    }
}

/// Vectorized GELU calculation for slices using AVX2 and FMA.
/// Evaluates `x * sigmoid(2 * sqrt(2/pi) * (x + 0.044715 * x^3))` using fast vector exponential.
/// Numerical precision is within 1e-4 (typically < 1e-6) of `gelu_f32`.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision)]
pub unsafe fn gelu_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = data.len();
    let c1 = _mm256_set1_ps(1.5957691_f32);
    let c2 = _mm256_set1_ps(0.071354816_f32);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let sign_bit = _mm256_set1_ps(-0.0);

    let chunks = n / 16;
    let ptr = data.as_mut_ptr();

    for i in 0..chunks {
        let p = ptr.add(i * 16);
        let v0 = _mm256_loadu_ps(p);
        let v1 = _mm256_loadu_ps(p.add(8));

        let x2_0 = _mm256_mul_ps(v0, v0);
        let x2_1 = _mm256_mul_ps(v1, v1);
        let z0 = _mm256_mul_ps(v0, _mm256_fmadd_ps(x2_0, c2, c1));
        let z1 = _mm256_mul_ps(v1, _mm256_fmadd_ps(x2_1, c2, c1));

        let abs_z0 = _mm256_andnot_ps(sign_bit, z0);
        let abs_z1 = _mm256_andnot_ps(sign_bit, z1);
        let neg_abs_z0 = _mm256_or_ps(abs_z0, sign_bit);
        let neg_abs_z1 = _mm256_or_ps(abs_z1, sign_bit);

        let exp_neg0 = exp256_ps(neg_abs_z0);
        let exp_neg1 = exp256_ps(neg_abs_z1);

        let sig_abs0 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg0));
        let sig_abs1 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg1));

        let ge_mask0 = _mm256_cmp_ps(z0, zero, _CMP_GE_OQ);
        let ge_mask1 = _mm256_cmp_ps(z1, zero, _CMP_GE_OQ);

        let sig0 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs0), sig_abs0, ge_mask0);
        let sig1 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs1), sig_abs1, ge_mask1);

        _mm256_storeu_ps(p, _mm256_mul_ps(v0, sig0));
        _mm256_storeu_ps(p.add(8), _mm256_mul_ps(v1, sig1));
    }

    let rem_start = chunks * 16;
    let rem = n - rem_start;
    if rem >= 8 {
        let p = ptr.add(rem_start);
        let v0 = _mm256_loadu_ps(p);
        let x2_0 = _mm256_mul_ps(v0, v0);
        let z0 = _mm256_mul_ps(v0, _mm256_fmadd_ps(x2_0, c2, c1));
        let abs_z0 = _mm256_andnot_ps(sign_bit, z0);
        let neg_abs_z0 = _mm256_or_ps(abs_z0, sign_bit);
        let exp_neg0 = exp256_ps(neg_abs_z0);
        let sig_abs0 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg0));
        let ge_mask0 = _mm256_cmp_ps(z0, zero, _CMP_GE_OQ);
        let sig0 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs0), sig_abs0, ge_mask0);
        _mm256_storeu_ps(p, _mm256_mul_ps(v0, sig0));
    }

    let tail_start = rem_start + (rem / 8) * 8;
    for v in &mut data[tail_start..] {
        *v = gelu_f32(*v);
    }
}

/// Out-of-place vectorized GELU calculation for slices using AVX2 and FMA.
///
/// # Safety
/// Requires AVX2 and FMA CPU features. `dst.len()` must be at least `src.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::excessive_precision)]
pub unsafe fn gelu_into_avx2(src: &[f32], dst: &mut [f32]) {
    use std::arch::x86_64::*;

    let n = src.len();
    debug_assert_eq!(n, dst.len());
    let c1 = _mm256_set1_ps(1.5957691_f32);
    let c2 = _mm256_set1_ps(0.071354816_f32);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let sign_bit = _mm256_set1_ps(-0.0);

    let chunks = n / 16;
    let s_ptr = src.as_ptr();
    let d_ptr = dst.as_mut_ptr();

    for i in 0..chunks {
        let sp = s_ptr.add(i * 16);
        let dp = d_ptr.add(i * 16);
        let v0 = _mm256_loadu_ps(sp);
        let v1 = _mm256_loadu_ps(sp.add(8));

        let x2_0 = _mm256_mul_ps(v0, v0);
        let x2_1 = _mm256_mul_ps(v1, v1);
        let z0 = _mm256_mul_ps(v0, _mm256_fmadd_ps(x2_0, c2, c1));
        let z1 = _mm256_mul_ps(v1, _mm256_fmadd_ps(x2_1, c2, c1));

        let abs_z0 = _mm256_andnot_ps(sign_bit, z0);
        let abs_z1 = _mm256_andnot_ps(sign_bit, z1);
        let neg_abs_z0 = _mm256_or_ps(abs_z0, sign_bit);
        let neg_abs_z1 = _mm256_or_ps(abs_z1, sign_bit);

        let exp_neg0 = exp256_ps(neg_abs_z0);
        let exp_neg1 = exp256_ps(neg_abs_z1);

        let sig_abs0 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg0));
        let sig_abs1 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg1));

        let ge_mask0 = _mm256_cmp_ps(z0, zero, _CMP_GE_OQ);
        let ge_mask1 = _mm256_cmp_ps(z1, zero, _CMP_GE_OQ);

        let sig0 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs0), sig_abs0, ge_mask0);
        let sig1 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs1), sig_abs1, ge_mask1);

        _mm256_storeu_ps(dp, _mm256_mul_ps(v0, sig0));
        _mm256_storeu_ps(dp.add(8), _mm256_mul_ps(v1, sig1));
    }

    let rem_start = chunks * 16;
    let rem = n - rem_start;
    if rem >= 8 {
        let sp = s_ptr.add(rem_start);
        let dp = d_ptr.add(rem_start);
        let v0 = _mm256_loadu_ps(sp);
        let x2_0 = _mm256_mul_ps(v0, v0);
        let z0 = _mm256_mul_ps(v0, _mm256_fmadd_ps(x2_0, c2, c1));
        let abs_z0 = _mm256_andnot_ps(sign_bit, z0);
        let neg_abs_z0 = _mm256_or_ps(abs_z0, sign_bit);
        let exp_neg0 = exp256_ps(neg_abs_z0);
        let sig_abs0 = _mm256_div_ps(one, _mm256_add_ps(one, exp_neg0));
        let ge_mask0 = _mm256_cmp_ps(z0, zero, _CMP_GE_OQ);
        let sig0 = _mm256_blendv_ps(_mm256_sub_ps(one, sig_abs0), sig_abs0, ge_mask0);
        _mm256_storeu_ps(dp, _mm256_mul_ps(v0, sig0));
    }

    let tail_start = rem_start + (rem / 8) * 8;
    for (o, &v) in dst[tail_start..].iter_mut().zip(&src[tail_start..]) {
        *o = gelu_f32(v);
    }
}
#[inline]
pub fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
pub fn sigmoid(x: &Tensor2) -> Tensor2 {
    let mut out = x.clone();
    sigmoid_inplace(&mut out);
    out
}

#[inline]
pub fn sigmoid_into(x: &Tensor2, out: &mut Tensor2) {
    unsafe {
        out.prepare_uninit(x.data.len());
    }
    for (o, &v) in out.data.iter_mut().zip(x.data.iter()) {
        *o = sigmoid_f32(v);
    }
    out.shape = x.shape;
}

#[inline]
pub fn sigmoid_inplace(x: &mut Tensor2) {
    for v in x.data.iter_mut() {
        *v = sigmoid_f32(*v);
    }
}

// ===========================================================================
// Attention auxiliary functions
// ===========================================================================

/// Transpose one attention head's K block (`k[j * d_model + col + t]`,
/// strided row-major) into the t-major panel `out[t * kv_len + j]`.
#[inline]
pub fn transpose_head_block_tmajor(
    k: &[f32],
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    out: &mut [f32],
) {
    debug_assert!(out.len() >= dk * kv_len);
    for j in 0..kv_len {
        let in_off = j * d_model + col;
        for t in 0..dk {
            out[t * kv_len + j] = k[in_off + t];
        }
    }
}

/// Scalar reference: sequential t-chain per `(i, j)`. Reads the original
/// head block directly.
#[inline]
pub fn score_rows_scalar(
    q: &[f32],
    k: &[f32],
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    scale: f32,
    out: &mut [f32],
) {
    for i in 0..q_len {
        let q_off = i * d_model + col;
        let o_off = i * kv_len;
        for j in 0..kv_len {
            let k_off = j * d_model + col;
            let mut acc = 0.0f32;
            for t in 0..dk {
                acc += q[q_off + t] * k[k_off + t];
            }
            out[o_off + j] = acc * scale;
        }
    }
}

/// Scalar reference over a t-major transposed panel `kt[t * kv_len + j]`.
/// Numerically bit-identical to [`score_rows_scalar`]; runs in the same t
/// accumulation order and rounds identical products. Used by dispatch when
/// `q_len > 1` (where the transpose is already amortised) on platforms
/// without an AVX2 kernel, and by callers who already hold a transposed
/// head-major panel (whose memory layout the strided kernel cannot read).
#[inline]
pub fn score_rows_scalar_tmajor(
    q: &[f32],
    kt: &[f32],
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    scale: f32,
    out: &mut [f32],
) {
    for i in 0..q_len {
        let q_off = i * d_model + col;
        let o_off = i * kv_len;
        for j in 0..kv_len {
            let mut acc = 0.0f32;
            for t in 0..dk {
                acc += q[q_off + t] * kt[t * kv_len + j];
            }
            out[o_off + j] = acc * scale;
        }
    }
}

/// AVX2 attention score kernel over a t-major transposed K panel.
///
/// # Safety
///
/// The caller must ensure that the CPU supports AVX2 and FMA instructions,
/// and that `q`, `kt`, and `out` have sufficient capacity for their respective dimensions.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
pub unsafe fn score_rows_avx2_tmajor(
    q: &[f32],
    kt: &[f32],
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    scale: f32,
    out: &mut [f32],
) {
    use std::arch::x86_64::*;
    const W: usize = 8;
    let groups = kv_len / W;
    let jtail = groups * W;
    let scale_v = _mm256_set1_ps(scale);

    for i in 0..q_len {
        let q_off = i * d_model + col;
        let o_off = i * kv_len;
        let orow = &mut out[o_off..o_off + kv_len];

        // 16-wide unroll (two 8-wide AVX registers) across j columns.
        let mut g = 0;
        while g + 1 < groups {
            let j0 = g * W;
            let mut acc0 = _mm256_setzero_ps();
            let mut acc1 = _mm256_setzero_ps();
            for t in 0..dk {
                let q_val = *q.get_unchecked(q_off + t);
                let q_broadcast = _mm256_set1_ps(q_val);
                let kt_row = kt.as_ptr().add(t * kv_len + j0);
                let k0 = _mm256_loadu_ps(kt_row);
                let k1 = _mm256_loadu_ps(kt_row.add(W));
                acc0 = _mm256_fmadd_ps(q_broadcast, k0, acc0);
                acc1 = _mm256_fmadd_ps(q_broadcast, k1, acc1);
            }
            acc0 = _mm256_mul_ps(acc0, scale_v);
            acc1 = _mm256_mul_ps(acc1, scale_v);
            _mm256_storeu_ps(orow.get_unchecked_mut(j0) as *mut f32, acc0);
            _mm256_storeu_ps(orow.get_unchecked_mut(j0 + W) as *mut f32, acc1);
            g += 2;
        }
        if groups % 2 != 0 {
            let j0 = (groups - 1) * W;
            let mut accs = _mm256_setzero_ps();
            for t in 0..dk {
                let q_val = *q.get_unchecked(q_off + t);
                let q_broadcast = _mm256_set1_ps(q_val);
                let kt_row = kt.as_ptr().add(t * kv_len + j0);
                let k0 = _mm256_loadu_ps(kt_row);
                accs = _mm256_fmadd_ps(q_broadcast, k0, accs);
            }
            accs = _mm256_mul_ps(accs, scale_v);
            _mm256_storeu_ps(orow.get_unchecked_mut(j0) as *mut f32, accs);
        }
        // j tail: exact scalar chain over the transposed panel.
        for j in jtail..kv_len {
            let mut acc = 0.0f32;
            for t in 0..dk {
                acc += *q.get_unchecked(q_off + t) * *kt.get_unchecked(t * kv_len + j);
            }
            *orow.get_unchecked_mut(j) = acc * scale;
        }
    }
}

/// Score rows with the runtime-selected kernel.
#[inline]
pub fn score_rows_dispatch(
    q: &[f32],
    k: Option<&[f32]>,
    kt: Option<&[f32]>,
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    scale: f32,
    out: &mut [f32],
) {
    if let Some(kt) = kt {
        #[cfg(target_arch = "x86_64")]
        {
            if crate::utils::arch::has_avx2() {
                return unsafe {
                    score_rows_avx2_tmajor(q, kt, q_len, kv_len, d_model, col, dk, scale, out)
                };
            }
        }
        score_rows_scalar_tmajor(q, kt, q_len, kv_len, d_model, col, dk, scale, out);
        return;
    }
    let k = k.expect("score_rows_dispatch requires k when kt is None");
    score_rows_scalar(q, k, q_len, kv_len, d_model, col, dk, scale, out);
}

/// Attention context accumulation helper: `out_ctx[i][col+t] = Σ_j scores[i][j] · v[j][col+t]`.
#[inline]
pub fn accumulate_head_v(
    scores: &[f32],
    v: &[f32],
    q_len: usize,
    kv_len: usize,
    d_model: usize,
    col: usize,
    dk: usize,
    out_ctx: &mut [f32],
) {
    for i in 0..q_len {
        let o_off = i * d_model + col;
        let out_slice = &mut out_ctx[o_off..o_off + dk];
        out_slice.fill(0.0);
        for j in 0..kv_len {
            let w = scores[i * kv_len + j];
            let v_off = j * d_model + col;
            let v_slice = &v[v_off..v_off + dk];
            for (o, &val) in out_slice.iter_mut().zip(v_slice.iter()) {
                *o += w * val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_scores_inplace_matches_scalar() {
        let cases = [
            (0, 0),
            (1, 0),
            (0, 8),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 7),
            (1, 8),
            (1, 9),
            (1, 15),
            (1, 16),
            (1, 17),
            (2, 31),
            (2, 32),
            (2, 33),
            (4, 64),
            (8, 127),
            (8, 128),
            (4, 256),
            (2, 384),
        ];

        for (q_len, kv_len) in cases {
            if q_len == 0 || kv_len == 0 {
                let mut data = vec![0.0; 10];
                softmax_scores_inplace(&mut data, q_len, kv_len);
                continue;
            }
            let total = q_len * kv_len;
            let original: Vec<f32> = (0..total)
                .map(|i| ((i * 19 + 7) % 73) as f32 * 0.1 - 3.5)
                .collect();
            let mut ref_scores = original.clone();
            let mut opt_scores = original;

            softmax_scores_inplace_scalar(&mut ref_scores, q_len, kv_len);
            softmax_scores_inplace(&mut opt_scores, q_len, kv_len);

            for (idx, (a, b)) in ref_scores.iter().zip(opt_scores.iter()).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-5,
                    "Mismatch at {idx} for ({q_len}, {kv_len}): ref={a}, opt={b}, diff={}",
                    (a - b).abs()
                );
            }

            for row in opt_scores.chunks_exact(kv_len) {
                let row_sum: f32 = row.iter().sum();
                assert!(
                    (row_sum - 1.0).abs() <= 1e-5,
                    "Row sum mismatch for ({q_len}, {kv_len}): got {row_sum}"
                );
            }
        }
    }

    #[test]
    fn test_softmax_underflow_masking() {
        let mut scores = vec![-10000.0, -5000.0, 0.0, -1000.0, -2000.0, -100.0, -90.0, 1.0];
        softmax_scores_inplace(&mut scores, 1, 8);
        assert_eq!(scores[0], 0.0);
        assert_eq!(scores[1], 0.0);
        assert_eq!(scores[3], 0.0);
        assert_eq!(scores[4], 0.0);
        assert_eq!(scores[5], 0.0);
        assert_eq!(scores[6], 0.0);
        let sum: f32 = scores.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-5);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_exp256_ps_accuracy() {
        if !crate::utils::arch::has_avx2() {
            return;
        }
        use std::arch::x86_64::*;
        let mut vals = [0.0f32; 8];
        for step in 0..1000 {
            let base = -(step as f32) * 0.088;
            for i in 0..8 {
                vals[i] = base - (i as f32) * 0.01;
            }
            unsafe {
                let v = _mm256_loadu_ps(vals.as_ptr());
                let exp_v = exp256_ps(v);
                let mut out = [0.0f32; 8];
                _mm256_storeu_ps(out.as_mut_ptr(), exp_v);

                for i in 0..8 {
                    let expected = vals[i].exp();
                    let got = out[i];
                    if expected > 1e-30 {
                        let rel = (expected - got).abs() / expected;
                        assert!(
                            rel <= 1e-5,
                            "Rel error at {}: exp={}, got={}, rel={rel}",
                            vals[i],
                            expected,
                            got
                        );
                    } else {
                        assert!(
                            (expected - got).abs() <= 1e-6,
                            "Abs error at {}: exp={}, got={}",
                            vals[i],
                            expected,
                            got
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_softmax_tensor_api() {
        let t = Tensor2::from_vec([2, 4], vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 1.5, -1.5]);
        let sm = softmax(&t);
        assert_eq!(sm.shape, [2, 4]);
        for row in sm.data.chunks_exact(4) {
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() <= 1e-5);
        }

        let mut out = Tensor2::zeros([0, 0]);
        softmax_into(&t, &mut out);
        assert_eq!(out.data, sm.data);

        let mut t_mut = t.clone();
        softmax_inplace(&mut t_mut);
        assert_eq!(t_mut.data, sm.data);
    }

    #[test]
    fn test_softmax_empty_tensor() {
        for shape in [[0, 0], [0, 5], [5, 0]] {
            let mut t = Tensor2::zeros(shape);
            t.softmax_inplace();
            assert_eq!(t.shape, shape);
            assert!(t.data.is_empty());

            let sm = t.softmax();
            assert_eq!(sm.shape, shape);
            assert!(sm.data.is_empty());

            let mut out = Tensor2::zeros([1, 1]);
            t.softmax_into(&mut out);
            assert_eq!(out.shape, shape);
            assert!(out.data.is_empty());
        }

        let mut default_t = Tensor2::default();
        default_t.softmax_inplace();
        assert_eq!(default_t.shape, [0, 0]);
        assert!(default_t.data.is_empty());
    }
}
