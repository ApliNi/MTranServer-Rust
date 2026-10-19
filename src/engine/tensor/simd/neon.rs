//! ARM NEON accelerated SIMD kernels.

#![cfg(any(target_arch = "aarch64", target_arch = "arm"))]

use super::scalar::matmul_i8_scalar;
use std::arch::aarch64::*;

/// ARM NEON single-row f32 matmul.
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[target_feature(enable = "neon")]
pub unsafe fn matmul_row_neon(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmlaq_f32, vst1q_f32,
    };
    const NR: usize = 4;
    const MAX_STACK_GROUPS: usize = 256;
    let groups = n / NR;
    let tail = n % NR;
    let mut stack_acc = [vdupq_n_f32(0.0); MAX_STACK_GROUPS];
    let mut heap_acc: Vec<float32x4_t>;
    let acc_slice: &mut [float32x4_t] = if groups <= MAX_STACK_GROUPS {
        &mut stack_acc[..groups]
    } else {
        heap_acc = vec![vdupq_n_f32(0.0); groups];
        heap_acc.as_mut_slice()
    };
    for t in 0..k {
        let av = vdupq_n_f32(a[t]);
        for (g, ac) in acc_slice.iter_mut().enumerate() {
            let bv = vld1q_f32(b[t * n + g * NR..].as_ptr());
            *ac = vmlaq_f32(*ac, av, bv);
        }
    }
    for (g, &ac) in acc_slice.iter().enumerate() {
        let mut v = ac;
        if let Some(bb) = bias {
            v = vaddq_f32(v, vld1q_f32(bb[g * NR..].as_ptr()));
        }
        if relu {
            v = vmaxq_f32(v, vdupq_n_f32(0.0));
        }
        vst1q_f32(out[g * NR..].as_mut_ptr(), v);
    }
    if tail > 0 {
        for j in 0..tail {
            let c = groups * NR + j;
            let mut s = 0.0f32;
            for t in 0..k {
                s += a[t] * b[t * n + c];
            }
            let s = if let Some(bb) = bias { s + bb[c] } else { s };
            out[c] = if relu && s > 0.0 {
                s
            } else if relu {
                0.0
            } else {
                s
            };
        }
    }
}
/// ARM NEON `M×K @ K×N` f32 kernel.
///
/// 4-wide `float32x4_t` accumulators with fused multiply–accumulate
/// (`vmlaq_f32`), N-blocked as above. A scalar tail handles `n % 4`.
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[target_feature(enable = "neon")]
pub unsafe fn matmul_neon(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmlaq_f32, vst1q_f32,
    };

    const NR: usize = 4;
    let groups = n / NR;
    let tail = n % NR;
    let mut accs: Vec<float32x4_t> = vec![vdupq_n_f32(0.0); groups];
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        accs.fill(vdupq_n_f32(0.0));
        for t in 0..k {
            let av = vdupq_n_f32(a_row[t]);
            for (g, ac) in accs.iter_mut().enumerate() {
                let bv = vld1q_f32(b[t * n + g * NR..].as_ptr());
                *ac = vmlaq_f32(*ac, av, bv);
            }
        }
        if let Some(bb) = bias {
            for (g, ac) in accs.iter_mut().enumerate() {
                *ac = vaddq_f32(*ac, vld1q_f32(bb[g * NR..].as_ptr()));
            }
        }
        for (g, ac) in accs.iter().enumerate() {
            let mut v = *ac;
            if relu {
                v = vmaxq_f32(v, vdupq_n_f32(0.0));
            }
            vst1q_f32(o_row[g * NR..].as_mut_ptr(), v);
        }
        if tail > 0 {
            let c = groups * NR;
            for j in 0..tail {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += a_row[t] * b[t * n + c + j];
                }
                let s = if let Some(bb) = bias {
                    s + bb[c + j]
                } else {
                    s
                };
                o_row[c + j] = if relu && s > 0.0 {
                    s
                } else if relu {
                    0.0
                } else {
                    s
                };
            }
        }
    }
}
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn quantize_row_neon(data: &[f32], qa: f32, out: &mut [i8]) {
    use std::arch::aarch64::*;
    let len = data.len();
    let chunks = len / 8;
    let tail = len % 8;
    let qa_vec = vdupq_n_f32(qa);
    let half_pos = vdupq_n_f32(0.5);
    let half_neg = vdupq_n_f32(-0.5);
    let zero = vdupq_n_f32(0.0);
    let min_clamp = vdupq_n_f32(-127.0);
    let max_clamp = vdupq_n_f32(127.0);
    let neg127 = vdup_n_s8(-127);

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    for i in 0..chunks {
        let v0 = vld1q_f32(src.add(i * 8));
        let v1 = vld1q_f32(src.add(i * 8 + 4));

        let scaled0 = vmulq_f32(v0, qa_vec);
        let scaled1 = vmulq_f32(v1, qa_vec);

        let mask0 = vcgeq_f32(scaled0, zero);
        let mask1 = vcgeq_f32(scaled1, zero);

        let offset0 = vbslq_f32(mask0, half_pos, half_neg);
        let offset1 = vbslq_f32(mask1, half_pos, half_neg);

        let rounded0 = vaddq_f32(scaled0, offset0);
        let rounded1 = vaddq_f32(scaled1, offset1);

        let ord_mask0 = vceqq_f32(rounded0, rounded0);
        let ord_mask1 = vceqq_f32(rounded1, rounded1);

        let clamped0 = vminq_f32(vmaxq_f32(rounded0, min_clamp), max_clamp);
        let clamped1 = vminq_f32(vmaxq_f32(rounded1, min_clamp), max_clamp);

        let valid_clamped0 =
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(clamped0), ord_mask0));
        let valid_clamped1 =
            vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(clamped1), ord_mask1));

        let r0 = vcvtq_s32_f32(valid_clamped0);
        let r1 = vcvtq_s32_f32(valid_clamped1);

        let packed_16_0 = vqmovn_s32(r0);
        let packed_16_1 = vqmovn_s32(r1);
        let packed_16 = vcombine_s16(packed_16_0, packed_16_1);
        let packed_8 = vqmovn_s16(packed_16);
        let clamped_8 = vmax_s8(packed_8, neg127);

        vst1_s8(dst.add(i * 8), clamped_8);
    }

    let off = chunks * 8;
    for j in 0..tail {
        let v = *src.add(off + j);
        let scaled = v * qa;
        let offset = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let r = (scaled + offset) as i32;
        *dst.add(off + j) = r.clamp(-127, 127) as i8;
    }
}
/// ARM NEON `i8×i8 → i32` gemm kernel.
///
/// The i8×i8 → i16 widening (vmull_s8) followed by i32 accumulation
/// (vpadalq_s32) is left on the portable scalar kernel for ARM builds;
/// NEON's 128-bit throughput here is not competitive with the SSE/AVX2
/// lanes this engine actually runs on, and correctness identical to the
/// i32-accumulation reference is preserved.
#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
#[target_feature(enable = "neon")]
pub unsafe fn matmul_i8_neon(
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
    matmul_i8_scalar(x, w, m, k, n, bias, row_sums, out, denom, relu)
}
