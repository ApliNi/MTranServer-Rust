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

/// ARM NEON `M×K @ K×N` f32 kernel with memory & register blocking.
///
/// Output columns are tiled in cache-fitting memory blocks (`N_TILE = 64`),
/// keeping the B panel cache-resident across all M rows. Within each tile,
/// an `MR×NR` (4×8) 2D register tile accumulates 8 `float32x4_t` registers
/// in-flight, streaming K dot products with zero heap allocations.
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
        vaddq_f32, vdupq_n_f32, vld1q_f32, vmaxq_f32, vmlaq_f32, vst1q_f32,
    };

    if m == 0 || k == 0 || n == 0 {
        return;
    }
    if m == 1 {
        matmul_row_neon(a, b, k, n, bias, out, relu);
        return;
    }

    const N_TILE: usize = 64;
    let zero = vdupq_n_f32(0.0);
    let b_ptr = b.as_ptr();
    let bias_ptr = bias.map(|bb| bb.as_ptr());

    for j_start in (0..n).step_by(N_TILE) {
        let j_len = (n - j_start).min(N_TILE);
        let n_blocks8 = j_len / 8;
        let n_rem8 = j_len % 8;
        let n_blocks4 = n_rem8 / 4;
        let tail = n_rem8 % 4;

        let mut i = 0;
        // 4-row register tile (MR = 4, NR = 8)
        while i + 3 < m {
            let a0 = a.as_ptr().add(i * k);
            let a1 = a.as_ptr().add((i + 1) * k);
            let a2 = a.as_ptr().add((i + 2) * k);
            let a3 = a.as_ptr().add((i + 3) * k);

            let o0 = out.as_mut_ptr().add(i * n);
            let o1 = out.as_mut_ptr().add((i + 1) * n);
            let o2 = out.as_mut_ptr().add((i + 2) * n);
            let o3 = out.as_mut_ptr().add((i + 3) * n);

            for blk in 0..n_blocks8 {
                let col = j_start + blk * 8;
                let mut c00 = zero;
                let mut c01 = zero;
                let mut c10 = zero;
                let mut c11 = zero;
                let mut c20 = zero;
                let mut c21 = zero;
                let mut c30 = zero;
                let mut c31 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let av1 = vdupq_n_f32(*a1.add(t));
                    let av2 = vdupq_n_f32(*a2.add(t));
                    let av3 = vdupq_n_f32(*a3.add(t));

                    let bp = b_ptr.add(t * n + col);
                    let bv0 = vld1q_f32(bp);
                    let bv1 = vld1q_f32(bp.add(4));

                    c00 = vmlaq_f32(c00, av0, bv0);
                    c01 = vmlaq_f32(c01, av0, bv1);
                    c10 = vmlaq_f32(c10, av1, bv0);
                    c11 = vmlaq_f32(c11, av1, bv1);
                    c20 = vmlaq_f32(c20, av2, bv0);
                    c21 = vmlaq_f32(c21, av2, bv1);
                    c30 = vmlaq_f32(c30, av3, bv0);
                    c31 = vmlaq_f32(c31, av3, bv1);
                }

                if let Some(bp) = bias_ptr {
                    let b0 = vld1q_f32(bp.add(col));
                    let b1 = vld1q_f32(bp.add(col + 4));
                    c00 = vaddq_f32(c00, b0);
                    c01 = vaddq_f32(c01, b1);
                    c10 = vaddq_f32(c10, b0);
                    c11 = vaddq_f32(c11, b1);
                    c20 = vaddq_f32(c20, b0);
                    c21 = vaddq_f32(c21, b1);
                    c30 = vaddq_f32(c30, b0);
                    c31 = vaddq_f32(c31, b1);
                }

                if relu {
                    c00 = vmaxq_f32(c00, zero);
                    c01 = vmaxq_f32(c01, zero);
                    c10 = vmaxq_f32(c10, zero);
                    c11 = vmaxq_f32(c11, zero);
                    c20 = vmaxq_f32(c20, zero);
                    c21 = vmaxq_f32(c21, zero);
                    c30 = vmaxq_f32(c30, zero);
                    c31 = vmaxq_f32(c31, zero);
                }

                vst1q_f32(o0.add(col), c00);
                vst1q_f32(o0.add(col + 4), c01);
                vst1q_f32(o1.add(col), c10);
                vst1q_f32(o1.add(col + 4), c11);
                vst1q_f32(o2.add(col), c20);
                vst1q_f32(o2.add(col + 4), c21);
                vst1q_f32(o3.add(col), c30);
                vst1q_f32(o3.add(col + 4), c31);
            }

            if n_blocks4 > 0 {
                let col = j_start + n_blocks8 * 8;
                let mut c0 = zero;
                let mut c1 = zero;
                let mut c2 = zero;
                let mut c3 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let av1 = vdupq_n_f32(*a1.add(t));
                    let av2 = vdupq_n_f32(*a2.add(t));
                    let av3 = vdupq_n_f32(*a3.add(t));
                    let bv = vld1q_f32(b_ptr.add(t * n + col));

                    c0 = vmlaq_f32(c0, av0, bv);
                    c1 = vmlaq_f32(c1, av1, bv);
                    c2 = vmlaq_f32(c2, av2, bv);
                    c3 = vmlaq_f32(c3, av3, bv);
                }

                if let Some(bp) = bias_ptr {
                    let b = vld1q_f32(bp.add(col));
                    c0 = vaddq_f32(c0, b);
                    c1 = vaddq_f32(c1, b);
                    c2 = vaddq_f32(c2, b);
                    c3 = vaddq_f32(c3, b);
                }

                if relu {
                    c0 = vmaxq_f32(c0, zero);
                    c1 = vmaxq_f32(c1, zero);
                    c2 = vmaxq_f32(c2, zero);
                    c3 = vmaxq_f32(c3, zero);
                }

                vst1q_f32(o0.add(col), c0);
                vst1q_f32(o1.add(col), c1);
                vst1q_f32(o2.add(col), c2);
                vst1q_f32(o3.add(col), c3);
            }

            if tail > 0 {
                let col = j_start + n_blocks8 * 8 + n_blocks4 * 4;
                for j in 0..tail {
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    for t in 0..k {
                        let bv = *b_ptr.add(t * n + col + j);
                        s0 += *a0.add(t) * bv;
                        s1 += *a1.add(t) * bv;
                        s2 += *a2.add(t) * bv;
                        s3 += *a3.add(t) * bv;
                    }
                    if let Some(bp) = bias_ptr {
                        let bb = *bp.add(col + j);
                        s0 += bb;
                        s1 += bb;
                        s2 += bb;
                        s3 += bb;
                    }
                    *o0.add(col + j) = if relu { s0.max(0.0) } else { s0 };
                    *o1.add(col + j) = if relu { s1.max(0.0) } else { s1 };
                    *o2.add(col + j) = if relu { s2.max(0.0) } else { s2 };
                    *o3.add(col + j) = if relu { s3.max(0.0) } else { s3 };
                }
            }
            i += 4;
        }

        // 2-row register tile (MR = 2, NR = 8)
        while i + 1 < m {
            let a0 = a.as_ptr().add(i * k);
            let a1 = a.as_ptr().add((i + 1) * k);
            let o0 = out.as_mut_ptr().add(i * n);
            let o1 = out.as_mut_ptr().add((i + 1) * n);

            for blk in 0..n_blocks8 {
                let col = j_start + blk * 8;
                let mut c00 = zero;
                let mut c01 = zero;
                let mut c10 = zero;
                let mut c11 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let av1 = vdupq_n_f32(*a1.add(t));
                    let bp = b_ptr.add(t * n + col);
                    let bv0 = vld1q_f32(bp);
                    let bv1 = vld1q_f32(bp.add(4));

                    c00 = vmlaq_f32(c00, av0, bv0);
                    c01 = vmlaq_f32(c01, av0, bv1);
                    c10 = vmlaq_f32(c10, av1, bv0);
                    c11 = vmlaq_f32(c11, av1, bv1);
                }

                if let Some(bp) = bias_ptr {
                    let b0 = vld1q_f32(bp.add(col));
                    let b1 = vld1q_f32(bp.add(col + 4));
                    c00 = vaddq_f32(c00, b0);
                    c01 = vaddq_f32(c01, b1);
                    c10 = vaddq_f32(c10, b0);
                    c11 = vaddq_f32(c11, b1);
                }

                if relu {
                    c00 = vmaxq_f32(c00, zero);
                    c01 = vmaxq_f32(c01, zero);
                    c10 = vmaxq_f32(c10, zero);
                    c11 = vmaxq_f32(c11, zero);
                }

                vst1q_f32(o0.add(col), c00);
                vst1q_f32(o0.add(col + 4), c01);
                vst1q_f32(o1.add(col), c10);
                vst1q_f32(o1.add(col + 4), c11);
            }

            if n_blocks4 > 0 {
                let col = j_start + n_blocks8 * 8;
                let mut c0 = zero;
                let mut c1 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let av1 = vdupq_n_f32(*a1.add(t));
                    let bv = vld1q_f32(b_ptr.add(t * n + col));

                    c0 = vmlaq_f32(c0, av0, bv);
                    c1 = vmlaq_f32(c1, av1, bv);
                }

                if let Some(bp) = bias_ptr {
                    let b = vld1q_f32(bp.add(col));
                    c0 = vaddq_f32(c0, b);
                    c1 = vaddq_f32(c1, b);
                }

                if relu {
                    c0 = vmaxq_f32(c0, zero);
                    c1 = vmaxq_f32(c1, zero);
                }

                vst1q_f32(o0.add(col), c0);
                vst1q_f32(o1.add(col), c1);
            }

            if tail > 0 {
                let col = j_start + n_blocks8 * 8 + n_blocks4 * 4;
                for j in 0..tail {
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    for t in 0..k {
                        let bv = *b_ptr.add(t * n + col + j);
                        s0 += *a0.add(t) * bv;
                        s1 += *a1.add(t) * bv;
                    }
                    if let Some(bp) = bias_ptr {
                        let bb = *bp.add(col + j);
                        s0 += bb;
                        s1 += bb;
                    }
                    *o0.add(col + j) = if relu { s0.max(0.0) } else { s0 };
                    *o1.add(col + j) = if relu { s1.max(0.0) } else { s1 };
                }
            }
            i += 2;
        }

        // 1-row remainder (MR = 1, NR = 8)
        if i < m {
            let a0 = a.as_ptr().add(i * k);
            let o0 = out.as_mut_ptr().add(i * n);

            for blk in 0..n_blocks8 {
                let col = j_start + blk * 8;
                let mut c00 = zero;
                let mut c01 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let bp = b_ptr.add(t * n + col);
                    let bv0 = vld1q_f32(bp);
                    let bv1 = vld1q_f32(bp.add(4));

                    c00 = vmlaq_f32(c00, av0, bv0);
                    c01 = vmlaq_f32(c01, av0, bv1);
                }

                if let Some(bp) = bias_ptr {
                    let b0 = vld1q_f32(bp.add(col));
                    let b1 = vld1q_f32(bp.add(col + 4));
                    c00 = vaddq_f32(c00, b0);
                    c01 = vaddq_f32(c01, b1);
                }

                if relu {
                    c00 = vmaxq_f32(c00, zero);
                    c01 = vmaxq_f32(c01, zero);
                }

                vst1q_f32(o0.add(col), c00);
                vst1q_f32(o0.add(col + 4), c01);
            }

            if n_blocks4 > 0 {
                let col = j_start + n_blocks8 * 8;
                let mut c0 = zero;

                for t in 0..k {
                    let av0 = vdupq_n_f32(*a0.add(t));
                    let bv = vld1q_f32(b_ptr.add(t * n + col));
                    c0 = vmlaq_f32(c0, av0, bv);
                }

                if let Some(bp) = bias_ptr {
                    let b = vld1q_f32(bp.add(col));
                    c0 = vaddq_f32(c0, b);
                }

                if relu {
                    c0 = vmaxq_f32(c0, zero);
                }

                vst1q_f32(o0.add(col), c0);
            }

            if tail > 0 {
                let col = j_start + n_blocks8 * 8 + n_blocks4 * 4;
                for j in 0..tail {
                    let mut s0 = 0.0f32;
                    for t in 0..k {
                        let bv = *b_ptr.add(t * n + col + j);
                        s0 += *a0.add(t) * bv;
                    }
                    if let Some(bp) = bias_ptr {
                        s0 += *bp.add(col + j);
                    }
                    *o0.add(col + j) = if relu { s0.max(0.0) } else { s0 };
                }
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
