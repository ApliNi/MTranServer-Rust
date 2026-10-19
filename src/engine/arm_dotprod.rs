//! ARMv8.2-a (and later) i8×i8→i32 dot-product (SDOT) GEMM kernels.
//!
//! The `vdot` (SDOT) instruction computes four independent 4×4 i8 dot
//! products and adds them to a 128-bit i32 accumulator in one operation.
//! Because every accumulation step is an exact integer i32 addition (the
//! products fit i32 and the cross-block additions are associative), the
//! result is bit-identical to the portable scalar kernel while the i32
//! accumulator never narrows. `dotprod` is the Armv8.2-a extension (FEAT_DotProd);
//! these kernels are only selected when the running CPU reports it, and only
//! compiled on `aarch64` (they do not exist on 32-bit ARM or x86).
//!
//! The layout note from the scalar kernels applies equally here: each output
//! row's dot product walks the `w` row linearly, so the 16-byte SDOT operand
//! pairs align with the scalar `Σ x[t]·w[t]` loop exactly.

/// `M×K @ K×N` i8 gemm with fused bias + dequantization (+ optional relu),
/// using SDOT (vdot) for the inner i32 accumulation.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_i8_dotprod(
    x: &[i8],
    w: &[i8],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    _row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    use std::arch::aarch64::*;
    let inv = denom.recip();
    const BLK: usize = 16;
    const NR: usize = 4;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        let or = &mut out[i * n..(i + 1) * n];
        let x_ptr = xr.as_ptr();

        for g in 0..groups {
            let j0 = g * NR;
            let wr0 = w.as_ptr().add(j0 * k);
            let wr1 = w.as_ptr().add((j0 + 1) * k);
            let wr2 = w.as_ptr().add((j0 + 2) * k);
            let wr3 = w.as_ptr().add((j0 + 3) * k);

            let mut acc0 = vdupq_n_s32(0);
            let mut acc1 = vdupq_n_s32(0);
            let mut acc2 = vdupq_n_s32(0);
            let mut acc3 = vdupq_n_s32(0);

            for b in 0..blocks {
                let off = b * BLK;
                let xv = vld1q_s8(x_ptr.add(off));
                let wv0 = vld1q_s8(wr0.add(off));
                let wv1 = vld1q_s8(wr1.add(off));
                let wv2 = vld1q_s8(wr2.add(off));
                let wv3 = vld1q_s8(wr3.add(off));

                acc0 = vdotq_s32(acc0, xv, wv0);
                acc1 = vdotq_s32(acc1, xv, wv1);
                acc2 = vdotq_s32(acc2, xv, wv2);
                acc3 = vdotq_s32(acc3, xv, wv3);
            }

            let sums = [
                vaddvq_s32(acc0),
                vaddvq_s32(acc1),
                vaddvq_s32(acc2),
                vaddvq_s32(acc3),
            ];

            for c in 0..NR {
                let j = j0 + c;
                let mut total = sums[c];
                if tail > 0 {
                    let off = blocks * BLK;
                    let wr = &w[j * k..(j + 1) * k];
                    for t in 0..tail {
                        total += xr[off + t] as i32 * wr[off + t] as i32;
                    }
                }
                let mut v = total as f32 * inv;
                if let Some(bb) = bias {
                    v += bb[j];
                }
                or[j] = if relu && v > 0.0 {
                    v
                } else if relu {
                    0.0
                } else {
                    v
                };
            }
        }

        for jt in 0..jtail {
            let j = groups * NR + jt;
            let wr_ptr = w.as_ptr().add(j * k);
            let mut acc = vdupq_n_s32(0);
            for b in 0..blocks {
                let off = b * BLK;
                let xv = vld1q_s8(x_ptr.add(off));
                let wv = vld1q_s8(wr_ptr.add(off));
                acc = vdotq_s32(acc, xv, wv);
            }
            let mut total = vaddvq_s32(acc);
            if tail > 0 {
                let off = blocks * BLK;
                let wr = &w[j * k..(j + 1) * k];
                for t in 0..tail {
                    total += xr[off + t] as i32 * wr[off + t] as i32;
                }
            }
            let mut v = total as f32 * inv;
            if let Some(bb) = bias {
                v += bb[j];
            }
            or[j] = if relu && v > 0.0 {
                v
            } else if relu {
                0.0
            } else {
                v
            };
        }
    }
}

/// `[1, K] @ [K, N]` i8 row kernel with fused bias + dequantization (+
/// optional relu), using SDOT.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_row_i8_dotprod(
    x: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    _row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    matmul_i8_dotprod(x, w, 1, k, n, bias, _row_sums, out, denom, relu)
}
