//! AVX-VNNI accelerated SIMD kernels using vpdpbusd instructions.

#![cfg(target_arch = "x86_64")]

use super::avx2::{hsum_4x_i32_avx2, hsum_8x_i32_avx2, hsum_i32_avx2};
use super::compute_row_sums;
#[allow(unused_imports)]
use std::arch::x86_64::*;
#[allow(unused_imports)]
use std::borrow::Cow;

/// AVX-VNNI i8 single-row kernel — drop-in counterpart of
/// [`matmul_row_i8_avx2`] with the per-block `maddubs`/`madd` widening chain
/// replaced by one `vpdpbusd` per column: `(u8)(x + 128) · (i8)w` pairs are
/// accumulated exactly in i32 lanes, so after subtracting the same per-column
/// `128·Σw` correction the result is bit-identical to the scalar reference.
/// `row_sums` carries `Σw` precomputed at load time (see
/// [`Quant8Tensor::row_sums`]); `None` falls back to an in-kernel scan.
/// The `vpdpbusd` VEX encoding is emitted via `asm!` because the Rust
/// intrinsic is gated behind `avx512vnni`, which also selects EVEX
/// encodings unavailable at runtime on AVX-VNNI-only CPUs.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline]
pub unsafe fn matmul_row_i8_avxvnni_prebiased(
    xb_buf: &[std::arch::x86_64::__m256i],
    x: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    if crate::utils::arch::vnni_uses_evex() {
        matmul_row_i8_avxvnni_prebiased_impl::<true>(xb_buf, x, w, k, n, bias, row_sums, out, denom, relu);
    } else {
        matmul_row_i8_avxvnni_prebiased_impl::<false>(xb_buf, x, w, k, n, bias, row_sums, out, denom, relu);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline]
unsafe fn matmul_row_i8_avxvnni_prebiased_impl<const EVEX: bool>(
    xb_buf: &[std::arch::x86_64::__m256i],
    x: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    use std::arch::asm;
    use std::arch::x86_64::*;

    macro_rules! dpbusd {
        ($acc:expr, $a:expr, $b:expr) => {{
            let acc = $acc;
            let a = $a;
            let b = $b;
            let out;
            if EVEX {
                asm!(
                    "{{evex}} vpdpbusd {acc}, {a}, {b}",
                    acc = inout(ymm_reg) acc => out,
                    a = in(ymm_reg) a,
                    b = in(ymm_reg) b,
                    options(pure, nomem, nostack)
                );
            } else {
                asm!(
                    "{{vex}} vpdpbusd {acc}, {a}, {b}",
                    acc = inout(ymm_reg) acc => out,
                    a = in(ymm_reg) a,
                    b = in(ymm_reg) b,
                    options(pure, nomem, nostack)
                );
            }
            out
        }};
    }
    const BLK: usize = 32;
    const NR: usize = 8;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    let inv_denom = denom.recip();
    let inv_denom_vec = _mm256_set1_ps(inv_denom);
    let zero_vec = _mm256_setzero_ps();
    let wsums: std::borrow::Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => std::borrow::Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };

    let b_pairs = blocks / 2;
    let b_rem = blocks % 2;

    let w_ptr = w.as_ptr();
    for g in 0..groups {
        let j0 = g * NR;
        let mut acc = [_mm256_setzero_si256(); NR];
        let p0 = w_ptr.add(j0 * k);
        let p1 = p0.add(k);
        let p2 = p1.add(k);
        let p3 = p2.add(k);
        let p4 = p3.add(k);
        let p5 = p4.add(k);
        let p6 = p5.add(k);
        let p7 = p6.add(k);

        macro_rules! step_8c {
            ($xb:expr, $off:expr) => {
                let wv0 = _mm256_loadu_si256(p0.add($off) as *const __m256i);
                acc[0] = dpbusd!(acc[0], $xb, wv0);
                let wv1 = _mm256_loadu_si256(p1.add($off) as *const __m256i);
                acc[1] = dpbusd!(acc[1], $xb, wv1);
                let wv2 = _mm256_loadu_si256(p2.add($off) as *const __m256i);
                acc[2] = dpbusd!(acc[2], $xb, wv2);
                let wv3 = _mm256_loadu_si256(p3.add($off) as *const __m256i);
                acc[3] = dpbusd!(acc[3], $xb, wv3);
                let wv4 = _mm256_loadu_si256(p4.add($off) as *const __m256i);
                acc[4] = dpbusd!(acc[4], $xb, wv4);
                let wv5 = _mm256_loadu_si256(p5.add($off) as *const __m256i);
                acc[5] = dpbusd!(acc[5], $xb, wv5);
                let wv6 = _mm256_loadu_si256(p6.add($off) as *const __m256i);
                acc[6] = dpbusd!(acc[6], $xb, wv6);
                let wv7 = _mm256_loadu_si256(p7.add($off) as *const __m256i);
                acc[7] = dpbusd!(acc[7], $xb, wv7);
            };
        }

        for bp in 0..b_pairs {
            let b0 = bp * 2;
            let b1 = b0 + 1;
            let off0 = b0 * BLK;
            let off1 = b1 * BLK;
            let xb0 = xb_buf[b0];
            let xb1 = xb_buf[b1];

            step_8c!(xb0, off0);
            step_8c!(xb1, off1);
        }

        if b_rem > 0 {
            let b = b_pairs * 2;
            let off = b * BLK;
            let xb = xb_buf[b];
            step_8c!(xb, off);
        }

        if tail == 0 {
            let raw_totals = hsum_8x_i32_avx2(&acc);
            let wsum_v = _mm256_loadu_si256(wsums.as_ptr().add(j0) as *const __m256i);
            let total_v = _mm256_sub_epi32(raw_totals, wsum_v);
            let mut vf = _mm256_cvtepi32_ps(total_v);
            vf = _mm256_mul_ps(vf, inv_denom_vec);
            if let Some(bb) = bias {
                let bias_v = _mm256_loadu_ps(bb.as_ptr().add(j0));
                vf = _mm256_add_ps(vf, bias_v);
            }
            if relu {
                vf = _mm256_max_ps(vf, zero_vec);
            }
            _mm256_storeu_ps(out.as_mut_ptr().add(j0), vf);
        } else {
            let w_ptrs = [p0, p1, p2, p3, p4, p5, p6, p7];
            for c in 0..NR {
                let j = j0 + c;
                let mut total: i32 = hsum_i32_avx2(acc[c]);
                let off = blocks * BLK;
                let wr = w_ptrs[c];
                let mut tail_wsum: i32 = 0;
                for t in 0..tail {
                    let w_val = *wr.add(off + t) as i32;
                    tail_wsum += w_val;
                    total += x[off + t] as i32 * w_val;
                }
                total -= wsums[j] - tail_wsum * 128;

                let mut v = total as f32 * inv_denom;
                if let Some(bb) = bias {
                    v += bb[j];
                }
                out[j] = if relu && v > 0.0 {
                    v
                } else if relu {
                    0.0
                } else {
                    v
                };
            }
        }
    }

    for jt in 0..jtail {
        let j = groups * NR + jt;
        let wr = &w[j * k..(j + 1) * k];
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        for bp in 0..b_pairs {
            let b0 = bp * 2;
            let b1 = b0 + 1;
            let off0 = b0 * BLK;
            let off1 = b1 * BLK;
            let wv0 = _mm256_loadu_si256(wr.as_ptr().add(off0) as *const __m256i);
            let wv1 = _mm256_loadu_si256(wr.as_ptr().add(off1) as *const __m256i);
            acc0 = dpbusd!(acc0, xb_buf[b0], wv0);
            acc1 = dpbusd!(acc1, xb_buf[b1], wv1);
        }
        let mut acc = _mm256_add_epi32(acc0, acc1);
        if b_rem > 0 {
            let b = b_pairs * 2;
            let off = b * BLK;
            let wv = _mm256_loadu_si256(wr.as_ptr().add(off) as *const __m256i);
            acc = dpbusd!(acc, xb_buf[b], wv);
        }
        let mut total: i32 = hsum_i32_avx2(acc);
        let mut tail_wsum: i32 = 0;
        if tail > 0 {
            let off = blocks * BLK;
            for t in 0..tail {
                tail_wsum += wr[off + t] as i32;
                total += x[off + t] as i32 * wr[off + t] as i32;
            }
        }
        total -= wsums[j] - tail_wsum * 128;

        let mut v = total as f32 * inv_denom;
        if let Some(bb) = bias {
            v += bb[j];
        }
        out[j] = if relu && v > 0.0 {
            v
        } else if relu {
            0.0
        } else {
            v
        };
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
#[inline]
pub unsafe fn matmul_row_i8_avxvnni(
    x: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out: &mut [f32],
    denom: f32,
    relu: bool,
) {
    use std::arch::x86_64::*;

    const BLK: usize = 32;
    let blocks = k / BLK;
    let x_bias_vec = _mm256_set1_epi8(0x80u8 as i8);
    let b_pairs = blocks / 2;
    let b_rem = blocks % 2;

    VNNI_XB_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < blocks {
            buf.resize(blocks, _mm256_setzero_si256());
        }
        for bp in 0..b_pairs {
            let b0 = bp * 2;
            let b1 = b0 + 1;
            let xv0 = _mm256_loadu_si256(x.as_ptr().add(b0 * BLK) as *const __m256i);
            let xv1 = _mm256_loadu_si256(x.as_ptr().add(b1 * BLK) as *const __m256i);
            buf[b0] = _mm256_add_epi8(xv0, x_bias_vec);
            buf[b1] = _mm256_add_epi8(xv1, x_bias_vec);
        }
        if b_rem > 0 {
            let b = b_pairs * 2;
            let xv = _mm256_loadu_si256(x.as_ptr().add(b * BLK) as *const __m256i);
            buf[b] = _mm256_add_epi8(xv, x_bias_vec);
        }
        let xb_buf = &buf[..blocks];
        matmul_row_i8_avxvnni_prebiased(xb_buf, x, w, k, n, bias, row_sums, out, denom, relu);
    });
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn matmul_2rows_i8_avxvnni(
    xb0_buf: &[std::arch::x86_64::__m256i],
    xb1_buf: &[std::arch::x86_64::__m256i],
    x0: &[i8],
    x1: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out0: &mut [f32],
    out1: &mut [f32],
    denom: f32,
    relu: bool,
) {
    if crate::utils::arch::vnni_uses_evex() {
        matmul_2rows_i8_avxvnni_impl::<true>(
            xb0_buf, xb1_buf, x0, x1, w, k, n, bias, row_sums, out0, out1, denom, relu,
        );
    } else {
        matmul_2rows_i8_avxvnni_impl::<false>(
            xb0_buf, xb1_buf, x0, x1, w, k, n, bias, row_sums, out0, out1, denom, relu,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn matmul_2rows_i8_avxvnni_impl<const EVEX: bool>(
    xb0_buf: &[std::arch::x86_64::__m256i],
    xb1_buf: &[std::arch::x86_64::__m256i],
    x0: &[i8],
    x1: &[i8],
    w: &[i8],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    row_sums: Option<&[i32]>,
    out0: &mut [f32],
    out1: &mut [f32],
    denom: f32,
    relu: bool,
) {
    use std::arch::asm;
    use std::arch::x86_64::*;

    macro_rules! dpbusd {
        ($acc:expr, $a:expr, $b:expr) => {{
            let acc = $acc;
            let a = $a;
            let b = $b;
            let out;
            if EVEX {
                asm!(
                    "{{evex}} vpdpbusd {acc}, {a}, {b}",
                    acc = inout(ymm_reg) acc => out,
                    a = in(ymm_reg) a,
                    b = in(ymm_reg) b,
                    options(pure, nomem, nostack)
                );
            } else {
                asm!(
                    "{{vex}} vpdpbusd {acc}, {a}, {b}",
                    acc = inout(ymm_reg) acc => out,
                    a = in(ymm_reg) a,
                    b = in(ymm_reg) b,
                    options(pure, nomem, nostack)
                );
            }
            out
        }};
    }

    const BLK: usize = 32;
    const NR: usize = 4;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    let inv_denom = denom.recip();
    let inv_denom_128 = _mm_set1_ps(inv_denom);
    let zero_128 = _mm_setzero_ps();
    let wsums: std::borrow::Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => std::borrow::Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };
    let w_ptr = w.as_ptr();
    for g in 0..groups {
        let j0 = g * NR;
        let mut acc0 = [_mm256_setzero_si256(); NR];
        let mut acc1 = [_mm256_setzero_si256(); NR];
        let p0 = w_ptr.add(j0 * k);
        let p1 = p0.add(k);
        let p2 = p1.add(k);
        let p3 = p2.add(k);

        for b in 0..blocks {
            let off = b * BLK;
            let xb0 = xb0_buf[b];
            let xb1 = xb1_buf[b];
            let wv0 = _mm256_loadu_si256(p0.add(off) as *const __m256i);
            acc0[0] = dpbusd!(acc0[0], xb0, wv0);
            acc1[0] = dpbusd!(acc1[0], xb1, wv0);
            let wv1 = _mm256_loadu_si256(p1.add(off) as *const __m256i);
            acc0[1] = dpbusd!(acc0[1], xb0, wv1);
            acc1[1] = dpbusd!(acc1[1], xb1, wv1);
            let wv2 = _mm256_loadu_si256(p2.add(off) as *const __m256i);
            acc0[2] = dpbusd!(acc0[2], xb0, wv2);
            acc1[2] = dpbusd!(acc1[2], xb1, wv2);
            let wv3 = _mm256_loadu_si256(p3.add(off) as *const __m256i);
            acc0[3] = dpbusd!(acc0[3], xb0, wv3);
            acc1[3] = dpbusd!(acc1[3], xb1, wv3);
        }

        if tail == 0 {
            let raw_totals0 = hsum_4x_i32_avx2(&acc0);
            let raw_totals1 = hsum_4x_i32_avx2(&acc1);
            let wsum_v = _mm_loadu_si128(wsums.as_ptr().add(j0) as *const __m128i);
            let total0_v = _mm_sub_epi32(raw_totals0, wsum_v);
            let total1_v = _mm_sub_epi32(raw_totals1, wsum_v);

            let mut vf0 = _mm_cvtepi32_ps(total0_v);
            let mut vf1 = _mm_cvtepi32_ps(total1_v);
            vf0 = _mm_mul_ps(vf0, inv_denom_128);
            vf1 = _mm_mul_ps(vf1, inv_denom_128);
            if let Some(bb) = bias {
                let bias_v = _mm_loadu_ps(bb.as_ptr().add(j0));
                vf0 = _mm_add_ps(vf0, bias_v);
                vf1 = _mm_add_ps(vf1, bias_v);
            }
            if relu {
                vf0 = _mm_max_ps(vf0, zero_128);
                vf1 = _mm_max_ps(vf1, zero_128);
            }
            _mm_storeu_ps(out0.as_mut_ptr().add(j0), vf0);
            _mm_storeu_ps(out1.as_mut_ptr().add(j0), vf1);
        } else {
            let w_ptrs = [p0, p1, p2, p3];
            for c in 0..NR {
                let j = j0 + c;
                let mut total0: i32 = hsum_i32_avx2(acc0[c]);
                let mut total1: i32 = hsum_i32_avx2(acc1[c]);

                let mut tail_wsum: i32 = 0;
                let off = blocks * BLK;
                let wr = w_ptrs[c];
                for t in 0..tail {
                    let w_val = *wr.add(off + t) as i32;
                    tail_wsum += w_val;
                    total0 += x0[off + t] as i32 * w_val;
                    total1 += x1[off + t] as i32 * w_val;
                }
                let correction = wsums[j] - tail_wsum * 128;
                total0 -= correction;
                total1 -= correction;

                let mut v0 = total0 as f32 * inv_denom;
                let mut v1 = total1 as f32 * inv_denom;
                if let Some(bb) = bias {
                    v0 += bb[j];
                    v1 += bb[j];
                }
                out0[j] = if relu && v0 > 0.0 {
                    v0
                } else if relu {
                    0.0
                } else {
                    v0
                };
                out1[j] = if relu && v1 > 0.0 {
                    v1
                } else if relu {
                    0.0
                } else {
                    v1
                };
            }
        }
    }

    for jt in 0..jtail {
        let j = groups * NR + jt;
        let wr = &w[j * k..(j + 1) * k];
        let mut acc0 = _mm256_setzero_si256();
        let mut acc1 = _mm256_setzero_si256();
        for b in 0..blocks {
            let off = b * BLK;
            let wv = _mm256_loadu_si256(wr.as_ptr().add(off) as *const __m256i);
            acc0 = dpbusd!(acc0, xb0_buf[b], wv);
            acc1 = dpbusd!(acc1, xb1_buf[b], wv);
        }
        let mut total0: i32 = hsum_i32_avx2(acc0);
        let mut total1: i32 = hsum_i32_avx2(acc1);
        let mut tail_wsum: i32 = 0;
        if tail > 0 {
            let off = blocks * BLK;
            for t in 0..tail {
                tail_wsum += wr[off + t] as i32;
                total0 += x0[off + t] as i32 * wr[off + t] as i32;
                total1 += x1[off + t] as i32 * wr[off + t] as i32;
            }
        }
        let correction = wsums[j] - tail_wsum * 128;
        total0 -= correction;
        total1 -= correction;

        let mut v0 = total0 as f32 * inv_denom;
        let mut v1 = total1 as f32 * inv_denom;
        if let Some(bb) = bias {
            v0 += bb[j];
            v1 += bb[j];
        }
        out0[j] = if relu && v0 > 0.0 {
            v0
        } else if relu {
            0.0
        } else {
            v0
        };
        out1[j] = if relu && v1 > 0.0 {
            v1
        } else if relu {
            0.0
        } else {
            v1
        };
    }
}
thread_local! {
    static VNNI_XB_BUF: std::cell::RefCell<Vec<std::arch::x86_64::__m256i>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_avxvnni(
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
    use std::arch::x86_64::*;

    const N_TILE: usize = 64;
    const BLK: usize = 32;
    let blocks = k / BLK;
    let fallback: std::borrow::Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => std::borrow::Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };
    let wsums: &[i32] = &fallback;

    let x_bias_vec = _mm256_set1_epi8(0x80u8 as i8);
    let total_blocks = m * blocks;
    VNNI_XB_BUF.with(|cell| {
        let mut buf = cell.borrow_mut();
        if buf.len() < total_blocks {
            buf.resize(total_blocks, _mm256_setzero_si256());
        }
        for row in 0..m {
            let xr = &x[row * k..(row + 1) * k];
            let row_dest = &mut buf[row * blocks..(row + 1) * blocks];
            for b in 0..blocks {
                let off = b * BLK;
                let xv = _mm256_loadu_si256(xr.as_ptr().add(off) as *const __m256i);
                row_dest[b] = _mm256_add_epi8(xv, x_bias_vec);
            }
        }
        let xb_slice = &buf[..total_blocks];

        for j_start in (0..n).step_by(N_TILE) {
            let j_len = (n - j_start).min(N_TILE);
            let w_tile = &w[j_start * k..(j_start + j_len) * k];
            let bias_tile = bias.map(|b| &b[j_start..j_start + j_len]);
            let rs_tile = Some(&wsums[j_start..j_start + j_len]);

            let mut i = 0;
            while i + 1 < m {
                let xb0 = &xb_slice[i * blocks..(i + 1) * blocks];
                let xb1 = &xb_slice[(i + 1) * blocks..(i + 2) * blocks];
                let xr0 = &x[i * k..(i + 1) * k];
                let xr1 = &x[(i + 1) * k..(i + 2) * k];
                let (top, bottom) = out.split_at_mut((i + 1) * n);
                let out_r0 = &mut top[i * n + j_start..i * n + j_start + j_len];
                let out_r1 = &mut bottom[j_start..j_start + j_len];
                matmul_2rows_i8_avxvnni(
                    xb0, xb1, xr0, xr1, w_tile, k, j_len, bias_tile, rs_tile, out_r0, out_r1,
                    denom, relu,
                );
                i += 2;
            }
            if i < m {
                let xb = &xb_slice[i * blocks..(i + 1) * blocks];
                let xr = &x[i * k..(i + 1) * k];
                let out_row = &mut out[i * n + j_start..i * n + j_start + j_len];
                matmul_row_i8_avxvnni_prebiased(
                    xb, xr, w_tile, k, j_len, bias_tile, rs_tile, out_row, denom, relu,
                );
            }
        }
    });
}
