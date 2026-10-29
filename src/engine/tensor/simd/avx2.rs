//! AVX2, AVX, and FMA accelerated SIMD kernels.

#![cfg(target_arch = "x86_64")]

use super::compute_row_sums;
#[allow(unused_imports)]
use std::arch::x86_64::*;
#[allow(unused_imports)]
use std::borrow::Cow;

/// AVX2 i8 single-row kernel.
///
/// Follows the upstream intgemm row kernel's structure: the outer loop walks
/// the activation row in 32-byte blocks — loading and biasing each activation
/// block exactly once — and the inner loop keeps eight independent per-column
/// i32 accumulators live, so one activation load feeds eight widening chains.
///
/// Each column accumulates `Σ(x+128)·w` exactly: the biased byte is split
/// into nibbles (`xb = 16·hi + lo`, both in `[0, 15]`), two `maddubs` passes
/// keep every i16 pairwise sum ≤ 2·15·127 = 3810 — far from saturation — and
/// the recombination `16·Σhi·w + Σlo·w` is exact i32 integer arithmetic, so
/// full-range weights (whose direct `maddubs` pair sums reach ±64770 and
/// would clip in i16) stay bit-identical to the scalar reference. The
/// per-column `128·Σw` correction is subtracted once from the final
/// horizontal sum, matching [`matmul_row_i8_avxvnni`]; columns below the
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_i32_avx2(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extracti128_si256(v, 1);
    let lo = _mm256_castsi256_si128(v);
    let sum128 = _mm_add_epi32(lo, hi);
    let hi64 = _mm_shuffle_epi32(sum128, 0x4E);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0xB1);
    let sum32 = _mm_add_epi32(sum64, hi32);
    _mm_cvtsi128_si32(sum32)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_8x_i32_avx2(
    acc: &[std::arch::x86_64::__m256i; 8],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    // Step 1: Pairwise 32-bit unpacks within each 128-bit lane
    let t0 = _mm256_unpacklo_epi32(acc[0], acc[1]);
    let t1 = _mm256_unpackhi_epi32(acc[0], acc[1]);
    let v01 = _mm256_add_epi32(t0, t1);

    let t2 = _mm256_unpacklo_epi32(acc[2], acc[3]);
    let t3 = _mm256_unpackhi_epi32(acc[2], acc[3]);
    let v23 = _mm256_add_epi32(t2, t3);

    let t4 = _mm256_unpacklo_epi32(acc[4], acc[5]);
    let t5 = _mm256_unpackhi_epi32(acc[4], acc[5]);
    let v45 = _mm256_add_epi32(t4, t5);

    let t6 = _mm256_unpacklo_epi32(acc[6], acc[7]);
    let t7 = _mm256_unpackhi_epi32(acc[6], acc[7]);
    let v67 = _mm256_add_epi32(t6, t7);

    // Step 2: 64-bit unpacks within each 128-bit lane
    let w_lo_0123 = _mm256_unpacklo_epi64(v01, v23);
    let w_hi_0123 = _mm256_unpackhi_epi64(v01, v23);
    let sum0123 = _mm256_add_epi32(w_lo_0123, w_hi_0123);

    let w_lo_4567 = _mm256_unpacklo_epi64(v45, v67);
    let w_hi_4567 = _mm256_unpackhi_epi64(v45, v67);
    let sum4567 = _mm256_add_epi32(w_lo_4567, w_hi_4567);

    // Step 3: Cross-lane permutation to assemble low 128-bit halves (lane 0) and high 128-bit halves (lane 1)
    let sum_lo = _mm256_permute2x128_si256(sum0123, sum4567, 0x20);
    let sum_hi = _mm256_permute2x128_si256(sum0123, sum4567, 0x31);

    _mm256_add_epi32(sum_lo, sum_hi)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn hsum_4x_i32_avx2(
    acc: &[std::arch::x86_64::__m256i; 4],
) -> std::arch::x86_64::__m128i {
    use std::arch::x86_64::*;
    let t0 = _mm256_unpacklo_epi32(acc[0], acc[1]);
    let t1 = _mm256_unpackhi_epi32(acc[0], acc[1]);
    let v01 = _mm256_add_epi32(t0, t1);

    let t2 = _mm256_unpacklo_epi32(acc[2], acc[3]);
    let t3 = _mm256_unpackhi_epi32(acc[2], acc[3]);
    let v23 = _mm256_add_epi32(t2, t3);

    let w_lo = _mm256_unpacklo_epi64(v01, v23);
    let w_hi = _mm256_unpackhi_epi64(v01, v23);
    let sum0123 = _mm256_add_epi32(w_lo, w_hi);

    let lo128 = _mm256_castsi256_si128(sum0123);
    let hi128 = _mm256_extracti128_si256(sum0123, 1);
    _mm_add_epi32(lo128, hi128)
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn matmul_row_i8_avx2_prebiased(
    lo_buf: &[std::arch::x86_64::__m256i],
    hi_buf: &[std::arch::x86_64::__m256i],
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
    const NR: usize = 8;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    let inv_denom = denom.recip();
    let inv_denom_vec = _mm256_set1_ps(inv_denom);
    let zero_vec = _mm256_setzero_ps();
    let ones = _mm256_set1_epi16(1);
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

        for bp in 0..b_pairs {
            let b0 = bp * 2;
            let b1 = b0 + 1;
            let off0 = b0 * BLK;
            let off1 = b1 * BLK;
            let lo8_0 = lo_buf[b0];
            let hi8_0 = hi_buf[b0];
            let lo8_1 = lo_buf[b1];
            let hi8_1 = hi_buf[b1];

            macro_rules! step_c_2x {
                ($c:expr, $p:expr) => {
                    let wv0 = _mm256_loadu_si256($p.add(off0) as *const __m256i);
                    let wv1 = _mm256_loadu_si256($p.add(off1) as *const __m256i);
                    let lo_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_0, wv0), ones);
                    let hi_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_0, wv0), ones);
                    let t32_0 = _mm256_add_epi32(lo_part0, _mm256_slli_epi32::<4>(hi_part0));
                    let lo_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_1, wv1), ones);
                    let hi_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_1, wv1), ones);
                    let t32_1 = _mm256_add_epi32(lo_part1, _mm256_slli_epi32::<4>(hi_part1));
                    let sum = _mm256_add_epi32(t32_0, t32_1);
                    acc[$c] = _mm256_add_epi32(acc[$c], sum);
                };
            }
            step_c_2x!(0, p0);
            step_c_2x!(1, p1);
            step_c_2x!(2, p2);
            step_c_2x!(3, p3);
            step_c_2x!(4, p4);
            step_c_2x!(5, p5);
            step_c_2x!(6, p6);
            step_c_2x!(7, p7);
        }

        if b_rem > 0 {
            let b = b_pairs * 2;
            let off = b * BLK;
            let lo8 = lo_buf[b];
            let hi8 = hi_buf[b];
            macro_rules! step_c {
                ($c:expr, $p:expr) => {
                    let wv = _mm256_loadu_si256($p.add(off) as *const __m256i);
                    let lo_part = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8, wv), ones);
                    let hi_part = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8, wv), ones);
                    let t32 = _mm256_add_epi32(lo_part, _mm256_slli_epi32::<4>(hi_part));
                    acc[$c] = _mm256_add_epi32(acc[$c], t32);
                };
            }
            step_c!(0, p0);
            step_c!(1, p1);
            step_c!(2, p2);
            step_c!(3, p3);
            step_c!(4, p4);
            step_c!(5, p5);
            step_c!(6, p6);
            step_c!(7, p7);
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
        let mut acc = _mm256_setzero_si256();
        for bp in 0..b_pairs {
            let b0 = bp * 2;
            let b1 = b0 + 1;
            let off0 = b0 * BLK;
            let off1 = b1 * BLK;
            let wv0 = _mm256_loadu_si256(wr.as_ptr().add(off0) as *const __m256i);
            let wv1 = _mm256_loadu_si256(wr.as_ptr().add(off1) as *const __m256i);
            let lo_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo_buf[b0], wv0), ones);
            let hi_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi_buf[b0], wv0), ones);
            let t32_0 = _mm256_add_epi32(lo_part0, _mm256_slli_epi32::<4>(hi_part0));
            let lo_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo_buf[b1], wv1), ones);
            let hi_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi_buf[b1], wv1), ones);
            let t32_1 = _mm256_add_epi32(lo_part1, _mm256_slli_epi32::<4>(hi_part1));
            let sum = _mm256_add_epi32(t32_0, t32_1);
            acc = _mm256_add_epi32(acc, sum);
        }
        if b_rem > 0 {
            let b = b_pairs * 2;
            let off = b * BLK;
            let wv = _mm256_loadu_si256(wr.as_ptr().add(off) as *const __m256i);
            let lo_part = _mm256_madd_epi16(_mm256_maddubs_epi16(lo_buf[b], wv), ones);
            let hi_part = _mm256_madd_epi16(_mm256_maddubs_epi16(hi_buf[b], wv), ones);
            let t32 = _mm256_add_epi32(lo_part, _mm256_slli_epi32::<4>(hi_part));
            acc = _mm256_add_epi32(acc, t32);
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
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn matmul_row_i8_avx2(
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
    let nibble_mask = _mm256_set1_epi8(0x0F);

    let mut lo_stack = [std::mem::MaybeUninit::<__m256i>::uninit(); 128];
    let mut hi_stack = [std::mem::MaybeUninit::<__m256i>::uninit(); 128];
    let (lo_heap, hi_heap);
    let (lo_buf, hi_buf): (&[__m256i], &[__m256i]) = if blocks <= 128 {
        for b in 0..blocks {
            let off = b * BLK;
            let xv = _mm256_loadu_si256(x.as_ptr().add(off) as *const __m256i);
            let xb = _mm256_add_epi8(xv, x_bias_vec);
            lo_stack[b].write(_mm256_and_si256(xb, nibble_mask));
            hi_stack[b].write(_mm256_and_si256(_mm256_srli_epi16::<4>(xb), nibble_mask));
        }
        let lo_slice = std::slice::from_raw_parts(lo_stack.as_ptr() as *const __m256i, blocks);
        let hi_slice = std::slice::from_raw_parts(hi_stack.as_ptr() as *const __m256i, blocks);
        (lo_slice, hi_slice)
    } else {
        let mut lo_v = Vec::with_capacity(blocks);
        let mut hi_v = Vec::with_capacity(blocks);
        for b in 0..blocks {
            let off = b * BLK;
            let xv = _mm256_loadu_si256(x.as_ptr().add(off) as *const __m256i);
            let xb = _mm256_add_epi8(xv, x_bias_vec);
            lo_v.push(_mm256_and_si256(xb, nibble_mask));
            hi_v.push(_mm256_and_si256(_mm256_srli_epi16::<4>(xb), nibble_mask));
        }
        lo_heap = lo_v;
        hi_heap = hi_v;
        (&lo_heap[..], &hi_heap[..])
    };

    matmul_row_i8_avx2_prebiased(lo_buf, hi_buf, x, w, k, n, bias, row_sums, out, denom, relu);
}
/// AVX (no FMA) single-row f32 matmul.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_row_avx(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::x86_64::{
        __m256, _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };
    const NR: usize = 8;
    const MAX_STACK_GROUPS: usize = 256;
    let groups = n / NR;
    let tail = n % NR;
    let mut stack_acc = [_mm256_setzero_ps(); MAX_STACK_GROUPS];
    let mut heap_acc: Vec<__m256>;
    let acc_slice: &mut [__m256] = if groups <= MAX_STACK_GROUPS {
        &mut stack_acc[..groups]
    } else {
        heap_acc = vec![_mm256_setzero_ps(); groups];
        heap_acc.as_mut_slice()
    };
    for t in 0..k {
        let av = _mm256_set1_ps(a[t]);
        for (g, ac) in acc_slice.iter_mut().enumerate() {
            let bv = _mm256_loadu_ps(b[t * n + g * NR..].as_ptr());
            *ac = _mm256_add_ps(_mm256_mul_ps(av, bv), *ac);
        }
    }
    for (g, &ac) in acc_slice.iter().enumerate() {
        let mut v = ac;
        if let Some(bb) = bias {
            v = _mm256_add_ps(v, _mm256_loadu_ps(bb[g * NR..].as_ptr()));
        }
        if relu {
            v = _mm256_max_ps(v, _mm256_setzero_ps());
        }
        _mm256_storeu_ps(out[g * NR..].as_mut_ptr(), v);
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
/// AVX2+FMA single-row f32 matmul. `relu` optionally applies
/// `max(0, x)` to every output element in the store step (the FFN's fused
/// activation epilogue); the vector path uses `maxps(v, 0)` and the scalar
/// tail uses the same `> 0.0` predicate as the standalone relu helper, so a
/// `-0.0` product maps to `+0.0` in both.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_row_avx2(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let zero = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();
    let o_ptr = out.as_mut_ptr();
    let bias_ptr = bias.map(|bb| bb.as_ptr());

    let k_pairs = k / 2;
    let k_rem = k % 2;

    let n_blocks = n / 32;
    let n_rem = n % 32;

    // Outer loop traverses output columns N with stride 32 (4 YMM accumulators c0..c3 resident in registers)
    for blk in 0..n_blocks {
        let col = blk * 32;
        let mut c0 = zero;
        let mut c1 = zero;
        let mut c2 = zero;
        let mut c3 = zero;

        for t in 0..k_pairs {
            let t0 = t * 2;
            let t1 = t0 + 1;
            let av0 = _mm256_set1_ps(*a_ptr.add(t0));
            let av1 = _mm256_set1_ps(*a_ptr.add(t1));
            let bp0 = b_ptr.add(t0 * n + col);
            let bp1 = b_ptr.add(t1 * n + col);

            c0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0), c0);
            c1 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0.add(8)), c1);
            c2 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0.add(16)), c2);
            c3 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0.add(24)), c3);

            c0 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1), c0);
            c1 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1.add(8)), c1);
            c2 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1.add(16)), c2);
            c3 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1.add(24)), c3);
        }

        if k_rem > 0 {
            let t = k_pairs * 2;
            let av = _mm256_set1_ps(*a_ptr.add(t));
            let bp = b_ptr.add(t * n + col);
            c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
            c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
            c2 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(16)), c2);
            c3 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(24)), c3);
        }

        if let Some(bp) = bias_ptr {
            c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
            c1 = _mm256_add_ps(c1, _mm256_loadu_ps(bp.add(col + 8)));
            c2 = _mm256_add_ps(c2, _mm256_loadu_ps(bp.add(col + 16)));
            c3 = _mm256_add_ps(c3, _mm256_loadu_ps(bp.add(col + 24)));
        }

        if relu {
            c0 = _mm256_max_ps(c0, zero);
            c1 = _mm256_max_ps(c1, zero);
            c2 = _mm256_max_ps(c2, zero);
            c3 = _mm256_max_ps(c3, zero);
        }

        _mm256_storeu_ps(o_ptr.add(col), c0);
        _mm256_storeu_ps(o_ptr.add(col + 8), c1);
        _mm256_storeu_ps(o_ptr.add(col + 16), c2);
        _mm256_storeu_ps(o_ptr.add(col + 24), c3);
    }

    let mut cur_col = n_blocks * 32;

    // Remainder 16: 2 YMM accumulators
    if n_rem >= 16 {
        let col = cur_col;
        let mut c0 = zero;
        let mut c1 = zero;

        for t in 0..k_pairs {
            let t0 = t * 2;
            let t1 = t0 + 1;
            let av0 = _mm256_set1_ps(*a_ptr.add(t0));
            let av1 = _mm256_set1_ps(*a_ptr.add(t1));
            let bp0 = b_ptr.add(t0 * n + col);
            let bp1 = b_ptr.add(t1 * n + col);

            c0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0), c0);
            c1 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0.add(8)), c1);

            c0 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1), c0);
            c1 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1.add(8)), c1);
        }

        if k_rem > 0 {
            let t = k_pairs * 2;
            let av = _mm256_set1_ps(*a_ptr.add(t));
            let bp = b_ptr.add(t * n + col);
            c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
            c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
        }

        if let Some(bp) = bias_ptr {
            c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
            c1 = _mm256_add_ps(c1, _mm256_loadu_ps(bp.add(col + 8)));
        }

        if relu {
            c0 = _mm256_max_ps(c0, zero);
            c1 = _mm256_max_ps(c1, zero);
        }

        _mm256_storeu_ps(o_ptr.add(col), c0);
        _mm256_storeu_ps(o_ptr.add(col + 8), c1);

        cur_col += 16;
    }

    // Remainder 8: 1 YMM accumulator
    if (n - cur_col) >= 8 {
        let col = cur_col;
        let mut c0 = zero;

        for t in 0..k_pairs {
            let t0 = t * 2;
            let t1 = t0 + 1;
            let av0 = _mm256_set1_ps(*a_ptr.add(t0));
            let av1 = _mm256_set1_ps(*a_ptr.add(t1));
            let bp0 = b_ptr.add(t0 * n + col);
            let bp1 = b_ptr.add(t1 * n + col);

            c0 = _mm256_fmadd_ps(av0, _mm256_loadu_ps(bp0), c0);
            c0 = _mm256_fmadd_ps(av1, _mm256_loadu_ps(bp1), c0);
        }

        if k_rem > 0 {
            let t = k_pairs * 2;
            let av = _mm256_set1_ps(*a_ptr.add(t));
            let bp = b_ptr.add(t * n + col);
            c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
        }

        if let Some(bp) = bias_ptr {
            c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
        }

        if relu {
            c0 = _mm256_max_ps(c0, zero);
        }

        _mm256_storeu_ps(o_ptr.add(col), c0);

        cur_col += 8;
    }

    // Scalar tail
    let tail = n - cur_col;
    if tail > 0 {
        for j in 0..tail {
            let col = cur_col + j;
            let mut s = 0.0f32;
            for t in 0..k {
                s += *a_ptr.add(t) * *b_ptr.add(t * n + col);
            }
            let s = if let Some(bp) = bias_ptr {
                s + *bp.add(col)
            } else {
                s
            };
            *o_ptr.add(col) = if relu && s > 0.0 {
                s
            } else if relu {
                0.0
            } else {
                s
            };
        }
    }
}
/// AVX (no FMA) `M×K @ K×N` f32 kernel.
///
/// Identical structure to the AVX2+FMA kernel but uses separable mul + add
/// (`_mm256_mul_ps` / `_mm256_add_ps`) so it runs on CPUs that have AVX
/// without FMA (e.g. Sandy/Ivy Bridge, early AMD Bulldozer). A scalar tail
/// handles `n % 8`.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx")]
pub unsafe fn matmul_avx(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_mul_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let zero = _mm256_setzero_ps();
    let b_ptr = b.as_ptr();
    let bias_ptr = bias.map(|bb| bb.as_ptr());

    if m == 1 {
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        let n_blocks = n / 32;
        let n_rem = n % 32;

        for blk in 0..n_blocks {
            let col = blk * 32;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;

            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                let bp = b_ptr.add(t * n + col);
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(av, _mm256_loadu_ps(bp)));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(8))));
                c2 = _mm256_add_ps(c2, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(16))));
                c3 = _mm256_add_ps(c3, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(24))));
            }

            if let Some(bp) = bias_ptr {
                c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
                c1 = _mm256_add_ps(c1, _mm256_loadu_ps(bp.add(col + 8)));
                c2 = _mm256_add_ps(c2, _mm256_loadu_ps(bp.add(col + 16)));
                c3 = _mm256_add_ps(c3, _mm256_loadu_ps(bp.add(col + 24)));
            }

            if relu {
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
                c2 = _mm256_max_ps(c2, zero);
                c3 = _mm256_max_ps(c3, zero);
            }

            _mm256_storeu_ps(o_ptr.add(col), c0);
            _mm256_storeu_ps(o_ptr.add(col + 8), c1);
            _mm256_storeu_ps(o_ptr.add(col + 16), c2);
            _mm256_storeu_ps(o_ptr.add(col + 24), c3);
        }

        let rem_col = n_blocks * 32;
        let rem_groups = n_rem / 8;
        for g in 0..rem_groups {
            let col = rem_col + g * 8;
            let mut c = zero;
            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                c = _mm256_add_ps(
                    c,
                    _mm256_mul_ps(av, _mm256_loadu_ps(b_ptr.add(t * n + col))),
                );
            }
            if let Some(bp) = bias_ptr {
                c = _mm256_add_ps(c, _mm256_loadu_ps(bp.add(col)));
            }
            if relu {
                c = _mm256_max_ps(c, zero);
            }
            _mm256_storeu_ps(o_ptr.add(col), c);
        }

        let tail_col = rem_col + rem_groups * 8;
        let tail = n_rem % 8;
        if tail > 0 {
            for j in 0..tail {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += *a_ptr.add(t) * *b_ptr.add(t * n + tail_col + j);
                }
                let s = if let Some(bp) = bias_ptr {
                    s + *bp.add(tail_col + j)
                } else {
                    s
                };
                *o_ptr.add(tail_col + j) = if relu && s > 0.0 {
                    s
                } else if relu {
                    0.0
                } else {
                    s
                };
            }
        }
        return;
    }

    let m_pairs = m / 2;
    let m_rem = m % 2;

    let n_blocks = n / 16;
    let n_rem = n % 16;
    let n_8 = n_rem / 8;
    let tail = n_rem % 8;

    for mi in 0..m_pairs {
        let i0 = mi * 2;
        let i1 = i0 + 1;
        let a0_ptr = a.as_ptr().add(i0 * k);
        let a1_ptr = a.as_ptr().add(i1 * k);
        let o0_ptr = out.as_mut_ptr().add(i0 * n);
        let o1_ptr = out.as_mut_ptr().add(i1 * n);

        for blk in 0..n_blocks {
            let col = blk * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c10 = zero;
            let mut c11 = zero;

            for t in 0..k {
                let av0 = _mm256_set1_ps(*a0_ptr.add(t));
                let av1 = _mm256_set1_ps(*a1_ptr.add(t));

                let bp = b_ptr.add(t * n + col);
                let bv0 = _mm256_loadu_ps(bp);
                let bv1 = _mm256_loadu_ps(bp.add(8));

                c00 = _mm256_add_ps(c00, _mm256_mul_ps(av0, bv0));
                c01 = _mm256_add_ps(c01, _mm256_mul_ps(av0, bv1));
                c10 = _mm256_add_ps(c10, _mm256_mul_ps(av1, bv0));
                c11 = _mm256_add_ps(c11, _mm256_mul_ps(av1, bv1));
            }

            if let Some(bp) = bias_ptr {
                let b0 = _mm256_loadu_ps(bp.add(col));
                let b1 = _mm256_loadu_ps(bp.add(col + 8));
                c00 = _mm256_add_ps(c00, b0);
                c01 = _mm256_add_ps(c01, b1);
                c10 = _mm256_add_ps(c10, b0);
                c11 = _mm256_add_ps(c11, b1);
            }

            if relu {
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
            }

            _mm256_storeu_ps(o0_ptr.add(col), c00);
            _mm256_storeu_ps(o0_ptr.add(col + 8), c01);
            _mm256_storeu_ps(o1_ptr.add(col), c10);
            _mm256_storeu_ps(o1_ptr.add(col + 8), c11);
        }

        if n_8 > 0 {
            let col = n_blocks * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            for t in 0..k {
                let av0 = _mm256_set1_ps(*a0_ptr.add(t));
                let av1 = _mm256_set1_ps(*a1_ptr.add(t));
                let bv = _mm256_loadu_ps(b_ptr.add(t * n + col));
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(av0, bv));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(av1, bv));
            }
            if let Some(bp) = bias_ptr {
                let b = _mm256_loadu_ps(bp.add(col));
                c0 = _mm256_add_ps(c0, b);
                c1 = _mm256_add_ps(c1, b);
            }
            if relu {
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
            }
            _mm256_storeu_ps(o0_ptr.add(col), c0);
            _mm256_storeu_ps(o1_ptr.add(col), c1);
        }

        if tail > 0 {
            let tail_col = n_blocks * 16 + n_8 * 8;
            for j in 0..tail {
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                for t in 0..k {
                    let bv = *b_ptr.add(t * n + tail_col + j);
                    s0 += *a0_ptr.add(t) * bv;
                    s1 += *a1_ptr.add(t) * bv;
                }
                if let Some(bp) = bias_ptr {
                    s0 += *bp.add(tail_col + j);
                    s1 += *bp.add(tail_col + j);
                }
                *o0_ptr.add(tail_col + j) = if relu && s0 > 0.0 {
                    s0
                } else if relu {
                    0.0
                } else {
                    s0
                };
                *o1_ptr.add(tail_col + j) = if relu && s1 > 0.0 {
                    s1
                } else if relu {
                    0.0
                } else {
                    s1
                };
            }
        }
    }

    if m_rem > 0 {
        let i = m - 1;
        let a_ptr = a.as_ptr().add(i * k);
        let o_ptr = out.as_mut_ptr().add(i * n);

        for blk in 0..n_blocks {
            let col = blk * 16;
            let mut c0 = zero;
            let mut c1 = zero;

            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                let bp = b_ptr.add(t * n + col);
                c0 = _mm256_add_ps(c0, _mm256_mul_ps(av, _mm256_loadu_ps(bp)));
                c1 = _mm256_add_ps(c1, _mm256_mul_ps(av, _mm256_loadu_ps(bp.add(8))));
            }

            if let Some(bp) = bias_ptr {
                c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
                c1 = _mm256_add_ps(c1, _mm256_loadu_ps(bp.add(col + 8)));
            }

            if relu {
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
            }

            _mm256_storeu_ps(o_ptr.add(col), c0);
            _mm256_storeu_ps(o_ptr.add(col + 8), c1);
        }

        if n_8 > 0 {
            let col = n_blocks * 16;
            let mut c = zero;
            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                c = _mm256_add_ps(
                    c,
                    _mm256_mul_ps(av, _mm256_loadu_ps(b_ptr.add(t * n + col))),
                );
            }
            if let Some(bp) = bias_ptr {
                c = _mm256_add_ps(c, _mm256_loadu_ps(bp.add(col)));
            }
            if relu {
                c = _mm256_max_ps(c, zero);
            }
            _mm256_storeu_ps(o_ptr.add(col), c);
        }

        if tail > 0 {
            let tail_col = n_blocks * 16 + n_8 * 8;
            for j in 0..tail {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += *a_ptr.add(t) * *b_ptr.add(t * n + tail_col + j);
                }
                let s = if let Some(bp) = bias_ptr {
                    s + *bp.add(tail_col + j)
                } else {
                    s
                };
                *o_ptr.add(tail_col + j) = if relu && s > 0.0 {
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
/// AVX2+FMA `M×K @ K×N` kernel.
///
/// N-blocked as described at the header: one `__m256` accumulator per 8-column
/// tile per output row, the whole K dot product accumulated in registers with
/// fused multiply–add, then the tile stored; the `[k, 8]` B-panel stays in L1
/// and is reused across the M loop. A scalar tail handles `n % 8`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn matmul_avx2_fma(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_max_ps, _mm256_set1_ps,
        _mm256_setzero_ps, _mm256_storeu_ps,
    };

    let zero = _mm256_setzero_ps();
    let b_ptr = b.as_ptr();
    let bias_ptr = bias.map(|bb| bb.as_ptr());

    if m == 1 {
        matmul_row_avx2(a, b, k, n, bias, out, relu);
        return;
    }

    // M >= 2: 2x2 M-N register tiled GEMM (2 rows of A reuse 1 row chunk of B).
    let m_pairs = m / 2;
    let m_rem = m % 2;

    let n_blocks = n / 16;
    let n_rem = n % 16;
    let n_8 = n_rem / 8;
    let tail = n_rem % 8;

    for mi in 0..m_pairs {
        let i0 = mi * 2;
        let i1 = i0 + 1;
        let a0_ptr = a.as_ptr().add(i0 * k);
        let a1_ptr = a.as_ptr().add(i1 * k);
        let o0_ptr = out.as_mut_ptr().add(i0 * n);
        let o1_ptr = out.as_mut_ptr().add(i1 * n);

        for blk in 0..n_blocks {
            let col = blk * 16;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c10 = zero;
            let mut c11 = zero;

            for t in 0..k {
                let av0 = _mm256_set1_ps(*a0_ptr.add(t));
                let av1 = _mm256_set1_ps(*a1_ptr.add(t));

                let bp = b_ptr.add(t * n + col);
                let bv0 = _mm256_loadu_ps(bp);
                let bv1 = _mm256_loadu_ps(bp.add(8));

                c00 = _mm256_fmadd_ps(av0, bv0, c00);
                c01 = _mm256_fmadd_ps(av0, bv1, c01);
                c10 = _mm256_fmadd_ps(av1, bv0, c10);
                c11 = _mm256_fmadd_ps(av1, bv1, c11);
            }

            if let Some(bp) = bias_ptr {
                let b0 = _mm256_loadu_ps(bp.add(col));
                let b1 = _mm256_loadu_ps(bp.add(col + 8));
                c00 = _mm256_add_ps(c00, b0);
                c01 = _mm256_add_ps(c01, b1);
                c10 = _mm256_add_ps(c10, b0);
                c11 = _mm256_add_ps(c11, b1);
            }

            if relu {
                c00 = _mm256_max_ps(c00, zero);
                c01 = _mm256_max_ps(c01, zero);
                c10 = _mm256_max_ps(c10, zero);
                c11 = _mm256_max_ps(c11, zero);
            }

            _mm256_storeu_ps(o0_ptr.add(col), c00);
            _mm256_storeu_ps(o0_ptr.add(col + 8), c01);
            _mm256_storeu_ps(o1_ptr.add(col), c10);
            _mm256_storeu_ps(o1_ptr.add(col + 8), c11);
        }

        if n_8 > 0 {
            let col = n_blocks * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            for t in 0..k {
                let av0 = _mm256_set1_ps(*a0_ptr.add(t));
                let av1 = _mm256_set1_ps(*a1_ptr.add(t));
                let bv = _mm256_loadu_ps(b_ptr.add(t * n + col));
                c0 = _mm256_fmadd_ps(av0, bv, c0);
                c1 = _mm256_fmadd_ps(av1, bv, c1);
            }
            if let Some(bp) = bias_ptr {
                let b = _mm256_loadu_ps(bp.add(col));
                c0 = _mm256_add_ps(c0, b);
                c1 = _mm256_add_ps(c1, b);
            }
            if relu {
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
            }
            _mm256_storeu_ps(o0_ptr.add(col), c0);
            _mm256_storeu_ps(o1_ptr.add(col), c1);
        }

        if tail > 0 {
            let col = n_blocks * 16 + n_8 * 8;
            for j in 0..tail {
                let mut s0 = 0.0f32;
                let mut s1 = 0.0f32;
                for t in 0..k {
                    let bv = *b_ptr.add(t * n + col + j);
                    s0 += *a0_ptr.add(t) * bv;
                    s1 += *a1_ptr.add(t) * bv;
                }
                if let Some(bp) = bias_ptr {
                    s0 += *bp.add(col + j);
                    s1 += *bp.add(col + j);
                }
                *o0_ptr.add(col + j) = if relu && s0 > 0.0 {
                    s0
                } else if relu {
                    0.0
                } else {
                    s0
                };
                *o1_ptr.add(col + j) = if relu && s1 > 0.0 {
                    s1
                } else if relu {
                    0.0
                } else {
                    s1
                };
            }
        }
    }

    if m_rem > 0 {
        let i = m - 1;
        let a_ptr = a.as_ptr().add(i * k);
        let o_ptr = out.as_mut_ptr().add(i * n);

        for blk in 0..n_blocks {
            let col = blk * 16;
            let mut c0 = zero;
            let mut c1 = zero;

            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                let bp = b_ptr.add(t * n + col);
                c0 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp), c0);
                c1 = _mm256_fmadd_ps(av, _mm256_loadu_ps(bp.add(8)), c1);
            }

            if let Some(bp) = bias_ptr {
                c0 = _mm256_add_ps(c0, _mm256_loadu_ps(bp.add(col)));
                c1 = _mm256_add_ps(c1, _mm256_loadu_ps(bp.add(col + 8)));
            }

            if relu {
                c0 = _mm256_max_ps(c0, zero);
                c1 = _mm256_max_ps(c1, zero);
            }

            _mm256_storeu_ps(o_ptr.add(col), c0);
            _mm256_storeu_ps(o_ptr.add(col + 8), c1);
        }

        if n_8 > 0 {
            let col = n_blocks * 16;
            let mut c = zero;
            for t in 0..k {
                let av = _mm256_set1_ps(*a_ptr.add(t));
                c = _mm256_fmadd_ps(av, _mm256_loadu_ps(b_ptr.add(t * n + col)), c);
            }
            if let Some(bp) = bias_ptr {
                c = _mm256_add_ps(c, _mm256_loadu_ps(bp.add(col)));
            }
            if relu {
                c = _mm256_max_ps(c, zero);
            }
            _mm256_storeu_ps(o_ptr.add(col), c);
        }

        if tail > 0 {
            let col = n_blocks * 16 + n_8 * 8;
            for j in 0..tail {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += *a_ptr.add(t) * *b_ptr.add(t * n + col + j);
                }
                let s = if let Some(bp) = bias_ptr {
                    s + *bp.add(col + j)
                } else {
                    s
                };
                *o_ptr.add(col + j) = if relu && s > 0.0 {
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
/// Quantise activations to i8 with per-value scaling and clamp to `[-127, 127]`.
///
/// `qa` is the weight-tensor's `QuantMultA` (activation quantizer). Rounding
/// matches the marian intgemm path (`round(x * qa)` without a rounding bias).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn quantize_row_avx2(data: &[f32], qa: f32, out: &mut [i8]) {
    use std::arch::x86_64::*;
    let len = data.len();
    let chunks16 = len / 16;
    let mut offset = 0;
    let qa_vec = _mm256_set1_ps(qa);
    let half_pos = _mm256_set1_ps(0.5);
    let half_neg = _mm256_set1_ps(-0.5);
    let zero = _mm256_setzero_ps();
    let min_clamp = _mm256_set1_ps(-127.0);
    let max_clamp = _mm256_set1_ps(127.0);

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    for _ in 0..chunks16 {
        let v0 = _mm256_loadu_ps(src.add(offset));
        let scaled0 = _mm256_mul_ps(v0, qa_vec);
        let mask0 = _mm256_cmp_ps(scaled0, zero, _CMP_GE_OQ);
        let off0 = _mm256_blendv_ps(half_neg, half_pos, mask0);
        let rounded0 = _mm256_add_ps(scaled0, off0);
        let ord_mask0 = _mm256_cmp_ps(rounded0, rounded0, _CMP_EQ_OQ);
        let clamped0 = _mm256_min_ps(_mm256_max_ps(rounded0, min_clamp), max_clamp);
        let valid_clamped0 = _mm256_and_ps(clamped0, ord_mask0);
        let r0 = _mm256_cvttps_epi32(valid_clamped0);

        let v1 = _mm256_loadu_ps(src.add(offset + 8));
        let scaled1 = _mm256_mul_ps(v1, qa_vec);
        let mask1 = _mm256_cmp_ps(scaled1, zero, _CMP_GE_OQ);
        let off1 = _mm256_blendv_ps(half_neg, half_pos, mask1);
        let rounded1 = _mm256_add_ps(scaled1, off1);
        let ord_mask1 = _mm256_cmp_ps(rounded1, rounded1, _CMP_EQ_OQ);
        let clamped1 = _mm256_min_ps(_mm256_max_ps(rounded1, min_clamp), max_clamp);
        let valid_clamped1 = _mm256_and_ps(clamped1, ord_mask1);
        let r1 = _mm256_cvttps_epi32(valid_clamped1);

        let r0_lo = _mm256_castsi256_si128(r0);
        let r0_hi = _mm256_extracti128_si256(r0, 1);
        let r1_lo = _mm256_castsi256_si128(r1);
        let r1_hi = _mm256_extracti128_si256(r1, 1);

        let p16_0 = _mm_packs_epi32(r0_lo, r0_hi);
        let p16_1 = _mm_packs_epi32(r1_lo, r1_hi);
        let packed = _mm_packs_epi16(p16_0, p16_1);

        _mm_storeu_si128(dst.add(offset) as *mut __m128i, packed);
        offset += 16;
    }

    if len - offset >= 8 {
        let v = _mm256_loadu_ps(src.add(offset));
        let scaled = _mm256_mul_ps(v, qa_vec);
        let mask = _mm256_cmp_ps(scaled, zero, _CMP_GE_OQ);
        let off = _mm256_blendv_ps(half_neg, half_pos, mask);
        let rounded = _mm256_add_ps(scaled, off);
        let ord_mask = _mm256_cmp_ps(rounded, rounded, _CMP_EQ_OQ);
        let clamped = _mm256_min_ps(_mm256_max_ps(rounded, min_clamp), max_clamp);
        let valid_clamped = _mm256_and_ps(clamped, ord_mask);
        let r = _mm256_cvttps_epi32(valid_clamped);

        let r_lo = _mm256_castsi256_si128(r);
        let r_hi = _mm256_extracti128_si256(r, 1);
        let p16 = _mm_packs_epi32(r_lo, r_hi);
        let packed = _mm_packs_epi16(p16, p16);

        (dst.add(offset) as *mut i64).write_unaligned(_mm_cvtsi128_si64(packed));
        offset += 8;
    }

    let tail = len - offset;
    for j in 0..tail {
        let v = *src.add(offset + j);
        let scaled = v * qa;
        let offset_val = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let r = (scaled + offset_val) as i32;
        *dst.add(offset + j) = r.clamp(-127, 127) as i8;
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
#[inline]
pub unsafe fn matmul_2rows_i8_avx2(
    lo0_buf: &[std::arch::x86_64::__m256i],
    hi0_buf: &[std::arch::x86_64::__m256i],
    lo1_buf: &[std::arch::x86_64::__m256i],
    hi1_buf: &[std::arch::x86_64::__m256i],
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
    use std::arch::x86_64::*;

    const BLK: usize = 32;
    const NR: usize = 4;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    let inv_denom = denom.recip();
    let inv_denom_128 = _mm_set1_ps(inv_denom);
    let zero_128 = _mm_setzero_ps();
    let ones = _mm256_set1_epi16(1);
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
            let lo8_0 = lo0_buf[b];
            let hi8_0 = hi0_buf[b];
            let lo8_1 = lo1_buf[b];
            let hi8_1 = hi1_buf[b];
            macro_rules! step_2r {
                ($c:expr, $p:expr) => {
                    let wv = _mm256_loadu_si256($p.add(off) as *const __m256i);
                    let lo_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_0, wv), ones);
                    let hi_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_0, wv), ones);
                    let t32_0 = _mm256_add_epi32(lo_part0, _mm256_slli_epi32::<4>(hi_part0));
                    acc0[$c] = _mm256_add_epi32(acc0[$c], t32_0);

                    let lo_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_1, wv), ones);
                    let hi_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_1, wv), ones);
                    let t32_1 = _mm256_add_epi32(lo_part1, _mm256_slli_epi32::<4>(hi_part1));
                    acc1[$c] = _mm256_add_epi32(acc1[$c], t32_1);
                };
            }
            step_2r!(0, p0);
            step_2r!(1, p1);
            step_2r!(2, p2);
            step_2r!(3, p3);
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
            let lo8_0 = lo0_buf[b];
            let hi8_0 = hi0_buf[b];
            let lo8_1 = lo1_buf[b];
            let hi8_1 = hi1_buf[b];
            let wv = _mm256_loadu_si256(wr.as_ptr().add(off) as *const __m256i);
            let lo_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_0, wv), ones);
            let hi_part0 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_0, wv), ones);
            let t32_0 = _mm256_add_epi32(lo_part0, _mm256_slli_epi32::<4>(hi_part0));
            acc0 = _mm256_add_epi32(acc0, t32_0);

            let lo_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo8_1, wv), ones);
            let hi_part1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi8_1, wv), ones);
            let t32_1 = _mm256_add_epi32(lo_part1, _mm256_slli_epi32::<4>(hi_part1));
            acc1 = _mm256_add_epi32(acc1, t32_1);
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
    static AVX2_LO_BUF: std::cell::RefCell<Vec<std::arch::x86_64::__m256i>> =
        const { std::cell::RefCell::new(Vec::new()) };
    static AVX2_HI_BUF: std::cell::RefCell<Vec<std::arch::x86_64::__m256i>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn with_avx2_prebiased<
    R,
    F: FnOnce(&[std::arch::x86_64::__m256i], &[std::arch::x86_64::__m256i]) -> R,
>(
    x: &[i8],
    m: usize,
    k: usize,
    f: F,
) -> R {
    use std::arch::x86_64::*;
    const BLK: usize = 32;
    let blocks = k / BLK;
    let x_bias_vec = _mm256_set1_epi8(0x80u8 as i8);
    let nibble_mask = _mm256_set1_epi8(0x0F);
    let total_blocks = m * blocks;
    AVX2_LO_BUF.with(|cell_lo| {
        AVX2_HI_BUF.with(|cell_hi| {
            let mut lo_vec = cell_lo.borrow_mut();
            let mut hi_vec = cell_hi.borrow_mut();
            if lo_vec.len() < total_blocks {
                lo_vec.resize(total_blocks, _mm256_setzero_si256());
            }
            if hi_vec.len() < total_blocks {
                hi_vec.resize(total_blocks, _mm256_setzero_si256());
            }

            for row in 0..m {
                let xr = &x[row * k..(row + 1) * k];
                let lo_dest = &mut lo_vec[row * blocks..(row + 1) * blocks];
                let hi_dest = &mut hi_vec[row * blocks..(row + 1) * blocks];
                for b in 0..blocks {
                    let off = b * BLK;
                    let xv = _mm256_loadu_si256(xr.as_ptr().add(off) as *const __m256i);
                    let xb = _mm256_add_epi8(xv, x_bias_vec);
                    lo_dest[b] = _mm256_and_si256(xb, nibble_mask);
                    hi_dest[b] = _mm256_and_si256(_mm256_srli_epi16::<4>(xb), nibble_mask);
                }
            }

            f(&lo_vec[..total_blocks], &hi_vec[..total_blocks])
        })
    })
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_avx2_prebiased(
    lo_slice: &[std::arch::x86_64::__m256i],
    hi_slice: &[std::arch::x86_64::__m256i],
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
    const N_TILE: usize = 64;
    const BLK: usize = 32;
    let blocks = k / BLK;
    let fallback: std::borrow::Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => std::borrow::Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };
    let wsums: &[i32] = &fallback;

    for j_start in (0..n).step_by(N_TILE) {
        let j_len = (n - j_start).min(N_TILE);
        let w_tile = &w[j_start * k..(j_start + j_len) * k];
        let bias_tile = bias.map(|b| &b[j_start..j_start + j_len]);
        let rs_tile = Some(&wsums[j_start..j_start + j_len]);

        let mut i = 0;
        while i + 1 < m {
            let lo0 = &lo_slice[i * blocks..(i + 1) * blocks];
            let hi0 = &hi_slice[i * blocks..(i + 1) * blocks];
            let lo1 = &lo_slice[(i + 1) * blocks..(i + 2) * blocks];
            let hi1 = &hi_slice[(i + 1) * blocks..(i + 2) * blocks];
            let xr0 = &x[i * k..(i + 1) * k];
            let xr1 = &x[(i + 1) * k..(i + 2) * k];
            let (top, bottom) = out.split_at_mut((i + 1) * n);
            let out_r0 = &mut top[i * n + j_start..i * n + j_start + j_len];
            let out_r1 = &mut bottom[j_start..j_start + j_len];
            matmul_2rows_i8_avx2(
                lo0, hi0, lo1, hi1, xr0, xr1, w_tile, k, j_len, bias_tile, rs_tile, out_r0, out_r1,
                denom, relu,
            );
            i += 2;
        }
        if i < m {
            let lo = &lo_slice[i * blocks..(i + 1) * blocks];
            let hi = &hi_slice[i * blocks..(i + 1) * blocks];
            let xr = &x[i * k..(i + 1) * k];
            let out_row = &mut out[i * n + j_start..i * n + j_start + j_len];
            matmul_row_i8_avx2_prebiased(
                lo, hi, xr, w_tile, k, j_len, bias_tile, rs_tile, out_row, denom, relu,
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
unsafe fn matmul_i8_avx2_prebiased_parallel_recursive(
    lo: &[std::arch::x86_64::__m256i],
    hi: &[std::arch::x86_64::__m256i],
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
    if m < 2 * super::GEMM_MIN_ROWS {
        matmul_i8_avx2_prebiased(lo, hi, x, w, m, k, n, bias, row_sums, out, denom, relu);
        return;
    }
    let m_mid = m / 2;
    let blocks = k / 32;
    let (lo_top, lo_bot) = lo.split_at(m_mid * blocks);
    let (hi_top, hi_bot) = hi.split_at(m_mid * blocks);
    let (x_top, x_bot) = x.split_at(m_mid * k);
    let (o_top, o_bot) = out.split_at_mut(m_mid * n);
    rayon::join(
        || {
            matmul_i8_avx2_prebiased_parallel_recursive(
                lo_top, hi_top, x_top, w, m_mid, k, n, bias, row_sums, o_top, denom, relu,
            )
        },
        || {
            matmul_i8_avx2_prebiased_parallel_recursive(
                lo_bot, hi_bot, x_bot, w, m - m_mid, k, n, bias, row_sums, o_bot, denom, relu,
            )
        },
    );
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn matmul_i8_avx2_prebiased_split(
    lo: &[std::arch::x86_64::__m256i],
    hi: &[std::arch::x86_64::__m256i],
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
    if super::gemm_should_split(m, k, n) {
        super::ACTIVE_SPLITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let _guard = super::SplitGuard;
        matmul_i8_avx2_prebiased_parallel_recursive(
            lo, hi, x, w, m, k, n, bias, row_sums, out, denom, relu,
        );
    } else {
        matmul_i8_avx2_prebiased(lo, hi, x, w, m, k, n, bias, row_sums, out, denom, relu);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_avx2(
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
    with_avx2_prebiased(x, m, k, |lo_slice, hi_slice| {
        matmul_i8_avx2_prebiased(
            lo_slice, hi_slice, x, w, m, k, n, bias, row_sums, out, denom, relu,
        );
    });
}
