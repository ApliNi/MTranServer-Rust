//! SSE4.1 / SSSE3 / SSE2 accelerated SIMD kernels.

#![cfg(target_arch = "x86_64")]

use super::compute_row_sums;
#[allow(unused_imports)]
use std::arch::x86_64::*;
#[allow(unused_imports)]
use std::borrow::Cow;

/// SSE4.1/SSSE3 i8 single-row kernel — 128-bit counterpart of
/// [`matmul_row_i8_avx2`]: k-outer with four independent per-column i32
/// accumulators sharing each biased activation load, mirroring
/// [`matmul_i8_sse41`]'s per-block chain over 16-byte blocks. The biased
/// activation byte is split into nibbles so every i16 pairwise sum stays
/// ≤ 2·15·127 = 3810 (exact for full-range weights), and the per-column
/// `128·Σw` correction is subtracted once from the final horizontal sum.
/// Column groups below the 4-group fall back to the same chain one column
/// at a time.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_row_i8_sse41(
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
    const BLK: usize = 16;
    const NR: usize = 4;
    let blocks = k / BLK;
    let tail = k % BLK;
    let groups = n / NR;
    let jtail = n % NR;

    let inv_denom = denom.recip();
    let ones = _mm_set1_epi16(1);
    let x_bias_vec = _mm_set1_epi8(0x80u8 as i8);
    let nibble_mask = _mm_set1_epi8(0x0F);
    let wsums: std::borrow::Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => std::borrow::Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };

    let mut lo_stack = [_mm_setzero_si128(); 64];
    let mut hi_stack = [_mm_setzero_si128(); 64];
    let (mut lo_heap, mut hi_heap);
    let (lo_buf, hi_buf): (&mut [__m128i], &mut [__m128i]) = if blocks <= 64 {
        (&mut lo_stack[..blocks], &mut hi_stack[..blocks])
    } else {
        lo_heap = vec![_mm_setzero_si128(); blocks];
        hi_heap = vec![_mm_setzero_si128(); blocks];
        (&mut lo_heap[..], &mut hi_heap[..])
    };

    for b in 0..blocks {
        let off = b * BLK;
        let xv = _mm_loadu_si128(x.as_ptr().add(off) as *const __m128i);
        let xb = _mm_add_epi8(xv, x_bias_vec);
        *lo_buf.get_unchecked_mut(b) = _mm_and_si128(xb, nibble_mask);
        *hi_buf.get_unchecked_mut(b) = _mm_and_si128(_mm_srli_epi16::<4>(xb), nibble_mask);
    }

    for g in 0..groups {
        let j0 = g * NR;
        let mut acc = [_mm_setzero_si128(); NR];
        let wr_ptrs = [
            w.as_ptr().add(j0 * k),
            w.as_ptr().add((j0 + 1) * k),
            w.as_ptr().add((j0 + 2) * k),
            w.as_ptr().add((j0 + 3) * k),
        ];
        for b in 0..blocks {
            let off = b * BLK;
            let lo8 = *lo_buf.get_unchecked(b);
            let hi8 = *hi_buf.get_unchecked(b);
            for c in 0..NR {
                let wv = _mm_loadu_si128(wr_ptrs[c].add(off) as *const __m128i);
                let lo_part = _mm_madd_epi16(_mm_maddubs_epi16(lo8, wv), ones);
                let hi_part = _mm_madd_epi16(_mm_maddubs_epi16(hi8, wv), ones);
                let t32 = _mm_add_epi32(lo_part, _mm_slli_epi32::<4>(hi_part));
                acc[c] = _mm_add_epi32(acc[c], t32);
            }
        }

        for c in 0..NR {
            let j = j0 + c;
            let sum = _mm_hadd_epi32(acc[c], acc[c]);
            let sum = _mm_hadd_epi32(sum, sum);
            // Undo the unsigned-x bias: Σ(x+128)·w − 128·Σw over the
            // vectorised blocks. `wsums` carries the full-row ×128 sum;
            // the scalar tail below adds unbiased products, so its weights
            // are removed from the correction.
            let mut total: i32 = _mm_cvtsi128_si32(sum);
            let mut tail_wsum: i32 = 0;
            if tail > 0 {
                let off = blocks * BLK;
                let wr = &w[j * k..(j + 1) * k];
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

    for jt in 0..jtail {
        let j = groups * NR + jt;
        let wr_ptr = w.as_ptr().add(j * k);
        let mut acc = _mm_setzero_si128();
        for b in 0..blocks {
            let off = b * BLK;
            let lo8 = *lo_buf.get_unchecked(b);
            let hi8 = *hi_buf.get_unchecked(b);
            let wv = _mm_loadu_si128(wr_ptr.add(off) as *const __m128i);
            let lo_part = _mm_madd_epi16(_mm_maddubs_epi16(lo8, wv), ones);
            let hi_part = _mm_madd_epi16(_mm_maddubs_epi16(hi8, wv), ones);
            let t32 = _mm_add_epi32(lo_part, _mm_slli_epi32::<4>(hi_part));
            acc = _mm_add_epi32(acc, t32);
        }
        let sum = _mm_hadd_epi32(acc, acc);
        let sum = _mm_hadd_epi32(sum, sum);
        let mut total: i32 = _mm_cvtsi128_si32(sum);
        // Same tail-aware bias correction as the group loop above.
        let mut tail_wsum: i32 = 0;
        if tail > 0 {
            let off = blocks * BLK;
            let wr = &w[j * k..(j + 1) * k];
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
/// SSE4.1 single-row f32 matmul: 4-wide groups, one broadcast per k.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn matmul_row_sse41(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    use std::arch::x86_64::{
        __m128, _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
        _mm_storeu_ps,
    };
    const NR: usize = 4;
    const MAX_STACK_GROUPS: usize = 256;
    let groups = n / NR;
    let tail = n % NR;
    let mut stack_acc = [_mm_setzero_ps(); MAX_STACK_GROUPS];
    let mut heap_acc: Vec<__m128>;
    let acc_slice: &mut [__m128] = if groups <= MAX_STACK_GROUPS {
        &mut stack_acc[..groups]
    } else {
        heap_acc = vec![_mm_setzero_ps(); groups];
        heap_acc.as_mut_slice()
    };
    for t in 0..k {
        let av = _mm_set1_ps(a[t]);
        for (g, ac) in acc_slice.iter_mut().enumerate() {
            let bv = _mm_loadu_ps(b[t * n + g * NR..].as_ptr());
            *ac = _mm_add_ps(_mm_mul_ps(av, bv), *ac);
        }
    }
    for (g, &ac) in acc_slice.iter().enumerate() {
        let mut v = ac;
        if let Some(bb) = bias {
            v = _mm_add_ps(v, _mm_loadu_ps(bb[g * NR..].as_ptr()));
        }
        if relu {
            v = _mm_max_ps(v, _mm_setzero_ps());
        }
        _mm_storeu_ps(out[g * NR..].as_mut_ptr(), v);
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
/// SSE4.1 `M×K @ K×N` f32 kernel.
///
/// N-blocked: 4-wide `__m128` accumulators, one per 4-column tile; the inner
/// loop broadcasts one activation and does mul+add against a contiguous `b`
/// tile (the layout already streams weight rows linearly), then emits the
/// tile and reuses it (L1-resident) across all M rows. A scalar tail handles
/// `n % 4`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_sse41(
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
        _mm_add_ps, _mm_loadu_ps, _mm_max_ps, _mm_mul_ps, _mm_set1_ps, _mm_setzero_ps,
        _mm_storeu_ps,
    };

    let zero = _mm_setzero_ps();
    let b_ptr = b.as_ptr();
    let bias_ptr = bias.map(|bb| bb.as_ptr());

    if m == 1 {
        let a_ptr = a.as_ptr();
        let o_ptr = out.as_mut_ptr();
        let n_blocks = n / 16;
        let n_rem = n % 16;

        for blk in 0..n_blocks {
            let col = blk * 16;
            let mut c0 = zero;
            let mut c1 = zero;
            let mut c2 = zero;
            let mut c3 = zero;

            for t in 0..k {
                let av = _mm_set1_ps(*a_ptr.add(t));
                let bp = b_ptr.add(t * n + col);
                c0 = _mm_add_ps(c0, _mm_mul_ps(av, _mm_loadu_ps(bp)));
                c1 = _mm_add_ps(c1, _mm_mul_ps(av, _mm_loadu_ps(bp.add(4))));
                c2 = _mm_add_ps(c2, _mm_mul_ps(av, _mm_loadu_ps(bp.add(8))));
                c3 = _mm_add_ps(c3, _mm_mul_ps(av, _mm_loadu_ps(bp.add(12))));
            }

            if let Some(bp) = bias_ptr {
                c0 = _mm_add_ps(c0, _mm_loadu_ps(bp.add(col)));
                c1 = _mm_add_ps(c1, _mm_loadu_ps(bp.add(col + 4)));
                c2 = _mm_add_ps(c2, _mm_loadu_ps(bp.add(col + 8)));
                c3 = _mm_add_ps(c3, _mm_loadu_ps(bp.add(col + 12)));
            }

            if relu {
                c0 = _mm_max_ps(c0, zero);
                c1 = _mm_max_ps(c1, zero);
                c2 = _mm_max_ps(c2, zero);
                c3 = _mm_max_ps(c3, zero);
            }

            _mm_storeu_ps(o_ptr.add(col), c0);
            _mm_storeu_ps(o_ptr.add(col + 4), c1);
            _mm_storeu_ps(o_ptr.add(col + 8), c2);
            _mm_storeu_ps(o_ptr.add(col + 12), c3);
        }

        let rem_col = n_blocks * 16;
        let rem_groups = n_rem / 4;
        for g in 0..rem_groups {
            let col = rem_col + g * 4;
            let mut c = zero;
            for t in 0..k {
                let av = _mm_set1_ps(*a_ptr.add(t));
                c = _mm_add_ps(c, _mm_mul_ps(av, _mm_loadu_ps(b_ptr.add(t * n + col))));
            }
            if let Some(bp) = bias_ptr {
                c = _mm_add_ps(c, _mm_loadu_ps(bp.add(col)));
            }
            if relu {
                c = _mm_max_ps(c, zero);
            }
            _mm_storeu_ps(o_ptr.add(col), c);
        }

        let tail_col = rem_col + rem_groups * 4;
        let tail = n_rem % 4;
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

    let n_blocks = n / 8;
    let n_rem = n % 8;
    let n_4 = n_rem / 4;
    let tail = n_rem % 4;

    for mi in 0..m_pairs {
        let i0 = mi * 2;
        let i1 = i0 + 1;
        let a0_ptr = a.as_ptr().add(i0 * k);
        let a1_ptr = a.as_ptr().add(i1 * k);
        let o0_ptr = out.as_mut_ptr().add(i0 * n);
        let o1_ptr = out.as_mut_ptr().add(i1 * n);

        for blk in 0..n_blocks {
            let col = blk * 8;
            let mut c00 = zero;
            let mut c01 = zero;
            let mut c10 = zero;
            let mut c11 = zero;

            for t in 0..k {
                let av0 = _mm_set1_ps(*a0_ptr.add(t));
                let av1 = _mm_set1_ps(*a1_ptr.add(t));

                let bp = b_ptr.add(t * n + col);
                let bv0 = _mm_loadu_ps(bp);
                let bv1 = _mm_loadu_ps(bp.add(4));

                c00 = _mm_add_ps(c00, _mm_mul_ps(av0, bv0));
                c01 = _mm_add_ps(c01, _mm_mul_ps(av0, bv1));
                c10 = _mm_add_ps(c10, _mm_mul_ps(av1, bv0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(av1, bv1));
            }

            if let Some(bp) = bias_ptr {
                let b0 = _mm_loadu_ps(bp.add(col));
                let b1 = _mm_loadu_ps(bp.add(col + 4));
                c00 = _mm_add_ps(c00, b0);
                c01 = _mm_add_ps(c01, b1);
                c10 = _mm_add_ps(c10, b0);
                c11 = _mm_add_ps(c11, b1);
            }

            if relu {
                c00 = _mm_max_ps(c00, zero);
                c01 = _mm_max_ps(c01, zero);
                c10 = _mm_max_ps(c10, zero);
                c11 = _mm_max_ps(c11, zero);
            }

            _mm_storeu_ps(o0_ptr.add(col), c00);
            _mm_storeu_ps(o0_ptr.add(col + 4), c01);
            _mm_storeu_ps(o1_ptr.add(col), c10);
            _mm_storeu_ps(o1_ptr.add(col + 4), c11);
        }

        if n_4 > 0 {
            let col = n_blocks * 8;
            let mut c0 = zero;
            let mut c1 = zero;
            for t in 0..k {
                let av0 = _mm_set1_ps(*a0_ptr.add(t));
                let av1 = _mm_set1_ps(*a1_ptr.add(t));
                let bv = _mm_loadu_ps(b_ptr.add(t * n + col));
                c0 = _mm_add_ps(c0, _mm_mul_ps(av0, bv));
                c1 = _mm_add_ps(c1, _mm_mul_ps(av1, bv));
            }
            if let Some(bp) = bias_ptr {
                let b = _mm_loadu_ps(bp.add(col));
                c0 = _mm_add_ps(c0, b);
                c1 = _mm_add_ps(c1, b);
            }
            if relu {
                c0 = _mm_max_ps(c0, zero);
                c1 = _mm_max_ps(c1, zero);
            }
            _mm_storeu_ps(o0_ptr.add(col), c0);
            _mm_storeu_ps(o1_ptr.add(col), c1);
        }

        if tail > 0 {
            let tail_col = n_blocks * 8 + n_4 * 4;
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
            let col = blk * 8;
            let mut c0 = zero;
            let mut c1 = zero;

            for t in 0..k {
                let av = _mm_set1_ps(*a_ptr.add(t));
                let bp = b_ptr.add(t * n + col);
                c0 = _mm_add_ps(c0, _mm_mul_ps(av, _mm_loadu_ps(bp)));
                c1 = _mm_add_ps(c1, _mm_mul_ps(av, _mm_loadu_ps(bp.add(4))));
            }

            if let Some(bp) = bias_ptr {
                c0 = _mm_add_ps(c0, _mm_loadu_ps(bp.add(col)));
                c1 = _mm_add_ps(c1, _mm_loadu_ps(bp.add(col + 4)));
            }

            if relu {
                c0 = _mm_max_ps(c0, zero);
                c1 = _mm_max_ps(c1, zero);
            }

            _mm_storeu_ps(o_ptr.add(col), c0);
            _mm_storeu_ps(o_ptr.add(col + 4), c1);
        }

        if n_4 > 0 {
            let col = n_blocks * 8;
            let mut c = zero;
            for t in 0..k {
                let av = _mm_set1_ps(*a_ptr.add(t));
                c = _mm_add_ps(c, _mm_mul_ps(av, _mm_loadu_ps(b_ptr.add(t * n + col))));
            }
            if let Some(bp) = bias_ptr {
                c = _mm_add_ps(c, _mm_loadu_ps(bp.add(col)));
            }
            if relu {
                c = _mm_max_ps(c, zero);
            }
            _mm_storeu_ps(o_ptr.add(col), c);
        }

        if tail > 0 {
            let tail_col = n_blocks * 8 + n_4 * 4;
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
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline]
pub unsafe fn quantize_row_sse2(data: &[f32], qa: f32, out: &mut [i8]) {
    let len = data.len();
    let chunks = len / 4;
    let tail = len % 4;
    let qa_vec = _mm_set1_ps(qa);
    let half_pos = _mm_set1_ps(0.5);
    let half_neg = _mm_set1_ps(-0.5);
    let zero = _mm_setzero_ps();
    let min_clamp = _mm_set1_ps(-127.0);
    let max_clamp = _mm_set1_ps(127.0);

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    for i in 0..chunks {
        let v = _mm_loadu_ps(src.add(i * 4));
        let scaled = _mm_mul_ps(v, qa_vec);
        let mask = _mm_cmpge_ps(scaled, zero);
        let offset = _mm_or_ps(_mm_and_ps(mask, half_pos), _mm_andnot_ps(mask, half_neg));
        let rounded = _mm_add_ps(scaled, offset);
        let ord_mask = _mm_cmpeq_ps(rounded, rounded);
        let clamped = _mm_min_ps(_mm_max_ps(rounded, min_clamp), max_clamp);
        let valid_clamped = _mm_and_ps(clamped, ord_mask);
        let r = _mm_cvttps_epi32(valid_clamped);
        let packed_16 = _mm_packs_epi32(r, r);
        let packed_8 = _mm_packs_epi16(packed_16, packed_16);
        let val32 = _mm_cvtsi128_si32(packed_8);
        let bytes = val32.to_le_bytes();
        for j in 0..4 {
            let b = bytes[j] as i8;
            *dst.add(i * 4 + j) = if b < -127 { -127 } else { b };
        }
    }

    let off = chunks * 4;
    for j in 0..tail {
        let v = *src.add(off + j);
        let scaled = v * qa;
        let offset = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let r = (scaled + offset) as i32;
        *dst.add(off + j) = r.clamp(-127, 127) as i8;
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.1")]
#[inline]
pub unsafe fn quantize_row_sse41(data: &[f32], qa: f32, out: &mut [i8]) {
    let len = data.len();
    let chunks = len / 8;
    let tail = len % 8;
    let qa_vec = _mm_set1_ps(qa);
    let half_pos = _mm_set1_ps(0.5);
    let half_neg = _mm_set1_ps(-0.5);
    let zero = _mm_setzero_ps();
    let min_clamp = _mm_set1_ps(-127.0);
    let max_clamp = _mm_set1_ps(127.0);
    let neg127 = _mm_set1_epi8(-127);

    let src = data.as_ptr();
    let dst = out.as_mut_ptr();

    for i in 0..chunks {
        let v0 = _mm_loadu_ps(src.add(i * 8));
        let v1 = _mm_loadu_ps(src.add(i * 8 + 4));

        let scaled0 = _mm_mul_ps(v0, qa_vec);
        let scaled1 = _mm_mul_ps(v1, qa_vec);

        let mask0 = _mm_cmpge_ps(scaled0, zero);
        let mask1 = _mm_cmpge_ps(scaled1, zero);

        let offset0 = _mm_or_ps(_mm_and_ps(mask0, half_pos), _mm_andnot_ps(mask0, half_neg));
        let offset1 = _mm_or_ps(_mm_and_ps(mask1, half_pos), _mm_andnot_ps(mask1, half_neg));

        let rounded0 = _mm_add_ps(scaled0, offset0);
        let rounded1 = _mm_add_ps(scaled1, offset1);

        let ord_mask0 = _mm_cmpeq_ps(rounded0, rounded0);
        let ord_mask1 = _mm_cmpeq_ps(rounded1, rounded1);

        let clamped0 = _mm_min_ps(_mm_max_ps(rounded0, min_clamp), max_clamp);
        let clamped1 = _mm_min_ps(_mm_max_ps(rounded1, min_clamp), max_clamp);

        let valid_clamped0 = _mm_and_ps(clamped0, ord_mask0);
        let valid_clamped1 = _mm_and_ps(clamped1, ord_mask1);

        let r0 = _mm_cvttps_epi32(valid_clamped0);
        let r1 = _mm_cvttps_epi32(valid_clamped1);

        let packed_16 = _mm_packs_epi32(r0, r1);
        let packed_8 = _mm_packs_epi16(packed_16, packed_16);
        let clamped_8 = _mm_max_epi8(packed_8, neg127);

        std::ptr::copy_nonoverlapping(&clamped_8 as *const _ as *const i8, dst.add(i * 8), 8);
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
/// SSE4.1/SSSE3 `i8×i8 → i32` gemm kernel.
///
/// 128-bit counterpart of [`matmul_i8_avx2`]: per 16-element block it
/// `_mm_maddubs_epi16`s the biased activation and the weights (pairwise
/// u8×i8 → i16), widens to i32 and accumulates, with the same hoisted
/// `128·Σw` correction subtracted once per output column.
/// N-blocked over 16-column weight tiles; the scalar tail handles `k % 16`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "ssse3")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_sse41(
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
    const BLK: usize = 16;
    let blocks = k / BLK;
    let tail = k % BLK;

    let inv_denom = denom.recip();
    let ones = _mm_set1_epi16(1);
    let x_bias_vec = _mm_set1_epi8(0x80u8 as i8);
    let nibble_mask = _mm_set1_epi8(0x0F);

    // Per-column `128·Σw` correction hoisted out of the m×n loop (see the
    // AVX2 kernel for the derivation); precomputed at load time.
    let fallback: Cow<[i32]> = match row_sums {
        Some(rs) if rs.len() >= n => Cow::Borrowed(&rs[..n]),
        _ => compute_row_sums(w, n, k),
    };
    let row_sum_w: &[i32] = &fallback;

    const NR: usize = 4;
    let groups = n / NR;
    let jtail = n % NR;

    let mut lo_stack = [_mm_setzero_si128(); 64];
    let mut hi_stack = [_mm_setzero_si128(); 64];
    let (mut lo_heap, mut hi_heap);
    let (lo_buf, hi_buf): (&mut [__m128i], &mut [__m128i]) = if blocks <= 64 {
        (&mut lo_stack[..blocks], &mut hi_stack[..blocks])
    } else {
        lo_heap = vec![_mm_setzero_si128(); blocks];
        hi_heap = vec![_mm_setzero_si128(); blocks];
        (&mut lo_heap[..], &mut hi_heap[..])
    };

    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        let or = &mut out[i * n..(i + 1) * n];
        let xr_ptr = xr.as_ptr();

        for b in 0..blocks {
            let off = b * BLK;
            let xv = _mm_loadu_si128(xr_ptr.add(off) as *const __m128i);
            let xb = _mm_add_epi8(xv, x_bias_vec);
            *lo_buf.get_unchecked_mut(b) = _mm_and_si128(xb, nibble_mask);
            *hi_buf.get_unchecked_mut(b) = _mm_and_si128(_mm_srli_epi16::<4>(xb), nibble_mask);
        }
        for g in 0..groups {
            let j0 = g * NR;
            let mut acc = [_mm_setzero_si128(); NR];
            let wr_ptrs = [
                w.as_ptr().add(j0 * k),
                w.as_ptr().add((j0 + 1) * k),
                w.as_ptr().add((j0 + 2) * k),
                w.as_ptr().add((j0 + 3) * k),
            ];

            for b in 0..blocks {
                let off = b * BLK;
                let lo8 = *lo_buf.get_unchecked(b);
                let hi8 = *hi_buf.get_unchecked(b);

                for c in 0..NR {
                    let wv = _mm_loadu_si128(wr_ptrs[c].add(off) as *const __m128i);
                    let lo_part = _mm_madd_epi16(_mm_maddubs_epi16(lo8, wv), ones);
                    let hi_part = _mm_madd_epi16(_mm_maddubs_epi16(hi8, wv), ones);
                    let t32 = _mm_add_epi32(lo_part, _mm_slli_epi32::<4>(hi_part));
                    acc[c] = _mm_add_epi32(acc[c], t32);
                }
            }

            for c in 0..NR {
                let j = j0 + c;
                let sum = _mm_hadd_epi32(acc[c], acc[c]);
                let sum = _mm_hadd_epi32(sum, sum);
                let mut total: i32 = _mm_cvtsi128_si32(sum);

                let mut tail_wsum = 0i32;
                if tail > 0 {
                    let off = blocks * BLK;
                    let wr = &w[j * k..(j + 1) * k];
                    for t in 0..tail {
                        tail_wsum += wr[off + t] as i32;
                        total += xr[off + t] as i32 * wr[off + t] as i32;
                    }
                }
                total -= row_sum_w[j] - tail_wsum * 128;

                let mut v = total as f32 * inv_denom;
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
            let mut acc = _mm_setzero_si128();

            for b in 0..blocks {
                let off = b * BLK;
                let lo8 = *lo_buf.get_unchecked(b);
                let hi8 = *hi_buf.get_unchecked(b);
                let wv = _mm_loadu_si128(wr_ptr.add(off) as *const __m128i);
                let lo_part = _mm_madd_epi16(_mm_maddubs_epi16(lo8, wv), ones);
                let hi_part = _mm_madd_epi16(_mm_maddubs_epi16(hi8, wv), ones);
                let t32 = _mm_add_epi32(lo_part, _mm_slli_epi32::<4>(hi_part));
                acc = _mm_add_epi32(acc, t32);
            }

            let sum = _mm_hadd_epi32(acc, acc);
            let sum = _mm_hadd_epi32(sum, sum);
            let mut total: i32 = _mm_cvtsi128_si32(sum);

            let mut tail_wsum = 0i32;
            if tail > 0 {
                let off = blocks * BLK;
                let wr = &w[j * k..(j + 1) * k];
                for t in 0..tail {
                    tail_wsum += wr[off + t] as i32;
                    total += xr[off + t] as i32 * wr[off + t] as i32;
                }
            }
            total -= row_sum_w[j] - tail_wsum * 128;

            let mut v = total as f32 * inv_denom;
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
