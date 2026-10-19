//! Scalar fallback kernels for tensor operations.
//!
//! Provides portable reference implementations that run without SIMD instruction
//! sets, and serve as bit-identical validation baselines for vector kernels.

#[allow(clippy::too_many_arguments)]
pub fn matmul_row_i8_scalar(
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
    let inv = denom.recip();
    for j in 0..n {
        let wr = &w[j * k..(j + 1) * k];
        let mut acc: i32 = 0;
        for t in 0..k {
            acc += x[t] as i32 * wr[t] as i32;
        }
        let mut v = acc as f32 * inv;
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
pub fn matmul_row_scalar(
    a: &[f32],
    b: &[f32],
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    for j in 0..n {
        let mut s = 0.0f32;
        for t in 0..k {
            s += a[t] * b[t * n + j];
        }
        let s = if let Some(bb) = bias { s + bb[j] } else { s };
        // Same `> 0.0` predicate as the standalone relu helper (a `-0.0`
        // product stores `0.0`).
        out[j] = if relu && s > 0.0 {
            s
        } else if relu {
            0.0
        } else {
            s
        };
    }
}
/// Scalar `M×K @ K×N` reference kernel (i-k-j accumulation, compiler
/// auto-vectorises the inner dot product). Fully overwrites `out`; an optional
/// bias is added to every output row and `relu` applies `max(0, x)` in the
/// same pass (same `> 0.0` predicate as the standalone relu helper).
#[allow(clippy::too_many_arguments)]
pub fn matmul_scalar(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    out: &mut [f32],
    relu: bool,
) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let o_row = &mut out[i * n..(i + 1) * n];
        o_row.fill(0.0);
        for t in 0..k {
            let av = a_row[t];
            let b_row = &b[t * n..(t + 1) * n];
            for j in 0..n {
                o_row[j] += av * b_row[j];
            }
        }
        if let Some(bb) = bias {
            for j in 0..n {
                o_row[j] += bb[j];
            }
        }
        if relu {
            for v in o_row.iter_mut() {
                *v = if *v > 0.0 { *v } else { 0.0 };
            }
        }
    }
}
/// Generic i8 fallback used when the platform has no viable SIMD int8 kernel.
/// y = x·w (int32 accumulation), then dequantized by `denom` (+ optional bias).
#[allow(clippy::too_many_arguments)]
pub fn matmul_i8_scalar(
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
    let inv = denom.recip();
    for i in 0..m {
        let xr = &x[i * k..(i + 1) * k];
        let or = &mut out[i * n..(i + 1) * n];
        for j in 0..n {
            let wr = &w[j * k..(j + 1) * k];
            let mut acc: i32 = 0;
            for t in 0..k {
                acc += xr[t] as i32 * wr[t] as i32;
            }
            let mut v = acc as f32 * inv;
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
#[inline]
pub unsafe fn quantize_row_scalar(data: &[f32], qa: f32, out: &mut [i8]) {
    for (o, &v) in out.iter_mut().zip(data.iter()) {
        let scaled = v * qa;
        let offset = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let r = (scaled + offset) as i32;
        *o = (r.clamp(-127, 127)) as i8;
    }
}
