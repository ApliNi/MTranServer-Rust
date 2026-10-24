//! Tensor unit and regression tests.

use super::*;
use crate::engine::tensor::ops::*;
#[allow(unused_imports)]
use crate::engine::tensor::simd::scalar::*;
use crate::engine::tensor::simd::*;
#[cfg(target_arch = "x86_64")]
use crate::engine::tensor::simd::{avx2::*, sse41::*, vnni::*};

#[cfg(test)]
mod relu_fusion_tests {
    use super::*;

    /// xorshift64* — deterministic, no external rng dependency.
    fn next_rand(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (u32::MAX as f32) - 0.5
    }

    fn run_case(m: usize, k: usize, n: usize) {
        let mut rng_state = 12345u64;
        let a_data: Vec<f32> = (0..m * k).map(|_| next_rand(&mut rng_state)).collect();
        let b_data: Vec<f32> = (0..k * n).map(|_| next_rand(&mut rng_state)).collect();
        let bias_data: Vec<f32> = (0..n).map(|_| next_rand(&mut rng_state)).collect();

        let x = Tensor2::from_vec([m, k], a_data);
        let w = Tensor2::from_vec([k, n], b_data);
        let bias = Tensor2::from_vec([1, n], bias_data);

        let mut fused = Tensor2::zeros([m, n]);
        x.matmul_biased_relu_into(&w, Some(&bias), &mut fused);

        let plain = x.matmul_biased(&w, Some(&bias));
        let mut expected = plain.data.clone();
        for v in expected.iter_mut() {
            *v = if *v > 0.0 { *v } else { 0.0 };
        }

        assert_eq!(
            fused.data, expected,
            "relu fusion mismatch at m={m} k={k} n={n}"
        );
    }

    #[test]
    fn relu_fusion_matches_reference() {
        for &(m, k, n) in &[
            (1usize, 384usize, 1536usize), // decoder hot path (row kernels)
            (1, 383, 1536),                // odd k
            (1, 384, 1537),                // n % 32 = 1 scalar tail
            (1, 385, 1550),                // odd k, tail
            (1, 15, 23),                   // small odd k, tail
            (1, 1, 1),                     // minimal 1x1
            (1, 8, 16),
            (1, 32, 40),
            (1, 64, 48),
            (1, 128, 56),
            (16, 384, 1536), // encoder FFN (general kernels)
            (1, 64, 64),
            (3, 32, 12), // n % 4 != 0 && n % 8 != 0 scalar tails
            (5, 17, 21),
            (7, 100, 130),
        ] {
            run_case(m, k, n);
        }
    }

    /// i8 path: the hoisted-correction kernel must match a scalar reference
    /// exactly (int32 accumulation, same dequantization).
    #[test]
    fn quantized_relu_matches_scalar_reference() {
        let mut rng_state = 67890u64;
        let m = 5usize;
        let k = 384usize;
        let n = 1536usize;

        let x_data: Vec<f32> = (0..m * k).map(|_| next_rand(&mut rng_state)).collect();
        let w_data: Vec<i8> = (0..n * k)
            .map(|_| ((next_rand(&mut rng_state) * 40.0) as i32).clamp(-127, 127) as i8)
            .collect();
        let bias_data: Vec<f32> = (0..n).map(|_| next_rand(&mut rng_state) * 0.1).collect();

        let x = Tensor2::from_vec([m, k], x_data.clone());
        let wq = Quant8Tensor {
            shape: vec![n, k],
            data: w_data.clone().into(),
            quant_mult: 64.0,
            row_sums: None,
        };
        let bias = Tensor2::from_vec([1, n], bias_data.clone());

        let mut fused = Tensor2::zeros([m, n]);
        x.quantized_matmul_biased_relu_into(&wq, 64.0, Some(&bias), &mut fused);

        // Scalar reference in f32 domain.
        for i in 0..m {
            for j in 0..n {
                let wr = &w_data[j * k..(j + 1) * k];
                let xr = &x_data[i * k..(i + 1) * k];
                // Quantize activations identically to i8_quantize.
                let mut acc: i64 = 0;
                for t in 0..k {
                    let scaled = xr[t] * 64.0;
                    let r = if scaled >= 0.0 {
                        (scaled + 0.5) as i32
                    } else {
                        (scaled - 0.5) as i32
                    };
                    acc += r.clamp(-127, 127) as i64 * wr[t] as i64;
                }
                let mut v = acc as f32 / (64.0 * 64.0);
                v += bias_data[j];
                let expected = if v > 0.0 { v } else { 0.0 };
                assert_eq!(fused.data[i * n + j], expected, "q8 mismatch at {i},{j}");
            }
        }
    }
}

#[cfg(test)]
mod i8_gemm_tests {
    use super::*;

    /// xorshift64* — deterministic, no external rng dependency.
    fn next_rand(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (u32::MAX as f32) - 0.5
    }

    /// Runs one gemm case against an exact i64 scalar reference. `x_range`
    /// / `w_range` scale the uniform random draws, so callers can exercise
    /// mild data (which the kernels already handled) or the full i8 range
    /// real intgemm weights use — where the pre-nibble maddubs chains used
    /// to saturate.
    #[allow(clippy::too_many_arguments)]
    fn run_gemm_case(
        m: usize,
        k: usize,
        n: usize,
        x_range: f32,
        w_range: f32,
        kernel: unsafe fn(
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
        ),
        name: &str,
    ) {
        let mut rng_state = 4242u64;
        let x_data: Vec<i8> = (0..m * k)
            .map(|_| ((next_rand(&mut rng_state) * x_range) as i32).clamp(-127, 127) as i8)
            .collect();
        let w_data: Vec<i8> = (0..n * k)
            .map(|_| ((next_rand(&mut rng_state) * w_range) as i32).clamp(-127, 127) as i8)
            .collect();
        let bias_data: Vec<f32> = (0..n).map(|_| next_rand(&mut rng_state) * 0.1).collect();

        let mut got = vec![0f32; m * n];
        unsafe {
            kernel(
                &x_data,
                &w_data,
                m,
                k,
                n,
                Some(&bias_data),
                None,
                &mut got,
                64.0 * 64.0,
                false,
            );
        }
        for i in 0..m {
            for j in 0..n {
                let wr = &w_data[j * k..(j + 1) * k];
                let xr = &x_data[i * k..(i + 1) * k];
                let mut acc: i64 = 0;
                for t in 0..k {
                    acc += xr[t] as i64 * wr[t] as i64;
                }
                let mut v = acc as f32 / (64.0 * 64.0);
                v += bias_data[j];
                assert_eq!(got[i * n + j], v, "{name} gemm mismatch at {i},{j}");
            }
        }
    }

    #[test]
    fn avx2_gemm_matches_scalar_reference() {
        // k covers exact blocks (384 = 12×32), block tails (k % 32 != 0),
        // and n covers the 32-column group and sub-group tails.
        for &(m, k, n) in &[
            (16usize, 384usize, 1536usize),
            (7usize, 100usize, 130usize),
            (3usize, 32usize, 12usize),
            (5usize, 50usize, 40usize),
            (2usize, 70usize, 45usize),
        ] {
            run_gemm_case(m, k, n, 100.0, 40.0, matmul_i8_avx2, "avx2");
        }
    }

    #[test]
    fn avx2_gemm_full_range_matches_scalar_reference() {
        // Full-range weights and activations: pair sums of the biased bytes
        // reach ±64770, which would saturate i16 without the nibble split.
        for &(m, k, n) in &[
            (16usize, 384usize, 1536usize),
            (7usize, 100usize, 130usize),
            (3usize, 32usize, 12usize),
            (5usize, 50usize, 40usize),
            (2usize, 70usize, 45usize),
        ] {
            run_gemm_case(m, k, n, 127.0, 127.0, matmul_i8_avx2, "avx2 full-range");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avxvnni_gemm_matches_scalar_reference() {
        if !crate::utils::arch::has_avxvnni() {
            return;
        }
        // Shapes mirror the AVX2 case: exact blocks, tails, sub-group columns.
        for &(m, k, n) in &[
            (16usize, 384usize, 1536usize),
            (7usize, 100usize, 130usize),
            (3usize, 32usize, 12usize),
            (5usize, 50usize, 40usize),
            (2usize, 70usize, 45usize),
        ] {
            run_gemm_case(m, k, n, 127.0, 127.0, matmul_i8_avxvnni, "avxvnni");
        }
    }

    /// Single-row kernels (the decode hot path): exact i64 reference, mild
    /// and full-range data, exact blocks and tails.
    #[allow(clippy::too_many_arguments)]
    fn run_row_case(
        k: usize,
        n: usize,
        x_range: f32,
        w_range: f32,
        kernel: unsafe fn(
            &[i8],
            &[i8],
            usize,
            usize,
            Option<&[f32]>,
            Option<&[i32]>,
            &mut [f32],
            f32,
            bool,
        ),
        name: &str,
    ) {
        let mut rng_state = 909u64;
        let x_data: Vec<i8> = (0..k)
            .map(|_| ((next_rand(&mut rng_state) * x_range) as i32).clamp(-127, 127) as i8)
            .collect();
        let w_data: Vec<i8> = (0..n * k)
            .map(|_| ((next_rand(&mut rng_state) * w_range) as i32).clamp(-127, 127) as i8)
            .collect();
        let bias_data: Vec<f32> = (0..n).map(|_| next_rand(&mut rng_state) * 0.1).collect();

        let mut got = vec![0f32; n];
        unsafe {
            kernel(
                &x_data,
                &w_data,
                k,
                n,
                Some(&bias_data),
                None,
                &mut got,
                64.0 * 64.0,
                false,
            );
        }
        for j in 0..n {
            let wr = &w_data[j * k..(j + 1) * k];
            let mut acc: i64 = 0;
            for t in 0..k {
                acc += x_data[t] as i64 * wr[t] as i64;
            }
            let mut v = acc as f32 / (64.0 * 64.0);
            v += bias_data[j];
            assert_eq!(got[j], v, "{name} row mismatch at {j}");
        }
    }

    #[test]
    fn row_i8_kernels_match_scalar_reference() {
        // 384 = exact 32-blocks and 16-blocks; 100/50/45 cover k tails.
        for &(k, n) in &[
            (384usize, 1536usize),
            (100usize, 130usize),
            (32usize, 12usize),
            (50usize, 40usize),
            (45usize, 45usize),
        ] {
            run_row_case(k, n, 100.0, 40.0, matmul_row_i8_avx2, "avx2 row");
            run_row_case(
                k,
                n,
                127.0,
                127.0,
                matmul_row_i8_avx2,
                "avx2 row full-range",
            );
            #[cfg(target_arch = "x86_64")]
            {
                run_row_case(
                    k,
                    n,
                    127.0,
                    127.0,
                    matmul_row_i8_sse41,
                    "sse41 row full-range",
                );
                if crate::utils::arch::has_avxvnni() {
                    run_row_case(k, n, 127.0, 127.0, matmul_row_i8_avxvnni, "avxvnni row");
                }
            }
        }
    }
}

#[cfg(test)]
mod score_tests {
    use super::*;

    fn next_rand(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (u32::MAX as f32) - 0.5
    }

    fn run_case(q_len: usize, kv_len: usize, d_model: usize, heads: usize) {
        let dk = d_model / heads;
        let mut rng = 777u64;
        let q: Vec<f32> = (0..q_len * d_model).map(|_| next_rand(&mut rng)).collect();
        let k: Vec<f32> = (0..kv_len * d_model).map(|_| next_rand(&mut rng)).collect();
        let scale = 1.0 / (dk as f32).sqrt();

        for h in 0..heads {
            let col = h * dk;
            let mut reference = vec![0f32; q_len * kv_len];
            score_rows_scalar(
                &q,
                &k,
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                &mut reference,
            );

            let mut kt = vec![0f32; dk * kv_len];
            transpose_head_block_tmajor(&k, kv_len, d_model, col, dk, &mut kt);
            let mut got = vec![0f32; q_len * kv_len];
            score_rows_dispatch(
                &q,
                Some(&k),
                Some(&kt),
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                &mut got,
            );
            for (i, (r, g)) in reference.iter().zip(got.iter()).enumerate() {
                assert!((r - g).abs() <= 1e-5, "score mismatch at {i}: {r} vs {g}");
            }
        }
    }

    #[test]
    fn score_rows_match_scalar_bit_identically() {
        // Encoder self-attention shapes, decode-step shapes, and tails.
        for &(q_len, kv_len, d_model, heads) in &[
            (16usize, 16usize, 384usize, 8usize),
            (50, 50, 384, 8),
            (128, 128, 384, 8),
            (1, 16, 384, 8),
            (1, 129, 384, 8),
            (7, 33, 384, 8),
            (3, 8, 384, 8),
            (5, 9, 72, 4),
            (2, 5, 384, 8),
            (3, 5, 384, 8),
            (5, 5, 384, 8),
            (2, 7, 384, 8),
            (6, 7, 384, 8),
        ] {
            run_case(q_len, kv_len, d_model, heads);
        }
    }

    #[test]
    fn score_dispatch_tmajor_only_matches_scalar_bit_identically() {
        // Head-major callers pass `k = None` (their layout does not fit the
        // strided kernel); every dispatch tier must stay bit-identical to
        // the strided scalar reference, including the scalar t-major
        // fallback used below the AVX2 tier.
        let (q_len, kv_len, d_model, heads) = (1usize, 33usize, 384usize, 8usize);
        let dk = d_model / heads;
        let mut rng = 4242u64;
        let q: Vec<f32> = (0..q_len * d_model).map(|_| next_rand(&mut rng)).collect();
        let k: Vec<f32> = (0..kv_len * d_model).map(|_| next_rand(&mut rng)).collect();
        let scale = 1.0 / (dk as f32).sqrt();
        for h in 0..heads {
            let col = h * dk;
            let mut reference = vec![0f32; q_len * kv_len];
            score_rows_scalar(
                &q,
                &k,
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                &mut reference,
            );

            let mut kt = vec![0f32; dk * kv_len];
            transpose_head_block_tmajor(&k, kv_len, d_model, col, dk, &mut kt);
            let mut got = vec![0f32; q_len * kv_len];
            score_rows_dispatch(
                &q,
                None,
                Some(&kt),
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                &mut got,
            );
            for (i, (r, g)) in reference.iter().zip(got.iter()).enumerate() {
                assert!(
                    (r - g).abs() <= 1e-5,
                    "tmajor dispatch mismatch at {i}: {r} vs {g}"
                );
            }
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod quantize_tests {
    use super::*;

    fn scalar_quant(v: f32, qa: f32) -> i8 {
        let scaled = v * qa;
        let offset = if scaled >= 0.0 { 0.5 } else { -0.5 };
        let r = (scaled + offset) as i32;
        r.clamp(-127, 127) as i8
    }

    #[test]
    fn sse2_quant_matches_scalar_bit_identically() {
        let qa = 100.0f32;
        let special_vals = vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            1e20,
            -1e20,
            1e-10,
            -1e-10,
            127.0,
            -127.0,
            128.0,
            -128.0,
            1.27,
            -1.27,
            1.28,
            -1.28,
            1.269,
            -1.269,
            0.0049,
            0.0050,
            0.0051,
            -0.0049,
            -0.0050,
            -0.0051,
        ];
        for len in [0, 1, 2, 3, 4, 7, 8, 11, 16, 25, special_vals.len()] {
            let slice = &special_vals[..len];
            let mut got_sse2 = vec![0i8; len];
            unsafe {
                quantize_row_sse2(slice, qa, &mut got_sse2);
            }
            for (idx, &v) in slice.iter().enumerate() {
                let exp = scalar_quant(v, qa);
                assert_eq!(
                    got_sse2[idx], exp,
                    "SSE2 mismatch at len {len}, idx {idx}, val {v}: got {}, exp {}",
                    got_sse2[idx], exp
                );
            }
            if is_x86_feature_detected!("sse4.1") {
                let mut got_sse41 = vec![0i8; len];
                unsafe {
                    quantize_row_sse41(slice, qa, &mut got_sse41);
                }
                for (idx, &v) in slice.iter().enumerate() {
                    let exp = scalar_quant(v, qa);
                    assert_eq!(
                        got_sse41[idx], exp,
                        "SSE4.1 mismatch at len {len}, idx {idx}, val {v}: got {}, exp {}",
                        got_sse41[idx], exp
                    );
                }
            }

            if is_x86_feature_detected!("avx2") {
                let mut got_avx2 = vec![0i8; len];
                unsafe {
                    quantize_row_avx2(slice, qa, &mut got_avx2);
                }
                for (idx, &v) in slice.iter().enumerate() {
                    let exp = scalar_quant(v, qa);
                    assert_eq!(
                        got_avx2[idx], exp,
                        "AVX2 mismatch at len {len}, idx {idx}, val {v}: got {}, exp {}",
                        got_avx2[idx], exp
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod fused_add_layer_norm_tests {
    use super::*;

    fn next_rand(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 33) as f32 / (u32::MAX as f32) - 0.5
    }

    #[test]
    fn test_fused_add_layer_norm_matches_unfused() {
        let mut rng = 4242u64;
        let eps = 1e-5f32;
        for &(rows, d) in &[(1, 384), (3, 512), (2, 64), (1, 17), (4, 33)] {
            let a_data: Vec<f32> = (0..rows * d).map(|_| next_rand(&mut rng)).collect();
            let b_data: Vec<f32> = (0..rows * d).map(|_| next_rand(&mut rng)).collect();
            let scale_data: Vec<f32> = (0..d).map(|_| next_rand(&mut rng) + 1.0).collect();
            let bias_data: Vec<f32> = (0..d).map(|_| next_rand(&mut rng)).collect();

            let a = Tensor2::from_vec([rows, d], a_data.clone());
            let b = Tensor2::from_vec([rows, d], b_data.clone());
            let scale = Tensor2::from_vec([1, d], scale_data);
            let bias = Tensor2::from_vec([1, d], bias_data);

            // Unfused reference: add then layer norm
            let mut sum = Tensor2::zeros([rows, d]);
            a.add_into(&b, &mut sum);
            let mut expected = Tensor2::zeros([rows, d]);
            sum.layer_norm_into(&scale, &bias, eps, &mut expected);

            // Fused into separate out
            let mut fused_out = Tensor2::zeros([rows, d]);
            fused_add_layer_norm_into(&a, &b, &scale, &bias, eps, &mut fused_out);

            for i in 0..rows * d {
                let diff = (fused_out.data[i] - expected.data[i]).abs();
                assert!(
                    diff < 1e-4,
                    "Mismatch at r*d={rows}x{d}, idx {i}: got {}, exp {}, diff {}",
                    fused_out.data[i],
                    expected.data[i],
                    diff
                );
            }

            // Fused with in-place aliasing (out aliases a)
            let mut in_place_a = a.clone();
            fused_add_layer_norm_in_place(&mut in_place_a, &b, &scale, &bias, eps);
            for i in 0..rows * d {
                let diff = (in_place_a.data[i] - expected.data[i]).abs();
                assert!(
                    diff < 1e-4,
                    "In-place mismatch at r*d={rows}x{d}, idx {i}: got {}, exp {}, diff {}",
                    in_place_a.data[i],
                    expected.data[i],
                    diff
                );
            }
        }
    }

    #[test]
    fn test_fused_add_layer_norm_avx2_vs_scalar_precision() {
        let mut rng = 9999u64;
        let eps = 1e-5f32;
        for &d in &[17, 33, 64, 128, 384, 512, 1024] {
            let a: Vec<f32> = (0..d).map(|_| next_rand(&mut rng)).collect();
            let b: Vec<f32> = (0..d).map(|_| next_rand(&mut rng)).collect();
            let scale: Vec<f32> = (0..d).map(|_| next_rand(&mut rng) + 1.0).collect();
            let bias: Vec<f32> = (0..d).map(|_| next_rand(&mut rng)).collect();

            let mut out_scalar = vec![0.0f32; d];
            unsafe {
                ops::fused_add_layer_norm_scalar_raw(
                    a.as_ptr(),
                    b.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_scalar.as_mut_ptr(),
                    d,
                );
            }

            #[cfg(target_arch = "x86_64")]
            if crate::utils::arch::has_avx2() {
                let mut out_avx2 = vec![0.0f32; d];
                unsafe {
                    ops::fused_add_layer_norm_avx2_raw(
                        a.as_ptr(),
                        b.as_ptr(),
                        scale.as_ptr(),
                        bias.as_ptr(),
                        eps,
                        out_avx2.as_mut_ptr(),
                        d,
                    );
                }

                for i in 0..d {
                    let diff = (out_avx2[i] - out_scalar[i]).abs();
                    let denom = out_scalar[i].abs().max(1e-5);
                    let rel = diff / denom;
                    assert!(
                        rel < 2e-7,
                        "AVX2 vs Scalar relative mismatch at d={d}, idx {i}: avx2={}, scalar={}, diff={}, rel={}",
                        out_avx2[i],
                        out_scalar[i],
                        diff,
                        rel
                    );
                }
            }
        }
    }
    #[test]
    fn test_gelu_accuracy_and_vectorization() {
        use crate::engine::tensor::{Tensor2, gelu, gelu_f32, gelu_inplace, gelu_into};

        // Test over a broad range of values: negative, zero, positive, tail cases
        let test_vals: Vec<f32> = (-200..=200).map(|i| i as f32 * 0.1).collect();
        let t = Tensor2::from_vec([10, 41], test_vals.clone());

        let mut t_inplace = t.clone();
        gelu_inplace(&mut t_inplace);

        let mut t_into = Tensor2::zeros([10, 41]);
        gelu_into(&t, &mut t_into);

        let t_out = gelu(&t);

        assert_eq!(t_inplace.rows(), 10);
        assert_eq!(t_inplace.cols(), 41);
        assert_eq!(t_into.rows(), 10);
        assert_eq!(t_into.cols(), 41);
        assert_eq!(t_out.rows(), 10);
        assert_eq!(t_out.cols(), 41);

        for (i, &x) in test_vals.iter().enumerate() {
            let expected = gelu_f32(x);
            let diff_inplace = (t_inplace.data[i] - expected).abs();
            let diff_into = (t_into.data[i] - expected).abs();
            let diff_out = (t_out.data[i] - expected).abs();

            assert!(
                diff_inplace < 1e-4,
                "GELU inplace mismatch at x={x}: got {}, expected {}, diff {}",
                t_inplace.data[i],
                expected,
                diff_inplace
            );
            assert!(
                diff_into < 1e-4,
                "GELU into mismatch at x={x}: got {}, expected {}, diff {}",
                t_into.data[i],
                expected,
                diff_into
            );
            assert!(
                diff_out < 1e-4,
                "GELU out mismatch at x={x}: got {}, expected {}, diff {}",
                t_out.data[i],
                expected,
                diff_out
            );
        }
    }

    #[test]
    fn test_tensor2_shape_rows_cols() {
        use crate::engine::tensor::Tensor2;
        let mut t = Tensor2::zeros([3, 7]);
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 7);
        assert_eq!(t.shape, [3, 7]);

        t.set_shape_2d(5, 4);
        assert_eq!(t.rows(), 5);
        assert_eq!(t.cols(), 4);
        assert_eq!(t.shape, [5, 4]);
    }

    #[test]
    fn test_quantized_matmul_pair_multi_row_matches_individual() {
        let mut rng = 987654321u64;
        let mut rand_f = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as f32 / (u32::MAX as f32) - 0.5
        };

        for &(m, k, n1, n2) in &[
            (1usize, 32usize, 32usize, 48usize),
            (2, 32, 64, 32),
            (5, 64, 48, 64),
            (16, 64, 64, 64),
            (33, 64, 96, 64),
        ] {
            let x_data: Vec<f32> = (0..m * k).map(|_| rand_f()).collect();
            let w1_data: Vec<i8> = (0..n1 * k).map(|_| ((rand_f() * 40.0) as i32).clamp(-127, 127) as i8).collect();
            let w2_data: Vec<i8> = (0..n2 * k).map(|_| ((rand_f() * 40.0) as i32).clamp(-127, 127) as i8).collect();
            let b1_data: Vec<f32> = (0..n1).map(|_| rand_f() * 0.1).collect();
            let b2_data: Vec<f32> = (0..n2).map(|_| rand_f() * 0.1).collect();

            let x = Tensor2::from_vec([m, k], x_data);
            let rs1 = crate::engine::tensor::simd::compute_row_sums(&w1_data, n1, k).into_owned();
            let rs2 = crate::engine::tensor::simd::compute_row_sums(&w2_data, n2, k).into_owned();
            let w1 = Quant8Tensor { shape: vec![n1, k], data: w1_data.into(), quant_mult: 64.0, row_sums: Some(rs1) };
            let w2 = Quant8Tensor { shape: vec![n2, k], data: w2_data.into(), quant_mult: 64.0, row_sums: Some(rs2) };
            let bias1 = Tensor2::from_vec([1, n1], b1_data);
            let bias2 = Tensor2::from_vec([1, n2], b2_data);

            let qa = 50.0f32;

            // With bias
            let mut p_out1 = Tensor2::zeros([m, n1]);
            let mut p_out2 = Tensor2::zeros([m, n2]);
            x.quantized_matmul_pair_into(&w1, Some(&bias1), &mut p_out1, &w2, Some(&bias2), &mut p_out2, qa);

            let mut ind_out1 = Tensor2::zeros([m, n1]);
            let mut ind_out2 = Tensor2::zeros([m, n2]);
            x.quantized_matmul_biased_into(&w1, qa, Some(&bias1), &mut ind_out1);
            x.quantized_matmul_biased_into(&w2, qa, Some(&bias2), &mut ind_out2);

            assert_eq!(p_out1.data, ind_out1.data, "pair out1 mismatch at m={m} k={k} n1={n1}");
            assert_eq!(p_out2.data, ind_out2.data, "pair out2 mismatch at m={m} k={k} n2={n2}");

            // Without bias
            let mut p_out1_nobias = Tensor2::zeros([m, n1]);
            let mut p_out2_nobias = Tensor2::zeros([m, n2]);
            x.quantized_matmul_pair_into(&w1, None, &mut p_out1_nobias, &w2, None, &mut p_out2_nobias, qa);

            let mut ind_out1_nobias = Tensor2::zeros([m, n1]);
            let mut ind_out2_nobias = Tensor2::zeros([m, n2]);
            x.quantized_matmul_biased_into(&w1, qa, None, &mut ind_out1_nobias);
            x.quantized_matmul_biased_into(&w2, qa, None, &mut ind_out2_nobias);

            assert_eq!(p_out1_nobias.data, ind_out1_nobias.data, "pair out1 (nobias) mismatch at m={m}");
            assert_eq!(p_out2_nobias.data, ind_out2_nobias.data, "pair out2 (nobias) mismatch at m={m}");
        }
    }

    #[test]
    fn test_quantized_matmul_triple_multi_row_matches_individual() {
        let mut rng = 543216789u64;
        let mut rand_f = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (rng >> 33) as f32 / (u32::MAX as f32) - 0.5
        };

        for &(m, k, n1, n2, n3) in &[
            (1usize, 32usize, 32usize, 48usize, 32usize),
            (2, 32, 48, 32, 64),
            (7, 64, 32, 64, 48),
            (18, 64, 64, 48, 64),
            (35, 64, 64, 64, 64),
        ] {
            let x_data: Vec<f32> = (0..m * k).map(|_| rand_f()).collect();
            let w1_data: Vec<i8> = (0..n1 * k).map(|_| ((rand_f() * 40.0) as i32).clamp(-127, 127) as i8).collect();
            let w2_data: Vec<i8> = (0..n2 * k).map(|_| ((rand_f() * 40.0) as i32).clamp(-127, 127) as i8).collect();
            let w3_data: Vec<i8> = (0..n3 * k).map(|_| ((rand_f() * 40.0) as i32).clamp(-127, 127) as i8).collect();
            let b1_data: Vec<f32> = (0..n1).map(|_| rand_f() * 0.1).collect();
            let b2_data: Vec<f32> = (0..n2).map(|_| rand_f() * 0.1).collect();
            let b3_data: Vec<f32> = (0..n3).map(|_| rand_f() * 0.1).collect();

            let x = Tensor2::from_vec([m, k], x_data);
            let rs1 = crate::engine::tensor::simd::compute_row_sums(&w1_data, n1, k).into_owned();
            let rs2 = crate::engine::tensor::simd::compute_row_sums(&w2_data, n2, k).into_owned();
            let rs3 = crate::engine::tensor::simd::compute_row_sums(&w3_data, n3, k).into_owned();
            let w1 = Quant8Tensor { shape: vec![n1, k], data: w1_data.into(), quant_mult: 64.0, row_sums: Some(rs1) };
            let w2 = Quant8Tensor { shape: vec![n2, k], data: w2_data.into(), quant_mult: 64.0, row_sums: Some(rs2) };
            let w3 = Quant8Tensor { shape: vec![n3, k], data: w3_data.into(), quant_mult: 64.0, row_sums: Some(rs3) };
            let bias1 = Tensor2::from_vec([1, n1], b1_data);
            let bias2 = Tensor2::from_vec([1, n2], b2_data);
            let bias3 = Tensor2::from_vec([1, n3], b3_data);

            let qa = 48.0f32;

            let mut t_out1 = Tensor2::zeros([m, n1]);
            let mut t_out2 = Tensor2::zeros([m, n2]);
            let mut t_out3 = Tensor2::zeros([m, n3]);
            x.quantized_matmul_triple_into(
                &w1, Some(&bias1), &mut t_out1,
                &w2, Some(&bias2), &mut t_out2,
                &w3, Some(&bias3), &mut t_out3,
                qa,
            );

            let mut ind_out1 = Tensor2::zeros([m, n1]);
            let mut ind_out2 = Tensor2::zeros([m, n2]);
            let mut ind_out3 = Tensor2::zeros([m, n3]);
            x.quantized_matmul_biased_into(&w1, qa, Some(&bias1), &mut ind_out1);
            x.quantized_matmul_biased_into(&w2, qa, Some(&bias2), &mut ind_out2);
            x.quantized_matmul_biased_into(&w3, qa, Some(&bias3), &mut ind_out3);

            assert_eq!(t_out1.data, ind_out1.data, "triple out1 mismatch at m={m} k={k} n1={n1}");
            assert_eq!(t_out2.data, ind_out2.data, "triple out2 mismatch at m={m} k={k} n2={n2}");
            assert_eq!(t_out3.data, ind_out3.data, "triple out3 mismatch at m={m} k={k} n3={n3}");
        }
    }
}
