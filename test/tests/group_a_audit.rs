use std::borrow::Cow;
use std::collections::HashSet;
use std::fs;
use std::thread;

use clap::Parser;
use mtranserver_lib::engine::markup::split::{
    split_long_sentence_semantic_with_profile, split_sentences,
};
use mtranserver_lib::engine::markup::{
    extract_inline_markup, extract_unified_inline, InlineParseOptions,
};
use mtranserver_lib::engine::rules::detect::{
    detect_text_language, detect_text_language_with_confidence,
};
use mtranserver_lib::engine::sentencepiece::{
    normalize_nmt_nfkc, normalize_nmt_nfkc_into,
};
use mtranserver_lib::engine::shortlist::{Shortlist, ShortlistPolicy};
use mtranserver_lib::engine::tensor::{ops, Tensor2};
use mtranserver_lib::service::detector::InlineLang;
use mtranserver_lib::utils::cache::{
    clear, compute_key, contains, get, init, invalidate, len, set,
};

// ===========================================================================
// Stage 1 Audit Tests (Cache, Language Detection, Bounds)
// ===========================================================================

#[test]
fn test_cache_block_hash_boundary_lengths() {
    let lengths = [
        0, 1, 2, 3, 7, 8, 9, 14, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
    ];
    let mut keys = HashSet::new();

    for &len in &lengths {
        let text_a = "a".repeat(len);
        let k_a = compute_key(&[&text_a]);
        assert!(
            keys.insert(k_a),
            "Hash collision for repeated 'a' at length {}",
            len
        );

        if len > 0 {
            let text_zero = "\0".repeat(len);
            let k_zero = compute_key(&[&text_zero]);
            assert!(
                keys.insert(k_zero),
                "Hash collision for repeated '\\0' at length {}",
                len
            );

            let mut seq_bytes = vec![0u8; len];
            for (i, b) in seq_bytes.iter_mut().enumerate() {
                *b = (i % 251 + 1) as u8;
            }
            let seq_str = String::from_utf8_lossy(&seq_bytes);
            let k_seq = compute_key(&[&seq_str]);
            assert!(
                keys.insert(k_seq),
                "Hash collision for sequence at length {}",
                len
            );
        }
    }
}

#[test]
fn test_cache_block_hash_argument_boundaries() {
    let k1 = compute_key(&["foo", "bar"]);
    let k2 = compute_key(&["foobar"]);
    assert_ne!(k1, k2, "compute_key must distinguish delimiter boundaries");

    let k3 = compute_key(&["", "foobar"]);
    let k4 = compute_key(&["foobar", ""]);
    assert_ne!(k3, k4);
    assert_ne!(k1, k3);

    let k_empty = compute_key(&[]);
    let k_one_empty = compute_key(&[""]);
    let k_two_empty = compute_key(&["", ""]);
    assert_ne!(k_empty, k_one_empty);
    assert_ne!(k_one_empty, k_two_empty);
}

#[test]
fn test_cache_concurrency_stress_rwlock() {
    if !mtranserver_lib::config::is_initialized() {
        let cli =
            mtranserver_lib::config::Cli::try_parse_from(["mtranserver", "--cache-size", "512"])
                .unwrap();
        mtranserver_lib::config::init(cli);
    }
    init();
    clear();

    const NUM_THREADS: usize = 8;
    const NUM_ITERS: usize = 2000;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|tid| {
            thread::spawn(move || {
                for i in 0..NUM_ITERS {
                    let key_str = format!("key_{}_{}", tid, i % 64);
                    let val_str = format!("val_{}_{}", tid, i);
                    let k = compute_key(&["en", "zh", &key_str]);

                    match i % 4 {
                        0 => {
                            set(k, val_str);
                        }
                        1 => {
                            let _ = get(k);
                        }
                        2 => {
                            let _ = contains(k);
                        }
                        3 => {
                            if i % 16 == 3 {
                                invalidate(k);
                            } else {
                                let _ = get(k);
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Worker thread panicked under cache concurrency");
    }

    assert!(len() <= 512);
}

#[test]
fn test_inline_lang_utf8_safety_and_boundaries() {
    let l1 = InlineLang::new("zh-Hans");
    assert_eq!(l1.as_str(), "zh-Hans");
    assert_eq!(l1.len(), 7);

    let l15 = InlineLang::new("123456789012345");
    assert_eq!(l15.as_str(), "123456789012345");
    assert_eq!(l15.len(), 15);

    let l16 = InlineLang::new("1234567890123456");
    assert_eq!(l16.as_str(), "123456789012345");
    assert_eq!(l16.len(), 15);

    let text_2byte = "12345678901234ü";
    let l_2byte = InlineLang::new(text_2byte);
    let s_2byte = l_2byte.as_str();
    assert!(
        std::str::from_utf8(s_2byte.as_bytes()).is_ok(),
        "InlineLang produced invalid UTF-8 for 2-byte char"
    );
    assert_eq!(s_2byte, "12345678901234");

    let text_4byte = "12345678901234🔥";
    let l_4byte = InlineLang::new(text_4byte);
    let s_4byte = l_4byte.as_str();
    assert!(
        std::str::from_utf8(s_4byte.as_bytes()).is_ok(),
        "InlineLang produced invalid UTF-8 for 4-byte emoji"
    );
    assert_eq!(s_4byte, "12345678901234");

    let text_3byte = "1234567890123中";
    let l_3byte = InlineLang::new(text_3byte);
    let s_3byte = l_3byte.as_str();
    assert!(
        std::str::from_utf8(s_3byte.as_bytes()).is_ok(),
        "InlineLang produced invalid UTF-8 for 3-byte char"
    );
    assert_eq!(s_3byte, "1234567890123");
}

#[test]
fn test_whatlang_128_char_truncation_utf8_boundaries() {
    let cjk_127: String = "中".repeat(127);
    let cjk_128: String = "中".repeat(128);
    let cjk_129: String = "中".repeat(129);
    let cjk_200: String = "中".repeat(200);

    assert_eq!(detect_text_language(&cjk_127), "zh-Hans");
    assert_eq!(detect_text_language(&cjk_128), "zh-Hans");
    assert_eq!(detect_text_language(&cjk_129), "zh-Hans");
    assert_eq!(detect_text_language(&cjk_200), "zh-Hans");

    let ru_text = "Это длинный текст на русском языке для проверки корректности обрезки строк без паники при детекции языка через whatlang 128 символов.";
    let ru_long: String = ru_text.repeat(3);
    assert_eq!(detect_text_language(&ru_long), "ru");
}

#[test]
fn test_tier0_fast_bypass_natural_language_collision() {
    let (c2, _) = detect_text_language_with_confidence("El cargo público oficial del ministro en el gobierno español.");
    let (c3, _) = detect_text_language_with_confidence("git buradan hemen ve bir daha asla geri dönme");
    let (c4, _) = detect_text_language_with_confidence("Hon tar bussen till skolan varje morgon.");
    let (c5, _) = detect_text_language_with_confidence("var du der i går?");
    let (c6, _) = detect_text_language_with_confidence("import et export de marchandises en France");

    assert_eq!(c2, "es");
    assert_eq!(c3, "tr");
    assert_eq!(c4, "sv");
    assert!(c5 == "da" || c5 == "nb" || c5 == "sv");
    assert_eq!(c6, "fr");
}

// ===========================================================================
// Stage 2 Audit Tests: AVX2 LayerNorm, SentencePiece, Shortlist, Markup
// ===========================================================================

fn lcg_rand(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32 / (u32::MAX as f32)) - 0.5
}

#[test]
fn test_audit_step2_layernorm_dimensions_and_numerical_stability() {
    let dimensions = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 128, 256, 384, 512, 768, 1024];
    let eps = 1e-5f32;
    let mut rng = 88888u64;

    for &d in &dimensions {
        // 1. Standard pseudo-random inputs
        let a_rand: Vec<f32> = (0..d).map(|_| lcg_rand(&mut rng) * 10.0).collect();
        let b_rand: Vec<f32> = (0..d).map(|_| lcg_rand(&mut rng) * 10.0).collect();
        let scale: Vec<f32> = (0..d).map(|_| lcg_rand(&mut rng).abs() + 0.5).collect();
        let bias: Vec<f32> = (0..d).map(|_| lcg_rand(&mut rng)).collect();

        // Scalar reference
        let mut out_scalar = vec![0.0f32; d];
        unsafe {
            ops::fused_add_layer_norm_scalar_raw(
                a_rand.as_ptr(),
                b_rand.as_ptr(),
                scale.as_ptr(),
                bias.as_ptr(),
                eps,
                out_scalar.as_mut_ptr(),
                d,
            );
        }

        // AVX2 raw kernel
        #[cfg(target_arch = "x86_64")]
        if mtranserver_lib::utils::arch::has_avx2() {
            let mut out_avx2 = vec![0.0f32; d];
            unsafe {
                ops::fused_add_layer_norm_avx2_raw(
                    a_rand.as_ptr(),
                    b_rand.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_avx2.as_mut_ptr(),
                    d,
                );
            }

            for i in 0..d {
                assert!(!out_avx2[i].is_nan(), "AVX2 LayerNorm produced NaN at d={d}, idx={i}");
                assert!(!out_avx2[i].is_infinite(), "AVX2 LayerNorm produced Inf at d={d}, idx={i}");
                let diff = (out_avx2[i] - out_scalar[i]).abs();
                let denom = out_scalar[i].abs().max(1e-5);
                let rel = diff / denom;
                assert!(
                    rel < 5e-6,
                    "AVX2 vs Scalar mismatch at d={d}, idx={i}: avx2={}, scalar={}, diff={}, rel={}",
                    out_avx2[i],
                    out_scalar[i],
                    diff,
                    rel
                );
            }
        }

        // High-level Tensor2 API: out-of-place vs in-place equivalence
        let t_a = Tensor2::from_vec([1, d], a_rand.clone());
        let t_b = Tensor2::from_vec([1, d], b_rand.clone());
        let t_scale = Tensor2::from_vec([1, d], scale.clone());
        let t_bias = Tensor2::from_vec([1, d], bias.clone());

        let mut t_out = Tensor2::zeros([1, d]);
        ops::fused_add_layer_norm_into(&t_a, &t_b, &t_scale, &t_bias, eps, &mut t_out);

        let mut t_inplace = t_a.clone();
        ops::fused_add_layer_norm_in_place(&mut t_inplace, &t_b, &t_scale, &t_bias, eps);

        for i in 0..d {
            let diff = (t_out.data[i] - t_inplace.data[i]).abs();
            assert!(
                diff < 1e-6,
                "Inplace vs into mismatch at d={d}, idx={i}: into={}, inplace={}",
                t_out.data[i],
                t_inplace.data[i]
            );
        }

        // 2. Extreme pattern: All Zeros
        let a_zeros = vec![0.0f32; d];
        let b_zeros = vec![0.0f32; d];
        let mut out_zeros = vec![0.0f32; d];
        #[cfg(target_arch = "x86_64")]
        if mtranserver_lib::utils::arch::has_avx2() {
            unsafe {
                ops::fused_add_layer_norm_avx2_raw(
                    a_zeros.as_ptr(),
                    b_zeros.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_zeros.as_mut_ptr(),
                    d,
                );
            }
            for i in 0..d {
                assert!(!out_zeros[i].is_nan(), "All zeros caused NaN at d={d}, idx={i}");
                // Mean is 0, variance is 0, (x - mean) * inv * scale + bias = bias[i]
                let diff = (out_zeros[i] - bias[i]).abs();
                assert!(diff < 1e-5, "All zeros must yield bias at d={d}, idx={i}");
            }
        }

        // 3. Extreme pattern: Identical constant values (Zero variance condition)
        let a_const = vec![5432.1f32; d];
        let b_const = vec![1234.5f32; d];
        let mut out_const = vec![0.0f32; d];
        #[cfg(target_arch = "x86_64")]
        if mtranserver_lib::utils::arch::has_avx2() {
            unsafe {
                ops::fused_add_layer_norm_avx2_raw(
                    a_const.as_ptr(),
                    b_const.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_const.as_mut_ptr(),
                    d,
                );
            }
            for i in 0..d {
                assert!(!out_const[i].is_nan(), "Identical constants caused NaN at d={d}, idx={i}");
                let diff = (out_const[i] - bias[i]).abs();
                assert!(diff < 1e-4, "Identical constants must yield bias at d={d}, idx={i}");
            }
        }

        // 4. Extreme pattern: Near-identical values with tiny variance (Negative variance rounding defense)
        let a_near: Vec<f32> = (0..d).map(|i| 10_000.0 + (i as f32) * 1e-6).collect();
        let b_near = vec![5_000.0f32; d];
        let mut out_near = vec![0.0f32; d];
        #[cfg(target_arch = "x86_64")]
        if mtranserver_lib::utils::arch::has_avx2() {
            unsafe {
                ops::fused_add_layer_norm_avx2_raw(
                    a_near.as_ptr(),
                    b_near.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_near.as_mut_ptr(),
                    d,
                );
            }
            for i in 0..d {
                assert!(!out_near[i].is_nan(), "Near-identical values caused NaN at d={d}, idx={i}");
                assert!(!out_near[i].is_infinite(), "Near-identical values caused Inf at d={d}, idx={i}");
            }
        }

        // 5. Extreme pattern: Large magnitudes (f64 accumulator overflow defense)
        let a_large = vec![1e7f32; d];
        let b_large = vec![2e7f32; d];
        let mut out_large = vec![0.0f32; d];
        #[cfg(target_arch = "x86_64")]
        if mtranserver_lib::utils::arch::has_avx2() {
            unsafe {
                ops::fused_add_layer_norm_avx2_raw(
                    a_large.as_ptr(),
                    b_large.as_ptr(),
                    scale.as_ptr(),
                    bias.as_ptr(),
                    eps,
                    out_large.as_mut_ptr(),
                    d,
                );
            }
            for i in 0..d {
                assert!(!out_large[i].is_nan(), "Large values caused NaN at d={d}, idx={i}");
                assert!(!out_large[i].is_infinite(), "Large values caused Inf at d={d}, idx={i}");
            }
        }
    }
}

#[test]
fn test_audit_step2_sentencepiece_normalization_equivalence() {
    let test_cases = [
        // Pure printable ASCII
        "The quick brown fox jumps over the lazy dog 1234567890!@#$%^&*()",
        // Fullwidth ASCII characters
        "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？",
        "＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿",
        "｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
        // Unicode whitespace variants
        "space\u{00A0}nbsp\u{1680}ogham\u{2000}enquad\u{2003}emspace\u{202F}nnbsp\u{205F}mosp\u{3000}ideospace",
        // Filtered control chars (soft hyphen, zero-width spaces, BOM, directional marks)
        "pre\u{00AD}fix\u{200B}mid\u{200C}join\u{200D}end\u{FEFF}bom\u{202A}dir\u{202C}ok",
        // Multi-byte CJK & Emojis
        "你好世界，这是一个基于 Rust 的快速神经机器翻译系统！🚀🔥🎉",
        "こんにちは世界！ひらがな、カタカナ、漢字が混在するテキストです。",
        "안녕하세요 세계! 한국어 번역 테스트 문자열입니다.",
        // Mixed text
        "Fullwidth：【Ｈｅｌｌｏ】　ＮＢＳＰ\u{00A0}and\u{3000}soft\u{00AD}hyphen！",
        // Empty string
        "",
        // Single character edge cases
        "a",
        "\u{00A0}",
        "\u{00AD}",
        "！",
    ];

    let mut reusable_buf = String::new();

    for &text in &test_cases {
        let cow_res = normalize_nmt_nfkc(text);
        let modified = normalize_nmt_nfkc_into(text, &mut reusable_buf);

        if modified {
            assert!(
                matches!(cow_res, Cow::Owned(_)),
                "normalize_nmt_nfkc must return Cow::Owned when modified is true for text: {:?}",
                text
            );
            assert_eq!(
                cow_res.as_ref(),
                reusable_buf.as_str(),
                "Mismatch between normalize_nmt_nfkc and normalize_nmt_nfkc_into for: {:?}",
                text
            );
        } else {
            assert!(
                matches!(cow_res, Cow::Borrowed(_)),
                "normalize_nmt_nfkc must return Cow::Borrowed when modified is false for text: {:?}",
                text
            );
            assert_eq!(
                cow_res.as_ref(),
                text,
                "Unmodified text must borrow original for: {:?}",
                text
            );
        }
    }

    // Scratch reuse stress: ensure repeated calls do not accumulate or leak state
    reusable_buf.clear();
    for _ in 0..10 {
        normalize_nmt_nfkc_into("全角ＡＢＣ", &mut reusable_buf);
        assert_eq!(reusable_buf, "全角ABC");
        normalize_nmt_nfkc_into("Plain ASCII", &mut reusable_buf);
        // Plain ASCII returns false and does not clear or touch norm_buf
        normalize_nmt_nfkc_into("１２３", &mut reusable_buf);
        assert_eq!(reusable_buf, "123");
    }
}

#[test]
fn test_audit_step2_shortlist_streaming_and_candidate_generation() {
    // 1. Construct valid binary shortlist format (.s2t.bin)
    // Magic: 0xF11A48D5013417F5 (u64 le)
    let magic = 0xF11A48D5013417F5u64;
    let checksum = 0u64;
    let first_num = 50u64;
    let best_num = 20u64;
    let word_to_offset = vec![0u64, 2u64, 5u64, 5u64];
    let short_lists = vec![100u32, 101u32, 200u32, 201u32, 202u32];

    let mut bin = Vec::new();
    bin.extend_from_slice(&magic.to_le_bytes());
    bin.extend_from_slice(&checksum.to_le_bytes());
    bin.extend_from_slice(&first_num.to_le_bytes());
    bin.extend_from_slice(&best_num.to_le_bytes());
    bin.extend_from_slice(&(word_to_offset.len() as u64).to_le_bytes());
    bin.extend_from_slice(&(short_lists.len() as u64).to_le_bytes());

    for &w in &word_to_offset {
        bin.extend_from_slice(&w.to_le_bytes());
    }
    for &s in &short_lists {
        bin.extend_from_slice(&s.to_le_bytes());
    }

    let sl_from_bytes = Shortlist::from_bytes(&bin).expect("from_bytes failed");
    assert_eq!(sl_from_bytes.first_num(), first_num);
    assert_eq!(sl_from_bytes.best_num(), best_num);
    assert_eq!(sl_from_bytes.word_to_offset(), word_to_offset.as_slice());
    assert_eq!(sl_from_bytes.short_lists(), short_lists.as_slice());

    // Write to a temporary file and verify Shortlist::load streaming path
    let temp_dir = std::env::temp_dir();
    let temp_file = temp_dir.join(format!("test_shortlist_{}.bin", std::process::id()));
    fs::write(&temp_file, &bin).expect("Failed to write temp shortlist");

    let sl_from_file = Shortlist::load(&temp_file).expect("Shortlist::load failed");
    assert_eq!(sl_from_file.first_num(), first_num);
    assert_eq!(sl_from_file.best_num(), best_num);
    assert_eq!(sl_from_file.word_to_offset(), word_to_offset.as_slice());
    assert_eq!(sl_from_file.short_lists(), short_lists.as_slice());

    let _ = fs::remove_file(&temp_file);

    // 2. Error handling: invalid magic, truncated file
    let mut bad_bin = bin.clone();
    bad_bin[0] = 0xFF;
    assert!(Shortlist::from_bytes(&bad_bin).is_err());

    let truncated_file = temp_dir.join(format!("test_shortlist_trunc_{}.bin", std::process::id()));
    fs::write(&truncated_file, &bin[..20]).unwrap();
    assert!(Shortlist::load(&truncated_file).is_err());
    let _ = fs::remove_file(&truncated_file);

    // 3. Candidate generation correctness and zeroed state
    let mut seen = Vec::new();
    let mut out = Vec::new();

    // Source token 0 maps to [100, 101]
    sl_from_bytes.generate_into(&[0], 500, &mut seen, &mut out, ShortlistPolicy::Strict);
    // first_num (50) + 2 candidate tokens
    assert_eq!(out.len(), 50 + 2);
    assert!(out.contains(&100));
    assert!(out.contains(&101));
    // Verify seen is left clean (all zeros)
    assert!(seen.iter().all(|&w| w == 0), "Seen buffer was not cleanly zeroed");

    // Repeated source token deduplication
    out.clear();
    sl_from_bytes.generate_into(&[0, 0, 0], 500, &mut seen, &mut out, ShortlistPolicy::Strict);
    assert_eq!(out.len(), 50 + 2);
    assert!(seen.iter().all(|&w| w == 0));

    // Empty source IDs fallback
    out.clear();
    sl_from_bytes.generate_into(&[], 500, &mut seen, &mut out, ShortlistPolicy::Strict);
    assert_eq!(out.len(), first_num as usize);
}

#[test]
fn test_audit_step2_markup_inline_and_split_equivalence() {
    let options = InlineParseOptions {
        parse_html: true,
        parse_markdown: true,
    };

    let test_inputs = [
        "Plain text without any markup triggers whatsoever.",
        "Simple <b>bold</b> and <i>italic</i> tags.",
        "Void tag <img src=\"test.png\" alt=\"image\"/> and <br> separator.",
        "Nested tags: <div><p>Paragraph with <b>bold <i>and italic</i></b> text.</p></div>",
        "Markdown links [Rust](https://rust-lang.org) and `inline_code()` with **bold** text.",
        "Markdown strikethrough ~~deleted text~~ and *single asterisk italic*.",
        "False markup triggers: 3 < 5 && 7 > 2, price * 1.2 = total, array[0] = 1, foo_bar_baz.",
        "Unterminated tags: <b>unclosed tag text continues here.",
        "Multilingual with markup: <p>这是一个<b>重要</b>的通知：请查看[链接](http://example.com)！</p>",
    ];

    for &input in &test_inputs {
        let (plain, spans) = extract_unified_inline(input);
        let (plain_opt, spans_opt) = extract_inline_markup(input, options);
        assert_eq!(plain, plain_opt);
        assert_eq!(spans, spans_opt);
        // Verification 1: plain text must be valid UTF-8 and non-empty for non-empty text
        assert!(!plain.is_empty(), "Plain text was unexpectedly empty for: {:?}", input);

        // Verification 2: spans offsets must lie within plain text bounds
        for span in &spans {
            assert!(
                span.source_range.start <= plain.len(),
                "Span start {} > plain len {} for input: {:?}",
                span.source_range.start,
                plain.len(),
                input
            );
            assert!(
                span.source_range.end <= plain.len(),
                "Span end {} > plain len {} for input: {:?}",
                span.source_range.end,
                plain.len(),
                input
            );
            assert!(
                span.source_range.start <= span.source_range.end,
                "Inverted span range {:?} for input: {:?}",
                span.source_range,
                input
            );
        }
    }

    // Sentence splitting regression check
    let paragraph = "Hello world. How are you? I'm fine! Thank you.";
    let sents = split_sentences(paragraph);
    assert_eq!(sents.len(), 4);
    assert_eq!(sents[0].text, "Hello world.");
    assert_eq!(sents[1].text, "How are you?");
    assert_eq!(sents[2].text, "I'm fine!");
    assert_eq!(sents[3].text, "Thank you.");

    // Long sentence semantic splitting check
    let long_text = "This is a very long segment that definitely exceeds the maximum chunk length and needs to be split semantically at punctuation or conjunction boundaries.";
    let split_chunks = split_long_sentence_semantic_with_profile(long_text, 40, None);
    assert!(split_chunks.len() >= 2);
    let reconstructed: String = split_chunks.iter().map(|s| format!("{}{}", s.text, s.separator)).collect();
    assert_eq!(reconstructed, long_text);
}
// ===========================================================================
// Stage 3 Audit Tests: API Protocol, Plugins, Zero-Copy & Error Compatibility
// ===========================================================================

use mtranserver_lib::api::plugins::deepl::{DeeplRequest, DeeplResponse};
use mtranserver_lib::api::plugins::deeplx::{DeeplxRequest, DeeplxResponse};
use mtranserver_lib::api::plugins::google::{GoogleRequest, GoogleResponse};
use mtranserver_lib::api::plugins::hcfy::{HcfyRequest, HcfyResponse};
use mtranserver_lib::api::plugins::imme::{ImmeRequest, ImmeResponse};
use mtranserver_lib::api::plugins::kiss::{KissRequest, KissResponse};
use mtranserver_lib::api::plugins::{parse_source_lang, parse_target_lang, OneOrMany};
use mtranserver_lib::api::translate::validate_batch;

#[test]
fn test_audit_step3_static_languages_and_system() {
    let list1 = mtranserver_lib::models::get_language_list_response();
    let list2 = mtranserver_lib::models::get_language_list_response();
    assert!(std::sync::Arc::ptr_eq(&list1, &list2));
    let json = serde_json::to_string(&list1).expect("serialize LanguageListResponse");
    assert!(json.contains("\"languages\":"));
    assert!(json.contains("\"pairs\":"));
}

#[test]
fn test_audit_step3_one_or_many_deserialization() {
    let one_json = r#""single text""#;
    let one: OneOrMany<String> = serde_json::from_str(one_json).unwrap();
    assert_eq!(one.as_slice(), &["single text"]);
    assert_eq!(one, OneOrMany::One("single text".to_string()));

    let many_json = r#"["text 1", "text 2"]"#;
    let many: OneOrMany<String> = serde_json::from_str(many_json).unwrap();
    assert_eq!(many.as_slice(), &["text 1", "text 2"]);
    assert_eq!(
        many,
        OneOrMany::Many(vec!["text 1".to_string(), "text 2".to_string()])
    );

    let empty_many_json = r#"[]"#;
    let empty_many: OneOrMany<String> = serde_json::from_str(empty_many_json).unwrap();
    assert_eq!(empty_many.as_slice(), &[] as &[String]);

    assert!(serde_json::from_str::<OneOrMany<String>>(r#"123"#).is_err());
    assert!(serde_json::from_str::<OneOrMany<String>>(r#"{"text": "val"}"#).is_err());
}

#[test]
fn test_audit_step3_deepl_protocol_compatibility() {
    let json_single = r#"{
        "text": "Hello world",
        "target_lang": "ZH",
        "tag_handling": "xml"
    }"#;
    let req: DeeplRequest = serde_json::from_str(json_single).unwrap();
    assert_eq!(req.text.as_slice(), &["Hello world"]);
    assert_eq!(req.source_lang, None);
    assert_eq!(req.target_lang, "ZH");
    assert_eq!(req.tag_handling.as_deref(), Some("xml"));

    let json_multi = r#"{
        "text": ["Hello", "World"],
        "source_lang": "EN",
        "target_lang": "JA"
    }"#;
    let req_multi: DeeplRequest = serde_json::from_str(json_multi).unwrap();
    assert_eq!(req_multi.text.as_slice(), &["Hello", "World"]);
    assert_eq!(req_multi.source_lang.as_deref(), Some("EN"));
    assert_eq!(req_multi.target_lang, "JA");
    assert_eq!(req_multi.tag_handling, None);

    let resp = DeeplResponse {
        translations: vec![
            mtranserver_lib::api::plugins::deepl::DeeplTranslation {
                detected_source_language: Cow::Borrowed("EN"),
                text: "你好".to_string(),
            },
            mtranserver_lib::api::plugins::deepl::DeeplTranslation {
                detected_source_language: Cow::Borrowed("EN"),
                text: "世界".to_string(),
            },
        ],
    };
    let serialized = serde_json::to_string(&resp).unwrap();
    assert_eq!(
        serialized,
        r#"{"translations":[{"detected_source_language":"EN","text":"你好"},{"detected_source_language":"EN","text":"世界"}]}"#
    );
}
#[test]
fn test_audit_step3_deeplx_protocol_compatibility() {
    let json = r#"{"text":"DeepLX test","source_lang":"EN","target_lang":"ZH"}"#;
    let req: DeeplxRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.text, "DeepLX test");
    assert_eq!(req.source_lang, "EN");
    assert_eq!(req.target_lang, "ZH");

    let json_null = r#"{"text":"test","source_lang":null,"target_lang":"ZH"}"#;
    let req_null: DeeplxRequest = serde_json::from_str(json_null).unwrap();
    assert_eq!(req_null.source_lang, "");

    let resp = DeeplxResponse {
        alternatives: &[],
        code: 200,
        data: "测试".to_string(),
        id: 123456789,
        method: "Free",
        source_lang: Cow::Borrowed("EN"),
        target_lang: Cow::Borrowed("ZH"),
    };
    let json_resp = serde_json::to_string(&resp).unwrap();
    assert!(json_resp.contains(r#""code":200"#));
    assert!(json_resp.contains(r#""data":"测试""#));
    assert!(json_resp.contains(r#""method":"Free""#));
    assert!(json_resp.contains(r#""alternatives":[]"#));
    assert!(json_resp.contains(r#""id":123456789"#));
}

#[test]
fn test_audit_step3_google_protocol_compatibility() {
    let json_single = r#"{"q":"Hello","target":"zh"}"#;
    let req_single: GoogleRequest = serde_json::from_str(json_single).unwrap();
    assert_eq!(req_single.q.as_slice(), &["Hello"]);
    assert_eq!(req_single.source, "");
    assert_eq!(req_single.target, "zh");

    let json_multi = r#"{"q":["Hello","World"],"source":"en","target":"zh"}"#;
    let req_multi: GoogleRequest = serde_json::from_str(json_multi).unwrap();
    assert_eq!(req_multi.q.as_slice(), &["Hello", "World"]);
    assert_eq!(req_multi.source, "en");
    assert_eq!(req_multi.target, "zh");

    let resp = GoogleResponse {
        data: mtranserver_lib::api::plugins::google::GoogleTranslateData {
            translations: vec![
                mtranserver_lib::api::plugins::google::GoogleTranslation {
                    translatedText: "你好".to_string(),
                    detectedSourceLanguage: Some("en".to_string()),
                },
            ],
        },
    };
    let serialized = serde_json::to_string(&resp).unwrap();
    assert_eq!(
        serialized,
        r#"{"data":{"translations":[{"translatedText":"你好","detectedSourceLanguage":"en"}]}}"#
    );

}

#[test]
fn test_audit_step3_hcfy_protocol_compatibility() {
    let json = r#"{"text":"Hello world","source":"英语","destination":["中文(简体)"]}"#;
    let req: HcfyRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.text, "Hello world");
    assert_eq!(req.source.as_deref(), Some("英语"));
    assert_eq!(req.destination, vec!["中文(简体)".to_string()]);

    let json_empty = r#"{"text":"","destination":[]}"#;
    let req_empty: HcfyRequest = serde_json::from_str(json_empty).unwrap();
    assert_eq!(req_empty.source, None);
    assert!(req_empty.destination.is_empty());

    let resp = HcfyResponse {
        text: "你好世界".to_string(),
        from: "英语".to_string(),
        to: "中文(简体)".to_string(),
        result: None,
    };
    let serialized = serde_json::to_string(&resp).unwrap();
    assert_eq!(
        serialized,
        r#"{"text":"你好世界","from":"英语","to":"中文(简体)"}"#
    );
}
#[test]
fn test_audit_step3_kiss_protocol_compatibility() {
    let json_text = r#"{"text":"Hello","from":"en","to":"zh"}"#;
    let req: KissRequest = serde_json::from_str(json_text).unwrap();
    assert_eq!(req.text.as_deref(), Some("Hello"));
    assert_eq!(req.texts, None);
    assert_eq!(req.from, "en");
    assert_eq!(req.to, "zh");

    let json_texts = r#"{"texts":["A","B"],"from":null,"to":"zh"}"#;
    let req2: KissRequest = serde_json::from_str(json_texts).unwrap();
    assert_eq!(req2.text, None);
    assert_eq!(
        req2.texts,
        Some(vec!["A".to_string(), "B".to_string()])
    );
    assert_eq!(req2.from, "");

    let resp_single = KissResponse {
        text: Some("你好".to_string()),
        src: Some("en".to_string()),
        translations: None,
    };
    let s_single = serde_json::to_string(&resp_single).unwrap();
    assert_eq!(s_single, r#"{"text":"你好","src":"en"}"#);

    let resp_multi = KissResponse {
        text: None,
        src: None,
        translations: Some(vec![
            mtranserver_lib::api::plugins::kiss::KissTranslation {
                text: "甲".to_string(),
                src: "en".to_string(),
            },
        ]),
    };
    let s_multi = serde_json::to_string(&resp_multi).unwrap();
    assert_eq!(s_multi, r#"{"translations":[{"text":"甲","src":"en"}]}"#);
}

#[test]
fn test_audit_step3_imme_protocol_compatibility() {
    let json = r#"{"text_list":["Hello","World"],"source_lang":"en","target_lang":"zh"}"#;
    let req: ImmeRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.text_list, vec!["Hello", "World"]);
    assert_eq!(req.source_lang, "en");
    assert_eq!(req.target_lang, "zh");

    let json_null = r#"{"text_list":[],"source_lang":null,"target_lang":"zh"}"#;
    let req_null: ImmeRequest = serde_json::from_str(json_null).unwrap();
    assert_eq!(req_null.source_lang, "");

    let resp = ImmeResponse {
        translations: vec![
            mtranserver_lib::api::plugins::imme::ImmeTranslation {
                detected_source_lang: "en".to_string(),
                text: "你好".to_string(),
            },
        ],
    };
    let s = serde_json::to_string(&resp).unwrap();
    assert_eq!(
        s,
        r#"{"translations":[{"detected_source_lang":"en","text":"你好"}]}"#
    );
}

#[test]
fn test_audit_step3_language_and_batch_validation() {
    assert_eq!(parse_source_lang(None), "auto");
    assert_eq!(parse_source_lang(Some("")), "auto");
    assert_eq!(parse_source_lang(Some("auto")), "auto");
    assert_eq!(parse_source_lang(Some("AUTO")), "auto");
    assert_eq!(parse_source_lang(Some("  en  ")), "en");
    assert_eq!(parse_source_lang(Some("zh-cn")), "zh-Hans");
    assert_eq!(parse_source_lang(Some("zh-hans")), "zh-Hans");

    assert!(parse_target_lang("", "to").is_err());
    assert!(parse_target_lang("   ", "to").is_err());
    let ok_lang = parse_target_lang("zh-hant", "to").unwrap();
    assert_eq!(ok_lang, "zh-Hant");

    let safe_texts = vec!["a".repeat(1000); 10];
    assert!(validate_batch(&safe_texts).is_ok());

    let over_chars = vec!["a".repeat(35000), "b".repeat(35000)];
    assert!(validate_batch(&over_chars).is_err());
}
