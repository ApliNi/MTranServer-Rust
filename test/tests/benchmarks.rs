use std::path::{Path, PathBuf};
use std::time::Instant;

use mtranserver_lib::engine::markup::document::parse_document_blocks;
use mtranserver_lib::engine::markup::split::split_sentences;
use mtranserver_lib::engine::rules::detect::detect_dominant_language;
use mtranserver_lib::engine::sentencepiece::{DecodeScratch, EncodeScratch, SentencePiece};
use mtranserver_lib::engine::shortlist::{Shortlist, ShortlistPolicy};
use mtranserver_lib::engine::tensor::{Quant8Tensor, Tensor2};
use mtranserver_lib::utils::cache::{compute_key, compute_key_with_version};

fn find_model_file(rel: &str) -> Option<PathBuf> {
    let p1 = Path::new(rel);
    if p1.exists() {
        return Some(p1.to_path_buf());
    }
    let p2 = Path::new("..").join(rel);
    if p2.exists() {
        return Some(p2);
    }
    None
}

#[test]
#[ignore = "Hotspot performance microbenchmark suite"]
fn bench_all_hotspots() {
    println!("\n=======================================================");
    println!("       MTranServer-Rust Performance Baseline Report     ");
    println!("=======================================================");

    bench_sentencepiece();
    bench_tensor_and_simd();
    bench_shortlist();
    bench_markup_and_split();
    bench_language_detection();
    bench_cache();

    println!("=======================================================\n");
}

fn bench_sentencepiece() {
    let spm_path = match find_model_file("data/models/zh-Hans_en/vocab.zhen.spm") {
        Some(p) => p,
        None => {
            println!("[SPM Bench] Skipped: vocab.zhen.spm not found.");
            return;
        }
    };
    let spm = SentencePiece::load(&spm_path).expect("load spm");

    let short_text = "你好，世界！今天天气真好。";
    let med_text = "人工智能与机器学习正在深刻改变各行各业，自然语言处理技术的发展让跨语言沟通变得前所未有的顺畅。无论是在线翻译、语音识别还是大语言模型，都在推动全球化的信息交流。";
    let long_text = med_text.repeat(5);

    let mut scratch = EncodeScratch::default();
    let mut decode_scratch = DecodeScratch::default();
    let mut decode_buf = String::new();

    // Warmup
    for _ in 0..100 {
        let _ = spm.encode(short_text);
        let _ = spm.encode_into(short_text, &mut scratch);
    }

    println!("\n--- [1] SentencePiece Tokenizer Benchmark ---");
    let iters = 2000;

    // Short text encode
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = spm.encode(short_text);
    }
    let d_short_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = spm.encode_into(short_text, &mut scratch);
    }
    let d_short_scratch = t0.elapsed();

    println!(
        "Short ({} chars) encode:\n  Alloc:   {:8.3} µs/op\n  Scratch: {:8.3} µs/op ({:.2}x)",
        short_text.chars().count(),
        d_short_alloc.as_secs_f64() * 1e6 / iters as f64,
        d_short_scratch.as_secs_f64() * 1e6 / iters as f64,
        d_short_alloc.as_nanos() as f64 / d_short_scratch.as_nanos().max(1) as f64
    );

    // Medium text encode
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = spm.encode(med_text);
    }
    let d_med_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = spm.encode_into(med_text, &mut scratch);
    }
    let d_med_scratch = t0.elapsed();

    println!(
        "Med ({} chars) encode:\n  Alloc:   {:8.3} µs/op\n  Scratch: {:8.3} µs/op ({:.2}x)",
        med_text.chars().count(),
        d_med_alloc.as_secs_f64() * 1e6 / iters as f64,
        d_med_scratch.as_secs_f64() * 1e6 / iters as f64,
        d_med_alloc.as_nanos() as f64 / d_med_scratch.as_nanos().max(1) as f64
    );

    // Long text encode
    let long_iters = 500;
    let t0 = Instant::now();
    for _ in 0..long_iters {
        let _ = spm.encode(&long_text);
    }
    let d_long_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..long_iters {
        let _ = spm.encode_into(&long_text, &mut scratch);
    }
    let d_long_scratch = t0.elapsed();

    println!(
        "Long ({} chars) encode:\n  Alloc:   {:8.3} µs/op\n  Scratch: {:8.3} µs/op ({:.2}x)",
        long_text.chars().count(),
        d_long_alloc.as_secs_f64() * 1e6 / long_iters as f64,
        d_long_scratch.as_secs_f64() * 1e6 / long_iters as f64,
        d_long_alloc.as_nanos() as f64 / d_long_scratch.as_nanos().max(1) as f64
    );

    // Decode benchmark
    let token_ids = spm.encode(med_text);
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = spm.decode(&token_ids);
    }
    let d_dec_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        spm.decode_into(&token_ids, &mut decode_scratch, &mut decode_buf);
    }
    let d_dec_scratch = t0.elapsed();

    println!(
        "Med ({} tokens) decode:\n  Alloc:   {:8.3} µs/op\n  Scratch: {:8.3} µs/op ({:.2}x)",
        token_ids.len(),
        d_dec_alloc.as_secs_f64() * 1e6 / iters as f64,
        d_dec_scratch.as_secs_f64() * 1e6 / iters as f64,
        d_dec_alloc.as_nanos() as f64 / d_dec_scratch.as_nanos().max(1) as f64
    );
}

fn bench_tensor_and_simd() {
    println!("\n--- [2] Tensor & SIMD Operations Benchmark ---");

    let d_model = 512;
    let eps = 1e-5;
    let x_single = Tensor2::from_vec([1, d_model], (0..d_model).map(|i| (i as f32) * 0.01).collect());
    let x_batch16 = Tensor2::from_vec([16, d_model], (0..16 * d_model).map(|i| (i as f32) * 0.001).collect());
    let scale = Tensor2::from_vec([1, d_model], vec![1.0; d_model]);
    let bias = Tensor2::from_vec([1, d_model], vec![0.0; d_model]);
    let mut out_single = Tensor2::zeros([1, d_model]);
    let mut out_batch16 = Tensor2::zeros([16, d_model]);

    // LayerNorm 1x512
    let iters = 20_000;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = x_single.layer_norm(&scale, &bias, eps);
    }
    let d_ln_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        x_single.layer_norm_into(&scale, &bias, eps, &mut out_single);
    }
    let d_ln_into = t0.elapsed();

    println!(
        "LayerNorm (1x512):\n  Alloc: {:8.3} µs/op\n  Into:  {:8.3} µs/op ({:.2}x)",
        d_ln_alloc.as_secs_f64() * 1e6 / iters as f64,
        d_ln_into.as_secs_f64() * 1e6 / iters as f64,
        d_ln_alloc.as_nanos() as f64 / d_ln_into.as_nanos().max(1) as f64
    );

    // LayerNorm 16x512
    let iters_16 = 5_000;
    let t0 = Instant::now();
    for _ in 0..iters_16 {
        x_batch16.layer_norm_into(&scale, &bias, eps, &mut out_batch16);
    }
    let d_ln_16 = t0.elapsed();
    println!(
        "LayerNorm (16x512) Into:\n         {:8.3} µs/op",
        d_ln_16.as_secs_f64() * 1e6 / iters_16 as f64
    );

    // Fused Add + LayerNorm vs Add + LayerNorm
    let other = Tensor2::from_vec([1, d_model], vec![0.5; d_model]);
    let mut temp = Tensor2::zeros([1, d_model]);
    let t0 = Instant::now();
    for _ in 0..iters {
        x_single.add_into(&other, &mut temp);
        temp.layer_norm_into(&scale, &bias, eps, &mut out_single);
    }
    let d_separate = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        x_single.fused_add_layer_norm_into(&other, &scale, &bias, eps, &mut out_single);
    }
    let d_fused = t0.elapsed();
    println!(
        "Add + LayerNorm (1x512):\n  Separate: {:8.3} µs/op\n  Fused:    {:8.3} µs/op ({:.2}x)",
        d_separate.as_secs_f64() * 1e6 / iters as f64,
        d_fused.as_secs_f64() * 1e6 / iters as f64,
        d_separate.as_nanos() as f64 / d_fused.as_nanos().max(1) as f64
    );

    // Quantized GEMM 1x512 * 512x512
    let k = 512;
    let n = 512;
    let w_data: Vec<i8> = (0..k * n).map(|i| (i % 255) as i8).collect();
    let q8_512 = Quant8Tensor::new(vec![n, k], w_data, 127.0, None);
    let qa = 1.0;
    let mut gemm_out = Tensor2::zeros([1, n]);

    let gemm_iters = 10_000;
    let t0 = Instant::now();
    for _ in 0..gemm_iters {
        x_single.quantized_matmul_biased_into(&q8_512, qa, None, &mut gemm_out);
    }
    let d_gemm_512 = t0.elapsed();
    println!(
        "QuantGEMM (1x512 * 512x512):\n         {:8.3} µs/op ({:.1} GFLOPS)",
        d_gemm_512.as_secs_f64() * 1e6 / gemm_iters as f64,
        (2.0 * 1.0 * k as f64 * n as f64 * gemm_iters as f64) / (d_gemm_512.as_secs_f64() * 1e9)
    );

    // Quantized GEMM 1x512 * 512x2048 (FFN projection)
    let n_ff = 2048;
    let w_ff_data: Vec<i8> = (0..k * n_ff).map(|i| (i % 255) as i8).collect();
    let q8_ff = Quant8Tensor::new(vec![n_ff, k], w_ff_data, 127.0, None);
    let mut gemm_ff_out = Tensor2::zeros([1, n_ff]);
    let gemm_ff_iters = 5_000;
    let t0 = Instant::now();
    for _ in 0..gemm_ff_iters {
        x_single.quantized_matmul_biased_into(&q8_ff, qa, None, &mut gemm_ff_out);
    }
    let d_gemm_ff = t0.elapsed();
    println!(
        "QuantGEMM (1x512 * 512x2048 FFN):\n         {:8.3} µs/op ({:.1} GFLOPS)",
        d_gemm_ff.as_secs_f64() * 1e6 / gemm_ff_iters as f64,
        (2.0 * 1.0 * k as f64 * n_ff as f64 * gemm_ff_iters as f64) / (d_gemm_ff.as_secs_f64() * 1e9)
    );

    // Softmax 1x512 & Inplace
    let mut sm_in = x_single.clone();
    let mut sm_out = Tensor2::zeros([1, d_model]);
    let sm_iters = 20_000;
    let t0 = Instant::now();
    for _ in 0..sm_iters {
        sm_in.softmax_into(&mut sm_out);
    }
    let d_sm_into = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..sm_iters {
        sm_in.softmax_inplace();
    }
    let d_sm_inplace = t0.elapsed();
    println!(
        "Softmax (1x512):\n  Into:    {:8.3} µs/op\n  Inplace: {:8.3} µs/op ({:.2}x)",
        d_sm_into.as_secs_f64() * 1e6 / sm_iters as f64,
        d_sm_inplace.as_secs_f64() * 1e6 / sm_iters as f64,
        d_sm_into.as_nanos() as f64 / d_sm_inplace.as_nanos().max(1) as f64
    );
}

fn bench_shortlist() {
    println!("\n--- [3] Lexical Shortlist Candidate Generation ---");

    let sl_path = match find_model_file("data/models/zh-Hans_en/lex.50.50.zhen.s2t.bin") {
        Some(p) => p,
        None => {
            println!("[Shortlist Bench] Skipped: lex.50.50.zhen.s2t.bin not found.");
            return;
        }
    };
    let sl = Shortlist::load(&sl_path).expect("load shortlist");
    let trg_vocab = 32_000;

    let src_short = vec![12u32, 45, 88, 120, 501, 892, 1045, 2301, 4055, 6000];
    let src_med: Vec<u32> = (0..50).map(|i| (i * 317 + 13) % 15000).collect();
    let src_long: Vec<u32> = (0..200).map(|i| (i * 127 + 7) % 25000).collect();

    let mut seen_buf = Vec::new();
    let mut out_buf = Vec::with_capacity(4096);
    let iters = 10_000;

    // Short (10 tokens)
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = sl.generate(&src_short, trg_vocab, ShortlistPolicy::Strict);
    }
    let d_alloc = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        sl.generate_into(&src_short, trg_vocab, &mut seen_buf, &mut out_buf, ShortlistPolicy::Strict);
    }
    let d_into = t0.elapsed();
    let count_short = out_buf.len();

    println!(
        "Short (10 tokens, produced {} candidates):\n  Alloc: {:8.3} µs/op\n  Into:  {:8.3} µs/op ({:.2}x)",
        count_short,
        d_alloc.as_secs_f64() * 1e6 / iters as f64,
        d_into.as_secs_f64() * 1e6 / iters as f64,
        d_alloc.as_nanos() as f64 / d_into.as_nanos().max(1) as f64
    );

    // Med (50 tokens)
    let t0 = Instant::now();
    for _ in 0..iters {
        sl.generate_into(&src_med, trg_vocab, &mut seen_buf, &mut out_buf, ShortlistPolicy::Strict);
    }
    let d_med = t0.elapsed();
    let count_med = out_buf.len();
    println!(
        "Med (50 tokens, produced {} candidates):\n  Into:  {:8.3} µs/op",
        count_med,
        d_med.as_secs_f64() * 1e6 / iters as f64
    );

    // Long (200 tokens)
    let iters_long = 2_000;
    let t0 = Instant::now();
    for _ in 0..iters_long {
        sl.generate_into(&src_long, trg_vocab, &mut seen_buf, &mut out_buf, ShortlistPolicy::Strict);
    }
    let d_long = t0.elapsed();
    let count_long = out_buf.len();
    println!(
        "Long (200 tokens, produced {} candidates):\n  Into:  {:8.3} µs/op",
        count_long,
        d_long.as_secs_f64() * 1e6 / iters_long as f64
    );
}

fn bench_markup_and_split() {
    println!("\n--- [4] Markup & Sentence Splitting Benchmark ---");

    let en_paragraph = "Hello Dr. Smith! How are you doing today? We have received your order #12345. \
    Please confirm your address at https://example.com/verify. The package will arrive on Dec. 25, 2026. \
    Thank you for choosing our service! Have a wonderful day.";

    let zh_paragraph = "你好，世界！这是一篇测试文章。段落中包含多个中文字符，以及标点符号。比如：感叹号！问号？还有省略号……\
    测试算法在面对复合中文句子时的分词与断句表现，确保准确切分。";

    let md_text = "# Header 1\n\n\
    This is **bold** text and *italic* text in a [link](https://example.com).\n\n\
    ```rust\n\
    fn main() {\n    println!(\"Hello\");\n}\n\
    ```\n\n\
    - Item 1 with `code`\n\
    - Item 2 with <span>HTML</span> tag\n\n\
    Another paragraph with multiple sentences. First sentence here! Second sentence there?";

    let iters = 5_000;

    // English sentence split (without lang -> runs detect_dominant_language)
    let t0 = Instant::now();
    let mut en_count = 0;
    for _ in 0..iters {
        let res = split_sentences(en_paragraph);
        en_count = res.len();
    }
    let d_en = t0.elapsed();

    // English sentence split with explicit lang
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = mtranserver_lib::engine::markup::split::split_sentences_with_lang(en_paragraph, Some("en"));
    }
    let d_en_with_lang = t0.elapsed();

    println!(
        "English Paragraph Split ({} sentences):\n  Without lang (auto-detect): {:8.3} µs/op\n  With lang=\"en\" (bypasses detect): {:8.3} µs/op ({:.2}x)",
        en_count,
        d_en.as_secs_f64() * 1e6 / iters as f64,
        d_en_with_lang.as_secs_f64() * 1e6 / iters as f64,
        d_en.as_nanos() as f64 / d_en_with_lang.as_nanos().max(1) as f64
    );

    // Chinese sentence split
    let t0 = Instant::now();
    let mut zh_count = 0;
    for _ in 0..iters {
        let res = split_sentences(zh_paragraph);
        zh_count = res.len();
    }
    let d_zh = t0.elapsed();
    println!(
        "Chinese Paragraph Split ({} sentences):\n         {:8.3} µs/op",
        zh_count,
        d_zh.as_secs_f64() * 1e6 / iters as f64
    );

    // Markdown document parsing
    let t0 = Instant::now();
    let mut block_count = 0;
    for _ in 0..iters {
        let blocks = parse_document_blocks(md_text);
        block_count = blocks.len();
    }
    let d_md = t0.elapsed();
    println!(
        "Markdown Document Block Parse ({} blocks):\n         {:8.3} µs/op",
        block_count,
        d_md.as_secs_f64() * 1e6 / iters as f64
    );
}

fn bench_language_detection() {
    println!("\n--- [5] Language Detection Benchmark ---");

    let samples = [
        ("Greeting EN", "Hello world"),
        ("Greeting IT", "buon pomeriggio"),
        ("Script ZH", "你好世界"),
        ("Script JA", "こんにちは世界"),
        ("Script RU", "Привет, мир!"),
        ("Latin ES", "Buenos días, ¿cómo estás hoy?"),
        ("Latin FR", "Bonjour, où est la bibliothèque s'il vous plaît?"),
        ("Bypass Code", "SELECT * FROM users WHERE id = 42;"),
    ];

    let iters = 20_000;
    for (name, text) in samples {
        let t0 = Instant::now();
        let mut detected = "";
        for _ in 0..iters {
            detected = detect_dominant_language(text);
        }
        let dur = t0.elapsed();
        println!(
            "Detect {:14} -> {:7}: {:6.3} µs/op",
            format!("\"{}\"", name),
            format!("\"{}\"", detected),
            dur.as_secs_f64() * 1e6 / iters as f64
        );
    }
}

fn bench_cache() {
    println!("\n--- [6] Cache Key Hashing & Shard Concurrency Benchmark ---");

    let short_args = ["auto", "zh-Hans", "Hello world!"];
    let long_args = [
        "en",
        "es",
        "Artificial intelligence and deep learning models have rapidly revolutionized natural language processing. \
        Modern offline translation systems require careful balance between memory footprint, latency, and throughput. \
        By utilizing specialized SIMD instructions such as AVX2, AVX-512, and VNNI, transformer inference achieves \
        remarkable performance on consumer and server hardware alike.",
    ];

    let iters = 100_000;

    // Hash short args
    let t0 = Instant::now();
    for _ in 0..iters {
        let k = compute_key(std::hint::black_box(&short_args));
        std::hint::black_box(k);
    }
    let d_short_hash = t0.elapsed();

    // Hash long args
    let t0 = Instant::now();
    for _ in 0..iters {
        let k = compute_key(std::hint::black_box(&long_args));
        std::hint::black_box(k);
    }
    let d_long_hash = t0.elapsed();

    // Hash with version
    let t0 = Instant::now();
    for _ in 0..iters {
        let k = compute_key_with_version(std::hint::black_box(&short_args), std::hint::black_box(42));
        std::hint::black_box(k);
    }
    let d_ver_hash = t0.elapsed();
    println!(
        "Dual FNV-1a Hashing:\n  Short ({} bytes): {:8.1} ns/op ({:.1} MB/s)\n  Long ({} bytes):  {:8.1} ns/op ({:.1} MB/s)\n  With Version:    {:8.1} ns/op",
        short_args.iter().map(|s| s.len()).sum::<usize>(),
        d_short_hash.as_nanos() as f64 / iters as f64,
        (short_args.iter().map(|s| s.len()).sum::<usize>() * iters) as f64 / (d_short_hash.as_secs_f64() * 1_000_000.0),
        long_args.iter().map(|s| s.len()).sum::<usize>(),
        d_long_hash.as_nanos() as f64 / iters as f64,
        (long_args.iter().map(|s| s.len()).sum::<usize>() * iters) as f64 / (d_long_hash.as_secs_f64() * 1_000_000.0),
        d_ver_hash.as_nanos() as f64 / iters as f64
    );
}
