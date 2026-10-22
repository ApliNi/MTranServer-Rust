use std::path::PathBuf;
use std::time::Instant;
use mtranserver_lib::config::{self, Config, ParseFormats};
use mtranserver_lib::{engine, models, service, utils};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub id: usize,
    pub category: String,
    pub from: String,
    pub to: String,
    pub is_pivot: bool,
    pub input: String,
    pub expected_features: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCaseResult {
    pub id: usize,
    pub category: String,
    pub from: String,
    pub to: String,
    pub is_pivot: bool,
    pub input: String,
    pub output: String,
    pub detected_lang: String,
    pub duration_ms: u64,
    pub model_load_ms: u64,
    pub tag_preserved: bool,
    pub no_repetition: bool,
    pub no_corruption: bool,
    pub accuracy_score: u8,
    pub fluency_score: u8,
    pub integrity_score: u8,
    pub overall_rating: String,
    pub notes: String,
}

fn setup_service() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root_dir = if manifest_dir.ends_with("test") {
        manifest_dir.parent().unwrap().to_path_buf()
    } else {
        manifest_dir
    };

    let data_dir = root_dir.join("data");
    let model_dir = data_dir.join("models");
    let log_dir = data_dir.join("logs");

    let cfg = Config {
        log_level: "info".into(),
        config_dir: data_dir.clone(),
        model_dir,
        host: "127.0.0.1".into(),
        port: "8080".into(),
        enable_web_ui: false,
        enable_offline_mode: true,
        worker_idle_timeout: 3600,
        api_token: "".into(),
        log_dir,
        log_to_file: false,
        log_requests: false,
        max_sentence_length: 1024,
        max_request_body_size: 10 * 1024 * 1024,
        max_batch_size: 100,
        check_model_update: false,
        model_update_interval_secs: 0,
        cache_size: 2000,
        fullwidth_zh_punctuation: true,
        keep_models: vec![],
        keep_top_n: 0,
        max_loaded_models: 0,
        max_threads: 0,
        max_concurrency: 0,
        max_concurrency_per_ip: 0,
        trust_proxy: false,
        log_retention_days: 7,
        preprocess_symbols: false,
        parse_formats: ParseFormats::ALL,
        profile: false,
        bench_qps_input_length: 128,
    };

    config::set_for_testing(cfg);
    service::init();
    utils::cache::init();
    engine::rules::init_rules_from_config();
}

fn check_repetition(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() >= 6 {
        for window in words.windows(3) {
            if window[0] == window[1] && window[1] == window[2] && window[0].len() > 1 {
                return false;
            }
        }
    }
    // Also check repeated single characters (e.g. ？？？？？)
    let chars: Vec<char> = text.chars().collect();
    if chars.len() >= 8 {
        for window in chars.windows(6) {
            if window.iter().all(|&c| c == window[0]) && !window[0].is_whitespace() {
                return false;
            }
        }
    }
    true
}

fn check_corruption(text: &str) -> bool {
    if text.contains('\u{FFFD}') {
        return false;
    }
    true
}

fn check_tag_preservation(input: &str, output: &str) -> bool {
    // Check HTML tags
    let tags = ["<a", "</a>", "<strong>", "</strong>", "<b>", "</b>", "<code>", "</code>", "<p>", "</p>", "<span>", "</span>"];
    for &tag in &tags {
        if input.contains(tag) && !output.contains(tag) {
            return false;
        }
    }
    // Check Markdown code backticks
    let input_backticks = input.matches('`').count();
    let output_backticks = output.matches('`').count();
    if input_backticks > 0 && output_backticks < input_backticks {
        return false;
    }
    true
}
#[tokio::test]
#[ignore = "Requires local Marian NMT models in data/models/"]
async fn run_comprehensive_translation_quality_eval() {
    setup_service();
    if let Err(e) = models::init_records().await {
        panic!("Failed to init records: {}", e);
    }

    let test_cases = vec![
        // ==================== EN -> ZH-HANS ====================
        TestCase {
            id: 1,
            category: "日常口语与礼貌用语".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Good morning! How are you doing today? Would you like a cup of coffee?".into(),
            expected_features: vec!["问候", "疑问句", "全角标点"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 2,
            category: "技术与计算机长句".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "The asynchronous runtime utilizes an epoll-based reactor to poll I/O events with zero CPU overhead when idle.".into(),
            expected_features: vec!["异步运行时", "epoll", "I/O", "CPU"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 3,
            category: "HTML富文本标签保护".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Please click <a href=\"https://example.com/login\">here</a> to <strong>sign in</strong> to your account.".into(),
            expected_features: vec!["<a>", "<strong>"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 4,
            category: "Markdown与行内代码保护".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "First install `docker-compose`, then run **production** mode with `make start`.".into(),
            expected_features: vec!["`docker-compose`", "**", "`make start`"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 5,
            category: "占位符与变量保护".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Welcome {user_name}! Your balance is $100 and your token is %s.".into(),
            expected_features: vec!["{user_name}", "$100", "%s"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 6,
            category: "中文标点规范化与盘古之白".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Memory limit is 16GB, cache size is 512MB, running on port 8080. Is it stable?".into(),
            expected_features: vec!["16 GB", "512 MB", "8080", "，", "？"].into_iter().map(String::from).collect(),
        },

        // ==================== ZH-HANS -> EN ====================
        TestCase {
            id: 7,
            category: "日常口语与礼貌用语".into(),
            from: "zh-Hans".into(),
            to: "en".into(),
            is_pivot: false,
            input: "今天天气真不错，我们一起去公园散步吧？".into(),
            expected_features: vec!["weather", "walk", "park"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 8,
            category: "技术与计算机长句".into(),
            from: "zh-Hans".into(),
            to: "en".into(),
            is_pivot: false,
            input: "本系统基于无锁循环队列实现高性能进程间通信，有效降低了上下文切换带来的延迟。".into(),
            expected_features: vec!["lock-free", "inter-process communication", "latency"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 9,
            category: "HTML富文本标签保护".into(),
            from: "zh-Hans".into(),
            to: "en".into(),
            is_pivot: false,
            input: "请访问我们的<a href=\"https://example.org\">官方网站</a>并下载<strong>最新版本</strong>。".into(),
            expected_features: vec!["<a>", "<strong>"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 10,
            category: "Markdown与行内代码保护".into(),
            from: "zh-Hans".into(),
            to: "en".into(),
            is_pivot: false,
            input: "在 `config.toml` 中配置 `max_threads = 4`，并保持 **debug** 关闭。".into(),
            expected_features: vec!["`config.toml`", "`max_threads = 4`", "**"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 11,
            category: "占位符与变量保护".into(),
            from: "zh-Hans".into(),
            to: "en".into(),
            is_pivot: false,
            input: "订单号 `{order_id}` 已成功创建，金额为 $50.00，请勿泄露。".into(),
            expected_features: vec!["`{order_id}`", "$50.00"].into_iter().map(String::from).collect(),
        },

        // ==================== EN <-> DE (German) ====================
        TestCase {
            id: 12,
            category: "德语日常对话".into(),
            from: "en".into(),
            to: "de".into(),
            is_pivot: false,
            input: "Can you recommend a nice restaurant near the central train station?".into(),
            expected_features: vec!["Restaurant", "Bahnhof"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 13,
            category: "德语技术长句与引文格式".into(),
            from: "en".into(),
            to: "de".into(),
            is_pivot: false,
            input: "The distributed database guarantees sequential consistency across all replica nodes.".into(),
            expected_features: vec!["Datenbank", "Konsistenz"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 14,
            category: "德语对话反向翻译 (de->en)".into(),
            from: "de".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Guten Tag! Ich möchte einen Termin für nächste Woche vereinbaren.".into(),
            expected_features: vec!["appointment", "next week"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 15,
            category: "德语技术长句反向翻译 (de->en)".into(),
            from: "de".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Die Schnittstelle ermöglicht eine nahtlose Integration in bestehende Cloud-Infrastrukturen.".into(),
            expected_features: vec!["interface", "seamless", "cloud"].into_iter().map(String::from).collect(),
        },

        // ==================== EN <-> FR (French) ====================
        TestCase {
            id: 16,
            category: "法语日常对话与排版空格".into(),
            from: "en".into(),
            to: "fr".into(),
            is_pivot: false,
            input: "Could you please send me the presentation slides before the meeting starts?".into(),
            expected_features: vec!["réunion", "présentation"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 17,
            category: "法语标点与技术排版 (en->fr)".into(),
            from: "en".into(),
            to: "fr".into(),
            is_pivot: false,
            input: "High-throughput services require careful load balancing. Are you ready?".into(),
            expected_features: vec!["charge", "?"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 18,
            category: "法语日常反向翻译 (fr->en)".into(),
            from: "fr".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Merci beaucoup pour votre aide précieuse pendant ce projet difficile.".into(),
            expected_features: vec!["thank", "help", "project"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 19,
            category: "法语技术反向翻译 (fr->en)".into(),
            from: "fr".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Cette fonction effectue une copie superficielle de la structure de données.".into(),
            expected_features: vec!["function", "copy", "data structure"].into_iter().map(String::from).collect(),
        },

        // ==================== EN <-> ES (Spanish) ====================
        TestCase {
            id: 20,
            category: "西语日常对话与倒问号 (en->es)".into(),
            from: "en".into(),
            to: "es".into(),
            is_pivot: false,
            input: "Where is the nearest pharmacy? I need some medicine.".into(),
            expected_features: vec!["farmacia", "medicina"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 21,
            category: "西语倒感叹号与感叹句 (en->es)".into(),
            from: "en".into(),
            to: "es".into(),
            is_pivot: false,
            input: "What a beautiful sunset! Have a safe flight!".into(),
            expected_features: vec!["puesta de sol", "vuelo"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 22,
            category: "西语日常反向翻译 (es->en)".into(),
            from: "es".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Hola amigos, ¿cómo ha estado su fin de semana?".into(),
            expected_features: vec!["hello", "weekend"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 23,
            category: "西语技术反向翻译 (es->en)".into(),
            from: "es".into(),
            to: "en".into(),
            is_pivot: false,
            input: "El algoritmo calcula la ruta más corta utilizando grafos dirigidos ponderados.".into(),
            expected_features: vec!["algorithm", "shortest path", "graph"].into_iter().map(String::from).collect(),
        },

        // ==================== EN <-> JA (Japanese) ====================
        TestCase {
            id: 24,
            category: "日语日常礼貌用语 (en->ja)".into(),
            from: "en".into(),
            to: "ja".into(),
            is_pivot: false,
            input: "Thank you very much for your kind assistance yesterday.".into(),
            expected_features: vec!["ありがとう", "昨日"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 25,
            category: "日语技术长句 (en->ja)".into(),
            from: "en".into(),
            to: "ja".into(),
            is_pivot: false,
            input: "The memory management subsystem automatically reclaims unreferenced buffers.".into(),
            expected_features: vec!["メモリ", "バッファ"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 26,
            category: "日语日常反向翻译 (ja->en)".into(),
            from: "ja".into(),
            to: "en".into(),
            is_pivot: false,
            input: "本日はご来訪いただき、誠にありがとうございます。".into(),
            expected_features: vec!["thank", "visit", "today"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 27,
            category: "日语技术反向翻译 (ja->en)".into(),
            from: "ja".into(),
            to: "en".into(),
            is_pivot: false,
            input: "このモデルは高速な推推理処理を実現するために量子化技術を採用しています。".into(),
            expected_features: vec!["model", "processing", "quantization"].into_iter().map(String::from).collect(),
        },

        // ==================== EN <-> RU (Russian) ====================
        TestCase {
            id: 28,
            category: "俄语日常祝福 (en->ru)".into(),
            from: "en".into(),
            to: "ru".into(),
            is_pivot: false,
            input: "Have a wonderful trip and enjoy your vacation with family!".into(),
            expected_features: vec!["семь", "отпуск"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 29,
            category: "俄语技术长句 (en->ru)".into(),
            from: "en".into(),
            to: "ru".into(),
            is_pivot: false,
            input: "The kernel scheduler balances threads among all available CPU cores.".into(),
            expected_features: vec!["ядер", "поток"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 30,
            category: "俄语日常反向翻译 (ru->en)".into(),
            from: "ru".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Привет! Рад тебя видеть, как твои дела?".into(),
            expected_features: vec!["hello", "see you", "how are you"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 31,
            category: "俄语技术反向翻译 (ru->en)".into(),
            from: "ru".into(),
            to: "en".into(),
            is_pivot: false,
            input: "Этот протокол обеспечивает надежную передачу данных по ненадежным сетям.".into(),
            expected_features: vec!["protocol", "reliable", "network"].into_iter().map(String::from).collect(),
        },

        // ==================== PIVOT 枢纽路由 (X -> EN -> Y) ====================
        TestCase {
            id: 32,
            category: "Pivot枢纽翻译 (zh-Hans -> de)".into(),
            from: "zh-Hans".into(),
            to: "de".into(),
            is_pivot: true,
            input: "欢迎使用本开源项目，我们期待您的贡献！".into(),
            expected_features: vec!["Willkommen", "Open-Source"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 33,
            category: "Pivot枢纽翻译 (zh-Hans -> ja)".into(),
            from: "zh-Hans".into(),
            to: "ja".into(),
            is_pivot: true,
            input: "欢迎来到东京，祝您旅行愉快！".into(),
            expected_features: vec!["東京", "旅行"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 34,
            category: "Pivot枢纽翻译 (de -> zh-Hans)".into(),
            from: "de".into(),
            to: "zh-Hans".into(),
            is_pivot: true,
            input: "Das Wetter heute ist wirklich schön, die Sonne scheint den ganzen Tag.".into(),
            expected_features: vec!["天气", "太阳"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 35,
            category: "Pivot枢纽翻译 (fr -> es)".into(),
            from: "fr".into(),
            to: "es".into(),
            is_pivot: true,
            input: "Le soleil brille aujourd'hui et les oiseaux chantent dans le parc.".into(),
            expected_features: vec!["sol", "parque"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 36,
            category: "Pivot枢纽翻译 (ru -> zh-Hans)".into(),
            from: "ru".into(),
            to: "zh-Hans".into(),
            is_pivot: true,
            input: "Наука и технологии играют решающую роль в развитии человечества.".into(),
            expected_features: vec!["科学", "技术", "人类"].into_iter().map(String::from).collect(),
        },

        // ==================== 自动语言识别 AUTO-DETECT ====================
        TestCase {
            id: 37,
            category: "自动语种检测 (auto -> zh-Hans)".into(),
            from: "auto".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Artificial intelligence is transforming every industry rapidly.".into(),
            expected_features: vec!["人工智能", "行业"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 38,
            category: "自动语种检测 (auto -> en)".into(),
            from: "auto".into(),
            to: "en".into(),
            is_pivot: false,
            input: "自动语言检测能够识别输入的文本语言并选择最优路由。".into(),
            expected_features: vec!["language detection", "route"].into_iter().map(String::from).collect(),
        },

        // ==================== 标点边界与特例保护 ====================
        TestCase {
            id: 39,
            category: "URL与网络端口保留 (en->zh-Hans)".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Visit https://example.com:8080/api/v1 or connect to [2001:db8::1]:443.".into(),
            expected_features: vec!["https://example.com:8080/api/v1", "[2001:db8::1]:443"].into_iter().map(String::from).collect(),
        },
        TestCase {
            id: 40,
            category: "混合代码块与换行多段 (en->zh-Hans)".into(),
            from: "en".into(),
            to: "zh-Hans".into(),
            is_pivot: false,
            input: "Section 1: Introduction.\nSection 2: Configuration.\nRun `cargo test --all` to verify.".into(),
            expected_features: vec!["`cargo test --all`"].into_iter().map(String::from).collect(),
        },
    ];

    println!("\n========================================================");
    println!("    MTranServer-Rust 真实翻译服务质量评测开始");
    println!("    用例总数: {}", test_cases.len());
    println!("========================================================\n");

    let mut results: Vec<TestCaseResult> = Vec::new();
    let mut total_accuracy = 0u32;
    let mut total_fluency = 0u32;
    let mut total_integrity = 0u32;

    for tc in test_cases {
        let t0 = Instant::now();
        let res = service::translate_with_pivot_timed(&tc.from, &tc.to, &tc.input).await;
        let _elapsed = t0.elapsed();

        match res {
            Ok(output) => {
                let out_text = output.text.to_string();
                let tag_preserved = check_tag_preservation(&tc.input, &out_text);
                let no_rep = check_repetition(&out_text);
                let no_corr = check_corruption(&out_text);

                // Scoring heuristics:
                let mut acc = 5u8;
                let mut flu = 5u8;
                let mut integ = 5u8;

                if out_text.trim().is_empty() {
                    acc = 1;
                    flu = 1;
                    integ = 1;
                }
                if !no_corr {
                    acc = acc.saturating_sub(2);
                    flu = flu.saturating_sub(2);
                }
                if !no_rep {
                    acc = acc.saturating_sub(2);
                    flu = flu.saturating_sub(2);
                }
                if !tag_preserved {
                    integ = integ.saturating_sub(2);
                }

                // Domain check: see if expected features roughly appear or semantic keywords match
                let lower_out = out_text.to_lowercase();
                let mut matched_features = 0;
                for feat in &tc.expected_features {
                    if lower_out.contains(&feat.to_lowercase()) {
                        matched_features += 1;
                    }
                }

                let overall_rating = if acc >= 4 && flu >= 4 && integ >= 4 && no_rep && no_corr {
                    "优秀 (Excellent)".to_string()
                } else if acc >= 3 && flu >= 3 {
                    "良好 (Good)".to_string()
                } else {
                    "需要改进 (Needs Improvement)".to_string()
                };

                let note = format!(
                    "特征匹配: {}/{}, 译文长度: {} 字符",
                    matched_features,
                    tc.expected_features.len(),
                    out_text.chars().count()
                );

                total_accuracy += acc as u32;
                total_fluency += flu as u32;
                total_integrity += integ as u32;

                let case_res = TestCaseResult {
                    id: tc.id,
                    category: tc.category.clone(),
                    from: tc.from.clone(),
                    to: tc.to.clone(),
                    is_pivot: tc.is_pivot,
                    input: tc.input.clone(),
                    output: out_text.clone(),
                    detected_lang: output.detected_language.clone(),
                    duration_ms: output.translation_duration.as_millis() as u64,
                    model_load_ms: output.model_load_duration.as_millis() as u64,
                    tag_preserved,
                    no_repetition: no_rep,
                    no_corruption: no_corr,
                    accuracy_score: acc,
                    fluency_score: flu,
                    integrity_score: integ,
                    overall_rating,
                    notes: note,
                };

                println!(
                    "[{:02}] [{}] {} -> {}{}:",
                    case_res.id,
                    case_res.category,
                    case_res.from,
                    case_res.to,
                    if case_res.is_pivot { " (Pivot)" } else { "" }
                );
                println!("     源文: {}", case_res.input);
                println!("     译文: {}", case_res.output);
                println!(
                    "     耗时: {}ms (加载 {}ms) | 评分: 准确性 {}/5, 流畅度 {}/5, 格式完整性 {}/5 | 结论: {}",
                    case_res.duration_ms, case_res.model_load_ms, case_res.accuracy_score, case_res.fluency_score, case_res.integrity_score, case_res.overall_rating
                );
                println!("--------------------------------------------------------");

                results.push(case_res);
            }
            Err(e) => {
                panic!("Test case {} failed: {:?}", tc.id, e);
            }
        }
    }

    let n = results.len() as f64;
    let avg_acc = total_accuracy as f64 / n;
    let avg_flu = total_fluency as f64 / n;
    let avg_integ = total_integrity as f64 / n;

    println!("\n========================================================");
    println!("                  评 测 汇 总 统 计                    ");
    println!("========================================================");
    println!("测试用例总数: {}", results.len());
    println!("平均准确性评分 (Accuracy):    {:.2} / 5.0", avg_acc);
    println!("平均流畅度评分 (Fluency):     {:.2} / 5.0", avg_flu);
    println!("平均格式完整性 (Integrity):   {:.2} / 5.0", avg_integ);
    println!("标签保护成功率:               {:.1}%", (results.iter().filter(|r| r.tag_preserved).count() as f64 / n) * 100.0);
    println!("无异常乱码率:                 {:.1}%", (results.iter().filter(|r| r.no_corruption).count() as f64 / n) * 100.0);
    println!("无循环重复词率:               {:.1}%", (results.iter().filter(|r| r.no_repetition).count() as f64 / n) * 100.0);
    println!("========================================================\n");

    // Also verify Batch and Long Text Translation
    test_batch_and_long_text().await;
}

async fn test_batch_and_long_text() {
    println!("========================================================");
    println!("           批量翻译与长文本分句翻译测试                ");
    println!("========================================================");

    // 1. Batch translation
    let batch_inputs = vec![
        "First batch sentence.".to_string(),
        "Second batch sentence.".to_string(),
        "Third batch sentence.".to_string(),
    ];
    let batch_res = service::translate_batch_with_pivot("en", "zh-Hans", &batch_inputs).await.unwrap();
    println!("[批量翻译 en -> zh-Hans] 结果数量: {}", batch_res.len());
    for (i, (translated, detected)) in batch_res.iter().enumerate() {
        println!("  [{}] '{}' -> '{}' (detected: {})", i + 1, batch_inputs[i], translated, detected);
    }
    assert_eq!(batch_res.len(), 3);

    // 2. Long text translation (sentence splitting)
    let long_text = "The Marian NMT engine is exceptionally fast. It executes translation using 8-bit integer matrix multiplication. Memory consumption is minimized while retaining high translation quality across languages.";
    let (long_translated, duration) = service::translate_long_text("en", "zh-Hans", long_text, ParseFormats::ALL).await.unwrap();
    println!("\n[长文本分句翻译 en -> zh-Hans] (耗时 {:?}):", duration);
    println!("  源文: {}", long_text);
    println!("  译文: {}", long_translated);
    assert!(!long_translated.is_empty());

    println!("========================================================\n");
}
