use std::path::PathBuf;

use mtranserver_lib::config::ParseFormats;
use mtranserver_lib::engine::{EngineConfig, TranslationEngine};
use mtranserver_test::loader::load_text;

fn model_dir(pair: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = if manifest_dir.ends_with("test") {
        manifest_dir.parent().unwrap().to_path_buf()
    } else {
        manifest_dir
    };
    root.join("data").join("models").join(pair)
}

#[test]
#[ignore = "requires local models in data/models/"]
fn test_html_modes_comparison() {
    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        println!("Skipping real model test: model file not found");
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "zh-Hans",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig::default(),
    )
    .expect("load en_zh engine");

    let input_html = r#"<div class="card" id="main"><h2 class="title">User Guide</h2><p>Please visit <a href="https://example.com/help" class="link">our website</a> for more details.</p></div>"#;

    // 1. Dom 模式 (默认火狐官方方案)
    let dom_result = engine.translate_with_formats(
        input_html,
        ParseFormats {
            html: true,
            md: false,
        },
    );
    println!("\n[Mode 1: DOM (Default Firefox Architecture)]");
    println!("Input:  {}", input_html);
    println!("Output: {}", dom_result);
    assert!(dom_result.contains(r#"<div class="card" id="main">"#));
    assert!(dom_result.contains(r#"<h2 class="title">"#));
    assert!(dom_result.contains(r#"</h2>"#));
    assert!(dom_result.contains(r#"<p>"#));
    assert!(dom_result.contains(r#"<a href="https://example.com/help" class="link">"#));
    assert!(dom_result.contains(r#"</a>"#));
    assert!(dom_result.contains(r#"</p></div>"#));

    let native_result = engine.translate_with_formats(input_html, ParseFormats::NONE);
    println!("\n[Mode 2: Native (Direct Model Passthrough)]");
    println!("Input:  {}", input_html);
    println!("Output: {}", native_result);
}

#[test]
#[ignore = "requires local models in data/models/"]
fn test_real_composite_markdown_document() {
    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        println!("Skipping test: en_zh-Hans model not found");
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "zh-Hans",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig::default(),
    )
    .expect("load en_zh engine");

    let test_md = load_text("documents/composite.md");

    let translated = engine.translate_with_formats(&test_md, ParseFormats::ALL);
    let normalized = translated.replace("\r\n", "\n");
    // 1. YAML Frontmatter must remain 100% verbatim
    assert!(normalized.starts_with("---\ntitle: \"A Complex HTML + Markdown Mashup\"\nauthor: Jane Doe\ntags: [markdown, html, demo]\n---"));

    // 2. Python fenced code block must remain 100% verbatim
    assert!(normalized.contains("```python\ndef greet(name: str) -> str:\n    return f\"Hello, {name}!\"\n\nprint(greet(\"Markdown + HTML\"))\n```"));

    // 3. HTML fenced code block must remain 100% verbatim
    assert!(normalized.contains("```html\n<section class=\"hero\">\n  <h1>Hello, <span class=\"accent\">world</span>!</h1>\n  <p>This is <strong>HTML</strong> inside a Markdown code fence.</p>\n</section>\n```"));

    // 4. Markdown Table delimiter row must be strictly verbatim
    assert!(normalized.contains("| :--- | :---: | ---: |"));

    // 5. Task list markers preserved
    assert!(normalized.contains("- [x] "));
    assert!(normalized.contains("- [ ] "));

    // 6. Inline markup preserved
    assert!(normalized.contains("<strong>"));
    assert!(normalized.contains("</strong>"));
    assert!(normalized.contains("<em>"));
    assert!(normalized.contains("</em>"));
    assert!(normalized.contains("<kbd>"));
    assert!(
        normalized.contains("<sub>2</sub>")
            || normalized.contains("<sub>2")
            || normalized.contains("<sub>")
            || normalized.contains("H2O")
    );
    assert!(normalized.contains("<sup>"));
}

#[test]
#[ignore = "requires local models in data/models/"]
fn test_user_speculative_decoding_translation() {
    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        println!("Skipping test: en_zh-Hans model not found");
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "zh-Hans",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig::default(),
    )
    .expect("load en_zh engine");

    let text = "Autoregressive language models generate one token per forward pass, making inference latency\nproportional to output length. Speculative decoding (Chen et al., 2023; Ge et al., 2022; Leviathan\net al., 2023) accelerates the inference of a target model \u{1D440}\u{1D461} using a lightweight draft model \u{1D440}\u{1D451}. At\neach decoding cycle, the draft model proposes \u{1D6FE} candidate tokens \u{1D465}1, . . . , \u{1D465}\u{1D6FE}. The target model\nverifies all candidates in a single forward pass, accepting the longest prefix consistent with its\nown distribution.";

    let res = engine.translate(text);
    assert!(res.contains("Mt"));
    assert!(res.contains("Md"));
    assert!(res.contains('γ'));
    assert!(res.contains("x1"));
    // Ellipsis and surrounding commas must be preserved rather than mangled into "。 。 。"
    assert!(res.contains("......"));
    assert!(
        res.contains("x1、......、xγ")
            || res.contains("x1、......、")
            || res.contains("......、xγ")
    );
    assert!(!res.contains("。 。 。"));
    assert!(!res.contains('\u{1F451}'));
    assert!(!res.contains('澾'));
}

#[test]
#[ignore = "requires local models in data/models/"]
fn test_real_whois_notification_translation() {
    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        println!("Skipping test: en_zh-Hans model not found");
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "zh-Hans",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig::default(),
    )
    .expect("load en_zh engine");

    let input = r#"Domain Info Notification for example.org
This message is a reminder to help you keep the contact data associated with your domain registration up-to-date. Our records include the following information:

Domain Name: example.org
Registrar Name: Example Registrar Inc

Registrant Name: John Doe
Registrant Organization: Acme Corp
Registrant Phone Number: +1.5550100
Registrant Email: user@example.org
Registrant Address: California
Registrant City: San Francisco
Registrant State/Province: Bay Area
Registrant Country: US
Registrant Postal Code: 00000

Administrative Contact Name: John Doe
Administrative Contact Organization: Acme Corp
Administrative Contact Address: California
Administrative Contact City: San Francisco
Administrative State/Province: Bay Area
Administrative Contact Country: US
Administrative Contact Postal Code: 00000
Administrative Contact Phone: +1.5550100
Administrative Contact Email: user@example.org
Technical Contact Name: John Doe
Technical Contact Organization: Acme Corp
Technical Contact Address: California
Technical Contact City: San Francisco
Technical State/Province: Bay Area
Technical Contact Country: US
Technical Contact Postal Code: 00000
Technical Contact Phone: +1.5550100
Technical Contact Email: user@example.org
Original Creation Date: Mon, Feb 9th, 2026
Expiration Date: Tue, Feb 9th, 2027
Nameservers:

ns1.example.org
ns2.example.org
If any of the information above is inaccurate, you must correct it by logging in to your account at Example dashboard.

If the information above is correct, you do not need to take any action.

Please remember that under the terms of your registration agreement, the provision of false contact information may be grounds for cancellation of your domain name registration.

Thank you for your attention.

The Example Registrar team
"#;

    let res = engine.translate_with_formats(input, ParseFormats::ALL);
    // 1. Verify key newlines are strictly preserved
    assert!(res.contains("example.org\n"), "actual: {}", res);
    assert!(res.contains("ns1.example.org\n"), "actual: {}", res);
    assert!(res.contains("ns2.example.org\n"), "actual: {}", res);
    assert!(
        res.contains("旧金山\n") || res.contains("San Francisco\n"),
        "actual: {}",
        res
    );
    assert!(!res.contains("加利福尼亚市"));
}

#[test]
#[ignore = "requires local models in data/models/"]
fn test_subword_continuation_markup_alignment_en_pt() {
    let dir = model_dir("en_pt");
    let model_path = dir.join("model.enpt.intgemm.alphas.bin");
    let vocab_path = dir.join("vocab.enpt.spm");
    let shortlist_path = dir.join("lex.50.50.enpt.s2t.bin");

    if !model_path.exists() {
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "pt",
        &model_path,
        &shortlist_path,
        &vocab_path,
        &vocab_path,
        EngineConfig::default(),
    )
    .expect("Failed to load en_pt engine");

    let input = "This is <b>bold</b> and <i>italic</i> text.";
    let output = engine.translate_with_formats(input, ParseFormats::ALL);
    println!("en->pt markup output: {}", output);
    // 1. Must not split "negrito" into "<b>neg</b>rito"
    assert!(
        !output.contains("<b>neg</b>rito"),
        "Subword continuation tore markup tag inside word: {}",
        output
    );
    assert!(
        output.contains("<b>em negrito</b>") || output.contains("<b>negrito</b>"),
        "Expected bold tag around negrito: {}",
        output
    );
    // 2. Whitespace absorption: <i> must not swallow leading space (e.g. `e<i> itálico</i>`)
    assert!(
        output.contains(" <i>itálico</i>"),
        "Expected clean space before <i>: {}",
        output
    );
}

#[tokio::test]
#[ignore = "Requires local Marian NMT models in data/models/"]
async fn test_emoji_whitespace_and_period_spacing() {
    use mtranserver_test::regression::{init_test_environment, translate_case};

    init_test_environment()
        .await
        .expect("Failed to init service");

    let input = "🚀 Launching version 2.0 today! ✨ Lightning fast performance and 🔥 zero memory leaks. Approved 👍 by team.";
    let res = translate_case("zh-Hans", input)
        .await
        .expect("Translation failed");
    println!("REAL TRANSLATION RESULT:\n{}", res);
    assert!(res.contains("和 🔥"));
    assert!(res.contains("。由团队批准") || res.contains("。已由团队批准"));
    assert!(!res.contains("。 "));
}

#[test]
#[ignore = "Requires local Marian NMT models in data/models/"]
fn test_beam_search_runtime() {
    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        return;
    }
    let engine_greedy = TranslationEngine::new(
        "en",
        "zh",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig {
            beam_size: 1,
            ..EngineConfig::default()
        },
    )
    .expect("load en_zh engine greedy");

    let engine_beam = TranslationEngine::new(
        "en",
        "zh",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig {
            beam_size: 2,
            ..EngineConfig::default()
        },
    )
    .expect("load en_zh engine beam");

    let text = "Hello world";
    let trans_greedy = engine_greedy.translate(text);
    let trans_beam = engine_beam.translate(text);
    assert!(!trans_greedy.is_empty());
    assert!(!trans_beam.is_empty());
}

#[test]
#[ignore = "Requires local Marian NMT models in data/models/"]
fn test_forward_with_weights_and_alignments_runtime() {
    use mtranserver_lib::engine::transformer::Transformer;

    let dir = model_dir("en_zh-Hans");
    if !dir.join("model.enzh.intgemm.alphas.bin").exists() {
        return;
    }

    let engine = TranslationEngine::new(
        "en",
        "zh-Hans",
        &dir.join("model.enzh.intgemm.alphas.bin"),
        &dir.join("lex.50.50.enzh.s2t.bin"),
        &dir.join("srcvocab.enzh.spm"),
        &dir.join("trgvocab.enzh.spm"),
        EngineConfig::default(),
    )
    .expect("load en_zh engine");

    let source_text = "Hello world from Rust";
    let source_ids = engine.spm_src().encode(source_text);
    assert!(!source_ids.is_empty());

    let mut seen = Vec::new();
    let mut shortlist_indices = Vec::new();
    engine.shortlist().generate_into(
        &source_ids,
        engine.spm_trg().vocab_size(),
        &mut seen,
        &mut shortlist_indices,
        engine.shortlist_policy(),
    );

    let mut out_tokens = Vec::new();
    let mut alignments = Vec::new();

    Transformer::forward_with_weights_and_alignments_into(
        engine.prepared_weights().weights(),
        engine.spm_src(),
        engine.spm_trg(),
        &source_ids,
        &shortlist_indices,
        engine.forward_config(),
        &mut out_tokens,
        Some(&mut alignments),
    );

    assert!(!out_tokens.is_empty(), "out_tokens should not be empty");
    assert_eq!(
        alignments.len(),
        out_tokens.len(),
        "each out_token must have an alignment entry"
    );

    for &src_idx in &alignments {
        assert!(
            src_idx < source_ids.len(),
            "alignment index {} must be strictly less than source_ids.len() {}",
            src_idx,
            source_ids.len()
        );
    }
}
