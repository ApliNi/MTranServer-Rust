use std::time::Duration;

use mtranserver_lib::config::{self, Cli};
use mtranserver_lib::service::{
    ensure_engine_loaded, ensure_required_engines, translate_batch_with_pivot_formats,
    translate_long_text, translate_long_text_with_engines, translate_segment, translate_with_pivot,
    translate_with_pivot_detected, translate_with_pivot_timed, PreparedEngines, ServiceError,
};

static INIT: std::sync::Once = std::sync::Once::new();

fn ensure_config_init() {
    INIT.call_once(|| {
        if !config::is_initialized() {
            use clap::Parser;
            let cli = Cli::parse_from(["mtranserver", "--parse-formats", "false"]);
            config::init(cli);
        }
    });
}

#[tokio::test]
async fn test_same_language_timing_zero() {
    ensure_config_init();
    let out = translate_with_pivot_timed("en", "en", "test text")
        .await
        .unwrap();
    assert_eq!(out.text, "test text");
    assert_eq!(out.detected_language, "en");
    assert_eq!(out.translation_duration, Duration::ZERO);
    assert_eq!(out.model_load_duration, Duration::ZERO);
}

#[tokio::test]
async fn test_translate_wrappers_same_language() {
    ensure_config_init();
    let text = translate_with_pivot("fr", "fr", "bonjour").await.unwrap();
    assert_eq!(text, "bonjour");

    let (text, detected) = translate_with_pivot_detected("fr", "fr", "bonjour")
        .await
        .unwrap();
    assert_eq!(text, "bonjour");
    assert_eq!(detected, "fr");
}

#[tokio::test]
async fn test_translate_segment_same_language() {
    ensure_config_init();
    let (text, duration) = translate_segment("de", "de", "hallo", config::ParseFormats::default())
        .await
        .unwrap();
    assert_eq!(text, "hallo");
    assert_eq!(duration, Duration::ZERO);
}

#[tokio::test]
async fn test_ensure_required_engines_same_language() {
    ensure_config_init();
    let (engines, d) = ensure_required_engines("zh", "zh").await.unwrap();
    assert_eq!(d, Duration::ZERO);
    assert!(matches!(engines, PreparedEngines::SameLanguage));
}

#[tokio::test]
async fn test_translate_long_text_same_language() {
    ensure_config_init();
    let text = "First sentence. Second sentence. Third sentence.";
    let (out, duration) = translate_long_text("en", "en", text, config::ParseFormats::default())
        .await
        .unwrap();
    assert_eq!(out, text);
    assert!(duration < Duration::from_secs(5));
}

#[tokio::test]
async fn test_translate_batch_same_language() {
    ensure_config_init();
    let texts = vec!["Hello".to_string(), "World".to_string()];
    let results = translate_batch_with_pivot_formats("en", "en", &texts, None)
        .await
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0], ("Hello".to_string(), "en".to_string()));
    assert_eq!(results[1], ("World".to_string(), "en".to_string()));
}

#[tokio::test]
async fn test_translate_batch_empty() {
    ensure_config_init();
    let results = translate_batch_with_pivot_formats("en", "zh", &[], None)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_translate_long_text_western_spacing() {
    ensure_config_init();
    let engines = PreparedEngines::SameLanguage;
    let text = "First sentence。Second sentence！Third sentence？";
    let (out, _) = translate_long_text_with_engines(
        &engines,
        "zh",
        "en",
        text,
        config::ParseFormats::default(),
    )
    .await
    .unwrap();
    assert_eq!(out, "First sentence。 Second sentence！ Third sentence？");

    let text_spaced = "First sentence。\nSecond sentence！ Third sentence？";
    let (out_spaced, _) = translate_long_text_with_engines(
        &engines,
        "zh",
        "en",
        text_spaced,
        config::ParseFormats::default(),
    )
    .await
    .unwrap();
    assert_eq!(out_spaced, "First sentence。\nSecond sentence！ Third sentence？");

    let (out_zh, _) = translate_long_text_with_engines(
        &engines,
        "ja",
        "zh",
        "第一句。第二句！第三句？",
        config::ParseFormats::default(),
    )
    .await
    .unwrap();
    assert_eq!(out_zh, "第一句。第二句！第三句？");
}

#[tokio::test]
async fn test_ensure_engine_loaded_concurrent_not_found() {
    ensure_config_init();
    let mut handles = Vec::new();
    for _ in 0..10 {
        handles.push(tokio::spawn(async {
            ensure_engine_loaded("xx_fake", "yy_fake").await
        }));
    }
    for h in handles {
        let res = h.await.unwrap();
        assert!(res.is_err());
        match res.err().unwrap() {
            ServiceError::ModelNotFound(_) => {}
            other => panic!("expected ModelNotFound, got: {other:?}"),
        }
    }
}
