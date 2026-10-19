use std::borrow::Cow;
use mtranserver_lib::engine::text::{
    char_count, chunk_by_word_boundary, exceeds_char_limit, hide_emojis, mapped_separator,
    preprocess_symbols, restore_emojis, restore_emojis_with_spans,
    tag_aware_spans,
};
use mtranserver_test::loader::load_json;
use serde::Deserialize;


#[derive(Debug, Deserialize)]
struct EmojiCase {
    input: String,
    borrowed: bool,
    replacements: Vec<String>,
    hidden: String,
}

#[derive(Debug, Deserialize)]
struct TagSpanCase {
    input: String,
    spans: Vec<(bool, String)>,
}

#[derive(Debug, Deserialize)]
struct SeparatorCase {
    sep: String,
    lang: String,
    expected: String,
}

#[derive(Debug, Deserialize)]
struct CharLimitCase {
    input: String,
    limit: usize,
    exceeds: bool,
    count: usize,
}


#[test]
fn test_emojis_data_driven() {
    let cases: Vec<EmojiCase> = load_json("text/emojis.json");
    for (idx, case) in cases.iter().enumerate() {
        let (hidden, repls) = hide_emojis(&case.input);
        assert_eq!(
            matches!(hidden, Cow::Borrowed(_)),
            case.borrowed,
            "Case #{} emoji borrowed check failed",
            idx
        );
        assert_eq!(hidden.as_ref(), &case.hidden, "Case #{} hidden string failed", idx);
        if !case.borrowed {
            assert_eq!(&repls, &case.replacements, "Case #{} replacements failed", idx);
            let restored = restore_emojis(&hidden, &repls);
            assert_eq!(restored.as_ref(), &case.input, "Case #{} restore failed", idx);
        }
    }
}

#[test]
fn test_remap_spans_with_emojis() {
    let text = "Prefix [EE0] middle [EE1] suffix";
    let mut spans = vec![
        0..6,   // "Prefix"
        6..7,   // " "
        7..12,  // "[EE0]"
        12..13, // " "
        13..19, // "middle"
        19..20, // " "
        20..25, // "[EE1]"
        25..32, // " suffix"
    ];
    let repls = vec!["😊".to_string(), "🚀".to_string()];
    let restored = restore_emojis_with_spans(text, &repls, &mut spans);
    assert_eq!(restored, "Prefix 😊 middle 🚀 suffix");

    let bytes = restored.as_bytes();
    assert_eq!(&bytes[spans[0].clone()], b"Prefix");
    assert_eq!(&bytes[spans[2].clone()], "😊".as_bytes());
    assert_eq!(&bytes[spans[4].clone()], b"middle");
    assert_eq!(&bytes[spans[6].clone()], "🚀".as_bytes());
    assert_eq!(&bytes[spans[7].clone()], b" suffix");
}

#[test]
fn test_tag_aware_spans_data_driven() {
    let cases: Vec<TagSpanCase> = load_json("text/tag_spans.json");
    for (idx, case) in cases.iter().enumerate() {
        let actual: Vec<(bool, String)> = tag_aware_spans(&case.input)
            .map(|(b, s)| (b, s.to_string()))
            .collect();
        assert_eq!(&actual, &case.spans, "Case #{} tag spans failed", idx);
    }
}

#[test]
fn test_separators_data_driven() {
    let cases: Vec<SeparatorCase> = load_json("text/separators.json");
    for (idx, case) in cases.iter().enumerate() {
        let actual = mapped_separator(&case.sep, &case.lang);
        assert_eq!(actual, case.expected, "Case #{} separator failed", idx);
    }
}

#[test]
fn test_char_limit_and_count_data_driven() {
    let cases: Vec<CharLimitCase> = load_json("text/char_limits.json");
    for (idx, case) in cases.iter().enumerate() {
        assert_eq!(
            exceeds_char_limit(&case.input, case.limit),
            case.exceeds,
            "Case #{} exceeds limit failed",
            idx
        );
        assert_eq!(
            char_count(&case.input),
            case.count,
            "Case #{} char count failed",
            idx
        );
    }
}

#[test]
fn test_chunk_by_word_boundary_respects_limit() {
    let text = "the quick brown fox jumps over the lazy dog";
    let chunks = chunk_by_word_boundary(text, 15);
    for chunk in &chunks {
        assert!(chunk.chars().count() <= 15);
    }
    let joined: String = chunks.concat();
    assert_eq!(joined, text);
}

#[test]
fn test_preprocess_symbols_long_text_quoted_runs() {
    let mut text = String::new();
    for i in 0..200 {
        text.push_str(&format!(
            "Sentence {} with \"quoted_token_{}\" and unquoted_item_{} and `backtick_token_{}`. ",
            i, i, i, i
        ));
    }
    let processed = preprocess_symbols(&text);
    assert!(processed.contains("\"quoted_token_0\""));
    assert!(processed.contains("`backtick_token_0`"));
    assert!(processed.contains("unquoted item_0"));
    assert!(processed.contains("\"quoted_token_199\""));
    assert!(processed.contains("`backtick_token_199`"));
    assert!(processed.contains("unquoted item_199"));
}
