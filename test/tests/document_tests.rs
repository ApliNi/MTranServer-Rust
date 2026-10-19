use std::ops::Range;
use mtranserver_lib::config::ParseFormats;
use mtranserver_lib::engine::markup::document::translate_document_with_alignment;
use mtranserver_test::loader::load_text;

type MockTranslateResult = (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>);

fn mock_translate(text: &str) -> MockTranslateResult {
    let mut trg = String::new();
    let mut src_spans = Vec::new();
    let mut trg_spans = Vec::new();
    let mut alignments = Vec::new();

    let mut i = 0;
    let bytes = text.as_bytes();
    let len = bytes.len();

    while i < len {
        let ch = text[i..].chars().next().unwrap();
        let ch_len = ch.len_utf8();

        if ch.is_whitespace() {
            trg.push(ch);
            i += ch_len;
        } else {
            let start = i;
            while i < len {
                let c = text[i..].chars().next().unwrap();
                if c.is_whitespace() {
                    break;
                }
                i += c.len_utf8();
            }
            let word = &text[start..i];
            let trg_start = trg.len();
            trg.push_str("TR(");
            trg.push_str(word);
            trg.push(')');
            let trg_end = trg.len();

            let token_idx = src_spans.len();
            src_spans.push(start..i);
            trg_spans.push(trg_start..trg_end);
            alignments.push(token_idx);
        }
    }

    if src_spans.is_empty() {
        src_spans.push(0..text.len());
        trg_spans.push(0..trg.len());
        alignments.push(0);
    }

    (trg, src_spans, trg_spans, alignments)
}

#[test]
fn test_composite_document_preserves_structure() {
    let doc = load_text("documents/composite.md");
    let translated = translate_document_with_alignment(&doc, ParseFormats::ALL, mock_translate);

    // Verify Frontmatter is preserved
    assert!(translated.contains("title: \"A Complex HTML + Markdown Mashup\""));
    assert!(translated.contains("author: Jane Doe"));

    // Verify code blocks are untouched
    assert!(translated.contains("def greet(name: str) -> str:"));
    assert!(translated.contains("<section class=\"hero\">"));

    // Verify HTML tags are preserved
    assert!(translated.contains("<div class=\"callout\""));
    assert!(translated.contains("<table>"));
    assert!(translated.contains("<caption>"));

    // Verify body text was translated
    assert!(translated.contains("TR(Welcome)"));
}
