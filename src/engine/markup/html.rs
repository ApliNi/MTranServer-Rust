//! HTML-specific markup parser and reconstructor.
//!
//! Categorizes HTML elements into Block-level (sentence boundaries) and
//! Inline-level (markup annotations). Inline markup is extracted with exact
//! byte offsets, allowing the enclosing block to be translated as an uninterrupted,
//! grammatically sound sentence before projecting inline tags back by alignment.

use super::{render_projected_markup, snap_markup_spans_to_word_boundaries, MarkupSpan};
use std::collections::HashSet;
use std::sync::LazyLock;

/// Recognized block-level tag names that represent natural syntactic and layout boundaries.
static BLOCK_TAGS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "address", "article", "aside", "blockquote", "canvas", "dd", "div", "dl", "dt",
        "fieldset", "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "hr", "li", "main", "nav", "noscript", "ol", "p", "pre", "section", "table",
        "tfoot", "ul", "video", "tr", "td", "th", "tbody", "thead",
    ]
    .into_iter()
    .collect()
});

static ATOMIC_TAGS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    [
        "code", "pre", "kbd", "samp", "math", "svg", "script", "style", "sub", "sup",
    ]
    .into_iter()
    .collect()
});

/// Case-insensitive ASCII substring search without allocating heap memory.
#[inline]
fn find_ignore_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    let n_bytes = needle.as_bytes();
    let h_bytes = haystack.as_bytes();
    if h_bytes.len() < n_bytes.len() {
        return None;
    }
    h_bytes
        .windows(n_bytes.len())
        .position(|window| window.eq_ignore_ascii_case(n_bytes))
}
/// Extracted unit representing a contiguous block or inline-enclosed sentence.
#[derive(Debug, Default)]
pub struct HtmlBlockUnit {
    /// Opening block markup or inter-block whitespace/comments.
    pub prefix: String,
    /// Uninterrupted plain text accumulated inside this block.
    pub text: String,
    /// Extracted inline markup spans with source byte offsets into `text`.
    pub inline_spans: Vec<MarkupSpan>,
}

/// Parse an HTML string into discrete blocks and inline spans.
pub fn parse_html_document(html: &str) -> Vec<HtmlBlockUnit> {
    if !html.contains('<') {
        return vec![HtmlBlockUnit {
            prefix: String::new(),
            text: html.to_string(),
            inline_spans: Vec::new(),
        }];
    }

    let mut units = Vec::new();
    let mut current_unit = HtmlBlockUnit::default();
    let mut tag_stack: Vec<(String, String, usize)> = Vec::new(); // (tag_name, open_tag_string, text_start_offset)

    let bytes = html.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut text_start = 0;

    while i < len {
        if bytes[i] == b'<' {
            let is_tag_start = i + 1 < len
                && matches!(bytes[i + 1], b'a'..=b'z' | b'A'..=b'Z' | b'!' | b'?' | b'/');
            if is_tag_start {
                if let Some(tag_end) = find_tag_end(bytes, i) {
                    // Flush preceding text
                    if i > text_start {
                        let text_segment = &html[text_start..i];
                        current_unit.text.push_str(text_segment);
                    }

                    let raw_tag = &html[i..tag_end];
                    let (is_closing, tag_name, is_self_closing) = parse_tag_info(raw_tag);

                    let is_block = BLOCK_TAGS.contains(tag_name.as_str());

                    if is_block {
                        // Block boundary encountered: flush current unit if it has text or tags
                        if !current_unit.text.trim().is_empty() || !current_unit.inline_spans.is_empty() {
                            units.push(std::mem::take(&mut current_unit));
                        } else {
                            // Inter-block whitespace belongs to layout prefix, not block sentence text
                            current_unit.prefix.push_str(&current_unit.text);
                            current_unit.text.clear();
                        }
                        current_unit.prefix.push_str(raw_tag);
                    } else if !is_closing && ATOMIC_TAGS.contains(tag_name.as_str()) {
                        // Atomic/verbatim inline tag (e.g. <code>...</code>, <sub>...</sub>):
                        // Find matching closing tag and preserve the entire run verbatim.
                        let close_pat = format!("</{}>", tag_name);
                        if let Some(close_rel) = find_ignore_ascii_case(&html[tag_end..], &close_pat) {
                            let close_end = tag_end + close_rel + close_pat.len();
                            let atomic_raw = &html[i..close_end];
                            let pos = current_unit.text.len();
                            current_unit.inline_spans.push(MarkupSpan::new_self_closing(
                                pos,
                                atomic_raw.to_string(),
                                tag_stack.len(),
                            ));
                            i = close_end;
                            text_start = i;
                            continue;
                        } else {
                            // Unterminated atomic tag: treat as regular opening tag
                            let start_offset = current_unit.text.len();
                            tag_stack.push((tag_name, raw_tag.to_string(), start_offset));
                        }
                    } else {
                        // Regular inline tag handling
                        if is_self_closing {
                            let pos = current_unit.text.len();
                            current_unit.inline_spans.push(MarkupSpan::new_self_closing(
                                pos,
                                raw_tag.to_string(),
                                tag_stack.len(),
                            ));
                        } else if is_closing {
                            if let Some(pos) = tag_stack.iter().rposition(|t| t.0 == tag_name) {
                                let (open_name, open_str, start_offset) = tag_stack.remove(pos);
                                let end_offset = current_unit.text.len();
                                current_unit.inline_spans.push(MarkupSpan::new_paired(
                                    start_offset..end_offset,
                                    open_str,
                                    raw_tag.to_string(),
                                    tag_stack.len(),
                                ));
                                let _ = open_name;
                            } else {
                                // Stray closing tag: treat as raw text or append directly
                                current_unit.text.push_str(raw_tag);
                            }
                        } else {
                            // Opening inline tag
                            let start_offset = current_unit.text.len();
                            tag_stack.push((tag_name, raw_tag.to_string(), start_offset));
                        }
                    }

                    i = tag_end;
                    text_start = i;
                    continue;
                }
            }
        }
        i += 1;
    }

    // Flush trailing text
    if text_start < len {
        let text_segment = &html[text_start..];
        current_unit.text.push_str(text_segment);
    }

    if !current_unit.prefix.is_empty() || !current_unit.text.is_empty() || !current_unit.inline_spans.is_empty() {
        units.push(current_unit);
    }

    units
}

/// Translate HTML with alignment-projected inline tags.
///
/// `translate_fn` accepts plain text and returns:
/// `(translated_text, source_token_spans, target_token_spans, alignments)`
pub fn translate_html_with_alignment<F>(html: &str, translate_fn: F) -> String
where
    F: FnMut(&str) -> (String, Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>, Vec<usize>),
{
    translate_html_with_alignment_and_lang(html, None, translate_fn)
}

/// Translate HTML text with word-alignment projected markup and language-aware syntax rules.
pub fn translate_html_with_alignment_and_lang<F>(
    html: &str,
    target_lang: Option<&str>,
    mut translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>, Vec<usize>),
{
    if !html.contains('<') {
        let (out, _, _, _) = translate_fn(html);
        return out;
    }

    let units = parse_html_document(html);
    let mut out = String::with_capacity(html.len() + 64);

    for mut unit in units {
        out.push_str(&unit.prefix);

        let trimmed_text = unit.text.trim();
        if trimmed_text.is_empty() {
            out.push_str(&unit.text);
            continue;
        }

        let leading_ws_len = unit.text.len() - unit.text.trim_start().len();
        let leading_ws = &unit.text[..leading_ws_len];
        let trailing_ws_start = leading_ws_len + trimmed_text.len();
        let trailing_ws = &unit.text[trailing_ws_start..];

        out.push_str(leading_ws);

        if leading_ws_len > 0 {
            for span in &mut unit.inline_spans {
                span.source_range.start = span.source_range.start.saturating_sub(leading_ws_len);
                span.source_range.end = span.source_range.end.saturating_sub(leading_ws_len);
            }
        }

        // Translate the clean block text as one continuous sentence
        let (translated_text, src_spans, trg_spans, alignments) = translate_fn(trimmed_text);

        if unit.inline_spans.is_empty() {
            out.push_str(&translated_text);
        } else {
            // Project inline markup positions using alignments and syntax rules
            super::project_markup_spans_with_lang(
                &mut unit.inline_spans,
                &src_spans,
                &trg_spans,
                &alignments,
                translated_text.len(),
                target_lang,
            );
            snap_markup_spans_to_word_boundaries(&mut unit.inline_spans, &translated_text);
            // Render reconstructed inline HTML
            let rendered = render_projected_markup(&translated_text, &unit.inline_spans);
            out.push_str(&rendered);
        }

        out.push_str(trailing_ws);
    }

    out
}

#[inline]
fn find_tag_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut in_quotes = false;
    let mut quote_char = 0u8;
    for (idx, &b) in bytes[start + 1..].iter().enumerate() {
        if in_quotes {
            if b == quote_char {
                in_quotes = false;
            }
        } else if b == b'"' || b == b'\'' {
            in_quotes = true;
            quote_char = b;
        } else if b == b'>' {
            return Some(start + 1 + idx + 1);
        }
    }
    None
}

/// Parse basic tag info: (is_closing, tag_name_lowercase, is_self_closing)
fn parse_tag_info(tag: &str) -> (bool, String, bool) {
    let trimmed = tag.trim_start_matches('<').trim_end_matches('>');
    let is_closing = trimmed.starts_with('/');
    let after_slash = if is_closing { &trimmed[1..] } else { trimmed };
    let is_self_closing = after_slash.ends_with('/')
        || matches!(
            after_slash.trim().to_ascii_lowercase().as_str(),
            "br" | "hr" | "img" | "input" | "meta" | "link"
        );

    let tag_name_end = after_slash
        .find(|c: char| c.is_whitespace() || c == '/')
        .unwrap_or(after_slash.len());
    let tag_name = after_slash[..tag_name_end].to_ascii_lowercase();

    (is_closing, tag_name, is_self_closing)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_inline_html() {
        let html = "<p>Please visit <a href=\"/help\">our website</a> for <b>more details</b>.</p>";
        let units = parse_html_document(html);

        assert_eq!(units.len(), 2);
        assert_eq!(units[0].prefix, "<p>");
        assert_eq!(units[0].text, "Please visit our website for more details.");
        assert_eq!(units[1].prefix, "</p>");
        assert_eq!(units[0].inline_spans.len(), 2);

        // First inline span: <a>
        assert_eq!(units[0].inline_spans[0].open_markup, "<a href=\"/help\">");
        assert_eq!(units[0].inline_spans[0].close_markup, "</a>");
        assert_eq!(
            &units[0].text[units[0].inline_spans[0].source_range.clone()],
            "our website"
        );

        // Second inline span: <b>
        assert_eq!(units[0].inline_spans[1].open_markup, "<b>");
        assert_eq!(units[0].inline_spans[1].close_markup, "</b>");
        assert_eq!(
            &units[0].text[units[0].inline_spans[1].source_range.clone()],
            "more details"
        );
    }

    #[test]
    fn test_translate_html_with_alignment_mock() {
        let html = "<p>Please visit <a href=\"/help\">our website</a> for more details.</p>";
        let res = translate_html_with_alignment(html, |plain| {
            assert_eq!(plain, "Please visit our website for more details.");
            // Mock translation:
            let translated = "詳細は当社のウェブサイトをクリックしてください。".to_string();
            let src_spans = vec![0..6, 7..12, 13..16, 17..24, 25..28, 29..33, 34..42];
            let trg_spans = vec![0..9, 9..18, 18..36, 36..48, 48..69];
            // Alignment mapping: target -> source
            let alignments = vec![6, 2, 3, 5, 0];
            (translated, src_spans, trg_spans, alignments)
        });

        assert_eq!(
            res,
            "<p>詳細は<a href=\"/help\">当社のウェブサイト</a>をクリックしてください。</p>"
        );
    }

    #[test]
    fn test_html_tag_inside_text_edge_cases() {
        let html = "<p>If x < 10 and y > 20, then true.</p>";
        let res = translate_html_with_alignment(html, |plain| {
            assert_eq!(plain, "If x < 10 and y > 20, then true.");
            (plain.to_string(), vec![], vec![], vec![])
        });
        assert_eq!(res, html);
    }
}
