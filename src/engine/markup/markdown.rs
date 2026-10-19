//! Markdown parser and word-alignment projector.
//!
//! Deconstructs Markdown documents into block-level elements (paragraphs, headers,
//! blockquotes, lists, code fences) and extracts inline formatting (links, bold,
//! italic, inline code, strikethrough) as [`MarkupSpan`]s.
//!
//! The enclosing blocks are submitted to the neural translation engine as complete,
//! natural sentences. The resulting Cross-Attention word alignments project the
//! Markdown syntax back onto the translated target text, preserving both natural
//! linguistic semantics and strict markup structure across word reordering.

use super::{render_projected_markup, snap_markup_spans_to_word_boundaries, MarkupSpan};

/// A block-level segment of a Markdown document.
#[derive(Debug, Clone, Default)]
pub struct MarkdownBlockUnit {
    /// Markdown block prefix (e.g. "### ", "- ", "> ", or newlines).
    pub prefix: String,
    /// Extracted plain text content for neural translation.
    pub text: String,
    /// Extracted inline markup annotations.
    pub inline_spans: Vec<MarkupSpan>,
    /// Whether this block is a verbatim block (e.g. fenced code block) that must not be translated.
    pub is_verbatim: bool,
    /// Markdown block suffix (trailing newlines).
    pub suffix: String,
}

/// Parse a Markdown document into block units and inline markup spans.
pub fn parse_markdown_document(md: &str) -> Vec<MarkdownBlockUnit> {
    if md.trim().is_empty() {
        return vec![MarkdownBlockUnit {
            prefix: String::new(),
            text: md.to_string(),
            inline_spans: Vec::new(),
            is_verbatim: false,
            suffix: String::new(),
        }];
    }

    let mut units = Vec::new();
    let lines: Vec<&str> = md.split('\n').collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // 1. Fenced code block (``` or ~~~)
        let trimmed_start = line.trim_start();
        if trimmed_start.starts_with("```") || trimmed_start.starts_with("~~~") {
            let fence = if trimmed_start.starts_with("```") { "```" } else { "~~~" };
            let mut code_block = String::new();
            code_block.push_str(line);
            code_block.push('\n');
            i += 1;

            while i < lines.len() {
                let code_line = lines[i];
                code_block.push_str(code_line);
                if i + 1 < lines.len() {
                    code_block.push('\n');
                }
                if code_line.trim_start().starts_with(fence) {
                    i += 1;
                    break;
                }
                i += 1;
            }

            units.push(MarkdownBlockUnit {
                prefix: code_block,
                text: String::new(),
                inline_spans: Vec::new(),
                is_verbatim: true,
                suffix: String::new(),
            });
            continue;
        }

        // 2. Empty line: preserve as structural formatting
        if line.trim().is_empty() {
            units.push(MarkdownBlockUnit {
                prefix: format!("{}\n", line),
                text: String::new(),
                inline_spans: Vec::new(),
                is_verbatim: true,
                suffix: String::new(),
            });
            i += 1;
            continue;
        }

        // 3. Header (# h1, ## h2, ...)
        if trimmed_start.starts_with('#') {
            let hash_count = trimmed_start.chars().take_while(|&c| c == '#').count();
            if hash_count <= 6 && trimmed_start[hash_count..].starts_with(' ') {
                let leading_ws = &line[..line.len() - trimmed_start.len()];
                let prefix = format!("{}{}{} ", leading_ws, "#".repeat(hash_count), "");
                let content = trimmed_start[hash_count..].trim_start();
                let (plain_text, inlines) = extract_inline_markdown(content);
                units.push(MarkdownBlockUnit {
                    prefix,
                    text: plain_text,
                    inline_spans: inlines,
                    is_verbatim: false,
                    suffix: "\n".to_string(),
                });
                i += 1;
                continue;
            }
        }

        // 4. Blockquote (> quote)
        if trimmed_start.starts_with('>') {
            let leading_ws = &line[..line.len() - trimmed_start.len()];
            let content = trimmed_start.trim_start_matches('>').trim_start();
            let prefix = format!("{}> ", leading_ws);
            let (plain_text, inlines) = extract_inline_markdown(content);
            units.push(MarkdownBlockUnit {
                prefix,
                text: plain_text,
                inline_spans: inlines,
                is_verbatim: false,
                suffix: "\n".to_string(),
            });
            i += 1;
            continue;
        }

        // 5. Unordered list (- , * , + )
        if (trimmed_start.starts_with("- ") || trimmed_start.starts_with("* ") || trimmed_start.starts_with("+ "))
            && !trimmed_start.starts_with("---") && !trimmed_start.starts_with("***")
        {
            let bullet_char = trimmed_start.chars().next().unwrap();
            let leading_ws = &line[..line.len() - trimmed_start.len()];
            let prefix = format!("{}{}{} ", leading_ws, bullet_char, "");
            let content = &trimmed_start[2..];
            let (plain_text, inlines) = extract_inline_markdown(content);
            units.push(MarkdownBlockUnit {
                prefix,
                text: plain_text,
                inline_spans: inlines,
                is_verbatim: false,
                suffix: "\n".to_string(),
            });
            i += 1;
            continue;
        }

        // 6. Ordered list (1. , 2. , etc.)
        let digit_count = trimmed_start.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_count > 0 && trimmed_start[digit_count..].starts_with(". ") {
            let num_str = &trimmed_start[..digit_count];
            let leading_ws = &line[..line.len() - trimmed_start.len()];
            let prefix = format!("{}{}. ", leading_ws, num_str);
            let content = &trimmed_start[digit_count + 2..];
            let (plain_text, inlines) = extract_inline_markdown(content);
            units.push(MarkdownBlockUnit {
                prefix,
                text: plain_text,
                inline_spans: inlines,
                is_verbatim: false,
                suffix: "\n".to_string(),
            });
            i += 1;
            continue;
        }

        // 7. General paragraph (single line or accumulated lines)
        let (plain_text, inlines) = extract_inline_markdown(line);
        units.push(MarkdownBlockUnit {
            prefix: String::new(),
            text: plain_text,
            inline_spans: inlines,
            is_verbatim: false,
            suffix: if i + 1 < lines.len() { "\n".to_string() } else { String::new() },
        });
        i += 1;
    }

    units
}

/// Extract inline Markdown annotations from text, producing clean plain text and `MarkupSpan`s.
pub fn extract_inline_markdown(text: &str) -> (String, Vec<MarkupSpan>) {
    let mut plain = String::with_capacity(text.len());
    let mut spans = Vec::new();
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;

    while i < len {
        // 1. Link or Image: [link text](url) or ![alt](url)
        let is_image = bytes[i] == b'!' && i + 1 < len && bytes[i + 1] == b'[';
        let is_link = bytes[i] == b'[';

        if is_image || is_link {
            let link_start = i;
            let bracket_open = if is_image { i + 1 } else { i };

            if let Some(bracket_close) = find_matching(bytes, bracket_open, b'[', b']') {
                if bracket_close + 1 < len && bytes[bracket_close + 1] == b'(' {
                    let paren_open = bracket_close + 1;
                    if let Some(paren_close) = find_matching(bytes, paren_open, b'(', b')') {
                        let link_text = &text[bracket_open + 1..bracket_close];
                        let url = &text[paren_open + 1..paren_close];

                        if is_image {
                            // Image: treat as self-closing point markup
                            let pos = plain.len();
                            let raw_img = &text[link_start..=paren_close];
                            spans.push(MarkupSpan::new_self_closing(pos, raw_img.to_string(), 0));
                        } else {
                            // Regular hyperlink: translate link_text and project [ and ](url)
                            let start_offset = plain.len();
                            plain.push_str(link_text);
                            let end_offset = plain.len();

                            spans.push(MarkupSpan::new_paired(
                                start_offset..end_offset,
                                "[".to_string(),
                                format!("]({})", url),
                                0,
                            ));
                        }

                        i = paren_close + 1;
                        continue;
                    }
                }
            }
        }

        // 2. Inline code: `code`
        if bytes[i] == b'`' {
            let code_start = i;
            let mut backtick_count = 0;
            while i < len && bytes[i] == b'`' {
                backtick_count += 1;
                i += 1;
            }
            let fence = &text[code_start..i];
            if let Some(pos) = text[i..].find(fence) {
                let code_content = &text[i..i + pos];
                let start_offset = plain.len();
                plain.push_str(code_content);
                let end_offset = plain.len();

                spans.push(MarkupSpan::new_paired(
                    start_offset..end_offset,
                    fence.to_string(),
                    fence.to_string(),
                    0,
                ));

                i = i + pos + backtick_count;
                continue;
            } else {
                // Unterminated backtick
                plain.push_str(fence);
                continue;
            }
        }

        // 3. Bold / Italic: ***, **, *, ___, __, _
        if bytes[i] == b'*' || bytes[i] == b'_' {
            let marker_char = bytes[i];
            let start_idx = i;
            let mut marker_count = 0;
            while i < len && bytes[i] == marker_char {
                marker_count += 1;
                i += 1;
            }
            let marker = &text[start_idx..i];

            if marker_count >= 1 && marker_count <= 3 {
                if let Some(closing_rel) = text[i..].find(marker) {
                    let inner_content = &text[i..i + closing_rel];
                    if !inner_content.trim().is_empty() {
                        let start_offset = plain.len();
                        plain.push_str(inner_content);
                        let end_offset = plain.len();

                        spans.push(MarkupSpan::new_paired(
                            start_offset..end_offset,
                            marker.to_string(),
                            marker.to_string(),
                            0,
                        ));

                        i = i + closing_rel + marker_count;
                        continue;
                    }
                }
            }

            plain.push_str(marker);
            continue;
        }

        // 4. Strikethrough: ~~text~~
        if bytes[i] == b'~' && i + 1 < len && bytes[i + 1] == b'~' {
            let start_idx = i;
            let marker = "~~";
            i += 2;
            if let Some(closing_rel) = text[i..].find(marker) {
                let inner_content = &text[i..i + closing_rel];
                if !inner_content.trim().is_empty() {
                    let start_offset = plain.len();
                    plain.push_str(inner_content);
                    let end_offset = plain.len();

                    spans.push(MarkupSpan::new_paired(
                        start_offset..end_offset,
                        marker.to_string(),
                        marker.to_string(),
                        0,
                    ));

                    i = i + closing_rel + 2;
                    continue;
                }
            }
            plain.push_str(&text[start_idx..i]);
            continue;
        }

        // Regular character
        let c = text[i..].chars().next().unwrap();
        plain.push(c);
        i += c.len_utf8();
    }

    (plain, spans)
}

/// Helper to find matching bracket or parenthesis, respecting nested pairs.
fn find_matching(bytes: &[u8], open_idx: usize, open_byte: u8, close_byte: u8) -> Option<usize> {
    let mut depth = 0;
    let mut j = open_idx;
    while j < bytes.len() {
        if bytes[j] == open_byte {
            depth += 1;
        } else if bytes[j] == close_byte {
            depth -= 1;
            if depth == 0 {
                return Some(j);
            }
        }
        j += 1;
    }
    None
}

/// Translate Markdown text with word-alignment projected markup.
pub fn translate_markdown_with_alignment<F>(md: &str, translate_fn: F) -> String
where
    F: FnMut(&str) -> (String, Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>, Vec<usize>),
{
    translate_markdown_with_alignment_and_lang(md, None, translate_fn)
}

/// Translate Markdown text with word-alignment projected markup and language-aware syntax rules.
pub fn translate_markdown_with_alignment_and_lang<F>(
    md: &str,
    target_lang: Option<&str>,
    mut translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<std::ops::Range<usize>>, Vec<std::ops::Range<usize>>, Vec<usize>),
{
    let units = parse_markdown_document(md);
    let mut out = String::with_capacity(md.len() + 64);

    for mut unit in units {
        out.push_str(&unit.prefix);

        if unit.is_verbatim || unit.text.trim().is_empty() {
            out.push_str(&unit.text);
            out.push_str(&unit.suffix);
            continue;
        }

        let (target_text, src_spans, trg_spans, alignments) = translate_fn(&unit.text);

        if unit.inline_spans.is_empty() {
            out.push_str(&target_text);
        } else {
            super::project_markup_spans_with_lang(
                &mut unit.inline_spans,
                &src_spans,
                &trg_spans,
                &alignments,
                target_text.len(),
                target_lang,
            );
            snap_markup_spans_to_word_boundaries(&mut unit.inline_spans, &target_text);
            let rendered = render_projected_markup(&target_text, &unit.inline_spans);
            out.push_str(&rendered);
        }

        out.push_str(&unit.suffix);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_inline_markdown() {
        let text = "Please visit [our website](https://example.com) for **more details** and `config.json`.";
        let (plain, spans) = extract_inline_markdown(text);

        assert_eq!(plain, "Please visit our website for more details and config.json.");
        assert_eq!(spans.len(), 3);

        // Link span
        assert_eq!(spans[0].open_markup, "[");
        assert_eq!(spans[0].close_markup, "](https://example.com)");
        assert_eq!(&plain[spans[0].source_range.clone()], "our website");

        // Bold span
        assert_eq!(spans[1].open_markup, "**");
        assert_eq!(spans[1].close_markup, "**");
        assert_eq!(&plain[spans[1].source_range.clone()], "more details");

        // Inline code span
        assert_eq!(spans[2].open_markup, "`");
        assert_eq!(spans[2].close_markup, "`");
        assert_eq!(&plain[spans[2].source_range.clone()], "config.json");
    }

    #[test]
    fn test_translate_markdown_word_reordering() {
        let md = "Please visit [our website](https://example.com) for more details.";
        let res = translate_markdown_with_alignment(md, |plain| {
            assert_eq!(plain, "Please visit our website for more details.");
            // Mock translation with word reordering:
            let translated = "詳細は当社のウェブサイトをご覧ください。".to_string();
            let src_spans = vec![0..6, 7..12, 13..16, 17..24, 25..28, 29..37];
            let trg_spans = vec![0..9, 9..18, 18..36, 36..39, 39..51, 51..72];
            let alignments = vec![5, 2, 3, 4, 1, 0];
            (translated, src_spans, trg_spans, alignments)
        });

        assert_eq!(
            res,
            "詳細は[当社のウェブサイト](https://example.com)をご覧ください。"
        );
    }

    #[test]
    fn test_fenced_code_block_verbatim() {
        let md = "# Title\n\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\n\n- List item";
        let units = parse_markdown_document(md);
        assert_eq!(units.len(), 5);
        assert!(units[2].is_verbatim);
        assert!(units[2].prefix.contains("```rust"));
    }
}
