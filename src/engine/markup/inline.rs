//! Unified inline dual-syntax markup & entity extractor.
//!
//! Provides simultaneous, lossless extraction of both HTML inline tags (paired,
//! void, and atomic) and Markdown inline markers (links, images, bold, italic,
//! strikethrough, backtick code), alongside atomic protection for HTML entities.
//!
//! Separates formatting and verbatim code/entities from natural language text,
//! yielding a clean plain text suitable for neural translation (Marian NMT)
//! and a sequence of [`MarkupSpan`]s that map onto exact byte offsets in the plain text.

use super::MarkupSpan;

/// Configuration options controlling which inline syntaxes to recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InlineParseOptions {
    /// Whether to parse HTML inline tags, void elements, atomic tags, and HTML entities.
    pub parse_html: bool,
    /// Whether to parse Markdown inline markup (links, images, bold, italic, strike, code).
    pub parse_markdown: bool,
}

impl Default for InlineParseOptions {
    fn default() -> Self {
        Self {
            parse_html: true,
            parse_markdown: true,
        }
    }
}

impl InlineParseOptions {
    /// Parse both HTML and Markdown inline markup.
    pub const ALL: Self = Self {
        parse_html: true,
        parse_markdown: true,
    };

    /// Parse HTML tags and entities only, treating Markdown syntax as plain text.
    pub const HTML_ONLY: Self = Self {
        parse_html: true,
        parse_markdown: false,
    };

    /// Parse Markdown inline syntax only, treating HTML tags and entities as plain text.
    pub const MARKDOWN_ONLY: Self = Self {
        parse_html: false,
        parse_markdown: true,
    };
}

/// HTML tags whose inner contents must be protected verbatim as atomic literals (not translated).
#[inline]
pub fn is_atomic_tag(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "code" | "pre" | "kbd" | "math" | "samp" | "svg" | "script" | "style"
    )
}

/// HTML void (self-closing) elements that never have a closing tag.
#[inline]
pub fn is_void_tag(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "img"
            | "br"
            | "hr"
            | "wbr"
            | "input"
            | "meta"
            | "link"
            | "col"
            | "area"
            | "base"
            | "param"
            | "source"
            | "track"
    )
}

/// Recognized standard HTML inline elements.
#[inline]
pub fn is_recognized_inline_tag(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "a" | "b"
            | "strong"
            | "i"
            | "em"
            | "code"
            | "kbd"
            | "sup"
            | "sub"
            | "mark"
            | "del"
            | "span"
            | "abbr"
            | "cite"
            | "q"
            | "u"
            | "s"
            | "small"
            | "var"
            | "time"
            | "bdo"
            | "bdi"
            | "ruby"
            | "rt"
            | "rp"
            | "data"
            | "dfn"
            | "ins"
            | "label"
            | "button"
    )
}

/// Extract unified inline markup spans from text using default settings (both HTML and Markdown enabled).
pub fn extract_unified_inline(text: &str) -> (String, Vec<MarkupSpan>) {
    extract_inline_markup(text, InlineParseOptions::default())
}
/// Maximum allowed nesting depth for recursive inline markup extraction.
pub const MAX_INLINE_DEPTH: usize = 32;

/// Extract inline markup annotations according to configured format options.
///
/// Returns:
/// - `plain_text`: Sanitized natural language text suitable for NMT translation.
/// - `spans`: Extracted `MarkupSpan` annotations referencing byte ranges in `plain_text`.
pub fn extract_inline_markup(text: &str, options: InlineParseOptions) -> (String, Vec<MarkupSpan>) {
    let (plain, mut spans) = extract_inline_impl(text, options, 0);
    // Sort spans primarily by start offset ascending, then by depth ascending (outer first)
    spans.sort_by(|a, b| {
        a.source_range
            .start
            .cmp(&b.source_range.start)
            .then_with(|| a.depth.cmp(&b.depth))
    });
    (plain, spans)
}

/// Core recursive parser for inline markup extraction.
fn extract_inline_impl(
    text: &str,
    options: InlineParseOptions,
    depth: usize,
) -> (String, Vec<MarkupSpan>) {
    let mut plain = String::with_capacity(text.len());
    let mut spans = Vec::with_capacity(4);

    if depth >= MAX_INLINE_DEPTH {
        plain.push_str(text);
        return (plain, spans);
    }

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    while i < len {
        // 1. HTML Comment or CDATA
        if options.parse_html {
            if text[i..].starts_with("<!--") {
                if let Some(pos) = text[i..].find("-->") {
                    let end_idx = i + pos + 3;
                    let raw = &text[i..end_idx];
                    let pos = plain.len();
                    spans.push(MarkupSpan::new_self_closing(pos, raw.to_string(), depth));
                    i = end_idx;
                    continue;
                }
            } else if text[i..].starts_with("<![CDATA[") {
                if let Some(pos) = text[i..].find("]]>") {
                    let end_idx = i + pos + 3;
                    let raw = &text[i..end_idx];
                    let pos = plain.len();
                    spans.push(MarkupSpan::new_self_closing(pos, raw.to_string(), depth));
                    i = end_idx;
                    continue;
                }
            }
        }

        // 2. HTML Entities: atomic protection (is_self_closing = true, not sent to translation model)
        if options.parse_html && bytes[i] == b'&' {
            if let Some(entity_len) = try_parse_html_entity(&text[i..]) {
                let entity_raw = &text[i..i + entity_len];
                let pos = plain.len();
                spans.push(MarkupSpan::new_self_closing(pos, entity_raw.to_string(), depth));
                i += entity_len;
                continue;
            }
        }

        // 3. Markdown Inline Code: `...` (atomic literal protection)
        if options.parse_markdown && bytes[i] == b'`' {
            let start_idx = i;
            let mut fence_len = 0;
            while i < len && bytes[i] == b'`' {
                fence_len += 1;
                i += 1;
            }
            if let Some((_content_end, after_fence)) = find_backtick_closing(&text[i..], fence_len) {
                let close_end = i + after_fence;
                let raw_code = &text[start_idx..close_end];
                let pos = plain.len();
                spans.push(MarkupSpan::new_self_closing(pos, raw_code.to_string(), depth));
                i = close_end;
                continue;
            } else {
                // Unterminated backtick fence: treat as plain text
                plain.push_str(&text[start_idx..i]);
                continue;
            }
        }

        // 4. HTML Tags
        if options.parse_html && bytes[i] == b'<' {
            let is_tag_start = i + 1 < len
                && matches!(bytes[i + 1], b'a'..=b'z' | b'A'..=b'Z' | b'!' | b'?' | b'/');
            if is_tag_start {
                if let Some(tag_end) = find_tag_end(bytes, i) {
                    let raw_tag = &text[i..tag_end];
                    let (is_closing, tag_name, is_explicit_self_closing) = parse_tag_info(raw_tag);

                    // 4a. Atomic tags (<code>...</code>, <pre>...</pre>, <math>...</math>, <kbd>...</kbd>, etc.)
                    if !is_closing && is_atomic_tag(tag_name.as_str()) {
                        if is_explicit_self_closing {
                            let pos = plain.len();
                            spans.push(MarkupSpan::new_self_closing(pos, raw_tag.to_string(), depth));
                            i = tag_end;
                            continue;
                        } else if let Some((_, close_end_rel)) = find_matching_close_tag(&text[tag_end..], &tag_name) {
                            let close_end = tag_end + close_end_rel;
                            let atomic_raw = &text[i..close_end];
                            let pos = plain.len();
                            spans.push(MarkupSpan::new_self_closing(pos, atomic_raw.to_string(), depth));
                            i = close_end;
                            continue;
                        } else {
                            // Unterminated atomic tag: protect tag as self-closing point
                            let pos = plain.len();
                            spans.push(MarkupSpan::new_self_closing(pos, raw_tag.to_string(), depth));
                            i = tag_end;
                            continue;
                        }
                    }

                    // 4b. Void elements or self-closing tags (<img ...>, <br>, <hr>, etc.)
                    let is_void = is_void_tag(tag_name.as_str()) || is_explicit_self_closing;
                    if is_void {
                        let pos = plain.len();
                        spans.push(MarkupSpan::new_self_closing(pos, raw_tag.to_string(), depth));
                        i = tag_end;
                        continue;
                    }

                    // 4c. Stray closing tag
                    if is_closing {
                        let pos = plain.len();
                        spans.push(MarkupSpan::new_self_closing(pos, raw_tag.to_string(), depth));
                        i = tag_end;
                        continue;
                    }

                    // 4d. Paired inline tag
                    if let Some((close_start_rel, close_end_rel)) = find_matching_close_tag(&text[tag_end..], &tag_name) {
                        let inner_text = &text[tag_end..tag_end + close_start_rel];
                        let close_tag_raw = &text[tag_end + close_start_rel..tag_end + close_end_rel];

                        let (inner_plain, inner_spans) = extract_inline_impl(inner_text, options, depth + 1);

                        let start_offset = plain.len();
                        plain.push_str(&inner_plain);
                        let end_offset = plain.len();

                        spans.push(MarkupSpan::new_paired(
                            start_offset..end_offset,
                            raw_tag.to_string(),
                            close_tag_raw.to_string(),
                            depth,
                        ));

                        for mut span in inner_spans {
                            span.source_range.start += start_offset;
                            span.source_range.end += start_offset;
                            spans.push(span);
                        }

                        i = tag_end + close_end_rel;
                        continue;
                    } else if is_recognized_inline_tag(tag_name.as_str()) {
                        // Unterminated recognized inline tag: treat as self-closing point
                        let pos = plain.len();
                        spans.push(MarkupSpan::new_self_closing(pos, raw_tag.to_string(), depth));
                        i = tag_end;
                        continue;
                    }
                    // Non-recognized angle bracket constructs (e.g. List<String>) fall through as plain text
                }
            }
        }

        // 5. Markdown Image: ![alt](url)
        if options.parse_markdown && bytes[i] == b'!' && i + 1 < len && bytes[i + 1] == b'[' {
            let bracket_open = i + 1;
            if let Some(bracket_close) = find_matching_bracket(bytes, bracket_open, b'[', b']') {
                if bracket_close + 1 < len && bytes[bracket_close + 1] == b'(' {
                    let paren_open = bracket_close + 1;
                    if let Some(paren_close) = find_matching_bracket(bytes, paren_open, b'(', b')') {
                        let raw_img = &text[i..=paren_close];
                        let pos = plain.len();
                        spans.push(MarkupSpan::new_self_closing(pos, raw_img.to_string(), depth));
                        i = paren_close + 1;
                        continue;
                    }
                }
            }
        }

        // 6. Markdown Link: [text](url)
        if options.parse_markdown && bytes[i] == b'[' {
            let bracket_open = i;
            if let Some(bracket_close) = find_matching_bracket(bytes, bracket_open, b'[', b']') {
                if bracket_close + 1 < len && bytes[bracket_close + 1] == b'(' {
                    let paren_open = bracket_close + 1;
                    if let Some(paren_close) = find_matching_bracket(bytes, paren_open, b'(', b')') {
                        let inner_text = &text[bracket_open + 1..bracket_close];
                        let url_part = &text[paren_open + 1..paren_close];

                        let (inner_plain, inner_spans) = extract_inline_impl(inner_text, options, depth + 1);

                        let start_offset = plain.len();
                        plain.push_str(&inner_plain);
                        let end_offset = plain.len();

                        spans.push(MarkupSpan::new_paired(
                            start_offset..end_offset,
                            "[".to_string(),
                            format!("]({})", url_part),
                            depth,
                        ));

                        for mut span in inner_spans {
                            span.source_range.start += start_offset;
                            span.source_range.end += start_offset;
                            spans.push(span);
                        }

                        i = paren_close + 1;
                        continue;
                    }
                }
            }
        }

        // 7. Markdown Strikethrough: ~~text~~
        if options.parse_markdown && text[i..].starts_with("~~") {
            let can_open = text[i + 2..].chars().next().is_some_and(|c| !c.is_whitespace());
            if can_open {
                if let Some(close_rel) = find_strikethrough_close(text, i + 2) {
                    let inner_text = &text[i + 2..close_rel];
                    if !inner_text.trim().is_empty() {
                        let (inner_plain, inner_spans) = extract_inline_impl(inner_text, options, depth + 1);

                        let start_offset = plain.len();
                        plain.push_str(&inner_plain);
                        let end_offset = plain.len();

                        spans.push(MarkupSpan::new_paired(
                            start_offset..end_offset,
                            "~~".to_string(),
                            "~~".to_string(),
                            depth,
                        ));

                        for mut span in inner_spans {
                            span.source_range.start += start_offset;
                            span.source_range.end += start_offset;
                            spans.push(span);
                        }

                        i = close_rel + 2;
                        continue;
                    }
                }
            }
        }

        // 8. Markdown Bold / Italic: ***, **, *, ___, __, _
        if options.parse_markdown && (bytes[i] == b'*' || bytes[i] == b'_') {
            let marker_char = bytes[i];
            let start_idx = i;
            let mut marker_len = 0;
            while i < len && bytes[i] == marker_char {
                marker_len += 1;
                i += 1;
            }

            if marker_len <= 3 {
                let marker = &text[start_idx..i];
                let is_underscore = marker_char == b'_';

                let left_flanking_ok = if is_underscore {
                    start_idx == 0 || !text[..start_idx].chars().next_back().unwrap().is_alphanumeric()
                } else {
                    true
                };

                let not_followed_by_space = text[i..].chars().next().is_some_and(|c| !c.is_whitespace());

                if left_flanking_ok && not_followed_by_space {
                    if let Some(close_idx) = find_markdown_delimiter_close(text, i, marker_char, marker_len) {
                        let inner_text = &text[i..close_idx];
                        if !inner_text.trim().is_empty() {
                            let (inner_plain, inner_spans) = extract_inline_impl(inner_text, options, depth + 1);

                            let start_offset = plain.len();
                            plain.push_str(&inner_plain);
                            let end_offset = plain.len();

                            spans.push(MarkupSpan::new_paired(
                                start_offset..end_offset,
                                marker.to_string(),
                                marker.to_string(),
                                depth,
                            ));

                            for mut span in inner_spans {
                                span.source_range.start += start_offset;
                                span.source_range.end += start_offset;
                                spans.push(span);
                            }

                            i = close_idx + marker_len;
                            continue;
                        }
                    }
                }
                plain.push_str(marker);
                continue;
            } else {
                plain.push_str(&text[start_idx..i]);
                continue;
            }
        }

        // 9. Regular characters (chunk contiguous plain text to avoid per-char looping)
        let run_start = i;
        let c = text[i..].chars().next().unwrap();
        i += c.len_utf8();
        while i < len {
            let b = bytes[i];
            let is_trigger = (options.parse_html && (b == b'<' || b == b'&'))
                || (options.parse_markdown
                    && (b == b'`'
                        || b == b'*'
                        || b == b'_'
                        || b == b'~'
                        || b == b'['
                        || (b == b'!' && i + 1 < len && bytes[i + 1] == b'[')));
            if is_trigger {
                break;
            }
            if b < 0x80 {
                i += 1;
            } else if (b & 0xE0) == 0xC0 && i + 1 < len {
                i += 2;
            } else if (b & 0xF0) == 0xE0 && i + 2 < len {
                i += 3;
            } else if i + 3 < len {
                i += 4;
            } else {
                i += 1;
            }
        }
        plain.push_str(&text[run_start..i]);
    }

    (plain, spans)
}

/// Find the end index (pointing after '>') of an HTML tag starting at `start`.
/// Correctly ignores '>' characters enclosed in attribute quotes.
pub(crate) fn find_tag_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut in_quote = None;
    let mut i = start + 1;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_quote {
            if b == q {
                in_quote = None;
            }
        } else {
            match b {
                b'"' | b'\'' => in_quote = Some(b),
                b'>' => return Some(i + 1),
                _ => {}
            }
        }
        i += 1;
    }
    None
}

/// Extract basic tag metadata without heap allocation: `(is_closing, tag_name, is_self_closing)`.
#[inline]
pub(crate) fn parse_tag_info_borrowed(tag: &str) -> (bool, &str, bool) {
    let trimmed = tag.trim_start_matches('<').trim_end_matches('>');
    let is_closing = trimmed.starts_with('/');
    let clean = if is_closing {
        trimmed[1..].trim_start()
    } else {
        trimmed
    };

    let tag_name_end = clean
        .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
        .unwrap_or(clean.len());
    let tag_name = &clean[..tag_name_end];

    let is_self_closing = trimmed.ends_with('/')
        || tag_name.eq_ignore_ascii_case("br")
        || tag_name.eq_ignore_ascii_case("hr")
        || tag_name.eq_ignore_ascii_case("img")
        || tag_name.eq_ignore_ascii_case("wbr")
        || tag_name.eq_ignore_ascii_case("input")
        || tag_name.eq_ignore_ascii_case("meta")
        || tag_name.eq_ignore_ascii_case("link");

    (is_closing, tag_name, is_self_closing)
}

/// Extract basic tag metadata: `(is_closing, tag_name_lowercase, is_self_closing)`.
#[inline]
pub(crate) fn parse_tag_info(tag: &str) -> (bool, String, bool) {
    let (is_closing, name, is_self_closing) = parse_tag_info_borrowed(tag);
    (is_closing, name.to_ascii_lowercase(), is_self_closing)
}

/// Search for matching closing tag `</tag_name>` in `html`, accounting for nested instances.
/// Returns `Some((close_tag_start_offset, close_tag_end_offset))` relative to `html`.
/// Performs 0 heap allocations and zero-copy case-insensitive tag matching.
pub(crate) fn find_matching_close_tag(html: &str, tag_name: &str) -> Option<(usize, usize)> {
    if !html.contains("</") {
        return None;
    }
    let bytes = html.as_bytes();
    let mut depth = 0;
    let mut i = 0;

    while i < bytes.len() {
        let next = match bytes[i..].iter().position(|&b| b == b'<') {
            Some(pos) => i + pos,
            None => break,
        };
        i = next;

        if i + 1 < bytes.len() {
            let next_b = bytes[i + 1];
            if next_b == b'/' || next_b.is_ascii_alphabetic() {
                if let Some(tag_end) = find_tag_end(bytes, i) {
                    let raw_tag = &html[i..tag_end];
                    let (is_closing, name, is_self_closing) = parse_tag_info_borrowed(raw_tag);
                    if name.eq_ignore_ascii_case(tag_name) {
                        if is_closing {
                            if depth == 0 {
                                return Some((i, tag_end));
                            }
                            depth -= 1;
                        } else if !is_self_closing {
                            depth += 1;
                        }
                    }
                    i = tag_end;
                    continue;
                }
            }
        }
        i += 1;
    }
    None
}

/// Try to recognize a valid HTML entity starting at the beginning of `text`.
/// Supports:
/// - Hexadecimal numeric entities: `&#x1f600;`, `&#X20;`
/// - Decimal numeric entities: `&#39;`, `&#123;`
/// - Named entities: `&copy;`, `&amp;`, `&lt;`, `&gt;`, `&quot;`, `&trade;`, `&nbsp;`, `&mdash;`, etc.
///
/// Returns the byte length of the matched entity if valid.
pub(crate) fn try_parse_html_entity(text: &str) -> Option<usize> {
    if !text.starts_with('&') {
        return None;
    }
    let rest = &text[1..];
    if rest.starts_with("#x") || rest.starts_with("#X") {
        let hex_part = &rest[2..];
        let mut count = 0;
        for b in hex_part.bytes() {
            if b == b';' {
                if (1..=8).contains(&count) {
                    return Some(1 + 2 + count + 1);
                }
                return None;
            }
            if b.is_ascii_hexdigit() {
                count += 1;
            } else {
                return None;
            }
        }
        None
    } else if let Some(dec_part) = rest.strip_prefix('#') {
        let mut count = 0;
        for b in dec_part.bytes() {
            if b == b';' {
                if (1..=8).contains(&count) {
                    return Some(1 + 1 + count + 1);
                }
                return None;
            }
            if b.is_ascii_digit() {
                count += 1;
            } else {
                return None;
            }
        }
        None
    } else {
        // Named entity: &[a-zA-Z][a-zA-Z0-9]{1,31};
        let mut count = 0;
        let mut first = true;
        for b in rest.bytes() {
            if b == b';' {
                if (1..=32).contains(&count) {
                    return Some(1 + count + 1);
                }
                return None;
            }
            if first {
                if !b.is_ascii_alphabetic() {
                    return None;
                }
                first = false;
            } else if !b.is_ascii_alphanumeric() {
                return None;
            }
            count += 1;
            if count > 32 {
                return None;
            }
        }
        None
    }
}

/// Decode a recognized HTML named or common punctuation entity into its UTF-8 string slice.
pub fn decode_single_html_entity(raw: &str) -> Option<&'static str> {
    match raw {
        "&amp;" => Some("&"),
        "&lt;" => Some("<"),
        "&gt;" => Some(">"),
        "&quot;" => Some("\""),
        "&apos;" | "&#39;" | "&#039;" => Some("'"),
        "&nbsp;" => Some(" "),
        "&mdash;" => Some("—"),
        "&ndash;" => Some("–"),
        "&hellip;" => Some("…"),
        "&lsquo;" => Some("‘"),
        "&rsquo;" => Some("’"),
        "&ldquo;" => Some("“"),
        "&rdquo;" => Some("”"),
        "&copy;" => Some("©"),
        "&reg;" => Some("®"),
        "&trade;" => Some("™"),
        "&cent;" => Some("¢"),
        "&pound;" => Some("£"),
        "&yen;" => Some("¥"),
        "&euro;" => Some("€"),
        "&times;" => Some("×"),
        "&divide;" => Some("÷"),
        "&deg;" => Some("°"),
        "&plusmn;" => Some("±"),
        "&le;" => Some("≤"),
        "&ge;" => Some("≥"),
        "&ne;" => Some("≠"),
        "&infin;" => Some("∞"),
        _ => None,
    }
}

/// Decode any numeric entity (decimal or hex) into a char.
pub fn decode_numeric_html_entity(raw: &str) -> Option<char> {
    let inner = raw.strip_prefix('&')?.strip_suffix(';')?;
    if let Some(hex_str) = inner.strip_prefix("#x").or_else(|| inner.strip_prefix("#X")) {
        let code = u32::from_str_radix(hex_str, 16).ok()?;
        char::from_u32(code)
    } else if let Some(dec_str) = inner.strip_prefix('#') {
        let code = dec_str.parse::<u32>().ok()?;
        char::from_u32(code)
    } else {
        None
    }
}

/// Unescape HTML character entities in text to plain natural language UTF-8 characters.
pub fn unescape_html_entities(text: &str) -> std::borrow::Cow<'_, str> {
    if !text.contains('&') {
        return std::borrow::Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut last_copied = 0;
    let mut i = 0;
    let bytes = text.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'&' {
            if let Some(entity_len) = try_parse_html_entity(&text[i..]) {
                let entity_raw = &text[i..i + entity_len];
                let decoded_str = decode_single_html_entity(entity_raw);
                let decoded_char = if decoded_str.is_none() {
                    decode_numeric_html_entity(entity_raw)
                } else {
                    None
                };
                if decoded_str.is_some() || decoded_char.is_some() {
                    out.push_str(&text[last_copied..i]);
                    if let Some(s) = decoded_str {
                        out.push_str(s);
                    } else if let Some(c) = decoded_char {
                        out.push(c);
                    }
                    i += entity_len;
                    last_copied = i;
                    continue;
                }
            }
        }
        i += 1;
    }
    if last_copied == 0 {
        std::borrow::Cow::Borrowed(text)
    } else {
        if last_copied < text.len() {
            out.push_str(&text[last_copied..]);
        }
        std::borrow::Cow::Owned(out)
    }
}

pub(crate) fn find_matching_bracket(
    bytes: &[u8],
    open_idx: usize,
    open_byte: u8,
    close_byte: u8,
) -> Option<usize> {
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

/// Find closing backtick fence of exact length `fence_len` in text.
/// Returns `Some((content_end_offset, after_fence_offset))`.
pub(crate) fn find_backtick_closing(text: &str, fence_len: usize) -> Option<(usize, usize)> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i;
            while i < bytes.len() && bytes[i] == b'`' {
                i += 1;
            }
            let count = i - start;
            if count == fence_len {
                return Some((start, i));
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Find closing delimiter run for bold/italic markers (* or _).
pub(crate) fn find_markdown_delimiter_close(
    text: &str,
    start_offset: usize,
    marker_char: u8,
    marker_len: usize,
) -> Option<usize> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = start_offset;

    while i < len {
        if bytes[i] == marker_char {
            let run_start = i;
            while i < len && bytes[i] == marker_char {
                i += 1;
            }
            let count = i - run_start;
            if count == marker_len {
                // Right-flanking check: preceding char must not be whitespace
                if run_start > 0 {
                    let prev_char = text[..run_start].chars().next_back().unwrap();
                    if prev_char.is_whitespace() {
                        continue;
                    }
                }
                // For underscore, following char must not be alphanumeric
                if marker_char == b'_' && i < len {
                    let next_char = text[i..].chars().next().unwrap();
                    if next_char.is_alphanumeric() {
                        continue;
                    }
                }
                return Some(run_start);
            }
        } else {
            i += 1;
        }
    }
    None
}

/// Find closing `~~` delimiter for strikethrough.
pub(crate) fn find_strikethrough_close(text: &str, start_offset: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = start_offset;

    while i + 1 < len {
        if bytes[i] == b'~' && bytes[i + 1] == b'~' {
            let run_start = i;
            if run_start > 0 {
                let prev_char = text[..run_start].chars().next_back().unwrap();
                if prev_char.is_whitespace() {
                    i += 2;
                    continue;
                }
            }
            return Some(run_start);
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::super::render_projected_markup;
    use super::*;

    fn assert_round_trip(original: &str) {
        let (plain, mut spans) = extract_unified_inline(original);
        for span in spans.iter_mut() {
            span.target_range = Some(span.source_range.clone());
        }
        let rendered = render_projected_markup(&plain, &spans);
        assert_eq!(rendered, original, "Round-trip failed for: {}", original);
    }

    #[test]
    fn test_html_inline_paired_tags() {
        let text = "Text with <a>link</a>, <b>bold</b>, <strong>strong</strong>, <i>italic</i>, <em>emphasis</em>, <sup>sup</sup>, <sub>sub</sub>, <mark>mark</mark>, <del>del</del>, <span>span</span>, <abbr title=\"API\">abbr</abbr>, <cite>cite</cite>, <q>quote</q>.";
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(
            plain,
            "Text with link, bold, strong, italic, emphasis, sup, sub, mark, del, span, abbr, cite, quote."
        );
        assert_eq!(spans.len(), 13);
        assert_round_trip(text);
    }

    #[test]
    fn test_html_attributes_complete_retention() {
        let text = r#"Check out <a href="https://example.com?query=1&amp;filter=true" class="btn btn-primary" target="_blank" data-id="123">Our Portal</a> now."#;
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(plain, "Check out Our Portal now.");
        assert_eq!(spans.len(), 1);
        assert_eq!(
            spans[0].open_markup,
            r#"<a href="https://example.com?query=1&amp;filter=true" class="btn btn-primary" target="_blank" data-id="123">"#
        );
        assert_eq!(spans[0].close_markup, "</a>");
        assert_round_trip(text);
    }

    #[test]
    fn test_html_void_tags() {
        let text = r#"Image: <img src="pic.jpg" alt="Photo >_<" class="rounded" /> and break <br> and rule <hr/> done."#;
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(plain, "Image:  and break  and rule  done.");
        assert_eq!(spans.len(), 3);
        assert!(spans.iter().all(|s| s.is_self_closing));
        assert_round_trip(text);
    }

    #[test]
    fn test_markdown_links_and_images() {
        let text = "Read [documentation](https://docs.rs) or view ![logo](https://example.com/logo.png).";
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(plain, "Read documentation or view .");
        assert_eq!(spans.len(), 2);

        // Link is paired
        assert_eq!(spans[0].open_markup, "[");
        assert_eq!(spans[0].close_markup, "](https://docs.rs)");
        assert!(!spans[0].is_self_closing);

        // Image is self-closing point
        assert_eq!(spans[1].open_markup, "![logo](https://example.com/logo.png)");
        assert!(spans[1].is_self_closing);

        assert_round_trip(text);
    }

    #[test]
    fn test_markdown_formatting() {
        let text = "This is **bold1** and __bold2__ with *italic1* and _italic2_ and ~~deleted~~.";
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(plain, "This is bold1 and bold2 with italic1 and italic2 and deleted.");
        assert_eq!(spans.len(), 5);
        assert_round_trip(text);
    }

    #[test]
    fn test_html_entities_atomic_protection() {
        let text = "Entities: &amp;, &lt;, &gt;, &quot;, &#39;, &copy;, &reg;, &trade;, &le;, &ge;, &ne;, &infin;, &alpha;, &beta;, &gamma;, &sum;, &times;, &divide;, &nbsp;, &mdash;, &ndash;, &#123;, &#x1f600;.";
        let (plain, spans) = extract_unified_inline(text);
        // None of the entity contents are sent in plain text!
        assert_eq!(plain, "Entities: , , , , , , , , , , , , , , , , , , , , , , .");
        assert_eq!(spans.len(), 23);
        assert!(spans.iter().all(|s| s.is_self_closing));
        assert_round_trip(text);
    }

    #[test]
    fn test_consecutive_entities_no_collapse() {
        let text = "Bitwise &amp;&amp; and copyright &copy;&copy; test";
        let (plain, spans) = extract_unified_inline(text);
        assert_eq!(plain, "Bitwise  and copyright  test");
        assert_eq!(spans.len(), 4);
        assert_round_trip(text);
    }

    #[test]
    fn test_atomic_code_tags_and_backticks() {
        let text = "Compare `std::vector<int>` and <code>a < b && c > d</code> and <kbd>Ctrl+C</kbd> and <math><mi>x</mi></math> and <pre>formatted\ncode</pre>.";
        let (plain, spans) = extract_unified_inline(text);
        // Code content is omitted from plain text to prevent NMT translation distortion
        assert_eq!(plain, "Compare  and  and  and  and .");
        assert_eq!(spans.len(), 5);
        assert!(spans.iter().all(|s| s.is_self_closing));
        assert_eq!(spans[0].open_markup, "`std::vector<int>`");
        assert_eq!(spans[1].open_markup, "<code>a < b && c > d</code>");
        assert_eq!(spans[2].open_markup, "<kbd>Ctrl+C</kbd>");
        assert_eq!(spans[3].open_markup, "<math><mi>x</mi></math>");
        assert_eq!(spans[4].open_markup, "<pre>formatted\ncode</pre>");
        assert_round_trip(text);
    }

    #[test]
    fn test_nested_dual_syntax() {
        let text = "Here is **[a bold link with `code` &amp; &copy;](https://example.com)** and <i>[italic link](https://example.net)</i>.";
        let (_plain, _spans) = extract_unified_inline(text);
        assert_round_trip(text);
    }

    #[test]
    fn test_nested_html_tags() {
        let text = r#"<div class="card"><p>Welcome <b><a href="/help">user <i>profile</i></a></b> &copy; 2026</p></div>"#;
        let (plain, _spans) = extract_unified_inline(text);
        assert_eq!(plain, "Welcome user profile  2026");
        assert_round_trip(text);
    }

    #[test]
    fn test_false_positives_resistance() {
        let text = "Formula: x < 10 && y > 20, generic List<String>, var foo_bar_baz, math 2 * 3 * 4 = 24, and AT&T.";
        let (plain, spans) = extract_unified_inline(text);
        // None of these mathematical, generic-type, or variable constructs should trigger markup extraction
        assert_eq!(plain, text);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_parse_formats_control() {
        let text = "Use **bold** and <i>italic</i> and &copy; 2026";

        // 1. ALL
        let (plain_all, spans_all) = extract_inline_markup(text, InlineParseOptions::ALL);
        assert_eq!(plain_all, "Use bold and italic and  2026");
        assert_eq!(spans_all.len(), 3);

        // 2. HTML ONLY: Markdown ** is not parsed
        let (plain_html, spans_html) = extract_inline_markup(text, InlineParseOptions::HTML_ONLY);
        assert_eq!(plain_html, "Use **bold** and italic and  2026");
        assert_eq!(spans_html.len(), 2);

        // 3. MARKDOWN ONLY: HTML <i> and &copy; are not parsed
        let (plain_md, spans_md) = extract_inline_markup(text, InlineParseOptions::MARKDOWN_ONLY);
        assert_eq!(plain_md, "Use bold and <i>italic</i> and &copy; 2026");
        assert_eq!(spans_md.len(), 1);
    }

    #[test]
    fn test_unescape_html_entities() {
        let text = "It&#39;s a test &amp; you&#39;re welcome! Price: &euro;100 &copy; 2026. Hex: &#x1F600;.";
        let unescaped = unescape_html_entities(text);
        assert_eq!(
            unescaped,
            "It's a test & you're welcome! Price: €100 © 2026. Hex: 😀."
        );
        let plain = "No entities here.";
        assert!(matches!(unescape_html_entities(plain), std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_tag_classification_functions() {
        assert!(is_atomic_tag("code"));
        assert!(is_atomic_tag("pre"));
        assert!(is_atomic_tag("script"));
        assert!(!is_atomic_tag("div"));
        assert!(!is_atomic_tag("span"));

        assert!(is_void_tag("img"));
        assert!(is_void_tag("br"));
        assert!(is_void_tag("hr"));
        assert!(!is_void_tag("p"));
        assert!(!is_void_tag("b"));

        assert!(is_recognized_inline_tag("a"));
        assert!(is_recognized_inline_tag("b"));
        assert!(is_recognized_inline_tag("strong"));
        assert!(is_recognized_inline_tag("span"));
        assert!(!is_recognized_inline_tag("div"));
        assert!(!is_recognized_inline_tag("table"));
    }

    #[test]
    fn test_inline_recursion_depth_limit() {
        let depth = 50;
        let mut text = String::new();
        for _ in 0..depth {
            text.push_str("<b>");
        }
        text.push_str("nested");
        for _ in 0..depth {
            text.push_str("</b>");
        }

        let (plain, spans) = extract_unified_inline(&text);
        assert_eq!(spans.len(), MAX_INLINE_DEPTH);
        assert!(plain.contains("<b>"));
        assert!(plain.contains("nested"));
    }

    #[test]
    fn test_inline_deep_nesting_no_stack_overflow() {
        let depth = 5000;
        let mut text = String::with_capacity(depth * 7 + 10);
        for _ in 0..depth {
            text.push_str("<i>");
        }
        text.push_str("deep");
        for _ in 0..depth {
            text.push_str("</i>");
        }
        let (plain, spans) = extract_unified_inline(&text);
        assert_eq!(spans.len(), MAX_INLINE_DEPTH);
        assert!(plain.contains("deep"));
    }
}
