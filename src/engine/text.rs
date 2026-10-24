//! Text pre/post-processing for translation requests.
//!
//! Ported from core/engine.ts: HTML sanitization, `{0}`/`[0]` placeholder
//! tagging, emoji hiding, long-text splitting, and the corresponding restore
//! passes. Kept separate from [`super`] so the engine module only orchestrates
//! model calls.

use std::borrow::Cow;

use crate::utils::lang;

/// One long-text splitting candidate: the literal separator used to split,
/// plus the rejoins emitted for CJK vs non-CJK target languages.
pub struct Separator {
    pub sep: &'static str,
    pub cjk: &'static str,
    pub latin: &'static str,
}

/// Long-text splitting candidates, tried in order until one produces parts
/// that all fit within the sentence-length limit. This single table replaces
/// the previous pair of parallel lists (the split order array and the
/// `mapped_separator` match arms) that had to be kept in sync manually.
pub const SEPARATORS: &[Separator] = &[
    Separator {
        sep: "\r\n",
        cjk: "\r\n",
        latin: "\r\n",
    },
    Separator {
        sep: "\n",
        cjk: "\n",
        latin: "\n",
    },
    Separator {
        sep: " - ",
        cjk: " - ",
        latin: " - ",
    },
    Separator {
        sep: ". ",
        cjk: "。",
        latin: ". ",
    },
    Separator {
        sep: "。",
        cjk: "。",
        latin: ". ",
    },
    Separator {
        sep: "！",
        cjk: "！",
        latin: "! ",
    },
    Separator {
        sep: "!",
        cjk: "！",
        latin: "! ",
    },
    Separator {
        sep: "？",
        cjk: "？",
        latin: "? ",
    },
    Separator {
        sep: "?",
        cjk: "？",
        latin: "? ",
    },
    Separator {
        sep: "; ",
        cjk: "；",
        latin: "; ",
    },
    Separator {
        sep: "；",
        cjk: "；",
        latin: "; ",
    },
    Separator {
        sep: "：",
        cjk: "：",
        latin: ": ",
    },
    Separator {
        sep: ": ",
        cjk: "：",
        latin: ": ",
    },
    Separator {
        sep: "，",
        cjk: "，",
        latin: ", ",
    },
    Separator {
        sep: ", ",
        cjk: "，",
        latin: ", ",
    },
];

/// Rejoin separator for `sep`, mapped by target language (CJK vs non-CJK).
pub fn mapped_separator<'a>(sep: &'a str, target_lang: &str) -> &'a str {
    if let Some(prof) = crate::engine::rules::RuleRegistry::global().profile(target_lang) {
        if let Some(custom) = prof.typography.separators.get(sep) {
            for s in SEPARATORS {
                if s.cjk == custom.as_str() {
                    return s.cjk;
                }
                if s.latin == custom.as_str() {
                    return s.latin;
                }
            }
        }
    }
    let is_cjk = lang::is_cjk_code(target_lang);
    SEPARATORS
        .iter()
        .find(|s| s.sep == sep)
        .map(|s| if is_cjk { s.cjk } else { s.latin })
        .unwrap_or(sep)
}

/// Normalizes typographical variants, academic symbols, and zero-width markers
/// that can trigger byte-fallback tokenization loops.
pub fn normalize_academic_and_special_symbols(text: &str) -> Cow<'_, str> {
    let has_special = text.chars().any(|c| {
        matches!(
            c,
            '∗' | '•' | '‧'
                | '\u{200B}'
                | '\u{FEFF}'
                | '\u{2060}'
        )
    });
    if !has_special {
        return Cow::Borrowed(text);
    }

    let mut out = String::with_capacity(text.len());

    for c in text.chars() {
        match c {
            '∗' | '•' | '‧' => {
                out.push('*');
            }
            '\u{200B}' | '\u{FEFF}' | '\u{2060}' => {}
            _ => {
                out.push(c);
            }
        }
    }

    Cow::Owned(out)
}



/// Hide emoji characters and template interpolation variables `{{...}}` behind `[EE<i>]` placeholders.
pub fn hide_emojis(text: &str) -> (Cow<'_, str>, Vec<String>) {
    let has_emoji_lead = text.as_bytes().iter().any(|&b| b >= 0xE2);
    let has_template = text.contains("{{");
    if !has_emoji_lead && !has_template {
        return (Cow::Borrowed(text), vec![]);
    }

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut replacements: Vec<String> = Vec::new();
    let mut clean = String::with_capacity(len + 16);
    let mut i = 0;
    let mut last = 0;

    while i < len {
        if has_template && bytes[i] == b'{' && i + 1 < len && bytes[i + 1] == b'{' {
            if let Some(pos) = text[i + 2..].find("}}") {
                let end = i + 2 + pos + 2;
                let inner = &text[i + 2..end - 2];
                if !inner.contains('\n') && !inner.contains('\r') {
                    if i > last {
                        clean.push_str(&text[last..i]);
                    }
                    clean.push_str("[EE");
                    push_usize(&mut clean, replacements.len());
                    clean.push(']');
                    replacements.push(text[i..end].to_string());
                    i = end;
                    last = end;
                    continue;
                }
            }
        }

        if has_emoji_lead && bytes[i] >= 0xE2 {
            if let Some(c) = text[i..].chars().next() {
                if is_emoji(c) {
                    if i > last {
                        clean.push_str(&text[last..i]);
                    }
                    clean.push_str("[EE");
                    push_usize(&mut clean, replacements.len());
                    clean.push(']');
                    replacements.push(c.to_string());
                    let c_len = c.len_utf8();
                    i += c_len;
                    last = i;
                    continue;
                }
            }
        }

        let c_len = text[i..].chars().next().map_or(1, |c| c.len_utf8());
        i += c_len;
    }

    if replacements.is_empty() {
        return (Cow::Borrowed(text), vec![]);
    }

    if last < len {
        clean.push_str(&text[last..]);
    }
    (Cow::Owned(clean), replacements)
}

pub fn restore_emojis<'a>(text: &'a str, replacements: &[String]) -> Cow<'a, str> {
    restore_emojis_with_spans(text, replacements, &mut [])
}

/// Remap token byte spans in-place based on non-overlapping string replacements.
/// `replacements` contains `(pos_in_old, old_len, new_len)` sorted by position ascending.
pub fn remap_spans(
    spans: &mut [std::ops::Range<usize>],
    replacements: &[(usize, usize, usize)],
) {
    if spans.is_empty() || replacements.is_empty() {
        return;
    }

    let remap_point = |p: usize| -> usize {
        let mut shift: isize = 0;
        for &(pos, old_len, new_len) in replacements {
            let old_end = pos + old_len;
            let delta = new_len as isize - old_len as isize;
            if p >= old_end {
                shift += delta;
            } else if p > pos {
                let offset = p - pos;
                let p_mapped = pos as isize + shift + offset.min(new_len) as isize;
                return p_mapped.max(0) as usize;
            } else {
                break;
            }
        }
        (p as isize + shift).max(0) as usize
    };

    for span in spans.iter_mut() {
        let new_s = remap_point(span.start);
        let new_e = remap_point(span.end);
        span.start = new_s.min(new_e);
        span.end = new_e;
    }
}
pub fn restore_emojis_with_spans<'a>(
    text: &'a str,
    replacements: &[String],
    spans: &mut [std::ops::Range<usize>],
) -> Cow<'a, str> {
    restore_emojis_with_source_and_spans(text, replacements, "", spans)
}

/// Restore emoji placeholders `[EE{i}]` with original emoji strings,
/// taking into account the source text's preceding whitespace context.
pub fn restore_emojis_with_source_and_spans<'a>(
    text: &'a str,
    replacements: &[String],
    source: &str,
    spans: &mut [std::ops::Range<usize>],
) -> Cow<'a, str> {
    if replacements.is_empty() {
        return Cow::Borrowed(text);
    }

    let track_spans = !spans.is_empty();
    let mut scan_offset = 0;
    let mut result: Option<String> = None;
    let mut deltas: Vec<(usize, usize, usize)> = Vec::new();

    let repl_leading_ws: Vec<bool> = if source.is_empty() {
        Vec::new()
    } else {
        let mut ws = Vec::with_capacity(replacements.len());
        let mut src_cursor = 0;
        for r in replacements {
            if let Some(pos) = source[src_cursor..].find(r.as_str()) {
                let abs_pos = src_cursor + pos;
                let has_ws = abs_pos > 0
                    && source[..abs_pos]
                        .chars()
                        .next_back()
                        .map_or(false, |c| c.is_whitespace());
                ws.push(has_ws);
                src_cursor = abs_pos + r.len();
            } else {
                ws.push(false);
            }
        }
        ws
    };

    while let Some(start) = text[scan_offset..].find("[EE") {
        let match_start = scan_offset + start;
        let after_prefix = &text[match_start + 3..];
        let bytes = after_prefix.as_bytes();
        let mut digit_end = 0;
        while digit_end < bytes.len() && bytes[digit_end].is_ascii_digit() {
            digit_end += 1;
        }

        let matched = if digit_end > 0 && digit_end < bytes.len() && bytes[digit_end] == b']' {
            if let Ok(index) = after_prefix[..digit_end].parse::<usize>() {
                if let Some(original) = replacements.get(index) {
                    let match_len = 3 + digit_end + 1;
                    let src_has_ws = repl_leading_ws.get(index).copied().unwrap_or(false);
                    Some((match_len, original.as_str(), src_has_ws))
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        if let Some((match_len, original, src_has_ws)) = matched {
            let res = match &mut result {
                Some(r) => {
                    r.push_str(&text[scan_offset..match_start]);
                    r
                }
                None => {
                    let mut r = String::with_capacity(text.len() + 16);
                    r.push_str(&text[..match_start]);
                    result = Some(r);
                    result.as_mut().unwrap()
                }
            };

            let mut added_space = false;
            if src_has_ws {
                if let Some(prev) = res.chars().next_back() {
                    if !prev.is_whitespace() && !crate::utils::lang::is_cjk_fullwidth_punct(prev) {
                        res.push(' ');
                        added_space = true;
                    }
                }
            }

            res.push_str(original);

            if track_spans {
                deltas.push((match_start, match_len, original.len() + if added_space { 1 } else { 0 }));
            }

            scan_offset = match_start + match_len;
        } else {
            if let Some(r) = &mut result {
                r.push_str(&text[scan_offset..match_start + 3]);
            }
            scan_offset = match_start + 3;
        }
    }

    match result {
        None => Cow::Borrowed(text),
        Some(mut res) => {
            res.push_str(&text[scan_offset..]);
            if track_spans && !deltas.is_empty() {
                remap_spans(spans, &deltas);
            }
            Cow::Owned(res)
        }
    }
}

#[inline(always)]
fn is_emoji(c: char) -> bool {
    let u = c as u32;
    if u < 0x2600 {
        return false;
    }
    // Emoji presentation: misc symbols and pictographs, transport and
    // map symbols (U+1F680-U+1F6FF), supplementals (U+1F900-U+1F9FF)
    (0x1F300..=0x1F9FF).contains(&u)
        // Misc symbols, dingbats and arrows (U+2600-U+27BF)
        || (0x2600..=0x27BF).contains(&u)
        // Region indicator (U+1F1E6-U+1F1FF)
        || (0x1F1E6..=0x1F1FF).contains(&u)
        // Symbols and pictographs extended-A (U+1FA70-U+1FAFF)
        || (0x1FA70..=0x1FAFF).contains(&u)
        // Symbols for legacy computing (U+1FB00-U+1FBFF)
        || (0x1FB00..=0x1FBFF).contains(&u)
        // Variation selectors (U+FE00-U+FE0F)
        || (0xFE00..=0xFE0F).contains(&u)
}

/// Fast check whether `text.chars().count() > limit`, avoiding walking characters
/// when byte length already guarantees `< limit` or `> limit * 4`.
#[inline]
pub fn exceeds_char_limit(text: &str, limit: usize) -> bool {
    let byte_len = text.len();
    if byte_len <= limit {
        return false;
    }
    if byte_len > limit.saturating_mul(4) {
        return true;
    }
    text.chars().nth(limit).is_some()
}

/// Fast character count: uses byte length if ASCII, otherwise counts Unicode characters.
#[inline]
pub fn char_count(text: &str) -> usize {
    if text.is_ascii() {
        text.len()
    } else {
        text.chars().count()
    }
}

pub fn chunk_by_word_boundary(text: &str, limit: usize) -> Vec<String> {
    let mut chunks: Vec<String> = vec![];
    let mut current = String::new();
    let mut current_len: usize = 0;

    for part in text.split_inclusive(|c: char| c.is_whitespace()) {
        let part_len = part.chars().count();
        if current_len + part_len <= limit {
            current.push_str(part);
            current_len += part_len;
        } else {
            if !current.is_empty() {
                chunks.push(current);
                current = String::new();
                current_len = 0;
            }
            if part_len > limit {
                let sub = chunk_by_length(part, limit);
                chunks.extend(sub);
            } else {
                current = part.to_string();
                current_len = part_len;
            }
        }
    }

    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// Rewrite compound tokens into plain word sequences before translation:
/// separators between letters of the same case (`snap-compacted`,
/// `user_name`) become spaces, and a lowercase→uppercase boundary
/// (`getUserData`) gets a space too. Three token shapes are kept verbatim:
/// a digit adjacent to the separator (dates, versions: `2024-01-15`,
/// `v2.0.1-beta`), any token containing an embedded dot (filenames,
/// domains: `package-lock.json`), and caseless scripts (CJK prose:
/// Split text into contiguous spans alternating between outside-of-tag (plain
/// text) and inside-of-tag (HTML/XML markup `<...>`).
///
/// Yields `(in_tag, span)` tuples. Neither `in_tag` flag is emitted twice in a
/// row; spans cover the entire input slice with zero allocations.
pub fn tag_aware_spans(text: &str) -> impl Iterator<Item = (bool, &str)> {
    struct TagSpanIter<'a> {
        text: &'a str,
        pos: usize,
        in_tag: bool,
    }

    impl<'a> Iterator for TagSpanIter<'a> {
        type Item = (bool, &'a str);

        fn next(&mut self) -> Option<Self::Item> {
            if self.pos >= self.text.len() {
                return None;
            }

            let start = self.pos;
            let bytes = self.text.as_bytes();

            if self.in_tag {
                let mut i = self.pos;
                while i < bytes.len() {
                    if bytes[i] == b'>' {
                        i += 1;
                        self.in_tag = false;
                        self.pos = i;
                        return Some((true, &self.text[start..i]));
                    }
                    i += 1;
                }
                self.pos = bytes.len();
                Some((true, &self.text[start..]))
            } else {
                let mut i = self.pos;
                while i < bytes.len() {
                    if bytes[i] == b'<' {
                        if i > start {
                            self.pos = i;
                            return Some((false, &self.text[start..i]));
                        }
                        self.in_tag = true;
                        i += 1;
                        while i < bytes.len() {
                            if bytes[i] == b'>' {
                                i += 1;
                                self.in_tag = false;
                                break;
                            }
                            i += 1;
                        }
                        self.pos = i;
                        return Some((true, &self.text[start..i]));
                    }
                    i += 1;
                }
                self.pos = bytes.len();
                Some((false, &self.text[start..]))
            }
        }
    }

    TagSpanIter {
        text,
        pos: 0,
        in_tag: false,
    }
}
fn is_alphanumeric_code(token: &str) -> bool {
    let has_digit = token.chars().any(|c| c.is_ascii_digit());
    let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
    if has_digit && has_upper && token.chars().all(|c| c.is_ascii_alphanumeric()) {
        return true;
    }
    false
}

fn get_prev_alpha(run: &str, pos: usize) -> &str {
    let bytes = run[..pos].as_bytes();
    let mut start = pos;
    while start > 0 && bytes[start - 1].is_ascii_alphabetic() {
        start -= 1;
    }
    &run[start..pos]
}

fn get_next_alpha(run: &str, pos: usize) -> &str {
    let bytes = run[pos..].as_bytes();
    let mut len = 0;
    while len < bytes.len() && bytes[len].is_ascii_alphabetic() {
        len += 1;
    }
    &run[pos..pos + len]
}

fn is_hyphen_protected(prev: &str, next: &str) -> bool {
    crate::engine::rules::RuleRegistry::global().is_hyphen_protected(prev, next)
}

fn is_quoted_run(span: &str, start: usize, end: usize) -> bool {
    let before = &span[..start];
    let after = &span[end..];

    if before.is_empty() || after.is_empty() {
        return false;
    }

    let prev_char = before.chars().next_back();
    let next_char = after.chars().next();

    // Direct wrapping: "token", 'token', `token`, “token”, ‘token’
    if let (Some(p), Some(n)) = (prev_char, next_char) {
        if (p == '"' && n == '"')
            || (p == '\'' && n == '\'')
            || (p == '`' && n == '`')
            || (p == '“' && n == '”')
            || (p == '‘' && n == '’')
        {
            return true;
        }
    }

    let before_bytes = before.as_bytes();
    let after_bytes = after.as_bytes();

    // Fast short-circuit & single-pass scan of `after`:
    // An enclosing quote requires a closing quote in `after`.
    let mut has_backtick_after = false;
    let mut has_dquote_after = false;
    let mut has_squote_after = false;
    let mut has_curly_after = false;

    for &b in after_bytes {
        match b {
            b'`' => has_backtick_after = true,
            b'"' => has_dquote_after = true,
            b'\'' => has_squote_after = true,
            0xE2 => has_curly_after = true,
            _ => {}
        }
        if has_backtick_after && has_dquote_after && has_squote_after && has_curly_after {
            break;
        }
    }

    // If `after` contains no quote/backtick bytes, no enclosing quote can match.
    if !(has_backtick_after || has_dquote_after || has_squote_after || has_curly_after) {
        return false;
    }

    // Single-pass scan of `before` collecting counts/presence only for quote types present in `after`.
    let mut backticks_before = 0usize;
    let mut dquotes_before = 0usize;
    let mut has_squote_before = false;
    let mut has_curly_before = false;

    for &b in before_bytes {
        match b {
            b'`' if has_backtick_after => backticks_before += 1,
            b'"' if has_dquote_after => dquotes_before += 1,
            b'\'' if has_squote_after => has_squote_before = true,
            0xE2 if has_curly_after => has_curly_before = true,
            _ => {}
        }
    }

    // If `before` contains no matching opening quotes, short-circuit.
    if !(backticks_before > 0 || dquotes_before > 0 || has_squote_before || has_curly_before) {
        return false;
    }

    // Enclosing backticks: `... token ...`
    if has_backtick_after && backticks_before % 2 == 1 {
        return true;
    }

    // Enclosing double quotes: " ... token ... "
    if has_dquote_after && dquotes_before % 2 == 1 {
        return true;
    }

    // Enclosing curly double quotes: “ ... token ... ”
    if has_curly_after && has_curly_before {
        if let Some(open_idx) = before.rfind('“') {
            let last_close_curly = before.rfind('”');
            if last_close_curly.map_or(true, |close_idx| close_idx < open_idx) && after.contains('”') {
                return true;
            }
        }
    }

    // Enclosing single quotes: ' ... token ... '
    if has_squote_after && has_squote_before {
        if let Some(open_idx) = before.rfind('\'') {
            let is_open_quote = open_idx == 0
                || before[..open_idx]
                    .chars()
                    .next_back()
                    .map_or(true, |c| c.is_whitespace() || c.is_ascii_punctuation());
            if is_open_quote {
                if let Some(close_idx) = after.find('\'') {
                    let is_close_quote = after[close_idx + 1..]
                        .chars()
                        .next()
                        .map_or(true, |c| c.is_whitespace() || c.is_ascii_punctuation());
                    if is_close_quote {
                        return true;
                    }
                }
            }
        }
    }

    false
}


pub fn preprocess_symbols(text: &str) -> Cow<'_, str> {
    // Cheap reject: a rule can only fire on a separator or a camel hump.
    let mut prev = '\0';
    let candidate = text.chars().any(|c| {
        let hit = c == '-'
            || c == '_'
            || (prev.is_lowercase() && c.is_uppercase());
        prev = c;
        hit
    });
    if !candidate {
        return Cow::Borrowed(text);
    }

    let is_token_char = |c: char| c.is_alphanumeric() || c == '.' || c == '-' || c == '_';
    // Cased letters only: Latin, Greek, Cyrillic qualify; CJK and other
    // caseless scripts do not.
    let is_cased = |c: char| c.is_lowercase() || c.is_uppercase();

    // Characters indicating structured context (URLs, paths, email, query, namespaces)
    let is_struct_char = |c: char| matches!(c, '/' | '\\' | ':' | '@' | '?' | '#' | '=' | '&');

    let mut out: Option<String> = None;
    let mut copied_up_to = 0;
    let mut span_start = 0;

    for (in_tag, span) in tag_aware_spans(text) {
        if in_tag {
            span_start += span.len();
            continue;
        }
        let span_has_quotes = span
            .as_bytes()
            .iter()
            .any(|&b| matches!(b, b'`' | b'"' | b'\'' | 0xE2));

        let mut i = 0;
        while i < span.len() {
            let c = span[i..].chars().next().unwrap();
            if !is_token_char(c) {
                i += c.len_utf8();
                continue;
            }

            // Lookbehind before this token run starts
            let prev_char = if i > 0 {
                span[..i].chars().next_back().unwrap_or('\0')
            } else {
                '\0'
            };

            // Collect the whole token run: alphanumerics joined by `.`, `-`, `_`.
            let start = i;
            i += c.len_utf8();
            while i < span.len() {
                let c = span[i..].chars().next().unwrap();
                if !is_token_char(c) {
                    break;
                }
                i += c.len_utf8();
            }
            let run = &span[start..i];

            // Lookahead right after this token run ends
            let next_char = if i < span.len() {
                span[i..].chars().next().unwrap_or('\0')
            } else {
                '\0'
            };

            // Guard 1: if adjacent to a structured entity boundary (URL path / query / email / namespace),
            // or if adjacent to a dot (domain segment / file extension / property call), keep verbatim.
            if is_struct_char(prev_char)
                || is_struct_char(next_char)
                || prev_char == '.'
                || next_char == '.'
            {
                continue;
            }

            // Guard 2: Embedded dot (`lock.json`, `e.g.`) marks a filename/domain/
            // abbreviation — keep the run exactly as written.
            let has_embedded_dot = {
                let mut prev = '\0';
                let mut hit = false;
                let mut chars = run.chars().peekable();
                while let Some(c) = chars.next() {
                    if c == '.'
                        && prev.is_alphanumeric()
                        && chars.peek().is_some_and(|&n| n.is_alphanumeric())
                    {
                        hit = true;
                        break;
                    }
                    prev = c;
                }
                hit
            };
            if has_embedded_dot {
                continue;
            }

            // Guard 3: Words/tokens wrapped in double quotes, single quotes, or backticks
            // must not be split.
            if span_has_quotes && is_quoted_run(span, start, i) {
                continue;
            }

            // Guard 4: Tokens consisting of alphanumeric codes (e.g. B2B, IPv6, P2P, H2O, MP3, A4)
            // must not be split.
            if is_alphanumeric_code(run) {
                continue;
            }

            let run_start_in_text = span_start + start;
            let mut prev = '\0';
            let mut chars = run.char_indices().peekable();
            while let Some((run_offset, c)) = chars.next() {
                if c == '-' {
                    let prev_word = get_prev_alpha(run, run_offset);
                    let next_word = get_next_alpha(run, run_offset + 1);
                    if is_hyphen_protected(prev_word, next_word) {
                        prev = c;
                        continue;
                    }
                    let next = chars.peek().map(|&(_, ch)| ch).unwrap_or('\0');
                    if is_cased(prev)
                        && is_cased(next)
                        && prev.is_uppercase() == next.is_uppercase()
                    {
                        let char_pos = run_start_in_text + run_offset;
                        let s = out.get_or_insert_with(|| String::with_capacity(text.len() + 16));
                        s.push_str(&text[copied_up_to..char_pos]);
                        s.push(' ');
                        copied_up_to = char_pos + c.len_utf8();
                    }
                } else if c == '_' {
                    let next = chars.peek().map(|&(_, ch)| ch).unwrap_or('\0');
                    if is_cased(prev)
                        && is_cased(next)
                        && prev.is_uppercase() == next.is_uppercase()
                    {
                        let char_pos = run_start_in_text + run_offset;
                        let s = out.get_or_insert_with(|| String::with_capacity(text.len() + 16));
                        s.push_str(&text[copied_up_to..char_pos]);
                        s.push(' ');
                        copied_up_to = char_pos + c.len_utf8();
                    }
                } else {
                    // Camel hump: lowercase letter before an uppercase one.
                    // Numbers and caseless scripts are unaffected.
                    if prev.is_lowercase() && c.is_uppercase() {
                        let char_pos = run_start_in_text + run_offset;
                        let s = out.get_or_insert_with(|| String::with_capacity(text.len() + 16));
                        s.push_str(&text[copied_up_to..char_pos]);
                        s.push(' ');
                        copied_up_to = char_pos;
                    }
                }
                prev = c;
            }
        }
        span_start += span.len();
    }

    if let Some(mut s) = out {
        s.push_str(&text[copied_up_to..]);
        Cow::Owned(s)
    } else {
        Cow::Borrowed(text)
    }
}

fn chunk_by_length(text: &str, chunk_size: usize) -> Vec<String> {
    if chunk_size == 0 {
        return vec![text.to_string()];
    }
    let chars: Vec<char> = text.chars().collect();
    let mut chunks = vec![];
    let mut i = 0;
    while i < chars.len() {
        let end = (i + chunk_size).min(chars.len());
        chunks.push(chars[i..end].iter().collect());
        i = end;
    }
    chunks
}
#[inline]
fn push_usize(buf: &mut String, mut n: usize) {
    if n < 10 {
        buf.push((b'0' + n as u8) as char);
    } else {
        let mut tmp = [0u8; 20];
        let mut len = 0;
        while n > 0 {
            tmp[len] = b'0' + (n % 10) as u8;
            n /= 10;
            len += 1;
        }
        tmp[..len].reverse();
        // SAFETY: only ASCII digits
        buf.push_str(unsafe { std::str::from_utf8_unchecked(&tmp[..len]) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hide_and_restore_placeholders() {
        let input = "set SECRET=\"{{APP_SECRET_KEY}}\" and emoji 👋 done.";
        let (clean, repls) = hide_emojis(input);
        assert_eq!(clean, "set SECRET=\"[EE0]\" and emoji [EE1] done.");
        assert_eq!(repls, vec!["{{APP_SECRET_KEY}}".to_string(), "👋".to_string()]);

        let restored = restore_emojis_with_source_and_spans(&clean, &repls, input, &mut []);
        assert_eq!(restored, input);
    }
}

