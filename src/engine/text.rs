//! Text pre/post-processing for translation requests.
//!
//! Ported from core/engine.ts: HTML sanitization, `{0}`/`[0]` placeholder
//! tagging, emoji hiding, long-text splitting, and the corresponding restore
//! passes. Kept separate from [`super`] so the engine module only orchestrates
//! model calls.

use std::borrow::Cow;

use unicode_segmentation::UnicodeSegmentation;

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
    if text.is_ascii() {
        return Cow::Borrowed(text);
    }
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



/// Helper to detect template interpolation variables (`{{...}}`, `{identifier}`, `${VAR}`)
/// starting at byte index `start`.
/// Returns the end byte index (one past the closing delimiter) if a valid variable placeholder is matched.
fn match_template_or_var_placeholder(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if start >= len {
        return None;
    }

    // 1. Shell variable: ${VAR} or ${HTTP_PORT:-8080}; Ruby / template variable: #{expr} or #{order_id}
    if (bytes[start] == b'$' || bytes[start] == b'#') && start + 1 < len && bytes[start + 1] == b'{' {
        let rest = &text[start + 2..];
        if let Some(pos) = rest.find('}') {
            let inner = &rest[..pos];
            if !inner.is_empty()
                && inner.len() <= 128
                && !inner.contains(|c: char| c.is_whitespace() || c == '{' || c == '}' || c == '"' || c == '\'')
                && inner.chars().any(|c| c.is_alphanumeric())
            {
                return Some(start + 2 + pos + 1);
            }
        }
        return None;
    }

    // 2. Double-brace template variable: {{...}}
    if bytes[start] == b'{' && start + 1 < len && bytes[start + 1] == b'{' {
        let rest = &text[start + 2..];
        if let Some(pos) = rest.find("}}") {
            let inner = &rest[..pos];
            if !inner.contains('\n') && !inner.contains('\r') {
                return Some(start + 2 + pos + 2);
            }
        }
        return None;
    }

    // 3. Single-brace variable: {username}, {order_id}, {address}, {0}, {1}, {user.name}, etc.
    if bytes[start] == b'{' {
        let rest = &text[start + 1..];
        if let Some(pos) = rest.find('}') {
            let inner = &rest[..pos];
            if !inner.is_empty()
                && inner.len() <= 128
                && !inner.contains(|c: char| c.is_whitespace() || c == '{' || c == '}')
                && inner.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ':')
                && inner.chars().any(|c| c.is_alphanumeric())
            {
                return Some(start + 1 + pos + 1);
            }
        }
    }

    None
}

/// Helper to detect C / POSIX / Python-style printf format placeholders (`%s`, `%d`, `%.2f`, `%%`, etc.)
/// starting at byte index `start`.
/// Returns the end byte index if a valid format specifier is matched.
pub(crate) fn match_printf_placeholder(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    if start >= len || bytes[start] != b'%' {
        return None;
    }
    if start + 1 >= len {
        return None;
    }

    // 1. Escaped percent: %%
    if bytes[start + 1] == b'%' {
        return Some(start + 2);
    }

    let mut i = start + 1;

    // 2. Python mapping key: %(name)
    if bytes[i] == b'(' {
        let pos = bytes[i + 1..].iter().position(|&b| b == b')')?;
        let key = &bytes[i + 1..i + 1 + pos];
        if !key.is_empty() && key.iter().all(|&b| b.is_ascii_alphanumeric() || b == b'_') {
            i = i + 1 + pos + 1;
        } else {
            return None;
        }
    } else {
        // POSIX positional argument: %1$
        let mut p = i;
        while p < len && bytes[p].is_ascii_digit() {
            p += 1;
        }
        if p > i && p < len && bytes[p] == b'$' {
            i = p + 1;
        }
    }

    // 3. Flags: zero or more of -, +, ' ', #, 0, '
    while i < len && matches!(bytes[i], b'-' | b'+' | b' ' | b'#' | b'0' | b'\'') {
        i += 1;
    }

    // 4. Width: digits or *
    if i < len && bytes[i] == b'*' {
        i += 1;
    } else {
        while i < len && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }

    // 5. Precision: . followed by digits or *
    if i < len && bytes[i] == b'.' {
        i += 1;
        if i < len && bytes[i] == b'*' {
            i += 1;
        } else {
            while i < len && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
    }

    // 6. Length modifier: hh, h, ll, l, j, z, t, L
    if i < len {
        if bytes[i..].starts_with(b"hh") || bytes[i..].starts_with(b"ll") {
            i += 2;
        } else if matches!(bytes[i], b'h' | b'l' | b'j' | b'z' | b't' | b'L') {
            i += 1;
        }
    }

    // 7. Conversion Specifier
    if i >= len {
        return None;
    }
    let spec = bytes[i];
    let is_specifier = matches!(
        spec,
        b's' | b'd' | b'i' | b'o' | b'u' | b'x' | b'X' | b'f' | b'F' | b'e' | b'E' | b'g' | b'G' | b'a' | b'A' | b'c' | b'p'
    );
    if !is_specifier {
        return None;
    }

    // 8. Terminal boundary defense: character immediately following specifier cannot be ASCII alphanumeric or '_'
    let end = i + 1;
    if end < len {
        let next_b = bytes[end];
        if next_b.is_ascii_alphanumeric() || next_b == b'_' {
            return None;
        }
    }

    Some(end)
}

/// Hide emoji characters and template interpolation variables (`{{...}}`, `{var}`, `${VAR}`, `%s`/`%d`) behind `[EE<i>]` placeholders.
pub fn hide_emojis(text: &str) -> (Cow<'_, str>, Vec<String>) {
    let has_emoji_lead = text.as_bytes().iter().any(|&b| b >= 0xE2);
    let has_template_or_var = text.contains('{') || text.contains('$') || text.contains('%') || text.contains("#{");
    if !has_emoji_lead && !has_template_or_var {
        return (Cow::Borrowed(text), vec![]);
    }

    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut replacements: Vec<String> = Vec::new();
    let mut clean = String::with_capacity(len + 16);
    let mut i = 0;
    let mut last = 0;

    while i < len {
        if has_template_or_var {
            if bytes[i] == b'{' || ((bytes[i] == b'$' || bytes[i] == b'#') && i + 1 < len && bytes[i + 1] == b'{') {
                if let Some(end) = match_template_or_var_placeholder(text, i) {
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
            } else if bytes[i] == b'%' {
                if let Some(end) = match_printf_placeholder(text, i) {
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

        if has_emoji_lead {
            let is_candidate = bytes[i] >= 0xE2 || matches!(bytes[i], b'0'..=b'9' | b'#' | b'*');
            if is_candidate {
                if let Some(g) = text[i..].graphemes(true).next() {
                    if is_emoji_grapheme(g) {
                        if i > last {
                            clean.push_str(&text[last..i]);
                        }
                        clean.push_str("[EE");
                        push_usize(&mut clean, replacements.len());
                        clean.push(']');
                        replacements.push(g.to_string());
                        i += g.len();
                        last = i;
                        continue;
                    }
                    i += g.len();
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
#[inline]
fn scan_next_placeholder(text: &str, start: usize) -> Option<(usize, usize, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut i = start;

    while i < len {
        let (open_len, is_open) = if bytes[i] == b'[' {
            (1, true)
        } else if bytes[i..].starts_with("［".as_bytes()) {
            (3, true)
        } else if bytes[i..].starts_with("【".as_bytes()) {
            (3, true)
        } else {
            (1, false)
        };

        if !is_open {
            i += 1;
            continue;
        }

        let mut p = i + open_len;
        while p < len && (bytes[p] == b' ' || bytes[p] == b'\t') {
            p += 1;
        }

        if p + 2 <= len
            && (bytes[p] == b'E' || bytes[p] == b'e')
            && (bytes[p + 1] == b'E' || bytes[p + 1] == b'e')
        {
            p += 2;
            while p < len && (bytes[p] == b' ' || bytes[p] == b'\t') {
                p += 1;
            }

            let num_start = p;
            while p < len && bytes[p].is_ascii_digit() {
                p += 1;
            }

            if p > num_start {
                if let Ok(idx) = text[num_start..p].parse::<usize>() {
                    while p < len && (bytes[p] == b' ' || bytes[p] == b'\t') {
                        p += 1;
                    }
                    let close_len = if p < len && bytes[p] == b']' {
                        1
                    } else if bytes[p..].starts_with("］".as_bytes()) {
                        3
                    } else if bytes[p..].starts_with("】".as_bytes()) {
                        3
                    } else {
                        0
                    };
                    let match_len = (p + close_len) - i;
                    return Some((i, match_len, idx));
                }
            }
        }

        i += open_len;
    }
    None
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

    let mut used = vec![false; replacements.len()];

    while let Some((match_start, match_len, index)) = scan_next_placeholder(text, scan_offset) {
        if let Some(original) = replacements.get(index) {
            used[index] = true;
            let src_has_ws = repl_leading_ws.get(index).copied().unwrap_or(false);
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
                r.push_str(&text[scan_offset..match_start + match_len]);
            }
            scan_offset = match_start + match_len;
        }
    }

    let mut final_res = match result {
        None => {
            if used.iter().any(|&u| !u) {
                Some(text.to_string())
            } else {
                None
            }
        }
        Some(mut res) => {
            res.push_str(&text[scan_offset..]);
            if track_spans && !deltas.is_empty() {
                remap_spans(spans, &deltas);
            }
            Some(res)
        }
    };

    // Safety net: restore any dropped template variables that were swallowed by the NMT model
    for (idx, &is_used) in used.iter().enumerate() {
        if !is_used {
            let original = &replacements[idx];
            if original.starts_with('{') || original.starts_with('$') || original.starts_with('#') || original.starts_with('%') {
                let res = final_res.get_or_insert_with(|| text.to_string());

                // Locate nearest restored placeholder to maintain relative order
                let insert_pos = if let Some(next_idx) = (idx + 1..replacements.len()).find(|&k| used[k]) {
                    let next_str = &replacements[next_idx];
                    res.find(next_str)
                } else if let Some(prev_idx) = (0..idx).rev().find(|&k| used[k]) {
                    let prev_str = &replacements[prev_idx];
                    res.find(prev_str).map(|p| p + prev_str.len())
                } else {
                    None
                };

                match insert_pos {
                    Some(pos) => {
                        let mut insert_str = String::with_capacity(original.len() + 2);
                        if pos > 0 && !res[..pos].ends_with(' ') {
                            insert_str.push(' ');
                        }
                        insert_str.push_str(original);
                        if pos < res.len() && !res[pos..].starts_with(' ') {
                            insert_str.push(' ');
                        }
                        res.insert_str(pos, &insert_str);
                    }
                    None => {
                        let term_pos = res.trim_end().rfind(|c: char| matches!(c, '.' | '。' | '!' | '！' | '?' | '？'));
                        if let Some(p) = term_pos {
                            let mut insert_str = String::with_capacity(original.len() + 2);
                            if p > 0 && !res[..p].ends_with(' ') {
                                insert_str.push(' ');
                            }
                            insert_str.push_str(original);
                            res.insert_str(p, &insert_str);
                        } else {
                            if !res.is_empty() && !res.ends_with(' ') {
                                res.push(' ');
                            }
                            res.push_str(original);
                        }
                    }
                }
            }
        }
    }
    match final_res {
        None => Cow::Borrowed(text),
        Some(res) => Cow::Owned(res),
    }
}

#[inline(always)]
pub fn is_emoji_char(c: char) -> bool {
    let u = c as u32;
    if u < 0x20E3 {
        return false;
    }
    (0x1F000..=0x1FBFF).contains(&u)
        || (0x2600..=0x27BF).contains(&u)
        || (0x2300..=0x23FF).contains(&u)
        || (0x2B00..=0x2BFF).contains(&u)
        || u == 0x20E3
        || u == 0xFE0F
}

#[inline(always)]
pub fn is_emoji_grapheme(g: &str) -> bool {
    g.chars().any(is_emoji_char)
}

#[inline(always)]
pub fn is_emoji(c: char) -> bool {
    is_emoji_char(c)
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

    #[test]
    fn test_composite_emojis_protection() {
        let input = "Family 👨‍👩‍👧‍👦, like 👍🏽, China 🇨🇳, keycap 1️⃣, fire ❤️‍🔥, heart ❤️!";
        let (clean, repls) = hide_emojis(input);
        assert_eq!(clean, "Family [EE0], like [EE1], China [EE2], keycap [EE3], fire [EE4], heart [EE5]!");
        assert_eq!(
            repls,
            vec![
                "👨‍👩‍👧‍👦".to_string(),
                "👍🏽".to_string(),
                "🇨🇳".to_string(),
                "1️⃣".to_string(),
                "❤️‍🔥".to_string(),
                "❤️".to_string(),
            ]
        );

        let restored = restore_emojis_with_source_and_spans(&clean, &repls, input, &mut []);
        assert_eq!(restored, input);
    }
    #[test]
    fn test_single_brace_and_shell_var_placeholders() {
        let input = "Welcome, {username}! Your order #{order_id} has shipped to {address}. Items: {0}, {1}, ${VAR}, and ${PORT:-8080}.";
        let (clean, repls) = hide_emojis(input);
        assert_eq!(
            clean,
            "Welcome, [EE0]! Your order [EE1] has shipped to [EE2]. Items: [EE3], [EE4], [EE5], and [EE6]."
        );
        assert_eq!(
            repls,
            vec![
                "{username}".to_string(),
                "#{order_id}".to_string(),
                "{address}".to_string(),
                "{0}".to_string(),
                "{1}".to_string(),
                "${VAR}".to_string(),
                "${PORT:-8080}".to_string(),
            ]
        );

        let restored = restore_emojis_with_source_and_spans(&clean, &repls, input, &mut []);
        assert_eq!(restored, input);

        // Negative cases that should not be matched as placeholders
        let neg = "This is not { a variable } nor empty {} nor json {\"key\": 1} nor {a, b}.";
        let (neg_clean, neg_repls) = hide_emojis(neg);
        assert_eq!(neg_clean, neg);
        assert!(neg_repls.is_empty());
    }

    #[test]
    fn test_printf_format_placeholders() {
        let input = "User %s consumed %d quota, remaining %.2f%%. Retries=%d, pos=%1$08d, name=%(user_name)s.";
        let (clean, repls) = hide_emojis(input);
        assert_eq!(
            clean,
            "User [EE0] consumed [EE1] quota, remaining [EE2][EE3]. Retries=[EE4], pos=[EE5], name=[EE6]."
        );
        assert_eq!(
            repls,
            vec![
                "%s".to_string(),
                "%d".to_string(),
                "%.2f".to_string(),
                "%%".to_string(),
                "%d".to_string(),
                "%1$08d".to_string(),
                "%(user_name)s".to_string(),
            ]
        );
        let restored = restore_emojis_with_source_and_spans(&clean, &repls, input, &mut []);
        assert_eq!(restored, input);

        // Negative cases (natural language percentages)
        let neg = "50% off and 10% discount on 100% load.";
        let (neg_clean, neg_repls) = hide_emojis(neg);
        assert_eq!(neg_clean, neg);
        assert!(neg_repls.is_empty());
    }

    #[test]
    fn test_corrupted_placeholder_restoration_and_fallback() {
        let repls = vec!["{username}".to_string(), "👋".to_string(), "${CONFIG}".to_string()];
        // Model generated lowercase [ee0], extra space [EE 1], and dropped ${CONFIG}
        let model_output = "Hello [ee0], see [EE 1]!";
        let restored = restore_emojis_with_source_and_spans(model_output, &repls, "", &mut []);
        // Must restore both username and emoji, AND append missing variable ${CONFIG}
        assert!(restored.contains("Hello {username}"));
        assert!(restored.contains("👋"));
        assert!(restored.contains("${CONFIG}"));
    }
}
