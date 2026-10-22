//! Intelligent sentence splitting and markup span remapping for paragraphs and multiline text.
//!
//! Provides sentence boundary detection with abbreviation and decimal protection,
//! safe convergence of cross-sentence [`MarkupSpan`]s, and an execution pipeline
//! for translating split sentences independently with token alignment projection
//! before reassembling with original separators.

use std::borrow::Cow;
use std::ops::Range;
use rayon::prelude::*;

use std::sync::Arc;
use crate::engine::rules::{LanguageRuleProfile, RuleRegistry};
use super::{render_projected_markup, snap_markup_spans_to_word_boundaries, MarkupSpan};

/// A segmented sentence within a paragraph or document block.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SplitSentence<'a> {
    /// Plain text of the sentence to be translated.
    pub text: Cow<'a, str>,
    /// Byte range `[start..end]` of `text` in the original input string.
    pub text_range: Range<usize>,
    /// Delimiter or whitespace following this sentence (e.g. `" "`, `"\n"`, `"\r\n"`, or `""`).
    pub separator: Cow<'a, str>,
    /// Localized markup spans mapped to `0..text.len()`.
    pub inline_spans: Vec<MarkupSpan>,
}

impl<'a> SplitSentence<'a> {
    /// Create a new sentence unit without spans.
    pub fn new(
        text: impl Into<Cow<'a, str>>,
        text_range: Range<usize>,
        separator: impl Into<Cow<'a, str>>,
    ) -> Self {
        Self {
            text: text.into(),
            text_range,
            separator: separator.into(),
            inline_spans: Vec::new(),
        }
    }

    /// Create a sentence unit from borrowed string slices without heap allocation.
    pub fn from_borrowed(text: &'a str, text_range: Range<usize>, separator: &'a str) -> Self {
        Self {
            text: Cow::Borrowed(text),
            text_range,
            separator: Cow::Borrowed(separator),
            inline_spans: Vec::new(),
        }
    }

    /// Whether this sentence has empty or whitespace-only text.
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }

    /// Full byte range in the source text including trailing separator.
    pub fn full_range(&self) -> Range<usize> {
        self.text_range.start..(self.text_range.end + self.separator.len())
    }

    /// Convert into an owned static sentence unit.
    pub fn into_owned(self) -> SplitSentence<'static> {
        SplitSentence {
            text: Cow::Owned(self.text.into_owned()),
            text_range: self.text_range,
            separator: Cow::Owned(self.separator.into_owned()),
            inline_spans: self.inline_spans,
        }
    }
}

#[inline]
fn consume_while<F>(text: &str, mut offset: usize, pred: F) -> usize
where
    F: Fn(char) -> bool,
{
    while offset < text.len() {
        match text.get(offset..).and_then(|s| s.chars().next()) {
            Some(c) if pred(c) => offset += c.len_utf8(),
            _ => break,
        }
    }
    offset
}

/// Heuristically detect dominant language from text character statistics and detector service.
pub fn detect_dominant_lang(text: &str) -> &'static str {
    crate::engine::rules::detect::detect_dominant_language(text)
}

/// Unified dominant language detection delegating to rules::detect.
#[inline]
pub fn detect_dominant_language(text: &str) -> &'static str {
    crate::engine::rules::detect::detect_dominant_language(text)
}

/// Resolve a language rule profile for text or explicit language code.
///
/// If no external rules have been loaded, returns None (pure Universal Semiotic Topology).
pub(crate) fn resolve_profile_for_text(
    text: &str,
    lang: Option<&str>,
) -> Option<Arc<LanguageRuleProfile>> {
    let registry = RuleRegistry::global();
    if registry.is_empty() {
        return None;
    }
    if let Some(l) = lang {
        return registry.get(l);
    }
    let detected = detect_dominant_lang(text);
    registry.get(detected)
}

/// Split a paragraph or multiline text into sentences, preserving all original separators.
///
/// Handles sentence-ending boundaries:
/// - Western punctuation: `". "`, `"! "`, `"? "`
/// - CJK punctuation: `'。'`, `'！'`, `'？'`, `'；'`
/// - Newlines: `"\n"`, `"\r\n"`, and multiline breaks
///
/// Protects:
/// - Abbreviations (e.g. `e.g.`, `i.e.`, `Dr.`, `Mr.`, `vs.`, `Fig.`, `No.`)
/// - Decimals and version numbers (e.g. `49.99`, `3.14`, `v2.0`, `1.2.3`)
/// - Acronyms and name initials (e.g. `John F. Kennedy`)
pub fn split_sentences_with_profile<'a>(
    text: &'a str,
    profile: Option<&LanguageRuleProfile>,
) -> Vec<SplitSentence<'a>> {
    if text.is_empty() {
        return Vec::new();
    }

    let estimated_cap = (text.len() / 64).clamp(1, 64);
    let mut sentences = Vec::with_capacity(estimated_cap);
    let bytes = text.as_bytes();
    let len = text.len();
    let mut cursor = 0;

    let mut i = 0;
    while i < len {
        let c = match text.get(i..).and_then(|s| s.chars().next()) {
            Some(c) => c,
            None => break,
        };
        let c_len = c.len_utf8();

        // 1. CJK & RTL sentence-ending punctuation: 。 ！ ？ / ؟ ۔ ׃
        if is_cjk_punctuation(c) || is_rtl_terminal_punctuation(c) {
            let punct_end = consume_while(text, i + c_len, |next_char| {
                is_cjk_punctuation(next_char)
                    || is_rtl_terminal_punctuation(next_char)
                    || is_western_sentence_punctuation(next_char)
                    || is_closing_bracket_or_quote(next_char)
            });
            let sep_end = consume_while(text, punct_end, |next_char| next_char.is_whitespace());

            sentences.push(SplitSentence::from_borrowed(
                &text[cursor..punct_end],
                cursor..punct_end,
                &text[punct_end..sep_end],
            ));
            cursor = sep_end;
            i = sep_end;
            continue;
        }
        // 1.5 CJK semicolon '；': conditional split
        // Avoid over-splitting extremely short parallel clauses (< 15 chars) in the initial phase,
        // maintaining higher-order clause (Tier 2 MajorClause) or conditional split behavior similar to Western semicolons.
        if c == '；' {
            let clause_chars = text.get(cursor..i).map_or(0, |s| s.chars().count());
            let next_after = text.get(i + c_len..).unwrap_or("").trim_start();
            let next_is_sentence_starter = next_after
                .chars()
                .next()
                .map(|nc| nc.is_uppercase() || is_discourse_chunk_starter_with_profile(next_after, profile))
                .unwrap_or(false);
            if clause_chars >= 15 || next_is_sentence_starter {
                let punct_end = consume_while(text, i + c_len, |next_char| {
                    is_cjk_punctuation(next_char)
                        || is_rtl_terminal_punctuation(next_char)
                        || is_western_sentence_punctuation(next_char)
                        || is_closing_bracket_or_quote(next_char)
                });
                let sep_end = consume_while(text, punct_end, |next_char| next_char.is_whitespace());

                sentences.push(SplitSentence::from_borrowed(
                    &text[cursor..punct_end],
                    cursor..punct_end,
                    &text[punct_end..sep_end],
                ));
                cursor = sep_end;
                i = sep_end;
                continue;
            }
        }

        // 2. Western sentence-ending punctuation: . ! ?
        if is_western_sentence_punctuation(c) && should_split_western(text, i, c, profile) {
            let punct_end = consume_while(text, i + c_len, |next_char| {
                is_western_sentence_punctuation(next_char) || is_closing_bracket_or_quote(next_char)
            });

            // Western boundary requires following whitespace or end of string
            let has_following_ws = punct_end == len
                || text.get(punct_end..)
                    .and_then(|s| s.chars().next())
                    .map(|ch| ch.is_whitespace())
                    .unwrap_or(false);

            if has_following_ws {
                let sep_end = consume_while(text, punct_end, |next_char| next_char.is_whitespace());

                sentences.push(SplitSentence::from_borrowed(
                    &text[cursor..punct_end],
                    cursor..punct_end,
                    &text[punct_end..sep_end],
                ));
                cursor = sep_end;
                i = sep_end;
                continue;
            }
        }

        // 3. Bare newline boundary (\n, \r\n) without preceding punctuation
        if c == '\n' {
            let line_end = if i > cursor && bytes.get(i - 1) == Some(&b'\r') {
                i - 1
            } else {
                i
            };

            // Trim trailing horizontal spaces before newline into separator
            let mut text_end = line_end;
            while text_end > cursor
                && (bytes.get(text_end - 1) == Some(&b' ') || bytes.get(text_end - 1) == Some(&b'\t'))
            {
                text_end -= 1;
            }

            let sep_end = consume_while(text, i + 1, |next_char| next_char == '\n' || next_char == '\r');

            sentences.push(SplitSentence::from_borrowed(
                &text[cursor..text_end],
                cursor..text_end,
                &text[text_end..sep_end],
            ));
            cursor = sep_end;
            i = sep_end;
            continue;
        }

        i += c_len;
    }

    if cursor < len {
        sentences.push(SplitSentence::from_borrowed(
            &text[cursor..len],
            cursor..len,
            "",
        ));
    }

    sentences
}

/// Default sentence length limit for bounded hierarchical splitting.
pub const DEFAULT_MAX_SENTENCE_LEN: usize = 512;
pub const MAX_SENTENCE_LEN: usize = DEFAULT_MAX_SENTENCE_LEN;
/// Split text into sentences with language-aware rule profile lookup and adaptive discourse chunk fusion.
pub fn split_sentences_with_lang<'a>(text: &'a str, lang: Option<&str>) -> Vec<SplitSentence<'a>> {
    split_sentences_bounded_with_lang(text, DEFAULT_MAX_SENTENCE_LEN, lang)
}

/// Split a paragraph or multiline text into sentences, preserving all original separators and fusing discourse chunks.
pub fn split_sentences<'a>(text: &'a str) -> Vec<SplitSentence<'a>> {
    split_sentences_with_lang(text, None)
}

/// Check if character is CJK sentence-ending punctuation: 。 ！ ？
#[inline]
fn is_cjk_punctuation(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '．')
}

/// Check if character is RTL sentence-ending terminal punctuation: ؟ (Arabic question mark), ۔ (Urdu full stop), ׃ (Hebrew Sof Pasuq).
#[inline]
fn is_rtl_terminal_punctuation(c: char) -> bool {
    matches!(c, '؟' | '۔' | '׃')
}

/// Check if character is Western sentence-ending punctuation: . ! ?
#[inline]
fn is_western_sentence_punctuation(c: char) -> bool {
    matches!(c, '.' | '!' | '?')
}

/// Check if character is closing bracket, parenthesis, or quote.
#[inline]
fn is_closing_bracket_or_quote(c: char) -> bool {
    matches!(
        c,
        '"' | '\''
            | ')'
            | ']'
            | '}'
            | '”'
            | '’'
            | '」'
            | '』'
            | '）'
            | '】'
            | '›'
            | '»'
            | '«'
            | '‹'
    )
}

/// Check if abbreviation token is strictly an absolute non-sentence-ending marker.
#[inline]
fn is_absolute_abbreviation(token: &str) -> bool {
    let clean = token.strip_suffix('.').unwrap_or(token);
    matches!(
        clean.to_ascii_lowercase().as_str(),
        "e.g" | "eg" | "i.e" | "ie" | "vs" | "v" | "cf" | "ca" | "viz"
    )
}

/// Check if abbreviation token is a label or reference prefix (Fig., Eq., No.).
#[inline]
fn is_label_abbreviation(token: &str) -> bool {
    let clean = token.strip_suffix('.').unwrap_or(token);
    matches!(
        clean.to_ascii_lowercase().as_str(),
        "fig" | "figs" | "eq" | "eqs" | "no" | "nos" | "ref" | "refs" | "sec" | "chap" | "vol" | "vols" | "tab"
    )
}

/// Check if abbreviation token is an honorific or personal title prefix (Dr., Prof., Mr.).
#[inline]
fn is_honorific_abbreviation(token: &str) -> bool {
    let clean = token.strip_suffix('.').unwrap_or(token);
    matches!(
        clean.to_ascii_lowercase().as_str(),
        "dr" | "mr" | "mrs" | "ms" | "prof" | "rev" | "hon" | "st" | "pres" | "gov" | "sen" | "rep"
    )
}

/// Evaluate whether Western punctuation at index `pos` represents a true sentence split boundary.
#[inline]
fn matches_conditional_end(token: &str) -> bool {
    let clean = token.strip_suffix('.').unwrap_or(token);
    clean.eq_ignore_ascii_case("etc")
        || clean.eq_ignore_ascii_case("al")
        || clean.eq_ignore_ascii_case("inc")
        || clean.eq_ignore_ascii_case("ltd")
        || clean.eq_ignore_ascii_case("co")
        || clean.eq_ignore_ascii_case("corp")
        || clean.eq_ignore_ascii_case("usw")
        || clean.eq_ignore_ascii_case("bzw")
}

/// Check if remainder starts with a high-frequency Western sentence starter word.
#[inline]
fn is_high_frequency_sentence_starter(remainder: &str) -> bool {
    let word_end = remainder
        .find(|c: char| !c.is_alphabetic())
        .unwrap_or(remainder.len());
    let word = &remainder[..word_end];
    matches!(
        word,
        "The"
            | "A"
            | "An"
            | "This"
            | "That"
            | "These"
            | "Those"
            | "He"
            | "She"
            | "It"
            | "They"
            | "We"
            | "You"
            | "In"
            | "On"
            | "At"
            | "If"
            | "When"
            | "There"
            | "Please"
            | "Wait"
            | "Turn"
            | "Switch"
            | "Open"
            | "Close"
            | "Click"
            | "Press"
            | "Start"
            | "Stop"
            | "Check"
            | "Look"
            | "Do"
            | "Make"
            | "Take"
            | "Get"
            | "See"
            | "Go"
            | "Come"
            | "Run"
            | "Use"
            | "Set"
            | "Now"
            | "Then"
            | "Here"
            | "So"
            | "However"
            | "Also"
    )
}

/// Check if text preceding a Japanese comma or dunhao ends with a head-final subordinating conjunction marker.
#[inline]
fn is_japanese_head_final_clause_marker(before: &str) -> bool {
    let s = before.trim_end();
    s.ends_with("ので")
        || s.ends_with("から")
        || s.ends_with("たら")
        || s.ends_with("なら")
        || s.ends_with("ば")
        || s.ends_with("けれども")
        || s.ends_with("けれど")
        || s.ends_with("ものの")
}

fn should_split_western(
    text: &str,
    pos: usize,
    c: char,
    profile: Option<&LanguageRuleProfile>,
) -> bool {
    let bytes = text.as_bytes();
    if c == '.' {
        // Spaced ellipsis protection: e.g. ". . ." or ". . . ,"
        let after_dot = text.get(pos + 1..).unwrap_or("").trim_start();
        if after_dot.starts_with('.') {
            return false;
        }
        let before_dot = text.get(..pos).unwrap_or("").trim_end();
        if before_dot.ends_with('.') {
            if let Some(next_c) = after_dot.chars().next() {
                if next_c == ',' || next_c == ';' || next_c.is_lowercase() || next_c == ')' || next_c == ']' {
                    return false;
                }
            }
        }
        // A dot directly followed by comma or semicolon never terminates a sentence
        if after_dot.starts_with(',') || after_dot.starts_with(';') {
            return false;
        }
        // 0. Extract preceding word token ending at pos
        let mut word_start = pos;
        while word_start > 0 {
            let prev_ch = match text.get(..word_start).and_then(|s| s.chars().next_back()) {
                Some(ch) => ch,
                None => break,
            };
            if prev_ch.is_alphanumeric() || prev_ch == '.' || prev_ch == '_' || prev_ch == '-' {
                word_start -= prev_ch.len_utf8();
            } else {
                break;
            }
        }

        let token = text.get(word_start..=pos).unwrap_or("");

        // List item prefix guard: paragraphs or lines starting with ^\d+\.
        if word_start < pos && text.get(word_start..pos).is_some_and(|s| s.chars().all(|ch| ch.is_ascii_digit())) {
            let before = text.get(..word_start).unwrap_or("").trim_end_matches([' ', '\t']);
            let is_line_or_para_start = before.is_empty()
                || before.ends_with('\n')
                || before.ends_with('\r')
                || before.ends_with('.')
                || before.ends_with('!')
                || before.ends_with('?');
            if is_line_or_para_start {
                if let Some(next_ch) = text.get(pos + 1..).and_then(|s| s.chars().next()) {
                    if next_ch.is_whitespace() || is_closing_bracket_or_quote(next_ch) {
                        return false;
                    }
                }
            }
        }

        // 1. Decimals: digit before and digit after (e.g. 49.99, 3.14)
        let prev_digit = pos > 0 && bytes.get(pos - 1).is_some_and(|b| b.is_ascii_digit());
        let next_digit = bytes.get(pos + 1).is_some_and(|b| b.is_ascii_digit());
        if prev_digit && next_digit {
            return false;
        }

        // 2. Version number like v2.0 or v.1: preceded by 'v'/'V' and succeeded by digit
        if bytes.get(pos + 1).is_some_and(|b| b.is_ascii_digit()) {
            if pos > 0 && (bytes.get(pos - 1) == Some(&b'v') || bytes.get(pos - 1) == Some(&b'V')) {
                return false;
            }
            if pos > 1 && bytes.get(pos - 1).is_some_and(|b| b.is_ascii_digit()) {
                let mut p = pos - 1;
                while p > 0 && bytes.get(p - 1).is_some_and(|b| b.is_ascii_digit()) {
                    p -= 1;
                }
                if p > 0 && (bytes.get(p - 1) == Some(&b'v') || bytes.get(p - 1) == Some(&b'V')) {
                    return false;
                }
            }
        }
        // 3. Non-boundary dot: followed by letter or digit without whitespace (e.g. example.com, config.json)
        if let Some(next_ch) = text.get(pos + 1..).and_then(|s| s.chars().next()) {
            if next_ch.is_alphanumeric() {
                return false;
            }
        }

        // 4. Dynamic abbreviation / nonbreaking prefix check via LanguageRuleProfile or Universal Semiotic Topology
        let is_nonbreaking = if let Some(prof) = profile {
            prof.is_nonbreaking_prefix(text, pos, word_start)
        } else {
            is_absolute_abbreviation(token)
                || is_label_abbreviation(token)
                || is_honorific_abbreviation(token)
                || matches_conditional_end(token)
        };

        if is_nonbreaking {
            // Check lookahead after dot and following whitespace
            let remainder = text.get(pos + 1..).unwrap_or("").trim_start();
            if let Some(next_c) = remainder.chars().next() {
                // If followed by lowercase, digit, or comma, it is within the sentence
                if next_c.is_lowercase()
                    || next_c.is_ascii_digit()
                    || next_c == ','
                    || next_c == ';'
                {
                    return false;
                }

                // Non-conditional abbreviations (e.g. Dr., Fig., e.g., Mr., vs.) NEVER end a sentence,
                // preventing is_high_frequency_sentence_starter from blindly splitting after them (e.g. Dr. He, Fig. A, e.g. The).
                if is_absolute_abbreviation(token)
                    || is_label_abbreviation(token)
                    || is_honorific_abbreviation(token)
                {
                    return false;
                }

                let is_cond_end = profile.map_or_else(
                    || matches_conditional_end(token),
                    |prof| prof.is_conditional_end(token) || matches_conditional_end(token),
                );
                if is_cond_end && next_c.is_uppercase() {
                    return true;
                }

                if is_cond_end && is_high_frequency_sentence_starter(remainder) {
                    return true;
                }
                return false;
            } else {
                // End of text: sentence boundary
                return true;
            }
        }

        // 5. Universal topological single uppercase letter initial (e.g. "John F. Kennedy", "А. С. Пушкин")
        let letters_only = text.get(word_start..pos).unwrap_or("").trim();
        let mut chars = letters_only.chars();
        if let Some(ch) = chars.next() {
            if chars.next().is_none() && ch.is_uppercase() {
                let remainder = text.get(pos + 1..).unwrap_or("").trim_start();
                if let Some(next_c) = remainder.chars().next() {
                    if next_c.is_uppercase() {
                        return false;
                    }
                    if next_c.is_lowercase() {
                        let first_token = remainder.split_whitespace().next().unwrap_or("");
                        let has_punct = first_token
                            .chars()
                            .skip(1)
                            .any(|c| matches!(c, '.' | '!' | '?' | ';' | ':' | ',' | ')' | ']' | '}' | '。' | '！' | '？' | '；' | '：' | '，'));
                        if !has_punct {
                            return false;
                        }
                    }
                }
            }
        }

        true
    } else {
        // For '!' or '?': protect if immediately followed by alphanumeric (e.g. identifiers/code)
        if let Some(next_ch) = text.get(pos + 1..).and_then(|s| s.chars().next()) {
            if next_ch.is_alphanumeric() {
                return false;
            }
        }
        true
    }
}

pub fn remap_spans_to_sentences(sentences: &mut [SplitSentence<'_>], spans: &[MarkupSpan]) {
    if sentences.is_empty() || spans.is_empty() {
        return;
    }

    for span in spans {
        let span_start = span.source_range.start;
        let span_end = span.source_range.end;

        if span.is_self_closing || span_start == span_end {
            // Self-closing / point markup span: binary search for the enclosing or nearest sentence
            let idx = sentences.partition_point(|s| s.text_range.end < span_start);

            if idx < sentences.len() && span_start >= sentences[idx].text_range.start {
                let local_pos = span_start - sentences[idx].text_range.start;
                let mut local_span = span.clone();
                local_span.source_range = local_pos..local_pos;
                local_span.target_range = None;
                sentences[idx].inline_spans.push(local_span);
            } else if idx > 0 {
                let local_pos = sentences[idx - 1].text.len();
                let mut local_span = span.clone();
                local_span.source_range = local_pos..local_pos;
                local_span.target_range = None;
                sentences[idx - 1].inline_spans.push(local_span);
            } else {
                let mut local_span = span.clone();
                local_span.source_range = 0..0;
                local_span.target_range = None;
                sentences[0].inline_spans.push(local_span);
            }
        } else {
            // Paired markup span with non-empty range [span_start..span_end]
            let mut overlapped = false;
            let start_idx = sentences.partition_point(|s| s.text_range.end <= span_start);

            for sentence in &mut sentences[start_idx..] {
                if sentence.text_range.start >= span_end {
                    break;
                }

                let s_start = sentence.text_range.start;
                let s_end = sentence.text_range.end;

                // Compute intersection [inter_start..inter_end]
                let inter_start = span_start.max(s_start);
                let inter_end = span_end.min(s_end);

                if inter_start < inter_end {
                    overlapped = true;
                    let local_start = inter_start - s_start;
                    let local_end = inter_end - s_start;

                    let mut local_span = span.clone();
                    local_span.source_range = local_start..local_end;
                    local_span.target_range = None;
                    sentence.inline_spans.push(local_span);
                }
            }

            // If the span was entirely inside a separator between sentences:
            if !overlapped && start_idx > 0 {
                let i = start_idx - 1;
                let s_end = sentences[i].text_range.end;
                let next_start = if i + 1 < sentences.len() {
                    sentences[i + 1].text_range.start
                } else {
                    s_end + sentences[i].separator.len()
                };

                if span_start >= s_end && span_end <= next_start {
                    let local_pos = sentences[i].text.len();
                    let mut local_span = span.clone();
                    local_span.source_range = local_pos..local_pos;
                    local_span.target_range = None;
                    sentences[i].inline_spans.push(local_span);
                }
            }
        }
    }

    // Sort inline spans in each sentence by start position and depth
    for sentence in sentences.iter_mut() {
        if sentence.inline_spans.len() > 1 {
            sentence.inline_spans.sort_by(|a, b| {
                a.source_range
                    .start
                    .cmp(&b.source_range.start)
                    .then_with(|| a.depth.cmp(&b.depth))
            });
        }
    }
}

/// Split text into sentences and automatically remap markup spans to local sentence coordinates.
pub fn split_sentences_with_spans<'a>(text: &'a str, spans: &[MarkupSpan]) -> Vec<SplitSentence<'a>> {
    let mut sentences = split_sentences(text);
    remap_spans_to_sentences(&mut sentences, spans);
    sentences
}

/// Semantic split candidate priority tier.
///
/// Tiers strictly enforce semantic hierarchy when a sentence exceeds the length threshold:
/// 1. `Terminal`: Primary sentence-ending punctuation (period, exclamation, question mark, newline).
/// 2. `MajorClause`: Coordination and major clause boundaries (semicolons, colons).
/// 3. `ConjunctionClause`: Subordinating or coordinating conjunction clauses after commas (e.g. `, and `, `, which `, `，但是`).
/// 4. `EmDash`: Pauses and dashes (em-dash, spaced hyphen).
/// 5. `MinorComma`: Regular commas and enumeration pauses (commas, dunhao).
/// 6. `Whitespace`: Word boundaries and whitespace.
/// 7. `Fallback`: Character-level cutoff when no semantic delimiter exists within threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SemanticSplitTier {
    Terminal = 1,
    MajorClause = 2,
    ConjunctionClause = 3,
    EmDash = 4,
    MinorComma = 5,
    EnumerationComma = 6,
    Whitespace = 7,
    Fallback = 8,
}

#[derive(Debug, Clone)]
pub(crate) struct SemanticCandidate {
    tier: SemanticSplitTier,
    punct_end: usize,
    sep_end: usize,
}
/// Default minimum chunk character length to prevent ultra-short fragments (e.g. "Hi,").
pub const DEFAULT_MIN_CHUNK_LEN: usize = 20;

/// Default minimum chunk character length for CJK languages (ideographic density is ~3x higher).
pub const DEFAULT_CJK_MIN_CHUNK_LEN: usize = 5;

#[inline]
fn is_word_boundary_at_split(text: &str, pos: usize) -> bool {
    match text.get(pos..).and_then(|s| s.chars().next()) {
        Some(c) => !c.is_alphanumeric() && c != '_',
        None => true,
    }
}

/// Check if a word is recognized as a verb, considering language-specific morphology and exceptions.
#[inline]
fn is_predicate_verb_token(word: &str, profile: Option<&LanguageRuleProfile>) -> bool {
    if word.is_empty() {
        return false;
    }
    if let Some(prof) = profile {
        if prof.is_predicate_verb(word) {
            return true;
        }
        let len = word.len();
        if len >= 5 {
            if word.ends_with("ed") || word.ends_with("ED") {
                let lower = word.to_ascii_lowercase();
                if prof.is_non_verb_ed(&lower) {
                    return false;
                }
                return true;
            }
            if word.ends_with("ing") || word.ends_with("ING") {
                let lower = word.to_ascii_lowercase();
                if prof.is_non_verb_ing(&lower) {
                    return false;
                }
                return true;
            }
            if word.ends_with("ize") || word.ends_with("ise") || word.ends_with("ify") {
                return true;
            }
        }
    }

    // Fallback: check global registry for any registered language verb
    if RuleRegistry::global().is_predicate_verb(word) {
        return true;
    }

    let len = word.len();
    if len >= 5 {
        if word.ends_with("ed") || word.ends_with("ED") {
            const DEFAULT_NON_VERB_ED: &[&str] = &[
                "naked", "sacred", "hundred", "wicked", "tired", "bed", "red", "sled", "shed",
            ];
            for &nv in DEFAULT_NON_VERB_ED {
                if word.eq_ignore_ascii_case(nv) {
                    return false;
                }
            }
            return true;
        }
        if word.ends_with("ing") || word.ends_with("ING") {
            const DEFAULT_NON_VERB_ING: &[&str] = &[
                "morning", "evening", "something", "nothing", "anything", "everything",
                "building", "ceiling", "feeling", "meeting", "meaning", "painting",
                "training", "wedding", "string", "spring", "king", "ring", "wing", "thing",
            ];
            for &nv in DEFAULT_NON_VERB_ING {
                if word.eq_ignore_ascii_case(nv) {
                    return false;
                }
            }
            return true;
        }
        if word.ends_with("ize") || word.ends_with("ise") || word.ends_with("ify") {
            return true;
        }
    }

    false
}

/// Check if a clause segment contains any predicate verb.
#[inline]
fn contains_predicate_verb(clause: &str, profile: Option<&LanguageRuleProfile>) -> bool {
    for raw_word in clause.split_whitespace() {
        let clean_word = raw_word.trim_matches(|c: char| !c.is_alphabetic());
        if clean_word.is_empty() {
            continue;
        }
        if is_predicate_verb_token(clean_word, profile) {
            return true;
        }
    }
    for c in clause.chars() {
        if super::is_cjk_character(c) {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            if let Some(prof) = profile {
                if prof.is_predicate_verb(s) {
                    return true;
                }
            }
            if RuleRegistry::global().is_predicate_verb(s) {
                return true;
            }
        }
    }
    false
}

/// Guard against treating Oxford comma and parallel noun lists as major clauses (Tier 3).
#[inline]
fn is_parallel_coordination_guard(after: &str, profile: Option<&LanguageRuleProfile>) -> bool {
    let trimmed = after.trim_start_matches(|c: char| {
        c.is_whitespace()
            || is_closing_bracket_or_quote(c)
            || matches!(c, '"' | '\'' | '“' | '”' | '‘' | '’' | '「' | '『' | '«' | '»')
    });

    // Check coordinating conjunction prefixes across English, German, French, Spanish, Russian
    let (is_coord, rest_after_conj) = if let Some(rest) = trimmed.strip_prefix("and") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("AND") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("or") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("OR") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("nor") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("NOR") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("und") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("UND") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("oder") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("ODER") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("et") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("ET") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("ou") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("OU") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("y") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("Y") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("o") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("O") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("или") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("Или") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("ИЛИ") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("и") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else if let Some(rest) = trimmed.strip_prefix("И") {
        if is_word_boundary_at_split(rest, 0) { (true, rest) } else { (false, "") }
    } else {
        (false, "")
    };

    if !is_coord {
        return false;
    }

    let rest_clause = rest_after_conj.trim_start();
    let clause_end = rest_clause
        .find(|c: char| {
            matches!(
                c,
                ',' | ';'
                    | ':'
                    | '.'
                    | '!'
                    | '?'
                    | '\n'
                    | '\r'
                    | '—'
                    | '–'
                    | '，'
                    | '。'
                    | '！'
                    | '？'
                    | '；'
                    | '؟'
                    | '۔'
                    | '؛'
                    | '،'
                    | '׃'
            )
        })
        .unwrap_or(rest_clause.len());
    let clause = &rest_clause[..clause_end];

    !contains_predicate_verb(clause, profile)
}

/// Check if string starts with an English conjunction/clause marker on a word boundary without heap allocation.
#[inline]
fn is_conjunction_clause_marker(after: &str, profile: Option<&LanguageRuleProfile>) -> bool {
    if let Some(prof) = profile {
        prof.is_conjunction_clause_marker(after)
    } else {
        false
    }
}

fn find_best_semantic_split_candidate(
    text: &str,
    cursor: usize,
    limit_byte_pos: usize,
    profile: Option<&LanguageRuleProfile>,
) -> SemanticCandidate {
    let mut best: Option<SemanticCandidate> = None;
    let bytes = text.as_bytes();
    let len = text.len();

    let is_cjk = profile.map(|p| !p.word_boundary).unwrap_or_else(|| {
        text[cursor..limit_byte_pos].chars().any(|c| (c as u32) >= 0x3000)
    });
    let total_chars = text[cursor..limit_byte_pos].chars().count();
    let effective_min_len = if is_cjk {
        DEFAULT_CJK_MIN_CHUNK_LEN.min(total_chars / 3).max(2)
    } else {
        DEFAULT_MIN_CHUNK_LEN.min(total_chars / 3).max(3)
    };
    let min_edge = (effective_min_len / 2).max(1);
    let mut i = cursor;
    while i < limit_byte_pos {
        let c = match text.get(i..).and_then(|s| s.chars().next()) {
            Some(c) => c,
            None => break,
        };
        let c_len = c.len_utf8();

        let mut candidate: Option<SemanticCandidate> = None;

        // 1. Newline (Tier 1: Terminal)
        if c == '\n' {
            let punct_end = if i > cursor && bytes.get(i - 1) == Some(&b'\r') {
                i - 1
            } else {
                i
            };
            let sep_end = i + 1;
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::Terminal,
                punct_end,
                sep_end,
            });
        }
        // 2. CJK & RTL Terminal: 。 ！ ？ / ؟ ۔ ׃ (Tier 1: Terminal)
        else if is_cjk_punctuation(c) || is_rtl_terminal_punctuation(c) {
            let punct_end = consume_while(text, i + c_len, |nc| {
                is_cjk_punctuation(nc)
                    || is_rtl_terminal_punctuation(nc)
                    || is_western_sentence_punctuation(nc)
                    || is_closing_bracket_or_quote(nc)
            });
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::Terminal,
                punct_end,
                sep_end,
            });
        }
        // 3. Western Terminal: . ! ? (Tier 1: Terminal)
        else if is_western_sentence_punctuation(c) && should_split_western(text, i, c, profile) {
            let punct_end = consume_while(text, i + c_len, |nc| {
                is_western_sentence_punctuation(nc) || is_closing_bracket_or_quote(nc)
            });
            let is_boundary = punct_end == len
                || text
                    .get(punct_end..)
                    .and_then(|s| s.chars().next())
                    .is_some_and(|ch| ch.is_whitespace() || is_closing_bracket_or_quote(ch));
            if is_boundary {
                let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
                candidate = Some(SemanticCandidate {
                    tier: SemanticSplitTier::Terminal,
                    punct_end,
                    sep_end,
                });
            }
        }
        // 4. Semicolon & Colon (Tier 2: MajorClause)
        else if c == '；' || c == '：' || c == '؛' {
            let punct_end = consume_while(text, i + c_len, is_closing_bracket_or_quote);
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::MajorClause,
                punct_end,
                sep_end,
            });
        } else if c == ';' {
            let is_boundary = i + 1 == len
                || text
                    .get(i + 1..)
                    .and_then(|s| s.chars().next())
                    .is_some_and(|ch| ch.is_whitespace() || is_closing_bracket_or_quote(ch));
            if is_boundary {
                let punct_end = consume_while(text, i + 1, is_closing_bracket_or_quote);
                let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
                candidate = Some(SemanticCandidate {
                    tier: SemanticSplitTier::MajorClause,
                    punct_end,
                    sep_end,
                });
            }
        } else if c == ':' {
            let not_url = !text[i..].starts_with("://")
                && !text[..i].ends_with("http")
                && !text[..i].ends_with("https");
            let not_digits = !(i > 0
                && bytes.get(i - 1).is_some_and(|b| b.is_ascii_digit())
                && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit()));
            let not_double_colon =
                !text[i..].starts_with("::") && !(i > 0 && bytes.get(i - 1) == Some(&b':'));
            let is_boundary = i + 1 == len
                || text
                    .get(i + 1..)
                    .and_then(|s| s.chars().next())
                    .is_some_and(|ch| ch.is_whitespace() || is_closing_bracket_or_quote(ch));

            if not_url && not_digits && not_double_colon && is_boundary {
                let punct_end = consume_while(text, i + 1, is_closing_bracket_or_quote);
                let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
                candidate = Some(SemanticCandidate {
                    tier: SemanticSplitTier::MajorClause,
                    punct_end,
                    sep_end,
                });
            }
        }
        // 5. Dashes (Tier 4: EmDash)
        else if text[i..].starts_with("\u{2014}\u{2014}") {
            let dash_len = "\u{2014}\u{2014}".len();
            let punct_end = i + dash_len;
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::EmDash,
                punct_end,
                sep_end,
            });
        } else if c == '\u{2014}' {
            let punct_end = i + c_len;
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::EmDash,
                punct_end,
                sep_end,
            });
        } else if text[i..].starts_with(" - ") || text[i..].starts_with(" – ") {
            let dash_len = if text[i..].starts_with(" - ") {
                2
            } else {
                1 + '–'.len_utf8()
            };
            let punct_end = i + dash_len;
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::EmDash,
                punct_end,
                sep_end,
            });
        }
        // 6. Commas & Dunhao (Tier 3: ConjunctionClause, Tier 5: MinorComma)
        else if c == '，' || c == '、' || c == '،' {
            let punct_end = consume_while(text, i + c_len, is_closing_bracket_or_quote);
            let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            let is_conj = is_conjunction_clause_marker(text.get(sep_end..).unwrap_or(""), profile)
                || is_japanese_head_final_clause_marker(text.get(..i).unwrap_or(""));
            let is_coord_guard = is_parallel_coordination_guard(text.get(sep_end..).unwrap_or(""), profile);
            let tier = if is_conj && !is_coord_guard {
                SemanticSplitTier::ConjunctionClause
            } else if c == '、' || is_coord_guard {
                SemanticSplitTier::EnumerationComma
            } else {
                SemanticSplitTier::MinorComma
            };
            candidate = Some(SemanticCandidate {
                tier,
                punct_end,
                sep_end,
            });
        } else if c == ',' {
            let not_number = !(i > 0
                && bytes.get(i - 1).is_some_and(|b| b.is_ascii_digit())
                && bytes.get(i + 1).is_some_and(|b| b.is_ascii_digit()));
            let is_boundary = i + 1 == len
                || text
                    .get(i + 1..)
                    .and_then(|s| s.chars().next())
                    .is_some_and(|ch| ch.is_whitespace() || is_closing_bracket_or_quote(ch))
                || is_conjunction_clause_marker(text.get(i + 1..).unwrap_or(""), profile);

            if not_number && is_boundary {
                let punct_end = consume_while(text, i + 1, is_closing_bracket_or_quote);
                let sep_end = consume_while(text, punct_end, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
                let is_conj = is_conjunction_clause_marker(text.get(sep_end..).unwrap_or(""), profile);
                let is_coord_guard = is_parallel_coordination_guard(text.get(sep_end..).unwrap_or(""), profile);
                let tier = if is_conj && !is_coord_guard {
                    SemanticSplitTier::ConjunctionClause
                } else if is_coord_guard {
                    SemanticSplitTier::EnumerationComma
                } else {
                    SemanticSplitTier::MinorComma
                };
                candidate = Some(SemanticCandidate {
                    tier,
                    punct_end,
                    sep_end,
                });
            }
        }
        // 7. Whitespace and word boundaries (Tier 6: Whitespace)
        else if c.is_whitespace() && c != '\n' && c != '\r' {
            let punct_end = i;
            let sep_end = consume_while(text, i + c_len, |nc| nc.is_whitespace() && nc != '\n' && nc != '\r');
            candidate = Some(SemanticCandidate {
                tier: SemanticSplitTier::Whitespace,
                punct_end,
                sep_end,
            });
        }
        if let Some(cand) = candidate {
            if cand.punct_end > cursor && cand.punct_end <= limit_byte_pos {
                let dist_from_cursor = text[cursor..cand.punct_end].chars().count();
                let too_close_to_cursor = dist_from_cursor < effective_min_len;
                let rem = &text[cand.sep_end..len];
                let too_close_to_edge = !rem.is_empty() && rem.chars().take(min_edge).count() < min_edge;
                if !too_close_to_cursor && !too_close_to_edge {
                    let update = match &best {
                        None => true,
                        Some(prev) => {
                            if cand.tier < prev.tier {
                                true
                            } else if cand.tier == prev.tier {
                                if cand.tier == SemanticSplitTier::MinorComma
                                    || cand.tier == SemanticSplitTier::EnumerationComma
                                {
                                    // Topological Distance Balancing: pick candidate closest to midpoint
                                    let mid = cursor + (limit_byte_pos - cursor) / 2;
                                    let cand_dist = cand.punct_end.abs_diff(mid);
                                    let prev_dist = prev.punct_end.abs_diff(mid);
                                    cand_dist <= prev_dist
                                } else {
                                    cand.punct_end >= prev.punct_end
                                }
                            } else {
                                false
                            }
                        }
                    };
                    if update {
                        best = Some(cand);
                    }
                }
            }
        }

        i += c_len;
    }

    best.unwrap_or(SemanticCandidate {
        tier: SemanticSplitTier::Fallback,
        punct_end: limit_byte_pos,
        sep_end: limit_byte_pos,
    })
}

/// Semantically split a long sentence (or block of text) when it exceeds `max_len`.
///
/// Priority hierarchy for split boundaries:
/// 1. Primary sentence-ending punctuation (`。`, `！`, `？`, `.`, `!`, `?`, newlines).
/// 2. Coordination and major clause punctuation (`;`, `；`, `:`, `：`).
/// 3. Subordinating or coordinating conjunction clauses (`, and `, `, which `, `，但是`, `，因为`).
/// 4. Pauses and dashes (`——`, `—`, ` - `, ` – `).
/// 5. Minor commas and pauses (`,`, `，`, `、`).
/// 6. Whitespace and word boundaries.
/// 7. Hard boundary fallback when no delimiters exist.
pub fn split_long_sentence_semantic_with_profile<'a>(
    text: &'a str,
    max_len: usize,
    profile: Option<&LanguageRuleProfile>,
) -> Vec<SplitSentence<'a>> {
    if text.is_empty() {
        return Vec::new();
    }
    let max_len = max_len.max(1);
    let estimated_cap = (text.len() / max_len).clamp(1, 32);
    let mut sentences = Vec::with_capacity(estimated_cap);
    let mut cursor = 0;
    let len = text.len();

    while cursor < len {
        let remaining = &text[cursor..];
        let limit_byte_pos = match remaining.char_indices().nth(max_len) {
            Some((idx, _)) => cursor + idx,
            None => {
                sentences.push(SplitSentence::from_borrowed(
                    remaining,
                    cursor..len,
                    "",
                ));
                break;
            }
        };

        let candidate =
            find_best_semantic_split_candidate(text, cursor, limit_byte_pos, profile);

        let (split_punct_end, split_sep_end) = (candidate.punct_end, candidate.sep_end);
        let sentence_text = &text[cursor..split_punct_end];
        let separator = &text[split_punct_end..split_sep_end];
        let text_range = cursor..split_punct_end;

        sentences.push(SplitSentence::from_borrowed(
            sentence_text,
            text_range,
            separator,
        ));

        cursor = split_sep_end;
    }

    sentences
}

/// Semantically split a long sentence (or block of text) when it exceeds `max_len` with language-aware rules.
pub fn split_long_sentence_semantic_with_lang<'a>(
    text: &'a str,
    max_len: usize,
    lang: Option<&str>,
) -> Vec<SplitSentence<'a>> {
    let profile = resolve_profile_for_text(text, lang);
    split_long_sentence_semantic_with_profile(text, max_len, profile.as_deref())
}

/// Semantically split a long sentence (or block of text) when it exceeds `max_len`.
pub fn split_long_sentence_semantic<'a>(text: &'a str, max_len: usize) -> Vec<SplitSentence<'a>> {
    split_long_sentence_semantic_with_lang(text, max_len, None)
}

/// Check if string starts with an anaphoric word or discourse connective using language profile.
#[inline]
pub fn is_discourse_chunk_starter_with_profile(
    text: &str,
    profile: Option<&LanguageRuleProfile>,
) -> bool {
    if let Some(prof) = profile {
        prof.is_discourse_chunk_starter(text)
    } else if let Some(prof) = resolve_profile_for_text(text, None) {
        prof.is_discourse_chunk_starter(text)
    } else {
        false
    }
}

/// Check if string starts with an anaphoric word or discourse connective with explicit language.
#[inline]
pub fn is_discourse_chunk_starter_with_lang(text: &str, lang: Option<&str>) -> bool {
    let profile = resolve_profile_for_text(text, lang);
    is_discourse_chunk_starter_with_profile(text, profile.as_deref())
}

/// Check if string starts with an anaphoric word or discourse connective without heap allocation.
#[inline]
pub fn is_discourse_chunk_starter(text: &str) -> bool {
    is_discourse_chunk_starter_with_lang(text, None)
}

/// Fuses adjacent discourse chunks and anaphoric short sentences in-place within bounded length.
pub fn fuse_discourse_chunks_with_profile<'a>(
    sentences: &mut Vec<SplitSentence<'a>>,
    max_len: usize,
    profile: Option<&LanguageRuleProfile>,
) {
    if sentences.len() < 2 {
        return;
    }

    let detected_profile = if profile.is_none() {
        sentences.first().and_then(|s| resolve_profile_for_text(&s.text, None))
    } else {
        None
    };
    let profile = profile.or(detected_profile.as_deref());

    let mut write_idx = 0;
    for read_idx in 1..sentences.len() {
        let follow_up_chars = sentences[read_idx].text.chars().count();
        let is_cjk = profile
            .map(|p| !p.word_boundary)
            .unwrap_or_else(|| {
                sentences[read_idx].text.chars().any(|c| {
                    matches!(c, '\u{3040}'..='\u{30ff}' | '\u{4e00}'..='\u{9fff}' | '。' | '，' | '、')
                })
            });
        let discourse_limit = if is_cjk { 15 } else { 40 };
        let is_short_discourse = follow_up_chars > 0
            && follow_up_chars <= discourse_limit
            && is_discourse_chunk_starter_with_profile(&sentences[read_idx].text, profile);

        let sep_has_newline = sentences[write_idx].separator.contains('\n');
        let prev_text = sentences[write_idx].text.trim_end();
        let ends_with_dialogue_or_mood = prev_text
            .chars()
            .next_back()
            .map(|last_c| {
                matches!(
                    last_c,
                    '?' | '？' | '!' | '！' | '"' | '”' | '»' | '」' | '』' | '؟'
                )
            })
            .unwrap_or(false);

        let next_text = sentences[read_idx].text.trim_start();
        let starts_with_dialogue_dash = next_text.starts_with('—')
            || next_text.starts_with('–')
            || (next_text.starts_with("- ") && next_text.len() > 2);

        let fused_len = sentences[write_idx].text.chars().count()
            + sentences[write_idx].separator.chars().count()
            + follow_up_chars;

        if is_short_discourse
            && !sep_has_newline
            && !ends_with_dialogue_or_mood
            && !starts_with_dialogue_dash
            && fused_len <= max_len
        {
            let next_sent = std::mem::take(&mut sentences[read_idx]);
            let fuse_offset = sentences[write_idx].text.len() + sentences[write_idx].separator.len();
            let sep = std::mem::take(&mut sentences[write_idx].separator);
            sentences[write_idx].text.to_mut().push_str(&sep);
            sentences[write_idx].text.to_mut().push_str(&next_sent.text);
            sentences[write_idx].separator = next_sent.separator;
            sentences[write_idx].text_range.end = next_sent.text_range.end;

            for mut span in next_sent.inline_spans {
                span.source_range.start += fuse_offset;
                span.source_range.end += fuse_offset;
                span.target_range = None;
                sentences[write_idx].inline_spans.push(span);
            }
        } else {
            write_idx += 1;
            if write_idx != read_idx {
                sentences.swap(write_idx, read_idx);
            }
        }
    }

    sentences.truncate(write_idx + 1);
}

/// Fuses adjacent discourse chunks and anaphoric short sentences in-place with language lookup.
pub fn fuse_discourse_chunks_with_lang<'a>(
    sentences: &mut Vec<SplitSentence<'a>>,
    max_len: usize,
    lang: Option<&str>,
) {
    let sample = sentences.first().map(|s| s.text.as_ref()).unwrap_or("");
    let profile = resolve_profile_for_text(sample, lang);
    fuse_discourse_chunks_with_profile(sentences, max_len, profile.as_deref());
}

/// Fuses adjacent discourse chunks and anaphoric short sentences in-place within bounded length.
pub fn fuse_discourse_chunks_inplace<'a>(sentences: &mut Vec<SplitSentence<'a>>, max_len: usize) {
    fuse_discourse_chunks_with_lang(sentences, max_len, None);
}

/// Produce an initial hierarchical sentence split with a pre-resolved language rule profile.
pub fn split_sentences_hierarchical_with_profile<'a>(
    text: &'a str,
    max_len: usize,
    profile: Option<&LanguageRuleProfile>,
) -> Vec<SplitSentence<'a>> {
    if text.is_empty() {
        return Vec::new();
    }
    let max_len = max_len.max(1);
    let initial_sentences = split_sentences_with_profile(text, profile);
    let mut result = Vec::with_capacity(initial_sentences.len().max(2));

    for sentence in initial_sentences {
        if sentence.text.chars().count() <= max_len {
            result.push(sentence);
        } else {
            let sub_sentences = split_long_sentence_semantic_with_profile(
                &text[sentence.text_range.clone()],
                max_len,
                profile,
            );
            let sub_count = sub_sentences.len();
            let base_offset = sentence.text_range.start;

            for (idx, mut sub) in sub_sentences.into_iter().enumerate() {
                sub.text_range.start += base_offset;
                sub.text_range.end += base_offset;
                if idx + 1 == sub_count {
                    if sub.separator.is_empty() {
                        sub.separator = sentence.separator.clone();
                    } else {
                        sub.separator.to_mut().push_str(&sentence.separator);
                    }
                }
                result.push(sub);
            }
        }
    }

    result
}

/// Produce an initial hierarchical sentence split with language rules, without discourse chunk fusion.
pub fn split_sentences_hierarchical_with_lang<'a>(
    text: &'a str,
    max_len: usize,
    lang: Option<&str>,
) -> Vec<SplitSentence<'a>> {
    let profile = resolve_profile_for_text(text, lang);
    split_sentences_hierarchical_with_profile(text, max_len, profile.as_deref())
}

/// Split text into sentences using standard boundaries, and further split any sentence
/// exceeding `max_len` using the semantic hierarchy and language rules.
pub fn split_sentences_bounded_with_lang<'a>(
    text: &'a str,
    max_len: usize,
    lang: Option<&str>,
) -> Vec<SplitSentence<'a>> {
    let profile = resolve_profile_for_text(text, lang);
    let mut result = split_sentences_hierarchical_with_profile(text, max_len, profile.as_deref());
    fuse_discourse_chunks_with_profile(&mut result, max_len, profile.as_deref());
    result
}

/// Split text into sentences using standard boundaries, and further split any sentence
/// exceeding `max_len` using the semantic hierarchy.
pub fn split_sentences_bounded<'a>(text: &'a str, max_len: usize) -> Vec<SplitSentence<'a>> {
    split_sentences_bounded_with_lang(text, max_len, None)
}

/// Split text into bounded sentences and automatically remap markup spans to local sentence coordinates.
pub fn split_sentences_with_spans_bounded_with_lang<'a>(
    text: &'a str,
    spans: &[MarkupSpan],
    max_len: usize,
    lang: Option<&str>,
) -> Vec<SplitSentence<'a>> {
    let mut sentences = split_sentences_bounded_with_lang(text, max_len, lang);
    remap_spans_to_sentences(&mut sentences, spans);
    sentences
}

/// Split text into bounded sentences and automatically remap markup spans to local sentence coordinates.
pub fn split_sentences_with_spans_bounded<'a>(
    text: &'a str,
    spans: &[MarkupSpan],
    max_len: usize,
) -> Vec<SplitSentence<'a>> {
    split_sentences_with_spans_bounded_with_lang(text, spans, max_len, None)
}

/// Convenience helper to split text into semantic chunks of at most `max_len` characters with language rules.
pub fn split_long_text_semantic_with_lang(
    text: &str,
    max_len: usize,
    lang: Option<&str>,
) -> Vec<String> {
    split_long_sentence_semantic_with_lang(text, max_len, lang)
        .into_iter()
        .map(|s| s.text.into_owned())
        .collect()
}

/// Convenience helper to split text into semantic chunks of at most `max_len` characters.
pub fn split_long_text_semantic(text: &str, max_len: usize) -> Vec<String> {
    split_long_text_semantic_with_lang(text, max_len, None)
}

#[inline]
pub fn translate_single_split_sentence<F>(
    sentence: SplitSentence<'_>,
    translate_fn: &mut F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    translate_single_split_sentence_with_syntax(sentence, translate_fn, None)
}

#[inline]
pub fn translate_single_split_sentence_with_lang<F>(
    sentence: SplitSentence<'_>,
    lang: Option<&str>,
    translate_fn: &mut F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| &p.syntax);
    translate_single_split_sentence_with_syntax(sentence, translate_fn, syntax)
}

pub fn translate_single_split_sentence_with_syntax<F>(
    mut sentence: SplitSentence<'_>,
    translate_fn: &mut F,
    syntax: Option<&crate::engine::rules::SyntaxRules>,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    if sentence.text.trim().is_empty() {
        if sentence.inline_spans.is_empty() {
            let mut out = String::with_capacity(sentence.text.len() + sentence.separator.len());
            out.push_str(&sentence.text);
            out.push_str(&sentence.separator);
            return out;
        } else {
            for span in &mut sentence.inline_spans {
                if span.target_range.is_none() {
                    span.target_range = Some(span.source_range.clone());
                }
            }
            let mut rendered = super::render_projected_markup(&sentence.text, &sentence.inline_spans);
            rendered.push_str(&sentence.separator);
            return rendered;
        }
    }
    // Clean leading and trailing whitespace for optimal translation tokenization
    let leading_ws_len = sentence.text.len() - sentence.text.trim_start().len();
    let trailing_ws_start = sentence.text.trim_end().len();
    let trimmed_text = &sentence.text[leading_ws_len..trailing_ws_start];

    let leading_ws = &sentence.text[..leading_ws_len];
    let trailing_ws = &sentence.text[trailing_ws_start..];

    let mut out = String::with_capacity(
        leading_ws.len() + trimmed_text.len() + trailing_ws.len() + sentence.separator.len() + 32,
    );
    out.push_str(leading_ws);

    if sentence.inline_spans.is_empty() {
        let (target_text, _, _, _) = translate_fn(trimmed_text);
        out.push_str(&target_text);
    } else {
        if leading_ws_len > 0 || trailing_ws_start < sentence.text.len() {
            for span in &mut sentence.inline_spans {
                span.source_range.start = span
                    .source_range
                    .start
                    .saturating_sub(leading_ws_len)
                    .min(trimmed_text.len());
                span.source_range.end = span
                    .source_range
                    .end
                    .saturating_sub(leading_ws_len)
                    .min(trimmed_text.len());
            }
        }
        let (target_text, src_spans, trg_spans, alignments) = translate_fn(trimmed_text);
        super::project_markup_spans_with_syntax(
            &mut sentence.inline_spans,
            &src_spans,
            &trg_spans,
            &alignments,
            target_text.len(),
            syntax,
        );
        snap_markup_spans_to_word_boundaries(&mut sentence.inline_spans, &target_text);
        let rendered = render_projected_markup(&target_text, &sentence.inline_spans);
        out.push_str(&rendered);
    }

    out.push_str(trailing_ws);
    out.push_str(&sentence.separator);
    out
}

#[inline]
pub fn translate_single_split_sentence_dual<P, A>(
    sentence: SplitSentence<'_>,
    plain_fn: &mut P,
    align_fn: &mut A,
) -> String
where
    P: FnMut(&str) -> String + ?Sized,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    translate_single_split_sentence_dual_with_syntax(sentence, plain_fn, align_fn, None)
}

#[inline]
pub fn translate_single_split_sentence_dual_with_lang<P, A>(
    sentence: SplitSentence<'_>,
    lang: Option<&str>,
    plain_fn: &mut P,
    align_fn: &mut A,
) -> String
where
    P: FnMut(&str) -> String + ?Sized,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| &p.syntax);
    translate_single_split_sentence_dual_with_syntax(sentence, plain_fn, align_fn, syntax)
}

#[inline]
pub fn translate_single_split_sentence_dual_with_syntax<P, A>(
    mut sentence: SplitSentence<'_>,
    plain_fn: &mut P,
    align_fn: &mut A,
    syntax: Option<&crate::engine::rules::SyntaxRules>,
) -> String
where
    P: FnMut(&str) -> String + ?Sized,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + ?Sized,
{
    if sentence.text.trim().is_empty() {
        if sentence.inline_spans.is_empty() {
            let mut out = String::with_capacity(sentence.text.len() + sentence.separator.len());
            out.push_str(&sentence.text);
            out.push_str(&sentence.separator);
            return out;
        } else {
            for span in &mut sentence.inline_spans {
                if span.target_range.is_none() {
                    span.target_range = Some(span.source_range.clone());
                }
            }
            let mut rendered = super::render_projected_markup(&sentence.text, &sentence.inline_spans);
            rendered.push_str(&sentence.separator);
            return rendered;
        }
    }
    // Clean leading and trailing whitespace for optimal translation tokenization
    let leading_ws_len = sentence.text.len() - sentence.text.trim_start().len();
    let trailing_ws_start = sentence.text.trim_end().len();
    let trimmed_text = &sentence.text[leading_ws_len..trailing_ws_start];

    let leading_ws = &sentence.text[..leading_ws_len];
    let trailing_ws = &sentence.text[trailing_ws_start..];

    let mut out = String::with_capacity(
        leading_ws.len() + trimmed_text.len() + trailing_ws.len() + sentence.separator.len() + 32,
    );
    out.push_str(leading_ws);

    if sentence.inline_spans.is_empty() {
        let target_text = plain_fn(trimmed_text);
        out.push_str(&target_text);
    } else {
        if leading_ws_len > 0 || trailing_ws_start < sentence.text.len() {
            for span in &mut sentence.inline_spans {
                span.source_range.start = span
                    .source_range
                    .start
                    .saturating_sub(leading_ws_len)
                    .min(trimmed_text.len());
                span.source_range.end = span
                    .source_range
                    .end
                    .saturating_sub(leading_ws_len)
                    .min(trimmed_text.len());
            }
        }
        let (target_text, src_spans, trg_spans, alignments) = align_fn(trimmed_text);
        super::project_markup_spans_with_syntax(
            &mut sentence.inline_spans,
            &src_spans,
            &trg_spans,
            &alignments,
            target_text.len(),
            syntax,
        );
        snap_markup_spans_to_word_boundaries(&mut sentence.inline_spans, &target_text);
        let rendered = render_projected_markup(&target_text, &sentence.inline_spans);
        out.push_str(&rendered);
    }

    out.push_str(trailing_ws);
    out.push_str(&sentence.separator);
    out
}

/// Translate already split sentences sequentially with token alignment markup projection,
/// reassembling the projected results with original sentence separators.
pub fn translate_prepared_split_sentences_with_alignment<F>(
    sentences: Vec<SplitSentence<'_>>,
    translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    translate_prepared_split_sentences_with_alignment_and_lang(sentences, None, translate_fn)
}

/// Translate already split sentences sequentially with token alignment markup projection and target language syntax rules.
pub fn translate_prepared_split_sentences_with_alignment_and_lang<F>(
    sentences: Vec<SplitSentence<'_>>,
    lang: Option<&str>,
    mut translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    if sentences.is_empty() {
        return String::new();
    }
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| &p.syntax);

    let cap = sentences.iter().map(|s| s.text.len() + s.separator.len()).sum::<usize>() + 64;
    let mut out = String::with_capacity(cap);

    for sentence in sentences {
        let part = translate_single_split_sentence_with_syntax(sentence, &mut translate_fn, syntax);
        out.push_str(&part);
    }

    out
}

/// Translate already split sentences sequentially using dual-track translation:
/// sentences without inline spans are translated using `plain_fn`, while sentences
/// with inline spans are translated using `align_fn` with token alignment markup projection.
pub fn translate_prepared_split_sentences_dual<P, A>(
    sentences: Vec<SplitSentence<'_>>,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: FnMut(&str) -> String,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    translate_prepared_split_sentences_dual_with_lang(sentences, None, plain_fn, align_fn)
}

/// Translate already split sentences sequentially using dual-track translation and target language syntax rules.
pub fn translate_prepared_split_sentences_dual_with_lang<P, A>(
    sentences: Vec<SplitSentence<'_>>,
    lang: Option<&str>,
    mut plain_fn: P,
    mut align_fn: A,
) -> String
where
    P: FnMut(&str) -> String,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    if sentences.is_empty() {
        return String::new();
    }
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| &p.syntax);

    let cap = sentences.iter().map(|s| s.text.len() + s.separator.len()).sum::<usize>() + 64;
    let mut out = String::with_capacity(cap);

    for sentence in sentences {
        let part = translate_single_split_sentence_dual_with_syntax(sentence, &mut plain_fn, &mut align_fn, syntax);
        out.push_str(&part);
    }

    out
}

/// Translate already split sentences concurrently across multiple threads via Rayon,
/// reassembling the projected results in the original sentence order.
pub fn translate_prepared_split_sentences_with_alignment_parallel<F>(
    sentences: Vec<SplitSentence<'_>>,
    translate_fn: F,
) -> String
where
    F: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    translate_prepared_split_sentences_with_alignment_parallel_and_lang(sentences, None, translate_fn)
}

/// Translate already split sentences concurrently across multiple threads via Rayon with target language syntax rules.
pub fn translate_prepared_split_sentences_with_alignment_parallel_and_lang<F>(
    mut sentences: Vec<SplitSentence<'_>>,
    lang: Option<&str>,
    translate_fn: F,
) -> String
where
    F: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    if sentences.is_empty() {
        return String::new();
    }
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| p.syntax.clone());

    if sentences.len() == 1 {
        let mut f = &translate_fn;
        if let Some(single) = sentences.pop() {
            return translate_single_split_sentence_with_syntax(single, &mut f, syntax.as_ref());
        }
    }

    let translated_parts: Vec<String> = sentences
        .into_par_iter()
        .map(|sent| {
            let mut f = &translate_fn;
            translate_single_split_sentence_with_syntax(sent, &mut f, syntax.as_ref())
        })
        .collect();

    let total_len: usize = translated_parts.iter().map(|s| s.len()).sum();
    let mut out = String::with_capacity(total_len);
    for part in translated_parts {
        out.push_str(&part);
    }
    out
}

/// Translate already split sentences concurrently across multiple threads via Rayon using dual-track translation:
/// sentences without inline spans use `plain_fn`, while sentences with inline spans use `align_fn`.
pub fn translate_prepared_split_sentences_dual_parallel<P, A>(
    sentences: Vec<SplitSentence<'_>>,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    translate_prepared_split_sentences_dual_parallel_with_lang(sentences, None, plain_fn, align_fn)
}

/// Translate already split sentences concurrently across multiple threads via Rayon with target language syntax rules.
pub fn translate_prepared_split_sentences_dual_parallel_with_lang<P, A>(
    mut sentences: Vec<SplitSentence<'_>>,
    lang: Option<&str>,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    if sentences.is_empty() {
        return String::new();
    }
    let profile = lang.and_then(|l| RuleRegistry::global().get(l));
    let syntax = profile.as_ref().map(|p| p.syntax.clone());

    if sentences.len() == 1 {
        let mut p = |s: &str| plain_fn(s);
        let mut a = |s: &str| align_fn(s);
        if let Some(single) = sentences.pop() {
            return translate_single_split_sentence_dual_with_syntax(single, &mut p, &mut a, syntax.as_ref());
        }
    }

    let translated_parts: Vec<String> = sentences
        .into_par_iter()
        .map(|sent| {
            let mut p = |s: &str| plain_fn(s);
            let mut a = |s: &str| align_fn(s);
            translate_single_split_sentence_dual_with_syntax(sent, &mut p, &mut a, syntax.as_ref())
        })
        .collect();

    let total_len: usize = translated_parts.iter().map(|s| s.len()).sum();
    let mut out = String::with_capacity(total_len);
    for part in translated_parts {
        out.push_str(&part);
    }
    out
}

/// Translate a block of text by splitting it into sentences, translating each sentence
/// independently with cross-attention token alignment markup projection, and reassembling
/// the projected results with original sentence separators.
pub fn translate_split_sentences_with_alignment<F>(
    text: &str,
    spans: &[MarkupSpan],
    translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    if text.is_empty() {
        return String::new();
    }

    let sentences = split_sentences_with_spans(text, spans);
    translate_prepared_split_sentences_with_alignment(sentences, translate_fn)
}

/// Translate a block of text by splitting it into sentences using dual-track translation:
/// sentences without inline spans use `plain_fn`, while sentences with inline spans use `align_fn`.
pub fn translate_split_sentences_dual<P, A>(
    text: &str,
    spans: &[MarkupSpan],
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: FnMut(&str) -> String,
    A: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    if text.is_empty() {
        return String::new();
    }

    let sentences = split_sentences_with_spans(text, spans);
    translate_prepared_split_sentences_dual(sentences, plain_fn, align_fn)
}

/// Translate a block of text by splitting it into sentences and executing their translation
/// concurrently across multiple threads via Rayon, reassembling the projected results in the original
/// sentence order.
pub fn translate_split_sentences_with_alignment_parallel<F>(
    text: &str,
    spans: &[MarkupSpan],
    translate_fn: F,
) -> String
where
    F: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    if text.is_empty() {
        return String::new();
    }

    let sentences = split_sentences_with_spans(text, spans);
    if sentences.len() <= 1 {
        let mut f = translate_fn;
        return translate_prepared_split_sentences_with_alignment(sentences, &mut f);
    }
    translate_prepared_split_sentences_with_alignment_parallel(sentences, translate_fn)
}

/// Translate a block of text by splitting it into sentences and executing their translation
/// concurrently across multiple threads via Rayon using dual-track translation.
pub fn translate_split_sentences_dual_parallel<P, A>(
    text: &str,
    spans: &[MarkupSpan],
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    if text.is_empty() {
        return String::new();
    }

    let sentences = split_sentences_with_spans(text, spans);
    if sentences.len() <= 1 {
        let mut p = plain_fn;
        let mut a = align_fn;
        return translate_prepared_split_sentences_dual(sentences, &mut p, &mut a);
    }
    translate_prepared_split_sentences_dual_parallel(sentences, plain_fn, align_fn)
}

/// Translate or transform segmented sentences using a custom per-sentence closure,
/// preserving original separators.
pub fn process_split_sentences<F>(
    text: &str,
    spans: &[MarkupSpan],
    mut process_fn: F,
) -> String
where
    F: FnMut(&mut SplitSentence<'_>) -> String,
{
    let sentences = split_sentences_with_spans(text, spans);
    let mut out = String::with_capacity(text.len() + 64);

    for mut sentence in sentences {
        let separator = std::mem::take(&mut sentence.separator);
        let transformed = process_fn(&mut sentence);
        out.push_str(&transformed);
        out.push_str(&separator);
    }

    out
}

/// Reassemble split sentences with their original separators without modification.
pub fn reassemble_sentences(sentences: &[SplitSentence<'_>]) -> String {
    let cap = sentences.iter().map(|s| s.text.len() + s.separator.len()).sum();
    let mut out = String::with_capacity(cap);
    for s in sentences {
        out.push_str(&s.text);
        out.push_str(&s.separator);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sentences_smoke() {
        let text = "Hello world. How are you?";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "Hello world.");
        assert_eq!(sentences[1].text, "How are you?");
        assert_eq!(reassemble_sentences(&sentences), text);
    }

    #[test]
    fn test_remap_spans_intra_sentence() {
        let text = "First sentence. Second bold sentence.";
        // "bold" is at range 23..27
        let span = MarkupSpan::new_paired(23..27, "<b>".to_string(), "</b>".to_string(), 0);
        let sentences = split_sentences_with_spans(text, &[span]);

        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].inline_spans.len(), 0);
        assert_eq!(sentences[1].inline_spans.len(), 1);

        let s1_span = &sentences[1].inline_spans[0];
        assert_eq!(s1_span.source_range, 7..11);
        assert_eq!(&sentences[1].text[s1_span.source_range.clone()], "bold");
    }

    #[test]
    fn test_remap_spans_cross_sentence_convergence() {
        // Span covers both sentences: "Sentence one. Sentence two."
        let text = "Sentence one. Sentence two.";
        let span = MarkupSpan::new_paired(0..27, "**".to_string(), "**".to_string(), 0);
        let sentences = split_sentences_with_spans(text, &[span]);

        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].inline_spans.len(), 1);
        assert_eq!(sentences[1].inline_spans.len(), 1);

        // Clamped to sentence 0 boundaries: "Sentence one."
        assert_eq!(sentences[0].inline_spans[0].source_range, 0..13);
        assert_eq!(
            &sentences[0].text[sentences[0].inline_spans[0].source_range.clone()],
            "Sentence one."
        );

        // Clamped to sentence 1 boundaries: "Sentence two."
        assert_eq!(sentences[1].inline_spans[0].source_range, 0..13);
        assert_eq!(
            &sentences[1].text[sentences[1].inline_spans[0].source_range.clone()],
            "Sentence two."
        );
    }

    #[test]
    fn test_remap_self_closing_span() {
        let text = "First sentence. Second sentence.";
        // Point span inside sentence 1
        let span = MarkupSpan::new_self_closing(6, "<point/>".to_string(), 0);
        let sentences = split_sentences_with_spans(text, &[span]);

        assert_eq!(sentences[0].inline_spans.len(), 1);
        assert_eq!(sentences[0].inline_spans[0].source_range, 6..6);
    }

    #[test]
    fn test_translate_split_sentences_with_alignment_serial_and_parallel() {
        let text = "Visit our website for info. Please contact our support team.";
        let spans = [
            MarkupSpan::new_paired(10..17, "<b>".to_string(), "</b>".to_string(), 0),
            MarkupSpan::new_paired(35..42, "<a>".to_string(), "</a>".to_string(), 0),
        ];

        let mock_align = |sent: &str| {
            if sent.starts_with("Visit") {
                (
                    "当社のウェブサイトをご覧ください。".to_string(),
                    vec![0..5, 6..9, 10..17, 18..21, 22..27],
                    vec![0..9, 9..24, 24..48],
                    vec![1, 2, 0],
                )
            } else {
                (
                    "サポートチームにお問い合わせください。".to_string(),
                    vec![0..6, 7..14, 15..18, 19..26, 27..32],
                    vec![0..21, 21..51],
                    vec![3, 1],
                )
            }
        };

        let assert_valid = |res: &str| {
            assert!(res.contains("<b>"));
            assert!(res.contains("</b>"));
            assert!(res.contains("<a>"));
            assert!(res.contains("</a>"));
            assert!(res.contains(' '));
        };

        let res_seq = translate_split_sentences_with_alignment(text, &spans, mock_align);
        assert_valid(&res_seq);

        let res_par = translate_split_sentences_with_alignment_parallel(text, &spans, mock_align);
        assert_valid(&res_par);
        assert_eq!(res_seq, res_par);

        let prepared = split_sentences_with_spans(text, &spans);
        let res_prep_seq = translate_prepared_split_sentences_with_alignment(prepared.clone(), mock_align);
        assert_valid(&res_prep_seq);
        assert_eq!(res_seq, res_prep_seq);

        let res_prep_par = translate_prepared_split_sentences_with_alignment_parallel(prepared, mock_align);
        assert_valid(&res_prep_par);
        assert_eq!(res_seq, res_prep_par);
    }
    #[test]
    fn test_split_empty_and_blank() {
        assert_eq!(split_sentences(""), Vec::new());

        let blank = "   \n\t  ";
        let sentences = split_sentences(blank);
        assert_eq!(reassemble_sentences(&sentences), blank);
    }

    #[test]
    fn test_split_nested_cross_sentence_spans() {
        // Text: "First sentence. Second sentence. Third sentence."
        let text = "First sentence. Second sentence. Third sentence.";
        // Outer bold span covering all 3 sentences: 0..48
        let bold_span = MarkupSpan::new_paired(0..48, "<b>".to_string(), "</b>".to_string(), 0);
        // Inner italic span covering only sentence 2: 16..32
        let italic_span = MarkupSpan::new_paired(16..32, "<i>".to_string(), "</i>".to_string(), 1);

        let sentences = split_sentences_with_spans(text, &[bold_span, italic_span]);
        assert_eq!(sentences.len(), 3);

        // Sentence 0: only bold span [0..15]
        assert_eq!(sentences[0].inline_spans.len(), 1);
        assert_eq!(sentences[0].inline_spans[0].open_markup, "<b>");
        assert_eq!(sentences[0].inline_spans[0].source_range, 0..15);

        // Sentence 1: bold span [0..16] and italic span [0..16]
        assert_eq!(sentences[1].inline_spans.len(), 2);
        assert_eq!(sentences[1].inline_spans[0].open_markup, "<b>");
        assert_eq!(sentences[1].inline_spans[0].source_range, 0..16);
        assert_eq!(sentences[1].inline_spans[1].open_markup, "<i>");
        assert_eq!(sentences[1].inline_spans[1].source_range, 0..16);

        // Sentence 2: only bold span [0..15]
        assert_eq!(sentences[2].inline_spans.len(), 1);
        assert_eq!(sentences[2].inline_spans[0].open_markup, "<b>");
        assert_eq!(sentences[2].inline_spans[0].source_range, 0..15);
    }

    #[test]
    fn test_split_mixed_cjk_and_western() {
        let text = "Rust 是最好用的语言。The system has zero-cost abstractions! 你觉得呢？I agree.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 4);
        assert_eq!(sentences[0].text, "Rust 是最好用的语言。");
        assert_eq!(sentences[0].separator, "");
        assert_eq!(sentences[1].text, "The system has zero-cost abstractions!");
        assert_eq!(sentences[1].separator, " ");
        assert_eq!(sentences[2].text, "你觉得呢？");
        assert_eq!(sentences[2].separator, "");
        assert_eq!(sentences[3].text, "I agree.");
        assert_eq!(sentences[3].separator, "");

        assert_eq!(reassemble_sentences(&sentences), text);
    }

    #[test]
    fn test_process_split_sentences() {
        let text = "First sentence. Second sentence.";
        let result = process_split_sentences(text, &[], |sentence| {
            format!("[{}]", sentence.text)
        });
        assert_eq!(result, "[First sentence.] [Second sentence.]");
    }

    #[test]
    fn test_remap_spans_binary_search_edge_cases() {
        let text = "First sentence.  Second sentence.  Third sentence.";
        // Sentence 0: "First sentence." (0..15), sep "  " (15..17)
        // Sentence 1: "Second sentence." (17..33), sep "  " (33..35)
        // Sentence 2: "Third sentence." (35..50), sep "" (50..50)
        let point_before = MarkupSpan::new_self_closing(0, "<p0/>".to_string(), 0);
        let point_in_sep = MarkupSpan::new_self_closing(16, "<p1/>".to_string(), 0);
        let point_at_end = MarkupSpan::new_self_closing(50, "<p2/>".to_string(), 0);
        let range_across = MarkupSpan::new_paired(5..40, "<b>".to_string(), "</b>".to_string(), 0);
        let range_in_sep = MarkupSpan::new_paired(15..17, "<i>".to_string(), "</i>".to_string(), 0);

        let sents = split_sentences_with_spans(text, &[point_before, point_in_sep, point_at_end, range_across, range_in_sep]);
        assert_eq!(sents.len(), 3);

        // Sentence 0: point_before at 0..0, range_across from 5..15, range_in_sep attached to end of sent 0 (15..15), point_in_sep attached to end of sent 0 (15..15)
        assert_eq!(sents[0].inline_spans.iter().filter(|s| s.open_markup == "<p0/>").count(), 1);
        assert_eq!(sents[0].inline_spans.iter().find(|s| s.open_markup == "<p0/>").unwrap().source_range, 0..0);
        assert_eq!(sents[0].inline_spans.iter().filter(|s| s.open_markup == "<p1/>").count(), 1);
        assert_eq!(sents[0].inline_spans.iter().find(|s| s.open_markup == "<p1/>").unwrap().source_range, 15..15);

        // Sentence 1: range_across from 0..16
        assert_eq!(sents[1].inline_spans.iter().filter(|s| s.open_markup == "<b>").count(), 1);
        assert_eq!(sents[1].inline_spans.iter().find(|s| s.open_markup == "<b>").unwrap().source_range, 0..16);

        // Sentence 2: range_across from 0..5, point_at_end at 15..15
        assert_eq!(sents[2].inline_spans.iter().filter(|s| s.open_markup == "<b>").count(), 1);
        assert_eq!(sents[2].inline_spans.iter().find(|s| s.open_markup == "<b>").unwrap().source_range, 0..5);
        assert_eq!(sents[2].inline_spans.iter().filter(|s| s.open_markup == "<p2/>").count(), 1);
        assert_eq!(sents[2].inline_spans.iter().find(|s| s.open_markup == "<p2/>").unwrap().source_range, 15..15);
    }

    #[test]
    #[allow(clippy::single_range_in_vec_init)]
    fn test_translate_split_sentences_dual_serial_and_parallel() {
        let text = "Sentence one is plain. Sentence two has bold text.";
        let span = MarkupSpan::new_paired(40..44, "<b>".to_string(), "</b>".to_string(), 0);
        let sentences = split_sentences_with_spans(text, std::slice::from_ref(&span));
        assert_eq!(sentences.len(), 2);
        assert!(sentences[0].inline_spans.is_empty());
        assert!(!sentences[1].inline_spans.is_empty());

        let mut plain_called = 0;
        let mut align_called = 0;

        let res_seq = translate_prepared_split_sentences_dual(
            sentences,
            |sent| {
                plain_called += 1;
                assert_eq!(sent, "Sentence one is plain.");
                "TR(Plain sentence)".to_string()
            },
            |sent| {
                align_called += 1;
                assert_eq!(sent, "Sentence two has bold text.");
                (
                    "TR(Sentence two has bold text.)".to_string(),
                    vec![0..8, 9..12, 13..16, 17..21, 22..26, 26..27],
                    vec![0..3, 3..11, 12..15, 16..19, 20..24, 25..29, 29..31],
                    vec![0, 0, 1, 2, 3, 4, 5],
                )
            },
        );

        assert_eq!(plain_called, 1);
        assert_eq!(align_called, 1);
        assert!(res_seq.contains("TR(Plain sentence) "));
        assert!(res_seq.contains("<b>"));

        let res_par = translate_split_sentences_dual_parallel(
            text,
            &[span],
            |sent| format!("PLAIN({})", sent),
            |sent| {
                (
                    format!("ALIGN({})", sent),
                    vec![0..sent.len()],
                    vec![0..(sent.len() + 7)],
                    vec![0],
                )
            },
        );

        assert!(res_par.contains("PLAIN(Sentence one is plain.) "));
        assert!(res_par.contains("<b>"));
    }

    #[test]
    fn test_semantic_split_hierarchy_priority() {
        // Priority 1: Terminal punctuation should be chosen over semicolons, commas, and spaces
        // within the allowed max_len window.
        let text_with_period = "Hello, world; this is tier 1. And here is extra content that continues.";
        let parts1 = split_long_sentence_semantic(text_with_period, 35);
        assert_eq!(parts1[0].text, "Hello, world; this is tier 1.");
        assert_eq!(parts1[0].separator, " ");

        // Priority 2: Clause punctuation (semicolon) chosen over commas and spaces when no terminal exists
        let text_with_semi = "First clause has a comma, like this; second clause continues further.";
        let parts2 = split_long_sentence_semantic(text_with_semi, 45);
        assert_eq!(parts2[0].text, "First clause has a comma, like this;");
        assert_eq!(parts2[0].separator, " ");

        // Priority 3: Comma chosen over spaces when no terminal or clause punctuation exists
        let text_with_comma = "This sentence contains several words, but no semicolons or periods.";
        let parts3 = split_long_sentence_semantic(text_with_comma, 45);
        assert_eq!(parts3[0].text, "This sentence contains several words,");
        assert_eq!(parts3[0].separator, " ");

        // Priority 4: Whitespace chosen when no punctuation exists
        let text_with_space = "An unbroken sequence of words with no punctuation whatsoever";
        let parts4 = split_long_sentence_semantic(text_with_space, 30);
        assert_eq!(parts4[0].text, "An unbroken sequence of words");

        // Priority 5: Fallback hard cutoff when a single continuous word exceeds max_len
        let text_unbroken = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let parts5 = split_long_sentence_semantic(text_unbroken, 10);
        assert_eq!(parts5.len(), 4);
        assert_eq!(parts5[0].text, "ABCDEFGHIJ");
        assert_eq!(parts5[1].text, "KLMNOPQRST");
        assert_eq!(parts5[2].text, "UVWXYZ0123");
        assert_eq!(parts5[3].text, "456789");
        assert_eq!(reassemble_sentences(&parts5), text_unbroken);
    }

    #[test]
    fn test_semantic_split_cjk_clauses_and_dunhao() {
        let cjk_text = "我们今天开会讨论了关于深度学习模型的若干重要议题，包括分词器的归一化策略；以及在面对超长文本时的语义截断和拆分逻辑，这些技术能够有效提升机器翻译的整体质量。";
        // Split with max_len = 30
        let sentences = split_long_sentence_semantic(cjk_text, 30);
        assert!(sentences.len() >= 3);

        // Verify all parts stay within threshold (or exact boundary)
        for s in &sentences {
            assert!(s.text.chars().count() <= 30);
        }

        // Perfect lossless reconstruction
        assert_eq!(reassemble_sentences(&sentences), cjk_text);

        // Verify Chinese enumeration comma (dunhao)
        let dunhao_text = "项目包含苹果、香蕉、橙子、梨子、葡萄、西瓜以及草莓等多种常见水果。";
        let dunhao_parts = split_long_sentence_semantic(dunhao_text, 15);
        assert!(dunhao_parts.len() >= 2);
        assert_eq!(reassemble_sentences(&dunhao_parts), dunhao_text);
    }

    #[test]
    fn test_semantic_split_western_numbers_and_quotes_protected() {
        // Numbers with commas (1,000,000) and decimals (3.14159) must NOT be split
        let text = "The company reported $1,000,000 in revenue with 3.14159 ratio, but profits fell significantly this quarter.";
        let parts = split_long_sentence_semantic(text, 65);
        // Should split at the comma after ratio (", but"), not inside "$1,000,000" or "3.14159"
        assert_eq!(parts[0].text, "The company reported $1,000,000 in revenue with 3.14159 ratio,");
        assert_eq!(parts[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts), text);

        // Quotes after punctuation should stay with the punctuation
        let quote_text = "He shouted: \"Stop right there!\" Then she quickly turned around.";
        let quote_parts = split_long_sentence_semantic(quote_text, 35);
        assert_eq!(quote_parts[0].text, "He shouted: \"Stop right there!\"");
        assert_eq!(quote_parts[0].separator, " ");
        assert_eq!(reassemble_sentences(&quote_parts), quote_text);
    }

    #[test]
    fn test_split_sentences_bounded_with_spans() {
        // Mix of short sentence and a very long sentence
        let text = "Short sentence. This is an extremely long sentence that discusses neural machine translation, Transformer attention mechanisms, and SentencePiece tokenization without stopping.";
        let span = MarkupSpan::new_paired(0..14, "<b>".to_string(), "</b>".to_string(), 0);
        let span_long = MarkupSpan::new_paired(44..78, "<i>".to_string(), "</i>".to_string(), 0);

        let bounded = split_sentences_with_spans_bounded(text, &[span, span_long], 50);
        assert!(bounded.len() >= 4);

        // First sentence was short and remains intact
        assert_eq!(bounded[0].text, "Short sentence.");
        assert_eq!(bounded[0].inline_spans.len(), 1);

        // Lossless reconstruction
        assert_eq!(reassemble_sentences(&bounded), text);

        // All text ranges are valid and strictly monotonic
        for i in 1..bounded.len() {
            assert!(bounded[i].text_range.start >= bounded[i - 1].text_range.end);
        }
    }

    #[test]
    fn test_split_long_text_semantic_helper() {
        let text = "Part one of the document; part two of the document; part three of the document.";
        let chunks = split_long_text_semantic(text, 30);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Part one of the document;");
        assert_eq!(chunks[1], "part two of the document;");
        assert_eq!(chunks[2], "part three of the document.");
    }

    #[test]
    fn test_semantic_split_western_conjunction_clauses() {
        // Case 1: Prioritize ", which " over a subsequent appositive comma within the window.
        // If both had the same tier, the later comma (after university) would be chosen.
        // With ConjunctionClause (tier 3) > MinorComma (tier 5), the earlier ", which " split is preferred.
        let text_which = "We visited Cambridge, which is home to the university, a historic institution in England.";
        let parts_which = split_long_sentence_semantic(text_which, 70);
        assert_eq!(parts_which[0].text, "We visited Cambridge,");
        assert_eq!(parts_which[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_which), text_which);

        // Case 2: Prioritize ", and " over preceding appositive commas within the window.
        let text_and = "Alexander the Great, the king of Macedonia, defeated the Persian Empire, and established a vast territory.";
        let parts_and = split_long_sentence_semantic(text_and, 80);
        assert_eq!(
            parts_and[0].text,
            "Alexander the Great, the king of Macedonia, defeated the Persian Empire,"
        );
        assert_eq!(parts_and[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_and), text_and);

        // Case 3: Prioritize ", and " when it appears earlier than an appositive comma.
        let text_and_app = "The server processed the request, and the client received the response, a JSON payload with metadata.";
        let parts_and_app = split_long_sentence_semantic(text_and_app, 75);
        assert_eq!(parts_and_app[0].text, "The server processed the request,");
        assert_eq!(parts_and_app[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_and_app), text_and_app);

        // Case 4: Prioritize ", because " over subsequent appositive commas.
        let text_because = "The experiment failed, because the sensor, an uncalibrated optical device, gave inaccurate readings.";
        let parts_because = split_long_sentence_semantic(text_because, 50);
        assert_eq!(parts_because[0].text, "The experiment failed,");
        assert_eq!(parts_because[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_because), text_because);

        // Case 5: Prioritize ", although " over preceding appositive commas.
        let text_although = "Dr. Watson, an experienced physician, arrived at Baker Street, although the weather was stormy.";
        let parts_although = split_long_sentence_semantic(text_although, 80);
        assert_eq!(
            parts_although[0].text,
            "Dr. Watson, an experienced physician, arrived at Baker Street,"
        );
        assert_eq!(parts_although[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_although), text_although);
    }

    #[test]
    fn test_semantic_split_cjk_conjunction_clauses() {
        // Case 1: Prioritize "，但是" over a subsequent regular comma within the window.
        let text_danshi = "这项研究虽然取得了一定的进展，但是实验结果并不理想，仍有许多细节需要进一步完善和推敲。";
        let parts_danshi = split_long_sentence_semantic(text_danshi, 30);
        assert_eq!(parts_danshi[0].text, "这项研究虽然取得了一定的进展，");
        assert_eq!(parts_danshi[0].separator, "");
        assert_eq!(reassemble_sentences(&parts_danshi), text_danshi);

        // Case 2: Prioritize "，因为" over a subsequent regular comma within the window.
        let text_yinwei = "这套系统在生产环境中表现优异，因为架构设计非常模块化，接口规范严谨，得到了团队的一致好评。";
        let parts_yinwei = split_long_sentence_semantic(text_yinwei, 30);
        assert_eq!(parts_yinwei[0].text, "这套系统在生产环境中表现优异，");
        assert_eq!(parts_yinwei[0].separator, "");
        assert_eq!(reassemble_sentences(&parts_yinwei), text_yinwei);

        // Case 3: Prioritize "，因此"
        let text_yinci = "团队在方案评估阶段经过了多次充分论证，因此最终选定了微服务架构，以确保系统的高可用性。";
        let parts_yinci = split_long_sentence_semantic(text_yinci, 35);
        assert_eq!(parts_yinci[0].text, "团队在方案评估阶段经过了多次充分论证，");
        assert_eq!(parts_yinci[0].separator, "");
        assert_eq!(reassemble_sentences(&parts_yinci), text_yinci);

        // Case 4: Prioritize "，然而"
        let text_raner = "新算法在基准测试中取得了非常突出的精度，然而推理延迟却大幅上升，需要进行进一步的量化优化。";
        let parts_raner = split_long_sentence_semantic(text_raner, 35);
        assert_eq!(parts_raner[0].text, "新算法在基准测试中取得了非常突出的精度，");
        assert_eq!(parts_raner[0].separator, "");
        assert_eq!(reassemble_sentences(&parts_raner), text_raner);

        // Case 5: Prioritize "，从而"
        let text_conger = "编译器对抽象语法树进行了深度常量折叠，从而显著减少了运行时的指令开销，提升了执行效率。";
        let parts_conger = split_long_sentence_semantic(text_conger, 35);
        assert_eq!(parts_conger[0].text, "编译器对抽象语法树进行了深度常量折叠，");
        assert_eq!(parts_conger[0].separator, "");
        assert_eq!(reassemble_sentences(&parts_conger), text_conger);
    }

    #[test]
    fn test_semantic_split_tier_emdash_vs_comma_hierarchy() {
        // EmDash (Tier 4) vs MinorComma (Tier 5): EmDash takes precedence over MinorComma
        let text_emdash = "He opened the door, looked inside — nobody was in the room at that late hour.";
        let parts_dash = split_long_sentence_semantic(text_emdash, 40);
        assert_eq!(parts_dash[0].text, "He opened the door, looked inside —");
        assert_eq!(parts_dash[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_dash), text_emdash);

        // ConjunctionClause (Tier 3) vs EmDash (Tier 4): ConjunctionClause takes precedence
        let text_conj_dash = "He opened the door — looked inside, and walked away quietly into the night.";
        let parts_conj_dash = split_long_sentence_semantic(text_conj_dash, 40);
        assert_eq!(parts_conj_dash[0].text, "He opened the door — looked inside,");
        assert_eq!(parts_conj_dash[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_conj_dash), text_conj_dash);

        // MajorClause (Tier 2) vs ConjunctionClause (Tier 3): MajorClause takes precedence
        let text_semi_conj = "He opened the door; looked inside, and walked away quietly into the night.";
        let parts_semi_conj = split_long_sentence_semantic(text_semi_conj, 40);
        assert_eq!(parts_semi_conj[0].text, "He opened the door;");
        assert_eq!(parts_semi_conj[0].separator, " ");
        assert_eq!(reassemble_sentences(&parts_semi_conj), text_semi_conj);
    }

    #[test]
    fn test_is_discourse_chunk_starter() {
        // English pronouns / anaphora
        assert!(is_discourse_chunk_starter("it worked"));
        assert!(is_discourse_chunk_starter("It worked well."));
        assert!(is_discourse_chunk_starter("it's great"));
        assert!(is_discourse_chunk_starter("this is fine"));
        assert!(is_discourse_chunk_starter("that happened"));
        assert!(is_discourse_chunk_starter("these were good"));
        assert!(is_discourse_chunk_starter("he said so"));
        assert!(is_discourse_chunk_starter("she agreed"));
        assert!(is_discourse_chunk_starter("they went there"));

        // English discourse connectives
        assert!(is_discourse_chunk_starter("and so on."));
        assert!(is_discourse_chunk_starter("And so on."));
        assert!(is_discourse_chunk_starter("but wait, there's more."));
        assert!(is_discourse_chunk_starter("so it goes."));
        assert!(is_discourse_chunk_starter("thus concluded the report."));
        assert!(is_discourse_chunk_starter("hence the confusion."));
        assert!(is_discourse_chunk_starter("therefore we decide."));

        // English non-starters (word boundary / different words)
        assert!(!is_discourse_chunk_starter("item 1 is ready"));
        assert!(!is_discourse_chunk_starter("italic font"));
        assert!(!is_discourse_chunk_starter("hello world"));
        assert!(!is_discourse_chunk_starter("Andrew said yes"));
        assert!(!is_discourse_chunk_starter("software engineer"));
        assert!(!is_discourse_chunk_starter("button pressed"));
        assert!(!is_discourse_chunk_starter("thesis defense"));

        // Chinese discourse / anaphoric starters
        assert!(is_discourse_chunk_starter("它也是如此。"));
        assert!(is_discourse_chunk_starter("“它也是如此。”"));
        assert!(is_discourse_chunk_starter("它们非常有效。"));
        assert!(is_discourse_chunk_starter("这也是一个方法。"));
        assert!(is_discourse_chunk_starter("那是过去的事了。"));
        assert!(is_discourse_chunk_starter("其中包含核心逻辑。"));
        assert!(is_discourse_chunk_starter("此外还需要注意细节。"));
        assert!(is_discourse_chunk_starter("他表示同意。"));
        assert!(is_discourse_chunk_starter("她继续前行。"));
        assert!(is_discourse_chunk_starter("因而获得了成功。"));
        assert!(is_discourse_chunk_starter("由此可见，此举可行。"));
        assert!(is_discourse_chunk_starter("因此建议采纳。"));
        assert!(is_discourse_chunk_starter("但是情况有变。"));
        assert!(is_discourse_chunk_starter("而且性能优异。"));

        // Chinese non-starters
        assert!(!is_discourse_chunk_starter("今天天气很好。"));
        assert!(!is_discourse_chunk_starter("开发人员注意。"));
    }

    #[test]
    fn test_fuse_discourse_chunks_english_anaphora() {
        // Verify English anaphoric short sentence "It worked well." fuses into preceding sentence within single paragraph
        let text = "We tried the new method. It worked well.";
        let sentences = split_sentences_bounded(text, 100);

        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "We tried the new method. It worked well.");
        assert_eq!(sentences[0].separator, "");
        assert_eq!(sentences[0].text_range, 0..text.len());
        assert_eq!(reassemble_sentences(&sentences), text);
    }

    #[test]
    fn test_fuse_discourse_chunks_newline_protection() {
        // Verify sentences separated by newline boundaries are not fused
        let text_lf = "We tried the new method.\nIt worked well.";
        let sentences_lf = split_sentences_bounded(text_lf, 100);
        assert_eq!(sentences_lf.len(), 2);
        assert_eq!(sentences_lf[0].text, "We tried the new method.");
        assert_eq!(sentences_lf[0].separator, "\n");
        assert_eq!(sentences_lf[1].text, "It worked well.");
        assert_eq!(sentences_lf[1].separator, "");
        assert_eq!(reassemble_sentences(&sentences_lf), text_lf);

        // Also test CRLF
        let text_crlf = "We tried the new method.\r\nIt worked well.";
        let sentences_crlf = split_sentences_bounded(text_crlf, 100);
        assert_eq!(sentences_crlf.len(), 2);
        assert_eq!(sentences_crlf[0].text, "We tried the new method.");
        assert_eq!(sentences_crlf[0].separator, "\r\n");
        assert_eq!(sentences_crlf[1].text, "It worked well.");
        assert_eq!(sentences_crlf[1].separator, "");
        assert_eq!(reassemble_sentences(&sentences_crlf), text_crlf);
    }

    #[test]
    fn test_fuse_discourse_chunks_max_len_exceeded() {
        // Verify candidate sentences exceeding max length budget are not fused
        let text = "We tried the new method. It worked well.";
        // Total chars: 24 ("We tried the new method.") + 1 (" ") + 15 ("It worked well.") = 40.
        // With max_len = 35, length budget is exceeded so they must not be fused.
        let sentences = split_sentences_bounded(text, 35);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "We tried the new method.");
        assert_eq!(sentences[0].separator, " ");
        assert_eq!(sentences[1].text, "It worked well.");
        assert_eq!(sentences[1].separator, "");
        assert_eq!(reassemble_sentences(&sentences), text);
    }

    #[test]
    fn test_fuse_discourse_chunks_chinese_anaphora() {
        let text = "我们尝试了新方法。它也是如此。";
        let sentences = split_sentences_bounded(text, 100);
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0].text, "我们尝试了新方法。它也是如此。");
        assert_eq!(sentences[0].separator, "");
        assert_eq!(sentences[0].text_range, 0..text.len());
        assert_eq!(reassemble_sentences(&sentences), text);

        // Across newline: must not fuse
        let text_nl = "我们尝试了新方法。\n它也是如此。";
        let sentences_nl = split_sentences_bounded(text_nl, 100);
        assert_eq!(sentences_nl.len(), 2);
        assert_eq!(sentences_nl[0].text, "我们尝试了新方法。");
        assert_eq!(sentences_nl[1].text, "它也是如此。");
        assert_eq!(reassemble_sentences(&sentences_nl), text_nl);
    }

    #[test]
    fn test_fuse_discourse_chunks_transitive_and_spans() {
        // Multiple discourse follow-ups: "It worked.", "And so on."
        let text = "We tried the new method. It worked. And so on.";
        let span_prev = MarkupSpan::new_paired(0..8, "<b>".to_string(), "</b>".to_string(), 0);
        let span_it = MarkupSpan::new_paired(25..34, "<i>".to_string(), "</i>".to_string(), 0);
        let span_and = MarkupSpan::new_paired(36..46, "<u>".to_string(), "</u>".to_string(), 0);

        let bounded = split_sentences_with_spans_bounded(text, &[span_prev, span_it, span_and], 100);
        assert_eq!(bounded.len(), 1);
        assert_eq!(bounded[0].text, text);
        assert_eq!(bounded[0].inline_spans.len(), 3);
        assert_eq!(&bounded[0].text[bounded[0].inline_spans[0].source_range.clone()], "We tried");
        assert_eq!(&bounded[0].text[bounded[0].inline_spans[1].source_range.clone()], "It worked");
        assert_eq!(&bounded[0].text[bounded[0].inline_spans[2].source_range.clone()], "And so on.");
        assert_eq!(reassemble_sentences(&bounded), text);
    }

    #[test]
    fn test_is_chinese_conjunction_conditional() {
        let prof = RuleRegistry::global().get("zh").expect("zh profile loaded");
        assert!(prof.is_conjunction_clause_marker("如果天气好"));
        assert!(prof.is_conjunction_clause_marker("要是明天出发"));
        assert!(prof.is_conjunction_clause_marker("假如条件满足"));
        assert!(prof.is_conjunction_clause_marker(" 如果发生异常"));
        assert!(prof.is_conjunction_clause_marker("「如果」"));
    }

    #[test]
    fn test_is_english_conjunction_conditional() {
        let prof = RuleRegistry::global().get("en").expect("en profile loaded");
        assert!(prof.is_conjunction_clause_marker("if this happens"));
        assert!(prof.is_conjunction_clause_marker("whether to continue"));
        assert!(prof.is_conjunction_clause_marker(" if system fails"));
    }

    #[test]
    fn test_fuse_discourse_chunks_extended_pronouns() {
        // Possessive and objective pronouns: "Its", "Their", "His", "There"
        let text_its = "The dog was happy. Its tail was wagging.";
        let res_its = split_sentences_bounded(text_its, 100);
        assert_eq!(res_its.len(), 1);
        assert_eq!(res_its[0].text, text_its);

        let text_their = "The users were waiting. Their feedback was positive.";
        let res_their = split_sentences_bounded(text_their, 100);
        assert_eq!(res_their.len(), 1);
        assert_eq!(res_their[0].text, text_their);

        let text_corner = "我们尝试了新方法。「它也是如此。」";
        let res_corner = split_sentences_bounded(text_corner, 100);
        assert_eq!(res_corner.len(), 1);
        assert_eq!(res_corner[0].text, text_corner);
    }

    #[test]
    fn test_dash_multiple_spaces_absorbed() {
        let text = "First part -   second part with length budget.";
        let res = split_long_sentence_semantic(text, 35);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].text, "First part -");
        assert_eq!(res[0].separator, "   ");
        assert_eq!(res[1].text, "second part with length budget.");
        assert_eq!(reassemble_sentences(&res), text);
    }

    #[test]
    fn test_fuse_discourse_chunks_non_discourse_not_fused() {
        // Follow-up sentence is <= 40 chars, but not a discourse starter
        let text = "We tried the new method. Doctors recommend rest.";
        let sentences = split_sentences_bounded(text, 100);
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "We tried the new method.");
        assert_eq!(sentences[1].text, "Doctors recommend rest.");
        assert_eq!(reassemble_sentences(&sentences), text);
    }

    #[test]
    fn test_pure_topological_fallback_without_rules() {
        // Without rules (profile: None), splitting relies purely on punctuation topology
        let text = "First sentence with 49.99 dollars! Second sentence (v2.0) is here? Third sentence is normal.";
        let parts = split_sentences_with_profile(text, None);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].text, "First sentence with 49.99 dollars!");
        assert_eq!(parts[1].text, "Second sentence (v2.0) is here?");
        assert_eq!(parts[2].text, "Third sentence is normal.");
        assert_eq!(reassemble_sentences(&parts), text);

        // Topological Distance Balancing fallback on long sentence
        let long_commas = "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu nu.";
        let split_fallback = split_long_sentence_semantic_with_profile(long_commas, 40, None);
        assert_eq!(split_fallback.len(), 3);
        assert_eq!(reassemble_sentences(&split_fallback), long_commas);
    }

    #[test]
    fn test_enhanced_semantic_split_with_profile() {
        let zh_profile = RuleRegistry::global().get("zh").expect("zh profile");
        // Chinese sentence with conjunction "如果" vs normal comma
        let zh_text = "数据包已经完全到达，如果是合法格式，那么系统立刻开始解析。";
        // When profile is provided, "如果" should be elevated to Tier 3
        let parts = split_long_sentence_semantic_with_profile(zh_text, 25, Some(&zh_profile));
        assert!(parts.len() >= 2);
        assert_eq!(reassemble_sentences(&parts), zh_text);
    }

    #[test]
    fn test_detect_dominant_lang_delegation() {
        for (input, expected) in [
            ("こんにちは世界", "ja"),
            ("テスト", "ja"),
            ("这是一个纯中文测试句子。", "zh"),
            ("This is a standard English sentence.", "en"),
            ("Mixed English and テスト", "ja"),
            ("Mixed English and 中文测试", "zh"),
            ("Привет мир!", "ru"),
            ("مرحبا بالعالم", "ar"),
            ("Das ist ein großes Haus.", "de"),
            ("¡Hola! ¿Cómo estás?", "es"),
        ] {
            assert_eq!(detect_dominant_lang(input), expected);
        }
    }
    #[test]
    fn test_split_sentences_production_discourse_fusion() {
        let en_text = "We tried the new method. It worked well.";
        let en_sentences = split_sentences(en_text);
        assert_eq!(en_sentences.len(), 1);
        assert_eq!(en_sentences[0].text, "We tried the new method. It worked well.");

        let zh_text = "我们尝试了新方法。它也是如此。";
        let zh_sentences = split_sentences_with_lang(zh_text, Some("zh"));
        assert_eq!(zh_sentences.len(), 1);
        assert_eq!(zh_sentences[0].text, "我们尝试了新方法。它也是如此。");
    }

    #[test]
    fn test_split_sentences_hierarchical_preserves_unfused() {
        let text = "We tried the new method. It worked well.";
        let sentences = split_sentences_hierarchical_with_lang(text, DEFAULT_MAX_SENTENCE_LEN, Some("en"));
        assert_eq!(sentences.len(), 2);
        assert_eq!(sentences[0].text, "We tried the new method.");
        assert_eq!(sentences[1].text, "It worked well.");
    }

    #[test]
    fn test_long_sentence_chunking_trailing_separator_fidelity() {
        let long_sentence = "Alpha beta gamma delta, epsilon zeta eta theta, iota kappa lambda mu nu.";
        let full_text = format!("{}\n\nSecond paragraph begins here.", long_sentence);
        let sentences = split_sentences_bounded(&full_text, 40);
        assert!(sentences.len() >= 3);
        let long_parts: Vec<&SplitSentence> = sentences
            .iter()
            .filter(|s| !s.text.starts_with("Second"))
            .collect();
        let last_long_part = long_parts.last().unwrap();
        assert!(
            last_long_part.separator.contains("\n\n"),
            "Last sub-chunk separator must inherit original sentence trailing separator: {:?}",
            last_long_part.separator
        );
        assert_eq!(reassemble_sentences(&sentences), full_text);
    }

    #[test]
    fn test_fuse_discourse_chunks_linear_complexity_large_n() {
        let mut sentences = Vec::with_capacity(1000);
        for i in 0..500 {
            sentences.push(SplitSentence::new(
                format!("Statement number {} was recorded.", i),
                0..10,
                " ".to_string(),
            ));
            sentences.push(SplitSentence::new(
                "It worked properly.".to_string(),
                10..20,
                " ".to_string(),
            ));
        }
        let prof = RuleRegistry::global().get("en");
        fuse_discourse_chunks_with_profile(&mut sentences, 200, prof.as_deref());
        assert_eq!(sentences.len(), 500);
        for sent in &sentences {
            assert!(sent.text.contains("It worked properly."));
        }
    }

    #[test]
    fn test_abbreviation_lookahead_high_frequency_starters() {
        let text1 = "We brought tools, etc. The meeting was productive.";
        let s1 = split_sentences(text1);
        assert_eq!(s1.len(), 2);
        assert_eq!(s1[0].text, "We brought tools, etc.");
        assert_eq!(s1[1].text, "The meeting was productive.");

        let text2 = "We consulted Dr. Smith regarding the treatment.";
        let s2 = split_sentences(text2);
        assert_eq!(s2.len(), 1);

        let text3 = "Checked with Smith et al. Please do not delay.";
        let s3 = split_sentences(text3);
        assert_eq!(s3.len(), 2);
        assert_eq!(s3[0].text, "Checked with Smith et al.");
        assert_eq!(s3[1].text, "Please do not delay.");
    }

    #[test]
    fn test_name_initials_unicode_and_lowercase_protection() {
        let text1 = "John F. was president during the Cold War.";
        let s1 = split_sentences(text1);
        assert_eq!(s1.len(), 1);
        assert_eq!(s1[0].text, "John F. was president during the Cold War.");

        let text2 = "А. С. Пушкин был великим поэтом. Все читают его стихи.";
        let s2 = split_sentences_with_lang(text2, Some("ru"));
        assert_eq!(s2.len(), 2);
        assert_eq!(s2[0].text, "А. С. Пушкин был великим поэтом.");
        assert_eq!(s2[1].text, "Все читают его стихи.");
    }

    #[test]
    fn test_list_number_prefix_protection() {
        let text = "1. First item is ready.\n2. Second item is processing.\n3. Third item is finished.";
        let s = split_sentences(text);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].text, "1. First item is ready.");
        assert_eq!(s[1].text, "2. Second item is processing.");
        assert_eq!(s[2].text, "3. Third item is finished.");
    }

    #[test]
    fn test_oxford_comma_and_parallel_nouns_guard() {
        let text_oxford = "I purchased apples, oranges, and bananas at the supermarket yesterday morning.";
        let parts = split_long_sentence_semantic(text_oxford, 50);
        assert_eq!(reassemble_sentences(&parts), text_oxford);
        assert!(!parts.iter().any(|p| p.text.ends_with("oranges,") && p.separator == " "));
    }

    #[test]
    fn test_min_chunk_len_avoids_short_fragments() {
        let text = "Hi, this is a detailed explanation of our system architecture and deployment pipeline.";
        let parts = split_long_sentence_semantic(text, 45);
        assert!(!parts.iter().any(|p| p.text == "Hi,"));
        for part in &parts {
            assert!(part.text.chars().count() >= 20 || part.text.is_empty());
        }
        assert_eq!(reassemble_sentences(&parts), text);
    }

    #[test]
    fn test_japanese_dunhao_clause_conjunction() {
        let text_ja = "このシステムは非常に安定しており、しかしながら大量のリクエストが発生した場合には注意が必要です。";
        let parts_ja = split_long_sentence_semantic_with_lang(text_ja, 30, Some("ja"));
        assert!(parts_ja.len() >= 2);
        assert_eq!(parts_ja[0].text, "このシステムは非常に安定しており、");
        assert_eq!(reassemble_sentences(&parts_ja), text_ja);
    }

    #[test]
    fn test_fuse_discourse_chunks_mood_and_dialogue_guard() {
        let mut s_q = vec![
            SplitSentence::new("Are you sure?".to_string(), 0..13, " ".to_string()),
            SplitSentence::new("It worked well.".to_string(), 14..29, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut s_q, 100, None);
        assert_eq!(s_q.len(), 2);

        let mut s_excl = vec![
            SplitSentence::new("Look out!".to_string(), 0..9, " ".to_string()),
            SplitSentence::new("It is dangerous.".to_string(), 10..26, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut s_excl, 100, None);
        assert_eq!(s_excl.len(), 2);

        let mut s_quote = vec![
            SplitSentence::new("He said \"Wait.\"".to_string(), 0..15, " ".to_string()),
            SplitSentence::new("It was silent.".to_string(), 16..30, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut s_quote, 100, None);
        assert_eq!(s_quote.len(), 2);
    }

    #[test]
    fn test_rtl_punctuation_topology() {
        // Arabic question mark '؟'
        let ar_text = "كيف حالك؟ أنا بخير.";
        let ar_sents = split_sentences(ar_text);
        assert_eq!(ar_sents.len(), 2);
        assert_eq!(ar_sents[0].text, "كيف حالك؟");
        assert_eq!(ar_sents[1].text, "أنا بخير.");
        assert_eq!(reassemble_sentences(&ar_sents), ar_text);

        // Urdu full stop '۔'
        let ur_text = "یہ ایک کتاب ہے۔ قلم میز پر ہے۔";
        let ur_sents = split_sentences(ur_text);
        assert_eq!(ur_sents.len(), 2);
        assert_eq!(ur_sents[0].text, "یہ ایک کتاب ہے۔");
        assert_eq!(ur_sents[1].text, "قلم میز پر ہے۔");
        assert_eq!(reassemble_sentences(&ur_sents), ur_text);

        // Hebrew Sof Pasuq '׃'
        let he_text = "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃ וְהָאָרֶץ הָיְתָה תֹהוּ וָבֹהוּ";
        let he_sents = split_sentences(he_text);
        assert_eq!(he_sents.len(), 2);
        assert_eq!(he_sents[0].text, "בְּרֵאשִׁית בָּרָא אֱלֹהִים אֵת הַשָּׁמַיִם וְאֵת הָאָרֶץ׃");
        assert_eq!(reassemble_sentences(&he_sents), he_text);

        // Arabic semicolon '؛' and comma '،' in semantic split
        let ar_long = "هذا هو القسم الأول من النص الطويل الذي يحتوي على تفاصيل كثيرة؛ والقسم الثاني يناقش قضايا أخرى، والقسم الثالث يختم الفكرة.";
        let ar_semantic = split_long_sentence_semantic(ar_long, 40);
        assert!(ar_semantic.len() >= 2);
        assert_eq!(reassemble_sentences(&ar_semantic), ar_long);
    }

    #[test]
    fn test_cjk_semicolon_short_and_long() {
        // Short parallel clauses should stay unified in stage 1
        let short_parallel = "虚心使人进步；骄傲使人落后。";
        let short_sents = split_sentences(short_parallel);
        assert_eq!(short_sents.len(), 1);
        assert_eq!(short_sents[0].text, "虚心使人进步；骄傲使人落后。");

        // Long clauses (>= 15 chars) split in stage 1
        let long_parallel = "这是一段用于测试长句逻辑的并列前置从句；而这是紧接着的另一段同样具有充分长度的后置从句。";
        let long_sents = split_sentences(long_parallel);
        assert_eq!(long_sents.len(), 2);
        assert_eq!(long_sents[0].text, "这是一段用于测试长句逻辑的并列前置从句；");
        assert_eq!(long_sents[1].text, "而这是紧接着的另一段同样具有充分长度的后置从句。");
        assert_eq!(reassemble_sentences(&long_sents), long_parallel);
    }

    #[test]
    fn test_abbreviations_absolute_vs_conditional() {
        // Dr. He (surname He must not trigger high frequency sentence starter split)
        let dr_he = "Dr. He will present the findings tomorrow.";
        let dr_sents = split_sentences(dr_he);
        assert_eq!(dr_sents.len(), 1);
        assert_eq!(dr_sents[0].text, dr_he);

        // Fig. A (label must not trigger high frequency sentence starter split)
        let fig_a = "Please refer to Fig. A for details.";
        let fig_sents = split_sentences(fig_a);
        assert_eq!(fig_sents.len(), 1);
        assert_eq!(fig_sents[0].text, fig_a);

        // e.g. The (inline abbreviation must not trigger split)
        let eg_text = "We use classic tools, e.g. The GNU compiler collection.";
        let eg_sents = split_sentences(eg_text);
        assert_eq!(eg_sents.len(), 1);
        assert_eq!(eg_sents[0].text, eg_text);

        // etc. followed by capitalized sentence starter DOES split
        let etc_text = "We brought apples, oranges, etc. The meeting ended.";
        let etc_sents = split_sentences(etc_text);
        assert_eq!(etc_sents.len(), 2);
        assert_eq!(etc_sents[0].text, "We brought apples, oranges, etc.");
        assert_eq!(etc_sents[1].text, "The meeting ended.");
    }

    #[test]
    fn test_fuse_discourse_chunks_cjk_vs_western_threshold() {
        // CJK: follow-up discourse chunk > 15 chars does NOT fuse
        let mut cjk_sents_long = vec![
            SplitSentence::new("这是一段前置说明。".to_string(), 0..9, " ".to_string()),
            SplitSentence::new("而且这也是一段非常长的后续补充说明文本。".to_string(), 10..30, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut cjk_sents_long, 100, None);
        assert_eq!(cjk_sents_long.len(), 2);

        // CJK: follow-up discourse chunk <= 15 chars DOES fuse
        let mut cjk_sents_short = vec![
            SplitSentence::new("这是前置说明。".to_string(), 0..7, " ".to_string()),
            SplitSentence::new("而且确实如此。".to_string(), 8..14, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut cjk_sents_short, 100, None);
        assert_eq!(cjk_sents_short.len(), 1);
        assert_eq!(cjk_sents_short[0].text, "这是前置说明。 而且确实如此。");

        // Western: follow-up discourse chunk of 25 chars (<= 40) DOES fuse
        let mut en_sents = vec![
            SplitSentence::new("The system is running.".to_string(), 0..22, " ".to_string()),
            SplitSentence::new("However it needs attention.".to_string(), 23..50, "".to_string()),
        ];
        fuse_discourse_chunks_with_profile(&mut en_sents, 100, None);
        assert_eq!(en_sents.len(), 1);
        assert_eq!(en_sents[0].text, "The system is running. However it needs attention.");
    }

    #[test]
    fn test_parallel_coordination_guard_multilingual() {
        // English: "and" without verb guards against splitting
        assert!(is_parallel_coordination_guard("and oranges, bananas", None));
        // English: "and" with predicate verb does not guard
        assert!(!is_parallel_coordination_guard("and she went home", None));

        // German: "und" / "oder"
        assert!(is_parallel_coordination_guard("und Orangen, Birnen", None));
        assert!(is_parallel_coordination_guard("oder Bananen", None));
        assert!(!is_parallel_coordination_guard("und sie geht nach Hause", None));

        // French: "et" / "ou"
        assert!(is_parallel_coordination_guard("et des oranges", None));
        assert!(is_parallel_coordination_guard("ou des bananes", None));
        assert!(!is_parallel_coordination_guard("et elle va à la maison", None));

        // Spanish: "y" / "o"
        assert!(is_parallel_coordination_guard("y manzanas", None));
        assert!(is_parallel_coordination_guard("o peras", None));
        assert!(!is_parallel_coordination_guard("y ella va a la casa", None));

        // Russian: "и" / "или"
        assert!(is_parallel_coordination_guard("и апельсины", None));
        assert!(is_parallel_coordination_guard("или бананы", None));
        assert!(!is_parallel_coordination_guard("и она идет домой", None));
    }

    #[test]
    fn test_split_sentence_zero_copy_borrowing() {
        let text = "First sentence here. Second sentence follows. Third sentence ends here.";
        let sentences = split_sentences(text);
        assert_eq!(sentences.len(), 3);
        for s in &sentences {
            assert!(matches!(s.text, Cow::Borrowed(_)), "text should be zero-copy Cow::Borrowed");
            assert!(matches!(s.separator, Cow::Borrowed(_)), "separator should be zero-copy Cow::Borrowed");
        }
        assert_eq!(reassemble_sentences(&sentences), text);

        let long_text = "This is a long sentence that will be semantically divided into smaller chunks, because it exceeds max length.";
        let semantic_sents = split_long_sentence_semantic(long_text, 30);
        assert!(semantic_sents.len() > 1);
        for s in &semantic_sents {
            assert!(matches!(s.text, Cow::Borrowed(_)), "semantic split text should be Cow::Borrowed");
        }
    }

    #[test]
    fn test_slice_boundary_safety_and_edge_cases() {
        let edge_cases = [
            "",
            "   ",
            ".",
            "...",
            ". . .",
            "!",
            "?",
            ":",
            ";",
            " - ",
            "——",
            "，",
            "。",
            "！！！？？？",
            "Edge cases: . ! ? ; : - —— , ， 。",
            "Numbers: 123.456, v1.0.0, 49.99$ and 3.1415926.",
            "Trailing dots without spaces....",
            "Multi-byte emojis: 👨‍👩‍👧‍👦. Next part: 🚀? And done! 🌟",
            "Arabic and Hebrew: هل أنت بخير؟ نعم، أنا بخير. שלום לך׃ מה נשמע؟",
            "Mixed ;;;; :::: ---- —— 、、、 。。。",
        ];

        for &case in &edge_cases {
            let sents = split_sentences(case);
            let reassembled = reassemble_sentences(&sents);
            assert_eq!(reassembled, case);

            let bounded = split_sentences_bounded(case, 10);
            let reassembled_bounded = reassemble_sentences(&bounded);
            assert_eq!(reassembled_bounded, case);
        }
    }

    #[test]
    fn test_extreme_boundaries_and_unicode_edge_cases() {
        let extreme_cases = vec![
            String::new(),
            "\0".to_string(),
            "\0\0\0Hello\0world\0".to_string(),
            "\u{FFFD}\u{FFFD}\u{FFFD}".to_string(),
            "Invalid UTF-8 lossy: \u{FFFD}\u{FFFD}. Followed by normal text.".to_string(),
            "\r".to_string(),
            "\n".to_string(),
            "\r\n".to_string(),
            "\r\r\n\n\r\n\r".to_string(),
            "\t  \r\n  \t  \n".to_string(),
            " \t \r\n . \r\n ! \r\n ? \r\n".to_string(),
            "👨‍👩‍👧‍👦🧑🏽‍💻👩🏻‍🔬🚀".to_string(),
            "👨‍👩‍👧‍👦. 🧑🏽‍💻! 👩🏻‍🔬? 🚀... Done.".to_string(),
            "Combined accents: e\u{0301}cole, na\u{0308}ive, re\u{0301}sume\u{0301}.".to_string(),
            "Zero width: A\u{200B}B\u{200C}C\u{200D}D. Next sentence.".to_string(),
            "Rare punctuation: Armenian \u{0589} Ethiopic \u{1362} Arabic \u{061F} \u{061B} \u{060C} \u{06D4} Hebrew \u{05C3} Danda \u{0964} \u{0965}.".to_string(),
            "Brackets: 《三国演义》（上册）［明］罗贯中【清】毛宗岗「卷一」『第一回』。".to_string(),
            "Repeated delimiters: ...... !?!?!?!?!? ;;;;;; :::::: -------- —————— 、、、、、、 。。。。。。".to_string(),
            "Mixed code: fn test(x: &str) -> Option<usize> { if x.len() > 0 { return Some(42); } None }".to_string(),
            "Mixed URLs: Visit https://example.com/a.b.c/d?e=1&f=2#anchor. Then go to http://127.0.0.1:8080/api. Next.".to_string(),
            "Mixed acronyms and floats: U.S.A. GDP grew by 2.5% in Q3. Dr. J. F. Smith Jr. paid $49.99 for v2.0.1.".to_string(),
        ];

        for case in &extreme_cases {
            // Test 1: standard split
            let sents = split_sentences(case);
            assert_eq!(&reassemble_sentences(&sents), case);

            // Test 2: bounded split with various lengths (including extreme 1, 2, 5, 20)
            for max_len in [1, 2, 5, 10, 25, 100, 512] {
                let bounded = split_sentences_bounded(case, max_len);
                assert_eq!(
                    &reassemble_sentences(&bounded),
                    case,
                    "Failed reassembly invariant for max_len={max_len} on case: {:?}",
                    case
                );
            }
        }
    }

    #[test]
    fn test_long_unbroken_text_and_stress() {
        // 1. Long unbroken text without any split points (Latin)
        let unbroken_en = "a".repeat(10_000);
        let sents_en = split_sentences_bounded(&unbroken_en, 256);
        assert_eq!(&reassemble_sentences(&sents_en), &unbroken_en);
        assert!(sents_en.len() > 1);

        // 2. Long unbroken text without any split points (CJK)
        let unbroken_cjk = "测".repeat(10_000);
        let sents_cjk = split_sentences_bounded(&unbroken_cjk, 256);
        assert_eq!(&reassemble_sentences(&sents_cjk), &unbroken_cjk);
        assert!(sents_cjk.len() > 1);

        // 3. High sentence count stress: 1000 short sentences
        let mut stress_doc = String::with_capacity(30_000);
        for i in 0..1000 {
            stress_doc.push_str(&format!("Sentence number {} has some data. ", i));
        }
        let stress_sents = split_sentences(&stress_doc);
        assert_eq!(stress_sents.len(), 1000);
        assert_eq!(&reassemble_sentences(&stress_sents), &stress_doc);

        for s in &stress_sents {
            assert!(matches!(s.text, Cow::Borrowed(_)));
            assert!(matches!(s.separator, Cow::Borrowed(_)));
        }
    }

    #[test]
    fn test_zero_copy_pointer_validation() {
        let text = "First clause here, and second clause continues. Third sentence concludes.";
        let sents = split_sentences(text);
        let text_start = text.as_ptr() as usize;
        let text_end = text_start + text.len();

        for s in &sents {
            if let Cow::Borrowed(b_text) = s.text {
                let ptr = b_text.as_ptr() as usize;
                assert!(ptr >= text_start && ptr + b_text.len() <= text_end);
            } else {
                panic!("Expected borrowed Cow");
            }

            if !s.separator.is_empty() {
                if let Cow::Borrowed(b_sep) = s.separator {
                    let ptr = b_sep.as_ptr() as usize;
                    assert!(ptr >= text_start && ptr + b_sep.len() <= text_end);
                } else {
                    panic!("Expected borrowed Cow separator");
                }
            }

            // Test into_owned()
            let owned = s.clone().into_owned();
            assert!(matches!(owned.text, Cow::Owned(_)));
            assert_eq!(owned.text, s.text);
            assert_eq!(owned.separator, s.separator);
            assert_eq!(owned.text_range, s.text_range);
        }
    }

    #[test]
    fn test_cjk_min_chunk_len_and_dialogue_dash() {
        // CJK clauses with >= 5 chars should be eligible for semantic splitting when exceeding budget
        let cjk = "今天的天气非常好，我们一起去公园散步吧。";
        let parts = split_long_sentence_semantic(cjk, 12);
        assert!(parts.len() >= 2);
        let dialogue = "— Bonjour. — Bonjour, comment allez-vous ?";
        let sents = split_sentences_bounded(dialogue, 100);
        assert_eq!(sents.len(), 2);
    }
}
