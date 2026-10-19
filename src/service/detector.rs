//! Language detection service with sharded concurrent LRU caching.
//!
//! Bridges asynchronous API requests to the unified zero-allocation detection engine
//! (`crate::engine::rules::detect`). Maintains thread-safe sharded LRU caches
//! for ultra-fast repetitive request resolution and sentence boundary segmentation.
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use lru::LruCache;
use parking_lot::Mutex;
use unicode_segmentation::UnicodeSegmentation;


/// Shard count for the detection cache: power of two, selected by a byte
/// mask on the key hash. Independent per-shard locks keep concurrent
/// detections from serialising on one global mutex.
const DETECT_CACHE_SHARDS: usize = 16;

/// Per-shard capacity so the aggregate bound stays at the old total.
const DETECT_CACHE_CAPACITY_PER_SHARD: usize = 32;

#[derive(Clone)]
struct CachedDetection {
    language: String,
    confidence: f32,
}

type DetectShard = Mutex<Option<LruCache<u64, CachedDetection>>>;

static DETECT_CACHE_SHARD_TABLE: [DetectShard; DETECT_CACHE_SHARDS] = [
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
    Mutex::new(None),
];

/// FNV-1a over the bytes that actually influence the detection result:
/// the entry kind, the sample cap, and the (already bounded) input text.
/// A fixed-size `u64` key avoids copying whole texts into cache keys.
fn detect_cache_key(kind: char, max_bytes: usize, text: &str) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            hash ^= b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    feed(&[kind as u8]);
    feed(&max_bytes.to_le_bytes());
    feed(&text.len().to_le_bytes());
    let capped = text.len().min(max_bytes.max(MAX_HEURISTIC_SAMPLE_BYTES));
    feed(&text.as_bytes()[..capped]);
    hash
}

fn detect_cache_shard(key: u64) -> &'static DetectShard {
    &DETECT_CACHE_SHARD_TABLE[(key as usize) & (DETECT_CACHE_SHARDS - 1)]
}

#[derive(Clone)]
pub struct TextSegment {
    pub text: String,
    pub language: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

const DEFAULT_CONFIDENCE_THRESHOLD: f32 = 0.5;
const MAX_DETECTION_BYTES: usize = 511;
const MAX_FALLBACK_DETECTION_BYTES: usize = 1023;

/// Detect the dominant language of a text. Returns language code or "en" if undetectable.
///
/// Mirrors `detectLanguage` in the original TypeScript: skip leading
/// non-alphanumeric content, try the pure-CJK heuristic first, then fall back to whatlang.
pub async fn detect_language(text: &str) -> String {
    detect_language_with_max_bytes(text, MAX_DETECTION_BYTES).await
}

async fn detect_language_with_max_bytes(text: &str, max_bytes: usize) -> String {
    let cache_key = detect_cache_key('W', max_bytes, text);
    if let Some(cached) = detect_cache_get(cache_key) {
        return cached.language;
    }

    let result = detect_language_with_max_bytes_uncached(text, max_bytes).await;
    detect_cache_put(
        cache_key,
        CachedDetection {
            language: result.clone(),
            confidence: 1.0,
        },
    );
    result
}

fn detect_cache_get(key: u64) -> Option<CachedDetection> {
    let shard = &mut *detect_cache_shard(key).lock();
    shard.as_mut()?.get(&key).cloned()
}

fn detect_cache_put(key: u64, value: CachedDetection) {
    let shard = &mut *detect_cache_shard(key).lock();
    let cache = shard.get_or_insert_with(|| {
        LruCache::new(std::num::NonZeroUsize::new(DETECT_CACHE_CAPACITY_PER_SHARD).unwrap())
    });
    cache.put(key, value);
}

/// Upper bound on the text fed to the heuristic scanners. Every signal the
/// heuristics use (script runs, distinctive letters, function words) appears
/// within the first few kilobytes, so capping the sample keeps detection cost
/// constant regardless of input size — matching the dual-engine path.
const MAX_HEURISTIC_SAMPLE_BYTES: usize = 4096;

async fn detect_language_with_max_bytes_uncached(text: &str, max_bytes: usize) -> String {
    if text.is_empty() {
        return String::new();
    }

    let text = sanitize_input(text);
    let start_index = get_valid_content_start_index(&text);
    if start_index >= text.len() {
        return "en".to_string();
    }

    let safe_slice = if text.is_char_boundary(start_index) {
        &text[start_index..]
    } else {
        let mut b = start_index;
        while b > 0 && !text.is_char_boundary(b) {
            b -= 1;
        }
        &text[b..]
    };

    let sample = truncate_by_utf8_bytes(
        safe_slice,
        max_bytes.max(MAX_HEURISTIC_SAMPLE_BYTES),
    );
    let (code, _) = crate::engine::rules::detect::detect_text_language_with_confidence(sample);
    code.to_string()
}

/// Detect language with confidence score.
pub async fn detect_language_with_confidence(text: &str, min_confidence: f32) -> (String, f32) {
    if text.is_empty() {
        return (String::new(), 0.0);
    }

    let text = sanitize_input(text);
    let start_index = get_valid_content_start_index(&text);
    if start_index >= text.len() {
        return ("en".to_string(), 0.0);
    }

    let safe_slice = if text.is_char_boundary(start_index) {
        &text[start_index..]
    } else {
        let mut b = start_index;
        while b > 0 && !text.is_char_boundary(b) {
            b -= 1;
        }
        &text[b..]
    };

    let (code, confidence) = detect_language_from(safe_slice, MAX_DETECTION_BYTES).await;
    if confidence < min_confidence {
        return (String::new(), confidence);
    }
    (code, confidence)
}

/// Detect multiple language segments in mixed text.
///
/// Splits by sentence boundaries (UAX #29), detects language per sentence with
/// confidence gating, applies unified script/dominant language fallbacks with
/// contextual smoothing (inheriting previous language for punctuation/numbers/short segments),
/// merges adjacent identical-language segments, and limits languages to top matches.
pub async fn detect_multiple_languages(text: &str) -> Vec<TextSegment> {
    if text.is_empty() {
        return Vec::new();
    }

    // Whole-text fallback language
    let fallback = detect_language_with_max_bytes(text, MAX_FALLBACK_DETECTION_BYTES).await;
    let effective_fallback = if fallback.is_empty() {
        "en".to_string()
    } else {
        fallback
    };


    // Split by sentence boundaries (mirrors Intl.Segmenter)
    let spans = split_sentence_spans(text);
    if spans.len() <= 1 {
        return vec![TextSegment {
            text: text.to_string(),
            language: effective_fallback,
            start: 0,
            end: text.len(),
            confidence: 1.0,
        }];
    }

    let mut segments: Vec<TextSegment> = Vec::with_capacity(spans.len());
    let mut previous_lang: Option<String> = None;

    for (start_byte, end_byte) in spans {
        let sentence = &text[start_byte..end_byte];
        let has_letters = sentence.chars().any(|c| c.is_alphabetic());

        let mut final_lang = if !has_letters {
            // Pure punctuation, digits, or whitespace: smooth with previous sentence
            previous_lang
                .as_ref()
                .cloned()
                .unwrap_or_else(|| effective_fallback.clone())
        } else {
            let (detected_lang, confidence) = detect_language_from(sentence, MAX_DETECTION_BYTES).await;
            if confidence >= DEFAULT_CONFIDENCE_THRESHOLD && !detected_lang.is_empty() && detected_lang != "un" {
                detected_lang
            } else {
                let dom = detect_dominant_language(sentence);
                if !dom.is_empty() && dom != "en" {
                    dom.to_string()
                } else if !detected_lang.is_empty() && detected_lang != "un" {
                    detected_lang
                } else {
                    previous_lang
                        .as_ref()
                        .cloned()
                        .unwrap_or_else(|| effective_fallback.clone())
                }
            }
        };

        if final_lang == "zh-Hans" || final_lang == "zh-Hant" || final_lang == "zh" {
            final_lang = distinguish_chinese_variant(sentence).to_string();
        }

        previous_lang = Some(final_lang.clone());

        segments.push(TextSegment {
            text: sentence.to_string(),
            language: final_lang,
            start: start_byte,
            end: end_byte,
            confidence: 1.0,
        });
    }

    // Merge adjacent segments with same language
    let merged = merge_adjacent_segments(segments, text);
    // Limit to MAXIMUM_LANGUAGES_IN_ONE_TEXT (relaxed to 6 to match max model capacity)
    limit_languages(merged, text, 6)
}

/// Detect the language of raw text from its start (no trim / no leading-content skip),
/// returning `(normalized code, confidence)`. Pure-CJK heuristic runs first.
async fn detect_language_from(text: &str, max_bytes: usize) -> (String, f32) {
    let cache_key = detect_cache_key('F', max_bytes, text);
    if let Some(cached) = detect_cache_get(cache_key) {
        return (cached.language, cached.confidence);
    }

    let (lang, conf) = detect_language_from_uncached(text, max_bytes).await;
    detect_cache_put(
        cache_key,
        CachedDetection {
            language: lang.clone(),
            confidence: conf,
        },
    );
    (lang, conf)
}

async fn detect_language_from_uncached(text: &str, max_bytes: usize) -> (String, f32) {
    let text = sanitize_input(text);
    let sample = truncate_by_utf8_bytes(&text, max_bytes.max(MAX_HEURISTIC_SAMPLE_BYTES));
    let (code, conf) = crate::engine::rules::detect::detect_text_language_with_confidence(sample);
    (code.to_string(), conf)
}

/// Detect languages whose script maps unambiguously to a single model language,
/// before falling back to n-gram engines. For these scripts there is no
/// classification ambiguity, so the result is exact and cheaper than the engines.
#[allow(dead_code)]
fn detect_language_by_script(text: &str, start_index: usize) -> Option<&'static str> {
    let scan_text = if start_index > 0 && text.is_char_boundary(start_index) {
        &text[start_index..]
    } else if start_index > 0 {
        let mut b = start_index;
        while b > 0 && !text.is_char_boundary(b) {
            b -= 1;
        }
        &text[b..]
    } else {
        text
    };
    let mut has_arabic = false;
    let mut has_persian = false;
    let mut has_urdu = false;
    let mut greek_count = 0usize;
    let mut latin_count = 0usize;

    for c in scan_text.chars() {
        let v = c as u32;
        if (0x0370..=0x03FF).contains(&v) || (0x1F00..=0x1FFF).contains(&v) {
            greek_count += 1;
            continue;
        }
        if c.is_ascii_alphabetic() {
            latin_count += 1;
            continue;
        }
        match v {
            0x10A0..=0x10FF | 0x2D00..=0x2D2F => return Some("ka"),
            0x0590..=0x05FF => return Some("he"),
            0x0E00..=0x0E7F => return Some("th"),
            0x0980..=0x09FF => return Some("bn"),
            0x0A80..=0x0AFF => return Some("gu"),
            0x0900..=0x097F => return Some("hi"),
            0x0B80..=0x0BFF => return Some("ta"),
            0x0C00..=0x0C7F => return Some("te"),
            0x0C80..=0x0CFF => return Some("kn"),
            0x0D00..=0x0D7F => return Some("ml"),
            // Urdu-specific letters: Tteh, Ddal, Rreh, Noon Ghunna, Heh Doachashmee, Choti Heh, Yeh Barree
            0x0679 | 0x0688 | 0x0691 | 0x06BA | 0x06BE | 0x06C1 | 0x06D2 => {
                has_urdu = true;
            }
            // Persian-specific letters (gaf, che, zhe, pe, farsi yeh) — MUST precede the
            // broad Arabic range below so they are not swallowed by it.
            0x06AF | 0x0686 | 0x0698 | 0x067E | 0x06CC => {
                has_persian = true;
            }
            // Arabic script (including Persian/Urdu extensions).
            0x0600..=0x06FF
            | 0x0750..=0x077F
            | 0x08A0..=0x08FF
            | 0xFB50..=0xFDFF
            | 0xFE70..=0xFEFF => {
                has_arabic = true;
            }
            _ => {}
        }
    }

    if has_urdu {
        return Some("ur");
    }
    if has_persian {
        return Some("fa");
    }
    if has_arabic {
        return Some("ar");
    }
    if (greek_count >= 2 && greek_count > latin_count)
        || (greek_count > 0 && latin_count == 0)
    {
        return Some("el");
    }

    None
}

/// Detect a language from its uniquely-identifying letters/diacritics, before
/// running n-gram engines. Several model languages use specific characters not
/// shared with their nearest neighbours (Albanian ë, Icelandic þ/ð, Hungarian
/// ő/ű, Romanian ș/ț, Czech ř/ě/ů, Slovak ľ/ŕ). When one of these appears, the
/// language is certain, so we return it with full confidence — cheaper and more
#[allow(dead_code)]
fn detect_language_by_chars(text: &str, start_index: usize) -> Option<&'static str> {
    // Single pass: collect the is/az flags (position-independent) and, for
    // every other language, the earliest byte offset of any of its markers.
    // Marker code points are disjoint across languages, so "first hit in
    // left-to-right order" is exactly "smallest offset wins".
    let mut has_ith = false;
    let mut has_az = false;
    let mut first_marker: Option<(usize, &'static str)> = None;

    for (i, c) in text.char_indices().skip_while(|&(i, _)| i < start_index) {
        let v = c as u32;
        // Strongest markers first — scan the whole text, because single-char
        // early-return would mis-fire when a shorter Turkish/Azerbaijani word
        // contains an overlapping letter (e.g. Azerbaijani `Sabahınız` ı before `ə`).
        if v == 0x00DE || v == 0x00FE || v == 0x00D0 || v == 0x00F0 {
            has_ith = true;
        }
        if v == 0x0259 {
            has_az = true;
        }

        let lang = match v {
            0x011F | 0x0131 => Some("tr"),
            0x0150 | 0x0170 | 0x0151 | 0x0171 => Some("hu"),
            0x0219 | 0x021B => Some("ro"),
            0x0158 | 0x0159 | 0x011A | 0x011B | 0x016E | 0x016F => Some("cs"),
            0x013D | 0x013E | 0x0154 | 0x0155 => Some("sk"),
            0x0141 | 0x0142 | 0x0104 | 0x0105 | 0x0118 | 0x0119 => Some("pl"),
            0x0116 | 0x0117 | 0x012E | 0x012F | 0x0172 | 0x0173 => Some("lt"),
            0x013C | 0x0146 | 0x0122 | 0x0123 | 0x0136 | 0x0137 => Some("lv"),
            _ => None,
        };
        if let Some(lang) = lang {
            if first_marker.map_or(true, |(pos, _)| i < pos) {
                first_marker = Some((i, lang));
            }
        }
    }

    if has_ith {
        return Some("is");
    }
    if has_az {
        return Some("az");
    }
    first_marker.map(|(_, lang)| lang)
}
/// Detect a language from rare, highly-distinctive function/content words,
/// consulted before the n-gram engines. Unlike the character heuristic this
/// targets languages whatlang models poorly or confuses with neighbours, using
/// a handful of high-precision tokens per language (kept deliberately small).
/// Whole-word matching prevents false positives from accidental substrings.
#[allow(dead_code)]
fn detect_language_by_words(text: &str) -> Option<&'static str> {
    // (distinctive word, language). Normalized lowercase; matched as whole words.
    const WORDTABLE: &[(&str, &str)] = &[
        // Galician — "hoxe" (today) is unambiguous.
        ("hoxe", "gl"),
        // Albanian hello vs Afrikaans.
        ("përshëndetje", "sq"),
        // Basque.
        ("kaixo", "eu"),
        ("egun on", "eu"),
        // Malay vs Indonesian.
        ("khabar", "ms"),
        // Indonesian.
        ("selamat", "id"),
        // Nynorsk vs Bokmål/Danish.
        ("korleis", "nn"),
        ("ikkje", "nn"),
        // Croatian vs Slovenian ("danas" = today).
        ("danas", "hr"),
        // Slovenian.
        ("danes", "sl"),
        // Icelandic (no þ/ð in this text).
        ("hvernig", "is"),
        // Serbian Cyrillic дvs Macedonian (денес).
        ("данас", "sr"),
        ("хвала", "sr"),
        // Ukrainian vs Russian generic.
        ("з вами", "uk"),
        ("справи", "uk"),
        // Belarusian.
        ("добры дзень", "be"),
        // Bulgarian — "днес" (today), "здравейте" (hello), "благодаря" (thanks).
        ("днес", "bg"),
        ("здравейте", "bg"),
        // Spanish vs Portuguese short-word tie-breaker.
        ("hola", "es"),
        // Italian vs Portuguese short-word tie-breaker.
        ("ciao", "it"),
        // Danish vs Norwegian Bokmål (near-identical orthography, so use the
        // couple of distinguishing interrogatives). Icelandic already caught by ð.
        ("hvad", "da"),
        ("hedder", "da"),
        ("hva", "nb"),
        ("heter", "nb"),
    ];
    let lower = to_lowercase_borrowed(text);
    for (word, lang) in WORDTABLE {
        // Multi-word phrases never survive token splitting; match them against
        // the raw lowercased text with word boundaries so "egun on" still hits.
        let hit = if word.contains(' ') {
            contains_word_phrase(&lower, word)
        } else {
            find_whole_word(&lower, word, 0).is_some()
        };
        if hit {
            return Some(*lang);
        }
    }
    None
}

/// Lowercase `text`; borrows when it contains no uppercase characters.
#[allow(dead_code)]
fn to_lowercase_borrowed(text: &str) -> Cow<'_, str> {
    if text.chars().any(|c| c.is_uppercase()) {
        Cow::Owned(text.to_lowercase())
    } else {
        Cow::Borrowed(text)
    }
}

/// True if all tokens of `phrase` appear in order as whole words in `lower`.
#[allow(dead_code)]
fn contains_word_phrase(lower: &str, phrase: &str) -> bool {
    let mut search_from = 0usize;
    for tok in phrase.split(' ') {
        let Some((pos, len)) = find_whole_word(lower, tok, search_from) else {
            return false;
        };
        search_from = pos + len;
    }
    true
}

/// Find `needle` as a whole word in `lower` starting at or after `from`.
#[allow(dead_code)]
fn find_whole_word(lower: &str, needle: &str, from: usize) -> Option<(usize, usize)> {
    if from >= lower.len() || !lower.is_char_boundary(from) {
        return None;
    }
    let mut i = from;
    while i < lower.len() {
        let rel = lower[i..].find(needle)?;
        let pos = i + rel;
        let before_ok = pos == 0
            || lower[..pos]
                .chars()
                .next_back()
                .map_or(true, |c| !c.is_alphanumeric());
        let after = pos + needle.len();
        let after_ok = after >= lower.len()
            || lower[after..]
                .chars()
                .next()
                .map_or(true, |c| !c.is_alphanumeric());
        if before_ok && after_ok {
            return Some((pos, needle.len()));
        }
        if let Some(ch) = lower[pos..].chars().next() {
            i = pos + ch.len_utf8();
        } else {
            break;
        }
    }
    None
}

/// Remove null bytes and control characters from input text.
///
/// Matches the TS `/[\\x01-\\x08\\x0B-\\x0C\\x0E-\\x1F\\x7F]/` pattern, which
/// only strips the real U+0001.. control points — never the low byte of a
/// multi-byte code point such as 丁 (U+4E01).
fn sanitize_input(text: &str) -> Cow<'_, str> {
    if !text.as_bytes().iter().any(|&b| {
        b == 0
            || (1..=8).contains(&b)
            || (11..=12).contains(&b)
            || (14..=31).contains(&b)
            || b == 127
    }) {
        return Cow::Borrowed(text);
    }
    let clean = |c: char| {
        let v = c as u32;
        c != '\0'
            && !(1..=8).contains(&v)
            && !(11..=12).contains(&v)
            && !(14..=31).contains(&v)
            && v != 127
    };
    match text.chars().position(|c| !clean(c)) {
        None => Cow::Borrowed(text),
        Some(_) => {
            let mut out = String::with_capacity(text.len());
            for c in text.chars() {
                if clean(c) {
                    out.push(c);
                }
            }
            Cow::Owned(out)
        }
    }
}

/// Truncate a string to at most `max_bytes` UTF-8 bytes without splitting a char,
/// mirroring `truncateByUtf8Bytes` in the original TypeScript.
fn truncate_by_utf8_bytes(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[inline]
pub fn is_traditional_feature_char(c: char) -> bool {
    crate::engine::rules::detect::is_traditional_char(c)
}

#[inline]
pub fn is_simplified_feature_char(c: char) -> bool {
    crate::engine::rules::detect::is_simplified_char(c)
}

/// Distinguish Traditional Chinese (`zh-Hant`) from Simplified Chinese (`zh-Hans`)
/// based on statistical hits of distinctive feature characters.
#[inline]
pub fn distinguish_chinese_variant(text: &str) -> &'static str {
    crate::engine::rules::detect::distinguish_chinese_variant(text)
}

/// Pure CJK heuristic: detect ja/ko/zh (zh-Hans vs zh-Hant) by script ranges.
pub fn detect_pure_cjk_language(text: &str, start_index: usize) -> Option<&'static str> {
    let safe_start = if text.is_char_boundary(start_index) {
        start_index
    } else {
        let mut b = start_index;
        while b > 0 && !text.is_char_boundary(b) {
            b -= 1;
        }
        b
    };
    let target_limit = text.len().min(safe_start.saturating_add(2000));
    let mut limit = target_limit;
    while limit > 0 && !text.is_char_boundary(limit) {
        limit -= 1;
    }

    let mut has_han = false;
    let mut has_kana = false;
    let mut has_hangul = false;
    let mut cjk_count = 0usize;
    let mut ascii_letter_count = 0usize;

    for (i, c) in text.char_indices() {
        if i < safe_start {
            continue;
        }
        if i >= limit {
            break;
        }

        let code = c as u32;

        if (0x41..=0x5A).contains(&code) || (0x61..=0x7A).contains(&code) {
            ascii_letter_count += 1;
            continue;
        }

        if (0x3040..=0x30FF).contains(&code) {
            has_kana = true;
            cjk_count += 1;
        } else if (0xAC00..=0xD7AF).contains(&code) {
            has_hangul = true;
            cjk_count += 1;
        } else if (0x4E00..=0x9FFF).contains(&code) {
            has_han = true;
            cjk_count += 1;
        }
    }

    // If ASCII letters dominate or no CJK was found, do not classify as CJK.
    if cjk_count == 0 || (ascii_letter_count > 0 && cjk_count < ascii_letter_count * 2) {
        return None;
    }

    if has_kana {
        Some("ja")
    } else if has_hangul {
        Some("ko")
    } else if has_han {
        let scan = if safe_start < limit {
            &text[safe_start..limit]
        } else {
            &text[..limit]
        };
        Some(distinguish_chinese_variant(scan))
    } else {
        None
    }
}

/// Get the index where actual content starts (skip leading non-letter/number chars).
pub fn get_valid_content_start_index(text: &str) -> usize {
    text.char_indices()
        .find(|(_, c)| c.is_alphabetic() || c.is_numeric())
        .map(|(i, _)| i)
        .unwrap_or(0)
}


fn merge_adjacent_segments(segments: Vec<TextSegment>, original_text: &str) -> Vec<TextSegment> {
    let mut iter = segments.into_iter();
    let Some(mut current) = iter.next() else {
        return Vec::new();
    };
    let mut merged: Vec<TextSegment> = Vec::new();
    for next in iter {
        if current.language == next.language {
            current.text = original_text[current.start..next.end].to_string();
            current.end = next.end;
            if next.confidence > current.confidence {
                current.confidence = next.confidence;
            }
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    merged
}

fn limit_languages(
    mut segments: Vec<TextSegment>,
    original_text: &str,
    max_langs: usize,
) -> Vec<TextSegment> {
    if segments.len() <= 1 {
        return segments;
    }
    // Count bytes per language
    let mut lang_bytes: HashMap<String, usize> = HashMap::new();
    for seg in &segments {
        *lang_bytes.entry(seg.language.clone()).or_insert(0) += seg.end - seg.start;
    }
    if lang_bytes.len() <= max_langs {
        return segments;
    }
    // Sort languages by byte count descending
    let mut sorted: Vec<(String, usize)> = lang_bytes.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let keep_langs: HashSet<String> = sorted
        .iter()
        .take(max_langs)
        .map(|(l, _)| l.clone())
        .collect();
    let primary_lang = sorted.first().map(|(l, _)| l.as_str()).unwrap_or("en");

    for seg in &mut segments {
        if !keep_langs.contains(&seg.language) {
            seg.language.clear();
            seg.language.push_str(primary_lang);
        }
    }
    merge_adjacent_segments(segments, original_text)
}

/// Split text into sentence spans using Unicode sentence boundary rules (UAX #29),
/// matching Intl.Segmenter with granularity: 'sentence'. Returns `(start, end)` byte offsets.
pub fn split_sentence_spans(text: &str) -> Vec<(usize, usize)> {
    let mut spans = Vec::with_capacity(text.len() / 64 + 4);
    for (start, s) in text.split_sentence_bound_indices() {
        if !s.is_empty() {
            spans.push((start, start + s.len()));
        }
    }
    spans
}

/// Split text into sentences using Unicode sentence boundary rules (UAX #29),
/// matching Intl.Segmenter with granularity: 'sentence'.
pub fn split_sentence_boundaries(text: &str) -> Vec<(String, usize)> {
    split_sentence_spans(text)
        .into_iter()
        .map(|(start, end)| (text[start..end].to_string(), start))
        .collect()
}

/// Heuristically and synchronously detect the dominant language of a text for language rule profiles.
///
/// Combines pure-CJK script inspection (ja, ko, zh), unique scripts (ru, ar, he, el, th, hi),
/// distinctive character diacritics (de, es, pt, fr, it), distinctive words, and dual-engine fallback (whatlang / whichlang).
pub fn detect_dominant_language(text: &str) -> &'static str {
    crate::engine::rules::detect::detect_dominant_language(text)
}


#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_detect_language_multithread() {
        assert_eq!(detect_language("Hello world, this is a test.").await, "en");
        assert_eq!(detect_language("你好，世界！").await, "zh-Hans");
        assert_eq!(detect_language("こんにちは世界").await, "ja");
        assert_eq!(detect_language("안녕하세요").await, "ko");
        assert_eq!(detect_language("").await, "");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_detect_language_current_thread_fallback() {
        assert_eq!(
            detect_language("Hello world from single thread.").await,
            "en"
        );
        assert_eq!(detect_language("你好，世界！单线程测试").await, "zh-Hans");
        assert_eq!(
            detect_language("Bonjour le monde, ceci est un test.").await,
            "fr"
        );
    }

    #[tokio::test]
    async fn test_detect_with_confidence_and_caching() {
        let text = "This is an English sentence for detector cache verification.";
        let (lang1, conf1) = detect_language_with_confidence(text, 0.1).await;
        assert_eq!(lang1, "en");
        assert!(conf1 > 0.1);

        // Second call should hit the cache and produce identical results
        let (lang2, conf2) = detect_language_with_confidence(text, 0.1).await;
        assert_eq!(lang1, lang2);
        assert_eq!(conf1, conf2);

        // Standard detect_language cache test
        let l1 = detect_language(text).await;
        let l2 = detect_language(text).await;
        assert_eq!(l1, l2);
        assert_eq!(l1, "en");
    }

    #[tokio::test]
    async fn test_detect_multiple_languages() {
        let mixed = "Hello world! This is English. 你好世界！这是中文。";
        let segments = detect_multiple_languages(mixed).await;
        assert!(!segments.is_empty());
        let langs: Vec<&str> = segments.iter().map(|s| s.language.as_str()).collect();
        assert!(langs.contains(&"en"));
        assert!(langs.contains(&"zh-Hans"));
    }

    #[test]
    fn test_detect_language_by_script() {
        assert_eq!(detect_language_by_script("Ελληνικά", 0), Some("el"));
        assert_eq!(detect_language_by_script("עברית", 0), Some("he"));
        assert_eq!(detect_language_by_script("ภาษาไทย", 0), Some("th"));
        assert_eq!(detect_language_by_script("বাংলা", 0), Some("bn"));
        assert_eq!(detect_language_by_script("ગુજરાતી", 0), Some("gu"));
        assert_eq!(detect_language_by_script("हिन्दी", 0), Some("hi"));
        assert_eq!(detect_language_by_script("தமிழ்", 0), Some("ta"));
        assert_eq!(detect_language_by_script("తెలుగు", 0), Some("te"));
        assert_eq!(detect_language_by_script("ಕನ್ನಡ", 0), Some("kn"));
        assert_eq!(detect_language_by_script("മലയാളം", 0), Some("ml"));
        assert_eq!(detect_language_by_script("فارسی چ گ پ ژ", 0), Some("fa"));
        assert_eq!(detect_language_by_script("العربية", 0), Some("ar"));
        assert_eq!(detect_language_by_script("   Ελληνικά", 3), Some("el"));
    }

    #[test]
    fn test_detect_language_by_words_cyrillic_uppercase() {
        // Test Cyrillic uppercase in word-based detection
        assert_eq!(detect_language_by_words("ДНЕС"), Some("bg"));
        assert_eq!(detect_language_by_words("ЗДРАВЕЙТЕ"), Some("bg"));
        assert_eq!(detect_language_by_words("ХВАЛА"), Some("sr"));
    }

    #[test]
    fn test_distinguish_chinese_variant() {
        let trad_text = "我們正在實施繁體中文的語言識別優化，測試繁簡特徵字。這點會讓我們學習與說話更標準。";
        assert_eq!(distinguish_chinese_variant(trad_text), "zh-Hant");

        let simp_text = "我们正在实施简体中文的语言识别优化，测试繁简特征字。这点会让我们学习与说话更标准。";
        assert_eq!(distinguish_chinese_variant(simp_text), "zh-Hans");

        // Neutral / no distinctive characters defaults to zh-Hans
        assert_eq!(distinguish_chinese_variant("你好世界"), "zh-Hans");
    }

    #[tokio::test]
    async fn test_detect_traditional_chinese() {
        let trad = "臺灣地區的經濟發展與國際貿易關係研究報告，涵蓋資訊網絡技術及各類設備。";
        assert_eq!(detect_language(trad).await, "zh-Hant");

        let simp = "中国各地区的经济发展与国际贸易关系研究报告，涵盖信息网络技术及各类设备。";
        assert_eq!(detect_language(simp).await, "zh-Hans");
    }

    #[tokio::test]
    async fn test_detect_multiple_languages_traditional() {
        let mixed = "Hello world! This is English. 這是一段繁體中文測試內容，包含許多繁體字與特徵字。";
        let segments = detect_multiple_languages(mixed).await;
        assert!(!segments.is_empty());
        let langs: Vec<&str> = segments.iter().map(|s| s.language.as_str()).collect();
        assert!(langs.contains(&"en"));
        assert!(langs.contains(&"zh-Hant"));
    }
    #[tokio::test]
    async fn test_detect_multiple_languages_latin_and_cyrillic() {
        let mixed = "Hello world! This is English. Привет мир! Это русский текст.";
        let segments = detect_multiple_languages(mixed).await;
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].language, "en");
        assert_eq!(segments[1].language, "ru");
    }

    #[tokio::test]
    async fn test_detect_multiple_languages_trilingual() {
        let mixed = "Hello world! This is English. Привет мир! Это русский язык. 你好世界！这是简体中文。";
        let segments = detect_multiple_languages(mixed).await;
        assert_eq!(segments.len(), 3);
        let langs: Vec<&str> = segments.iter().map(|s| s.language.as_str()).collect();
        assert_eq!(langs, vec!["en", "ru", "zh-Hans"]);
    }

    #[tokio::test]
    async fn test_detect_multiple_languages_intra_script_latin() {
        let mixed = "This is a simple English sentence for detection. ¡Hola amigo, esta es una frase en español con muchas palabras!";
        let segments = detect_multiple_languages(mixed).await;
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].language, "en");
        assert_eq!(segments[1].language, "es");
    }
    #[test]
    fn test_detect_dominant_language_sync() {
        assert_eq!(detect_dominant_language("こんにちは世界"), "ja");
        assert_eq!(detect_dominant_language("안녕하세요 세계"), "ko");
        assert_eq!(detect_dominant_language("你好，这是一个测试。"), "zh");
        assert_eq!(detect_dominant_language("Привет мир!"), "ru");
        assert_eq!(detect_dominant_language("مرحبا بالعالم"), "ar");
        assert_eq!(detect_dominant_language("שלום עולם"), "he");
        assert_eq!(detect_dominant_language("¡Hola! ¿Cómo estás?"), "es");
        assert_eq!(detect_dominant_language("Das ist ein großes Problem."), "de");
        assert_eq!(detect_dominant_language("C'est une belle journée."), "fr");
        assert_eq!(detect_dominant_language("Não temos informações."), "pt");
        assert_eq!(detect_dominant_language("This is a simple English sentence."), "en");
    }

    #[tokio::test]
    async fn test_detect_greek_math_symbol_and_diaeresis_guards() {
        // Single Greek letter in English math/science context must not be misclassified as Greek ("el")
        assert_eq!(detect_language_by_script("The angle θ is measured in radians.", 0), None);
        assert_eq!(detect_language("The angle θ is measured in radians.").await, "en");
        assert_eq!(detect_dominant_language("The learning rate α is set to 0.001."), "en");

        // French and Dutch words with 'ë' (0x00EB) must not be misclassified as Albanian ("sq")
        assert_eq!(detect_language_by_chars("Joyeux Noël à tous et bonne année!", 0), None);
        assert_eq!(detect_language("Joyeux Noël à tous et bonne année!").await, "fr");
        assert_eq!(detect_language("Welkom in België, een prachtig land.").await, "nl");
    }

    #[tokio::test]
    async fn test_detect_urdu_and_georgian_and_cjk_ascii_mixed() {
        // Urdu with distinctive characters (ٹ, ڈ, ڑ, ے) should detect as "ur", not "ar"
        let urdu_sample = "یہ ایک اردو کی کتاب ہے جس میں معلومات ہیں۔";
        assert_eq!(detect_language(urdu_sample).await, "ur");

        // Georgian script (Mkhedruli) should detect as "ka"
        let georgian_sample = "გამარჯობა მსოფლიო, ეს არის ქართული ტექსტი.";
        assert_eq!(detect_language(georgian_sample).await, "ka");

        // CJK mixed with English brand/acronym should still be recognized by CJK heuristic
        let mixed_cjk = "我的 iPhone 真的很实用，推荐大家使用。";
        assert_eq!(detect_pure_cjk_language(mixed_cjk, 0), Some("zh-Hans"));
    }
}
