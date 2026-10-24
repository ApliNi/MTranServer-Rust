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

/// Compact stack-allocated language tag (up to 15 UTF-8 bytes) without heap allocation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct InlineLang {
    bytes: [u8; 15],
    len: u8,
}

impl InlineLang {
    #[inline]
    pub const fn empty() -> Self {
        Self {
            bytes: [0; 15],
            len: 0,
        }
    }

    #[inline]
    pub fn new(s: &str) -> Self {
        let mut len = s.len().min(15);
        while !s.is_char_boundary(len) {
            len -= 1;
        }
        let b = s.as_bytes();
        let mut bytes = [0u8; 15];
        bytes[..len].copy_from_slice(&b[..len]);
        Self {
            bytes,
            len: len as u8,
        }
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(&self.bytes[..self.len as usize]) }
    }
}

impl std::ops::Deref for InlineLang {
    type Target = str;
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_str()
    }
}

impl AsRef<str> for InlineLang {
    #[inline]
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Display for InlineLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::fmt::Debug for InlineLang {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.as_str())
    }
}

impl From<&str> for InlineLang {
    #[inline]
    fn from(s: &str) -> Self {
        Self::new(s)
    }
}

impl From<&String> for InlineLang {
    #[inline]
    fn from(s: &String) -> Self {
        Self::new(s.as_str())
    }
}

impl From<InlineLang> for String {
    #[inline]
    fn from(lang: InlineLang) -> Self {
        lang.as_str().to_string()
    }
}

impl PartialEq<str> for InlineLang {
    #[inline]
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}

impl PartialEq<&str> for InlineLang {
    #[inline]
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[derive(Clone, Copy, Debug)]
struct CachedDetection {
    language: InlineLang,
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
        return cached.language.to_string();
    }

    let result = detect_language_with_max_bytes_uncached(text, max_bytes).await;
    detect_cache_put(
        cache_key,
        CachedDetection {
            language: InlineLang::new(&result),
            confidence: 1.0,
        },
    );
    result
}

fn detect_cache_get(key: u64) -> Option<CachedDetection> {
    let shard = &mut *detect_cache_shard(key).lock();
    shard.as_mut()?.get(&key).copied()
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
    let sample_limit = max_bytes.max(MAX_HEURISTIC_SAMPLE_BYTES);
    let bounded = truncate_by_utf8_bytes(text, sample_limit * 2);
    let text = sanitize_input(bounded);
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

    let sample = truncate_by_utf8_bytes(safe_slice, sample_limit);
    let (code, _) = crate::engine::rules::detect::detect_text_language_with_confidence(sample);
    code.to_string()
}

/// Detect language with confidence score.
pub async fn detect_language_with_confidence(text: &str, min_confidence: f32) -> (String, f32) {
    if text.is_empty() {
        return (String::new(), 0.0);
    }
    let bounded = truncate_by_utf8_bytes(text, MAX_DETECTION_BYTES * 2);
    let text = sanitize_input(bounded);
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
        return (cached.language.to_string(), cached.confidence);
    }

    let (lang, conf) = detect_language_from_uncached(text, max_bytes).await;
    detect_cache_put(
        cache_key,
        CachedDetection {
            language: InlineLang::new(&lang),
            confidence: conf,
        },
    );
    (lang, conf)
}

async fn detect_language_from_uncached(text: &str, max_bytes: usize) -> (String, f32) {
    let limit = max_bytes.max(MAX_HEURISTIC_SAMPLE_BYTES);
    let bounded = truncate_by_utf8_bytes(text, limit);
    let sanitized = sanitize_input(bounded);
    let (code, conf) = crate::engine::rules::detect::detect_text_language_with_confidence(&sanitized);
    (code.to_string(), conf)
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
        .unwrap_or(text.len())
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
        assert_eq!(detect_language("The angle θ is measured in radians.").await, "en");
        assert_eq!(detect_dominant_language("The learning rate α is set to 0.001."), "en");

        // French and Dutch words with 'ë' (0x00EB) must not be misclassified as Albanian ("sq")
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

    #[test]
    fn test_inline_lang_utf8_boundary() {
        let short = InlineLang::new("en");
        assert_eq!(short.as_str(), "en");

        let exact_15 = InlineLang::new("123456789012345");
        assert_eq!(exact_15.as_str(), "123456789012345");

        // Truncation on ASCII
        let long_ascii = InlineLang::new("12345678901234567890");
        assert_eq!(long_ascii.as_str(), "123456789012345");

        // Multi-byte character crossing 15-byte boundary:
        // 14 ASCII bytes + 3-byte Chinese char '中' (0xE4, 0xB8, 0xAD).
        // The character boundary at <= 15 is 14 bytes. It must truncate cleanly to 14 bytes without UB.
        let boundary_cross = InlineLang::new("abcdefghijklmn中");
        assert_eq!(boundary_cross.as_str(), "abcdefghijklmn");
    }

    #[tokio::test]
    async fn test_pure_punctuation_fast_path() {
        // Pure punctuation / symbols should return text.len() to trigger fast path
        assert_eq!(get_valid_content_start_index("---===---"), 9);
        assert_eq!(get_valid_content_start_index("!@#$%^&*()_+"), 12);
        assert_eq!(get_valid_content_start_index("   "), 3);
        assert_eq!(get_valid_content_start_index(""), 0);

        // Leading symbols with content should return index of first alphanumeric character
        assert_eq!(get_valid_content_start_index("---Hello"), 3);
        assert_eq!(get_valid_content_start_index("...123"), 3);
        assert_eq!(get_valid_content_start_index("你好世界"), 0);

        // Fast path detection returns default "en" without running full heavy detectors
        assert_eq!(detect_language("---===---").await, "en");
        assert_eq!(detect_language("!@#$%^&*()_+").await, "en");
        let (lang, conf) = detect_language_with_confidence("......", 0.5).await;
        assert_eq!(lang, "en");
        assert_eq!(conf, 0.0);
    }
}
