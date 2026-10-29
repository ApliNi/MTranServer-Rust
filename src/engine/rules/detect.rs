//! Four-tier zero-allocation unified language detection engine.
//!
//! Provides a single source of truth for language identification across both:
//! 1. Canonical tags: standard BCP-47 canonical tags (e.g. "zh-Hans", "zh-Hant", "ja", "en", "ru", "uk")
//! 2. Profile keys: zero-allocation profile keys mapped to RuleRegistry (57 profiles).
//!
//! Architecture:
//! - Tier 0: Fast bypass & noise filter (empty, CLI commands, HTTP logs, technical alerts)
//! - Tier 1: Single-pass Unicode script scan (ko, ja, el, he, th, ka, bn, ta, te, gu, kn, ml)
//! - Tier 2: Orthography single-character exclusive/distinctive markers (CJK Hanzi variants, Cyrillic, Devanagari, Arabic, Latin diacritics)
//! - Tier 3: Minimal essential lexical routing (CORE_GREETINGS, twin language terms: da vs nb vs nn vs sv, ms vs id, gl, eu, sq, af)
//! - Tier 4: Statistical 3-gram whatlang global fallback

use super::{CompiledDetectionModel, RuleRegistry};

#[inline]
pub fn is_traditional_char(c: char) -> bool {
    RuleRegistry::global().detection_model().is_zh_hant_char(c)
}

#[inline]
pub fn is_simplified_char(c: char) -> bool {
    RuleRegistry::global().detection_model().is_zh_hans_char(c)
}

/// Distinguish Traditional Chinese from Simplified Chinese.
pub fn distinguish_chinese_variant(text: &str) -> &'static str {
    let model = RuleRegistry::global().detection_model();
    distinguish_chinese_variant_with_model(text, &model)
}

/// Distinguish Traditional Chinese from Simplified Chinese using a precompiled detection model.
pub fn distinguish_chinese_variant_with_model(
    text: &str,
    model: &CompiledDetectionModel,
) -> &'static str {
    let mut trad_count = 0usize;
    let mut simp_count = 0usize;

    for c in text.chars() {
        if model.is_zh_hant_char(c) {
            trad_count += 1;
        } else if model.is_zh_hans_char(c) {
            simp_count += 1;
        }
    }

    if trad_count > simp_count {
        "zh-Hant"
    } else {
        "zh-Hans"
    }
}

/// Zero-heap-allocation case-insensitive ASCII comparison.
#[inline]
pub fn cmp_ignore_ascii_case(a: &str, b: &str) -> std::cmp::Ordering {
    let mut bytes_a = a.as_bytes().iter().map(|b| b.to_ascii_lowercase());
    let mut bytes_b = b.as_bytes().iter().map(|b| b.to_ascii_lowercase());
    loop {
        match (bytes_a.next(), bytes_b.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(ca), Some(cb)) => {
                let ord = ca.cmp(&cb);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

/// Zero-heap-allocation case-insensitive string comparison across full Unicode.
#[inline]
pub fn cmp_ignore_unicode_case(a: &str, b: &str) -> std::cmp::Ordering {
    if a.is_ascii() && b.is_ascii() {
        return cmp_ignore_ascii_case(a, b);
    }
    let mut a_chars = a.chars().flat_map(|c| c.to_lowercase());
    let mut b_chars = b.chars().flat_map(|c| c.to_lowercase());
    loop {
        match (a_chars.next(), b_chars.next()) {
            (None, None) => return std::cmp::Ordering::Equal,
            (None, Some(_)) => return std::cmp::Ordering::Less,
            (Some(_), None) => return std::cmp::Ordering::Greater,
            (Some(ca), Some(cb)) => {
                let ord = ca.cmp(&cb);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
        }
    }
}

/// Zero-heap-allocation case-insensitive string equality across full Unicode.
#[inline]
pub fn eq_ignore_unicode_case(a: &str, b: &str) -> bool {
    if a.len() == b.len() && a.eq_ignore_ascii_case(b) {
        return true;
    }
    cmp_ignore_unicode_case(a, b) == std::cmp::Ordering::Equal
}

/// Minimal compile-time read-only greetings table for ultra-short polite phrases.
/// Sorted lexicographically for consistent binary or linear lookup.
pub const CORE_GREETINGS: &[(&str, &str)] = &[
    ("arrivederci", "it"),
    ("auf wiedersehen", "de"),
    ("boa noite", "pt"),
    ("boa tarde", "pt"),
    ("bom dia", "pt"),
    ("bonjour", "fr"),
    ("bonne nuit", "fr"),
    ("buenas noches", "es"),
    ("buenas tardes", "es"),
    ("buenos dias", "es"),
    ("buenos días", "es"),
    ("buon appetito", "it"),
    ("buon pomeriggio", "it"),
    ("buona notte", "it"),
    ("buona sera", "it"),
    ("buongiorno", "it"),
    ("ciao", "it"),
    ("dank je wel", "nl"),
    ("danke", "de"),
    ("dobrý den", "cs"),
    ("dobrý deň", "sk"),
    ("god morgon", "sv"),
    ("goedemiddag", "nl"),
    ("goedemorgen", "nl"),
    ("goedenavond", "nl"),
    ("good afternoon", "en"),
    ("good evening", "en"),
    ("good morning", "en"),
    ("good night", "en"),
    ("grazie", "it"),
    ("grazie mille", "it"),
    ("guten abend", "de"),
    ("guten morgen", "de"),
    ("guten tag", "de"),
    ("hello", "en"),
    ("hola", "es"),
    ("merci", "fr"),
    ("merci beaucoup", "fr"),
    ("muchas gracias", "es"),
    ("muito obrigado", "pt"),
    ("obrigada", "pt"),
    ("obrigado", "pt"),
    ("por favor", "pt"),
    ("s'il vous plait", "fr"),
    ("s'il vous plaît", "fr"),
    ("see you later", "en"),
    ("tack sa mycket", "sv"),
    ("tack så mycket", "sv"),
    ("thank you", "en"),
    ("thanks", "en"),
    ("vielen dank", "de"),
    ("xin chao", "vi"),
    ("xin chào", "vi"),
    ("à bientôt", "fr"),
    ("早安", "zh-Hant"),
];

/// Match a short greeting phrase against `CORE_GREETINGS` with zero heap allocations.
#[inline]
pub fn lookup_greeting(text: &str) -> Option<&'static str> {
    if let Some(res) = RuleRegistry::global()
        .detection_model()
        .lookup_greeting(text)
    {
        return Some(res);
    }
    let trimmed = text.trim_matches(|c: char| {
        c.is_whitespace()
            || c.is_ascii_punctuation()
            || matches!(
                c,
                '«' | '»'
                    | '“'
                    | '”'
                    | '‘'
                    | '’'
                    | '¡'
                    | '¿'
                    | '…'
                    | '—'
                    | '–'
                    | '·'
                    | '•'
            )
    });
    if trimmed.is_empty() {
        return None;
    }
    for &(greeting, lang) in CORE_GREETINGS {
        if eq_ignore_unicode_case(trimmed, greeting) {
            return Some(lang);
        }
    }
    None
}

/// Find  as a whole word in  without allocating memory,
/// with Unicode-safe character boundary guarantees.
pub fn find_whole_word_safe(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() || haystack.len() < needle.len() {
        return false;
    }

    let mut start = 0;
    while start + needle.len() <= haystack.len() {
        if let Some(pos) = haystack[start..].find(needle) {
            let actual_pos = start + pos;
            let end_pos = actual_pos + needle.len();

            let before_ok = if actual_pos == 0 {
                true
            } else {
                haystack[..actual_pos]
                    .chars()
                    .next_back()
                    .map_or(true, |c| !c.is_alphanumeric() && c != '_')
            };

            let after_ok = if end_pos >= haystack.len() {
                true
            } else {
                haystack[end_pos..]
                    .chars()
                    .next()
                    .map_or(true, |c| !c.is_alphanumeric() && c != '_')
            };

            if before_ok && after_ok {
                return true;
            }

            // Advance by at least one char to avoid infinite loop
            if let Some(c) = haystack[actual_pos..].chars().next() {
                start = actual_pos + c.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    false
}

/// Statistics collected during Tier 1 single-pass scanning.
#[derive(Default)]
struct ScriptStats {
    han: usize,
    kana: usize,
    hangul: usize,
    cyrillic: usize,
    arabic: usize,
    devanagari: usize, // Note: excludes Danda (U+0964, U+0965)
    bengali: usize,
    tamil: usize,
    telugu: usize,
    gujarati: usize,
    kannada: usize,
    malayalam: usize,
    thai: usize,
    georgian: usize,
    hebrew: usize,
    greek: usize,
    latin: usize,

    // Specific orthography flags & sub-counts
    has_ja_unique_han: bool,
    trad_han: usize,
    simp_han: usize,

    // Cyrillic sub-flags
    has_uk_exclusive: bool,       // ї, є, ґ
    has_cyrillic_i: bool,         // і
    has_be: bool,                 // ў
    has_sr: bool,                 // ђ, ћ, ј, љ, њ, џ
    cyrillic_yer: usize,          // ъ
    cyrillic_ru_exclusive: usize, // ы, э, ё

    // Arabic sub-flags
    has_urdu: bool,
    has_persian: bool,

    // Devanagari sub-flags
    has_marathi_exclusive: bool, // ळ (U+0933)

    // Latin sub-flags
    vietnamese_diacritics: usize,
    has_icelandic: bool,
    has_azerbaijani: bool,
    has_czech: bool,
    has_slovak: bool,
    has_polish: bool,
    has_hungarian: bool,
    has_romanian: bool,
    romanian_vowels: usize,
    has_turkish: bool,
    has_german_eszett: bool,
    has_spanish_unique: bool,
    has_portuguese_tilde: bool,
    has_french_unique: bool,
    scand_ae_oe: usize,      // æ, ø
    scand_a_o_umlaut: usize, // ä, ö
}

/// Scan text and collect script and orthography statistics in a single pass.
fn scan_script_stats(text: &str, model: &CompiledDetectionModel) -> ScriptStats {
    let mut stats = ScriptStats::default();

    for c in text.chars() {
        let code = c as u32;

        // CJK
        if (0x4E00..=0x9FFF).contains(&code) || (0x3400..=0x4DBF).contains(&code) {
            stats.han += 1;
            if model.is_ja_unique_han(c) {
                stats.has_ja_unique_han = true;
            }
            if model.is_zh_hant_char(c) {
                stats.trad_han += 1;
            } else if model.is_zh_hans_char(c) {
                stats.simp_han += 1;
            }
            continue;
        }

        // Japanese Kana
        if (0x3040..=0x309F).contains(&code)
            || (0x30A0..=0x30FF).contains(&code)
            || (0x31F0..=0x31FF).contains(&code)
        {
            stats.kana += 1;
            continue;
        }

        // Korean Hangul
        if (0xAC00..=0xD7AF).contains(&code)
            || (0x1100..=0x11FF).contains(&code)
            || (0x3130..=0x318F).contains(&code)
        {
            stats.hangul += 1;
            continue;
        }

        // Cyrillic
        if (0x0400..=0x04FF).contains(&code) || (0x0500..=0x052F).contains(&code) {
            stats.cyrillic += 1;
            if model.cyrillic.is_uk_unique(c) {
                stats.has_uk_exclusive = true;
            } else if model.cyrillic.is_cyrillic_i(c) {
                stats.has_cyrillic_i = true;
            } else if model.cyrillic.is_be_unique(c) {
                stats.has_be = true;
            } else if model.cyrillic.is_sr_unique(c) {
                stats.has_sr = true;
            } else if model.cyrillic.is_bg_distinctive(c) {
                stats.cyrillic_yer += 1;
            } else if model.cyrillic.is_ru_distinctive(c) {
                stats.cyrillic_ru_exclusive += 1;
            }
            continue;
        }

        // Arabic
        if (0x0600..=0x06FF).contains(&code)
            || (0x0750..=0x077F).contains(&code)
            || (0x08A0..=0x08FF).contains(&code)
            || (0xFB50..=0xFDFF).contains(&code)
            || (0xFE70..=0xFEFF).contains(&code)
        {
            stats.arabic += 1;
            if model.arabic.is_ur_unique(c) {
                stats.has_urdu = true;
            } else if model.arabic.is_fa_unique(c) {
                stats.has_persian = true;
            }
            continue;
        }

        // Devanagari (excluding Danda U+0964 and Double Danda U+0965)
        if (0x0900..=0x0963).contains(&code) || (0x0966..=0x097F).contains(&code) {
            stats.devanagari += 1;
            if model.devanagari.is_mr_unique(c) {
                // Marathi 'ळ'
                stats.has_marathi_exclusive = true;
            }
            continue;
        }

        // South Asian scripts
        if (0x0980..=0x09FF).contains(&code) {
            stats.bengali += 1;
            continue;
        }
        if (0x0B80..=0x0BFF).contains(&code) {
            stats.tamil += 1;
            continue;
        }
        if (0x0C00..=0x0C7F).contains(&code) {
            stats.telugu += 1;
            continue;
        }
        if (0x0A80..=0x0AFF).contains(&code) {
            stats.gujarati += 1;
            continue;
        }
        if (0x0C80..=0x0CFF).contains(&code) {
            stats.kannada += 1;
            continue;
        }
        if (0x0D00..=0x0D7F).contains(&code) {
            stats.malayalam += 1;
            continue;
        }

        // Thai
        if (0x0E00..=0x0E7F).contains(&code) {
            stats.thai += 1;
            continue;
        }

        // Georgian
        if (0x10A0..=0x10FF).contains(&code) || (0x2D00..=0x2D2F).contains(&code) {
            stats.georgian += 1;
            continue;
        }

        // Hebrew
        if (0x0590..=0x05FF).contains(&code) {
            stats.hebrew += 1;
            continue;
        }

        // Greek
        if (0x0370..=0x03FF).contains(&code) || (0x1F00..=0x1FFF).contains(&code) {
            stats.greek += 1;
            continue;
        }

        // Latin letters & diacritics
        if c.is_ascii_alphabetic() {
            stats.latin += 1;
            continue;
        }

        // Extended Latin & IPA Extensions (0x00C0..=0x02AF includes 'ə' 0x0259)
        if (0x00C0..=0x02AF).contains(&code) || (0x1E00..=0x1EFF).contains(&code) {
            stats.latin += 1;

            if model.latin.is_vi_char(c) {
                stats.vietnamese_diacritics += 1;
            } else if model.latin.is_is_char(c) {
                stats.has_icelandic = true;
            } else if model.latin.is_az_char(c) {
                stats.has_azerbaijani = true;
            } else if model.latin.is_cs_char(c) {
                stats.has_czech = true;
            } else if model.latin.is_sk_char(c) {
                stats.has_slovak = true;
            } else if model.latin.is_pl_char(c) {
                stats.has_polish = true;
            } else if model.latin.is_hu_char(c) {
                stats.has_hungarian = true;
            } else if model.latin.is_ro_cedilla(c) {
                stats.has_romanian = true;
            } else if model.latin.is_ro_vowel(c) {
                stats.romanian_vowels += 1;
            } else if model.latin.is_tr_char(c) {
                stats.has_turkish = true;
            } else if model.latin.is_de_eszett(c) {
                stats.has_german_eszett = true;
            } else if model.latin.is_es_char(c) {
                stats.has_spanish_unique = true;
            } else if model.latin.is_pt_tilde(c) {
                stats.has_portuguese_tilde = true;
            } else if model.latin.is_fr_unique(c) {
                stats.has_french_unique = true;
            } else if model.latin.is_scand_ae_oe(c) {
                stats.scand_ae_oe += 1;
            } else if model.latin.is_scand_a_o_umlaut(c) {
                stats.scand_a_o_umlaut += 1;
            }
        }
    }

    stats
}

#[inline]
fn starts_with_ascii_ci(text: &str, prefix: &str) -> bool {
    text.len() >= prefix.len()
        && text.as_bytes()[..prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
}

#[inline]
fn contains_ascii_ci(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }
    let needle_bytes = needle.as_bytes();
    haystack
        .as_bytes()
        .windows(needle_bytes.len())
        .any(|w| w.eq_ignore_ascii_case(needle_bytes))
}

#[inline]
fn starts_with_any(text: &str, prefixes: &[&str]) -> bool {
    prefixes.iter().any(|&p| text.starts_with(p))
}

/// Tier 0: Fast bypass for technical logs, CLI commands, HTTP messages, code snippets, and common greetings.
fn tier0_fast_bypass(text: &str) -> Option<&'static str> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return Some("en");
    }

    // CLI Commands - fast dispatch by first word to avoid false positives and eliminate linear scanning
    if let Some((tool, rest)) = trimmed.split_once(' ') {
        let is_cli = match tool {
            "curl" => starts_with_any(rest, &["-", "http://", "https://", "\"http", "'http"]),
            "git" => starts_with_any(
                rest,
                &[
                    "checkout ", "commit ", "push ", "pull ", "status", "clone ", "add ",
                    "diff ", "log", "branch", "merge ", "rebase ", "reset ", "stash", "remote",
                    "init", "config ", "--",
                ],
            ),
            "docker" => starts_with_any(
                rest,
                &[
                    "run ", "build ", "compose ", "ps", "exec ", "images", "stop ", "rm ", "--",
                ],
            ),
            "npm" => starts_with_any(
                rest,
                &[
                    "install", "run ", "i ", "start", "test", "build", "publish", "init",
                    "update", "--",
                ],
            ),
            "pip" => starts_with_any(rest, &["install ", "uninstall ", "list", "freeze", "--"]),
            "cargo" => starts_with_any(
                rest,
                &[
                    "build", "test", "run", "check", "clippy", "update", "publish", "new ",
                    "init", "--",
                ],
            ),
            "sudo" => starts_with_any(
                rest,
                &[
                    "apt ", "apt-get ", "systemctl ", "docker ", "rm ", "chmod ", "chown ",
                    "mkdir ", "cp ", "mv ", "service ", "--",
                ],
            ),
            "kubectl" => starts_with_any(
                rest,
                &["get ", "apply ", "describe ", "delete ", "logs ", "exec ", "--"],
            ),
            "yarn" => starts_with_any(
                rest,
                &["add ", "install", "run ", "build", "start", "test", "--"],
            ),
            "pnpm" => starts_with_any(
                rest,
                &["add ", "install", "run ", "build", "start", "test", "--"],
            ),
            "systemctl" => starts_with_any(
                rest,
                &["start ", "stop ", "restart ", "status ", "enable ", "disable ", "--"],
            ),
            "tar" => starts_with_any(rest, &["-", "x", "c", "t", "z"]),
            "apt" => starts_with_any(rest, &["install ", "update", "upgrade"]),
            "apt-get" => starts_with_any(rest, &["install ", "update"]),
            _ => false,
        };
        if is_cli {
            return Some("en");
        }
    }

    // HTTP / Network / Log headers
    if trimmed.starts_with("GET ")
        || trimmed.starts_with("POST ")
        || trimmed.starts_with("PUT ")
        || trimmed.starts_with("DELETE ")
        || trimmed.starts_with("PATCH ")
        || trimmed.starts_with("HEAD ")
        || trimmed.starts_with("OPTIONS ")
        || trimmed.starts_with("### [ALERT]")
        || trimmed.starts_with("### [ERROR")
        || trimmed.starts_with("### [WARN")
        || trimmed.starts_with("### [INFO")
        || trimmed.contains("(CRITICAL FAILURE)")
        || trimmed.contains("status HTTP/1.1")
        || trimmed.contains("HTTP/1.1")
        || trimmed.contains("HTTP/2.0")
    {
        return Some("en");
    }

    // Common English greeting prefixes and exact phrases
    const GREETING_PREFIXES: &[&str] = &[
        "hello ",
        "hello,",
        "hello!",
        "hello?",
        "good morning ",
        "good morning,",
        "good morning!",
        "good afternoon ",
        "good afternoon,",
        "good afternoon!",
        "good evening ",
        "good evening,",
        "good evening!",
        "good night ",
        "good night,",
        "good night!",
        "thank you ",
        "thank you,",
        "thank you!",
        "thanks ",
        "thanks,",
        "thanks!",
        "welcome to ",
        "see you ",
    ];

    const EXACT_GREETINGS: &[&str] = &[
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "good night",
        "thank you",
        "thanks",
    ];

    for &exact in EXACT_GREETINGS {
        if trimmed.eq_ignore_ascii_case(exact) {
            return Some("en");
        }
    }

    for &prefix in GREETING_PREFIXES {
        if starts_with_ascii_ci(trimmed, prefix) {
            return Some("en");
        }
    }

    // Pure ASCII code / query / syntax structures
    if trimmed.is_ascii() {
        // SQL queries - require characteristic keywords
        if (starts_with_ascii_ci(trimmed, "select ")
            && (contains_ascii_ci(trimmed, " from ") || trimmed.ends_with(';')))
            || starts_with_ascii_ci(trimmed, "insert into ")
            || (starts_with_ascii_ci(trimmed, "update ") && contains_ascii_ci(trimmed, " set "))
            || starts_with_ascii_ci(trimmed, "delete from ")
            || starts_with_ascii_ci(trimmed, "create table ")
            || starts_with_ascii_ci(trimmed, "drop table ")
            || starts_with_ascii_ci(trimmed, "alter table ")
            || starts_with_ascii_ci(trimmed, "truncate table ")
        {
            return Some("en");
        }

        // Programming language declarations & statements
        // Verified against code syntax delimiters to prevent collision with natural languages (e.g., Danish 'let', 'var', French 'import')
        if (trimmed.starts_with("fn ")
            && (trimmed.contains('(') || trimmed.contains("->") || trimmed.contains('{')))
            || trimmed.starts_with("pub fn ")
            || (trimmed.starts_with("def ") && trimmed.contains('(') && trimmed.contains(':'))
            || (trimmed.starts_with("function ") && trimmed.contains('('))
            || ((trimmed.starts_with("const ")
                || trimmed.starts_with("let ")
                || trimmed.starts_with("var "))
                && (trimmed.contains('=') || trimmed.contains(':') || trimmed.ends_with(';')))
            || (trimmed.starts_with("import ")
                && (trimmed.contains("from ")
                    || trimmed.contains('{')
                    || trimmed.contains('*')
                    || trimmed.ends_with(';')
                    || trimmed.starts_with("import static ")
                    || trimmed.starts_with("import java.")
                    || trimmed.starts_with("import std.")))
            || (trimmed.starts_with("export ")
                && (trimmed.starts_with("export default ")
                    || trimmed.starts_with("export const ")
                    || trimmed.starts_with("export function ")
                    || trimmed.starts_with("export class ")
                    || trimmed.starts_with("export {")))
            || (trimmed.starts_with("class ")
                && (trimmed.contains('{')
                    || trimmed.contains(':')
                    || trimmed.contains("extends ")
                    || trimmed.contains("implements ")))
            || trimmed.starts_with("public class ")
            || trimmed.starts_with("public static void ")
            || (trimmed.starts_with("package ") && trimmed.ends_with(';'))
            || (trimmed.starts_with("return ") && (trimmed.ends_with(';') || trimmed.contains(';')))
            || trimmed.starts_with("#include <")
            || trimmed.starts_with("console.log(")
            || trimmed.starts_with("println!(")
        {
            return Some("en");
        }

        // JSON / XML / HTML structure
        if (trimmed.starts_with('{') && (trimmed.contains("\":") || trimmed.contains("\": ")))
            || trimmed.starts_with("[{")
            || trimmed.starts_with("<?xml")
            || trimmed.starts_with("<!DOCTYPE")
            || starts_with_ascii_ci(trimmed, "<html")
            || starts_with_ascii_ci(trimmed, "<script")
            || starts_with_ascii_ci(trimmed, "<style")
            || starts_with_ascii_ci(trimmed, "<div")
        {
            return Some("en");
        }
    }

    None
}

/// Core single-pass unified language detection engine with precompiled model and confidence.
pub fn detect_text_language_with_model_and_confidence(
    text: &str,
    model: &CompiledDetectionModel,
) -> (&'static str, f32) {
    // -------------------------------------------------------------
    // Tier 0: Fast bypass & noise filter
    // -------------------------------------------------------------
    if let Some(bypass) = tier0_fast_bypass(text) {
        return (bypass, 1.0);
    }

    let trimmed = text.trim();
    if trimmed.is_empty() {
        return ("en", 0.0);
    }

    // -------------------------------------------------------------
    // Tier 1: Single-pass O(N) script & orthography scanning
    // -------------------------------------------------------------
    let stats = scan_script_stats(trimmed, model);

    // -------------------------------------------------------------
    // -------------------------------------------------------------
    // Tier 1: Single-pass Unicode script scan
    // South Asian scripts: unambiguous Unicode script blocks
    if stats.bengali > 0 && stats.bengali >= stats.devanagari {
        return ("bn", 1.0);
    }
    if stats.tamil > 0 {
        return ("ta", 1.0);
    }
    if stats.telugu > 0 {
        return ("te", 1.0);
    }
    if stats.gujarati > 0 {
        return ("gu", 1.0);
    }
    if stats.kannada > 0 {
        return ("kn", 1.0);
    }
    if stats.malayalam > 0 {
        return ("ml", 1.0);
    }
    if stats.thai > 0 {
        return ("th", 1.0);
    }
    if stats.georgian > 0 {
        return ("ka", 1.0);
    }
    if stats.hebrew > 0 {
        return ("he", 1.0);
    }
    // Greek script: ensure not just 1-2 math symbols in an English sentence
    if (stats.greek >= 2 && stats.greek > stats.latin) || (stats.greek > 0 && stats.latin == 0) {
        return ("el", 1.0);
    }

    // Japanese Kana: any Kana immediately confirms Japanese
    if stats.kana > 0 {
        return ("ja", 1.0);
    }

    // Korean Hangul: Hangul present and dominates Hanzi
    if stats.hangul > 0 && stats.hangul >= stats.han {
        return ("ko", 1.0);
    }
    // -------------------------------------------------------------
    // Tier 2: Orthography single-character exclusive / distinctive features
    // -------------------------------------------------------------

    // (A) CJK Hanzi disambiguation (Japanese Kanji vs Simplified/Traditional Chinese)
    let cjk_chars = stats.han + stats.kana + stats.hangul;
    let w_cjk = cjk_chars * 3;

    if stats.han > 0 && (w_cjk >= stats.latin / 2 || stats.latin == 0) {
        // (1) Japanese unique Han character (Kokuji / Shinjitai) hit
        if stats.has_ja_unique_han {
            return ("ja", 1.0);
        }
        // (2) High-frequency Japanese Kanji compound terms
        for term in &model.ja_terms {
            if trimmed.contains(term.as_str()) {
                return ("ja", 1.0);
            }
        }
        // (3) Chinese variant classification
        if stats.trad_han > stats.simp_han {
            return ("zh-Hant", 1.0);
        }
        if stats.simp_han > stats.trad_han {
            return ("zh-Hans", 1.0);
        }
        if let Some(lang) = lookup_greeting(trimmed) {
            if matches!(lang, "zh-Hant" | "zh-Hans" | "ja") {
                return (lang, 1.0);
            }
        }
        return ("zh-Hans", 1.0);
    }
    if stats.cyrillic > 0
        && (stats.latin == 0 || stats.cyrillic >= stats.latin || stats.cyrillic * 2 >= stats.latin)
    {
        // Serbian
        if stats.has_sr
            || model
                .cyrillic
                .sr_terms
                .iter()
                .any(|t| trimmed.contains(t.as_str()))
        {
            return ("sr", 1.0);
        }

        // Belarusian: exclusive 'ў', or has 'і' + 'ы' (Ukrainian has no 'ы'), or keywords
        if stats.has_be
            || (stats.has_cyrillic_i && stats.cyrillic_ru_exclusive > 0 && !stats.has_uk_exclusive)
            || model
                .cyrillic
                .be_terms
                .iter()
                .any(|t| trimmed.contains(t.as_str()))
        {
            return ("be", 1.0);
        }

        // Ukrainian: exclusive 'ї', 'є', 'ґ', or has 'і' without Russian/Belarusian exclusive letters
        if stats.has_uk_exclusive
            || (stats.has_cyrillic_i && stats.cyrillic_ru_exclusive == 0)
            || model
                .cyrillic
                .uk_terms
                .iter()
                .any(|t| trimmed.contains(t.as_str()))
        {
            return ("uk", 1.0);
        }

        // Bulgarian: 'ъ' without Russian-exclusive letters, or Bulgarian keywords
        if (stats.cyrillic_yer > 0 && stats.cyrillic_ru_exclusive == 0)
            || model
                .cyrillic
                .bg_terms
                .iter()
                .any(|t| trimmed.contains(t.as_str()))
        {
            return ("bg", 1.0);
        }

        // Russian
        if stats.cyrillic_ru_exclusive > 0 {
            return ("ru", 1.0);
        }

        return ("ru", 1.0);
    }

    // (C) Devanagari family
    if stats.devanagari > 0 {
        if stats.has_marathi_exclusive
            || model
                .devanagari
                .mr_terms
                .iter()
                .any(|t| trimmed.contains(t.as_str()))
            || model
                .devanagari
                .mr_words
                .iter()
                .any(|w| find_whole_word_safe(trimmed, w))
        {
            return ("mr", 1.0);
        }
        return ("hi", 1.0);
    }

    // (D) Arabic family
    if stats.arabic > 0 {
        if stats.has_urdu {
            return ("ur", 1.0);
        }
        if stats.has_persian {
            return ("fa", 1.0);
        }
        return ("ar", 1.0);
    }

    // (E) Latin distinctive orthography
    if stats.vietnamese_diacritics > 0 {
        return ("vi", 1.0);
    }
    if stats.has_icelandic {
        return ("is", 1.0);
    }
    if stats.has_azerbaijani {
        return ("az", 1.0);
    }
    if stats.has_czech {
        return ("cs", 1.0);
    }
    if stats.has_slovak {
        return ("sk", 1.0);
    }
    if stats.has_polish {
        return ("pl", 1.0);
    }
    if stats.has_hungarian {
        return ("hu", 1.0);
    }
    if stats.has_romanian
        || (stats.romanian_vowels >= 2
            && model
                .latin
                .ro_words
                .iter()
                .any(|w| find_whole_word_safe(trimmed, w.as_str())))
    {
        return ("ro", 1.0);
    }
    if stats.has_turkish {
        return ("tr", 1.0);
    }
    if stats.has_french_unique {
        return ("fr", 1.0);
    }
    if stats.has_german_eszett {
        return ("de", 1.0);
    }
    if model.latin.es_punct.iter().any(|&c| trimmed.contains(c)) || stats.has_spanish_unique {
        return ("es", 1.0);
    }
    if stats.has_portuguese_tilde
        && model
            .latin
            .pt_terms
            .iter()
            .any(|t| trimmed.contains(t.as_str()))
    {
        return ("pt", 1.0);
    }

    // -------------------------------------------------------------
    // Tier 3: Minimal essential lexical routing
    // -------------------------------------------------------------

    // (A) High-frequency greetings fast path (zero heap allocation)
    if let Some(lang) = lookup_greeting(trimmed) {
        return (lang, 1.0);
    }

    // (B) Twin / near-relative language lexical discriminators
    if let Some(lang) = match_latin_lexical(trimmed, &model.latin) {
        return (lang, 1.0);
    }
    // Scandinavian disambiguation
    if stats.scand_ae_oe > 0 {
        if model
            .latin
            .scand_nb_terms
            .iter()
            .any(|t| trimmed.contains(t.as_str()))
        {
            return ("nb", 1.0);
        }
        if model
            .latin
            .scand_da_terms
            .iter()
            .any(|t| trimmed.contains(t.as_str()))
        {
            return ("da", 1.0);
        }
        return ("da", 1.0);
    }

    if stats.scand_a_o_umlaut > 0
        && (model
            .latin
            .scand_sv_terms
            .iter()
            .any(|t| trimmed.contains(t.as_str()))
            || model
                .latin
                .scand_sv_words
                .iter()
                .any(|w| find_whole_word_safe(trimmed, w)))
    {
        return ("sv", 1.0);
    }

    // -------------------------------------------------------------
    // Tier 4: Statistical 3-gram fallback (Whatlang)
    // -------------------------------------------------------------
    const MAX_WHATLANG_CHARS: usize = 128;
    let sample = match trimmed.char_indices().nth(MAX_WHATLANG_CHARS) {
        Some((byte_idx, _)) => &trimmed[..byte_idx],
        None => trimmed,
    };
    if let Some(info) = whatlang::detect(sample) {
        let code = whatlang_code_to_bcp47(info.lang().code());
        let mapped = map_detected_code_to_static(code);
        let conf = info.confidence() as f32;
        let min_conf = if sample.is_ascii() && !info.is_reliable() {
            0.25
        } else {
            0.1
        };
        if info.is_reliable() || conf >= min_conf {
            return (mapped, conf.max(0.6));
        }
    }

    ("en", 0.0)
}

/// Core single-pass unified language detection engine with confidence.
///
/// Guaranteed zero heap allocations on synchronous detection paths.
pub fn detect_text_language_with_confidence(text: &str) -> (&'static str, f32) {
    let model = RuleRegistry::global().detection_model();
    detect_text_language_with_model_and_confidence(text, &model)
}

/// Core single-pass unified language detection engine.
///
/// Guaranteed zero heap allocations on synchronous detection paths.
#[inline]
pub fn detect_text_language(text: &str) -> &'static str {
    detect_text_language_with_confidence(text).0
}

/// Detect dominant language mapped to RuleRegistry profile keys (57 profiles).
///
/// Guaranteed zero heap allocation.
#[inline]
pub fn detect_dominant(text: &str) -> &'static str {
    let lang = detect_text_language(text);
    match lang {
        "zh-Hans" | "zh-Hant" | "zh" => "zh",
        other => other,
    }
}

#[inline]
pub fn detect_dominant_language(text: &str) -> &'static str {
    detect_dominant(text)
}
fn whatlang_code_to_bcp47(code: &str) -> &'static str {
    match code {
        "eng" => "en",
        "rus" => "ru",
        "cmn" => "zh-Hans",
        "spa" => "es",
        "por" => "pt",
        "ita" => "it",
        "ben" => "bn",
        "fra" => "fr",
        "deu" => "de",
        "ukr" => "uk",
        "kat" => "ka",
        "ara" => "ar",
        "hin" => "hi",
        "jpn" => "ja",
        "heb" => "he",
        "yid" => "yi",
        "pol" => "pl",
        "kor" => "ko",
        "nob" => "nb",
        "dan" => "da",
        "swe" => "sv",
        "fin" => "fi",
        "tur" => "tr",
        "nld" => "nl",
        "hun" => "hu",
        "ces" => "cs",
        "ell" => "el",
        "bul" => "bg",
        "bel" => "be",
        "mar" => "mr",
        "kan" => "kn",
        "ron" => "ro",
        "slv" => "sl",
        "hrv" => "hr",
        "srp" => "sr",
        "mkd" => "mk",
        "lit" => "lt",
        "lav" => "lv",
        "est" => "et",
        "tam" => "ta",
        "vie" => "vi",
        "urd" => "ur",
        "tha" => "th",
        "guj" => "gu",
        "aze" => "az",
        "ind" => "id",
        "tel" => "te",
        "pes" => "fa",
        "mal" => "ml",
        "slk" => "sk",
        "cat" => "ca",
        "glg" => "gl",
        "eus" => "eu",
        "sqi" => "sq",
        "isl" => "is",
        "afr" => "af",
        "nno" => "nn",
        "bos" => "bs",
        _ => "en",
    }
}

/// Map detected language code to static slice corresponding to the 57 supported language profiles.
pub fn map_detected_code_to_static(code: &str) -> &'static str {
    match code {
        "en" => "en",
        "zh" | "zh-Hans" | "zh-hans" | "zh-CN" | "zh_CN" | "zh_hans" => "zh-Hans",
        "zh-Hant" | "zh-hant" | "zh-TW" | "zh-HK" | "zh_TW" | "zh_HK" | "zh_hant" => "zh-Hant",
        "ja" => "ja",
        "ko" => "ko",
        "de" => "de",
        "fr" => "fr",
        "es" => "es",
        "ru" => "ru",
        "it" => "it",
        "pt" => "pt",
        "nl" => "nl",
        "ar" => "ar",
        "he" => "he",
        "el" => "el",
        "th" => "th",
        "hi" => "hi",
        "tr" => "tr",
        "vi" => "vi",
        "pl" => "pl",
        "cs" => "cs",
        "uk" => "uk",
        "be" => "be",
        "bg" => "bg",
        "sr" => "sr",
        "sk" => "sk",
        "ro" => "ro",
        "sv" => "sv",
        "da" => "da",
        "nb" => "nb",
        "fi" => "fi",
        "hu" => "hu",
        "et" => "et",
        "ca" => "ca",
        "gl" => "gl",
        "fa" => "fa",
        "id" => "id",
        "bn" => "bn",
        "gu" => "gu",
        "kn" => "kn",
        "ml" => "ml",
        "mr" => "mr",
        "ta" => "ta",
        "te" => "te",
        "ur" => "ur",
        "is" => "is",
        "sq" => "sq",
        "ms" => "ms",
        "bs" => "bs",
        "az" => "az",
        "af" => "af",
        "nn" => "nn",
        "hr" => "hr",
        "sl" => "sl",
        "lt" => "lt",
        "lv" => "lv",
        "eu" => "eu",
        "ka" => "ka",
        _ => "en",
    }
}
#[inline]
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    if haystack.len() < needle.len() {
        return false;
    }
    let n_bytes = needle.as_bytes();
    let first = n_bytes[0].to_ascii_lowercase();
    let h_bytes = haystack.as_bytes();
    let max_idx = h_bytes.len() - n_bytes.len();
    for i in 0..=max_idx {
        if h_bytes[i].to_ascii_lowercase() == first
            && h_bytes[i..i + n_bytes.len()].eq_ignore_ascii_case(n_bytes)
        {
            return true;
        }
    }
    false
}

#[inline]
fn match_latin_lexical(
    text: &str,
    latin: &super::detection_model::CompiledLatinDetection,
) -> Option<&'static str> {
    let mut best: Option<(usize, &'static str)> = None;

    let mut check_tok = |tok: &str| -> bool {
        let mut buf = [0u8; 32];
        let search_res = if tok.len() <= 32 && tok.is_ascii() {
            let b = &mut buf[..tok.len()];
            b.copy_from_slice(tok.as_bytes());
            b.make_ascii_lowercase();
            let lower_tok = unsafe { std::str::from_utf8_unchecked(b) };
            latin
                .latin_lexical_words
                .binary_search_by(|(w, _, _)| w.as_str().cmp(lower_tok))
        } else {
            latin
                .latin_lexical_words
                .binary_search_by(|(w, _, _)| cmp_ignore_unicode_case(w.as_str(), tok))
        };
        if let Ok(idx) = search_res {
            let (_, lang, prio) = latin.latin_lexical_words[idx];
            if prio == 0 {
                best = Some((0, lang));
                return true;
            }
            if best.map_or(true, |(p, _)| prio < p) {
                best = Some((prio, lang));
            }
        }
        false
    };

    let mut word_start = None;
    for (i, c) in text.char_indices() {
        if c.is_alphanumeric() {
            if word_start.is_none() {
                word_start = Some(i);
            }
        } else if let Some(start) = word_start {
            let tok = &text[start..i];
            word_start = None;
            if check_tok(tok) {
                return best.map(|(_, lang)| lang);
            }
        }
    }
    if let Some(start) = word_start {
        let tok = &text[start..];
        if check_tok(tok) {
            return best.map(|(_, lang)| lang);
        }
    }

    for (phrase, lang, prio) in latin.latin_lexical_phrases.iter() {
        if best.map_or(true, |(p, _)| *prio < p)
            && (text.contains(phrase.as_str()) || contains_ignore_ascii_case(text, phrase.as_str()))
        {
            if *prio == 0 {
                return Some(lang);
            }
            best = Some((*prio, lang));
        }
    }

    best.map(|(_, lang)| lang)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detection_smoke() {
        assert_eq!(detect_text_language(""), "en");
        assert_eq!(detect_text_language("Hello world"), "en");
        assert_eq!(detect_text_language("こんにちは世界"), "ja");
        assert_eq!(detect_text_language("你好世界"), "zh-Hans");
    }

    #[test]
    fn test_tier0_cli_dispatch() {
        let cli_samples = [
            "curl -v https://example.com",
            "git status",
            "git checkout main",
            "docker run -it ubuntu",
            "npm install express",
            "pip install requests",
            "cargo build --release",
            "sudo apt update",
            "kubectl get pods",
            "yarn add lodash",
            "pnpm test",
            "systemctl restart nginx",
            "tar -xzvf archive.tar.gz",
            "apt upgrade",
            "apt-get install curl",
        ];
        for cmd in cli_samples {
            assert_eq!(tier0_fast_bypass(cmd), Some("en"), "Failed for CLI: {cmd}");
        }

        let non_cli = [
            "cargo",
            "git",
            "docker",
            "tar",
            "cargo un paquete en el auto",
            "git git git",
            "curl",
            "kubectl",
            "sudo",
        ];
        for text in non_cli {
            assert_ne!(tier0_fast_bypass(text), Some("en"), "False positive for: {text}");
        }
    }
}
