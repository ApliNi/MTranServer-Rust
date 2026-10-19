use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;

/// Language alias table: lowercase input code -> canonical code.
/// Purely static content, built once on first use.
static ALIASES: LazyLock<HashMap<&'static str, &'static str>> = LazyLock::new(|| {
    let mut m = HashMap::with_capacity(64);
    m.insert("zh", "zh-Hans");
    m.insert("zh-cn", "zh-Hans");
    m.insert("zh-sg", "zh-Hans");
    m.insert("zh-hans", "zh-Hans");
    m.insert("cmn", "zh-Hans");
    m.insert("chinese", "zh-Hans");
    m.insert("zh-tw", "zh-Hant");
    m.insert("zh-hk", "zh-Hant");
    m.insert("zh-mo", "zh-Hant");
    m.insert("zh-hant", "zh-Hant");
    m.insert("cht", "zh-Hant");
    m.insert("en-us", "en");
    m.insert("en-gb", "en");
    m.insert("en-au", "en");
    m.insert("en-ca", "en");
    m.insert("en-nz", "en");
    m.insert("en-ie", "en");
    m.insert("en-za", "en");
    m.insert("en-jm", "en");
    m.insert("en-bz", "en");
    m.insert("en-tt", "en");
    m.insert("fr-fr", "fr");
    m.insert("fr-ca", "fr");
    m.insert("fr-be", "fr");
    m.insert("fr-ch", "fr");
    m.insert("es-es", "es");
    m.insert("es-mx", "es");
    m.insert("es-ar", "es");
    m.insert("es-co", "es");
    m.insert("es-cl", "es");
    m.insert("es-pe", "es");
    m.insert("es-ve", "es");
    m.insert("pt-pt", "pt");
    m.insert("pt-br", "pt");
    m.insert("de-de", "de");
    m.insert("de-at", "de");
    m.insert("de-ch", "de");
    m.insert("it-it", "it");
    m.insert("it-ch", "it");
    m.insert("ja-jp", "ja");
    m.insert("jp", "ja");
    m.insert("ko-kr", "ko");
    m.insert("kr", "ko");
    m.insert("ru-ru", "ru");
    m.insert("no", "nb");
    m.insert("iw", "he");
    m.insert("in", "id");
    m.insert("jw", "jv");
    // ISO 639-2 / 639-3 3-letter codes to canonical BCP-47
    m.insert("eng", "en");
    m.insert("rus", "ru");
    m.insert("deu", "de");
    m.insert("ger", "de");
    m.insert("fra", "fr");
    m.insert("fre", "fr");
    m.insert("spa", "es");
    m.insert("por", "pt");
    m.insert("ita", "it");
    m.insert("nld", "nl");
    m.insert("dut", "nl");
    m.insert("pol", "pl");
    m.insert("tur", "tr");
    m.insert("vie", "vi");
    m.insert("ara", "ar");
    m.insert("kor", "ko");
    m.insert("jpn", "ja");
    m.insert("hin", "hi");
    m.insert("tha", "th");
    m.insert("ell", "el");
    m.insert("gre", "el");
    m.insert("heb", "he");
    m.insert("ukr", "uk");
    m.insert("ron", "ro");
    m.insert("rum", "ro");
    m.insert("ces", "cs");
    m.insert("cze", "cs");
    m.insert("swe", "sv");
    m.insert("hun", "hu");
    m.insert("fin", "fi");
    m.insert("dan", "da");
    m.insert("bul", "bg");
    m.insert("nob", "nb");
    m.insert("nor", "nb");
    m.insert("hrv", "hr");
    m.insert("srp", "sr");
    m.insert("slk", "sk");
    m.insert("slo", "sk");
    m.insert("slv", "sl");
    m.insert("lit", "lt");
    m.insert("lav", "lv");
    m.insert("est", "et");
    m.insert("cat", "ca");
    m.insert("glg", "gl");
    m.insert("eus", "eu");
    m.insert("baq", "eu");
    m.insert("fas", "fa");
    m.insert("per", "fa");
    m.insert("pes", "fa");
    m.insert("ben", "bn");
    m.insert("guj", "gu");
    m.insert("kan", "kn");
    m.insert("mal", "ml");
    m.insert("mar", "mr");
    m.insert("tam", "ta");
    m.insert("tel", "te");
    m.insert("urd", "ur");
    m.insert("isl", "is");
    m.insert("ice", "is");
    m.insert("sqi", "sq");
    m.insert("alb", "sq");
    m.insert("msa", "ms");
    m.insert("may", "ms");
    m.insert("bel", "be");
    m.insert("bos", "bs");
    m.insert("aze", "az");
    m.insert("afr", "af");
    m.insert("nno", "nn");
    m.insert("ind", "id");
    m.insert("kat", "ka");
    m.insert("geo", "ka");
    m
});

/// Normalize a language code, mirroring the original NormalizeLanguageCode.
pub fn normalize_language_code(code: &str) -> String {
    normalize_language_code_cow(code).into_owned()
}

/// Normalize a language code, borrowing when no allocation is needed.
pub fn normalize_language_code_cow<'a>(code: &'a str) -> Cow<'a, str> {
    if code.is_empty() {
        return Cow::Borrowed("");
    }

    // Fast-path for the most common pure lowercase ASCII language codes.
    match code {
        "en" => return Cow::Borrowed("en"),
        "zh" => return Cow::Borrowed("zh-Hans"),
        "ja" => return Cow::Borrowed("ja"),
        "es" => return Cow::Borrowed("es"),
        "fr" => return Cow::Borrowed("fr"),
        "de" => return Cow::Borrowed("de"),
        "ru" => return Cow::Borrowed("ru"),
        "it" => return Cow::Borrowed("it"),
        "pt" => return Cow::Borrowed("pt"),
        "ko" => return Cow::Borrowed("ko"),
        "ar" => return Cow::Borrowed("ar"),
        "tr" => return Cow::Borrowed("tr"),
        "vi" => return Cow::Borrowed("vi"),
        "pl" => return Cow::Borrowed("pl"),
        "nl" => return Cow::Borrowed("nl"),
        "el" => return Cow::Borrowed("el"),
        "th" => return Cow::Borrowed("th"),
        "hi" => return Cow::Borrowed("hi"),
        "id" => return Cow::Borrowed("id"),
        "uk" => return Cow::Borrowed("uk"),
        "ro" => return Cow::Borrowed("ro"),
        "cs" => return Cow::Borrowed("cs"),
        "sv" => return Cow::Borrowed("sv"),
        "hu" => return Cow::Borrowed("hu"),
        "fi" => return Cow::Borrowed("fi"),
        "da" => return Cow::Borrowed("da"),
        "bg" => return Cow::Borrowed("bg"),
        "he" => return Cow::Borrowed("he"),
        "nb" => return Cow::Borrowed("nb"),
        "hr" => return Cow::Borrowed("hr"),
        "sr" => return Cow::Borrowed("sr"),
        "sk" => return Cow::Borrowed("sk"),
        "sl" => return Cow::Borrowed("sl"),
        "lt" => return Cow::Borrowed("lt"),
        "lv" => return Cow::Borrowed("lv"),
        "et" => return Cow::Borrowed("et"),
        "ca" => return Cow::Borrowed("ca"),
        "gl" => return Cow::Borrowed("gl"),
        "eu" => return Cow::Borrowed("eu"),
        "fa" => return Cow::Borrowed("fa"),
        "bn" => return Cow::Borrowed("bn"),
        "gu" => return Cow::Borrowed("gu"),
        "kn" => return Cow::Borrowed("kn"),
        "ml" => return Cow::Borrowed("ml"),
        "mr" => return Cow::Borrowed("mr"),
        "ta" => return Cow::Borrowed("ta"),
        "te" => return Cow::Borrowed("te"),
        "ur" => return Cow::Borrowed("ur"),
        "is" => return Cow::Borrowed("is"),
        "sq" => return Cow::Borrowed("sq"),
        "ms" => return Cow::Borrowed("ms"),
        "be" => return Cow::Borrowed("be"),
        "bs" => return Cow::Borrowed("bs"),
        "az" => return Cow::Borrowed("az"),
        "af" => return Cow::Borrowed("af"),
        "nn" => return Cow::Borrowed("nn"),
        "ka" => return Cow::Borrowed("ka"),
        "zh-Hans" | "zh-hans" => return Cow::Borrowed("zh-Hans"),
        "zh-Hant" | "zh-hant" => return Cow::Borrowed("zh-Hant"),
        "auto" => return Cow::Borrowed("auto"),
        _ => {}
    }
    let normalized = normalize_to_lowercase(code);

    if let Some(&v) = ALIASES.get(normalized.as_ref()) {
        return Cow::Borrowed(v);
    }

    // Hierarchical right-to-left subtag fallback (e.g. "zh-hant-tw" -> "zh-hant" -> "zh")
    let mut current_len = normalized.len();
    while let Some((prefix, _)) = normalized[..current_len].rsplit_once('-') {
        if let Some(&v) = ALIASES.get(prefix) {
            return Cow::Borrowed(v);
        }
        current_len = prefix.len();
    }

    if let Some(&v) = ALIASES.get(&normalized[..current_len]) {
        return Cow::Borrowed(v);
    }

    match normalized {
        Cow::Borrowed(s) => Cow::Borrowed(&s[..current_len]),
        Cow::Owned(mut s) => {
            s.truncate(current_len);
            Cow::Owned(s)
        }
    }
}

/// Lowercase `code` and replace `_` with `-`; borrows when no change is needed.
fn normalize_to_lowercase(code: &str) -> Cow<'_, str> {
    if code.bytes().any(|b| b.is_ascii_uppercase() || b == b'_') {
        Cow::Owned(code.to_lowercase().replace('_', "-"))
    } else {
        Cow::Borrowed(code)
    }
}

/// Check if a language tag is known or recognized.
pub fn is_known_language_tag(code: &str) -> bool {
    let lower = code.to_ascii_lowercase();
    match lower.as_str() {
        "en" | "zh" | "ja" | "es" | "fr" | "de" | "ru" | "it" | "pt" | "ko" | "ar" | "tr"
        | "vi" | "pl" | "nl" | "el" | "th" | "hi" | "id" | "uk" | "ro" | "cs" | "sv"
        | "hu" | "fi" | "da" | "bg" | "he" | "nb" | "hr" | "sr" | "sk" | "sl" | "lt"
        | "lv" | "et" | "ca" | "gl" | "eu" | "fa" | "bn" | "gu" | "kn" | "ml" | "mr"
        | "ta" | "te" | "ur" | "is" | "sq" | "ms" | "be" | "bs" | "az" | "af" | "nn"
        | "ka" | "zh-hans" | "zh-hant" | "auto" => true,
        _ => ALIASES.contains_key(lower.as_str()),
    }
}

/// Split a language pair token (e.g. "en_zh-Hans", "zh-Hans-en", "en-es") into (from, to).
/// Handles hyphenated language codes like `zh-Hans`, `zh-Hant`, and `pt-PT`.
pub fn split_language_pair(token: &str) -> Option<(&str, &str)> {
    if let Some((from, to)) = token.split_once('_') {
        return Some((from, to));
    }
    let hyphen_count = token.matches('-').count();
    if hyphen_count == 1 {
        return token.split_once('-');
    }
    if hyphen_count > 1 {
        for (idx, _) in token.match_indices('-') {
            let left = &token[..idx];
            let right = &token[idx + 1..];
            if is_known_language_tag(left) && is_known_language_tag(right) {
                return Some((left, right));
            }
        }
        return token.rsplit_once('-');
    }
    None
}
/// Check if a language code is CJK (Chinese/Japanese/Korean).
pub fn is_cjk_code(code: &str) -> bool {
    code.get(..2).is_some_and(|p| {
        p.eq_ignore_ascii_case("zh")
            || p.eq_ignore_ascii_case("ja")
            || p.eq_ignore_ascii_case("ko")
    })
}

/// Returns true if `c` is a CJK (Chinese, Japanese, Korean) ideograph or phonetic character.
#[inline]
pub fn is_cjk_char(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'     // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}'   // CJK Extension A
        | '\u{20000}'..='\u{2A6DF}' // Extension B
        | '\u{2A700}'..='\u{2B73F}' // Extension C
        | '\u{2B740}'..='\u{2B81F}' // Extension D
        | '\u{2B820}'..='\u{2CEAF}' // Extension E
        | '\u{2CEB0}'..='\u{2EBEF}' // Extension F
        | '\u{30000}'..='\u{3134F}' // Extension G
        | '\u{31350}'..='\u{323AF}' // Extension H/I
        | '\u{F900}'..='\u{FAFF}'   // CJK Compatibility Ideographs
        | '\u{2F800}'..='\u{2FA1F}' // CJK Compatibility Supplement
        | '\u{3040}'..='\u{309F}'   // Hiragana
        | '\u{30A0}'..='\u{30FF}'   // Katakana
        | '\u{31F0}'..='\u{31FF}'   // Katakana Phonetic Extensions
        | '\u{AC00}'..='\u{D7AF}'   // Hangul Syllables
        | '\u{1100}'..='\u{11FF}'   // Hangul Jamo
        | '\u{3130}'..='\u{318F}'   // Hangul Compatibility Jamo
        | '\u{3100}'..='\u{312F}'   // Bopomofo
        | '\u{3005}'                // CJK iteration mark '々'
        | '\u{3007}'                // Ideographic number zero '〇'
    )
}

/// Check if a character belongs to CJK or continuous-script scripts (Chinese ideographs,
/// Japanese Kana, Korean Hangul, Thai, Lao, Myanmar, Khmer).
#[inline]
pub fn is_cjk_or_continuous_script(c: char) -> bool {
    is_cjk_char(c)
        || matches!(
            c,
            '\u{0E00}'..='\u{0E7F}'     // Thai
            | '\u{0E80}'..='\u{0EFF}'   // Lao
            | '\u{1000}'..='\u{109F}'   // Myanmar
            | '\u{1780}'..='\u{17FF}'   // Khmer
        )
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_language_code_fast_path() {
        assert_eq!(normalize_language_code_cow("en"), "en");
        assert!(matches!(
            normalize_language_code_cow("en"),
            Cow::Borrowed("en")
        ));

        assert_eq!(normalize_language_code_cow("zh"), "zh-Hans");
        assert!(matches!(
            normalize_language_code_cow("zh"),
            Cow::Borrowed("zh-Hans")
        ));

        assert_eq!(normalize_language_code_cow("ja"), "ja");
        assert!(matches!(
            normalize_language_code_cow("ja"),
            Cow::Borrowed("ja")
        ));

        assert_eq!(normalize_language_code_cow("es"), "es");
        assert_eq!(normalize_language_code_cow("fr"), "fr");
        assert_eq!(normalize_language_code_cow("de"), "de");
        assert_eq!(normalize_language_code_cow("ru"), "ru");
        assert_eq!(normalize_language_code_cow("it"), "it");
        assert_eq!(normalize_language_code_cow("pt"), "pt");
        assert_eq!(normalize_language_code_cow("ko"), "ko");
        assert_eq!(normalize_language_code_cow("ar"), "ar");
        assert_eq!(normalize_language_code_cow("auto"), "auto");
    }

    #[test]
    fn test_normalize_language_code_aliases_and_case() {
        assert_eq!(normalize_language_code_cow("zh-cn"), "zh-Hans");
        assert_eq!(normalize_language_code_cow("zh-TW"), "zh-Hant");
        assert_eq!(normalize_language_code_cow("zh-Hant-TW"), "zh-Hant");
        assert_eq!(normalize_language_code_cow("zh-Hant-HK"), "zh-Hant");
        assert_eq!(normalize_language_code_cow("zh-Hant-MO"), "zh-Hant");
        assert_eq!(normalize_language_code_cow("zh-TW-u-co-pinyin"), "zh-Hant");
        assert_eq!(normalize_language_code_cow("zh-Hans-CN"), "zh-Hans");
        assert_eq!(normalize_language_code_cow("EN-US"), "en");
        assert_eq!(normalize_language_code_cow("cmn"), "zh-Hans");
        assert_eq!(normalize_language_code_cow("jp"), "ja");
        assert_eq!(normalize_language_code_cow("kr"), "ko");
        assert_eq!(normalize_language_code_cow("no"), "nb");
        assert_eq!(normalize_language_code_cow("iw"), "he");
        assert_eq!(normalize_language_code_cow("in"), "id");
        assert_eq!(normalize_language_code_cow("jw"), "jv");
        assert_eq!(normalize_language_code_cow(""), "");
    }

    #[test]
    fn test_is_cjk_code() {
        assert!(is_cjk_code("zh"));
        assert!(is_cjk_code("zh-Hans"));
        assert!(is_cjk_code("ja"));
        assert!(is_cjk_code("ko"));
        assert!(!is_cjk_code("en"));
        assert!(!is_cjk_code("fr"));
        assert!(is_cjk_code("ZH-CN"));
        assert!(is_cjk_code("JA"));
        assert!(is_cjk_code("KO"));
        assert!(!is_cjk_code("z"));
        assert!(!is_cjk_code(""));
    }
    #[test]
    fn test_split_language_pair() {
        assert_eq!(split_language_pair("en-es"), Some(("en", "es")));
        assert_eq!(split_language_pair("en_es"), Some(("en", "es")));
        assert_eq!(split_language_pair("zh-Hans-en"), Some(("zh-Hans", "en")));
        assert_eq!(split_language_pair("en-zh-Hans"), Some(("en", "zh-Hans")));
        assert_eq!(split_language_pair("zh-Hans_en"), Some(("zh-Hans", "en")));
        assert_eq!(split_language_pair("zh-Hans-zh-Hant"), Some(("zh-Hans", "zh-Hant")));
        assert_eq!(split_language_pair("pt-PT-en"), Some(("pt-PT", "en")));
        assert_eq!(split_language_pair("invalid"), None);
    }
}
