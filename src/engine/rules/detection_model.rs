//! Compiled and compact read-only detection model for unified language identification.
//!
//! Compiles profiles from `RuleRegistry` (loaded from `lang_rules/*.toml`) into
//! contiguous, zero-heap-allocation lookup structures (sorted slices for binary search).

use std::collections::HashMap;
use std::sync::Arc;

use super::LanguageRuleProfile;

/// Compiled unified detection model containing all script and orthography rule indexes.
#[derive(Debug, Clone, Default)]
pub struct CompiledDetectionModel {
    /// Japanese Kokuji and exclusive Shinjitai Han characters (sorted for binary search).
    pub ja_unique_han: Box<[char]>,
    /// Simplified Chinese distinctive feature characters (sorted for binary search).
    pub zh_hans_distinctive: Box<[char]>,
    /// Traditional Chinese distinctive feature characters (sorted for binary search).
    pub zh_hant_distinctive: Box<[char]>,
    /// High-frequency Japanese Kanji compound terms.
    pub ja_terms: Box<[String]>,
    /// Cyrillic subfamily orthography and distinctive terms.
    pub cyrillic: CompiledCyrillicDetection,
    /// Devanagari family distinctions (Marathi vs Hindi).
    pub devanagari: CompiledDevanagariDetection,
    /// Arabic script distinctions (Urdu vs Persian vs Arabic).
    pub arabic: CompiledArabicDetection,
    /// Latin subfamily and orthography markers.
    pub latin: CompiledLatinDetection,
    /// High-frequency greetings loaded from language profiles.
    pub greetings: Box<[(String, &'static str)]>,
}
/// Compiled Cyrillic sub-family character and term indexes.
#[derive(Debug, Clone, Default)]
pub struct CompiledCyrillicDetection {
    /// Ukrainian-unique characters (e.g. ї, є, ґ).
    pub uk_unique: Box<[char]>,
    /// Belarusian-unique characters (e.g. ў).
    pub be_unique: Box<[char]>,
    /// Serbian-unique characters (e.g. ђ, ћ, ј, љ, њ, џ).
    pub sr_unique: Box<[char]>,
    /// Bulgarian distinctive characters (e.g. ъ).
    pub bg_distinctive: Box<[char]>,
    /// Bulgarian exclude characters.
    pub bg_exclude: Box<[char]>,
    /// Russian distinctive characters (e.g. ы, э, ё).
    pub ru_distinctive: Box<[char]>,
    /// Ukrainian distinctive characters (e.g. і).
    pub uk_distinctive: Box<[char]>,
    /// Ukrainian distinctive substring terms.
    pub uk_terms: Box<[String]>,
    /// Belarusian distinctive substring terms.
    pub be_terms: Box<[String]>,
    /// Serbian distinctive substring terms.
    pub sr_terms: Box<[String]>,
    /// Bulgarian distinctive substring terms.
    pub bg_terms: Box<[String]>,
}

impl CompiledCyrillicDetection {
    #[inline]
    pub fn is_uk_unique(&self, c: char) -> bool {
        self.uk_unique.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_be_unique(&self, c: char) -> bool {
        self.be_unique.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_sr_unique(&self, c: char) -> bool {
        self.sr_unique.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_bg_distinctive(&self, c: char) -> bool {
        self.bg_distinctive.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_bg_exclude(&self, c: char) -> bool {
        self.bg_exclude.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_ru_distinctive(&self, c: char) -> bool {
        self.ru_distinctive.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_cyrillic_i(&self, c: char) -> bool {
        self.uk_distinctive.binary_search(&c).is_ok() || c == 'і' || c == 'І'
    }
}

/// Compiled Devanagari sub-family character and term indexes.
#[derive(Debug, Clone, Default)]
pub struct CompiledDevanagariDetection {
    /// Marathi unique characters (e.g. ळ U+0933).
    pub mr_unique: Box<[char]>,
    /// Marathi distinctive substring terms.
    pub mr_terms: Box<[String]>,
    /// Marathi distinctive whole words.
    pub mr_words: Box<[String]>,
}

impl CompiledDevanagariDetection {
    #[inline]
    pub fn is_mr_unique(&self, c: char) -> bool {
        self.mr_unique.binary_search(&c).is_ok()
    }
}

/// Compiled Arabic sub-family character indexes.
#[derive(Debug, Clone, Default)]
pub struct CompiledArabicDetection {
    /// Urdu-unique characters (e.g. ٹ, ڈ, ڑ, etc.).
    pub ur_unique: Box<[char]>,
    /// Persian-unique characters (e.g. گ, چ, ژ, پ, ی).
    pub fa_unique: Box<[char]>,
}

impl CompiledArabicDetection {
    #[inline]
    pub fn is_ur_unique(&self, c: char) -> bool {
        self.ur_unique.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_fa_unique(&self, c: char) -> bool {
        self.fa_unique.binary_search(&c).is_ok()
    }
}

/// Compiled Latin sub-family and orthography indexes.
#[derive(Debug, Clone, Default)]
pub struct CompiledLatinDetection {
    pub vi_chars: Box<[char]>,
    pub is_chars: Box<[char]>,
    pub az_chars: Box<[char]>,
    pub cs_chars: Box<[char]>,
    pub sk_chars: Box<[char]>,
    pub pl_chars: Box<[char]>,
    pub hu_chars: Box<[char]>,
    pub ro_cedilla: Box<[char]>,
    pub ro_vowels: Box<[char]>,
    pub tr_chars: Box<[char]>,
    pub de_eszett: Box<[char]>,
    pub es_chars: Box<[char]>,
    pub es_punct: Box<[char]>,
    pub pt_tilde: Box<[char]>,
    pub scand_ae_oe: Box<[char]>,
    pub scand_a_o_umlaut: Box<[char]>,

    pub ro_words: Box<[String]>,
    pub pt_terms: Box<[String]>,
    pub nn_words: Box<[String]>,
    pub nn_terms: Box<[String]>,
    pub nb_terms: Box<[String]>,
    pub da_terms: Box<[String]>,
    pub gl_terms: Box<[String]>,
    pub gl_words: Box<[String]>,
    pub eu_terms: Box<[String]>,
    pub sq_terms: Box<[String]>,
    pub af_terms: Box<[String]>,
    pub ms_terms: Box<[String]>,
    pub id_terms: Box<[String]>,
    pub scand_nb_terms: Box<[String]>,
    pub scand_da_terms: Box<[String]>,
    pub scand_sv_terms: Box<[String]>,
    pub scand_sv_words: Box<[String]>,
    pub latin_lexical_words: Box<[(String, &'static str, usize)]>,
    pub latin_lexical_phrases: Box<[(String, &'static str, usize)]>,
}

impl CompiledLatinDetection {
    #[inline]
    pub fn is_vi_char(&self, c: char) -> bool {
        self.vi_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_is_char(&self, c: char) -> bool {
        self.is_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_az_char(&self, c: char) -> bool {
        self.az_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_cs_char(&self, c: char) -> bool {
        self.cs_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_sk_char(&self, c: char) -> bool {
        self.sk_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_pl_char(&self, c: char) -> bool {
        self.pl_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_hu_char(&self, c: char) -> bool {
        self.hu_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_ro_cedilla(&self, c: char) -> bool {
        self.ro_cedilla.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_ro_vowel(&self, c: char) -> bool {
        self.ro_vowels.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_tr_char(&self, c: char) -> bool {
        self.tr_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_de_eszett(&self, c: char) -> bool {
        self.de_eszett.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_es_char(&self, c: char) -> bool {
        self.es_chars.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_es_punct(&self, c: char) -> bool {
        self.es_punct.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_pt_tilde(&self, c: char) -> bool {
        self.pt_tilde.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_scand_ae_oe(&self, c: char) -> bool {
        self.scand_ae_oe.binary_search(&c).is_ok()
    }

    #[inline]
    pub fn is_scand_a_o_umlaut(&self, c: char) -> bool {
        self.scand_a_o_umlaut.binary_search(&c).is_ok()
    }
}

impl CompiledDetectionModel {
    /// Fast zero-heap-allocation lookup of high-frequency greetings.
    #[inline]
    pub fn lookup_greeting(&self, text: &str) -> Option<&'static str> {
        let trimmed = text.trim_matches(|c: char| {
            c.is_whitespace()
                || c.is_ascii_punctuation()
                || matches!(c, '，' | '。' | '！' | '？' | '；' | '：' | '“' | '”' | '‘' | '’' | '、')
        });
        if trimmed.is_empty() {
            return None;
        }

        // Fast stack-allocated path for ASCII greetings (covers 95%+ of queries)
        if trimmed.is_ascii() && trimmed.len() <= 64 {
            let mut buf = [0u8; 64];
            let len = trimmed.len();
            buf[..len].copy_from_slice(trimmed.as_bytes());
            buf[..len].make_ascii_lowercase();
            // SAFETY: original text was valid ASCII, make_ascii_lowercase preserves ASCII validity
            let lower = unsafe { std::str::from_utf8_unchecked(&buf[..len]) };
            return self
                .greetings
                .binary_search_by(|(g, _)| g.as_str().cmp(lower))
                .ok()
                .map(|idx| self.greetings[idx].1);
        }

        // Zero-heap-allocation comparison for non-ASCII or long greetings
        self.greetings
            .binary_search_by(|(g, _)| {
                let mut b1 = g.chars();
                let mut b2 = trimmed.chars().flat_map(|c| c.to_lowercase());
                loop {
                    match (b1.next(), b2.next()) {
                        (None, None) => return std::cmp::Ordering::Equal,
                        (None, Some(_)) => return std::cmp::Ordering::Less,
                        (Some(_), None) => return std::cmp::Ordering::Greater,
                        (Some(c1), Some(c2)) => {
                            let ord = c1.cmp(&c2);
                            if ord != std::cmp::Ordering::Equal {
                                return ord;
                            }
                        }
                    }
                }
            })
            .ok()
            .map(|idx| self.greetings[idx].1)
    }

    /// Check if character is a Japanese-unique Han character (Kokuji or exclusive Shinjitai).
    #[inline]
    pub fn is_ja_unique_han(&self, c: char) -> bool {
        self.ja_unique_han.binary_search(&c).is_ok()
    }

    /// Check if character is a Simplified Chinese distinctive feature character.
    #[inline]
    pub fn is_zh_hans_char(&self, c: char) -> bool {
        self.zh_hans_distinctive.binary_search(&c).is_ok()
    }

    /// Check if character is a Traditional Chinese distinctive feature character.
    #[inline]
    pub fn is_zh_hant_char(&self, c: char) -> bool {
        self.zh_hant_distinctive.binary_search(&c).is_ok()
    }

    /// Alias for Simplified Chinese distinctive character check.
    #[inline]
    pub fn is_simplified_char(&self, c: char) -> bool {
        self.is_zh_hans_char(c)
    }

    /// Alias for Traditional Chinese distinctive character check.
    #[inline]
    pub fn is_traditional_char(&self, c: char) -> bool {
        self.is_zh_hant_char(c)
    }

    /// Alias for Japanese unique Han character check.
    #[inline]
    pub fn is_kokuji(&self, c: char) -> bool {
        self.is_ja_unique_han(c)
    }

    /// Alias for Japanese unique Han character check.
    #[inline]
    pub fn is_shinjitai(&self, c: char) -> bool {
        self.is_ja_unique_han(c)
    }
    /// Compile detection model from active language rule profiles.
    pub fn from_profiles(profiles: &HashMap<String, Arc<LanguageRuleProfile>>) -> Self {
        // Helper to extract sorted unique chars
        let sort_chars = |mut chars: Vec<char>| -> Box<[char]> {
            chars.sort_unstable();
            chars.dedup();
            chars.into_boxed_slice()
        };

        // 2. CJK (Japanese unique Han characters, Kanji compound terms, and Chinese variant distinctive characters)
        let mut ja_unique_han = Vec::new();
        let mut ja_terms = Vec::new();
        let mut zh_hans_distinctive = Vec::new();
        let mut zh_hant_distinctive = Vec::new();

        if let Some(ja_prof) = profiles.get("ja") {
            if let Some(det) = &ja_prof.detection {
                ja_unique_han.extend(det.unique_chars.iter().copied().filter(|&c| {
                    let code = c as u32;
                    (0x4E00..=0x9FFF).contains(&code)
                        || (0x3400..=0x4DBF).contains(&code)
                        || (0x20000..=0x2A6DF).contains(&code)
                }));
                ja_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(zh_prof) = profiles.get("zh") {
            if let Some(det) = &zh_prof.detection {
                zh_hans_distinctive.extend_from_slice(&det.distinctive_chars);
            }
        }
        if let Some(hant_prof) = profiles.get("zh-hant") {
            if let Some(det) = &hant_prof.detection {
                zh_hant_distinctive.extend_from_slice(&det.distinctive_chars);
            }
        }

        let ja_unique_han = sort_chars(ja_unique_han);
        let zh_hans_distinctive = sort_chars(zh_hans_distinctive);
        let zh_hant_distinctive = sort_chars(zh_hant_distinctive);
        let ja_terms = ja_terms.into_boxed_slice();
        // 3. Cyrillic
        let mut uk_unique = Vec::new();
        let mut be_unique = Vec::new();
        let mut sr_unique = Vec::new();
        let mut bg_distinctive = Vec::new();
        let mut bg_exclude = Vec::new();
        let mut ru_distinctive = Vec::new();
        let mut uk_distinctive = Vec::new();
        let mut uk_terms = Vec::new();
        let mut be_terms = Vec::new();
        let mut sr_terms = Vec::new();
        let mut bg_terms = Vec::new();

        if let Some(prof) = profiles.get("uk") {
            if let Some(det) = &prof.detection {
                uk_unique.extend_from_slice(&det.unique_chars);
                uk_distinctive.extend_from_slice(&det.distinctive_chars);
                uk_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("be") {
            if let Some(det) = &prof.detection {
                be_unique.extend_from_slice(&det.unique_chars);
                be_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("sr") {
            if let Some(det) = &prof.detection {
                sr_unique.extend_from_slice(&det.unique_chars);
                sr_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("bg") {
            if let Some(det) = &prof.detection {
                bg_distinctive.extend_from_slice(&det.distinctive_chars);
                bg_exclude.extend_from_slice(&det.exclude_chars);
                bg_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("ru") {
            if let Some(det) = &prof.detection {
                ru_distinctive.extend_from_slice(&det.distinctive_chars);
            }
        }

        let cyrillic = CompiledCyrillicDetection {
            uk_unique: sort_chars(uk_unique),
            be_unique: sort_chars(be_unique),
            sr_unique: sort_chars(sr_unique),
            bg_distinctive: sort_chars(bg_distinctive),
            bg_exclude: sort_chars(bg_exclude),
            ru_distinctive: sort_chars(ru_distinctive),
            uk_distinctive: sort_chars(uk_distinctive),
            uk_terms: uk_terms.into_boxed_slice(),
            be_terms: be_terms.into_boxed_slice(),
            sr_terms: sr_terms.into_boxed_slice(),
            bg_terms: bg_terms.into_boxed_slice(),
        };

        // 4. Devanagari
        let mut mr_unique = Vec::new();
        let mut mr_terms = Vec::new();
        let mut mr_words = Vec::new();
        if let Some(prof) = profiles.get("mr") {
            if let Some(det) = &prof.detection {
                mr_unique.extend_from_slice(&det.unique_chars);
                mr_terms.extend_from_slice(&det.substring_terms);
                mr_words.extend_from_slice(&det.whole_words);
            }
        }
        let devanagari = CompiledDevanagariDetection {
            mr_unique: sort_chars(mr_unique),
            mr_terms: mr_terms.into_boxed_slice(),
            mr_words: mr_words.into_boxed_slice(),
        };

        // 5. Arabic
        let mut ur_unique = Vec::new();
        let mut fa_unique = Vec::new();
        if let Some(prof) = profiles.get("ur") {
            if let Some(det) = &prof.detection {
                ur_unique.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("fa") {
            if let Some(det) = &prof.detection {
                fa_unique.extend_from_slice(&det.unique_chars);
            }
        }
        let arabic = CompiledArabicDetection {
            ur_unique: sort_chars(ur_unique),
            fa_unique: sort_chars(fa_unique),
        };

        // 6. Latin
        let mut vi_chars = Vec::new();
        let mut is_chars = Vec::new();
        let mut az_chars = Vec::new();
        let mut cs_chars = Vec::new();
        let mut sk_chars = Vec::new();
        let mut pl_chars = Vec::new();
        let mut hu_chars = Vec::new();
        let mut ro_cedilla = Vec::new();
        let mut ro_vowels = Vec::new();
        let mut tr_chars = Vec::new();
        let mut de_eszett = Vec::new();
        let mut es_chars = Vec::new();
        let mut es_punct = Vec::new();
        let mut pt_tilde = Vec::new();
        let mut scand_ae_oe = Vec::new();
        let mut scand_a_o_umlaut = Vec::new();

        let mut ro_words = Vec::new();
        let mut pt_terms = Vec::new();
        let mut nn_words = Vec::new();
        let mut nn_terms = Vec::new();
        let mut nb_terms = Vec::new();
        let mut da_terms = Vec::new();
        let mut gl_terms = Vec::new();
        let mut gl_words = Vec::new();
        let mut eu_terms = Vec::new();
        let mut sq_terms = Vec::new();
        let mut af_terms = Vec::new();
        let mut ms_terms = Vec::new();
        let mut id_terms = Vec::new();
        let mut scand_nb_terms = Vec::new();
        let mut scand_da_terms = Vec::new();
        let mut scand_sv_terms = Vec::new();
        let mut scand_sv_words = Vec::new();

        if let Some(prof) = profiles.get("vi") {
            if let Some(det) = &prof.detection {
                vi_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("is") {
            if let Some(det) = &prof.detection {
                is_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("az") {
            if let Some(det) = &prof.detection {
                az_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("cs") {
            if let Some(det) = &prof.detection {
                cs_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("sk") {
            if let Some(det) = &prof.detection {
                sk_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("pl") {
            if let Some(det) = &prof.detection {
                pl_chars.extend_from_slice(&det.distinctive_chars);
            }
        }
        if let Some(prof) = profiles.get("hu") {
            if let Some(det) = &prof.detection {
                hu_chars.extend_from_slice(&det.unique_chars);
            }
        }
        if let Some(prof) = profiles.get("ro") {
            if let Some(det) = &prof.detection {
                for &c in &det.distinctive_chars {
                    if matches!(c, 'ș' | 'ț' | 'Ș' | 'Ț') {
                        ro_cedilla.push(c);
                    } else if matches!(c, 'ă' | 'Ă' | 'î' | 'Î' | 'â' | 'Â') {
                        ro_vowels.push(c);
                    }
                }
                ro_words.extend_from_slice(&det.whole_words);
            }
        }
        if let Some(prof) = profiles.get("tr") {
            if let Some(det) = &prof.detection {
                tr_chars.extend_from_slice(&det.distinctive_chars);
            }
        }
        if let Some(prof) = profiles.get("de") {
            if let Some(det) = &prof.detection {
                de_eszett.extend_from_slice(&det.distinctive_chars);
            }
        }
        if let Some(prof) = profiles.get("es") {
            if let Some(det) = &prof.detection {
                for &c in &det.distinctive_chars {
                    if matches!(c, 'ñ' | 'Ñ') {
                        es_chars.push(c);
                    } else if matches!(c, '¡' | '¿') {
                        es_punct.push(c);
                    }
                }
            }
        }
        if let Some(prof) = profiles.get("pt") {
            if let Some(det) = &prof.detection {
                pt_tilde.extend_from_slice(&det.distinctive_chars);
                pt_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("nn") {
            if let Some(det) = &prof.detection {
                scand_ae_oe.extend_from_slice(&det.distinctive_chars);
                nn_words.extend_from_slice(&det.whole_words);
                nn_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("nb") {
            if let Some(det) = &prof.detection {
                scand_ae_oe.extend_from_slice(&det.distinctive_chars);
                nb_terms.extend_from_slice(&det.substring_terms);
                scand_nb_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("da") {
            if let Some(det) = &prof.detection {
                scand_ae_oe.extend_from_slice(&det.distinctive_chars);
                da_terms.extend_from_slice(&det.substring_terms);
                scand_da_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("gl") {
            if let Some(det) = &prof.detection {
                gl_terms.extend_from_slice(&det.substring_terms);
                gl_words.extend_from_slice(&det.whole_words);
            }
        }
        if let Some(prof) = profiles.get("eu") {
            if let Some(det) = &prof.detection {
                eu_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("sq") {
            if let Some(det) = &prof.detection {
                sq_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("af") {
            if let Some(det) = &prof.detection {
                af_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("ms") {
            if let Some(det) = &prof.detection {
                ms_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("id") {
            if let Some(det) = &prof.detection {
                id_terms.extend_from_slice(&det.substring_terms);
            }
        }
        if let Some(prof) = profiles.get("sv") {
            if let Some(det) = &prof.detection {
                for &c in &det.distinctive_chars {
                    if matches!(c, 'ä' | 'ö' | 'Ä' | 'Ö') {
                        scand_a_o_umlaut.push(c);
                    }
                }
                scand_sv_terms.extend_from_slice(&det.substring_terms);
                for w in &det.whole_words {
                    if matches!(w.as_str(), "och" | "är") {
                        scand_sv_words.push(w.clone());
                    }
                }
            }
        }

        let mut greetings = Vec::new();
        for (lang_code, prof) in profiles {
            if let Some(det) = &prof.detection {
                let static_lang = super::detect::map_detected_code_to_static(lang_code);
                for g in &det.greetings {
                    let trimmed = g.trim();
                    if !trimmed.is_empty() {
                        greetings.push((trimmed.to_lowercase(), static_lang));
                    }
                }
            }
        }
        greetings.sort_by(|a, b| a.0.cmp(&b.0));
        greetings.dedup_by(|a, b| a.0 == b.0);
        let mut latin_lexical_words = Vec::new();
        let mut latin_lexical_phrases = Vec::new();
        let groups: &[(&[String], &'static str, usize)] = &[
            (&nn_words, "nn", 0),
            (&nn_terms, "nn", 0),
            (&nb_terms, "nb", 1),
            (&da_terms, "da", 2),
            (&gl_terms, "gl", 3),
            (&gl_words, "gl", 3),
            (&eu_terms, "eu", 4),
            (&sq_terms, "sq", 5),
            (&af_terms, "af", 6),
            (&ms_terms, "ms", 7),
            (&id_terms, "id", 8),
        ];
        for (items, lang, prio) in groups {
            for item in *items {
                if item.contains(' ') {
                    latin_lexical_phrases.push((item.clone(), *lang, *prio));
                } else {
                    latin_lexical_words.push((item.clone(), *lang, *prio));
                }
            }
        }
        latin_lexical_words.sort_by(|a, b| a.0.cmp(&b.0));
        latin_lexical_words.dedup_by(|a, b| a.0 == b.0);

        let latin = CompiledLatinDetection {
            vi_chars: sort_chars(vi_chars),
            is_chars: sort_chars(is_chars),
            az_chars: sort_chars(az_chars),
            cs_chars: sort_chars(cs_chars),
            sk_chars: sort_chars(sk_chars),
            pl_chars: sort_chars(pl_chars),
            hu_chars: sort_chars(hu_chars),
            ro_cedilla: sort_chars(ro_cedilla),
            ro_vowels: sort_chars(ro_vowels),
            tr_chars: sort_chars(tr_chars),
            de_eszett: sort_chars(de_eszett),
            es_chars: sort_chars(es_chars),
            es_punct: sort_chars(es_punct),
            pt_tilde: sort_chars(pt_tilde),
            scand_ae_oe: sort_chars(scand_ae_oe),
            scand_a_o_umlaut: sort_chars(scand_a_o_umlaut),

            ro_words: ro_words.into_boxed_slice(),
            pt_terms: pt_terms.into_boxed_slice(),
            nn_words: nn_words.into_boxed_slice(),
            nn_terms: nn_terms.into_boxed_slice(),
            nb_terms: nb_terms.into_boxed_slice(),
            da_terms: da_terms.into_boxed_slice(),
            gl_terms: gl_terms.into_boxed_slice(),
            gl_words: gl_words.into_boxed_slice(),
            eu_terms: eu_terms.into_boxed_slice(),
            sq_terms: sq_terms.into_boxed_slice(),
            af_terms: af_terms.into_boxed_slice(),
            ms_terms: ms_terms.into_boxed_slice(),
            id_terms: id_terms.into_boxed_slice(),
            scand_nb_terms: scand_nb_terms.into_boxed_slice(),
            scand_da_terms: scand_da_terms.into_boxed_slice(),
            scand_sv_terms: scand_sv_terms.into_boxed_slice(),
            scand_sv_words: scand_sv_words.into_boxed_slice(),
            latin_lexical_words: latin_lexical_words.into_boxed_slice(),
            latin_lexical_phrases: latin_lexical_phrases.into_boxed_slice(),
        };

        Self {
            ja_unique_han,
            zh_hans_distinctive,
            zh_hant_distinctive,
            ja_terms,
            cyrillic,
            devanagari,
            arabic,
            latin,
            greetings: greetings.into_boxed_slice(),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::rules::RuleRegistry;

    #[test]
    fn test_compiled_detection_model_from_registry() {
        let registry = RuleRegistry::global();
        let model = registry.detection_model();

        // CJK
        assert!(model.is_ja_unique_han('込'));
        assert!(model.is_ja_unique_han('発'));
        assert!(model.is_zh_hans_char('国'));
        assert!(model.is_zh_hant_char('國'));
        assert!(model.ja_terms.iter().any(|t| t == "新幹線"));
        // Cyrillic
        assert!(model.cyrillic.is_uk_unique('ї'));
        assert!(model.cyrillic.is_be_unique('ў'));
        assert!(model.cyrillic.is_sr_unique('ђ'));
        assert!(model.cyrillic.is_bg_distinctive('ъ'));
        assert!(model.cyrillic.is_ru_distinctive('ы'));
        assert!(model.cyrillic.uk_terms.iter().any(|t| t == "Україна"));

        // Devanagari
        assert!(model.devanagari.is_mr_unique('ळ'));
        assert!(model.devanagari.mr_terms.iter().any(|t| t == "महाराष्ट्र"));
        assert!(model.devanagari.mr_words.iter().any(|w| w == "आहे"));

        // Arabic
        assert!(model.arabic.is_ur_unique('ٹ'));
        assert!(model.arabic.is_fa_unique('گ'));

        // Latin
        assert!(model.latin.is_vi_char('ơ'));
        assert!(model.latin.is_is_char('þ'));
        assert!(model.latin.is_az_char('ə'));
        assert!(model.latin.is_cs_char('ř'));
        assert!(model.latin.is_sk_char('ľ'));
        assert!(model.latin.is_pl_char('ł'));
        assert!(model.latin.is_hu_char('ő'));
        assert!(model.latin.is_ro_cedilla('ș'));
        assert!(model.latin.is_ro_vowel('ă'));
        assert!(model.latin.is_tr_char('ğ'));
        assert!(model.latin.is_de_eszett('ß'));
        assert!(model.latin.is_es_char('ñ'));
        assert!(model.latin.is_es_punct('¡'));
        assert!(model.latin.is_pt_tilde('ã'));
        assert!(model.latin.is_scand_ae_oe('æ'));
        assert!(model.latin.is_scand_a_o_umlaut('ä'));
    }
}
