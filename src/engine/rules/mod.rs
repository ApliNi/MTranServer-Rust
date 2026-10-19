//! Language rule profile system for data-driven sentence splitting,
//! abbreviation protection, and discourse stitching.
//!
//! Separates language-specific lexicographical strategies from the core
//! semiotic topology engine. Loaded silently at startup from `lang_rules/`.

pub mod detect;
pub mod detection_model;
pub use detect::{detect_dominant_language, detect_text_language, detect_text_language_with_confidence};
pub use detection_model::CompiledDetectionModel;
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, RwLock};

use rust_embed::RustEmbed;
use serde::{Deserialize, Serialize};

/// Embedded language rules folder.
///
/// Automatically bundles all `.toml` files in `lang_rules/` into the binary at compile time.
/// New language rule profiles dropped into `lang_rules/` are embedded without requiring code changes.
#[derive(RustEmbed)]
#[folder = "lang_rules/"]
struct BuiltinRules;

#[derive(Debug, Clone, Deserialize)]
pub struct RawRuleProfile {
    pub metadata: RawMetadata,
    #[serde(default)]
    pub abbreviations: RawAbbreviations,
    #[serde(default)]
    pub conjunctions: RawConjunctions,
    #[serde(default)]
    pub discourse: RawDiscourse,
    #[serde(default)]
    pub generation: GenerationRules,
    #[serde(default)]
    pub typography: TypographyRules,
    #[serde(default)]
    pub syntax: SyntaxRules,
    #[serde(default = "default_conditional_end_prefixes")]
    pub conditional_end_prefixes: Vec<String>,
    #[serde(default)]
    pub detection: Option<DetectionRules>,
    #[serde(default)]
    pub postprocess: PostprocessRules,
    #[serde(default)]
    pub hyphenation: HyphenationRules,
}

pub type RawLanguageRuleProfile = RawRuleProfile;

#[derive(Debug, Clone, Deserialize)]
pub struct RawMetadata {
    pub language: String,
    #[serde(default)]
    pub version: String,
    #[serde(default = "default_true")]
    pub word_boundary: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawAbbreviations {
    #[serde(default)]
    pub nonbreaking_prefixes: Vec<String>,
    #[serde(default)]
    pub conditional_end_prefixes: Option<Vec<String>>,
}

pub fn default_conditional_end_prefixes() -> Vec<String> {
    vec![
        "etc.".to_string(),
        "al.".to_string(),
        "inc.".to_string(),
        "ltd.".to_string(),
        "co.".to_string(),
        "corp.".to_string(),
        "usw.".to_string(),
        "bzw.".to_string(),
    ]
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawConjunctions {
    #[serde(default)]
    pub clause_conjunctions: Vec<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RawDiscourse {
    #[serde(default)]
    pub discourse_starters: Vec<String>,
}

/// Generation constraints and token density rules for adaptive decoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerationRules {
    #[serde(default = "default_token_density_ratio", alias = "density")]
    pub token_density_ratio: f32,
    #[serde(default = "default_min_length_factor")]
    pub min_length_factor: f32,
    #[serde(default = "default_max_length_factor")]
    pub max_length_factor: f32,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    #[serde(default)]
    pub no_repeat_ngram_size: Option<usize>,
}

pub fn default_token_density_ratio() -> f32 {
    1.0
}

pub fn default_min_length_factor() -> f32 {
    0.5
}

pub fn default_max_length_factor() -> f32 {
    2.5
}

impl Default for GenerationRules {
    fn default() -> Self {
        Self {
            token_density_ratio: default_token_density_ratio(),
            min_length_factor: default_min_length_factor(),
            max_length_factor: default_max_length_factor(),
            repetition_penalty: None,
            no_repeat_ngram_size: None,
        }
    }
}
/// Language-specific typography and punctuation styling rules.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TypographyRules {
    #[serde(default, alias = "pangu")]
    pub pangu_spacing: bool,
    #[serde(default)]
    pub french_spacing: bool,
    #[serde(default)]
    pub spanish_inverted: bool,
    #[serde(default, alias = "quote")]
    pub quote_style: Option<String>,
    #[serde(default)]
    pub separators: HashMap<String, String>,
    #[serde(default)]
    pub inverted_compounds: Vec<String>,
    #[serde(default)]
    pub interrogatives: Vec<String>,
    #[serde(default)]
    pub exclamatories: Vec<String>,
    #[serde(default)]
    pub introductory_words: Vec<String>,
}

/// Language-specific postprocessing rules (e.g. title prefix preservation, translation replacements).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostprocessRules {
    #[serde(default = "default_true")]
    pub preserve_title_prefix: bool,
    #[serde(default)]
    pub replacements: Vec<(String, String)>,
}

impl Default for PostprocessRules {
    fn default() -> Self {
        Self {
            preserve_title_prefix: true,
            replacements: Vec::new(),
        }
    }
}
/// Language-specific morphology rules for hyphenated compound tokens.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct HyphenationRules {
    #[serde(default)]
    pub prefixes: Vec<String>,
    #[serde(default)]
    pub suffixes: Vec<String>,
    #[serde(default)]
    pub connectors: Vec<String>,
}
/// Syntactic topology rules for phrase clustering and tree projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SyntaxRules {
    #[serde(default = "default_clustering_gap_threshold", alias = "gap")]
    pub clustering_gap_threshold: usize,
    #[serde(default = "default_head_direction")]
    pub head_direction: String,
    #[serde(default)]
    pub predicate_verbs: Vec<String>,
    #[serde(default)]
    pub non_verb_ed: Vec<String>,
    #[serde(default)]
    pub non_verb_ing: Vec<String>,
}

pub fn default_clustering_gap_threshold() -> usize {
    3
}

pub fn default_head_direction() -> String {
    "head_initial".to_string()
}

impl Default for SyntaxRules {
    fn default() -> Self {
        Self {
            clustering_gap_threshold: default_clustering_gap_threshold(),
            head_direction: default_head_direction(),
            predicate_verbs: Vec::new(),
            non_verb_ed: Vec::new(),
            non_verb_ing: Vec::new(),
        }
    }
}
/// Language detection rules for script and orthography classification.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DetectionRules {
    pub script: Option<String>,
    #[serde(default)]
    pub unique_script: bool,
    #[serde(default)]
    pub unique_chars: Vec<char>,
    #[serde(default)]
    pub distinctive_chars: Vec<char>,
    #[serde(default)]
    pub exclude_chars: Vec<char>,
    #[serde(default)]
    pub whole_words: Vec<String>,
    #[serde(default)]
    pub substring_terms: Vec<String>,
    #[serde(default)]
    pub greetings: Vec<String>,
}

/// Fast-lookup index for hot-path sentence splitting queries.
#[derive(Debug, Clone, Default)]
pub(crate) struct RuleIndex {
    abbreviation_set: HashSet<String>,
    complex_prefixes: Vec<String>,
    conditional_end_set: HashSet<String>,
    conjunction_by_first_word: HashMap<String, Vec<String>>,
    conjunction_fallback: Vec<String>,
    conjunction_by_first_char: HashMap<char, Vec<String>>,
    discourse_by_first_word: HashMap<String, Vec<String>>,
    discourse_fallback: Vec<String>,
    discourse_by_first_char: HashMap<char, Vec<String>>,
    pub(crate) predicate_verbs: HashSet<String>,
    pub(crate) non_verb_ed: HashSet<String>,
    pub(crate) non_verb_ing: HashSet<String>,
    pub(crate) hyphen_prefixes: HashSet<String>,
    pub(crate) hyphen_suffixes: HashSet<String>,
    pub(crate) hyphen_connectors: HashSet<String>,
    pub(crate) inverted_compounds: Vec<String>,
    pub(crate) interrogatives: HashSet<String>,
    pub(crate) exclamatories: HashSet<String>,
    pub(crate) introductory_words: HashSet<String>,
}

#[inline]
fn extract_first_token(s: &str) -> &str {
    let end = s
        .char_indices()
        .find(|&(_, c)| !c.is_alphanumeric() && c != '_')
        .map(|(i, _)| i)
        .unwrap_or(s.len());
    &s[..end]
}

impl RuleIndex {
    fn build(
        nonbreaking_prefixes: &[String],
        conditional_end_prefixes: &[String],
        clause_conjunctions: &[String],
        discourse_starters: &[String],
        word_boundary: bool,
        hyphenation: &HyphenationRules,
        syntax: &SyntaxRules,
        typography: &TypographyRules,
    ) -> Self {
        let mut abbreviation_set = HashSet::with_capacity(nonbreaking_prefixes.len() * 2);
        let mut complex_prefixes = Vec::new();

        for prefix in nonbreaking_prefixes {
            let no_dot = prefix.strip_suffix('.').unwrap_or(prefix);
            if prefix.contains(' ') || no_dot.contains('.') {
                complex_prefixes.push(prefix.clone());
            } else {
                abbreviation_set.insert(prefix.to_lowercase());
                abbreviation_set.insert(no_dot.to_lowercase());
            }
        }

        let mut conditional_end_set = HashSet::with_capacity(conditional_end_prefixes.len() * 4);
        for prefix in conditional_end_prefixes {
            let no_dot = prefix.strip_suffix('.').unwrap_or(prefix);
            conditional_end_set.insert(prefix.to_lowercase());
            conditional_end_set.insert(no_dot.to_lowercase());
            let no_space = prefix.replace(' ', "");
            conditional_end_set.insert(no_space.to_lowercase());
            let no_space_no_dot = no_dot.replace(' ', "");
            conditional_end_set.insert(no_space_no_dot.to_lowercase());
        }

        let mut conjunction_by_first_word: HashMap<String, Vec<String>> = HashMap::new();
        let mut conjunction_fallback = Vec::new();
        let mut conjunction_by_first_char: HashMap<char, Vec<String>> = HashMap::new();

        for conj in clause_conjunctions {
            if word_boundary {
                let first_tok = extract_first_token(conj);
                if !first_tok.is_empty() {
                    conjunction_by_first_word
                        .entry(first_tok.to_lowercase())
                        .or_default()
                        .push(conj.clone());
                } else {
                    conjunction_fallback.push(conj.clone());
                }
            } else if let Some(first_char) = conj.chars().next() {
                conjunction_by_first_char
                    .entry(first_char)
                    .or_default()
                    .push(conj.clone());
            }
        }

        let mut discourse_by_first_word: HashMap<String, Vec<String>> = HashMap::new();
        let mut discourse_fallback = Vec::new();
        let mut discourse_by_first_char: HashMap<char, Vec<String>> = HashMap::new();

        for starter in discourse_starters {
            if word_boundary {
                let first_tok = extract_first_token(starter);
                if !first_tok.is_empty() {
                    discourse_by_first_word
                        .entry(first_tok.to_lowercase())
                        .or_default()
                        .push(starter.clone());
                } else {
                    discourse_fallback.push(starter.clone());
                }
            } else if let Some(first_char) = starter.chars().next() {
                discourse_by_first_char
                    .entry(first_char)
                    .or_default()
                    .push(starter.clone());
            }
        }

        let predicate_verbs = syntax
            .predicate_verbs
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let non_verb_ed = syntax
            .non_verb_ed
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let non_verb_ing = syntax
            .non_verb_ing
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let hyphen_prefixes = hyphenation
            .prefixes
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let hyphen_suffixes = hyphenation
            .suffixes
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let hyphen_connectors = hyphenation
            .connectors
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let mut inverted_compounds = typography.inverted_compounds.clone();
        inverted_compounds.sort_by(|a, b| b.len().cmp(&a.len()));
        let interrogatives = typography
            .interrogatives
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let exclamatories = typography
            .exclamatories
            .iter()
            .map(|s| s.to_lowercase())
            .collect();
        let introductory_words = typography
            .introductory_words
            .iter()
            .map(|s| s.to_lowercase())
            .collect();

        Self {
            abbreviation_set,
            complex_prefixes,
            conditional_end_set,
            conjunction_by_first_word,
            conjunction_fallback,
            conjunction_by_first_char,
            discourse_by_first_word,
            discourse_fallback,
            discourse_by_first_char,
            predicate_verbs,
            non_verb_ed,
            non_verb_ing,
            hyphen_prefixes,
            hyphen_suffixes,
            hyphen_connectors,
            inverted_compounds,
            interrogatives,
            exclamatories,
            introductory_words,
        }
    }
}

/// Compiled, zero-allocation language rule profile for fast-path lookups.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageRuleProfile {
    pub language: String,
    pub version: String,
    pub word_boundary: bool,
    /// Nonbreaking prefixes sorted by descending byte length.
    pub nonbreaking_prefixes: Vec<String>,
    /// Conditional sentence-ending prefixes (e.g. "etc.", "al.", "inc.").
    #[serde(default = "default_conditional_end_prefixes")]
    pub conditional_end_prefixes: Vec<String>,
    /// Clause conjunctions sorted by descending byte length.
    pub clause_conjunctions: Vec<String>,
    /// Discourse starters sorted by descending byte length.
    pub discourse_starters: Vec<String>,
    pub generation: GenerationRules,
    pub typography: TypographyRules,
    pub syntax: SyntaxRules,
    #[serde(default)]
    pub detection: Option<DetectionRules>,
    #[serde(default)]
    pub postprocess: PostprocessRules,
    #[serde(default)]
    pub hyphenation: HyphenationRules,
    #[serde(skip)]
    index: Arc<RuleIndex>,
}

impl PartialEq for LanguageRuleProfile {
    fn eq(&self, other: &Self) -> bool {
        self.language == other.language
            && self.version == other.version
            && self.word_boundary == other.word_boundary
            && self.nonbreaking_prefixes == other.nonbreaking_prefixes
            && self.conditional_end_prefixes == other.conditional_end_prefixes
            && self.clause_conjunctions == other.clause_conjunctions
            && self.discourse_starters == other.discourse_starters
            && self.generation == other.generation
            && self.typography == other.typography
            && self.syntax == other.syntax
            && self.detection == other.detection
            && self.postprocess == other.postprocess
            && self.hyphenation == other.hyphenation
    }
}

impl LanguageRuleProfile {
    /// Parse a profile from a TOML string.
    pub fn from_toml_str(content: &str) -> Result<Self, toml::de::Error> {
        let raw: RawRuleProfile = toml::from_str(content)?;

        let mut nonbreaking_prefixes = raw.abbreviations.nonbreaking_prefixes;
        nonbreaking_prefixes.retain(|s| !s.trim().is_empty());
        nonbreaking_prefixes.sort_by(|a, b| b.len().cmp(&a.len()));
        let mut conditional_end_prefixes = raw
            .abbreviations
            .conditional_end_prefixes
            .unwrap_or(raw.conditional_end_prefixes);
        if conditional_end_prefixes.is_empty() {
            conditional_end_prefixes = default_conditional_end_prefixes();
        }
        conditional_end_prefixes.retain(|s| !s.trim().is_empty());
        conditional_end_prefixes.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut clause_conjunctions = raw.conjunctions.clause_conjunctions;
        clause_conjunctions.retain(|s| !s.trim().is_empty());
        clause_conjunctions.sort_by(|a, b| b.len().cmp(&a.len()));

        let mut discourse_starters = raw.discourse.discourse_starters;
        discourse_starters.retain(|s| !s.trim().is_empty());
        discourse_starters.sort_by(|a, b| b.len().cmp(&a.len()));

        let index = Arc::new(RuleIndex::build(
            &nonbreaking_prefixes,
            &conditional_end_prefixes,
            &clause_conjunctions,
            &discourse_starters,
            raw.metadata.word_boundary,
            &raw.hyphenation,
            &raw.syntax,
            &raw.typography,
        ));

        Ok(Self {
            language: normalize_lang_code(&raw.metadata.language),
            version: raw.metadata.version,
            word_boundary: raw.metadata.word_boundary,
            nonbreaking_prefixes,
            conditional_end_prefixes,
            clause_conjunctions,
            discourse_starters,
            generation: raw.generation,
            typography: raw.typography,
            syntax: raw.syntax,
            detection: raw.detection,
            postprocess: raw.postprocess,
            hyphenation: raw.hyphenation,
            index,
        })
    }

    /// Check if a token or text fragment ending with a period is a conditional sentence-ending abbreviation.
    ///
    /// Evaluates tokens and multi-word abbreviations such as "etc.", "al.", "inc.", "usw.", "т. д.", "z. B."
    /// with Unicode-aware case insensitivity, optional space variations, and reverse matching.
    /// Guaranteed 0 heap allocation.
    #[inline]
    pub fn is_conditional_end(&self, token: &str) -> bool {
        let token_trimmed = token.trim();
        let token_no_dot = token_trimmed.strip_suffix('.').unwrap_or(token_trimmed);

        if !self.index.conditional_end_set.is_empty() {
            let lower_trimmed = token_trimmed.to_lowercase();
            let lower_no_dot = token_no_dot.to_lowercase();
            if self.index.conditional_end_set.contains(&lower_trimmed)
                || (!lower_no_dot.is_empty() && self.index.conditional_end_set.contains(&lower_no_dot))
            {
                return true;
            }
            let no_space = lower_trimmed.replace(' ', "");
            let no_space_no_dot = lower_no_dot.replace(' ', "");
            if self.index.conditional_end_set.contains(&no_space)
                || (!no_space_no_dot.is_empty() && self.index.conditional_end_set.contains(&no_space_no_dot))
            {
                return true;
            }
        }

        for prefix in &self.conditional_end_prefixes {
            let prefix_no_dot = prefix.strip_suffix('.').unwrap_or(prefix);
            if eq_ignore_case(token_trimmed, prefix)
                || (!token_no_dot.is_empty()
                    && !prefix_no_dot.is_empty()
                    && eq_ignore_case(token_no_dot, prefix_no_dot))
            {
                return true;
            }
            if eq_ignore_case_and_spaces(token_trimmed, prefix)
                || (!token_no_dot.is_empty()
                    && !prefix_no_dot.is_empty()
                    && eq_ignore_case_and_spaces(token_no_dot, prefix_no_dot))
            {
                return true;
            }
            if (prefix.contains(' ') || prefix.contains('.')) && ends_with_ignore_case(token_trimmed, prefix) {
                let match_start = token_trimmed.len().saturating_sub(prefix.len());
                if match_start == 0 || !token_trimmed[..match_start].chars().next_back().map_or(false, |c| c.is_alphanumeric()) {
                    return true;
                }
            }
            if (prefix.contains(' ') || prefix.contains('.'))
                && !token_no_dot.is_empty()
                && !prefix_no_dot.is_empty()
                && ends_with_ignore_case(token_no_dot, prefix_no_dot)
            {
                let match_start = token_no_dot.len().saturating_sub(prefix_no_dot.len());
                if match_start == 0 || !token_no_dot[..match_start].chars().next_back().map_or(false, |c| c.is_alphanumeric()) {
                    return true;
                }
            }
        }
        false
    }

    /// Check if the text ending at `pos` (typically a period position) matches any conditional sentence-ending prefix.
    ///
    /// Evaluates multi-token abbreviations (e.g. "т. д.", "z. B.", "p. ex.") as well as single-token
    /// abbreviations in their context, with full Unicode case insensitivity and boundary validation.
    /// Guaranteed 0 heap allocation.
    #[inline]
    pub fn is_conditional_end_at(&self, text: &str, pos: usize) -> bool {
        if pos >= text.len() || !text.is_char_boundary(pos) {
            return false;
        }
        let end = pos + 1;
        if !text.is_char_boundary(end) {
            return false;
        }
        let text_up_to_pos = &text[..end];
        let text_up_no_dot = text_up_to_pos.strip_suffix('.').unwrap_or(text_up_to_pos);

        for prefix in &self.conditional_end_prefixes {
            let prefix_no_dot = prefix.strip_suffix('.').unwrap_or(prefix);

            if ends_with_ignore_case(text_up_to_pos, prefix) {
                let matched_len = prefix.len();
                let start = text_up_to_pos.len().saturating_sub(matched_len);
                if start == 0 || text[..start].chars().next_back().map_or(true, |c| !c.is_alphanumeric()) {
                    return true;
                }
            }
            if ends_with_ignore_case(text_up_no_dot, prefix_no_dot) {
                let matched_len = prefix_no_dot.len();
                let start = text_up_no_dot.len().saturating_sub(matched_len);
                if start == 0 || text[..start].chars().next_back().map_or(true, |c| !c.is_alphanumeric()) {
                    return true;
                }
            }

            if prefix.contains(' ') {
                if let Some(matched_len) = match_suffix_ignore_case_and_spaces(text_up_to_pos, prefix) {
                    let start = text_up_to_pos.len().saturating_sub(matched_len);
                    if start == 0 || text[..start].chars().next_back().map_or(true, |c| !c.is_alphanumeric()) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Check if the text ending at period position matches any nonbreaking prefix.
    ///
    /// Evaluates both multi-token abbreviations (e.g. "e. g.", "z. B.") and single
    /// token prefixes (e.g. "Mr.", "Dr.", "etc.").
    /// Guaranteed 0 heap allocation.
    #[inline]
    pub fn is_nonbreaking_prefix(&self, text: &str, pos: usize, word_start: usize) -> bool {
        if word_start > pos || pos >= text.len() {
            return false;
        }
        let end = pos + 1;
        if !text.is_char_boundary(word_start) || !text.is_char_boundary(end) {
            return false;
        }

        let token = &text[word_start..end];
        let text_up_to_pos = &text[..end];
        let text_from_word_start = &text[word_start..];
        let token_no_dot = token.strip_suffix('.').unwrap_or(token);

        if !self.index.abbreviation_set.is_empty() {
            let lower_token = token.to_lowercase();
            let lower_no_dot = token_no_dot.to_lowercase();

            let matched = self.index.abbreviation_set.contains(&lower_token)
                || (!lower_no_dot.is_empty() && self.index.abbreviation_set.contains(&lower_no_dot));

            if matched {
                let left_boundary = word_start == 0
                    || text[..word_start]
                        .chars()
                        .next_back()
                        .map_or(true, |c| !c.is_alphanumeric());

                if left_boundary {
                    let right_boundary = token.ends_with('.')
                        || text[end..].starts_with('.')
                        || end >= text.len()
                        || text[end..]
                            .chars()
                            .next()
                            .map_or(true, |c| !c.is_alphanumeric());

                    if right_boundary
                        && is_valid_abbreviation_context(text, word_start, end, token, token_no_dot)
                    {
                        return true;
                    }
                }
            }

            for prefix in &self.index.complex_prefixes {
                let prefix_no_dot = prefix.strip_suffix('.').unwrap_or(prefix);

                if prefix.contains(' ') {
                    let text_up_no_dot = text_up_to_pos.strip_suffix('.').unwrap_or(text_up_to_pos);
                    if ends_with_ignore_case(text_up_to_pos, prefix)
                        || ends_with_ignore_case(text_up_no_dot, prefix_no_dot)
                        || starts_with_ignore_case(text_from_word_start, prefix)
                    {
                        return true;
                    }
                    if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix_no_dot) {
                        if text_from_word_start[matched_len..].starts_with('.') {
                            return true;
                        }
                    }
                } else {
                    let left_boundary = word_start == 0
                        || text[..word_start]
                            .chars()
                            .next_back()
                            .map_or(true, |c| !c.is_alphanumeric());
                    if !left_boundary {
                        continue;
                    }

                    let right_boundary = token.ends_with('.')
                        || text[end..].starts_with('.')
                        || end >= text.len()
                        || text[end..]
                            .chars()
                            .next()
                            .map_or(true, |c| !c.is_alphanumeric());
                    if !right_boundary {
                        continue;
                    }

                    let matches_token = eq_ignore_case(token, prefix)
                        || (!token_no_dot.is_empty()
                            && !prefix_no_dot.is_empty()
                            && eq_ignore_case(token_no_dot, prefix_no_dot));

                    if matches_token
                        && is_valid_abbreviation_context(text, word_start, end, token, prefix_no_dot)
                    {
                        return true;
                    }

                    if prefix.contains('.') {
                        if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix) {
                            let prefix_end = word_start + matched_len;
                            let prefix_token = &text[word_start..prefix_end];
                            if is_valid_abbreviation_context(
                                text,
                                word_start,
                                prefix_end,
                                prefix_token,
                                prefix_no_dot,
                            ) {
                                return true;
                            }
                        }
                        if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix_no_dot) {
                            if text_from_word_start[matched_len..].starts_with('.') {
                                let prefix_end = word_start + matched_len + 1;
                                let prefix_token = &text[word_start..prefix_end];
                                if is_valid_abbreviation_context(
                                    text,
                                    word_start,
                                    prefix_end,
                                    prefix_token,
                                    prefix_no_dot,
                                ) {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            return false;
        }

        for prefix in &self.nonbreaking_prefixes {
            let prefix_no_dot = prefix.strip_suffix('.').unwrap_or(prefix);

            if prefix.contains(' ') {
                let text_up_no_dot = text_up_to_pos.strip_suffix('.').unwrap_or(text_up_to_pos);
                if ends_with_ignore_case(text_up_to_pos, prefix)
                    || ends_with_ignore_case(text_up_no_dot, prefix_no_dot)
                    || starts_with_ignore_case(text_from_word_start, prefix)
                {
                    return true;
                }
                if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix_no_dot) {
                    if text_from_word_start[matched_len..].starts_with('.') {
                        return true;
                    }
                }
            } else {
                let left_boundary = word_start == 0
                    || text[..word_start]
                        .chars()
                        .next_back()
                        .map_or(true, |c| !c.is_alphanumeric());
                if !left_boundary {
                    continue;
                }

                let right_boundary = token.ends_with('.')
                    || text[end..].starts_with('.')
                    || end >= text.len()
                    || text[end..]
                        .chars()
                        .next()
                        .map_or(true, |c| !c.is_alphanumeric());
                if !right_boundary {
                    continue;
                }

                let matches_token = eq_ignore_case(token, prefix)
                    || (!token_no_dot.is_empty()
                        && !prefix_no_dot.is_empty()
                        && eq_ignore_case(token_no_dot, prefix_no_dot));

                if matches_token
                    && is_valid_abbreviation_context(text, word_start, end, token, prefix_no_dot)
                {
                    return true;
                }

                if prefix.contains('.') {
                    if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix) {
                        let prefix_end = word_start + matched_len;
                        let prefix_token = &text[word_start..prefix_end];
                        if is_valid_abbreviation_context(
                            text,
                            word_start,
                            prefix_end,
                            prefix_token,
                            prefix_no_dot,
                        ) {
                            return true;
                        }
                    }
                    if let Some(matched_len) = match_prefix_len_ignore_case(text_from_word_start, prefix_no_dot) {
                        if text_from_word_start[matched_len..].starts_with('.') {
                            let prefix_end = word_start + matched_len + 1;
                            let prefix_token = &text[word_start..prefix_end];
                            if is_valid_abbreviation_context(
                                text,
                                word_start,
                                prefix_end,
                                prefix_token,
                                prefix_no_dot,
                            ) {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    /// Check if text following a comma or clause separator starts with any clause conjunction.
    ///
    /// Skips leading whitespace and quotation marks before matching.
    /// Guaranteed 0 heap allocation.
    #[inline]
    pub fn matches_clause_conjunction(&self, mut after: &str) -> bool {
        while let Some(c) = after.chars().next() {
            if c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | '“' | '‘' | '「' | '『' | '（' | '(' | '《' | '<'
                )
            {
                after = &after[c.len_utf8()..];
            } else {
                break;
            }
        }

        if after.is_empty() {
            return false;
        }

        if self.word_boundary {
            let first_tok = extract_first_token(after);
            if !first_tok.is_empty() {
                let lower_tok = first_tok.to_lowercase();
                if let Some(candidates) = self.index.conjunction_by_first_word.get(&lower_tok) {
                    for conj in candidates {
                        if let Some(clen) = match_prefix_len_ignore_case(after, conj) {
                            if is_word_boundary_at(after, clen) {
                                return true;
                            }
                        }
                    }
                }
            }
            for conj in &self.index.conjunction_fallback {
                if let Some(clen) = match_prefix_len_ignore_case(after, conj) {
                    if is_word_boundary_at(after, clen) {
                        return true;
                    }
                }
            }
            if self.index.conjunction_by_first_word.is_empty() && !self.clause_conjunctions.is_empty() {
                for conj in &self.clause_conjunctions {
                    if let Some(clen) = match_prefix_len_ignore_case(after, conj) {
                        if is_word_boundary_at(after, clen) {
                            return true;
                        }
                    }
                }
            }
        } else {
            if let Some(first_char) = after.chars().next() {
                if let Some(candidates) = self.index.conjunction_by_first_char.get(&first_char) {
                    for conj in candidates {
                        if after.starts_with(conj) {
                            return true;
                        }
                    }
                }
            }
            if self.index.conjunction_by_first_char.is_empty() && !self.clause_conjunctions.is_empty() {
                for conj in &self.clause_conjunctions {
                    if after.starts_with(conj) {
                        return true;
                    }
                }
            }
        }

        false
    }

    /// Check if text following a comma or clause separator starts with any clause conjunction.
    #[inline]
    pub fn is_conjunction_clause_marker(&self, after: &str) -> bool {
        self.matches_clause_conjunction(after)
    }

    /// Check if the short sentence starts with an anaphoric or discourse starter.
    ///
    /// Guaranteed 0 heap allocation.
    #[inline]
    pub fn is_discourse_chunk_starter(&self, mut text: &str) -> bool {
        while let Some(c) = text.chars().next() {
            if c.is_whitespace()
                || matches!(
                    c,
                    '"' | '\'' | '“' | '‘' | '「' | '『' | '（' | '(' | '《' | '<'
                )
            {
                text = &text[c.len_utf8()..];
            } else {
                break;
            }
        }

        if text.is_empty() {
            return false;
        }

        if self.word_boundary {
            let first_tok = extract_first_token(text);
            if !first_tok.is_empty() {
                let lower_tok = first_tok.to_lowercase();
                if let Some(candidates) = self.index.discourse_by_first_word.get(&lower_tok) {
                    for starter in candidates {
                        if let Some(slen) = match_prefix_len_ignore_case(text, starter) {
                            if is_word_boundary_at(text, slen) {
                                return true;
                            }
                        }
                    }
                }
            }
            for starter in &self.index.discourse_fallback {
                if let Some(slen) = match_prefix_len_ignore_case(text, starter) {
                    if is_word_boundary_at(text, slen) {
                        return true;
                    }
                }
            }
            if self.index.discourse_by_first_word.is_empty() && !self.discourse_starters.is_empty() {
                for starter in &self.discourse_starters {
                    if let Some(slen) = match_prefix_len_ignore_case(text, starter) {
                        if is_word_boundary_at(text, slen) {
                            return true;
                        }
                    }
                }
            }
        } else {
            if let Some(first_char) = text.chars().next() {
                if let Some(candidates) = self.index.discourse_by_first_char.get(&first_char) {
                    for starter in candidates {
                        if text.starts_with(starter) {
                            return true;
                        }
                    }
                }
            }
            if self.index.discourse_by_first_char.is_empty() && !self.discourse_starters.is_empty() {
                for starter in &self.discourse_starters {
                    if text.starts_with(starter) {
                        return true;
                    }
                }
            }
        }

        false
    }

    #[inline]
    pub fn postprocess(&self) -> &PostprocessRules {
        &self.postprocess
    }

    #[inline]
    pub fn hyphenation(&self) -> &HyphenationRules {
        &self.hyphenation
    }

    /// Check if two tokens around a hyphen qualify for morphological protection without heap allocation.
    #[inline]
    pub fn is_hyphen_protected(&self, prev: &str, next: &str) -> bool {
        if prev.is_empty() || next.is_empty() {
            return true;
        }
        let prev_lower = prev.to_ascii_lowercase();
        let next_lower = next.to_ascii_lowercase();

        if self.index.hyphen_prefixes.contains(&prev_lower) {
            return true;
        }
        if self.index.hyphen_suffixes.contains(&next_lower) {
            return true;
        }
        if self.index.hyphen_connectors.contains(&prev_lower)
            || self.index.hyphen_connectors.contains(&next_lower)
        {
            return true;
        }
        let is_alphanumeric_code = |token: &str| {
            let has_digit = token.chars().any(|c| c.is_ascii_digit());
            let has_upper = token.chars().any(|c| c.is_ascii_uppercase());
            has_digit && has_upper && token.chars().all(|c| c.is_ascii_alphanumeric())
        };
        if is_alphanumeric_code(prev) || is_alphanumeric_code(next) {
            return true;
        }
        if prev_lower == next_lower {
            return true;
        }
        false
    }

    /// Check if a word is recognized as a predicate verb in this language.
    #[inline]
    pub fn is_predicate_verb(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.predicate_verbs.contains(&lower)
    }

    /// Check if word is known non-verb ending in -ed (e.g. "naked", "hundred").
    #[inline]
    pub fn is_non_verb_ed(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.non_verb_ed.contains(&lower)
    }

    /// Check if word is known non-verb ending in -ing (e.g. "morning", "ceiling").
    #[inline]
    pub fn is_non_verb_ing(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.non_verb_ing.contains(&lower)
    }

    /// Check if word is a Spanish interrogative word (e.g. "qué", "cómo", "dónde").
    #[inline]
    pub fn is_spanish_interrogative(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.interrogatives.contains(&lower)
    }

    /// Check if word is a Spanish exclamatory word (e.g. "qué", "cuánto", "ojalá").
    #[inline]
    pub fn is_spanish_exclamatory(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.exclamatories.contains(&lower)
    }

    /// Check if word is a Spanish introductory conversational word (e.g. "hola", "oye", "mira").
    #[inline]
    pub fn is_spanish_introductory(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        self.index.introductory_words.contains(&lower)
    }

    /// Check if a clause starts with a Spanish compound interrogative (e.g. "por qué", "para qué", "de dónde").
    #[inline]
    pub fn starts_with_spanish_compound_interrogative(&self, clause: &str) -> bool {
        let trimmed = clause.trim_start();
        for comp in &self.index.inverted_compounds {
            if trimmed.len() >= comp.len()
                && trimmed.as_bytes()[..comp.len()].eq_ignore_ascii_case(comp.as_bytes())
            {
                let rest = &trimmed[comp.len()..];
                if rest.is_empty()
                    || rest.starts_with(char::is_whitespace)
                    || rest.starts_with(|c: char| {
                        matches!(
                            c,
                            ',' | '.' | '?' | '!' | ':' | ';' | '-' | '—' | ')' | ']' | '}' | '"' | '\''
                        )
                    })
                {
                    return true;
                }
            }
        }
        false
    }
}

/// Check if a word (without trailing period) is an ambiguous everyday word that could collide
/// with an abbreviation prefix (e.g. "no" in "yes or no" vs "No. 5", "in" in "plugged in" vs "10 in.").
#[inline]
fn is_ambiguous_common_word(w: &str) -> bool {
    w.eq_ignore_ascii_case("no")
        || w.eq_ignore_ascii_case("in")
        || w.eq_ignore_ascii_case("it")
        || w.eq_ignore_ascii_case("is")
        || w.eq_ignore_ascii_case("to")
        || w.eq_ignore_ascii_case("at")
        || w.eq_ignore_ascii_case("on")
        || w.eq_ignore_ascii_case("or")
        || w.eq_ignore_ascii_case("so")
        || w.eq_ignore_ascii_case("as")
        || w.eq_ignore_ascii_case("if")
        || w.eq_ignore_ascii_case("by")
        || w.eq_ignore_ascii_case("an")
        || w.eq_ignore_ascii_case("us")
}

/// Verify if an abbreviation prefix candidate in the text context represents a genuine abbreviation.
///
/// For ambiguous everyday words (e.g. "no.", "in."), ensures that a trailing dot is present
/// and that the contextual surroundings confirm an abbreviation usage (e.g. followed by digits,
/// numbering symbols, Roman numerals, or measurement values).
#[inline]
fn is_valid_abbreviation_context(
    text: &str,
    word_start: usize,
    end: usize,
    token: &str,
    word_no_dot: &str,
) -> bool {
    if !is_ambiguous_common_word(word_no_dot) {
        return true;
    }

    let has_dot = token.ends_with('.') || text[end..].starts_with('.');
    if !has_dot {
        return false;
    }

    let after = if token.ends_with('.') {
        &text[end..]
    } else if text[end..].starts_with('.') {
        &text[end + 1..]
    } else {
        &text[end..]
    };
    let remainder = after.trim_start();

    if word_no_dot.eq_ignore_ascii_case("no") {
        if remainder.is_empty() {
            return false;
        }
        let first = remainder.chars().next().unwrap();
        if first.is_ascii_digit() || first == '#' || first == '№' {
            return true;
        }
        let word = remainder
            .split(|c: char| c.is_whitespace() || c == ',' || c == ';' || c == '.' || c == ':' || c == ')')
            .next()
            .unwrap_or("");
        if !word.is_empty() && word.len() <= 6 {
            let is_roman = word.chars().all(|c| {
                matches!(c, 'I' | 'V' | 'X' | 'L' | 'C' | 'D' | 'M' | 'i' | 'v' | 'x' | 'l' | 'c' | 'd' | 'm')
            });
            if is_roman && !word.eq_ignore_ascii_case("i") {
                return true;
            }
            if word.eq_ignore_ascii_case("i") {
                let after_first = remainder[word.len()..].trim_start();
                if after_first.is_empty() || after_first.starts_with(|c: char| !c.is_alphabetic()) {
                    return true;
                }
            }
            if word.chars().any(|c| c.is_ascii_digit()) {
                return true;
            }
        }
        false
    } else if word_no_dot.eq_ignore_ascii_case("in") {
        let before = text[..word_start].trim_end();
        let preceded_by_digit = before.ends_with(|c: char| c.is_ascii_digit());
        let followed_by_digit = remainder.starts_with(|c: char| c.is_ascii_digit());
        preceded_by_digit || followed_by_digit
    } else {
        remainder.starts_with(|c: char| c.is_ascii_digit())
            || text[..word_start].trim_end().ends_with(|c: char| c.is_ascii_digit())
    }
}

#[inline]
pub fn is_word_boundary_at(text: &str, pos: usize) -> bool {
    if pos == text.len() {
        return true;
    }
    if pos > text.len() || !text.is_char_boundary(pos) {
        return false;
    }
    let next_byte = text.as_bytes()[pos];
    if next_byte < 0x80 {
        !next_byte.is_ascii_alphanumeric() && next_byte != b'_'
    } else if let Some(c) = text[pos..].chars().next() {
        !c.is_alphanumeric() && c != '_'
    } else {
        true
    }
}

/// Check if two strings are equal ignoring Unicode case without heap allocation.
#[inline]
pub fn eq_ignore_case(a: &str, b: &str) -> bool {
    if a.is_ascii() && b.is_ascii() {
        a.eq_ignore_ascii_case(b)
    } else {
        a.chars()
            .flat_map(|c| c.to_lowercase())
            .eq(b.chars().flat_map(|c| c.to_lowercase()))
    }
}

/// Match prefix ignoring Unicode case, returning the byte length of the matched prefix in `s`.
/// Guaranteed 0 heap allocation.
pub fn match_prefix_len_ignore_case(s: &str, prefix: &str) -> Option<usize> {
    if prefix.is_empty() {
        return Some(0);
    }
    if s.is_ascii() && prefix.is_ascii() {
        let plen = prefix.len();
        if s.len() >= plen && s.as_bytes()[..plen].eq_ignore_ascii_case(prefix.as_bytes()) {
            return Some(plen);
        }
        return None;
    }

    let mut s_char_indices = s.char_indices();
    let mut prefix_lower = prefix.chars().flat_map(|c| c.to_lowercase());

    let mut last_byte_end = 0;
    let mut current_s_lower: Option<std::char::ToLowercase> = None;

    loop {
        let p_c = match prefix_lower.next() {
            Some(c) => c,
            None => {
                if let Some(iter) = &mut current_s_lower {
                    if iter.next().is_some() {
                        return None;
                    }
                }
                return Some(last_byte_end);
            }
        };

        let s_c = if let Some(c) = current_s_lower.as_mut().and_then(|iter| iter.next()) {
            c
        } else {
            match s_char_indices.next() {
                Some((idx, ch)) => {
                    last_byte_end = idx + ch.len_utf8();
                    let mut lower = ch.to_lowercase();
                    let first = lower.next().unwrap();
                    current_s_lower = Some(lower);
                    first
                }
                None => return None,
            }
        };

        if s_c != p_c {
            return None;
        }
    }
}

/// Check if string starts with prefix ignoring Unicode case without heap allocation.
#[inline]
pub fn starts_with_ignore_case(s: &str, prefix: &str) -> bool {
    match_prefix_len_ignore_case(s, prefix).is_some()
}

/// Check if string ends with suffix ignoring Unicode case without heap allocation.
#[inline]
pub fn ends_with_ignore_case(s: &str, suffix: &str) -> bool {
    if suffix.is_empty() {
        return true;
    }
    if s.is_ascii() && suffix.is_ascii() {
        let slen = suffix.len();
        return s.len() >= slen
            && s.as_bytes()[s.len() - slen..].eq_ignore_ascii_case(suffix.as_bytes());
    }

    let suffix_char_count = suffix.chars().count();
    let s_char_count = s.chars().count();
    if s_char_count < suffix_char_count {
        return false;
    }

    let start_idx = s
        .char_indices()
        .rev()
        .nth(suffix_char_count - 1)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    eq_ignore_case(&s[start_idx..], suffix)
}

#[inline]
pub fn starts_with_ignore_ascii_case(s: &str, prefix: &str) -> bool {
    starts_with_ignore_case(s, prefix)
}

#[inline]
pub fn ends_with_ignore_ascii_case(s: &str, suffix: &str) -> bool {
    ends_with_ignore_case(s, suffix)
}

/// Compare two strings ignoring Unicode case and internal whitespace.
/// Guaranteed 0 heap allocation.
#[inline]
pub fn eq_ignore_case_and_spaces(a: &str, b: &str) -> bool {
    let mut a_chars = a.chars().filter(|c| !c.is_whitespace()).flat_map(|c| c.to_lowercase());
    let mut b_chars = b.chars().filter(|c| !c.is_whitespace()).flat_map(|c| c.to_lowercase());
    loop {
        match (a_chars.next(), b_chars.next()) {
            (Some(ca), Some(cb)) if ca == cb => continue,
            (None, None) => return true,
            _ => return false,
        }
    }
}

/// Match suffix ignoring Unicode case and whitespace variations.
/// Guaranteed 0 heap allocation.
#[inline]
pub fn match_suffix_ignore_case_and_spaces(text: &str, suffix: &str) -> Option<usize> {
    let mut t_iter = text.char_indices().rev().peekable();
    let mut s_iter = suffix.chars().rev().peekable();

    let mut first_match_byte_end = None;
    let mut last_match_byte_start = 0;

    while let Some(&s_ch) = s_iter.peek() {
        if s_ch.is_whitespace() {
            s_iter.next();
            while let Some(&(_, t_ch)) = t_iter.peek() {
                if t_ch.is_whitespace() {
                    t_iter.next();
                } else {
                    break;
                }
            }
            continue;
        }

        match t_iter.next() {
            Some((idx, t_ch)) => {
                if first_match_byte_end.is_none() {
                    first_match_byte_end = Some(idx + t_ch.len_utf8());
                }
                last_match_byte_start = idx;

                s_iter.next();
                let mut t_lower = t_ch.to_lowercase();
                let mut s_lower = s_ch.to_lowercase();
                let matches = loop {
                    match (t_lower.next(), s_lower.next()) {
                        (Some(a), Some(b)) if a == b => continue,
                        (None, None) => break true,
                        _ => break false,
                    }
                };
                if !matches {
                    return None;
                }
            }
            None => return None,
        }
    }

    first_match_byte_end.map(|end| end - last_match_byte_start)
}

/// Thread-safe registry for loaded language rule profiles.
pub struct RuleRegistry {
    profiles: RwLock<HashMap<String, Arc<LanguageRuleProfile>>>,
    pub detection_model: RwLock<Arc<CompiledDetectionModel>>,
    pub predicate_verbs: RwLock<Arc<HashSet<String>>>,
    rules_version: AtomicU64,
}

static REGISTRY: LazyLock<RuleRegistry> = LazyLock::new(RuleRegistry::default);

impl Default for RuleRegistry {
    fn default() -> Self {
        let reg = Self {
            profiles: RwLock::new(HashMap::new()),
            detection_model: RwLock::new(Arc::new(CompiledDetectionModel::default())),
            predicate_verbs: RwLock::new(Arc::new(HashSet::new())),
            rules_version: AtomicU64::new(1),
        };
        reg.load_builtin_profiles();
        reg
    }
}

impl RuleRegistry {
    /// Create a new rule registry loaded with default built-in profiles.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the compile-time built-in rule profiles from the embedded folder.
    fn load_builtin_profiles(&self) {
        self.reset_to_builtin();
    }

    /// Reset the registry profiles to the compile-time embedded profiles.
    pub fn reset_to_builtin(&self) {
        let files: Vec<_> = BuiltinRules::iter()
            .filter(|file| file.ends_with(".toml"))
            .collect();

        let builtins: HashMap<String, Arc<LanguageRuleProfile>> = files
            .into_par_iter()
            .filter_map(|file| {
                let embedded = BuiltinRules::get(&file)?;
                let content = std::str::from_utf8(&embedded.data).ok()?;
                let profile = LanguageRuleProfile::from_toml_str(content).ok()?;
                Some((profile.language.clone(), Arc::new(profile)))
            })
            .collect();

        let model = Arc::new(CompiledDetectionModel::from_profiles(&builtins));
        let mut all_verbs = HashSet::new();
        for prof in builtins.values() {
            for v in &prof.syntax.predicate_verbs {
                all_verbs.insert(v.to_lowercase());
            }
        }
        if let Ok(mut guard) = self.profiles.write() {
            *guard = builtins;
        }
        if let Ok(mut model_guard) = self.detection_model.write() {
            *model_guard = model;
        }
        if let Ok(mut verb_guard) = self.predicate_verbs.write() {
            *verb_guard = Arc::new(all_verbs);
        }
    }

    /// Return the current rules version.
    pub fn version(&self) -> u64 {
        self.rules_version.load(Ordering::Acquire)
    }

    /// Access the global singleton rule registry.
    pub fn global() -> &'static RuleRegistry {
        &REGISTRY
    }

    /// Retrieve the current compiled detection model.
    pub fn detection_model(&self) -> Arc<CompiledDetectionModel> {
        self.detection_model
            .read()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Arc::new(CompiledDetectionModel::default()))
    }

    /// Retrieve a language profile by language code (e.g. "en", "zh-Hans", "de_DE").
    /// Uses zero-heap-allocation lookup on normalized ASCII tags, with fallback to language families.
    pub fn get(&self, lang: &str) -> Option<Arc<LanguageRuleProfile>> {
        let normalized = normalize_lang_code_cow(lang);
        let guard = self.profiles.read().ok()?;
        if let Some(profile) = guard.get(normalized.as_ref()) {
            return Some(profile.clone());
        }
        if let Some(fallback) = family_fallback_for_lang(normalized.as_ref()) {
            if let Some(profile) = guard.get(fallback) {
                return Some(profile.clone());
            }
        }
        None
    }

    /// Insert or update a profile.
    pub fn insert(&self, profile: LanguageRuleProfile) {
        if let Ok(mut guard) = self.profiles.write() {
            guard.insert(profile.language.clone(), Arc::new(profile.clone()));
            let model = Arc::new(CompiledDetectionModel::from_profiles(&guard));
            if let Ok(mut model_guard) = self.detection_model.write() {
                *model_guard = model;
            }
            if let Ok(mut verb_guard) = self.predicate_verbs.write() {
                let mut new_verbs = (**verb_guard).clone();
                for v in &profile.syntax.predicate_verbs {
                    new_verbs.insert(v.to_lowercase());
                }
                *verb_guard = Arc::new(new_verbs);
            }
        }
    }

    /// Number of loaded profiles in the registry.
    pub fn len(&self) -> usize {
        self.profiles.read().map(|g| g.len()).unwrap_or(0)
    }

    /// Check whether registry has no profiles loaded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Alias for get(lang) returning language profile.
    #[inline]
    pub fn profile(&self, lang: &str) -> Option<Arc<LanguageRuleProfile>> {
        self.get(lang)
    }

    /// Check if word is a predicate verb across all registered language profiles with O(1) hash lookup.
    #[inline]
    pub fn is_predicate_verb(&self, word: &str) -> bool {
        let lower = word.to_lowercase();
        if let Ok(guard) = self.predicate_verbs.read() {
            guard.contains(&lower)
        } else {
            false
        }
    }

    /// Check if tokens around hyphen qualify for morphological protection using English / global morphology.
    #[inline]
    pub fn is_hyphen_protected(&self, prev: &str, next: &str) -> bool {
        if let Some(prof) = self.get("en") {
            prof.is_hyphen_protected(prev, next)
        } else {
            true
        }
    }

    /// Retrieve the embedded profile version for a given language tag, if present.
    pub fn embedded_version_for_lang(lang: &str) -> Option<String> {
        let normalized = normalize_lang_code(lang);
        let file_name = format!("{}.toml", normalized);
        if let Some(embedded) = BuiltinRules::get(&file_name) {
            if let Ok(content) = std::str::from_utf8(&embedded.data) {
                if let Ok(profile) = LanguageRuleProfile::from_toml_str(content) {
                    return Some(profile.version);
                }
            }
        }
        None
    }

    /// Load all `*.toml` files from the specified directory.
    /// Files loaded here override matching embedded profiles on a per-file basis.
    /// Emits a warning log if the embedded rule profile version is newer than the external profile.
    /// Returns the number of successfully loaded profiles.
    pub fn load_from_dir(&self, dir: &Path) -> usize {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return 0,
        };

        let mut count = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                if let Ok(content) = fs::read_to_string(&path) {
                    match LanguageRuleProfile::from_toml_str(&content) {
                        Ok(profile) => {
                            let lang = profile.language.clone();
                            if let Some(embedded_ver) = Self::embedded_version_for_lang(&lang) {
                                if compare_versions(&embedded_ver, &profile.version) == std::cmp::Ordering::Greater {
                                    tracing::warn!(
                                        "[lang_rules] External rule profile for '{}' (version: {}) from {} is older than embedded version ({})",
                                        lang,
                                        profile.version,
                                        path.display(),
                                        embedded_ver
                                    );
                                }
                            }

                            let is_override = self.get(&lang).is_some();
                            self.insert(profile);
                            count += 1;
                            if is_override {
                                tracing::info!(
                                    "[lang_rules] Profile for '{}' overridden from {}",
                                    lang,
                                    path.display()
                                );
                            } else {
                                tracing::info!(
                                    "[lang_rules] Profile for '{}' loaded from {}",
                                    lang,
                                    path.display()
                                );
                            }
                        }
                        Err(err) => {
                            tracing::warn!(
                                "[lang_rules] Failed to parse rule profile {}: {}",
                                path.display(),
                                err
                            );
                        }
                    }
                }
            }
        }

        count
    }

    /// Reload rule profiles: resets to embedded profiles, then merges external file-level overrides
    /// from candidate directories (e.g. `./data/lang_rules/`), and increments the version counter.
    pub fn reload_from_config(&self) -> usize {
        self.reload_from_config_with_data_dir(None)
    }

    /// Reload rule profiles with an optional explicit data directory override.
    pub fn reload_from_config_with_data_dir(&self, explicit_data_dir: Option<&Path>) -> usize {
        self.reset_to_builtin();

        let mut candidates_low_to_high = Vec::new();

        // 1. Lower-priority fallback: executable directory's lang_rules
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates_low_to_high.push(parent.join("lang_rules"));
            }
        }

        // 2. Working directory ./lang_rules
        candidates_low_to_high.push(PathBuf::from("lang_rules"));

        // 3. Executable directory's data/lang_rules
        if let Ok(exe) = std::env::current_exe() {
            if let Some(parent) = exe.parent() {
                candidates_low_to_high.push(parent.join("data").join("lang_rules"));
            }
        }

        // 4. Working directory ./data/lang_rules (standard file-level override location)
        candidates_low_to_high.push(PathBuf::from("data").join("lang_rules"));

        // 5. Configured config_dir/lang_rules (if initialized)
        if crate::config::is_initialized() {
            let config_dir = &crate::config::get().config_dir;
            candidates_low_to_high.push(config_dir.join("lang_rules"));
        }

        // 6. Explicitly passed data_dir if any (highest priority)
        if let Some(data) = explicit_data_dir {
            candidates_low_to_high.push(data.join("lang_rules"));
        }

        // Deduplicate while keeping highest-priority (last) occurrence
        let mut seen = std::collections::HashSet::new();
        let mut candidates = Vec::new();
        for dir in candidates_low_to_high.into_iter().rev() {
            let key = dir.canonicalize().unwrap_or_else(|_| dir.clone());
            if seen.insert(key) {
                candidates.push(dir);
            }
        }
        candidates.reverse();

        for dir in &candidates {
            if dir.is_dir() {
                self.load_from_dir(dir);
            }
        }

        let total_active = self.len();
        self.rules_version.fetch_add(1, Ordering::SeqCst);
        crate::utils::cache::clear();

        total_active
    }

    /// Automatically detect and initialize language rules with file-level overrides.
    pub fn auto_detect_and_init(data_dir: Option<&Path>) -> usize {
        Self::global().reload_from_config_with_data_dir(data_dir)
    }
}
/// Map unsupported small language codes to their closest linguistic family profile.
/// Enables unconfigured dialects and regional variants (e.g. Moldavian, Occitan, Faroese)
/// to leverage appropriate token density, sentence splitting, and discourse rules.
#[inline]
pub fn family_fallback_for_lang(lang: &str) -> Option<&'static str> {
    match lang {
        // South Slavic (Macedonian) -> Bulgarian profile
        "mk" => Some("bg"),
        // Balkan Romance (Moldavian) -> Romanian profile
        "mo" => Some("ro"),
        // Ibero-Romance / Occitano-Romance variants -> Catalan profile
        "ast" | "oc" => Some("ca"),
        // Insular Scandinavian (Faroese) -> Icelandic profile
        "fo" => Some("is"),
        // Indic / Indo-Aryan relatives
        "ne" | "si" => Some("hi"),
        "pa" => Some("ur"),
        // Austronesian variants -> Indonesian profile
        "ceb" => Some("id"),
        _ => None,
    }
}

/// Normalize language tag (e.g. "en-US" -> "en", "zh-Hans" -> "zh", "de_DE" -> "de").
#[inline]
pub fn normalize_lang_code(lang: &str) -> String {
    normalize_lang_code_cow(lang).into_owned()
}

/// Normalize language tag borrowing when possible to avoid heap allocation.
/// Preserves Traditional Chinese variants (zh-hant, zh-tw, zh-hk, zh-mo, cht) as "zh-hant".
#[inline]
pub fn normalize_lang_code_cow<'a>(lang: &'a str) -> Cow<'a, str> {
    let trimmed = lang.trim();
    if trimmed.is_empty() {
        return Cow::Borrowed("");
    }

    let has_sep = trimmed.contains(|c| c == '-' || c == '_');
    let has_upper = trimmed.bytes().any(|b| b.is_ascii_uppercase());

    if !has_sep && !has_upper {
        if trimmed == "cht" {
            return Cow::Borrowed("zh-hant");
        }
        return Cow::Borrowed(trimmed);
    }

    let lower = trimmed.to_ascii_lowercase();
    let lower_hyphen = if lower.contains('_') {
        lower.replace('_', "-")
    } else {
        lower
    };

    if lower_hyphen == "zh-hant"
        || lower_hyphen == "zh-tw"
        || lower_hyphen == "zh-hk"
        || lower_hyphen == "zh-mo"
        || lower_hyphen == "cht"
    {
        return Cow::Borrowed("zh-hant");
    }

    if let Some((primary, _)) = lower_hyphen.split_once('-') {
        Cow::Owned(primary.to_string())
    } else {
        Cow::Owned(lower_hyphen)
    }
}

pub use crate::utils::compare_versions;

/// Reload language rules from config for the global registry.
pub fn reload_from_config() -> usize {
    RuleRegistry::global().reload_from_config()
}

/// Access the current global language rules version.
pub fn rules_version() -> u64 {
    RuleRegistry::global().version()
}

/// Initialize and load language rules from configured directories.
/// Attempts data_dir/lang_rules first, then current working directory's lang_rules.
pub fn init_rules_from_config() -> usize {
    reload_from_config()
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_and_lookup_profile() {
        let toml_str = r#"
[metadata]
language = "en"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["Mr.", "e.g.", "i.e."]

[conjunctions]
clause_conjunctions = ["and", "but", "even though", "though"]

[discourse]
discourse_starters = ["it", "this", "they"]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();
        assert_eq!(profile.language, "en");
        assert!(profile.word_boundary);

        // Test nonbreaking prefix
        assert!(profile.is_nonbreaking_prefix("Hello Mr.", 8, 6));
        assert!(profile.is_nonbreaking_prefix("See e.g.", 7, 4));
        assert!(!profile.is_nonbreaking_prefix("Hello World.", 11, 6));

        // Test clause conjunctions
        assert!(profile.matches_clause_conjunction(" and then"));
        assert!(profile.is_conjunction_clause_marker(" even though it happened"));
        assert!(profile.matches_clause_conjunction(" \"but we know\""));
        assert!(!profile.matches_clause_conjunction(" android is an OS"));

        // Test discourse starters
        assert!(profile.is_discourse_chunk_starter("It worked fine."));
        assert!(profile.is_discourse_chunk_starter(" \"This is great.\""));
        assert!(!profile.is_discourse_chunk_starter("Item number 42."));

        // Test conditional end prefixes
        assert!(profile.is_conditional_end("etc."));
        assert!(profile.is_conditional_end("ETC."));
        assert!(profile.is_conditional_end("al."));
        assert!(profile.is_conditional_end("inc."));
        assert!(profile.is_conditional_end("usw."));
        assert!(profile.is_conditional_end("bzw."));
        assert!(!profile.is_conditional_end("Mr."));
        assert!(!profile.is_conditional_end("cat."));
    }
    #[test]
    fn test_conditional_end_and_unicode_case_insensitivity() {
        assert!(eq_ignore_case("Привет", "привет"));
        assert!(eq_ignore_case("Élève", "élève"));
        assert!(starts_with_ignore_case("Привет мир", "привет"));
        assert!(ends_with_ignore_case("мир Привет", "привет"));
        assert_eq!(
            match_prefix_len_ignore_case("Привет мир", "привет"),
            Some("Привет".len())
        );

        let toml_ru = r#"
[metadata]
language = "ru"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["т. е.", "гл."]
conditional_end_prefixes = ["т. д.", "и др."]

[conjunctions]
clause_conjunctions = ["однако", "но"]

[discourse]
discourse_starters = ["он", "это"]
"#;
        let profile_ru = LanguageRuleProfile::from_toml_str(toml_ru).unwrap();
        assert!(profile_ru.is_conditional_end("т.д."));
        assert!(profile_ru.is_conditional_end("т. д."));
        assert!(profile_ru.is_conditional_end("Т.Д."));
        assert!(profile_ru.is_conditional_end("Т. Д."));
        assert!(profile_ru.is_conditional_end("и др."));
        let ru_sent = "Книги, тетради и т. д.";
        assert!(profile_ru.is_conditional_end_at(ru_sent, ru_sent.len() - 1));
        let ru_sent_nospace = "Книги, тетради и т.д.";
        assert!(profile_ru.is_conditional_end_at(ru_sent_nospace, ru_sent_nospace.len() - 1));

        let toml_de = r#"
[metadata]
language = "de"
version = "1.0.0"

[abbreviations]
conditional_end_prefixes = ["z. B.", "usw.", "d. h."]
"#;
        let profile_de = LanguageRuleProfile::from_toml_str(toml_de).unwrap();
        assert!(profile_de.is_conditional_end("z. B."));
        assert!(profile_de.is_conditional_end("z.B."));
        assert!(profile_de.is_conditional_end("Z. B."));
        let de_sent = "Ein schönes Beispiel, z. B.";
        assert!(profile_de.is_conditional_end_at(de_sent, de_sent.len() - 1));
        let de_sent_nospace = "Ein schönes Beispiel, z.B.";
        assert!(profile_de.is_conditional_end_at(de_sent_nospace, de_sent_nospace.len() - 1));
        assert!(profile_ru.is_discourse_chunk_starter("Он пошел домой."));
        assert!(profile_ru.is_discourse_chunk_starter("это правда."));
        assert!(profile_ru.matches_clause_conjunction(" Однако все получилось."));
    }


    #[test]
    fn test_registry_builtin_and_lang_normalization() {
        let registry = RuleRegistry::new();
        assert_eq!(registry.len(), 57);
        assert!(!registry.is_empty());
        assert!(registry.get("en").is_some());
        assert!(registry.get("zh").is_some());
        assert!(registry.get("zh-hant").is_some());
        assert!(registry.get("zh-Hant").is_some());
        assert!(registry.get("zh-TW").is_some());
        assert!(registry.get("zh-HK").is_some());
        assert!(registry.get("cht").is_some());
        assert!(registry.get("ja").is_some());
        assert!(registry.get("ko").is_some());
        assert!(registry.get("de").is_some());
        assert!(registry.get("fr").is_some());
        assert!(registry.get("es").is_some());
        assert!(registry.get("ru").is_some());
        assert!(registry.get("it").is_some());
        assert!(registry.get("pt").is_some());
        assert!(registry.get("nl").is_some());
        assert!(registry.get("hi").is_some());
        assert!(registry.get("ar").is_some());
        assert!(registry.get("th").is_some());
        assert!(registry.get("tr").is_some());
        assert!(registry.get("vi").is_some());
        assert!(registry.get("pl").is_some());
        assert!(registry.get("uk").is_some());
        assert!(registry.get("cs").is_some());
        assert!(registry.get("sv").is_some());
        assert!(registry.get("fi").is_some());
        assert!(registry.get("he").is_some());
        assert!(registry.get("ro").is_some());
        assert!(registry.get("xx").is_none());
        let toml_custom = r#"
[metadata]
language = "custom-LANG"
version = "1.0.0"
word_boundary = true
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_custom).unwrap();
        assert_eq!(profile.language, "custom");
        registry.insert(profile);

        assert_eq!(registry.len(), 58);

        // Normalized lookups: "custom-LANG", "CUSTOM_LANG", "  custom  "
        assert!(registry.get("custom").is_some());
        assert!(registry.get("custom-LANG").is_some());
        assert!(registry.get("CUSTOM_LANG").is_some());
        assert!(registry.get("  custom  ").is_some());
        assert!(registry.get("fr-FR").is_some());
    }

    #[test]
    fn test_rule_registry_versioning_and_reload() {
        let registry = RuleRegistry::new();
        let v1 = registry.version();
        assert!(v1 >= 1);

        let count = registry.reload_from_config();
        assert!(count >= 11);
        let v2 = registry.version();
        assert_eq!(v2, v1 + 1);
    }

    #[test]
    fn test_cjk_profile_no_word_boundary() {
        let toml_str = r#"
[metadata]
language = "zh"
version = "1.0.0"
word_boundary = false

[conjunctions]
clause_conjunctions = ["但是", "因为", "如果"]

[discourse]
discourse_starters = ["它", "这", "因此"]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();
        assert_eq!(profile.language, "zh");
        assert!(!profile.word_boundary);

        assert!(profile.matches_clause_conjunction("但是后续流程依然顺利"));
        assert!(profile.is_conjunction_clause_marker(" 如果遇到错误"));
        assert!(profile.is_discourse_chunk_starter("它确实成功了"));
        assert!(profile.is_discourse_chunk_starter("「这非常重要」"));
    }

    #[test]
    fn test_char_boundary_defense() {
        let toml_str = r#"
[metadata]
language = "en"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["Dr.", "Mr."]

[conjunctions]
clause_conjunctions = ["and"]

[discourse]
discourse_starters = ["it"]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();

        assert!(!profile.is_nonbreaking_prefix("Hello", 100, 0));
        assert!(!profile.is_nonbreaking_prefix("Hello", 2, 4));
        assert!(!profile.is_nonbreaking_prefix("", 0, 0));

        let multibyte = "你好世界，测试字符边界";
        assert!(!profile.is_nonbreaking_prefix(multibyte, 5, 1));
        assert!(!profile.is_nonbreaking_prefix(multibyte, 1, 0));

        assert!(!profile.matches_clause_conjunction(""));
        assert!(!profile.matches_clause_conjunction("   "));
        assert!(!profile.matches_clause_conjunction("\"\""));

        assert!(!profile.is_discourse_chunk_starter(""));
        assert!(!profile.is_discourse_chunk_starter("   "));
        assert!(!profile.is_discourse_chunk_starter("\"\""));
    }

    #[test]
    fn test_word_boundary_multibyte() {
        let toml_str = r#"
[metadata]
language = "en"
version = "1.0.0"
word_boundary = true

[conjunctions]
clause_conjunctions = ["and"]

[discourse]
discourse_starters = ["it"]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();

        assert!(!profile.matches_clause_conjunction("andès"));
        assert!(!profile.matches_clause_conjunction("andeñ"));
        assert!(!profile.matches_clause_conjunction("andä"));
        assert!(!profile.matches_clause_conjunction("and中文"));

        assert!(profile.matches_clause_conjunction("and then"));
        assert!(profile.matches_clause_conjunction("and, then"));
        assert!(profile.matches_clause_conjunction("and—then"));
        assert!(profile.matches_clause_conjunction("and"));

        assert!(!profile.is_discourse_chunk_starter("itè"));
        assert!(!profile.is_discourse_chunk_starter("itemñ"));
        assert!(profile.is_discourse_chunk_starter("it is"));
        assert!(profile.is_discourse_chunk_starter("it—then"));
    }

    #[test]
    fn test_abbreviation_dot_tolerance() {
        let toml_str = r#"
[metadata]
language = "fr"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["Dr.", "Mme", "GmbH.", "z. B."]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();

        assert!(profile.is_nonbreaking_prefix("Hello Dr.", 8, 6));
        assert!(profile.is_nonbreaking_prefix("Hello Dr", 7, 6));

        assert!(profile.is_nonbreaking_prefix("Bonjour Mme.", 11, 8));
        assert!(profile.is_nonbreaking_prefix("Bonjour Mme", 10, 8));

        assert!(profile.is_nonbreaking_prefix("Firma GmbH.", 10, 6));
        assert!(profile.is_nonbreaking_prefix("Firma GmbH", 9, 6));

        assert!(profile.is_nonbreaking_prefix("Hier z. B.", 9, 8));
        assert!(profile.is_nonbreaking_prefix("Hier z. B", 8, 8));

        assert!(!profile.is_nonbreaking_prefix("Bonjour Monde.", 13, 8));
    }
    #[test]
    fn test_nonbreaking_prefix_context_and_homographs() {
        let toml_str = r#"
[metadata]
language = "en"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["No.", "no.", "Dr.", "in.", "e.g."]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_str).unwrap();

        // Standard title abbreviation
        assert!(profile.is_nonbreaking_prefix("Hello Dr. Smith", 8, 6));

        // "No." followed by digit or identifier
        let s1 = "Item No. 5 is ready";
        assert!(profile.is_nonbreaking_prefix(s1, 7, 5));
        let s2 = "Item No. IV is ready";
        assert!(profile.is_nonbreaking_prefix(s2, 7, 5));
        let s3 = "Item No. #42 is ready";
        assert!(profile.is_nonbreaking_prefix(s3, 7, 5));

        // Everyday word "no." at the end of a sentence
        let s_end = "He said no.";
        assert!(!profile.is_nonbreaking_prefix(s_end, 10, 8));

        // Everyday word "no" without dot inside sentence
        let s_mid = "There is no question.";
        assert!(!profile.is_nonbreaking_prefix(s_mid, 10, 9));

        // "in." measurement abbreviation preceded by digits
        let s_in1 = "Cable length is 10 in. approx";
        assert!(profile.is_nonbreaking_prefix(s_in1, 21, 19));

        // "in." common preposition at sentence end
        let s_in2 = "They plugged it in. Then left.";
        assert!(!profile.is_nonbreaking_prefix(s_in2, 18, 16));
    }

    #[test]
    fn test_registry_candidate_directories_merging() {
        let temp_base = std::env::temp_dir().join(format!("mtran_test_reg_{}", std::process::id()));
        let dir_fallback = temp_base.join("fallback");
        let dir_primary = temp_base.join("primary");
        let _ = fs::create_dir_all(&dir_fallback);
        let _ = fs::create_dir_all(&dir_primary);

        let toml_en_fallback = r#"[metadata]
language = "en"
version = "1.0.0"
[generation]
token_density_ratio = 1.0
"#;
        let toml_de_fallback = r#"[metadata]
language = "de"
version = "1.0.0"
[generation]
token_density_ratio = 1.3
"#;
        fs::write(dir_fallback.join("en.toml"), toml_en_fallback).unwrap();
        fs::write(dir_fallback.join("de.toml"), toml_de_fallback).unwrap();

        let toml_en_primary = r#"[metadata]
language = "en"
version = "2.0.0"
[generation]
token_density_ratio = 2.0
"#;
        fs::write(dir_primary.join("en.toml"), toml_en_primary).unwrap();

        let registry = RuleRegistry::new();
        let candidates = vec![dir_primary.clone(), dir_fallback.clone()];
        for dir in candidates.iter().rev() {
            registry.load_from_dir(dir);
        }

        let en = registry.get("en").unwrap();
        assert_eq!(en.version, "2.0.0");
        assert_eq!(en.generation.token_density_ratio, 2.0);
        let de = registry.get("de").unwrap();
        assert_eq!(de.generation.token_density_ratio, 1.3);
        assert_eq!(registry.len(), 57);
    }

    #[test]
    fn test_unicode_discourse_starters() {
        let toml_es = r#"
[metadata]
language = "es"
version = "1.0.0"
word_boundary = true

[discourse]
discourse_starters = ["él", "Él", "así", "Así"]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_es).unwrap();

        assert!(profile.is_discourse_chunk_starter("Él dijo la verdad."));
        assert!(profile.is_discourse_chunk_starter("él dijo la verdad."));
        assert!(profile.is_discourse_chunk_starter("Así fue todo."));
        assert!(profile.is_discourse_chunk_starter("así fue todo."));

        assert!(!profile.is_discourse_chunk_starter("Élite de la sociedad."));
        assert!(!profile.is_discourse_chunk_starter("asiento reservado."));
    }

    #[test]
    fn test_toml_sanitization() {
        let toml_dirty = r#"
[metadata]
language = "en-US"
version = "1.0.0"
word_boundary = true

[abbreviations]
nonbreaking_prefixes = ["", "   ", "Dr.", ""]

[conjunctions]
clause_conjunctions = ["", "and", "  "]

[discourse]
discourse_starters = ["it", "   ", ""]
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_dirty).unwrap();
        assert_eq!(profile.language, "en");
        assert_eq!(profile.nonbreaking_prefixes, vec!["Dr."]);
        assert_eq!(profile.clause_conjunctions, vec!["and"]);
        assert_eq!(profile.discourse_starters, vec!["it"]);
    }

    #[test]
    fn test_load_all_repo_profiles() {
        let registry = RuleRegistry::new();
        let loaded = registry.load_from_dir(Path::new("lang_rules"));
        assert!(loaded >= 50, "Expected at least 50 profiles loaded, got {loaded}");

        let expected_langs = [
            "en", "zh", "zh-hant", "ja", "de", "fr", "es", "ru", "it", "pt", "nl", "ko", "hi",
            "ar", "tr", "vi", "th", "id", "el", "pl", "uk", "cs", "ro", "sv", "hu", "fi", "he",
        ];
        for lang in &expected_langs {
            let prof = registry.get(lang).unwrap_or_else(|| panic!("Missing profile for {lang}"));
            assert_eq!(prof.language, *lang);
            assert!(!prof.clause_conjunctions.is_empty(), "Empty conjunctions for {lang}");
            assert!(!prof.discourse_starters.is_empty(), "Empty discourse starters for {lang}");
        }

        // Test Thai non-space segmentation property
        let th = registry.get("th").expect("th profile should exist");
        assert_eq!(th.language, "th");
        assert!(!th.word_boundary);

        // Test Arabic RTL profile
        let ar = registry.get("ar").expect("ar profile should exist");
        assert_eq!(ar.language, "ar");
        assert!(ar.word_boundary);

        // Test Polish and Ukrainian profiles
        let pl = registry.get("pl").expect("pl profile should exist");
        assert_eq!(pl.language, "pl");
        let uk = registry.get("uk").expect("uk profile should exist");
        assert_eq!(uk.language, "uk");

        // Test family fallback for regional variants
        let mk = registry.get("mk").expect("Family fallback for mk should resolve to bg");
        assert_eq!(mk.language, "bg");
        let mo = registry.get("mo").expect("Family fallback for mo should resolve to ro");
        assert_eq!(mo.language, "ro");
        let fo = registry.get("fo").expect("Family fallback for fo should resolve to is");
        assert_eq!(fo.language, "is");
        // Test Italian profile
        let it = registry.get("it").unwrap();
        assert!(it.is_nonbreaking_prefix("dott. Rossi", 4, 0));
        assert!(it.matches_clause_conjunction("perché siamo qui"));
        assert!(it.is_discourse_chunk_starter("Lui non sapeva."));

        // Test Portuguese profile
        let pt = registry.get("pt").unwrap();
        assert!(pt.is_nonbreaking_prefix("sr. Silva", 2, 0));
        assert!(pt.matches_clause_conjunction("embora seja difícil"));
        assert!(pt.is_discourse_chunk_starter("Ele disse que sim."));

        // Test Dutch profile
        let nl = registry.get("nl").unwrap();
        assert!(nl.is_nonbreaking_prefix("bijv. een voorbeeld", 4, 0));
        assert!(nl.matches_clause_conjunction("omdat het regent"));
        assert!(nl.is_discourse_chunk_starter("Hij is gekomen."));

        // Test Korean profile
        let ko = registry.get("ko").unwrap();
        assert!(ko.matches_clause_conjunction("하지만 결과는 달랐다"));
        assert!(ko.is_discourse_chunk_starter("그는 조용히 떠났다."));
        assert!(ko.is_discourse_chunk_starter("이것은 중요한 문제이다."));
    }

    #[test]
    fn test_backward_compatibility_and_defaults() {
        let minimal_toml = r#"
[metadata]
language = "xx"
version = "1.0.0"
"#;
        let profile = LanguageRuleProfile::from_toml_str(minimal_toml).unwrap();
        assert_eq!(profile.language, "xx");
        assert_eq!(profile.generation.token_density_ratio, 1.0);
        assert_eq!(profile.generation.min_length_factor, 0.5);
        assert_eq!(profile.generation.max_length_factor, 2.5);
        assert_eq!(profile.generation.repetition_penalty, None);
        assert_eq!(profile.generation.no_repeat_ngram_size, None);
        assert!(!profile.typography.pangu_spacing);
        assert!(!profile.typography.french_spacing);
        assert!(!profile.typography.spanish_inverted);
        assert_eq!(profile.typography.quote_style, None);
        assert_eq!(profile.syntax.clustering_gap_threshold, 3);
        assert_eq!(profile.syntax.head_direction, "head_initial");

        // Test shorthand aliases
        let alias_toml = r#"
[metadata]
language = "yy"
version = "1.0.0"

[generation]
density = 1.6

[typography]
pangu = true
quote = "guillemets"

[syntax]
gap = 5
"#;
        let profile_alias = LanguageRuleProfile::from_toml_str(alias_toml).unwrap();
        assert_eq!(profile_alias.generation.token_density_ratio, 1.6);
        assert_eq!(profile_alias.generation.min_length_factor, 0.5);
        assert_eq!(profile_alias.generation.max_length_factor, 2.5);
        assert!(profile_alias.typography.pangu_spacing);
        assert_eq!(profile_alias.typography.quote_style.as_deref(), Some("guillemets"));
        assert_eq!(profile_alias.syntax.clustering_gap_threshold, 5);
        assert_eq!(profile_alias.syntax.head_direction, "head_initial");
    }

    #[test]
    fn test_builtin_en_zh_generation_and_typography() {
        let registry = RuleRegistry::new();
        let en = registry.get("en").expect("Built-in EN profile missing");
        assert_eq!(en.generation.token_density_ratio, 1.0);
        assert_eq!(en.generation.min_length_factor, 0.5);
        assert_eq!(en.generation.max_length_factor, 2.5);
        assert_eq!(en.generation.repetition_penalty, Some(1.15));
        assert_eq!(en.generation.no_repeat_ngram_size, Some(3));
        assert!(!en.typography.pangu_spacing);
        assert_eq!(en.typography.quote_style.as_deref(), Some("curly"));
        assert_eq!(en.syntax.clustering_gap_threshold, 3);
        assert_eq!(en.syntax.head_direction, "head_initial");

        let zh = registry.get("zh").expect("Built-in ZH profile missing");
        assert_eq!(zh.generation.token_density_ratio, 0.65);
        assert_eq!(zh.generation.min_length_factor, 0.5);
        assert_eq!(zh.generation.max_length_factor, 2.5);
        assert_eq!(zh.generation.repetition_penalty, Some(1.15));
        assert_eq!(zh.generation.no_repeat_ngram_size, Some(5));
        assert!(zh.typography.pangu_spacing);
        assert_eq!(zh.typography.quote_style.as_deref(), Some("curly"));
        assert_eq!(zh.syntax.clustering_gap_threshold, 3);
        assert_eq!(zh.syntax.head_direction, "head_initial");
        for lang in &["ja", "ko", "de", "fr", "es", "ru", "it", "pt", "nl"] {
            assert!(registry.get(lang).is_some(), "Built-in profile for {lang} should be present");
        }
    }

    #[test]
    fn test_all_11_profiles_rules_configuration() {
        let registry = RuleRegistry::new();
        registry.load_from_dir(Path::new("lang_rules"));

        // en: density=1.0, pangu=false, quote="curly"
        let en = registry.get("en").unwrap();
        assert_eq!(en.generation.token_density_ratio, 1.0);
        assert_eq!(en.generation.repetition_penalty, Some(1.15));
        assert_eq!(en.generation.no_repeat_ngram_size, Some(3));
        assert!(!en.typography.pangu_spacing);
        assert_eq!(en.typography.quote_style.as_deref(), Some("curly"));

        // zh: density=0.65, pangu=true, quote="curly"
        let zh = registry.get("zh").unwrap();
        assert_eq!(zh.generation.token_density_ratio, 0.65);
        assert_eq!(zh.generation.repetition_penalty, Some(1.15));
        assert_eq!(zh.generation.no_repeat_ngram_size, Some(5));
        assert!(zh.typography.pangu_spacing);
        assert_eq!(zh.typography.quote_style.as_deref(), Some("curly"));

        // ja: density=1.75, pangu=true, quote="corner", syntax: gap=4, head_direction="head_final"
        let ja = registry.get("ja").unwrap();
        assert_eq!(ja.generation.token_density_ratio, 1.75);
        assert_eq!(ja.generation.repetition_penalty, Some(1.06));
        assert_eq!(ja.generation.no_repeat_ngram_size, Some(5));
        assert!(!ja.typography.pangu_spacing);
        assert_eq!(ja.typography.quote_style.as_deref(), Some("corner"));
        assert_eq!(ja.syntax.clustering_gap_threshold, 4);
        assert_eq!(ja.syntax.head_direction, "head_final");

        // de: density=1.30, quote="german", syntax: gap=4, head_direction="head_final"
        let de = registry.get("de").unwrap();
        assert_eq!(de.generation.token_density_ratio, 1.30);
        assert_eq!(de.generation.repetition_penalty, Some(1.15));
        assert_eq!(de.generation.no_repeat_ngram_size, Some(3));
        assert_eq!(de.typography.quote_style.as_deref(), Some("german"));
        assert_eq!(de.syntax.clustering_gap_threshold, 4);
        assert_eq!(de.syntax.head_direction, "head_final");

        // fr: density=1.15, french_spacing=true, quote="guillemets"
        let fr = registry.get("fr").unwrap();
        assert_eq!(fr.generation.token_density_ratio, 1.15);
        assert_eq!(fr.generation.repetition_penalty, Some(1.15));
        assert_eq!(fr.generation.no_repeat_ngram_size, Some(3));
        assert!(fr.typography.french_spacing);
        assert_eq!(fr.typography.quote_style.as_deref(), Some("guillemets"));

        // es: density=1.15, spanish_inverted=true, quote="curly"
        let es = registry.get("es").unwrap();
        assert_eq!(es.generation.token_density_ratio, 1.15);
        assert_eq!(es.generation.repetition_penalty, Some(1.15));
        assert_eq!(es.generation.no_repeat_ngram_size, Some(3));
        assert!(es.typography.spanish_inverted);
        assert_eq!(es.typography.quote_style.as_deref(), Some("curly"));

        // ru: density=1.25, quote="guillemets"
        let ru = registry.get("ru").unwrap();
        assert_eq!(ru.generation.token_density_ratio, 1.25);
        assert_eq!(ru.generation.repetition_penalty, Some(1.15));
        assert_eq!(ru.generation.no_repeat_ngram_size, Some(3));
        assert_eq!(ru.typography.quote_style.as_deref(), Some("guillemets"));

        // it: density=1.15, quote="curly"
        let it = registry.get("it").unwrap();
        assert_eq!(it.generation.token_density_ratio, 1.15);
        assert_eq!(it.generation.repetition_penalty, Some(1.15));
        assert_eq!(it.generation.no_repeat_ngram_size, Some(3));
        assert_eq!(it.typography.quote_style.as_deref(), Some("curly"));

        // pt: density=1.15, quote="curly"
        let pt = registry.get("pt").unwrap();
        assert_eq!(pt.generation.token_density_ratio, 1.15);
        assert_eq!(pt.generation.repetition_penalty, Some(1.15));
        assert_eq!(pt.generation.no_repeat_ngram_size, Some(3));
        assert_eq!(pt.typography.quote_style.as_deref(), Some("curly"));

        // nl: density=1.20, quote="curly"
        let nl = registry.get("nl").unwrap();
        assert_eq!(nl.generation.token_density_ratio, 1.20);
        assert_eq!(nl.generation.repetition_penalty, Some(1.15));
        assert_eq!(nl.generation.no_repeat_ngram_size, Some(3));
        assert_eq!(nl.typography.quote_style.as_deref(), Some("curly"));

        // ko: density=1.40, pangu=true, syntax: gap=4, head_direction="head_final"
        let ko = registry.get("ko").unwrap();
        assert_eq!(ko.generation.token_density_ratio, 1.40);
        assert_eq!(ko.generation.repetition_penalty, Some(1.06));
        assert_eq!(ko.generation.no_repeat_ngram_size, Some(5));
        assert!(!ko.typography.pangu_spacing);
        assert_eq!(ko.syntax.clustering_gap_threshold, 4);
        assert_eq!(ko.syntax.head_direction, "head_final");

        // zh-hant: density=0.65, pangu=false, quote="corner"
        let zh_hant = registry.get("zh-hant").unwrap();
        assert_eq!(zh_hant.generation.token_density_ratio, 0.65);
        assert_eq!(zh_hant.generation.repetition_penalty, Some(1.15));
        assert_eq!(zh_hant.generation.no_repeat_ngram_size, Some(5));
        assert!(!zh_hant.typography.pangu_spacing);
        assert_eq!(zh_hant.typography.quote_style.as_deref(), Some("corner"));
    }

    #[test]
    fn test_file_level_override_and_embedding() {
        let registry = RuleRegistry::new();
        // Embedded profiles are present out of the box
        let orig_zh = registry.get("zh").expect("Embedded zh profile missing");
        assert_eq!(orig_zh.generation.token_density_ratio, 0.65);
        let orig_en = registry.get("en").expect("Embedded en profile missing");
        assert_eq!(orig_en.generation.token_density_ratio, 1.0);
        let test_dir = std::env::temp_dir().join(format!("mtranserver_rules_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&test_dir);
        fs::create_dir_all(&test_dir).unwrap();

        // Write custom zh.toml overriding only the Chinese profile
        let custom_zh = r#"
[metadata]
language = "zh"
version = "2.0.0"

[generation]
density = 0.88
"#;
        fs::write(test_dir.join("zh.toml"), custom_zh).unwrap();

        // Write an additional new language profile "xx.toml"
        let custom_xx = r#"
[metadata]
language = "xx"
version = "1.0.0"

[generation]
density = 1.33
"#;
        fs::write(test_dir.join("xx.toml"), custom_xx).unwrap();

        // Perform file-level override from directory
        let loaded = registry.load_from_dir(&test_dir);
        assert_eq!(loaded, 2);

        // Verify zh was overridden
        let new_zh = registry.get("zh").unwrap();
        assert_eq!(new_zh.version, "2.0.0");
        assert_eq!(new_zh.generation.token_density_ratio, 0.88);

        // Verify xx was added
        let new_xx = registry.get("xx").unwrap();
        assert_eq!(new_xx.generation.token_density_ratio, 1.33);

        // Verify other embedded languages (e.g. en, ja) remain untouched
        let untouched_en = registry.get("en").unwrap();
        assert_eq!(untouched_en.generation.token_density_ratio, 1.0);
        let untouched_ja = registry.get("ja").unwrap();
        assert_eq!(untouched_ja.generation.token_density_ratio, 1.75);

        let _ = fs::remove_dir_all(&test_dir);
    }

    #[test]
    fn test_compare_versions_and_embedded_check() {
        assert_eq!(compare_versions("1.1.0", "1.0.0"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("1.0.0", "1.1.0"), std::cmp::Ordering::Less);
        assert_eq!(compare_versions("1.1.0", "1.1.0"), std::cmp::Ordering::Equal);
        assert_eq!(compare_versions("1.1", "1.1.0"), std::cmp::Ordering::Equal);
        assert_eq!(compare_versions("2.0.0", "1.9.9"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("v1.2.0", "1.1.0"), std::cmp::Ordering::Greater);
        assert_eq!(compare_versions("1.10.0", "1.2.0"), std::cmp::Ordering::Greater);

        let embedded_zh_ver = RuleRegistry::embedded_version_for_lang("zh").unwrap();
        assert_eq!(embedded_zh_ver, "1.1.0");

        assert_eq!(
            compare_versions(&embedded_zh_ver, "1.0.0"),
            std::cmp::Ordering::Greater
        );
        assert_eq!(
            compare_versions(&embedded_zh_ver, "2.0.0"),
            std::cmp::Ordering::Less
        );
    }

    #[test]
    fn test_nonbreaking_prefix_unicode_case_length_change() {
        let toml_str = "language = \"tr\"\nword_boundary = true\n\n[abbreviations]\nnonbreaking_prefixes = [\"ⱥ.b.\"]\n";
        let toml_str = format!("[metadata]\n{toml_str}");
        let profile = LanguageRuleProfile::from_toml_str(&toml_str).unwrap();
        let text = "Ⱥ.b.😀";
        assert!(profile.is_nonbreaking_prefix(text, 4, 0));
    }

    #[test]
    fn test_detection_rules_deserialization() {
        let toml_without_detection = r#"
[metadata]
language = "en"
version = "1.0.0"
word_boundary = true
"#;
        let profile = LanguageRuleProfile::from_toml_str(toml_without_detection).unwrap();
        assert!(profile.detection.is_none());

        let toml_with_detection = r#"
[metadata]
language = "ja"
version = "1.0.0"
word_boundary = false

[detection]
script = "Japanese"
unique_script = true
unique_chars = ['ぁ', 'ぃ']
distinctive_chars = ['の', 'は']
exclude_chars = ['a', 'b']
whole_words = ["こんにちは"]
substring_terms = ["です"]
"#;
        let profile_ja = LanguageRuleProfile::from_toml_str(toml_with_detection).unwrap();
        assert!(profile_ja.detection.is_some());
        let det = profile_ja.detection.unwrap();
        assert_eq!(det.script.as_deref(), Some("Japanese"));
        assert!(det.unique_script);
        assert_eq!(det.unique_chars, vec!['ぁ', 'ぃ']);
        assert_eq!(det.distinctive_chars, vec!['の', 'は']);
        assert_eq!(det.exclude_chars, vec!['a', 'b']);
        assert_eq!(det.whole_words, vec!["こんにちは".to_string()]);
        assert_eq!(det.substring_terms, vec!["です".to_string()]);
    }
}
