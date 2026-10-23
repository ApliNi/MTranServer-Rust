//! Marian/Bergamot NMT translation engine implemented in pure Rust.
//!
//! Reads the same model files as the original MTranServer (Mozilla Firefox
//! Translations models): a Marian binary model (.bin), lexical shortlist
//! (.s2t.bin) and SentencePiece vocabularies (.spm).
//!
//! The engine performs a full transformer encoder-decoder forward pass and
//! greedy (beam=1) decoding, replicating the C++ Bergamot v0.6.0 behavior.

pub mod error;
pub mod model;
pub mod sentencepiece;
pub mod shortlist;
pub mod markup;
pub mod rules;
pub mod tensor;
pub mod text;
pub mod transformer;
pub use error::EngineError;
/// ARM dot-product (SDOT) i8 GEMM kernels, compiled only on `aarch64`.
#[cfg(target_arch = "aarch64")]
mod arm_dotprod;

use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::utils::profiling;
pub use model::Model;
pub use sentencepiece::SentencePiece;
use sentencepiece::{DecodeScratch, EncodeScratch};
pub use shortlist::{Shortlist, ShortlistPolicy};
use transformer::ForwardConfig;
pub use transformer::PreparedEngineWeights;

/// Per-thread reusable buffers for the translation hot path.
#[derive(Default)]
struct TranslationScratch {
    encode: EncodeScratch,
    decode: DecodeScratch,
    seen: Vec<u64>,
    shortlist: Vec<u32>,
    out_tokens: Vec<u32>,
    alignments: Vec<usize>,
}
impl TranslationScratch {
    fn new() -> Self {
        Self::default()
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<TranslationScratch> =
        std::cell::RefCell::new(TranslationScratch::new());
}
/// Re-export parse formats for direct engine consumption.
pub use crate::config::ParseFormats;

/// Configuration for the translation engine.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct EngineConfig {
    pub max_length_break: usize,
    pub gemm_precision: String,
    /// Maximum characters per sentence before long-text splitting kicks in.
    pub max_sentence_length: usize,
    pub beam_size: usize,
    pub length_penalty: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub shortlist_policy: ShortlistPolicy,
    #[serde(default)]
    pub expansion_ratio: Option<f32>,
    #[serde(default)]
    pub min_length_factor: Option<f32>,
    #[serde(default)]
    pub max_length_factor: Option<f32>,
    #[serde(default)]
    pub parse_formats: ParseFormats,
    #[serde(default)]
    pub preprocess_symbols: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_length_break: 512,
            gemm_precision: "int8shiftAlphaAll".to_string(),
            max_sentence_length: 512,
            beam_size: 1,
            length_penalty: 1.0,
            repetition_penalty: 1.15,
            no_repeat_ngram_size: 3,
            shortlist_policy: ShortlistPolicy::Relaxed,
            expansion_ratio: None,
            min_length_factor: None,
            max_length_factor: None,
            parse_formats: ParseFormats::default(),
            preprocess_symbols: false,
        }
    }
}

impl EngineConfig {
    /// Construct an `EngineConfig` from the global server configuration [`crate::config::Config`].
    pub fn from_config(config: &crate::config::Config) -> Self {
        Self {
            max_length_break: 512,
            gemm_precision: "int8shiftAlphaAll".to_string(),
            max_sentence_length: config.max_sentence_length,
            beam_size: 1,
            length_penalty: 1.0,
            repetition_penalty: 1.15,
            no_repeat_ngram_size: 3,
            shortlist_policy: ShortlistPolicy::Relaxed,
            expansion_ratio: None,
            min_length_factor: None,
            max_length_factor: None,
            parse_formats: config.parse_formats,
            preprocess_symbols: config.preprocess_symbols,
        }
    }
}

impl From<&crate::config::Config> for EngineConfig {
    fn from(config: &crate::config::Config) -> Self {
        Self::from_config(config)
    }
}

/// Return current system time as a millisecond-precision Unix timestamp.
#[inline]
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Thread-safe lock-free engine usage counters and timestamps.
#[derive(Debug)]
pub struct EngineUsage {
    /// Total number of translation calls dispatched to this engine instance.
    pub call_count: AtomicU64,
    /// Unix timestamp in milliseconds when this engine was last invoked.
    pub last_used_ms: AtomicU64,
}

impl Default for EngineUsage {
    fn default() -> Self {
        Self::new()
    }
}

impl EngineUsage {
    /// Initialize a new usage tracker with zero invocations and current timestamp.
    pub fn new() -> Self {
        Self {
            call_count: AtomicU64::new(0),
            last_used_ms: AtomicU64::new(now_ms()),
        }
    }

    /// Atomically increment call count and update last used timestamp.
    #[inline]
    pub fn record_call(&self) {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        self.last_used_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Update the last used timestamp without incrementing call count.
    #[inline]
    pub fn touch(&self) {
        self.last_used_ms.store(now_ms(), Ordering::Relaxed);
    }

    /// Read current invocation count with relaxed ordering.
    #[inline]
    pub fn call_count(&self) -> u64 {
        self.call_count.load(Ordering::Relaxed)
    }

    /// Read last used Unix timestamp in milliseconds with relaxed ordering.
    #[inline]
    pub fn last_used_ms(&self) -> u64 {
        self.last_used_ms.load(Ordering::Relaxed)
    }

    /// Check if the engine has been idle for at least the given duration.
    #[inline]
    pub fn idle_for(&self, timeout: Duration) -> bool {
        let last = self.last_used_ms();
        let now = now_ms();
        now.saturating_sub(last) >= timeout.as_millis() as u64
    }
}

/// A loaded translation model bundle: model + vocabs + shortlist.
pub struct TranslationEngine {
    model: Arc<Model>,
    spm_src: Arc<SentencePiece>,
    spm_trg: Arc<SentencePiece>,
    shortlist: Arc<Shortlist>,
    forward_config: Arc<ForwardConfig>,
    prepared_weights: Arc<PreparedEngineWeights>,
    source_lang: String,
    target_lang: String,
    // Text processing state
    max_sentence_length: usize,
    shortlist_policy: ShortlistPolicy,
    parse_formats: ParseFormats,
    preprocess_symbols: bool,
    usage: Arc<EngineUsage>,
}

impl TranslationEngine {
    /// Create an engine from model files.
    pub fn new(
        source_lang: &str,
        target_lang: &str,
        model_path: &Path,
        lex_path: &Path,
        src_vocab_path: &Path,
        trg_vocab_path: &Path,
        config: EngineConfig,
    ) -> Result<Self, EngineError> {
        let max_sentence_length = config.max_sentence_length;
        let shortlist_policy = config.shortlist_policy;
        let parse_formats = config.parse_formats;
        let preprocess_symbols = config.preprocess_symbols;
        // Model, shortlist and vocab files are independent inputs (each parse
        // produces an owned value), so they load concurrently on scoped
        // threads — the cold start is bounded by the slowest file instead of
        // the sum. `ForwardConfig::resolve` needs the loaded model's YAML, so
        // it stays after the join.
        let (model, shortlist, spm_src, spm_trg) = std::thread::scope(
            |s| -> Result<(Model, Shortlist, Arc<SentencePiece>, Arc<SentencePiece>), EngineError> {
                let model_h = s.spawn(|| Model::load(model_path));
                let lex_h = s.spawn(|| Shortlist::load(lex_path));
                // Both directions of a language pair often share the same vocab
                // file; parse it once and hand out two handles instead of
                // duplicating the protobuf and its lookup table.
                let spm_h = s.spawn(
                    move || -> Result<(Arc<SentencePiece>, Arc<SentencePiece>), EngineError> {
                        if src_vocab_path == trg_vocab_path {
                            let spm = SentencePiece::load(src_vocab_path)?;
                            let spm = Arc::new(spm);
                            Ok((spm.clone(), spm))
                        } else {
                            let src = SentencePiece::load(src_vocab_path)?;
                            let trg = SentencePiece::load(trg_vocab_path)?;
                            Ok((Arc::new(src), Arc::new(trg)))
                        }
                    },
                );
                let model = model_h.join().map_err(|_| {
                    EngineError::WorkerPanicked("Model loader panicked".to_string())
                })??;
                let shortlist = lex_h.join().map_err(|_| {
                    EngineError::WorkerPanicked("Shortlist loader panicked".to_string())
                })??;
                let (spm_src, spm_trg) = spm_h.join().map_err(|_| {
                    EngineError::WorkerPanicked("Vocab loader panicked".to_string())
                })??;
                Ok((model, shortlist, spm_src, spm_trg))
            },
        )?;
        // Resolve every transformer knob (shape config + activation) once so
        // the per-request path never touches the model's config mutex. An
        // unsupported activation fails here, at model-load time.
        let forward_config = Arc::new(ForwardConfig::resolve_with_languages(
            &model,
            &config,
            source_lang,
            target_lang,
        )?);
        let model = Arc::new(model);
        let prepared_weights = Arc::new(PreparedEngineWeights::new(model.clone(), &forward_config));
        Ok(Self {
            model,
            spm_src,
            spm_trg,
            shortlist: Arc::new(shortlist),
            forward_config,
            prepared_weights,
            source_lang: source_lang.to_string(),
            target_lang: target_lang.to_string(),
            max_sentence_length,
            shortlist_policy,
            parse_formats,
            preprocess_symbols,
            usage: Arc::new(EngineUsage::new()),
        })
    }
    /// Reference to the shared atomic engine usage statistics.
    #[inline]
    pub fn usage(&self) -> &Arc<EngineUsage> {
        &self.usage
    }

    /// Associate an existing usage tracker instance with this engine.
    #[inline]
    pub fn set_usage(&mut self, usage: Arc<EngineUsage>) {
        self.usage = usage;
    }

    /// Record a translation invocation on this engine.
    #[inline]
    pub fn record_call(&self) {
        self.usage.record_call();
    }

    /// Update the last used timestamp on this engine without incrementing call count.
    #[inline]
    pub fn touch(&self) {
        self.usage.touch();
    }

    /// Last used Unix timestamp in milliseconds.
    #[inline]
    pub fn last_used_ms(&self) -> u64 {
        self.usage.last_used_ms()
    }

    /// Check whether this engine has been idle longer than the specified duration.
    #[inline]
    pub fn idle_for(&self, timeout: Duration) -> bool {
        self.usage.idle_for(timeout)
    }
    /// Source language code of this engine.
    #[inline]
    pub fn source_lang(&self) -> &str {
        &self.source_lang
    }

    /// Target language code of this engine.
    #[inline]
    pub fn target_lang(&self) -> &str {
        &self.target_lang
    }

    /// Translate a single text using configured parse formats.
    pub fn translate(&self, text: &str) -> String {
        self.translate_with_formats(text, self.parse_formats)
    }

    /// Returns the currently configured parse formats.
    #[inline]
    pub fn parse_formats(&self) -> ParseFormats {
        self.parse_formats
    }

    /// Update parse formats for subsequent translations.
    #[inline]
    pub fn set_parse_formats(&mut self, parse_formats: ParseFormats) {
        self.parse_formats = parse_formats;
    }

    /// Whether symbol preprocessing is enabled.
    #[inline]
    pub fn preprocess_symbols(&self) -> bool {
        self.preprocess_symbols
    }

    /// Enable or disable symbol preprocessing.
    #[inline]
    pub fn set_preprocess_symbols(&mut self, enable: bool) {
        self.preprocess_symbols = enable;
    }
    /// Translate a single text with caller-specified parse formats.
    pub fn translate_with_formats(
        &self,
        text: &str,
        parse_formats: crate::config::ParseFormats,
    ) -> String {
        if parse_formats.is_empty() {
            // Disabled (false/0): pass raw input directly to the model
            return self.translate_single_text(text);
        }

        markup::document::translate_document_dual_parallel_with_languages(
            text,
            parse_formats,
            Some(&self.source_lang),
            Some(&self.target_lang),
            |segment| self.translate_single_text(segment),
            |segment| self.translate_single_text_with_alignment(segment),
        )
    }
    /// Translate a single plain text or protected text segment. Handles placeholder tagging,
    /// symbol preprocessing, long-text splitting, and emoji preservation.
    fn translate_single_text(&self, text: &str) -> String {
        let normalized = text::normalize_academic_and_special_symbols(text);
        let (clean, emoji_repls) = text::hide_emojis(normalized.as_ref());

        let clean = if self.preprocess_symbols {
            text::preprocess_symbols(clean.as_ref())
        } else {
            clean
        };

        let is_oversized = text::exceeds_char_limit(clean.as_ref(), self.max_sentence_length);
        let mut translation = if is_oversized {
            self.translate_long_text(clean.as_ref())
        } else {
            self.translate_internal(clean.as_ref())
        };
        if !emoji_repls.is_empty() {
            if let Cow::Owned(s) =
                text::restore_emojis_with_source_and_spans(&translation, &emoji_repls, text, &mut [])
            {
                translation = s;
            }
        }

        text::postprocess_translation_by_language(
            &translation,
            text,
            &self.source_lang,
            &self.target_lang,
        )
        .into_owned()
    }

    /// Translate a single plain text segment with token-level cross-attention alignments.
    /// Returns `(translated_text, source_token_spans, target_token_spans, alignments)`.
    pub fn translate_single_text_with_alignment(
        &self,
        text: &str,
    ) -> markup::document::AlignmentOutput {
        let normalized = text::normalize_academic_and_special_symbols(text);
        let (clean, emoji_repls) = text::hide_emojis(normalized.as_ref());
        let clean = if self.preprocess_symbols {
            text::preprocess_symbols(clean.as_ref())
        } else {
            clean
        };

        let mut out_text = String::new();
        let mut src_spans = Vec::new();
        let mut trg_spans = Vec::new();
        let mut alignments = Vec::new();

        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => {
                self.execute_with_alignments_scratch(
                    clean.as_ref(),
                    &mut scratch,
                    &mut out_text,
                    &mut src_spans,
                    &mut trg_spans,
                    &mut alignments,
                );
            }
            Err(_) => {
                let mut local = TranslationScratch::new();
                self.execute_with_alignments_scratch(
                    clean.as_ref(),
                    &mut local,
                    &mut out_text,
                    &mut src_spans,
                    &mut trg_spans,
                    &mut alignments,
                );
            }
        });

        if !emoji_repls.is_empty() {
            if let Cow::Owned(s) =
                text::restore_emojis_with_source_and_spans(&out_text, &emoji_repls, text, &mut trg_spans)
            {
                out_text = s;
            }
            text::restore_emojis_with_source_and_spans(clean.as_ref(), &emoji_repls, text, &mut src_spans);
        }
        let out_text = text::postprocess_translation_by_language_with_spans(
            &out_text,
            text,
            &self.source_lang,
            &self.target_lang,
            &mut trg_spans,
        )
        .into_owned();

        (out_text, src_spans, trg_spans, alignments)
    }

    /// Core translation call: tokenize, run the transformer, decode.
    /// Returns the raw translated text (no post-processing).
    pub fn translate_internal(&self, text: &str) -> String {
        if text.trim().is_empty() {
            return text.to_string();
        }

        if text.contains('\n') {
            let mut result = String::with_capacity(text.len() + 16);
            for (i, line) in text.split('\n').enumerate() {
                if i > 0 {
                    result.push('\n');
                }
                if line.trim().is_empty() {
                    result.push_str(line);
                } else {
                    let trans = self.translate_internal(line);
                    result.push_str(&trans);
                }
            }
            return result;
        }

        let mut result = String::new();
        SCRATCH.with(|cell| match cell.try_borrow_mut() {
            Ok(mut scratch) => self.execute_with_scratch(text, &mut scratch, &mut result),
            Err(_) => {
                let mut local = TranslationScratch::new();
                self.execute_with_scratch(text, &mut local, &mut result);
            }
        });

        result
    }
    /// Internal pipeline helper to run tokenization, shortlist, forward, and decode.
    fn execute_with_scratch(
        &self,
        text: &str,
        scratch: &mut TranslationScratch,
        result: &mut String,
    ) {
        let ids = profiling::scope("svc/spm encode", || {
            self.spm_src.encode_into(text, &mut scratch.encode)
        });
        if ids.is_empty() {
            return;
        }
        profiling::scope("svc/shortlist", || {
            self.shortlist.generate_into(
                ids,
                self.spm_trg.vocab_size(),
                &mut scratch.seen,
                &mut scratch.shortlist,
                self.shortlist_policy,
            );
        });
        transformer::Transformer::forward_with_weights_into(
            self.prepared_weights.weights(),
            &self.spm_src,
            &self.spm_trg,
            &scratch.encode.ids,
            &scratch.shortlist,
            &self.forward_config,
            &mut scratch.out_tokens,
        );
        if scratch.out_tokens.is_empty() {
            return;
        }
        profiling::scope("svc/spm decode", || {
            result.reserve(scratch.out_tokens.len() * 4);
            self.spm_trg
                .decode_into(&scratch.out_tokens, &mut scratch.decode, result);
        });
    }
    /// Internal pipeline helper to run tokenization, shortlist, forward, and decode with alignments.
    fn execute_with_alignments_scratch(
        &self,
        text: &str,
        scratch: &mut TranslationScratch,
        result: &mut String,
        src_spans_out: &mut Vec<std::ops::Range<usize>>,
        trg_spans_out: &mut Vec<std::ops::Range<usize>>,
        alignments_out: &mut Vec<usize>,
    ) {
        let (ids, src_spans) = profiling::scope("svc/spm encode", || {
            self.spm_src.encode_with_spans_into(text, &mut scratch.encode)
        });
        if ids.is_empty() {
            return;
        }
        src_spans_out.extend_from_slice(src_spans);

        profiling::scope("svc/shortlist", || {
            self.shortlist.generate_into(
                ids,
                self.spm_trg.vocab_size(),
                &mut scratch.seen,
                &mut scratch.shortlist,
                self.shortlist_policy,
            );
        });

        scratch.alignments.clear();
        transformer::Transformer::forward_with_weights_and_alignments_into(
            self.prepared_weights.weights(),
            &self.spm_src,
            &self.spm_trg,
            &scratch.encode.ids,
            &scratch.shortlist,
            &self.forward_config,
            &mut scratch.out_tokens,
            Some(&mut scratch.alignments),
        );
        if scratch.out_tokens.is_empty() {
            return;
        }

        alignments_out.extend_from_slice(&scratch.alignments);

        profiling::scope("svc/spm decode", || {
            result.reserve(scratch.out_tokens.len() * 4);
            let trg_spans = self.spm_trg
                .decode_with_spans_into(&scratch.out_tokens, &mut scratch.decode, result);
            let start_idx = trg_spans_out.len();
            trg_spans_out.extend_from_slice(trg_spans);

            // Subword continuation alignment synchronization:
            // When a Western word is split across multiple subwords (e.g. "neg" + "rito"),
            // cross-attention on the continuation token frequently looks ahead to the subsequent
            // source token, which would tear rich-text markup spans in half. Synchronize continuation
            // alignments to the head subword's alignment to preserve morphological word integrity.
            let total_spans = trg_spans_out.len();
            for t in (start_idx + 1)..total_spans {
                if trg_spans_out[t - 1].end == trg_spans_out[t].start
                    && !trg_spans_out[t - 1].is_empty()
                    && !trg_spans_out[t].is_empty()
                {
                    let prev_last = result[trg_spans_out[t - 1].clone()].chars().next_back();
                    let next_first = result[trg_spans_out[t].clone()].chars().next();
                    if let (Some(c1), Some(c2)) = (prev_last, next_first) {
                        let is_western_word_char = |c: char| {
                            c.is_alphanumeric()
                                && !matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{3040}'..='\u{30FF}' | '\u{AC00}'..='\u{D7AF}')
                        };
                        if is_western_word_char(c1) && is_western_word_char(c2) && t < alignments_out.len() {
                            alignments_out[t] = alignments_out[t - 1];
                        }
                    }
                }
            }
        });
    }
    /// Source SentencePiece tokenizer (read-only access for diagnostics).
    pub fn spm_src(&self) -> &SentencePiece {
        &self.spm_src
    }

    /// Target vocabulary size (read-only access for diagnostics).
    pub fn trg_vocab_size(&self) -> usize {
        self.spm_trg.vocab_size()
    }

    /// Target SentencePiece tokenizer (read-only access for diagnostics).
    pub fn spm_trg(&self) -> &SentencePiece {
        &self.spm_trg
    }

    /// Lexical shortlist (read-only access for diagnostics).
    pub fn shortlist(&self) -> &Shortlist {
        &self.shortlist
    }

    /// Pre-resolved transformer layer weights.
    pub fn prepared_weights(&self) -> &PreparedEngineWeights {
        &self.prepared_weights
    }
    /// Model weights (read-only access for diagnostics).
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Forward pass configuration (read-only access for diagnostics).
    pub fn forward_config(&self) -> &ForwardConfig {
        &self.forward_config
    }
    /// Shortlist filtering policy used by this engine.
    #[inline]
    pub fn shortlist_policy(&self) -> ShortlistPolicy {
        self.shortlist_policy
    }
    /// Split long text by best separator and translate each part.
    pub fn translate_long_text(&self, text: &str) -> String {
        let separators = text::SEPARATORS;

        let mut best_sep = "";
        let mut min_max_len = usize::MAX;

        for sep in separators {
            // Skip separators that don't occur — avoids splitting for the
            // many absent candidates.
            if !text.contains(sep.sep) {
                continue;
            }
            let mut count = 0usize;
            let mut max_len = 0usize;
            let mut all_fit = true;

            for part in text.split(sep.sep) {
                count += 1;
                if text::exceeds_char_limit(part, self.max_sentence_length) {
                    all_fit = false;
                    let plen = text::char_count(part);
                    if plen > max_len {
                        max_len = plen;
                    }
                    if max_len >= min_max_len {
                        break;
                    }
                } else {
                    let plen = text::char_count(part);
                    if plen > max_len {
                        max_len = plen;
                    }
                }
            }

            if count > 1 {
                if all_fit {
                    best_sep = sep.sep;
                    break;
                }
                if max_len < min_max_len {
                    min_max_len = max_len;
                    best_sep = sep.sep;
                }
            }
        }

        let owned_parts: Vec<String>;
        let best_parts: Vec<&str> = if best_sep.is_empty() {
            owned_parts = text::chunk_by_word_boundary(text, self.max_sentence_length);
            owned_parts.iter().map(String::as_str).collect()
        } else {
            text.split(best_sep).collect()
        };

        let sep = text::mapped_separator(best_sep, &self.target_lang);

        // Translate every part, then join. Parts are independent — each one
        // runs the same encode → shortlist → forward → decode pipeline as the
        // sequential path — so they are dispatched to the shared rayon pool
        // once there is more than one. Every part's outputs depend only on its
        // own input text (no cross-part state), and joining happens in part
        // order, so results are byte-identical to the sequential loop; the
        // only shared state is the model itself, which is read-only. Each
        // task builds its own scratch buffers instead of touching this
        // thread's `SCRATCH` thread-local.
        let translated: Vec<String> = if best_parts.len() > 1 {
            translate_parts_parallel(self, &best_parts)
        } else {
            best_parts
                .iter()
                .map(|part| self.translate_part(part))
                .collect()
        };

        if translated.is_empty() {
            String::new()
        } else if translated.len() == 1 {
            translated.into_iter().next().unwrap()
        } else {
            translated.join(sep)
        }
    }

    /// Translate one already-split part: oversized chunks recurse into
    /// [`Self::translate_long_text`], everything else goes through
    /// [`Self::translate_internal`].
    fn translate_part(&self, part: &str) -> String {
        if part.trim().is_empty() {
            part.to_string()
        } else if text::exceeds_char_limit(part, self.max_sentence_length) {
            self.translate_long_text(part)
        } else {
            self.translate_internal(part)
        }
    }
}
/// Abstract translation engine trait allowing multiple inference backends
/// (native SIMD, full Transformer self-attention, and future high-capacity engines).
pub trait TranslationEngineTrait: Send + Sync {
    /// Translate input text using configured parse formats.
    fn translate(&self, text: &str) -> String;

    /// Translate input text with caller-specified markup formats.
    fn translate_with_formats(
        &self,
        text: &str,
        parse_formats: crate::config::ParseFormats,
    ) -> String;

    /// Returns target language code.
    fn target_lang(&self) -> &str;
}

impl TranslationEngineTrait for TranslationEngine {
    fn translate(&self, text: &str) -> String {
        self.translate(text)
    }

    fn translate_with_formats(
        &self,
        text: &str,
        parse_formats: crate::config::ParseFormats,
    ) -> String {
        self.translate_with_formats(text, parse_formats)
    }

    fn target_lang(&self) -> &str {
        &self.target_lang
    }
}

/// Type alias for dynamic translation engine instance.
pub type DynTranslationEngine = Arc<dyn TranslationEngineTrait>;

/// Translate independent text parts on the rayon pool, preserving order.
///
/// Results land in a pre-sized `Vec<String>` indexed by part, so the join is
/// trivially order-preserving. An oversized chunk recurses into
/// [`Self::translate_long_text`] *inside* its task, which may open a nested
/// rayon scope — supported by the shared global pool (workers keep
/// work-stealing while a scope waits, and the pool size never grows), so
/// nesting adds latency at worst, never oversubscription or deadlock. Each
/// recursion level strictly shortens the longest remaining chunk, so the
/// recursion depth is bounded.
fn translate_parts_parallel(engine: &TranslationEngine, parts: &[&str]) -> Vec<String> {
    if parts.len() <= 1 {
        return parts.iter().map(|part| engine.translate_part(part)).collect();
    }
    use rayon::prelude::*;
    parts
        .par_iter()
        .map(|part| engine.translate_part(part))
        .collect()
}
/// Fast check for Markdown formatting features before invoking full document parsing.
pub(crate) fn has_markdown_syntax(text: &str) -> bool {
    text.contains("```")
        || text.contains('`')
        || text.contains("](")
        || text.contains("![")
        || text.contains("**")
        || text.contains("~~")
        || text.contains("\n\n")
        || text.lines().any(|l| {
            let t = l.trim_start();
            t.starts_with("# ")
                || t.starts_with("## ")
                || t.starts_with("### ")
                || t.starts_with("> ")
                || t.starts_with("- ")
                || t.starts_with("* ")
                || t.starts_with("+ ")
        })
}
