//! Hybrid Document (Markdown, HTML, Frontmatter, Code Fences, Tables) AST Pipeline.
//!
//! Provides document-level block extraction, verbatim literal protection,
//! and block-level translation with inline markup projection and sentence splitting.
//!
//! Supports seamless composite documents such as GitHub Flavored Markdown with
//! embedded HTML blocks, YAML frontmatter, fenced code blocks, task lists,
//! Markdown tables, footnotes, and multiline paragraphs.

use std::ops::Range;

use crate::config::ParseFormats;

use super::inline::{extract_inline_markup, find_tag_end, parse_tag_info, InlineParseOptions};
use super::split::{
    remap_spans_to_sentences, split_sentences_with_lang,
    translate_prepared_split_sentences_dual_with_lang,
    translate_prepared_split_sentences_dual_parallel_with_lang,
    translate_prepared_split_sentences_with_alignment_and_lang,
    translate_prepared_split_sentences_with_alignment_parallel_and_lang, SplitSentence,
};
use super::{
    is_cjk_character, project_markup_spans_with_syntax, render_projected_markup,
    snap_markup_spans_to_word_boundaries, MarkupSpan,
};
/// Character length threshold beyond which multi-sentence paragraph blocks
/// are partitioned into individual sentences for neural machine translation.
pub const DEFAULT_SENTENCE_SPLIT_THRESHOLD: usize = 50;

/// Maximum allowed nesting depth for recursive document translation (blockquotes and nested HTML blocks).
pub const MAX_DOCUMENT_DEPTH: usize = 16;
/// Recognized standard HTML block-level tag names that form structural document boundaries.
#[inline]
pub fn is_html_block_tag(tag_name: &str) -> bool {
    matches!(
        tag_name,
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "canvas"
            | "dd"
            | "div"
            | "dl"
            | "dt"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "header"
            | "hr"
            | "li"
            | "main"
            | "nav"
            | "noscript"
            | "ol"
            | "p"
            | "pre"
            | "section"
            | "table"
            | "tfoot"
            | "ul"
            | "video"
            | "tr"
            | "td"
            | "th"
            | "tbody"
            | "thead"
            | "caption"
            | "details"
            | "summary"
            | "dialog"
            | "menu"
    )
}

/// A cell inside a Markdown table row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableCell {
    pub leading_ws: String,
    pub text: String,
    pub trailing_ws: String,
}

/// A parsed block within a hybrid document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DocumentBlock {
    /// Pure verbatim content that must never be sent to the translation model.
    /// Includes YAML frontmatter, fenced code blocks, thematic breaks, HTML comments, and scripts.
    Verbatim(String),

    /// A structural single-line or multi-line element with a fixed prefix, translatable content, and suffix.
    /// Used for headers (`#`), list items (`- `, `* `, `+ `, `1. `), task lists (`- [x] `, `- [ ] `),
    /// footnotes (`[^1]: `), etc.
    Structured {
        prefix: String,
        text: String,
        suffix: String,
    },

    /// A standard paragraph (possibly multiline with soft/hard line breaks) containing translatable text.
    Paragraph {
        text: String,
        suffix: String,
    },

    /// A Markdown table with a header row, a verbatim delimiter row, and zero or more body rows.
    Table {
        header_raw: String,
        delimiter_raw: String,
        body_rows: Vec<String>,
        suffix: String,
    },

    /// A Markdown blockquote (`> ...`), possibly multiline or containing nested blocks.
    Blockquote {
        raw: String,
    },

    /// An HTML block starting with an HTML block-level tag (e.g. `<div>`, `<table>`, `<details>`, `<p>`).
    HtmlBlock {
        raw: String,
    },
}

/// Standard translation output tuple containing target text, source token spans, target token spans, and alignments.
pub type AlignmentOutput = (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>);

pub type AlignmentTranslateFn<'a> =
    dyn FnMut(&str) -> AlignmentOutput + 'a;

pub type AlignmentTranslateSyncFn<'a> =
    dyn Fn(&str) -> AlignmentOutput + Sync + Send + 'a;

pub type PlainTranslateFn<'a> = dyn FnMut(&str) -> String + 'a;

pub type PlainTranslateSyncFn<'a> = dyn Fn(&str) -> String + Sync + Send + 'a;

pub enum TranslateHandler<'a> {
    SequentialDual {
        plain: &'a mut PlainTranslateFn<'a>,
        alignment: &'a mut AlignmentTranslateFn<'a>,
        lang: Option<&'a str>,
        target_lang: Option<&'a str>,
    },
    ParallelDual {
        plain: &'a PlainTranslateSyncFn<'a>,
        alignment: &'a AlignmentTranslateSyncFn<'a>,
        lang: Option<&'a str>,
        target_lang: Option<&'a str>,
    },
    SequentialSingle {
        alignment: &'a mut AlignmentTranslateFn<'a>,
        lang: Option<&'a str>,
        target_lang: Option<&'a str>,
    },
    ParallelSingle {
        alignment: &'a AlignmentTranslateSyncFn<'a>,
        lang: Option<&'a str>,
        target_lang: Option<&'a str>,
    },
}

impl<'a> TranslateHandler<'a> {
    #[inline]
    pub fn sequential_dual(
        plain: &'a mut PlainTranslateFn<'a>,
        alignment: &'a mut AlignmentTranslateFn<'a>,
    ) -> Self {
        Self::SequentialDual {
            plain,
            alignment,
            lang: None,
            target_lang: None,
        }
    }

    #[inline]
    pub fn parallel_dual(
        plain: &'a PlainTranslateSyncFn<'a>,
        alignment: &'a AlignmentTranslateSyncFn<'a>,
    ) -> Self {
        Self::ParallelDual {
            plain,
            alignment,
            lang: None,
            target_lang: None,
        }
    }

    #[inline]
    pub fn sequential_single(alignment: &'a mut AlignmentTranslateFn<'a>) -> Self {
        Self::SequentialSingle {
            alignment,
            lang: None,
            target_lang: None,
        }
    }

    #[inline]
    pub fn parallel_single(alignment: &'a AlignmentTranslateSyncFn<'a>) -> Self {
        Self::ParallelSingle {
            alignment,
            lang: None,
            target_lang: None,
        }
    }

    #[inline]
    pub fn with_lang(mut self, language: &'a str) -> Self {
        match &mut self {
            Self::SequentialDual { lang, .. }
            | Self::ParallelDual { lang, .. }
            | Self::SequentialSingle { lang, .. }
            | Self::ParallelSingle { lang, .. } => {
                *lang = Some(language);
            }
        }
        self
    }

    #[inline]
    pub fn lang(&self) -> Option<&'a str> {
        match self {
            Self::SequentialDual { lang, .. }
            | Self::ParallelDual { lang, .. }
            | Self::SequentialSingle { lang, .. }
            | Self::ParallelSingle { lang, .. } => *lang,
        }
    }

    #[inline]
    pub fn with_target_lang(mut self, language: &'a str) -> Self {
        match &mut self {
            Self::SequentialDual { target_lang, .. }
            | Self::ParallelDual { target_lang, .. }
            | Self::SequentialSingle { target_lang, .. }
            | Self::ParallelSingle { target_lang, .. } => {
                *target_lang = Some(language);
            }
        }
        self
    }

    #[inline]
    pub fn target_lang(&self) -> Option<&'a str> {
        match self {
            Self::SequentialDual { target_lang, .. }
            | Self::ParallelDual { target_lang, .. }
            | Self::SequentialSingle { target_lang, .. }
            | Self::ParallelSingle { target_lang, .. } => *target_lang,
        }
    }

    #[inline]
    pub fn translate_plain(&mut self, text: &str) -> String {
        match self {
            TranslateHandler::SequentialDual { plain, .. } => plain(text),
            TranslateHandler::ParallelDual { plain, .. } => plain(text),
            TranslateHandler::SequentialSingle { alignment, .. } => alignment(text).0,
            TranslateHandler::ParallelSingle { alignment, .. } => alignment(text).0,
        }
    }

    #[inline]
    pub fn translate_alignment(&mut self, text: &str) -> AlignmentOutput {
        match self {
            TranslateHandler::SequentialDual { alignment, .. } => alignment(text),
            TranslateHandler::ParallelDual { alignment, .. } => alignment(text),
            TranslateHandler::SequentialSingle { alignment, .. } => alignment(text),
            TranslateHandler::ParallelSingle { alignment, .. } => alignment(text),
        }
    }

    #[inline]
    pub fn translate_prepared_sentences(&mut self, sentences: Vec<SplitSentence<'_>>) -> String {
        let target_lang = self.target_lang().or_else(|| self.lang());
        match self {
            TranslateHandler::SequentialDual {
                plain, alignment, ..
            } => translate_prepared_split_sentences_dual_with_lang(sentences, target_lang, &mut **plain, &mut **alignment),
            TranslateHandler::ParallelDual {
                plain, alignment, ..
            } => translate_prepared_split_sentences_dual_parallel_with_lang(sentences, target_lang, *plain, *alignment),
            TranslateHandler::SequentialSingle { alignment, .. } => {
                translate_prepared_split_sentences_with_alignment_and_lang(sentences, target_lang, &mut **alignment)
            }
            TranslateHandler::ParallelSingle { alignment, .. } => {
                translate_prepared_split_sentences_with_alignment_parallel_and_lang(sentences, target_lang, *alignment)
            }
        }
    }

    #[inline]
    pub fn is_parallel(&self) -> bool {
        matches!(
            self,
            TranslateHandler::ParallelDual { .. } | TranslateHandler::ParallelSingle { .. }
        )
    }

    #[inline]
    pub fn reborrow(&mut self) -> TranslateHandler<'_> {
        match self {
            TranslateHandler::SequentialDual {
                plain,
                alignment,
                lang,
                target_lang,
            } => TranslateHandler::SequentialDual {
                plain: *plain,
                alignment: *alignment,
                lang: *lang,
                target_lang: *target_lang,
            },
            TranslateHandler::ParallelDual {
                plain,
                alignment,
                lang,
                target_lang,
            } => TranslateHandler::ParallelDual {
                plain: *plain,
                alignment: *alignment,
                lang: *lang,
                target_lang: *target_lang,
            },
            TranslateHandler::SequentialSingle {
                alignment,
                lang,
                target_lang,
            } => TranslateHandler::SequentialSingle {
                alignment: *alignment,
                lang: *lang,
                target_lang: *target_lang,
            },
            TranslateHandler::ParallelSingle {
                alignment,
                lang,
                target_lang,
            } => TranslateHandler::ParallelSingle {
                alignment: *alignment,
                lang: *lang,
                target_lang: *target_lang,
            },
        }
    }
}

/// Translate a composite document (Markdown + HTML + Frontmatter + Tables + Code Fences)
/// while preserving formatting, verbatim code, and projecting inline markup via token alignments.
pub fn translate_document_with_alignment<F>(
    doc: &str,
    parse_formats: ParseFormats,
    mut translate_fn: F,
) -> String
where
    F: FnMut(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>),
{
    translate_document_with_alignment_dyn(doc, parse_formats, &mut translate_fn)
}

/// Dynamic dispatch implementation of composite document translation to prevent monomorphization recursion.
pub fn translate_document_with_alignment_dyn(
    doc: &str,
    parse_formats: ParseFormats,
    translate_fn: &mut AlignmentTranslateFn<'_>,
) -> String {
    let mut handler = TranslateHandler::sequential_single(translate_fn);
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}

/// Translate a composite document (Markdown + HTML + Frontmatter + Tables + Code Fences)
/// concurrently across multiple threads via Rayon.
pub fn translate_document_with_alignment_parallel<F>(
    doc: &str,
    parse_formats: ParseFormats,
    translate_fn: F,
) -> String
where
    F: Fn(&str) -> (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>) + Sync + Send,
{
    translate_document_with_alignment_parallel_dyn(doc, parse_formats, &translate_fn)
}

/// Dynamic dispatch implementation of parallel composite document translation.
pub fn translate_document_with_alignment_parallel_dyn(
    doc: &str,
    parse_formats: ParseFormats,
    translate_fn: &AlignmentTranslateSyncFn<'_>,
) -> String {
    let mut handler = TranslateHandler::parallel_single(translate_fn);
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}

/// Translate a composite document (Markdown + HTML + Frontmatter + Tables + Code Fences)
/// using dual-track translation: plain text uses `plain_fn` directly,
/// while text with inline markup uses `align_fn` with token alignment markup projection.
pub fn translate_document_dual<P, A>(
    doc: &str,
    parse_formats: ParseFormats,
    mut plain_fn: P,
    mut align_fn: A,
) -> String
where
    P: FnMut(&str) -> String,
    A: FnMut(&str) -> AlignmentOutput,
{
    translate_document_dual_dyn(doc, parse_formats, &mut plain_fn, &mut align_fn)
}

pub fn translate_document_dual_dyn(
    doc: &str,
    parse_formats: ParseFormats,
    plain_fn: &mut PlainTranslateFn<'_>,
    align_fn: &mut AlignmentTranslateFn<'_>,
) -> String {
    let mut handler = TranslateHandler::sequential_dual(plain_fn, align_fn);
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}
/// Translate a composite document sequentially using dual-track translation with explicit languages.
pub fn translate_document_dual_with_languages<P, A>(
    doc: &str,
    parse_formats: ParseFormats,
    source_lang: Option<&str>,
    target_lang: Option<&str>,
    mut plain_fn: P,
    mut align_fn: A,
) -> String
where
    P: FnMut(&str) -> String,
    A: FnMut(&str) -> AlignmentOutput,
{
    let mut handler = TranslateHandler::sequential_dual(&mut plain_fn, &mut align_fn);
    if let Some(l) = source_lang {
        handler = handler.with_lang(l);
    }
    if let Some(tl) = target_lang {
        handler = handler.with_target_lang(tl);
    }
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}


/// Translate a composite document concurrently across multiple threads via Rayon using dual-track translation.
pub fn translate_document_dual_parallel<P, A>(
    doc: &str,
    parse_formats: ParseFormats,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> AlignmentOutput + Sync + Send,
{
    translate_document_dual_parallel_with_lang(doc, parse_formats, None, plain_fn, align_fn)
}

pub fn translate_document_dual_parallel_with_lang<P, A>(
    doc: &str,
    parse_formats: ParseFormats,
    lang: Option<&str>,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> AlignmentOutput + Sync + Send,
{
    translate_document_dual_parallel_with_languages(doc, parse_formats, lang, None, plain_fn, align_fn)
}

/// Translate a composite document concurrently across multiple threads via Rayon using dual-track translation
/// with explicit source and target language rule profiles.
pub fn translate_document_dual_parallel_with_languages<P, A>(
    doc: &str,
    parse_formats: ParseFormats,
    source_lang: Option<&str>,
    target_lang: Option<&str>,
    plain_fn: P,
    align_fn: A,
) -> String
where
    P: Fn(&str) -> String + Sync + Send,
    A: Fn(&str) -> AlignmentOutput + Sync + Send,
{
    let mut handler = TranslateHandler::parallel_dual(&plain_fn, &align_fn);
    if let Some(l) = source_lang {
        handler = handler.with_lang(l);
    }
    if let Some(tl) = target_lang {
        handler = handler.with_target_lang(tl);
    }
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}

pub fn translate_document_dual_parallel_dyn(
    doc: &str,
    parse_formats: ParseFormats,
    plain_fn: &PlainTranslateSyncFn<'_>,
    align_fn: &AlignmentTranslateSyncFn<'_>,
) -> String {
    let mut handler = TranslateHandler::parallel_dual(plain_fn, align_fn);
    translate_document_internal(doc, parse_formats, &mut handler, 0)
}

/// Translate a document with a pre-configured [`TranslateHandler`].
pub fn translate_document_with_handler(
    doc: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
) -> String {
    translate_document_internal(doc, parse_formats, handler, 0)
}

fn translate_document_internal(
    doc: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
    depth: usize,
) -> String {
    if doc.is_empty() {
        return String::new();
    }

    if depth >= MAX_DOCUMENT_DEPTH {
        return translate_translatable_content(doc, parse_formats, handler);
    }

    if parse_formats.is_empty() {
        return handler.translate_plain(doc);
    }

    let blocks = parse_document_blocks(doc);

    if blocks.len() > 1 {
        match handler {
            TranslateHandler::ParallelDual { plain, alignment, lang, target_lang } => {
                use rayon::prelude::*;
                let p = *plain;
                let a = *alignment;
                let l = *lang;
                let tl = *target_lang;
                let pieces: Vec<String> = blocks
                    .into_par_iter()
                    .map(|block| {
                        let mut h = TranslateHandler::ParallelDual {
                            plain: p,
                            alignment: a,
                            lang: l,
                            target_lang: tl,
                        };
                        translate_block(block, parse_formats, &mut h, depth)
                    })
                    .collect();
                return pieces.concat();
            }
            TranslateHandler::ParallelSingle { alignment, lang, target_lang } => {
                use rayon::prelude::*;
                let a = *alignment;
                let l = *lang;
                let tl = *target_lang;
                let pieces: Vec<String> = blocks
                    .into_par_iter()
                    .map(|block| {
                        let mut h = TranslateHandler::ParallelSingle {
                            alignment: a,
                            lang: l,
                            target_lang: tl,
                        };
                        translate_block(block, parse_formats, &mut h, depth)
                    })
                    .collect();
                return pieces.concat();
            }
            _ => {}
        }
    }

    let mut out = String::with_capacity(doc.len() + 64);
    for block in blocks {
        out.push_str(&translate_block(block, parse_formats, handler, depth));
    }

    out
}

fn translate_block(
    block: DocumentBlock,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
    depth: usize,
) -> String {
    let mut out = String::new();
    match block {
        DocumentBlock::Verbatim(raw) => {
            out.push_str(&raw);
        }
        DocumentBlock::Structured {
            prefix,
            text,
            suffix,
        } => {
            out.push_str(&prefix);
            let translated =
                translate_translatable_content(&text, parse_formats, handler);
            out.push_str(&translated);
            out.push_str(&suffix);
        }
        DocumentBlock::Paragraph { text, suffix } => {
            let translated =
                translate_translatable_content(&text, parse_formats, handler);
            out.push_str(&translated);
            out.push_str(&suffix);
        }
        DocumentBlock::Table {
            header_raw,
            delimiter_raw,
            body_rows,
            suffix,
        } => {
            let translated_header =
                translate_table_row(&header_raw, parse_formats, handler);
            out.push_str(&translated_header);
            out.push_str(&delimiter_raw);
            for body_row in body_rows {
                let translated_row =
                    translate_table_row(&body_row, parse_formats, handler);
                out.push_str(&translated_row);
            }
            out.push_str(&suffix);
        }
        DocumentBlock::Blockquote { raw } => {
            let translated_bq =
                translate_blockquote_block(&raw, parse_formats, handler, depth);
            out.push_str(&translated_bq);
        }
        DocumentBlock::HtmlBlock { raw } => {
            let translated_html = translate_html_block(&raw, parse_formats, handler, depth);
            out.push_str(&translated_html);
        }
    }
    out
}


/// Translate a single translatable text segment by extracting inline markup,
/// splitting into sentences if threshold is met, calling the translation model,
/// and projecting markup spans back onto target text.
fn translate_translatable_content(
    raw_text: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
) -> String {
    if raw_text.trim().is_empty() {
        return raw_text.to_string();
    }

    let leading_ws_len = raw_text.len() - raw_text.trim_start().len();
    let leading_ws = &raw_text[..leading_ws_len];
    let trailing_ws_start = raw_text.trim_end().len();
    let trailing_ws = &raw_text[trailing_ws_start..];
    let trimmed_text = &raw_text[leading_ws_len..trailing_ws_start];

    let options = InlineParseOptions {
        parse_html: parse_formats.html,
        parse_markdown: parse_formats.md,
    };

    let (plain_text, mut spans) = extract_inline_markup(trimmed_text, options);

    if plain_text.trim().is_empty() {
        return raw_text.to_string();
    }

    let should_split =
        plain_text.contains('\n') || plain_text.len() >= DEFAULT_SENTENCE_SPLIT_THRESHOLD;
    let translated_content = if should_split {
        let mut sentences = split_sentences_with_lang(&plain_text, handler.lang());
        if sentences.len() > 1 {
            if !spans.is_empty() {
                remap_spans_to_sentences(&mut sentences, &spans);
            }
            handler.translate_prepared_sentences(sentences)
        } else {
            translate_single_unit(&plain_text, &mut spans, handler)
        }
    } else {
        translate_single_unit(&plain_text, &mut spans, handler)
    };

    let mut out =
        String::with_capacity(leading_ws.len() + translated_content.len() + trailing_ws.len());
    out.push_str(leading_ws);
    out.push_str(&translated_content);
    out.push_str(trailing_ws);
    out
}

#[inline]
fn translate_single_unit(
    plain_text: &str,
    spans: &mut [MarkupSpan],
    handler: &mut TranslateHandler<'_>,
) -> String {
    if spans.is_empty() {
        handler.translate_plain(plain_text)
    } else {
        let (target_text, src_spans, trg_spans, alignments) =
            handler.translate_alignment(plain_text);
        let target_lang = handler.target_lang().or_else(|| handler.lang());
        let profile = target_lang.and_then(|l| crate::engine::rules::RuleRegistry::global().get(l));
        let syntax = profile.as_ref().map(|p| &p.syntax);
        project_markup_spans_with_syntax(
            spans,
            &src_spans,
            &trg_spans,
            &alignments,
            target_text.len(),
            syntax,
        );
        snap_markup_spans_to_word_boundaries(spans, &target_text);
        render_projected_markup(&target_text, spans)
    }
}

/// Translate cells in a Markdown table row while preserving pipe delimiters and cell spacing.
fn translate_table_row(
    row: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
) -> String {
    let (content, ending) = split_line_ending(row);
    let has_leading_pipe = content.trim_start().starts_with('|');
    let has_trailing_pipe = content.trim_end().ends_with('|');

    let parts: Vec<&str> = content.split('|').collect();
    if parts.is_empty() {
        return row.to_string();
    }

    let start_idx = if has_leading_pipe { 1 } else { 0 };
    let end_idx = if has_trailing_pipe {
        parts.len().saturating_sub(1)
    } else {
        parts.len()
    };

    let mut out = String::with_capacity(row.len() + 32);
    if has_leading_pipe {
        out.push('|');
    }

    if start_idx < end_idx {
        for (k, cell_raw) in parts[start_idx..end_idx].iter().enumerate() {
            let leading_ws_len = cell_raw.len() - cell_raw.trim_start().len();
            let leading_ws = &cell_raw[..leading_ws_len];
            let trailing_ws_start = cell_raw.trim_end().len();
            let trailing_ws = &cell_raw[trailing_ws_start..];
            let cell_text = &cell_raw[leading_ws_len..trailing_ws_start];

            out.push_str(leading_ws);
            if !cell_text.is_empty() {
                let translated =
                    translate_translatable_content(cell_text, parse_formats, handler);
                out.push_str(&translated);
            }
            out.push_str(trailing_ws);

            if k + 1 < (end_idx - start_idx) || has_trailing_pipe {
                out.push('|');
            }
        }
    }

    out.push_str(ending);
    out
}

/// Translate a blockquote group by stripping `>` markers, translating the inner content,
/// and re-prefixing each line with `>` or `> `.
fn translate_blockquote_block(
    raw: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
    depth: usize,
) -> String {
    let lines = split_lines_with_endings(raw);
    let mut stripped = String::with_capacity(raw.len());

    for line in &lines {
        let (content, ending) = split_line_ending(line);
        let trimmed_start = content.trim_start();
        if let Some(after_gt) = trimmed_start.strip_prefix('>') {
            let inner = after_gt.strip_prefix(' ').unwrap_or(after_gt);
            stripped.push_str(inner);
        } else {
            stripped.push_str(content);
        }
        stripped.push_str(ending);
    }

    let translated_inner =
        translate_document_internal(&stripped, parse_formats, &mut handler.reborrow(), depth + 1);

    // Re-apply `> ` prefix to each line
    let translated_lines = split_lines_with_endings(&translated_inner);
    let mut out = String::with_capacity(translated_inner.len() + translated_lines.len() * 2);

    for line in translated_lines {
        let (content, ending) = split_line_ending(line);
        if content.trim().is_empty() {
            out.push('>');
        } else {
            out.push_str("> ");
            out.push_str(content);
        }
        out.push_str(ending);
    }

    out
}

/// Translate an HTML block while preserving opening/closing block tags, HTML comments,
/// and translating text segments with inline markup.
fn translate_html_block(
    raw: &str,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
    depth: usize,
) -> String {
    let trimmed = raw.trim_start();
    if trimmed.starts_with("<!--")
        || trimmed.starts_with("<script")
        || trimmed.starts_with("<style")
    {
        return raw.to_string();
    }

    let mut out = String::with_capacity(raw.len() + 64);
    let bytes = raw.as_bytes();
    let len = bytes.len();
    let mut i = 0;
    let mut text_buf = String::new();

    while i < len {
        if bytes[i] == b'<' {
            let is_tag = i + 1 < len
                && matches!(bytes[i + 1], b'a'..=b'z' | b'A'..=b'Z' | b'!' | b'?' | b'/');
            if is_tag {
                if let Some(tag_end) = find_tag_end(bytes, i) {
                    let tag_str = &raw[i..tag_end];
                    let (_is_closing, tag_name, _is_self_closing) = parse_tag_info(tag_str);

                    let is_block_tag = is_html_block_tag(tag_name.as_str());

                    if is_block_tag {
                        // Flush accumulated text buffer as translatable content
                        flush_html_text_buffer(
                            &mut out,
                            &mut text_buf,
                            parse_formats,
                            handler,
                            depth,
                        );
                        out.push_str(tag_str);
                        i = tag_end;
                        continue;
                    } else {
                        // Inline HTML tag: accumulate in text buffer directly
                        text_buf.push_str(tag_str);
                        i = tag_end;
                        continue;
                    }
                }
            }
            text_buf.push('<');
            i += 1;
        }

        // Fast slice non-tag text up to the next '<'
        let next_tag = match raw[i..].find('<') {
            Some(rel) => i + rel,
            None => len,
        };
        if next_tag > i {
            text_buf.push_str(&raw[i..next_tag]);
            i = next_tag;
        }
    }

    // Flush any remaining text buffer
    flush_html_text_buffer(&mut out, &mut text_buf, parse_formats, handler, depth);
    out
}

/// Flush accumulated text buffer from an HTML block, translating if non-empty.
fn flush_html_text_buffer(
    out: &mut String,
    text_buf: &mut String,
    parse_formats: ParseFormats,
    handler: &mut TranslateHandler<'_>,
    depth: usize,
) {
    if text_buf.is_empty() {
        return;
    }

    if text_buf.trim().is_empty() {
        out.push_str(text_buf);
        text_buf.clear();
        return;
    }

    // Check if the text inside the HTML block has Markdown sub-blocks (e.g. lists, blank-line separated paragraphs)
    if contains_markdown_blocks(text_buf) {
        let translated =
            translate_document_internal(text_buf, parse_formats, &mut handler.reborrow(), depth + 1);
        out.push_str(&translated);
    } else {
        let translated =
            translate_translatable_content(text_buf, parse_formats, handler);
        out.push_str(&translated);
    }

    text_buf.clear();
}

/// Check if text inside an HTML container contains Markdown blocks (paragraphs, lists, headers).
fn contains_markdown_blocks(text: &str) -> bool {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() <= 1 {
        return false;
    }

    let mut blank_line_found = false;
    for line in &lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            blank_line_found = true;
        } else if blank_line_found {
            // Content after a blank line inside an HTML block (like `<details>`) indicates Markdown
            return true;
        }
        if trimmed.starts_with("- ")
            || trimmed.starts_with("* ")
            || trimmed.starts_with("+ ")
            || (trimmed.starts_with('#')
                && {
                    let hash_count = trimmed.chars().take_while(|&c| c == '#').count();
                    hash_count <= 6 && trimmed[hash_count..].starts_with(' ')
                })
        {
            return true;
        }
    }

    false
}

/// Parse a composite document into discrete [`DocumentBlock`] AST units.
#[allow(clippy::needless_range_loop)]
pub fn parse_document_blocks(doc: &str) -> Vec<DocumentBlock> {
    if doc.is_empty() {
        return Vec::new();
    }

    let lines = split_lines_with_endings(doc);
    let mut blocks = Vec::new();
    let num_lines = lines.len();
    let mut i = 0;

    // 1. YAML Frontmatter (document head starting at line 0 with `---`)
    if num_lines > 0 {
        let (content, _) = split_line_ending(lines[0]);
        if content.trim() == "---" {
            let mut end_idx = None;
            for j in 1..num_lines {
                let (line_content, _) = split_line_ending(lines[j]);
                let trimmed = line_content.trim();
                if trimmed == "---" || trimmed == "..." {
                    end_idx = Some(j);
                    break;
                }
            }

            if let Some(j) = end_idx {
                let frontmatter = lines[0..=j].concat();
                blocks.push(DocumentBlock::Verbatim(frontmatter));
                i = j + 1;
            }
        }
    }

    while i < num_lines {
        let line = lines[i];
        let (content, ending) = split_line_ending(line);
        let trimmed_start = content.trim_start();

        // 2. Empty line: preserve verbatim
        if content.trim().is_empty() {
            blocks.push(DocumentBlock::Verbatim(line.to_string()));
            i += 1;
            continue;
        }

        // 3. Fenced Code Block (``` or ~~~)
        if trimmed_start.starts_with("```") || trimmed_start.starts_with("~~~") {
            let fence_char = if trimmed_start.starts_with("```") {
                '`'
            } else {
                '~'
            };
            let fence_len = trimmed_start
                .chars()
                .take_while(|&c| c == fence_char)
                .count();

            if fence_len >= 3 {
                let mut end_idx = num_lines - 1;
                for j in (i + 1)..num_lines {
                    let (code_content, _) = split_line_ending(lines[j]);
                    let closing_start = code_content.trim_start();
                    let close_count = closing_start
                        .chars()
                        .take_while(|&c| c == fence_char)
                        .count();
                    if close_count >= fence_len {
                        end_idx = j;
                        break;
                    }
                }

                let code_block = lines[i..=end_idx].concat();
                blocks.push(DocumentBlock::Verbatim(code_block));
                i = end_idx + 1;
                continue;
            }
        }

        // 4. HTML Block
        if let Some((end_idx, is_verbatim)) = match_html_block(&lines, i) {
            let raw_html = lines[i..=end_idx].concat();
            if is_verbatim {
                blocks.push(DocumentBlock::Verbatim(raw_html));
            } else {
                blocks.push(DocumentBlock::HtmlBlock { raw: raw_html });
            }
            i = end_idx + 1;
            continue;
        }

        // 5. Markdown Table (header + delimiter line + body rows)
        if content.contains('|')
            && i + 1 < num_lines
            && is_table_delimiter_row(split_line_ending(lines[i + 1]).0)
        {
            let header_raw = lines[i].to_string();
            let delimiter_raw = lines[i + 1].to_string();
            let mut body_rows = Vec::new();
            let mut j = i + 2;

            while j < num_lines {
                let (body_content, _) = split_line_ending(lines[j]);
                if body_content.contains('|')
                    && !body_content.trim().is_empty()
                    && !is_table_delimiter_row(body_content)
                {
                    body_rows.push(lines[j].to_string());
                    j += 1;
                } else {
                    break;
                }
            }

            blocks.push(DocumentBlock::Table {
                header_raw,
                delimiter_raw,
                body_rows,
                suffix: String::new(),
            });
            i = j;
            continue;
        }

        // 6. Thematic Break / Horizontal Rule (---, ***, ___)
        if is_thematic_break(content.trim()) {
            blocks.push(DocumentBlock::Verbatim(line.to_string()));
            i += 1;
            continue;
        }

        // 7. Header (# ~ ######)
        if trimmed_start.starts_with('#') {
            let hash_count = trimmed_start.chars().take_while(|&c| c == '#').count();
            if hash_count <= 6 && trimmed_start[hash_count..].starts_with(' ') {
                let leading_ws = &content[..content.len() - trimmed_start.len()];
                let prefix = format!("{}{}{} ", leading_ws, "#".repeat(hash_count), "");
                let text = trimmed_start[hash_count + 1..].to_string();
                blocks.push(DocumentBlock::Structured {
                    prefix,
                    text,
                    suffix: ending.to_string(),
                });
                i += 1;
                continue;
            }
        }

        // 8. Blockquote (> ...)
        if trimmed_start.starts_with('>') {
            let mut j = i;
            while j < num_lines {
                let (bq_content, _) = split_line_ending(lines[j]);
                let bq_start = bq_content.trim_start();
                if bq_start.starts_with('>') {
                    j += 1;
                } else {
                    break;
                }
            }

            let bq_raw = lines[i..j].concat();
            blocks.push(DocumentBlock::Blockquote { raw: bq_raw });
            i = j;
            continue;
        }

        // 9. Task List Item (- [x] , - [ ] , * [x] , etc.)
        if let Some((prefix, text)) = try_parse_task_list_item(content) {
            blocks.push(DocumentBlock::Structured {
                prefix,
                text,
                suffix: ending.to_string(),
            });
            i += 1;
            continue;
        }

        // 10. Unordered List Item (- , * , + )
        if (trimmed_start.starts_with("- ")
            || trimmed_start.starts_with("* ")
            || trimmed_start.starts_with("+ "))
            && !is_thematic_break(content.trim())
        {
            let bullet = trimmed_start.chars().next().unwrap();
            let leading_ws = &content[..content.len() - trimmed_start.len()];
            let prefix = format!("{}{}{} ", leading_ws, bullet, "");
            let text = trimmed_start[2..].to_string();
            blocks.push(DocumentBlock::Structured {
                prefix,
                text,
                suffix: ending.to_string(),
            });
            i += 1;
            continue;
        }

        // 11. Ordered List Item (1. , 2. , etc.)
        let digit_count = trimmed_start.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_count > 0 && trimmed_start[digit_count..].starts_with(". ") {
            let num_str = &trimmed_start[..digit_count];
            let leading_ws = &content[..content.len() - trimmed_start.len()];
            let prefix = format!("{}{}. ", leading_ws, num_str);
            let text = trimmed_start[digit_count + 2..].to_string();
            blocks.push(DocumentBlock::Structured {
                prefix,
                text,
                suffix: ending.to_string(),
            });
            i += 1;
            continue;
        }

        // 12. Footnote Definition ([^1]: ...)
        if let Some((prefix, text)) = try_parse_footnote_def(content) {
            blocks.push(DocumentBlock::Structured {
                prefix,
                text,
                suffix: ending.to_string(),
            });
            i += 1;
            continue;
        }

        // 13. General Paragraph (accumulates multiline soft breaks)
        let (para_text, para_suffix, next_i) = accumulate_paragraph(&lines, i);
        blocks.push(DocumentBlock::Paragraph {
            text: para_text,
            suffix: para_suffix,
        });
        i = next_i;
    }

    blocks
}

/// Case-insensitive ASCII substring search returning starting byte offset.
#[inline]
fn find_ignore_ascii_case(haystack: &str, needle: &str) -> Option<usize> {
    let h_bytes = haystack.as_bytes();
    let n_bytes = needle.as_bytes();
    let n_len = n_bytes.len();
    if n_len == 0 {
        return Some(0);
    }
    if h_bytes.len() < n_len {
        return None;
    }
    let first_lower = n_bytes[0].to_ascii_lowercase();
    let first_upper = n_bytes[0].to_ascii_uppercase();
    let end = h_bytes.len() - n_len;

    for i in 0..=end {
        let b = h_bytes[i];
        if (b == first_lower || b == first_upper)
            && h_bytes[i..i + n_len].eq_ignore_ascii_case(n_bytes)
        {
            return Some(i);
        }
    }
    None
}

/// Case-insensitive ASCII substring containment check without heap allocations.
#[inline]
fn contains_ignore_ascii_case(haystack: &str, needle: &str) -> bool {
    find_ignore_ascii_case(haystack, needle).is_some()
}

/// Match an HTML block starting at `lines[start_idx]`.
/// Returns `Some((end_line_idx, is_verbatim))` if matched.
#[allow(clippy::needless_range_loop)]
fn match_html_block(lines: &[&str], start_idx: usize) -> Option<(usize, bool)> {
    let (content, _) = split_line_ending(lines[start_idx]);
    let trimmed = content.trim_start();

    if !trimmed.starts_with('<') {
        return None;
    }

    // HTML comment <!-- ... -->
    if trimmed.starts_with("<!--") {
        for j in start_idx..lines.len() {
            let (c, _) = split_line_ending(lines[j]);
            if c.contains("-->") {
                return Some((j, true));
            }
        }
        return Some((lines.len() - 1, true));
    }

    // CDATA <![CDATA[ ... ]]>
    if trimmed.starts_with("<![CDATA[") {
        for j in start_idx..lines.len() {
            let (c, _) = split_line_ending(lines[j]);
            if c.contains("]]>") {
                return Some((j, true));
            }
        }
        return Some((lines.len() - 1, true));
    }

    // Find tag name
    let tag_name_end = trimmed[1..]
        .find(|c: char| c.is_whitespace() || c == '>' || c == '/')
        .map(|idx| idx + 1)
        .unwrap_or(trimmed.len());
    let tag_name = trimmed[1..tag_name_end].to_ascii_lowercase();

    // Verbatim script and style blocks
    if tag_name == "script" {
        for j in start_idx..lines.len() {
            let (c, _) = split_line_ending(lines[j]);
            if contains_ignore_ascii_case(c, "</script>") {
                return Some((j, true));
            }
        }
        return Some((lines.len() - 1, true));
    }
    if tag_name == "style" {
        for j in start_idx..lines.len() {
            let (c, _) = split_line_ending(lines[j]);
            if contains_ignore_ascii_case(c, "</style>") {
                return Some((j, true));
            }
        }
        return Some((lines.len() - 1, true));
    }

    // Must be in recognized HTML block tags
    if !is_html_block_tag(tag_name.as_str()) {
        return None;
    }

    // Check if self-closing tag (e.g. `<hr>`, `<hr/>`, `<div ... />`)
    if let Some(tag_end) = find_tag_end(trimmed.as_bytes(), 0) {
        let raw_tag = &trimmed[..tag_end];
        let (_, _, is_self_closing) = parse_tag_info(raw_tag);
        if is_self_closing {
            return Some((start_idx, false));
        }
    }

    // Search for matching closing tag `</tag_name>` with nesting support
    let close_pat = format!("</{}>", tag_name);
    let open_pat = format!("<{}", tag_name);
    let mut depth: usize = 0;

    for j in start_idx..lines.len() {
        let (c, _) = split_line_ending(lines[j]);

        // Increment for open tags
        let mut search_idx = 0;
        while let Some(pos) = find_ignore_ascii_case(&c[search_idx..], &open_pat) {
            let absolute_pos = search_idx + pos;
            if let Some(tag_end) = find_tag_end(c.as_bytes(), absolute_pos) {
                let tag_str = &c[absolute_pos..tag_end];
                let (is_closing, parsed_name, is_self_closing) = parse_tag_info(tag_str);
                if !is_closing && parsed_name == tag_name && !is_self_closing {
                    depth += 1;
                }
                search_idx = tag_end;
            } else {
                search_idx = absolute_pos + open_pat.len();
            }
        }

        // Decrement for closing tags
        search_idx = 0;
        while let Some(pos) = find_ignore_ascii_case(&c[search_idx..], &close_pat) {
            let absolute_pos = search_idx + pos;
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some((j, false));
            }
            search_idx = absolute_pos + close_pat.len();
        }
    }

    // Unclosed tag fallback: extend to last line
    Some((lines.len() - 1, false))
}

/// Accumulate consecutive lines of a general paragraph, merging soft line breaks
/// while respecting hard line breaks (`  \n` or `\\\n`).
fn accumulate_paragraph(lines: &[&str], start_idx: usize) -> (String, String, usize) {
    let mut i = start_idx;
    let mut text_acc = String::new();
    let num_lines = lines.len();

    while i < num_lines {
        let line = lines[i];
        let (content, ending) = split_line_ending(line);
        let trimmed_start = content.trim_start();

        // Blank line ends paragraph
        if content.trim().is_empty() {
            break;
        }

        // If a new structural block starts (unless line is start_idx), end paragraph
        if i > start_idx
            && (trimmed_start.starts_with("```")
                || trimmed_start.starts_with("~~~")
                || (trimmed_start.starts_with('<') && match_html_block(lines, i).is_some())
                || (content.contains('|')
                    && i + 1 < num_lines
                    && is_table_delimiter_row(split_line_ending(lines[i + 1]).0))
                || is_thematic_break(content.trim())
                || (trimmed_start.starts_with('#')
                    && trimmed_start.chars().take_while(|&c| c == '#').count() <= 6
                    && trimmed_start[trimmed_start.chars().take_while(|&c| c == '#').count()..]
                        .starts_with(' '))
                || trimmed_start.starts_with('>')
                || try_parse_task_list_item(content).is_some()
                || ((trimmed_start.starts_with("- ")
                    || trimmed_start.starts_with("* ")
                    || trimmed_start.starts_with("+ "))
                    && !is_thematic_break(content.trim()))
                || try_parse_footnote_def(content).is_some()
            ) {
                break;
            }

        // Hard break detection: two trailing spaces or trailing backslash
        let is_hard_break = content.ends_with("  ") || content.ends_with('\\');

        if is_hard_break {
            let clean_content = content.strip_suffix('\\').unwrap_or(content);
            text_acc.push_str(clean_content);
            text_acc.push_str(ending);
        } else if i + 1 < num_lines
            && !split_line_ending(lines[i + 1]).0.trim().is_empty()
            && can_soft_merge_lines(content, split_line_ending(lines[i + 1]).0)
        {
            // Soft merge with next line
            let next_content = split_line_ending(lines[i + 1]).0.trim_start();
            let need_space = should_insert_space_between(content.trim_end(), next_content);
            text_acc.push_str(content.trim_end());
            if need_space {
                text_acc.push(' ');
            }
        } else {
            text_acc.push_str(content);
            text_acc.push_str(ending);
        }

        i += 1;
    }

    (text_acc, String::new(), i)
}

/// Check if a line is a thematic break / horizontal rule (e.g. `---`, `***`, `___`).
fn is_thematic_break(s: &str) -> bool {
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if clean.len() < 3 {
        return false;
    }
    let first = clean.chars().next().unwrap();
    (first == '-' || first == '*' || first == '_') && clean.chars().all(|c| c == first)
}

/// Check if a line is a Markdown table delimiter row (e.g. `| :--- | :---: | ---: |`).
fn is_table_delimiter_row(line: &str) -> bool {
    let trimmed = line.trim();
    if !trimmed.contains('|') {
        return false;
    }

    let cells: Vec<&str> = trimmed.split('|').collect();
    let mut valid_cell_count = 0;

    for cell in cells {
        let cell_trimmed = cell.trim();
        if cell_trimmed.is_empty() {
            continue;
        }
        let has_dash = cell_trimmed.contains('-');
        let valid_chars = cell_trimmed
            .chars()
            .all(|c| c == '-' || c == ':' || c.is_whitespace());
        if has_dash && valid_chars {
            valid_cell_count += 1;
        } else {
            return false;
        }
    }

    valid_cell_count > 0
}

/// Try parsing a task list item (e.g. `- [x] text` or `- [ ] text`).
fn try_parse_task_list_item(line: &str) -> Option<(String, String)> {
    let trimmed_start = line.trim_start();
    let bullet_char = trimmed_start.chars().next()?;
    if bullet_char != '-' && bullet_char != '*' && bullet_char != '+' {
        return None;
    }

    if trimmed_start.len() < 6 {
        return None;
    }

    let after_bullet = &trimmed_start[1..];
    if !after_bullet.starts_with(' ') {
        return None;
    }

    let check_part = &after_bullet[1..];
    if check_part.starts_with("[ ] ")
        || check_part.starts_with("[x] ")
        || check_part.starts_with("[X] ")
    {
        let leading_ws = &line[..line.len() - trimmed_start.len()];
        let checkbox = &check_part[..3];
        let prefix = format!("{}{} {} ", leading_ws, bullet_char, checkbox);
        let text = check_part[4..].to_string();
        return Some((prefix, text));
    }

    None
}

/// Try parsing a footnote definition (e.g. `[^1]: text`).
fn try_parse_footnote_def(line: &str) -> Option<(String, String)> {
    let trimmed_start = line.trim_start();
    if !trimmed_start.starts_with("[^") {
        return None;
    }

    if let Some(colon_pos) = trimmed_start.find("]: ") {
        let leading_ws = &line[..line.len() - trimmed_start.len()];
        let prefix = format!("{}{}", leading_ws, &trimmed_start[..colon_pos + 3]);
        let text = trimmed_start[colon_pos + 3..].to_string();
        return Some((prefix, text));
    }

    None
}

/// Check whether character is sentence-ending punctuation.
fn ends_with_sentence_punctuation(s: &str) -> bool {
    if let Some(last_char) = s.chars().last() {
        matches!(
            last_char,
            '.' | '!' | '?' | '。' | '！' | '？' | '；' | ';'
        )
    } else {
        false
    }
}

/// Check if a line represents a key-value or structured attribute pair (e.g. `Domain Name: example.com`).
fn is_key_value_line(line: &str) -> bool {
    let trimmed = line.trim();
    if let Some(colon_pos) = trimmed.find(':') {
        if colon_pos > 0 {
            let key = trimmed[..colon_pos].trim();
            if !key.is_empty()
                && key.len() <= 64
                && !key.starts_with("http")
                && !key.starts_with("ftp")
                && !ends_with_sentence_punctuation(key)
            {
                let after = &trimmed[colon_pos + 1..];
                if after.is_empty() || after.starts_with(' ') || after.starts_with('\t') {
                    return true;
                }
            }
        }
    }
    if let Some(colon_pos) = trimmed.find('：') {
        let key = trimmed[..colon_pos].trim();
        if !key.is_empty() && key.len() <= 64 && !ends_with_sentence_punctuation(key) {
            return true;
        }
    }
    false
}

/// Check if a line is a standalone single-token data item (e.g. domain name, email address, phone number).
fn is_data_item_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Single-word / non-spaced token resembling a hostname, email, or telephone
    if !trimmed.contains(' ') && !trimmed.contains('\t') {
        if trimmed.contains('@')
            || (trimmed.contains('.') && !trimmed.ends_with('.'))
            || (trimmed.starts_with('+')
                && trimmed.len() > 1
                && trimmed
                    .chars()
                    .skip(1)
                    .all(|c| c.is_ascii_digit() || c == '.' || c == '-'))
        {
            return true;
        }
    }
    false
}

/// Determine whether the current line can be softly merged with the next line in a Markdown paragraph.
fn can_soft_merge_lines(curr_content: &str, next_content: &str) -> bool {
    let curr_trimmed = curr_content.trim();
    let next_trimmed = next_content.trim();

    if curr_trimmed.is_empty() || next_trimmed.is_empty() {
        return false;
    }

    // 1. Current line is an explicit Markdown hard break
    if curr_content.ends_with("  ") || curr_content.ends_with('\\') {
        return false;
    }

    // 2. Current line ends with sentence punctuation or list intro colon
    if ends_with_sentence_punctuation(curr_trimmed)
        || curr_trimmed.ends_with(':')
        || curr_trimmed.ends_with('：')
    {
        return false;
    }

    // 3. Either line is a key-value pair or single-item data entry
    if is_key_value_line(curr_trimmed)
        || is_key_value_line(next_trimmed)
        || is_data_item_line(curr_trimmed)
        || is_data_item_line(next_trimmed)
    {
        return false;
    }

    // 4. Continuation heuristics:
    // If next line starts with a lowercase letter, it is definitely a continuation of a wrapped clause.
    if let Some(first_char) = next_trimmed.chars().next() {
        if first_char.is_lowercase() {
            return true;
        }
    }

    // If current line ends with a continuation symbol: comma, dash/hyphen
    if curr_trimmed.ends_with(',')
        || curr_trimmed.ends_with('，')
        || curr_trimmed.ends_with('-')
        || curr_trimmed.ends_with('—')
    {
        return true;
    }

    false
}
/// Determine whether a space is required between two merged lines (avoiding space between CJK characters).
fn should_insert_space_between(prev: &str, next: &str) -> bool {
    let prev_last = prev.chars().last();
    let next_first = next.chars().next();
    !matches!((prev_last, next_first), (Some(c1), Some(c2)) if is_cjk_character(c1) && is_cjk_character(c2))
}

/// Split text into lines while preserving their exact line endings (`\r\n` or `\n`).
fn split_lines_with_endings(text: &str) -> Vec<&str> {
    let mut lines = Vec::with_capacity(text.len() / 40 + 4);
    let mut start = 0;
    for (i, b) in text.as_bytes().iter().enumerate() {
        if *b == b'\n' {
            lines.push(&text[start..=i]);
            start = i + 1;
        }
    }
    if start < text.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Split a single line into its content and line ending (`\r\n`, `\n`, or empty).
fn split_line_ending(line: &str) -> (&str, &str) {
    if let Some(stripped) = line.strip_suffix("\r\n") {
        (stripped, "\r\n")
    } else if let Some(stripped) = line.strip_suffix('\n') {
        (stripped, "\n")
    } else {
        (line, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    type MockTranslateResult = (String, Vec<Range<usize>>, Vec<Range<usize>>, Vec<usize>);

    /// Word-level mock translation function for testing token alignments and markup projection.
    fn mock_translate(text: &str) -> MockTranslateResult {
        let mut trg = String::new();
        let mut src_spans = Vec::new();
        let mut trg_spans = Vec::new();
        let mut alignments = Vec::new();

        let mut i = 0;
        let bytes = text.as_bytes();
        let len = bytes.len();

        while i < len {
            let ch = text[i..].chars().next().unwrap();
            let ch_len = ch.len_utf8();

            if ch.is_whitespace() {
                trg.push(ch);
                i += ch_len;
            } else {
                let start = i;
                while i < len {
                    let c = text[i..].chars().next().unwrap();
                    if c.is_whitespace() {
                        break;
                    }
                    i += c.len_utf8();
                }
                let word = &text[start..i];
                let trg_start = trg.len();
                trg.push_str("TR(");
                trg.push_str(word);
                trg.push(')');
                let trg_end = trg.len();

                let token_idx = src_spans.len();
                src_spans.push(start..i);
                trg_spans.push(trg_start..trg_end);
                alignments.push(token_idx);
            }
        }

        if src_spans.is_empty() {
            src_spans.push(0..text.len());
            trg_spans.push(0..trg.len());
            alignments.push(0);
        }

        (trg, src_spans, trg_spans, alignments)
    }

    #[test]
    fn test_yaml_frontmatter_protection() {
        let doc = "---\ntitle: \"Hello World\"\nauthor: Test\n---\n\n# Header\n\nBody paragraph.";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.starts_with("---\ntitle: \"Hello World\"\nauthor: Test\n---\n"));
        assert!(res.contains("# TR(Header)"));
        assert!(res.contains("TR(Body) TR(paragraph.)"));
    }

    #[test]
    fn test_fenced_code_block_protection() {
        let doc = "Here is code:\n\n```python\ndef foo():\n    return 42\n```\n\nDone.";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.contains("```python\ndef foo():\n    return 42\n```"));
        assert!(res.contains("TR(Here) TR(is) TR(code:)"));
        assert!(res.contains("TR(Done.)"));
    }

    #[test]
    fn test_markdown_table_protection_and_translation() {
        let doc = "| Feature | Value |\n| :--- | ---: |\n| Bold | **text** |\n";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.contains("| :--- | ---: |"));
        assert!(res.contains("| TR(Feature) | TR(Value) |"));
        assert!(res.contains("**TR(text)**"));
    }

    #[test]
    fn test_task_list_preservation() {
        let doc = "- [x] Finished task\n- [ ] Pending **important** task\n";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.starts_with("- [x] TR(Finished) TR(task)\n"));
        assert!(res.contains("- [ ] TR(Pending) **TR(important)** TR(task)\n"));
    }

    #[test]
    fn test_html_blocks_and_inlines() {
        let doc = "<div class=\"note\">\n  <p>Hello <b>world</b> and <code>var x = 1;</code></p>\n</div>";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.starts_with("<div class=\"note\">\n  <p>"));
        assert!(res.contains("<b>TR(world)</b>"));
        assert!(res.contains("<code>var x = 1;</code>"));
        assert!(res.ends_with("</p>\n</div>"));
    }

    #[test]
    fn test_paragraph_sentence_splitting() {
        let long_para = "First sentence of the long paragraph here. Scientists conducted several experiments yesterday. Researchers analyzed the results carefully.";
        assert!(long_para.len() >= DEFAULT_SENTENCE_SPLIT_THRESHOLD);

        let mut sentences_translated = Vec::new();
        let res = translate_document_with_alignment(
            long_para,
            ParseFormats::default(),
            |sent| {
                sentences_translated.push(sent.to_string());
                mock_translate(sent)
            },
        );

        assert_eq!(sentences_translated.len(), 3);
        assert_eq!(
            sentences_translated[0],
            "First sentence of the long paragraph here."
        );
        assert_eq!(
            sentences_translated[1],
            "Scientists conducted several experiments yesterday."
        );
        assert_eq!(
            sentences_translated[2],
            "Researchers analyzed the results carefully."
        );
        assert!(res.contains("TR(First) TR(sentence) TR(of)"));
    }

    #[test]
    fn test_paragraph_sentence_splitting_discourse_fusion() {
        let long_para_multi = "This is a detailed paragraph explaining our novel architecture. It works remarkably well. And all tests pass.";
        assert!(long_para_multi.len() >= DEFAULT_SENTENCE_SPLIT_THRESHOLD);

        let mut sentences_translated = Vec::new();
        let _ = translate_document_with_alignment(
            long_para_multi,
            ParseFormats::default(),
            |sent| {
                sentences_translated.push(sent.to_string());
                mock_translate(sent)
            },
        );
        assert_eq!(sentences_translated.len(), 1);
        assert_eq!(
            sentences_translated[0],
            "This is a detailed paragraph explaining our novel architecture. It works remarkably well. And all tests pass."
        );
    }

    #[test]
    fn test_soft_and_hard_line_breaks() {
        let soft_break_doc = "This is line one\nthat continues seamlessly on line two.\n";
        let res = translate_document_with_alignment(
            soft_break_doc,
            ParseFormats::default(),
            mock_translate,
        );
        assert!(res.contains("TR(This) TR(is) TR(line) TR(one)"));

        let hard_break_doc = "First line with hard break  \nSecond line.\n";
        let res_hard = translate_document_with_alignment(
            hard_break_doc,
            ParseFormats::default(),
            mock_translate,
        );
        assert!(res_hard.contains("  \n"));
    }

    #[test]
    fn test_footnote_and_divider() {
        let doc = "Reference.[^1]\n\n---\n\n[^1]: Note with *italic* text.\n";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.contains("---\n"));
        assert!(res.contains("[^1]: TR(Note) TR(with) *TR(italic)* TR(text.)"));
    }

    #[test]
    fn test_blockquote_with_inner_content() {
        let doc = "> Quote line 1\n>\n> Quote line 2 with `code`\n";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);

        assert!(res.contains("> TR(Quote) TR(line) TR(1)"));
        assert!(res.contains(">\n"));
        assert!(res.contains("`code`"));
    }

    #[test]
    fn test_full_composite_document_roundtrip() {
        let input = r#"---
title: "Demo"
---

# Heading 1

Paragraph with **bold** and <em>italic</em>.

| Col A | Col B |
| :--- | ---: |
| Value 1 | Value 2 |

```rust
fn main() { println!("Hello"); }
```

- [x] Done
- [ ] Todo

<details open>
  <summary>Click to open</summary>
  <p>Inside details</p>
</details>
"#;

        let res = translate_document_with_alignment(input, ParseFormats::default(), mock_translate);

        // Verify all key structural anchors
        assert!(res.starts_with("---\ntitle: \"Demo\"\n---"));
        assert!(res.contains("# TR(Heading) TR(1)"));
        assert!(res.contains("**TR(bold)**"));
        assert!(res.contains("<em>TR(italic.)</em>"));
        assert!(res.contains("| :--- | ---: |"));
        assert!(res.contains("| TR(Value) TR(1) | TR(Value) TR(2) |"));
        assert!(res.contains("```rust\nfn main() { println!(\"Hello\"); }\n```"));
        assert!(res.contains("- [x] TR(Done)"));
        assert!(res.contains("- [ ] TR(Todo)"));
        assert!(res.contains("<details open>"));
        assert!(res.contains("<summary>TR(Click) TR(to) TR(open)</summary>"));
        assert!(res.contains("<p>TR(Inside) TR(details)</p>"));
        assert!(res.contains("</details>"));
    }

    #[test]
    fn test_standalone_image_and_void_tags_preservation() {
        let doc = "![Alt text](https://example.com/logo.png)\n\n<img src=\"pic.png\" />\n\n`var x = 1;`\n";
        let res = translate_document_with_alignment(doc, ParseFormats::default(), mock_translate);
        assert!(res.contains("![Alt text](https://example.com/logo.png)"));
        assert!(res.contains("<img src=\"pic.png\" />"));
        assert!(res.contains("`var x = 1;`"));
    }

    #[test]
    fn test_is_html_block_tag() {
        assert!(is_html_block_tag("div"));
        assert!(is_html_block_tag("p"));
        assert!(is_html_block_tag("table"));
        assert!(is_html_block_tag("details"));
        assert!(is_html_block_tag("summary"));
        assert!(!is_html_block_tag("span"));
        assert!(!is_html_block_tag("b"));
        assert!(!is_html_block_tag("a"));
        assert!(!is_html_block_tag("img"));
        assert!(!is_html_block_tag("unknown_tag"));
    }

    #[test]
    fn test_translate_document_with_alignment_parallel() {
        let doc = "# Hello World\n\nThis is paragraph one with **bold** text.\n\nThis is paragraph two. It has multiple sentences. And another sentence.\n";
        let res = translate_document_with_alignment_parallel(doc, ParseFormats::default(), mock_translate);
        assert!(res.contains("# TR(Hello) TR(World)"));
        assert!(res.contains("**TR(bold)**"));
        assert!(res.contains("TR(This) TR(is) TR(paragraph) TR(two.)"));
    }

    #[test]
    fn test_find_and_contains_ignore_ascii_case() {
        assert_eq!(find_ignore_ascii_case("Hello <SCRIPT>alert(1)</script>", "</script>"), Some(22));
        assert_eq!(find_ignore_ascii_case("Hello <SCRIPT>alert(1)</script>", "</SCRIPT>"), Some(22));
        assert_eq!(find_ignore_ascii_case("No script here", "</script>"), None);
        assert_eq!(find_ignore_ascii_case("short", "much longer pattern"), None);
        assert_eq!(find_ignore_ascii_case("abc", ""), Some(0));

        assert!(contains_ignore_ascii_case("<Style type=\"text/css\">", "<style"));
        assert!(contains_ignore_ascii_case("</STYLE>", "</style>"));
        assert!(!contains_ignore_ascii_case("plain text", "</style>"));
    }

    #[test]
    fn test_translate_document_dual_serial_and_parallel() {
        let doc = "# Plain Header\n\nPlain paragraph without any tags.\n\nParagraph with **bold** text.\n";
        let mut plain_called = 0;
        let mut align_called = 0;

        let res_seq = translate_document_dual(
            doc,
            ParseFormats::default(),
            |plain| {
                plain_called += 1;
                assert!(!plain.contains("**"));
                format!("PLAIN({})", plain)
            },
            |markup| {
                align_called += 1;
                mock_translate(markup)
            },
        );

        // Header is plain (1), first para is plain (1), second para has bold (1)
        assert_eq!(plain_called, 2);
        assert_eq!(align_called, 1);
        assert!(res_seq.contains("# PLAIN(Plain Header)"));
        assert!(res_seq.contains("PLAIN(Plain paragraph without any tags.)"));
        assert!(res_seq.contains("**TR(bold)**"));

        let res_par = translate_document_dual_parallel(
            doc,
            ParseFormats::default(),
            |plain| format!("PLAIN({})", plain),
            mock_translate,
        );
        assert_eq!(res_seq, res_par);
    }
    #[test]
    fn test_whois_notification_multiline_and_newline_preservation() {
        let input = r#"Domain Info Notification for example.com
This message is a reminder to help you keep the contact data associated with your domain registration up-to-date. Our records include the following information:

Domain Name: example.com
Registrar Name: Example Registrar Inc

Registrant Name: John Doe
Registrant Organization: Acme Corp
Registrant Phone Number: +1.5550100
Registrant Email: johndoe@example.com
Registrant Address: California
Registrant City: San Francisco
Registrant State/Province: Bay Area
Registrant Country: US
Registrant Postal Code: 00000

Administrative Contact Name: John Doe
Administrative Contact Organization: Acme Corp
Administrative Contact Address: California
Administrative Contact City: San Francisco
Administrative State/Province: Bay Area
Administrative Contact Country: US
Administrative Contact Postal Code: 00000
Administrative Contact Phone: +1.5550100
Administrative Contact Email: johndoe@example.com
Technical Contact Name: John Doe
Technical Contact Organization: Acme Corp
Technical Contact Address: California
Technical Contact City: San Francisco
Technical State/Province: Bay Area
Technical Contact Country: US
Technical Contact Postal Code: 00000
Technical Contact Phone: +1.5550100
Technical Contact Email: johndoe@example.com
Original Creation Date: Mon, Feb 9th, 2026
Expiration Date: Tue, Feb 9th, 2027
Nameservers:

ns1.example.com
ns2.example.com
If any of the information above is inaccurate, you must correct it by logging in to your account at Example dashboard.

If the information above is correct, you do not need to take any action.

Please remember that under the terms of your registration agreement, the provision of false contact information may be grounds for cancellation of your domain name registration.

Thank you for your attention.

The Example Registrar team
"#;

        let mut translated_segments = Vec::new();
        let res = translate_document_with_alignment(
            input,
            ParseFormats::default(),
            |text| {
                translated_segments.push(text.to_string());
                mock_translate(text)
            },
        );

        // 1. Verify newline line counts and structure
        assert!(res.contains("TR(Domain) TR(Name:) TR(example.com)\nTR(Registrar) TR(Name:) TR(Example) TR(Registrar) TR(Inc)"));
        assert!(res.contains("TR(Registrant) TR(Address:) TR(California)\nTR(Registrant) TR(City:) TR(San) TR(Francisco)"));
        assert!(res.contains("TR(Administrative) TR(Contact) TR(Address:) TR(California)\nTR(Administrative) TR(Contact) TR(City:) TR(San) TR(Francisco)"));
        assert!(res.contains("TR(ns1.example.com)\nTR(ns2.example.com)"));

        // 2. Crucial semantic isolation: Hunan and City must NOT be in the same translated segment
        assert!(!translated_segments.iter().any(|s| s.contains("California") && s.contains("San Francisco")));
        assert!(!translated_segments.iter().any(|s| s.contains("Domain Name:") && s.contains("Registrar Name:")));
    }

    #[test]
    fn test_short_multiline_under_threshold_preserves_newlines() {
        let input = "Name: Alice\nCity: Paris\n";
        let mut segments = Vec::new();
        let res = translate_document_with_alignment(
            input,
            ParseFormats::default(),
            |text| {
                segments.push(text.to_string());
                mock_translate(text)
            },
        );

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], "Name: Alice");
        assert_eq!(segments[1], "City: Paris");
        assert_eq!(res, "TR(Name:) TR(Alice)\nTR(City:) TR(Paris)\n");
    }

    #[test]
    fn test_document_blockquote_recursion_depth_limit() {
        let mut doc = String::new();
        for _ in 0..50 {
            doc.push_str("> ");
        }
        doc.push_str("Deep quote\n");

        let res = translate_document_with_alignment(&doc, ParseFormats::default(), mock_translate);
        assert!(res.contains("TR(Deep) TR(quote)"));
        let gt_count = res.chars().filter(|&c| c == '>').count();
        assert!(gt_count >= MAX_DOCUMENT_DEPTH);
    }

    #[test]
    fn test_document_deep_nesting_no_stack_overflow() {
        let mut doc = String::new();
        for _ in 0..2000 {
            doc.push_str("> ");
        }
        doc.push_str("Nested\n");

        let res = translate_document_with_alignment(&doc, ParseFormats::default(), mock_translate);
        assert!(res.contains("TR(Nested)"));
    }
}
