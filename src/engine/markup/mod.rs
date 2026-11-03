//! Universal markup preservation and alignment projection module.
//!
//! Provides document-level and sentence-level markup abstraction that decouples
//! markup parsing (HTML, Markdown, etc.) from translation models. Using
//! token-level cross-attention alignments emitted by the Marian transformer decoder,
//! inline markup spans are projected onto target text with full context retention
//! and nested tag tree validity, matching the Firefox Translations (Bergamot) architecture.

pub mod inline;
pub mod split;
pub mod document;

pub use inline::{extract_inline_markup, extract_unified_inline, unescape_html_entities};
pub use split::{
    detect_dominant_lang, fuse_discourse_chunks_inplace, is_discourse_chunk_starter,
    process_split_sentences, reassemble_sentences, remap_spans_to_sentences,
    split_long_sentence_semantic, split_long_text_semantic, split_sentences,
    split_sentences_bounded, split_sentences_hierarchical_with_lang,
    split_sentences_with_lang, split_sentences_with_spans,
    split_sentences_with_spans_bounded, translate_prepared_split_sentences_dual,
    translate_prepared_split_sentences_dual_parallel,
    translate_prepared_split_sentences_with_alignment,
    translate_prepared_split_sentences_with_alignment_parallel,
    translate_single_split_sentence, translate_single_split_sentence_dual,
    translate_split_sentences_dual, translate_split_sentences_dual_parallel,
    translate_split_sentences_with_alignment,
    translate_split_sentences_with_alignment_parallel, DEFAULT_MAX_SENTENCE_LEN,
    MAX_SENTENCE_LEN, SemanticSplitTier, SplitSentence,
};
pub use document::{
    is_html_block_tag, parse_document_blocks, translate_document_dual,
    translate_document_dual_with_languages,
    translate_document_dual_dyn, translate_document_dual_parallel,
    translate_document_dual_parallel_with_languages,
    translate_document_dual_parallel_dyn, translate_document_with_alignment,
    translate_document_with_alignment_dyn, translate_document_with_alignment_parallel,
    translate_document_with_alignment_parallel_dyn, translate_document_with_handler,
    AlignmentOutput, AlignmentTranslateFn, AlignmentTranslateSyncFn, DocumentBlock,
    PlainTranslateFn, PlainTranslateSyncFn, TableCell, TranslateHandler,
    DEFAULT_SENTENCE_SPLIT_THRESHOLD,
};

use std::ops::Range;
use crate::engine::rules::SyntaxRules;

/// Check if an opening markup tag represents an intra-word/character-level element
/// (e.g. `<sub>`, `<sup>`, `<ruby>`, `<rt>`, `<rp>`) that should not be expanded to word boundaries.
#[inline]
pub fn is_intra_word_markup(open_markup: &str) -> bool {
    let tag = open_markup.trim();
    if !tag.starts_with('<') {
        return false;
    }
    let rest = &tag[1..];
    let tag_name = rest
        .split(|c: char| c.is_ascii_whitespace() || c == '>' || c == '/')
        .next()
        .unwrap_or("");
    matches!(
        tag_name.to_ascii_lowercase().as_str(),
        "sub" | "sup" | "ruby" | "rt" | "rp"
    )
}

/// An inline markup annotation (e.g. `<b>...</b>`, `<a href="...">...</a>`, or `**...**`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MarkupSpan {
    /// Character byte range in the source plain text.
    pub source_range: Range<usize>,
    /// Projected character byte range in the target plain text.
    pub target_range: Option<Range<usize>>,
    /// Opening tag or marker (e.g. `<a href="/help">` or `**`).
    pub open_markup: String,
    /// Closing tag or marker (e.g. `</a>` or `**`). Empty for self-closing tags.
    pub close_markup: String,
    /// Whether this markup represents a self-closing element (e.g. `<br/>`, `<img .../>`).
    pub is_self_closing: bool,
    /// Original nesting depth in the parse tree (outer = 0, inner > 0).
    pub depth: usize,
    /// Whether this markup represents an intra-word/character-level element (e.g. `<sub>`, `<sup>`, `<ruby>`).
    pub is_intra_word: bool,
}

impl MarkupSpan {
    /// Create a paired enclosing markup span.
    pub fn new_paired(
        source_range: Range<usize>,
        open_markup: String,
        close_markup: String,
        depth: usize,
    ) -> Self {
        let is_intra_word = is_intra_word_markup(&open_markup);
        Self {
            source_range,
            target_range: None,
            open_markup,
            close_markup,
            is_self_closing: false,
            depth,
            is_intra_word,
        }
    }

    /// Create a self-closing markup point.
    pub fn new_self_closing(pos: usize, markup: String, depth: usize) -> Self {
        Self {
            source_range: pos..pos,
            target_range: None,
            open_markup: markup,
            close_markup: String::new(),
            is_self_closing: true,
            depth,
            is_intra_word: false,
        }
    }
}

/// Project source markup spans onto target text byte ranges using token alignments.
///
/// Implements the Firefox Translations (Bergamot) projection heuristic with 1D continuous density clustering:
/// 1. Maps each markup span's source byte range to overlapping source tokens.
/// 2. Accumulates aligned target tokens for those source tokens into a stack-allocated buffer (0 heap allocation).
/// 3. Filters cross-attention diffuse noise outliers using 1D continuous density clustering (gap <= 3)
///    to identify the major dense cluster and prevent span bloating.
/// 4. Computes the bounding target byte span `[min_trg_start..max_trg_end]` from the major cluster.
/// 5. Resolves crossing/overlapping markup trees to guarantee well-nested structure.
pub fn project_markup_spans(
    spans: &mut [MarkupSpan],
    source_token_spans: &[Range<usize>],
    target_token_spans: &[Range<usize>],
    alignments: &[usize], // target_token_idx -> source_token_idx
    target_text_len: usize,
) {
    project_markup_spans_with_syntax(
        spans,
        source_token_spans,
        target_token_spans,
        alignments,
        target_text_len,
        None,
    )
}

/// Project source markup spans onto target text byte ranges using token alignments and target language rule profile.
pub fn project_markup_spans_with_lang(
    spans: &mut [MarkupSpan],
    source_token_spans: &[Range<usize>],
    target_token_spans: &[Range<usize>],
    alignments: &[usize],
    target_text_len: usize,
    target_lang: Option<&str>,
) {
    let profile = target_lang.and_then(|lang| crate::engine::rules::RuleRegistry::global().get(lang));
    project_markup_spans_with_syntax(
        spans,
        source_token_spans,
        target_token_spans,
        alignments,
        target_text_len,
        profile.as_ref().map(|p| &p.syntax),
    );
}

/// Snaps a byte offset to the nearest valid character/token boundary within `[0, target_text_len]`.
///
/// Searches `target_token_spans` (both start and end offsets) as well as 0 and `target_text_len`
/// to find the closest legal boundary, preventing multi-byte UTF-8 slice panics during markup reconstruction.
#[inline]
pub(crate) fn snap_to_char_boundary(
    raw_pos: usize,
    target_token_spans: &[Range<usize>],
    target_text_len: usize,
) -> usize {
    let clamped = raw_pos.min(target_text_len);
    if target_token_spans.is_empty() {
        return clamped;
    }

    let mut best_boundary = clamped;
    let mut min_diff = usize::MAX;

    for boundary in [0, target_text_len] {
        let diff = clamped.abs_diff(boundary);
        if diff < min_diff {
            min_diff = diff;
            best_boundary = boundary;
        }
    }

    let idx = target_token_spans.partition_point(|tok| tok.start < clamped);
    let start_i = idx.saturating_sub(2);
    let end_i = (idx + 2).min(target_token_spans.len());

    for tok in &target_token_spans[start_i..end_i] {
        for boundary in [tok.start, tok.end] {
            let b = boundary.min(target_text_len);
            let diff = clamped.abs_diff(b);
            if diff < min_diff {
                min_diff = diff;
                best_boundary = b;
            }
        }
    }

    best_boundary
}

/// Project source markup spans onto target text byte ranges using token alignments and language syntax rules.
pub fn project_markup_spans_with_syntax(
    spans: &mut [MarkupSpan],
    source_token_spans: &[Range<usize>],
    target_token_spans: &[Range<usize>],
    alignments: &[usize], // target_token_idx -> source_token_idx
    target_text_len: usize,
    syntax: Option<&SyntaxRules>,
) {
    if spans.is_empty() || target_text_len == 0 {
        for span in spans.iter_mut() {
            span.target_range = Some(0..0);
        }
        return;
    }

    let num_target_tokens = target_token_spans.len().min(alignments.len());

    // Step 1 & 2: Project each span independently based on alignments.
    for span in spans.iter_mut() {
        if span.is_self_closing {
            let src_pos = span.source_range.start;
            // Find closest source token for point insertion
            let mut best_src_tok = 0;
            let mut min_dist = usize::MAX;
            for (i, tok_span) in source_token_spans.iter().enumerate() {
                let dist = if src_pos < tok_span.start {
                    tok_span.start - src_pos
                } else {
                    src_pos.saturating_sub(tok_span.end)
                };
                if dist < min_dist {
                    min_dist = dist;
                    best_src_tok = i;
                }
            }

            // Relative positioning within best_src_tok
            let (rel_is_end, rel_ratio) = if let Some(s_span) = source_token_spans.get(best_src_tok) {
                if src_pos >= s_span.end {
                    (true, 1.0f32)
                } else if src_pos <= s_span.start {
                    (false, 0.0f32)
                } else {
                    let len = (s_span.end - s_span.start).max(1);
                    (false, (src_pos - s_span.start) as f32 / len as f32)
                }
            } else {
                (false, 0.0f32)
            };

            // Collect all target tokens aligned to best_src_tok
            let mut matched_targets: Vec<usize> = Vec::new();
            for (t, &align) in alignments[..num_target_tokens].iter().enumerate() {
                if align == best_src_tok {
                    matched_targets.push(t);
                }
            }

            let trg_pos = if !matched_targets.is_empty() {
                let (min_t, max_t) = cluster_1d_major(&matched_targets, syntax);
                let t_start = target_token_spans.get(min_t).map_or(0, |s| s.start).min(target_text_len);
                let t_end = target_token_spans.get(max_t).map_or(target_text_len, |s| s.end).min(target_text_len);
                let pos = if rel_is_end {
                    t_end
                } else if rel_ratio <= 0.0 {
                    t_start
                } else {
                    t_start + ((t_end.saturating_sub(t_start)) as f32 * rel_ratio).round() as usize
                };
                Some(snap_to_char_boundary(pos, target_token_spans, target_text_len))
            } else {
                None
            };

            let trg_pos = trg_pos.unwrap_or_else(|| {
                let src_len = source_token_spans.last().map(|s| s.end).unwrap_or(0).max(1);
                let ratio = (src_pos as f32 / src_len as f32).clamp(0.0, 1.0);
                let raw_pos = ((target_text_len as f32 * ratio).round() as usize).min(target_text_len);
                snap_to_char_boundary(raw_pos, target_token_spans, target_text_len)
            });
            span.target_range = Some(trg_pos..trg_pos);
            continue;
        }

        // For paired enclosing spans: find source tokens that overlap source_range
        let src_start = span.source_range.start;
        let src_end = span.source_range.end;

        // Collect all target token indices matching source_range.
        // Uses stack buffer up to 64 tokens (0 heap allocation for typical spans);
        // seamlessly spills into heap Vec for exceptionally long spans to prevent truncation.
        let mut matched_stack = [0usize; 64];
        let mut matched_heap: Vec<usize> = Vec::new();
        let mut count = 0;
        for (t, &s_tok) in alignments[..num_target_tokens].iter().enumerate() {
            if let Some(s_span) = source_token_spans.get(s_tok) {
                if s_span.start < src_end && s_span.end > src_start {
                    if count < 64 {
                        matched_stack[count] = t;
                    } else if count == 64 {
                        matched_heap.reserve(128);
                        matched_heap.extend_from_slice(&matched_stack);
                        matched_heap.push(t);
                    } else {
                        matched_heap.push(t);
                    }
                    count += 1;
                }
            }
        }

        if count == 0 {
            // Fallback: proportional heuristic mapping if token alignment dropped this token
            let ratio_start = if !source_token_spans.is_empty() {
                let last_src_end = source_token_spans.last().unwrap().end.max(1);
                (src_start as f64 / last_src_end as f64).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let ratio_end = if !source_token_spans.is_empty() {
                let last_src_end = source_token_spans.last().unwrap().end.max(1);
                (src_end as f64 / last_src_end as f64).clamp(0.0, 1.0)
            } else {
                1.0
            };

            let raw_start = ((target_text_len as f64 * ratio_start) as usize).min(target_text_len);
            let raw_end = ((target_text_len as f64 * ratio_end) as usize).min(target_text_len).max(raw_start);

            let approx_start = snap_to_char_boundary(raw_start, target_token_spans, target_text_len);
            let approx_end = snap_to_char_boundary(raw_end, target_token_spans, target_text_len).max(approx_start);
            span.target_range = Some(approx_start..approx_end);
        } else {
            // 1D continuous density clustering to filter isolated attention noise outliers.
            let matched_slice: &[usize] = if count <= 64 {
                &matched_stack[..count]
            } else {
                &matched_heap
            };
            let (min_t, max_t) = cluster_1d_major(matched_slice, syntax);

            let raw_start = target_token_spans.get(min_t).map_or(0, |s| s.start).min(target_text_len);
            let raw_end = target_token_spans.get(max_t).map_or(target_text_len, |s| s.end).min(target_text_len);

            let trg_start = snap_to_char_boundary(raw_start, target_token_spans, target_text_len);
            let trg_end = snap_to_char_boundary(raw_end, target_token_spans, target_text_len).max(trg_start);
            span.target_range = Some(trg_start..trg_end);
        }
    }

    // Step 3: Enforce monotonic order guard and resolve nesting conflicts
    enforce_monotonic_order_guard(spans, target_text_len, syntax);
}

/// 1D continuous density clustering to filter isolated attention noise outliers.
/// Partitions target tokens into clusters where neighboring points have gap <= clustering_gap_threshold.
/// Returns (min_t, max_t) of the major dense cluster containing the most tokens.
/// In bimodal clustering with `head_direction == "head_final"`, prioritizes the final semantic core peak.
#[inline]
pub(crate) fn cluster_1d_major(
    matched: &[usize],
    syntax: Option<&SyntaxRules>,
) -> (usize, usize) {
    if matched.is_empty() {
        return (0, 0);
    }
    if matched.len() == 1 {
        return (matched[0], matched[0]);
    }

    let gap_threshold = syntax.map_or(3, |s| s.clustering_gap_threshold).max(1);
    let is_head_final = syntax.is_some_and(|s| s.head_direction == "head_final");

    if matched.len() == 2 {
        if matched[1].saturating_sub(matched[0]) <= gap_threshold {
            return (matched[0], matched[1]);
        }
        return if is_head_final {
            (matched[1], matched[1])
        } else if syntax.is_none() {
            (matched[0], matched[1])
        } else {
            (matched[0], matched[0])
        };
    }

    let mut best_start = 0;
    let mut best_len = 0;
    let mut best_span = usize::MAX;

    let mut cur_start = 0;
    for i in 1..matched.len() {
        if matched[i].saturating_sub(matched[i - 1]) <= gap_threshold {
            // Consecutive points within neighborhood distance gap <= gap_threshold belong to same cluster
        } else {
            let cur_len = i - cur_start;
            let cur_span = matched[i - 1] - matched[cur_start] + 1;
            let should_replace = if is_head_final {
                cur_len > best_len || (cur_len == best_len && cur_span <= best_span)
            } else {
                cur_len > best_len || (cur_len == best_len && cur_span < best_span)
            };
            if should_replace {
                best_len = cur_len;
                best_start = cur_start;
                best_span = cur_span;
            }
            cur_start = i;
        }
    }

    let cur_len = matched.len() - cur_start;
    let cur_span = matched[matched.len() - 1] - matched[cur_start] + 1;
    let should_replace = if is_head_final {
        cur_len > best_len || (cur_len == best_len && cur_span <= best_span)
    } else {
        cur_len > best_len || (cur_len == best_len && cur_span < best_span)
    };
    if should_replace {
        best_len = cur_len;
        best_start = cur_start;
    }

    (matched[best_start], matched[best_start + best_len - 1])
}

/// Enforces monotonic order and structural tree invariants on projected markup spans.
///
/// 1. Hierarchical containment: Ancestor spans in the source document strictly enclose
///    descendant spans in the target text.
/// 2. Monotonic sibling ordering: In languages without head-final constituent movement,
///    sibling spans maintain monotonic non-decreasing order ( \le T_{i+1}$) without
///    spurious cross-attention inversions.
/// 3. Sibling overlap partitioning: Disjoint sibling spans that overlap due to subword
///    expansion or attention blur are cleanly partitioned at word/character boundaries
///    rather than swallowing one another into false nesting.
/// 4. Self-closing point bounding: Point tags (, ) are bounded between
///    their neighboring source anchors and prevented from escaping their parent scopes.
pub fn enforce_monotonic_order_guard(
    spans: &mut [MarkupSpan],
    target_text_len: usize,
    syntax: Option<&SyntaxRules>,
) {
    if spans.len() <= 1 {
        if let Some(s) = spans.first_mut() {
            if let Some(r) = &mut s.target_range {
                r.start = r.start.min(target_text_len);
                r.end = r.end.min(target_text_len).max(r.start);
            }
        }
        return;
    }

    for s in spans.iter_mut() {
        if let Some(r) = &mut s.target_range {
            r.start = r.start.min(target_text_len);
            r.end = r.end.min(target_text_len).max(r.start);
        }
    }

    // Canonical source tree ordering:
    // Sort by source_range.start asc, depth asc (outer first), source_range.end desc
    spans.sort_by(|a, b| {
        a.source_range.start.cmp(&b.source_range.start)
            .then_with(|| a.depth.cmp(&b.depth))
            .then_with(|| b.source_range.end.cmp(&a.source_range.end))
    });

    let n = spans.len();
    let is_head_final = syntax.is_some_and(|s| s.head_direction == "head_final");

    // Pass 1: Hierarchical containment (Parent must enclose Child in target).
    for i in 0..n {
        let src_i = &spans[i].source_range;
        for j in 0..n {
            if i == j {
                continue;
            }
            let src_j = &spans[j].source_range;
            let is_ancestor = src_i.start <= src_j.start
                && src_i.end >= src_j.end
                && (src_i.start < src_j.start || src_i.end > src_j.end || spans[i].depth < spans[j].depth);

            if is_ancestor {
                let range_j = spans[j].target_range.clone();
                if let (Some(r_i), Some(r_j)) = (&mut spans[i].target_range, range_j) {
                    if !spans[i].is_self_closing {
                        r_i.start = r_i.start.min(r_j.start);
                        r_i.end = r_i.end.max(r_j.end).min(target_text_len);
                    }
                }
            }
        }
    }

    // Pass 2: Bound self-closing tags between neighboring source anchors
    for i in 0..n {
        if !spans[i].is_self_closing {
            continue;
        }
        let src_pos = spans[i].source_range.start;
        let mut min_bound = 0;
        for p in (0..i).rev() {
            if spans[p].source_range.end <= src_pos {
                if let Some(r) = &spans[p].target_range {
                    min_bound = min_bound.max(r.end);
                    break;
                }
            }
        }
        let mut max_bound = target_text_len;
        for s in (i + 1)..n {
            if spans[s].source_range.start >= src_pos {
                if let Some(r) = &spans[s].target_range {
                    max_bound = max_bound.min(r.start);
                    break;
                }
            }
        }

        if let Some(r) = &mut spans[i].target_range {
            if min_bound <= max_bound {
                r.start = r.start.clamp(min_bound, max_bound);
                r.end = r.start;
            } else {
                r.start = min_bound.min(target_text_len);
                r.end = r.start;
            }
        }
    }

    // Pass 3: Sibling monotonicity & Non-overlapping partitioning.
    enforce_monotonic_siblings(spans, target_text_len, !is_head_final);

    // Pass 4: Final hierarchical containment re-check
    for i in 0..n {
        let src_i = &spans[i].source_range;
        for j in 0..n {
            if i == j {
                continue;
            }
            let src_j = &spans[j].source_range;
            let is_ancestor = src_i.start <= src_j.start
                && src_i.end >= src_j.end
                && (src_i.start < src_j.start || src_i.end > src_j.end || spans[i].depth < spans[j].depth);

            if is_ancestor {
                let range_j = spans[j].target_range.clone();
                if let (Some(r_i), Some(r_j)) = (&mut spans[i].target_range, range_j) {
                    if !spans[i].is_self_closing {
                        r_i.start = r_i.start.min(r_j.start);
                        r_i.end = r_i.end.max(r_j.end).min(target_text_len);
                    }
                }
            }
        }
    }

    // Pass 5: Re-bound self-closing tags after sibling adjustments
    for i in 0..n {
        if !spans[i].is_self_closing {
            continue;
        }
        let src_pos = spans[i].source_range.start;
        let mut min_bound = 0;
        for p in (0..i).rev() {
            if spans[p].source_range.end <= src_pos {
                if let Some(r) = &spans[p].target_range {
                    min_bound = min_bound.max(r.end);
                    break;
                }
            }
        }
        let mut max_bound = target_text_len;
        for s in (i + 1)..n {
            if spans[s].source_range.start >= src_pos {
                if let Some(r) = &spans[s].target_range {
                    max_bound = max_bound.min(r.start);
                    break;
                }
            }
        }

        if let Some(r) = &mut spans[i].target_range {
            if min_bound <= max_bound {
                r.start = r.start.clamp(min_bound, max_bound);
                r.end = r.start;
            } else {
                r.start = min_bound.min(target_text_len);
                r.end = r.start;
            }
        }
    }

    // Sort by target_range.start and outer depth for rendering consistency
    spans.sort_by(|a, b| {
        let a_start = a.target_range.as_ref().map(|r| r.start).unwrap_or(0);
        let b_start = b.target_range.as_ref().map(|r| r.start).unwrap_or(0);
        a_start.cmp(&b_start).then_with(|| a.depth.cmp(&b.depth))
    });
}

fn enforce_monotonic_siblings(
    spans: &mut [MarkupSpan],
    target_text_len: usize,
    strictly_monotonic: bool,
) {
    let n = spans.len();
    if n <= 1 {
        return;
    }

    if strictly_monotonic {
        let mut paired_indices: Vec<usize> = Vec::new();
        for i in 0..n {
            if !spans[i].is_self_closing && spans[i].target_range.is_some() {
                paired_indices.push(i);
            }
        }

        if paired_indices.len() > 1 {
            let starts: Vec<usize> = paired_indices
                .iter()
                .map(|&idx| spans[idx].target_range.as_ref().unwrap().start)
                .collect();

            let lis_mask = compute_lis_mask(&starts);

            for (pos, &is_lis) in lis_mask.iter().enumerate() {
                if !is_lis {
                    let idx = paired_indices[pos];
                    let mut prev_bound = 0;
                    for p in (0..pos).rev() {
                        if lis_mask[p] {
                            let p_idx = paired_indices[p];
                            prev_bound = spans[p_idx].target_range.as_ref().unwrap().end;
                            break;
                        }
                    }
                    let mut next_bound = target_text_len;
                    for s in (pos + 1)..lis_mask.len() {
                        if lis_mask[s] {
                            let s_idx = paired_indices[s];
                            next_bound = spans[s_idx].target_range.as_ref().unwrap().start;
                            break;
                        }
                    }

                    if let Some(r) = &mut spans[idx].target_range {
                        let len = r.end.saturating_sub(r.start);
                        if prev_bound <= next_bound {
                            r.start = r.start.clamp(prev_bound, next_bound);
                            r.end = (r.start + len).min(next_bound).max(r.start);
                        } else {
                            r.start = prev_bound.min(target_text_len);
                            r.end = r.start;
                        }
                    }
                }
            }
        }
    }

    // Partition overlapping sibling spans
    for i in 0..n {
        if spans[i].is_self_closing {
            continue;
        }
        for j in (i + 1)..n {
            if spans[j].is_self_closing {
                continue;
            }
            let src_i = &spans[i].source_range;
            let src_j = &spans[j].source_range;
            let are_disjoint_in_source = src_i.end <= src_j.start || src_j.end <= src_i.start;
            if are_disjoint_in_source {
                let (r_i, r_j) = match (&spans[i].target_range, &spans[j].target_range) {
                    (Some(a), Some(b)) => (a.clone(), b.clone()),
                    _ => continue,
                };

                let (first, second, idx_first, idx_second) = if r_i.start <= r_j.start {
                    (r_i, r_j, i, j)
                } else {
                    (r_j, r_i, j, i)
                };

                if first.end > second.start {
                    let split = (first.end + second.start) / 2;
                    let split_clamped = split.clamp(first.start, second.end);

                    if let Some(r) = &mut spans[idx_first].target_range {
                        r.end = split_clamped.max(r.start);
                    }
                    if let Some(r) = &mut spans[idx_second].target_range {
                        r.start = split_clamped.min(r.end);
                    }
                }
            }
        }
    }
}

fn compute_lis_mask(arr: &[usize]) -> Vec<bool> {
    let n = arr.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dp = vec![1usize; n];
    let mut prev = vec![None; n];

    for i in 0..n {
        for j in 0..i {
            if arr[j] <= arr[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                prev[i] = Some(j);
            }
        }
    }

    let mut max_len = 0;
    let mut best_idx = 0;
    for (i, &len) in dp.iter().enumerate() {
        if len > max_len {
            max_len = len;
            best_idx = i;
        }
    }

    let mut mask = vec![false; n];
    let mut curr = Some(best_idx);
    while let Some(idx) = curr {
        mask[idx] = true;
        curr = prev[idx];
    }
    mask
}

/// Enforce well-nested hierarchy on target spans to avoid crossing tag bugs (e.g. ).
pub fn resolve_markup_nesting_conflicts(spans: &mut [MarkupSpan], target_text_len: usize) {
    enforce_monotonic_order_guard(spans, target_text_len, None);
}

pub(crate) use crate::utils::lang::is_cjk_or_continuous_script as is_cjk_character;

#[inline]
pub(crate) fn is_intra_word_connector(c: char) -> bool {
    matches!(c, '\'' | '\u{2019}' | '\u{02BB}' | '-' | '\u{2010}')
}

#[inline]
pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() && !is_cjk_character(c)
}

/// Finds the non-CJK word boundary interval  enclosing byte offset .
/// If  is outside any word (or at whitespace / punctuation / CJK character), returns .
pub(crate) fn find_word_boundaries(text: &str, pos: usize) -> (usize, usize) {
    if text.is_empty() || pos > text.len() {
        return (pos, pos);
    }
    let pos = clamp_to_char_boundary(text, pos);
    let len = text.len();
    let prev = text[..pos].chars().next_back();
    let next = text[pos..].chars().next();

    let is_inside = match (prev, next) {
        (Some(p), Some(n)) if is_word_char(p) && is_word_char(n) => true,
        (Some(p), Some(n)) if is_word_char(p) && is_intra_word_connector(n) => {
            let after = text[pos + n.len_utf8()..].chars().next();
            after.is_some_and(is_word_char)
        }
        (Some(p), Some(n)) if is_intra_word_connector(p) && is_word_char(n) => {
            let before = text[..pos.saturating_sub(p.len_utf8())].chars().next_back();
            before.is_some_and(is_word_char)
        }
        _ => false,
    };

    if !is_inside {
        return (pos, pos);
    }

    let mut word_start = pos;
    while word_start > 0 {
        let p = text[..word_start].chars().next_back().unwrap();
        if is_word_char(p) {
            word_start -= p.len_utf8();
        } else if is_intra_word_connector(p) {
            let before = text[..word_start.saturating_sub(p.len_utf8())].chars().next_back();
            if before.is_some_and(is_word_char) {
                word_start -= p.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let mut word_end = pos;
    while word_end < len {
        let n = text[word_end..].chars().next().unwrap();
        if is_word_char(n) {
            word_end += n.len_utf8();
        } else if is_intra_word_connector(n) {
            let after = text[word_end + n.len_utf8()..].chars().next();
            if after.is_some_and(is_word_char) {
                word_end += n.len_utf8();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    (word_start, word_end)
}

#[inline]
pub(crate) fn snap_point_to_word_boundary(text: &str, pos: usize) -> usize {
    let (w_start, w_end) = find_word_boundaries(text, pos);
    if w_start < w_end {
        if pos - w_start <= w_end - pos {
            w_start
        } else {
            w_end
        }
    } else {
        pos
    }
}

/// Snaps  and  byte offsets so they do not bisect non-CJK words or compound connectors.
#[inline]
pub(crate) fn snap_to_word_boundary(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    if text.is_empty() || start >= end {
        return (start, end);
    }
    let (w_start, _) = find_word_boundaries(text, start);
    if w_start < start {
        start = w_start;
    }
    let (_, w_end) = find_word_boundaries(text, end);
    if w_end > end {
        end = w_end;
    }
    (start, end)
}

/// Snaps projected markup span boundaries outwards to complete words for non-CJK text.
pub fn snap_markup_spans_to_word_boundaries(spans: &mut [MarkupSpan], target_text: &str) {
    snap_markup_spans_to_word_boundaries_with_syntax(spans, target_text, None);
}

/// Snaps projected markup span boundaries outwards to complete words with language syntax rules.
pub fn snap_markup_spans_to_word_boundaries_with_syntax(
    spans: &mut [MarkupSpan],
    target_text: &str,
    syntax: Option<&SyntaxRules>,
) {
    if spans.is_empty() || target_text.is_empty() {
        return;
    }
    for span in spans.iter_mut() {
        if span.is_intra_word {
            continue;
        }
        if span.is_self_closing {
            if let Some(target_range) = &span.target_range {
                let pos = clamp_to_char_boundary(target_text, target_range.start.min(target_text.len()));
                let snapped = snap_point_to_word_boundary(target_text, pos);
                span.target_range = Some(snapped..snapped);
            }
            continue;
        }
        if let Some(target_range) = &span.target_range {
            let start = clamp_to_char_boundary(target_text, target_range.start.min(target_text.len()));
            let end = clamp_to_char_boundary(target_text, target_range.end.min(target_text.len())).max(start);
            if start < end {
                let (snapped_start, snapped_end) = snap_to_word_boundary(target_text, start, end);
                span.target_range = Some(snapped_start..snapped_end);
            }
        }
    }
    enforce_monotonic_order_guard(spans, target_text.len(), syntax);
}

/// Render plain target text with projected markup spans inserted at their computed locations.
pub fn render_projected_markup(target_text: &str, spans: &[MarkupSpan]) -> String {
    if spans.is_empty() {
        return target_text.to_string();
    }

    #[derive(Eq, PartialEq)]
    enum TagEvent<'a> {
        Open(&'a str),
        Close(&'a str),
        SelfClosing(&'a str),
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum TagKind {
        Close,
        SelfClosing,
        Open,
    }

    let mut events = Vec::with_capacity(spans.len() * 2);
    let mut event_idx = 0;
    for span in spans {
        if let Some(target_range) = &span.target_range {
            let mut start = target_range.start.min(target_text.len());
            let end = target_range.end.min(target_text.len());

            // Whitespace boundary absorption:
            // Target token spans from SentencePiece often have leading whitespace (due to \u{2581}),
            // which would cause tags to swallow inter-word spaces (e.g.  instead of
            // ). Shrink whitespace inside paired tags to keep spaces outside.
            if !span.is_self_closing {
                if start < end {
                    while start < end && target_text.as_bytes()[start] == b' ' {
                        start += 1;
                    }
                }
                if start >= end {
                    continue;
                }
            }

            if span.is_self_closing {
                events.push((start, TagKind::SelfClosing, span.depth, start, event_idx, TagEvent::SelfClosing(&span.open_markup)));
            } else {
                events.push((end, TagKind::Close, span.depth, start, event_idx, TagEvent::Close(&span.close_markup)));
                event_idx += 1;
                events.push((start, TagKind::Open, span.depth, start, event_idx, TagEvent::Open(&span.open_markup)));
                event_idx += 1;
            }
        }
    }

    // Sort events using hierarchical tag nesting rules
    events.sort_by(|a, b| {
        if a.0 != b.0 {
            return a.0.cmp(&b.0);
        }
        use std::cmp::Ordering::*;
        match (a.1, b.1) {
            (TagKind::Close, TagKind::Close) => {
                b.3.cmp(&a.3)
                    .then_with(|| b.2.cmp(&a.2))
                    .then_with(|| b.4.cmp(&a.4))
            }
            (TagKind::Open, TagKind::Open) => {
                a.2.cmp(&b.2).then_with(|| a.4.cmp(&b.4))
            }
            (TagKind::Close, TagKind::Open) => {
                if a.3 == b.3 { Greater } else { Less }
            }
            (TagKind::Open, TagKind::Close) => {
                if a.3 == b.3 { Less } else { Greater }
            }
            (TagKind::SelfClosing, TagKind::Close) => {
                if a.2 > b.2 { Less } else { Greater }
            }
            (TagKind::Close, TagKind::SelfClosing) => {
                if b.2 > a.2 { Greater } else { Less }
            }
            (TagKind::SelfClosing, TagKind::Open) => {
                if a.2 > b.2 { Greater } else { Less }
            }
            (TagKind::Open, TagKind::SelfClosing) => {
                if b.2 > a.2 { Less } else { Greater }
            }
            (TagKind::SelfClosing, TagKind::SelfClosing) => a.4.cmp(&b.4),
        }
    });

    let mut out = String::with_capacity(target_text.len() + 64);
    let mut cur_idx = 0;

    for (pos, _, _, _, _, event) in events {
        if pos > cur_idx {
            let safe_pos = clamp_to_char_boundary(target_text, pos);
            if safe_pos > cur_idx {
                out.push_str(&target_text[cur_idx..safe_pos]);
                cur_idx = safe_pos;
            }
        }
        match event {
            TagEvent::Open(tag) => out.push_str(tag),
            TagEvent::Close(tag) => out.push_str(tag),
            TagEvent::SelfClosing(tag) => out.push_str(tag),
        }
    }

    if cur_idx < target_text.len() {
        out.push_str(&target_text[cur_idx..]);
    }

    out
}

#[inline]
fn clamp_to_char_boundary(s: &str, mut idx: usize) -> usize {
    if idx >= s.len() {
        return s.len();
    }
    while idx > 0 && !s.is_char_boundary(idx) {
        idx -= 1;
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_markup_spans_word_reordering() {
        // Source: "Please click [our website] for details."
        // Target: "詳細は[当社のウェブサイト]をクリックしてください。" (Japanese SOV reordering)
        let mut spans = vec![MarkupSpan::new_paired(
            13..24, // "our website"
            "<a href=\"#\">".to_string(),
            "</a>".to_string(),
            0,
        )];

        let source_token_spans = vec![
            0..6,   // "Please" (tok 0)
            7..12,  // "click"  (tok 1)
            13..16, // "our"    (tok 2)
            17..24, // "website"(tok 3)
            25..28, // "for"    (tok 4)
            29..37, // "details"(tok 5)
        ];

        let target_text = "詳細は当社のウェブサイトをクリックしてください。";
        let target_token_spans = vec![
            0..9,   // "詳細は" (tok 0 -> src tok 5 "details")
            9..18,  // "当社の" (tok 1 -> src tok 2 "our")
            18..36, // "ウェブサイト" (tok 2 -> src tok 3 "website")
            36..39, // "を" (tok 3 -> src tok 4)
            39..51, // "クリック" (tok 4 -> src tok 1 "click")
            51..72, // "してください。" (tok 5 -> src tok 0 "Please")
        ];

        let alignments = vec![5, 2, 3, 4, 1, 0]; // target token -> source token

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text.len(),
        );

        let rendered = render_projected_markup(target_text, &spans);
        assert_eq!(
            rendered,
            "詳細は<a href=\"#\">当社のウェブサイト</a>をクリックしてください。"
        );
    }

    #[test]
    fn test_resolve_markup_nesting_conflicts_crossing_tags() {
        let mut spans = vec![
            MarkupSpan {
                source_range: 0..10,
                target_range: Some(10..20),
                open_markup: "<a>".to_string(),
                close_markup: "</a>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
            MarkupSpan {
                source_range: 5..15,
                target_range: Some(5..15),
                open_markup: "<b>".to_string(),
                close_markup: "</b>".to_string(),
                is_self_closing: false,
                depth: 1,
                is_intra_word: false,
            },
        ];

        resolve_markup_nesting_conflicts(&mut spans, 30);

        // After conflict resolution, spans should no longer cross
        let r0 = spans[0].target_range.as_ref().unwrap();
        let r1 = spans[1].target_range.as_ref().unwrap();
        assert!(r0.end <= r1.start || r1.end <= r0.start || (r0.start <= r1.start && r0.end >= r1.end));
    }

    #[test]
    fn test_resolve_markup_nesting_conflicts_identical_end_depth_inverted() {
        let mut spans = vec![
            MarkupSpan {
                source_range: 5..15,
                target_range: Some(5..20),
                open_markup: "<b>".to_string(),
                close_markup: "</b>".to_string(),
                is_self_closing: false,
                depth: 1,
                is_intra_word: false,
            },
            MarkupSpan {
                source_range: 0..15,
                target_range: Some(10..20),
                open_markup: "<a>".to_string(),
                close_markup: "</a>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
        ];

        resolve_markup_nesting_conflicts(&mut spans, 30);

        let r0 = spans[0].target_range.as_ref().unwrap();
        let r1 = spans[1].target_range.as_ref().unwrap();
        assert_eq!(spans[0].open_markup, "<a>");
        assert_eq!(spans[1].open_markup, "<b>");
        assert!(r0.start <= r1.start && r0.end >= r1.end);

        let target_text = "012345678901234567890123456789";
        let rendered = render_projected_markup(target_text, &spans);
        assert_eq!(
            rendered,
            "01234<a><b>567890123456789</b></a>0123456789"
        );
    }

    #[test]
    fn test_render_projected_markup_raw_crossing_prevention_lifo() {
        let spans = vec![
            MarkupSpan {
                source_range: 5..15,
                target_range: Some(0..10),
                open_markup: "<b>".to_string(),
                close_markup: "</b>".to_string(),
                is_self_closing: false,
                depth: 1,
                is_intra_word: false,
            },
            MarkupSpan {
                source_range: 0..15,
                target_range: Some(5..10),
                open_markup: "<a>".to_string(),
                close_markup: "</a>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
        ];

        let target_text = "hello world";
        let rendered = render_projected_markup(target_text, &spans);
        assert_eq!(rendered, "<b>hello <a>worl</a></b>d");
    }

    #[test]
    fn test_snap_markup_spans_to_word_boundaries() {
        let text = "Texto em negrito e espaço reservado.";
        let neg_start = text.find("neg").unwrap();
        let neg_end = neg_start + 3; // bisects "negrito" into "neg" + "rito"

        let mut spans = vec![MarkupSpan {
            source_range: 0..4,
            target_range: Some(neg_start..neg_end),
            open_markup: "<b>".to_string(),
            close_markup: "</b>".to_string(),
            is_self_closing: false,
            depth: 0,
            is_intra_word: false,
        }];

        snap_markup_spans_to_word_boundaries(&mut spans, text);
        let rendered = render_projected_markup(text, &spans);
        assert_eq!(rendered, "Texto em <b>negrito</b> e espaço reservado.");

        // Verify CJK text preserves character-level boundaries without space-word expansion
        let cjk_text = "这是测试代码。";
        let test_start = cjk_text.find("测试").unwrap();
        let test_end = test_start + "测试".len();

        let mut cjk_spans = vec![MarkupSpan {
            source_range: 0..4,
            target_range: Some(test_start..test_end),
            open_markup: "<i>".to_string(),
            close_markup: "</i>".to_string(),
            is_self_closing: false,
            depth: 0,
            is_intra_word: false,
        }];

        snap_markup_spans_to_word_boundaries(&mut cjk_spans, cjk_text);
        let rendered_cjk = render_projected_markup(cjk_text, &cjk_spans);
        assert_eq!(rendered_cjk, "这是<i>测试</i>代码。");

        // Verify intra-word tags (sub, sup) are not snapped outwards to word boundaries
        let formula_text = "H2O and CO2";
        let sub_start = formula_text.find('2').unwrap();
        let sub_end = sub_start + 1; // exactly '2' in "H2O"
        let mut formula_spans = vec![MarkupSpan::new_paired(
            0..1,
            "<sub>".to_string(),
            "</sub>".to_string(),
            0,
        )];
        formula_spans[0].target_range = Some(sub_start..sub_end);
        assert!(formula_spans[0].is_intra_word);

        snap_markup_spans_to_word_boundaries(&mut formula_spans, formula_text);
        let rendered_formula = render_projected_markup(formula_text, &formula_spans);
        assert_eq!(rendered_formula, "H<sub>2</sub>O and CO2");
    }

    #[test]
    fn test_cluster_1d_major_edge_cases() {
        assert_eq!(cluster_1d_major(&[], None), (0, 0));
        assert_eq!(cluster_1d_major(&[42], None), (42, 42));
        assert_eq!(cluster_1d_major(&[5, 10], None), (5, 10));
    }

    #[test]
    fn test_project_markup_spans_over_64_tokens() {
        // Build 80 tokens in source and target
        let _src_text = "word ".repeat(80);
        let trg_text = "word ".repeat(80);
        let mut src_spans = Vec::new();
        let mut trg_spans = Vec::new();
        let mut cur = 0;
        for _ in 0..80 {
            src_spans.push(cur..cur + 4);
            trg_spans.push(cur..cur + 4);
            cur += 5;
        }
        let alignments: Vec<usize> = (0..80).collect();
        let mut spans = vec![MarkupSpan::new_paired(0..cur, "<b>".to_string(), "</b>".to_string(), 0)];

        project_markup_spans(&mut spans, &src_spans, &trg_spans, &alignments, trg_text.len());
        let target_range = spans[0].target_range.clone().unwrap();
        // Span should cover from 0 to the 80th token end (not truncated at 64th)
        assert_eq!(target_range.start, 0);
        assert_eq!(target_range.end, trg_spans[79].end);
    }

    #[test]
    fn test_cluster_1d_major_outlier_rejection() {
        // [1, 15, 16, 17]: isolated outlier at 1, major cluster at 15..17
        assert_eq!(cluster_1d_major(&[1, 15, 16, 17], None), (15, 17));

        // Outlier at suffix: [15, 16, 17, 30]
        assert_eq!(cluster_1d_major(&[15, 16, 17, 30], None), (15, 17));

        // Outliers at both prefix and suffix: [0, 15, 16, 17, 40]
        assert_eq!(cluster_1d_major(&[0, 15, 16, 17, 40], None), (15, 17));

        // Consecutive dense points: [2, 3, 4]
        assert_eq!(cluster_1d_major(&[2, 3, 4], None), (2, 4));

        // Neighborhood distance gap <= 3 preserved within same cluster: [2, 5, 8]
        assert_eq!(cluster_1d_major(&[2, 5, 8], None), (2, 8));

        // Major dense cluster wins over smaller cluster: [1, 2, 20, 21, 22]
        assert_eq!(cluster_1d_major(&[1, 2, 20, 21, 22], None), (20, 22));
    }

    #[test]
    fn test_cluster_1d_major_bimodal_head_direction() {
        let head_final = SyntaxRules {
            clustering_gap_threshold: 3,
            head_direction: "head_final".to_string(),
            ..Default::default()
        };
        let head_initial = SyntaxRules {
            clustering_gap_threshold: 3,
            head_direction: "head_initial".to_string(),
            ..Default::default()
        };
        // German verb-bracket structure (bimodal): auxiliary at [2, 3], participle/lexical verb at [20, 21]
        let de_verb_bracket = [2, 3, 20, 21];
        // Head-final prioritizes the semantic core peak at the end of the clause
        assert_eq!(cluster_1d_major(&de_verb_bracket, Some(&head_final)), (20, 21));
        // Head-initial prioritizes the initial peak
        assert_eq!(cluster_1d_major(&de_verb_bracket, Some(&head_initial)), (2, 3));

        // Disconnected two points across gap: [2, 20]
        assert_eq!(cluster_1d_major(&[2, 20], Some(&head_final)), (20, 20));
        assert_eq!(cluster_1d_major(&[2, 20], Some(&head_initial)), (2, 2));

        // Japanese/Korean SOV structure: subject/topic particle at [0, 1], main predicate at [14, 15]
        let ja_predicate = [0, 1, 14, 15];
        assert_eq!(cluster_1d_major(&ja_predicate, Some(&head_final)), (14, 15));
        assert_eq!(cluster_1d_major(&ja_predicate, Some(&head_initial)), (0, 1));
    }

    #[test]
    fn test_cluster_1d_major_density_awareness_head_final() {
        let head_final = SyntaxRules {
            clustering_gap_threshold: 3,
            head_direction: "head_final".to_string(),
            ..Default::default()
        };

        // Dense head [2, 3, 4] (len=3, span=3) vs sparse tail noise [20, 23, 26] (len=3, span=7).
        // Dense cluster strictly beats sparse tail noise.
        let tokens = [2, 3, 4, 20, 23, 26];
        assert_eq!(cluster_1d_major(&tokens, Some(&head_final)), (2, 4));

        // When posterior cluster is strictly denser: [2, 5, 8] (len=3, span=7) vs [20, 21, 22] (len=3, span=3).
        let tokens_posterior_dense = [2, 5, 8, 20, 21, 22];
        assert_eq!(cluster_1d_major(&tokens_posterior_dense, Some(&head_final)), (20, 22));

        // When both clusters have equal token count and equal span: [2, 3] (len=2, span=2) vs [20, 21] (len=2, span=2).
        let tokens_equal = [2, 3, 20, 21];
        assert_eq!(cluster_1d_major(&tokens_equal, Some(&head_final)), (20, 21));

        // When posterior cluster has more tokens: [2, 3] (len=2) vs [20, 22, 24] (len=3).
        let tokens_more = [2, 3, 20, 22, 24];
        assert_eq!(cluster_1d_major(&tokens_more, Some(&head_final)), (20, 24));
    }

    #[test]
    fn test_cluster_1d_major_custom_gap_threshold() {
        let wide_gap = SyntaxRules {
            clustering_gap_threshold: 4,
            head_direction: "head_initial".to_string(),
            ..Default::default()
        };
        let narrow_gap = SyntaxRules {
            clustering_gap_threshold: 2,
            head_direction: "head_initial".to_string(),
            ..Default::default()
        };

        // Points with distance 4 between them: [1, 5, 9]
        assert_eq!(cluster_1d_major(&[1, 5, 9], Some(&wide_gap)), (1, 9));
        // With narrow gap 2, points are separate clusters of len 1; initial point returned
        assert_eq!(cluster_1d_major(&[1, 5, 9], Some(&narrow_gap)), (1, 1));
    }

    #[test]
    fn test_project_markup_spans_1d_density_clustering_outlier_rejection() {
        // Construct an alignment scenario with matched target tokens [1, 15, 16, 17].
        // Target token 1 is an isolated cross-attention noise outlier.
        // Target tokens 15..=17 form the major dense cluster corresponding to the true translation phrase.
        let mut spans = vec![MarkupSpan::new_paired(
            6..17, // "our website" (source token 1)
            "<b>".to_string(),
            "</b>".to_string(),
            0,
        )];

        let source_token_spans = vec![
            0..5,   // "Click"       (tok 0)
            6..17,  // "our website" (tok 1, marked span)
            18..21, // "for"         (tok 2)
            22..29, // "details"     (tok 3)
        ];

        // 20 target tokens (0..=19), each 3 bytes long
        let target_token_spans: Vec<Range<usize>> = (0..20)
            .map(|i| (i * 3)..(i * 3 + 3))
            .collect();
        let target_text = "aa_bb_cc_dd_ee_ff_gg_hh_ii_jj_kk_ll_mm_nn_oo_pp_qq_rr_ss_tt_";
        assert_eq!(target_text.len(), 60);

        // Alignments: target token 1 has spurious attention to source token 1.
        // Target tokens 15, 16, 17 align to source token 1 (true phrase).
        // All others align to other source tokens.
        let mut alignments = vec![0usize; 20];
        alignments[0] = 0;
        alignments[1] = 1; // Outlier noise!
        alignments[2] = 0;
        alignments[3] = 2;
        alignments[4] = 2;
        alignments[5] = 3;
        alignments[6] = 3;
        alignments[7] = 0;
        alignments[8] = 0;
        alignments[9] = 2;
        alignments[10] = 2;
        alignments[11] = 0;
        alignments[12] = 0;
        alignments[13] = 3;
        alignments[14] = 3;
        alignments[15] = 1; // Main cluster
        alignments[16] = 1; // Main cluster
        alignments[17] = 1; // Main cluster
        alignments[18] = 0;
        alignments[19] = 0;

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text.len(),
        );

        // Target span must strictly lock to target tokens [15..17] (bytes 45..54),
        // rejecting outlier token 1 (bytes 3..6).
        assert_eq!(
            spans[0].target_range,
            Some(45..54),
            "Span should be locked to main cluster 15..17 (45..54), not bloated to 3..54"
        );

        let rendered = render_projected_markup(target_text, &spans);
        assert!(
            !rendered.starts_with("aa_<b>"),
            "Outlier token 1 must not be wrapped in tag: {}",
            rendered
        );
        assert_eq!(
            rendered,
            "aa_bb_cc_dd_ee_ff_gg_hh_ii_jj_kk_ll_mm_nn_oo_<b>pp_qq_rr_</b>ss_tt_"
        );
    }

    #[test]
    fn test_project_markup_spans_small_match_preserves_bounds() {
        // When matched tokens <= 2, direct min..max is preserved without clustering.
        let mut spans = vec![MarkupSpan::new_paired(
            6..17,
            "<i>".to_string(),
            "</i>".to_string(),
            0,
        )];

        let source_token_spans = vec![
            0..5,
            6..17,
            18..21,
        ];

        let target_token_spans: Vec<Range<usize>> = (0..5)
            .map(|i| (i * 3)..(i * 3 + 3))
            .collect();
        let target_text = "aa_bb_cc_dd_ee_";

        // Only tokens 1 and 3 match source token 1 (count = 2)
        let alignments = vec![0, 1, 0, 1, 0];

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text.len(),
        );

        // For <= 2 matches, min_t=1, max_t=3 -> span 3..12
        assert_eq!(spans[0].target_range, Some(3..12));
    }

    #[test]
    fn test_project_markup_spans_unmatched_self_closing_interpolation() {
        let mut spans = vec![MarkupSpan::new_self_closing(50, "<br/>".to_string(), 0)];
        let source_token_spans = vec![
            0..20,
            20..40,
            40..60, // Token 2 closest to src_pos 50
            60..80,
            80..100,
        ];
        let target_token_spans = vec![
            0..20,
            20..40,
            40..60,
            60..80,
            80..100,
        ];
        // Alignments do not map any target token to source token 2
        let alignments = vec![0, 1, 3, 4, 4];
        let target_text_len = 100;

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text_len,
        );

        let target_range = spans[0].target_range.clone().expect("target range set");
        assert_ne!(
            target_range.start, target_text_len,
            "Unmatched self-closing tag must not drift to end of target text"
        );
        // src_pos = 50 in src_len = 100 -> ratio = 0.5. In target_text_len = 100 -> raw_pos = 50.
        // Nearest boundary in target_token_spans (40..60) is 40 or 60.
        assert!(
            target_range.start == 40 || target_range.start == 60,
            "Target range start {} should align to nearest token boundary (40 or 60)",
            target_range.start
        );
        assert_eq!(target_range.start, target_range.end);
    }

    #[test]
    fn test_project_markup_spans_unmatched_paired_proportional_boundary_alignment() {
        // Multi-byte UTF-8 string: Japanese 3-byte characters, total 15 bytes.
        // "あいうえお": 'あ' 0..3, 'い' 3..6, 'う' 6..9, 'え' 9..12, 'お' 12..15
        let target_text = "あいうえお";
        let target_text_len = target_text.len();
        let target_token_spans = vec![0..3, 3..6, 6..9, 9..12, 12..15];

        let source_token_spans = vec![0..10, 10..20, 20..30];
        // Alignments do not map target tokens to source token 1 (unmatched, count == 0)
        let alignments = vec![0, 0, 0, 0, 0];
        let mut spans = vec![MarkupSpan::new_paired(
            10..20,
            "<b>".to_string(),
            "</b>".to_string(),
            0,
        )];

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text_len,
        );

        let trg = spans[0].target_range.clone().expect("target range set");
        // Must align to valid UTF-8 character boundary (multiples of 3)
        assert!(target_text.is_char_boundary(trg.start), "trg.start {} must be a char boundary", trg.start);
        assert!(target_text.is_char_boundary(trg.end), "trg.end {} must be a char boundary", trg.end);
        assert!(trg.end >= trg.start);

        // Rendering must not panic with slice error
        let rendered = render_projected_markup(target_text, &spans);
        assert!(!rendered.is_empty());
    }

    #[test]
    fn test_project_markup_spans_target_token_spans_char_boundary_alignment() {
        // Multi-byte UTF-8 string: "你好世界！"
        let target_text = "你好世界！";
        let target_text_len = target_text.len();
        let target_token_spans = vec![0..3, 3..6, 6..9, 9..12, 12..15];
        let source_token_spans = vec![0..5, 5..10];
        // Target token 1 (3..6) maps to source token 1 (5..10)
        let alignments = vec![0, 1, 0, 0, 0];
        let mut spans = vec![MarkupSpan::new_paired(
            5..10,
            "<i>".to_string(),
            "</i>".to_string(),
            0,
        )];

        project_markup_spans(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            target_text_len,
        );

        let trg = spans[0].target_range.clone().expect("target range set");
        assert!(target_text.is_char_boundary(trg.start));
        assert!(target_text.is_char_boundary(trg.end));
        let rendered = render_projected_markup(target_text, &spans);
        assert_eq!(rendered, "你<i>好</i>世界！");
    }

    #[test]
    fn test_snap_to_char_boundary_binary_search() {
        let spans = vec![3..7, 7..12, 15..20, 20..25];
        let text_len = 30;

        // Exhaustive verification against linear ground truth across all possible positions
        for raw_pos in 0..=35 {
            let snapped = snap_to_char_boundary(raw_pos, &spans, text_len);
            // Compute ground truth by checking all boundaries
            let clamped = raw_pos.min(text_len);
            let mut best = clamped;
            let mut min_diff = usize::MAX;
            for b in [0, text_len] {
                let diff = clamped.abs_diff(b);
                if diff < min_diff {
                    min_diff = diff;
                    best = b;
                }
            }
            for tok in &spans {
                for b in [tok.start, tok.end] {
                    let b = b.min(text_len);
                    let diff = clamped.abs_diff(b);
                    if diff < min_diff {
                        min_diff = diff;
                        best = b;
                    }
                }
            }
            assert_eq!(snapped, best, "mismatch at raw_pos {}", raw_pos);
        }

        // Edge cases
        assert_eq!(snap_to_char_boundary(10, &[], 20), 10);
        assert_eq!(snap_to_char_boundary(25, &[], 20), 20);
    }

    #[test]
    fn test_collapsed_markup_span_no_inversion() {
        let mut span = MarkupSpan::new_paired(0..0, "<i>".into(), "</i>".into(), 0);
        span.target_range = Some(5..5);
        let rendered = render_projected_markup("hello world", &[span]);
        assert_eq!(rendered, "hello world");
    }

    #[test]
    fn test_snap_markup_spans_intra_word_connector_hyphen_and_contraction() {
        let text = "This is state-of-the-art and don\x27t stop.";
        let state_start = text.find("state").unwrap();
        // Span bisecting "state-of-the-art" to just "state"
        let mut spans = vec![
            MarkupSpan::new_paired(
                0..5,
                "<b>".to_string(),
                "</b>".to_string(),
                0,
            ),
            MarkupSpan::new_paired(
                10..13,
                "<i>".to_string(),
                "</i>".to_string(),
                0,
            ),
        ];
        spans[0].target_range = Some(state_start..state_start + 5); // "state"
        let dont_start = text.find("don\x27t").unwrap();
        spans[1].target_range = Some(dont_start..dont_start + 3); // "don"

        snap_markup_spans_to_word_boundaries(&mut spans, text);

        let rendered = render_projected_markup(text, &spans);
        assert_eq!(
            rendered,
            "This is <b>state-of-the-art</b> and <i>don\x27t</i> stop."
        );
    }

    #[test]
    fn test_snap_point_to_word_boundary_self_closing() {
        let text = "hello world and don\x27t stop";
        // Void tag placed inside "hello" at byte 2 ("he<br/>llo")
        let mut spans = vec![
            MarkupSpan::new_self_closing(0, "<br/>".to_string(), 0),
            MarkupSpan::new_self_closing(10, "<img/>".to_string(), 0),
        ];
        spans[0].target_range = Some(2..2); // inside "hello"
        let dont_start = text.find("don\x27t").unwrap();
        spans[1].target_range = Some(dont_start + 3..dont_start + 3); // between "don" and "\x27t"

        snap_markup_spans_to_word_boundaries(&mut spans, text);

        // Byte 2 is closer to 0 than 5 -> snaps to 0
        assert_eq!(spans[0].target_range, Some(0..0));
        // Byte dont_start + 3 is closer to dont_start + 5 (end of "don\x27t") than dont_start
        assert_eq!(spans[1].target_range, Some(dont_start + 5..dont_start + 5));

        let rendered = render_projected_markup(text, &spans);
        assert_eq!(rendered, "<br/>hello world and don\x27t<img/> stop");
    }

    #[test]
    fn test_monotonic_order_guard_sequential_inversion_resolution() {
        // Source: "First and Second"
        // Target: "Erste und Zweite" (head-initial)
        // Spans: span 0 ("First"), span 1 ("Second")
        // Alignment noise inverted them: span 1 is at 0..5, span 0 is at 10..15
        let syntax = SyntaxRules {
            clustering_gap_threshold: 3,
            head_direction: "head_initial".to_string(),
            ..Default::default()
        };
        let mut spans = vec![
            MarkupSpan {
                source_range: 0..5,
                target_range: Some(10..15),
                open_markup: "<b>".to_string(),
                close_markup: "</b>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
            MarkupSpan {
                source_range: 10..16,
                target_range: Some(0..5),
                open_markup: "<i>".to_string(),
                close_markup: "</i>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
        ];

        enforce_monotonic_order_guard(&mut spans, 20, Some(&syntax));

        // After monotonic order guard, span 0 must precede span 1 in target
        assert_eq!(spans[0].open_markup, "<b>");
        assert_eq!(spans[1].open_markup, "<i>");
        let r0 = spans[0].target_range.as_ref().unwrap();
        let r1 = spans[1].target_range.as_ref().unwrap();
        assert!(r0.start <= r1.start, "r0.start ({}) must <= r1.start ({})", r0.start, r1.start);
        assert!(r0.end <= r1.start || r0.end <= r1.end);
    }

    #[test]
    fn test_monotonic_order_guard_sibling_overlap_partitioning() {
        // Disjoint sibling spans at depth 0 that overlap in target:
        // span 0 at 0..15, span 1 at 10..25.
        // Sibling spans must be partitioned, NOT merged into false nesting!
        let mut spans = vec![
            MarkupSpan {
                source_range: 0..5,
                target_range: Some(0..15),
                open_markup: "<b>".to_string(),
                close_markup: "</b>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
            MarkupSpan {
                source_range: 10..15,
                target_range: Some(10..25),
                open_markup: "<i>".to_string(),
                close_markup: "</i>".to_string(),
                is_self_closing: false,
                depth: 0,
                is_intra_word: false,
            },
        ];

        enforce_monotonic_order_guard(&mut spans, 30, None);

        let r0 = spans[0].target_range.as_ref().unwrap();
        let r1 = spans[1].target_range.as_ref().unwrap();
        // Spans must not overlap: r0.end <= r1.start
        assert!(r0.end <= r1.start, "Sibling spans must not overlap: r0.end={}, r1.start={}", r0.end, r1.start);
        assert_eq!(spans[0].open_markup, "<b>");
        assert_eq!(spans[1].open_markup, "<i>");
    }

    #[test]
    fn test_monotonic_order_guard_self_closing_tag_bounding() {
        // Source: "Span1 <br/> Span2"
        // Target: "Text1 ... Text2"
        // Span 0: 0..5 ("Span1") -> target 0..10
        // Span 1: 6..6 ("<br/>") -> target 35 (drifted beyond Span 2)
        // Span 2: 7..12 ("Span2") -> target 15..25
        let mut spans = vec![
            MarkupSpan::new_paired(0..5, "<b>".to_string(), "</b>".to_string(), 0),
            MarkupSpan::new_self_closing(6, "<br/>".to_string(), 0),
            MarkupSpan::new_paired(7..12, "<i>".to_string(), "</i>".to_string(), 0),
        ];
        spans[0].target_range = Some(0..10);
        spans[1].target_range = Some(35..35); // Outlier drift!
        spans[2].target_range = Some(15..25);

        enforce_monotonic_order_guard(&mut spans, 40, None);

        let br_span = spans.iter().find(|s| s.is_self_closing).unwrap();
        let br_pos = br_span.target_range.as_ref().unwrap().start;
        // <br/> must be bounded between Span 0 end (10) and Span 2 start (15)
        assert!(
            br_pos >= 10 && br_pos <= 15,
            "br_pos {} must be between 10 and 15",
            br_pos
        );
    }

    #[test]
    fn test_project_markup_spans_self_closing_after_word_placement() {
        // Source: "hello<br/> world"
        // Source tokens: 0..5 ("hello"), 6..11 ("world")
        // Target tokens: 0..6 ("bonjour"), 7..12 ("monde")
        let source_token_spans = vec![0..5, 6..11];
        let target_token_spans = vec![0..7, 8..13];
        let alignments = vec![0, 1]; // "bonjour" -> "hello", "monde" -> "world"
        let mut spans = vec![MarkupSpan::new_self_closing(5, "<br/>".to_string(), 0)];

        project_markup_spans_with_syntax(
            &mut spans,
            &source_token_spans,
            &target_token_spans,
            &alignments,
            13,
            None,
        );

        let r = spans[0].target_range.as_ref().unwrap();
        // Since src_pos is 5 (at end of source token 0), it must be placed at end of target token 0 (7),
        // NOT at the start of target token 0 (0)!
        assert_eq!(r.start, 7, "Self-closing tag after word must be placed at token end");
    }
}

