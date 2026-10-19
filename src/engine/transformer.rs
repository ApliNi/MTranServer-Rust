//! Transformer encoder-decoder forward pass.
//!
//! Mirrors `models/transformer.h` in the marian-fork (the engine behind
//! bergamot and the Mozilla Firefox Translations / OpusMT models).
//!
//! ## Model architecture (from the embedded `special:model.yml`)
//!
//! ```text
//! type: transformer
//! enc-depth: 6, dec-depth: 4
//! dim-emb: 384, transformer-dim-ffn: 1536, transformer-heads: 8
//! transformer-preprocess: ""            (no pre-LN / pre-dropout)
//! transformer-postprocess: dan          (dropout + residual + layer-norm)
//! transformer-postprocess-emb: d        (dropout only; no-op at inference)
//! transformer-postprocess-top: ""       (no final layer-norm)
//! transformer-ffn-activation: relu, transformer-ffn-depth: 2
//! transformer-decoder-autoreg: rnn      (SSRU replaces self-attention)
//! tied-embeddings: true                 (output projection = decoder_Wembᵀ)
//! layer-normalization: false            (SSRU has no internal LN)
//! transformer-train-position-embeddings: false   (sinusoidal positions)
//! ```
//!
//! ## Semantics
//!
//! `forward` performs a complete autoregressive translation:
//!
//! 1. Encoder: `source_ids + [EOS]` (no BOS — marian uses `--bos_id=-1`
//!    for SentencePiece), embed, scale by `sqrt(d_model)`, add sinusoidal
//!    positional embeddings, then run the encoder stack:
//!    `LayerAttention("encoder_l{i}_self")` + `LayerFFN("encoder_l{i}_ffn")`
//!    per layer, each with postprocess "dan" (residual add + layer-norm;
//!    dropout is a no-op at inference).
//! 2. Decoder autoregressive loop: each step feeds the embedding of the
//!    previously generated word (an all-zero embedding for the very first
//!    step — marian's RNN start state), scaled and position-encoded, through
//!    `dec-depth` layers of `SSRU -> cross-attention -> FFN` (marian's
//!    `DecoderLayerRNN` / `LayerAttention` / `LayerFFN`). The SSRU cell state
//!    of each layer is carried across steps.
//! 3. Project the last hidden state with the tied output layer
//!    (`decoder_Wembᵀ · h + decoder_ff_logit_out_b`), gather the shortlist
//!    entries, greedy-argmax over the shortlist to pick the next token, and
//!    repeat until EOS or `max_length_break`.
//! 4. Returns the generated token id sequence (one id per decoding step,
//!    including a trailing EOS or the extra post-truncation token).
//!
//! ## Weight naming and layout
//!
//! There is no graph prefix (unlike older single-factor models, these tensors
//! have no `F0_` prefix) and no `_pre`-suffixed layer-norm parameters
//! (because `transformer-preprocess` is empty, only post-processing
//! layer-norms exist). Layer names are 1-based.
//!
//! All linear weights are stored `[in, out]` — marian's `denseInline` uses
//! `param(prefix + "_W" + suffix, {x->shape()[-1], outDim})` — so the
//! forward computes `x @ W + b` with **no transposition**.
//!
//! LayerNorm weights follow marian's `layerNorm(x, prefix)`, i.e.
//! `prefix + "_ln_scale"` / `prefix + "_ln_bias"`, shape `[1, d_model]`.

pub use super::model::DecoderAutoregType;
use super::model::{Model, Quant8Tensor};
use super::EngineError;
use super::sentencepiece::SentencePiece;
use super::tensor::{
    gemv_biased, matmul_row_i8_dispatch, score_rows_dispatch,
    softmax_scores_inplace, transpose_head_block_tmajor, Tensor2,
};
use crate::utils::profiling;
use std::sync::Arc;

/// Layer-norm epsilon, matching marian's `LayerNormalization` default
/// (`epsilon = 1e-6`).
const LAYER_NORM_EPS: f32 = 1e-6;

/// A resolved affine-projection weight: either a dequantized f32 matrix or a
/// quantized int8 matrix (with its activation quantizer `quant_mult_a`),
/// used by the intgemm path.  Bias is always kept as a plain f32 `Tensor2`.
#[derive(Debug, Clone, Copy)]
pub enum LinW<'a> {
    F32(&'a Tensor2),
    Q8(&'a Quant8Tensor, f32),
}

/// Encoder K projection stored in head-major transposed layout:
/// `kt[h * dk * src_len + t * src_len + j] == tensor[j][h * dk + t]`, i.e.
/// `[heads][dk][src_len]`.
///
/// Decoder cross-attention computes scores via dot products along `dk` per source
/// position. The t-major transposition lets the score kernel stream contiguous
/// j-runs per t step without any runtime transpose.
#[derive(Clone, Default)]
struct HeadMajorK {
    src_len: usize,
    d_model: usize,
    heads: usize,
    dk: usize,
    kt: Vec<f32>,
}

impl HeadMajorK {
    fn empty() -> Self {
        Self::default()
    }
}

#[derive(Clone, Default)]
struct HeadMajorV {
    src_len: usize,
    d_model: usize,
    heads: usize,
    dk: usize,
    data: Vec<f32>,
}

impl HeadMajorV {
    fn empty() -> Self {
        Self::default()
    }
}

/// Permute a `[len, d_model]` K projection into `out` in [`HeadMajorK`] layout.
fn rearrange_k_heads_into(t: &Tensor2, heads: usize, out: &mut HeadMajorK) {
    let src_len = t.shape[0];
    let d_model = t.shape[1];
    let dk = d_model / heads;
    let total_len = src_len * d_model;
    out.src_len = src_len;
    out.d_model = d_model;
    out.heads = heads;
    out.dk = dk;
    if out.kt.len() != total_len {
        out.kt.resize(total_len, 0.0);
    }
    for j in 0..src_len {
        for h in 0..heads {
            let src = j * d_model + h * dk;
            for t2 in 0..dk {
                out.kt[h * dk * src_len + t2 * src_len + j] = t.data[src + t2];
            }
        }
    }
}

/// Permute a `[len, d_model]` V projection into `out` in [`HeadMajorV`] layout.
fn rearrange_v_heads_into(t: &Tensor2, heads: usize, out: &mut HeadMajorV) {
    let src_len = t.shape[0];
    let d_model = t.shape[1];
    let dk = d_model / heads;
    let total_len = src_len * d_model;
    out.src_len = src_len;
    out.d_model = d_model;
    out.heads = heads;
    out.dk = dk;
    if out.data.len() != total_len {
        out.data.resize(total_len, 0.0);
    }
    for j in 0..src_len {
        for h in 0..heads {
            let src = j * d_model + h * dk;
            let dst = h * src_len * dk + j * dk;
            out.data[dst..dst + dk].copy_from_slice(&t.data[src..src + dk]);
        }
    }
}

#[allow(dead_code)]
fn rearrange_k_heads(t: Tensor2, heads: usize) -> HeadMajorK {
    let mut out = HeadMajorK::empty();
    rearrange_k_heads_into(&t, heads, &mut out);
    out
}

#[allow(dead_code)]
fn rearrange_v_heads(t: Tensor2, heads: usize) -> HeadMajorV {
    let mut out = HeadMajorV::empty();
    rearrange_v_heads_into(&t, heads, &mut out);
    out
}
/// One encoder layer's resolved weights.
#[derive(Debug, Clone, Copy)]
pub struct EncoderLayerWeights<'a> {
    pub wq: LinW<'a>,
    pub bq: &'a Tensor2,
    pub wk: LinW<'a>,
    pub bk: &'a Tensor2,
    pub wv: LinW<'a>,
    pub bv: &'a Tensor2,
    pub wo: LinW<'a>,
    pub bo: &'a Tensor2,
    pub ln_attn_scale: &'a Tensor2,
    pub ln_attn_bias: &'a Tensor2,
    pub w1: LinW<'a>,
    pub b1: &'a Tensor2,
    pub w2: LinW<'a>,
    pub b2: &'a Tensor2,
    pub ln_ffn_scale: &'a Tensor2,
    pub ln_ffn_bias: &'a Tensor2,
}

impl<'a> EncoderLayerWeights<'a> {
    fn new(model: &'a Model, names: &EncoderLayerNames, use_q8: bool) -> Self {
        Self {
            wq: param_lw(model, &names.self_wq, use_q8),
            bq: param(model, &names.self_bq),
            wk: param_lw(model, &names.self_wk, use_q8),
            bk: param(model, &names.self_bk),
            wv: param_lw(model, &names.self_wv, use_q8),
            bv: param(model, &names.self_bv),
            wo: param_lw(model, &names.self_wo, use_q8),
            bo: param(model, &names.self_bo),
            ln_attn_scale: param(model, &names.self_ln_attn_scale),
            ln_attn_bias: param(model, &names.self_ln_attn_bias),
            w1: param_lw(model, &names.ffn_w1, use_q8),
            b1: param(model, &names.ffn_b1),
            w2: param_lw(model, &names.ffn_w2, use_q8),
            b2: param(model, &names.ffn_b2),
            ln_ffn_scale: param(model, &names.ffn_ln_scale),
            ln_ffn_bias: param(model, &names.ffn_ln_bias),
        }
    }
}

/// Alias for [`DecoderLayerStaticWeights`] representing decoder layer weights.
pub type DecoderLayerWeights<'a> = DecoderLayerStaticWeights<'a>;

/// One decoder layer's resolved static weights.
#[derive(Debug, Clone, Copy)]
pub struct DecoderLayerStaticWeights<'a> {
    // SSRU weights (present when autoreg_type == Rnn)
    pub rnn_w: Option<LinW<'a>>,
    pub rnn_wf: Option<LinW<'a>>,
    pub rnn_bf: Option<&'a Tensor2>,
    pub rnn_ln_scale: Option<&'a Tensor2>,
    pub rnn_ln_bias: Option<&'a Tensor2>,
    // Self-attention weights (present when autoreg_type == SelfAttention)
    pub self_wq: Option<LinW<'a>>,
    pub self_bq: Option<&'a Tensor2>,
    pub self_wk: Option<LinW<'a>>,
    pub self_bk: Option<&'a Tensor2>,
    pub self_wv: Option<LinW<'a>>,
    pub self_bv: Option<&'a Tensor2>,
    pub self_wo: Option<LinW<'a>>,
    pub self_bo: Option<&'a Tensor2>,
    pub self_ln_scale: Option<&'a Tensor2>,
    pub self_ln_bias: Option<&'a Tensor2>,
    // Cross-attention weights
    pub ctx_wq: LinW<'a>,
    pub ctx_bq: &'a Tensor2,
    pub ctx_wk: LinW<'a>,
    pub ctx_bk: &'a Tensor2,
    pub ctx_wv: LinW<'a>,
    pub ctx_bv: &'a Tensor2,
    pub ctx_wo: LinW<'a>,
    pub ctx_bo: &'a Tensor2,
    pub ctx_ln_scale: &'a Tensor2,
    pub ctx_ln_bias: &'a Tensor2,
    // FFN weights
    pub ffn_w1: LinW<'a>,
    pub ffn_b1: &'a Tensor2,
    pub ffn_w2: LinW<'a>,
    pub ffn_b2: &'a Tensor2,
    pub ffn_ln_scale: &'a Tensor2,
    pub ffn_ln_bias: &'a Tensor2,
}

impl<'a> DecoderLayerStaticWeights<'a> {
    fn new(
        model: &'a Model,
        names: &DecoderLayerNames,
        use_q8: bool,
        autoreg_type: DecoderAutoregType,
    ) -> Self {
        let (rnn_w, rnn_wf, rnn_bf, rnn_ln_scale, rnn_ln_bias) = match autoreg_type {
            DecoderAutoregType::Rnn => (
                Some(param_lw(model, &names.rnn_w, use_q8)),
                Some(param_lw(model, &names.rnn_wf, use_q8)),
                Some(param(model, &names.rnn_bf)),
                Some(param(model, &names.rnn_ln_scale)),
                Some(param(model, &names.rnn_ln_bias)),
            ),
            DecoderAutoregType::SelfAttention => (None, None, None, None, None),
        };

        let (
            self_wq,
            self_bq,
            self_wk,
            self_bk,
            self_wv,
            self_bv,
            self_wo,
            self_bo,
            self_ln_scale,
            self_ln_bias,
        ) = match autoreg_type {
            DecoderAutoregType::SelfAttention => {
                let resolve_p = |name: &str| {
                    if model.contains(name) {
                        param(model, name)
                    } else {
                        let alt = if name.contains("_W") {
                            name.replace("_W", "_w")
                        } else {
                            name.to_lowercase()
                        };
                        param(model, &alt)
                    }
                };
                let resolve_lw = |name: &str| {
                    if model.contains(name) {
                        param_lw(model, name, use_q8)
                    } else {
                        let alt = if name.contains("_W") {
                            name.replace("_W", "_w")
                        } else {
                            name.to_lowercase()
                        };
                        param_lw(model, &alt, use_q8)
                    }
                };
                let ln_scale = if model.contains(&names.self_ln_scale) {
                    param(model, &names.self_ln_scale)
                } else if model.contains(&names.self_ln_scale.replace("_Wo_ln_scale", "_ln_scale")) {
                    param(model, &names.self_ln_scale.replace("_Wo_ln_scale", "_ln_scale"))
                } else {
                    param(model, &names.self_ln_scale.replace("_Wo_ln_scale", "_ffn_ln_scale"))
                };
                let ln_bias = if model.contains(&names.self_ln_bias) {
                    param(model, &names.self_ln_bias)
                } else if model.contains(&names.self_ln_bias.replace("_Wo_ln_bias", "_ln_bias")) {
                    param(model, &names.self_ln_bias.replace("_Wo_ln_bias", "_ln_bias"))
                } else {
                    param(model, &names.self_ln_bias.replace("_Wo_ln_bias", "_ffn_ln_bias"))
                };
                (
                    Some(resolve_lw(&names.self_wq)),
                    Some(resolve_p(&names.self_bq)),
                    Some(resolve_lw(&names.self_wk)),
                    Some(resolve_p(&names.self_bk)),
                    Some(resolve_lw(&names.self_wv)),
                    Some(resolve_p(&names.self_bv)),
                    Some(resolve_lw(&names.self_wo)),
                    Some(resolve_p(&names.self_bo)),
                    Some(ln_scale),
                    Some(ln_bias),
                )
            }
            DecoderAutoregType::Rnn => (None, None, None, None, None, None, None, None, None, None),
        };

        Self {
            rnn_w,
            rnn_wf,
            rnn_bf,
            rnn_ln_scale,
            rnn_ln_bias,
            self_wq,
            self_bq,
            self_wk,
            self_bk,
            self_wv,
            self_bv,
            self_wo,
            self_bo,
            self_ln_scale,
            self_ln_bias,
            ctx_wq: param_lw(model, &names.ctx_wq, use_q8),
            ctx_bq: param(model, &names.ctx_bq),
            ctx_wk: param_lw(model, &names.ctx_wk, use_q8),
            ctx_bk: param(model, &names.ctx_bk),
            ctx_wv: param_lw(model, &names.ctx_wv, use_q8),
            ctx_bv: param(model, &names.ctx_bv),
            ctx_wo: param_lw(model, &names.ctx_wo, use_q8),
            ctx_bo: param(model, &names.ctx_bo),
            ctx_ln_scale: param(model, &names.ctx_ln_scale),
            ctx_ln_bias: param(model, &names.ctx_ln_bias),
            ffn_w1: param_lw(model, &names.ffn_w1, use_q8),
            ffn_b1: param(model, &names.ffn_b1),
            ffn_w2: param_lw(model, &names.ffn_w2, use_q8),
            ffn_b2: param(model, &names.ffn_b2),
            ffn_ln_scale: param(model, &names.ffn_ln_scale),
            ffn_ln_bias: param(model, &names.ffn_ln_bias),
        }
    }
}

/// Source embedding weight representation.
#[derive(Debug, Clone, Copy)]
pub enum SrcEmbedding<'a> {
    F32(&'a Tensor2),
    Q8(&'a Quant8Tensor),
}

/// Pre-resolved weight references bound to a model's lifetime.
#[derive(Debug)]
pub struct LayerWeightsView<'a> {
    pub encoder_layers: Vec<EncoderLayerWeights<'a>>,
    pub decoder_layers: Vec<DecoderLayerStaticWeights<'a>>,
    pub src_wemb: SrcEmbedding<'a>,
    pub output_bias: &'a Tensor2,
    pub trg_wemb_q8: Option<&'a Quant8Tensor>,
    pub trg_wemb_f32: Option<&'a Tensor2>,
    pub proj_qa: f32,
    pub d_ff: usize,
}

impl<'a> LayerWeightsView<'a> {
    pub fn new(model: &'a Model, config: &ForwardConfig) -> Self {
        let encoder_layers = config
            .encoder_names
            .iter()
            .map(|names| EncoderLayerWeights::new(model, names, config.use_q8))
            .collect();
        let decoder_layers: Vec<DecoderLayerStaticWeights<'a>> = config
            .decoder_names
            .iter()
            .map(|names| {
                DecoderLayerStaticWeights::new(
                    model,
                    names,
                    config.use_q8,
                    config.autoreg_type,
                )
            })
            .collect();

        let src_name = src_wemb_name(model);
        let src_wemb = if config.use_q8 {
            if let Some(q) = model.get_quant8(src_name) {
                SrcEmbedding::Q8(q)
            } else {
                SrcEmbedding::F32(param(model, src_name))
            }
        } else {
            SrcEmbedding::F32(param(model, src_name))
        };

        let trg_name = if model.contains("decoder_Wemb") {
            "decoder_Wemb"
        } else {
            "Wemb"
        };
        let (trg_wemb_q8, trg_wemb_f32, proj_qa) = if config.use_q8 {
            let q = model
                .get_quant8(trg_name)
                .expect("model is missing int8 embedding");
            let qa = model.quant_mult_a(trg_name).unwrap_or(1.0);
            (Some(q), None, qa)
        } else {
            (None, Some(param(model, trg_name)), 0.0)
        };

        let output_bias = param(model, "decoder_ff_logit_out_b");

        let d_ff = decoder_layers
            .first()
            .map(|l| match &l.ffn_w1 {
                LinW::F32(w) => w.shape[1],
                LinW::Q8(wq, _) => wq.shape[0],
            })
            .unwrap_or(1536);

        Self {
            encoder_layers,
            decoder_layers,
            src_wemb,
            output_bias,
            trg_wemb_q8,
            trg_wemb_f32,
            proj_qa,
            d_ff,
        }
    }
}

/// Pre-resolved transformer layer weights bound to an [`Arc<Model>`].
///
/// Holds the backing [`Model`] on the heap via `Arc` and caches pre-resolved
/// weight views in [`LayerWeightsView`], completely eliminating hash map
/// lookups and vector allocations on the translation hot path.
pub struct PreparedEngineWeights {
    model: Arc<Model>,
    weights: LayerWeightsView<'static>,
}

impl PreparedEngineWeights {
    /// Pre-resolve all weights for `model` according to `config`.
    pub fn new(model: Arc<Model>, config: &ForwardConfig) -> Self {
        let model_ptr: *const Model = &*model;
        // SAFETY: `model` is owned by `PreparedEngineWeights` via `Arc<Model>`,
        // keeping the underlying `Model` heap allocation alive and pinned.
        // The static reference is never exposed directly; callers only receive
        // `&LayerWeightsView<'_>` with a lifetime tied to `&self`.
        let static_model: &'static Model = unsafe { &*model_ptr };
        let weights = LayerWeightsView::new(static_model, config);
        Self { model, weights }
    }

    /// Access the pre-resolved layer weights.
    #[inline]
    pub fn weights(&self) -> &LayerWeightsView<'_> {
        &self.weights
    }

    /// Access the underlying [`Model`].
    #[inline]
    pub fn model(&self) -> &Arc<Model> {
        &self.model
    }
}

impl std::fmt::Debug for PreparedEngineWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedEngineWeights")
            .field("weights", &self.weights)
            .finish_non_exhaustive()
    }
}
/// KV-Cache for a single decoder layer during autoregressive generation.
#[derive(Debug, Clone, Default)]
pub struct DecoderLayerKVCache {
    pub k: Tensor2,
    pub v: Tensor2,
}

impl DecoderLayerKVCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(max_len: usize, d_model: usize) -> Self {
        let mut k = Tensor2::zeros([0, d_model]);
        k.data.reserve(max_len * d_model);
        let mut v = Tensor2::zeros([0, d_model]);
        v.data.reserve(max_len * d_model);
        Self { k, v }
    }

    pub fn reset(&mut self) {
        let d_model = self.k.shape[1];
        self.k.shape = [0, d_model];
        self.k.data.clear();
        self.v.shape = [0, d_model];
        self.v.data.clear();
    }

    pub fn step(&self) -> usize {
        self.k.shape[0]
    }

    pub fn append(&mut self, k_step: &[f32], v_step: &[f32], d_model: usize) {
        let curr_len = self.k.shape[0];
        self.k.data.extend_from_slice(k_step);
        self.v.data.extend_from_slice(v_step);
        self.k.set_shape_2d(curr_len + 1, d_model);
        self.v.set_shape_2d(curr_len + 1, d_model);
    }
}

/// Multi-layer KV-Cache holding all decoder layers' past keys and values.
#[derive(Debug, Clone, Default)]
pub struct DecoderKVCache {
    pub layers: Vec<DecoderLayerKVCache>,
}

impl DecoderKVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| DecoderLayerKVCache::new()).collect(),
        }
    }

    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
    }

    pub fn resize(&mut self, num_layers: usize) {
        if self.layers.len() != num_layers {
            self.layers = (0..num_layers).map(|_| DecoderLayerKVCache::new()).collect();
        } else {
            self.reset();
        }
    }
}

/// Per-forward-pass reusable intermediate buffers for the decode loop.
///
/// Every field is a scratch tensor whose data buffer is resized and
/// overwritten in place by the `_into` kernel variants, so an autoregressive
/// step performs no heap allocations. Sizes are established lazily on first
/// use (`resize`) or eagerly in [`Scratch::new`].
struct Scratch {
    /// SSRU relu output (pre-residual).
    a: Tensor2,
    /// SSRU forget-gate pre-activation.
    b: Tensor2,
    /// SSRU residual/LN result; also the cross-attention residual target.
    t1: Tensor2,
    /// Cross-attention residual add result.
    t2: Tensor2,
    /// Projected Q `[1, d_model]`.
    q: Tensor2,
    /// Projected K for current step self-attention `[1, d_model]`.
    self_k: Tensor2,
    /// Projected V for current step self-attention `[1, d_model]`.
    self_v: Tensor2,
    /// Attention context (post-Wo) `[1, d_model]`.
    attn: Tensor2,
    /// Attention weighted-V sum (pre-Wo) `[1, d_model]`, so the output
    /// projection reads one pre-sized buffer and writes the other.
    ctx_pre: Tensor2,
    /// Attention score rows `[q_len, kv_len]` (reused per head).
    scores: Tensor2,
    /// FFN first projection `[1, d_ff]`.
    h1: Tensor2,
}

impl Scratch {
    fn new(d_model: usize, d_ff: usize) -> Self {
        Self {
            a: Tensor2::zeros([1, d_model]),
            b: Tensor2::zeros([1, d_model]),
            t1: Tensor2::zeros([1, d_model]),
            t2: Tensor2::zeros([1, d_model]),
            q: Tensor2::zeros([1, d_model]),
            self_k: Tensor2::zeros([1, d_model]),
            self_v: Tensor2::zeros([1, d_model]),
            attn: Tensor2::zeros([1, d_model]),
            ctx_pre: Tensor2::zeros([1, d_model]),
            scores: Tensor2::zeros([1, 1]),
            h1: Tensor2::zeros([1, d_ff]),
        }
    }

    fn reset(&mut self, d_model: usize, d_ff: usize) {
        let reset_tensor = |t: &mut Tensor2, len: usize| {
            t.set_shape_2d(1, len);
            if t.data.len() != len {
                t.data.resize(len, 0.0);
            }
        };
        reset_tensor(&mut self.a, d_model);
        reset_tensor(&mut self.b, d_model);
        reset_tensor(&mut self.t1, d_model);
        reset_tensor(&mut self.t2, d_model);
        reset_tensor(&mut self.q, d_model);
        reset_tensor(&mut self.self_k, d_model);
        reset_tensor(&mut self.self_v, d_model);
        reset_tensor(&mut self.attn, d_model);
        reset_tensor(&mut self.ctx_pre, d_model);
        self.scores.set_shape_2d(1, 1);
        reset_tensor(&mut self.h1, d_ff);
}
}
/// Reusable forward decoder state buffers per thread.
struct ForwardBuffers {
    cell_states: Vec<Tensor2>,
    cell_scratches: Vec<Tensor2>,
    kv_cache: DecoderKVCache,
    scratch: Scratch,
    step_input: Tensor2,
    xq_row: Vec<i8>,
    logit_buf: Vec<f32>,
    dec_ids: Vec<u32>,
    short_proj_q8_data: Vec<i8>,
    short_row_sums: Vec<i32>,
    short_bias: Vec<f32>,
    short_proj_f32: Vec<f32>,
    cached_wemb_ptr: usize,
    cached_shortlist: Vec<u32>,
    enc_k: Vec<HeadMajorK>,
    enc_v: Vec<HeadMajorV>,
    linear_scratch: Tensor2,
    linear_scratch_v: Tensor2,
    enc_state: Tensor2,
    attn_accum: Vec<f32>,
}

impl ForwardBuffers {
    fn new(dec_layers: usize, d_model: usize, d_ff: usize) -> Self {
        Self {
            cell_states: (0..dec_layers)
                .map(|_| Tensor2::zeros([1, d_model]))
                .collect(),
            cell_scratches: (0..dec_layers)
                .map(|_| Tensor2::zeros([1, d_model]))
                .collect(),
            kv_cache: DecoderKVCache::new(dec_layers),
            scratch: Scratch::new(d_model, d_ff),
            step_input: Tensor2::zeros([1, d_model]),
            xq_row: vec![0; d_model],
            logit_buf: Vec::new(),
            dec_ids: Vec::new(),
            short_proj_q8_data: Vec::new(),
            short_row_sums: Vec::new(),
            short_bias: Vec::new(),
            short_proj_f32: Vec::new(),
            cached_wemb_ptr: 0,
            cached_shortlist: Vec::new(),
            enc_k: (0..dec_layers).map(|_| HeadMajorK::empty()).collect(),
            enc_v: (0..dec_layers).map(|_| HeadMajorV::empty()).collect(),
            linear_scratch: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            linear_scratch_v: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            enc_state: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            attn_accum: Vec::new(),
        }
    }

    fn reset(&mut self, dec_layers: usize, d_model: usize, d_ff: usize) {
        if self.cell_states.len() < dec_layers {
            self.cell_states
                .resize_with(dec_layers, || Tensor2::zeros([1, d_model]));
        }
        if self.cell_scratches.len() < dec_layers {
            self.cell_scratches
                .resize_with(dec_layers, || Tensor2::zeros([1, d_model]));
        }
        self.kv_cache.resize(dec_layers);
        if self.enc_k.len() < dec_layers {
            self.enc_k.resize_with(dec_layers, HeadMajorK::empty);
        }
        if self.enc_v.len() < dec_layers {
            self.enc_v.resize_with(dec_layers, HeadMajorV::empty);
        }
        for t in self.cell_states[..dec_layers].iter_mut() {
            t.set_shape_2d(1, d_model);
            if t.data.len() != d_model {
                t.data.resize(d_model, 0.0);
            }
            t.data.fill(0.0);
        }
        for t in self.cell_scratches[..dec_layers].iter_mut() {
            t.set_shape_2d(1, d_model);
            if t.data.len() != d_model {
                t.data.resize(d_model, 0.0);
            }
            t.data.fill(0.0);
        }
        self.scratch.reset(d_model, d_ff);
        self.step_input.set_shape_2d(1, d_model);
        if self.step_input.data.len() != d_model {
            self.step_input.data.resize(d_model, 0.0);
        }
        self.xq_row.resize(d_model, 0);
        self.logit_buf.clear();
        self.dec_ids.clear();
        // Retain short_proj_q8_data, short_row_sums, short_bias, short_proj_f32
        // and cached_shortlist across forward calls to reuse shortlist projections.
        self.attn_accum.clear();
    }
}
thread_local! {
    static FORWARD_BUFFERS: std::cell::RefCell<Option<ForwardBuffers>> =
        const { std::cell::RefCell::new(None) };
}

/// Per-forward-pass reusable intermediate buffers for the **encoder** stack.
///
/// Sizes are established lazily on first use (each `_into` kernel resizes the
/// target buffer), so a full source sentence spans every encoder layer without
/// allocating a fresh `[seq, d_model]`/`[seq, d_ff]` tensor per sub-layer.
struct EncoderScratch {
    /// Projected Q `[seq, d_model]`.
    q: Tensor2,
    /// Projected K `[seq, d_model]`.
    k: Tensor2,
    /// Projected V `[seq, d_model]`.
    v: Tensor2,
    /// Attention context (post-Wo) `[seq, d_model]`.
    ctx: Tensor2,
    /// Attention score rows `[seq, kv_len]` (reused per head).
    scores: Tensor2,
    /// Attention postprocess "dan" residual+LN scratch `[seq, d_model]`.
    res: Tensor2,
    /// FFN first projection / residual-LN scratch `[seq, d_ff]`.
    h1: Tensor2,
    /// T-major transposed K head panel for the dispatched score kernel
    /// (`[dk][kv_len]`, reused per head).
    kt_panel: Tensor2,
}

impl EncoderScratch {
    fn empty() -> Self {
        Self {
            q: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            k: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            v: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            ctx: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            scores: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            res: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            h1: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            kt_panel: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
        }
    }
}
struct EncoderBuffers {
    scratch: EncoderScratch,
    x: Tensor2,
    out: Tensor2,
    ids: Vec<u32>,
}

thread_local! {
    static ENCODER_BUFFERS: std::cell::RefCell<Option<EncoderBuffers>> =
        const { std::cell::RefCell::new(None) };
}

/// FFN activation function, resolved from `transformer-ffn-activation` once
/// at model-load time (unknown names fail loudly there, mirroring the
/// explicit-failure policy of [`param`]) so the per-request path can never
/// panic on a bad name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Relu,
    /// swish / silu (two names, one function)
    Silu,
}

impl Activation {
    /// Parse a marian activation name; `None` for unsupported names.
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "relu" => Some(Self::Relu),
            "swish" | "silu" => Some(Self::Silu),
            _ => None,
        }
    }
}

/// Pre-resolved weight names for one encoder layer (layer number baked in).
///
/// Built once by [`ForwardConfig::resolve`] so the per-request encoder pass
/// resolves weights by table lookup instead of re-`format!`ing the same
/// names on every call.
#[derive(Debug, Clone)]
pub struct EncoderLayerNames {
    // Self-attention group (`encoder_l{i}_self_*`)
    pub self_wq: String,
    pub self_bq: String,
    pub self_wk: String,
    pub self_bk: String,
    pub self_wv: String,
    pub self_bv: String,
    pub self_wo: String,
    pub self_bo: String,
    pub self_ln_attn_scale: String,
    pub self_ln_attn_bias: String,
    // Feed-forward group (`encoder_l{i}_ffn_*`)
    pub ffn_w1: String,
    pub ffn_b1: String,
    pub ffn_w2: String,
    pub ffn_b2: String,
    pub ffn_ln_scale: String,
    pub ffn_ln_bias: String,
}

impl EncoderLayerNames {
    /// Build the weight names for encoder layer `layer_idx` (already
    /// offset-adjusted; marian names its layers 1-based by default).
    fn build(layer_idx: usize) -> Self {
        let self_prefix = format!("encoder_l{layer_idx}_self");
        let ffn_prefix = format!("encoder_l{layer_idx}_ffn");
        Self {
            self_wq: format!("{self_prefix}_Wq"),
            self_bq: format!("{self_prefix}_bq"),
            self_wk: format!("{self_prefix}_Wk"),
            self_bk: format!("{self_prefix}_bk"),
            self_wv: format!("{self_prefix}_Wv"),
            self_bv: format!("{self_prefix}_bv"),
            self_wo: format!("{self_prefix}_Wo"),
            self_bo: format!("{self_prefix}_bo"),
            self_ln_attn_scale: format!("{self_prefix}_Wo_ln_scale"),
            self_ln_attn_bias: format!("{self_prefix}_Wo_ln_bias"),
            ffn_w1: format!("{ffn_prefix}_W1"),
            ffn_b1: format!("{ffn_prefix}_b1"),
            ffn_w2: format!("{ffn_prefix}_W2"),
            ffn_b2: format!("{ffn_prefix}_b2"),
            ffn_ln_scale: format!("{ffn_prefix}_ffn_ln_scale"),
            ffn_ln_bias: format!("{ffn_prefix}_ffn_ln_bias"),
        }
    }
}

/// Pre-resolved weight names for one decoder layer (layer number baked in),
/// mirroring [`EncoderLayerNames`] for the per-request decoder pass.
#[derive(Debug, Clone)]
pub struct DecoderLayerNames {
    // SSRU group (`decoder_l{i}_rnn_*`)
    pub rnn_w: String,
    pub rnn_wf: String,
    pub rnn_bf: String,
    pub rnn_ln_scale: String,
    pub rnn_ln_bias: String,
    // Self-attention group (`decoder_l{i}_self_*`)
    pub self_wq: String,
    pub self_bq: String,
    pub self_wk: String,
    pub self_bk: String,
    pub self_wv: String,
    pub self_bv: String,
    pub self_wo: String,
    pub self_bo: String,
    pub self_ln_scale: String,
    pub self_ln_bias: String,
    // Cross-attention group (`decoder_l{i}_context_*`)
    pub ctx_wq: String,
    pub ctx_bq: String,
    pub ctx_wk: String,
    pub ctx_bk: String,
    pub ctx_wv: String,
    pub ctx_bv: String,
    pub ctx_wo: String,
    pub ctx_bo: String,
    pub ctx_ln_scale: String,
    pub ctx_ln_bias: String,
    // Feed-forward group (`decoder_l{i}_ffn_*`)
    pub ffn_w1: String,
    pub ffn_b1: String,
    pub ffn_w2: String,
    pub ffn_b2: String,
    pub ffn_ln_scale: String,
    pub ffn_ln_bias: String,
}

impl DecoderLayerNames {
    /// Build the weight names for decoder layer `layer_idx` (already
    /// offset-adjusted).
    fn build(layer_idx: usize) -> Self {
        let prefix = format!("decoder_l{layer_idx}");
        let rnn_prefix = format!("{prefix}_rnn");
        let self_prefix = format!("{prefix}_self");
        let ctx_prefix = format!("{prefix}_context");
        let ffn_prefix = format!("{prefix}_ffn");
        Self {
            rnn_w: format!("{rnn_prefix}_W"),
            rnn_wf: format!("{rnn_prefix}_Wf"),
            rnn_bf: format!("{rnn_prefix}_bf"),
            rnn_ln_scale: format!("{rnn_prefix}_ffn_ln_scale"),
            rnn_ln_bias: format!("{rnn_prefix}_ffn_ln_bias"),
            self_wq: format!("{self_prefix}_Wq"),
            self_bq: format!("{self_prefix}_bq"),
            self_wk: format!("{self_prefix}_Wk"),
            self_bk: format!("{self_prefix}_bk"),
            self_wv: format!("{self_prefix}_Wv"),
            self_bv: format!("{self_prefix}_bv"),
            self_wo: format!("{self_prefix}_Wo"),
            self_bo: format!("{self_prefix}_bo"),
            self_ln_scale: format!("{self_prefix}_Wo_ln_scale"),
            self_ln_bias: format!("{self_prefix}_Wo_ln_bias"),
            ctx_wq: format!("{ctx_prefix}_Wq"),
            ctx_bq: format!("{ctx_prefix}_bq"),
            ctx_wk: format!("{ctx_prefix}_Wk"),
            ctx_bk: format!("{ctx_prefix}_bk"),
            ctx_wv: format!("{ctx_prefix}_Wv"),
            ctx_bv: format!("{ctx_prefix}_bv"),
            ctx_wo: format!("{ctx_prefix}_Wo"),
            ctx_bo: format!("{ctx_prefix}_bo"),
            ctx_ln_scale: format!("{ctx_prefix}_Wo_ln_scale"),
            ctx_ln_bias: format!("{ctx_prefix}_Wo_ln_bias"),
            ffn_w1: format!("{ffn_prefix}_W1"),
            ffn_b1: format!("{ffn_prefix}_b1"),
            ffn_w2: format!("{ffn_prefix}_W2"),
            ffn_b2: format!("{ffn_prefix}_b2"),
            ffn_ln_scale: format!("{ffn_prefix}_ffn_ln_scale"),
            ffn_ln_bias: format!("{ffn_prefix}_ffn_ln_bias"),
        }
    }
}

/// Everything [`forward`] needs, resolved once when the engine is created.
///
/// Reading the model YAML (`get_config_value`) takes the model's internal
/// mutex on every call, so the per-request path must not touch it: the
/// shape knobs and activation are parsed once here, together with the
/// engine-level knobs copied out of [`EngineConfig`].
#[derive(Debug, Clone)]
pub struct ForwardConfig {
    pub d_model: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    /// FFN activation, resolved from the model YAML at load time.
    pub activation: Activation,
    pub autoreg_type: DecoderAutoregType,
    pub use_q8: bool,
    /// Decoder stop length (from `EngineConfig::max_length_break`).
    pub max_length_break: usize,
    /// Layer-index offset (0- or 1-based naming), resolved once at load time
    /// so the request path never probes the model's item table.
    pub layer_offset: usize,
    /// Precomputed sinusoidal position table, one `d_model`-wide row per
    /// position `0..=max_length_break`, written by [`Self::resolve`] at load
    /// time. Positions are fixed and bounded, so the decoder indexes this
    /// table instead of re-evaluating `exp`/`sin`/`cos` on every forward
    /// step. Values are bit-identical to [`sinusoidal_positions`] (same
    /// per-element formula).
    pub position_table: Vec<f32>,
    /// Per-layer weight names (layer number baked in), prebuilt by
    /// [`Self::resolve`] so the request path resolves weights by table
    /// lookup instead of re-`format!`ing them on every forward pass. Names
    /// already include [`Self::layer_offset`]; callers index by layer
    /// position.
    pub encoder_names: Vec<EncoderLayerNames>,
    pub decoder_names: Vec<DecoderLayerNames>,
    pub beam_size: usize,
    pub length_penalty: f32,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub expansion_ratio: f32,
    pub min_length_factor: f32,
    pub max_length_factor: f32,
}

impl ForwardConfig {
    /// Resolve the transformer configuration from the model's embedded YAML
    /// plus the engine config. Called once from `TranslationEngine::new`;
    /// an unsupported `transformer-ffn-activation` aborts model loading here.
    pub fn resolve(model: &Model, config: &super::EngineConfig) -> Result<Self, EngineError> {
        Self::resolve_with_languages(model, config, "", "")
    }

    /// Resolve the transformer configuration including language-pair adaptive density factors.
    pub fn resolve_with_languages(
        model: &Model,
        config: &super::EngineConfig,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<Self, EngineError> {
        let activation = model
            .get_config_value("transformer-ffn-activation")
            .unwrap_or_else(|| "relu".to_string());
        let activation = Activation::from_name(&activation).ok_or_else(|| {
            EngineError::Format(format!("Unsupported transformer-ffn-activation '{activation}'"))
        })?;
        let d_model = config_int(model, "dim-emb", 384);
        let max_length_break = config.max_length_break;
        let enc_layers = config_int(model, "enc-depth", 6);
        let dec_layers = config_int(model, "dec-depth", 4);
        let layer_offset = layer_indexing(model);
        let num_timescales = d_model / 2;
        let log_timescale_increment = (10000.0f32).ln() / (num_timescales as f32 - 1.0);
        // Pre-warm the shared position table covering every position the
        // decoder can request (`0..=max_length_break`). Same formula as
        // `sinusoidal_positions`, computed once here.
        let mut position_table = vec![0.0f32; (max_length_break + 1) * d_model];
        for p in 0..=max_length_break {
            let pf = p as f32;
            for i in 0..num_timescales {
                let v = pf * ((i as f32) * -log_timescale_increment).exp();
                position_table[p * d_model + i] = v.sin();
                position_table[p * d_model + num_timescales + i] = v.cos();
            }
        }
        // Prebuild the per-layer weight-name tables: the `format!`s here run
        // once at engine construction, never on the request path.
        let encoder_names = (0..enc_layers)
            .map(|i| EncoderLayerNames::build(i + layer_offset))
            .collect();
        let decoder_names = (0..dec_layers)
            .map(|i| DecoderLayerNames::build(i + layer_offset))
            .collect();
        let beam_size = config.beam_size.clamp(1, 8);
        let length_penalty = config.length_penalty;
        let autoreg = model
            .get_config_value("transformer-decoder-autoreg")
            .map(|s| s.trim().to_ascii_lowercase());
        let autoreg_type = match autoreg.as_deref() {
            Some("rnn") => DecoderAutoregType::Rnn,
            Some("self-attention") => DecoderAutoregType::SelfAttention,
            None => {
                if model.get_config_value("dec-cell").as_deref() == Some("ssru") {
                    DecoderAutoregType::Rnn
                } else {
                    DecoderAutoregType::SelfAttention
                }
            }
            Some(_) => DecoderAutoregType::SelfAttention,
        };
        let (source_density, _src_min_f, _src_max_f) = if !source_lang.is_empty() {
            if let Some(prof) = crate::engine::rules::RuleRegistry::global().get(source_lang) {
                (prof.generation.token_density_ratio, prof.generation.min_length_factor, prof.generation.max_length_factor)
            } else {
                (1.0, 0.5, 2.5)
            }
        } else {
            (1.0, 0.5, 2.5)
        };

        let (target_density, trg_min_f, trg_max_f, trg_rep_penalty, trg_ngram_size) = if !target_lang.is_empty() {
            if let Some(prof) = crate::engine::rules::RuleRegistry::global().get(target_lang) {
                (
                    prof.generation.token_density_ratio,
                    prof.generation.min_length_factor,
                    prof.generation.max_length_factor,
                    prof.generation.repetition_penalty,
                    prof.generation.no_repeat_ngram_size,
                )
            } else {
                (1.0, 0.5, 2.5, None, None)
            }
        } else {
            (1.0, 0.5, 2.5, None, None)
        };

        let default_config = super::EngineConfig::default();
        let repetition_penalty = if (config.repetition_penalty - default_config.repetition_penalty).abs() < 1e-6
            || config.repetition_penalty <= 0.0
        {
            trg_rep_penalty.unwrap_or(config.repetition_penalty)
        } else {
            config.repetition_penalty
        };

        let mut no_repeat_ngram_size = if config.no_repeat_ngram_size == default_config.no_repeat_ngram_size {
            trg_ngram_size.unwrap_or(config.no_repeat_ngram_size)
        } else {
            config.no_repeat_ngram_size
        };

        // Guarantee smooth decoding for CJK languages with frequent single characters,
        // pre-resolving at configuration time to avoid per-forward hot-path vocabulary scanning.
        if (crate::utils::lang::is_cjk_code(target_lang) || target_lang.starts_with("zh"))
            && no_repeat_ngram_size > 0
            && no_repeat_ngram_size < 5
            && config.no_repeat_ngram_size == default_config.no_repeat_ngram_size
        {
            no_repeat_ngram_size = 5;
        }
        let raw_ratio = config.expansion_ratio.unwrap_or_else(|| {
            if source_density > 0.0 {
                target_density / source_density
            } else {
                1.0
            }
        });
        let expansion_ratio = if raw_ratio > 0.0 { raw_ratio } else { 1.0 };
        let min_length_factor = config.min_length_factor.unwrap_or(trg_min_f).max(0.01);
        let max_length_factor = config.max_length_factor.unwrap_or(trg_max_f).max(min_length_factor);

        Ok(Self {
            d_model,
            heads: config_int(model, "transformer-heads", 8),
            enc_layers,
            dec_layers,
            activation,
            autoreg_type,
            use_q8: config.gemm_precision.starts_with("int8"),
            max_length_break,
            layer_offset,
            position_table,
            encoder_names,
            decoder_names,
            beam_size,
            length_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            expansion_ratio,
            min_length_factor,
            max_length_factor,
        })
    }
    /// Bind and cache model weights into a lifetime-bound [`LayerWeightsView`],
    /// completely eliminating hash map lookups on the inference hot path.
    pub fn weights<'a>(&self, model: &'a Model) -> LayerWeightsView<'a> {
        LayerWeightsView::new(model, self)
    }
    /// Bind and cache model weights into an [`Arc`]-backed [`PreparedEngineWeights`],
    /// completely eliminating hash map lookups on the inference hot path.
    pub fn prepare_weights(&self, model: Arc<Model>) -> PreparedEngineWeights {
        PreparedEngineWeights::new(model, self)
    }
}
/// Transformer encoder-decoder namespace: [`Transformer::forward`] runs a
/// complete autoregressive translation over the loaded model.
pub struct Transformer;

/// Fetch a model tensor by name, panicking with a descriptive message if the
/// model does not contain it. Explicit failure is intentional: a silently
/// missing weight would produce garbage translations instead of a clear error.
fn param<'a>(model: &'a Model, name: &str) -> &'a super::model::Tensor {
    model
        .get(name)
        .unwrap_or_else(|| panic!("model is missing tensor '{}'", name))
}

/// Resolve an affine weight by name, preferring the quantized int8 form when
/// the engine runs in int8 gemm mode and the tensor is intgemm-quantized;
/// falls back to the dequantized f32 weight otherwise.
fn param_lw<'a>(model: &'a Model, name: &str, use_q8: bool) -> LinW<'a> {
    if use_q8 {
        if let Some(q) = model.get_quant8(name) {
            let qa = model.quant_mult_a(name).unwrap_or(1.0);
            return LinW::Q8(q, qa);
        }
    }
    LinW::F32(param(model, name))
}

// ---------------------------------------------------------------------------
// Helpers: embedding lookup, activation, sinusoidal positions
// ---------------------------------------------------------------------------

/// Dequantizes an i8 embedding row into f32 using AVX2 SIMD instructions.
///
/// # Safety
/// Requires AVX2 CPU features. `src.len()` must equal `dst.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn embed_src_row_avx2(src: &[i8], dst: &mut [f32], inv_qm: f32) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 8;
    let inv_qm_v = _mm256_set1_ps(inv_qm);

    for c in 0..chunks {
        let off = c * 8;
        let raw = (src.as_ptr().add(off) as *const i64).read_unaligned();
        let v8 = _mm_cvtsi64_si128(raw);
        let v32 = _mm256_cvtepi8_epi32(v8);
        let vps = _mm256_cvtepi32_ps(v32);
        let res = _mm256_mul_ps(vps, inv_qm_v);
        _mm256_storeu_ps(dst.as_mut_ptr().add(off), res);
    }

    for idx in chunks * 8..len {
        *dst.get_unchecked_mut(idx) = (*src.get_unchecked(idx) as f32) * inv_qm;
    }
}

/// Dequantizes an i8 embedding row and adds sinusoidal positional embeddings in a single fused AVX2 pass.
///
/// # Safety
/// Requires AVX2 CPU features. `src.len()`, `pos_slice.len()`, and `dst.len()` must match.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn embed_add_pos_avx2(src: &[i8], pos_slice: &[f32], dst: &mut [f32], qm_mult: f32) {
    use std::arch::x86_64::*;
    let len = src.len();
    let chunks = len / 8;
    let qm_mult_v = _mm256_set1_ps(qm_mult);

    for c in 0..chunks {
        let off = c * 8;
        let raw = (src.as_ptr().add(off) as *const i64).read_unaligned();
        let v8 = _mm_cvtsi64_si128(raw);
        let v32 = _mm256_cvtepi8_epi32(v8);
        let vps = _mm256_cvtepi32_ps(v32);
        let mult = _mm256_mul_ps(vps, qm_mult_v);
        let pos_v = _mm256_loadu_ps(pos_slice.as_ptr().add(off));
        let sum = _mm256_add_ps(mult, pos_v);
        _mm256_storeu_ps(dst.as_mut_ptr().add(off), sum);
    }

    for idx in chunks * 8..len {
        *dst.get_unchecked_mut(idx) =
            (*src.get_unchecked(idx) as f32) * qm_mult + *pos_slice.get_unchecked(idx);
    }
}

/// Embedding lookup into `out` directly from pre-resolved [`SrcEmbedding`].
fn embed_src_into(src_emb: &SrcEmbedding, ids: &[u32], unk_id: u32, out: &mut Tensor2) {
    match src_emb {
        SrcEmbedding::F32(weight) => {
            let max_vocab = weight.shape.first().copied().unwrap_or(0);
            let d_model = weight.shape.get(1).copied().unwrap_or(0);
            let total_len = ids.len() * d_model;
            out.set_shape_2d(ids.len(), d_model);
            if out.data.len() != total_len {
                out.data.resize(total_len, 0.0);
            }
            for (i, &id) in ids.iter().enumerate() {
                let dst_start = i * d_model;
                let eff_id = if (id as usize) < max_vocab {
                    Some(id as usize)
                } else if (unk_id as usize) < max_vocab {
                    tracing::warn!(
                        "Token id {} exceeds source embedding size {}, falling back to unk {}",
                        id, max_vocab, unk_id
                    );
                    Some(unk_id as usize)
                } else {
                    None
                };
                if let Some(valid_id) = eff_id {
                    let start = valid_id * d_model;
                    out.data[dst_start..dst_start + d_model]
                        .copy_from_slice(&weight.data[start..start + d_model]);
                } else {
                    out.data[dst_start..dst_start + d_model].fill(0.0);
                }
            }
        }
        SrcEmbedding::Q8(q) => {
            let max_vocab = q.shape.first().copied().unwrap_or(0);
            let d_model = q.shape.get(1).copied().unwrap_or(0);
            let total_len = ids.len() * d_model;
            out.set_shape_2d(ids.len(), d_model);
            if out.data.len() != total_len {
                out.data.resize(total_len, 0.0);
            }
            let inv_qm = 1.0 / q.quant_mult;
            #[cfg(target_arch = "x86_64")]
            let use_avx2 = crate::utils::arch::has_avx2();
            for (i, &id) in ids.iter().enumerate() {
                let dst_start = i * d_model;
                let eff_id = if (id as usize) < max_vocab {
                    Some(id as usize)
                } else if (unk_id as usize) < max_vocab {
                    tracing::warn!(
                        "Token id {} exceeds source quantized embedding size {}, falling back to unk {}",
                        id, max_vocab, unk_id
                    );
                    Some(unk_id as usize)
                } else {
                    None
                };
                if let Some(valid_id) = eff_id {
                    let start = valid_id * d_model;
                    let src = &q.data[start..start + d_model];
                    let dst = &mut out.data[dst_start..dst_start + d_model];
                    #[cfg(target_arch = "x86_64")]
                    if use_avx2 {
                        unsafe {
                            embed_src_row_avx2(src, dst, inv_qm);
                        }
                        continue;
                    }
                    for (d, &v) in dst.iter_mut().zip(src.iter()) {
                        *d = (v as f32) * inv_qm;
                    }
                } else {
                    out.data[dst_start..dst_start + d_model].fill(0.0);
                }
            }
        }
    }
}

/// Resolve the source embedding table name: shared models store one `Wemb`,
/// split models separate `encoder_Wemb` / `decoder_Wemb`.
fn src_wemb_name(model: &Model) -> &'static str {
    if model.contains("encoder_Wemb") {
        "encoder_Wemb"
    } else {
        "Wemb"
    }
}

/// Build sinusoidal positional embeddings for positions `0..seq_len`,
/// replicating marian's `inits::sinusoidalPositionEmbeddings(start)`.
///
/// The first half of the dimensions carries `sin`, the second half `cos`,
/// both at the timescale frequencies:
///
/// ```text
/// numTimescales = d_model / 2
/// logTimescaleIncrement = log(10000) / (numTimescales - 1)
/// PE(p, i)         = sin(p · exp(-i · logTimescaleIncrement))   i < d/2
/// PE(p, i + d/2)   = cos(p · exp(-i · logTimescaleIncrement))
/// ```
fn sinusoidal_positions(seq_len: usize, d_model: usize) -> Tensor2 {
    let num_timescales = d_model / 2;
    let log_timescale_increment = (10000.0f32).ln() / (num_timescales as f32 - 1.0);
    let mut data = vec![0.0f32; seq_len * d_model];
    for p in 0..seq_len {
        let pf = p as f32;
        for i in 0..num_timescales {
            let v = pf * ((i as f32) * -log_timescale_increment).exp();
            data[p * d_model + i] = v.sin();
            data[p * d_model + num_timescales + i] = v.cos();
        }
    }
    Tensor2::from_vec([seq_len, d_model], data)
}

// ---------------------------------------------------------------------------
// Linear layer and attention primitives
// ---------------------------------------------------------------------------

/// Buffer-reusing variant of [`linear`] writing into `out`.
fn linear_into(x: &Tensor2, w: &LinW, b: &Tensor2, out: &mut Tensor2) {
    match w {
        LinW::F32(wf) => x.matmul_biased_into(wf, Some(b), out),
        LinW::Q8(wq, qa) => x.quantized_matmul_biased_into(wq, *qa, Some(b), out),
    }
}

/// Buffer-reusing variant of [`linear`] with ReLU fused into the matmul's
/// store step: `out = max(0, x @ W + b)`. Avoids the separate elementwise
/// pass over the hidden `[seq, d_ff]` buffer.
fn linear_relu_into(x: &Tensor2, w: &LinW, b: &Tensor2, out: &mut Tensor2) {
    match w {
        LinW::F32(wf) => x.matmul_biased_relu_into(wf, Some(b), out),
        LinW::Q8(wq, qa) => x.quantized_matmul_biased_relu_into(wq, *qa, Some(b), out),
    }
}
#[inline]
fn weighted_accumulate_dk(dst: &mut [f32], src: &[f32], w: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::utils::arch::has_avx2() {
            unsafe {
                weighted_accumulate_dk_avx2(dst, src, w);
                return;
            }
        }
    }
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d += w * s;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn weighted_accumulate_dk_avx2(dst: &mut [f32], src: &[f32], w: f32) {
    use std::arch::x86_64::*;
    let len = dst.len();
    let blocks = len / 8;
    let wv = _mm256_set1_ps(w);
    let d_ptr = dst.as_mut_ptr();
    let s_ptr = src.as_ptr();
    for b in 0..blocks {
        let off = b * 8;
        let d = _mm256_loadu_ps(d_ptr.add(off));
        let s = _mm256_loadu_ps(s_ptr.add(off));
        let res = _mm256_fmadd_ps(wv, s, d);
        _mm256_storeu_ps(d_ptr.add(off), res);
    }
    for i in (blocks * 8)..len {
        *dst.get_unchecked_mut(i) += w * *src.get_unchecked(i);
    }
}
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn accumulate_head_v_dk48_avx2(
    dst: &mut [f32],
    v_data: &[f32],
    h_off: usize,
    scores_row: &[f32],
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let d_ptr = dst.as_mut_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    let mut a5 = _mm256_setzero_ps();

    let v_ptr = v_data.as_ptr();
    for j in 0..kv_len {
        let w = *scores_row.get_unchecked(j);
        let wv = _mm256_set1_ps(w);
        let v_off = h_off + j * 48;
        let vj_ptr = v_ptr.add(v_off);
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr), a0);
        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(8)), a1);
        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(16)), a2);
        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(24)), a3);
        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(32)), a4);
        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(40)), a5);
    }

    _mm256_storeu_ps(d_ptr, a0);
    _mm256_storeu_ps(d_ptr.add(8), a1);
    _mm256_storeu_ps(d_ptr.add(16), a2);
    _mm256_storeu_ps(d_ptr.add(24), a3);
    _mm256_storeu_ps(d_ptr.add(32), a4);
    _mm256_storeu_ps(d_ptr.add(40), a5);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn accumulate_strided_v_dk48_avx2(
    dst: &mut [f32],
    v_data: &[f32],
    d_model: usize,
    col: usize,
    scores_row: &[f32],
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let d_ptr = dst.as_mut_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    let mut a5 = _mm256_setzero_ps();

    let v_ptr = v_data.as_ptr().add(col);
    for j in 0..kv_len {
        let w = *scores_row.get_unchecked(j);
        let wv = _mm256_set1_ps(w);
        let row_ptr = v_ptr.add(j * d_model);
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr), a0);
        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(8)), a1);
        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(16)), a2);
        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(24)), a3);
        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(32)), a4);
        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(40)), a5);
    }

    _mm256_storeu_ps(d_ptr, a0);
    _mm256_storeu_ps(d_ptr.add(8), a1);
    _mm256_storeu_ps(d_ptr.add(16), a2);
    _mm256_storeu_ps(d_ptr.add(24), a3);
    _mm256_storeu_ps(d_ptr.add(32), a4);
    _mm256_storeu_ps(d_ptr.add(40), a5);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn accumulate_strided_v_dk64_avx2(
    dst: &mut [f32],
    v_data: &[f32],
    d_model: usize,
    col: usize,
    scores_row: &[f32],
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let d_ptr = dst.as_mut_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    let mut a5 = _mm256_setzero_ps();
    let mut a6 = _mm256_setzero_ps();
    let mut a7 = _mm256_setzero_ps();

    let v_ptr = v_data.as_ptr().add(col);
    for j in 0..kv_len {
        let w = *scores_row.get_unchecked(j);
        let wv = _mm256_set1_ps(w);
        let row_ptr = v_ptr.add(j * d_model);
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr), a0);
        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(8)), a1);
        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(16)), a2);
        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(24)), a3);
        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(32)), a4);
        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(40)), a5);
        a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(48)), a6);
        a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row_ptr.add(56)), a7);
    }

    _mm256_storeu_ps(d_ptr, a0);
    _mm256_storeu_ps(d_ptr.add(8), a1);
    _mm256_storeu_ps(d_ptr.add(16), a2);
    _mm256_storeu_ps(d_ptr.add(24), a3);
    _mm256_storeu_ps(d_ptr.add(32), a4);
    _mm256_storeu_ps(d_ptr.add(40), a5);
    _mm256_storeu_ps(d_ptr.add(48), a6);
    _mm256_storeu_ps(d_ptr.add(56), a7);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn accumulate_head_v_dk64_avx2(
    dst: &mut [f32],
    v_data: &[f32],
    h_off: usize,
    scores_row: &[f32],
    kv_len: usize,
) {
    use std::arch::x86_64::*;
    let d_ptr = dst.as_mut_ptr();
    let mut a0 = _mm256_setzero_ps();
    let mut a1 = _mm256_setzero_ps();
    let mut a2 = _mm256_setzero_ps();
    let mut a3 = _mm256_setzero_ps();
    let mut a4 = _mm256_setzero_ps();
    let mut a5 = _mm256_setzero_ps();
    let mut a6 = _mm256_setzero_ps();
    let mut a7 = _mm256_setzero_ps();

    let v_ptr = v_data.as_ptr();
    for j in 0..kv_len {
        let w = *scores_row.get_unchecked(j);
        let wv = _mm256_set1_ps(w);
        let v_off = h_off + j * 64;
        let vj_ptr = v_ptr.add(v_off);
        a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr), a0);
        a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(8)), a1);
        a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(16)), a2);
        a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(24)), a3);
        a4 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(32)), a4);
        a5 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(40)), a5);
        a6 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(48)), a6);
        a7 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(vj_ptr.add(56)), a7);
    }

    _mm256_storeu_ps(d_ptr, a0);
    _mm256_storeu_ps(d_ptr.add(8), a1);
    _mm256_storeu_ps(d_ptr.add(16), a2);
    _mm256_storeu_ps(d_ptr.add(24), a3);
    _mm256_storeu_ps(d_ptr.add(32), a4);
    _mm256_storeu_ps(d_ptr.add(40), a5);
    _mm256_storeu_ps(d_ptr.add(48), a6);
    _mm256_storeu_ps(d_ptr.add(56), a7);
}

/// Multi-head attention with **pre-projected** Q, K, V.
///
/// - `q`: `[q_len, d_model]` (already projected)
/// - `k, v`: `[kv_len, d_model]` (already projected)
/// - `wo, bo`: output projection `[d_model, d_model]` / `[1, d_model]`
/// - `heads`: number of attention heads
///
/// The attention context goes to `out_ctx`, the per-head score rows to `out_scores`
/// (both resized and overwritten). The weighted-V sum accumulates into `pre_ctx`,
/// which the output projection then reads in place — both buffers are resized and
/// overwritten, so the whole call allocates nothing beyond the first use.
/// Numerically identical to the allocating wrapper.
#[allow(clippy::too_many_arguments)]
fn multi_head_attention_qkv_into(
    q: &Tensor2,
    k: &Tensor2,
    v: &Tensor2,
    wo: &LinW,
    bo: &Tensor2,
    heads: usize,
    mask: Option<&Tensor2>,
    causal: bool,
    out_ctx: &mut Tensor2,
    pre_ctx: &mut Tensor2,
    out_scores: &mut Tensor2,
    kt_panel: &mut Tensor2,
) {
    let q_len = q.shape[0];
    let kv_len = k.shape[0];
    let d_model = q.shape[1];
    let dk = d_model / heads;
    let scale = 1.0 / (dk as f32).sqrt();

    let mask_cols = mask.map(|m| m.shape[1]).unwrap_or(0);

    if pre_ctx.data.len() == q_len * d_model {
        pre_ctx.data.fill(0.0);
    } else {
        pre_ctx.data.clear();
        pre_ctx.data.resize(q_len * d_model, 0.0);
    }
    out_scores.data.resize(q_len * kv_len, 0.0);
    let scores = out_scores.data.as_mut_slice();
    if q_len > 1 {
        kt_panel.data.resize(dk * kv_len, 0.0);
    }

    for h in 0..heads {
        let col = h * dk;

        // scores[i][j] = (Σ_t q[i][col+t] · k[j][col+t]) · scale
        //
        // Transposing the head block once per head into a t-major panel lets
        // the dispatched kernel vectorise across j while every score keeps
        // the exact scalar sequential t-chain (bit-identical output). The
        // transpose amortises over all q rows, so multi-row encoder calls
        // take the dispatched path; single-row callers keep the strided
        // scalar loop.
        if q_len > 1 {
            transpose_head_block_tmajor(&k.data, kv_len, d_model, col, dk, &mut kt_panel.data);
            score_rows_dispatch(
                &q.data,
                Some(&k.data),
                Some(&kt_panel.data),
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                scores,
            );
        } else {
            for i in 0..q_len {
                let q_off = i * d_model + col;
                for j in 0..kv_len {
                    let k_off = j * d_model + col;
                    let mut acc = 0.0f32;
                    for t in 0..dk {
                        acc += q.data[q_off + t] * k.data[k_off + t];
                    }
                    scores[i * kv_len + j] = acc * scale;
                }
            }
        }

        // Additive masks, applied directly to the scores buffer.
        if causal {
            let limit = kv_len.min(q_len);
            for i in 0..q_len {
                for j in (i + 1)..limit {
                    scores[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(m) = mask {
            for i in 0..q_len.min(m.shape[0]) {
                for j in 0..kv_len.min(mask_cols) {
                    scores[i * kv_len + j] += m.data[i * mask_cols + j];
                }
            }
        }

        softmax_scores_inplace(scores, q_len, kv_len);

        // pre_ctx[i][col+t] = Σ_j scores[i][j] · v[j][col+t]
        // (j-outer, t-inner so the contiguous v/out runs vectorise).
        let use_avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                crate::utils::arch::has_avx2()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };

        for i in 0..q_len {
            let o_off = i * d_model + col;
            #[cfg(target_arch = "x86_64")]
            if use_avx2 && dk == 48 {
                unsafe {
                    accumulate_strided_v_dk48_avx2(
                        &mut pre_ctx.data[o_off..o_off + 48],
                        &v.data,
                        d_model,
                        col,
                        &scores[i * kv_len..(i + 1) * kv_len],
                        kv_len,
                    );
                }
                continue;
            } else if use_avx2 && dk == 64 {
                unsafe {
                    accumulate_strided_v_dk64_avx2(
                        &mut pre_ctx.data[o_off..o_off + 64],
                        &v.data,
                        d_model,
                        col,
                        &scores[i * kv_len..(i + 1) * kv_len],
                        kv_len,
                    );
                }
                continue;
            }

            for j in 0..kv_len {
                let w = scores[i * kv_len + j];
                let v_off = j * d_model + col;
                weighted_accumulate_dk(
                    &mut pre_ctx.data[o_off..o_off + dk],
                    &v.data[v_off..v_off + dk],
                    w,
                );
            }
        }
    }
    pre_ctx.set_shape_2d(q_len, d_model);

    // Output projection (marian Wo/bo) reads the weighted-V sum from
    // `pre_ctx` and lands in `out_ctx` — two distinct pre-sized buffers, so
    // no temporary allocation is needed.
    linear_into(pre_ctx, wo, bo, out_ctx);
}

/// Cross-attention multi-head peak sharpness filtering.
///
/// Filters diffuse/noisy heads during cross-attention alignment accumulation by
/// weighting sharper heads above an expected uniform threshold with quadratic escalation,
/// while heavily damping diffuse heads with baseline noise weight 0.001.
///
/// Smoothly transitions above the threshold without inversion artifacts:
/// `weight = 0.001 + (head_max - threshold).max(0.0)^2`.
///
/// Prevents diffuse noise from distorting token alignment in rich-text/markup
/// translation without heap allocation or branch-heavy overhead.
#[inline]
pub(crate) fn filter_and_accumulate_cross_attention_head(
    accum: &mut [f32],
    scores: &[f32],
    kv_len: usize,
    h: usize,
) {
    let len = kv_len.min(accum.len()).min(scores.len());
    let mut head_max = 0.0f32;
    for &s in &scores[..len] {
        if s > head_max {
            head_max = s;
        }
    }
    let threshold = if kv_len > 1 {
        1.2 / kv_len as f32
    } else {
        0.0
    };
    let diff = (head_max - threshold).max(0.0);
    let weight = 0.001f32 + diff * diff;

    if h == 0 {
        for j in 0..len {
            accum[j] = scores[j] * weight;
        }
        if accum.len() > len {
            accum[len..].fill(0.0);
        }
    } else {
        for j in 0..len {
            accum[j] += scores[j] * weight;
        }
    }
}

/// Head-major variant of [`multi_head_attention_qkv_into`] for the decoder
/// cross-attention, taking pre-arranged K/V ([`HeadMajorK`] / [`HeadMajorV`], built once in
/// [`DecoderLayerWeights::new`]; heads/dk come from the structs).
///
/// Numerically identical to the row-major version: the score dot product
/// still sums `t = 0..dk` in order (the rearrangement keeps each source
/// position's `dk` elements adjacent), and the weighted V sum still
/// accumulates `+= w * v` j-outer / t-inner, so every float add happens in
/// the same order on the same values. Only the memory layout of the K/V
/// reads changes; scores keep their `[q_len, kv_len]` layout and `mask`/
/// `causal` are kept for signature symmetry with the original (the decoder
/// passes `None`/`false` today).
///
/// The weighted-V sum lands in `pre_ctx`, which the output projection then
/// reads in place; the post-Wo context lands in `out_ctx`. All three
/// buffers are distinct and pre-sized, so a decode step allocates nothing.
#[allow(clippy::too_many_arguments)]
fn multi_head_attention_qkv_headmajor_into(
    q: &Tensor2,
    k: &HeadMajorK,
    v: &HeadMajorV,
    wo: &LinW,
    bo: &Tensor2,
    mask: Option<&Tensor2>,
    causal: bool,
    out_ctx: &mut Tensor2,
    pre_ctx: &mut Tensor2,
    out_scores: &mut Tensor2,
    mut attn_accum: Option<&mut [f32]>,
) {
    let q_len = q.shape[0];
    let kv_len = k.src_len;
    let d_model = k.d_model;
    let heads = k.heads;
    let dk = k.dk;
    let scale = 1.0 / (dk as f32).sqrt();

    let mask_cols = mask.map(|m| m.shape[1]).unwrap_or(0);

    pre_ctx.data.resize(q_len * d_model, 0.0);
    out_scores.data.resize(q_len * kv_len, 0.0);
    let scores = out_scores.data.as_mut_slice();
    let has_avx2 = {
        #[cfg(target_arch = "x86_64")]
        {
            crate::utils::arch::has_avx2()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };
    let use_avx2_dk48 = has_avx2 && dk == 48;
    let use_avx2_dk64 = has_avx2 && dk == 64;

    for h in 0..heads {
        let col = h * dk;
        // Start of head h's `[kv_len, dk]` block in the rearranged K/V.
        let h_off = h * kv_len * dk;

        // scores[i][j] = (Σ_t q[i][col+t] · k[j][col+t]) · scale
        let kt_slice = &k.kt[h * dk * kv_len..(h + 1) * dk * kv_len];
        #[cfg(target_arch = "x86_64")]
        if has_avx2 {
            // SAFETY: guarded by `has_avx2` check; buffer slices are sized for attention heads.
            unsafe {
                crate::engine::tensor::score_rows_avx2_tmajor(
                    &q.data,
                    kt_slice,
                    q_len,
                    kv_len,
                    d_model,
                    col,
                    dk,
                    scale,
                    scores,
                );
            }
        } else {
            score_rows_dispatch(
                &q.data,
                None,
                Some(kt_slice),
                q_len,
                kv_len,
                d_model,
                col,
                dk,
                scale,
                scores,
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        score_rows_dispatch(
            &q.data,
            None,
            Some(kt_slice),
            q_len,
            kv_len,
            d_model,
            col,
            dk,
            scale,
            scores,
        );

        // Additive masks, applied directly to the scores buffer.
        if causal {
            let limit = kv_len.min(q_len);
            for i in 0..q_len {
                for j in (i + 1)..limit {
                    scores[i * kv_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(m) = mask {
            for i in 0..q_len.min(m.shape[0]) {
                for j in 0..kv_len.min(mask_cols) {
                    scores[i * kv_len + j] += m.data[i * mask_cols + j];
                }
            }
        }

        softmax_scores_inplace(scores, q_len, kv_len);

        if let Some(accum) = &mut attn_accum {
            filter_and_accumulate_cross_attention_head(accum, scores, kv_len, h);
        }
        // pre_ctx[i][col+t] = Σ_j scores[i][j] · v[j][col+t]
        // (j-outer, t-inner so the contiguous v/out runs vectorise).

        for i in 0..q_len {
            let o_off = i * d_model + col;
            #[cfg(target_arch = "x86_64")]
            if use_avx2_dk48 {
                unsafe {
                    accumulate_head_v_dk48_avx2(
                        &mut pre_ctx.data[o_off..o_off + 48],
                        &v.data,
                        h_off,
                        &scores[i * kv_len..(i + 1) * kv_len],
                        kv_len,
                    );
                }
                continue;
            } else if use_avx2_dk64 {
                unsafe {
                    accumulate_head_v_dk64_avx2(
                        &mut pre_ctx.data[o_off..o_off + 64],
                        &v.data,
                        h_off,
                        &scores[i * kv_len..(i + 1) * kv_len],
                        kv_len,
                    );
                }
                continue;
            }
            let slice = &mut pre_ctx.data[o_off..o_off + dk];
            slice.fill(0.0);
            for j in 0..kv_len {
                let w = scores[i * kv_len + j];
                let v_off = h_off + j * dk;
                weighted_accumulate_dk(slice, &v.data[v_off..v_off + dk], w);
            }
        }
    }
    pre_ctx.set_shape_2d(q_len, d_model);

    // Output projection (marian Wo/bo) reads the weighted-V sum from
    // `pre_ctx` and lands in `out_ctx` — two distinct pre-sized buffers, so
    // no temporary allocation is needed.
    linear_into(pre_ctx, wo, bo, out_ctx);
}
/// Buffer-reusing variant of [`multi_head_attention`]: projects Q/K/V into
/// the encoder scratch, accumulates the weighted-V sum in `scratch.res`
/// (dead until the residual add), and lands the post-Wo context in
/// `scratch.ctx`.
fn multi_head_attention_into(
    x: &Tensor2,
    wq: &LinW,
    bq: &Tensor2,
    wk: &LinW,
    bk: &Tensor2,
    wv: &LinW,
    bv: &Tensor2,
    wo: &LinW,
    bo: &Tensor2,
    heads: usize,
    mask: Option<&Tensor2>,
    causal: bool,
    scratch: &mut EncoderScratch,
) {
    if let (LinW::Q8(q_w, qa_q), LinW::Q8(k_w, qa_k), LinW::Q8(v_w, qa_v)) = (wq, wk, wv) {
        if qa_q == qa_k && qa_k == qa_v {
            x.quantized_matmul_triple_into(
                q_w,
                Some(bq),
                &mut scratch.q,
                k_w,
                Some(bk),
                &mut scratch.k,
                v_w,
                Some(bv),
                &mut scratch.v,
                *qa_q,
            );
        } else {
            linear_into(x, wq, bq, &mut scratch.q);
            linear_into(x, wk, bk, &mut scratch.k);
            linear_into(x, wv, bv, &mut scratch.v);
        }
    } else {
        linear_into(x, wq, bq, &mut scratch.q);
        linear_into(x, wk, bk, &mut scratch.k);
        linear_into(x, wv, bv, &mut scratch.v);
    }
    multi_head_attention_qkv_into(
        &scratch.q,
        &scratch.k,
        &scratch.v,
        wo,
        bo,
        heads,
        mask,
        causal,
        &mut scratch.ctx,
        &mut scratch.res,
        &mut scratch.scores,
        &mut scratch.kt_panel,
    );
}
// ---------------------------------------------------------------------------
// Encoder helpers
// ---------------------------------------------------------------------------

/// Read integer config or fallback to `default`.
fn config_int(model: &Model, key: &str, default: usize) -> usize {
    model
        .get_config_value(key)
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(default)
}

/// FFN sub-layer, writing the layer-normed result into `out`. `h1` is the
/// scratch buffer for the hidden projections.
/// The hidden projection fuses ReLU into the matmul store step (the
/// activation all bundled models use; see [`Activation::Relu`]), so no
/// separate elementwise pass over the `[seq, d_ff]` buffer is needed.
#[allow(clippy::too_many_arguments)]
fn ffn_layer_into(
    x: &Tensor2,
    w1: &LinW,
    b1: &Tensor2,
    w2: &LinW,
    b2: &Tensor2,
    ln_scale: &Tensor2,
    ln_bias: &Tensor2,
    eps: f32,
    h1: &mut Tensor2,
    out: &mut Tensor2,
) {
    linear_relu_into(x, w1, b1, h1); // [seq, d_ff], relu fused in the matmul
    linear_into(h1, w2, b2, out); // [seq, d_model]
    out.fused_add_layer_norm_in_place(x, ln_scale, ln_bias, eps);
}

/// One encoder layer: self-attention (with the "dan" postprocess) + FFN.
///
/// Buffer-reusing variant writing the final layer-normed result into `out`
/// (which must not alias the input `x`). `scratch` holds the Q/K/V
/// projections, the attention context/score rows, and the residual-LN scratch
/// reused across both sub-layers. The FFN activation is fused ReLU, matching
/// [`ffn_layer_into`] (all bundled models use ReLU).
#[allow(clippy::too_many_arguments)]
fn encoder_layer_into(
    x: &Tensor2,
    wq: &LinW,
    bq: &Tensor2,
    wk: &LinW,
    bk: &Tensor2,
    wv: &LinW,
    bv: &Tensor2,
    wo: &LinW,
    bo: &Tensor2,
    ln_attn_scale: &Tensor2,
    ln_attn_bias: &Tensor2,
    w1: &LinW,
    b1: &Tensor2,
    w2: &LinW,
    b2: &Tensor2,
    ln_ffn_scale: &Tensor2,
    ln_ffn_bias: &Tensor2,
    heads: usize,
    eps: f32,
    scratch: &mut EncoderScratch,
    out: &mut Tensor2,
) {
    // LayerAttention(self): preProcess("") no-op; full self-attention (the
    // batch is not padded, so no mask is needed); postProcess("dan").
    profiling::scope("enc/self-attn", || {
        multi_head_attention_into(
            x, wq, bq, wk, bk, wv, bv, wo, bo, heads, None, false, scratch,
        );
    });
    scratch
        .ctx
        .fused_add_layer_norm_in_place(x, ln_attn_scale, ln_attn_bias, eps);

    // LayerFFN: `ctx` (the attention postprocess result) is the FFN input;
    // `out` receives the layer output, so inputs never alias outputs.
    profiling::scope("enc/ffn", || {
        ffn_layer_into(
            &scratch.ctx,
            w1,
            b1,
            w2,
            b2,
            ln_ffn_scale,
            ln_ffn_bias,
            eps,
            &mut scratch.h1,
            out,
        );
    });
}

/// Detect whether the model uses 0-based or 1-based layer indexing.
///
/// Marian convention is 1-based (`_l1_`, `_l2_`), used by the OpusMT models;
/// some variants use 0-based (`_l0_`, `_l1_`).  Returns the offset.
fn layer_indexing(model: &Model) -> usize {
    if model.contains("encoder_l1_self_Wq") {
        1
    } else if model.contains("encoder_l0_self_Wq") {
        0
    } else {
        // Default to 1-based (Marian standard)
        1
    }
}

/// Run the full encoder stack.
///
/// Appends `[EOS]` to `source_ids` (no BOS — marian uses `--bos_id=-1`),
/// embeds with `sqrt(d_model)` scaling, adds sinusoidal positions, then
/// applies `enc-depth` `LayerAttention + LayerFFN` pairs.
/// Returns `[src_len, d_model]` encoder hidden states.
fn encoder_forward_into(
    weights: &LayerWeightsView,
    spm_src: &SentencePiece,
    source_ids: &[u32],
    config: &ForwardConfig,
    enc_state: &mut Tensor2,
) {
    let eos = spm_src.eos_id();
    let mut enc_bufs = ENCODER_BUFFERS
        .with(|cell| cell.borrow_mut().take())
        .unwrap_or_else(|| EncoderBuffers {
            scratch: EncoderScratch::empty(),
            x: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            out: Tensor2 {
                shape: [0, 0],
                data: Vec::new(),
            },
            ids: Vec::new(),
        });

    let mut ids = std::mem::take(&mut enc_bufs.ids);
    ids.clear();
    ids.reserve(source_ids.len() + 1);
    ids.extend_from_slice(source_ids);
    ids.push(eos);
    let seq_len = ids.len();
    let d_model = config.d_model;

    let mut x = std::mem::take(&mut enc_bufs.x);
    let scale = (d_model as f32).sqrt();
    let need = seq_len * d_model;

    let fused_done = {
        #[cfg(target_arch = "x86_64")]
        if crate::utils::arch::has_avx2() && config.position_table.len() >= need {
            if let SrcEmbedding::Q8(q8) = &weights.src_wemb {
                let max_vocab = q8.shape.first().copied().unwrap_or(0);
                x.set_shape_2d(seq_len, d_model);
                if x.data.len() != need {
                    x.data.resize(need, 0.0);
                }
                let qm_mult = scale / q8.quant_mult;
                let unk_id = spm_src.unk_id();
                let unk_valid = (unk_id as usize) < max_vocab;
                for (t, &id) in ids.iter().enumerate() {
                    let pos_slice = &config.position_table[t * d_model..(t + 1) * d_model];
                    let dst = &mut x.data[t * d_model..(t + 1) * d_model];
                    let eff_id = if (id as usize) < max_vocab {
                        Some(id as usize)
                    } else if unk_valid {
                        tracing::warn!(
                            "Source token id {} exceeds quantized embedding size {}, falling back to unk {}",
                            id, max_vocab, unk_id
                        );
                        Some(unk_id as usize)
                    } else {
                        tracing::warn!(
                            "Source token id {} and unk_id {} exceed quantized embedding size {}",
                            id, unk_id, max_vocab
                        );
                        None
                    };
                    if let Some(valid_id) = eff_id {
                        let src_start = valid_id * d_model;
                        let src = &q8.data[src_start..src_start + d_model];
                        unsafe {
                            embed_add_pos_avx2(src, pos_slice, dst, qm_mult);
                        }
                    } else {
                        dst.copy_from_slice(pos_slice);
                    }
                }
                true
            } else {
                false
            }
        } else {
            false
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    };

    if !fused_done {
        embed_src_into(&weights.src_wemb, &ids, spm_src.unk_id(), &mut x);
        if config.position_table.len() >= need {
            let positions = &config.position_table[..need];
            for (v, &p) in x.data[..need].iter_mut().zip(positions) {
                *v = *v * scale + p;
            }
        } else {
            let positions = sinusoidal_positions(seq_len, d_model);
            for (v, &p) in x.data[..need].iter_mut().zip(&positions.data) {
                *v = *v * scale + p;
            }
        }
    }

    let mut scratch = enc_bufs.scratch;
    let mut out = std::mem::take(&mut enc_bufs.out);

    for layer in &weights.encoder_layers {
        encoder_layer_into(
            &x,
            &layer.wq,
            layer.bq,
            &layer.wk,
            layer.bk,
            &layer.wv,
            layer.bv,
            &layer.wo,
            layer.bo,
            layer.ln_attn_scale,
            layer.ln_attn_bias,
            &layer.w1,
            layer.b1,
            &layer.w2,
            layer.b2,
            layer.ln_ffn_scale,
            layer.ln_ffn_bias,
            config.heads,
            LAYER_NORM_EPS,
            &mut scratch,
            &mut out,
        );
        std::mem::swap(&mut x, &mut out);
    }

    std::mem::swap(&mut x, enc_state);

    ENCODER_BUFFERS.with(|cell| {
        *cell.borrow_mut() = Some(EncoderBuffers {
            scratch,
            x,
            out,
            ids,
        });
    });
}
// ---------------------------------------------------------------------------
// Decoder helpers
// ---------------------------------------------------------------------------

/// SSRU cell (marian `rnn::SSRU` + `highway`), the decoder's auto-regressive
/// layer when `transformer-decoder-autoreg: rnn`:
///
/// ```text
/// x    = input @ rnn_W                      (no bias, layer-normalization: false)
/// f    = input @ rnn_Wf + rnn_bf            (forget gate)
/// cell = sigmoid(f) * cell_prev + (1 - sigmoid(f)) * x
/// out  = relu(cell)
/// ```
///
/// Writes the relu'd output into `s_x`; the new cell state is published
/// through `cell` (swapped with `s_cell`, so the allocation is reused).
/// The initial cell state is all zeros (marian `startState`).
fn decoder_ssru(
    input: &Tensor2,
    cell: &mut Tensor2,
    rnn_w: &LinW,
    rnn_wf: &LinW,
    rnn_bf: &Tensor2,
    s_x: &mut Tensor2,
    s_f: &mut Tensor2,
    s_cell: &mut Tensor2,
) {
    // x = input @ rnn_W (no bias), f = input @ rnn_Wf + rnn_bf.
    match (rnn_w, rnn_wf) {
        (LinW::Q8(wq, qa1), LinW::Q8(wqf, qa2)) if qa1 == qa2 => {
            input.quantized_matmul_pair_into(wq, None, s_x, wqf, Some(rnn_bf), s_f, *qa1);
        }
        _ => {
            match rnn_w {
                LinW::F32(w) => input.matmul_biased_into(w, None, s_x),
                LinW::Q8(wq, qa) => input.quantized_matmul_biased_into(wq, *qa, None, s_x),
            }
            match rnn_wf {
                LinW::F32(w) => input.matmul_biased_into(w, Some(rnn_bf), s_f),
                LinW::Q8(wq, qa) => input.quantized_matmul_biased_into(wq, *qa, Some(rnn_bf), s_f),
            }
        }
    }
    // Single pass over the gate: cell' = x + sig·(cell_prev - x).
    // The relu'd output is written in the same pass (out = max(0, cell')),
    // so the [1, d_model] buffers are traversed once instead of twice.
    s_cell.data.resize(input.data.len(), 0.0);
    let n = input.data.len();
    let cell_slice = &cell.data[..n];
    let s_f_slice = &s_f.data[..n];
    let s_x_slice = &mut s_x.data[..n];
    let s_cell_slice = &mut s_cell.data[..n];
    ssru_gate_into(cell_slice, s_f_slice, s_x_slice, s_cell_slice);
    if s_cell.shape != input.shape {
        s_cell.shape = input.shape;
    }

    // Publish the new cell state (swap keeps the scratch allocation warm for
    // the next step). The STORED state stays un-relu'd, exactly like the
    // original two-tensor version (`out = relu(cell)`; `cell` keeps the raw
    // gate result).
    std::mem::swap(cell, s_cell);
}

#[inline]
fn ssru_gate_into(cell: &[f32], s_f: &[f32], s_x: &mut [f32], s_cell: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if crate::utils::arch::has_avx2() {
            unsafe {
                ssru_gate_avx2(cell, s_f, s_x, s_cell);
            }
            return;
        }
    }
    ssru_gate_scalar(cell, s_f, s_x, s_cell);
}

fn ssru_gate_scalar(cell: &[f32], s_f: &[f32], s_x: &mut [f32], s_cell: &mut [f32]) {
    for i in 0..cell.len() {
        let f_v = s_f[i];
        let sig = 1.0 / (1.0 + (-f_v).exp());
        let c = sig * cell[i] + (1.0 - sig) * s_x[i];
        s_cell[i] = c;
        s_x[i] = if c > 0.0 { c } else { 0.0 };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn ssru_gate_avx2(cell: &[f32], s_f: &[f32], s_x: &mut [f32], s_cell: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = cell.len();
    let chunks = len / 8;
    let tail = len % 8;

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let c1 = _mm256_set1_ps(0.693359375);
    let c2 = _mm256_set1_ps(-2.1219444005469058e-4);
    let min_x = _mm256_set1_ps(-87.0);
    let max_x = _mm256_set1_ps(88.0);
    let one = _mm256_set1_ps(1.0);
    let zero = _mm256_setzero_ps();
    let bias_int = _mm256_set1_epi32(127);

    let p5 = _mm256_set1_ps(0.008333333333333333);
    let p4 = _mm256_set1_ps(0.041666666666666664);
    let p3 = _mm256_set1_ps(0.16666666666666666);
    let p2 = _mm256_set1_ps(0.5);

    let mut ptr_c = cell.as_ptr();
    let mut ptr_f = s_f.as_ptr();
    let mut ptr_x = s_x.as_mut_ptr();
    let mut ptr_sc = s_cell.as_mut_ptr();

    for _ in 0..chunks {
        let f_v = _mm256_loadu_ps(ptr_f);
        let c_v = _mm256_loadu_ps(ptr_c);
        let x_v = _mm256_loadu_ps(ptr_x);

        let neg_f = _mm256_sub_ps(zero, f_v);
        let clamped = _mm256_min_ps(_mm256_max_ps(neg_f, min_x), max_x);

        let z = _mm256_mul_ps(clamped, log2e);
        let k = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

        let r = _mm256_fnmadd_ps(k, c2, _mm256_fnmadd_ps(k, c1, clamped));

        let mut poly = _mm256_fmadd_ps(p5, r, p4);
        poly = _mm256_fmadd_ps(poly, r, p3);
        poly = _mm256_fmadd_ps(poly, r, p2);
        poly = _mm256_fmadd_ps(poly, r, one);
        poly = _mm256_fmadd_ps(poly, r, one);

        let k_i = _mm256_cvtps_epi32(k);
        let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(k_i, bias_int), 23);
        let pow2 = _mm256_castsi256_ps(exp_bits);

        let exp_val = _mm256_mul_ps(pow2, poly);
        let denom = _mm256_add_ps(one, exp_val);
        let sig = _mm256_div_ps(one, denom);

        let diff = _mm256_sub_ps(c_v, x_v);
        let c = _mm256_fmadd_ps(sig, diff, x_v);

        _mm256_storeu_ps(ptr_sc, c);
        let relu_c = _mm256_max_ps(c, zero);
        _mm256_storeu_ps(ptr_x, relu_c);

        ptr_c = ptr_c.add(8);
        ptr_f = ptr_f.add(8);
        ptr_x = ptr_x.add(8);
        ptr_sc = ptr_sc.add(8);
    }

    if tail > 0 {
        let off = chunks * 8;
        for i in off..len {
            let f_v = s_f[i];
            let sig = 1.0 / (1.0 + (-f_v).exp());
            let c = sig * cell[i] + (1.0 - sig) * s_x[i];
            s_cell[i] = c;
            s_x[i] = if c > 0.0 { c } else { 0.0 };
        }
    }
}

/// Run one decoder layer using pre-resolved weights and precomputed encoder
/// K/V (marian `DecoderLayerRNN` + `LayerAttention` + `LayerFFN`).  Every
/// sub-layer uses postprocess "dan".
///
/// - `x`: `[1, d_model]` — current step input
/// - `layer`: pre-resolved weights + precomputed encoder K/V for this layer
/// - `cell`: running SSRU cell state; updated in place so the next step sees
///   the new state
///
/// All intermediate tensors live in the caller-provided `Scratch` so a decode
/// step allocates nothing.
fn decoder_layer(
    x: &mut Tensor2,
    layer: &DecoderLayerStaticWeights,
    enc_k: &HeadMajorK,
    enc_v: &HeadMajorV,
    heads: usize,
    autoreg_type: DecoderAutoregType,
    cell: &mut Tensor2,
    s_cell_scratch: &mut Tensor2,
    kv_layer: &mut DecoderLayerKVCache,
    eps: f32,
    s: &mut Scratch,
    attn_accum: Option<&mut [f32]>,
) {
    match autoreg_type {
        DecoderAutoregType::Rnn => {
            let rnn_w = layer.rnn_w.as_ref().expect("missing rnn_w");
            let rnn_wf = layer.rnn_wf.as_ref().expect("missing rnn_wf");
            let rnn_bf = layer.rnn_bf.expect("missing rnn_bf");
            let rnn_ln_scale = layer.rnn_ln_scale.expect("missing rnn_ln_scale");
            let rnn_ln_bias = layer.rnn_ln_bias.expect("missing rnn_ln_bias");

            profiling::scope("dec/ssru", || {
                decoder_ssru(
                    x,
                    cell,
                    rnn_w,
                    rnn_wf,
                    rnn_bf,
                    &mut s.a,
                    &mut s.b,
                    s_cell_scratch,
                );
            });

            // dan postprocess: residual add then layer-norm, both into `x`.
            profiling::scope("dec/ssru+post ln", || {
                x.fused_add_layer_norm_in_place(&s.a, rnn_ln_scale, rnn_ln_bias, eps);
            });
        }
        DecoderAutoregType::SelfAttention => {
            let self_wq = layer.self_wq.as_ref().expect("missing self_wq");
            let self_bq = layer.self_bq.expect("missing self_bq");
            let self_wk = layer.self_wk.as_ref().expect("missing self_wk");
            let self_bk = layer.self_bk.expect("missing self_bk");
            let self_wv = layer.self_wv.as_ref().expect("missing self_wv");
            let self_bv = layer.self_bv.expect("missing self_bv");
            let self_wo = layer.self_wo.as_ref().expect("missing self_wo");
            let self_bo = layer.self_bo.expect("missing self_bo");
            let self_ln_scale = layer.self_ln_scale.expect("missing self_ln_scale");
            let self_ln_bias = layer.self_ln_bias.expect("missing self_ln_bias");

            profiling::scope("dec/self-attn", || {
                linear_into(x, self_wq, self_bq, &mut s.q);
                linear_into(x, self_wk, self_bk, &mut s.self_k);
                linear_into(x, self_wv, self_bv, &mut s.self_v);
                let d_model = x.shape[1];
                kv_layer.append(&s.self_k.data, &s.self_v.data, d_model);

                multi_head_attention_qkv_into(
                    &s.q,
                    &kv_layer.k,
                    &kv_layer.v,
                    self_wo,
                    self_bo,
                    heads,
                    None,
                    false,
                    &mut s.attn,
                    &mut s.ctx_pre,
                    &mut s.scores,
                    &mut s.t2,
                );
                x.fused_add_layer_norm_in_place(&s.attn, self_ln_scale, self_ln_bias, eps);
            });
        }
    }

    // ---- Cross-attention (encoder-decoder) ----
    // Q from the decoder state; K/V are precomputed in head-major layout
    // in the encoder K/V cache.
    profiling::scope("dec/cross", || {
        linear_into(x, &layer.ctx_wq, layer.ctx_bq, &mut s.q);
        multi_head_attention_qkv_headmajor_into(
            &s.q,
            enc_k,
            enc_v,
            &layer.ctx_wo,
            layer.ctx_bo,
            None,
            false,
            &mut s.attn,
            &mut s.ctx_pre,
            &mut s.scores,
            attn_accum,
        );
        x.fused_add_layer_norm_in_place(&s.attn, layer.ctx_ln_scale, layer.ctx_ln_bias, eps);
    });

    // ---- FFN ----
    profiling::scope("dec/ffn", || {
        ffn_layer_into(
            x,
            &layer.ffn_w1,
            layer.ffn_b1,
            &layer.ffn_w2,
            layer.ffn_b2,
            layer.ffn_ln_scale,
            layer.ffn_ln_bias,
            eps,
            &mut s.h1,
            &mut s.t1,
        );
        std::mem::swap(x, &mut s.t1);
    });
}

/// Run one decoder step: apply the `dec-depth` decoder layers to the step
/// input, maintaining the per-layer SSRU cell states or self-attention KV-caches across steps.
///
/// `step_input`: `[1, d_model]` — zero embedding for the first step or the
/// (scaled + position-encoded) embedding of the previously generated word.
/// All intermediates live in the caller-owned `Scratch`.
fn decoder_forward(
    step_input: &mut Tensor2,
    layers: &[DecoderLayerStaticWeights],
    enc_k: &[HeadMajorK],
    enc_v: &[HeadMajorV],
    heads: usize,
    autoreg_type: DecoderAutoregType,
    cell_states: &mut [Tensor2],
    cell_scratches: &mut [Tensor2],
    kv_cache: &mut DecoderKVCache,
    eps: f32,
    s: &mut Scratch,
    mut attn_accum: Option<&mut [f32]>,
) {
    let last_idx = layers.len().saturating_sub(1);
    for (i, (((layer, (k, v)), (cell, cell_scratch)), kv_layer)) in layers
        .iter()
        .zip(enc_k.iter().zip(enc_v.iter()))
        .zip(cell_states.iter_mut().zip(cell_scratches.iter_mut()))
        .zip(kv_cache.layers.iter_mut())
        .enumerate()
    {
        let layer_attn = if i == last_idx {
            attn_accum.as_deref_mut()
        } else {
            None
        };
        decoder_layer(
            step_input,
            layer,
            k,
            v,
            heads,
            autoreg_type,
            cell,
            cell_scratch,
            kv_layer,
            eps,
            s,
            layer_attn,
        );
    }
    // transformer-postprocess-top: "" -> no final layer-norm
}

// ---------------------------------------------------------------------------
// Output projection and shortlist
// ---------------------------------------------------------------------------

/// Compute logits for only the shortlist vocabulary entries.
///
/// With tied embeddings the output projection is `h @ Wembᵀ + b`, where
/// `Wemb` is `[vocab, d_model]`.  Pre-gather the shortlist rows into an
/// `[sh_len, d_model]` **int8** weight matrix — the same quantized bytes the
/// model stores on disk, so no precision is lost relative to any intgemm
/// path — plus the f32 bias slice. Every decoder step then runs a single
/// i8 gemv with fused dequantization/bias (the bergamot production layout),
/// scanning 4× less weight memory than the previous f32 projection.
///
/// Returns `(wq, sb)` with `wq.data[j * d_model + t] = Wemb_i8[shortlist[j]][t]`
/// and `sb[j]` the corresponding bias.
fn gather_shortlist_projection_into(
    wemb: &Quant8Tensor,
    bias: Option<&Tensor2>,
    shortlist: &[u32],
    out_data: &mut Vec<i8>,
    out_row_sums: &mut Vec<i32>,
    out_bias: &mut Vec<f32>,
) {
    let d_model = wemb.shape[1];
    let max_vocab = wemb.shape.first().copied().unwrap_or(0);
    let sh_len = shortlist.len();
    let total_elems = sh_len * d_model;

    out_data.clear();
    if out_data.capacity() < total_elems {
        out_data.reserve(total_elems - out_data.capacity());
    }

    out_bias.clear();
    if out_bias.capacity() < sh_len {
        out_bias.reserve(sh_len - out_bias.capacity());
    }

    out_row_sums.clear();
    if wemb.row_sums.is_some() && out_row_sums.capacity() < sh_len {
        out_row_sums.reserve(sh_len - out_row_sums.capacity());
    }

    let mut i = 0;
    while i < shortlist.len() {
        let start_v = shortlist[i] as usize;
        let mut run_len = 1;
        while i + run_len < shortlist.len() && shortlist[i + run_len] as usize == start_v + run_len
        {
            run_len += 1;
        }

        if start_v < max_vocab {
            let valid_len = run_len.min(max_vocab - start_v);
            let src_start = start_v * d_model;
            let src_end = (start_v + valid_len) * d_model;
            out_data.extend_from_slice(&wemb.data[src_start..src_end]);
            if valid_len < run_len {
                out_data.resize(out_data.len() + (run_len - valid_len) * d_model, 0);
            }
        } else {
            out_data.resize(out_data.len() + run_len * d_model, 0);
        }

        if let Some(b) = bias {
            if start_v + run_len <= b.data.len() {
                out_bias.extend_from_slice(&b.data[start_v..start_v + run_len]);
            } else {
                for v in start_v..start_v + run_len {
                    out_bias.push(b.data.get(v).copied().unwrap_or(0.0));
                }
            }
        }
        if let Some(src_rs) = &wemb.row_sums {
            if start_v + run_len <= src_rs.len() {
                out_row_sums.extend_from_slice(&src_rs[start_v..start_v + run_len]);
            } else {
                for v in start_v..start_v + run_len {
                    out_row_sums.push(src_rs.get(v).copied().unwrap_or(0));
                }
            }
        }
        i += run_len;
    }

    if bias.is_none() {
        out_bias.resize(sh_len, 0.0);
    }
}

/// Transposed f32 variant of [`gather_shortlist_projection_into`] for models
/// without intgemm8 embeddings: gathers the shortlist rows into a
/// `[k, sh_len]` B-layout matrix for [`gemv_biased`].
///
/// Writes into `out_proj` where `out_proj[t * sh_len + j] = Wemb[shortlist[j]][t]` and
/// `out_bias[j]` is the corresponding bias.
fn gather_shortlist_projection_f32_into(
    wemb: &Tensor2,
    bias: Option<&Tensor2>,
    shortlist: &[u32],
    out_proj: &mut Vec<f32>,
    out_bias: &mut Vec<f32>,
) {
    let d_model = wemb.shape[1];
    let max_vocab = wemb.shape.first().copied().unwrap_or(0);
    let sh_len = shortlist.len();
    let total_elems = sh_len * d_model;

    out_proj.clear();
    out_proj.resize(total_elems, 0.0);
    for (j, &v) in shortlist.iter().enumerate() {
        let v_idx = v as usize;
        if v_idx < max_vocab {
            let row = &wemb.data[v_idx * d_model..(v_idx + 1) * d_model];
            // Transpose into [k, sh_len] for the gemv kernel's B layout.
            for (t, &val) in row.iter().enumerate() {
                out_proj[t * sh_len + j] = val;
            }
        }
    }

    out_bias.clear();
    if out_bias.capacity() < sh_len {
        out_bias.reserve(sh_len - out_bias.capacity());
    }
    if let Some(b) = bias {
        for &v in shortlist {
            out_bias.push(b.data.get(v as usize).copied().unwrap_or(0.0));
        }
    } else {
        out_bias.resize(sh_len, 0.0);
    }
}

use crate::engine::tensor::quantize_row;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------
/// Greedy argmax over logits.
///
/// Strictly adheres to `score > max_score` tie-breaking, returning the
/// earliest index on ties (matching C++ `std::max_element` and the scalar reference).
/// A compact single loop is friendly to modern branch predictors: since only a tiny
/// fraction of scores update the maximum, the branch is predicted not-taken nearly 100%
/// of the time (1 cycle per element) with zero bubbles from multi-way branching.
#[inline]
pub(crate) fn argmax_greedy_scalar(logits: &[f32]) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut best_idx = 0;
    let mut max_score = logits[0];
    for (idx, &score) in logits.iter().enumerate().skip(1) {
        if score > max_score {
            max_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn argmax_greedy_avx2(logits: &[f32]) -> usize {
    use std::arch::x86_64::*;
    let n = logits.len();
    if n == 0 {
        return 0;
    }
    let ptr = logits.as_ptr();

    // Pass 1: find max_score using _mm256_max_ps reduction
    let chunks32 = n / 32;
    let mut max_v0 = _mm256_set1_ps(*ptr);
    let mut max_v1 = max_v0;
    let mut max_v2 = max_v0;
    let mut max_v3 = max_v0;

    for i in 0..chunks32 {
        let p = ptr.add(i * 32);
        max_v0 = _mm256_max_ps(max_v0, _mm256_loadu_ps(p));
        max_v1 = _mm256_max_ps(max_v1, _mm256_loadu_ps(p.add(8)));
        max_v2 = _mm256_max_ps(max_v2, _mm256_loadu_ps(p.add(16)));
        max_v3 = _mm256_max_ps(max_v3, _mm256_loadu_ps(p.add(24)));
    }
    max_v0 = _mm256_max_ps(_mm256_max_ps(max_v0, max_v1), _mm256_max_ps(max_v2, max_v3));

    let rem_start = chunks32 * 32;
    let rem_chunks8 = (n - rem_start) / 8;
    for i in 0..rem_chunks8 {
        max_v0 = _mm256_max_ps(max_v0, _mm256_loadu_ps(ptr.add(rem_start + i * 8)));
    }

    // Horizontal reduction of max_v0
    let v128 = _mm_max_ps(_mm256_castps256_ps128(max_v0), _mm256_extractf128_ps(max_v0, 1));
    let v64 = _mm_max_ps(v128, _mm_movehl_ps(v128, v128));
    let v32 = _mm_max_ss(v64, _mm_shuffle_ps(v64, v64, 1));
    let mut max_score = _mm_cvtss_f32(v32);

    let tail_start = rem_start + rem_chunks8 * 8;
    for i in tail_start..n {
        let s = *ptr.add(i);
        if s > max_score {
            max_score = s;
        }
    }

    // Pass 2: vectorized index search with _mm256_cmp_ps and _mm256_movemask_ps
    let max_v = _mm256_set1_ps(max_score);
    let chunks8 = n / 8;
    for c in 0..chunks8 {
        let v = _mm256_loadu_ps(ptr.add(c * 8));
        let eq = _mm256_cmp_ps(v, max_v, _CMP_EQ_OQ);
        let mask = _mm256_movemask_ps(eq);
        if mask != 0 {
            let lane = mask.trailing_zeros() as usize;
            return c * 8 + lane;
        }
    }

    for i in (chunks8 * 8)..n {
        if *ptr.add(i) == max_score {
            return i;
        }
    }

    0
}

#[inline(always)]
pub(crate) fn find_contiguous_prefix_len(indices: &[u32]) -> usize {
    let mut i = 0;
    while i < indices.len() && indices[i] == i as u32 {
        i += 1;
    }
    i
}

#[inline(always)]
pub(crate) fn shortlist_index_of(
    shortlist_indices: &[u32],
    prefix_len: usize,
    tok: u32,
) -> Option<usize> {
    let tok_usize = tok as usize;
    if tok_usize < prefix_len {
        Some(tok_usize)
    } else if prefix_len < shortlist_indices.len() {
        shortlist_indices[prefix_len..]
            .binary_search(&tok)
            .ok()
            .map(|idx| prefix_len + idx)
    } else {
        None
    }
}

#[inline]
pub(crate) fn argmax_greedy(logits: &[f32]) -> usize {
    if logits.len() < 64 {
        return argmax_greedy_scalar(logits);
    }
    #[cfg(target_arch = "x86_64")]
    if crate::utils::arch::has_avx2() {
        return unsafe { argmax_greedy_avx2(logits) };
    }
    argmax_greedy_scalar(logits)
}
/// Google NMT length penalty normalization factor:
/// LP(Y) = ((5 + |Y|) / 6.0)^alpha.
#[inline]
pub(crate) fn google_length_penalty(length: usize, alpha: f32) -> f32 {
    if (alpha - 0.0).abs() < 1e-5 {
        1.0
    } else if (alpha - 1.0).abs() < 1e-5 {
        (5.0 + length as f32) / 6.0
    } else {
        ((5.0 + length as f32) / 6.0).powf(alpha)
    }
}

/// In-place repetition penalty and n-gram blocking.
///
/// For already generated tokens:
///   if logit > 0: logit /= penalty
///   if logit <= 0: logit *= penalty
///
/// Default soft damping delta for candidate tokens completing an n-gram seen once in history.
pub const NGRAM_SOFT_DAMPING: f32 = 3.5;

/// Apply repetition penalty and 2-stage n-gram repetition prevention to `logit_buf` in-place.
///
/// 1. Repetition penalty:
///    - if logit > 0: `logit /= penalty`
///    - if logit <= 0: `logit *= penalty`
///
/// 2. N-gram 2-stage damping & frequency escalation:
///    - Candidate completing an n-gram matched 1 time (first repeat, e.g. legitimate rhetoric / 'step by step'):
///      `logit -= 3.5` (soft damping, preserving high-confidence tokens in reasonable positive range)
///    - Candidate completing an n-gram matched >= 2 times (infinite autoregressive loop tendency):
///      - if exempt (e.g. EOS, structural tokens): soft damped (`logit -= 3.5 * match_count`)
///      - otherwise: `logit = -1e9` (heavy blocking to eliminate dead loops)
#[inline]
#[allow(dead_code)]
pub(crate) fn apply_penalties(
    logit_buf: &mut [f32],
    shortlist_indices: &[u32],
    tokens: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
) {
    apply_penalties_with_exemptions(
        logit_buf,
        shortlist_indices,
        tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        &[],
    );
}

/// Apply repetition penalty and 2-stage n-gram repetition prevention with explicit token exemptions.
#[inline]
pub(crate) fn apply_penalties_with_exemptions(
    logit_buf: &mut [f32],
    shortlist_indices: &[u32],
    tokens: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    exempt_tokens: &[u32],
) {
    let prefix_len = find_contiguous_prefix_len(shortlist_indices);
    apply_penalties_with_exemptions_fast(
        logit_buf,
        shortlist_indices,
        prefix_len,
        tokens,
        repetition_penalty,
        no_repeat_ngram_size,
        exempt_tokens,
    );
}

/// Fast variant of `apply_penalties_with_exemptions` reusing precomputed contiguous prefix length.
#[inline]
pub(crate) fn apply_penalties_with_exemptions_fast(
    logit_buf: &mut [f32],
    shortlist_indices: &[u32],
    prefix_len: usize,
    tokens: &[u32],
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    exempt_tokens: &[u32],
) {
    if tokens.is_empty() {
        return;
    }

    // 1. Repetition penalty
    if repetition_penalty > 0.0 && (repetition_penalty - 1.0).abs() > 1e-5 {
        let inv_penalty = 1.0 / repetition_penalty;

        if tokens.len() <= 64 {
            // In-place single-pass linear deduplication (L1 cache scan faster than sort_unstable)
            for (i, &tok) in tokens.iter().enumerate() {
                if exempt_tokens.contains(&tok) {
                    continue;
                }
                if !tokens[..i].contains(&tok) {
                    if let Some(idx) = shortlist_index_of(shortlist_indices, prefix_len, tok) {
                        let logit = &mut logit_buf[idx];
                        if *logit > 0.0 {
                            *logit *= inv_penalty;
                        } else {
                            *logit *= repetition_penalty;
                        }
                    }
                }
            }
        } else {
            let mut stack_buf = [0u32; 256];
            let mut heap_buf;
            let unique_tokens: &[u32] = if tokens.len() <= stack_buf.len() {
                stack_buf[..tokens.len()].copy_from_slice(tokens);
                let s = &mut stack_buf[..tokens.len()];
                s.sort_unstable();
                let mut len = 0;
                for i in 0..s.len() {
                    if (i == 0 || s[i] != s[i - 1]) && !exempt_tokens.contains(&s[i]) {
                        s[len] = s[i];
                        len += 1;
                    }
                }
                &s[..len]
            } else {
                heap_buf = tokens.to_vec();
                heap_buf.sort_unstable();
                heap_buf.dedup();
                heap_buf.retain(|tok| !exempt_tokens.contains(tok));
                &heap_buf
            };
            for &tok in unique_tokens {
                if let Some(idx) = shortlist_index_of(shortlist_indices, prefix_len, tok) {
                    let logit = &mut logit_buf[idx];
                    if *logit > 0.0 {
                        *logit *= inv_penalty;
                    } else {
                        *logit *= repetition_penalty;
                    }
                }
            }
        }
    }

    // 2. N-gram 2-stage damping & frequency escalation
    if no_repeat_ngram_size > 0 && tokens.len() >= no_repeat_ngram_size {
        let n = no_repeat_ngram_size;
        let p_len = n - 1;

        // Zero heap allocation: fixed stack buffer for matching candidate tokens & counts.
        const MAX_CANDIDATES: usize = 256;
        let mut cands = [0u32; MAX_CANDIDATES];
        let mut counts = [0u8; MAX_CANDIDATES];
        let mut n_cand = 0;

        let mut record_cand = |cand: u32| {
            for i in 0..n_cand {
                if cands[i] == cand {
                    counts[i] = counts[i].saturating_add(1);
                    return;
                }
            }
            if n_cand < MAX_CANDIDATES {
                cands[n_cand] = cand;
                counts[n_cand] = 1;
                n_cand += 1;
            }
        };

        match p_len {
            0 => {
                for &tok in tokens {
                    record_cand(tok);
                }
            }
            1 => {
                let p0 = tokens[tokens.len() - 1];
                for w in tokens.windows(2) {
                    if w[0] == p0 {
                        record_cand(w[1]);
                    }
                }
            }
            2 => {
                let p0 = tokens[tokens.len() - 2];
                let p1 = tokens[tokens.len() - 1];
                for w in tokens.windows(3) {
                    if w[0] == p0 && w[1] == p1 {
                        record_cand(w[2]);
                    }
                }
            }
            3 => {
                let p0 = tokens[tokens.len() - 3];
                let p1 = tokens[tokens.len() - 2];
                let p2 = tokens[tokens.len() - 1];
                for w in tokens.windows(4) {
                    if w[0] == p0 && w[1] == p1 && w[2] == p2 {
                        record_cand(w[3]);
                    }
                }
            }
            _ => {
                let cur_prefix = &tokens[tokens.len() - p_len..];
                let p_first = cur_prefix[0];
                let p_last = cur_prefix[p_len - 1];
                for w in tokens.windows(n) {
                    if w[0] == p_first
                        && w[p_len - 1] == p_last
                        && &w[..p_len] == cur_prefix
                    {
                        record_cand(w[p_len]);
                    }
                }
            }
        }

        for i in 0..n_cand {
            let cand = cands[i];
            let match_count = counts[i];
            if let Some(idx) = shortlist_index_of(shortlist_indices, prefix_len, cand) {
                let logit = &mut logit_buf[idx];
                let is_exempt = exempt_tokens.contains(&cand);

                if match_count == 1 {
                    // Soft damping: allow high-confidence legitimate repetitions (e.g. "step by step")
                    // Smoothly damp positive and negative logits
                    *logit -= NGRAM_SOFT_DAMPING;
                } else if match_count >= 2 {
                    if is_exempt {
                        // Exempt tokens (e.g. EOS, structural markers) are not hard blocked
                        *logit -= NGRAM_SOFT_DAMPING * (match_count as f32);
                    } else {
                        // Heavy blocking: eliminate autoregressive dead loops
                        *logit = -1e9;
                    }
                }
            }
        }
    }
}

/// Compute numerically stable log-softmax over logits.
#[inline]
pub(crate) fn compute_log_softmax(logits: &[f32], out: &mut [f32]) {
    debug_assert_eq!(logits.len(), out.len());
    let mut max_val = f32::NEG_INFINITY;
    for &val in logits {
        if val > max_val {
            max_val = val;
        }
    }
    if max_val.is_infinite() {
        out.fill(f32::NEG_INFINITY);
        return;
    }
    let mut sum = 0.0f32;
    for (&val, dst) in logits.iter().zip(out.iter_mut()) {
        let diff = val - max_val;
        let exp_v = if diff < -80.0 { 0.0 } else { diff.exp() };
        *dst = exp_v;
        sum += exp_v;
    }
    let log_sum = sum.ln();
    for (&val, dst) in logits.iter().zip(out.iter_mut()) {
        *dst = val - max_val - log_sum;
    }
}

/// Apply adaptive dynamic EOS length guidance and repetition damping/gain on `logit_buf` in-place.
///
/// Prevents micro-distilled models under greedy and beam decoding from prematurely emitting EOS
/// on intermediate pauses/commas when decoding longer sentences, while strictly avoiding
/// forced hallucinations on short utterances.
///
/// - `L_min = ((source_len as f32 * eff_expansion_ratio * min_factor).round() as usize).clamp(min_bound, 120)`
///   where `min_bound = 1` if `source_len <= 1`, else `2`.
/// - `L_soft_max = ((source_len as f32 * eff_expansion_ratio * max_factor).round() as usize).max(L_min + 10)`
///   constrained for `expansion_ratio < 0.5` to not drop below `max(L_min + 15, round(source_len * 1.2))`.
/// - When `generated_len < L_min`: applies soft quadratic damping:
///   `logit -= 3.5 * (1.0 - t / L_min)^2`
/// - When `generated_len > L_soft_max`: applies bounded linear gain:
///   `logit += 4.0 * min(1.0, (t - L_soft_max) / 30.0)`
#[inline]
pub(crate) fn apply_adaptive_dynamic_eos_guidance(
    logit_buf: &mut [f32],
    eos_idx: usize,
    source_len: usize,
    generated_len: usize,
    expansion_ratio: f32,
    min_factor: f32,
    max_factor: f32,
) {
    if source_len == 0 {
        return;
    }
    let expansion_ratio = if expansion_ratio > 0.0 { expansion_ratio } else { 1.0 };
    let min_factor = if min_factor > 0.0 { min_factor } else { 0.5 };
    let max_factor = if max_factor > 0.0 { max_factor } else { 2.5 };

    let eff_expansion_ratio = if expansion_ratio < 0.5 || expansion_ratio > 2.0 {
        1.0 + (expansion_ratio - 1.0) * 0.75
    } else {
        expansion_ratio
    };

    let min_bound = if source_len <= 3 { 1 } else { 2 };
    let l_min = ((source_len as f32 * eff_expansion_ratio * min_factor).round() as usize).clamp(min_bound, 120);
    let mut l_soft_max = ((source_len as f32 * eff_expansion_ratio * max_factor).round() as usize).max(l_min + 10);
    if expansion_ratio < 0.5 {
        l_soft_max = l_soft_max.max(l_min + 15).max((source_len as f32 * 1.2).round() as usize);
    }
    l_soft_max = l_soft_max.min(480).max(l_min + 1);
    if let Some(logit) = logit_buf.get_mut(eos_idx) {
        if generated_len < l_min {
            let ratio = 1.0 - (generated_len as f32 / l_min as f32);
            *logit -= 3.5 * (ratio * ratio);
        } else if generated_len > l_soft_max {
            let delta = ((generated_len - l_soft_max) as f32 / 30.0).min(1.0);
            *logit += 4.0 * delta;
        }
    }
}

/// Backwards-compatible dynamic EOS bias wrapper preserving default parameters (expansion_ratio = 1.0, min_factor = 0.5, max_factor = 2.5).
#[allow(dead_code)]
#[inline]
pub(crate) fn apply_dynamic_eos_bias(
    logit_buf: &mut [f32],
    eos_idx: usize,
    source_len: usize,
    generated_len: usize,
) {
    apply_adaptive_dynamic_eos_guidance(
        logit_buf,
        eos_idx,
        source_len,
        generated_len,
        1.0,
        0.5,
        2.5,
    );
}


impl Transformer {
    /// Run the full transformer encoder-decoder forward pass with
    /// **autoregressive decoding** inside.
    ///
    /// 1. Encoder: append `[EOS]` to `source_ids` (no BOS), embed with
    ///    `sqrt(d_model)` scaling, add sinusoidal positions, run the encoder
    ///    stack (self-attention + FFN per layer, postprocess "dan") →
    ///    `enc_state`.
    /// 2. Decoder loop: start from a zero embedding (marian's RNN start
    ///    state — no BOS token), per step embed the previously generated
    ///    word, scale and position-encode it, run the decoder stack
    ///    (SSRU + cross-attention + FFN per layer, postprocess "dan") while
    ///    carrying the per-layer SSRU cell states, project the hidden state
    ///    with the tied output layer, gather shortlist entries, and
    ///    greedy-argmax the shortlist to pick the next token. Stop when
    ///    `[EOS]` is predicted or `max_length_break` is reached.
    ///
    /// Returns the generated token id sequence (one id per decoding step,
    /// including a trailing EOS or the extra post-truncation token).
    pub fn forward(
        model: &Model,
        spm_src: &SentencePiece,
        spm_trg: &SentencePiece,
        source_ids: &[u32],
        shortlist_indices: &[u32],
        config: &ForwardConfig,
    ) -> Vec<u32> {
        let weights = config.weights(model);
        Self::forward_with_weights(
            &weights,
            spm_src,
            spm_trg,
            source_ids,
            shortlist_indices,
            config,
        )
    }

    /// Forward pass with pre-resolved layer weights, avoiding hash map lookups.
    pub fn forward_with_weights(
        weights: &LayerWeightsView,
        spm_src: &SentencePiece,
        spm_trg: &SentencePiece,
        source_ids: &[u32],
        shortlist_indices: &[u32],
        config: &ForwardConfig,
    ) -> Vec<u32> {
        let mut out = Vec::new();
        Self::forward_with_weights_into(
            weights,
            spm_src,
            spm_trg,
            source_ids,
            shortlist_indices,
            config,
            &mut out,
        );
        out
    }

    /// Forward pass writing output tokens into the caller-owned buffer,
    /// eliminating per-request allocations on the decode hot path.
    pub fn forward_with_weights_into(
        weights: &LayerWeightsView,
        spm_src: &SentencePiece,
        spm_trg: &SentencePiece,
        source_ids: &[u32],
        shortlist_indices: &[u32],
        config: &ForwardConfig,
        out_tokens: &mut Vec<u32>,
    ) {
        Self::forward_with_weights_and_alignments_into(
            weights,
            spm_src,
            spm_trg,
            source_ids,
            shortlist_indices,
            config,
            out_tokens,
            None,
        );
    }

    /// Forward pass writing output tokens and optional token-level cross-attention
    /// alignments (target token -> source token argmax index).
    pub fn forward_with_weights_and_alignments_into(
        weights: &LayerWeightsView,
        spm_src: &SentencePiece,
        spm_trg: &SentencePiece,
        source_ids: &[u32],
        shortlist_indices: &[u32],
        config: &ForwardConfig,
        out_tokens: &mut Vec<u32>,
        mut alignments: Option<&mut Vec<usize>>,
    ) {
        out_tokens.clear();
        if let Some(a) = &mut alignments {
            a.clear();
        }
        if shortlist_indices.is_empty() {
            return;
        }

        let d_model = config.d_model;
        let d_ff = weights.d_ff;

        let mut fwd_bufs = FORWARD_BUFFERS.with(|cell| {
            let mut opt = cell.borrow_mut();
            match opt.take() {
                Some(mut bufs) => {
                    bufs.reset(config.dec_layers, d_model, d_ff);
                    bufs
                }
                None => ForwardBuffers::new(config.dec_layers, d_model, d_ff),
            }
        });

        // ---- 1. Encoder ----
        profiling::scope("encoder_forward", || {
            encoder_forward_into(
                weights,
                spm_src,
                source_ids,
                config,
                &mut fwd_bufs.enc_state,
            );
        });

        // ---- 2. Autoregressive decoder loop ----
        let eos_id = spm_trg.eos_id();

        // Precompute every decoder layer's encoder K/V into fwd_bufs.enc_k / enc_v.
        for (i, layer) in weights.decoder_layers.iter().enumerate() {
            if let (LinW::Q8(kw, qa_k), LinW::Q8(vw, qa_v)) = (&layer.ctx_wk, &layer.ctx_wv) {
                if qa_k == qa_v {
                    fwd_bufs.enc_state.quantized_matmul_pair_into(
                        kw,
                        Some(layer.ctx_bk),
                        &mut fwd_bufs.linear_scratch,
                        vw,
                        Some(layer.ctx_bv),
                        &mut fwd_bufs.linear_scratch_v,
                        *qa_k,
                    );
                    rearrange_k_heads_into(
                        &fwd_bufs.linear_scratch,
                        config.heads,
                        &mut fwd_bufs.enc_k[i],
                    );
                    rearrange_v_heads_into(
                        &fwd_bufs.linear_scratch_v,
                        config.heads,
                        &mut fwd_bufs.enc_v[i],
                    );
                    continue;
                }
            }
            linear_into(
                &fwd_bufs.enc_state,
                &layer.ctx_wk,
                layer.ctx_bk,
                &mut fwd_bufs.linear_scratch,
            );
            rearrange_k_heads_into(
                &fwd_bufs.linear_scratch,
                config.heads,
                &mut fwd_bufs.enc_k[i],
            );

            linear_into(
                &fwd_bufs.enc_state,
                &layer.ctx_wv,
                layer.ctx_bv,
                &mut fwd_bufs.linear_scratch,
            );
            rearrange_v_heads_into(
                &fwd_bufs.linear_scratch,
                config.heads,
                &mut fwd_bufs.enc_v[i],
            );
        }

        let ForwardBuffers {
            ref mut cell_states,
            ref mut cell_scratches,
            ref mut kv_cache,
            ref mut scratch,
            ref mut step_input,
            ref mut xq_row,
            ref mut logit_buf,
            dec_ids: _,
            ref mut short_proj_q8_data,
            ref mut short_row_sums,
            ref mut short_bias,
            ref mut short_proj_f32,
            ref mut cached_wemb_ptr,
            ref mut cached_shortlist,
            ref enc_k,
            ref enc_v,
            linear_scratch: _,
            linear_scratch_v: _,
            enc_state: _,
            ref mut attn_accum,
        } = fwd_bufs;
        let dec_layers = config.dec_layers;
        let enc_src_len = enc_k.first().map(|k| k.src_len).unwrap_or(source_ids.len());
        let scale = (d_model as f32).sqrt();
        // Tied output projection pre-gathered for the shortlist (P: fixed
        // across all decoder steps, cached across consecutive identical shortlist calls).
        let (proj_qa, proj_denom) = if let Some(wemb_q8) = weights.trg_wemb_q8 {
            let wemb_ptr = wemb_q8.data.as_ptr() as usize;
            let can_reuse = *cached_wemb_ptr == wemb_ptr
                && cached_shortlist.as_slice() == shortlist_indices
                && short_proj_q8_data.len() == shortlist_indices.len() * d_model;
            if !can_reuse {
                gather_shortlist_projection_into(
                    wemb_q8,
                    Some(weights.output_bias),
                    shortlist_indices,
                    short_proj_q8_data,
                    short_row_sums,
                    short_bias,
                );
                *cached_wemb_ptr = wemb_ptr;
                cached_shortlist.clear();
                cached_shortlist.extend_from_slice(shortlist_indices);
            }
            let proj_qa = weights.proj_qa;
            let proj_denom = proj_qa * wemb_q8.quant_mult;
            (proj_qa, proj_denom)
        } else if let Some(wemb_f32) = weights.trg_wemb_f32 {
            let wemb_ptr = wemb_f32.data.as_ptr() as usize;
            let can_reuse = *cached_wemb_ptr == wemb_ptr
                && cached_shortlist.as_slice() == shortlist_indices
                && short_proj_f32.len() == shortlist_indices.len() * d_model;
            if !can_reuse {
                gather_shortlist_projection_f32_into(
                    wemb_f32,
                    Some(weights.output_bias),
                    shortlist_indices,
                    short_proj_f32,
                    short_bias,
                );
                *cached_wemb_ptr = wemb_ptr;
                cached_shortlist.clear();
                cached_shortlist.extend_from_slice(shortlist_indices);
            }
            (0.0, 0.0)
        } else {
            panic!("model is missing target embedding");
        };
        let sh_len = shortlist_indices.len();
        let qm_mult = weights
            .trg_wemb_q8
            .map(|w| scale / w.quant_mult)
            .unwrap_or(0.0);
        logit_buf.resize(sh_len, 0.0);
        let need_align = alignments.is_some();
        if need_align {
            attn_accum.resize(enc_src_len, 0.0);
        }
        let has_avx2 = {
            #[cfg(target_arch = "x86_64")]
            {
                crate::utils::arch::has_avx2()
            }
            #[cfg(not(target_arch = "x86_64"))]
            {
                false
            }
        };
        let row_i8_kernel = matmul_row_i8_dispatch();
        let short_bias_slice = short_bias.as_slice();
        let short_row_sums_opt = if short_row_sums.is_empty() {
            None
        } else {
            Some(short_row_sums.as_slice())
        };
        // Hoist loop invariants out of autoregressive decoding loops
        let beam_size = config.beam_size;
        let heads = config.heads;
        let autoreg_type = config.autoreg_type;
        let repetition_penalty = config.repetition_penalty;
        let no_repeat_ngram_size = config.no_repeat_ngram_size;
        let max_length_break = config.max_length_break;
        let length_penalty = config.length_penalty;
        let position_table = config.position_table.as_slice();
        let has_penalties = (repetition_penalty > 0.0 && (repetition_penalty - 1.0).abs() > 1e-5)
            || no_repeat_ngram_size > 0;
        let prefix_len = find_contiguous_prefix_len(shortlist_indices);
        let eos_shortlist_idx = shortlist_index_of(shortlist_indices, prefix_len, eos_id);
        let eff_expansion = if config.expansion_ratio > 0.0 { config.expansion_ratio } else { 1.0 };
        let eff_max_factor = if config.max_length_factor > 0.0 { config.max_length_factor } else { 2.5 };
        let dyn_limit = if source_ids.is_empty() {
            max_length_break
        } else {
            ((source_ids.len() as f32 * eff_expansion * eff_max_factor).round() as usize + 64)
                .max(32)
                .min(max_length_break)
        };
        if beam_size <= 1 {
            loop {
                profiling::scope("dec/embed+pos", || {
                    let step_buf = &mut step_input.data;
                    // On the int8 path the row is dequantized straight into `step_buf`
                    let pos = out_tokens.len();
                    let pos_slice = &position_table[pos * d_model..(pos + 1) * d_model];
                    if let Some(&last) = out_tokens.last() {
                        if let Some(wemb_q8) = weights.trg_wemb_q8 {
                            let max_vocab = wemb_q8.shape.first().copied().unwrap_or(0);
                            let eff_last = if (last as usize) < max_vocab {
                                Some(last as usize)
                            } else if (spm_trg.unk_id() as usize) < max_vocab {
                                Some(spm_trg.unk_id() as usize)
                            } else {
                                None
                            };
                            if let Some(valid_id) = eff_last {
                                let start = valid_id * d_model;
                                let src = &wemb_q8.data[start..start + d_model];
                                #[cfg(target_arch = "x86_64")]
                                if has_avx2 {
                                    unsafe {
                                        embed_add_pos_avx2(src, pos_slice, step_buf, qm_mult);
                                    }
                                } else {
                                    for ((dst, &v), &pos) in
                                        step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                    {
                                        *dst = (v as f32) * qm_mult + pos;
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                for ((dst, &v), &pos) in
                                    step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                {
                                    *dst = (v as f32) * qm_mult + pos;
                                }
                            } else {
                                step_buf.copy_from_slice(pos_slice);
                            }
                        } else if let Some(w) = weights.trg_wemb_f32 {
                            let max_vocab = w.shape.first().copied().unwrap_or(0);
                            let d = w.shape[1];
                            let eff_last = if (last as usize) < max_vocab {
                                Some(last as usize)
                            } else if (spm_trg.unk_id() as usize) < max_vocab {
                                Some(spm_trg.unk_id() as usize)
                            } else {
                                None
                            };
                            if let Some(valid_id) = eff_last {
                                let src = &w.data[valid_id * d..(valid_id + 1) * d];
                                for ((dst, &v), &pos) in
                                    step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                {
                                    *dst = v * scale + pos;
                                }
                            } else {
                                step_buf.copy_from_slice(pos_slice);
                            }
                        }
                    } else {
                        step_buf.copy_from_slice(pos_slice);
                    }
                });
                if need_align {
                    attn_accum.fill(0.0);
                }
                profiling::scope("decoder_forward", || {
                    let attn_opt = if need_align {
                        Some(attn_accum.as_mut_slice())
                    } else {
                        None
                    };
                    decoder_forward(
                        step_input,
                        &weights.decoder_layers,
                        enc_k,
                        enc_v,
                        heads,
                        autoreg_type,
                        &mut cell_states[..dec_layers],
                        &mut cell_scratches[..dec_layers],
                        kv_cache,
                        LAYER_NORM_EPS,
                        scratch,
                        attn_opt,
                    );
                });
                // production path — 4× less weight traffic than an f32 gemv).
                if weights.trg_wemb_q8.is_some() {
                    profiling::scope("dec/out_proj", || {
                        quantize_row(&step_input.data, proj_qa, xq_row);
                        // SAFETY: the dispatch fn was selected from this CPU's feature
                        // flags; slices are valid and sized by construction above.
                        unsafe {
                            row_i8_kernel(
                                xq_row,
                                short_proj_q8_data,
                                d_model,
                                sh_len,
                                Some(short_bias_slice),
                                short_row_sums_opt,
                                logit_buf,
                                proj_denom,
                                false,
                            )
                        };
                    });
                } else {
                    profiling::scope("dec/out_proj", || {
                        gemv_biased(
                            &step_input.data,
                            short_proj_f32.as_slice(),
                            d_model,
                            sh_len,
                            short_bias.as_slice(),
                            logit_buf,
                        );
                    });
                }

                // Apply repetition penalty and n-gram blocking in-place
                if has_penalties {
                    apply_penalties_with_exemptions_fast(
                        logit_buf,
                        shortlist_indices,
                        prefix_len,
                        out_tokens.as_slice(),
                        repetition_penalty,
                        no_repeat_ngram_size,
                        &[eos_id],
                    );
                }

                // Adaptive dynamic EOS length-guided damping and repetition control before argmax
                if let Some(eos_idx) = eos_shortlist_idx {
                    apply_adaptive_dynamic_eos_guidance(
                        logit_buf,
                        eos_idx,
                        source_ids.len(),
                        out_tokens.len(),
                        config.expansion_ratio,
                        config.min_length_factor,
                        config.max_length_factor,
                    );
                }

                // Greedy argmax over the shortlist -> next token (full vocab id).
                let next_token = profiling::scope("dec/argmax", || {
                    let local_best_idx = argmax_greedy(logit_buf);
                    shortlist_indices[local_best_idx]
                });
                out_tokens.push(next_token);
                if let Some(align_vec) = &mut alignments {
                    let mut best_src = 0;
                    let mut max_val = f32::NEG_INFINITY;
                    for (j, &val) in attn_accum.iter().enumerate() {
                        if val > max_val {
                            max_val = val;
                            best_src = j;
                        }
                    }
                    let safe_src = best_src.min(source_ids.len().saturating_sub(1));
                    align_vec.push(safe_src);
                }
                if next_token == eos_id || out_tokens.len() >= dyn_limit {
                    break;
                }
            }
        } else {
            #[derive(Clone)]
            struct BeamHypothesis {
                tokens: Vec<u32>,
                score: f32,
                cell_states: Vec<Tensor2>,
                kv_cache: DecoderKVCache,
                alignments: Vec<usize>,
                eos_reached: bool,
            }

            struct Candidate {
                parent_idx: usize,
                token: u32,
                score: f32,
                norm_score: f32,
                align_src: usize,
                is_eos: bool,
                is_finished_beam: bool,
            }

            let initial_cells = cell_states[..dec_layers].to_vec();
            let mut initial_kv = kv_cache.clone();
            initial_kv.reset();
            let mut beams = vec![BeamHypothesis {
                tokens: Vec::new(),
                score: 0.0,
                cell_states: initial_cells,
                kv_cache: initial_kv,
                alignments: Vec::new(),
                eos_reached: false,
            }];
            let mut log_prob_buf = vec![0.0f32; sh_len];
            for _step in 0..dyn_limit {
                if beams.iter().all(|b| b.eos_reached) {
                    break;
                }

                let mut candidates: Vec<Candidate> = Vec::new();

                for (b_idx, beam) in beams.iter_mut().enumerate() {
                    if beam.eos_reached || beam.tokens.len() >= dyn_limit {
                        let len = beam.tokens.len().max(1);
                        let norm = beam.score / google_length_penalty(len, length_penalty);
                        candidates.push(Candidate {
                            parent_idx: b_idx,
                            token: eos_id,
                            score: beam.score,
                            norm_score: norm,
                            align_src: 0,
                            is_eos: true,
                            is_finished_beam: true,
                        });
                        continue;
                    }

                    // 1. Embed + pos
                    let step_buf = &mut step_input.data;
                    let pos = beam.tokens.len();
                    let pos_slice = &position_table[pos * d_model..(pos + 1) * d_model];
                    if let Some(&last) = beam.tokens.last() {
                        if let Some(wemb_q8) = weights.trg_wemb_q8 {
                            let max_vocab = wemb_q8.shape.first().copied().unwrap_or(0);
                            let eff_last = if (last as usize) < max_vocab {
                                Some(last as usize)
                            } else if (spm_trg.unk_id() as usize) < max_vocab {
                                Some(spm_trg.unk_id() as usize)
                            } else {
                                None
                            };
                            if let Some(valid_id) = eff_last {
                                let start = valid_id * d_model;
                                let src = &wemb_q8.data[start..start + d_model];
                                #[cfg(target_arch = "x86_64")]
                                if has_avx2 {
                                    unsafe {
                                        embed_add_pos_avx2(src, pos_slice, step_buf, qm_mult);
                                    }
                                } else {
                                    for ((dst, &v), &pos) in
                                        step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                    {
                                        *dst = (v as f32) * qm_mult + pos;
                                    }
                                }
                                #[cfg(not(target_arch = "x86_64"))]
                                for ((dst, &v), &pos) in
                                    step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                {
                                    *dst = (v as f32) * qm_mult + pos;
                                }
                            } else {
                                step_buf.copy_from_slice(pos_slice);
                            }
                        } else if let Some(w) = weights.trg_wemb_f32 {
                            let max_vocab = w.shape.first().copied().unwrap_or(0);
                            let d = w.shape[1];
                            let eff_last = if (last as usize) < max_vocab {
                                Some(last as usize)
                            } else if (spm_trg.unk_id() as usize) < max_vocab {
                                Some(spm_trg.unk_id() as usize)
                            } else {
                                None
                            };
                            if let Some(valid_id) = eff_last {
                                let src = &w.data[valid_id * d..(valid_id + 1) * d];
                                for ((dst, &v), &pos) in
                                    step_buf.iter_mut().zip(src.iter()).zip(pos_slice.iter())
                                {
                                    *dst = v * scale + pos;
                                }
                            } else {
                                step_buf.copy_from_slice(pos_slice);
                            }
                        }
                    } else {
                        step_buf.copy_from_slice(pos_slice);
                    }

                    // 2. Decoder layers forward
                    if need_align {
                        attn_accum.fill(0.0);
                    }
                    let attn_opt = if need_align {
                        Some(attn_accum.as_mut_slice())
                    } else {
                        None
                    };
                    decoder_forward(
                        step_input,
                        &weights.decoder_layers,
                        enc_k,
                        enc_v,
                        heads,
                        autoreg_type,
                        &mut beam.cell_states[..dec_layers],
                        &mut cell_scratches[..dec_layers],
                        &mut beam.kv_cache,
                        LAYER_NORM_EPS,
                        scratch,
                        attn_opt,
                    );

                    // 3. Out projection into logit_buf
                    if weights.trg_wemb_q8.is_some() {
                        quantize_row(&step_input.data, proj_qa, xq_row);
                        unsafe {
                            row_i8_kernel(
                                xq_row,
                                short_proj_q8_data,
                                d_model,
                                sh_len,
                                Some(short_bias_slice),
                                short_row_sums_opt,
                                logit_buf,
                                proj_denom,
                                false,
                            )
                        };
                    } else {
                        gemv_biased(
                            &step_input.data,
                            short_proj_f32.as_slice(),
                            d_model,
                            sh_len,
                            short_bias.as_slice(),
                            logit_buf,
                        );
                    }

                    let safe_src = if need_align {
                        let mut best_src = 0;
                        let mut max_val = f32::NEG_INFINITY;
                        for (j, &val) in attn_accum.iter().enumerate() {
                            if val > max_val {
                                max_val = val;
                                best_src = j;
                            }
                        }
                        best_src.min(source_ids.len().saturating_sub(1))
                    } else {
                        0
                    };

                    // 4. In-place penalties on logit_buf
                    if has_penalties {
                        apply_penalties_with_exemptions_fast(
                            logit_buf,
                            shortlist_indices,
                            prefix_len,
                            &beam.tokens,
                            repetition_penalty,
                            no_repeat_ngram_size,
                            &[eos_id],
                        );
                    }

                    if let Some(eos_idx) = eos_shortlist_idx {
                        apply_adaptive_dynamic_eos_guidance(
                            logit_buf,
                            eos_idx,
                            source_ids.len(),
                            beam.tokens.len(),
                            config.expansion_ratio,
                            config.min_length_factor,
                            config.max_length_factor,
                        );
                    }

                    // 5. Compute log-softmax
                    compute_log_softmax(logit_buf, &mut log_prob_buf);

                    let cur_len = beam.tokens.len() + 1;
                    let lp_factor = google_length_penalty(cur_len, length_penalty);

                    for (j, &tok) in shortlist_indices.iter().enumerate() {
                        let lp = log_prob_buf[j];
                        if lp < -1e8 {
                            continue;
                        }
                        let new_score = beam.score + lp;
                        let norm_score = new_score / lp_factor;
                        candidates.push(Candidate {
                            parent_idx: b_idx,
                            token: tok,
                            score: new_score,
                            norm_score,
                            align_src: safe_src,
                            is_eos: (tok == eos_id),
                            is_finished_beam: false,
                        });
                    }
                }

                if candidates.is_empty() {
                    break;
                }

                let k = beam_size.min(candidates.len());
                if candidates.len() > k {
                    candidates.select_nth_unstable_by(k - 1, |a, b| {
                        b.norm_score
                            .partial_cmp(&a.norm_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| a.token.cmp(&b.token))
                    });
                    candidates.truncate(k);
                }
                candidates.sort_unstable_by(|a, b| {
                    b.norm_score
                        .partial_cmp(&a.norm_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a.token.cmp(&b.token))
                });

                let mut next_beams = Vec::with_capacity(candidates.len());
                for cand in candidates {
                    if cand.is_finished_beam {
                        next_beams.push(beams[cand.parent_idx].clone());
                    } else {
                        let parent = &beams[cand.parent_idx];
                        let mut new_tokens = parent.tokens.clone();
                        new_tokens.push(cand.token);
                        let mut new_align = parent.alignments.clone();
                        if need_align {
                            new_align.push(cand.align_src);
                        }
                        let new_cells = parent.cell_states.clone();
                        let new_kv = parent.kv_cache.clone();
                        next_beams.push(BeamHypothesis {
                            tokens: new_tokens,
                            score: cand.score,
                            cell_states: new_cells,
                            kv_cache: new_kv,
                            alignments: new_align,
                            eos_reached: cand.is_eos,
                        });
                    }
                }
                beams = next_beams;
            }

            if let Some(best_beam) = beams.iter().max_by(|a, b| {
                let a_len = a.tokens.len().max(1);
                let b_len = b.tokens.len().max(1);
                let a_norm = a.score / google_length_penalty(a_len, length_penalty);
                let b_norm = b.score / google_length_penalty(b_len, length_penalty);
                a_norm
                    .partial_cmp(&b_norm)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                out_tokens.extend_from_slice(&best_beam.tokens);
                if let Some(align_vec) = &mut alignments {
                    align_vec.extend_from_slice(&best_beam.alignments);
                }
            }
        }

        FORWARD_BUFFERS.with(|cell| {
            *cell.borrow_mut() = Some(fwd_bufs);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EngineConfig, TranslationEngine};

    #[test]
    fn f32_shortlist_gather_transposes_rows() {
        let wemb = Tensor2::from_vec([3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let bias = Tensor2::from_vec([1, 3], vec![0.5, 0.0, -0.5]);
        let mut g = Vec::new();
        let mut sb = Vec::new();
        gather_shortlist_projection_f32_into(&wemb, Some(&bias), &[2, 0], &mut g, &mut sb);
        // Row j=2 then row j=0, transposed into [k=2, n=2] B layout.
        assert_eq!(g, vec![5.0, 1.0, 6.0, 2.0]);
        assert_eq!(sb, vec![-0.5, 0.5]);
    }

    #[test]
    fn q8_shortlist_gather_preserves_row_sums_and_slices() {
        let wemb_q8 = Quant8Tensor {
            shape: vec![3, 2],
            data: vec![1, 2, 3, 4, 5, 6].into(),
            quant_mult: 0.1,
            row_sums: Some(vec![3, 7, 11]),
        };
        let bias = Tensor2::from_vec([1, 3], vec![0.5, 0.0, -0.5]);
        let mut out_data = Vec::new();
        let mut out_row_sums = Vec::new();
        let mut sb = Vec::new();
        gather_shortlist_projection_into(
            &wemb_q8,
            Some(&bias),
            &[2, 0],
            &mut out_data,
            &mut out_row_sums,
            &mut sb,
        );
        assert_eq!(out_data, vec![5, 6, 1, 2]);
        assert_eq!(out_row_sums, vec![11, 3]);
        assert_eq!(sb, vec![-0.5, 0.5]);
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_accumulate_head_v_dk64_avx2_matches_fallback() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let kv_len = 17;
        let dk = 64;
        let mut dst_ref = vec![0.0f32; dk];
        let mut dst_avx = vec![1.234f32; dk];
        let scores: Vec<f32> = (0..kv_len).map(|j| (j as f32 + 0.5) * 0.1).collect();
        let v_data: Vec<f32> = (0..kv_len * dk)
            .map(|k| (k as f32 - 100.0) * 0.05)
            .collect();

        for j in 0..kv_len {
            let w = scores[j];
            let v_off = j * dk;
            weighted_accumulate_dk(&mut dst_ref, &v_data[v_off..v_off + dk], w);
        }

        unsafe {
            accumulate_head_v_dk64_avx2(&mut dst_avx, &v_data, 0, &scores, kv_len);
        }

        for (a, b) in dst_ref.iter().zip(dst_avx.iter()) {
            assert!(
                (a - b).abs() <= 1e-5,
                "Float bits must match within tolerance: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_argmax_greedy_matches_scalar_reference() {
        fn reference_argmax(logits: &[f32]) -> usize {
            let mut local_best_idx = 0;
            let mut max_score = logits.first().copied().unwrap_or(f32::NEG_INFINITY);
            for (idx, &score) in logits.iter().enumerate() {
                if score > max_score {
                    max_score = score;
                    local_best_idx = idx;
                }
            }
            local_best_idx
        }

        assert_eq!(argmax_greedy(&[]), reference_argmax(&[]));

        let test_lens = [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 64, 127, 256, 1000, 3999, 4000,
        ];
        for &len in &test_lens {
            // Monotonic increasing
            let inc: Vec<f32> = (0..len).map(|i| i as f32).collect();
            assert_eq!(argmax_greedy(&inc), reference_argmax(&inc), "inc len={len}");

            // Monotonic decreasing
            let dec: Vec<f32> = (0..len).map(|i| (len - i) as f32).collect();
            assert_eq!(argmax_greedy(&dec), reference_argmax(&dec), "dec len={len}");

            // All identical
            let same = vec![42.0f32; len];
            assert_eq!(
                argmax_greedy(&same),
                reference_argmax(&same),
                "same len={len}"
            );

            // Pseudo-random patterned values
            let mut pattern: Vec<f32> = (0..len)
                .map(|i| ((i * 37 + 13) % 101) as f32 - 50.0)
                .collect();
            assert_eq!(
                argmax_greedy(&pattern),
                reference_argmax(&pattern),
                "pattern len={len}"
            );

            // Pattern with multiple forced maximum ties across different streams
            if len >= 8 {
                let max_val = 999.0;
                pattern[0] = max_val;
                pattern[3] = max_val;
                pattern[5] = max_val;
                pattern[len - 1] = max_val;
                assert_eq!(
                    argmax_greedy(&pattern),
                    reference_argmax(&pattern),
                    "pattern with ties len={len}"
                );

                // Tie not at index 0
                pattern[0] = -999.0;
                pattern[1] = max_val;
                pattern[4] = max_val;
                assert_eq!(
                    argmax_greedy(&pattern),
                    reference_argmax(&pattern),
                    "tie not at 0 len={len}"
                );
            }
        }
    }

    #[test]
    fn test_forward_buffers_reset_across_d_model() {
        // Regression: switching a thread's FORWARD_BUFFERS between models with
        // different d_model must resize every data buffer, including
        // `step_input`. Previously reset() only called set_shape_2d on
        // step_input, leaving stale data from a larger model behind; the
        // decoder's first step then panicked with "copy_from_slice: source
        // slice length (256) does not match destination slice length (384)".
        let mut bufs = ForwardBuffers::new(4, 384, 1536);
        assert_eq!(bufs.step_input.data.len(), 384);

        // Shrink: d_model 384 -> 256 (e.g. en_zh-Hans -> el_en on one thread).
        bufs.reset(4, 256, 1024);
        assert_eq!(bufs.step_input.data.len(), 256);
        assert_eq!(bufs.xq_row.len(), 256);
        assert_eq!(bufs.cell_states[0].data.len(), 256);
        assert_eq!(bufs.scratch.a.data.len(), 256);
        assert_eq!(bufs.scratch.h1.data.len(), 1024);

        // Grow back: 256 -> 384.
        bufs.reset(4, 384, 1536);
        assert_eq!(bufs.step_input.data.len(), 384);
        assert_eq!(bufs.xq_row.len(), 384);
        assert_eq!(bufs.cell_states[0].data.len(), 384);
        assert_eq!(bufs.scratch.h1.data.len(), 1536);
    }

    #[test]
    fn test_forward_with_weights_and_alignments_runtime() {
        use crate::engine::{EngineConfig, TranslationEngine};
        use std::path::Path;

        let model_dir = Path::new("data/models/en_zh-Hans");
        if !model_dir.join("model.enzh.intgemm.alphas.bin").exists() {
            return;
        }

        let engine = TranslationEngine::new(
            "en",
            "zh-Hans",
            &model_dir.join("model.enzh.intgemm.alphas.bin"),
            &model_dir.join("lex.50.50.enzh.s2t.bin"),
            &model_dir.join("srcvocab.enzh.spm"),
            &model_dir.join("trgvocab.enzh.spm"),
            EngineConfig::default(),
        )
        .expect("load en_zh engine");

        let source_text = "Hello world from Rust";
        let source_ids = engine.spm_src().encode(source_text);
        assert!(!source_ids.is_empty());

        let mut seen = Vec::new();
        let mut shortlist_indices = Vec::new();
        engine.shortlist().generate_into(
            &source_ids,
            engine.spm_trg().vocab_size(),
            &mut seen,
            &mut shortlist_indices,
            engine.shortlist_policy(),
        );

        let mut out_tokens = Vec::new();
        let mut alignments = Vec::new();

        Transformer::forward_with_weights_and_alignments_into(
            engine.prepared_weights().weights(),
            engine.spm_src(),
            engine.spm_trg(),
            &source_ids,
            &shortlist_indices,
            engine.forward_config(),
            &mut out_tokens,
            Some(&mut alignments),
        );

        assert!(!out_tokens.is_empty(), "out_tokens should not be empty");
        assert_eq!(
            alignments.len(),
            out_tokens.len(),
            "each out_token must have an alignment entry"
        );

        for &src_idx in &alignments {
            assert!(
                src_idx < source_ids.len(),
                "alignment index {} must be strictly less than source_ids.len() {}",
                src_idx,
                source_ids.len()
            );
        }
    }
    #[test]
    fn test_google_length_penalty() {
        assert_eq!(google_length_penalty(1, 0.0), 1.0);
        assert_eq!(google_length_penalty(10, 0.0), 1.0);

        let lp1 = google_length_penalty(1, 1.0);
        assert!((lp1 - 1.0).abs() < 1e-5);
        let lp7 = google_length_penalty(7, 1.0);
        assert!((lp7 - 2.0).abs() < 1e-5);

        let lp_pow = google_length_penalty(7, 0.5);
        assert!((lp_pow - 2.0f32.sqrt()).abs() < 1e-5);
    }

    #[test]
    fn test_repetition_penalty_and_ngram_blocking() {
        let shortlist = vec![10, 20, 30, 40, 50];
        let mut logits = vec![2.0, -1.0, 3.0, 0.5, -2.0];
        let tokens = vec![20, 50];

        apply_penalties(&mut logits, &shortlist, &tokens, 2.0, 0);
        // token 20: index 1 in shortlist. logit was -1.0 (< 0) -> multiplied by 2.0 -> -2.0
        assert!((logits[1] - (-2.0)).abs() < 1e-5);
        // token 50: index 4 in shortlist. logit was -2.0 (< 0) -> multiplied by 2.0 -> -4.0
        assert!((logits[4] - (-4.0)).abs() < 1e-5);
        // token 10: not in tokens -> unchanged (2.0)
        assert!((logits[0] - 2.0).abs() < 1e-5);

        // Test positive logit repetition penalty
        let mut logits_pos = vec![4.0, 1.0];
        let shortlist_pos = vec![10, 20];
        let tokens_pos = vec![10];
        apply_penalties(&mut logits_pos, &shortlist_pos, &tokens_pos, 2.0, 0);
        // token 10: index 0. logit was 4.0 (> 0) -> divided by 2.0 -> 2.0
        assert!((logits_pos[0] - 2.0).abs() < 1e-5);

        // Test 3-gram blocking: first repeat receives soft damping (-3.5)
        let shortlist_ngram = vec![1, 2, 3, 4];
        let mut logits_ngram = vec![1.0, 1.0, 1.0, 1.0];
        let tokens_ngram = vec![1, 2, 3, 1, 2];
        apply_penalties(&mut logits_ngram, &shortlist_ngram, &tokens_ngram, 1.0, 3);
        // token 3 is at index 2 in shortlist_ngram -> matched once -> soft damped: 1.0 - 3.5 = -2.5
        assert!((logits_ngram[2] - (-2.5)).abs() < 1e-5);
        // other tokens are untouched
        assert_eq!(logits_ngram[0], 1.0);
        assert_eq!(logits_ngram[1], 1.0);
        assert_eq!(logits_ngram[3], 1.0);

        // Second repeat (loop tendency, match_count >= 2) -> hard blocked with -1e9
        let mut logits_ngram_loop = vec![1.0, 1.0, 1.0, 1.0];
        let tokens_ngram_loop = vec![1, 2, 3, 1, 2, 3, 1, 2];
        apply_penalties(&mut logits_ngram_loop, &shortlist_ngram, &tokens_ngram_loop, 1.0, 3);
        assert!(logits_ngram_loop[2] <= -1e8);
        assert_eq!(logits_ngram_loop[0], 1.0);
        assert_eq!(logits_ngram_loop[1], 1.0);
        assert_eq!(logits_ngram_loop[3], 1.0);
        // Test duplicate token deduplication (penalty applied only once)
        let mut logits_dedup = vec![4.0, -2.0];
        let shortlist_dedup = vec![10, 20];
        let tokens_dedup = vec![10, 20, 10, 20, 10];
        apply_penalties(&mut logits_dedup, &shortlist_dedup, &tokens_dedup, 2.0, 0);
        // token 10: 4.0 * (1/2.0) = 2.0 (applied once, not 4.0 / 8 = 0.5)
        assert!((logits_dedup[0] - 2.0).abs() < 1e-5);
        // token 20: -2.0 * 2.0 = -4.0 (applied once, not -8.0)
        assert!((logits_dedup[1] - (-4.0)).abs() < 1e-5);

        // Test large sequence (> 64 tokens) deduplication
        let mut large_tokens = vec![5; 70];
        large_tokens.push(10);
        let mut logits_large = vec![4.0, 4.0];
        let shortlist_large = vec![5, 10];
        apply_penalties(&mut logits_large, &shortlist_large, &large_tokens, 2.0, 0);
        assert!((logits_large[0] - 2.0).abs() < 1e-5);
        assert!((logits_large[1] - 2.0).abs() < 1e-5);

        // Test 2-gram blocking: first repeat (soft damped) vs second repeat (hard blocked)
        let shortlist_ngram2 = vec![1, 2, 3];
        let mut logits_ngram2 = vec![1.0, 1.0, 1.0];
        let tokens_ngram2 = vec![1, 2, 1];
        apply_penalties(&mut logits_ngram2, &shortlist_ngram2, &tokens_ngram2, 1.0, 2);
        assert!((logits_ngram2[1] - (-2.5)).abs() < 1e-5); // token 2 soft damped
        assert_eq!(logits_ngram2[0], 1.0);
        assert_eq!(logits_ngram2[2], 1.0);

        let mut logits_ngram2_loop = vec![1.0, 1.0, 1.0];
        let tokens_ngram2_loop = vec![1, 2, 1, 2, 1];
        apply_penalties(&mut logits_ngram2_loop, &shortlist_ngram2, &tokens_ngram2_loop, 1.0, 2);
        assert!(logits_ngram2_loop[1] <= -1e8); // token 2 hard blocked (match_count == 2)

        // Test 4-gram blocking: first repeat (soft damped) vs second repeat (hard blocked)
        let shortlist_ngram4 = vec![1, 2, 3, 4, 5];
        let mut logits_ngram4 = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let tokens_ngram4 = vec![1, 2, 3, 4, 1, 2, 3];
        apply_penalties(&mut logits_ngram4, &shortlist_ngram4, &tokens_ngram4, 1.0, 4);
        assert!((logits_ngram4[3] - (-2.5)).abs() < 1e-5); // token 4 soft damped
        assert_eq!(logits_ngram4[0], 1.0);
        assert_eq!(logits_ngram4[1], 1.0);
        assert_eq!(logits_ngram4[2], 1.0);
        assert_eq!(logits_ngram4[4], 1.0);

        let mut logits_ngram4_loop = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let tokens_ngram4_loop = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3];
        apply_penalties(&mut logits_ngram4_loop, &shortlist_ngram4, &tokens_ngram4_loop, 1.0, 4);
        assert!(logits_ngram4_loop[3] <= -1e8); // token 4 hard blocked (match_count == 2)
    }

    #[test]
    fn test_repetition_first_repeat_soft_damping() {
        // Test that first repeat of an n-gram applies soft damping,
        // allowing high-confidence tokens to stay in the positive range (e.g. legitimate rhetoric or 'step by step').
        let shortlist = vec![10, 20, 30];
        let mut logits = vec![6.0, 5.0, 1.0]; // Token 20 has high confidence logit = 5.0
        let tokens = vec![10, 20, 10]; // 2-gram prefix [10], cand 20 matched once
        apply_penalties(&mut logits, &shortlist, &tokens, 1.0, 2);

        // Token 20: 5.0 - 3.5 = 1.5 > 0.0, still in reasonable positive range!
        assert!(logits[1] > 0.0);
        assert!((logits[1] - 1.5).abs() < 1e-5);
        // Token 10 and 30 untouched
        assert_eq!(logits[0], 6.0);
        assert_eq!(logits[2], 1.0);
    }

    #[test]
    fn test_repetition_second_repeat_hard_blocking() {
        // Test that second and subsequent repetitions (match_count >= 2)
        // are strictly hard blocked with -1e9 to terminate infinite autoregressive loops.
        let shortlist = vec![10, 20, 30];
        let mut logits = vec![6.0, 10.0, 1.0]; // Even if token 20 had very high logit = 10.0
        let tokens = vec![10, 20, 10, 20, 10]; // 2-gram prefix [10], cand 20 matched twice
        apply_penalties(&mut logits, &shortlist, &tokens, 1.0, 2);

        // Token 20: strictly hard-blocked with -1e9
        assert!(logits[1] <= -1e8);
        assert_eq!(logits[0], 6.0);
        assert_eq!(logits[2], 1.0);
    }

    #[test]
    fn test_repetition_special_token_exempt() {
        // Test that special tokens (e.g. EOS or punctuation) passed via exempt_tokens
        // are not hard-blocked even when match_count >= 2.
        let shortlist = vec![2, 10, 20];
        let mut logits = vec![4.0, 1.0, 1.0]; // Token 2 is EOS or special token
        let tokens = vec![10, 2, 10, 2, 10]; // 2-gram prefix [10], cand 2 matched twice
        let eos_id = 2u32;
        apply_penalties_with_exemptions(&mut logits, &shortlist, &tokens, 1.0, 2, &[eos_id]);

        // Token 2 is exempt from hard blocking: not -1e9, instead soft damped
        assert!(logits[0] > -1e8);
        // Specifically 4.0 - 3.5 * 2 = -3.0
        assert!((logits[0] - (-3.0)).abs() < 1e-5);

        // Phase 1 repetition penalty must respect exempt_tokens
        let mut logits_rep = vec![5.0, 5.0];
        let shortlist_rep = vec![2, 10];
        let tokens_rep = vec![2, 10];
        apply_penalties_with_exemptions(&mut logits_rep, &shortlist_rep, &tokens_rep, 1.25, 0, &[2]);
        // Token 2 is exempt: logit remains 5.0
        assert_eq!(logits_rep[0], 5.0);
        // Token 10 is not exempt: logit penalized to 5.0 / 1.25 = 4.0
        assert!((logits_rep[1] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_compute_log_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut out = vec![0.0; 3];
        compute_log_softmax(&logits, &mut out);

        let max = 3.0f32;
        let sum_exp = (1.0f32 - max).exp() + (2.0f32 - max).exp() + (3.0f32 - max).exp();
        let expected0 = 1.0 - max - sum_exp.ln();
        let expected1 = 2.0 - max - sum_exp.ln();
        let expected2 = 3.0 - max - sum_exp.ln();
        assert!((out[0] - expected0).abs() < 1e-5);
        assert!((out[1] - expected1).abs() < 1e-5);
        assert!((out[2] - expected2).abs() < 1e-5);
    }

    #[test]
    fn test_beam_search_runtime() {
        let model_dir = std::path::Path::new("data/models/en_zh-Hans");
        if !model_dir.join("model.enzh.intgemm.alphas.bin").exists() {
            return;
        }
        let engine_greedy = TranslationEngine::new(
            "en",
            "zh",
            &model_dir.join("model.enzh.intgemm.alphas.bin"),
            &model_dir.join("lex.50.50.enzh.s2t.bin"),
            &model_dir.join("srcvocab.enzh.spm"),
            &model_dir.join("trgvocab.enzh.spm"),
            EngineConfig {
                beam_size: 1,
                ..EngineConfig::default()
            },
        )
        .expect("load en_zh engine greedy");

        let engine_beam = TranslationEngine::new(
            "en",
            "zh",
            &model_dir.join("model.enzh.intgemm.alphas.bin"),
            &model_dir.join("lex.50.50.enzh.s2t.bin"),
            &model_dir.join("srcvocab.enzh.spm"),
            &model_dir.join("trgvocab.enzh.spm"),
            EngineConfig {
                beam_size: 2,
                ..EngineConfig::default()
            },
        )
        .expect("load en_zh engine beam");

        let text = "Hello world";
        let trans_greedy = engine_greedy.translate(text);
        let trans_beam = engine_beam.translate(text);
        assert!(!trans_greedy.is_empty());
        assert!(!trans_beam.is_empty());
    }

    #[test]
    fn test_dynamic_eos_bias_short_utterance() {
        // Source len 0: must never modify logits
        let mut logits_empty = vec![1.0, 5.0, 2.0];
        apply_dynamic_eos_bias(&mut logits_empty, 1, 0, 0);
        assert_eq!(logits_empty[1], 5.0);

        // For single-token / single-word sentences (e.g. source_len = 1), L_min = clamp(1, 120) = 1, L_soft_max = 11.
        // Step 0 (gen_len = 0): ratio = 1.0 - 0/1 = 1.0; damping = 3.5 * 1.0^2 = 3.5
        let initial_eos_logit = 5.0f32;
        let eos_idx = 1;
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 1, 0);
        assert!((logits[1] - (initial_eos_logit - 3.5)).abs() < 1e-5);

        // Step 1 (gen_len = 1 >= L_min = 1): damping is 0 to allow accurate single-token translation without forced hallucination
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 1, 1);
        assert_eq!(logits[1], initial_eos_logit);

        // For short sentences with source_len = 4, L_min = clamp(2, 120) = 2
        // Step 1 (gen_len = 1 < L_min = 2): ratio = 1.0 - 1/2 = 0.5; damping = 3.5 * 0.25 = 0.875
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 4, 1);
        assert!((logits[1] - (initial_eos_logit - 0.875)).abs() < 1e-5);

        // Step 2 (gen_len = 2 >= L_min): damping smoothly reaches 0
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 4, 2);
        assert_eq!(logits[1], initial_eos_logit);
        // Step 11 (gen_len = 11 == L_soft_max): within normal range, logit unchanged
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 1, 11);
        assert_eq!(logits[1], initial_eos_logit);

        // Step 14 (gen_len = 14 > L_soft_max): gain = 4.0 * min(1.0, (14 - 11) / 30.0) = 4.0 * 0.1 = 0.4
        let mut logits = vec![1.0, initial_eos_logit, 2.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, 1, 14);
        assert!((logits[1] - (initial_eos_logit + 0.4)).abs() < 1e-5);
    }

    #[test]
    fn test_dynamic_eos_bias_long_utterance_soft_damping_and_decay() {
        // For source_len = 10, default factors (expansion_ratio = 1.0, min_factor = 0.5, max_factor = 2.5):
        // L_min = ((10 * 1.0 * 0.5).round() as usize).clamp(2, 120) = 5.
        // L_soft_max = ((10 * 1.0 * 2.5).round() as usize).max(5 + 10) = 25.
        let source_len = 10;
        let initial_eos_logit = 5.0f32;
        let eos_idx = 1;

        // Step 0: ratio = 1.0 - 0/5 = 1.0; damping = 3.5 * 1.0^2 = 3.5
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 0);
        assert!((logits[1] - (initial_eos_logit - 3.5)).abs() < 1e-5);

        // Step 1: ratio = 1.0 - 1/5 = 0.8; damping = 3.5 * 0.64 = 2.24
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 1);
        assert!((logits[1] - (initial_eos_logit - 2.24)).abs() < 1e-5);

        // Step 2: ratio = 1.0 - 2/5 = 0.6; damping = 3.5 * 0.36 = 1.26
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 2);
        assert!((logits[1] - (initial_eos_logit - 1.26)).abs() < 1e-5);

        // Step 3: ratio = 1.0 - 3/5 = 0.4; damping = 3.5 * 0.16 = 0.56
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 3);
        assert!((logits[1] - (initial_eos_logit - 0.56)).abs() < 1e-5);

        // Step 4: ratio = 1.0 - 4/5 = 0.2; damping = 3.5 * 0.04 = 0.14
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 4);
        assert!((logits[1] - (initial_eos_logit - 0.14)).abs() < 1e-5);

        // Step 5: gen_len == L_min (5); damping smoothly hits 0
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 5);
        assert_eq!(logits[1], initial_eos_logit);

        // Step 25: gen_len == L_soft_max (25); normal range
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 25);
        assert_eq!(logits[1], initial_eos_logit);

        // Step 40: gen_len = 40 > L_soft_max (25); delta = 15/30 = 0.5; gain = 4.0 * 0.5 = 2.0
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 40);
        assert!((logits[1] - (initial_eos_logit + 2.0)).abs() < 1e-5);

        // Step 55: gen_len = 55 > L_soft_max (25); delta = 30/30 = 1.0; gain = 4.0 * 1.0 = 4.0
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 55);
        assert!((logits[1] - (initial_eos_logit + 4.0)).abs() < 1e-5);

        // Step 70: gen_len = 70 > L_soft_max (25); delta capped at 1.0; gain = 4.0
        let mut logits = vec![0.0, initial_eos_logit, 0.0];
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 70);
        assert!((logits[1] - (initial_eos_logit + 4.0)).abs() < 1e-5);
    }

    #[test]
    fn test_dynamic_eos_bias_boundary_clamp() {
        let initial_eos_logit = 10.0f32;
        let eos_idx = 0;

        // Lower clamp: source_len = 1, expansion_ratio = 0.5, min_factor = 0.5 -> 0.25, clamped to min 1
        let mut logits = vec![initial_eos_logit];
        apply_adaptive_dynamic_eos_guidance(&mut logits, eos_idx, 1, 0, 0.5, 0.5, 2.5);
        assert!((logits[0] - (initial_eos_logit - 3.5)).abs() < 1e-5);

        let mut logits = vec![initial_eos_logit];
        apply_adaptive_dynamic_eos_guidance(&mut logits, eos_idx, 1, 1, 0.5, 0.5, 2.5);
        assert_eq!(logits[0], initial_eos_logit);

        let mut logits = vec![initial_eos_logit];
        apply_adaptive_dynamic_eos_guidance(&mut logits, eos_idx, 1, 2, 0.5, 0.5, 2.5);
        assert_eq!(logits[0], initial_eos_logit);

        // Upper clamp: source_len = 500, expansion_ratio = 1.0, min_factor = 0.5 -> 250, clamped to max 120
        let source_len = 500;
        let mut logits = vec![initial_eos_logit];
        apply_adaptive_dynamic_eos_guidance(&mut logits, eos_idx, source_len, 119, 1.0, 0.5, 2.5);
        let ratio = 1.0 - 119.0 / 120.0;
        let expected_damping = 3.5 * (ratio * ratio);
        assert!((logits[0] - (initial_eos_logit - expected_damping)).abs() < 1e-5);

        let mut logits = vec![initial_eos_logit];
        apply_adaptive_dynamic_eos_guidance(&mut logits, eos_idx, source_len, 120, 1.0, 0.5, 2.5);
        assert_eq!(logits[0], initial_eos_logit);
    }

    #[test]
    fn test_beam_search_dynamic_eos_bias_candidate_scoring() {
        let shortlist_indices = vec![10u32, 20, 2]; // 2 is eos_id
        let eos_id = 2u32;
        let eos_idx = shortlist_indices.iter().position(|&t| t == eos_id).unwrap();
        let source_len = 10; // L_min = 5

        // Case 1: Early beam step (beam.tokens.len() == 0 < L_min).
        // Model predicts EOS with higher raw logit than non-terminal tokens.
        let mut logits = vec![2.0f32, 2.5, 3.5];
        let mut log_prob_buf = vec![0.0f32; shortlist_indices.len()];

        // Raw log-softmax without dynamic EOS bias would prefer EOS
        compute_log_softmax(&logits, &mut log_prob_buf);
        assert!(log_prob_buf[eos_idx] > log_prob_buf[0]);
        assert!(log_prob_buf[eos_idx] > log_prob_buf[1]);

        // With dynamic EOS bias applied to logit_buf before compute_log_softmax (matching beam search step):
        apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, 0);
        // EOS logit is damped: 3.5 - 3.5 = 0.0
        assert!((logits[eos_idx] - 0.0).abs() < 1e-5);

        compute_log_softmax(&logits, &mut log_prob_buf);
        // Non-terminal token 20 (idx 1) now beats EOS (idx 2)
        assert!(log_prob_buf[1] > log_prob_buf[eos_idx]);

        // Case 2: Sufficient generation length (beam.tokens.len() == 5 >= L_min).
        // Dynamic EOS bias decay reaches 0; EOS is not damped and terminates naturally.
        let mut logits_step5 = vec![2.0f32, 2.5, 3.5];
        apply_dynamic_eos_bias(&mut logits_step5, eos_idx, source_len, 5);
        assert_eq!(logits_step5[eos_idx], 3.5);

        compute_log_softmax(&logits_step5, &mut log_prob_buf);
        assert!(log_prob_buf[eos_idx] > log_prob_buf[1]);
    }

    #[test]
    fn test_adaptive_dynamic_eos_guidance_table_driven() {
        struct TestCase {
            source_len: usize,
            expansion_ratio: f32,
            min_factor: f32,
            max_factor: f32,
            initial_logit: f32,
            step_checks: Vec<(usize, f32)>,
        }

        let cases = vec![
            // zh -> en expansion (expansion_ratio = 1.6)
            TestCase {
                source_len: 20,
                expansion_ratio: 1.6,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 6.0,
                step_checks: vec![
                    (0, 6.0 - 3.5),
                    (15, 6.0 - 3.5 * (1.0 - 15.0 / 16.0f32).powi(2)),
                    (16, 6.0),
                    (80, 6.0),
                    (95, 6.0 + 2.0),
                ],
            },
            // en -> zh compression (expansion_ratio = 0.7)
            TestCase {
                source_len: 30,
                expansion_ratio: 0.7,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 4.0,
                step_checks: vec![
                    (0, 4.0 - 3.5),
                    (11, 4.0),
                    (68, 4.0 + 2.0),
                ],
            },
            // ja -> zh compression smoothing (expansion_ratio = 0.37 < 0.5)
            TestCase {
                source_len: 30,
                expansion_ratio: 0.37,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 5.0,
                step_checks: vec![
                    (0, 5.0 - 3.5),
                    (8, 5.0),
                    (40, 5.0),
                    (55, 5.0 + 2.0),
                ],
            },
            // zh -> ja extreme expansion (expansion_ratio = 2.69 > 2.0)
            TestCase {
                source_len: 20,
                expansion_ratio: 2.69,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 7.0,
                step_checks: vec![
                    (0, 7.0 - 3.5),
                    (23, 7.0),
                    (113, 7.0),
                    (128, 7.0 + 2.0),
                ],
            },
            // short input adaptation (source_len <= 3: min_bound is 1)
            TestCase {
                source_len: 2,
                expansion_ratio: 1.0,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 5.0,
                step_checks: vec![
                    (0, 5.0 - 3.5),
                    (1, 5.0),
                ],
            },
            // short source lower bound check (source_len = 10, raw L_soft_max = 13, lower bound 18)
            TestCase {
                source_len: 10,
                expansion_ratio: 0.37,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 5.0,
                step_checks: vec![
                    (18, 5.0),
                    (33, 5.0 + 2.0),
                ],
            },
            // extremely long inputs (source_len = 400 clamped to 480)
            TestCase {
                source_len: 400,
                expansion_ratio: 1.0,
                min_factor: 0.5,
                max_factor: 2.5,
                initial_logit: 5.0,
                step_checks: vec![
                    (495, 5.0 + 2.0),
                ],
            },
        ];

        let eos_idx = 0;
        for case in cases {
            for (step, expected) in case.step_checks {
                let mut logits = vec![case.initial_logit];
                apply_adaptive_dynamic_eos_guidance(
                    &mut logits,
                    eos_idx,
                    case.source_len,
                    step,
                    case.expansion_ratio,
                    case.min_factor,
                    case.max_factor,
                );
                assert!(
                    (logits[0] - expected).abs() < 1e-5,
                    "Mismatch for ratio {} step {}: expected {}, got {}",
                    case.expansion_ratio,
                    step,
                    expected,
                    logits[0]
                );
            }
        }
    }

    #[test]
    fn test_adaptive_dynamic_eos_guidance_loop_mitigation() {
        let source_len = 10;
        let initial_logit = 1.0f32;
        let eos_idx = 0;

        for step in [26, 30, 40, 55, 80, 200] {
            let mut logits = vec![initial_logit];
            apply_dynamic_eos_bias(&mut logits, eos_idx, source_len, step);
            let expected_delta = ((step - 25) as f32 / 30.0).min(1.0);
            let expected_gain = 4.0 * expected_delta;
            assert!((logits[0] - (initial_logit + expected_gain)).abs() < 1e-5);
            assert!(logits[0] <= initial_logit + 4.0 + 1e-5);
        }
    }
    #[test]
    fn test_ngram_candidate_overflow_hard_blocking_priority() {
        // Build a shortlist of 350 tokens
        let shortlist: Vec<u32> = (0..350).collect();

        // Scenario 1: Unigram repetition prevention (prefix_len = 0).
        // Populate 260 distinct tokens where token 300 appears twice and all others appear once.
        let mut tokens = Vec::with_capacity(300);
        for t in 0..255 {
            tokens.push(t);
        }
        tokens.push(300); // First occurrence of repeating token
        for t in 255..270 {
            tokens.push(t);
        }
        tokens.push(300); // Second occurrence of repeating token

        let mut logits = vec![2.0f32; 350];
        apply_penalties_with_exemptions(&mut logits, &shortlist, &tokens, 1.0, 1, &[]);

        // Token 300 must be hard blocked (-1e9) despite buffer capacity being exceeded by singletons
        assert_eq!(logits[300], -1e9);

        // Singletons that were recorded should only have soft damping
        assert!((logits[0] - (2.0 - 3.5)).abs() < 1e-5);

        // Scenario 2: Bigram repetition prevention (prefix_len = 1).
        // Prefix token 9999 is followed by 255 unique tokens once, and token 310 twice.
        let prefix_tok = 9999u32;
        let mut bigram_tokens = Vec::new();
        for t in 0..255 {
            bigram_tokens.push(prefix_tok);
            bigram_tokens.push(t);
        }
        bigram_tokens.push(prefix_tok);
        bigram_tokens.push(310);
        for t in 255..270 {
            bigram_tokens.push(prefix_tok);
            bigram_tokens.push(t);
        }
        bigram_tokens.push(prefix_tok);
        bigram_tokens.push(310);
        bigram_tokens.push(prefix_tok); // Current tail ending with prefix_tok

        let mut bigram_logits = vec![2.0f32; 350];
        apply_penalties_with_exemptions(
            &mut bigram_logits,
            &shortlist,
            &bigram_tokens,
            1.0,
            2,
            &[],
        );

        // Token 310 must be hard blocked
        assert_eq!(bigram_logits[310], -1e9);
    }

    #[test]
    fn test_cross_attention_multihead_variance_filtering() {
        // kv_len = 5: threshold = 1.2 / 5.0 = 0.24
        let kv_len = 5;

        // Head 0: Sharp alignment head with clear peak at index 1 (score 0.80)
        let scores_sharp = [0.05f32, 0.80, 0.05, 0.05, 0.05];
        // Head 1: Uniform diffuse head with noise spread evenly across all positions (0.20 each)
        let scores_diffuse = [0.20f32, 0.20, 0.20, 0.20, 0.20];

        // 1. Simulate naive simple average accumulation (old unweighted behavior)
        let mut naive_accum = vec![0.0f32; kv_len];
        for j in 0..kv_len {
            naive_accum[j] = scores_sharp[j] + scores_diffuse[j];
        }
        let naive_peak = naive_accum[1]; // 0.80 + 0.20 = 1.00
        let naive_noise = naive_accum[0]; // 0.05 + 0.20 = 0.25
        let naive_snr = naive_peak / naive_noise; // 1.00 / 0.25 = 4.0

        // 2. Multi-head sharpness filtered accumulation
        let mut filtered_accum = vec![0.0f32; kv_len];
        filter_and_accumulate_cross_attention_head(&mut filtered_accum, &scores_sharp, kv_len, 0);
        filter_and_accumulate_cross_attention_head(&mut filtered_accum, &scores_diffuse, kv_len, 1);

        let filtered_peak = filtered_accum[1];
        let filtered_noise = filtered_accum[0];
        let filtered_snr = filtered_peak / filtered_noise;

        // Sharp head max = 0.80 > 0.24, weight = (0.80 - 0.24)^2 = 0.56^2 = 0.3136
        // Diffuse head max = 0.20 <= 0.24, weight = 0.001
        // Peak = 0.80 * 0.3136 + 0.20 * 0.001 = 0.25088 + 0.0002 = 0.25108
        // Noise (idx 0) = 0.05 * 0.3136 + 0.20 * 0.001 = 0.01568 + 0.0002 = 0.01588
        // Filtered SNR = 0.25108 / 0.01588 = 15.811
        assert!(
            filtered_snr > naive_snr * 3.5,
            "Peak-to-noise ratio should be amplified (filtered SNR: {filtered_snr}, naive SNR: {naive_snr})"
        );

        // Diffuse head noise contribution at index 0 should be suppressed by orders of magnitude
        let diffuse_noise_naive = scores_diffuse[0]; // 0.20
        let diffuse_noise_filtered = scores_diffuse[0] * 0.001; // 0.0002
        assert!(diffuse_noise_filtered < diffuse_noise_naive * 0.01);

        // Alignment peak index is unambiguously the sharp head's peak
        let best_idx = filtered_accum
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(best_idx, 1);

        // 3. Test head order invariance: diffuse first, then sharp
        let mut filtered_accum_rev = vec![0.0f32; kv_len];
        filter_and_accumulate_cross_attention_head(&mut filtered_accum_rev, &scores_diffuse, kv_len, 0);
        filter_and_accumulate_cross_attention_head(&mut filtered_accum_rev, &scores_sharp, kv_len, 1);
        for j in 0..kv_len {
            assert!(
                (filtered_accum[j] - filtered_accum_rev[j]).abs() < 1e-6,
                "Accumulation must be independent of head order at position {j}"
            );
        }

        // 4. Edge cases
        // Single token kv_len == 1: threshold is 0.0, weight = (1.0 - 0.0)^2 = 1.0
        let mut single_accum = vec![0.0f32; 1];
        filter_and_accumulate_cross_attention_head(&mut single_accum, &[1.0f32], 1, 0);
        assert!((single_accum[0] - 1.001).abs() < 1e-5);

        // 5. Monotonicity verification around threshold boundary
        let mut mono_base = vec![0.0f32; kv_len];
        filter_and_accumulate_cross_attention_head(&mut mono_base, &[0.24, 0.24, 0.24, 0.24, 0.04], kv_len, 0);
        let mut mono_slight = vec![0.0f32; kv_len];
        filter_and_accumulate_cross_attention_head(&mut mono_slight, &[0.25, 0.24, 0.24, 0.23, 0.04], kv_len, 0);
        assert!(
            mono_slight[0] > mono_base[0],
            "Slightly focused head must have higher weight than diffuse baseline"
        );

        // All diffuse heads: should not zero out or NaN, preserving relative weights with baseline 0.001
        let mut all_diffuse = vec![0.0f32; kv_len];
        filter_and_accumulate_cross_attention_head(&mut all_diffuse, &scores_diffuse, kv_len, 0);
        filter_and_accumulate_cross_attention_head(&mut all_diffuse, &scores_diffuse, kv_len, 1);
        for &val in &all_diffuse[..kv_len] {
            assert!((val - 0.0004).abs() < 1e-6);
        }
    }

    #[test]
    fn test_multi_head_attention_cross_attn_accum_variance_filtering() {
        let heads = 2;
        let dk = 4;
        let d_model = heads * dk;
        let kv_len = 3;
        let q_len = 1;

        let q = Tensor2::from_vec([q_len, d_model], vec![1.0; d_model]);
        let k_t = Tensor2::from_vec([kv_len, d_model], vec![0.0; kv_len * d_model]);
        let v_t = Tensor2::from_vec([kv_len, d_model], vec![0.0; kv_len * d_model]);
        let k = rearrange_k_heads(k_t, heads);
        let v = rearrange_v_heads(v_t, heads);

        let wo_weights = Tensor2::from_vec([d_model, d_model], vec![0.0; d_model * d_model]);
        let wo = LinW::F32(&wo_weights);
        let bo = Tensor2::from_vec([1, d_model], vec![0.0; d_model]);

        let mut out_ctx = Tensor2 { shape: [0, 0], data: vec![] };
        let mut pre_ctx = Tensor2 { shape: [0, 0], data: vec![] };
        let mut out_scores = Tensor2 { shape: [0, 0], data: vec![] };
        let mut accum = vec![0.0f32; kv_len];

        multi_head_attention_qkv_headmajor_into(
            &q,
            &k,
            &v,
            &wo,
            &bo,
            None,
            false,
            &mut out_ctx,
            &mut pre_ctx,
            &mut out_scores,
            Some(&mut accum),
        );

        // With zero K weights, all dot products are 0, softmax is uniform (1/3 each),
        // head_max = 1/3 = 0.3333. Threshold = 1.2 / 3 = 0.40.
        // Since 0.3333 <= 0.40, heads are diffuse, weight = 0.001.
        // For 2 heads, accum should be 2 * (1/3) * 0.001 = 0.0006666...
        for &val in &accum {
            assert!((val - 2.0 * (1.0 / 3.0) * 0.001).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resolve_with_languages_adaptive_generation_rules() {
        use std::path::Path;
        use crate::engine::{EngineConfig, Model};
        use crate::engine::rules::RuleRegistry;

        // Ensure all rule profiles from lang_rules directory are loaded into global registry
        RuleRegistry::global().load_from_dir(Path::new("lang_rules"));

        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // version 1
        buf.extend_from_slice(&0u64.to_le_bytes()); // 0 headers
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset 0
        let model = Model::from_bytes(&buf).expect("minimal test model");

        // 1. Target = Japanese (ja): repetition_penalty = 1.06, no_repeat_ngram_size = 5
        let fwd_ja = ForwardConfig::resolve_with_languages(&model, &EngineConfig::default(), "en", "ja").unwrap();
        assert!((fwd_ja.repetition_penalty - 1.06).abs() < 1e-5);
        assert_eq!(fwd_ja.no_repeat_ngram_size, 5);

        // 2. Target = Korean (ko): repetition_penalty = 1.06, no_repeat_ngram_size = 5
        let fwd_ko = ForwardConfig::resolve_with_languages(&model, &EngineConfig::default(), "en", "ko").unwrap();
        assert!((fwd_ko.repetition_penalty - 1.06).abs() < 1e-5);
        assert_eq!(fwd_ko.no_repeat_ngram_size, 5);

        // 3. Target = Chinese (zh): repetition_penalty = 1.15, no_repeat_ngram_size = 5
        let fwd_zh = ForwardConfig::resolve_with_languages(&model, &EngineConfig::default(), "en", "zh").unwrap();
        assert!((fwd_zh.repetition_penalty - 1.15).abs() < 1e-5);
        assert_eq!(fwd_zh.no_repeat_ngram_size, 5);

        // 4. Target = German (de): repetition_penalty = 1.15, no_repeat_ngram_size = 3
        let fwd_de = ForwardConfig::resolve_with_languages(&model, &EngineConfig::default(), "en", "de").unwrap();
        assert!((fwd_de.repetition_penalty - 1.15).abs() < 1e-5);
        assert_eq!(fwd_de.no_repeat_ngram_size, 3);

        // 5. Explicitly specified repetition_penalty / no_repeat_ngram_size should NOT be overridden
        let mut custom_cfg = EngineConfig::default();
        custom_cfg.repetition_penalty = 1.25;
        custom_cfg.no_repeat_ngram_size = 4;
        let fwd_custom = ForwardConfig::resolve_with_languages(&model, &custom_cfg, "en", "ja").unwrap();
        assert!((fwd_custom.repetition_penalty - 1.25).abs() < 1e-5);
        assert_eq!(fwd_custom.no_repeat_ngram_size, 4);
    }
}
