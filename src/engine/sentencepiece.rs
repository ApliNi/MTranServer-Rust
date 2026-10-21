//! Pure Rust SentencePiece tokenizer.
//!
//! Parses `.spm` protobuf model files and implements unigram/BPE tokenization.
//! No C++ dependencies. Mirrors the behavior of the original SentencePiece
//! library used by Bergamot.

use std::borrow::Cow;
use std::cell::RefCell;
use std::path::Path;
use super::error::EngineError;
/// SentencePiece model type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
enum ModelType {
    Unigram = 1,
    BPE = 2,
    Word = 3,
    Char = 4,
}

/// A vocabulary piece
#[derive(Debug, Clone)]
struct Piece {
    piece: String,
    score: f32,
    piece_type: i32, // SentencePiece proto Type: 1=NORMAL, 2=UNKNOWN, 3=CONTROL, 4=USER_DEFINED, 6=BYTE
    byte_val: Option<u8>,
    /// Bit set of sub-field numbers actually present in this message
    /// (see [`FIELD_PIECE`] and friends). Used to tell real pieces apart
    /// from other messages that share the same field slot.
    fields_seen: u8,
}

const FIELD_PIECE: u8 = 1 << 0;
const FIELD_SCORE: u8 = 1 << 1;
const FIELD_TYPE: u8 = 1 << 2;

/// Prefix tree over the vocabulary pieces, keyed by `char`.
///
/// Replaces `HashMap<String, u32>` lookups in the unigram DP hot path: a
/// probe there slices and hashes the candidate substring for every `(i, j)`
/// pair, while a trie walk advances one `char` at a time with no slicing,
/// hashing, or allocation, and returns every matching piece length in a
/// single pass. Pieces sharing a prefix (morphology, CJK n-grams) collapse
/// into shared nodes.
#[derive(Clone, Copy)]
struct CompactTrieNode {
    child_start: u32,
    child_count: u16,
    /// Vocabulary piece id, or `u32::MAX` if no piece terminates at this node.
    piece_id: u32,
}

impl CompactTrieNode {
    #[inline]
    fn find_child(&self, children: &[(char, u32)], c: char) -> Option<u32> {
        let start = self.child_start as usize;
        let count = self.child_count as usize;
        let slice = children.get(start..start + count)?;
        if count <= 8 {
            for &(ch, idx) in slice {
                if ch == c {
                    return Some(idx);
                }
            }
            None
        } else {
            match slice.binary_search_by_key(&c, |&(ch, _)| ch) {
                Ok(idx) => Some(slice[idx].1),
                Err(_) => None,
            }
        }
    }
}

/// Compact prefix tree over vocabulary pieces, flattened into contiguous arenas.
///
/// Eliminates per-node heap allocations and pointer chasing: all nodes reside in
/// `nodes`, and all edge transitions reside in a single flat `children` slice.
struct PieceTrie {
    nodes: Vec<CompactTrieNode>,
    children: Vec<(char, u32)>,
    root_ascii: [u32; 128],
    space_child: u32,
    space_ascii: [u32; 128],
}

impl PieceTrie {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            nodes: vec![CompactTrieNode {
                child_start: 0,
                child_count: 0,
                piece_id: u32::MAX,
            }],
            children: Vec::new(),
            root_ascii: [u32::MAX; 128],
            space_child: u32::MAX,
            space_ascii: [u32::MAX; 128],
        }
    }

    #[allow(dead_code)]
    #[inline]
    fn builder() -> PieceTrieBuilder {
        PieceTrieBuilder::new()
    }

    #[allow(dead_code)]
    #[inline]
    fn find_child(&self, node_idx: usize, c: char) -> Option<u32> {
        if node_idx == 0 {
            if c == '\u{2581}' {
                return if self.space_child != u32::MAX {
                    Some(self.space_child)
                } else {
                    None
                };
            } else if (c as u32) < 128 {
                let idx = self.root_ascii[c as usize];
                return if idx != u32::MAX { Some(idx) } else { None };
            }
        } else if node_idx == self.space_child as usize && (c as u32) < 128 {
            let idx = self.space_ascii[c as usize];
            return if idx != u32::MAX { Some(idx) } else { None };
        }
        let node = self.nodes.get(node_idx)?;
        node.find_child(&self.children, c)
    }

    /// Exact lookup (used by BPE merges and special tokens).
    #[inline]
    fn get(&self, piece: &str) -> Option<u32> {
        let nodes = self.nodes.as_slice();
        let mut cur_idx = 0usize;
        for c in piece.chars() {
            let next = self.find_child(cur_idx, c)?;
            cur_idx = next as usize;
        }
        let cur_node = nodes.get(cur_idx)?;
        if cur_node.piece_id != u32::MAX {
            Some(cur_node.piece_id)
        } else {
            None
        }
    }

    /// Walk the tree from `text` starting at byte offset `start`, invoking
    /// `on_piece(char_len, vocab_id)` for every piece that ends within
    /// `max_chars` characters. Equivalent to probing the exact substring
    /// `text[start..end]` for every `end` in the window — same match set,
    /// same ids.
    #[inline]
    fn walk_matches(
        &self,
        text: &str,
        start: usize,
        max_chars: usize,
        mut on_piece: impl FnMut(usize, u32),
    ) {
        let bytes = text.as_bytes();
        let nodes = self.nodes.as_slice();
        let children = self.children.as_slice();
        let mut cur_idx = 0usize;
        let mut cur_node = match nodes.first() {
            Some(n) => *n,
            None => return,
        };
        let mut char_len = 0usize;
        let mut off = start;
        while char_len < max_chars && off < bytes.len() {
            // Fast UTF-8 decode without slicing overhead
            let b0 = bytes[off];
            let (c, len) = if b0 < 0x80 {
                (b0 as char, 1)
            } else if (b0 & 0xE0) == 0xC0 && off + 1 < bytes.len() {
                let b1 = bytes[off + 1];
                let cp = (((b0 & 0x1F) as u32) << 6) | ((b1 & 0x3F) as u32);
                (char::from_u32(cp).unwrap_or('\u{FFFD}'), 2)
            } else if (b0 & 0xF0) == 0xE0 && off + 2 < bytes.len() {
                let b1 = bytes[off + 1];
                let b2 = bytes[off + 2];
                let cp = (((b0 & 0x0F) as u32) << 12)
                    | (((b1 & 0x3F) as u32) << 6)
                    | ((b2 & 0x3F) as u32);
                (char::from_u32(cp).unwrap_or('\u{FFFD}'), 3)
            } else if off + 3 < bytes.len() {
                let b1 = bytes[off + 1];
                let b2 = bytes[off + 2];
                let b3 = bytes[off + 3];
                let cp = (((b0 & 0x07) as u32) << 18)
                    | (((b1 & 0x3F) as u32) << 12)
                    | (((b2 & 0x3F) as u32) << 6)
                    | ((b3 & 0x3F) as u32);
                (char::from_u32(cp).unwrap_or('\u{FFFD}'), 4)
            } else {
                ('\u{FFFD}', 1)
            };
            let next = if cur_idx == 0 {
                if c == '\u{2581}' {
                    if self.space_child != u32::MAX {
                        Some(self.space_child)
                    } else {
                        None
                    }
                } else if (c as u32) < 128 {
                    let idx = self.root_ascii[c as usize];
                    if idx != u32::MAX {
                        Some(idx)
                    } else {
                        None
                    }
                } else {
                    cur_node.find_child(children, c)
                }
            } else if cur_idx == self.space_child as usize {
                if (c as u32) < 128 {
                    let idx = self.space_ascii[c as usize];
                    if idx != u32::MAX {
                        Some(idx)
                    } else {
                        None
                    }
                } else {
                    cur_node.find_child(children, c)
                }
            } else {
                cur_node.find_child(children, c)
            };
            match next {
                Some(next_idx) => {
                    let next_node = match nodes.get(next_idx as usize) {
                        Some(n) => *n,
                        None => break,
                    };
                    cur_node = next_node;
                    cur_idx = next_idx as usize;
                    char_len += 1;
                    off += len;
                    if cur_node.piece_id != u32::MAX {
                        on_piece(char_len, cur_node.piece_id);
                    }
                }
                None => break,
            }
        }
    }

    /// Walk the tree from the start of `text`, invoking `on_piece(char_len, vocab_id)`
    /// for every piece matching a prefix of `text`.
    #[allow(dead_code)]
    #[inline]
    fn walk_prefixes(&self, text: &str, on_piece: impl FnMut(usize, u32)) {
        self.walk_matches(text, 0, usize::MAX, on_piece);
    }
}

struct BuilderNode {
    children: Vec<(char, u32)>,
    piece_id: u32,
}

#[derive(Default)]
struct PieceTrieBuilder {
    nodes: Vec<BuilderNode>,
}

impl PieceTrieBuilder {
    fn new() -> Self {
        Self::with_capacity(64)
    }

    fn with_capacity(capacity: usize) -> Self {
        let mut builder = Self {
            nodes: Vec::with_capacity(capacity),
        };
        builder.nodes.push(BuilderNode {
            children: Vec::new(),
            piece_id: u32::MAX,
        });
        builder
    }

    fn insert(&mut self, piece: &str, id: u32) {
        let mut cur = 0usize;
        for c in piece.chars() {
            let next = match self.nodes[cur].children.binary_search_by_key(&c, |&(ch, _)| ch) {
                Ok(pos) => self.nodes[cur].children[pos].1 as usize,
                Err(pos) => {
                    let next_idx = self.nodes.len() as u32;
                    self.nodes.push(BuilderNode {
                        children: Vec::new(),
                        piece_id: u32::MAX,
                    });
                    self.nodes[cur].children.insert(pos, (c, next_idx));
                    next_idx as usize
                }
            };
            cur = next;
        }
        self.nodes[cur].piece_id = id;
    }

    fn build(self) -> PieceTrie {
        let mut root_ascii = [u32::MAX; 128];
        let mut space_child = u32::MAX;
        let mut space_ascii = [u32::MAX; 128];

        if let Some(root) = self.nodes.first() {
            for &(c, idx) in &root.children {
                if (c as u32) < 128 {
                    root_ascii[c as usize] = idx;
                }
                if c == '\u{2581}' {
                    space_child = idx;
                }
            }
        }

        if space_child != u32::MAX {
            if let Some(space_node) = self.nodes.get(space_child as usize) {
                for &(c, idx) in &space_node.children {
                    if (c as u32) < 128 {
                        space_ascii[c as usize] = idx;
                    }
                }
            }
        }

        let mut compact_nodes = Vec::with_capacity(self.nodes.len());
        let total_children = self.nodes.len().saturating_sub(1);
        let mut all_children = Vec::with_capacity(total_children);

        for node in self.nodes {
            let child_start = all_children.len() as u32;
            let child_count = node.children.len() as u16;
            all_children.extend(node.children);
            compact_nodes.push(CompactTrieNode {
                child_start,
                child_count,
                piece_id: node.piece_id,
            });
        }

        PieceTrie {
            nodes: compact_nodes,
            children: all_children,
            root_ascii,
            space_child,
            space_ascii,
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct DpNode {
    pub score: f32,
    pub split: u32,
    pub prev: u32,
}

/// Reusable heap buffers for the encode hot path.
///
/// `translate_internal` keeps one of these in thread-local storage so the
/// DP tables, backtrack list, and output vector live across requests instead
/// of being re-allocated on every call.
#[derive(Default)]
pub struct EncodeScratch {
    /// Sanitized SentencePiece input text
    pub sp_text: String,
    /// Byte offsets for each character in `sp_text`
    pub char_offsets: Vec<usize>,
    /// Viterbi DP: score, split piece id, and back-pointer per position
    pub dp: Vec<DpNode>,
    /// Reusable buffer for backtracked piece end positions
    pub ends: Vec<u32>,
    /// Output token IDs
    pub ids: Vec<u32>,
    /// Byte ranges in the original input text corresponding to each token in `ids`.
    pub token_spans: Vec<std::ops::Range<usize>>,
    /// Byte offsets mapping each character in `sp_text` back to `(start, end)` in original text.
    pub sp_to_orig: Vec<std::ops::Range<usize>>,
    /// Reusable buffer for normalized text
    pub norm_buf: String,
    /// Reusable byte spans mapping normalized characters to original text
    pub norm_to_orig: Vec<std::ops::Range<usize>>,
}

/// Reusable heap buffers for the decode hot path.
#[derive(Default)]
pub struct DecodeScratch {
    /// Buffer for accumulating byte-fallback runs
    pub byte_buf: Vec<u8>,
    /// Byte ranges in the decoded output text corresponding to each token.
    pub token_spans: Vec<std::ops::Range<usize>>,
}

thread_local! {
    static ENCODE_SCRATCH: RefCell<EncodeScratch> = RefCell::new(EncodeScratch::default());
    static DECODE_SCRATCH: RefCell<DecodeScratch> = RefCell::new(DecodeScratch::default());
}

#[derive(Debug, Clone)]
enum DecodePiece {
    /// Skip control and unknown tokens
    Skip,
    /// Byte fallback piece (<0xXX>)
    Byte(u8),
    /// Clean pre-decoded text: normal with leading space if piece had  
    Text {
        normal: String,
        has_leading_space: bool,
    },
}

impl SentencePiece {
    fn compute_decode_pieces(pieces: &[Piece]) -> Vec<DecodePiece> {
        pieces
            .iter()
            .map(|piece| {
                if piece.piece_type == 2 || piece.piece_type == 3 {
                    DecodePiece::Skip
                } else if let Some(b) = piece.byte_val {
                    DecodePiece::Byte(b)
                } else {
                    let has_leading_space = piece.piece.starts_with('\u{2581}');
                    let normal: String = piece
                        .piece
                        .chars()
                        .map(|c| if c == '\u{2581}' { ' ' } else { c })
                        .collect();
                    DecodePiece::Text {
                        normal,
                        has_leading_space,
                    }
                }
            })
            .collect()
    }
}

/// Compact double-array unit for SentencePiece charsmap Trie.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct DoubleArrayUnit(pub u32);

impl DoubleArrayUnit {
    #[inline(always)]
    pub fn has_leaf(&self) -> bool {
        ((self.0 >> 8) & 1) == 1
    }

    #[inline(always)]
    pub fn value(&self) -> u32 {
        self.0 & ((1 << 31) - 1)
    }

    #[inline(always)]
    pub fn label(&self) -> u32 {
        self.0 & ((1 << 31) | 0xFF)
    }

    #[inline(always)]
    pub fn offset(&self) -> u32 {
        (self.0 >> 10) << ((self.0 & (1 << 9)) >> 6)
    }
}

/// Official SentencePiece Normalizer based on precompiled Double-Array Trie (CharsMap).
#[derive(Clone, Debug)]
pub struct Normalizer {
    units: Vec<DoubleArrayUnit>,
    normalized_bytes: Vec<u8>,
    #[allow(dead_code)]
    add_dummy_prefix: bool,
    #[allow(dead_code)]
    remove_extra_whitespaces: bool,
    #[allow(dead_code)]
    escape_whitespaces: bool,
}

impl Normalizer {
    pub fn from_precompiled_charsmap(
        data: &[u8],
        add_dummy_prefix: bool,
        remove_extra_whitespaces: bool,
        escape_whitespaces: bool,
    ) -> Result<Self, EngineError> {
        if data.len() < 4 {
            return Err(EngineError::Format(
                "precompiled_charsmap too short".to_string(),
            ));
        }
        let trie_blob_size = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        let trie_end = 4 + trie_blob_size;
        if trie_end > data.len() || trie_blob_size % 4 != 0 {
            return Err(EngineError::Format(format!(
                "Invalid trie_blob_size: {trie_blob_size} with data len {}",
                data.len()
            )));
        }
        let trie_size = trie_blob_size / 4;
        let mut units = Vec::with_capacity(trie_size);
        for chunk in data[4..trie_end].chunks_exact(4) {
            let u = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            units.push(DoubleArrayUnit(u));
        }
        let normalized_bytes = data[trie_end..].to_vec();

        Ok(Self {
            units,
            normalized_bytes,
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
        })
    }

    #[inline]
    pub fn common_prefix_search(&self, input: &[u8]) -> Option<(usize, usize)> {
        if self.units.is_empty() {
            return None;
        }
        let units = self.units.as_slice();
        let mut node_pos = 0usize;
        let mut unit = *units.get(node_pos)?;
        node_pos ^= unit.offset() as usize;

        let mut longest = None;

        for (i, &b) in input.iter().enumerate() {
            node_pos ^= b as usize;
            let Some(next_unit) = units.get(node_pos) else {
                return longest;
            };
            if next_unit.label() != (b as u32) {
                return longest;
            }
            unit = *next_unit;
            node_pos ^= unit.offset() as usize;
            if unit.has_leaf() {
                if let Some(leaf) = units.get(node_pos) {
                    longest = Some((i + 1, leaf.value() as usize));
                }
            }
        }

        longest
    }

    pub fn normalize<'a>(&self, text: &'a str) -> Cow<'a, str> {
        // Fast path: pure printable ASCII (0x20..=0x7E) contains no matches in nmt_nfkc
        if text.bytes().all(|b| (0x20..=0x7E).contains(&b)) {
            return Cow::Borrowed(text);
        }

        let raw_bytes = text.as_bytes();
        let mut pos = 0;
        let mut out: Option<String> = None;
        let mut last_copied = 0;

        while pos < raw_bytes.len() {
            if let Some((match_len, val)) = self.common_prefix_search(&raw_bytes[pos..]) {
                let null_pos = self.normalized_bytes[val..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|p| val + p)
                    .unwrap_or(self.normalized_bytes.len());
                let repl = &self.normalized_bytes[val..null_pos];

                let buf = out.get_or_insert_with(|| {
                    String::with_capacity(text.len() + 16)
                });
                if pos > last_copied {
                    if let Ok(prefix) = std::str::from_utf8(&raw_bytes[last_copied..pos]) {
                        buf.push_str(prefix);
                    }
                }
                if let Ok(repl_str) = std::str::from_utf8(repl) {
                    buf.push_str(repl_str);
                }
                pos += match_len;
                last_copied = pos;
            } else {
                let b0 = raw_bytes[pos];
                let char_len = if b0 < 0x80 {
                    1
                } else if (b0 & 0xE0) == 0xC0 {
                    2
                } else if (b0 & 0xF0) == 0xE0 {
                    3
                } else {
                    4
                };
                pos += char_len.min(raw_bytes.len() - pos);
            }
        }

        match out {
            None => Cow::Borrowed(text),
            Some(mut buf) => {
                if last_copied < raw_bytes.len() {
                    if let Ok(suffix) = std::str::from_utf8(&raw_bytes[last_copied..]) {
                        buf.push_str(suffix);
                    }
                }
                Cow::Owned(buf)
            }
        }
    }

    pub fn normalize_with_spans_into(
        &self,
        text: &str,
        norm_buf: &mut String,
        norm_to_orig: &mut Vec<std::ops::Range<usize>>,
    ) {
        norm_buf.clear();
        norm_to_orig.clear();
        norm_buf.reserve(text.len() + 16);
        norm_to_orig.reserve(text.len() + 16);

        let raw_bytes = text.as_bytes();
        let mut pos = 0;

        while pos < raw_bytes.len() {
            if let Some((match_len, val)) = self.common_prefix_search(&raw_bytes[pos..]) {
                let orig_span = pos..pos + match_len;
                let null_pos = self.normalized_bytes[val..]
                    .iter()
                    .position(|&b| b == 0)
                    .map(|p| val + p)
                    .unwrap_or(self.normalized_bytes.len());
                let repl = &self.normalized_bytes[val..null_pos];
                if let Ok(repl_str) = std::str::from_utf8(repl) {
                    for c in repl_str.chars() {
                        norm_buf.push(c);
                        norm_to_orig.push(orig_span.clone());
                    }
                }
                pos += match_len;
            } else {
                let b0 = raw_bytes[pos];
                let char_len = if b0 < 0x80 {
                    1
                } else if (b0 & 0xE0) == 0xC0 {
                    2
                } else if (b0 & 0xF0) == 0xE0 {
                    3
                } else {
                    4
                }
                .min(raw_bytes.len() - pos);

                let orig_span = pos..pos + char_len;
                if let Ok(c_str) = std::str::from_utf8(&raw_bytes[pos..pos + char_len]) {
                    for c in c_str.chars() {
                        norm_buf.push(c);
                        norm_to_orig.push(orig_span.clone());
                    }
                }
                pos += char_len;
            }
        }
    }
}

/// SentencePiece tokenizer.
pub struct SentencePiece {
    pieces: Vec<Piece>,
    scores: Vec<f32>,
    decode_pieces: Vec<DecodePiece>,
    model_type: ModelType,
    vocab_size: usize,
    // Prefix tree over the pieces for fast lookup (see [`PieceTrie`]).
    trie: PieceTrie,
    // Longest piece in chars, precomputed at load time to bound the DP window
    max_piece_len: usize,
    // Byte fallback pieces (0x00-0xFF)
    byte_pieces: [u32; 256],
    // Special tokens
    bos_id: u32,
    eos_id: u32,
    unk_id: u32,
    pad_id: u32,
    // Optional official normalizer loaded from SPM model
    normalizer: Option<Normalizer>,
}
impl SentencePiece {
    /// Load a SentencePiece model from a `.spm` file.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let data = std::fs::read(path)
            .map_err(|e| EngineError::Io(format!("Failed to read SPM file: {}", e)))?;
        Self::from_bytes(&data)
    }

    /// Load a SentencePiece model from in-memory protobuf bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, EngineError> {
        // Parse protobuf: sentencepiece_model.ProtoSentencePieceProcessor
        let model = parse_spm_model(data)?;
        let pieces = model.pieces;
        let vocab_size = pieces.len();

        let model_type = match model.trainer_spec.model_type {
            1 => ModelType::Unigram,
            2 => ModelType::BPE,
            3 => ModelType::Word,
            4 => ModelType::Char,
            _ => ModelType::Unigram,
        };

        let mut trie_builder = PieceTrieBuilder::with_capacity(pieces.len());
        let mut byte_pieces = [u32::MAX; 256];
        // Longest piece in chars, accumulated while walking the table below
        // (a separate full pass over ~32k pieces costs a measurable fraction
        // of the load time).
        let mut max_piece_len = 1usize;

        for (i, piece) in pieces.iter().enumerate() {
            let id = i as u32;
            trie_builder.insert(&piece.piece, id);
            max_piece_len = max_piece_len.max(piece.piece.chars().count());

            if let Some(b) = piece.byte_val {
                byte_pieces[b as usize] = id;
            }
        }
        let trie = trie_builder.build();

        let bos_id = trie.get("<s>").unwrap_or(1);
        let eos_id = trie.get("</s>").unwrap_or(2);
        let unk_id = trie.get("<unk>").unwrap_or(0);
        let pad_id = trie.get("<pad>").unwrap_or(0);

        let decode_pieces = Self::compute_decode_pieces(&pieces);
        let scores = pieces.iter().map(|p| p.score).collect();

        let normalizer = model.normalizer;

        Ok(Self {
            pieces,
            scores,
            decode_pieces,
            model_type,
            vocab_size,
            trie,
            max_piece_len,
            byte_pieces,
            bos_id,
            eos_id,
            unk_id,
            pad_id,
            normalizer,
        })
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Get the EOS token id.
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }

    /// Get the UNK token id.
    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// Get the PAD token id.
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }
    /// Get piece text for a given token id.
    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| p.piece.as_str())
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return vec![];
        }

        ENCODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            self.encode_into(text, &mut scratch).to_vec()
        })
    }

    /// Encode text reusing caller-owned scratch buffers.
    pub fn encode_into<'a>(&self, text: &str, scratch: &'a mut EncodeScratch) -> &'a [u32] {
        scratch.ids.clear();
        scratch.norm_to_orig.clear();
        if text.is_empty() {
            return &scratch.ids;
        }
        let norm = match &self.normalizer {
            Some(normalizer) => normalizer.normalize(text),
            None => normalize_nmt_nfkc(text),
        };
        let norm_text = norm.as_ref();
        match self.model_type {
            ModelType::Unigram => self.encode_unigram_into(norm_text, scratch),
            ModelType::BPE => {
                let ids = self.encode_bpe(norm_text);
                scratch.ids.extend(ids);
            }
            _ => self.encode_unigram_into(norm_text, scratch),
        }
        &scratch.ids
    }

    /// Encode text reusing caller-owned scratch buffers, returning both token IDs and
    /// the source text byte ranges `Range<usize>` corresponding to each token.
    pub fn encode_with_spans_into<'a>(
        &self,
        text: &str,
        scratch: &'a mut EncodeScratch,
    ) -> (&'a [u32], &'a [std::ops::Range<usize>]) {
        scratch.ids.clear();
        scratch.token_spans.clear();
        if text.is_empty() {
            return (&scratch.ids, &scratch.token_spans);
        }
        if let Some(normalizer) = &self.normalizer {
            normalizer.normalize_with_spans_into(
                text,
                &mut scratch.norm_buf,
                &mut scratch.norm_to_orig,
            );
            let norm_len = scratch.norm_buf.len();
            let norm_ptr = scratch.norm_buf.as_ptr();
            let norm_text = unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(norm_ptr, norm_len))
            };
            match self.model_type {
                ModelType::Unigram => self.encode_unigram_with_spans_into(norm_text, None, scratch),
                ModelType::BPE => {
                    let ids = self.encode_bpe(norm_text);
                    let text_len = text.len();
                    scratch.ids.extend(ids);
                    scratch.token_spans.resize(scratch.ids.len(), 0..text_len);
                }
                _ => self.encode_unigram_with_spans_into(norm_text, None, scratch),
            }
            scratch.norm_to_orig.clear();
        } else {
            let norm = normalize_nmt_nfkc(text);
            let orig_text = match norm {
                Cow::Borrowed(_) => None,
                Cow::Owned(_) => Some(text),
            };
            let norm_text = norm.as_ref();
            match self.model_type {
                ModelType::Unigram => self.encode_unigram_with_spans_into(norm_text, orig_text, scratch),
                ModelType::BPE => {
                    let ids = self.encode_bpe(norm_text);
                    let text_len = text.len();
                    scratch.ids.extend(ids);
                    scratch.token_spans.resize(scratch.ids.len(), 0..text_len);
                }
                _ => self.encode_unigram_with_spans_into(norm_text, orig_text, scratch),
            }
        }
        (&scratch.ids, &scratch.token_spans)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::with_capacity(ids.len() * 4);
        DECODE_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            self.decode_into(ids, &mut scratch, &mut result);
        });
        result
    }

    /// Decode token IDs into an existing String using scratch buffers.
    pub fn decode_into(&self, ids: &[u32], scratch: &mut DecodeScratch, result: &mut String) {
        result.clear();
        scratch.byte_buf.clear();
        let byte_buf = &mut scratch.byte_buf;

        for &id in ids {
            if id == self.bos_id || id == self.eos_id || id == self.pad_id {
                flush_byte_buf(result, byte_buf);
                continue;
            }

            if let Some(piece) = self.decode_pieces.get(id as usize) {
                match piece {
                    DecodePiece::Skip => {
                        flush_byte_buf(result, byte_buf);
                    }
                    DecodePiece::Byte(b) => {
                        byte_buf.push(*b);
                    }
                    DecodePiece::Text {
                        normal,
                        has_leading_space,
                    } => {
                        flush_byte_buf(result, byte_buf);
                        if result.is_empty() && *has_leading_space {
                            result.push_str(&normal[1..]);
                        } else {
                            result.push_str(normal);
                        }
                    }
                }
            }
        }
        flush_byte_buf(result, byte_buf);
    }

    /// Decode token IDs into an existing String using scratch buffers, returning the
    /// generated text byte ranges `Range<usize>` corresponding to each input token.
    pub fn decode_with_spans_into<'a>(
        &self,
        ids: &[u32],
        scratch: &'a mut DecodeScratch,
        result: &mut String,
    ) -> &'a [std::ops::Range<usize>] {
        result.clear();
        scratch.byte_buf.clear();
        scratch.token_spans.clear();
        let byte_buf = &mut scratch.byte_buf;
        let token_spans = &mut scratch.token_spans;

        for &id in ids {
            let start_len = result.len();
            if id == self.bos_id || id == self.eos_id || id == self.pad_id {
                flush_byte_buf(result, byte_buf);
                token_spans.push(result.len()..result.len());
                continue;
            }

            if let Some(piece) = self.decode_pieces.get(id as usize) {
                match piece {
                    DecodePiece::Skip => {
                        flush_byte_buf(result, byte_buf);
                    }
                    DecodePiece::Byte(b) => {
                        byte_buf.push(*b);
                    }
                    DecodePiece::Text {
                        normal,
                        has_leading_space,
                    } => {
                        flush_byte_buf(result, byte_buf);
                        if result.is_empty() && *has_leading_space {
                            result.push_str(&normal[1..]);
                        } else {
                            result.push_str(normal);
                        }
                    }
                }
            }
            let end_len = result.len();
            token_spans.push(start_len..end_len);
        }
        flush_byte_buf(result, byte_buf);
        let final_len = result.len();
        if let Some(last) = token_spans.last_mut() {
            if last.end < final_len {
                last.end = final_len;
            }
        }
        token_spans
    }

    #[inline]
    fn prepare_sp_text(text: &str, sp_text: &mut String) {
        sp_text.clear();
        let needed_cap = text.len().saturating_add(text.len() / 2).saturating_add(8);
        if sp_text.capacity() < needed_cap {
            sp_text.reserve(needed_cap);
        }
        sp_text.push('\u{2581}');
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut i = 0;
        let mut last_chunk_start = 0;
        while i < len {
            if bytes[i] == b' ' {
                if i > last_chunk_start {
                    // SAFETY: Slicing at an ASCII boundary (b' ') preserves UTF-8 validity.
                    sp_text.push_str(unsafe {
                        std::str::from_utf8_unchecked(&bytes[last_chunk_start..i])
                    });
                }
                sp_text.push('\u{2581}');
                i += 1;
                last_chunk_start = i;
            } else {
                i += 1;
            }
        }
        if last_chunk_start < len {
            // SAFETY: Slicing from an ASCII boundary to end of valid UTF-8 string preserves validity.
            sp_text
                .push_str(unsafe { std::str::from_utf8_unchecked(&bytes[last_chunk_start..len]) });
        }
    }

    #[inline]
    fn prepare_char_offsets(sp_text: &str, char_offsets: &mut Vec<usize>) -> usize {
        char_offsets.clear();
        let bytes = sp_text.as_bytes();
        let b_len = bytes.len();
        if char_offsets.capacity() < b_len + 1 {
            char_offsets.reserve(b_len + 1);
        }
        let mut b_idx = 0;
        while b_idx < b_len {
            char_offsets.push(b_idx);
            let b = bytes[b_idx];
            if b < 0x80 {
                b_idx += 1;
            } else if (b & 0xE0) == 0xC0 {
                b_idx += 2;
            } else if (b & 0xF0) == 0xE0 {
                b_idx += 3;
            } else {
                b_idx += 4;
            }
        }
        char_offsets.push(b_len);
        char_offsets.len() - 1
    }

    #[inline]
    fn init_dp(n: usize, dp: &mut Vec<DpNode>) {
        let old_len = dp.len();
        if old_len < n + 1 {
            dp.resize(
                n + 1,
                DpNode {
                    score: -1e38f32,
                    split: 0,
                    prev: 0,
                },
            );
            for node in &mut dp[..old_len] {
                node.score = -1e38f32;
            }
        } else {
            for node in &mut dp[..=n] {
                node.score = -1e38f32;
            }
        }
        dp[0].score = 0.0;
    }

    #[inline]
    fn run_unigram_dp(
        &self,
        sp_text: &str,
        char_offsets: &[usize],
        dp: &mut [DpNode],
        n: usize,
    ) {
        let max_piece_len = self.max_piece_len;
        for i in 0..n {
            if dp[i].score == -1e38f32 {
                continue;
            }

            let base = dp[i].score;
            let start = char_offsets[i];
            let scores = &self.scores;
            self.trie
                .walk_matches(sp_text, start, max_piece_len, |len, id| {
                    let j = i + len;
                    let new_score = base + scores[id as usize];
                    if new_score > dp[j].score {
                        dp[j].score = new_score;
                        dp[j].split = id;
                        dp[j].prev = i as u32;
                    }
                });

            if dp[i + 1].score == -1e38f32 {
                let unk_score = self.scores.get(self.unk_id as usize).copied().unwrap_or(-10.0);
                let fallback_id = if let Some(c) = sp_text[start..].chars().next() {
                    let mut buf = [0u8; 4];
                    let bytes = c.encode_utf8(&mut buf).as_bytes();
                    if bytes.len() == 1 && self.byte_pieces[bytes[0] as usize] != u32::MAX {
                        self.byte_pieces[bytes[0] as usize]
                    } else {
                        self.unk_id
                    }
                } else {
                    self.unk_id
                };
                dp[i + 1].score = base + unk_score;
                dp[i + 1].split = fallback_id;
                dp[i + 1].prev = i as u32;
            }
        }
    }

    #[inline]
    fn backtrack_ends(dp: &[DpNode], n: usize, ends: &mut Vec<u32>) {
        ends.clear();
        if ends.capacity() < n {
            ends.reserve(n);
        }
        let mut pos = n;
        while pos > 0 {
            ends.push(pos as u32);
            pos = dp[pos].prev as usize;
        }
        ends.reverse();
    }

    /// Scratch-buffered variant of unigram encoding that reuses caller's
    /// heap allocations for DP tables and emits token IDs directly into `ids`
    /// without tracking character spans.
    fn encode_unigram_into(&self, text: &str, s: &mut EncodeScratch) {
        Self::prepare_sp_text(text, &mut s.sp_text);
        let n = Self::prepare_char_offsets(&s.sp_text, &mut s.char_offsets);
        if n == 0 {
            s.ids.clear();
            return;
        }

        Self::init_dp(n, &mut s.dp);
        self.run_unigram_dp(&s.sp_text, &s.char_offsets, &mut s.dp, n);
        Self::backtrack_ends(&s.dp, n, &mut s.ends);

        let ids = &mut s.ids;
        ids.clear();
        if ids.capacity() < s.ends.len() {
            ids.reserve(s.ends.len());
        }

        for &end_u32 in s.ends.iter() {
            let end = end_u32 as usize;
            let start = s.dp[end].prev as usize;
            let id = s.dp[end].split as usize;

            if id == self.unk_id as usize && end - start == 1 {
                if let Some(c) = s.sp_text[s.char_offsets[start]..].chars().next() {
                    if self.byte_fallback_encode_into(c, ids) {
                        continue;
                    }
                }
            }
            if id != self.bos_id as usize {
                let token_id = id as u32;
                if token_id == self.unk_id && ids.last() == Some(&self.unk_id) {
                    // Collapse consecutive unk_id
                } else {
                    ids.push(token_id);
                }
            }
        }
    }

    /// Scratch-buffered unigram encoding that tracks token byte ranges `token_spans`.
    fn encode_unigram_with_spans_into(
        &self,
        text: &str,
        orig_text: Option<&str>,
        s: &mut EncodeScratch,
    ) {
        Self::prepare_sp_text(text, &mut s.sp_text);

        let sp_to_orig = &mut s.sp_to_orig;
        sp_to_orig.clear();
        let max_chars = text.len() + 1;
        if sp_to_orig.capacity() < max_chars {
            sp_to_orig.reserve(max_chars);
        }
        sp_to_orig.push(0..0);

        if !s.norm_to_orig.is_empty() {
            for i in 0..s.norm_to_orig.len() {
                sp_to_orig.push(s.norm_to_orig[i].clone());
            }
        } else {
            match orig_text {
                Some(orig) => {
                    for (byte_idx, c) in orig.char_indices() {
                        if map_nmt_nfkc_char(c).is_some() {
                            sp_to_orig.push(byte_idx..byte_idx + c.len_utf8());
                        }
                    }
                }
                None => {
                    for (byte_idx, c) in text.char_indices() {
                        sp_to_orig.push(byte_idx..byte_idx + c.len_utf8());
                    }
                }
            }
        }

        let n = Self::prepare_char_offsets(&s.sp_text, &mut s.char_offsets);
        if n == 0 {
            s.ids.clear();
            s.token_spans.clear();
            return;
        }

        Self::init_dp(n, &mut s.dp);
        self.run_unigram_dp(&s.sp_text, &s.char_offsets, &mut s.dp, n);
        Self::backtrack_ends(&s.dp, n, &mut s.ends);

        let ids = &mut s.ids;
        ids.clear();
        if ids.capacity() < s.ends.len() {
            ids.reserve(s.ends.len());
        }

        let token_spans = &mut s.token_spans;
        token_spans.clear();
        if token_spans.capacity() < s.ends.len() {
            token_spans.reserve(s.ends.len());
        }

        for &end_u32 in s.ends.iter() {
            let end = end_u32 as usize;
            let start = s.dp[end].prev as usize;
            let id = s.dp[end].split as usize;

            let orig_start = s.sp_to_orig.get(start).map(|r| r.start).unwrap_or(0);
            let orig_end = if end > 0 {
                s.sp_to_orig.get(end - 1).map(|r| r.end).unwrap_or(orig_start)
            } else {
                orig_start
            };
            let span = orig_start..orig_end;

            if id == self.unk_id as usize && end - start == 1 {
                let prev_len = ids.len();
                if let Some(c) = s.sp_text[s.char_offsets[start]..].chars().next() {
                    if self.byte_fallback_encode_into(c, ids) {
                        let added = ids.len() - prev_len;
                        for _ in 0..added {
                            token_spans.push(span.clone());
                        }
                        continue;
                    }
                }
            }
            if id != self.bos_id as usize {
                let token_id = id as u32;
                if token_id == self.unk_id && ids.last() == Some(&self.unk_id) {
                    if let Some(last_span) = token_spans.last_mut() {
                        last_span.end = span.end;
                    }
                } else {
                    ids.push(token_id);
                    token_spans.push(span);
                }
            }
        }
    }

    /// Encode a character that has no literal piece using byte-fallback.
    ///
    /// Appends the token IDs for the character's `<0xXX>` byte pieces to `out`,
    /// or returns `false` if the vocabulary lacks the byte pieces (then UNK is used).
    fn byte_fallback_encode_into(&self, c: char, out: &mut Vec<u32>) -> bool {
        let mut buf = [0u8; 4];
        let bytes = c.encode_utf8(&mut buf).as_bytes();
        if bytes.is_empty() {
            return false;
        }
        let mut piece_ids = [0u32; 4];
        for (i, &b) in bytes.iter().enumerate() {
            let id = self.byte_pieces[b as usize];
            if id == u32::MAX {
                return false;
            }
            piece_ids[i] = id;
        }
        out.extend_from_slice(&piece_ids[..bytes.len()]);
        true
    }

    /// Encode a character that has no literal piece using byte-fallback.
    ///
    /// Returns the token IDs for the character's `<0xXX>` byte pieces, or
    /// `None` if the vocabulary lacks the byte pieces (then UNK is used).
    #[inline]
    pub fn byte_fallback_encode(&self, c: char) -> Option<Vec<u32>> {
        let mut ids = Vec::new();
        if self.byte_fallback_encode_into(c, &mut ids) {
            Some(ids)
        } else {
            None
        }
    }

    /// BPE model encoding.
    fn encode_bpe(&self, text: &str) -> Vec<u32> {
        // Simple BPE: split by space, encode each word with BPE merges
        // This is a simplified implementation
        let sp_text = format!("▁{}", text.replace(' ', "▁"));

        // Start with character-level tokenization
        let mut tokens: Vec<String> = sp_text.chars().map(|c| c.to_string()).collect();

        // Greedy BPE merge: find the best pair and merge
        let mut changed = true;
        while changed {
            changed = false;
            let mut best_pair: Option<(usize, String)> = None;
            let mut best_score = f32::NEG_INFINITY;

            for i in 0..tokens.len().saturating_sub(1) {
                let merged = format!("{}{}", tokens[i], tokens[i + 1]);
                if let Some(id) = self.trie.get(&merged) {
                    let score = self.pieces[id as usize].score;
                    if score > best_score {
                        best_score = score;
                        best_pair = Some((i, merged));
                    }
                }
            }

            if let Some((idx, merged)) = best_pair {
                tokens[idx] = merged;
                tokens.remove(idx + 1);
                changed = true;
            }
        }

        let mut ids = Vec::new();
        for token in &tokens {
            if let Some(id) = self.trie.get(token) {
                ids.push(id);
            } else {
                ids.push(self.unk_id);
            }
        }

        ids
    }
}
/// Standard lightweight NFKC normalization for NMT (SentencePiece style).
///
/// - Fullwidth ASCII (`！`..=`～`, `0xFF01`..=`0xFF5E`) -> halfwidth ASCII (`!`..=`~`, `0x21`..=`0x7E`).
/// - Fullwidth space (`\u{3000}`), NBSP (`\u{00A0}`), and other Unicode whitespace -> regular ASCII space (`0x20`).
/// - Soft hyphen (`\u{00AD}`) and zero-width control characters -> filtered out.
/// - Pure ASCII text is zero-copy borrowed (`Cow::Borrowed`).
pub fn normalize_nmt_nfkc(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        return Cow::Borrowed(text);
    }
    for (i, c) in text.char_indices() {
        if is_nmt_nfkc_special(c) {
            let mut out = String::with_capacity(text.len());
            out.push_str(&text[..i]);
            for c in text[i..].chars() {
                if let Some(mapped) = map_nmt_nfkc_char(c) {
                    out.push(mapped);
                }
            }
            return Cow::Owned(out);
        }
    }
    Cow::Borrowed(text)
}

#[inline]
fn is_nmt_nfkc_special(c: char) -> bool {
    matches!(
        c,
        '\u{FF01}'..='\u{FF5E}'
            | '\u{00AD}'
            | '\u{3000}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{200B}'..='\u{200F}'
            | '\u{202A}'..='\u{202E}'
            | '\u{2060}'..='\u{2064}'
            | '\u{2066}'..='\u{2069}'
            | '\u{FEFF}'
    )
}

#[inline]
fn map_nmt_nfkc_char(c: char) -> Option<char> {
    match c {
        // Fullwidth ASCII -> Halfwidth ASCII
        '\u{FF01}'..='\u{FF5E}' => {
            char::from_u32(c as u32 - 0xFF01 + 0x21)
        }
        // Unicode whitespace -> regular ASCII space
        '\u{3000}'
        | '\u{00A0}'
        | '\u{1680}'
        | '\u{2000}'..='\u{200A}'
        | '\u{202F}'
        | '\u{205F}' => Some(' '),
        // Soft hyphen and zero-width control characters -> filter
        '\u{00AD}'
        | '\u{200B}'..='\u{200F}'
        | '\u{202A}'..='\u{202E}'
        | '\u{2060}'..='\u{2064}'
        | '\u{2066}'..='\u{2069}'
        | '\u{FEFF}' => None,
        // All other characters remain untouched
        other => Some(other),
    }
}

/// Flush accumulated byte-fallback bytes as UTF-8.
///
/// Byte runs originate from whole source characters. Valid UTF-8 sequences are
/// decoded directly, and invalid bytes are converted to U+FFFD according to
/// standard UTF-8 replacement rules.
#[inline(always)]
fn flush_byte_buf(result: &mut String, buf: &mut Vec<u8>) {
    if !buf.is_empty() {
        flush_byte_buf_cold(result, buf);
    }
}

#[cold]
fn flush_byte_buf_cold(result: &mut String, buf: &mut Vec<u8>) {
    if buf.is_empty() {
        return;
    }
    match std::str::from_utf8(buf) {
        // Whole run is valid — the common path, no per-byte scan.
        Ok(text) => result.push_str(text),
        Err(e) => {
            let (valid, after_valid) = buf.split_at(e.valid_up_to());
            // SAFETY: `valid_up_to` marks a verified UTF-8 boundary.
            result.push_str(std::str::from_utf8(valid).unwrap());
            let mut rest = after_valid;
            while !rest.is_empty() {
                match std::str::from_utf8(rest) {
                    Ok(text) => {
                        result.push_str(text);
                        break;
                    }
                    Err(e) => {
                        // Decode the valid prefix before the bad block: an
                        // error skip must never swallow decodable data.
                        let (valid, after_valid) = rest.split_at(e.valid_up_to());
                        // SAFETY: `valid_up_to` marks a verified UTF-8 boundary.
                        result.push_str(std::str::from_utf8(valid).unwrap());
                        rest = after_valid;
                        // One replacement char per invalid byte of the
                        // maximal bad block (sentencepiece semantics), then
                        // resume decoding after it. A truncated lead at the
                        // end consumes the remaining bytes.
                        let skip = e.error_len().unwrap_or(rest.len());
                        for _ in 0..skip {
                            result.push('\u{FFFD}');
                        }
                        rest = &rest[skip..];
                    }
                }
            }
        }
    }
    // Clear in place so the allocation is reused across calls.
    buf.clear();
}

// ---------------------------------------------------------------------------
// Protobuf parsing – minimal implementation for SentencePiece model format
// ---------------------------------------------------------------------------

/// Parsed trainer spec from the SentencePiece model protobuf.
#[derive(Debug, Default)]
struct TrainerSpec {
    model_type: i32,
}

/// Parsed SentencePiece model.
#[derive(Debug)]
struct SpmModel {
    pieces: Vec<Piece>,
    trainer_spec: TrainerSpec,
    normalizer: Option<Normalizer>,
}

#[derive(Default)]
struct NormalizerSpecProto {
    name: String,
    precompiled_charsmap: Vec<u8>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

fn parse_normalizer_spec(data: &[u8]) -> Result<NormalizerSpecProto, EngineError> {
    let mut spec = NormalizerSpecProto {
        name: String::new(),
        precompiled_charsmap: Vec::new(),
        add_dummy_prefix: true,
        remove_extra_whitespaces: true,
        escape_whitespaces: true,
    };
    let mut pos = 0;
    while pos < data.len() {
        let (tag, tag_len) = decode_varint(&data[pos..]);
        if tag_len == 0 {
            return Err(EngineError::Format(
                "Unexpected EOF reading normalizer_spec tag".to_string(),
            ));
        }
        pos += tag_len;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;

        match wire_type {
            0 => {
                let (val, len) = decode_varint(&data[pos..]);
                if len == 0 {
                    return Err(EngineError::Format(
                        "Unexpected EOF reading normalizer_spec varint".to_string(),
                    ));
                }
                pos += len;
                match field_number {
                    3 => spec.add_dummy_prefix = val != 0,
                    4 => spec.remove_extra_whitespaces = val != 0,
                    5 => spec.escape_whitespaces = val != 0,
                    _ => {}
                }
            }
            1 => pos += 8,
            2 => {
                let (len, len_len) = decode_varint(&data[pos..]);
                if len_len == 0 {
                    return Err(EngineError::Format(
                        "Unexpected EOF reading normalizer_spec length".to_string(),
                    ));
                }
                pos += len_len;
                let len = len as usize;
                if pos.checked_add(len).map_or(true, |end| end > data.len()) {
                    return Err(EngineError::Format(
                        "Truncated field in normalizer_spec".to_string(),
                    ));
                }
                let f_data = &data[pos..pos + len];
                pos += len;
                match field_number {
                    1 => spec.name = String::from_utf8_lossy(f_data).to_string(),
                    2 => spec.precompiled_charsmap = f_data.to_vec(),
                    _ => {}
                }
            }
            5 => pos += 4,
            _ => break,
        }
    }
    Ok(spec)
}

/// Minimal protobuf parser for SentencePiece model format.
///
/// The SentencePiece model protobuf has this structure:
/// ```protobuf
/// message TrainerSpec { ... }
/// message NormalizerSpec { ... }
/// message SelfTestData { ... }
/// message ModelProto {
///   TrainerSpec trainer_spec = 1;
///   NormalizerSpec normalizer_spec = 2;
///   SelfTestData self_test_data = 3;
///   repeated SentencePiece pieces = 4;
///   message SentencePiece {
///     bytes piece = 1;
///     float score = 2;
///     enum Type { NORMAL=1; UNKNOWN=2; CONTROL=3; USER_DEFINED=4; BYTE=6; }
///     Type type = 3;
///   }
/// }
/// ```
fn parse_spm_model(data: &[u8]) -> Result<SpmModel, EngineError> {
    let mut pieces = Vec::with_capacity(32768);
    let mut trainer_spec = TrainerSpec::default();
    let mut normalizer = None;
    // Parse the top-level protobuf fields
    let mut pos = 0;
    while pos < data.len() {
        let (tag, tag_len) = decode_varint(&data[pos..]);
        if tag_len == 0 {
            break;
        }
        pos += tag_len;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;

        match wire_type {
            0 => {
                // Varint
                let (value, len) = decode_varint(&data[pos..]);
                if len == 0 {
                    break;
                }
                pos += len;
                if field_number == 1 {
                    // trainer_spec.model_type field (varint type)
                    trainer_spec.model_type = value as i32;
                }
            }
            1 => {
                // 64-bit (fixed64) — skip
                pos += 8;
            }
            2 => {
                // Length-delimited
                let (len, len_len) = decode_varint(&data[pos..]);
                if len_len == 0 {
                    return Err(EngineError::Format(format!(
                        "Truncated length prefix for field {} at offset {pos}",
                        field_number
                    )));
                }
                pos += len_len;
                let len = len as usize;
                if pos.checked_add(len).map_or(true, |end| end > data.len()) {
                    return Err(EngineError::Format(format!(
                        "Truncated length-delimited field {} at offset {pos} (len {len})",
                        field_number
                    )));
                }
                let field_data = &data[pos..pos + len];
                pos += len;

                match field_number {
                    1 => {
                        // Bergamot .spm files serialize SentencePiece entries at
                        // field 1 (each entry is a sub-message with field 1=str
                        // piece, field 2=fixed32 score, field 3=varint type)
                        // rather than at the standard field 4. Distinguish the two
                        // by which sub-field numbers were actually consumed instead
                        // of guessing from content shape.
                        let candidate = parse_sentence_piece(field_data);
                        if candidate.fields_seen & (FIELD_PIECE | FIELD_SCORE | FIELD_TYPE) != 0 {
                            pieces.push(candidate);
                        } else {
                            trainer_spec = parse_trainer_spec(field_data)?;
                        }
                    }
                    2 => {
                        // TrainerSpec (standard field number)
                        trainer_spec = parse_trainer_spec(field_data)?;
                    }
                    3 => {
                        // NormalizerSpec (standard field number)
                        let norm_spec = parse_normalizer_spec(field_data)?;
                        if !norm_spec.precompiled_charsmap.is_empty() {
                            normalizer = Normalizer::from_precompiled_charsmap(
                                &norm_spec.precompiled_charsmap,
                                norm_spec.add_dummy_prefix,
                                norm_spec.remove_extra_whitespaces,
                                norm_spec.escape_whitespaces,
                            )
                            .ok();
                        }
                    }
                    4 => {
                        // SentencePiece pieces (standard field number)
                        let piece = parse_sentence_piece(field_data);
                        pieces.push(piece);
                    }
                    _ => {}
                }
            }
            3 => {
                // Start group — skip
                let mut depth = 1;
                while depth > 0 && pos < data.len() {
                    let wt = data[pos] & 0x07;
                    let _fn_ = (data[pos] >> 3) as u32;
                    if wt == 3 {
                        // start group
                        depth += 1;
                    } else if wt == 4 {
                        // end group
                        depth -= 1;
                    }
                    pos += 1;
                    // skip value
                    match wt {
                        0 => {
                            let (_, l) = decode_varint(&data[pos..]);
                            pos += l;
                        }
                        1 => pos += 8,
                        2 => {
                            let (l, ll) = decode_varint(&data[pos..]);
                            let l = l as usize;
                            pos += ll + l;
                        }
                        _ => {}
                    }
                }
            }
            4 => {
                // End group — skip
            }
            5 => {
                // 32-bit (fixed32) — skip
                pos += 4;
            }
            _ => {
                return Err(EngineError::Format(format!(
                    "Unknown wire type {}",
                    wire_type
                )))
            }
        }
    }

    Ok(SpmModel {
        pieces,
        trainer_spec,
        normalizer,
    })
}

fn parse_trainer_spec(data: &[u8]) -> Result<TrainerSpec, EngineError> {
    let mut spec = TrainerSpec::default();
    let mut pos = 0;
    while pos < data.len() {
        let (tag, tag_len) = decode_varint(&data[pos..]);
        if tag_len == 0 {
            return Err(EngineError::Format(
                "Unexpected EOF reading trainer_spec tag".to_string(),
            ));
        }
        pos += tag_len;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;

        match wire_type {
            0 => {
                let (value, len) = decode_varint(&data[pos..]);
                if len == 0 {
                    return Err(EngineError::Format(
                        "Unexpected EOF reading trainer_spec varint".to_string(),
                    ));
                }
                pos += len;
                if field_number == 3 {
                    // type field in TrainerSpec
                    spec.model_type = value as i32;
                }
            }
            2 => {
                let (len, len_len) = decode_varint(&data[pos..]);
                if len_len == 0 {
                    return Err(EngineError::Format(
                        "Unexpected EOF reading trainer_spec length".to_string(),
                    ));
                }
                pos += len_len;
                let len = len as usize;
                if pos.checked_add(len).map_or(true, |end| end > data.len()) {
                    return Err(EngineError::Format(
                        "Truncated field in trainer_spec".to_string(),
                    ));
                }
                pos += len;
            }
            _ => {
                // skip other wire types
                match wire_type {
                    1 => pos += 8,
                    5 => pos += 4,
                    _ => {
                        return Err(EngineError::Format(format!(
                            "Unknown wire type {} in trainer_spec",
                            wire_type
                        )))
                    }
                }
            }
        }
    }
    Ok(spec)
}

fn parse_sentence_piece(data: &[u8]) -> Piece {
    let mut piece = String::new();
    let mut score = 0.0f32;
    let mut piece_type = 0i32;
    let mut fields_seen = 0u8;

    let mut pos = 0;
    while pos < data.len() {
        let (tag, tag_len) = decode_varint(&data[pos..]);
        if tag_len == 0 {
            break;
        }
        pos += tag_len;
        let wire_type = (tag & 0x07) as u8;
        let field_number = (tag >> 3) as u32;

        match wire_type {
            0 => {
                let (value, len) = decode_varint(&data[pos..]);
                if len == 0 {
                    break;
                }
                pos += len;
                if field_number == 3 {
                    // type
                    piece_type = value as i32;
                    fields_seen |= FIELD_TYPE;
                }
            }
            2 => {
                let (len, len_len) = decode_varint(&data[pos..]);
                if len_len == 0 {
                    break;
                }
                pos += len_len;
                let len = len as usize;
                if pos.checked_add(len).map_or(true, |end| end > data.len()) {
                    break; // truncated sub-message
                }
                let str_data = &data[pos..pos + len];
                pos += len;
                if field_number == 1 {
                    // piece (string). `from_utf8_lossy(..).to_string()` would
                    // allocate twice for invalid UTF-8 (Cow::Owned + to_string);
                    // this form allocates exactly once per piece and keeps the
                    // same replacement-char semantics.
                    piece = match std::str::from_utf8(str_data) {
                        Ok(s) => s.to_string(),
                        Err(_) => String::from_utf8_lossy(str_data).into_owned(),
                    };
                    fields_seen |= FIELD_PIECE;
                }
            }
            5 => {
                // fixed32 for float score
                let mut bytes = [0u8; 4];
                if pos + 4 <= data.len() {
                    bytes.copy_from_slice(&data[pos..pos + 4]);
                    pos += 4;
                } else {
                    break; // truncated fixed32
                }
                if field_number == 2 {
                    score = f32::from_le_bytes(bytes);
                    fields_seen |= FIELD_SCORE;
                }
            }
            _ => match wire_type {
                1 => pos += 8,
                _ => break,
            },
        }
    }
    let is_unk = piece == "<unk>";
    let piece_type = if piece_type == 0 && is_unk {
        2
    } else {
        piece_type
    };
    let byte_val = if piece_type == 6 && piece.starts_with("<0x") && piece.len() >= 5 {
        u8::from_str_radix(&piece[3..5], 16).ok()
    } else {
        None
    };
    Piece {
        piece,
        score,
        piece_type,
        byte_val,
        fields_seen,
    }
}

fn decode_varint(data: &[u8]) -> (u64, usize) {
    let mut value = 0u64;
    let mut shift = 0u32;
    let mut len = 0;

    for &byte in data {
        len += 1;
        // A varint never legitimately exceeds 10 bytes; stop before the
        // shift would overflow on malformed input.
        if shift >= 64 {
            break;
        }
        value |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }

    (value, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flush(bytes: &[u8]) -> String {
        let mut out = String::new();
        let mut buf = bytes.to_vec();
        flush_byte_buf(&mut out, &mut buf);
        assert!(buf.is_empty(), "buffer must be cleared");
        out
    }

    #[test]
    fn valid_run_passes_through() {
        assert_eq!(flush("中文 abc".as_bytes()), "中文 abc");
    }

    #[test]
    fn invalid_bytes_between_text_map_per_byte() {
        assert_eq!(flush(b"A\xFFB"), "A\u{FFFD}B");
        assert_eq!(flush(b"A\xFF\xC3\x28B"), "A\u{FFFD}\u{FFFD}(B");
    }

    #[test]
    fn valid_data_between_errors_survives() {
        assert_eq!(flush(b"\xFF\xE4\xB8\xAD\xFF"), "\u{FFFD}中\u{FFFD}");
        assert_eq!(flush(b"\xFF\x41\xE4"), "\u{FFFD}A\u{FFFD}");
        assert_eq!(flush(b"\xE9\x99\x94\xFF\xE4\xB8\xAD"), "陔\u{FFFD}中");
    }

    #[test]
    fn trailing_invalid_bytes_map_to_replacement() {
        assert_eq!(flush("中".as_bytes()), "中");
        assert_eq!(flush(b"\xE4\xB8\xAD\x90"), "中\u{FFFD}");
    }

    #[test]
    fn buffer_is_reusable_after_flush() {
        let mut out = String::new();
        let mut buf = b"\xFF".to_vec();
        flush_byte_buf(&mut out, &mut buf);
        buf.extend_from_slice("中".as_bytes());
        flush_byte_buf(&mut out, &mut buf);
        assert_eq!(out, "\u{FFFD}中");
    }

    #[test]
    fn decode_byte_fallback_tokens() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<0x21>".to_string(),
                    score: -5.0,
                    piece_type: 6,
                    byte_val: Some(b'!'),
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-1.0; 32],
            decode_pieces: SentencePiece::compute_decode_pieces(&[
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<0x21>".to_string(),
                    score: -5.0,
                    piece_type: 6,
                    byte_val: Some(b'!'),
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ]),
            model_type: ModelType::Unigram,
            vocab_size: 5,
            trie: PieceTrie::new(),
            max_piece_len: 6,
            byte_pieces: {
                let mut bp = [u32::MAX; 256];
                bp[b'!' as usize] = 4;
                bp
            },
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let decoded = sp.decode(&[3, 4]);
        assert_eq!(decoded, "hello!");
        let fallback = sp.byte_fallback_encode('!');
        assert_eq!(fallback, Some(vec![4]));
        let no_fallback = sp.byte_fallback_encode('?');
        assert_eq!(no_fallback, None);
        let mut decode_scratch = DecodeScratch::default();
        let mut decoded_into = String::new();
        sp.decode_into(&[3, 4], &mut decode_scratch, &mut decoded_into);
        assert_eq!(decoded_into, "hello!");
    }

    #[test]
    fn encode_into_retains_capacity() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁a".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-10.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            decode_pieces: vec![],
            model_type: ModelType::Unigram,
            vocab_size: 4,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("▁a", 3);
                t.build()
            },
            max_piece_len: 2,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let mut scratch = EncodeScratch::default();
        scratch.ids.reserve(64);
        let cap_before = scratch.ids.capacity();
        assert!(cap_before >= 64);

        let slice = sp.encode_into("a", &mut scratch);
        assert_eq!(slice, &[3]);
        assert_eq!(scratch.ids.capacity(), cap_before);

        let mut prefix_matches = Vec::new();
        sp.trie.walk_prefixes("▁a", |len, id| {
            prefix_matches.push((len, id));
        });
        assert_eq!(prefix_matches, vec![(2, 3)]);
    }
    #[test]
    fn test_encode_decode_spans_exact() {
        let path = std::path::Path::new("data/models/en_zh-Hans/srcvocab.enzh.spm");
        if !path.exists() {
            return;
        }
        let sp = SentencePiece::load(path).unwrap();
        let text = "Please click our website for more details.";
        let mut enc_scratch = EncodeScratch::default();
        let (ids, spans) = sp.encode_with_spans_into(text, &mut enc_scratch);
        assert_eq!(ids.len(), spans.len());
        for span in spans {
            assert!(span.start <= span.end);
            assert!(span.end <= text.len());
        }

        let mut dec_scratch = DecodeScratch::default();
        let mut result = String::new();
        let dec_spans = sp.decode_with_spans_into(ids, &mut dec_scratch, &mut result);
        assert_eq!(ids.len(), dec_spans.len());
        for span in dec_spans {
            assert!(span.start <= span.end);
            assert!(span.end <= result.len());
        }
    }

    #[test]
    fn test_piece_trie_ascii_and_space_fast_path() {
        let mut builder = PieceTrieBuilder::new();
        builder.insert("\u{2581}hello", 10);
        builder.insert("\u{2581}world", 11);
        builder.insert("hello", 12);
        builder.insert("world", 13);
        builder.insert("©copy", 14);
        let trie = builder.build();

        assert_ne!(trie.space_child, u32::MAX);
        assert_eq!(trie.get("\u{2581}hello"), Some(10));
        assert_eq!(trie.get("\u{2581}world"), Some(11));
        assert_eq!(trie.get("hello"), Some(12));
        assert_eq!(trie.get("world"), Some(13));
        assert_eq!(trie.get("©copy"), Some(14));
        assert_eq!(trie.get("notfound"), None);

        // find_child at root
        let space_node = trie.find_child(0, '\u{2581}').unwrap();
        assert_eq!(space_node, trie.space_child);

        let h_node = trie.find_child(0, 'h').unwrap();
        assert_eq!(trie.root_ascii[b'h' as usize], h_node);

        // find_child at space_child
        let h_after_space = trie.find_child(space_node as usize, 'h').unwrap();
        assert_eq!(trie.space_ascii[b'h' as usize], h_after_space);

        // non-ascii
        let copy_node = trie.find_child(0, '©').unwrap();
        assert_ne!(copy_node, u32::MAX);
    }

    #[test]
    fn test_encode_into_skips_spans() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-1.0; 32],
            decode_pieces: SentencePiece::compute_decode_pieces(&[
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ]),
            model_type: ModelType::Unigram,
            vocab_size: 5,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("\u{2581}hello", 3);
                t.insert("\u{2581}world", 4);
                t.build()
            },
            max_piece_len: 6,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let mut scratch_fast = EncodeScratch::default();
        let ids_fast = sp.encode_into("hello world", &mut scratch_fast);
        assert_eq!(ids_fast, &[3, 4]);
        assert!(scratch_fast.token_spans.is_empty());
        assert!(scratch_fast.sp_to_orig.is_empty());

        let mut scratch_spans = EncodeScratch::default();
        let (ids_spans, spans) = sp.encode_with_spans_into("hello world", &mut scratch_spans);
        assert_eq!(ids_spans, &[3, 4]);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], 0..5);
        assert_eq!(spans[1], 5..11);

        let mut dec_scratch = DecodeScratch::default();
        let mut decoded = String::new();
        sp.decode_into(&[3, 4], &mut dec_scratch, &mut decoded);
        assert_eq!(decoded, "hello world");
        assert!(dec_scratch.token_spans.is_empty());

        let mut dec_scratch_spans = DecodeScratch::default();
        let mut decoded_spans_res = String::new();
        let out_spans = sp.decode_with_spans_into(&[3, 4], &mut dec_scratch_spans, &mut decoded_spans_res);
        assert_eq!(decoded_spans_res, "hello world");
        assert_eq!(out_spans.len(), 2);
    }

    #[test]
    fn test_encode_and_decode_thread_local_and_scratch_reuse() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-1.0; 32],
            decode_pieces: SentencePiece::compute_decode_pieces(&[
                Piece {
                    piece: "<unk>".to_string(),
                    score: 0.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "\u{2581}world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ]),
            model_type: ModelType::Unigram,
            vocab_size: 5,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("\u{2581}hello", 3);
                t.insert("\u{2581}world", 4);
                t.build()
            },
            max_piece_len: 6,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        // 1. encode() with thread-local scratch
        let ids1 = sp.encode("hello world");
        assert_eq!(ids1, vec![3, 4]);

        // 2. decode() with thread-local scratch
        let text1 = sp.decode(&[3, 4]);
        assert_eq!(text1, "hello world");

        // 3. Reusing scratch across multiple varied-length calls
        let mut scratch = EncodeScratch::default();
        let r1 = sp.encode_into("hello world", &mut scratch).to_vec();
        assert_eq!(r1, vec![3, 4]);
        let dp_len1 = scratch.dp.len();

        // Shorter input: tests that dp does not truncate and state is clean
        let r2 = sp.encode_into("world", &mut scratch).to_vec();
        assert_eq!(r2, vec![4]);
        assert_eq!(scratch.dp.len(), dp_len1);

        // Longer input: tests that dp resizes without double-write
        let r3 = sp.encode_into("hello world hello world", &mut scratch).to_vec();
        assert_eq!(r3, vec![3, 4, 3, 4]);
        assert!(scratch.dp.len() >= dp_len1);
    }

    #[test]
    fn test_normalize_nmt_nfkc_ascii_borrowed() {
        let ascii = "Hello, world! 123 - SentencePiece test.";
        let norm = normalize_nmt_nfkc(ascii);
        assert!(matches!(norm, Cow::Borrowed(_)));
        assert_eq!(norm.as_ref(), ascii);
    }

    #[test]
    fn test_normalize_nmt_nfkc_cjk_clean_borrowed() {
        let pure_han = "你好世界这是纯汉字文本";
        let norm = normalize_nmt_nfkc(pure_han);
        assert!(matches!(norm, Cow::Borrowed(_)));
        assert_eq!(norm.as_ref(), pure_han);
    }

    #[test]
    fn test_normalize_nmt_nfkc_fullwidth_ascii() {
        let text = "Ｈｅｌｌｏ，　ｗｏｒｌｄ！１２３";
        let norm = normalize_nmt_nfkc(text);
        assert!(matches!(norm, Cow::Owned(_)));
        assert_eq!(norm.as_ref(), "Hello, world!123");
    }

    #[test]
    fn test_normalize_nmt_nfkc_unicode_spaces() {
        let text = "A\u{3000}B\u{00A0}C\u{2003}D\u{202F}E";
        let norm = normalize_nmt_nfkc(text);
        assert_eq!(norm.as_ref(), "A B C D E");
    }

    #[test]
    fn test_normalize_nmt_nfkc_soft_hyphen_and_zero_width() {
        let text = "hel\u{00AD}lo\u{200B} \u{FEFF}world\u{200D}!";
        let norm = normalize_nmt_nfkc(text);
        assert_eq!(norm.as_ref(), "hello world!");
    }

    #[test]
    fn test_unk_viterbi_competition() {
        // Create a SentencePiece model with:
        // piece 0: <unk> with score -2.0
        // piece 1: <s>
        // piece 2: </s>
        // piece 3: ▁a with score -1.0
        // piece 4: ▁b with score -1.0
        // piece 5: ▁z with score -50.0 (an extremely rare, distorted piece)
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: -2.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁a".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁b".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁z".to_string(),
                    score: -50.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-2.0, 0.0, 0.0, -1.0, -1.0, -50.0],
            decode_pieces: vec![],
            model_type: ModelType::Unigram,
            vocab_size: 6,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("▁a", 3);
                t.insert("▁b", 4);
                t.insert("▁z", 5);
                t.build()
            },
            max_piece_len: 2,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        // For "a": piece 3 (▁a) score -1.0 beats unk (-2.0 + -2.0 = -4.0)
        assert_eq!(sp.encode("a"), vec![3]);

        // For "z": piece 5 (▁z) has a valid piece match in trie
        assert_eq!(sp.encode("z"), vec![5]);
    }

    #[test]
    fn test_encode_with_spans_preserves_original_offsets() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: -10.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-10.0, 0.0, 0.0, -1.0],
            decode_pieces: vec![],
            model_type: ModelType::Unigram,
            vocab_size: 4,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("▁hello", 3);
                t.build()
            },
            max_piece_len: 6,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let mut scratch = EncodeScratch::default();
        // Input contains soft hyphen \u{00AD} inside "hel\u{00AD}lo"
        let orig = "hel\u{00AD}lo";
        let (ids, spans) = sp.encode_with_spans_into(orig, &mut scratch);
        assert_eq!(ids, &[3]);
        assert_eq!(spans.len(), 1);
        // The span should cover the whole original string including the soft-hyphen
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, orig.len());
    }

    #[test]
    fn test_encode_consecutive_unk_folding() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: -10.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁hello".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁".to_string(),
                    score: -2.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "world".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-10.0, 0.0, 0.0, -1.0, -1.0, -2.0, -1.0],
            decode_pieces: vec![],
            model_type: ModelType::Unigram,
            vocab_size: 7,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("▁hello", 3);
                t.insert("▁world", 4);
                t.insert("▁", 5);
                t.insert("world", 6);
                t.build()
            },
            max_piece_len: 6,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let mut scratch = EncodeScratch::default();

        // 1. Single UNK is preserved without folding
        let text_single = "hello ? world";
        let (ids, spans) = sp.encode_with_spans_into(text_single, &mut scratch);
        assert_eq!(ids, &[3, 5, 0, 4]);
        assert_eq!(spans.len(), 4);
        assert_eq!(&text_single[spans[0].clone()], "hello");
        assert_eq!(&text_single[spans[1].clone()], " ");
        assert_eq!(&text_single[spans[2].clone()], "?");
        assert_eq!(&text_single[spans[3].clone()], " world");

        // 2. Consecutive UNKs folded into single unk_id with span covering all unknown characters
        let text_consec = "hello ??? world";
        let (ids, spans) = sp.encode_with_spans_into(text_consec, &mut scratch);
        assert_eq!(ids, &[3, 5, 0, 4]);
        assert_eq!(spans.len(), 4);
        assert_eq!(&text_consec[spans[0].clone()], "hello");
        assert_eq!(&text_consec[spans[1].clone()], " ");
        assert_eq!(&text_consec[spans[2].clone()], "???");
        assert_eq!(spans[2].start, 6);
        assert_eq!(spans[2].end, 9);
        assert_eq!(&text_consec[spans[3].clone()], " world");

        // 3. Consecutive UNKs without spaces between pieces
        let text_tight = "hello???world";
        let (ids, spans) = sp.encode_with_spans_into(text_tight, &mut scratch);
        assert_eq!(ids, &[3, 0, 6]);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text_tight[spans[0].clone()], "hello");
        assert_eq!(&text_tight[spans[1].clone()], "???");
        assert_eq!(spans[1].start, 5);
        assert_eq!(spans[1].end, 8);
        assert_eq!(&text_tight[spans[2].clone()], "world");

        // 4. Consecutive UNKs with multibyte UTF-8 characters (e.g. emoji)
        let text_emoji = "hello 🚀🎉🔥 world";
        let (ids, spans) = sp.encode_with_spans_into(text_emoji, &mut scratch);
        assert_eq!(ids, &[3, 5, 0, 4]);
        assert_eq!(spans.len(), 4);
        assert_eq!(&text_emoji[spans[0].clone()], "hello");
        assert_eq!(&text_emoji[spans[1].clone()], " ");
        assert_eq!(&text_emoji[spans[2].clone()], "🚀🎉🔥");
        assert_eq!(spans[2].start, 6);
        assert_eq!(spans[2].end, 18);
        assert_eq!(&text_emoji[spans[3].clone()], " world");

        // 5. Consecutive UNKs at start
        let text_start = "??? hello";
        let (ids, spans) = sp.encode_with_spans_into(text_start, &mut scratch);
        assert_eq!(ids, &[5, 0, 3]);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text_start[spans[1].clone()], "???");
        assert_eq!(&text_start[spans[2].clone()], " hello");

        // 6. Only consecutive UNKs
        let text_only = "?????";
        let (ids, spans) = sp.encode_with_spans_into(text_only, &mut scratch);
        assert_eq!(ids, &[5, 0]);
        assert_eq!(&text_only[spans[1].clone()], "?????");
        assert_eq!(spans[1].start, 0);
        assert_eq!(spans[1].end, 5);

        // 7. Plain encode without spans (track_spans = false)
        assert_eq!(sp.encode("hello ??? world"), vec![3, 5, 0, 4]);
        assert_eq!(sp.encode("hello???world"), vec![3, 0, 6]);
        assert_eq!(sp.encode("hello 🚀🎉🔥 world"), vec![3, 5, 0, 4]);
        assert_eq!(sp.encode("?????"), vec![5, 0]);

        // 8. Disjoint UNKs are NOT folded together
        let text_disjoint = "hello ? world ? world";
        let (ids, spans) = sp.encode_with_spans_into(text_disjoint, &mut scratch);
        assert_eq!(ids, &[3, 5, 0, 4, 5, 0, 4]);
        assert_eq!(spans.len(), 7);
        assert_eq!(&text_disjoint[spans[2].clone()], "?");
        assert_eq!(&text_disjoint[spans[5].clone()], "?");
    }

    #[test]
    fn test_encode_consecutive_unk_no_space_piece() {
        let sp = SentencePiece {
            pieces: vec![
                Piece {
                    piece: "<unk>".to_string(),
                    score: -10.0,
                    piece_type: 2,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "<s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "</s>".to_string(),
                    score: 0.0,
                    piece_type: 3,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁a".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
                Piece {
                    piece: "▁b".to_string(),
                    score: -1.0,
                    piece_type: 1,
                    byte_val: None,
                    fields_seen: FIELD_PIECE | FIELD_TYPE,
                },
            ],
            scores: vec![-10.0, 0.0, 0.0, -1.0, -1.0],
            decode_pieces: vec![],
            model_type: ModelType::Unigram,
            vocab_size: 5,
            trie: {
                let mut t = PieceTrieBuilder::new();
                t.insert("▁a", 3);
                t.insert("▁b", 4);
                t.build()
            },
            max_piece_len: 2,
            byte_pieces: [u32::MAX; 256],
            bos_id: 1,
            eos_id: 2,
            unk_id: 0,
            pad_id: 0,
            normalizer: None,
        };

        let mut scratch = EncodeScratch::default();

        // Input contains consecutive UNKs between two valid pieces
        let text = "a??? b";
        let (ids, spans) = sp.encode_with_spans_into(text, &mut scratch);
        assert_eq!(ids, &[3, 0, 4]);
        assert_eq!(spans.len(), 3);
        assert_eq!(&text[spans[0].clone()], "a");
        assert_eq!(&text[spans[1].clone()], "???");
        assert_eq!(spans[1].start, 1);
        assert_eq!(spans[1].end, 4);
        assert_eq!(&text[spans[2].clone()], " b");

        // Entirely unknown text collapses to a single unk_id covering whole string
        let text_all_unk = "????";
        let (ids_all, spans_all) = sp.encode_with_spans_into(text_all_unk, &mut scratch);
        assert_eq!(ids_all, &[0]);
        assert_eq!(spans_all.len(), 1);
        assert_eq!(spans_all[0].start, 0);
        assert_eq!(spans_all[0].end, 4);
        assert_eq!(&text_all_unk[spans_all[0].clone()], "????");

        assert_eq!(sp.encode("a??? b"), vec![3, 0, 4]);
        assert_eq!(sp.encode("????"), vec![0]);
    }

    #[test]
    fn test_encode_consecutive_unk_with_real_model() {
        let path = std::path::Path::new("data/models/en_zh-Hans/srcvocab.enzh.spm");
        if !path.exists() {
            return;
        }
        let sp = SentencePiece::load(path).unwrap();
        let mut scratch = EncodeScratch::default();
        // Use private use area unicode characters which are not in vocabulary
        let text = "Hello \u{E000}\u{E001}\u{E002} world";
        let (ids, spans) = sp.encode_with_spans_into(text, &mut scratch);
        assert_eq!(ids.len(), spans.len());
        // Invariant: no two consecutive unk_id tokens
        for window in ids.windows(2) {
            assert!(
                !(window[0] == sp.unk_id && window[1] == sp.unk_id),
                "Found consecutive UNK tokens in real model output: {:?}",
                ids
            );
        }
        for span in spans {
            assert!(span.start <= span.end);
            assert!(span.end <= text.len());
        }
    }

    #[test]
    fn test_normalizer_math_symbols_and_smp_with_real_model() {
        let path = std::path::Path::new("data/models/en_zh-Hans/srcvocab.enzh.spm");
        if !path.exists() {
            return;
        }
        let sp = SentencePiece::load(path).unwrap();
        assert!(sp.normalizer.is_some());

        let text = "model 𝑀𝑡 using a lightweight draft model 𝑀𝑑. Proposes 𝛾 tokens 𝑥1, . . . , 𝑥𝛾.";
        let norm = sp.normalizer.as_ref().unwrap().normalize(text);
        assert_eq!(
            norm.as_ref(),
            "model Mt using a lightweight draft model Md. Proposes γ tokens x1, . . . , xγ."
        );

        let mut scratch = EncodeScratch::default();
        let (ids, spans) = sp.encode_with_spans_into(text, &mut scratch);
        assert_eq!(ids.len(), spans.len());

        let decoded = sp.decode(ids);
        assert!(decoded.contains("Mt"));
        assert!(decoded.contains("Md"));
        assert!(decoded.contains("γ"));
        assert!(decoded.contains("x1"));
    }

    #[test]
    fn test_sentencepiece_from_bytes() {
        let path = std::path::Path::new("data/models/en_zh-Hans/srcvocab.enzh.spm");
        if !path.exists() {
            return;
        }
        let data = std::fs::read(path).unwrap();
        let sp = SentencePiece::from_bytes(&data).unwrap();
        assert_eq!(sp.decode(&[sp.bos_id]), "");
    }
}
