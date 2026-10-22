//! Binary lexical shortlist parser.
//!
//! Reads the `.s2t.bin` shortlist files. Format (see shortlist.h):
//! ```text
//! struct Header {
//!   uint64 magic;         // BINARY_SHORTLIST_MAGIC = 0xF11A48D5013417F5
//!   uint64 checksum;
//!   uint64 firstNum;
//!   uint64 bestNum;
//!   uint64 wordToOffsetSize;
//!   uint64 shortListsSize;
//! }
//! uint64 wordToOffset[wordToOffsetSize]
//! WordIndex shortLists[shortListsSize]
//! ```
//!
//! wordToOffset has one entry per source word plus a sentinel entry at the end
//! equal to shortListsSize. For source word i, the shortlist entries are in
//! range `shortLists[wordToOffset[i] .. wordToOffset[i+1]]`.
use std::cell::RefCell;
use std::fs;
use std::io::Read;
use std::path::Path;

use super::error::EngineError;

const BINARY_SHORTLIST_MAGIC: u64 = 0xF11A48D5013417F5;
/// Policy controlling how lexical shortlists filter the target vocabulary.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize, clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum ShortlistPolicy {
    /// Strictly adhere to `first_num` and `best_num` from the shortlist header.
    Strict,
    /// Elevate `first_n` to at least 500 (capped at vocab size) and allow up to 100 candidate words per source token.
    #[default]
    Relaxed,
    /// Bypass the shortlist entirely and evaluate the full target vocabulary.
    Disabled,
}

impl std::fmt::Display for ShortlistPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Strict => write!(f, "strict"),
            Self::Relaxed => write!(f, "relaxed"),
            Self::Disabled => write!(f, "disabled"),
        }
    }
}

impl std::str::FromStr for ShortlistPolicy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "strict" => Ok(Self::Strict),
            "relaxed" => Ok(Self::Relaxed),
            "disabled" | "disable" | "none" | "off" | "false" | "0" => Ok(Self::Disabled),
            other => Err(format!(
                "Unknown shortlist policy '{}': expected 'strict', 'relaxed', or 'disabled'",
                other
            )),
        }
    }
}


thread_local! {
    static SHORTLIST_SEEN: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}
pub struct Shortlist {
    /// `firstNum` from the header: the count of most-frequent target words
    /// seeded into every generated shortlist.
    first_num: u64,
    best_num: u64,
    word_to_offset: Vec<u64>,
    short_lists: Vec<u32>,
}

impl Shortlist {
    #[inline]
    pub fn first_num(&self) -> u64 {
        self.first_num
    }

    #[inline]
    pub fn best_num(&self) -> u64 {
        self.best_num
    }

    /// Load a `.s2t.bin` shortlist file directly streaming into memory structures.
    ///
    /// Validates the magic number — returns `Err` on mismatch or read failure.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let mut file = fs::File::open(path)
            .map_err(|e| EngineError::Io(format!("Failed to open shortlist file: {}", e)))?;
        let mut header = [0u8; 48];
        file.read_exact(&mut header)
            .map_err(|e| EngineError::Io(format!("Failed to read shortlist header: {}", e)))?;

        let magic = u64::from_le_bytes(header[0..8].try_into().unwrap());
        if magic != BINARY_SHORTLIST_MAGIC {
            return Err(EngineError::Format(format!(
                "Invalid shortlist magic: {:#018x}",
                magic
            )));
        }
        let first_num = u64::from_le_bytes(header[16..24].try_into().unwrap());
        let best_num = u64::from_le_bytes(header[24..32].try_into().unwrap());
        let word_to_offset_size = u64::from_le_bytes(header[32..40].try_into().unwrap()) as usize;
        let short_lists_size = u64::from_le_bytes(header[40..48].try_into().unwrap()) as usize;

        let word_to_offset_bytes = word_to_offset_size
            .checked_mul(8)
            .ok_or_else(|| EngineError::Format("wordToOffsetSize overflow".to_string()))?;
        let short_lists_bytes = short_lists_size
            .checked_mul(4)
            .ok_or_else(|| EngineError::Format("shortListsSize overflow".to_string()))?;

        let mut word_to_offset = Vec::with_capacity(word_to_offset_size);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    word_to_offset.as_mut_ptr() as *mut u8,
                    word_to_offset_bytes,
                );
                file.read_exact(slice).map_err(|e| {
                    EngineError::Io(format!("Failed to read word_to_offset: {}", e))
                })?;
                word_to_offset.set_len(word_to_offset_size);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut buf = vec![0u8; word_to_offset_bytes];
            file.read_exact(&mut buf).map_err(|e| {
                EngineError::Io(format!("Failed to read word_to_offset: {}", e))
            })?;
            word_to_offset.extend(
                buf.chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
            );
        }

        let mut short_lists = Vec::with_capacity(short_lists_size);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(
                    short_lists.as_mut_ptr() as *mut u8,
                    short_lists_bytes,
                );
                file.read_exact(slice).map_err(|e| {
                    EngineError::Io(format!("Failed to read short_lists: {}", e))
                })?;
                short_lists.set_len(short_lists_size);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            let mut buf = vec![0u8; short_lists_bytes];
            file.read_exact(&mut buf).map_err(|e| {
                EngineError::Io(format!("Failed to read short_lists: {}", e))
            })?;
            short_lists.extend(
                buf.chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
            );
        }

        Ok(Self {
            first_num,
            best_num,
            word_to_offset,
            short_lists,
        })
    }
    /// Parse a shortlist from an in-memory byte slice.
    pub fn from_bytes(data: &[u8]) -> Result<Self, EngineError> {
        let (first_num, best_num, word_to_offset_size, short_lists_size, header_size) =
            Self::parse_header(data)?;

        let word_to_offset_bytes = word_to_offset_size * 8;
        let short_lists_bytes = short_lists_size * 4;

        let offset_start = header_size;
        let shortlist_start = offset_start + word_to_offset_bytes;

        let mut word_to_offset = Vec::with_capacity(word_to_offset_size);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(offset_start),
                    word_to_offset.as_mut_ptr() as *mut u8,
                    word_to_offset_bytes,
                );
                word_to_offset.set_len(word_to_offset_size);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            word_to_offset.extend(
                data[offset_start..offset_start + word_to_offset_bytes]
                    .chunks_exact(8)
                    .map(|c| u64::from_le_bytes(c.try_into().unwrap())),
            );
        }

        let mut short_lists = Vec::with_capacity(short_lists_size);
        #[cfg(target_endian = "little")]
        {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr().add(shortlist_start),
                    short_lists.as_mut_ptr() as *mut u8,
                    short_lists_bytes,
                );
                short_lists.set_len(short_lists_size);
            }
        }
        #[cfg(not(target_endian = "little"))]
        {
            short_lists.extend(
                data[shortlist_start..shortlist_start + short_lists_bytes]
                    .chunks_exact(4)
                    .map(|c| u32::from_le_bytes(c.try_into().unwrap())),
            );
        }

        Ok(Self {
            first_num,
            best_num,
            word_to_offset,
            short_lists,
        })
    }
    fn parse_header(data: &[u8]) -> Result<(u64, u64, usize, usize, usize), EngineError> {
        if data.len() < 48 {
            return Err(EngineError::Format("Shortlist file too small".to_string()));
        }

        let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
        if magic != BINARY_SHORTLIST_MAGIC {
            return Err(EngineError::Format(format!(
                "Invalid shortlist magic: {:#018x}",
                magic
            )));
        }
        let _checksum = u64::from_le_bytes(data[8..16].try_into().unwrap());
        let first_num = u64::from_le_bytes(data[16..24].try_into().unwrap());
        let best_num = u64::from_le_bytes(data[24..32].try_into().unwrap());
        let word_to_offset_size = u64::from_le_bytes(data[32..40].try_into().unwrap()) as usize;
        let short_lists_size = u64::from_le_bytes(data[40..48].try_into().unwrap()) as usize;

        let header_size = 48usize;
        let word_to_offset_bytes = word_to_offset_size
            .checked_mul(8)
            .ok_or_else(|| EngineError::Format("wordToOffsetSize overflow".to_string()))?;
        let short_lists_bytes = short_lists_size
            .checked_mul(4)
            .ok_or_else(|| EngineError::Format("shortListsSize overflow".to_string()))?;
        let total_size = header_size
            .checked_add(word_to_offset_bytes)
            .and_then(|s| s.checked_add(short_lists_bytes))
            .ok_or_else(|| EngineError::Format("total size overflow".to_string()))?;

        if data.len() < total_size {
            return Err(EngineError::Format(format!(
                "Shortlist file truncated: expected {} bytes, got {}",
                total_size,
                data.len()
            )));
        }

        Ok((
            first_num,
            best_num,
            word_to_offset_size,
            short_lists_size,
            header_size,
        ))
    }

    /// Construct a shortlist directly from owned vectors (for testing).
    pub fn from_vecs(
        first_num: u64,
        best_num: u64,
        word_to_offset: Vec<u64>,
        short_lists: Vec<u32>,
    ) -> Self {
        Self {
            first_num,
            best_num,
            word_to_offset,
            short_lists,
        }
    }

    #[inline]
    pub fn word_to_offset(&self) -> &[u64] {
        &self.word_to_offset
    }

    #[inline]
    pub fn short_lists(&self) -> &[u32] {
        &self.short_lists
    }

    /// Look up the shortlist entries for a single source token ID.
    ///
    /// Returns an empty slice if `source_id` is out of range.
    #[inline]
    pub fn get(&self, source_id: u32) -> &[u32] {
        let idx = source_id as usize;
        let w2o = self.word_to_offset();
        if idx >= w2o.len().saturating_sub(1) {
            return &[];
        }
        let start = w2o[idx] as usize;
        let end = w2o[idx + 1] as usize;
        let sl = self.short_lists();
        if start >= end || start >= sl.len() {
            return &[];
        }
        &sl[start..end.min(sl.len())]
    }

    /// Check whether a target token ID is present in the shortlist for `source_id`.
    ///
    /// Shortlist blocks are stored in sorted ascending order in `.s2t.bin` files,
    /// so this lookup checks via binary search with linear fallback.
    #[inline]
    pub fn contains_target(&self, source_id: u32, target_id: u32) -> bool {
        let entries = self.get(source_id);
        if entries.binary_search(&target_id).is_ok() {
            true
        } else {
            entries.contains(&target_id)
        }
    }

    /// Build a deduplicated, sorted list of target vocabulary IDs from the
    /// shortlist entries matching the given source token IDs.
    ///
    /// For each `source_id` that is a valid index into the offset table, the
    /// corresponding shortlist entries are collected. IDs equal to or greater
    /// than `trg_vocab_size` are filtered out. The result is deduplicated and
    /// sorted.
    ///
    /// If no entries are collected (all source IDs are out of range or all
    /// shortlist entries are exhausted), the fallback is the range
    /// `0..min(bestNum, trg_vocab_size)`.
    pub fn generate(&self, source_ids: &[u32], trg_vocab_size: usize, policy: ShortlistPolicy) -> Vec<u32> {
        let mut out = Vec::new();
        SHORTLIST_SEEN.with(|seen_cell| {
            let mut seen = seen_cell.borrow_mut();
            self.generate_into(source_ids, trg_vocab_size, &mut seen, &mut out, policy);
        });
        out
    }

    /// Like [`generate`], but reuses caller-provided bitset and output buffers
    /// across calls so `translate()` avoids heap allocations per request on a
    /// hot worker thread.
    pub fn generate_into(
        &self,
        source_ids: &[u32],
        trg_vocab_size: usize,
        seen: &mut Vec<u64>,
        out: &mut Vec<u32>,
        policy: ShortlistPolicy,
    ) {
        out.clear();
        if trg_vocab_size == 0 {
            return;
        }

        if policy == ShortlistPolicy::Disabled {
            out.reserve(trg_vocab_size.saturating_sub(out.capacity()));
            out.extend(0..trg_vocab_size as u32);
            return;
        }

        let first_n = match policy {
            ShortlistPolicy::Relaxed => (self.first_num as usize).max(500).min(trg_vocab_size),
            ShortlistPolicy::Strict => (self.first_num as usize).min(trg_vocab_size),
            ShortlistPolicy::Disabled => unreachable!(),
        };

        let max_candidates_per_source = match policy {
            ShortlistPolicy::Relaxed => {
                if self.best_num > 0 {
                    (self.best_num as usize).max(100)
                } else {
                    100
                }
            }
            ShortlistPolicy::Strict => {
                if self.best_num > 0 {
                    self.best_num as usize
                } else {
                    usize::MAX
                }
            }
            ShortlistPolicy::Disabled => unreachable!(),
        };

        // Cap the reservation to a reasonable estimate or trg_vocab_size.
        let expected = first_n.saturating_add(
            source_ids
                .len()
                .saturating_mul(max_candidates_per_source.min(trg_vocab_size)),
        );
        let cap = expected.min(trg_vocab_size).min(first_n + 4096);
        if out.capacity() < cap {
            out.reserve(cap - out.capacity());
        }

        // 1. Prefix 0..first_n is always included in the shortlist
        out.extend(0..first_n as u32);

        // 2. The remaining range first_n..trg_vocab_size is tracked via bitmap
        let remaining_vocab = trg_vocab_size - first_n;
        if remaining_vocab > 0 {
            let num_words = remaining_vocab.div_ceil(64);
            if seen.len() < num_words {
                seen.resize(num_words, 0);
            }

            // Deduplicate source IDs to scan each source ID at most once.
            // For typical sentences (<= 512 tokens), use a stack-allocated buffer without zeroing.
            let mut stack_src = [std::mem::MaybeUninit::<u32>::uninit(); 512];
            let mut heap_src;
            let deduped_src: &[u32] = if source_ids.len() <= 1 {
                source_ids
            } else if source_ids.len() <= stack_src.len() {
                let n = source_ids.len();
                for (dst, &src) in stack_src[..n].iter_mut().zip(source_ids.iter()) {
                    dst.write(src);
                }
                // SAFETY: exactly the first n elements are initialized.
                let slice = unsafe {
                    std::slice::from_raw_parts_mut(stack_src.as_mut_ptr() as *mut u32, n)
                };
                slice.sort_unstable();
                let mut len = 0;
                for i in 0..slice.len() {
                    if i == 0 || slice[i] != slice[i - 1] {
                        slice[len] = slice[i];
                        len += 1;
                    }
                }
                &slice[..len]
            } else {
                heap_src = source_ids.to_vec();
                heap_src.sort_unstable();
                heap_src.dedup();
                &heap_src
            };

            let w2o = self.word_to_offset();
            let sl = self.short_lists();
            let max_idx = w2o.len().saturating_sub(1);
            let sl_len = sl.len();

            let mut min_word_idx = num_words;
            let mut max_word_idx = 0;

            for &src_id in deduped_src {
                let idx = src_id as usize;
                if idx >= max_idx {
                    continue;
                }
                let start = w2o[idx] as usize;
                let end = w2o[idx + 1] as usize;
                if start >= end || start >= sl_len {
                    continue;
                }
                let block = &sl[start..end.min(sl_len)];
                if block.is_empty() {
                    continue;
                }

                let mut added = 0;
                for &word_id in block {
                    if added >= max_candidates_per_source {
                        break;
                    }
                    let wid = word_id as usize;
                    if wid < first_n || wid >= trg_vocab_size {
                        continue;
                    }
                    let rel = wid - first_n;
                    let w_idx = rel / 64;
                    if w_idx < num_words {
                        if w_idx < min_word_idx {
                            min_word_idx = w_idx;
                        }
                        if w_idx > max_word_idx {
                            max_word_idx = w_idx;
                        }
                        seen[w_idx] |= 1u64 << (rel % 64);
                        added += 1;
                    }
                }
            }

            if min_word_idx <= max_word_idx {
                // Exact single reservation for all candidate words
                let total_new_words: usize = seen[min_word_idx..=max_word_idx]
                    .iter()
                    .map(|&w| w.count_ones() as usize)
                    .sum();
                if out.capacity() < out.len() + total_new_words {
                    out.reserve(total_new_words);
                }

                for w_idx in min_word_idx..=max_word_idx {
                    let word = seen[w_idx];
                    if word != 0 {
                        seen[w_idx] = 0;
                        let base = (first_n + w_idx * 64) as u32;
                        let mut bits = word;
                        while bits != 0 {
                            let t = bits.trailing_zeros();
                            let token_id = base + t;
                            if (token_id as usize) < trg_vocab_size {
                                out.push(token_id);
                            }
                            bits &= bits - 1;
                        }
                    }
                }
            }
        }

        if out.is_empty() {
            let limit = match policy {
                ShortlistPolicy::Disabled => trg_vocab_size,
                ShortlistPolicy::Relaxed => (500usize).max(self.best_num as usize).min(trg_vocab_size),
                ShortlistPolicy::Strict => (self.best_num as usize).min(trg_vocab_size),
            };
            out.extend(0u32..limit as u32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_and_generate_into_are_identical() {
        let shortlist =
            Shortlist::from_vecs(100, 10, vec![0, 3, 5, 5], vec![10, 200, 500, 200, 300]);

        let source_ids = [0, 1];
        let trg_vocab_size = 600;

        assert_eq!(shortlist.get(0), &[10, 200, 500]);
        assert_eq!(shortlist.get(1), &[200, 300]);
        assert_eq!(shortlist.get(2), &[] as &[u32]);
        assert_eq!(shortlist.get(999), &[] as &[u32]);
        assert!(shortlist.contains_target(0, 200));
        assert!(!shortlist.contains_target(0, 999));

        let out1 = shortlist.generate(&source_ids, trg_vocab_size, ShortlistPolicy::Strict);

        let mut seen = Vec::new();
        let mut out2 = Vec::new();
        shortlist.generate_into(&source_ids, trg_vocab_size, &mut seen, &mut out2, ShortlistPolicy::Strict);

        assert_eq!(out1, out2);
        assert!(out1.contains(&0));
        assert!(out1.contains(&99));
        assert!(!out1.contains(&100));
        assert!(out1.contains(&200));
        assert!(out1.contains(&300));
        assert!(out1.contains(&500));
        // Must be strictly sorted and deduplicated
        for i in 1..out1.len() {
            assert!(out1[i - 1] < out1[i]);
        }
    }

    #[test]
    fn generate_into_handles_non_adjacent_duplicates_and_many_tokens() {
        let shortlist = Shortlist::from_vecs(50, 10, vec![0, 2, 4, 4], vec![60, 70, 80, 90]);

        // Duplicates non-adjacent, and > 128 elements to test heap path as well
        let mut source_ids = vec![1, 0, 1, 0, 1];
        let trg_vocab_size = 200;
        let out1 = shortlist.generate(&source_ids, trg_vocab_size, ShortlistPolicy::Strict);
        let mut seen = Vec::new();
        let mut out2 = Vec::new();
        shortlist.generate_into(&source_ids, trg_vocab_size, &mut seen, &mut out2, ShortlistPolicy::Strict);
        assert!(out1.contains(&60));
        assert!(out1.contains(&70));
        assert!(out1.contains(&80));
        assert!(out1.contains(&90));

        // Heap path (> 512 tokens)
        source_ids = vec![0; 300];
        source_ids.extend(vec![1; 300]);
        let out_large1 = shortlist.generate(&source_ids, trg_vocab_size, ShortlistPolicy::Strict);
        let mut out_large2 = Vec::new();
        shortlist.generate_into(&source_ids, trg_vocab_size, &mut seen, &mut out_large2, ShortlistPolicy::Strict);
        assert_eq!(out_large1, out_large2);
    }

    #[test]
    fn from_bytes_parses_binary_format() {
        let mut buf = Vec::new();
        // Header: 6 x u64
        buf.extend_from_slice(&BINARY_SHORTLIST_MAGIC.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // checksum
        buf.extend_from_slice(&10u64.to_le_bytes()); // first_num
        buf.extend_from_slice(&5u64.to_le_bytes()); // best_num
        buf.extend_from_slice(&3u64.to_le_bytes()); // word_to_offset_size (3 entries: 2 words + sentinel)
        buf.extend_from_slice(&4u64.to_le_bytes()); // short_lists_size (4 u32s)

        // word_to_offset: [0, 2, 4]
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());

        // short_lists: [100, 200, 300, 400]
        buf.extend_from_slice(&100u32.to_le_bytes());
        buf.extend_from_slice(&200u32.to_le_bytes());
        buf.extend_from_slice(&300u32.to_le_bytes());
        buf.extend_from_slice(&400u32.to_le_bytes());

        let sl = Shortlist::from_bytes(&buf).expect("parse shortlist from bytes");
        assert_eq!(sl.get(0), &[100, 200]);
        assert_eq!(sl.get(1), &[300, 400]);
        assert_eq!(sl.get(2), &[] as &[u32]);
        assert!(sl.contains_target(0, 100));
        assert!(sl.contains_target(0, 200));
        assert!(!sl.contains_target(0, 300));
    }

    #[test]
    fn test_shortlist_policy_disabled() {
        let shortlist =
            Shortlist::from_vecs(50, 10, vec![0, 2, 4, 4], vec![60, 70, 80, 90]);
        let source_ids = [0];
        let trg_vocab_size = 1000;

        let out = shortlist.generate(&source_ids, trg_vocab_size, ShortlistPolicy::Disabled);
        assert_eq!(out.len(), 1000);
        assert_eq!(out[0], 0);
        assert_eq!(out[999], 999);
        for (i, &v) in out.iter().enumerate() {
            assert_eq!(v as usize, i);
        }
    }

    #[test]
    fn test_shortlist_policy_relaxed_extends_first_n_and_candidates() {
        // first_num = 50, but Relaxed should elevate to at least 500
        let shortlist =
            Shortlist::from_vecs(50, 5, vec![0, 2, 4, 4], vec![600, 700, 800, 900]);
        let source_ids = [0];
        let trg_vocab_size = 1000;

        let out = shortlist.generate(&source_ids, trg_vocab_size, ShortlistPolicy::Relaxed);
        // First 500 must all be present
        for i in 0..500 {
            assert!(out.contains(&(i as u32)));
        }
        // Candidate words >= 500 (600, 700) should be included
        assert!(out.contains(&600));
        assert!(out.contains(&700));
    }

    #[test]
    fn test_shortlist_policy_parsing_and_serde() {
        assert_eq!("strict".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Strict);
        assert_eq!("STRICT".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Strict);
        assert_eq!("relaxed".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Relaxed);
        assert_eq!("disabled".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert_eq!("disable".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert_eq!("none".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert_eq!("off".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert_eq!("0".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert_eq!("false".parse::<ShortlistPolicy>().unwrap(), ShortlistPolicy::Disabled);
        assert!("invalid".parse::<ShortlistPolicy>().is_err());

        assert_eq!(ShortlistPolicy::default(), ShortlistPolicy::Relaxed);
        assert_eq!(ShortlistPolicy::Strict.to_string(), "strict");
        assert_eq!(ShortlistPolicy::Relaxed.to_string(), "relaxed");
        assert_eq!(ShortlistPolicy::Disabled.to_string(), "disabled");

        // Serde roundtrip
        let json = serde_json::to_string(&ShortlistPolicy::Strict).unwrap();
        assert_eq!(json, "\"strict\"");
        let parsed: ShortlistPolicy = serde_json::from_str("\"relaxed\"").unwrap();
        assert_eq!(parsed, ShortlistPolicy::Relaxed);
    }

    #[test]
    fn test_shortlist_out_of_bounds_and_dirty_leak_protection() {
        // Inverted offsets [10, 2] and out-of-order/out-of-range candidates
        let shortlist = Shortlist::from_vecs(
            10,
            5,
            vec![0, 10, 2, 8],
            vec![5, 50, 20, 1000, 15, 80, 2, 90],
        );

        // Inverted offset should return empty slice without panic
        assert_eq!(shortlist.get(1), &[] as &[u32]);

        let mut seen = Vec::new();
        let mut out = Vec::new();

        // Target vocab size is 100 (so 1000 is out of bounds)
        shortlist.generate_into(&[0, 2, 999], 100, &mut seen, &mut out, ShortlistPolicy::Strict);

        // Prefix 0..10 must be included
        for i in 0..10 {
            assert!(out.contains(&(i as u32)));
        }
        // Valid candidates (15, 20, 50) must be included
        assert!(out.contains(&15));
        assert!(out.contains(&20));
        assert!(out.contains(&50));
        // Out of bounds candidate (1000) must NOT be included
        assert!(!out.contains(&1000));
        for &tok in &out {
            assert!((tok as usize) < 100);
        }

        // Verify no dirty bits leaked into seen buffer
        assert!(seen.iter().all(|&w| w == 0));

        // Reusing seen buffer on a second call must not leak previous bits
        let mut out2 = Vec::new();
        shortlist.generate_into(&[2], 100, &mut seen, &mut out2, ShortlistPolicy::Strict);
        assert!(seen.iter().all(|&w| w == 0));
    }

    #[test]
    fn test_shortlist_large_dirty_dense_fallback() {
        // Generate > 512 candidate words in distinct 64-word buckets to trigger dense fast path
        let mut cands = Vec::new();
        for i in 0..600 {
            cands.push(100 + i * 64);
        }
        let shortlist = Shortlist::from_vecs(50, 1000, vec![0, cands.len() as u64], cands);
        let mut seen = Vec::new();
        let mut out = Vec::new();
        let trg_vocab = 50000;
        shortlist.generate_into(&[0], trg_vocab, &mut seen, &mut out, ShortlistPolicy::Strict);

        assert_eq!(out.len(), 50 + 600);
        for i in 1..out.len() {
            assert!(out[i - 1] < out[i]);
        }
        assert!(seen.iter().all(|&w| w == 0));
    }
}
