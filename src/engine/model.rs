//! Marian binary model format parser.
//!
//! Reads the `.bin` model files produced by marian/nematus. See
//! `common/binary.cpp` in the original C++ source:
//!
//! ```text
//! uint64  version         // BINARY_FILE_VERSION = 1
//! uint64  numHeaders
//! Header[numHeaders]      // each 4*uint64: nameLength, type, shapeLength, dataLength
//! char[]  names(...)
//! int32[] shapes(...)
//! uint64  offset          // aligned padding to 256-byte boundary
//! char[]  data(...)
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use super::error::EngineError;
pub use super::tensor::Tensor2 as Tensor;
/// A named tensor loaded from the model file (dequantized to `f32`).
///
/// Reuses the compute [`Tensor2`] type so weights live once in the model
/// (`Model::get` returns an `&Tensor2`) and the transformer reads them by
/// reference instead of cloning a fresh copy per layer call / decode step.
pub use super::tensor::{Quant8Data, Quant8Tensor};
/// Upper bound for a single tensor dimension read from a header.  Real
/// Marian tensors stay far below this; negative values become huge `usize`
/// dimensions when cast, and larger values indicate a corrupt file.
const MAX_DIMENSION: i32 = 1 << 28;

/// The type tag stored in a Marian binary header field.
///
/// These values come from `binary.h` in the original marian source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItemType {
    Float32,
    Int8,
    Int16,
    IntGemm8,
    IntGemm16,
    Unknown(u64),
}

fn type_from_u64(v: u64) -> ItemType {
    // New-style type values (from marian-fork types.h)
    //   signed_type = 0x0100, float_type = 0x0400, intgemm_type = 0x4000
    //   int8 = 0x0101, int16 = 0x0102, float32 = 0x0404
    //   intgemm8 = 0x4101, intgemm16 = 0x4102
    // Old-style values (from original marian-lite binary.h):
    //   FLOAT32=0x20, INT8=0x21, INT16=0x22, INTGemm8=0x31, INTGemm16=0x32
    match v {
        0x20 | 0x0404 => ItemType::Float32,
        0x21 | 0x0101 => ItemType::Int8,
        0x22 | 0x0102 => ItemType::Int16,
        0x31 | 0x4101 => ItemType::IntGemm8,
        0x32 | 0x4102 => ItemType::IntGemm16,
        other => ItemType::Unknown(other),
    }
}

#[inline]
fn to_2d_shape(shape: &[usize]) -> [usize; 2] {
    match shape.len() {
        0 => [0, 0],
        1 => [1, shape[0]],
        2 => [shape[0], shape[1]],
        _ => [shape[..shape.len() - 1].iter().product(), *shape.last().unwrap_or(&1)],
    }
}
/// Autoregressive mechanism used by the decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecoderAutoregType {
    /// Recurrent Bergamot SSRU (default for Firefox/Bergamot models).
    #[default]
    Rnn,
    /// Standard Transformer multi-head causal self-attention with KV-Cache.
    SelfAttention,
}


/// A tensor entry in the model's item table.
///
/// intgemm8 weight matrices whose f32 form is never read on the int8
/// inference path are stored as [`LazyEntry`]: the dequantized copy is
/// materialized only if some caller actually requests it (e.g. a future
/// `gemm_precision` that disables the quantized kernels).  Dequantization is
/// a per-element deterministic function of the stored bytes, so a lazily
/// built tensor is bit-identical to one built at load time.
enum ItemEntry {
    Ready(Tensor),
    Lazy(LazyEntry),
}

struct LazyEntry {
    quant_name: String,
    /// Original header shape `[rows, cols]` of the f32 weight.
    shape: Vec<usize>,
    /// How the quantized bytes relate to the f32 weight layout:
    /// `true` — disk stores the transpose (regular weight matrices), so
    /// materialization reads column-major; `false` — disk rows match f32
    /// rows (embedding tables, which are never transposed on disk).
    transposed: bool,
    slot: OnceLock<Tensor>,
}

/// Parsed Marian binary model.
pub struct Model {
    items: HashMap<String, ItemEntry>,
    /// intgemm8 weight matrices in on-disk layout (see [`Quant8Tensor`]).
    quant8: HashMap<String, Quant8Tensor>,
    /// Activation quantizers resolved at load time, keyed by weight name:
    /// `quant_mult_cache[w]` holds `{w}_QuantMultA.data[0]`.
    quant_mult_cache: HashMap<String, f32>,
    pub config_yaml: String,
    /// Lazily parsed config values keyed by name (`get_config_value`).
    config_cache: std::sync::Mutex<HashMap<String, Option<String>>>,
    _mmap: Option<Arc<memmap2::Mmap>>,
}

impl Model {
    /// Load a `.bin` model file.
    ///
    /// Uses memory-mapped zero-copy tensor loading when possible,
    /// falling back to owned buffer parsing.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(EngineError::Io(format!("Failed to read model file: {e}"))),
        };
        if let Ok(mmap) = unsafe { memmap2::Mmap::map(&file) } {
            let mmap = Arc::new(mmap);
            Self::parse_buffer(&mmap, Some(mmap.clone()), None)
        } else {
            let data = std::fs::read(path)
                .map_err(|e| EngineError::Io(format!("Failed to read model file: {e}")))?;
            Self::from_bytes(&data)
        }
    }

    /// Parse a model from an in-memory byte slice.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, EngineError> {
        Self::from_shared_bytes(Arc::from(buf))
    }

    /// Parse a model from shared bytes using zero-copy tensor slicing on any platform.
    pub fn from_shared_bytes(data: Arc<[u8]>) -> Result<Self, EngineError> {
        Self::parse_buffer(&data, None, Some(data.clone()))
    }

    fn parse_buffer(
        buf: &[u8],
        mmap_source: Option<Arc<memmap2::Mmap>>,
        shared_source: Option<Arc<[u8]>>,
    ) -> Result<Self, EngineError> {
        let mut pos: usize = 0;

        // --- version (must be 1) ---
        let version = read_u64_le(buf, &mut pos)?;
        if version != 1 {
            return Err(EngineError::Format(format!(
                "Unsupported binary file version {}, expected 1",
                version
            )));
        }

        // --- numHeaders ---
        let num_headers = read_u64_le(buf, &mut pos)? as usize;
        // Each header occupies at least 4 × u64 = 32 bytes on disk, so a
        // count above `remaining / 32` cannot be valid.  Reject it up front
        // instead of pre-allocating from an unbounded corrupt value.
        if num_headers > buf.len().saturating_sub(pos) / 32 {
            return Err(EngineError::Format(format!(
                "Invalid header count {num_headers} exceeds remaining file size"
            )));
        }

        // --- headers ---
        // Each header is 4 × u64: nameLength, type, shapeLength, dataLength
        struct Header {
            name_len: u64,
            type_val: u64,
            shape_len: u64,
            data_len: u64,
            name: String,
            shape: Vec<usize>,
        }

        let mut headers: Vec<Header> = Vec::with_capacity(num_headers);
        for _ in 0..num_headers {
            let name_len = read_u64_le(buf, &mut pos)?;
            let type_val = read_u64_le(buf, &mut pos)?;
            let shape_len = read_u64_le(buf, &mut pos)?;
            let data_len = read_u64_le(buf, &mut pos)?;
            headers.push(Header {
                name_len,
                type_val,
                shape_len,
                data_len,
                name: String::new(),
                shape: Vec::new(),
            });
        }

        // --- names ---
        for h in &mut headers {
            let name_len = h.name_len as usize;
            ensure_remaining(buf, pos, name_len)?;
            let end = pos + name_len;
            // name is null-terminated; strip the trailing \0
            let raw = &buf[pos..end];
            let name = if let Some(nul_pos) = raw.iter().position(|&b| b == 0) {
                String::from_utf8(raw[..nul_pos].to_vec())
                    .map_err(|e| EngineError::Format(format!("Invalid UTF-8 in item name: {e}")))?
            } else {
                String::from_utf8(raw.to_vec())
                    .map_err(|e| EngineError::Format(format!("Invalid UTF-8 in item name: {e}")))?
            };
            h.name = name;
            pos = end;
        }

        // --- shapes ---
        for h in &mut headers {
            let shape_len = h.shape_len as usize;
            ensure_remaining(buf, pos, shape_len * 4)?;
            let mut shape = Vec::with_capacity(shape_len);
            for _ in 0..shape_len {
                let dim = read_i32_le(buf, &mut pos)?;
                if !(0..=MAX_DIMENSION).contains(&dim) {
                    return Err(EngineError::DimensionMismatch(format!(
                        "Invalid tensor dimension {dim} in item \"{}\"",
                        h.name
                    )));
                }
                shape.push(dim as usize);
            }
            h.shape = shape;
        }

        // --- align data section to 256-byte boundary ---
        // The C++ reader reads a u64 "offset" value, then skips that many bytes
        // of padding.  The data section starts at the next 256-byte boundary
        // after the shapes + offset field.
        let data_offset_field = read_u64_le(buf, &mut pos)? as usize;
        pos = pos.checked_add(data_offset_field).ok_or_else(|| {
            EngineError::Format("Integer overflow aligning data section".to_string())
        })?;
        // --- data segments ---

        // Resolve the byte length this tensor occupies on disk from its
        // element count, and verify it against the remaining file bytes.
        fn checked_product(shape: &[usize], name: &str) -> Result<usize, EngineError> {
            shape.iter().try_fold(1usize, |acc, &d| {
                acc.checked_mul(d).ok_or_else(|| {
                    EngineError::DimensionMismatch(format!("Tensor \"{name}\" element count overflows usize"))
                })
            })
        }

        fn checked_consumed(
            num_elements: usize,
            elem_size: usize,
            pos: usize,
            data_len: usize,
            buf: &[u8],
            name: &str,
        ) -> Result<usize, EngineError> {
            let consumed = num_elements.checked_mul(elem_size).ok_or_else(|| {
                EngineError::DimensionMismatch(format!("Tensor \"{name}\" size overflows usize"))
            })?;
            if consumed > data_len {
                return Err(EngineError::Format(format!(
                    "Tensor \"{name}\" payload size {consumed} exceeds declared data segment {data_len}"
                )));
            }
            let avail = buf.len() - pos;
            if consumed > avail {
                return Err(EngineError::Format(format!(
                    "Unexpected end of file: need {} bytes at offset {}, file has {} bytes",
                    consumed,
                    pos,
                    buf.len()
                )));
            }
            Ok(consumed)
        }

        let mut items: HashMap<String, ItemEntry> = HashMap::new();
        let mut quant8: HashMap<String, Quant8Tensor> = HashMap::new();
        let mut quant_mult_cache: HashMap<String, f32> = HashMap::new();
        let mut config_yaml = String::new();

        for h in &headers {
            let item_type = type_from_u64(h.type_val);
            let shape = h.shape.clone();
            let data_len = h.data_len as usize;
            ensure_remaining(buf, pos, data_len)?;

            // Guard against the tiny model.yml item: shape is [yaml_len + 1], data_len bytes
            let is_model_yaml = h.name == "special:model.yml";

            // Handle special:model.yml as a raw string, regardless of its storage type
            if is_model_yaml {
                let yaml_len = checked_product(&shape, &h.name)?;
                let raw = &buf[pos..pos + yaml_len.min(data_len)];
                let yaml_str = if let Some(nul_pos) = raw.iter().position(|&b| b == 0) {
                    String::from_utf8(raw[..nul_pos].to_vec())
                        .map_err(|e| format!("Invalid UTF-8 in model.yml: {e}"))?
                } else {
                    String::from_utf8(raw.to_vec())
                        .map_err(|e| format!("Invalid UTF-8 in model.yml: {e}"))?
                };
                config_yaml = yaml_str;
                // Advance past the full data segment (includes padding)
                pos = pos
                    .checked_add(data_len)
                    .ok_or_else(|| "Integer overflow skipping model.yml padding".to_string())?;
                continue;
            }

            let entry = match item_type {
                ItemType::Float32 => {
                    let num_elements = checked_product(&shape, &h.name)?;
                    let consumed = checked_consumed(num_elements, 4, pos, data_len, buf, &h.name)?;
                    let bytes = &buf[pos..pos + consumed];
                    let mut data = Vec::with_capacity(num_elements);
                    data.extend(
                        bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                    // Skip any remaining padding bytes in this segment
                    let skip = data_len.saturating_sub(consumed);
                    pos += consumed + skip;
                    ItemEntry::Ready(Tensor { shape: to_2d_shape(&shape), data })
                }
                ItemType::Int8 => {
                    let num_elements = checked_product(&shape, &h.name)?;
                    checked_consumed(num_elements, 1, pos, data_len, buf, &h.name)?;
                    let bytes = &buf[pos..pos + num_elements];
                    let mut data = Vec::with_capacity(num_elements);
                    data.extend(bytes.iter().map(|&b| b as i8 as f32));
                    let skip = data_len.saturating_sub(num_elements);
                    pos += num_elements + skip;
                    ItemEntry::Ready(Tensor { shape: to_2d_shape(&shape), data })
                }
                ItemType::Int16 => {
                    let num_elements = checked_product(&shape, &h.name)?;
                    let consumed = checked_consumed(num_elements, 2, pos, data_len, buf, &h.name)?;
                    let bytes = &buf[pos..pos + consumed];
                    let mut data = Vec::with_capacity(num_elements);
                    data.extend(
                        bytes
                            .chunks_exact(2)
                            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32),
                    );
                    let skip = data_len.saturating_sub(consumed);
                    pos += consumed + skip;
                    ItemEntry::Ready(Tensor { shape: to_2d_shape(&shape), data })
                }
                ItemType::IntGemm8 => {
                    let num_elements = checked_product(&shape, &h.name)?;
                    // On-disk layout: `num_elements` int8 values followed by one
                    // f32 quantMult (any further bytes are segment padding).
                    checked_consumed(num_elements.saturating_add(4), 1, pos, data_len, buf, &h.name)?;
                    let data_start = pos;
                    pos += num_elements;
                    // Read the quantMult (float32 at the end of the int8 data)
                    let quant_mult = read_f32_le(buf, &mut pos)?;

                    let int8_slice: &[i8] = unsafe {
                        std::slice::from_raw_parts(
                            buf[data_start..data_start + num_elements].as_ptr() as *const i8,
                            num_elements,
                        )
                    };

                    let q8_data = if let Some(mmap) = &mmap_source {
                        Quant8Data::Mmap {
                            mmap: Arc::clone(mmap),
                            offset: data_start,
                            len: num_elements,
                        }
                    } else if let Some(shared) = &shared_source {
                        Quant8Data::Shared {
                            data: Arc::clone(shared),
                            offset: data_start,
                            len: num_elements,
                        }
                    } else {
                        Quant8Data::Owned(int8_slice.to_vec())
                    };
                    // Wemb matrices are stored WITHOUT transposition in the binary.
                    // Their f32 form is only consumed row-by-row (embedding lookup
                    // / decoder step input), so like every other intgemm8 tensor
                    // the dequantized copy is deferred until first use; consumers
                    // on the int8 path read rows through `dequant_row` instead of
                    // materializing the full table.
                    let is_wemb = h.name.contains("Wemb");
                    if !is_wemb {
                        // Build quant8 tensor in on-disk layout (no f32 transpose):
                        // 2-D non-Wemb already stored as [out, in] row-major; keep it.
                        let q8_shape = if h.shape.len() == 2 && h.shape[0] > 0 && h.shape[1] > 0 {
                            vec![h.shape[1], h.shape[0]]
                        } else {
                            h.shape.clone()
                        };
                        // Precompute the per-row `128·Σw` correction for the
                        // i8 gemm kernels once here — recomputing it per matmul
                        // call costs a full weight rescan (~46μs for a 1536×384
                        // FFN matrix). Every non-Wemb 2-D weight gets the table:
                        // the VNNI row kernels have no inline Σw fold, so even
                        // decoder step weights (m==1) consume it, and the m>1
                        // kernels reuse it across all rows. The one-off scan at
                        // load time is negligible; decoder-only skip list removed
                        // when VNNI replaced the maddubs chains.
                        let row_sums = if q8_shape.len() == 2 && q8_shape[0] > 0 && q8_shape[1] > 0 {
                            Some(crate::engine::tensor::simd::compute_row_sums_vec(int8_slice, q8_shape[0], q8_shape[1]))
                        } else {
                            None
                        };
                        quant8.insert(
                            h.name.clone(),
                            Quant8Tensor {
                                shape: q8_shape,
                                data: q8_data,
                                quant_mult,
                                row_sums,
                            },
                        );
                        // Skip padding bytes in this segment. The dequantized f32
                        // copy is not materialized: the int8 kernels consume the
                        // quant8 form and `get` rebuilds f32 lazily if ever asked.
                        let consumed = num_elements + 4;
                        let skip = data_len.saturating_sub(consumed);
                        pos = pos.checked_add(skip).ok_or_else(|| {
                            "Integer overflow skipping intgemm8 padding".to_string()
                        })?;
                        ItemEntry::Lazy(LazyEntry {
                            quant_name: h.name.clone(),
                            shape,
                            // Regular weights are stored transposed on disk.
                            transposed: true,
                            slot: OnceLock::new(),
                        })
                    } else {
                        // Wemb keeps its raw int8 rows in `quant8` ([vocab, d_model],
                        // untransposed): the shortlist output projection gathers rows
                        // straight from these bytes, and embedding lookups dequantize
                        // single rows via `dequant_row`. No eager f32 table is built.
                        let row_sums = if h.shape.len() == 2 && h.shape[0] > 0 && h.shape[1] > 0 {
                            Some(crate::engine::tensor::simd::compute_row_sums_vec(int8_slice, h.shape[0], h.shape[1]))
                        } else {
                            None
                        };
                        quant8.insert(
                            h.name.clone(),
                            Quant8Tensor {
                                shape: h.shape.clone(),
                                data: q8_data,
                                quant_mult,
                                row_sums,
                            },
                        );
                        let consumed = num_elements + 4;
                        let skip = data_len.saturating_sub(consumed);
                        pos = pos.checked_add(skip).ok_or_else(|| {
                            "Integer overflow skipping intgemm8 padding".to_string()
                        })?;
                        ItemEntry::Lazy(LazyEntry {
                            quant_name: h.name.clone(),
                            shape,
                            // Embedding tables keep row-major f32 order on disk.
                            transposed: false,
                            slot: OnceLock::new(),
                        })
                    }
                }
                ItemType::IntGemm16 => {
                    let num_elements = checked_product(&shape, &h.name)?;
                    // On-disk layout: `num_elements` int16 values followed by one
                    // f32 quantMult (any further bytes are segment padding).
                    checked_consumed(num_elements.saturating_mul(2).saturating_add(4), 1, pos, data_len, buf, &h.name)?;
                    let int16_vals: Vec<i16> = buf[pos..pos + num_elements * 2]
                        .chunks_exact(2)
                        .map(|c| i16::from_le_bytes([c[0], c[1]]))
                        .collect();
                    pos += num_elements * 2;
                    let quant_mult = read_f32_le(buf, &mut pos)?;
                    let mut dequant = Vec::with_capacity(num_elements);
                    for &v in &int16_vals {
                        dequant.push((v as f32) / quant_mult);
                    }
                    let is_wemb = h.name.contains("Wemb");
                    let tensor = if shape.len() == 2 && shape[0] > 0 && shape[1] > 0 && !is_wemb {
                        let rows = shape[0];
                        let cols = shape[1];
                        let mut transposed = Vec::with_capacity(num_elements);
                        for r in 0..rows {
                            for c in 0..cols {
                                transposed.push(dequant[c * rows + r]);
                            }
                        }
                        Tensor {
                            shape: [rows, cols],
                            data: transposed,
                        }
                    } else {
                        Tensor {
                            shape: to_2d_shape(&shape),
                            data: dequant,
                        }
                    };
                    let consumed = num_elements * 2 + 4;
                    let skip = data_len.saturating_sub(consumed);
                    pos = pos
                        .checked_add(skip)
                        .ok_or_else(|| "Integer overflow skipping intgemm16 padding".to_string())?;
                    ItemEntry::Ready(tensor)
                }
                ItemType::Unknown(v) => {
                    // Unknown type — skip the data segment
                    pos = pos.checked_add(data_len).ok_or_else(|| {
                        format!("Integer overflow skipping unknown type 0x{v:x} data")
                    })?;
                    continue;
                }
            };
            // Cache the activation quantizer for intgemm8 weights up front:
            // `quant_mult_a` is on the per-layer hot path and used to
            // re-format + look up `{name}_QuantMultA` on every call.
            //
            // `_QuantMultA` tensors may be stored as Float32 (most layer
            // weights) or IntGemm8 (e.g. the embedding).  Cache either form:
            // Float32 is ready immediately; IntGemm8 dequantizes its single
            // element (`data[0] / quant_mult`) from the quant8 table.  Without
            // this every layer falls back to `qa = 1.0`, rounding activations
            // to integers and corrupting the output.
            if h.name.ends_with("_QuantMultA") {
                let key = h.name.trim_end_matches("_QuantMultA").to_string();
                let qa = match &entry {
                    ItemEntry::Ready(t) => t.data.first().copied(),
                    _ => quant8
                        .get(&h.name)
                        .and_then(|q| q.data.first().map(|&v| v as f32 / q.quant_mult)),
                };
                if let Some(qa) = qa {
                    quant_mult_cache.insert(key, qa);
                }
            }

            items.insert(h.name.clone(), entry);
        }

        Ok(Model {
            items,
            quant8,
            quant_mult_cache,
            config_yaml,
            config_cache: std::sync::Mutex::new(HashMap::new()),
            _mmap: mmap_source,
        })
    }

    /// Look up a named tensor (e.g. `"F0_encoder_Wemb"`).
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        match self.items.get(name) {
            Some(ItemEntry::Ready(t)) => Some(t),
            Some(ItemEntry::Lazy(lazy)) => {
                let built = lazy.slot.get_or_init(|| {
                    // Reproduce the load-time materialization exactly. Regular
                    // weights are stored as the quantized transpose, so their
                    // f32 form is result[r][c] = disk[c * rows + r] / qm;
                    // embedding tables keep row-major order on disk and
                    // dequantize linearly.
                    let q = &self.quant8[&lazy.quant_name];
                    let shape = &lazy.shape;
                    if !lazy.transposed || shape.len() != 2 {
                        // Row-major (or non-2-D) tensors have no transposed
                        // layout; dequantize linearly instead of indexing [0]/[1].
                        return Tensor {
                            shape: to_2d_shape(shape),
                            data: q.data.iter().map(|&v| v as f32 / q.quant_mult).collect(),
                        };
                    }
                    let (rows, cols) = (shape[0], shape[1]);
                    let mut data = Vec::with_capacity(q.data.len());
                    for r in 0..rows {
                        for c in 0..cols {
                            data.push(q.data[c * rows + r] as f32 / q.quant_mult);
                        }
                    }
                    Tensor {
                        shape: [rows, cols],
                        data,
                    }
                });
                Some(built)
            }
            None => None,
        }
    }

    /// Look up an intgemm8 weight matrix kept in on-disk layout. Returns
    /// `None` for tensors that are not intgemm8-quantized (e.g. f32 weights).
    pub fn get_quant8(&self, name: &str) -> Option<&Quant8Tensor> {
        self.quant8.get(name)
    }
    /// Probe for a tensor's existence without materializing it.
    ///
    /// Returns `true` when the item table holds an entry with this exact
    /// name, including lazily stored weights. Use this for existence probes
    /// (`layer_indexing`) where only presence matters — `get` would force a
    /// full f32 materialization of an intgemm8 tensor just to answer
    /// `is_some()`.
    pub fn contains(&self, name: &str) -> bool {
        self.items.contains_key(name)
    }

    /// Dequantize one embedding row from an intgemm8 weight matrix.
    ///
    /// Embedding tables are consumed row-by-row (encoder lookup by token id,
    /// decoder step input), so the int8 path reads single rows straight from
    /// the quantized bytes instead of materializing the full f32 table. The
    /// per-element expression is identical to the load-time dequantization,
    /// so the values are bit-identical to a row of the equivalent f32 tensor.
    ///
    /// Returns `None` when `name` is not an intgemm8 tensor or `row` is out
    /// of range.
    pub fn dequant_row(&self, name: &str, row: usize) -> Option<Vec<f32>> {
        let q = self.quant8.get(name)?;
        let d_model = *q.shape.get(1)?;
        let start = row.checked_mul(d_model)?;
        let src = q.data.get(start..start + d_model)?;
        Some(src.iter().map(|&v| v as f32 / q.quant_mult).collect())
    }
    /// Dequantize one intgemm8 embedding row directly into a caller-owned
    /// buffer. Same per-element expression as [`Self::dequant_row`] (so the
    /// values are bit-identical), but avoids the per-row `Vec` allocation —
    /// the decoder step loop reuses one scratch buffer across all steps.
    ///
    /// Returns `false` when `name` is not an intgemm8 tensor or `row` is out
    /// of range, leaving `out` untouched.
    pub fn dequant_row_into(&self, name: &str, row: usize, out: &mut [f32]) -> bool {
        let q = match self.quant8.get(name) {
            Some(q) => q,
            None => return false,
        };
        let d_model = match q.shape.get(1) {
            Some(&d) if d == out.len() => d,
            _ => return false,
        };
        let start = match row.checked_mul(d_model) {
            Some(s) => s,
            None => return false,
        };
        let src = match q.data.get(start..start + d_model) {
            Some(s) => s,
            None => return false,
        };
        for (dst, &v) in out.iter_mut().zip(src) {
            *dst = v as f32 / q.quant_mult;
        }
        true
    }
    /// Activation quantizer `quant_mult` for the weight `name`, resolved at
    /// load time from the `{name}_QuantMultA` float32 scalar tensor.
    pub fn quant_mult_a(&self, name: &str) -> Option<f32> {
        self.quant_mult_cache.get(name).copied()
    }

    /// Parse a configuration value from the embedded YAML by simple text scanning.
    ///
    /// Looks for a line starting with `key:` and returns the trimmed value after the colon.
    /// Handles quoted values and inline comments.
    pub fn get_config_value(&self, key: &str) -> Option<String> {
        // Cache parsed values per key: `forward` re-reads the same handful of
        // config keys on every request, and each miss rescans the whole YAML.
        if let Some(v) = self
            .config_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(key)
            .cloned()
        {
            return v;
        }
        let v = Self::scan_config_value(&self.config_yaml, key);
        self.config_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_string(), v.clone());
        v
    }

    /// Raw YAML text scan for a configuration value (the original
    /// [`Self::get_config_value`] implementation).
    fn scan_config_value(config_yaml: &str, key: &str) -> Option<String> {
        let prefix = format!("{key}:");
        let lines: Vec<&str> = config_yaml.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            let trimmed = line.trim();
            if let Some(val_start) = trimmed.strip_prefix(prefix.as_str()) {
                let val = val_start.trim();
                if val.is_empty() {
                    // YAML block list form: "dim-vocabs:" followed by indented "- item"
                    let mut items: Vec<String> = Vec::new();
                    for next in &lines[idx + 1..] {
                        if next.trim_start().starts_with('-') {
                            let item = next.trim_start().trim_start_matches('-').trim();
                            if !item.is_empty() {
                                items.push(item.to_string());
                            }
                        } else {
                            break;
                        }
                    }
                    if items.is_empty() {
                        return None;
                    }
                    return Some(items.join(","));
                }
                // Strip trailing comment (everything after first # outside quotes)
                let mut in_quote = false;
                let mut comment_split = None;
                for (i, ch) in val.char_indices() {
                    match ch {
                        '"' | '\'' => in_quote = !in_quote,
                        '#' if !in_quote => {
                            comment_split = Some(i);
                            break;
                        }
                        _ => {}
                    }
                }
                let out = match comment_split {
                    Some(pos) => val[..pos].trim(),
                    None => val,
                };
                // Strip surrounding quotes
                let out = out
                    .strip_prefix('"')
                    .and_then(|s| s.strip_suffix('"'))
                    .or_else(|| out.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')))
                    .unwrap_or(out);
                return Some(out.to_string());
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — little-endian binary reading
// ---------------------------------------------------------------------------

fn ensure_remaining(buf: &[u8], pos: usize, needed: usize) -> Result<(), EngineError> {
    if pos.checked_add(needed).map_or(true, |end| end > buf.len()) {
        return Err(EngineError::Format(format!(
            "Unexpected end of file: need {} bytes at offset {}, file has {} bytes",
            needed,
            pos,
            buf.len()
        )));
    }
    Ok(())
}

fn read_u64_le(buf: &[u8], pos: &mut usize) -> Result<u64, EngineError> {
    ensure_remaining(buf, *pos, 8)?;
    let val = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
    *pos += 8;
    Ok(val)
}

fn read_i32_le(buf: &[u8], pos: &mut usize) -> Result<i32, EngineError> {
    ensure_remaining(buf, *pos, 4)?;
    let val = i32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(val)
}

fn read_f32_le(buf: &[u8], pos: &mut usize) -> Result<f32, EngineError> {
    ensure_remaining(buf, *pos, 4)?;
    let val = f32::from_le_bytes(buf[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(val)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_from_bytes_roundtrip() {
        // Construct a minimal valid binary model file in memory:
        // version: 1 (u64)
        // num_headers: 1 (u64)
        // header 0:
        //   name_len: 4 (u64)
        //   type: 3 (IntGemm8) (u64)
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // num_headers
        buf.extend_from_slice(&5u64.to_le_bytes()); // name_len (including \0)
        buf.extend_from_slice(&0x4101u64.to_le_bytes()); // type = IntGemm8
        buf.extend_from_slice(&2u64.to_le_bytes()); // shape_len
        buf.extend_from_slice(&8u64.to_le_bytes()); // data_len = 4 bytes int8 + 4 bytes f32
        buf.extend_from_slice(b"Wemb\0");
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        // uint64 offset padding
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Data:
        buf.extend_from_slice(&[10i8 as u8, 20i8 as u8, 30i8 as u8, 40i8 as u8]);
        buf.extend_from_slice(&2.0f32.to_le_bytes());

        let model = Model::from_bytes(&buf).expect("load from bytes");
        let q = model.quant8.get("Wemb").expect("has Wemb");
        assert_eq!(&q.data[..], &[10, 20, 30, 40]);
        assert_eq!(q.quant_mult, 2.0);
        let mut row = vec![0.0f32; 2];
        assert!(model.dequant_row_into("Wemb", 0, &mut row));
        assert_eq!(row, vec![5.0, 10.0]);
        assert!(model.dequant_row_into("Wemb", 1, &mut row));
        assert_eq!(row, vec![15.0, 20.0]);

        // Explicit zero-copy shared bytes loading test
        let shared: std::sync::Arc<[u8]> = std::sync::Arc::from(buf.as_slice());
        let shared_model = Model::from_shared_bytes(shared).expect("load from shared bytes");
        let shared_q = shared_model.quant8.get("Wemb").expect("has Wemb");
        assert_eq!(&shared_q.data[..], &[10, 20, 30, 40]);
    }

    #[test]
    fn model_from_bytes_intgemm16_roundtrip() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u64.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // num_headers
        buf.extend_from_slice(&5u64.to_le_bytes()); // name_len (including \0)
        buf.extend_from_slice(&0x4102u64.to_le_bytes()); // type = IntGemm16
        buf.extend_from_slice(&2u64.to_le_bytes()); // shape_len
        buf.extend_from_slice(&12u64.to_le_bytes()); // data_len = 4 elements * 2 bytes + 4 bytes f32 = 12
        buf.extend_from_slice(b"Wemb\0");
        buf.extend_from_slice(&2i32.to_le_bytes());
        buf.extend_from_slice(&2i32.to_le_bytes());
        // uint64 offset padding
        buf.extend_from_slice(&0u64.to_le_bytes());
        // Data: 4 x i16 values followed by f32 quantMult
        buf.extend_from_slice(&100i16.to_le_bytes());
        buf.extend_from_slice(&200i16.to_le_bytes());
        buf.extend_from_slice(&300i16.to_le_bytes());
        buf.extend_from_slice(&400i16.to_le_bytes());
        buf.extend_from_slice(&2.0f32.to_le_bytes());

        let model = Model::from_bytes(&buf).expect("load intgemm16 from bytes");
        let t = model.get("Wemb").expect("has Wemb tensor");
        assert_eq!(&t.data[..], &[50.0, 100.0, 150.0, 200.0]);
    }
}
