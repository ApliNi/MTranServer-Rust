//! Tensor abstractions and computational engine.
//!
//! Provides the core [`Tensor2`] (2D float tensor) and [`Quant8Tensor`]
//! (quantized 8-bit weight matrix) abstractions, high-level matrix multiplication
//! entry points, activation functions, and layer normalization routines.

pub mod ops;
pub mod simd;

#[cfg(test)]
mod tests;

pub use ops::*;
pub use simd::*;

/// Runtime math: `y = (x_q8 · w_q8) / (quant_mult_a * quant_mult)` where
/// `x_q8 = round(x * quant_mult_a)` clamps to `[-127, 127]` (matching the
/// marian-fork intgemm semantics), and `quant_mult` is the per-tensor scalar
/// stored in the weight segment trailer.  Because the weights are identical to
/// the dequantized f32 versions (modulo the file's own 8-bit precision),
/// results match `x @ (w_q8 / quant_mult)` up to activation quantization.
use std::sync::Arc;

/// Storage backend for quantized 8-bit tensor data: either owned heap memory
/// or memory-mapped read-only bytes for zero-copy loading.
#[derive(Debug, Clone)]
pub enum Quant8Data {
    Owned(Vec<i8>),
    Mmap {
        mmap: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
    Shared {
        data: Arc<[u8]>,
        offset: usize,
        len: usize,
    },
}

impl std::ops::Deref for Quant8Data {
    type Target = [i8];

    #[inline]
    fn deref(&self) -> &[i8] {
        match self {
            Quant8Data::Owned(v) => v.as_slice(),
            Quant8Data::Mmap { mmap, offset, len } => {
                let bytes = &mmap[*offset..*offset + *len];
                // Safety: i8 and u8 have identical layout (1 byte size and alignment).
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, *len) }
            }
            Quant8Data::Shared { data, offset, len } => {
                let bytes = &data[*offset..*offset + *len];
                // Safety: i8 and u8 have identical layout (1 byte size and alignment).
                unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const i8, *len) }
            }
        }
    }
}

impl From<Vec<i8>> for Quant8Data {
    #[inline]
    fn from(v: Vec<i8>) -> Self {
        Quant8Data::Owned(v)
    }
}

impl Default for Quant8Data {
    #[inline]
    fn default() -> Self {
        Quant8Data::Owned(Vec::new())
    }
}

impl AsRef<[i8]> for Quant8Data {
    #[inline]
    fn as_ref(&self) -> &[i8] {
        self
    }
}

impl std::borrow::Borrow<[i8]> for Quant8Data {
    #[inline]
    fn borrow(&self) -> &[i8] {
        self
    }
}

impl PartialEq for Quant8Data {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl Eq for Quant8Data {}

#[derive(Debug, Clone)]
pub struct Quant8Tensor {
    /// Row-major layout of `data`: `[out, in]` for 2-D non-`Wemb` weights,
    /// `[vocab, d_model]` for `Wemb`.
    pub shape: Vec<usize>,
    pub data: Quant8Data,
    /// Weight quantizer `quant_mult` (trailer scalar in the binary segment).
    pub quant_mult: f32,
    /// Per-row `128·Σw` correction for the unsigned-x maddubs kernels
    /// (`row_sums[j] = 128 · Σ_t data[j][t]`). Precomputed once at load time;
    /// `None` for tensors whose layout is not a 2-D weight matrix.
    pub row_sums: Option<Vec<i32>>,
}

impl Quant8Tensor {
    #[inline]
    pub fn new(
        shape: Vec<usize>,
        data: impl Into<Quant8Data>,
        quant_mult: f32,
        row_sums: Option<Vec<i32>>,
    ) -> Self {
        Self {
            shape,
            data: data.into(),
            quant_mult,
            row_sums,
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(0)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Tensor2 {
    pub shape: [usize; 2],
    pub data: Vec<f32>,
}

impl Tensor2 {
    #[inline]
    pub fn new(rows: usize, cols: usize, data: Vec<f32>) -> Self {
        Self {
            shape: [rows, cols],
            data,
        }
    }

    #[inline]
    pub fn zeros(shape: [usize; 2]) -> Self {
        let len = shape[0] * shape[1];
        Self {
            shape,
            data: vec![0.0; len],
        }
    }

    #[inline]
    pub fn from_vec(shape: [usize; 2], data: Vec<f32>) -> Self {
        Self {
            shape,
            data,
        }
    }

    #[inline]
    pub fn rows(&self) -> usize {
        self.shape[0]
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.shape[1]
    }

    /// Sets a 2D shape on `self` without heap allocation.
    #[inline]
    pub fn set_shape_2d(&mut self, m: usize, n: usize) {
        self.shape = [m, n];
    }

    /// Sets shape from a 2D array without heap allocation.
    #[inline]
    pub fn set_shape(&mut self, shape: [usize; 2]) {
        self.shape = shape;
    }
    /// Overwrites `self.data` with the contents of `src` without zero-initializing.
    #[inline]
    pub fn copy_from_slice(&mut self, src: &[f32]) {
        self.data.clear();
        self.data.extend_from_slice(src);
    }

    /// Prepares `self.data` to hold `len` elements, resizing with default values if needed.
    ///
    /// # Safety
    /// Calling this is safe as memory is initialized with default values. Marked unsafe for
    /// compatibility with existing callers.
    pub unsafe fn prepare_uninit(&mut self, len: usize) {
        prepare_uninit_vec(&mut self.data, len);
    }

    /// Buffer-reusing variant of [`Self::matmul_biased`]: resizes and
    /// overwrites `out`'s data so hot callers (the decode loop) avoid
    /// per-step output allocations. Numerically identical to
    /// [`Self::matmul_biased`].
    #[inline]
    pub fn matmul_biased_into(&self, rhs: &Tensor2, bias: Option<&Tensor2>, out: &mut Tensor2) {
        self.matmul_biased_relu_generic_into(rhs, bias, out, false);
    }

    /// Buffer-reusing matmul with ReLU fused into the kernels' store step:
    /// `out = max(0, self @ rhs + bias)`. Numerically identical to
    /// [`Self::matmul_biased_into`] followed by an elementwise `max(0, x)`
    /// pass (the same `> 0.0` predicate), minus that extra traversal.
    #[inline]
    pub fn matmul_biased_relu_into(
        &self,
        rhs: &Tensor2,
        bias: Option<&Tensor2>,
        out: &mut Tensor2,
    ) {
        self.matmul_biased_relu_generic_into(rhs, bias, out, true);
    }

    #[inline]
    fn matmul_biased_relu_generic_into(
        &self,
        rhs: &Tensor2,
        bias: Option<&Tensor2>,
        out: &mut Tensor2,
        relu: bool,
    ) {
        let m = self.shape[0];
        let k = self.shape[1];
        let n = rhs.shape[1];
        debug_assert_eq!(
            k, rhs.shape[0],
            "matmul: inner dimensions must match, self {:?} vs rhs {:?}",
            self.shape, rhs.shape
        );
        if let Some(b) = bias {
            debug_assert_eq!(b.data.len(), n, "matmul: bias {} != n {}", b.data.len(), n);
        }
        // SAFETY: capacity is guaranteed >= m * n, and the kernels overwrite all m * n elements.
        unsafe {
            prepare_uninit_vec(&mut out.data, m * n);
        }
        // SAFETY: the dispatch fn was selected from the CPU feature flags of
        // the running machine; slices are valid and freshly allocated.
        let bias_slice = bias.map(|b| b.data.as_slice());
        if m == 1 {
            // Single-row path: reuse the activation broadcast across all
            // columns (see `matmul_row_f32_dispatch`).
            unsafe {
                matmul_row_f32_dispatch()(
                    &self.data,
                    &rhs.data,
                    k,
                    n,
                    bias_slice,
                    &mut out.data,
                    relu,
                )
            };
        } else {
            unsafe {
                matmul_f32_split(
                    &self.data,
                    &rhs.data,
                    m,
                    k,
                    n,
                    bias_slice,
                    &mut out.data,
                    relu,
                )
            };
        }
        out.shape = [m, n];
    }

    /// `self @ rhs` with an optional broadcast bias added to every output row.
    ///
    /// `bias` must describe a `[1, n]` (or `[n]`) tensor; it is fused into the
    /// matmul kernels' store step, so `linear` avoids the separate bias clone
    /// and an extra elementwise pass/alloc of the old `x.matmul(w).add(&b)`.
    #[inline]
    pub fn matmul_biased(&self, rhs: &Tensor2, bias: Option<&Tensor2>) -> Tensor2 {
        let mut out = Tensor2 {
            shape: [0, 0],
            data: Vec::new(),
        };
        self.matmul_biased_into(rhs, bias, &mut out);
        out
    }

    /// Quantized matmul with an optional broadcast bias fused into the store.
    ///
    /// `self` is `[m, k]` f32 activations.  `w` is `[n, k]` i8 weights
    /// (row-major, one output neuron's weight vector per row — the on-disk
    /// layout for non-Wemb tensors).  Activations are quantised on-the-fly
    /// with `qa`, the result is dequantised by `qa * w.quant_mult`.
    ///
    /// Uses int32 accumulation (no saturation) for correctness.
    #[inline]
    pub fn quantized_matmul_biased(
        &self,
        w: &Quant8Tensor,
        qa: f32,
        bias: Option<&Tensor2>,
    ) -> Tensor2 {
        let mut out = Tensor2 {
            shape: [0, 0],
            data: Vec::new(),
        };
        self.quantized_matmul_biased_into(w, qa, bias, &mut out);
        out
    }

    /// Buffer-reusing variant of [`Self::quantized_matmul_biased`]: resizes
    /// and overwrites `out`'s data so hot callers (the decode loop) avoid
    /// per-step output allocations. Numerically identical to
    /// [`Self::quantized_matmul_biased`].
    #[inline]
    pub fn quantized_matmul_biased_into(
        &self,
        w: &Quant8Tensor,
        qa: f32,
        bias: Option<&Tensor2>,
        out: &mut Tensor2,
    ) {
        self.quantized_matmul_relu_generic_into(w, qa, bias, out, false);
    }

    /// Buffer-reusing quantized matmul with ReLU fused into the kernels'
    /// dequantize-and-store step: `out = max(0, x·W + bias)`. Numerically
    /// identical to [`Self::quantized_matmul_biased_into`] followed by an
    /// elementwise `max(0, x)` pass, minus that extra traversal.
    #[inline]
    pub fn quantized_matmul_biased_relu_into(
        &self,
        w: &Quant8Tensor,
        qa: f32,
        bias: Option<&Tensor2>,
        out: &mut Tensor2,
    ) {
        self.quantized_matmul_relu_generic_into(w, qa, bias, out, true);
    }
    /// Performs multiple quantized matmuls with the SAME input `self` and same `qa`,
    /// eliminating redundant input quantization into `I8_ACT_BUF`.
    #[allow(clippy::type_complexity)]
    pub fn quantized_matmul_multi_into(
        &self,
        ops: &mut [(&Quant8Tensor, Option<&Tensor2>, &mut Tensor2)],
        qa: f32,
    ) {
        if ops.is_empty() {
            return;
        }
        let m = self.shape[0];
        let k = self.shape[1];

        for (w, bias, out) in ops.iter_mut() {
            debug_assert_eq!(w.shape.len(), 2, "quantized_matmul_multi: w must be 2D");
            let n = w.shape[0];
            debug_assert_eq!(k, w.shape[1], "quantized_matmul_multi: inner dim mismatch");
            debug_assert_eq!(w.data.len(), n * k, "quantized_matmul_multi: w data size mismatch");
            if let Some(b) = bias {
                debug_assert_eq!(b.data.len(), n, "quantized_matmul_multi: bias data size mismatch");
            }
            unsafe {
                prepare_uninit_vec(&mut out.data, m * n);
            }
        }

        I8_ACT_BUF.with(|cell| {
            let mut opt_buf = cell.try_borrow_mut();
            let mut local_buf;
            let buf_ref = match &mut opt_buf {
                Ok(b) => {
                    i8_quantize_into(&self.data, qa, b);
                    &**b
                }
                Err(_) => {
                    local_buf = Vec::new();
                    i8_quantize_into(&self.data, qa, &mut local_buf);
                    &local_buf[..]
                }
            };

            #[cfg(target_arch = "x86_64")]
            let handled = if crate::utils::arch::has_avxvnni() {
                unsafe {
                    simd::vnni::with_vnni_prebiased(buf_ref, m, k, |xb_slice| {
                        for (w, bias, out) in ops.iter_mut() {
                            let n = w.shape[0];
                            let denom = qa * w.quant_mult;
                            let bias_slice = bias.map(|b| b.data.as_slice());
                            let row_sums = w.row_sums.as_deref();
                            if m == 1 {
                                simd::vnni::matmul_row_i8_avxvnni_prebiased(
                                    xb_slice,
                                    buf_ref,
                                    &w.data,
                                    k,
                                    n,
                                    bias_slice,
                                    row_sums,
                                    &mut out.data,
                                    denom,
                                    false,
                                );
                            } else {
                                simd::vnni::matmul_i8_avxvnni_prebiased_split(
                                    xb_slice,
                                    buf_ref,
                                    &w.data,
                                    m,
                                    k,
                                    n,
                                    bias_slice,
                                    row_sums,
                                    &mut out.data,
                                    denom,
                                    false,
                                );
                            }
                        }
                    });
                }
                true
            } else if crate::utils::arch::has_avx2() {
                unsafe {
                    simd::avx2::with_avx2_prebiased(buf_ref, m, k, |lo_slice, hi_slice| {
                        for (w, bias, out) in ops.iter_mut() {
                            let n = w.shape[0];
                            let denom = qa * w.quant_mult;
                            let bias_slice = bias.map(|b| b.data.as_slice());
                            let row_sums = w.row_sums.as_deref();
                            if m == 1 {
                                simd::avx2::matmul_row_i8_avx2_prebiased(
                                    lo_slice,
                                    hi_slice,
                                    buf_ref,
                                    &w.data,
                                    k,
                                    n,
                                    bias_slice,
                                    row_sums,
                                    &mut out.data,
                                    denom,
                                    false,
                                );
                            } else {
                                simd::avx2::matmul_i8_avx2_prebiased_split(
                                    lo_slice,
                                    hi_slice,
                                    buf_ref,
                                    &w.data,
                                    m,
                                    k,
                                    n,
                                    bias_slice,
                                    row_sums,
                                    &mut out.data,
                                    denom,
                                    false,
                                );
                            }
                        }
                    });
                }
                true
            } else {
                false
            };

            #[cfg(not(target_arch = "x86_64"))]
            let handled = false;

            if !handled {
                if m == 1 {
                    let row_fn = matmul_row_i8_dispatch();
                    for (w, bias, out) in ops.iter_mut() {
                        let n = w.shape[0];
                        let denom = qa * w.quant_mult;
                        let bias_slice = bias.map(|b| b.data.as_slice());
                        let row_sums = w.row_sums.as_deref();
                        unsafe {
                            row_fn(
                                buf_ref,
                                &w.data,
                                k,
                                n,
                                bias_slice,
                                row_sums,
                                &mut out.data,
                                denom,
                                false,
                            );
                        }
                    }
                } else {
                    for (w, bias, out) in ops.iter_mut() {
                        let n = w.shape[0];
                        let denom = qa * w.quant_mult;
                        let bias_slice = bias.map(|b| b.data.as_slice());
                        let row_sums = w.row_sums.as_deref();
                        unsafe {
                            matmul_i8_split(
                                buf_ref,
                                &w.data,
                                m,
                                k,
                                n,
                                bias_slice,
                                row_sums,
                                &mut out.data,
                                denom,
                                false,
                            );
                        }
                    }
                }
            }
        });

        for (w, _, out) in ops.iter_mut() {
            out.shape = [m, w.shape[0]];
        }
    }

    /// Performs two quantized matmuls with the SAME input `self` and same `qa`,
    /// eliminating redundant input quantization into `I8_ACT_BUF`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn quantized_matmul_pair_into(
        &self,
        w1: &Quant8Tensor,
        bias1: Option<&Tensor2>,
        out1: &mut Tensor2,
        w2: &Quant8Tensor,
        bias2: Option<&Tensor2>,
        out2: &mut Tensor2,
        qa: f32,
    ) {
        let mut ops = [(w1, bias1, out1), (w2, bias2, out2)];
        self.quantized_matmul_multi_into(&mut ops, qa);
    }

    /// Performs three quantized matmuls with the SAME input `self` and same `qa`,
    /// eliminating redundant input quantization into `I8_ACT_BUF`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn quantized_matmul_triple_into(
        &self,
        w1: &Quant8Tensor,
        bias1: Option<&Tensor2>,
        out1: &mut Tensor2,
        w2: &Quant8Tensor,
        bias2: Option<&Tensor2>,
        out2: &mut Tensor2,
        w3: &Quant8Tensor,
        bias3: Option<&Tensor2>,
        out3: &mut Tensor2,
        qa: f32,
    ) {
        let mut ops = [(w1, bias1, out1), (w2, bias2, out2), (w3, bias3, out3)];
        self.quantized_matmul_multi_into(&mut ops, qa);
    }

    #[inline]
    fn quantized_matmul_relu_generic_into(
        &self,
        w: &Quant8Tensor,
        qa: f32,
        bias: Option<&Tensor2>,
        out: &mut Tensor2,
        relu: bool,
    ) {
        debug_assert_eq!(w.shape.len(), 2, "quantized_matmul: w must be 2D");
        let m = self.shape[0];
        let k = self.shape[1];
        let n = w.shape[0];
        debug_assert_eq!(k, w.shape[1], "quantized_matmul: inner dim mismatch");
        debug_assert_eq!(
            w.data.len(),
            n * k,
            "quantized_matmul: w data size mismatch"
        );
        if let Some(b) = bias {
            debug_assert_eq!(
                b.data.len(),
                n,
                "quantized_matmul: bias {} != n {}",
                b.data.len(),
                n
            );
        }

        // Quantise activations to i8 (clamped [-127, 127]). The single-row
        // path quantises into a thread-local buffer, so the decode loop's
        // several quantized matmuls per step reuse one allocation instead of
        // materializing a fresh `Vec<i8>` each time; the multi-row path
        // quantises into a per-call vector as before.
        let denom = qa * w.quant_mult;
        // SAFETY: capacity is guaranteed >= m * n, and the kernels overwrite all elements.
        unsafe {
            prepare_uninit_vec(&mut out.data, m * n);
        }
        let row_sums = w.row_sums.as_deref();
        let bias_slice = bias.map(|b| b.data.as_slice());
        I8_ACT_BUF.with(|cell| {
            let mut opt_buf = cell.try_borrow_mut();
            let mut local_buf;
            let buf_ref = match &mut opt_buf {
                Ok(b) => {
                    i8_quantize_into(&self.data, qa, b);
                    &**b
                }
                Err(_) => {
                    local_buf = Vec::new();
                    i8_quantize_into(&self.data, qa, &mut local_buf);
                    &local_buf[..]
                }
            };
            if m == 1 {
                // SAFETY: the dispatch fn was selected from the CPU feature
                // flags of the running machine; the activation row and the
                // freshly resized output are valid, distinct slices.
                unsafe {
                    matmul_row_i8_dispatch()(
                        buf_ref,
                        &w.data,
                        k,
                        n,
                        bias_slice,
                        row_sums,
                        &mut out.data,
                        denom,
                        relu,
                    )
                };
            } else {
                // SAFETY: the dispatch fn was selected from the CPU feature flags
                // of the running machine; slices are valid and freshly resized.
                unsafe {
                    matmul_i8_split(
                        buf_ref,
                        &w.data,
                        m,
                        k,
                        n,
                        bias_slice,
                        row_sums,
                        &mut out.data,
                        denom,
                        relu,
                    )
                };
            }
        });
        out.shape = [m, n];
    }

    #[inline]
    pub fn layer_norm(&self, scale: &Tensor2, bias: &Tensor2, eps: f32) -> Tensor2 {
        ops::layer_norm(self, scale, bias, eps)
    }

    #[inline]
    pub fn layer_norm_into(&self, scale: &Tensor2, bias: &Tensor2, eps: f32, out: &mut Tensor2) {
        ops::layer_norm_into(self, scale, bias, eps, out)
    }
    #[inline]
    pub fn fused_add_layer_norm_into(
        &self,
        other: &Tensor2,
        scale: &Tensor2,
        bias: &Tensor2,
        eps: f32,
        out: &mut Tensor2,
    ) {
        ops::fused_add_layer_norm_into(self, other, scale, bias, eps, out)
    }

    #[inline]
    pub fn fused_add_layer_norm_in_place(
        &mut self,
        other: &Tensor2,
        scale: &Tensor2,
        bias: &Tensor2,
        eps: f32,
    ) {
        ops::fused_add_layer_norm_in_place(self, other, scale, bias, eps)
    }

    #[inline]
    pub fn add(&self, other: &Tensor2) -> Tensor2 {
        ops::add(self, other)
    }

    #[inline]
    pub fn add_into(&self, other: &Tensor2, out: &mut Tensor2) {
        ops::add_into(self, other, out)
    }

    #[inline]
    pub fn softmax(&self) -> Tensor2 {
        ops::softmax(self)
    }

    #[inline]
    pub fn softmax_into(&self, out: &mut Tensor2) {
        ops::softmax_into(self, out)
    }

    #[inline]
    pub fn softmax_inplace(&mut self) {
        ops::softmax_inplace(self)
    }

    #[inline]
    pub fn relu(&self) -> Tensor2 {
        ops::relu(self)
    }

    #[inline]
    pub fn relu_into(&self, out: &mut Tensor2) {
        ops::relu_into(self, out)
    }

    #[inline]
    pub fn relu_inplace(&mut self) {
        ops::relu_inplace(self)
    }

    #[inline]
    pub fn gelu(&self) -> Tensor2 {
        ops::gelu(self)
    }

    #[inline]
    pub fn gelu_into(&self, out: &mut Tensor2) {
        ops::gelu_into(self, out)
    }

    #[inline]
    pub fn gelu_inplace(&mut self) {
        ops::gelu_inplace(self)
    }

    #[inline]
    pub fn sigmoid(&self) -> Tensor2 {
        ops::sigmoid(self)
    }

    #[inline]
    pub fn sigmoid_into(&self, out: &mut Tensor2) {
        ops::sigmoid_into(self, out)
    }

    #[inline]
    pub fn sigmoid_inplace(&mut self) {
        ops::sigmoid_inplace(self)
    }
}

/// Buffer-reusing i8 quantizer: sets len and overwrites `out`.
#[inline]
fn i8_quantize_into(data: &[f32], qa: f32, out: &mut Vec<i8>) {
    let len = data.len();
    // SAFETY: capacity is guaranteed to be >= len, and every element up to len is
    // initialized in `quantize_row`.
    unsafe {
        prepare_uninit_vec(out, len);
    }
    quantize_row(data, qa, out);
}

/// Helper to ensure a `Vec` has at least `len` capacity and length `len`,
/// initializing newly extended elements to default values.
///
/// # Safety
/// Calling this is safe as memory is initialized with default values. Marked unsafe for
/// compatibility with existing callers.
#[inline]
pub(crate) unsafe fn prepare_uninit_vec<T: Copy + Default>(vec: &mut Vec<T>, len: usize) {
    vec.resize(len, T::default());
}
// Per-thread scratch for the single-row quantized matmul's activation
// buffer (`i8_quantize_into`), so the decode loop's several quantized
// matmuls per step share one allocation instead of materializing a fresh
// `Vec<i8>` each time. Threads run one translation at a time, so buffer
// reuse across calls on the same thread is safe.
thread_local! {
    static I8_ACT_BUF: std::cell::RefCell<Vec<i8>> =
        const { std::cell::RefCell::new(Vec::new()) };
}
