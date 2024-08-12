use std::iter::zip;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::engine::{
    tables::{self, Mul128, Multiply128lutT},
    utils, Engine, GfElement, GF_ORDER,
};

// ======================================================================
// Ssse3 - PUBLIC

/// Optimized [`Engine`] using SSSE3 instructions.
///
/// [`Ssse3`] is an optimized engine that follows the same algorithm as
/// [`NoSimd`] but takes advantage of the x86 SSSE3 SIMD instructions.
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone)]
pub struct Ssse3 {
    mul128: &'static Mul128,
}

impl Ssse3 {
    /// Creates new [`Ssse3`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = tables::initialize_mul128();

        Self { mul128 }
    }
}

impl Engine for Ssse3 {
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.fftb_128(x_chunk, y_chunk, log_m);
        }
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.ifftb_128(x_chunk, y_chunk, log_m);
        }
    }

    #[inline(always)]
    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_ssse3(x, log_m);
        }
    }

    #[inline(always)]
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_ssse3(erasures, truncated_size) }
    }
}

// ======================================================================
// Ssse3 - IMPL Default

impl Default for Ssse3 {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Ssse3 - PRIVATE
//
//

impl Ssse3 {
    #[target_feature(enable = "ssse3")]
    unsafe fn mul_ssse3(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr() as *mut __m128i;
            unsafe {
                let x0_lo = _mm_loadu_si128(x_ptr);
                let x1_lo = _mm_loadu_si128(x_ptr.add(1));
                let x0_hi = _mm_loadu_si128(x_ptr.add(2));
                let x1_hi = _mm_loadu_si128(x_ptr.add(3));
                let (prod0_lo, prod0_hi) = Self::mul_128(x0_lo, x0_hi, lut);
                let (prod1_lo, prod1_hi) = Self::mul_128(x1_lo, x1_hi, lut);
                _mm_storeu_si128(x_ptr, prod0_lo);
                _mm_storeu_si128(x_ptr.add(1), prod1_lo);
                _mm_storeu_si128(x_ptr.add(2), prod0_hi);
                _mm_storeu_si128(x_ptr.add(3), prod1_hi);
            }
        }
    }

    // Impelemntation of LEO_MUL_128
    #[inline(always)]
    fn mul_128(value_lo: __m128i, value_hi: __m128i, lut: &Multiply128lutT) -> (__m128i, __m128i) {
        let mut prod_lo: __m128i;
        let mut prod_hi: __m128i;

        unsafe {
            let t0_lo = _mm_loadu_si128(&lut.lo[0] as *const u128 as *const __m128i);
            let t1_lo = _mm_loadu_si128(&lut.lo[1] as *const u128 as *const __m128i);
            let t2_lo = _mm_loadu_si128(&lut.lo[2] as *const u128 as *const __m128i);
            let t3_lo = _mm_loadu_si128(&lut.lo[3] as *const u128 as *const __m128i);

            let t0_hi = _mm_loadu_si128(&lut.hi[0] as *const u128 as *const __m128i);
            let t1_hi = _mm_loadu_si128(&lut.hi[1] as *const u128 as *const __m128i);
            let t2_hi = _mm_loadu_si128(&lut.hi[2] as *const u128 as *const __m128i);
            let t3_hi = _mm_loadu_si128(&lut.hi[3] as *const u128 as *const __m128i);

            let clr_mask = _mm_set1_epi8(0x0f);

            let data_0 = _mm_and_si128(value_lo, clr_mask);
            prod_lo = _mm_shuffle_epi8(t0_lo, data_0);
            prod_hi = _mm_shuffle_epi8(t0_hi, data_0);

            let data_1 = _mm_and_si128(_mm_srli_epi64(value_lo, 4), clr_mask);
            prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t1_lo, data_1));
            prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t1_hi, data_1));

            let data_0 = _mm_and_si128(value_hi, clr_mask);
            prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t2_lo, data_0));
            prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t2_hi, data_0));

            let data_1 = _mm_and_si128(_mm_srli_epi64(value_hi, 4), clr_mask);
            prod_lo = _mm_xor_si128(prod_lo, _mm_shuffle_epi8(t3_lo, data_1));
            prod_hi = _mm_xor_si128(prod_hi, _mm_shuffle_epi8(t3_hi, data_1));
        }

        (prod_lo, prod_hi)
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_128
    #[inline(always)]
    fn muladd_128(
        mut x_lo: __m128i,
        mut x_hi: __m128i,
        y_lo: __m128i,
        y_hi: __m128i,
        lut: &Multiply128lutT,
    ) -> (__m128i, __m128i) {
        let (prod_lo, prod_hi) = Self::mul_128(y_lo, y_hi, lut);
        unsafe {
            x_lo = _mm_xor_si128(x_lo, prod_lo);
            x_hi = _mm_xor_si128(x_hi, prod_hi);
        }
        (x_lo, x_hi)
    }
}

// ======================================================================
// Ssse3 - PRIVATE - FFT (fast Fourier transform)

impl Ssse3 {
    // Implementation of LEO_FFTB_128
    #[inline(always)]
    fn fftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr = x.as_mut_ptr() as *mut __m128i;
        let y_ptr = y.as_mut_ptr() as *mut __m128i;
        unsafe {
            let mut x0_lo = _mm_loadu_si128(x_ptr);
            let mut x1_lo = _mm_loadu_si128(x_ptr.add(1));
            let mut x0_hi = _mm_loadu_si128(x_ptr.add(2));
            let mut x1_hi = _mm_loadu_si128(x_ptr.add(3));

            let mut y0_lo = _mm_loadu_si128(y_ptr);
            let mut y1_lo = _mm_loadu_si128(y_ptr.add(1));
            let mut y0_hi = _mm_loadu_si128(y_ptr.add(2));
            let mut y1_hi = _mm_loadu_si128(y_ptr.add(3));

            (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
            (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

            _mm_storeu_si128(x_ptr, x0_lo);
            _mm_storeu_si128(x_ptr.add(1), x1_lo);
            _mm_storeu_si128(x_ptr.add(2), x0_hi);
            _mm_storeu_si128(x_ptr.add(3), x1_hi);

            y0_lo = _mm_xor_si128(y0_lo, x0_lo);
            y1_lo = _mm_xor_si128(y1_lo, x1_lo);
            y0_hi = _mm_xor_si128(y0_hi, x0_hi);
            y1_hi = _mm_xor_si128(y1_hi, x1_hi);

            _mm_storeu_si128(y_ptr, y0_lo);
            _mm_storeu_si128(y_ptr.add(1), y1_lo);
            _mm_storeu_si128(y_ptr.add(2), y0_hi);
            _mm_storeu_si128(y_ptr.add(3), y1_hi);
        }
    }
}

// ======================================================================
// Ssse3 - PRIVATE - IFFT (inverse fast Fourier transform)

impl Ssse3 {
    // Implementation of LEO_IFFTB_128
    #[inline(always)]
    fn ifftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr = x.as_mut_ptr() as *mut __m128i;
        let y_ptr = y.as_mut_ptr() as *mut __m128i;

        unsafe {
            let mut x0_lo = _mm_loadu_si128(x_ptr);
            let mut x1_lo = _mm_loadu_si128(x_ptr.add(1));
            let mut x0_hi = _mm_loadu_si128(x_ptr.add(2));
            let mut x1_hi = _mm_loadu_si128(x_ptr.add(3));

            let mut y0_lo = _mm_loadu_si128(y_ptr);
            let mut y1_lo = _mm_loadu_si128(y_ptr.add(1));
            let mut y0_hi = _mm_loadu_si128(y_ptr.add(2));
            let mut y1_hi = _mm_loadu_si128(y_ptr.add(3));

            y0_lo = _mm_xor_si128(y0_lo, x0_lo);
            y1_lo = _mm_xor_si128(y1_lo, x1_lo);
            y0_hi = _mm_xor_si128(y0_hi, x0_hi);
            y1_hi = _mm_xor_si128(y1_hi, x1_hi);

            _mm_storeu_si128(y_ptr, y0_lo);
            _mm_storeu_si128(y_ptr.add(1), y1_lo);
            _mm_storeu_si128(y_ptr.add(2), y0_hi);
            _mm_storeu_si128(y_ptr.add(3), y1_hi);

            (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
            (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

            _mm_storeu_si128(x_ptr, x0_lo);
            _mm_storeu_si128(x_ptr.add(1), x1_lo);
            _mm_storeu_si128(x_ptr.add(2), x0_hi);
            _mm_storeu_si128(x_ptr.add(3), x1_hi);
        }
    }
}

// ======================================================================
// Ssse3 - PRIVATE - Evaluate polynomial

impl Ssse3 {
    #[target_feature(enable = "ssse3")]
    unsafe fn eval_poly_ssse3(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly_fallback(erasures, truncated_size)
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
