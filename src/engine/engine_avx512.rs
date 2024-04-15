use std::iter::zip;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::engine::{
    self,
    tables::{self, Mul128, Multiply128lutT, Skew},
    Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Avx512 - PUBLIC

/// Optimized [`Engine`] using AVX512 instructions.
///
/// [`Avx512`] is an optimized engine that follows the same algorithm as
/// [`NoSimd`] but takes advantage of the x86 avx512f,avx512vl and avx512bw SIMD instructions.
///
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone)]
pub struct Avx512 {
    mul128: &'static Mul128,
    skew: &'static Skew,
}

impl Avx512 {
    /// Creates new [`Avx512`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = tables::initialize_mul128();
        let skew = tables::initialize_skew();

        Self { mul128, skew }
    }
}

impl Engine for Avx512 {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_avx512(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.ifft_private_avx512(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        unsafe {
            self.mul_avx512(x, log_m);
        }
    }

    //    #[inline(always)]
    fn xor(x: &mut [u8], y: &[u8]) {
        let x: &mut [u64] = bytemuck::cast_slice_mut(x);
        let y: &[u64] = bytemuck::cast_slice(y);

        for (x64, y64) in zip(x.iter_mut(), y.iter()) {
            *x64 ^= y64;
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_avx512(erasures, truncated_size) }
    }
}

// ======================================================================
// Avx512 - IMPL Default

impl Default for Avx512 {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Avx512 - PRIVATE
//
//

impl Avx512 {
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    unsafe fn mul_avx512(&self, x: &mut [u8], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        let mut x_chunks_iter = x.chunks_exact_mut(128);
        for chunk in &mut x_chunks_iter {
            let x_ptr = chunk.as_mut_ptr() as *mut __m256i;
            unsafe {
                let mut x0_lo = _mm256_loadu_si256(x_ptr);
                let mut x0_hi = _mm256_loadu_si256(x_ptr.add(1));
                let mut x1_lo = _mm256_loadu_si256(x_ptr.add(2));
                let mut x1_hi = _mm256_loadu_si256(x_ptr.add(3));

                (x0_lo, x0_hi, x1_lo, x1_hi) = Self::mul_512(x0_lo, x0_hi, x1_lo, x1_hi, lut);

                _mm256_storeu_si256(x_ptr, x0_lo);
                _mm256_storeu_si256(x_ptr.add(1), x0_hi);
                _mm256_storeu_si256(x_ptr.add(2), x1_lo);
                _mm256_storeu_si256(x_ptr.add(3), x1_hi);
            }
        }

        // 64 byte left?
        if let Ok(chunk) = TryInto::<&mut [u8; 64]>::try_into(x_chunks_iter.into_remainder()) {
            let x_ptr = chunk.as_mut_ptr() as *mut __m256i;
            unsafe {
                let mut x_lo = _mm256_loadu_si256(x_ptr);
                let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));
                let zero = _mm256_setzero_si256();

                (x_lo, x_hi, _, _) = Self::mul_512(x_lo, x_hi, zero, zero, lut);

                _mm256_storeu_si256(x_ptr, x_lo);
                _mm256_storeu_si256(x_ptr.add(1), x_hi);
            }
        }
    }

    // Impelemntation of LEO_MUL_256
    #[inline(always)]
    fn mul_512(
        value0_lo: __m256i,
        value0_hi: __m256i,
        value1_lo: __m256i,
        value1_hi: __m256i,
        lut: &Multiply128lutT,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            let value_lo = _mm512_inserti64x4(_mm512_castsi256_si512(value0_lo), value1_lo, 1);
            let value_hi = _mm512_inserti64x4(_mm512_castsi256_si512(value0_hi), value1_hi, 1);

            let t0_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.lo[0] as *const u128 as *const __m128i,
            ));
            let t1_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.lo[1] as *const u128 as *const __m128i,
            ));
            let t2_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.lo[2] as *const u128 as *const __m128i,
            ));
            let t3_lo = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.lo[3] as *const u128 as *const __m128i,
            ));

            let t0_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.hi[0] as *const u128 as *const __m128i,
            ));
            let t1_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.hi[1] as *const u128 as *const __m128i,
            ));
            let t2_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.hi[2] as *const u128 as *const __m128i,
            ));
            let t3_hi = _mm512_broadcast_i32x4(_mm_loadu_si128(
                &lut.hi[3] as *const u128 as *const __m128i,
            ));

            let clr_mask = _mm512_set1_epi8(0x0f);

            let data_0 = _mm512_and_si512(value_lo, clr_mask);
            let mut prod_lo = _mm512_shuffle_epi8(t0_lo, data_0);
            let mut prod_hi = _mm512_shuffle_epi8(t0_hi, data_0);

            let data_1 = _mm512_and_si512(_mm512_srli_epi64(value_lo, 4), clr_mask);
            prod_lo = _mm512_xor_si512(prod_lo, _mm512_shuffle_epi8(t1_lo, data_1));
            prod_hi = _mm512_xor_si512(prod_hi, _mm512_shuffle_epi8(t1_hi, data_1));

            let data_0 = _mm512_and_si512(value_hi, clr_mask);
            prod_lo = _mm512_xor_si512(prod_lo, _mm512_shuffle_epi8(t2_lo, data_0));
            prod_hi = _mm512_xor_si512(prod_hi, _mm512_shuffle_epi8(t2_hi, data_0));

            let data_1 = _mm512_and_si512(_mm512_srli_epi64(value_hi, 4), clr_mask);
            prod_lo = _mm512_xor_si512(prod_lo, _mm512_shuffle_epi8(t3_lo, data_1));
            prod_hi = _mm512_xor_si512(prod_hi, _mm512_shuffle_epi8(t3_hi, data_1));

            let prod0_lo = _mm512_extracti64x4_epi64(prod_lo, 0);
            let prod0_hi = _mm512_extracti64x4_epi64(prod_hi, 0);
            let prod1_lo = _mm512_extracti64x4_epi64(prod_lo, 1);
            let prod1_hi = _mm512_extracti64x4_epi64(prod_hi, 1);

            (prod0_lo, prod0_hi, prod1_lo, prod1_hi)
        }
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_256
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn muladd_512(
        mut x0_lo: __m256i,
        mut x0_hi: __m256i,
        mut x1_lo: __m256i,
        mut x1_hi: __m256i,
        y0_lo: __m256i,
        y0_hi: __m256i,
        y1_lo: __m256i,
        y1_hi: __m256i,
        lut: &Multiply128lutT,
    ) -> (__m256i, __m256i, __m256i, __m256i) {
        unsafe {
            let (prod0_lo, prod0_hi, prod1_lo, prod1_hi) =
                Self::mul_512(y0_lo, y0_hi, y1_lo, y1_hi, lut);

            x0_lo = _mm256_xor_si256(x0_lo, prod0_lo);
            x0_hi = _mm256_xor_si256(x0_hi, prod0_hi);
            x1_lo = _mm256_xor_si256(x1_lo, prod1_lo);
            x1_hi = _mm256_xor_si256(x1_hi, prod1_hi);

            (x0_lo, x0_hi, x1_lo, x1_hi)
        }
    }
}

// ======================================================================
// Avx512 - PRIVATE - FFT (fast Fourier transform)

impl Avx512 {
    // Implementation of LEO_FFTB_256
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        let mut x_chunks_iter = x.chunks_exact_mut(128);
        let mut y_chunks_iter = y.chunks_exact_mut(128);
        for (x_chunk, y_chunk) in zip(&mut x_chunks_iter, &mut y_chunks_iter) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut __m256i;
            let y_ptr = y_chunk.as_mut_ptr() as *mut __m256i;

            unsafe {
                let mut x0_lo = _mm256_loadu_si256(x_ptr);
                let mut x0_hi = _mm256_loadu_si256(x_ptr.add(1));
                let mut x1_lo = _mm256_loadu_si256(x_ptr.add(2));
                let mut x1_hi = _mm256_loadu_si256(x_ptr.add(3));

                let mut y0_lo = _mm256_loadu_si256(y_ptr);
                let mut y0_hi = _mm256_loadu_si256(y_ptr.add(1));
                let mut y1_lo = _mm256_loadu_si256(y_ptr.add(2));
                let mut y1_hi = _mm256_loadu_si256(y_ptr.add(3));

                (x0_lo, x0_hi, x1_lo, x1_hi) =
                    Self::muladd_512(x0_lo, x0_hi, x1_lo, x1_hi, y0_lo, y0_hi, y1_lo, y1_hi, lut);

                y0_lo = _mm256_xor_si256(y0_lo, x0_lo);
                y0_hi = _mm256_xor_si256(y0_hi, x0_hi);
                y1_lo = _mm256_xor_si256(y1_lo, x1_lo);
                y1_hi = _mm256_xor_si256(y1_hi, x1_hi);

                _mm256_storeu_si256(y_ptr, y0_lo);
                _mm256_storeu_si256(y_ptr.add(1), y0_hi);
                _mm256_storeu_si256(y_ptr.add(2), y1_lo);
                _mm256_storeu_si256(y_ptr.add(3), y1_hi);

                _mm256_storeu_si256(x_ptr, x0_lo);
                _mm256_storeu_si256(x_ptr.add(1), x0_hi);
                _mm256_storeu_si256(x_ptr.add(2), x1_lo);
                _mm256_storeu_si256(x_ptr.add(3), x1_hi);
            }
        }

        // 64 bytes left?
        let x_chunk = TryInto::<&mut [u8; 64]>::try_into(x_chunks_iter.into_remainder());
        let y_chunk = TryInto::<&mut [u8; 64]>::try_into(y_chunks_iter.into_remainder());
        if let (Ok(x_chunk), Ok(y_chunk)) = (x_chunk, y_chunk) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut __m256i;
            let y_ptr = y_chunk.as_mut_ptr() as *mut __m256i;

            unsafe {
                let mut x_lo = _mm256_loadu_si256(x_ptr);
                let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

                let mut y_lo = _mm256_loadu_si256(y_ptr);
                let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

                let zero = _mm256_setzero_si256();
                (x_lo, x_hi, _, _) =
                    Self::muladd_512(x_lo, x_hi, zero, zero, y_lo, y_hi, zero, zero, lut);

                y_lo = _mm256_xor_si256(y_lo, x_lo);
                y_hi = _mm256_xor_si256(y_hi, x_hi);

                _mm256_storeu_si256(y_ptr, y_lo);
                _mm256_storeu_si256(y_ptr.add(1), y_hi);

                _mm256_storeu_si256(x_ptr, x_lo);
                _mm256_storeu_si256(x_ptr.add(1), x_hi);
            }
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            self.fft_butterfly_partial(s0, s2, log_m02);
            self.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            self.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            self.fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    unsafe fn fft_private_avx512(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.fft_private(data, pos, size, truncated_size, skew_delta);
    }

    #[inline(always)]
    fn fft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
                }

                r += dist4;
            }
            dist4 = dist;
            dist >>= 2;
        }

        // FINAL ODD LAYER

        if dist4 == 2 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + skew_delta];

                let (x, y) = data.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    Self::xor(y, x);
                } else {
                    self.fft_butterfly_partial(x, y, log_m)
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Avx512 - PRIVATE - IFFT (inverse fast Fourier transform)

impl Avx512 {
    // Implementation of LEO_IFFTB_256
    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        let mut x_chunks_iter = x.chunks_exact_mut(128);
        let mut y_chunks_iter = y.chunks_exact_mut(128);
        for (x_chunk, y_chunk) in zip(&mut x_chunks_iter, &mut y_chunks_iter) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut __m256i;
            let y_ptr = y_chunk.as_mut_ptr() as *mut __m256i;

            unsafe {
                let mut x0_lo = _mm256_loadu_si256(x_ptr);
                let mut x0_hi = _mm256_loadu_si256(x_ptr.add(1));
                let mut x1_lo = _mm256_loadu_si256(x_ptr.add(2));
                let mut x1_hi = _mm256_loadu_si256(x_ptr.add(3));

                let mut y0_lo = _mm256_loadu_si256(y_ptr);
                let mut y0_hi = _mm256_loadu_si256(y_ptr.add(1));
                let mut y1_lo = _mm256_loadu_si256(y_ptr.add(2));
                let mut y1_hi = _mm256_loadu_si256(y_ptr.add(3));

                y0_lo = _mm256_xor_si256(y0_lo, x0_lo);
                y0_hi = _mm256_xor_si256(y0_hi, x0_hi);
                y1_lo = _mm256_xor_si256(y1_lo, x1_lo);
                y1_hi = _mm256_xor_si256(y1_hi, x1_hi);

                _mm256_storeu_si256(y_ptr, y0_lo);
                _mm256_storeu_si256(y_ptr.add(1), y0_hi);
                _mm256_storeu_si256(y_ptr.add(2), y1_lo);
                _mm256_storeu_si256(y_ptr.add(3), y1_hi);

                (x0_lo, x0_hi, x1_lo, x1_hi) =
                    Self::muladd_512(x0_lo, x0_hi, x1_lo, x1_hi, y0_lo, y0_hi, y1_lo, y1_hi, lut);

                _mm256_storeu_si256(x_ptr, x0_lo);
                _mm256_storeu_si256(x_ptr.add(1), x0_hi);
                _mm256_storeu_si256(x_ptr.add(2), x1_lo);
                _mm256_storeu_si256(x_ptr.add(3), x1_hi);
            }
        }

        // 64 bytes left?
        let x_chunk = TryInto::<&mut [u8; 64]>::try_into(x_chunks_iter.into_remainder());
        let y_chunk = TryInto::<&mut [u8; 64]>::try_into(y_chunks_iter.into_remainder());
        if let (Ok(x_chunk), Ok(y_chunk)) = (x_chunk, y_chunk) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut __m256i;
            let y_ptr = y_chunk.as_mut_ptr() as *mut __m256i;

            unsafe {
                let mut x_lo = _mm256_loadu_si256(x_ptr);
                let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

                let mut y_lo = _mm256_loadu_si256(y_ptr);
                let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

                y_lo = _mm256_xor_si256(y_lo, x_lo);
                y_hi = _mm256_xor_si256(y_hi, x_hi);

                _mm256_storeu_si256(y_ptr, y_lo);
                _mm256_storeu_si256(y_ptr.add(1), y_hi);

                let zero = _mm256_setzero_si256();
                (x_lo, x_hi, _, _) =
                    Self::muladd_512(x_lo, x_hi, zero, zero, y_lo, y_hi, zero, zero, lut);

                _mm256_storeu_si256(x_ptr, x_lo);
                _mm256_storeu_si256(x_ptr.add(1), x_hi);
            }
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            Self::xor(s1, s0);
        } else {
            self.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            Self::xor(s3, s2);
        } else {
            self.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            Self::xor(s2, s0);
            Self::xor(s3, s1);
        } else {
            self.ifft_butterfly_partial(s0, s2, log_m02);
            self.ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    unsafe fn ifft_private_avx512(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.ifft_private(data, pos, size, truncated_size, skew_delta)
    }

    #[inline(always)]
    fn ifft_private(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = self.skew[base];
                let log_m02 = self.skew[base + dist];
                let log_m23 = self.skew[base + dist * 2];

                for i in r..r + dist {
                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = self.skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                Self::xor_within(data, pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = data.split_at_mut(pos + dist);
                for i in 0..dist {
                    self.ifft_butterfly_partial(
                        &mut a[pos + i], // data[pos + i]
                        &mut b[i],       // data[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }
}

// ======================================================================
// Avx512 - PRIVATE - Evaluate polynomial

impl Avx512 {
    #[target_feature(enable = "avx512f,avx512vl,avx512bw")]
    unsafe fn eval_poly_avx512(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        engine::eval_poly(erasures, truncated_size)
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Avx2;
    use crate::engine::NoSimd;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_avx2() {
        let mut rng = rand::thread_rng();

        let mut x = vec![0u8; 128];

        rng.fill::<[u8]>(x.as_mut());

        let mut x_clone = x.clone();

        let avx2 = Avx2::new();
        let nosimd = NoSimd::new();

        avx2.mul(x.as_mut(), 128);
        nosimd.mul(x_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
    }

    #[test]
    fn test_mul() {
        let mut rng = rand::thread_rng();
        let mut x = vec![0u8; 128];
        rng.fill::<[u8]>(x.as_mut());
        let mut x_clone = x.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.mul(x.as_mut(), 128);
        nosimd.mul(x_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
    }

    #[test]
    fn test_mul_64() {
        let mut rng = rand::thread_rng();
        let mut x = vec![0u8; 64];
        rng.fill::<[u8]>(x.as_mut());
        let mut x_clone = x.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.mul(x.as_mut(), 128);
        nosimd.mul(x_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
    }

    #[test]
    fn test_ifft_butterfly_partial() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut x = vec![0u8; 256];
        let mut y = vec![0u8; 256];

        rng.fill::<[u8]>(x.as_mut());
        rng.fill::<[u8]>(y.as_mut());

        let mut x_clone = x.clone();
        let mut y_clone = y.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.ifft_butterfly_partial(x.as_mut(), y.as_mut(), 128);
        nosimd.ifft_butterfly_partial(x_clone.as_mut(), y_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
        assert_eq!(y, y_clone);
    }

    #[test]
    fn test_ifft_butterfly_partial_64() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut x = vec![0u8; 64];
        let mut y = vec![0u8; 64];

        rng.fill::<[u8]>(x.as_mut());
        rng.fill::<[u8]>(y.as_mut());

        let mut x_clone = x.clone();
        let mut y_clone = y.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.ifft_butterfly_partial(x.as_mut(), y.as_mut(), 128);
        nosimd.ifft_butterfly_partial(x_clone.as_mut(), y_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
        assert_eq!(y, y_clone);
    }

    #[test]
    fn test_fft_butterfly_partial() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut x = vec![0u8; 256];
        let mut y = vec![0u8; 256];

        rng.fill::<[u8]>(x.as_mut());
        rng.fill::<[u8]>(y.as_mut());

        let mut x_clone = x.clone();
        let mut y_clone = y.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.fft_butterfly_partial(x.as_mut(), y.as_mut(), 128);
        nosimd.fft_butterfly_partial(x_clone.as_mut(), y_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
        assert_eq!(y, y_clone);
    }

    #[test]
    fn test_fft_butterfly_partial_64() {
        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut x = vec![0u8; 64];
        let mut y = vec![0u8; 64];

        rng.fill::<[u8]>(x.as_mut());
        rng.fill::<[u8]>(y.as_mut());

        let mut x_clone = x.clone();
        let mut y_clone = y.clone();

        let avx512 = Avx512::new();
        let nosimd = NoSimd::new();

        avx512.fft_butterfly_partial(x.as_mut(), y.as_mut(), 128);
        nosimd.fft_butterfly_partial(x_clone.as_mut(), y_clone.as_mut(), 128);

        assert_eq!(x, x_clone);
        assert_eq!(y, y_clone);
    }
}
