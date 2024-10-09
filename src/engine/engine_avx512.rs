use std::iter::zip;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::engine::{
    tables::{self, Mul128, Multiply128lutT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
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
        let mul128 = &*tables::MUL128;
        let skew = &*tables::SKEW;

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

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_avx512(x, log_m);
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
    unsafe fn mul_avx512(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr() as *mut i32;
            unsafe {
                let x = _mm512_loadu_si512(x_ptr);
                let prod = Self::mul_512(x, lut);
                _mm512_storeu_si512(x_ptr, prod);
            }
        }
    }

    // Impelemntation of LEO_MUL_256
    #[inline(always)]
    fn mul_512(value: __m512i, lut: &Multiply128lutT) -> __m512i {
        unsafe {
            let t0_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.lo[0] as *const u128 as *const __m128i,
            ));
            let t1_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.lo[1] as *const u128 as *const __m128i,
            ));
            let t2_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.lo[2] as *const u128 as *const __m128i,
            ));
            let t3_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.lo[3] as *const u128 as *const __m128i,
            ));

            let t0_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.hi[0] as *const u128 as *const __m128i,
            ));
            let t1_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.hi[1] as *const u128 as *const __m128i,
            ));
            let t2_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.hi[2] as *const u128 as *const __m128i,
            ));
            let t3_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(
                &lut.hi[3] as *const u128 as *const __m128i,
            ));

            let t0_t2_lo = _mm512_inserti64x4(_mm512_castsi256_si512(t0_lo), t2_lo, 1);
            let t0_t2_hi = _mm512_inserti64x4(_mm512_castsi256_si512(t0_hi), t2_hi, 1);
            let t1_t3_lo = _mm512_inserti64x4(_mm512_castsi256_si512(t1_lo), t3_lo, 1);
            let t1_t3_hi = _mm512_inserti64x4(_mm512_castsi256_si512(t1_hi), t3_hi, 1);

            let clr_mask = _mm512_set1_epi8(0x0f);

            let data = _mm512_and_si512(value, clr_mask);
            let mut prod_lo_512 = _mm512_shuffle_epi8(t0_t2_lo, data);
            let mut prod_hi_512 = _mm512_shuffle_epi8(t0_t2_hi, data);

            let data = _mm512_and_si512(_mm512_srli_epi64(value, 4), clr_mask);
            prod_lo_512 = _mm512_xor_si512(prod_lo_512, _mm512_shuffle_epi8(t1_t3_lo, data));
            prod_hi_512 = _mm512_xor_si512(prod_hi_512, _mm512_shuffle_epi8(t1_t3_hi, data));

            // XOR first half with second half of vector
            let prod_lo = _mm256_xor_si256(
                _mm512_castsi512_si256(prod_lo_512),
                _mm512_extracti64x4_epi64(prod_lo_512, 1),
            );
            let prod_hi = _mm256_xor_si256(
                _mm512_castsi512_si256(prod_hi_512),
                _mm512_extracti64x4_epi64(prod_hi_512, 1),
            );

            _mm512_inserti64x4(_mm512_castsi256_si512(prod_lo), prod_hi, 1)
        }
    }

    //// {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_256
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn muladd_512(x: __m512i, y: __m512i, lut: &Multiply128lutT) -> __m512i {
        unsafe {
            let prod = Self::mul_512(y, lut);
            _mm512_xor_si512(x, prod)
        }
    }
}

// ======================================================================
// Avx512 - PRIVATE - FFT (fast Fourier transform)

impl Avx512 {
    // Implementation of LEO_FFTB_256
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut i32;
            let y_ptr = y_chunk.as_mut_ptr() as *mut i32;

            unsafe {
                let mut x = _mm512_loadu_si512(x_ptr);
                let mut y = _mm512_loadu_si512(y_ptr);

                x = Self::muladd_512(x, y, lut);
                y = _mm512_xor_si512(y, x);

                _mm512_storeu_si512(x_ptr, x);
                _mm512_storeu_si512(y_ptr, y);
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
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            self.fft_butterfly_partial(s0, s2, log_m02);
            self.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            self.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
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
                    utils::xor(y, x);
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
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for (x_chunk, y_chunk) in zip(&mut x.iter_mut(), &mut y.iter_mut()) {
            let x_ptr = x_chunk.as_mut_ptr() as *mut i32;
            let y_ptr = y_chunk.as_mut_ptr() as *mut i32;

            unsafe {
                let mut x = _mm512_loadu_si512(x_ptr);
                let mut y = _mm512_loadu_si512(y_ptr);

                y = _mm512_xor_si512(y, x);
                x = Self::muladd_512(x, y, lut);

                _mm512_storeu_si512(x_ptr, x);
                _mm512_storeu_si512(y_ptr, y);
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
            utils::xor(s1, s0);
        } else {
            self.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            self.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
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
                utils::xor_within(data, pos + dist, pos, dist);
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
        utils::eval_poly(erasures, truncated_size)
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
        let mut x = vec![[0u8; 64]; 2];
        rng.fill::<[u8]>(x.as_flattened_mut());
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
        let mut x = vec![[0u8; 64]; 2];
        rng.fill::<[u8]>(x.as_flattened_mut());
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
        let mut x = vec![[0u8; 64]; 1];
        rng.fill::<[u8]>(x.as_flattened_mut());
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
        let mut x = vec![[0u8; 64]; 4];
        let mut y = vec![[0u8; 64]; 4];

        rng.fill::<[u8]>(x.as_flattened_mut());
        rng.fill::<[u8]>(y.as_flattened_mut());

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
        let mut x = vec![[0u8; 64]; 1];
        let mut y = vec![[0u8; 64]; 1];

        rng.fill::<[u8]>(x.as_flattened_mut());
        rng.fill::<[u8]>(y.as_flattened_mut());

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
        let mut x = vec![[0u8; 64]; 4];
        let mut y = vec![[0u8; 64]; 4];

        rng.fill::<[u8]>(x.as_flattened_mut());
        rng.fill::<[u8]>(y.as_flattened_mut());

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
        let mut x = vec![[0u8; 64]; 1];
        let mut y = vec![[0u8; 64]; 1];

        rng.fill::<[u8]>(x.as_flattened_mut());
        rng.fill::<[u8]>(y.as_flattened_mut());

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
