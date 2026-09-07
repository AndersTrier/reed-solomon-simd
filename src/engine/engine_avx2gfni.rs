use core::iter::zip;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::engine::{
    tables::{self, MulGfni, Multiply64GfniT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Avx2Gfni - PUBLIC

/// Optimized [`Engine`] using AVX2 and GFNI instructions.
///
/// [`Avx2Gfni`] follows the same algorithm as [`NoSimd`], but instead of
/// looking up the four nibble tables of a multiplication with `vpshufb`, it
/// evaluates the multiplication directly as a `GF(2)` matrix product with
/// `vgf2p8affineqb`. That replaces the 8 `vpshufb`, 6 `vpxor`, 4 `vpand` and
/// 2 `vpsrlq` of [`Avx2`] with 4 `vgf2p8affineqb` and 2 `vpxor` per 64 byte
/// chunk, and shrinks the multiplication table from 8 MiB to 2 MiB.
///
/// GFNI is available on AMD Zen 4 and later, and on Intel Ice Lake / Alder
/// Lake and later.
///
/// [`NoSimd`]: crate::engine::NoSimd
/// [`Avx2`]: crate::engine::Avx2
#[derive(Clone, Copy)]
pub struct Avx2Gfni {
    mul_gfni: &'static MulGfni,
    skew: &'static Skew,
}

impl Avx2Gfni {
    /// Creates new [`Avx2Gfni`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [tables]: crate::engine::tables
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul_gfni = tables::get_mul_gfni();
        let skew = tables::get_skew();

        Self { mul_gfni, skew }
    }
}

impl Engine for Avx2Gfni {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_avx2gfni(data, pos, size, truncated_size, skew_delta);
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
            self.ifft_private_avx2gfni(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_avx2gfni(x, log_m);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_avx2gfni(erasures, truncated_size) }
    }
}

// ======================================================================
// Avx2Gfni - IMPL Default

impl Default for Avx2Gfni {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Avx2Gfni - PRIVATE

#[derive(Copy, Clone)]
struct LutAvx2Gfni {
    lo_from_lo: __m256i,
    lo_from_hi: __m256i,
    hi_from_lo: __m256i,
    hi_from_hi: __m256i,
}

impl From<&Multiply64GfniT> for LutAvx2Gfni {
    #[inline(always)]
    fn from(lut: &Multiply64GfniT) -> Self {
        unsafe {
            Self {
                lo_from_lo: _mm256_set1_epi64x(lut.lo_from_lo.cast_signed()),
                lo_from_hi: _mm256_set1_epi64x(lut.lo_from_hi.cast_signed()),
                hi_from_lo: _mm256_set1_epi64x(lut.hi_from_lo.cast_signed()),
                hi_from_hi: _mm256_set1_epi64x(lut.hi_from_hi.cast_signed()),
            }
        }
    }
}

impl Avx2Gfni {
    #[target_feature(enable = "avx2,gfni")]
    unsafe fn mul_avx2gfni(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx2Gfni::from(&self.mul_gfni[log_m as usize]);

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr().cast::<__m256i>();
            unsafe {
                let x_lo = _mm256_loadu_si256(x_ptr);
                let x_hi = _mm256_loadu_si256(x_ptr.add(1));
                let (prod_lo, prod_hi) = Self::mul_256(x_lo, x_hi, lut);
                _mm256_storeu_si256(x_ptr, prod_lo);
                _mm256_storeu_si256(x_ptr.add(1), prod_hi);
            }
        }
    }

    // `{value_lo, value_hi} * log_m`, as four 8x8 `GF(2)` matrix products.
    #[inline(always)]
    fn mul_256(value_lo: __m256i, value_hi: __m256i, lut: LutAvx2Gfni) -> (__m256i, __m256i) {
        unsafe {
            let prod_lo = _mm256_xor_si256(
                _mm256_gf2p8affine_epi64_epi8::<0>(value_lo, lut.lo_from_lo),
                _mm256_gf2p8affine_epi64_epi8::<0>(value_hi, lut.lo_from_hi),
            );
            let prod_hi = _mm256_xor_si256(
                _mm256_gf2p8affine_epi64_epi8::<0>(value_lo, lut.hi_from_lo),
                _mm256_gf2p8affine_epi64_epi8::<0>(value_hi, lut.hi_from_hi),
            );

            (prod_lo, prod_hi)
        }
    }

    // `{x_lo, x_hi} ^= {y_lo, y_hi} * log_m`
    #[inline(always)]
    fn muladd_256(
        mut x_lo: __m256i,
        mut x_hi: __m256i,
        y_lo: __m256i,
        y_hi: __m256i,
        lut: LutAvx2Gfni,
    ) -> (__m256i, __m256i) {
        let (prod_lo, prod_hi) = Self::mul_256(y_lo, y_hi, lut);
        unsafe {
            x_lo = _mm256_xor_si256(x_lo, prod_lo);
            x_hi = _mm256_xor_si256(x_hi, prod_hi);
        }
        (x_lo, x_hi)
    }
}

// ======================================================================
// Avx2Gfni - PRIVATE - FFT (fast Fourier transform)

impl Avx2Gfni {
    #[inline(always)]
    fn fftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutAvx2Gfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let mut x_lo = _mm256_loadu_si256(x_ptr);
            let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

            let mut y_lo = _mm256_loadu_si256(y_ptr);
            let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut);

            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);

            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);

            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx2Gfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::fftb_256(x_chunk, y_chunk, lut);
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

    #[target_feature(enable = "avx2,gfni")]
    unsafe fn fft_private_avx2gfni(
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
                    self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
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
                    self.fft_butterfly_partial(x, y, log_m);
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Avx2Gfni - PRIVATE - IFFT (inverse fast Fourier transform)

impl Avx2Gfni {
    #[inline(always)]
    fn ifftb_256(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutAvx2Gfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m256i>();
        let y_ptr = y.as_mut_ptr().cast::<__m256i>();

        unsafe {
            let mut x_lo = _mm256_loadu_si256(x_ptr);
            let mut x_hi = _mm256_loadu_si256(x_ptr.add(1));

            let mut y_lo = _mm256_loadu_si256(y_ptr);
            let mut y_hi = _mm256_loadu_si256(y_ptr.add(1));

            y_lo = _mm256_xor_si256(y_lo, x_lo);
            y_hi = _mm256_xor_si256(y_hi, x_hi);

            _mm256_storeu_si256(y_ptr, y_lo);
            _mm256_storeu_si256(y_ptr.add(1), y_hi);

            (x_lo, x_hi) = Self::muladd_256(x_lo, x_hi, y_lo, y_hi, lut);

            _mm256_storeu_si256(x_ptr, x_lo);
            _mm256_storeu_si256(x_ptr.add(1), x_hi);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx2Gfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::ifftb_256(x_chunk, y_chunk, lut);
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

    #[target_feature(enable = "avx2,gfni")]
    unsafe fn ifft_private_avx2gfni(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        // Drop unsafe privileges
        self.ifft_private(data, pos, size, truncated_size, skew_delta);
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
                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02);
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
// Avx2Gfni - PRIVATE - Evaluate polynomial

impl Avx2Gfni {
    #[target_feature(enable = "avx2,gfni")]
    unsafe fn eval_poly_avx2gfni(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate,
// but those only run on hosts which actually have GFNI. The test below models
// `mul_256` in scalar code so that the lane arrangement of the lookup table is
// checked everywhere.

#[cfg(test)]
mod tests {
    use crate::engine::tables::{self, gfni_affine};
    use crate::engine::{Engine, NoSimd};

    // Scalar model of `mul_256`: the first 32 bytes of a chunk are the low
    // bytes of 32 `GfElement`s, the last 32 bytes are their high bytes.
    fn mul_256_model(chunk: &mut [u8; 64], log_m: u16) {
        let lut = &tables::get_mul_gfni()[log_m as usize];

        let (value_lo, value_hi) = (&chunk[..32].to_vec(), &chunk[32..].to_vec());

        for i in 0..32 {
            chunk[i] =
                gfni_affine(lut.lo_from_lo, value_lo[i]) ^ gfni_affine(lut.lo_from_hi, value_hi[i]);
            chunk[32 + i] =
                gfni_affine(lut.hi_from_lo, value_lo[i]) ^ gfni_affine(lut.hi_from_hi, value_hi[i]);
        }
    }

    #[test]
    fn mul_256_model_matches_nosimd() {
        let nosimd = NoSimd::new();

        for log_m in [0u16, 1, 2, 128, 1000, 32768, 65534, 65535] {
            let mut chunk = [0u8; 64];
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
            }

            let mut expected = [chunk];
            nosimd.mul(&mut expected, log_m);

            mul_256_model(&mut chunk, log_m);

            assert_eq!(chunk, expected[0], "log_m {log_m}");
        }
    }
}
