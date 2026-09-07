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
// Avx512Gfni - PUBLIC

/// Optimized [`Engine`] using AVX-512 and GFNI instructions.
///
/// [`Avx512Gfni`] is the 512 bit wide variant of [`Avx2Gfni`]: a 64 byte
/// chunk holds the low bytes of 32 `GfElement`s followed by their high
/// bytes, so a single `zmm` register holds a whole chunk and the four 8x8
/// `GF(2)` matrices of a multiplication collapse into two
/// `vgf2p8affineqb`, one on the chunk and one on the chunk with its halves
/// swapped.
///
/// The half swap is not avoidable: `vgf2p8affineqb` is byte local, and a
/// multiplication has to mix a chunk's low bytes with its high bytes, which
/// sit in the other 256 bit half. So this engine only pays off where a `zmm`
/// `vgf2p8affineqb` retires at the same rate as a `ymm` one. Measured
/// reciprocal throughputs:
///
/// | CPU                                | `ymm` | `zmm` |
/// | ---------------------------------- | ----- | ----- |
/// | Zen 4                              | 0.50c | 1.00c |
/// | Zen 5, 256 bit dispatch            | 0.50c | 1.00c |
/// | Zen 5, 512 bit dispatch            | 0.50c | 0.50c |
/// | Ice Lake / Rocket Lake / Alder Lake| 0.50c | 1.00c |
///
/// Everywhere but the third row the extra width buys no throughput and the
/// `vshufi64x2` is pure loss, so [`Avx2Gfni`] is faster: 2.50 versus 3.50
/// port cycles per chunk on Zen 4. Since CPUID does not expose the dispatch
/// width (same silicon, both of the Zen 5 rows), [`DefaultEngine`] never
/// picks this engine; select it explicitly after benchmarking.
///
/// [`Avx2Gfni`]: crate::engine::Avx2Gfni
/// [`DefaultEngine`]: crate::engine::DefaultEngine
#[derive(Clone, Copy)]
pub struct Avx512Gfni {
    mul_gfni: &'static MulGfni,
    skew: &'static Skew,
}

impl Avx512Gfni {
    /// Creates new [`Avx512Gfni`], initializing all [tables]
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

impl Engine for Avx512Gfni {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_avx512gfni(data, pos, size, truncated_size, skew_delta);
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
            self.ifft_private_avx512gfni(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_avx512gfni(x, log_m);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_avx512gfni(erasures, truncated_size) }
    }
}

// ======================================================================
// Avx512Gfni - IMPL Default

impl Default for Avx512Gfni {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Avx512Gfni - PRIVATE

#[derive(Copy, Clone)]
struct LutAvx512Gfni {
    // `{lo_from_lo, hi_from_hi}`, applied to a chunk as loaded.
    straight: __m512i,
    // `{lo_from_hi, hi_from_lo}`, applied to a chunk with swapped halves.
    crossed: __m512i,
}

impl From<&Multiply64GfniT> for LutAvx512Gfni {
    #[inline(always)]
    fn from(lut: &Multiply64GfniT) -> Self {
        unsafe {
            Self {
                straight: _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_set1_epi64x(lut.lo_from_lo.cast_signed())),
                    _mm256_set1_epi64x(lut.hi_from_hi.cast_signed()),
                    1,
                ),
                crossed: _mm512_inserti64x4(
                    _mm512_castsi256_si512(_mm256_set1_epi64x(lut.lo_from_hi.cast_signed())),
                    _mm256_set1_epi64x(lut.hi_from_lo.cast_signed()),
                    1,
                ),
            }
        }
    }
}

impl Avx512Gfni {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
    unsafe fn mul_avx512gfni(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx512Gfni::from(&self.mul_gfni[log_m as usize]);

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr().cast::<__m512i>();
            unsafe {
                let x = _mm512_loadu_si512(x_ptr);
                _mm512_storeu_si512(x_ptr, Self::mul_512(x, lut));
            }
        }
    }

    // `value * log_m`, as four 8x8 `GF(2)` matrix products.
    //
    // `value` is `{value_lo, value_hi}` and the result is
    // `{lo_from_lo * value_lo + lo_from_hi * value_hi,
    //   hi_from_lo * value_lo + hi_from_hi * value_hi}`,
    // which the swapped copy of `value` lines up with `crossed`.
    #[inline(always)]
    fn mul_512(value: __m512i, lut: LutAvx512Gfni) -> __m512i {
        unsafe {
            let swapped = _mm512_shuffle_i64x2::<0x4e>(value, value);

            _mm512_xor_si512(
                _mm512_gf2p8affine_epi64_epi8::<0>(value, lut.straight),
                _mm512_gf2p8affine_epi64_epi8::<0>(swapped, lut.crossed),
            )
        }
    }

    // `x ^= y * log_m`
    #[inline(always)]
    fn muladd_512(x: __m512i, y: __m512i, lut: LutAvx512Gfni) -> __m512i {
        unsafe { _mm512_xor_si512(x, Self::mul_512(y, lut)) }
    }
}

// ======================================================================
// Avx512Gfni - PRIVATE - FFT (fast Fourier transform)

impl Avx512Gfni {
    #[inline(always)]
    fn fftb_512(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutAvx512Gfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m512i>();
        let y_ptr = y.as_mut_ptr().cast::<__m512i>();

        unsafe {
            let mut x = _mm512_loadu_si512(x_ptr);
            let mut y = _mm512_loadu_si512(y_ptr);

            x = Self::muladd_512(x, y, lut);
            y = _mm512_xor_si512(y, x);

            _mm512_storeu_si512(x_ptr, x);
            _mm512_storeu_si512(y_ptr, y);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx512Gfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::fftb_512(x_chunk, y_chunk, lut);
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

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
    unsafe fn fft_private_avx512gfni(
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
// Avx512Gfni - PRIVATE - IFFT (inverse fast Fourier transform)

impl Avx512Gfni {
    #[inline(always)]
    fn ifftb_512(x: &mut [u8; 64], y: &mut [u8; 64], lut: LutAvx512Gfni) {
        let x_ptr = x.as_mut_ptr().cast::<__m512i>();
        let y_ptr = y.as_mut_ptr().cast::<__m512i>();

        unsafe {
            let mut x = _mm512_loadu_si512(x_ptr);
            let mut y = _mm512_loadu_si512(y_ptr);

            y = _mm512_xor_si512(y, x);
            x = Self::muladd_512(x, y, lut);

            _mm512_storeu_si512(x_ptr, x);
            _mm512_storeu_si512(y_ptr, y);
        }
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        let lut = LutAvx512Gfni::from(&self.mul_gfni[log_m as usize]);

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            Self::ifftb_512(x_chunk, y_chunk, lut);
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

    #[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
    unsafe fn ifft_private_avx512gfni(
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
// Avx512Gfni - PRIVATE - Evaluate polynomial

impl Avx512Gfni {
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,gfni")]
    unsafe fn eval_poly_avx512gfni(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate,
// but those only run on hosts which actually have GFNI. The test below models
// `mul_512` in scalar code so that the `straight`/`crossed` lane arrangement
// and the half swap are checked everywhere.

#[cfg(test)]
mod tests {
    use crate::engine::tables::{self, gfni_affine};
    use crate::engine::{Engine, NoSimd};

    // Scalar model of `mul_512`.
    fn mul_512_model(chunk: &mut [u8; 64], log_m: u16) {
        let lut = &tables::get_mul_gfni()[log_m as usize];

        // `_mm512_shuffle_i64x2::<0x4e>(value, value)`
        let mut swapped = [0u8; 64];
        swapped[..32].copy_from_slice(&chunk[32..]);
        swapped[32..].copy_from_slice(&chunk[..32]);

        for i in 0..64 {
            let (straight, crossed) = if i < 32 {
                (lut.lo_from_lo, lut.lo_from_hi)
            } else {
                (lut.hi_from_hi, lut.hi_from_lo)
            };

            chunk[i] = gfni_affine(straight, chunk[i]) ^ gfni_affine(crossed, swapped[i]);
        }
    }

    #[test]
    fn mul_512_model_matches_nosimd() {
        let nosimd = NoSimd::new();

        for log_m in [0u16, 1, 2, 128, 1000, 32768, 65534, 65535] {
            let mut chunk = [0u8; 64];
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte = (i as u8).wrapping_mul(37).wrapping_add(11);
            }

            let mut expected = [chunk];
            nosimd.mul(&mut expected, log_m);

            mul_512_model(&mut chunk, log_m);

            assert_eq!(chunk, expected[0], "log_m {log_m}");
        }
    }
}
