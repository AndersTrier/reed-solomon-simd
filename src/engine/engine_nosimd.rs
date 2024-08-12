use std::iter::zip;

use crate::engine::{
    tables::{self, Mul16},
    utils, Engine, GfElement,
};

// ======================================================================
// NoSimd - PUBLIC

/// Optimized [`Engine`] without SIMD.
///
/// [`NoSimd`] is a basic optimized engine which works on all CPUs.
#[derive(Clone)]
pub struct NoSimd {
    mul16: &'static Mul16,
}

impl NoSimd {
    /// Creates new [`NoSimd`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul16 = tables::initialize_mul16();

        Self { mul16 }
    }
}

impl Engine for NoSimd {
    #[inline(always)]
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        self.mul_add(x, y, log_m);
        utils::xor(y, x);
    }

    #[inline(always)]
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        utils::xor(y, x);
        self.mul_add(x, y, log_m);
    }

    #[inline(always)]
    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul16[log_m as usize];

        for x_chunk in x.iter_mut() {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);

            for i in 0..32 {
                let lo = x_lo[i];
                let hi = x_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] = prod as u8;
                x_hi[i] = (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// NoSimd - IMPL Default

impl Default for NoSimd {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// NoSimd - PRIVATE

impl NoSimd {
    /// `x[] ^= y[] * log_m`
    #[inline(always)]
    fn mul_add(&self, x: &mut [[u8; 64]], y: &[[u8; 64]], log_m: GfElement) {
        let lut = &self.mul16[log_m as usize];

        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter()) {
            let (x_lo, x_hi) = x_chunk.split_at_mut(32);
            let (y_lo, y_hi) = y_chunk.split_at(32);

            for i in 0..32 {
                let lo = y_lo[i];
                let hi = y_hi[i];
                let prod = lut[0][usize::from(lo & 15)]
                    ^ lut[1][usize::from(lo >> 4)]
                    ^ lut[2][usize::from(hi & 15)]
                    ^ lut[3][usize::from(hi >> 4)];
                x_lo[i] ^= prod as u8;
                x_hi[i] ^= (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
