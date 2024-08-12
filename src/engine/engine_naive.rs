use crate::engine::{
    tables::{self, Exp, Log},
    utils, Engine, GfElement,
};

// ======================================================================
// Naive - PUBLIC

/// Simple reference implementation of [`Engine`].
///
/// - [`Naive`] is meant for those who want to study
///   the source code to understand [`Engine`].
/// - [`Naive`] also includes some debug assertions
///   which are not present in other implementations.
#[derive(Clone)]
pub struct Naive {
    exp: &'static Exp,
    log: &'static Log,
}

impl Naive {
    /// Creates new [`Naive`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let (exp, log) = tables::initialize_exp_log();

        Self { exp, log }
    }
}

impl Engine for Naive {
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        self.mul_add(x, y, log_m);
        utils::xor(y, x);
    }

    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
        utils::xor(y, x);
        self.mul_add(x, y, log_m);
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        for chunk in x.iter_mut() {
            for i in 0..32 {
                let lo = chunk[i] as GfElement;
                let hi = chunk[i + 32] as GfElement;
                let prod = tables::mul(lo | (hi << 8), log_m, self.exp, self.log);
                chunk[i] = prod as u8;
                chunk[i + 32] = (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// Naive - IMPL Default

impl Default for Naive {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Naive - PRIVATE

impl Naive {
    /// `x[] ^= y[] * log_m`
    fn mul_add(&self, x: &mut [[u8; 64]], y: &[[u8; 64]], log_m: GfElement) {
        debug_assert_eq!(x.len(), y.len());

        for (x_chunk, y_chunk) in std::iter::zip(x.iter_mut(), y.iter()) {
            for i in 0..32 {
                let lo = y_chunk[i] as GfElement;
                let hi = y_chunk[i + 32] as GfElement;
                let prod = tables::mul(lo | (hi << 8), log_m, self.exp, self.log);
                x_chunk[i] ^= prod as u8;
                x_chunk[i + 32] ^= (prod >> 8) as u8;
            }
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
