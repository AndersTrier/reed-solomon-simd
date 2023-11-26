use crate::engine::{
    self,
    tables::{self, Exp, Log, Skew},
    Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
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
    skew: &'static Skew,
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
        let skew = tables::initialize_skew();

        Self { exp, log, skew }
    }
}

impl Engine for Naive {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        debug_assert!(size.is_power_of_two());
        debug_assert!(truncated_size <= size);

        let mut dist = size / 2;
        while dist > 0 {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + dist + skew_delta - 1];
                for i in r..r + dist {
                    let (a, b) = data.dist2_mut(pos + i, dist);

                    // FFT BUTTERFLY

                    if log_m != GF_MODULUS {
                        self.mul_add(a, b, log_m);
                    }
                    Self::xor(b, a);
                }
                r += dist * 2;
            }
            dist /= 2;
        }
    }

    fn fwht(data: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        debug_assert!(truncated_size <= GF_ORDER);

        let mut dist = 1;
        while dist < GF_ORDER {
            let mut r = 0;
            while r < truncated_size {
                for i in r..r + dist {
                    let sum = engine::add_mod(data[i], data[i + dist]);
                    let dif = engine::sub_mod(data[i], data[i + dist]);
                    data[i] = sum;
                    data[i + dist] = dif;
                }
                r += dist * 2;
            }
            dist *= 2;
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
        debug_assert!(size.is_power_of_two());
        debug_assert!(truncated_size <= size);

        let mut dist = 1;
        while dist < size {
            let mut r = 0;
            while r < truncated_size {
                let log_m = self.skew[r + dist + skew_delta - 1];
                for i in r..r + dist {
                    let (a, b) = data.dist2_mut(pos + i, dist);

                    // IFFT BUTTERFLY

                    Self::xor(b, a);
                    if log_m != GF_MODULUS {
                        self.mul_add(a, b, log_m);
                    }
                }
                r += dist * 2;
            }
            dist *= 2;
        }
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        let shard_bytes = x.len();
        debug_assert!(shard_bytes & 63 == 0);

        let mut pos = 0;
        while pos < shard_bytes {
            for i in 0..32 {
                let lo = x[pos + i] as GfElement;
                let hi = x[pos + i + 32] as GfElement;
                let prod = tables::mul(lo | (hi << 8), log_m, self.exp, self.log);
                x[pos + i] = prod as u8;
                x[pos + i + 32] = (prod >> 8) as u8;
            }
            pos += 64;
        }
    }

    fn xor(x: &mut [u8], y: &[u8]) {
        let shard_bytes = x.len();
        debug_assert!(shard_bytes & 63 == 0);
        debug_assert_eq!(shard_bytes, y.len());

        for i in 0..shard_bytes {
            x[i] ^= y[i];
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
    fn mul_add(&self, x: &mut [u8], y: &[u8], log_m: GfElement) {
        let shard_bytes = x.len();
        debug_assert!(shard_bytes & 63 == 0);
        debug_assert_eq!(shard_bytes, y.len());

        let mut pos = 0;
        while pos < shard_bytes {
            for i in 0..32 {
                let lo = y[pos + i] as GfElement;
                let hi = y[pos + i + 32] as GfElement;
                let prod = tables::mul(lo | (hi << 8), log_m, self.exp, self.log);
                x[pos + i] ^= prod as u8;
                x[pos + i + 32] ^= (prod >> 8) as u8;
            }
            pos += 64;
        }
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
