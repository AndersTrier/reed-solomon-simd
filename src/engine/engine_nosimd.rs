use crate::engine::{
    tables::{self, Mul16, Skew},
    Engine, GfElement, ShardsRefMut, GF_MODULUS,
};

// ======================================================================
// NoSimd - PUBLIC

/// Optimized [`Engine`] without SIMD.
///
/// [`NoSimd`] is a basic optimized engine which works on all CPUs.
#[derive(Clone)]
pub struct NoSimd {
    mul16: &'static Mul16,
    skew: &'static Skew,
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
        let skew = tables::initialize_skew();

        Self { mul16, skew }
    }
}

impl Engine for NoSimd {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.fft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.ifft_private(data, pos, size, truncated_size, skew_delta);
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        let lut = &self.mul16[log_m as usize];

        let mut pos = 0;
        while pos < x.len() {
            for i in 0..32 {
                let lo = x[pos + i] as usize;
                let hi = x[pos + i + 32] as usize;
                let prod = lut[0][lo & 15] ^ lut[1][lo >> 4] ^ lut[2][hi & 15] ^ lut[3][hi >> 4];
                x[pos + i] = prod as u8;
                x[pos + i + 32] = (prod >> 8) as u8;
            }
            pos += 64;
        }
    }

    fn xor(x: &mut [u8], y: &[u8]) {
        let x64: &mut [u64] = bytemuck::cast_slice_mut(x);
        let y64: &[u64] = bytemuck::cast_slice(y);

        for i in 0..x64.len() {
            x64[i] ^= y64[i];
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
    fn mul_add(&self, x: &mut [u8], y: &[u8], log_m: GfElement) {
        let lut = &self.mul16[log_m as usize];

        let mut pos = 0;
        while pos < x.len() {
            for i in 0..32 {
                let lo = y[pos + i] as usize;
                let hi = y[pos + i + 32] as usize;
                let prod = lut[0][lo & 15] ^ lut[1][lo >> 4] ^ lut[2][hi & 15] ^ lut[3][hi >> 4];
                x[pos + i] ^= prod as u8;
                x[pos + i + 32] ^= (prod >> 8) as u8;
            }
            pos += 64;
        }
    }
}

// ======================================================================
// NoSimd - PRIVATE - FFT (fast Fourier transform)

impl NoSimd {
    /// TODO
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    pub fn fft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        self.mul_add(x, y, log_m);
        Self::xor(y, x);
    }

    /// TODO
    #[inline(always)]
    pub fn fft_butterfly_two_layers(
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
// NoSimd - PRIVATE - IFFT (inverse fast Fourier transform)

impl NoSimd {
    /// TODO
    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    pub fn ifft_butterfly_partial(&self, x: &mut [u8], y: &mut [u8], log_m: GfElement) {
        Self::xor(y, x);
        self.mul_add(x, y, log_m);
    }

    /// TODO
    #[inline(always)]
    pub fn ifft_butterfly_two_layers(
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
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
