use crate::engine::{tables, utils, Engine, GfElement, ShardsRefMut, GF_MODULUS};

impl<'a> ShardsRefMut<'a> {
    // ======================================================================
    // FFT (fast Fourier transform)

    /// In-place decimation-in-time FFT (fast Fourier transform).
    ///
    /// - FFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid FFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid FFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    #[inline(always)]
    pub(crate) fn fft(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let skew = tables::initialize_skew();

        // TWO LAYERS AT TIME

        let mut dist4 = size;
        let mut dist = size >> 2;
        while dist != 0 {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = skew[base];
                let log_m02 = skew[base + dist];
                let log_m23 = skew[base + dist * 2];

                for i in r..r + dist {
                    self.fft_butterfly_two_layers(engine, pos + i, dist, log_m01, log_m23, log_m02)
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
                let log_m = skew[r + skew_delta];

                let (x, y) = self.dist2_mut(pos + r, 1);

                if log_m == GF_MODULUS {
                    utils::xor(y, x);
                } else {
                    engine.fft_butterfly_partial(x, y, log_m)
                }

                r += 2;
            }
        }
    }

    #[inline(always)]
    fn fft_butterfly_two_layers(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = self.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            engine.fft_butterfly_partial(s0, s2, log_m02);
            engine.fft_butterfly_partial(s1, s3, log_m02);
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            engine.fft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            engine.fft_butterfly_partial(s2, s3, log_m23);
        }
    }

    /// FFT with `skew_delta = pos + size`.
    #[inline(always)]
    pub(crate) fn fft_skew_end(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        size: usize,
        truncated_size: usize,
    ) {
        self.fft(engine, pos, size, truncated_size, pos + size)
    }

    // ======================================================================
    // IFFT (inverse fast Fourier transform)

    /// In-place decimation-in-time IFFT (inverse fast Fourier transform).
    ///
    /// - IFFT is done on chunk `data[pos .. pos + size]`
    /// - `size` must be `2^n`
    /// - Before function call `data[pos .. pos + size]` must be valid.
    /// - After function call
    ///     - `data[pos .. pos + truncated_size]`
    ///       contains valid IFFT result.
    ///     - `data[pos + truncated_size .. pos + size]`
    ///       contains valid IFFT result if this contained
    ///       only `0u8`:s and garbage otherwise.
    #[inline(always)]
    pub(crate) fn ifft(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let skew = tables::initialize_skew();

        // TWO LAYERS AT TIME

        let mut dist = 1;
        let mut dist4 = 4;
        while dist4 <= size {
            let mut r = 0;
            while r < truncated_size {
                let base = r + dist + skew_delta - 1;

                let log_m01 = skew[base];
                let log_m02 = skew[base + dist];
                let log_m23 = skew[base + dist * 2];

                for i in r..r + dist {
                    self.ifft_butterfly_two_layers(engine, pos + i, dist, log_m01, log_m23, log_m02)
                }

                r += dist4;
            }
            dist = dist4;
            dist4 <<= 2;
        }

        // FINAL ODD LAYER

        if dist < size {
            let log_m = skew[dist + skew_delta - 1];
            if log_m == GF_MODULUS {
                self.xor_within(pos + dist, pos, dist);
            } else {
                let (mut a, mut b) = self.split_at_mut(pos + dist);
                for i in 0..dist {
                    engine.ifft_butterfly_partial(
                        &mut a[pos + i], // self[pos + i]
                        &mut b[i],       // self[pos + i + dist]
                        log_m,
                    );
                }
            }
        }
    }

    #[inline(always)]
    fn ifft_butterfly_two_layers(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        dist: usize,
        log_m01: GfElement,
        log_m23: GfElement,
        log_m02: GfElement,
    ) {
        let (s0, s1, s2, s3) = self.dist4_mut(pos, dist);

        // FIRST LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            engine.ifft_butterfly_partial(s0, s1, log_m01);
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            engine.ifft_butterfly_partial(s2, s3, log_m23);
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            engine.ifft_butterfly_partial(s0, s2, log_m02);
            engine.ifft_butterfly_partial(s1, s3, log_m02);
        }
    }

    /// IFFT with `skew_delta = pos + size`.
    #[inline(always)]
    pub(crate) fn ifft_skew_end(
        &mut self,
        engine: &impl Engine,
        pos: usize,
        size: usize,
        truncated_size: usize,
    ) {
        self.ifft(engine, pos, size, truncated_size, pos + size)
    }
}

// ======================================================================
// TESTS
