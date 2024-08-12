use crate::engine::{utils, Engine, GfElement, ShardsRefMut, GF_MODULUS};

#[inline(always)]
fn fft_butterfly_two_layers<E: Engine>(
    engine: &E,
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

//#[inline(always)]
//fn fft_private<E: Engine>(
//    &engine: E,
//    data: &mut ShardsRefMut,
//    pos: usize,
//    size: usize,
//    truncated_size: usize,
//    skew_delta: usize,
//) {
//    // TWO LAYERS AT TIME
//
//    let mut dist4 = size;
//    let mut dist = size >> 2;
//    while dist != 0 {
//        let mut r = 0;
//        while r < truncated_size {
//            let base = r + dist + skew_delta - 1;
//
//            let log_m01 = self.skew[base];
//            let log_m02 = self.skew[base + dist];
//            let log_m23 = self.skew[base + dist * 2];
//
//            for i in r..r + dist {
//                self.fft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
//            }
//
//            r += dist4;
//        }
//        dist4 = dist;
//        dist >>= 2;
//    }
//
//    // FINAL ODD LAYER
//
//    if dist4 == 2 {
//        let mut r = 0;
//        while r < truncated_size {
//            let log_m = self.skew[r + skew_delta];
//
//            let (x, y) = data.dist2_mut(pos + r, 1);
//
//            if log_m == GF_MODULUS {
//                utils::xor(y, x);
//            } else {
//                self.fft_butterfly_partial(x, y, log_m)
//            }
//
//            r += 2;
//        }
//    }
//}
//
//

// ======================================================================
// NoSimd - PRIVATE - IFFT (inverse fast Fourier transform)

//impl NoSimd {
//    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
//    #[inline(always)]
//    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement) {
//        utils::xor(y, x);
//        self.mul_add(x, y, log_m);
//    }
//
//    #[inline(always)]
//    fn ifft_butterfly_two_layers(
//        &self,
//        data: &mut ShardsRefMut,
//        pos: usize,
//        dist: usize,
//        log_m01: GfElement,
//        log_m23: GfElement,
//        log_m02: GfElement,
//    ) {
//        let (s0, s1, s2, s3) = data.dist4_mut(pos, dist);
//
//        // FIRST LAYER
//
//        if log_m01 == GF_MODULUS {
//            utils::xor(s1, s0);
//        } else {
//            self.ifft_butterfly_partial(s0, s1, log_m01);
//        }
//
//        if log_m23 == GF_MODULUS {
//            utils::xor(s3, s2);
//        } else {
//            self.ifft_butterfly_partial(s2, s3, log_m23);
//        }
//
//        // SECOND LAYER
//
//        if log_m02 == GF_MODULUS {
//            utils::xor(s2, s0);
//            utils::xor(s3, s1);
//        } else {
//            self.ifft_butterfly_partial(s0, s2, log_m02);
//            self.ifft_butterfly_partial(s1, s3, log_m02);
//        }
//    }
//
//    #[inline(always)]
//    fn ifft_private(
//        &self,
//        data: &mut ShardsRefMut,
//        pos: usize,
//        size: usize,
//        truncated_size: usize,
//        skew_delta: usize,
//    ) {
//        // TWO LAYERS AT TIME
//
//        let mut dist = 1;
//        let mut dist4 = 4;
//        while dist4 <= size {
//            let mut r = 0;
//            while r < truncated_size {
//                let base = r + dist + skew_delta - 1;
//
//                let log_m01 = self.skew[base];
//                let log_m02 = self.skew[base + dist];
//                let log_m23 = self.skew[base + dist * 2];
//
//                for i in r..r + dist {
//                    self.ifft_butterfly_two_layers(data, pos + i, dist, log_m01, log_m23, log_m02)
//                }
//
//                r += dist4;
//            }
//            dist = dist4;
//            dist4 <<= 2;
//        }
//
//        // FINAL ODD LAYER
//
//        if dist < size {
//            let log_m = self.skew[dist + skew_delta - 1];
//            if log_m == GF_MODULUS {
//                utils::xor_within(data, pos + dist, pos, dist);
//            } else {
//                let (mut a, mut b) = data.split_at_mut(pos + dist);
//                for i in 0..dist {
//                    self.ifft_butterfly_partial(
//                        &mut a[pos + i], // data[pos + i]
//                        &mut b[i],       // data[pos + i + dist]
//                        log_m,
//                    );
//                }
//            }
//        }
//    }
//}
//
//// ======================================================================
//// TESTS
