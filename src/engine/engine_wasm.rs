use core::arch::wasm32::*;
use core::iter::zip;

use crate::engine::{
    tables::{self, Mul128, Multiply128lutT, Skew},
    utils, Engine, GfElement, ShardsRefMut, GF_MODULUS, GF_ORDER,
};

// ======================================================================
// Wasm - PUBLIC

/// Optimized [`Engine`] using WebAssembly SIMD128 instructions.
///
/// [`Wasm`] is an optimized engine that follows the same algorithm as
/// [`Neon`] but takes advantage of the WebAssembly SIMD128 instructions.
///
/// [`Neon`]: crate::engine::Neon
/// [`NoSimd`]: crate::engine::NoSimd
#[derive(Clone, Copy)]
pub struct Wasm {
    mul128: &'static Mul128,
    skew: &'static Skew,
}

impl Wasm {
    /// Creates new [`Wasm`], initializing all [tables]
    /// needed for encoding or decoding.
    ///
    /// Currently only difference between encoding/decoding is
    /// [`LogWalsh`] (128 kiB) which is only needed for decoding.
    ///
    /// [`LogWalsh`]: crate::engine::tables::LogWalsh
    pub fn new() -> Self {
        let mul128 = tables::get_mul128();
        let skew = tables::get_skew();

        Self { mul128, skew }
    }
}

impl Engine for Wasm {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        unsafe {
            self.fft_private_wasm(data, pos, size, truncated_size, skew_delta);
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
            self.ifft_private_wasm(data, pos, size, truncated_size, skew_delta);
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        unsafe {
            self.mul_wasm(x, log_m);
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        unsafe { Self::eval_poly_wasm(erasures, truncated_size) }
    }
}

// ======================================================================
// Wasm - IMPL Default

impl Default for Wasm {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// Wasm - PRIVATE

impl Wasm {
    #[target_feature(enable = "simd128")]
    unsafe fn mul_wasm(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];

        for chunk in x.iter_mut() {
            let x_ptr = chunk.as_mut_ptr();
            let (prod0_lo, prod0_hi) = Self::mul_128(
                v128_load(x_ptr.cast::<v128>()),
                v128_load(x_ptr.add(16 * 2).cast::<v128>()),
                lut,
            );
            let (prod1_lo, prod1_hi) = Self::mul_128(
                v128_load(x_ptr.add(16).cast::<v128>()),
                v128_load(x_ptr.add(16 * 3).cast::<v128>()),
                lut,
            );

            v128_store(x_ptr.cast::<v128>(), prod0_lo);
            v128_store(x_ptr.add(16).cast::<v128>(), prod1_lo);
            v128_store(x_ptr.add(16 * 2).cast::<v128>(), prod0_hi);
            v128_store(x_ptr.add(16 * 3).cast::<v128>(), prod1_hi);
        }
    }

    // Implementation of LEO_MUL_128
    #[inline(always)]
    unsafe fn mul_128(value_lo: v128, value_hi: v128, lut: &Multiply128lutT) -> (v128, v128) {
        let t0_lo = v128_load(core::ptr::from_ref::<u128>(&lut.lo[0]).cast::<v128>());
        let t1_lo = v128_load(core::ptr::from_ref::<u128>(&lut.lo[1]).cast::<v128>());
        let t2_lo = v128_load(core::ptr::from_ref::<u128>(&lut.lo[2]).cast::<v128>());
        let t3_lo = v128_load(core::ptr::from_ref::<u128>(&lut.lo[3]).cast::<v128>());

        let t0_hi = v128_load(core::ptr::from_ref::<u128>(&lut.hi[0]).cast::<v128>());
        let t1_hi = v128_load(core::ptr::from_ref::<u128>(&lut.hi[1]).cast::<v128>());
        let t2_hi = v128_load(core::ptr::from_ref::<u128>(&lut.hi[2]).cast::<v128>());
        let t3_hi = v128_load(core::ptr::from_ref::<u128>(&lut.hi[3]).cast::<v128>());

        let clr_mask = u8x16_splat(0x0F);
        let shift_4 = 4u32;

        let data_0 = v128_and(value_lo, clr_mask);
        let mut prod_lo = i8x16_swizzle(t0_lo, data_0);
        let mut prod_hi = i8x16_swizzle(t0_hi, data_0);

        let data_1 = u8x16_shr(value_lo, shift_4);
        prod_lo = v128_xor(prod_lo, i8x16_swizzle(t1_lo, data_1));
        prod_hi = v128_xor(prod_hi, i8x16_swizzle(t1_hi, data_1));

        let data_0 = v128_and(value_hi, clr_mask);
        prod_lo = v128_xor(prod_lo, i8x16_swizzle(t2_lo, data_0));
        prod_hi = v128_xor(prod_hi, i8x16_swizzle(t2_hi, data_0));

        let data_1 = u8x16_shr(value_hi, shift_4);
        prod_lo = v128_xor(prod_lo, i8x16_swizzle(t3_lo, data_1));
        prod_hi = v128_xor(prod_hi, i8x16_swizzle(t3_hi, data_1));

        (prod_lo, prod_hi)
    }

    // {x_lo, x_hi} ^= {y_lo, y_hi} * log_m
    // Implementation of LEO_MULADD_128
    #[inline(always)]
    unsafe fn muladd_128(
        x_lo: v128,
        x_hi: v128,
        y_lo: v128,
        y_hi: v128,
        lut: &Multiply128lutT,
    ) -> (v128, v128) {
        let (prod_lo, prod_hi) = Self::mul_128(y_lo, y_hi, lut);
        let x_lo = v128_xor(x_lo, prod_lo);
        let x_hi = v128_xor(x_hi, prod_hi);
        (x_lo, x_hi)
    }
}

// ======================================================================
// Wasm - PRIVATE - FFT (fast Fourier transform)

impl Wasm {
    // Implementation of LEO_FFTB_128
    #[inline(always)]
    unsafe fn fftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr = x.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();

        let mut x0_lo = v128_load(x_ptr.cast::<v128>());
        let mut x1_lo = v128_load(x_ptr.add(16).cast::<v128>());
        let mut x0_hi = v128_load(x_ptr.add(16 * 2).cast::<v128>());
        let mut x1_hi = v128_load(x_ptr.add(16 * 3).cast::<v128>());

        let mut y0_lo = v128_load(y_ptr.cast::<v128>());
        let mut y1_lo = v128_load(y_ptr.add(16).cast::<v128>());
        let mut y0_hi = v128_load(y_ptr.add(16 * 2).cast::<v128>());
        let mut y1_hi = v128_load(y_ptr.add(16 * 3).cast::<v128>());

        (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
        (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

        v128_store(x_ptr.cast::<v128>(), x0_lo);
        v128_store(x_ptr.add(16).cast::<v128>(), x1_lo);
        v128_store(x_ptr.add(16 * 2).cast::<v128>(), x0_hi);
        v128_store(x_ptr.add(16 * 3).cast::<v128>(), x1_hi);

        y0_lo = v128_xor(y0_lo, x0_lo);
        y1_lo = v128_xor(y1_lo, x1_lo);
        y0_hi = v128_xor(y0_hi, x0_hi);
        y1_hi = v128_xor(y1_hi, x1_hi);

        v128_store(y_ptr.cast::<v128>(), y0_lo);
        v128_store(y_ptr.add(16).cast::<v128>(), y1_lo);
        v128_store(y_ptr.add(16 * 2).cast::<v128>(), y0_hi);
        v128_store(y_ptr.add(16 * 3).cast::<v128>(), y1_hi);
    }

    // Partial butterfly, caller must do `GF_MODULUS` check with `xor`.
    #[inline(always)]
    unsafe fn fft_butterfly_partial(
        &self,
        x: &mut [[u8; 64]],
        y: &mut [[u8; 64]],
        log_m: GfElement,
    ) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.fftb_128(x_chunk, y_chunk, log_m);
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
            unsafe {
                self.fft_butterfly_partial(s0, s2, log_m02);
                self.fft_butterfly_partial(s1, s3, log_m02);
            }
        }

        // SECOND LAYER

        if log_m01 == GF_MODULUS {
            utils::xor(s1, s0);
        } else {
            unsafe {
                self.fft_butterfly_partial(s0, s1, log_m01);
            }
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            unsafe {
                self.fft_butterfly_partial(s2, s3, log_m23);
            }
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn fft_private_wasm(
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
                    unsafe {
                        self.fft_butterfly_partial(x, y, log_m);
                    }
                }

                r += 2;
            }
        }
    }
}

// ======================================================================
// Wasm - PRIVATE - IFFT (inverse fast Fourier transform)

impl Wasm {
    // Implementation of LEO_IFFTB_128
    #[inline(always)]
    unsafe fn ifftb_128(&self, x: &mut [u8; 64], y: &mut [u8; 64], log_m: GfElement) {
        let lut = &self.mul128[log_m as usize];
        let x_ptr = x.as_mut_ptr();
        let y_ptr = y.as_mut_ptr();

        let mut x0_lo = v128_load(x_ptr.cast::<v128>());
        let mut x1_lo = v128_load(x_ptr.add(16).cast::<v128>());
        let mut x0_hi = v128_load(x_ptr.add(16 * 2).cast::<v128>());
        let mut x1_hi = v128_load(x_ptr.add(16 * 3).cast::<v128>());

        let mut y0_lo = v128_load(y_ptr.cast::<v128>());
        let mut y1_lo = v128_load(y_ptr.add(16).cast::<v128>());
        let mut y0_hi = v128_load(y_ptr.add(16 * 2).cast::<v128>());
        let mut y1_hi = v128_load(y_ptr.add(16 * 3).cast::<v128>());

        y0_lo = v128_xor(y0_lo, x0_lo);
        y1_lo = v128_xor(y1_lo, x1_lo);
        y0_hi = v128_xor(y0_hi, x0_hi);
        y1_hi = v128_xor(y1_hi, x1_hi);

        v128_store(y_ptr.cast::<v128>(), y0_lo);
        v128_store(y_ptr.add(16).cast::<v128>(), y1_lo);
        v128_store(y_ptr.add(16 * 2).cast::<v128>(), y0_hi);
        v128_store(y_ptr.add(16 * 3).cast::<v128>(), y1_hi);

        (x0_lo, x0_hi) = Self::muladd_128(x0_lo, x0_hi, y0_lo, y0_hi, lut);
        (x1_lo, x1_hi) = Self::muladd_128(x1_lo, x1_hi, y1_lo, y1_hi, lut);

        v128_store(x_ptr.cast::<v128>(), x0_lo);
        v128_store(x_ptr.add(16).cast::<v128>(), x1_lo);
        v128_store(x_ptr.add(16 * 2).cast::<v128>(), x0_hi);
        v128_store(x_ptr.add(16 * 3).cast::<v128>(), x1_hi);
    }

    #[inline(always)]
    unsafe fn ifft_butterfly_partial(
        &self,
        x: &mut [[u8; 64]],
        y: &mut [[u8; 64]],
        log_m: GfElement,
    ) {
        for (x_chunk, y_chunk) in zip(x.iter_mut(), y.iter_mut()) {
            self.ifftb_128(x_chunk, y_chunk, log_m);
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
            unsafe {
                self.ifft_butterfly_partial(s0, s1, log_m01);
            }
        }

        if log_m23 == GF_MODULUS {
            utils::xor(s3, s2);
        } else {
            unsafe {
                self.ifft_butterfly_partial(s2, s3, log_m23);
            }
        }

        // SECOND LAYER

        if log_m02 == GF_MODULUS {
            utils::xor(s2, s0);
            utils::xor(s3, s1);
        } else {
            unsafe {
                self.ifft_butterfly_partial(s0, s2, log_m02);
                self.ifft_butterfly_partial(s1, s3, log_m02);
            }
        }
    }

    #[target_feature(enable = "simd128")]
    unsafe fn ifft_private_wasm(
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
                    unsafe {
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
    // Wasm SIMD128 Runtime Detection

    /// Returns true if the wasm runtime support simd128
    #[cfg(target_arch = "wasm32")]
    pub fn wasm_simd128_supported() -> bool {
        use std::sync::OnceLock;
        static SIMD128_SUPPORTED: OnceLock<bool> = OnceLock::new();

        *SIMD128_SUPPORTED.get_or_init(|| {
            // Minimal SIMD128 module: (module (func (result v128) (i8x16.splat i32.const 0)))
            // If WebAssembly.validate() returns true, SIMD128 is supported.
            const SIMD128_MODULE: &[u8] = &[
                0x00, 0x61, 0x73, 0x6d, // magic
                0x01, 0x00, 0x00, 0x00, // version
                0x01, 0x05,             // type section: id=1, size=5
                0x01,                   // 1 type
                0x60, 0x00, 0x01, 0x7b, // func type: () -> v128
                0x03, 0x02,             // function section: id=3, size=2
                0x01, 0x00,             // 1 func, type 0
                0x0a, 0x08,             // code section: id=10, size=8
                0x01,                   // 1 body
                0x06,                   // body size=6
                0x00,                   // 0 locals
                0x41, 0x00,             // i32.const 0
                0xfd, 0x0f,             // i8x16.splat
                0x0b,                   // end
            ];

            // WebAssembly.validate() is synchronous and returns a boolean.
            // It validates the module without executing it.
            // If the runtime supports SIMD128, the v128 type (0x7b) is valid.
            // If not, validation fails.
            js_sys::WebAssembly::validate(&js_sys::Uint8Array::from(SIMD128_MODULE))
                .unwrap_or(false)
        })
    }
}

// ======================================================================
// Wasm - PRIVATE - Evaluate polynomial

impl Wasm {
    #[target_feature(enable = "simd128")]
    unsafe fn eval_poly_wasm(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        utils::eval_poly(erasures, truncated_size);
    }
}
