//! Low-level building blocks for Reed-Solomon encoding/decoding.
//!
//! **This is an advanced module which is not needed for [simple usage] or [basic usage].**
//!
//! This module is relevant if you want to
//! - use [`rate`] module and need an [`Engine`] to use with it.
//! - create your own [`Engine`].
//! - understand/benchmark/test at low level.
//!
//! # Engines
//!
//! An [`Engine`] is an implementation of basic low-level algorithms
//! needed for Reed-Solomon encoding/decoding.
//!
//! - [`Naive`]
//!     - Simple reference implementation.
//! - [`NoSimd`]
//!     - Basic optimized engine without SIMD so that it works on all CPUs.
//! - [`Avx2`]
//!     - Optimized engine that takes advantage of the x86(-64) AVX2 SIMD instructions.
//! - [`Ssse3`]
//!     - Optimized engine that takes advantage of the x86(-64) SSSE3 SIMD instructions.
//! - [`Neon`]
//!     - Optimized engine that takes advantage of the AArch64 Neon SIMD instructions.
//! - [`DefaultEngine`]
//!     - Default engine which is used when no specific engine is given.
//!     - Automatically selects best engine at runtime.
//!
//! [simple usage]: crate#simple-usage
//! [basic usage]: crate#basic-usage
//! [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
//! [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
//! [`rate`]: crate::rate

pub(crate) use self::shards::Shards;

pub use self::{engine_nosimd::NoSimd, shards::ShardsRefMut};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::{engine_avx2::Avx2, engine_ssse3::Ssse3};

#[cfg(target_arch = "aarch64")]
pub use self::engine_neon::Neon;

//mod engine_default;
mod engine_nosimd;

///FIXME
pub type DefaultEngine = NoSimd;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_ssse3;

#[cfg(target_arch = "aarch64")]
mod engine_neon;

mod fft;
mod fwht;
mod shards;
pub(crate) mod utils;

pub mod tables;

// ======================================================================
// CONST - PUBLIC

/// Size of Galois field element [`GfElement`] in bits.
pub const GF_BITS: usize = 16;

/// Galois field order, i.e. number of elements.
pub const GF_ORDER: usize = 65536;

/// `GF_ORDER - 1`
pub const GF_MODULUS: GfElement = 65535;

/// Galois field polynomial.
pub const GF_POLYNOMIAL: usize = 0x1002D;

/// TODO
pub const CANTOR_BASIS: [GfElement; GF_BITS] = [
    0x0001, 0xACCA, 0x3C0E, 0x163E, 0xC582, 0xED2E, 0x914C, 0x4012, 0x6C98, 0x10D8, 0x6A72, 0xB900,
    0xFDB8, 0xFB34, 0xFF38, 0x991E,
];

// ======================================================================
// TYPE ALIASES - PUBLIC

/// Galois field element.
pub type GfElement = u16;

// ======================================================================
// Engine - PUBLIC

/// Implementation of basic low-level algorithms needed
/// for Reed-Solomon encoding/decoding.
///
/// These algorithms are not properly documented.
///
/// [`Naive`] engine is provided for those who want to
/// study the source code to understand [`Engine`].
pub trait Engine {
    // ============================================================
    // REQUIRED

    /// TODO    
    fn fft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement);

    /// TODO
    fn ifft_butterfly_partial(&self, x: &mut [[u8; 64]], y: &mut [[u8; 64]], log_m: GfElement);

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
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    );

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
    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    );

    /// `x[] *= log_m`
    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement);

    // ============================================================
    // PROVIDED

    /// Evaluate polynomial.
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize)
    where
        Self: Sized,
    {
        utils::eval_poly(erasures, truncated_size)
    }

    /// FFT with `skew_delta = pos + size`.
    #[inline(always)]
    fn fft_skew_end(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
    ) {
        self.fft(data, pos, size, truncated_size, pos + size)
    }

    /// IFFT with `skew_delta = pos + size`.
    #[inline(always)]
    fn ifft_skew_end(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
    ) {
        self.ifft(data, pos, size, truncated_size, pos + size)
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.
