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

pub use self::{
    engine_default::DefaultEngine, engine_naive::Naive, engine_nosimd::NoSimd, shards::ShardsRefMut,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub use self::{engine_avx2::Avx2, engine_ssse3::Ssse3};

#[cfg(target_arch = "aarch64")]
pub use self::engine_neon::Neon;

mod engine_default;
mod engine_naive;
mod engine_nosimd;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_avx2;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod engine_ssse3;

#[cfg(target_arch = "aarch64")]
mod engine_neon;

mod fwht;
mod shards;

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
// FUNCTIONS - PUBLIC - Galois field operations

/// Some kind of addition.
#[inline(always)]
pub fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = u32::from(x) + u32::from(y);
    (sum + (sum >> GF_BITS)) as GfElement
}

/// Some kind of subtraction.
#[inline(always)]
pub fn sub_mod(x: GfElement, y: GfElement) -> GfElement {
    let dif = u32::from(x).wrapping_sub(u32::from(y));
    dif.wrapping_add(dif >> GF_BITS) as GfElement
}

// ======================================================================
// FUNCTIONS - CRATE - Evaluate polynomial

// We have this function here instead of inside 'trait Engine' to allow
// it to be included and compiled with SIMD features enabled within the
// SIMD engines.
#[inline(always)]
pub(crate) fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    let log_walsh = tables::initialize_log_walsh();

    fwht::fwht(erasures, truncated_size);

    for (e, factor) in std::iter::zip(erasures.iter_mut(), log_walsh.iter()) {
        *e = ((u32::from(*e) * u32::from(*factor)) % u32::from(GF_MODULUS)) as GfElement;
    }

    fwht::fwht(erasures, GF_ORDER);
}

// ======================================================================
// FUNCTIONS - PUBLIC - misc

/// Returns smallest value that is greater than or equal to `a` and multiple of `b`,
/// or `None` if `b` is zero or operation would overflow.
///
/// - This function is available as [`usize::checked_next_multiple_of`] in nightly Rust.
///
/// # Examples
///
/// ```rust
/// use reed_solomon_simd::engine;
///
/// assert_eq!(engine::checked_next_multiple_of(20, 10), Some(20));
/// assert_eq!(engine::checked_next_multiple_of(27, 10), Some(30));
/// ```
///
/// [`usize::checked_next_multiple_of`]: https://doc.rust-lang.org/std/primitive.usize.html#method.checked_next_multiple_of
pub fn checked_next_multiple_of(a: usize, b: usize) -> Option<usize> {
    if b == 0 {
        None
    } else {
        let mut x = a / b;
        x += if a % b != 0 { 1 } else { 0 };
        x.checked_mul(b)
    }
}

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
    fn mul(&self, x: &mut [u8], log_m: GfElement);

    /// `x[] ^= y[]`
    fn xor(x: &mut [u8], y: &[u8])
    where
        Self: Sized;

    // ============================================================
    // PROVIDED

    /// In-place FWHT (fast Walsh-Hadamard transform).
    ///
    /// - This is used only in [`Engine::eval_poly`],
    ///   both directly and indirectly via [`initialize_log_walsh`].
    /// - `truncated_size` must be handled so that
    ///   [`Engine::eval_poly`] returns correct result.
    ///
    /// [`initialize_log_walsh`]: self::tables::initialize_log_walsh
    fn fwht(data: &mut [GfElement; GF_ORDER], truncated_size: usize)
    where
        Self: Sized,
    {
        fwht::fwht(data, truncated_size)
    }

    /// Evaluate polynomial.
    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize)
    where
        Self: Sized,
    {
        eval_poly(erasures, truncated_size)
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

    /// Formal derivative.
    fn formal_derivative(data: &mut ShardsRefMut)
    where
        Self: Sized,
    {
        for i in 1..data.len() {
            let width: usize = ((i ^ (i - 1)) + 1) >> 1;
            Self::xor_within(data, i - width, i, width);
        }
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

    /// `data[x .. x + count] ^= data[y .. y + count]`
    ///
    /// Ranges must not overlap.
    #[inline(always)]
    fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize)
    where
        Self: Sized,
    {
        let (xs, ys) = data.flat2_mut(x, y, count);
        Self::xor(xs, ys);
    }
}

// ======================================================================
// TESTS

// Engines are tested indirectly via roundtrip tests of HighRate and LowRate.

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // checked_next_multiple_of

    #[test]
    fn test_checked_next_multiple_of() {
        assert_eq!(checked_next_multiple_of(10, 0), None);
        assert_eq!(checked_next_multiple_of(usize::MAX, 2), None);

        assert_eq!(checked_next_multiple_of(99, 20), Some(100));
        assert_eq!(checked_next_multiple_of(100, 20), Some(100));
        assert_eq!(checked_next_multiple_of(101, 20), Some(120));
    }
}
