//! A collection of utility functions and helpers to facilitate the implementation of the [`Engine`] trait.
//!
//! [`Engine`]: crate::engine::Engine

use crate::engine::{fwht, tables, Engine, GfElement, ShardStorage, GF_BITS, GF_ORDER};
use core::iter::zip;

// ======================================================================
// FUNCTIONS - PUBLIC

/// Evaluate Polynomial using Fast Walsh-Hadamard Transform (FWHT).
///
/// This function is designed to be inlined and be compiled with SIMD
/// features enabled within an Engine's implementation of `eval_poly`.
///
/// See [`Avx2`] for an example on how to do this.
///
/// [`Avx2`]: crate::engine::Avx2
#[inline(always)]
pub fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
    let log_walsh = tables::get_log_walsh();

    fwht::fwht(erasures, truncated_size);

    for (e, factor) in zip(erasures.iter_mut(), log_walsh.iter()) {
        let product = u32::from(*e) * u32::from(*factor);
        *e = add_mod(product as GfElement, (product >> GF_BITS) as GfElement);
    }

    fwht::fwht(erasures, GF_ORDER);
}

/// `x[] ^= y[]`
#[inline(always)]
pub fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
    debug_assert_eq!(xs.len(), ys.len());

    for (x_chunk, y_chunk) in zip(xs.iter_mut(), ys.iter()) {
        for (x, y) in zip(x_chunk.iter_mut(), y_chunk.iter()) {
            *x ^= y;
        }
    }
}

/// `data[x .. x + count] ^= data[y .. y + count]`
///
/// # Safety
///
/// Shard-ranges `x .. x + count` and `y .. y + count` must be within `data.len()` and must not
/// overlap.
#[inline(always)]
pub unsafe fn xor_within<S: ShardStorage>(data: &mut S, x: usize, y: usize, count: usize) {
    // SAFETY: Guaranteed by the caller.
    unsafe { data.xor_within(x, y, count) }
}

// ======================================================================
// FUNCTIONS - CRATE - Galois field operations

/// Some kind of addition.
#[inline(always)]
pub(crate) fn add_mod(x: GfElement, y: GfElement) -> GfElement {
    let sum = u32::from(x) + u32::from(y);
    (sum + (sum >> GF_BITS)) as GfElement
}

/// Some kind of subtraction.
#[inline(always)]
pub(crate) fn sub_mod(x: GfElement, y: GfElement) -> GfElement {
    let dif = u32::from(x).wrapping_sub(u32::from(y));
    dif.wrapping_add(dif >> GF_BITS) as GfElement
}

// ======================================================================
// FUNCTIONS - CRATE

/// FFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn fft_skew_end<S: ShardStorage>(
    engine: &impl Engine,
    data: &mut S,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.fft(data, pos, size, truncated_size, pos + size);
}

/// IFFT with `skew_delta = pos + size`.
#[inline(always)]
pub(crate) fn ifft_skew_end<S: ShardStorage>(
    engine: &impl Engine,
    data: &mut S,
    pos: usize,
    size: usize,
    truncated_size: usize,
) {
    engine.ifft(data, pos, size, truncated_size, pos + size);
}

// Formal derivative.
//
// `data.len()` must be a power of two.
pub(crate) fn formal_derivative<S: ShardStorage>(data: &mut S) {
    debug_assert!(data.len().is_power_of_two());

    for i in 1..data.len() {
        let width: usize = 1 << i.trailing_zeros();
        // SAFETY: `width` is at most `i` and `i + width <= data.len()` because `data.len()` is a
        // power of two, so shard-ranges `i - width .. i` and `i .. i + width` are within bounds and
        // don't overlap
        unsafe { xor_within(data, i - width, i, width) };
    }
}
