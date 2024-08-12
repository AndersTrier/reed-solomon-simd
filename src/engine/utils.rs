use crate::engine::{fwht, tables, GfElement, ShardsRefMut, GF_BITS, GF_ORDER};
use std::iter::zip;

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
        let product = u32::from(*e) * u32::from(*factor);
        *e = add_mod(product as GfElement, (product >> GF_BITS) as GfElement);
    }

    fwht::fwht(erasures, GF_ORDER);
}

/// `x[] ^= y[]`
#[inline(always)]
pub(crate) fn xor(xs: &mut [[u8; 64]], ys: &[[u8; 64]]) {
    debug_assert_eq!(xs.len(), ys.len());

    for (x_chunk, y_chunk) in zip(xs.iter_mut(), ys.iter()) {
        for (x, y) in zip(x_chunk.iter_mut(), y_chunk.iter()) {
            *x ^= y;
        }
    }
}

///// FFT with `skew_delta = pos + size`.
//#[inline(always)]
//fn fft_skew_end(
//    &self,
//    data: &mut ShardsRefMut,
//    pos: usize,
//    size: usize,
//    truncated_size: usize,
//) {
//    self.fft(data, pos, size, truncated_size, pos + size)
//}

/// Formal derivative.
pub(crate) fn formal_derivative(data: &mut ShardsRefMut) {
    for i in 1..data.len() {
        let width: usize = 1 << i.trailing_zeros();
        xor_within(data, i - width, i, width);
    }
}

///// IFFT with `skew_delta = pos + size`.
//#[inline(always)]
//fn ifft_skew_end(
//    &self,
//    data: &mut ShardsRefMut,
//    pos: usize,
//    size: usize,
//    truncated_size: usize,
//) {
//    self.ifft(data, pos, size, truncated_size, pos + size)
//}

/// `data[x .. x + count] ^= data[y .. y + count]`
///
/// Ranges must not overlap.
#[inline(always)]
pub(crate) fn xor_within(data: &mut ShardsRefMut, x: usize, y: usize, count: usize) {
    let (xs, ys) = data.flat2_mut(x, y, count);
    xor(xs, ys);
}
