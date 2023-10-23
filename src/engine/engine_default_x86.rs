use once_cell::sync::OnceCell;

use crate::engine::{Avx2, Engine, GfElement, NoSimd, ShardsRefMut, Ssse3, GF_ORDER};

// ======================================================================
// STATIC - PRIVATE

static CHOSEN_ENGINE: OnceCell<InnerEngine> = OnceCell::new();

// ======================================================================
// FUNCTIONS - PRIVATE

fn select_best_engine() -> InnerEngine {
    if is_x86_feature_detected!("avx2") {
        return InnerEngine::Avx2(Avx2::new());
    }

    if is_x86_feature_detected!("ssse3") {
        return InnerEngine::Ssse3(Ssse3::new());
    }

    InnerEngine::NoSimd(NoSimd::new())
}

// ======================================================================
// InnerEngine - PRIVATE

enum InnerEngine {
    NoSimd(NoSimd),
    Avx2(Avx2),
    Ssse3(Ssse3),
}

fn initialize_best_engine() -> &'static InnerEngine {
    CHOSEN_ENGINE.get_or_init(select_best_engine)
}

// ======================================================================
// DefaultEngine - PUBLIC

impl Default for DefaultEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// [`Engine`] that on x86 platforms at runtime chooses the best Engine
///
/// The engine is chosen in the following order of preference:
/// 1. [`Avx2`]
/// 2. [`Ssse3`]
/// 3. [`NoSimd`]
#[derive(Clone)]
pub struct DefaultEngine();

impl DefaultEngine {
    /// TODO
    pub fn new() -> Self {
        initialize_best_engine();
        Self()
    }
}

impl Engine for DefaultEngine {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let engine = initialize_best_engine();
        match engine {
            InnerEngine::NoSimd(e) => e.fft(data, pos, size, truncated_size, skew_delta),
            InnerEngine::Avx2(e) => e.fft(data, pos, size, truncated_size, skew_delta),
            InnerEngine::Ssse3(e) => e.fft(data, pos, size, truncated_size, skew_delta),
        };
    }

    fn fwht(data: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        let engine = initialize_best_engine();
        match engine {
            InnerEngine::NoSimd(_) => NoSimd::fwht(data, truncated_size),
            InnerEngine::Avx2(_) => Avx2::fwht(data, truncated_size),
            InnerEngine::Ssse3(_) => Ssse3::fwht(data, truncated_size),
        };
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        let engine = initialize_best_engine();
        match engine {
            InnerEngine::NoSimd(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
            InnerEngine::Avx2(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
            InnerEngine::Ssse3(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
        };
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        let engine = initialize_best_engine();
        match engine {
            InnerEngine::NoSimd(e) => e.mul(x, log_m),
            InnerEngine::Avx2(e) => e.mul(x, log_m),
            InnerEngine::Ssse3(e) => e.mul(x, log_m),
        }
    }

    fn xor(x: &mut [u8], y: &[u8]) {
        let engine = initialize_best_engine();
        match engine {
            InnerEngine::NoSimd(_) => NoSimd::xor(x, y),
            InnerEngine::Avx2(_) => Avx2::xor(x, y),
            InnerEngine::Ssse3(_) => Ssse3::xor(x, y),
        };
    }
}
