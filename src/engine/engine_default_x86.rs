use crate::engine::{Avx2, Engine, GfElement, NoSimd, ShardsRefMut, Ssse3, GF_ORDER};

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] that on x86 platforms at runtime chooses the best Engine.
#[derive(Clone)]
pub enum DefaultEngine {
    Avx2(Avx2),
    Ssse3(Ssse3),
    NoSimd(NoSimd),
}

impl Default for DefaultEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultEngine {
    /// Creates new [`DefaultEngine`] by chosing and initializing the underlying engine.
    ///
    /// The engine is chosen in the following order of preference:
    /// 1. [`Avx2`]
    /// 2. [`Ssse3`]
    /// 3. [`NoSimd`]
    pub fn new() -> Self {
        if is_x86_feature_detected!("avx2") {
            return DefaultEngine::Avx2(Avx2::new());
        }

        if is_x86_feature_detected!("ssse3") {
            return DefaultEngine::Ssse3(Ssse3::new());
        }

        DefaultEngine::NoSimd(NoSimd::new())
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
        match self {
            DefaultEngine::Avx2(e) => e.fft(data, pos, size, truncated_size, skew_delta),
            DefaultEngine::Ssse3(e) => e.fft(data, pos, size, truncated_size, skew_delta),
            DefaultEngine::NoSimd(e) => e.fft(data, pos, size, truncated_size, skew_delta),
        };
    }

    fn fwht(data: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        if is_x86_feature_detected!("avx2") {
            return Avx2::fwht(data, truncated_size);
        }

        if is_x86_feature_detected!("ssse3") {
            return Ssse3::fwht(data, truncated_size);
        }

        NoSimd::fwht(data, truncated_size)
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        match self {
            DefaultEngine::Avx2(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
            DefaultEngine::Ssse3(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
            DefaultEngine::NoSimd(e) => e.ifft(data, pos, size, truncated_size, skew_delta),
        };
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        match self {
            DefaultEngine::Avx2(_) => Avx2::mul(x, log_m),
            DefaultEngine::Ssse3(_) => Ssse3::mul(x, log_m),
            DefaultEngine::NoSimd(_) => NoSimd::mul(x, log_m),
        };
    }

    fn xor(x: &mut [u8], y: &[u8]) {
        if is_x86_feature_detected!("avx2") {
            return Avx2::xor(x, y);
        }

        if is_x86_feature_detected!("ssse3") {
            return Ssse3::xor(x, y);
        }

        NoSimd::xor(x, y)
    }
}
