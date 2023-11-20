use crate::engine::{Avx2, Engine, GfElement, NoSimd, ShardsRefMut, Ssse3, GF_ORDER};

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] that on x86 platforms at runtime chooses the best Engine.
pub struct DefaultEngine(Box<dyn Engine>);

impl DefaultEngine {
    /// Creates new [`DefaultEngine`] by chosing and initializing the underlying engine.
    ///
    /// The engine is chosen in the following order of preference:
    /// 1. [`Avx2`]
    /// 2. [`Ssse3`]
    /// 3. [`NoSimd`]
    pub fn new() -> Self {
        if is_x86_feature_detected!("avx2") {
            return DefaultEngine(Box::new(Avx2::new()));
        }

        if is_x86_feature_detected!("ssse3") {
            return DefaultEngine(Box::new(Ssse3::new()));
        }

        DefaultEngine(Box::new(NoSimd::new()))
    }
}

// ======================================================================
// DefaultEngine - IMPL Default

impl Default for DefaultEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// DefaultEngine - IMPL Engine

impl Engine for DefaultEngine {
    fn fft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.0.fft(data, pos, size, truncated_size, skew_delta)
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
        self.0.ifft(data, pos, size, truncated_size, skew_delta)
    }

    fn mul(&self, x: &mut [u8], log_m: GfElement) {
        self.0.mul(x, log_m)
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
