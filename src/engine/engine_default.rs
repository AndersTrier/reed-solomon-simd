use crate::engine::{Engine, GfElement, NoSimd, ShardStorage, GF_ORDER};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::engine::{Avx2, Ssse3};

#[cfg(target_arch = "aarch64")]
use crate::engine::Neon;

/// The engine that [`DefaultEngine`] dispatches to.
///
/// [`Engine`] is generic over [`ShardStorage`] and therefore not object safe,
/// so the selected engine is stored in an enum instead of a trait object.
enum Inner {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Avx2(Avx2),
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    Ssse3(Ssse3),
    #[cfg(target_arch = "aarch64")]
    Neon(Neon),
    NoSimd(NoSimd),
}

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] that at runtime selects the best Engine.
pub struct DefaultEngine(Inner);

impl DefaultEngine {
    /// Creates new [`DefaultEngine`] by chosing and initializing the underlying engine.
    ///
    /// On x86(-64) the engine is chosen in the following order of preference:
    /// 1. [`Avx2`]
    /// 2. [`Ssse3`]
    /// 3. [`NoSimd`]
    ///
    /// On `AArch64` the engine is chosen in the following order of preference:
    /// 1. [`Neon`]
    /// 2. [`NoSimd`]
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            cpufeatures::new!(has_avx2, "avx2");
            if has_avx2::get() {
                return Self(Inner::Avx2(Avx2::new()));
            }

            cpufeatures::new!(has_ssse3, "ssse3");
            if has_ssse3::get() {
                return Self(Inner::Ssse3(Ssse3::new()));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            cpufeatures::new!(has_neon, "neon");
            if has_neon::get() {
                return Self(Inner::Neon(Neon::new()));
            }
        }

        Self(Inner::NoSimd(NoSimd::new()))
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
    fn fft<S: ShardStorage>(
        &self,
        data: &mut S,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        match &self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Avx2(engine) => engine.fft(data, pos, size, truncated_size, skew_delta),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Ssse3(engine) => engine.fft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "aarch64")]
            Inner::Neon(engine) => engine.fft(data, pos, size, truncated_size, skew_delta),
            Inner::NoSimd(engine) => engine.fft(data, pos, size, truncated_size, skew_delta),
        }
    }

    fn ifft<S: ShardStorage>(
        &self,
        data: &mut S,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        match &self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Avx2(engine) => engine.ifft(data, pos, size, truncated_size, skew_delta),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Ssse3(engine) => engine.ifft(data, pos, size, truncated_size, skew_delta),
            #[cfg(target_arch = "aarch64")]
            Inner::Neon(engine) => engine.ifft(data, pos, size, truncated_size, skew_delta),
            Inner::NoSimd(engine) => engine.ifft(data, pos, size, truncated_size, skew_delta),
        }
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        match &self.0 {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Avx2(engine) => engine.mul(x, log_m),
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            Inner::Ssse3(engine) => engine.mul(x, log_m),
            #[cfg(target_arch = "aarch64")]
            Inner::Neon(engine) => engine.mul(x, log_m),
            Inner::NoSimd(engine) => engine.mul(x, log_m),
        }
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            cpufeatures::new!(has_avx2, "avx2");
            if has_avx2::get() {
                return Avx2::eval_poly(erasures, truncated_size);
            }

            cpufeatures::new!(has_ssse3, "ssse3");
            if has_ssse3::get() {
                return Ssse3::eval_poly(erasures, truncated_size);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            cpufeatures::new!(has_neon, "neon");
            if has_neon::get() {
                return Neon::eval_poly(erasures, truncated_size);
            }
        }

        NoSimd::eval_poly(erasures, truncated_size);
    }
}
