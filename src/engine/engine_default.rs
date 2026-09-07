use crate::engine::{Engine, GfElement, NoSimd, ShardsRefMut, GF_ORDER};
#[cfg(not(feature = "std"))]
use alloc::boxed::Box;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::engine::calibrate::{gfni_engine, GfniEngine};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::engine::{Avx2, Avx2Gfni, Avx512, Avx512Gfni, Ssse3};

#[cfg(target_arch = "aarch64")]
use crate::engine::Neon;

// ======================================================================
// DefaultEngine - PUBLIC

/// [`Engine`] that at runtime selects the best Engine.
pub struct DefaultEngine(Box<dyn Engine + Send + Sync>);

impl DefaultEngine {
    /// Creates new [`DefaultEngine`] by chosing and initializing the underlying engine.
    ///
    /// On x86(-64) the engine is chosen in the following order of preference:
    /// 1. [`Avx2Gfni`] or [`Avx512Gfni`]
    /// 2. [`Avx512`]
    /// 3. [`Avx2`]
    /// 4. [`Ssse3`]
    /// 5. [`NoSimd`]
    ///
    /// Which of the two GFNI engines wins depends on how wide the core's
    /// 512 bit datapath really is, and CPUID does not say: Zen 5 reports the
    /// same CPUID in its 256 bit and 512 bit dispatch modes, and Intel server
    /// parts advertise `AVX_VNNI` whether or not their GFNI unit is full rate.
    /// So the two are timed against each other the first time a
    /// [`DefaultEngine`] is built on a CPU that has both, and the outcome is
    /// cached for the rest of the process.
    ///
    /// On `AArch64` the engine is chosen in the following order of preference:
    /// 1. [`Neon`]
    /// 2. [`NoSimd`]
    pub fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            match gfni_engine() {
                GfniEngine::Avx2 => {
                    return Self(Box::new(Avx2Gfni::new()));
                }
                GfniEngine::Avx512 => {
                    return Self(Box::new(Avx512Gfni::new()));
                }
                GfniEngine::None => {}
            }

            cpufeatures::new!(has_avx512, "avx512f", "avx512vl", "avx512bw");
            if has_avx512::get() {
                return Self(Box::new(Avx512::new()));
            }

            cpufeatures::new!(has_avx2, "avx2");
            if has_avx2::get() {
                return Self(Box::new(Avx2::new()));
            }

            cpufeatures::new!(has_ssse3, "ssse3");
            if has_ssse3::get() {
                return Self(Box::new(Ssse3::new()));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            cpufeatures::new!(has_neon, "neon");
            if has_neon::get() {
                return Self(Box::new(Neon::new()));
            }
        }

        Self(Box::new(NoSimd::new()))
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
        self.0.fft(data, pos, size, truncated_size, skew_delta);
    }

    fn ifft(
        &self,
        data: &mut ShardsRefMut,
        pos: usize,
        size: usize,
        truncated_size: usize,
        skew_delta: usize,
    ) {
        self.0.ifft(data, pos, size, truncated_size, skew_delta);
    }

    fn mul(&self, x: &mut [[u8; 64]], log_m: GfElement) {
        self.0.mul(x, log_m);
    }

    fn eval_poly(erasures: &mut [GfElement; GF_ORDER], truncated_size: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            cpufeatures::new!(has_avx512, "avx512f", "avx512vl", "avx512bw");
            if has_avx512::get() {
                return Avx512::eval_poly(erasures, truncated_size);
            }

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
