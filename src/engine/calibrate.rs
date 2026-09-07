//! Runtime calibration of engine selection.
//!
//! Some choices between engines cannot be made from CPUID. The clearest case is
//! [`Avx2Gfni`] versus [`Avx512Gfni`]: which one wins depends on how wide the
//! core's 512 bit datapath actually is, and on how much the halved instruction
//! and load/store count is worth on that core. Neither is exposed by CPUID.
//! Zen 5 reports identical CPUID in its 256 bit and 512 bit dispatch modes, and
//! on Intel server parts `vgf2p8affineqb zmm` measures at half the rate of the
//! `ymm` form -- the same ratio as on Zen 4 -- yet the 512 bit engine still
//! wins there, because its advantage is the halved instruction and load/store
//! count rather than the datapath width. There is no bit to test for that.
//!
//! So we measure instead. Both candidates run their `mul` kernel over an L1
//! resident buffer, and the faster one wins. The procedure takes a few
//! milliseconds and happens at most once per process, next to the multiply
//! table initialization that building either engine costs anyway.
//!
//! The CPUID checks are part of the same decision rather than a separate step
//! before it, so that once [`gfni_engine`] has resolved, picking an engine
//! costs a single relaxed atomic load on every CPU, GFNI or not.
//!
//! [`Avx2Gfni`]: crate::engine::Avx2Gfni
//! [`Avx512Gfni`]: crate::engine::Avx512Gfni

use crate::engine::{Avx2Gfni, Avx512Gfni, Engine, GfElement};
use alloc::vec;
use alloc::vec::Vec;
use core::hint::black_box;
use core::sync::atomic::{compiler_fence, AtomicU8, Ordering};

#[cfg(target_arch = "x86")]
use core::arch::x86::_rdtsc;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::_rdtsc;

// ======================================================================
// CONSTANTS

/// Buffer handed to the kernels, in 64 byte chunks. 4 KiB stays comfortably
/// inside L1d on every CPU that has GFNI, so the measurement reflects the
/// kernel rather than the memory hierarchy.
const CHUNKS: usize = 64;

/// `mul` calls per timed measurement. Long enough that the `rdtsc` overhead and
/// out of order slop around it are noise, short enough to be interrupted rarely.
const CALLS: usize = 32;

/// Timed measurements per candidate. The minimum over these is what counts, so
/// a few disturbed measurements do not change the outcome.
const ROUNDS: usize = 8;

/// TSC cycles of untimed warm up before any measurement is taken.
///
/// The first milliseconds of vector code in a process are not representative.
/// On Skylake-SP class cores the upper half of the vector unit is powered down,
/// and 512 bit instructions run at a fraction of their steady state rate until
/// it comes up; the multiply tables are still being faulted in as well. Both
/// effects were measured at roughly 4x, i.e. far larger than the difference the
/// calibration is trying to resolve, and they outlast a warm up counted in
/// iterations rather than in time. The TSC ticks at the CPU's nominal
/// frequency, so this is somewhere between one and four milliseconds.
const WARMUP: u64 = 4_000_000;

/// Arbitrary non-trivial multiplier. `mul` is data independent, so the exact
/// value only matters for not being a special case.
const LOG_M: GfElement = 0x1234;

// ======================================================================
// GfniEngine - PUBLIC CRATE

/// Which GFNI engine to use, if any.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GfniEngine {
    /// The CPU has no GFNI, or no AVX2 to go with it.
    None,
    Avx2,
    Avx512,
}

/// The GFNI engine to use on this CPU, resolved once per process.
///
/// Mirrors what `cpufeatures` does for CPUID: the fast path is a single relaxed
/// atomic load, and a race between two threads resolving at the same time is
/// benign because both arrive at the same answer.
#[inline]
pub(crate) fn gfni_engine() -> GfniEngine {
    match CHOICE.load(Ordering::Relaxed) {
        NONE => GfniEngine::None,
        AVX2 => GfniEngine::Avx2,
        AVX512 => GfniEngine::Avx512,
        _ => decide(),
    }
}

// ======================================================================
// PRIVATE

static CHOICE: AtomicU8 = AtomicU8::new(UNDECIDED);

const UNDECIDED: u8 = 0;
const NONE: u8 = 1;
const AVX2: u8 = 2;
const AVX512: u8 = 3;

#[cold]
fn decide() -> GfniEngine {
    cpufeatures::new!(has_avx512_gfni, "avx512f", "avx512vl", "avx512bw", "gfni");
    cpufeatures::new!(has_avx2_gfni, "avx2", "gfni");

    let (engine, code) = if has_avx512_gfni::get() {
        if faster(&Avx2Gfni::new(), &Avx512Gfni::new()) {
            (GfniEngine::Avx2, AVX2)
        } else {
            (GfniEngine::Avx512, AVX512)
        }
    } else if has_avx2_gfni::get() {
        (GfniEngine::Avx2, AVX2)
    } else {
        (GfniEngine::None, NONE)
    };

    CHOICE.store(code, Ordering::Relaxed);
    engine
}

/// `true` if `a` is at least as fast as `b`.
fn faster<A: Engine, B: Engine>(a: &A, b: &B) -> bool {
    let mut buf: Vec<[u8; 64]> = vec![[0; 64]; CHUNKS];

    unsafe {
        let start = _rdtsc();
        while _rdtsc().wrapping_sub(start) < WARMUP {
            run(a, &mut buf);
            run(b, &mut buf);
        }
    }

    let mut best_a = u64::MAX;
    let mut best_b = u64::MAX;

    for round in 0..ROUNDS {
        // Alternate the order so that a frequency ramp, or a downclock caused by
        // the wider kernel, cannot systematically penalise whichever goes second.
        if round % 2 == 0 {
            let (x, y) = (time(a, &mut buf), time(b, &mut buf));
            best_a = best_a.min(x);
            best_b = best_b.min(y);
        } else {
            let (y, x) = (time(b, &mut buf), time(a, &mut buf));
            best_b = best_b.min(y);
            best_a = best_a.min(x);
        }
    }

    best_a <= best_b
}

fn time<E: Engine>(engine: &E, buf: &mut [[u8; 64]]) -> u64 {
    // `rdtsc` can be reordered by a few tens of cycles, which is irrelevant next
    // to the thousands of cycles being measured. The compiler fences matter more:
    // without them the loop could be hoisted out of the timed region.
    unsafe {
        compiler_fence(Ordering::SeqCst);
        let start = _rdtsc();
        compiler_fence(Ordering::SeqCst);

        run(engine, buf);

        compiler_fence(Ordering::SeqCst);
        let end = _rdtsc();
        compiler_fence(Ordering::SeqCst);

        end.wrapping_sub(start)
    }
}

#[inline(always)]
fn run<E: Engine>(engine: &E, buf: &mut [[u8; 64]]) {
    for _ in 0..CALLS {
        engine.mul(black_box(buf), LOG_M);
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{NoSimd, ShardsRefMut};

    /// [`NoSimd`] with its `mul` kernel made `N` times as expensive, so that the
    /// expected winner of a race is known without relying on how two real
    /// engines happen to compare on the CPU running the test.
    struct Slower<const N: usize>(NoSimd);

    impl<const N: usize> Slower<N> {
        fn new() -> Self {
            Self(NoSimd::new())
        }
    }

    impl<const N: usize> Engine for Slower<N> {
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
            for _ in 0..N {
                self.0.mul(black_box(x), log_m);
            }
        }
    }

    #[test]
    fn picks_the_faster_engine() {
        assert!(faster(&NoSimd::new(), &Slower::<8>::new()));
        assert!(!faster(&Slower::<8>::new(), &NoSimd::new()));
    }

    #[test]
    fn resolves_once_and_agrees_with_cpuid() {
        cpufeatures::new!(has_avx512_gfni, "avx512f", "avx512vl", "avx512bw", "gfni");
        cpufeatures::new!(has_avx2_gfni, "avx2", "gfni");

        let engine = gfni_engine();

        // Resolving is idempotent, and the second call takes the cached path.
        assert_eq!(engine, gfni_engine());
        assert_ne!(CHOICE.load(Ordering::Relaxed), UNDECIDED);

        if has_avx512_gfni::get() {
            assert!(engine == GfniEngine::Avx2 || engine == GfniEngine::Avx512);
        } else if has_avx2_gfni::get() {
            assert_eq!(engine, GfniEngine::Avx2);
        } else {
            assert_eq!(engine, GfniEngine::None);
        }
    }
}
