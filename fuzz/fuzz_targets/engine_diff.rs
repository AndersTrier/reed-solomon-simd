#![no_main]
//! Differential testing of engines.
//!
//! For a fuzzer-chosen rate (`HighRate`, `LowRate` or `DefaultRate`) every engine must
//! produce bit-identical recovery shards and bit-identical restored shards:
//! - `Naive` (reference implementation, only for small shard counts as it's slow)
//! - `NoSimd`
//! - `Ssse3` / `Avx2` on x86(-64), `Neon` on AArch64, when supported by the CPU
//!
//! `Engine::mul` is additionally compared against `Naive` directly.

use arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
#[cfg(target_arch = "aarch64")]
use reed_solomon_simd::engine::Neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use reed_solomon_simd::engine::{Avx2, Ssse3};
use reed_solomon_simd::{
    engine::{Engine, GfElement, Naive, NoSimd},
    rate::{DefaultRate, HighRate, LowRate, Rate, RateDecoder, RateEncoder},
};

/// Upper bound on `original_count` and `recovery_count`.
const MAX_COUNT: usize = 1024;

/// `Naive` is only included when `original_count + recovery_count` is at
/// most this, as it's much slower than the other engines.
const NAIVE_MAX_TOTAL_COUNT: usize = 256;

/// `shard_bytes` is `2 * [1, MAX_HALF_SHARD_BYTES]`.
const MAX_HALF_SHARD_BYTES: usize = 80;

/// Maximum number of 64-byte chunks for the `mul` comparison.
const MAX_MUL_CHUNKS: usize = 4;

struct Case {
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    original: Vec<Vec<u8>>,
    /// Indexes into `original ++ recovery` given to the decoder.
    received: Vec<usize>,
}

/// `(recovery shards, restored original shards)`
type Output = (Vec<Vec<u8>>, Vec<(usize, Vec<u8>)>);

// ======================================================================
// RATE-LEVEL COMPARISON

/// Runs `$Rate` with every available engine and asserts identical output.
macro_rules! check_engines {
    ($Rate:ident, $case:expr) => {{
        let case: &Case = $case;
        #[allow(unused_mut)]
        let mut outputs: Vec<(&str, Option<Output>)> = Vec::new();

        if case.original_count + case.recovery_count <= NAIVE_MAX_TOTAL_COUNT {
            outputs.push(("Naive", run::<_, $Rate<_>>(Naive::new(), case)));
        }
        outputs.push(("NoSimd", run::<_, $Rate<_>>(NoSimd::new(), case)));

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if std::is_x86_feature_detected!("ssse3") {
                outputs.push(("Ssse3", run::<_, $Rate<_>>(Ssse3::new(), case)));
            }
            if std::is_x86_feature_detected!("avx2") {
                outputs.push(("Avx2", run::<_, $Rate<_>>(Avx2::new(), case)));
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if std::arch::is_aarch64_feature_detected!("neon") {
                outputs.push(("Neon", run::<_, $Rate<_>>(Neon::new(), case)));
            }
        }

        assert_all_equal(stringify!($Rate), case, &outputs);
    }};
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let _ = fuzz(&mut u);
});

fn fuzz(u: &mut Unstructured) -> Result<()> {
    mul_diff(u)?;

    // One rate per input: decoding has a fixed cost (`eval_poly` over all
    // `GF_ORDER` elements) so running every rate per input is slow.
    let rate = u.int_in_range(0..=2)?;
    let case = case(u)?;
    match rate {
        0 => check_engines!(HighRate, &case),
        1 => check_engines!(LowRate, &case),
        _ => check_engines!(DefaultRate, &case),
    }
    Ok(())
}

fn case(u: &mut Unstructured) -> Result<Case> {
    let original_count = u.int_in_range(1..=MAX_COUNT)?;
    let recovery_count = u.int_in_range(1..=MAX_COUNT)?;
    let shard_bytes = 2 * u.int_in_range(1..=MAX_HALF_SHARD_BYTES)?;

    let seed: u8 = u.arbitrary().unwrap_or(0);
    let original = (0..original_count)
        .map(|i| {
            (0..shard_bytes)
                .map(|j| {
                    u.arbitrary()
                        .unwrap_or((i as u8).wrapping_mul(7) ^ (j as u8) ^ seed)
                })
                .collect()
        })
        .collect();

    // Exactly the minimum number of shards, chosen by fuzzer-driven shuffle.
    let mut received: Vec<usize> = (0..original_count + recovery_count).collect();
    for i in (1..received.len()).rev() {
        let j = u.int_in_range(0..=i).unwrap_or(i);
        received.swap(i, j);
    }
    received.truncate(original_count);

    Ok(Case {
        original_count,
        recovery_count,
        shard_bytes,
        original,
        received,
    })
}

/// Encodes and decodes `case` with given engine and rate.
/// Returns `None` if the rate doesn't support the shard counts.
fn run<E: Engine, R: Rate<E>>(engine: E, case: &Case) -> Option<Output> {
    let Case {
        original_count,
        recovery_count,
        shard_bytes,
        ..
    } = *case;

    if !R::supports(original_count, recovery_count) {
        return None;
    }

    let mut encoder =
        R::encoder(original_count, recovery_count, shard_bytes, engine, None).unwrap();
    for shard in &case.original {
        encoder.add_original_shard(shard).unwrap();
    }
    let recovery: Vec<Vec<u8>> = encoder
        .encode()
        .unwrap()
        .recovery_iter()
        .map(<[u8]>::to_vec)
        .collect();
    let (engine, _) = encoder.into_parts();

    let mut decoder =
        R::decoder(original_count, recovery_count, shard_bytes, engine, None).unwrap();
    for &index in &case.received {
        if index < original_count {
            decoder
                .add_original_shard(index, &case.original[index])
                .unwrap();
        } else {
            let index = index - original_count;
            decoder.add_recovery_shard(index, &recovery[index]).unwrap();
        }
    }
    let restored: Vec<(usize, Vec<u8>)> = decoder
        .decode()
        .unwrap()
        .restored_original_iter()
        .map(|(index, shard)| (index, shard.to_vec()))
        .collect();

    for (index, shard) in &restored {
        assert_eq!(
            shard, &case.original[*index],
            "wrong restoration of original {index}"
        );
    }

    Some((recovery, restored))
}

fn assert_all_equal(rate: &str, case: &Case, outputs: &[(&str, Option<Output>)]) {
    let (reference_name, reference) = &outputs[0];
    for (name, output) in &outputs[1..] {
        assert!(
            output == reference,
            "{rate}: {name} differs from {reference_name} \
             (original_count={}, recovery_count={}, shard_bytes={})",
            case.original_count,
            case.recovery_count,
            case.shard_bytes,
        );
    }
}

// ======================================================================
// ENGINE-LEVEL COMPARISON

fn mul_diff(u: &mut Unstructured) -> Result<()> {
    let log_m: GfElement = u.arbitrary()?;
    let chunk_count = u.int_in_range(1..=MAX_MUL_CHUNKS)?;
    let mut input = vec![[0u8; 64]; chunk_count];
    for chunk in &mut input {
        u.fill_buffer(chunk)?;
    }

    let mut expected = input.clone();
    Naive::new().mul(&mut expected, log_m);

    let check = |name: &str, engine: &dyn Engine| {
        let mut actual = input.clone();
        engine.mul(&mut actual, log_m);
        assert!(
            actual == expected,
            "{name}::mul differs from Naive (log_m={log_m})"
        );
    };

    check("NoSimd", &NoSimd::new());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("ssse3") {
            check("Ssse3", &Ssse3::new());
        }
        if std::is_x86_feature_detected!("avx2") {
            check("Avx2", &Avx2::new());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            check("Neon", &Neon::new());
        }
    }

    Ok(())
}
