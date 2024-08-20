use rand::Rng;
use std::collections::HashMap;

use reed_solomon_simd::engine::NoSimd;
use reed_solomon_simd::rate::{DefaultRateEncoder, RateEncoder};
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use reed_solomon_simd::engine::{Avx2, Ssse3};

#[cfg(target_arch = "aarch64")]
use reed_solomon_simd::engine::Neon;

#[test]
fn readme_example_1() -> Result<(), reed_solomon_simd::Error> {
    let original = [
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
        b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
        b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
    ];

    let recovery = reed_solomon_simd::encode(
        3,        // total number of original shards
        5,        // total number of recovery shards
        original, // all original shards
    )?;

    let restored = reed_solomon_simd::decode(
        3, // total number of original shards
        5, // total number of recovery shards
        [
            // provided original shards with indexes
            (1, &original[1]),
        ],
        [
            // provided recovery shards with indexes
            (1, &recovery[1]),
            (4, &recovery[4]),
        ],
    )?;

    assert_eq!(restored[&0], original[0]);
    assert_eq!(restored[&2], original[2]);

    Ok(())
}

#[test]
fn readme_example_2() -> Result<(), reed_solomon_simd::Error> {
    let original = [
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
        b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
        b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
    ];

    let mut encoder = ReedSolomonEncoder::new(
        3,  // total number of original shards
        5,  // total number of recovery shards
        64, // shard size in bytes
    )?;

    for original in original {
        encoder.add_original_shard(original)?;
    }

    let result = encoder.encode()?;
    let recovery: Vec<_> = result.recovery_iter().collect();

    let mut decoder = ReedSolomonDecoder::new(
        3,  // total number of original shards
        5,  // total number of recovery shards
        64, // shard size in bytes
    )?;

    decoder.add_original_shard(1, original[1])?;
    decoder.add_recovery_shard(1, recovery[1])?;
    decoder.add_recovery_shard(4, recovery[4])?;

    let result = decoder.decode()?;
    let restored: HashMap<_, _> = result.restored_original_iter().collect();

    assert_eq!(restored[&0], original[0]);
    assert_eq!(restored[&2], original[2]);

    Ok(())
}

#[test]
fn engine_intercompatibility() {
    let engine_nosimd = NoSimd::new();
    let mut encoder = DefaultRateEncoder::new(3, 5, 64, engine_nosimd, None).unwrap();

    let mut original_count;
    let mut recovery_count;
    let mut rng = rand::thread_rng();
    loop {
        original_count = rng.gen_range(0..65536);
        recovery_count = rng.gen_range(0..65536);
        if DefaultRateEncoder::<NoSimd>::supports(original_count, recovery_count) {
            break;
        }
    }

    let original = [
        b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
        b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
        b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
    ];

    for original in original {
        encoder.add_original_shard(original).unwrap();
    }

    let result = encoder.encode().unwrap();
    let recovery: Vec<_> = result.recovery_iter().collect();

    let mut decoder = ReedSolomonDecoder::new(
        3,  // total number of original shards
        5,  // total number of recovery shards
        64, // shard size in bytes
    )
    .unwrap();

    decoder.add_original_shard(1, original[1]).unwrap();
    decoder.add_recovery_shard(1, recovery[1]).unwrap();
    decoder.add_recovery_shard(4, recovery[4]).unwrap();

    let result = decoder.decode().unwrap();
    let restored: HashMap<_, _> = result.restored_original_iter().collect();

    assert_eq!(restored[&0], original[0]);
    assert_eq!(restored[&2], original[2]);
}
