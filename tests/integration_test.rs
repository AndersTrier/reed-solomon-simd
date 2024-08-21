use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

fn generate_shards(shard_count: usize, shard_bytes: usize, seed: u8) -> Vec<Vec<u8>> {
    let mut rng = ChaCha8Rng::from_seed([seed; 32]);
    let mut shards = vec![vec![0u8; shard_bytes]; shard_count];
    for shard in &mut shards {
        rng.fill::<[u8]>(shard);
    }
    shards
}

#[test]
fn engine_intercompatibility() {
    let mut original_count;
    let mut recovery_count;
    let mut rng = ChaCha8Rng::from_seed([3; 32]);
    loop {
        original_count = rng.gen_range(0..65536);
        recovery_count = rng.gen_range(0..65536);
        if DefaultRateEncoder::<NoSimd>::supports(original_count, recovery_count) {
            break;
        }
    }

    let mut shard_bytes = rng.gen_range(2..513);

    if shard_bytes % 2 != 0 {
        shard_bytes -= 1;
    }

    let engine_nosimd = NoSimd::new();
    let mut encoder = DefaultRateEncoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        engine_nosimd,
        None,
    )
    .unwrap();

    let original = generate_shards(original_count, shard_bytes, 0);

    for shard in &original {
        encoder.add_original_shard(shard).unwrap();
    }

    let result = encoder.encode().unwrap();
    let recovery: Vec<_> = result.recovery_iter().collect();

    let mut decoder = ReedSolomonDecoder::new(original_count, recovery_count, shard_bytes).unwrap();

    // Let's add a random amount of shards
    let shards_to_add = rng.gen_range(original_count..(original_count + recovery_count + 1));
    let mut recovery_added = 0;

    for (idx, shard) in recovery.iter().enumerate().take(shards_to_add) {
        decoder.add_recovery_shard(idx, shard).unwrap();
        recovery_added += 1;
    }

    for (idx, shard) in original
        .iter()
        .enumerate()
        .take(shards_to_add - recovery_added)
    {
        decoder.add_original_shard(idx, shard).unwrap();
    }

    let result = decoder.decode().unwrap();

    for (idx, restored_shard) in result.restored_original_iter() {
        assert_eq!(restored_shard, original[idx]);
    }
}
