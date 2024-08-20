use std::collections::HashMap;
use reed_solomon_simd::{ReedSolomonEncoder, ReedSolomonDecoder};

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
