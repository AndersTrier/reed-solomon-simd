#![no_main]
//! Roundtrip: encode, erase an arbitrary subset of shards, feed the rest to
//! the decoder in arbitrary order and check that restoration is exact.
//!
//! Each input runs two rounds on the *same* `ReedSolomonEncoder` /
//! `ReedSolomonDecoder` (the second one via `reset`) with independent
//! parameters, to catch stale state in reused working space and in
//! high <-> low rate switches.

use arbitrary::{Result, Unstructured};
use libfuzzer_sys::fuzz_target;
use reed_solomon_simd::{Error, ReedSolomonDecoder, ReedSolomonEncoder};

/// Upper bound on `original_count` and `recovery_count`.
/// Kept modest so that executions stay fast; see `fuzz/PLAN.md`.
const MAX_COUNT: usize = 1024;

/// `shard_bytes` is `2 * [1, MAX_HALF_SHARD_BYTES]`, which covers shard
/// sizes below, equal to and above the 64-byte chunk size, including the
/// special encoding of a partial last chunk.
const MAX_HALF_SHARD_BYTES: usize = 100;

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let mut encoder = None;
    let mut decoder = None;
    for _ in 0..2 {
        if round(&mut u, &mut encoder, &mut decoder).is_err() {
            break;
        }
    }
});

fn round(
    u: &mut Unstructured,
    encoder: &mut Option<ReedSolomonEncoder>,
    decoder: &mut Option<ReedSolomonDecoder>,
) -> Result<()> {
    let original_count = u.int_in_range(1..=MAX_COUNT)?;
    let recovery_count = u.int_in_range(1..=MAX_COUNT)?;
    let shard_bytes = 2 * u.int_in_range(1..=MAX_HALF_SHARD_BYTES)?;

    if !ReedSolomonEncoder::supports(original_count, recovery_count) {
        return Ok(());
    }

    let original = original_shards(u, original_count, shard_bytes);

    // ENCODE

    let encoder = match encoder {
        Some(encoder) => {
            encoder
                .reset(original_count, recovery_count, shard_bytes)
                .unwrap();
            encoder
        }
        None => encoder
            .insert(ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes).unwrap()),
    };
    for shard in &original {
        encoder.add_original_shard(shard).unwrap();
    }
    let recovery: Vec<Vec<u8>> = encoder
        .encode()
        .unwrap()
        .recovery_iter()
        .map(<[u8]>::to_vec)
        .collect();
    assert_eq!(recovery.len(), recovery_count);
    assert!(recovery.iter().all(|shard| shard.len() == shard_bytes));

    // CHOOSE SURVIVING SHARDS
    //
    // `Shard::Original(i)` / `Shard::Recovery(i)` in a fuzzer-chosen order;
    // the first `received` of them are given to the decoder.

    let mut shards: Vec<Shard> = (0..original_count)
        .map(Shard::Original)
        .chain((0..recovery_count).map(Shard::Recovery))
        .collect();
    shuffle(u, &mut shards);
    let received = u.int_in_range(0..=shards.len())?;

    // DECODE

    let decoder = match decoder {
        Some(decoder) => {
            decoder
                .reset(original_count, recovery_count, shard_bytes)
                .unwrap();
            decoder
        }
        None => decoder
            .insert(ReedSolomonDecoder::new(original_count, recovery_count, shard_bytes).unwrap()),
    };

    let mut original_received = vec![false; original_count];
    for &shard in &shards[..received] {
        match shard {
            Shard::Original(i) => {
                decoder.add_original_shard(i, &original[i]).unwrap();
                original_received[i] = true;
            }
            Shard::Recovery(i) => decoder.add_recovery_shard(i, &recovery[i]).unwrap(),
        }
    }

    match decoder.decode() {
        Ok(result) => {
            assert!(
                received >= original_count,
                "decode succeeded with {received} < {original_count} shards"
            );
            for (i, &was_received) in original_received.iter().enumerate() {
                match result.restored_original(i) {
                    Some(restored) => {
                        assert!(!was_received, "restored original {i} which was given");
                        assert_eq!(
                            restored, original[i],
                            "wrong restoration of original {i} \
                             (original_count={original_count}, \
                             recovery_count={recovery_count}, shard_bytes={shard_bytes})"
                        );
                    }
                    None => assert!(was_received, "missing original {i} was not restored"),
                }
            }
        }
        Err(Error::NotEnoughShards { .. }) => assert!(received < original_count),
        Err(err) => panic!("unexpected decode error: {err:?}"),
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum Shard {
    Original(usize),
    Recovery(usize),
}

/// Original shard data from fuzzer input, padded deterministically once the
/// input is exhausted (inputs are much smaller than `count * shard_bytes`).
fn original_shards(u: &mut Unstructured, count: usize, shard_bytes: usize) -> Vec<Vec<u8>> {
    let seed: u8 = u.arbitrary().unwrap_or(0);
    (0..count)
        .map(|i| {
            (0..shard_bytes)
                .map(|j| {
                    u.arbitrary()
                        .unwrap_or((i as u8).wrapping_mul(31) ^ (j as u8) ^ seed)
                })
                .collect()
        })
        .collect()
}

/// Fisher-Yates shuffle driven by fuzzer input; identity once exhausted.
fn shuffle<T>(u: &mut Unstructured, items: &mut [T]) {
    for i in (1..items.len()).rev() {
        let j = u.int_in_range(0..=i).unwrap_or(i);
        items.swap(i, j);
    }
}
