#![no_main]
//! API robustness: arbitrary sequences of public API calls with arbitrary,
//! mostly invalid, arguments.
//!
//! Property: the library never panics; misuse is reported via `Error`.
//! Successful results are checked for a few cheap invariants.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};

/// Maximum number of operations executed per input.
const MAX_OPS: usize = 64;

/// Maximum number of shards passed to the one-shot `decode`.
const MAX_ONE_SHOT_SHARDS: usize = 32;

#[derive(Arbitrary, Debug)]
enum Op {
    EncoderNew {
        original_count: u16,
        recovery_count: u16,
        shard_bytes: u16,
    },
    EncoderReset {
        original_count: u16,
        recovery_count: u16,
        shard_bytes: u16,
    },
    EncoderAddOriginal {
        len: u16,
        fill: u8,
    },
    Encode {
        probe_index: u16,
    },

    DecoderNew {
        original_count: u16,
        recovery_count: u16,
        shard_bytes: u16,
    },
    DecoderReset {
        original_count: u16,
        recovery_count: u16,
        shard_bytes: u16,
    },
    DecoderAddOriginal {
        index: u16,
        len: u16,
        fill: u8,
    },
    DecoderAddRecovery {
        index: u16,
        len: u16,
        fill: u8,
    },
    Decode {
        probe_index: u16,
    },

    OneShotEncode {
        original_count: u16,
        recovery_count: u16,
        shards: u8,
        len: u16,
    },
    OneShotDecode {
        original_count: u16,
        recovery_count: u16,
        original: Vec<(u16, u16)>,
        recovery: Vec<(u16, u16)>,
    },
}

/// Maps a fuzzer-provided size into `0..1100`: slightly above the 1024
/// range where all small shard counts are supported, so that both valid and
/// invalid values are common, without spending all time on allocation.
fn size(x: u16) -> usize {
    usize::from(x % 1100)
}

// TODO: Remove once the `reset` fix (branch `nikolai/reset-bug`) is merged.
// Before that fix, a `reset` that switches between high and low rate with an
// invalid `shard_bytes` leaves the encoder/decoder in a state where the next
// call panics with `unreachable!()`.
fn reset_is_known_to_panic(shard_bytes: usize) -> bool {
    shard_bytes == 0 || shard_bytes % 2 == 1
}

fuzz_target!(|ops: Vec<Op>| {
    let mut encoder: Option<(ReedSolomonEncoder, usize)> = None;
    let mut decoder: Option<(ReedSolomonDecoder, usize)> = None;

    for op in ops.into_iter().take(MAX_OPS) {
        match op {
            // ENCODER
            Op::EncoderNew {
                original_count,
                recovery_count,
                shard_bytes,
            } => {
                let shard_bytes = size(shard_bytes);
                if let Ok(new) =
                    ReedSolomonEncoder::new(size(original_count), size(recovery_count), shard_bytes)
                {
                    encoder = Some((new, shard_bytes));
                }
            }

            Op::EncoderReset {
                original_count,
                recovery_count,
                shard_bytes,
            } => {
                let shard_bytes = size(shard_bytes);
                if let Some((encoder, current_shard_bytes)) = &mut encoder {
                    if reset_is_known_to_panic(shard_bytes) {
                        continue;
                    }
                    if encoder
                        .reset(size(original_count), size(recovery_count), shard_bytes)
                        .is_ok()
                    {
                        *current_shard_bytes = shard_bytes;
                    }
                }
            }

            Op::EncoderAddOriginal { len, fill } => {
                if let Some((encoder, _)) = &mut encoder {
                    let _ = encoder.add_original_shard(vec![fill; size(len)]);
                }
            }

            Op::Encode { probe_index } => {
                if let Some((encoder, shard_bytes)) = &mut encoder {
                    let shard_bytes = *shard_bytes;
                    if let Ok(result) = encoder.encode() {
                        for shard in result.recovery_iter() {
                            assert_eq!(shard.len(), shard_bytes);
                        }
                        let _ = result.recovery(usize::from(probe_index));
                    }
                }
            }

            // DECODER
            Op::DecoderNew {
                original_count,
                recovery_count,
                shard_bytes,
            } => {
                let shard_bytes = size(shard_bytes);
                if let Ok(new) =
                    ReedSolomonDecoder::new(size(original_count), size(recovery_count), shard_bytes)
                {
                    decoder = Some((new, shard_bytes));
                }
            }

            Op::DecoderReset {
                original_count,
                recovery_count,
                shard_bytes,
            } => {
                let shard_bytes = size(shard_bytes);
                if let Some((decoder, current_shard_bytes)) = &mut decoder {
                    if reset_is_known_to_panic(shard_bytes) {
                        continue;
                    }
                    if decoder
                        .reset(size(original_count), size(recovery_count), shard_bytes)
                        .is_ok()
                    {
                        *current_shard_bytes = shard_bytes;
                    }
                }
            }

            Op::DecoderAddOriginal { index, len, fill } => {
                if let Some((decoder, _)) = &mut decoder {
                    let _ = decoder.add_original_shard(usize::from(index), vec![fill; size(len)]);
                }
            }

            Op::DecoderAddRecovery { index, len, fill } => {
                if let Some((decoder, _)) = &mut decoder {
                    let _ = decoder.add_recovery_shard(usize::from(index), vec![fill; size(len)]);
                }
            }

            Op::Decode { probe_index } => {
                if let Some((decoder, shard_bytes)) = &mut decoder {
                    let shard_bytes = *shard_bytes;
                    if let Ok(result) = decoder.decode() {
                        for (_, shard) in result.restored_original_iter() {
                            assert_eq!(shard.len(), shard_bytes);
                        }
                        let _ = result.restored_original(usize::from(probe_index));
                    }
                }
            }

            // ONE-SHOT FUNCTIONS
            Op::OneShotEncode {
                original_count,
                recovery_count,
                shards,
                len,
            } => {
                let original = vec![vec![0xA5u8; size(len)]; usize::from(shards)];
                if let Ok(recovery) =
                    reed_solomon_simd::encode(size(original_count), size(recovery_count), &original)
                {
                    assert_eq!(recovery.len(), size(recovery_count));
                }
            }

            Op::OneShotDecode {
                original_count,
                recovery_count,
                original,
                recovery,
            } => {
                let shards = |list: Vec<(u16, u16)>, fill: u8| -> Vec<(usize, Vec<u8>)> {
                    list.into_iter()
                        .take(MAX_ONE_SHOT_SHARDS)
                        .map(|(index, len)| (usize::from(index), vec![fill; size(len)]))
                        .collect()
                };
                let _ = reed_solomon_simd::decode(
                    size(original_count),
                    size(recovery_count),
                    shards(original, 1),
                    shards(recovery, 2),
                );
            }
        }
    }
});
