#![doc = include_str!(concat!(env!("OUT_DIR"), "/README-rustdocified.md"))]
#![deny(missing_docs)]
#![warn(clippy::pedantic, clippy::cargo)]
// Allow some lints from the pedantic category
#![allow(
    clippy::must_use_candidate,
    clippy::cast_ptr_alignment,
    clippy::inline_always,
    clippy::missing_errors_doc,
    clippy::cast_possible_truncation,
    clippy::large_stack_arrays,
    clippy::wildcard_imports
)]

use core::fmt;
use std::collections::HashMap;

pub use crate::{
    decoder_result::{DecoderResult, RestoredOriginal},
    encoder_result::{EncoderResult, Recovery},
    reed_solomon::{ReedSolomonDecoder, ReedSolomonEncoder},
};

#[cfg(test)]
#[macro_use]
mod test_util;

mod decoder_result;
mod encoder_result;
mod reed_solomon;

pub mod algorithm {
    #![doc = include_str!("algorithm.md")]
}
pub mod engine;
pub mod rate;

// ======================================================================
// Error - PUBLIC

/// Represents all possible errors that can occur in this library.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Error {
    /// Given shard has different size than given or inferred shard size.
    ///
    /// - Shard size is given explicitly to encoders/decoders
    ///   and inferred for [`reed_solomon_simd::encode`]
    ///   and [`reed_solomon_simd::decode`].
    ///
    /// [`reed_solomon_simd::encode`]: crate::encode
    /// [`reed_solomon_simd::decode`]: crate::decode
    DifferentShardSize {
        /// Given or inferred shard size.
        shard_bytes: usize,
        /// Size of the given shard.
        got: usize,
    },

    /// Decoder was given two original shards with same index.
    DuplicateOriginalShardIndex {
        /// Given duplicate index.
        index: usize,
    },

    /// Decoder was given two recovery shards with same index.
    DuplicateRecoveryShardIndex {
        /// Given duplicate index.
        index: usize,
    },

    /// Decoder was given original shard with invalid index,
    /// i.e. `index >= original_count`.
    InvalidOriginalShardIndex {
        /// Configured number of original shards.
        original_count: usize,
        /// Given invalid index.
        index: usize,
    },

    /// Decoder was given recovery shard with invalid index,
    /// i.e. `index >= recovery_count`.
    InvalidRecoveryShardIndex {
        /// Configured number of recovery shards.
        recovery_count: usize,
        /// Given invalid index.
        index: usize,
    },

    /// Given or inferred shard size is invalid:
    /// Size must be non-zero and even.
    ///
    /// - Shard size is given explicitly to encoders/decoders
    ///   and inferred for [`reed_solomon_simd::encode`]
    ///   and [`reed_solomon_simd::decode`].
    ///
    /// [`reed_solomon_simd::encode`]: crate::encode
    /// [`reed_solomon_simd::decode`]: crate::decode
    InvalidShardSize {
        /// Given or inferred shard size.
        shard_bytes: usize,
    },

    /// Decoder was given too few shards.
    ///
    /// Decoding requires as many shards as there were original shards
    /// in total, in any combination of original shards and recovery shards.
    NotEnoughShards {
        /// Configured number of original shards.
        original_count: usize,
        /// Number of original shards given to decoder.
        original_received_count: usize,
        /// Number of recovery shards given to decoder.
        recovery_received_count: usize,
    },

    /// Encoder was given less than `original_count` original shards.
    TooFewOriginalShards {
        /// Configured number of original shards.
        original_count: usize,
        /// Number of original shards given to encoder.
        original_received_count: usize,
    },

    /// Encoder was given more than `original_count` original shards.
    TooManyOriginalShards {
        /// Configured number of original shards.
        original_count: usize,
    },

    /// Given `original_count` / `recovery_count` combination is not supported.
    UnsupportedShardCount {
        /// Given number of original shards.
        original_count: usize,
        /// Given number of recovery shards.
        recovery_count: usize,
    },
}

// ======================================================================
// Error - IMPL DISPLAY

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DifferentShardSize { shard_bytes, got } => {
                write!(
                    f,
                    "different shard size: expected {shard_bytes} bytes, got {got} bytes"
                )
            }

            Self::DuplicateOriginalShardIndex { index } => {
                write!(f, "duplicate original shard index: {index}")
            }

            Self::DuplicateRecoveryShardIndex { index } => {
                write!(f, "duplicate recovery shard index: {index}")
            }

            Self::InvalidOriginalShardIndex {
                original_count,
                index,
            } => {
                write!(
                    f,
                    "invalid original shard index: {index} >= original_count {original_count}",
                )
            }

            Self::InvalidRecoveryShardIndex {
                recovery_count,
                index,
            } => {
                write!(
                    f,
                    "invalid recovery shard index: {index} >= recovery_count {recovery_count}",
                )
            }

            Self::InvalidShardSize { shard_bytes } => {
                write!(
                    f,
                    "invalid shard size: {shard_bytes} bytes (must non-zero and multiple of 2)"
                )
            }

            Self::NotEnoughShards {
                original_count,
                original_received_count,
                recovery_received_count,
            } => {
                write!(
                    f,
                    "not enough shards: {original_received_count} original + {recovery_received_count} recovery < {original_count} original_count",
                )
            }

            Self::TooFewOriginalShards {
                original_count,
                original_received_count,
            } => {
                write!(
                    f,
                    "too few original shards: got {original_received_count} shards while original_count is {original_count}"
                )
            }

            Self::TooManyOriginalShards { original_count } => {
                write!(
                    f,
                    "too many original shards: got more than original_count ({original_count}) shards"
                )
            }

            Self::UnsupportedShardCount {
                original_count,
                recovery_count,
            } => {
                write!(
                    f,
                    "unsupported shard count: {original_count} original shards with {recovery_count} recovery shards"
                )
            }
        }
    }
}

// ======================================================================
// Error - IMPL ERROR

impl core::error::Error for Error {}

// ======================================================================
// FUNCTIONS - PUBLIC

/// Encodes in one go using [`ReedSolomonEncoder`],
/// returning generated recovery shards.
///
/// - Original shards have indexes `0..original_count`
///   corresponding to the order in which they are given.
/// - Recovery shards have indexes `0..recovery_count`
///   corresponding to their position in the returned `Vec`.
/// - These same indexes must be used when decoding.
///
/// See [simple usage](crate#simple-usage) for an example.
pub fn encode<T>(
    original_count: usize,
    recovery_count: usize,
    original: T,
) -> Result<Vec<Vec<u8>>, Error>
where
    T: IntoIterator,
    T::Item: AsRef<[u8]>,
{
    if !ReedSolomonEncoder::supports(original_count, recovery_count) {
        return Err(Error::UnsupportedShardCount {
            original_count,
            recovery_count,
        });
    }

    let mut original = original.into_iter();

    let (shard_bytes, first) = if let Some(first) = original.next() {
        (first.as_ref().len(), first)
    } else {
        return Err(Error::TooFewOriginalShards {
            original_count,
            original_received_count: 0,
        });
    };

    let mut encoder = ReedSolomonEncoder::new(original_count, recovery_count, shard_bytes)?;

    encoder.add_original_shard(first)?;
    for original in original {
        encoder.add_original_shard(original)?;
    }

    let result = encoder.encode()?;

    Ok(result.recovery_iter().map(<[u8]>::to_vec).collect())
}

/// Decodes in one go using [`ReedSolomonDecoder`],
/// returning restored original shards with their indexes.
///
/// - Given shard indexes must be the same that were used in encoding.
///
/// See [simple usage](crate#simple-usage) for an example and more details.
pub fn decode<O, R, OT, RT>(
    original_count: usize,
    recovery_count: usize,
    original: O,
    recovery: R,
) -> Result<HashMap<usize, Vec<u8>>, Error>
where
    O: IntoIterator<Item = (usize, OT)>,
    R: IntoIterator<Item = (usize, RT)>,
    OT: AsRef<[u8]>,
    RT: AsRef<[u8]>,
{
    if !ReedSolomonDecoder::supports(original_count, recovery_count) {
        return Err(Error::UnsupportedShardCount {
            original_count,
            recovery_count,
        });
    }

    let original = original.into_iter();
    let mut recovery = recovery.into_iter();

    let (shard_bytes, first_recovery) = if let Some(first_recovery) = recovery.next() {
        (first_recovery.1.as_ref().len(), first_recovery)
    } else {
        // NO RECOVERY SHARDS

        let original_received_count = original.count();
        if original_received_count == original_count {
            // Nothing to do, original data is complete.
            return Ok(HashMap::new());
        }

        return Err(Error::NotEnoughShards {
            original_count,
            original_received_count,
            recovery_received_count: 0,
        });
    };

    let mut decoder = ReedSolomonDecoder::new(original_count, recovery_count, shard_bytes)?;

    for (index, original) in original {
        decoder.add_original_shard(index, original)?;
    }

    decoder.add_recovery_shard(first_recovery.0, first_recovery.1)?;
    for (index, recovery) in recovery {
        decoder.add_recovery_shard(index, recovery)?;
    }

    let mut result = HashMap::new();
    for (index, original) in decoder.decode()?.restored_original_iter() {
        result.insert(index, original.to_vec());
    }

    Ok(result)
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{engine::DefaultEngine, rate::DefaultRate};

    // ============================================================
    // ROUNDTRIP

    #[test]
    fn roundtrip() {
        let original = test_util::generate_original(2, 1024, 123);

        let recovery = encode(2, 3, &original).unwrap();

        test_util::assert_hash(&recovery, test_util::LOW_2_3);

        let restored = decode(2, 3, [(0, ""); 0], [(0, &recovery[0]), (1, &recovery[1])]).unwrap();

        assert_eq!(restored.len(), 2);
        assert_eq!(restored[&0], original[0]);
        assert_eq!(restored[&1], original[1]);
    }

    // ==================================================
    // trait Send

    #[test]
    fn test_send() {
        fn assert_send<T: Send>() {}
        assert_send::<ReedSolomonEncoder>();
        assert_send::<ReedSolomonDecoder>();
        assert_send::<DefaultEngine>();
        assert_send::<DefaultRate<DefaultEngine>>();
        assert_send::<DecoderResult>();
        assert_send::<EncoderResult>();
        assert_send::<Error>();
    }

    // ==================================================
    // trait Sync

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<ReedSolomonEncoder>();
        assert_sync::<ReedSolomonDecoder>();
        assert_sync::<DefaultEngine>();
        assert_sync::<DefaultRate<DefaultEngine>>();
        assert_sync::<DecoderResult>();
        assert_sync::<EncoderResult>();
        assert_sync::<Error>();
    }

    // ============================================================
    // encode

    mod encode {
        use super::super::*;

        // ==================================================
        // ERRORS

        #[test]
        fn different_shard_size_with_different_original_shard_sizes() {
            assert_eq!(
                encode(2, 1, &[&[0u8; 64] as &[u8], &[0u8; 128]]),
                Err(Error::DifferentShardSize {
                    shard_bytes: 64,
                    got: 128
                })
            );
        }

        #[test]
        fn invalid_shard_size_with_empty_shard() {
            assert_eq!(
                encode(1, 1, &[&[0u8; 0]]),
                Err(Error::InvalidShardSize { shard_bytes: 0 })
            );
        }

        #[test]
        fn too_few_original_shards_with_zero_shards_given() {
            assert_eq!(
                encode(1, 1, &[] as &[&[u8]]),
                Err(Error::TooFewOriginalShards {
                    original_count: 1,
                    original_received_count: 0,
                })
            );
        }

        #[test]
        fn too_many_original_shards() {
            assert_eq!(
                encode(1, 1, &[[0u8; 64], [0u8; 64]]),
                Err(Error::TooManyOriginalShards { original_count: 1 })
            );
        }

        #[test]
        fn unsupported_shard_count_with_zero_original_count() {
            assert_eq!(
                encode(0, 1, &[] as &[&[u8]]),
                Err(Error::UnsupportedShardCount {
                    original_count: 0,
                    recovery_count: 1,
                })
            );
        }

        #[test]
        fn unsupported_shard_count_with_zero_recovery_count() {
            assert_eq!(
                encode(1, 0, &[[0u8; 64]]),
                Err(Error::UnsupportedShardCount {
                    original_count: 1,
                    recovery_count: 0,
                })
            );
        }
    }

    // ============================================================
    // decode

    mod decode {
        use super::super::*;

        #[test]
        fn no_original_missing_with_no_recovery_given() {
            let restored = decode(1, 1, [(0, &[0u8; 64])], [(0, ""); 0]).unwrap();
            assert!(restored.is_empty());
        }

        // ==================================================
        // ERRORS

        #[test]
        fn different_shard_size_with_different_original_shard_sizes() {
            assert_eq!(
                decode(
                    2,
                    1,
                    [(0, &[0u8; 64] as &[u8]), (1, &[0u8; 128])],
                    [(0, &[0u8; 64])],
                ),
                Err(Error::DifferentShardSize {
                    shard_bytes: 64,
                    got: 128
                })
            );
        }

        #[test]
        fn different_shard_size_with_different_recovery_shard_sizes() {
            assert_eq!(
                decode(
                    1,
                    2,
                    [(0, &[0u8; 64])],
                    [(0, &[0u8; 64] as &[u8]), (1, &[0u8; 128])],
                ),
                Err(Error::DifferentShardSize {
                    shard_bytes: 64,
                    got: 128
                })
            );
        }

        #[test]
        fn different_shard_size_with_empty_original_shard() {
            assert_eq!(
                decode(1, 1, [(0, &[0u8; 0])], [(0, &[0u8; 64])]),
                Err(Error::DifferentShardSize {
                    shard_bytes: 64,
                    got: 0
                })
            );
        }

        #[test]
        fn duplicate_original_shard_index() {
            assert_eq!(
                decode(2, 1, [(0, &[0u8; 64]), (0, &[0u8; 64])], [(0, &[0u8; 64])]),
                Err(Error::DuplicateOriginalShardIndex { index: 0 })
            );
        }

        #[test]
        fn duplicate_recovery_shard_index() {
            assert_eq!(
                decode(1, 2, [(0, &[0u8; 64])], [(0, &[0u8; 64]), (0, &[0u8; 64])]),
                Err(Error::DuplicateRecoveryShardIndex { index: 0 })
            );
        }

        #[test]
        fn invalid_original_shard_index() {
            assert_eq!(
                decode(1, 1, [(1, &[0u8; 64])], [(0, &[0u8; 64])]),
                Err(Error::InvalidOriginalShardIndex {
                    original_count: 1,
                    index: 1,
                })
            );
        }

        #[test]
        fn invalid_recovery_shard_index() {
            assert_eq!(
                decode(1, 1, [(0, &[0u8; 64])], [(1, &[0u8; 64])]),
                Err(Error::InvalidRecoveryShardIndex {
                    recovery_count: 1,
                    index: 1,
                })
            );
        }

        #[test]
        fn invalid_shard_size_with_empty_recovery_shard() {
            assert_eq!(
                decode(1, 1, [(0, &[0u8; 64])], [(0, &[0u8; 0])]),
                Err(Error::InvalidShardSize { shard_bytes: 0 })
            );
        }

        #[test]
        fn not_enough_shards() {
            assert_eq!(
                decode(1, 1, [(0, ""); 0], [(0, ""); 0]),
                Err(Error::NotEnoughShards {
                    original_count: 1,
                    original_received_count: 0,
                    recovery_received_count: 0,
                })
            );
        }

        #[test]
        fn unsupported_shard_count_with_zero_original_count() {
            assert_eq!(
                decode(0, 1, [(0, ""); 0], [(0, ""); 0]),
                Err(Error::UnsupportedShardCount {
                    original_count: 0,
                    recovery_count: 1,
                })
            );
        }

        #[test]
        fn unsupported_shard_count_with_zero_recovery_count() {
            assert_eq!(
                decode(1, 0, [(0, ""); 0], [(0, ""); 0]),
                Err(Error::UnsupportedShardCount {
                    original_count: 1,
                    recovery_count: 0,
                })
            );
        }
    }
}
