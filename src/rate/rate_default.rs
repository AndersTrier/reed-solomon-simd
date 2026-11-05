use core::{cmp::Ordering, marker::PhantomData};

use crate::{
    engine::{Engine, GF_ORDER},
    rate::{
        DecoderWork, EncoderWork, HighRateDecoder, HighRateEncoder, LowRateDecoder, LowRateEncoder,
        Rate, RateDecoder, RateEncoder,
    },
    DecoderResult, EncoderResult, Error,
};

// ======================================================================
// FUNCTIONS - PRIVATE

fn use_high_rate(original_count: usize, recovery_count: usize) -> Result<bool, Error> {
    if original_count > GF_ORDER || recovery_count > GF_ORDER {
        return Err(Error::UnsupportedShardCount {
            original_count,
            recovery_count,
        });
    }

    let original_count_pow2 = original_count.next_power_of_two();
    let recovery_count_pow2 = recovery_count.next_power_of_two();

    let smaller_pow2 = core::cmp::min(original_count_pow2, recovery_count_pow2);
    let larger = core::cmp::max(original_count, recovery_count);

    if original_count == 0 || recovery_count == 0 || smaller_pow2 + larger > GF_ORDER {
        return Err(Error::UnsupportedShardCount {
            original_count,
            recovery_count,
        });
    }

    match original_count_pow2.cmp(&recovery_count_pow2) {
        Ordering::Less => {
            // The "correct" rate is generally faster here,
            // and also must be used if `recovery_count > 32768`.

            Ok(false)
        }

        Ordering::Greater => {
            // The "correct" rate is generally faster here,
            // and also must be used if `original_count > 32768`.

            Ok(true)
        }

        Ordering::Equal => {
            // Here counter-intuitively the "wrong" rate is generally faster
            // in decoding if `original_count` and `recovery_count` differ a lot.

            if original_count <= recovery_count {
                // Using the "wrong" rate on purpose.
                Ok(true)
            } else {
                // Using the "wrong" rate on purpose.
                Ok(false)
            }
        }
    }
}

// ======================================================================
// DefaultRate - PUBLIC

/// Reed-Solomon encoder/decoder generator using high or low rate as appropriate.
pub struct DefaultRate<E: Engine>(PhantomData<E>);

impl<E: Engine> Rate<E> for DefaultRate<E> {
    type RateEncoder = DefaultRateEncoder<E>;
    type RateDecoder = DefaultRateDecoder<E>;

    fn supports(original_count: usize, recovery_count: usize) -> bool {
        use_high_rate(original_count, recovery_count).is_ok()
    }
}

// ======================================================================
// InnerEncoder - PRIVATE

#[derive(Default)]
enum InnerEncoder<E: Engine> {
    High(HighRateEncoder<E>),
    Low(LowRateEncoder<E>),

    // This is only used temporarily during `reset`, never anywhere else.
    #[default]
    None,
}

// ======================================================================
// DefaultRateEncoder - PUBLIC

/// Reed-Solomon encoder using high or low rate as appropriate.
///
/// This is basically same as [`ReedSolomonEncoder`]
/// except with slightly different API which allows
/// specifying [`Engine`] and [`EncoderWork`].
///
/// [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
pub struct DefaultRateEncoder<E: Engine>(InnerEncoder<E>);

impl<E: Engine> RateEncoder<E> for DefaultRateEncoder<E> {
    type Rate = DefaultRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error> {
        match &mut self.0 {
            InnerEncoder::High(high) => high.add_original_shard(original_shard),
            InnerEncoder::Low(low) => low.add_original_shard(original_shard),
            InnerEncoder::None => unreachable!(),
        }
    }

    fn encode(&mut self) -> Result<EncoderResult<'_>, Error> {
        match &mut self.0 {
            InnerEncoder::High(high) => high.encode(),
            InnerEncoder::Low(low) => low.encode(),
            InnerEncoder::None => unreachable!(),
        }
    }

    fn into_parts(self) -> (E, EncoderWork) {
        match self.0 {
            InnerEncoder::High(high) => high.into_parts(),
            InnerEncoder::Low(low) => low.into_parts(),
            InnerEncoder::None => unreachable!(),
        }
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self, Error> {
        let inner = if use_high_rate(original_count, recovery_count)? {
            InnerEncoder::High(HighRateEncoder::new(
                original_count,
                recovery_count,
                shard_bytes,
                engine,
                work,
            )?)
        } else {
            InnerEncoder::Low(LowRateEncoder::new(
                original_count,
                recovery_count,
                shard_bytes,
                engine,
                work,
            )?)
        };

        Ok(Self(inner))
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        // Validate everything before `mem::take` below, so that an invalid
        // configuration returns an error without leaving `self.0` as `None`.
        DefaultRate::<E>::validate(original_count, recovery_count, shard_bytes)?;
        let new_rate_is_high = use_high_rate(original_count, recovery_count)?;

        self.0 = match core::mem::take(&mut self.0) {
            InnerEncoder::High(mut high) => {
                if new_rate_is_high {
                    high.reset(original_count, recovery_count, shard_bytes)?;
                    InnerEncoder::High(high)
                } else {
                    let (engine, work) = high.into_parts();
                    InnerEncoder::Low(LowRateEncoder::new(
                        original_count,
                        recovery_count,
                        shard_bytes,
                        engine,
                        Some(work),
                    )?)
                }
            }

            InnerEncoder::Low(mut low) => {
                if new_rate_is_high {
                    let (engine, work) = low.into_parts();
                    InnerEncoder::High(HighRateEncoder::new(
                        original_count,
                        recovery_count,
                        shard_bytes,
                        engine,
                        Some(work),
                    )?)
                } else {
                    low.reset(original_count, recovery_count, shard_bytes)?;
                    InnerEncoder::Low(low)
                }
            }

            InnerEncoder::None => unreachable!(),
        };

        Ok(())
    }
}

// ======================================================================
// InnerDecoder - PRIVATE

#[derive(Default)]
enum InnerDecoder<E: Engine> {
    High(HighRateDecoder<E>),
    Low(LowRateDecoder<E>),

    // This is only used temporarily during `reset`, never anywhere else.
    #[default]
    None,
}

// ======================================================================
// DefaultRateDecoder - PUBLIC

/// Reed-Solomon decoder using high or low rate as appropriate.
///
/// This is basically same as [`ReedSolomonDecoder`]
/// except with slightly different API which allows
/// specifying [`Engine`] and [`DecoderWork`].
///
/// [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
pub struct DefaultRateDecoder<E: Engine>(InnerDecoder<E>);

impl<E: Engine> RateDecoder<E> for DefaultRateDecoder<E> {
    type Rate = DefaultRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error> {
        match &mut self.0 {
            InnerDecoder::High(high) => high.add_original_shard(index, original_shard),
            InnerDecoder::Low(low) => low.add_original_shard(index, original_shard),
            InnerDecoder::None => unreachable!(),
        }
    }

    fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error> {
        match &mut self.0 {
            InnerDecoder::High(high) => high.add_recovery_shard(index, recovery_shard),
            InnerDecoder::Low(low) => low.add_recovery_shard(index, recovery_shard),
            InnerDecoder::None => unreachable!(),
        }
    }

    fn decode(&mut self) -> Result<DecoderResult<'_>, Error> {
        match &mut self.0 {
            InnerDecoder::High(high) => high.decode(),
            InnerDecoder::Low(low) => low.decode(),
            InnerDecoder::None => unreachable!(),
        }
    }

    fn into_parts(self) -> (E, DecoderWork) {
        match self.0 {
            InnerDecoder::High(high) => high.into_parts(),
            InnerDecoder::Low(low) => low.into_parts(),
            InnerDecoder::None => unreachable!(),
        }
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self, Error> {
        let inner = if use_high_rate(original_count, recovery_count)? {
            InnerDecoder::High(HighRateDecoder::new(
                original_count,
                recovery_count,
                shard_bytes,
                engine,
                work,
            )?)
        } else {
            InnerDecoder::Low(LowRateDecoder::new(
                original_count,
                recovery_count,
                shard_bytes,
                engine,
                work,
            )?)
        };

        Ok(Self(inner))
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        // Validate everything before `mem::take` below, so that an invalid
        // configuration returns an error without leaving `self.0` as `None`.
        DefaultRate::<E>::validate(original_count, recovery_count, shard_bytes)?;
        let new_rate_is_high = use_high_rate(original_count, recovery_count)?;

        self.0 = match core::mem::take(&mut self.0) {
            InnerDecoder::High(mut high) => {
                if new_rate_is_high {
                    high.reset(original_count, recovery_count, shard_bytes)?;
                    InnerDecoder::High(high)
                } else {
                    let (engine, work) = high.into_parts();
                    InnerDecoder::Low(LowRateDecoder::new(
                        original_count,
                        recovery_count,
                        shard_bytes,
                        engine,
                        Some(work),
                    )?)
                }
            }

            InnerDecoder::Low(mut low) => {
                if new_rate_is_high {
                    let (engine, work) = low.into_parts();
                    InnerDecoder::High(HighRateDecoder::new(
                        original_count,
                        recovery_count,
                        shard_bytes,
                        engine,
                        Some(work),
                    )?)
                } else {
                    low.reset(original_count, recovery_count, shard_bytes)?;
                    InnerDecoder::Low(low)
                }
            }

            InnerDecoder::None => unreachable!(),
        };

        Ok(())
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    use crate::{engine::NoSimd, test_util};

    // ============================================================
    // ROUNDTRIPS - SINGLE ROUND

    #[test]
    fn roundtrips_tiny() {
        for (original_count, recovery_count, seed, recovery_hash) in test_util::DEFAULT_TINY {
            roundtrip_single!(
                DefaultRate,
                *original_count,
                *recovery_count,
                1024,
                recovery_hash,
                &[*recovery_count..*original_count],
                &[0..core::cmp::min(*original_count, *recovery_count)],
                *seed,
            );
        }
    }

    // ============================================================
    // ROUNDTRIPS - TWO ROUNDS

    #[test]
    fn two_rounds_implicit_reset() {
        roundtrip_two_rounds!(
            DefaultRate,
            false,
            (2, 3, 1024, test_util::LOW_2_3, &[], &[0, 2], 123),
            (2, 3, 1024, test_util::LOW_2_3_223, &[0], &[1], 223),
        );
    }

    #[test]
    fn two_rounds_reset_high_to_high() {
        roundtrip_two_rounds!(
            DefaultRate,
            true,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (5, 3, 1024, test_util::HIGH_5_3, &[1, 3], &[0, 1, 2], 153),
        );
    }

    #[test]
    fn two_rounds_reset_high_to_low() {
        roundtrip_two_rounds!(
            DefaultRate,
            true,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (2, 3, 1024, test_util::LOW_2_3, &[], &[0, 2], 123),
        );
    }

    #[test]
    fn two_rounds_reset_low_to_high() {
        roundtrip_two_rounds!(
            DefaultRate,
            true,
            (2, 3, 1024, test_util::LOW_2_3, &[], &[0, 1], 123),
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
        );
    }

    #[test]
    fn two_rounds_reset_low_to_low() {
        roundtrip_two_rounds!(
            DefaultRate,
            true,
            (2, 3, 1024, test_util::LOW_2_3, &[], &[0, 2], 123),
            (3, 5, 1024, test_util::LOW_3_5, &[], &[0, 2, 4], 135),
        );
    }

    // ============================================================
    // RESET WITH INVALID SHARD SIZE
    //
    // Regression: a `reset` that switched rate (high <-> low) with an invalid
    // `shard_bytes` returned `Err` but left the encoder/decoder in the internal
    // `None` state, so the next call panicked with `unreachable!()`.

    // (original_count, recovery_count) for initial high rate, then low rate
    // and vice versa.
    const RATE_SWITCHES: [((usize, usize), (usize, usize)); 2] =
        [((3, 2), (2, 3)), ((2, 3), (3, 2))];

    fn roundtrip_with(
        encoder: &mut DefaultRateEncoder<NoSimd>,
        decoder: &mut DefaultRateDecoder<NoSimd>,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) {
        let original: Vec<Vec<u8>> = (0..original_count)
            .map(|i| (0..shard_bytes).map(|j| (i * 31 + j) as u8).collect())
            .collect();

        for shard in &original {
            encoder.add_original_shard(shard).unwrap();
        }
        let result = encoder.encode().unwrap();
        let recovery: Vec<&[u8]> = result.recovery_iter().collect();
        assert_eq!(recovery.len(), recovery_count);

        // Drop original shard 0, replace it with recovery shard 0.
        for (i, shard) in original.iter().enumerate().skip(1) {
            decoder.add_original_shard(i, shard).unwrap();
        }
        decoder.add_recovery_shard(0, recovery[0]).unwrap();
        let result = decoder.decode().unwrap();
        assert_eq!(result.restored_original(0).unwrap(), &original[0][..]);
    }

    #[test]
    fn reset_invalid_shard_size_keeps_state() {
        for ((oc1, rc1), (oc2, rc2)) in RATE_SWITCHES {
            for bad_shard_bytes in [0, 3] {
                let mut encoder =
                    DefaultRateEncoder::new(oc1, rc1, 64, NoSimd::new(), None).unwrap();
                let mut decoder =
                    DefaultRateDecoder::new(oc1, rc1, 64, NoSimd::new(), None).unwrap();

                let expected = Err(Error::InvalidShardSize {
                    shard_bytes: bad_shard_bytes,
                });
                assert_eq!(encoder.reset(oc2, rc2, bad_shard_bytes), expected);
                assert_eq!(decoder.reset(oc2, rc2, bad_shard_bytes), expected);

                // Failed reset leaves previous configuration usable.
                roundtrip_with(&mut encoder, &mut decoder, oc1, rc1, 64);

                // Subsequent valid reset (with rate switch) works.
                encoder.reset(oc2, rc2, 128).unwrap();
                decoder.reset(oc2, rc2, 128).unwrap();
                roundtrip_with(&mut encoder, &mut decoder, oc2, rc2, 128);
            }
        }
    }

    // ============================================================
    // use_high_rate

    #[test]
    fn use_high_rate() {
        fn err(original_count: usize, recovery_count: usize) -> Result<bool, Error> {
            Err(Error::UnsupportedShardCount {
                original_count,
                recovery_count,
            })
        }

        for (original_count, recovery_count, expected) in [
            (0, 1, err(0, 1)),
            (1, 0, err(1, 0)),
            // CORRECT/WRONG RATE
            (3, 3, Ok(true)),
            (3, 4, Ok(true)),
            (3, 5, Ok(false)),
            (4, 3, Ok(false)),
            (5, 3, Ok(true)),
            // LOW RATE LIMIT
            (4096, 61440, Ok(false)),
            (4096, 61441, err(4096, 61441)),
            (4097, 61440, err(4097, 61440)),
            // HIGH RATE LIMIT
            (61440, 4096, Ok(true)),
            (61440, 4097, err(61440, 4097)),
            (61441, 4096, err(61441, 4096)),
            // OVERFLOW CHECK
            (usize::MAX, usize::MAX, err(usize::MAX, usize::MAX)),
        ] {
            assert_eq!(
                super::use_high_rate(original_count, recovery_count),
                expected
            );
        }
    }
}
