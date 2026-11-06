use core::marker::PhantomData;

use crate::{
    engine::{self, tables, Engine, GfElement, GF_MODULUS, GF_ORDER},
    rate::{DecoderWork, EncoderWork, Rate, RateDecoder, RateEncoder},
    DecoderResult, EncoderResult, Error,
};

// ======================================================================
// HighRate - PUBLIC

/// Reed-Solomon encoder/decoder generator using only high rate.
pub struct HighRate<E: Engine>(PhantomData<E>);

impl<E: Engine> Rate<E> for HighRate<E> {
    type RateEncoder = HighRateEncoder<E>;
    type RateDecoder = HighRateDecoder<E>;

    fn supports(original_count: usize, recovery_count: usize) -> bool {
        original_count > 0
            && recovery_count > 0
            && original_count < GF_ORDER
            && recovery_count < GF_ORDER
            && recovery_count.next_power_of_two() + original_count <= GF_ORDER
    }
}

// ======================================================================
// HighRateEncoder - PUBLIC

/// Reed-Solomon encoder using only high rate.
pub struct HighRateEncoder<E: Engine> {
    engine: E,
    work: EncoderWork,
}

impl<E: Engine> RateEncoder<E> for HighRateEncoder<E> {
    type Rate = HighRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error> {
        self.work.add_original_shard(original_shard)
    }

    fn encode(&mut self) -> Result<EncoderResult<'_>, Error> {
        let (mut work, original_count, recovery_count) = self.work.encode_begin()?;
        let chunk_size = recovery_count.next_power_of_two();
        let engine = &self.engine;

        // FUSED TOP STAGE
        //
        // When `chunk_size` is a power of 4 and all originals fit in its first quarter,
        // the IFFT's last radix-4 pass sees three zero quarters, so it and the FFT's first
        // radix-4 pass reduce to scaling the quarter by four constants.

        let quarter = chunk_size / 4;
        if chunk_size >= 4 && chunk_size.trailing_zeros() % 2 == 0 && original_count <= quarter {
            work.zero(original_count..quarter);
            engine.ifft(&mut work, 0, quarter, original_count, chunk_size);

            let scales = top_scales(chunk_size);
            let log = &tables::get_exp_log().log;
            // Group 0 is the source of the others, so it is scaled last.
            for g in (0..recovery_count.div_ceil(quarter)).rev() {
                let (src, dst) = work.flat2_mut(0, g.max(1) * quarter, quarter);
                let group = if g == 0 {
                    src
                } else {
                    dst.copy_from_slice(src);
                    dst
                };
                match scales[g] {
                    0 => group.fill([0; 64]),
                    1 => {}
                    c => engine.mul(group, log[c as usize]),
                }
            }

            for g in 0..recovery_count.div_ceil(quarter) {
                let pos = g * quarter;
                engine.fft(
                    &mut work,
                    pos,
                    quarter,
                    quarter.min(recovery_count - pos),
                    pos,
                );
            }

            self.work.undo_last_chunk_encoding();
            return Ok(EncoderResult::new(&mut self.work));
        }

        // FIRST CHUNK

        let first_count = core::cmp::min(original_count, chunk_size);

        work.zero(first_count..chunk_size);
        engine::ifft_skew_end(engine, &mut work, 0, chunk_size, first_count);

        if original_count > chunk_size {
            // FULL CHUNKS

            let mut chunk_start = chunk_size;
            while chunk_start + chunk_size <= original_count {
                engine::ifft_skew_end(engine, &mut work, chunk_start, chunk_size, chunk_size);
                engine::xor_within(&mut work, 0, chunk_start, chunk_size);
                chunk_start += chunk_size;
            }

            // FINAL PARTIAL CHUNK

            let last_count = original_count % chunk_size;
            if last_count > 0 {
                work.zero(chunk_start + last_count..);
                engine::ifft_skew_end(engine, &mut work, chunk_start, chunk_size, last_count);
                engine::xor_within(&mut work, 0, chunk_start, chunk_size);
            }
        }

        // FFT

        engine.fft(&mut work, 0, chunk_size, recovery_count, 0);

        // UNDO LAST CHUNK ENCODING

        self.work.undo_last_chunk_encoding();

        // DONE

        Ok(EncoderResult::new(&mut self.work))
    }

    fn into_parts(self) -> (E, EncoderWork) {
        (self.engine, self.work)
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self, Error> {
        let mut work = work.unwrap_or_default();
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut work)?;
        Ok(Self { engine, work })
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut self.work)
    }
}

// ======================================================================
// HighRateEncoder - PRIVATE

/// Constants `c[g]` such that, for input `(y, 0, 0, 0)` at rows `i + g * size / 4`,
/// the IFFT's last radix-4 pass followed by the FFT's first one yields `c[g] * y`.
/// Mirrors the butterflies of `Naive::ifft` / `Naive::fft` at `dist = size / 4, size / 2`.
fn top_scales(size: usize) -> [GfElement; 4] {
    let tables::ExpLog { exp, log } = tables::get_exp_log();
    let skew = tables::get_skew();
    let q = size / 4;
    let mut v: [GfElement; 4] = [1, 0, 0, 0];

    let ifftb = |v: &mut [GfElement; 4], a: usize, b: usize, log_m: GfElement| {
        v[b] ^= v[a];
        if log_m != GF_MODULUS {
            v[a] ^= tables::mul(v[b], log_m, exp, log);
        }
    };
    let fftb = |v: &mut [GfElement; 4], a: usize, b: usize, log_m: GfElement| {
        if log_m != GF_MODULUS {
            v[a] ^= tables::mul(v[b], log_m, exp, log);
        }
        v[b] ^= v[a];
    };

    // IFFT, skew_delta = size
    ifftb(&mut v, 0, 1, skew[q + size - 1]);
    ifftb(&mut v, 2, 3, skew[3 * q + size - 1]);
    ifftb(&mut v, 0, 2, skew[2 * q + size - 1]);
    ifftb(&mut v, 1, 3, skew[2 * q + size - 1]);

    // FFT, skew_delta = 0
    fftb(&mut v, 0, 2, skew[2 * q - 1]);
    fftb(&mut v, 1, 3, skew[2 * q - 1]);
    fftb(&mut v, 0, 1, skew[q - 1]);
    fftb(&mut v, 2, 3, skew[3 * q - 1]);

    v
}

impl<E: Engine> HighRateEncoder<E> {
    fn reset_work(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work: &mut EncoderWork,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;
        work.reset(
            original_count,
            recovery_count,
            shard_bytes,
            Self::work_count(original_count, recovery_count),
        );
        Ok(())
    }

    fn work_count(original_count: usize, recovery_count: usize) -> usize {
        debug_assert!(Self::supports(original_count, recovery_count));

        let chunk_size = recovery_count.next_power_of_two();

        original_count.next_multiple_of(chunk_size)
    }
}

// ======================================================================
// HighRateDecoder - PUBLIC

/// Reed-Solomon decoder using only high rate.
pub struct HighRateDecoder<E: Engine> {
    engine: E,
    work: DecoderWork,
}

impl<E: Engine> RateDecoder<E> for HighRateDecoder<E> {
    type Rate = HighRate<E>;

    fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error> {
        self.work.add_original_shard(index, original_shard)
    }

    fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error> {
        self.work.add_recovery_shard(index, recovery_shard)
    }

    fn decode(&mut self) -> Result<DecoderResult<'_>, Error> {
        let Some((mut work, original_count, recovery_count, received)) =
            self.work.decode_begin()?
        else {
            // Nothing to do, original data is complete.
            return Ok(DecoderResult::new(&mut self.work));
        };

        let chunk_size = recovery_count.next_power_of_two();
        let original_end = chunk_size + original_count;
        let work_count = work.len();

        // ERASURE LOCATIONS

        let mut erasures = [0; GF_ORDER];

        for i in 0..recovery_count {
            if !received[i] {
                erasures[i] = 1;
            }
        }

        erasures[recovery_count..chunk_size].fill(1);

        for i in chunk_size..original_end {
            if !received[i] {
                erasures[i] = 1;
            }
        }

        // EVALUATE POLYNOMIAL

        E::eval_poly(&mut erasures, original_end);

        // MULTIPLY SHARDS

        // work[               .. recovery_count] = recovery * erasures
        // work[recovery_count .. chunk_size    ] = 0
        // work[chunk_size     .. original_end  ] = original * erasures
        // work[original_end   ..               ] = 0

        for i in 0..recovery_count {
            if received[i] {
                self.engine.mul(&mut work[i], erasures[i]);
            } else {
                work[i].fill([0; 64]);
            }
        }

        work.zero(recovery_count..chunk_size);

        for i in chunk_size..original_end {
            if received[i] {
                self.engine.mul(&mut work[i], erasures[i]);
            } else {
                work[i].fill([0; 64]);
            }
        }

        work.zero(original_end..);

        // IFFT / FORMAL DERIVATIVE / FFT

        self.engine.ifft(&mut work, 0, work_count, original_end, 0);
        engine::formal_derivative(&mut work);
        self.engine.fft(&mut work, 0, work_count, original_end, 0);

        // REVEAL ERASURES

        for i in chunk_size..original_end {
            if !received[i] {
                self.engine.mul(&mut work[i], GF_MODULUS - erasures[i]);
            }
        }

        // UNDO LAST CHUNK ENCODING

        self.work.undo_last_chunk_encoding();

        // DONE

        Ok(DecoderResult::new(&mut self.work))
    }

    fn into_parts(self) -> (E, DecoderWork) {
        (self.engine, self.work)
    }

    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self, Error> {
        let mut work = work.unwrap_or_default();
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut work)?;
        Ok(Self { engine, work })
    }

    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::reset_work(original_count, recovery_count, shard_bytes, &mut self.work)
    }
}

// ======================================================================
// HighRateDecoder - PRIVATE

impl<E: Engine> HighRateDecoder<E> {
    fn reset_work(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work: &mut DecoderWork,
    ) -> Result<(), Error> {
        Self::validate(original_count, recovery_count, shard_bytes)?;

        // work[..recovery_count     ]  =  recovery
        // work[recovery_count_pow2..]  =  original
        work.reset(
            original_count,
            recovery_count,
            shard_bytes,
            recovery_count.next_power_of_two(),
            0,
            Self::work_count(original_count, recovery_count),
        );

        Ok(())
    }

    fn work_count(original_count: usize, recovery_count: usize) -> usize {
        debug_assert!(Self::supports(original_count, recovery_count));

        (recovery_count.next_power_of_two() + original_count).next_power_of_two()
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util;

    // ============================================================
    // ROUNDTRIPS - SINGLE ROUND

    #[test]
    fn roundtrip_all_originals_missing() {
        roundtrip_single!(
            HighRate,
            3,
            3,
            1024,
            test_util::EITHER_3_3,
            &[],
            &[0..3],
            133,
        );
    }

    #[test]
    fn roundtrip_no_originals_missing() {
        roundtrip_single!(HighRate, 3, 2, 1024, test_util::HIGH_3_2, &[0..3], &[], 132);
    }

    #[test]
    fn roundtrips_tiny() {
        for (original_count, recovery_count, seed, recovery_hash) in test_util::HIGH_TINY {
            roundtrip_single!(
                HighRate,
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

    #[test]
    #[ignore]
    fn roundtrip_3000_30000() {
        roundtrip_single!(
            HighRate,
            3000,
            30000,
            64,
            test_util::HIGH_3000_30000_14,
            &[],
            &[0..3000],
            14,
        );
    }

    #[test]
    #[ignore]
    fn roundtrip_32768_32768() {
        roundtrip_single!(
            HighRate,
            32768,
            32768,
            64,
            test_util::EITHER_32768_32768_11,
            &[],
            &[0..32768],
            11,
        );
    }

    #[test]
    #[ignore]
    fn roundtrip_60000_3000() {
        roundtrip_single!(
            HighRate,
            60000,
            3000,
            64,
            test_util::HIGH_60000_3000_12,
            &[3000..60000],
            &[0..3000],
            12,
        );
    }

    #[test]
    fn roundtrip_34000_2000_shard_size_8() {
        roundtrip_single!(
            HighRate,
            34000,
            2000,
            8,
            test_util::HIGH_34000_2000_123_8,
            &[0..32000],
            &[0..2000],
            123
        );
    }

    // ============================================================
    // FUSED TOP STAGE

    // Hashes recorded with the unfused encoder (v3.1.0), seed 7.
    // (8, 24) does not qualify (chunk size 32 is not a power of 4) and is a control.
    #[rustfmt::skip]
    const FUSED: &[(usize, usize, usize, &str)] = &[
        (1, 3, 64, "81e0ffedb5e49961dc08cabd8a80caf32e0a1d6b996468ccadcf9304c9339891"),
        (1, 3, 70, "9e9883a40005e2e3db90788b35b128a9b3006bf317291ddd0d73e0b5189b575d"),
        (1, 4, 64, "71c39cfd4b58d7d53db3677fee41af26041392a79640bf26115e119eff3827dd"),
        (1, 4, 70, "18188d1b8aed2da0a299c0b9f5dbbe0d987569c6cd39c1292f6287ed3fd5fd06"),
        (4, 9, 64, "1c97b0a4db4fe500bd686bd3749c3e73eba307bc1fc7b6a6168ecac089f47dbe"),
        (4, 9, 70, "036c851aedc9043104fabafa863b883fcc5efa057de5e98a2700b82fcd7355eb"),
        (4, 16, 64, "57ecb262056e78c9cf8b161d499db0e0664e7b162c96107077287070cf3e3636"),
        (4, 16, 70, "3678456eefa65da781e947e2bae1e531b539fefa939fca3a3dc3cd38a8af8913"),
        (16, 48, 64, "03e7251016c197280065ac684279b7971f1373d9ded3aa7ede3302fb5001f18b"),
        (16, 48, 70, "04f074a97533d17f37f73e2a7c4221d180f09d73d94686b91d9659baab733c6c"),
        (16, 64, 64, "9efdb347d694375eaa0278e40c8f5711d6e53f0785e7651b2a8fa45d1ae174e7"),
        (16, 64, 70, "5e9cb30cc3289d70be25cdf1365e2f8ea1946243321d56f7b202bbd640abd583"),
        (3, 12, 64, "437aaddfa9f7b51280c608237c43c4463905dcab26187867054e34208a8de130"),
        (3, 12, 70, "a506f420d7719a9b339098e42c1677e776dcfd8f6599ef13f5bc38c2fba17953"),
        (8, 24, 64, "c2a6d414be442544ec951b7fdc54fa90c8a8b08013f0864f4a7c16c83c68d9cc"),
        (8, 24, 70, "7a4b0586450d140b27711e237d55a9dd7e8f370c1aef6d6ac479eb26d5ae8ffc"),
    ];

    fn assert_fused<E: Engine>(engine: fn() -> E, k: usize, n: usize, b: usize, hash: &str) {
        let mut encoder = HighRateEncoder::new(k, n, b, engine(), None).unwrap();
        for original in test_util::generate_original(k, b, 7) {
            encoder.add_original_shard(original).unwrap();
        }
        test_util::assert_hash(encoder.encode().unwrap().recovery_iter(), hash);
    }

    fn assert_fused_all_engines(k: usize, n: usize, b: usize, hash: &str) {
        assert_fused(crate::engine::Naive::new, k, n, b, hash);
        assert_fused(crate::engine::NoSimd::new, k, n, b, hash);
        assert_fused(crate::engine::DefaultEngine::new, k, n, b, hash);
    }

    #[test]
    fn fused_top_stage() {
        for (k, n, b, hash) in FUSED {
            assert_fused_all_engines(*k, *n, *b, hash);
        }
    }

    #[test]
    #[ignore]
    fn fused_top_stage_4096_12288() {
        assert_fused_all_engines(
            4096,
            12288,
            64,
            "ed8a8d4a9dd3cad43f68ed4070ce500116d96c5dc7a44a53516f831472ae82f7",
        );
        assert_fused_all_engines(
            4096,
            12288,
            70,
            "73491d671d528d5fec587e93f34d5ace968442a6e5d93daba9ba387b0267bdbb",
        );
    }

    // ============================================================
    // ROUNDTRIPS - TWO ROUNDS

    #[test]
    fn two_rounds_implicit_reset() {
        roundtrip_two_rounds!(
            HighRate,
            false,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (3, 2, 1024, test_util::HIGH_3_2_232, &[0], &[0, 1], 232),
        );
    }

    #[test]
    fn two_rounds_explicit_reset() {
        roundtrip_two_rounds!(
            HighRate,
            true,
            (3, 2, 1024, test_util::HIGH_3_2, &[1], &[0, 1], 132),
            (5, 2, 1024, test_util::HIGH_5_2, &[0, 2, 4], &[0, 1], 152),
        );
    }

    // ============================================================
    // HighRate

    mod high_rate {
        use crate::{
            engine::NoSimd,
            rate::{HighRate, Rate},
            Error,
        };

        #[test]
        fn decoder() {
            assert_eq!(
                HighRate::<NoSimd>::decoder(4096, 61440, 64, NoSimd::new(), None).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::decoder(61440, 4096, 64, NoSimd::new(), None).is_ok());
        }

        #[test]
        fn encoder() {
            assert_eq!(
                HighRate::<NoSimd>::encoder(4096, 61440, 64, NoSimd::new(), None).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::encoder(61440, 4096, 64, NoSimd::new(), None).is_ok());
        }

        #[test]
        fn supports() {
            assert!(!HighRate::<NoSimd>::supports(0, 1));
            assert!(!HighRate::<NoSimd>::supports(1, 0));

            assert!(!HighRate::<NoSimd>::supports(4096, 61440));

            assert!(HighRate::<NoSimd>::supports(61440, 4096));
            assert!(!HighRate::<NoSimd>::supports(61440, 4097));
            assert!(!HighRate::<NoSimd>::supports(61441, 4096));

            assert!(!HighRate::<NoSimd>::supports(usize::MAX, usize::MAX));
        }

        #[test]
        fn validate() {
            assert_eq!(
                HighRate::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRate::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRate::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }
    }

    // ============================================================
    // HighRateEncoder

    mod high_rate_encoder {
        use crate::{
            engine::NoSimd,
            rate::{HighRateEncoder, RateEncoder},
            Error,
        };

        // ==================================================
        // ERRORS

        test_rate_encoder_errors! {HighRateEncoder}

        // ==================================================
        // supports

        #[test]
        fn supports() {
            assert!(!HighRateEncoder::<NoSimd>::supports(4096, 61440));
            assert!(HighRateEncoder::<NoSimd>::supports(61440, 4096));
        }

        // ==================================================
        // validate

        #[test]
        fn validate() {
            assert_eq!(
                HighRateEncoder::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRateEncoder::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRateEncoder::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }

        // ==================================================
        // work_count

        #[test]
        fn work_count() {
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(1, 1), 1);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4096, 1024), 4096);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4097, 1024), 5120);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(4097, 1025), 6144);
            assert_eq!(HighRateEncoder::<NoSimd>::work_count(32768, 32768), 32768);
        }
    }

    // ============================================================
    // HighRateDecoder

    mod high_rate_decoder {
        use crate::{
            engine::NoSimd,
            rate::{HighRateDecoder, RateDecoder},
            Error,
        };

        // ==================================================
        // ERRORS

        test_rate_decoder_errors! {HighRateDecoder}

        // ==================================================
        // supports

        #[test]
        fn supports() {
            assert!(!HighRateDecoder::<NoSimd>::supports(4096, 61440));
            assert!(HighRateDecoder::<NoSimd>::supports(61440, 4096));
        }

        // ==================================================
        // validate

        #[test]
        fn validate() {
            assert_eq!(
                HighRateDecoder::<NoSimd>::validate(1, 1, 123).err(),
                Some(Error::InvalidShardSize { shard_bytes: 123 })
            );

            assert_eq!(
                HighRateDecoder::<NoSimd>::validate(4096, 61440, 64).err(),
                Some(Error::UnsupportedShardCount {
                    original_count: 4096,
                    recovery_count: 61440,
                })
            );

            assert!(HighRateDecoder::<NoSimd>::validate(61440, 4096, 64).is_ok());
        }

        // ==================================================
        // work_count

        #[test]
        fn work_count() {
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(1, 1), 2);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(2048, 1025), 4096);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(2049, 1025), 8192);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(3072, 1024), 4096);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(3073, 1024), 8192);
            assert_eq!(HighRateDecoder::<NoSimd>::work_count(32768, 32768), 65536);
        }
    }
}
