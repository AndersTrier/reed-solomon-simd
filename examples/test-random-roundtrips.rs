use fixedbitset::FixedBitSet;
use rand::Rng;

use reed_solomon_simd::{
    engine::{Engine, Naive, NoSimd, GF_ORDER},
    rate::{
        DecoderWork, DefaultRate, EncoderWork, HighRate, LowRate, Rate, RateDecoder, RateEncoder,
    },
    Error,
};

// ======================================================================
// CONST

// Large shard sizes shouldn't need to be tested that much
// as algorithms handle data in 64-byte blocks.
const MIN_SHARD_BYTES_LOG: f64 = 6.0; // 2^6 = 64
const MAX_SHARD_BYTES_LOG: f64 = 6.0;
// const MAX_SHARD_BYTES_LOG: f64 = 8.0; // 2^8 = 256

const MIN_ORIGINAL_COUNT_LOG: f64 = 0.0; // 2^0 = 1
const MAX_ORIGINAL_COUNT_LOG: f64 = 16.0; // 2^16 = 65536

const MIN_RECOVERY_COUNT_LOG: f64 = 0.0; // 2^0 = 1
const MAX_RECOVERY_COUNT_LOG: f64 = 16.0; // 2^16 = 65536

// ======================================================================
// MACROS

macro_rules! roundtrip {
    (
        $Rate: ident,
        $original: expr,
        $original_count: expr,
        $recovery_count: expr,
        $shard_bytes: expr,
        $loss_indexes: expr,
        $encoder_work: expr,
        $decoder_work: expr $(,)?
    ) => {
        let recovery_naive = roundtrip::<_, $Rate<_>>(
            $original,
            $original_count,
            $recovery_count,
            $shard_bytes,
            $loss_indexes,
            $encoder_work,
            $decoder_work,
            Naive::new,
        )
        .unwrap();

        let recovery_nosimd = roundtrip::<_, $Rate<_>>(
            $original,
            $original_count,
            $recovery_count,
            $shard_bytes,
            $loss_indexes,
            $encoder_work,
            $decoder_work,
            NoSimd::new,
        )
        .unwrap();

        assert_eq!(recovery_naive, recovery_nosimd);
    };
}

// ======================================================================
// MAIN

fn main() {
    let mut encoder_work = Some(EncoderWork::new());
    let mut decoder_work = Some(DecoderWork::new());

    let mut rng = rand::thread_rng();

    let max_shard_bytes = MAX_SHARD_BYTES_LOG.exp2() as usize;
    let max_original_count = MAX_ORIGINAL_COUNT_LOG.exp2() as usize;
    let mut original = vec![vec![0u8; max_shard_bytes]; max_original_count];
    for original in &mut original {
        rng.fill::<[u8]>(original);
    }

    let mut test_number = 1;

    loop {
        // Actual data shouldn't matter at all,
        // but just in case keep changing data occasionally.
        if test_number % 100 == 0 {
            for original in &mut original {
                rng.fill::<[u8]>(original);
            }
        }

        let shard_bytes_log: f64 = rng.gen_range(MIN_SHARD_BYTES_LOG..=MAX_SHARD_BYTES_LOG);
        let shard_bytes: usize = ((shard_bytes_log.exp2() / 64.0) as usize) * 64;

        let mut original_count;
        let mut recovery_count;
        loop {
            let original_count_log: f64 =
                rng.gen_range(MIN_ORIGINAL_COUNT_LOG..=MAX_ORIGINAL_COUNT_LOG);
            let recovery_count_log: f64 =
                rng.gen_range(MIN_RECOVERY_COUNT_LOG..=MAX_RECOVERY_COUNT_LOG);

            original_count = original_count_log.exp2() as usize;
            recovery_count = recovery_count_log.exp2() as usize;

            if std::cmp::min(original_count, recovery_count).next_power_of_two()
                + std::cmp::max(original_count, recovery_count)
                <= GF_ORDER
            {
                break;
            }
        }

        // 50% chance of max loss
        let loss_count = if rng.gen::<bool>() {
            recovery_count
        } else {
            rng.gen_range(1..=recovery_count)
        };

        let loss_indexes: FixedBitSet =
            rand::seq::index::sample(&mut rng, original_count + recovery_count, loss_count)
                .iter()
                .collect();

        eprintln!();
        eprintln!("{}", test_number);
        eprintln!("original_count: {}", original_count);
        eprintln!("recovery_count: {}", recovery_count);
        eprintln!("loss_count    : {}", loss_count);
        eprintln!("shard_bytes   : {}", shard_bytes);

        roundtrip!(
            DefaultRate,
            &original,
            original_count,
            recovery_count,
            shard_bytes,
            &loss_indexes,
            &mut encoder_work,
            &mut decoder_work,
        );

        if HighRate::<NoSimd>::supports(original_count, recovery_count) {
            println!("- High");
            roundtrip!(
                HighRate,
                &original,
                original_count,
                recovery_count,
                shard_bytes,
                &loss_indexes,
                &mut encoder_work,
                &mut decoder_work,
            );
        }

        if LowRate::<NoSimd>::supports(original_count, recovery_count) {
            println!("- Low");
            roundtrip!(
                LowRate,
                &original,
                original_count,
                recovery_count,
                shard_bytes,
                &loss_indexes,
                &mut encoder_work,
                &mut decoder_work,
            );
        }

        test_number += 1;
    }
}

// ======================================================================
// FUNCTIONS

fn roundtrip<E, R>(
    original: &[Vec<u8>],
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    loss_indexes: &FixedBitSet,
    encoder_work: &mut Option<EncoderWork>,
    decoder_work: &mut Option<DecoderWork>,
    new_engine: fn() -> E,
) -> Result<Vec<Vec<u8>>, Error>
where
    E: Engine,
    R: Rate<E>,
{
    // ENCODE

    let mut encoder = R::encoder(
        original_count,
        recovery_count,
        shard_bytes,
        new_engine(),
        encoder_work.take(),
    )?;

    for original in &original[..original_count] {
        encoder.add_original_shard(&original[..shard_bytes])?;
    }

    let result = encoder.encode()?;
    let recovery: Vec<_> = result.recovery_iter().map(|s| s.to_vec()).collect();
    drop(result);

    // DECODE

    let mut decoder = R::decoder(
        original_count,
        recovery_count,
        shard_bytes,
        new_engine(),
        decoder_work.take(),
    )?;

    for n in 0..original_count {
        if !loss_indexes[n] {
            decoder.add_original_shard(n, &original[n][..shard_bytes])?;
        }
    }

    for n in 0..recovery_count {
        if !loss_indexes[original_count + n] {
            decoder.add_recovery_shard(n, &recovery[n])?;
        }
    }

    // CHECK

    let result = decoder.decode()?;
    for n in 0..original_count {
        if loss_indexes[n] {
            assert_eq!(
                result.restored_original(n).unwrap(),
                &original[n][..shard_bytes]
            );
        }
    }
    drop(result);

    // DONE

    *encoder_work = Some(encoder.into_parts().1);
    *decoder_work = Some(decoder.into_parts().1);

    Ok(recovery)
}
