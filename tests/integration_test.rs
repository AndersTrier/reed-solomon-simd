use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use reed_solomon_simd::engine::{DefaultEngine, Engine, Naive, NoSimd};
use reed_solomon_simd::rate::{
    DefaultRateDecoder, DefaultRateEncoder, HighRate, LowRate, Rate, RateDecoder, RateEncoder,
};
use reed_solomon_simd::Error;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use reed_solomon_simd::engine::{Avx2, Ssse3};

#[cfg(target_arch = "aarch64")]
use reed_solomon_simd::engine::Neon;

// ======================================================================
// TESTS - HELPERS

fn random_shard_count(rng: &mut impl Rng) -> (usize, usize) {
    loop {
        let original_count = rng.gen_range(0..65536);
        let recovery_count = rng.gen_range(0..65536);
        if DefaultRateEncoder::<NoSimd>::supports(original_count, recovery_count) {
            return (original_count, recovery_count);
        }
    }
}

fn random_buf(bytes: usize) -> Vec<u8> {
    let mut rng = ChaCha8Rng::from_seed([0; 32]);
    let mut buf = vec![0u8; bytes];
    rng.fill::<[u8]>(buf.as_mut());
    buf
}

fn test_rate<E: Engine + Default, R: Rate<E>>(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
) -> Result<(), Error> {
    R::validate(original_count, recovery_count, shard_bytes)?;

    let mut encoder = R::encoder(
        original_count,
        recovery_count,
        shard_bytes,
        E::default(),
        None,
    )?;

    let original_buf = random_buf(original_count * shard_bytes);
    let original: Vec<&[u8]> = original_buf.chunks_exact(shard_bytes).collect();

    for shard in &original {
        encoder.add_original_shard(shard)?;
    }

    let result = encoder.encode()?;
    let recovery: Vec<&[u8]> = result.recovery_iter().collect();

    let mut decoder = R::decoder(
        original_count,
        recovery_count,
        shard_bytes,
        E::default(),
        None,
    )?;

    // Add minimum amount of shards
    let shards_to_add = original_count;
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

    let result = decoder.decode()?;

    for (idx, shard) in result.restored_original_iter() {
        assert_eq!(shard, original[idx]);
    }
    Ok(())
}

fn compare_to_nosimd<E: Engine + Default>(
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
) -> Result<(), Error> {
    let original_buf = random_buf(original_count * shard_bytes);
    let original: Vec<&[u8]> = original_buf.chunks_exact(shard_bytes).collect();

    // ENCODE

    let mut encoder_prospect = DefaultRateEncoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        E::default(),
        None,
    )?;

    let mut encoder_nosimd = DefaultRateEncoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        NoSimd::new(),
        None,
    )?;

    for shard in &original {
        encoder_prospect.add_original_shard(shard)?;
        encoder_nosimd.add_original_shard(shard)?;
    }

    let result_prospect = encoder_prospect.encode()?;
    let result_nosimd = encoder_nosimd.encode()?;

    let recovery_prospect: Vec<&[u8]> = result_prospect.recovery_iter().collect();
    let recovery_nosimd: Vec<&[u8]> = result_nosimd.recovery_iter().collect();

    assert_eq!(recovery_nosimd, recovery_prospect);

    // DECODE

    let mut decoder_prospect = DefaultRateDecoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        E::default(),
        None,
    )?;

    let mut decoder_nosimd = DefaultRateDecoder::new(
        original_count,
        recovery_count,
        shard_bytes,
        NoSimd::default(),
        None,
    )?;

    // Add minimum amount of shards
    let shards_to_add = original_count;
    let mut recovery_added = 0;

    for (idx, shard) in recovery_prospect.iter().enumerate().take(shards_to_add) {
        decoder_prospect.add_recovery_shard(idx, shard).unwrap();
        decoder_nosimd.add_recovery_shard(idx, shard).unwrap();
        recovery_added += 1;
    }

    for (idx, shard) in original
        .iter()
        .enumerate()
        .take(shards_to_add - recovery_added)
    {
        decoder_prospect.add_original_shard(idx, shard).unwrap();
        decoder_nosimd.add_original_shard(idx, shard).unwrap();
    }

    let result_prospect = decoder_prospect.decode()?;
    let result_nosimd = decoder_nosimd.decode()?;

    assert!(result_nosimd
        .restored_original_iter()
        .eq(result_prospect.restored_original_iter()));

    Ok(())
}

// ======================================================================
// TESTS

#[test]
fn high_rate() -> Result<(), Error> {
    test_rate::<DefaultEngine, HighRate<DefaultEngine>>(35000, 1000, 64)
}

#[test]
fn low_rate() -> Result<(), Error> {
    test_rate::<DefaultEngine, LowRate<DefaultEngine>>(1000, 35000, 64)
}

#[test]
fn engine_naive() -> Result<(), Error> {
    compare_to_nosimd::<Naive>(128, 32, 64)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_ssse3() -> Result<(), Error> {
    if is_x86_feature_detected!("ssse3") {
        compare_to_nosimd::<Ssse3>(128, 32, 64)
    } else {
        eprintln!("Skipping test: SSSE3 not supported on this processor.");
        Ok(())
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
fn x86_avx2() -> Result<(), Error> {
    if is_x86_feature_detected!("avx2") {
        compare_to_nosimd::<Avx2>(128, 32, 64)
    } else {
        eprintln!("Skipping test: AVX2 not supported on this processor.");
        Ok(())
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_neon() -> Result<(), Error> {
    if std::arch::is_aarch64_feature_detected!("neon") {
        compare_to_nosimd::<Neon>(128, 32, 64)
    } else {
        eprintln!("Skipping test: NEON not supported on this processor.");
        Ok(())
    }
}

// ======================================================================
// TESTS - IGNORED
//
// Takes a bit longer. Runs much faster in release mode:
// cargo test --release -- --ignored

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
#[ignore]
fn x86_ssse3_random_roundtrips() -> Result<(), Error> {
    if !is_x86_feature_detected!("ssse3") {
        eprintln!("Skipping test: SSSE3 not supported on this processor.");
        return Ok(());
    }

    let mut rng = ChaCha8Rng::from_seed([0; 32]);

    for _ in 0..5 {
        let (original_count, recovery_count) = random_shard_count(&mut rng);
        let chunk_count: usize = rng.gen_range(1..=3);
        compare_to_nosimd::<Ssse3>(original_count, recovery_count, chunk_count * 64)?;
    }

    Ok(())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[test]
#[ignore]
fn x86_avx2_random_roundtrips() -> Result<(), Error> {
    if !is_x86_feature_detected!("avx2") {
        eprintln!("Skipping test: AVX2 not supported on this processor.");
        return Ok(());
    }

    let mut rng = ChaCha8Rng::from_seed([0; 32]);

    for _ in 0..5 {
        let (original_count, recovery_count) = random_shard_count(&mut rng);
        let chunk_count: usize = rng.gen_range(1..=3);
        compare_to_nosimd::<Avx2>(original_count, recovery_count, chunk_count * 64)?;
    }

    Ok(())
}

#[cfg(target_arch = "aarch64")]
#[test]
#[ignore]
fn aarch64_neon_random_roundtrips() -> Result<(), Error> {
    if !std::arch::is_aarch64_feature_detected!("neon") {
        eprintln!("Skipping test: NEON not supported on this processor.");
        return Ok(());
    }

    let mut rng = ChaCha8Rng::from_seed([0; 32]);

    for _ in 0..5 {
        let (original_count, recovery_count) = random_shard_count(&mut rng);
        let chunk_count: usize = rng.gen_range(1..=3);
        compare_to_nosimd::<Neon>(original_count, recovery_count, chunk_count * 64)?;
    }

    Ok(())
}
