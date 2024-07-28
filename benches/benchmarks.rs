use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use reed_solomon_simd::{
    engine::{DefaultEngine, Engine, Naive, NoSimd, ShardsRefMut},
    rate::{
        HighRateDecoder, HighRateEncoder, LowRateDecoder, LowRateEncoder, RateDecoder, RateEncoder,
    },
    ReedSolomonDecoder, ReedSolomonEncoder,
};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use reed_solomon_simd::engine::{Avx2, Ssse3};

#[cfg(target_arch = "aarch64")]
use reed_solomon_simd::engine::Neon;

// ======================================================================
// CONST

const SHARD_BYTES: usize = 1024;

// ======================================================================
// UTIL

fn generate_shards_64(shard_count: usize, chunk_count: usize, seed: u8) -> Vec<Vec<[u8; 64]>> {
    let mut rng = ChaCha8Rng::from_seed([seed; 32]);
    let mut shards = vec![vec![[0u8; 64]; chunk_count]; shard_count];
    for shard in &mut shards {
        rng.fill::<[u8]>(shard.as_flattened_mut());
    }
    shards
}

fn generate_shards(shard_count: usize, shard_bytes: usize, seed: u8) -> Vec<Vec<u8>> {
    assert_eq!(shard_bytes % 64, 0);
    generate_shards_64(shard_count, shard_bytes / 64, seed)
        .into_iter()
        .map(|s| s.into_flattened())
        .collect()
}

// ======================================================================
// BENCHMARKS - MAIN

fn benchmarks_main(c: &mut Criterion) {
    let mut group = c.benchmark_group("main");

    for (original_count, recovery_count) in [
        // 2^n. original_count == recovery_count
        (32, 32),
        (64, 64),
        (128, 128),
        (256, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (8192, 8192),
        (16384, 16384),
        (32768, 32768),
        // And some other combinations
        (128, 1024),
        (1000, 100),
        (1000, 10000),
        (1024, 128),
        (1024, 8192),
        (8192, 1024),
        (8192, 16384),
        (8192, 57344),
        (10000, 1000),
        (16384, 8192),
        (16385, 16385), // 2^n + 1
        (57344, 8192),
    ] {
        if original_count >= 1000 && recovery_count >= 1000 {
            group.sample_size(10);
        } else {
            group.sample_size(100);
        }

        let original = generate_shards(original_count, SHARD_BYTES, 0);
        let recovery =
            reed_solomon_simd::encode(original_count, recovery_count, &original).unwrap();

        group.throughput(Throughput::Bytes(
            ((original_count + recovery_count) * SHARD_BYTES) as u64,
        ));

        // ReedSolomonEncoder

        let mut encoder =
            ReedSolomonEncoder::new(original_count, recovery_count, SHARD_BYTES).unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("ReedSolomonEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // ReedSolomonDecoder

        let max_original_loss_count = std::cmp::min(original_count, recovery_count);

        for loss_percent in [1, 100] {
            // We round up to make sure at least one shard is lost for low shard counts.
            // '+ 99' as div_ceil() is not in stable yet (int_roundings #88581).
            let original_loss_count = (max_original_loss_count * loss_percent + 99) / 100;
            let original_provided_count = original_count - original_loss_count;
            let recovery_provided_count = original_loss_count;

            let mut decoder =
                ReedSolomonDecoder::new(original_count, recovery_count, SHARD_BYTES).unwrap();

            let id = format!("{}:{} ({}%)", original_count, recovery_count, loss_percent);

            group.bench_with_input(
                BenchmarkId::new("ReedSolomonDecoder", &id),
                &recovery,
                |b, recovery| {
                    b.iter(|| {
                        for index in 0..original_provided_count {
                            decoder.add_original_shard(index, &original[index]).unwrap();
                        }
                        for index in 0..recovery_provided_count {
                            decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                        }
                        decoder.decode().unwrap();
                    });
                },
            );
        }
    }

    group.finish();
}

// ======================================================================
// BENCHMARKS - RATE

fn benchmarks_rate(c: &mut Criterion) {
    // benchmarks_rate_one(c, "rate-Naive", Naive::new);
    benchmarks_rate_one(c, "rate", DefaultEngine::new);
}

fn benchmarks_rate_one<E: Engine>(c: &mut Criterion, name: &str, new_engine: fn() -> E) {
    let mut group = c.benchmark_group(name);
    group.sample_size(10);

    for (original_count, recovery_count) in [
        (1024, 1024),
        (1024, 1025),
        (1025, 1024),
        (1024, 2048),
        (2048, 1024),
        (1025, 1025),
        (1025, 2048),
        (2048, 1025),
        (2048, 2048),
    ] {
        let original = generate_shards(original_count, SHARD_BYTES, 0);
        let recovery =
            reed_solomon_simd::encode(original_count, recovery_count, &original).unwrap();

        group.throughput(Throughput::Bytes(
            ((original_count + recovery_count) * SHARD_BYTES) as u64,
        ));

        // ENCODE

        let id = format!("{}:{}", original_count, recovery_count);

        // HighRateEncoder

        let mut encoder = HighRateEncoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("HighRateEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // LowRateEncoder

        let mut encoder = LowRateEncoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        group.bench_with_input(
            BenchmarkId::new("LowRateEncoder", &id),
            &original,
            |b, original| {
                b.iter(|| {
                    for original in original {
                        encoder.add_original_shard(original).unwrap();
                    }
                    encoder.encode().unwrap();
                });
            },
        );

        // DECODE

        let original_loss_count = std::cmp::min(original_count, recovery_count);
        let original_provided_count = original_count - original_loss_count;
        let recovery_provided_count = original_loss_count;

        // HighRateDecoder

        let mut decoder = HighRateDecoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("HighRateDecoder", &id),
            &recovery,
            |b, recovery| {
                b.iter(|| {
                    for index in 0..original_provided_count {
                        decoder.add_original_shard(index, &original[index]).unwrap();
                    }
                    for index in 0..recovery_provided_count {
                        decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                    }
                    decoder.decode().unwrap();
                });
            },
        );

        // LowRateDecoder

        let mut decoder = LowRateDecoder::new(
            original_count,
            recovery_count,
            SHARD_BYTES,
            new_engine(),
            None,
        )
        .unwrap();

        let id = format!("{}:{}", original_count, recovery_count);

        group.bench_with_input(
            BenchmarkId::new("LowRateDecoder", &id),
            &recovery,
            |b, recovery| {
                b.iter(|| {
                    for index in 0..original_provided_count {
                        decoder.add_original_shard(index, &original[index]).unwrap();
                    }
                    for index in 0..recovery_provided_count {
                        decoder.add_recovery_shard(index, &recovery[index]).unwrap();
                    }
                    decoder.decode().unwrap();
                });
            },
        );
    }

    group.finish();
}

// ======================================================================
// BENCHMARKS - ENGINES

fn benchmarks_engine(c: &mut Criterion) {
    benchmarks_engine_one(c, "engine-Naive", Naive::new());
    benchmarks_engine_one(c, "engine-NoSimd", NoSimd::new());

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("ssse3") {
            benchmarks_engine_one(c, "engine-Ssse3", Ssse3::new());
        }
        if is_x86_feature_detected!("avx2") {
            benchmarks_engine_one(c, "engine-Avx2", Avx2::new());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            benchmarks_engine_one(c, "engine-Neon", Neon::new());
        }
    }
}

fn benchmarks_engine_one<E: Engine>(c: &mut Criterion, name: &str, engine: E) {
    let mut group = c.benchmark_group(name);

    // XOR MUL

    let mut x = &mut generate_shards_64(1, SHARD_BYTES, 0)[0];
    let y = &generate_shards_64(1, SHARD_BYTES, 1)[0];

    group.bench_function("xor", |b| {
        b.iter(|| E::xor(black_box(&mut x), black_box(&y)))
    });

    group.bench_function("mul", |b| {
        b.iter(|| engine.mul(black_box(&mut x), black_box(12345)))
    });

    // XOR_WITHIN

    let shards_256_data = &mut generate_shards_64(1, 256 * SHARD_BYTES, 0)[0];
    let mut shards_256 = ShardsRefMut::new(256, SHARD_BYTES, shards_256_data.as_mut());

    group.bench_function("xor_within 128*2", |b| {
        b.iter(|| {
            E::xor_within(
                black_box(&mut shards_256),
                black_box(0),
                black_box(128),
                black_box(128),
            )
        })
    });

    // FORMAL DERIVATIVE

    let shards_128_data = &mut generate_shards_64(1, 128 * SHARD_BYTES, 0)[0];
    let mut shards_128 = ShardsRefMut::new(128, SHARD_BYTES, shards_128_data.as_mut());

    group.bench_function("formal_derivative 128", |b| {
        b.iter(|| E::formal_derivative(black_box(&mut shards_128)))
    });

    // FFT IFFT

    group.bench_function("FFT 128", |b| {
        b.iter(|| {
            engine.fft(
                black_box(&mut shards_128),
                black_box(0),
                black_box(128),
                black_box(128),
                black_box(128),
            )
        })
    });

    group.bench_function("IFFT 128", |b| {
        b.iter(|| {
            engine.ifft(
                black_box(&mut shards_128),
                black_box(0),
                black_box(128),
                black_box(128),
                black_box(128),
            )
        })
    });

    group.finish();
}

// ======================================================================
// MAIN

criterion_group!(benches_main, benchmarks_main);
criterion_group!(benches_rate, benchmarks_rate);
criterion_group!(benches_engine, benchmarks_engine);
criterion_main!(benches_main, benches_rate, benches_engine);
