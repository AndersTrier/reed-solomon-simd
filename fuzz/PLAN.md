# Fuzzing plan

Status and next steps for fuzzing `reed-solomon-simd`. Items are ordered by
expected value per effort.

## Done

- `fuzz/` with three cargo-fuzz targets: `roundtrip`, `api_sequence`,
  `engine_diff` (see [README](README.md)).
- Initial runs (x86_64, AVX2 machine, ASan): ~5 minutes per target.
  - `api_sequence` found a panic within seconds: a `reset` that switches
    between high and low rate with an invalid `shard_bytes` returns `Err`
    but leaves `DefaultRateEncoder`/`DefaultRateDecoder` (and thus
    `ReedSolomonEncoder`/`ReedSolomonDecoder`) in an internal `None` state;
    the next call panics with `unreachable!()`. Fixed on branch
    `nikolai/reset-bug` with a regression test.
  - `roundtrip` and `engine_diff`: no findings.
- Sanity check: planting a bug in `Avx2::mul` (skip all but the first
  chunk) is caught by `engine_diff` within seconds.

## 1. Land what exists

- [ ] Merge the `reset` fix.
- [ ] Remove the `reset_is_known_to_panic` workaround from `api_sequence`.
- [ ] Add a minimized reproducer of the `reset` bug to the `api_sequence`
      seed corpus (`fuzz/corpus/api_sequence/` would need to be committed,
      or a separate `fuzz/regressions/` directory replayed in CI).

## 2. Continuous fuzzing in CI

- [ ] PR smoke job: build all targets on nightly and run each for ~60 s.
      Run on both `ubuntu-24.04` (SSSE3/AVX2) and `ubuntu-24.04-arm`
      (Neon) — `engine_diff` only compares the engines the host CPU has, so
      without the ARM runner the Neon engine is never fuzzed.
- [ ] Persist the corpus between runs (`actions/cache` keyed per target and
      architecture), so that coverage accumulates instead of restarting.
- [ ] Scheduled job (e.g. daily, 30-60 min per target, `-s none` for
      `engine_diff`) that uploads `fuzz/artifacts/` on failure.
- [ ] Optionally: ClusterFuzzLite for corpus management and crash
      deduplication, or apply to OSS-Fuzz if the crate's adoption warrants it.

## 3. Better inputs

- [ ] Seed corpus at known boundaries, since the interesting dimensions
      are structural rather than byte-level:
  - shard counts: 1, powers of two and ±1 around them, 32768, the
    high/low rate limits (4096 / 61440), `original_count == recovery_count`
    (where `DefaultRate` deliberately picks the "wrong" rate);
  - `shard_bytes`: 2, 62, 64, 66, 126, 128, 130.
- [ ] Bias `int_in_range` choices toward these boundaries in the targets
      themselves (e.g. pick "boundary vs uniform" first, then the value).
- [ ] Measure coverage (`cargo fuzz coverage` + `llvm-cov`) and check that
      all error paths, rate-switch branches in `reset`, and the
      partial-last-chunk paths are reached.

## 4. Large shard counts

Current targets cap counts at 1024 for throughput. Counts up to 65535 run
through the same code but deeper FFT recursion and different
`truncated_size` handling.

- [ ] Separate `roundtrip_large` target (counts up to `GF_ORDER`, few
      fixed-size shards, `-s none`), run only in the scheduled job.
- [ ] Alternatively, a `proptest`-based test with a weighted count
      distribution, run with `--include-ignored` like the existing
      random roundtrips (`examples/test-random-roundtrips.rs` already covers
      this space randomly; a property test would shrink failures).

## 5. More oracles

- [ ] **Version compatibility**: the README promises that shards are
      compatible across versions when `shard_bytes` is a multiple of 64.
      Differential target against `reed-solomon-simd` 2.x (renamed
      dependency): identical recovery shards, and cross-version decode.
- [ ] **Leopard compatibility**: differential against `leopard-codec`
      (already a dev-dependency) for multiple-of-64 shard sizes, if its
      shard layout matches.
- [ ] **Engine primitives**: direct differential of `Engine::fft`/`ifft`
      and `eval_poly` against `Naive`. Requires pinning down the valid
      domain of `pos`, `size`, `truncated_size` and `skew_delta` (from the
      call sites in `rate_high.rs` / `rate_low.rs`) so the target doesn't
      report precondition violations as bugs.
- [ ] **Lower-level API**: `HighRate`/`LowRate` encoders/decoders directly,
      and moving `EncoderWork`/`DecoderWork` between differently configured
      encoders/decoders via `into_parts`.

## 6. Complementary tooling (not fuzzing)

- [ ] Run unit tests under Miri for the `NoSimd`/`Naive` engines (and the
      SIMD engines as far as Miri supports their intrinsics).
- [ ] `Avx2::new()`, `Ssse3::new()` and `Neon::new()` are safe public
      constructors that don't check CPU support; calling e.g. `Avx2::mul`
      on a CPU without AVX2 executes AVX2 instructions, which is undefined
      behavior. Fuzzing can't find this. Fix by making the constructors
      `unsafe` or adding a checked `try_new()` (breaking change, needs
      a major version).

## Out of scope

- Corrupted (as opposed to missing) shards: Reed-Solomon erasure coding
  can't detect corruption, so "wrong data in, wrong data out" isn't a bug.
