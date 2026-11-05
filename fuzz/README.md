# Fuzzing

Fuzz targets for `reed-solomon-simd`, using [cargo-fuzz] (libFuzzer).

[cargo-fuzz]: https://github.com/rust-fuzz/cargo-fuzz

## Setup

```sh
rustup toolchain install nightly
cargo install cargo-fuzz
```

## Targets

| Target         | Property                                                                                                                                                                                       |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `roundtrip`    | Encode, erase an arbitrary subset of shards, decode with shards given in arbitrary order: restoration is exact, and decoding fails iff too few shards were given. Runs two rounds per input on the same encoder/decoder (via `reset`) to exercise working-space reuse and high/low rate switches. |
| `api_sequence` | Arbitrary sequences of public API calls with arbitrary, mostly invalid, arguments never panic.                                                                                                |
| `engine_diff`  | All engines (`Naive`, `NoSimd`, and `Ssse3`/`Avx2`/`Neon` when supported by the CPU) produce bit-identical recovery and restored shards for a given rate, and identical `Engine::mul` results. |

Shard counts are capped at 1024 and shard sizes at 200 bytes to keep
executions fast. Shard sizes cover the partial 64-byte last chunk handling.

## Running

From the repository root:

```sh
# Run until a failure is found (Ctrl-C to stop).
cargo +nightly fuzz run roundtrip

# Time-limited run, in release mode with debug assertions (default).
cargo +nightly fuzz run -O api_sequence -- -max_total_time=300

# Several workers in parallel.
cargo +nightly fuzz run engine_diff --jobs 8 -- -max_total_time=600
```

All `unsafe` code in the crate is SIMD loads/stores on `[u8; 64]` chunks,
so most bugs these targets can find are logic bugs or panics rather than
memory-safety bugs. AddressSanitizer is enabled by default; disabling it
(`-s none`) gives roughly 4x more executions per second, which is useful
for longer differential runs of `engine_diff`.

## Reproducing and minimizing failures

Failing inputs are written to `fuzz/artifacts/<target>/`.

```sh
cargo +nightly fuzz run <target> fuzz/artifacts/<target>/crash-...
cargo +nightly fuzz tmin <target> fuzz/artifacts/<target>/crash-...
cargo +nightly fuzz fmt <target> fuzz/artifacts/<target>/crash-...   # api_sequence: print the ops
```

## Coverage

```sh
cargo +nightly fuzz coverage <target>
```

See the [cargo-fuzz book](https://rust-fuzz.github.io/book/cargo-fuzz/coverage.html)
for turning the result into a report.
