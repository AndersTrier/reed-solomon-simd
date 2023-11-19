# reed-solomon-simd

Reed-Solomon erasure coding, featuring:

- Any combination of 1 - 32768 original shards with 1 - 32768 recovery shards.
- `O(n log n)` complexity.
- Runtime selection of best SIMD implementation for x86(-64) (SSSE3 and AVX2) with
fallback to plain Rust.
- Entirely written in Rust.

## Benchmarks

| Original : Recovery | Encode       | Decode (1% loss; 100% loss) |
| ------------------- | ------------ | --------------------------- |
| 64: 64              | 7.1111 GiB/s | 264.10 MiB/s ; 261.10 MiB/s |
| 128 : 128           | 5.5354 GiB/s | 464.81 MiB/s ; 461.00 MiB/s |
| 100 : 1 000         | 5.1622 GiB/s | 600.51 MiB/s ; 600.01 MiB/s |
| 1 000 : 100         | 4.2336 GiB/s | 746.80 MiB/s ; 749.49 MiB/s |
| 1 000 : 1 000       | 3.6074 GiB/s | 1.0584 GiB/s ; 1.0471 GiB/s |
| 1 000 : 10 000      | 3.1045 GiB/s | 769.13 MiB/s ; 793.63 MiB/s |
| 10 000 : 1 000      | 2.4273 GiB/s | 718.68 MiB/s ; 730.49 MiB/s |
| 10 000 : 10 000     | 1.9195 GiB/s | 453.12 MiB/s ; 443.75 MiB/s |
| 32 768 : 32 768     | 1.4868 GiB/s | 621.78 MiB/s ; 603.40 MiB/s |

- Single core AVX2 on an AMD Ryzen 5 3600 (Zen 2, 2019).
- MiB/s and GiB/s are w.r.t the total amount of data,
  i.e. original shards + recovery shards.
    - For decoder this includes missing shards.
- Shards are 1024 bytes.
- Encode benchmark
    - Includes [`add_original_shard`][RSE::add_original_shard] and
      [`encode`][RSE::encode] of [`ReedSolomonEncoder`].
- Decode benchmark
    - Has two MiB/s values for 1% and 100% original shard loss, of maximum possible.
    - Provides minimum required amount of shards to decoder.
    - Includes [`add_original_shard`][RSD::add_original_shard],
      [`add_recovery_shard`][RSD::add_recovery_shard] and
      [`decode`][RSD::decode] of [`ReedSolomonDecoder`].


I invite you to clone [reed-solomon-simd] and run your own benchmark:
```sh
$ cargo bench main
```

## Simple usage

1. Divide data into equal-sized original shards.
   Shard size must be multiple of 64 bytes.
2. Decide how many recovery shards you want.
3. Generate recovery shards with [`reed_solomon_simd::encode`].
4. When some original shards get lost, restore them with [`reed_solomon_simd::decode`].
    - You must provide at least as many shards as there were original shards in total,
      in any combination of original shards and recovery shards.

### Example

Divide data into 3 original shards of 64 bytes each and generate 5 recovery shards.
Assume then that original shards #0 and #2 are lost
and restore them by providing 1 original shard and 2 recovery shards.

```rust
let original = [
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
    b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
    b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
];

let recovery = reed_solomon_simd::encode(
    3, // total number of original shards
    5, // total number of recovery shards
    original, // all original shards
)?;

let restored = reed_solomon_simd::decode(
    3, // total number of original shards
    5, // total number of recovery shards
    [  // provided original shards with indexes
        (1, &original[1]),
    ],
    [  // provided recovery shards with indexes
        (1, &recovery[1]),
        (4, &recovery[4]),
    ],
)?;

assert_eq!(restored[&0], original[0]);
assert_eq!(restored[&2], original[2]);
# Ok::<(), reed_solomon_simd::Error>(())
```

## Basic usage

[`ReedSolomonEncoder`] and [`ReedSolomonDecoder`] give more control
of the encoding/decoding process.

Here's the above example using these instead:

```rust
use reed_solomon_simd::{ReedSolomonDecoder, ReedSolomonEncoder};
use std::collections::HashMap;

let original = [
    b"Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ",
    b"eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut e",
    b"nim ad minim veniam, quis nostrud exercitation ullamco laboris n",
];

let mut encoder = ReedSolomonEncoder::new(
    3, // total number of original shards
    5, // total number of recovery shards
    64, // shard size in bytes
)?;

for original in original {
    encoder.add_original_shard(original)?;
}

let result = encoder.encode()?;
let recovery: Vec<_> = result.recovery_iter().collect();

let mut decoder = ReedSolomonDecoder::new(
    3, // total number of original shards
    5, // total number of recovery shards
    64, // shard size in bytes
)?;

decoder.add_original_shard(1, original[1])?;
decoder.add_recovery_shard(1, recovery[1])?;
decoder.add_recovery_shard(4, recovery[4])?;

let result = decoder.decode()?;
let restored: HashMap<_, _> = result.restored_original_iter().collect();

assert_eq!(restored[&0], original[0]);
assert_eq!(restored[&2], original[2]);
# Ok::<(), reed_solomon_simd::Error>(())
```

## Advanced usage

See [`rate`][mod:rate] module for advanced encoding/decoding
using chosen [`Engine`] and [`Rate`].

## Benchmarks against other crates

Use `cargo run --release --example quick-comparison`
to run few simple benchmarks against [`reed-solomon-16`], [`reed-solomon-erasure`]
and [`reed-solomon-novelpoly`] crates.

This crate is the fastest in all cases on my AMD Ryzen 5 3600, except in the
case of decoding with about 42 or fewer recovery shards.
There's also a one-time initialization (< 10 ms) for computing tables
which can dominate at really small data amounts.

[`reed-solomon-16`]: https://crates.io/crates/reed-solomon-16
[`reed-solomon-erasure`]: https://crates.io/crates/reed-solomon-erasure
[`reed-solomon-novelpoly`]: https://crates.io/crates/reed-solomon-novelpoly

## Running tests

Some larger tests are marked `#[ignore]` and are not run with `cargo test`.
Use `cargo test -- --ignored` to run those.

## Safety

This crate only uses `unsafe` code in the [`Ssse3`] and [`Avx2`] [`Engine`]s.

## Credits

This crate is a fork of Markus Laire's [`reed-solomon-16`] crate, which in turn
is based on [Leopard-RS] by Christopher A. Taylor.

[Leopard-RS]: https://github.com/catid/leopard
[reed-solomon-simd]: https://github.com/AndersTrier/reed-solomon-simd

[`Naive`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/engine/struct.Naive.html
[`NoSimd`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/engine/struct.NoSimd.html
[`Ssse3`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/engine/struct.Ssse3.html
[`Avx2`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/engine/struct.Avx2.html

[`ReedSolomonEncoder`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonEncoder.html
[RSE::add_original_shard]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonEncoder.html#method.add_original_shard
[RSE::encode]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonEncoder.html#method.encode

[`ReedSolomonDecoder`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonDecoder.html
[RSD::add_original_shard]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonDecoder.html#method.add_original_shard
[RSD::add_recovery_shard]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonDecoder.html#method.add_recovery_shard
[RSD::decode]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/struct.ReedSolomonDecoder.html#method.decode

[`Engine`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/engine/trait.Engine.html
[`Rate`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/rate/trait.Rate.html

[mod:rate]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/rate/index.html

[`reed_solomon_simd::encode`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/fn.encode.html
[`reed_solomon_simd::decode`]: https://docs.rs/reed-solomon-simd/2.0.0/reed_solomon_simd/fn.decode.html
