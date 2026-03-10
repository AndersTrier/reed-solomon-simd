Algorithm documentation.

As I don't understand algorithm fully myself,
I'll just document some parts which I do understand.

# Shard

- Reed-Solomon `GF(2^16)` erasure coding works on 16-bit elements ([`GfElement`]).
- A **shard** is a byte-array which is interpreted as an array of [`GfElement`]:s.

A naive implementation could e.g. require shards to be a multiple of **2 bytes**
and then interpret each byte-pair as low/high parts of a single [`GfElement`]:

```text
[ low_0, high_0, low_1, high_1, ...]
```

However that approach isn't good for SIMD optimizations.
Instead shards are required to be a multiple of **64 bytes**.
In each 64-byte block first 32 bytes are low parts of 32 [`GfElement`]:s
and last 32 bytes are high parts of those 32 [`GfElement`]:s.

```text
[ low_0, low_1, ..., low_31, high_0, high_1, ..., high_31 ]
```

A shard then consists of one or more of these 64-byte blocks:

```text
// -------- first 64-byte block --------- | --------- second 64-byte block ---------- | ...
[ low_0, ..., low_31, high_0, ..., high_31, low_32, ..., low_63, high_32, ..., high_63, ... ]
```

# Rate

Encoding and decoding both have two variations:

- **High rate** refers to having more original shards than recovery shards.
    - High rate must be used when there are over 32768 original shards.
    - High rate encoding uses **chunks** of `recovery_count.next_power_of_two()` shards.
- **Low rate** refers to having more recovery shards than original shards.
    - Low rate must be used when there are over 32768 recovery shards.
    - Low rate encoding uses **chunks** of `original_count.next_power_of_two()` shards.
- Because of padding either rate can be used when there are
  at most 32768 original shards and at most 32768 recovery shards.
    - High rate and low rate are not [^1] compatible with each other,
      i.e. decoding must use same rate that encoding used.
    - With multiple chunks "correct" rate is generally faster in encoding
      and not-slower in decoding.
    - With single chunk "wrong" rate is generally faster in decoding
      if `original_count` and `recovery_count` differ a lot.

[^1]: They seem to be compatible with single chunk. However I don't quite
    understand why and I don't recommend relying on this.

## Benchmarks

- These benchmarks are from `cargo bench rate`
  and use similar setup than [main benchmarks],
  except with maximum possible shard loss.

| original : recovery | Chunks  | `HighRateEncoder` | `LowRateEncoder` | `HighRateDecoder` | `LowRateDecoder` |
| ------------------- | ------- | ----------------- | ---------------- | ----------------- | ---------------- |
| 1024 : 1024         | 1x 1024 | 175 MiB/s         | 176 MiB/s        | 76 MiB/s          | 75 MiB/s         |
| 1024 : 1025 (Low)   | 2x 1024 | 140               | **153**          | 47                | **59**           |
| 1025 : 1024 (High)  | 2x 1024 | **152**           | 132              | **60**            | 46               |
| 1024 : 2048 (Low)   | 2x 1024 | 157               | **169**          | 70                | 70               |
| 2048 : 1024 (High)  | 2x 1024 | **167**           | 151              | 69                | 68               |
| 1025 : 1025         | 1x 2048 | 125               | 126              | 44                | 43               |
| 1025 : 2048 (Low)   | 1x 2048 | 144               | 144              | **65** **!!!**    | 53               |
| 2048 : 1025 (High)  | 1x 2048 | 144               | 145              | 53                | **62** **!!!**   |
| 2048 : 2048         | 1x 2048 | 156               | 157              | 70                | 69               |

[main benchmarks]: crate#benchmarks

# Encoding

Encoding takes original shards as input and generates recovery shards.

## High rate encoding

- Encoding is done in **chunks** of `recovery_count.next_power_of_two()` shards.
- Original shards are split into chunks and last chunk
  is padded with zero-filled shards if needed.
    - In theory original shards are padded to [`GF_ORDER`]` - chunk_size` shards
      but since `IFFT([0u8; x]) == [0u8; x]` and `xor` with `0` is no-op,
      the chunks which contain only `0u8`:s can be ignored.
- Recovery shards fit into a single chunk
  which is padded with unused shards if needed.
- Recovery chunk is generated with following algorithm

```text
recovery_chunk = FFT(
    IFFT(original_chunk_0, skew_0) xor
    IFFT(original_chunk_1, skew_1) xor
    ...
)
```

This is implemented in [`HighRateEncoder`].

## Low rate encoding

- Encoding is done in **chunks** of `original_count.next_power_of_two()` shards.
- Original shards fit into a single chunk
  which is padded with zero-filled shards if needed.
- Recovery shards are generated in chunks and last chunk
  is padded with unused shards if needed.
    - In theory recovery shards are padded to [`GF_ORDER`]` - chunk_size` shards
      but chunks which contain only unused shards can be ignored.
- Recovery chunks are generated with following algorithm

```text
recovery_chunk_0 = FFT( IFFT(original_chunk), skew_0 )
recovery_chunk_1 = FFT( IFFT(original_chunk), skew_1 )
...
```

This is implemented in [`LowRateEncoder`].

# Decoding

Decoding recovers erased original shards from any `original_count` received shards
(original or recovery). It requires exactly `original_count` received shards.

## Mathematical basis

Let `Ω` be the set of erased positions and `R` the received positions.
Define the **erasure locator polynomial**:

```text
E(x) = ∏_{j ∈ Ω} (x ⊕ α_j)    over GF(2^16)
```

where `α_j` is the evaluation point for position `j`. Key properties:
- `E(α_j) = 0` for all `j ∈ Ω` (erased positions are roots)
- `E(α_j) ≠ 0` for all `j ∈ R` (received positions are not roots)

The **formal derivative** `D` satisfies the Leibniz product rule over any field:

```text
D(f · g) = D(f)·g + f·D(g)
```

At an erased position where `E(α_j) = 0`, this gives:

```text
D(C · E)(α_j) = C(α_j) · D(E)(α_j)
```

where `C(x)` is the codeword polynomial. So the erased value is recoverable as:

```text
C(α_j) = D(C · E)(α_j) / D(E)(α_j)
```

`D(E)(α_j) = ∏_{k ∈ Ω, k≠j}(α_j - α_k)` is nonzero whenever all evaluation
points are distinct, which holds as long as `original_count + recovery_count ≤ GF_ORDER`.

In GF(2^16) with characteristic 2, the formal derivative kills all even-power terms:
`D(x^{2k}) = 0`. Its kernel is the set of polynomials in `x²`, but this does not
affect correctness since `E` has only simple roots (from distinct evaluation points).

## Decoding algorithm

The decoder operates on a flat work buffer laid out as:

```text
High rate:  work[0 .. recovery_count]            = received recovery shards
            work[chunk_size .. chunk_size+n]      = received original shards

Low rate:   work[0 .. original_count]            = received original shards
            work[chunk_size .. chunk_size+m]     = received recovery shards
```

Missing shards and padding gaps are zeroed.

### Step 1 — Compute erasure locator values (FWHT)

Build indicator vector `erasures[i] = 1` if position `i` is erased, else `0`.
Evaluate `E(α_i)` for all `i` simultaneously using the Fast Walsh-Hadamard Transform:

```text
1. erasures ← FWHT(erasures, active_size)
2. erasures[i] ← erasures[i] * log_walsh[i]   (mod GF_MODULUS, pointwise)
3. erasures ← FWHT(erasures, GF_ORDER)
```

Result: `erasures[i] = log(E(α_i))` for each position `i`. The `log_walsh` table is
the precomputed FWHT of the logarithm table: `FWHT(log[·])`. This works because
`log E(x) = Σ_{j∈Ω} log(x ⊕ α_j)` decomposes into a convolution in the Walsh-Hadamard
domain, which the FWHT diagonalizes.

### Step 2 — Scale received shards by E

```text
work[i] ← work[i] · E(α_i)   for each received i  (multiply using log: erasures[i])
work[i] ← 0                   for each erased i
```

This forms `R̃(α_i) = C(α_i) · E(α_i)` at received positions and `0` at erased
positions, making `R̃` a well-defined polynomial in the work buffer.

### Step 3 — IFFT → formal derivative → FFT

```text
work ← IFFT(work)          // evaluation domain → coefficient domain
work ← D(work)             // formal derivative in coefficient domain
work ← FFT(work)           // coefficient domain → evaluation domain
```

The formal derivative in the Cantor-basis coefficient representation is
(see `formal_derivative` in `utils.rs`):

```text
for i = 1 to len-1:
    width = 1 << trailing_zeros(i)
    work[i - width] ^= work[i]
```

This butterfly pattern implements the matrix of `D` in the subspace polynomial
(Cantor) basis. It runs in O(n log n).

After these three steps, `work` holds `D(R̃)(α_i)` at each evaluation point, which
at erased positions equals `C(α_i) · D(E)(α_i)`.

### Step 4 — Reveal erased shards

```text
for each erased position j:
    work[j] ← work[j] · E(α_j)^{-1}
```

implemented as:

```text
engine.mul(&mut work[j], GF_MODULUS - erasures[j])
```

`GF_MODULUS - erasures[j]` is `-log(E(α_j))` in the log domain, i.e., multiplication
by `E(α_j)^{-1}`. Since `D(E)(α_j) = E(α_j)` holds in this Cantor-basis construction
(up to the sign absorbed into the erasure polynomial definition), this single multiply
recovers `C(α_j)`.

## High rate decoding

Implemented in [`HighRateDecoder`]. Uses `chunk_size = recovery_count.next_power_of_two()`.
`work_count = (chunk_size + original_count).next_power_of_two()`.
The IFFT/D/FFT triple operates on the full `work_count`-sized buffer, spanning both
the recovery region `[0, chunk_size)` and the original region `[chunk_size, chunk_size+n)`.

## Low rate decoding

Implemented in [`LowRateDecoder`]. Uses `chunk_size = original_count.next_power_of_two()`.
`work_count = (chunk_size + recovery_count).next_power_of_two()`.
Same structure with original and recovery regions swapped.


[`GfElement`]: crate::engine::GfElement
[`HighRateEncoder`]: crate::rate::HighRateEncoder
[`HighRateDecoder`]: crate::rate::HighRateDecoder
[`LowRateEncoder`]: crate::rate::LowRateEncoder
[`LowRateDecoder`]: crate::rate::LowRateDecoder

[`GF_ORDER`]: crate::engine::GF_ORDER
