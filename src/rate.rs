//! Advanced encoding/decoding using chosen [`Engine`] and [`Rate`].
//!
//! **This is an advanced module which is not needed for [simple usage] or [basic usage].**
//!
//! This module is relevant if you want to
//! - encode/decode using other [`Engine`] than [`DefaultEngine`].
//! - re-use working space of one encoder/decoder in another.
//! - understand/benchmark/test high or low rate directly.
//!
//! # Rates
//!
//! See [algorithm > Rate] for details about high/low rate.
//!
//! - [`DefaultRate`], [`DefaultRateEncoder`], [`DefaultRateDecoder`]
//!     - Encoding/decoding using high or low rate as appropriate.
//!     - These are basically same as [`ReedSolomonEncoder`]
//!       and [`ReedSolomonDecoder`] except with slightly different API
//!       which allows specifying [`Engine`] and working space.
//! - [`HighRate`], [`HighRateEncoder`], [`HighRateDecoder`]
//!     - Encoding/decoding using only high rate.
//! - [`LowRate`], [`LowRateEncoder`], [`LowRateDecoder`]
//!     - Encoding/decoding using only low rate.
//!
//! # Received shards
//!
//! [`ReceivedShards`] is a bit-per-shard abstraction over which shards were received,
//! with implementations for `[bool]`, [`ReceivedShardBits`] and more.
//!
//! [simple usage]: crate#simple-usage
//! [basic usage]: crate#basic-usage
//! [algorithm > Rate]: crate::algorithm#rate
//! [`ReedSolomonEncoder`]: crate::ReedSolomonEncoder
//! [`ReedSolomonDecoder`]: crate::ReedSolomonDecoder
//! [`DefaultEngine`]: crate::engine::DefaultEngine

use crate::{engine::Engine, DecoderResult, EncoderResult, Error};
use core::ops::Range;
use fixedbitset::FixedBitSet;

pub use self::{
    decoder_work::DecoderWork,
    encoder_work::EncoderWork,
    rate_default::{DefaultRate, DefaultRateDecoder, DefaultRateEncoder},
    rate_high::{HighRate, HighRateDecoder, HighRateEncoder},
    rate_low::{LowRate, LowRateDecoder, LowRateEncoder},
};

mod decoder_work;
mod encoder_work;
mod rate_default;
mod rate_high;
mod rate_low;

// ======================================================================
// ReceivedShards - PUBLIC

/// Which shards were received and which are missing.
///
/// This is a bit-per-shard abstraction over the "received" flags that decoding needs, allowing
/// callers to use whatever representation suits them best: one `bool` per shard, a bit-packed
/// array of words (including stack-allocated ones for small shard counts), a [`FixedBitSet`] or a
/// custom type.
///
/// The trait is word-oriented: the only required method returns the received flags of 64 shards
/// at once, so that decoding can skip over whole words of received shards instead of inspecting
/// every shard individually. Everything else is provided on top of that, but implementations are
/// free to override the provided methods with something more efficient.
///
/// Implementations are provided for:
/// - `[bool]` and `[bool; N]`: one byte per shard, index `i` corresponds to element `i`
/// - [`ReceivedShardBits`]: bit-packed representation over `[u64]`
/// - [`FixedBitSet`]
/// - `&T` for any `T: ReceivedShards + ?Sized`
///
/// # Examples
///
/// ```
/// use reed_solomon_simd::rate::{ReceivedShardBits, ReceivedShards};
///
/// // Shards 0 and 2 were received, shard 1 was lost.
/// let flags = [true, false, true];
/// assert!(flags.received(0));
/// assert!(!flags.received(1));
/// assert_eq!(flags.received_count(0..3), 2);
/// assert_eq!(flags.missing_in(0..3).collect::<Vec<_>>(), [1]);
///
/// // Same, bit-packed: shard `i` is bit `i % 64` of word `i / 64`, least significant bit first.
/// let bits = ReceivedShardBits([0b101u64]);
/// assert_eq!(bits.received_word(0), 0b101);
/// assert!(bits.received(0));
/// assert!(!bits.received(1));
/// assert!(bits.received(2));
/// ```
pub trait ReceivedShards {
    /// Returns the received flags of shards `word_index * 64 .. word_index * 64 + 64`,
    /// bit-packed with the least significant bit first: bit `i` of the returned word is the flag
    /// of shard `word_index * 64 + i`, and a set bit means the shard was received.
    ///
    /// Bits which correspond to indices outside the range of the underlying representation must
    /// be zero, and word indices which are entirely out of range must return `0` instead of
    /// panicking.
    ///
    /// # Examples
    ///
    /// ```
    /// use reed_solomon_simd::rate::ReceivedShards;
    ///
    /// // Shards 0 and 2 were received, shard 1 was lost.
    /// let flags = [true, false, true];
    /// assert_eq!(flags.received_word(0), 0b101);
    /// assert_eq!(flags.received_word(1), 0);
    /// ```
    fn received_word(&self, word_index: usize) -> u64;

    /// Returns `true` if the shard at the given index was received.
    ///
    /// Indices which are out of range of the underlying representation return `false` instead
    /// of panicking.
    #[inline(always)]
    fn received(&self, index: usize) -> bool {
        self.received_word(index / 64) >> (index % 64) & 1 == 1
    }

    /// Returns the number of received shards with index in `range`.
    #[inline]
    fn received_count(&self, range: Range<usize>) -> usize {
        let Range { start, end } = range;

        if start == end {
            return 0;
        }

        let mut count = 0;
        for word_index in start / 64..end.div_ceil(64) {
            let word_start = word_index * 64;
            let lo = start.saturating_sub(word_start);
            let hi = (end - word_start).min(64);
            count += (self.received_word(word_index) & low_bits(hi) & !low_bits(lo)).count_ones()
                as usize;
        }

        count
    }

    /// Returns the indices in `range` of the shards which were **not** received,
    /// in ascending order.
    ///
    /// Whole words of received shards are skipped, so this costs `O(range.len() / 64)` when
    /// nothing is missing.
    #[inline]
    fn missing_in(&self, range: Range<usize>) -> impl Iterator<Item = usize> {
        MissingIn::new(self, range)
    }
}

/// Mask with the lowest `count` bits set.
///
/// Clearing the lowest `count` bits instead is `!low_bits(count)`.
#[inline(always)]
fn low_bits(count: usize) -> u64 {
    debug_assert!(count <= 64);

    // Shifting by 64 would overflow, `unbounded_shr` yields 0 instead
    u64::MAX.unbounded_shr((64 - count) as u32)
}

/// Iterator over the missing shard indices of a [`ReceivedShards`], see
/// [`ReceivedShards::missing_in`].
struct MissingIn<'a, T: ?Sized> {
    received: &'a T,
    /// Missing shards of the current word which have not been yielded yet, bit `i` corresponding
    /// to shard `word_start + i`.
    missing: u64,
    /// Index of the first shard of the current word.
    word_start: usize,
    /// Exclusive end of the range.
    end: usize,
}

impl<'a, T: ReceivedShards + ?Sized> MissingIn<'a, T> {
    #[inline]
    fn new(received: &'a T, range: Range<usize>) -> Self {
        let Range { start, end } = range;

        if start >= end {
            return Self {
                received,
                missing: 0,
                word_start: 0,
                end: 0,
            };
        }

        let word_start = start / 64 * 64;

        let mut this = Self {
            received,
            missing: 0,
            word_start,
            end,
        };
        // The first word starts part way into the range
        this.missing = this.load() & !low_bits(start - word_start);
        this
    }

    /// Loads the missing shards of the current word, dropping any past the end of the range.
    #[inline]
    fn load(&self) -> u64 {
        let len = (self.end - self.word_start).min(64);
        !self.received.received_word(self.word_start / 64) & low_bits(len)
    }
}

impl<T: ReceivedShards + ?Sized> Iterator for MissingIn<'_, T> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        loop {
            if self.missing != 0 {
                let bit = self.missing.trailing_zeros() as usize;
                self.missing &= self.missing - 1;
                return Some(self.word_start + bit);
            }

            // Whole word was received (or is exhausted), move on to the next one.
            self.word_start += 64;
            if self.word_start >= self.end {
                return None;
            }
            self.missing = self.load();
        }
    }
}

impl<T: ReceivedShards + ?Sized> ReceivedShards for &T {
    #[inline(always)]
    fn received_word(&self, word_index: usize) -> u64 {
        T::received_word(self, word_index)
    }

    #[inline(always)]
    fn received(&self, index: usize) -> bool {
        T::received(self, index)
    }

    #[inline]
    fn received_count(&self, range: Range<usize>) -> usize {
        T::received_count(self, range)
    }

    #[inline]
    fn missing_in(&self, range: Range<usize>) -> impl Iterator<Item = usize> {
        T::missing_in(self, range)
    }
}

/// One `bool` per shard. Out-of-range indices are treated as not received.
impl ReceivedShards for [bool] {
    #[inline]
    fn received_word(&self, word_index: usize) -> u64 {
        let Some(start) = word_index.checked_mul(64) else {
            return 0;
        };
        let Some(flags) = self.get(start..) else {
            return 0;
        };

        let mut word = 0;

        if let Some(flags) = flags.first_chunk::<64>() {
            // Fixed size loop which the compiler can vectorize.
            for (i, flag) in flags.iter().enumerate() {
                word |= u64::from(*flag) << i;
            }
        } else {
            for (i, flag) in flags.iter().enumerate() {
                word |= u64::from(*flag) << i;
            }
        }

        word
    }

    #[inline(always)]
    fn received(&self, index: usize) -> bool {
        self.get(index) == Some(&true)
    }
}

/// One `bool` per shard. Out-of-range indices are treated as not received.
impl<const N: usize> ReceivedShards for [bool; N] {
    #[inline]
    fn received_word(&self, word_index: usize) -> u64 {
        self.as_slice().received_word(word_index)
    }

    #[inline(always)]
    fn received(&self, index: usize) -> bool {
        self.as_slice().received(index)
    }
}

/// Out-of-range indices are treated as not received.
impl ReceivedShards for FixedBitSet {
    #[inline]
    fn received_word(&self, word_index: usize) -> u64 {
        // `FixedBitSet` blocks are `usize`, which may be narrower than 64 bits.
        const BLOCK_BITS: usize = usize::BITS as usize;

        let Some(start) = word_index.checked_mul(64) else {
            return 0;
        };
        let len = self.len();
        if start >= len {
            return 0;
        }

        let blocks = self.as_slice();
        let mut word = 0;
        let mut shift = 0;
        while shift < 64 {
            if let Some(block) = blocks.get(start / BLOCK_BITS + shift / BLOCK_BITS) {
                word |= (*block as u64) << shift;
            }
            shift += BLOCK_BITS;
        }

        // Bits beyond the length of the set must be zero.
        word & low_bits((len - start).min(64))
    }

    #[inline(always)]
    fn received(&self, index: usize) -> bool {
        self.contains(index)
    }
}

/// Bit-packed [`ReceivedShards`] over a sequence of `u64` words.
///
/// The shard at `index` is stored in bit `index % 64` of word `index / 64`, in other words the
/// least significant bit of the first word corresponds to shard `0`, the most significant bit of
/// the first word to shard `63`, the least significant bit of the second word to shard `64` and
/// so on.
///
/// Indices beyond the last word are treated as not received, so `ceil(shard_count / 64)` words
/// are enough (but a longer sequence is fine as well).
///
/// The inner value can be anything that dereferences to a `u64` slice, for instance `[u64; N]`
/// (no allocation needed), `&[u64]` or `Vec<u64>`.
///
/// # Examples
///
/// ```
/// use reed_solomon_simd::rate::{ReceivedShardBits, ReceivedShards};
///
/// let bits = ReceivedShardBits([0b1000_0011u64, 0b1]);
/// assert!(bits.received(0));
/// assert!(bits.received(1));
/// assert!(!bits.received(2));
/// assert!(bits.received(7));
/// assert!(bits.received(64));
/// assert!(!bits.received(65));
/// assert!(!bits.received(1000));
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ReceivedShardBits<T>(pub T);

impl<T: AsRef<[u64]>> ReceivedShardBits<T> {
    /// Returns the underlying words.
    pub fn as_words(&self) -> &[u64] {
        self.0.as_ref()
    }
}

impl<T: AsRef<[u64]>> ReceivedShards for ReceivedShardBits<T> {
    #[inline(always)]
    fn received_word(&self, word_index: usize) -> u64 {
        match self.0.as_ref().get(word_index) {
            Some(word) => *word,
            None => 0,
        }
    }
}

// ======================================================================
// Rate - PUBLIC

/// Reed-Solomon encoder/decoder generator using specific rate.
pub trait Rate<E: Engine> {
    // ============================================================
    // REQUIRED

    /// Encoder of this rate.
    type RateEncoder: RateEncoder<E>;
    /// Decoder of this rate.
    type RateDecoder: RateDecoder<E>;

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    fn supports(original_count: usize, recovery_count: usize) -> bool;

    // ============================================================
    // PROVIDED

    /// Creates new encoder. This is same as [`RateEncoder::new`].
    fn encoder(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self::RateEncoder, Error> {
        Self::RateEncoder::new(original_count, recovery_count, shard_bytes, engine, work)
    }

    /// Creates new decoder. This is same as [`RateDecoder::new`].
    fn decoder(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self::RateDecoder, Error> {
        Self::RateDecoder::new(original_count, recovery_count, shard_bytes, engine, work)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        if !Self::supports(original_count, recovery_count) {
            Err(Error::UnsupportedShardCount {
                original_count,
                recovery_count,
            })
        } else if shard_bytes == 0 || shard_bytes & 1 != 0 {
            Err(Error::InvalidShardSize { shard_bytes })
        } else {
            Ok(())
        }
    }
}

// ======================================================================
// RateEncoder - PUBLIC

/// Reed-Solomon encoder using specific rate.
pub trait RateEncoder<E: Engine>
where
    Self: Sized,
{
    // ============================================================
    // REQUIRED

    /// Rate of this encoder.
    type Rate: Rate<E>;

    /// Like [`ReedSolomonEncoder::add_original_shard`](crate::ReedSolomonEncoder::add_original_shard).
    fn add_original_shard<T: AsRef<[u8]>>(&mut self, original_shard: T) -> Result<(), Error>;

    /// Like [`ReedSolomonEncoder::encode`](crate::ReedSolomonEncoder::encode).
    fn encode(&mut self) -> Result<EncoderResult<'_>, Error>;

    /// Consumes this encoder returning its [`Engine`] and [`EncoderWork`]
    /// so that they can be re-used by another encoder.
    fn into_parts(self) -> (E, EncoderWork);

    /// Like [`ReedSolomonEncoder::new`](crate::ReedSolomonEncoder::new)
    /// with [`Engine`] to use and optional working space to be re-used.
    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<EncoderWork>,
    ) -> Result<Self, Error>;

    /// Like [`ReedSolomonEncoder::reset`](crate::ReedSolomonEncoder::reset).
    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error>;

    // ============================================================
    // PROVIDED

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// This is same as [`Rate::supports`].
    fn supports(original_count: usize, recovery_count: usize) -> bool {
        Self::Rate::supports(original_count, recovery_count)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    ///
    /// This is same as [`Rate::validate`].
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::Rate::validate(original_count, recovery_count, shard_bytes)
    }
}

// ======================================================================
// RateDecoder - PUBLIC

/// Reed-Solomon decoder using specific rate.
pub trait RateDecoder<E: Engine>
where
    Self: Sized,
{
    // ============================================================
    // REQUIRED

    /// Rate of this decoder.
    type Rate: Rate<E>;

    /// Like [`ReedSolomonDecoder::add_original_shard`](crate::ReedSolomonDecoder::add_original_shard).
    fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        original_shard: T,
    ) -> Result<(), Error>;

    /// Like [`ReedSolomonDecoder::add_recovery_shard`](crate::ReedSolomonDecoder::add_recovery_shard).
    fn add_recovery_shard<T: AsRef<[u8]>>(
        &mut self,
        index: usize,
        recovery_shard: T,
    ) -> Result<(), Error>;

    /// Like [`ReedSolomonDecoder::decode`](crate::ReedSolomonDecoder::decode).
    fn decode(&mut self) -> Result<DecoderResult<'_>, Error>;

    /// Consumes this decoder returning its [`Engine`] and [`DecoderWork`]
    /// so that they can be re-used by another decoder.
    fn into_parts(self) -> (E, DecoderWork);

    /// Like [`ReedSolomonDecoder::new`](crate::ReedSolomonDecoder::new)
    /// with [`Engine`] to use and optional working space to be re-used.
    fn new(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        engine: E,
        work: Option<DecoderWork>,
    ) -> Result<Self, Error>;

    /// Like [`ReedSolomonDecoder::reset`](crate::ReedSolomonDecoder::reset).
    fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error>;

    // ============================================================
    // PROVIDED

    /// Returns `true` if given `original_count` / `recovery_count`
    /// combination is supported.
    ///
    /// This is same as [`Rate::supports`].
    fn supports(original_count: usize, recovery_count: usize) -> bool {
        Self::Rate::supports(original_count, recovery_count)
    }

    /// Returns `Ok(())` if given `original_count` / `recovery_count`
    /// combination is supported and given `shard_bytes` is valid.
    ///
    /// This is same as [`Rate::validate`].
    fn validate(
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
    ) -> Result<(), Error> {
        Self::Rate::validate(original_count, recovery_count, shard_bytes)
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    extern crate alloc;

    use super::{ReceivedShardBits, ReceivedShards};
    use alloc::vec;
    use alloc::vec::Vec;
    use core::ops::Range;
    use fixedbitset::FixedBitSet;

    /// Naive reference implementation of [`ReceivedShards::received_count`].
    fn naive_count(flags: &[bool], range: Range<usize>) -> usize {
        range
            .filter(|index| flags.get(*index) == Some(&true))
            .count()
    }

    /// Naive reference implementation of [`ReceivedShards::missing_in`].
    fn naive_missing(flags: &[bool], range: Range<usize>) -> Vec<usize> {
        range
            .filter(|index| flags.get(*index) != Some(&true))
            .collect()
    }

    /// Checks all [`ReceivedShards`] methods of every representation of `flags` against the naive
    /// reference implementations.
    fn check_all(flags: &[bool]) {
        let mut words = vec![0u64; flags.len().div_ceil(64)];
        let mut set = FixedBitSet::with_capacity(flags.len());
        for (index, flag) in flags.iter().enumerate() {
            if *flag {
                words[index / 64] |= 1 << (index % 64);
                set.insert(index);
            }
        }
        let bits = ReceivedShardBits(words.as_slice());

        let checked_len = flags.len() + 130;

        for index in 0..checked_len {
            let expected = flags.get(index) == Some(&true);
            assert_eq!(flags.received(index), expected, "bools, index {index}");
            assert_eq!(bits.received(index), expected, "bits, index {index}");
            assert_eq!(set.received(index), expected, "set, index {index}");
            assert_eq!((&flags).received(index), expected, "&bools, index {index}");
        }

        for word_index in 0..checked_len.div_ceil(64) {
            let expected: u64 = (0..64)
                .map(|bit| u64::from(flags.get(word_index * 64 + bit) == Some(&true)) << bit)
                .sum();
            assert_eq!(
                flags.received_word(word_index),
                expected,
                "bools, word {word_index}"
            );
            assert_eq!(
                bits.received_word(word_index),
                expected,
                "bits, word {word_index}"
            );
            assert_eq!(
                set.received_word(word_index),
                expected,
                "set, word {word_index}"
            );
        }

        // Word indices which are far out of range must not panic.
        for representation in [
            flags.received_word(usize::MAX),
            bits.received_word(usize::MAX),
            set.received_word(usize::MAX),
        ] {
            assert_eq!(representation, 0);
        }

        let mut ranges = vec![
            0..0,
            0..1,
            5..5,
            0..64,
            0..65,
            63..65,
            64..128,
            3..70,
            1..checked_len,
            0..checked_len,
            checked_len..checked_len + 200,
        ];
        if flags.len() > 2 {
            ranges.push(1..flags.len() - 1);
            ranges.push(0..flags.len());
        }

        for range in ranges {
            let count = naive_count(flags, range.clone());
            assert_eq!(
                flags.received_count(range.clone()),
                count,
                "bools, {range:?}"
            );
            assert_eq!(bits.received_count(range.clone()), count, "bits, {range:?}");
            assert_eq!(set.received_count(range.clone()), count, "set, {range:?}");
            assert_eq!(
                (&flags).received_count(range.clone()),
                count,
                "&bools, {range:?}"
            );

            let missing = naive_missing(flags, range.clone());
            assert_eq!(
                flags.missing_in(range.clone()).collect::<Vec<_>>(),
                missing,
                "bools, {range:?}"
            );
            assert_eq!(
                bits.missing_in(range.clone()).collect::<Vec<_>>(),
                missing,
                "bits, {range:?}"
            );
            assert_eq!(
                set.missing_in(range.clone()).collect::<Vec<_>>(),
                missing,
                "set, {range:?}"
            );
            assert_eq!(
                (&flags).missing_in(range.clone()).collect::<Vec<_>>(),
                missing,
                "&bools, {range:?}"
            );
        }
    }

    #[test]
    fn patterns() {
        // Nothing received.
        check_all(&[false; 200]);
        // Everything received.
        check_all(&[true; 200]);
        // A single missing shard in the middle of a word.
        let mut flags = [true; 200];
        flags[70] = false;
        check_all(&flags);
        // Sparse.
        check_all(&core::array::from_fn::<bool, 200, _>(|index| {
            index % 37 != 5
        }));
        // Dense.
        check_all(&core::array::from_fn::<bool, 200, _>(|index| {
            index % 3 == 0
        }));
        // Word boundaries.
        check_all(&core::array::from_fn::<bool, 200, _>(|index| {
            !matches!(index, 0 | 63 | 64 | 127 | 128 | 191)
        }));
        // Lengths which are not a multiple of 64.
        for len in [0usize, 1, 63, 64, 65, 127, 128, 129] {
            let flags: Vec<bool> = (0..len).map(|index| index % 5 != 1).collect();
            check_all(&flags);
        }
    }

    #[test]
    fn received_count_ranges() {
        let flags: [bool; 130] = core::array::from_fn(|index| index % 2 == 0);

        assert_eq!(flags.received_count(0..0), 0);
        assert_eq!(flags.received_count(0..1), 1);
        assert_eq!(flags.received_count(1..2), 0);
        assert_eq!(flags.received_count(0..64), 32);
        assert_eq!(flags.received_count(0..130), 65);
        // Out of range indices are not received.
        assert_eq!(flags.received_count(0..1000), 65);
        assert_eq!(flags.received_count(130..1000), 0);
        // Empty and reversed ranges.
        assert_eq!(flags.received_count(50..50), 0);
        assert_eq!(flags.received_count(50..10), 0);
        assert_eq!(flags.missing_in(50..10).count(), 0);
    }

    #[test]
    fn missing_in_skips_full_words() {
        // 1000 shards, all received except the last one.
        let mut words = [u64::MAX; 16];
        words[15] = u64::MAX >> (1024 - 999);
        let bits = ReceivedShardBits(words);

        assert_eq!(bits.missing_in(0..999).collect::<Vec<_>>(), Vec::new());
        assert_eq!(bits.missing_in(0..1000).collect::<Vec<_>>(), [999]);
        assert_eq!(bits.received_count(0..1000), 999);
    }

    #[test]
    fn bool_slice() {
        let flags = [true, false, true];

        assert!(flags.received(0));
        assert!(!flags.received(1));
        assert!(flags.received(2));

        // Out of range.
        assert!(!flags.received(3));
        assert!(!flags.received(usize::MAX));

        // Slice and reference give same results.
        let slice: &[bool] = &flags;
        for index in 0..10 {
            assert_eq!(slice.received(index), flags.received(index));
            assert_eq!((&slice).received(index), flags.received(index));
        }

        // Empty.
        let empty: [bool; 0] = [];
        assert!(!empty.received(0));
    }

    #[test]
    fn bits() {
        let bits = ReceivedShardBits([0b1000_0011u64, 0b1 << 63]);

        assert_eq!(bits.as_words(), &[0b1000_0011u64, 0b1 << 63]);

        // Bit order within first word: LSB first.
        assert!(bits.received(0));
        assert!(bits.received(1));
        for index in 2..7 {
            assert!(!bits.received(index));
        }
        assert!(bits.received(7));
        for index in 8..127 {
            assert!(!bits.received(index));
        }

        // Most significant bit of second word is shard 127.
        assert!(bits.received(127));

        // Out of range.
        assert!(!bits.received(128));
        assert!(!bits.received(usize::MAX));

        // Empty.
        assert!(!ReceivedShardBits([0u64; 0]).received(0));

        // Works with slices/vectors too.
        let words = [0u64, 0b100];
        assert!(ReceivedShardBits(words.as_slice()).received(66));
        assert!(ReceivedShardBits(&words[..]).received(66));
    }

    #[test]
    fn bits_match_bools() {
        let flags: [bool; 200] = core::array::from_fn(|index| index % 7 == 3);

        let mut words = [0u64; 4];
        for (index, received) in flags.iter().enumerate() {
            if *received {
                words[index / 64] |= 1 << (index % 64);
            }
        }
        let bits = ReceivedShardBits(words);

        for index in 0..300 {
            assert_eq!(bits.received(index), flags.received(index));
        }
    }

    #[test]
    fn fixedbitset() {
        let mut set = FixedBitSet::with_capacity(100);
        set.insert(0);
        set.insert(99);

        assert!(set.received(0));
        assert!(!set.received(1));
        assert!(set.received(99));

        // Out of range.
        assert!(!set.received(100));
        assert!(!set.received(usize::MAX));

        // Reference.
        assert!((&set).received(0));
        assert!(!(&set).received(100));
    }
}
