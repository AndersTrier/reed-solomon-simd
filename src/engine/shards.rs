#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
use core::ops::{Bound, Index, IndexMut, Range, RangeBounds};

use crate::engine::utils;

// ======================================================================
// Shards - CRATE

pub(crate) struct Shards {
    shard_count: usize,
    // Shard length in 64 byte chunks
    shard_len_64: usize,

    // Flat Vec of `shard_count * shard_len_64 * 64` bytes.
    data: Vec<[u8; 64]>,
}

impl Shards {
    pub(crate) fn as_ref_mut(&mut self) -> ShardsRefMut<'_> {
        ShardsRefMut::new(self.shard_count, self.shard_len_64, self.data.as_mut())
    }

    pub(crate) fn new() -> Self {
        Self {
            shard_count: 0,
            shard_len_64: 0,
            data: Vec::new(),
        }
    }

    pub(crate) fn resize(&mut self, shard_count: usize, shard_len_64: usize) {
        self.shard_count = shard_count;
        self.shard_len_64 = shard_len_64;

        self.data
            .resize(self.shard_count * self.shard_len_64, [0; 64]);
    }

    pub(crate) fn insert(&mut self, index: usize, shard: &[u8]) {
        debug_assert_eq!(shard.len() % 2, 0);

        let whole_chunk_count = shard.len() / 64;
        let tail_len = shard.len() % 64;

        let (src_chunks, src_tail) = shard.split_at(shard.len() - tail_len);

        let dst = &mut self[index];
        dst[..whole_chunk_count]
            .as_flattened_mut()
            .copy_from_slice(src_chunks);

        // Last chunk is special if shard.len() % 64 != 0.
        // See src/algorithm.md for an explanation.
        if tail_len > 0 {
            let (src_lo, src_hi) = src_tail.split_at(tail_len / 2);
            let (dst_lo, dst_hi) = dst[whole_chunk_count].split_at_mut(32);
            dst_lo[..src_lo.len()].copy_from_slice(src_lo);
            dst_hi[..src_hi.len()].copy_from_slice(src_hi);
        }
    }

    // Undoes the encoding of the last chunk for the given range of shards
    pub(crate) fn undo_last_chunk_encoding(&mut self, shard_bytes: usize, range: Range<usize>) {
        let whole_chunk_count = shard_bytes / 64;
        let tail_len = shard_bytes % 64;

        if tail_len == 0 {
            return;
        }

        for idx in range {
            let last_chunk = &mut self[idx][whole_chunk_count];
            last_chunk.copy_within(32..32 + tail_len / 2, tail_len / 2);
        }
    }
}

// ======================================================================
// Shards - IMPL Index

impl Index<usize> for Shards {
    type Output = [[u8; 64]];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.shard_len_64..(index + 1) * self.shard_len_64]
    }
}

// ======================================================================
// Shards - IMPL IndexMut

impl IndexMut<usize> for Shards {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.shard_len_64..(index + 1) * self.shard_len_64]
    }
}

/// Storage of the shards that the [`Engine`] algorithms work on.
///
/// A shard storage holds a fixed number of shards, all of which have the same length in 64 byte
/// chunks. Shards are addressed by index `0 .. len()`, and it is up to the implementation where the
/// individual shards live in memory: [`ShardsRefMut`] keeps them consecutively in a single flat
/// buffer, but an implementation could equally well keep every shard in a separate allocation.
///
/// The methods which hand out multiple mutable references at once, as well as those which move data
/// between shard ranges, are `unsafe` **to call**: instead of checking in every iteration of the
/// hot loops that the given indices don't overlap, the obligation to only pass non-overlapping
/// indices is placed on the caller.
///
/// # Safety
///
/// This trait is `unsafe` to implement, because unsafe code in this crate relies on the following
/// guarantees:
/// - [`ShardStorage::len()`]() returns the same value for as long as the storage is borrowed
/// - indexing with any `index < len()` succeeds, and all shards have the same length
/// - shards do not overlap: different indices refer to disjoint memory
/// - [`ShardStorage::dist2_mut()`] and [`ShardStorage::dist4_mut()`] return exactly the shards that
///   the given indices refer to, in the given order
///
/// [`Engine`]: crate::engine::Engine
pub unsafe trait ShardStorage: Index<usize, Output = [[u8; 64]]> + IndexMut<usize> {
    /// Returns number of shards
    fn len(&self) -> usize;

    /// Returns `true` if this contains no shards
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns mutable references to shards at `pos` and `pos + dist`.
    ///
    /// See source code of [`Naive::fft`] for an example.
    ///
    /// # Safety
    ///
    /// `dist` must be non-zero and `pos + dist` must be less than
    /// [`len()`](ShardStorage::len).
    ///
    /// [`Naive::fft`]: crate::engine::Naive#method.fft
    unsafe fn dist2_mut(&mut self, pos: usize, dist: usize) -> (&mut [[u8; 64]], &mut [[u8; 64]]);

    /// Returns mutable references to shards at
    /// `pos`, `pos + dist`, `pos + dist * 2` and `pos + dist * 3`.
    ///
    /// See source code of [`NoSimd::fft`] for an example (specifically the private method
    /// `fft_butterfly_two_layers`).
    ///
    /// # Safety
    ///
    /// `dist` must be non-zero and `pos + dist * 3` must be less than
    /// [`len()`](ShardStorage::len).
    ///
    /// [`NoSimd::fft`]: crate::engine::NoSimd#method.fft
    #[allow(clippy::type_complexity)]
    unsafe fn dist4_mut(
        &mut self,
        pos: usize,
        dist: usize,
    ) -> (
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
    );

    /// Fills the given shard-range with `0u8`:s.
    ///
    /// # Panics
    ///
    /// If the range is out of bounds.
    fn zero(&mut self, range: impl RangeBounds<usize>);

    /// `self[x .. x + count] ^= self[y .. y + count]`
    ///
    /// The provided implementation does this shard by shard. Implementations which store
    /// consecutive shards consecutively in memory can override this with a single pass over both
    /// ranges.
    ///
    /// # Safety
    ///
    /// Shard-ranges `x .. x + count` and `y .. y + count` must be within [`ShardStorage::len()`]
    /// and must not overlap.
    unsafe fn xor_within(&mut self, x: usize, y: usize, count: usize) {
        for i in 0..count {
            // SAFETY: The ranges don't overlap and are within bounds, so `dist`
            // is non-zero and both shards exist.
            let (xs, ys) = unsafe {
                if x < y {
                    self.dist2_mut(x + i, y - x)
                } else {
                    let (ys, xs) = self.dist2_mut(y + i, x - y);
                    (xs, ys)
                }
            };
            utils::xor(xs, ys);
        }
    }

    /// `self[dest .. dest + count] = self[src .. src + count]`
    ///
    /// The provided implementation does this shard by shard. Implementations which store
    /// consecutive shards consecutively in memory can override this with a single copy.
    ///
    /// # Safety
    ///
    /// Shard-ranges `src .. src + count` and `dest .. dest + count` must be within
    /// [`ShardStorage::len()`] and must not overlap.
    unsafe fn copy_within(&mut self, src: usize, dest: usize, count: usize) {
        for i in 0..count {
            // SAFETY: The ranges don't overlap and are within bounds, so `dist`
            // is non-zero and both shards exist.
            unsafe {
                if src < dest {
                    let (from, to) = self.dist2_mut(src + i, dest - src);
                    to.copy_from_slice(from);
                } else {
                    let (to, from) = self.dist2_mut(dest + i, src - dest);
                    to.copy_from_slice(from);
                }
            }
        }
    }
}

/// Mutable reference to a shard array.
pub struct ShardsRefMut<'a> {
    shard_count: usize,
    shard_len_64: usize,

    data: &'a mut [[u8; 64]],
}

impl<'a> ShardsRefMut<'a> {
    /// Creates a new [`ShardsRefMut`] that references given `data`.
    ///
    /// # Panics
    ///
    /// If `data.len() < shard_count * shard_len_64`.
    pub fn new(shard_count: usize, shard_len_64: usize, data: &'a mut [[u8; 64]]) -> Self {
        assert!(data.len() >= shard_count * shard_len_64);

        Self {
            shard_count,
            shard_len_64,
            data: &mut data[..shard_count * shard_len_64],
        }
    }

    /// Splits this [`ShardsRefMut`] into two so that the first includes shards `0..mid` and the
    /// second includes shards `mid..`.
    pub fn split_at_mut(&mut self, mid: usize) -> (ShardsRefMut<'_>, ShardsRefMut<'_>) {
        let (a, b) = self.data.split_at_mut(mid * self.shard_len_64);

        (
            ShardsRefMut::new(mid, self.shard_len_64, a),
            ShardsRefMut::new(self.shard_count - mid, self.shard_len_64, b),
        )
    }

    // Returns mutable references to flat-arrays of shard-ranges
    // `x .. x + count` and `y .. y + count`.
    //
    // Ranges must not overlap.
    fn flat2_mut(
        &mut self,
        mut x: usize,
        mut y: usize,
        mut count: usize,
    ) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
        x *= self.shard_len_64;
        y *= self.shard_len_64;
        count *= self.shard_len_64;

        if x < y {
            let (head, tail) = self.data.split_at_mut(y);
            (&mut head[x..x + count], &mut tail[..count])
        } else {
            let (head, tail) = self.data.split_at_mut(x);
            (&mut tail[..count], &mut head[y..y + count])
        }
    }
}

// SAFETY: Shards are stored consecutively in a flat buffer at a fixed stride of `shard_len_64`
// chunks, hence distinct indices refer to distinct shards of equal length, and
// `dist2_mut()`/`dist4_mut()` return the requested shards.
unsafe impl ShardStorage for ShardsRefMut<'_> {
    fn len(&self) -> usize {
        self.shard_count
    }

    fn is_empty(&self) -> bool {
        self.shard_count == 0
    }

    unsafe fn dist2_mut(
        &mut self,
        mut pos: usize,
        mut dist: usize,
    ) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
        pos *= self.shard_len_64;
        dist *= self.shard_len_64;

        let (a, b) = self.data[pos..].split_at_mut(dist);
        (&mut a[..self.shard_len_64], &mut b[..self.shard_len_64])
    }

    unsafe fn dist4_mut(
        &mut self,
        mut pos: usize,
        mut dist: usize,
    ) -> (
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
    ) {
        pos *= self.shard_len_64;
        dist *= self.shard_len_64;

        let (ab, cd) = self.data[pos..].split_at_mut(dist * 2);
        let (a, b) = ab.split_at_mut(dist);
        let (c, d) = cd.split_at_mut(dist);

        (
            &mut a[..self.shard_len_64],
            &mut b[..self.shard_len_64],
            &mut c[..self.shard_len_64],
            &mut d[..self.shard_len_64],
        )
    }

    fn zero(&mut self, range: impl RangeBounds<usize>) {
        let start = match range.start_bound() {
            Bound::Included(start) => start * self.shard_len_64,
            Bound::Excluded(start) => (start + 1) * self.shard_len_64,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end) => (end + 1) * self.shard_len_64,
            Bound::Excluded(end) => end * self.shard_len_64,
            Bound::Unbounded => self.shard_count * self.shard_len_64,
        };

        self.data[start..end].fill([0; 64]);
    }

    unsafe fn xor_within(&mut self, x: usize, y: usize, count: usize) {
        let (xs, ys) = self.flat2_mut(x, y, count);
        utils::xor(xs, ys);
    }

    unsafe fn copy_within(&mut self, mut src: usize, mut dest: usize, mut count: usize) {
        src *= self.shard_len_64;
        dest *= self.shard_len_64;
        count *= self.shard_len_64;

        self.data.copy_within(src..src + count, dest);
    }
}

// ======================================================================
// ShardsRefMut - IMPL Index

impl Index<usize> for ShardsRefMut<'_> {
    type Output = [[u8; 64]];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.shard_len_64..(index + 1) * self.shard_len_64]
    }
}

// ======================================================================
// ShardsRefMut - IMPL IndexMut

impl IndexMut<usize> for ShardsRefMut<'_> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.shard_len_64..(index + 1) * self.shard_len_64]
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::{self, Engine, NoSimd, ShardStorage, ShardsRefMut};
    #[cfg(not(feature = "std"))]
    use alloc::{vec, vec::Vec};
    use core::ops::{Bound, Index, IndexMut, RangeBounds};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    const SHARD_COUNT: usize = 32;
    const SHARD_LEN_64: usize = 3;
    // Number of unused chunks between two consecutive shards
    const PADDING: usize = 2;

    /// [`ShardStorage`] which stores shards with padding in between, meaning that its layout is
    /// deliberately not the flat layout of [`ShardsRefMut`]
    struct PaddedShards {
        data: Vec<[u8; 64]>,
    }

    impl PaddedShards {
        fn new(shards: &[[u8; 64]]) -> Self {
            let data = vec![[0; 64]; SHARD_COUNT * (SHARD_LEN_64 + PADDING)];
            let mut this = Self { data };

            for index in 0..SHARD_COUNT {
                this[index].copy_from_slice(&shards[index * SHARD_LEN_64..][..SHARD_LEN_64]);
            }

            this
        }

        // Contents of all shards, without the padding in between
        fn shards(&self) -> Vec<[u8; 64]> {
            (0..SHARD_COUNT)
                .flat_map(|index| self[index].to_vec())
                .collect()
        }

        // Contents of the padding between the shards, which no operation is ever allowed to touch
        fn padding(&self) -> Vec<[u8; 64]> {
            (0..SHARD_COUNT)
                .flat_map(|index| {
                    self.data[index * (SHARD_LEN_64 + PADDING) + SHARD_LEN_64..][..PADDING].to_vec()
                })
                .collect()
        }
    }

    impl Index<usize> for PaddedShards {
        type Output = [[u8; 64]];
        fn index(&self, index: usize) -> &Self::Output {
            &self.data[index * (SHARD_LEN_64 + PADDING)..][..SHARD_LEN_64]
        }
    }

    impl IndexMut<usize> for PaddedShards {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[index * (SHARD_LEN_64 + PADDING)..][..SHARD_LEN_64]
        }
    }

    // SAFETY: Shards are stored at a fixed stride, hence distinct indices refer to distinct shards
    // of equal length, and `dist2_mut()`/`dist4_mut()` return the requested shards
    unsafe impl ShardStorage for PaddedShards {
        fn len(&self) -> usize {
            SHARD_COUNT
        }

        unsafe fn dist2_mut(
            &mut self,
            pos: usize,
            dist: usize,
        ) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
            let stride = SHARD_LEN_64 + PADDING;
            let (head, tail) = self.data.split_at_mut((pos + dist) * stride);

            (
                &mut head[pos * stride..][..SHARD_LEN_64],
                &mut tail[..SHARD_LEN_64],
            )
        }

        unsafe fn dist4_mut(
            &mut self,
            pos: usize,
            dist: usize,
        ) -> (
            &mut [[u8; 64]],
            &mut [[u8; 64]],
            &mut [[u8; 64]],
            &mut [[u8; 64]],
        ) {
            let stride = SHARD_LEN_64 + PADDING;
            let (head, tail) = self.data.split_at_mut((pos + dist * 2) * stride);
            let (a, b) = head.split_at_mut((pos + dist) * stride);
            let (c, d) = tail.split_at_mut(dist * stride);

            (
                &mut a[pos * stride..][..SHARD_LEN_64],
                &mut b[..SHARD_LEN_64],
                &mut c[..SHARD_LEN_64],
                &mut d[..SHARD_LEN_64],
            )
        }

        fn zero(&mut self, range: impl RangeBounds<usize>) {
            let start = match range.start_bound() {
                Bound::Included(start) => *start,
                Bound::Excluded(start) => start + 1,
                Bound::Unbounded => 0,
            };

            let end = match range.end_bound() {
                Bound::Included(end) => end + 1,
                Bound::Excluded(end) => *end,
                Bound::Unbounded => SHARD_COUNT,
            };

            for index in start..end {
                self[index].fill([0; 64]);
            }
        }
    }

    // Runs the same sequence of operations on both storages
    fn exercise<S: ShardStorage>(engine: &NoSimd, data: &mut S) {
        engine.ifft(data, 0, SHARD_COUNT, SHARD_COUNT, 0);
        // SAFETY: Shard-ranges `0 .. 8` and `8 .. 16` are within bounds and don't overlap
        unsafe { data.copy_within(0, 8, 8) };
        // SAFETY: Shard-ranges `16 .. 24` and `0 .. 8` are within bounds and don't overlap
        unsafe { engine::xor_within(data, 16, 0, 8) };
        data.zero(24..28);
        engine::formal_derivative(data);
        engine.fft(data, 0, SHARD_COUNT, SHARD_COUNT, 0);
        engine.mul(&mut data[3], 12345);
    }

    // Verifies that the algorithms produce the same result with a shard storage which is not laid
    // out flat in memory
    #[test]
    fn padded_shards_match_flat_shards() {
        let engine = NoSimd::new();

        let mut rng = ChaCha8Rng::from_seed([0; 32]);
        let mut flat_data = vec![[0; 64]; SHARD_COUNT * SHARD_LEN_64];
        rng.fill(flat_data.as_mut_slice().as_flattened_mut());

        let mut padded = PaddedShards::new(&flat_data);
        let mut flat = ShardsRefMut::new(SHARD_COUNT, SHARD_LEN_64, &mut flat_data);

        assert_eq!(flat.len(), padded.len());
        assert_eq!(padded.shards(), flat_data_of(&mut flat));

        exercise(&engine, &mut flat);
        exercise(&engine, &mut padded);

        assert_eq!(padded.shards(), flat_data_of(&mut flat));
        assert_eq!(padded.padding(), vec![[0; 64]; SHARD_COUNT * PADDING]);
    }

    fn flat_data_of(shards: &mut ShardsRefMut<'_>) -> Vec<[u8; 64]> {
        (0..shards.len())
            .flat_map(|index| shards[index].to_vec())
            .collect()
    }
}
