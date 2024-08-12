use std::ops::{Bound, Index, IndexMut, RangeBounds};

use crate::engine::utils;

// ======================================================================
// Shards - CRATE

pub(crate) struct Shards {
    shard_count: usize,
    shard_bytes: usize,

    // Flat array of `shard_count * shard_bytes` bytes.
    data: Vec<[u8; 64]>,
}

impl Shards {
    pub(crate) fn as_ref_mut(&mut self) -> ShardsRefMut {
        ShardsRefMut::new(self.shard_count, self.shard_bytes, self.data.as_mut())
    }

    pub(crate) fn new() -> Self {
        Self {
            shard_count: 0,
            shard_bytes: 0,
            data: Vec::new(),
        }
    }

    pub(crate) fn resize(&mut self, shard_count: usize, shard_bytes: usize) {
        assert!(shard_bytes > 0 && shard_bytes & 63 == 0);

        self.shard_count = shard_count;
        self.shard_bytes = shard_bytes;

        self.data.resize(shard_count * (shard_bytes / 64), [0; 64]);
    }
}

// ======================================================================
// Shards - IMPL Index

impl Index<usize> for Shards {
    type Output = [[u8; 64]];
    fn index(&self, index: usize) -> &Self::Output {
        let shard_chunk_count = self.shard_bytes / 64;
        &self.data[index * shard_chunk_count..(index + 1) * shard_chunk_count]
    }
}

// ======================================================================
// Shards - IMPL IndexMut

impl IndexMut<usize> for Shards {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let shard_chunk_count = self.shard_bytes / 64;
        &mut self.data[index * shard_chunk_count..(index + 1) * shard_chunk_count]
    }
}

// ======================================================================
// ShardsRefMut - PUBLIC

/// Mutable reference to shard array implemented as flat byte array.
pub struct ShardsRefMut<'a> {
    shard_count: usize,
    shard_bytes: usize,

    // Flat array of `shard_count * shard_bytes` bytes.
    data: &'a mut [[u8; 64]],
}

impl<'a> ShardsRefMut<'a> {
    /// Returns mutable references to shards at `pos` and `pos + dist`.
    ///
    /// See source code of [`Naive::fft`] for an example.
    ///
    /// # Panics
    ///
    /// If `dist` is `0`.
    ///
    /// [`Naive::fft`]: crate::engine::Naive#method.fft
    pub fn dist2_mut(
        &mut self,
        mut pos: usize,
        mut dist: usize,
    ) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
        let shard_chunk_count = self.shard_bytes / 64;

        pos *= shard_chunk_count;
        dist *= shard_chunk_count;

        let (a, b) = self.data[pos..].split_at_mut(dist);
        (&mut a[..shard_chunk_count], &mut b[..shard_chunk_count])
    }

    /// Returns mutable references to shards at
    /// `pos`, `pos + dist`, `pos + dist * 2` and `pos + dist * 3`.
    ///
    /// See source code of [`NoSimd::fft`] for an example
    /// (specifically the private method `fft_butterfly_two_layers`).
    ///
    /// # Panics
    ///
    /// If `dist` is `0`.
    ///
    /// [`NoSimd::fft`]: crate::engine::NoSimd#method.fft
    #[allow(clippy::type_complexity)]
    pub fn dist4_mut(
        &mut self,
        mut pos: usize,
        mut dist: usize,
    ) -> (
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
        &mut [[u8; 64]],
    ) {
        let shard_chunk_count = self.shard_bytes / 64;

        pos *= shard_chunk_count;
        dist *= shard_chunk_count;

        let (ab, cd) = self.data[pos..].split_at_mut(dist * 2);
        let (a, b) = ab.split_at_mut(dist);
        let (c, d) = cd.split_at_mut(dist);

        (
            &mut a[..shard_chunk_count],
            &mut b[..shard_chunk_count],
            &mut c[..shard_chunk_count],
            &mut d[..shard_chunk_count],
        )
    }

    /// Returns `true` if this contains no shards.
    pub fn is_empty(&self) -> bool {
        self.shard_count == 0
    }

    /// Returns number of shards.
    pub fn len(&self) -> usize {
        self.shard_count
    }

    /// Creates new [`ShardsRefMut`] that references given `data`.
    ///
    /// # Panics
    ///
    /// If `data` is smaller than `shard_count * shard_bytes` bytes.
    pub fn new(shard_count: usize, shard_bytes: usize, data: &'a mut [[u8; 64]]) -> Self {
        Self {
            shard_count,
            shard_bytes,
            data: &mut data[..shard_count * (shard_bytes / 64)],
        }
    }

    /// Splits this [`ShardsRefMut`] into two so that
    /// first includes shards `0..mid` and second includes shards `mid..`.
    pub fn split_at_mut(&mut self, mid: usize) -> (ShardsRefMut, ShardsRefMut) {
        let shard_chunk_count = self.shard_bytes / 64;

        let (a, b) = self.data.split_at_mut(mid * shard_chunk_count);

        (
            ShardsRefMut::new(mid, self.shard_bytes, a),
            ShardsRefMut::new(self.shard_count - mid, self.shard_bytes, b),
        )
    }

    /// Fills the given shard-range with `0u8`:s.
    pub fn zero<R: RangeBounds<usize>>(&mut self, range: R) {
        let shard_chunk_count = self.shard_bytes / 64;

        let start = match range.start_bound() {
            Bound::Included(start) => start * shard_chunk_count,
            Bound::Excluded(start) => (start + 1) * shard_chunk_count,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end) => (end + 1) * shard_chunk_count,
            Bound::Excluded(end) => end * shard_chunk_count,
            Bound::Unbounded => self.shard_count * shard_chunk_count,
        };

        self.data[start..end].fill([0; 64]);
    }
}

// ======================================================================
// ShardsRefMut - IMPL Index

impl<'a> Index<usize> for ShardsRefMut<'a> {
    type Output = [[u8; 64]];
    fn index(&self, index: usize) -> &Self::Output {
        let shard_chunk_count = self.shard_bytes / 64;
        &self.data[index * shard_chunk_count..(index + 1) * shard_chunk_count]
    }
}

// ======================================================================
// ShardsRefMut - IMPL IndexMut

impl<'a> IndexMut<usize> for ShardsRefMut<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let shard_chunk_count = self.shard_bytes / 64;
        &mut self.data[index * shard_chunk_count..(index + 1) * shard_chunk_count]
    }
}

// ======================================================================
// ShardsRefMut - CRATE

impl<'a> ShardsRefMut<'a> {
    pub(crate) fn copy_within(&mut self, mut src: usize, mut dest: usize, mut count: usize) {
        let shard_chunk_count = self.shard_bytes / 64;

        src *= shard_chunk_count;
        dest *= shard_chunk_count;
        count *= shard_chunk_count;

        self.data.copy_within(src..src + count, dest);
    }

    // Returns mutable references to flat-arrays of shard-ranges
    // `x .. x + count` and `y .. y + count`.
    //
    // Ranges must not overlap.
    pub(crate) fn flat2_mut(
        &mut self,
        mut x: usize,
        mut y: usize,
        mut count: usize,
    ) -> (&mut [[u8; 64]], &mut [[u8; 64]]) {
        let shard_chunk_count = self.shard_bytes / 64;

        x *= shard_chunk_count;
        y *= shard_chunk_count;
        count *= shard_chunk_count;

        if x < y {
            let (head, tail) = self.data.split_at_mut(y);
            (&mut head[x..x + count], &mut tail[..count])
        } else {
            let (head, tail) = self.data.split_at_mut(x);
            (&mut tail[..count], &mut head[y..y + count])
        }
    }

    /// Formal derivative.
    pub(crate) fn formal_derivative(&mut self) {
        for i in 1..self.len() {
            let width: usize = 1 << i.trailing_zeros();
            self.xor_within(i - width, i, width);
        }
    }

    /// `data[x .. x + count] ^= data[y .. y + count]`
    ///
    /// Ranges must not overlap.
    #[inline(always)]
    pub(crate) fn xor_within(&mut self, x: usize, y: usize, count: usize) {
        let (xs, ys) = self.flat2_mut(x, y, count);
        utils::xor(xs, ys);
    }
}
