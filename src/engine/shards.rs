use std::ops::{Bound, Index, IndexMut, RangeBounds};

// ======================================================================
// Shards - CRATE

pub(crate) struct Shards {
    shard_count: usize,
    shard_bytes: usize,

    // Flat array of `shard_count * shard_bytes` bytes.
    pub(crate) data: Vec<u8>,
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

        self.data.resize(shard_count * shard_bytes, 0);
    }
}

// ======================================================================
// Shards - IMPL Index

impl Index<usize> for Shards {
    type Output = [u8];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.shard_bytes..(index + 1) * self.shard_bytes]
    }
}

// ======================================================================
// Shards - IMPL IndexMut

impl IndexMut<usize> for Shards {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.shard_bytes..(index + 1) * self.shard_bytes]
    }
}

// ======================================================================
// ShardsRefMut - PUBLIC

/// Mutable reference to shard array implemented as flat byte array.
pub struct ShardsRefMut<'a> {
    shard_count: usize,
    shard_bytes: usize,

    // Flat array of `shard_count * shard_bytes` bytes.
    data: &'a mut [u8],
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
    pub fn dist2_mut(&mut self, mut pos: usize, mut dist: usize) -> (&mut [u8], &mut [u8]) {
        pos *= self.shard_bytes;
        dist *= self.shard_bytes;

        let (a, b) = self.data[pos..].split_at_mut(dist);
        (&mut a[..self.shard_bytes], &mut b[..self.shard_bytes])
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
    pub fn dist4_mut(
        &mut self,
        mut pos: usize,
        mut dist: usize,
    ) -> (&mut [u8], &mut [u8], &mut [u8], &mut [u8]) {
        pos *= self.shard_bytes;
        dist *= self.shard_bytes;

        let (ab, cd) = self.data[pos..].split_at_mut(dist * 2);
        let (a, b) = ab.split_at_mut(dist);
        let (c, d) = cd.split_at_mut(dist);

        (
            &mut a[..self.shard_bytes],
            &mut b[..self.shard_bytes],
            &mut c[..self.shard_bytes],
            &mut d[..self.shard_bytes],
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
    pub fn new(shard_count: usize, shard_bytes: usize, data: &'a mut [u8]) -> Self {
        Self {
            shard_count,
            shard_bytes,
            data: &mut data[..shard_count * shard_bytes],
        }
    }

    /// Splits this [`ShardsRefMut`] into two so that
    /// first includes shards `0..mid` and second includes shards `mid..`.
    pub fn split_at_mut(&mut self, mid: usize) -> (ShardsRefMut, ShardsRefMut) {
        let (a, b) = self.data.split_at_mut(mid * self.shard_bytes);
        (
            ShardsRefMut::new(mid, self.shard_bytes, a),
            ShardsRefMut::new(self.shard_count - mid, self.shard_bytes, b),
        )
    }

    /// Fills the given shard-range with `0u8`:s.
    pub fn zero<R: RangeBounds<usize>>(&mut self, range: R) {
        let start = match range.start_bound() {
            Bound::Included(start) => start * self.shard_bytes,
            Bound::Excluded(start) => (start + 1) * self.shard_bytes,
            Bound::Unbounded => 0,
        };

        let end = match range.end_bound() {
            Bound::Included(end) => (end + 1) * self.shard_bytes,
            Bound::Excluded(end) => end * self.shard_bytes,
            Bound::Unbounded => self.shard_count * self.shard_bytes,
        };

        self.data[start..end].fill(0);
    }
}

// ======================================================================
// ShardsRefMut - IMPL Index

impl<'a> Index<usize> for ShardsRefMut<'a> {
    type Output = [u8];
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index * self.shard_bytes..(index + 1) * self.shard_bytes]
    }
}

// ======================================================================
// ShardsRefMut - IMPL IndexMut

impl<'a> IndexMut<usize> for ShardsRefMut<'a> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index * self.shard_bytes..(index + 1) * self.shard_bytes]
    }
}

// ======================================================================
// ShardsRefMut - CRATE

impl<'a> ShardsRefMut<'a> {
    pub(crate) fn copy_within(&mut self, mut src: usize, mut dest: usize, mut count: usize) {
        src *= self.shard_bytes;
        dest *= self.shard_bytes;
        count *= self.shard_bytes;

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
    ) -> (&mut [u8], &mut [u8]) {
        x *= self.shard_bytes;
        y *= self.shard_bytes;
        count *= self.shard_bytes;

        if x < y {
            let (head, tail) = self.data.split_at_mut(y);
            (&mut head[x..x + count], &mut tail[..count])
        } else {
            let (head, tail) = self.data.split_at_mut(x);
            (&mut tail[..count], &mut head[y..y + count])
        }
    }
}
