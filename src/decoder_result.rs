use crate::rate::DecoderWork;

// ======================================================================
// DecoderResult - PUBLIC

/// Result of decoding. Contains the restored original shards.
///
/// This struct is created by [`ReedSolomonDecoder::decode`]
/// and [`RateDecoder::decode`].
///
/// [`RateDecoder::decode`]: crate::rate::RateDecoder::decode
/// [`ReedSolomonDecoder::decode`]: crate::ReedSolomonDecoder::decode
pub struct DecoderResult<'a> {
    work: &'a mut DecoderWork,
}

impl DecoderResult<'_> {
    /// Returns restored original shard with given `index`
    /// or `None` if given `index` doesn't correspond to
    /// a missing original shard.
    pub fn restored_original(&self, index: usize) -> Option<&[u8]> {
        self.work.restored_original(index)
    }

    /// Returns iterator over all restored original shards
    /// and their indexes, ordered by indexes.
    pub fn restored_original_iter(&self) -> RestoredOriginal {
        RestoredOriginal::new(self.work)
    }
}

// ======================================================================
// DecoderResult - CRATE

impl<'a> DecoderResult<'a> {
    pub(crate) fn new(work: &'a mut DecoderWork) -> Self {
        Self { work }
    }
}

// ======================================================================
// DecoderResult - IMPL DROP

impl Drop for DecoderResult<'_> {
    fn drop(&mut self) {
        self.work.reset_received();
    }
}

// ======================================================================
// RestoredOriginal - PUBLIC

/// Iterator over restored original shards and their indexes.
///
/// This struct is created by [`DecoderResult::restored_original_iter`].
pub struct RestoredOriginal<'a> {
    remaining: usize,
    next_index: usize,
    work: &'a DecoderWork,
}

// ======================================================================
// RestoredOriginal - IMPL Iterator

impl<'a> Iterator for RestoredOriginal<'a> {
    type Item = (usize, &'a [u8]);
    fn next(&mut self) -> Option<(usize, &'a [u8])> {
        if self.remaining == 0 {
            return None;
        }

        let mut index = self.next_index;
        while index < self.work.original_count() {
            if let Some(original) = self.work.restored_original(index) {
                self.next_index = index + 1;
                self.remaining -= 1;
                return Some((index, original));
            }
            index += 1;
        }

        debug_assert!(
            false,
            "Inconsistency in internal data structures. Please report."
        );

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

// ======================================================================
// RestoredOriginal - IMPL ExactSizeIterator

impl ExactSizeIterator for RestoredOriginal<'_> {}

// ======================================================================
// RestoredOriginal - CRATE

impl<'a> RestoredOriginal<'a> {
    pub(crate) fn new(work: &'a DecoderWork) -> Self {
        Self {
            remaining: work.missing_original_count(),
            next_index: 0,
            work,
        }
    }
}

// ======================================================================
// TESTS

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{test_util, ReedSolomonDecoder, ReedSolomonEncoder};

    #[cfg(not(feature = "std"))]
    use alloc::vec::Vec;

    fn simple_roundtrip(shard_size: usize) {
        let original = test_util::generate_original(3, shard_size, 0);

        let mut encoder = ReedSolomonEncoder::new(3, 2, shard_size).unwrap();
        let mut decoder = ReedSolomonDecoder::new(3, 2, shard_size).unwrap();

        for original in &original {
            encoder.add_original_shard(original).unwrap();
        }

        let result = encoder.encode().unwrap();
        let recovery: Vec<_> = result.recovery_iter().collect();

        assert!(recovery.iter().all(|slice| slice.len() == shard_size));

        decoder.add_original_shard(1, &original[1]).unwrap();
        decoder.add_recovery_shard(0, recovery[0]).unwrap();
        decoder.add_recovery_shard(1, recovery[1]).unwrap();

        let result: DecoderResult = decoder.decode().unwrap();

        assert_eq!(result.restored_original(0).unwrap(), original[0]);
        assert!(result.restored_original(1).is_none());
        assert_eq!(result.restored_original(2).unwrap(), original[2]);
        assert!(result.restored_original(3).is_none());

        let mut iter: RestoredOriginal = result.restored_original_iter();
        assert_eq!(iter.next(), Some((0, original[0].as_slice())));
        assert_eq!(iter.next(), Some((2, original[2].as_slice())));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }

    #[test]
    // DecoderResult::restored_original
    // DecoderResult::restored_original_iter
    // RestoredOriginal
    fn decoder_result() {
        simple_roundtrip(1024);
    }

    #[test]
    fn shard_size_not_divisible_by_64() {
        for shard_size in [2, 4, 6, 30, 32, 34, 62, 64, 66, 126, 128, 130] {
            simple_roundtrip(shard_size);
        }
    }

    #[test]
    fn decoder_result_size_hint() {
        let shard_size = 64;
        let original = test_util::generate_original(3, shard_size, 0);

        let mut encoder = ReedSolomonEncoder::new(3, 2, shard_size).unwrap();
        let mut decoder = ReedSolomonDecoder::new(3, 2, shard_size).unwrap();

        for original in &original {
            encoder.add_original_shard(original).unwrap();
        }

        let result = encoder.encode().unwrap();
        let recovery: Vec<_> = result.recovery_iter().collect();

        decoder.add_original_shard(1, &original[1]).unwrap();
        decoder.add_recovery_shard(0, recovery[0]).unwrap();
        decoder.add_recovery_shard(1, recovery[1]).unwrap();

        let result: DecoderResult = decoder.decode().unwrap();

        let mut iter: RestoredOriginal = result.restored_original_iter();

        assert_eq!(iter.len(), 2);

        assert!(iter.next().is_some());
        assert_eq!(iter.len(), 1);

        assert!(iter.next().is_some());
        assert_eq!(iter.len(), 0);

        assert!(iter.next().is_none());
        assert_eq!(iter.len(), 0);
    }
}
