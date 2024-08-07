use crate::{
    engine::{Shards, ShardsRefMut},
    Error,
};

// ======================================================================
// EncoderWork - PUBLIC

/// Working space for [`RateEncoder`].
///
/// [`RateEncoder`]: crate::rate::RateEncoder
pub struct EncoderWork {
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,

    original_received_count: usize,
    shards: Shards,
}

impl EncoderWork {
    /// Creates new [`EncoderWork`] which initially
    /// has no working space allocated.
    pub fn new() -> Self {
        Self {
            original_count: 0,
            recovery_count: 0,
            shard_bytes: 0,

            original_received_count: 0,
            shards: Shards::new(),
        }
    }
}

// ======================================================================
// EncoderWork - IMPL Default

impl Default for EncoderWork {
    fn default() -> Self {
        Self::new()
    }
}

// ======================================================================
// EncoderWork - CRATE

impl EncoderWork {
    pub(crate) fn add_original_shard<T: AsRef<[u8]>>(
        &mut self,
        original_shard: T,
    ) -> Result<(), Error> {
        let original_shard = original_shard.as_ref();

        if self.original_received_count == self.original_count {
            Err(Error::TooManyOriginalShards {
                original_count: self.original_count,
            })
        } else if original_shard.len() != self.shard_bytes {
            Err(Error::DifferentShardSize {
                shard_bytes: self.shard_bytes,
                got: original_shard.len(),
            })
        } else {
            self.shards[self.original_received_count]
                .as_flattened_mut()
                .copy_from_slice(original_shard);
            self.original_received_count += 1;
            Ok(())
        }
    }

    pub(crate) fn encode_begin(&mut self) -> Result<(ShardsRefMut, usize, usize), Error> {
        if self.original_received_count != self.original_count {
            Err(Error::TooFewOriginalShards {
                original_count: self.original_count,
                original_received_count: self.original_received_count,
            })
        } else {
            Ok((
                self.shards.as_ref_mut(),
                self.original_count,
                self.recovery_count,
            ))
        }
    }

    // This must only be called by `EncoderResult`.
    pub(crate) fn recovery(&self, index: usize) -> Option<&[u8]> {
        if index < self.recovery_count {
            Some(self.shards[index].as_flattened())
        } else {
            None
        }
    }

    pub(crate) fn reset(
        &mut self,
        original_count: usize,
        recovery_count: usize,
        shard_bytes: usize,
        work_count: usize,
    ) {
        self.original_count = original_count;
        self.recovery_count = recovery_count;
        self.shard_bytes = shard_bytes;

        self.original_received_count = 0;
        self.shards.resize(work_count, shard_bytes);
    }

    pub(crate) fn reset_received(&mut self) {
        self.original_received_count = 0;
    }
}
