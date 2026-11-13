//! A fixed compensation delay using the mixer's existing delay implementation.
use crate::MAX_CHANNELS;
use crate::strip::{DelayLine, MAX_LATENCY};

/// Allocate on the control thread, then process without changing the layout.
/// Useful for a hosted effect's dry fallback at its prepared algorithmic latency.
pub struct PreparedDelay {
    inner: DelayLine,
}

impl PreparedDelay {
    pub fn new(samples: usize, channels: usize) -> Result<Self, String> {
        if samples > MAX_LATENCY || channels == 0 || channels > MAX_CHANNELS {
            return Err("compensation delay or channel count exceeds the supported budget".into());
        }
        let mut inner = DelayLine::default();
        inner.set(samples, channels);
        Ok(Self { inner })
    }

    /// Processes the prepared channels and the available portion of `frames`.
    /// Extra channels are left untouched. Does not allocate or resize storage.
    pub fn process(&mut self, io: &mut [Vec<f32>], frames: usize) {
        self.inner.process(io, frames);
    }

    /// Control-thread reset; clearing a large delay is not a callback operation.
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepared_delay_preserves_timing_across_unequal_blocks_and_reset() {
        let mut delay = PreparedDelay::new(3, 2).unwrap();
        let mut first = vec![vec![1.0, 2.0], vec![10.0, 20.0]];
        delay.process(&mut first, 2);
        assert_eq!(first, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
        let mut next = vec![vec![3.0, 4.0, 5.0, 6.0], vec![30.0, 40.0, 50.0, 60.0]];
        delay.process(&mut next, 4);
        assert_eq!(next, vec![vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 10.0, 20.0, 30.0]]);
        delay.clear();
        let mut silence = vec![vec![0.0; 4]; 2];
        delay.process(&mut silence, 4);
        assert_eq!(silence, vec![vec![0.0; 4]; 2]);
    }

    #[test]
    fn prepared_delay_rejects_unbounded_layouts() {
        assert!(PreparedDelay::new(MAX_LATENCY + 1, 2).is_err());
        assert!(PreparedDelay::new(3, 0).is_err());
        assert!(PreparedDelay::new(3, MAX_CHANNELS + 1).is_err());
        let mut dry = PreparedDelay::new(0, 1).unwrap();
        let mut audio = vec![vec![0.25, -0.5]];
        dry.process(&mut audio, 2);
        assert_eq!(audio, vec![vec![0.25, -0.5]]);
    }
}
