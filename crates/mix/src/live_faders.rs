//! Bounded, allocation-free live fader overrides shared with the audio callback.
use soundcraft_model::TrackId;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};

const SLOTS: usize = 128;
const EMPTY: u64 = 0;
const CLAIMED: u64 = u64::MAX;

pub struct LiveFaders {
    active: AtomicUsize,
    tracks: [AtomicU64; SLOTS],
    values: [AtomicU32; SLOTS],
}

impl Default for LiveFaders {
    fn default() -> Self {
        Self {
            active: AtomicUsize::new(0),
            tracks: std::array::from_fn(|_| AtomicU64::new(EMPTY)),
            values: std::array::from_fn(|_| AtomicU32::new(f32::NAN.to_bits())),
        }
    }
}

impl LiveFaders {
    /// Set or clear a track's live fader. A full table refuses a new track.
    pub fn set(&self, track: TrackId, db: Option<f32>) -> bool {
        if track.0 == EMPTY || track.0 == CLAIMED || db.is_some_and(|v| !v.is_finite() || !(-144.0..=12.0).contains(&v)) {
            return false;
        }
        let bits = db.unwrap_or(f32::NAN).to_bits();
        for (id, value) in self.tracks.iter().zip(&self.values) {
            if id.load(Ordering::Acquire) == track.0 {
                value.store(bits, Ordering::Release);
                if db.is_none() {
                    id.store(EMPTY, Ordering::Release);
                    self.active.fetch_sub(1, Ordering::Relaxed);
                }
                return true;
            }
        }
        if db.is_none() {
            return true;
        }
        for (id, value) in self.tracks.iter().zip(&self.values) {
            if id.compare_exchange(EMPTY, CLAIMED, Ordering::AcqRel, Ordering::Acquire).is_ok() {
                value.store(bits, Ordering::Release);
                id.store(track.0, Ordering::Release);
                self.active.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        }
        false
    }

    /// The callback reads only atomics; inactive slots return no override.
    pub fn get(&self, track: TrackId) -> Option<f32> {
        if self.active.load(Ordering::Relaxed) == 0 {
            return None;
        }
        self.tracks
            .iter()
            .zip(&self.values)
            .find_map(|(id, value)| {
                let first = id.load(Ordering::Acquire);
                if first != track.0 {
                    return None;
                }
                let db = f32::from_bits(value.load(Ordering::Acquire));
                (id.load(Ordering::Acquire) == first).then_some(db)
            })
            .filter(|v| v.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_table_refuses_new_tracks_and_cleared_slots_can_be_reused() {
        let faders = LiveFaders::default();
        for id in 1..=SLOTS as u64 {
            assert!(faders.set(TrackId(id), Some(-6.0)));
        }
        assert!(!faders.set(TrackId(SLOTS as u64 + 1), Some(-6.0)));
        assert!(faders.set(TrackId(1), None));
        assert_eq!(faders.get(TrackId(1)), None);
        assert!(faders.set(TrackId(SLOTS as u64 + 1), Some(-9.0)));
        assert_eq!(faders.get(TrackId(SLOTS as u64 + 1)), Some(-9.0));
    }
}
