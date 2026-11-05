//! Preparation and state handoff for the built-in-only mobile mixer.
use super::{MAX_LATENCY, MixEngine, PluginSlot, Samples, Session, TrackKind, main_channels};
use std::sync::Arc;

impl MixEngine {
    /// Prepare a fresh built-in mixer on a control thread, including compensation delays for
    /// the parameters at `pos`. A disposable silent probe resolves plugin latency without
    /// advancing the real instances' smoothers, oscillators, filters, or effect tails.
    ///
    /// This allocates, constructs DSP, and temporarily holds a second mixer. Compensation
    /// reserves each processor's declared maximum latency, including future automation.
    /// Do not call in a callback.
    pub fn prepare_realtime(&mut self, session: &Session, pos: Samples) {
        self.ensure_synced(session);
        let mut probe = MixEngine::new(self.sample_rate, self.max_block);
        probe.monitor_only = true;
        probe.ensure_synced(session);
        let mut output = vec![vec![0.0; self.max_block]; main_channels(session)];
        probe.render(session, pos, 1, &mut output);
        for (id, strip) in &mut self.strips {
            if let Some(prepared) = probe.strips.get_mut(id) {
                for (delay, prepared) in [
                    (&mut strip.align, &mut prepared.align),
                    (&mut strip.align_pre, &mut prepared.align_pre),
                    (&mut strip.out_extra, &mut prepared.out_extra),
                ] {
                    std::mem::swap(delay, prepared);
                    delay.clear();
                }
                strip.latency = prepared.latency;
            }
        }
        self.latency = probe.latency;
        self.track_latency = probe.track_latency;
        self.aux_latency = probe.aux_latency;
        let mut max_track = 0;
        let mut max_aux = 0;
        for track in &session.tracks {
            let Some(strip) = self.strips.get(&track.id) else { continue };
            let maximum = strip.plugins.iter().flatten().fold(0usize, |sum, slot| sum.saturating_add(slot.plugin.max_latency())).min(MAX_LATENCY);
            match track.kind {
                TrackKind::Aux | TrackKind::Folder => max_aux = max_aux.max(maximum),
                TrackKind::Master | TrackKind::Vca => {}
                _ => max_track = max_track.max(maximum),
            }
        }
        for strip in self.strips.values_mut() {
            let channels = strip.buf.len();
            strip.align.reserve(max_track.max(max_aux), channels);
            strip.align_pre.reserve(max_track, channels);
            strip.out_extra.reserve(max_aux, channels);
        }
    }

    /// Move compatible live DSP state into an independently prepared replacement. This swaps
    /// ownership or copies into already prepared storage; it never allocates or destroys plugins.
    /// The caller must retire `previous` on a control thread after the handoff.
    ///
    /// Channel/rate/block-size changes start fresh where state is incompatible. Compensation
    /// keeps its live history when it fits the new prepared capacity, copying into larger
    /// prepared storage when needed. Clip-effect chains are not transferred here yet.
    pub fn inherit_runtime(&mut self, previous: &mut Self) {
        if self.sample_rate != previous.sample_rate || self.max_block != previous.max_block {
            return;
        }
        self.last_end = previous.last_end;
        self.input_channels = previous.input_channels;
        self.monitor_only = previous.monitor_only;
        self.recording = previous.recording;
        self.metronome = previous.metronome;
        std::mem::swap(&mut self.input, &mut previous.input);
        for (id, strip) in &mut self.strips {
            let Some(old) = previous.strips.get_mut(id) else { continue };
            for (next, old) in strip.plugins.iter_mut().zip(&mut old.plugins) {
                transfer_slot(next, old);
            }
            let same_sample = match (&strip.sample, &old.sample) {
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            };
            if same_sample && transfer_slot(&mut strip.instrument, &mut old.instrument) {
                std::mem::swap(&mut strip.held, &mut old.held);
                strip.chase = old.chase;
            }
            for (delay, old) in
                [(&mut strip.align, &mut old.align), (&mut strip.align_pre, &mut old.align_pre), (&mut strip.out_extra, &mut old.out_extra)]
            {
                if delay.bufs.len() != old.bufs.len() {
                    continue;
                }
                let old_cap = old.bufs.first().map_or(0, Vec::len);
                let next_cap = delay.bufs.first().map_or(0, Vec::len);
                if old_cap >= next_cap {
                    std::mem::swap(delay, old);
                } else {
                    // Keep the larger prepared rings; swapping in smaller old rings would
                    // reintroduce a callback allocation when later automation needs it. Copy the
                    // live history oldest first, as DelayLine growth does, so reads continue.
                    let start = old.idx.min(old_cap);
                    for (next, old) in delay.bufs.iter_mut().zip(&old.bufs) {
                        let (head, tail) = old.split_at(start);
                        next[..tail.len()].copy_from_slice(tail);
                        next[tail.len()..old_cap].copy_from_slice(head);
                        next[old_cap..].fill(0.0);
                    }
                    delay.len = old.len;
                    delay.idx = old_cap;
                }
            }
        }
    }
}

fn transfer_slot(next: &mut Option<PluginSlot>, old: &mut Option<PluginSlot>) -> bool {
    let (Some(next), Some(old)) = (next, old) else { return false };
    if next.id != old.id || next.ch != old.ch {
        return false;
    }
    if next.applied_keys == old.applied_keys {
        // Retain values already applied to the live plugin. Subsequent document values still
        // flow through apply_values, so real edits update smoothers instead of resetting them.
        std::mem::swap(&mut next.applied, &mut old.applied);
    }
    // Keep the new parameter identities and their prepared cache when its shape changed.
    // Retain the larger lane scratch even when parameter identities stayed the same: an
    // automation-only edit can require more storage without changing the plugin parameters.
    if old.lanes.capacity() > next.lanes.capacity() {
        std::mem::swap(&mut next.lanes, &mut old.lanes);
    }
    std::mem::swap(&mut next.plugin, &mut old.plugin);
    std::mem::swap(&mut next.failed, &mut old.failed);
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use soundcraft_audio_io::AudioBuffer;
    use soundcraft_dsp::Plugin;
    use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackId};

    fn session() -> (Session, TrackId, TrackId) {
        let mut session = Session::default();
        let vocal = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let latent = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let samples = (0..48_000).map(|i| if i < 512 { (i as f32 * 0.21).sin() * 0.4 } else { 0.0 }).collect();
        session.pool.insert(SourceId(1), Arc::new(SourceAudio::new(AudioBuffer { sample_rate: 48_000, channels: vec![samples] })));
        let clip = session.new_clip_id();
        session.track_mut(vocal).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(clip, "vocal", SourceId(1), 0, 0, 48_000));
        let mut reverb = Insert::new("room_reverb");
        reverb.params.insert("mix".into(), 60.0);
        session.track_mut(vocal).unwrap().mixer.inserts[0] = Some(reverb);
        let mut compressor = Insert::new("compressor");
        compressor.params.insert("lookahead".into(), 5.0);
        session.track_mut(latent).unwrap().mixer.inserts[0] = Some(compressor);
        (session, vocal, latent)
    }

    #[test]
    fn preparation_preserves_first_block_pcm_and_prepares_nonzero_delays() {
        let (session, vocal, _) = session();
        let mut prepared = MixEngine::new(48_000.0, 512);
        prepared.prepare_realtime(&session, 0);
        assert_eq!(prepared.strips[&vocal].align.len, 240);
        let delay = prepared.strips[&vocal].align.bufs[0].as_ptr();
        let mut reference = MixEngine::new(48_000.0, 512);
        let mut a = vec![vec![0.0; 512]; 2];
        let mut b = a.clone();
        for block in 0..20 {
            prepared.render(&session, block * 512, 512, &mut a);
            reference.render(&session, block * 512, 512, &mut b);
            assert_eq!(a, b, "silent probing must not advance the real DSP or parameter ramps");
            assert_eq!(prepared.strips[&vocal].align.bufs[0].as_ptr(), delay);
        }
    }

    #[test]
    fn prepared_replacements_retain_reverb_tails_delays_and_parameter_smoothing() {
        let (mut session, vocal, latent) = session();
        let mut live = MixEngine::new(48_000.0, 512);
        live.prepare_realtime(&session, 0);
        let mut reference = MixEngine::new(48_000.0, 512);
        let mut a = vec![vec![0.0; 512]; 2];
        let mut b = a.clone();
        let mut ringing_after_edit = false;
        for block in 0..80 {
            if [12, 24, 36, 48].contains(&block) {
                if block < 48 {
                    session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
                    if block == 24 {
                        session.track_mut(latent).unwrap().mixer.inserts[1] = Some(Insert::new("compressor"));
                    }
                } else {
                    session.track_mut(vocal).unwrap().mixer.inserts[0].as_mut().unwrap().params.insert("mix".into(), 30.0);
                }
                let mut next = MixEngine::new(48_000.0, 512);
                next.prepare_realtime(&session, block * 512);
                let new_layout = next.synced;
                let old_plugin = std::ptr::from_ref(live.strips[&vocal].plugins[0].as_ref().unwrap().plugin.as_ref());
                let required_delay_capacity = next.strips[&vocal].align.bufs[0].capacity();
                let prepared_buffer = next.strips[&vocal].buf[0].as_ptr();
                next.inherit_runtime(&mut live);
                assert_eq!(next.synced, new_layout);
                assert!(std::ptr::eq(old_plugin, std::ptr::from_ref(next.strips[&vocal].plugins[0].as_ref().unwrap().plugin.as_ref())));
                assert!(next.strips[&vocal].align.bufs[0].capacity() >= required_delay_capacity);
                assert_eq!(next.strips[&vocal].buf[0].as_ptr(), prepared_buffer);
                assert_eq!(next.last_end, block * 512);
                live = next;
            }
            live.render(&session, block * 512, 512, &mut a);
            reference.render(&session, block * 512, 512, &mut b);
            assert_eq!(a, b, "replacement differs from the persistent mixer at block {block}");
            if block > 12 && a.iter().flatten().any(|x| x.abs() > 1e-6) {
                ringing_after_edit = true;
            }
        }
        assert!(ringing_after_edit, "test must exercise a live reverb tail after a structural edit");
    }

    #[test]
    fn parameter_shape_changes_reuse_dsp_with_the_new_cache() {
        let (mut session, _, track) = session();
        let insert = session.track_mut(track).unwrap().mixer.inserts[0].as_mut().unwrap();
        insert.params.clear();
        insert.params.insert("attack".into(), 30.0);
        let mut previous = MixEngine::new(48_000.0, 512);
        previous.prepare_realtime(&session, 0);
        let mut output = vec![vec![0.0; 512]; 2];
        previous.render(&session, 0, 512, &mut output);
        let insert = session.track_mut(track).unwrap().mixer.inserts[0].as_mut().unwrap();
        insert.params.clear();
        insert.params.insert("release".into(), 30.0);
        let mut next = MixEngine::new(48_000.0, 512);
        next.prepare_realtime(&session, 512);
        let cache = next.strips[&track].plugins[0].as_ref().unwrap().applied.as_ptr();
        let plugin = std::ptr::from_ref(previous.strips[&track].plugins[0].as_ref().unwrap().plugin.as_ref());
        next.inherit_runtime(&mut previous);
        next.render(&session, 512, 512, &mut output);
        let slot = next.strips[&track].plugins[0].as_ref().unwrap();
        assert!(std::ptr::eq(plugin, std::ptr::from_ref(slot.plugin.as_ref())));
        assert_eq!(slot.applied.as_ptr(), cache);
        assert_eq!(slot.plugin.param("release"), Some(30.0));
    }

    #[test]
    fn sample_rate_changes_keep_the_fresh_instances() {
        let (session, vocal, _) = session();
        let mut previous = MixEngine::new(48_000.0, 512);
        previous.prepare_realtime(&session, 0);
        let mut next = MixEngine::new(44_100.0, 512);
        next.prepare_realtime(&session, 512);
        let plugin = std::ptr::from_ref(next.strips[&vocal].plugins[0].as_ref().unwrap().plugin.as_ref());
        next.inherit_runtime(&mut previous);
        assert!(std::ptr::eq(plugin, std::ptr::from_ref(next.strips[&vocal].plugins[0].as_ref().unwrap().plugin.as_ref())));
    }

    struct Watched {
        inner: Box<dyn Plugin>,
        drops: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl Drop for Watched {
        fn drop(&mut self) {
            self.drops.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }
    impl Plugin for Watched {
        fn info(&self) -> &'static soundcraft_dsp::PluginInfo {
            self.inner.info()
        }
        fn prepare(&mut self, sr: f32, mb: usize, ch: usize) {
            self.inner.prepare(sr, mb, ch);
        }
        fn reset(&mut self) {
            self.inner.reset();
        }
        fn set_param(&mut self, id: &str, value: f32) -> bool {
            self.inner.set_param(id, value)
        }
        fn param(&self, id: &str) -> Option<f32> {
            self.inner.param(id)
        }
        fn process(&mut self, io: &mut [Vec<f32>], frames: usize) {
            self.inner.process(io, frames);
        }
    }

    #[test]
    fn handoff_defers_destruction_of_replaced_instances_to_the_retiring_thread() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (session, track, _) = session();
        let mut previous = MixEngine::new(48_000.0, 512);
        previous.prepare_realtime(&session, 0);
        let mut next = MixEngine::new(48_000.0, 512);
        next.prepare_realtime(&session, 0);
        let live_drops = Arc::new(AtomicUsize::new(0));
        let spare_drops = Arc::new(AtomicUsize::new(0));
        for (engine, drops) in [(&mut previous, Arc::clone(&live_drops)), (&mut next, Arc::clone(&spare_drops))] {
            let slot = engine.strips.get_mut(&track).unwrap().plugins[0].as_mut().unwrap();
            let inner = std::mem::replace(&mut slot.plugin, soundcraft_dsp::create("room_reverb").unwrap());
            slot.plugin = Box::new(Watched { inner, drops });
        }
        next.inherit_runtime(&mut previous);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0);
        assert_eq!(spare_drops.load(Ordering::SeqCst), 0, "handoff must retain the prepared spare too");
        std::thread::spawn(move || drop(previous)).join().unwrap();
        assert_eq!(spare_drops.load(Ordering::SeqCst), 1);
        assert_eq!(live_drops.load(Ordering::SeqCst), 0, "the sounding instance remains live");
        drop(next);
        assert_eq!(live_drops.load(Ordering::SeqCst), 1);
    }
}
