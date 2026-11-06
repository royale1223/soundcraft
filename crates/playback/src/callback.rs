use super::{
    Arc, AtomicBool, BLOCK, BTreeMap, Cmd, HashMap, HashSet, MAX_CHANNELS, MAX_CONTROL_COMMANDS_PER_CALLBACK, MeterSnapshot, MixEngine, MpscReceiver,
    OLD_SESSIONS, Ordering, OutputTrouble, Plugin, PluginEditor, PluginStateReader, PluginStates, PoisonError, PreparedInstance, Range, Receiver,
    RetiredSession, ReturnReceiver, ReturnSender, Samples, Sender, Session, Shared, SyncSender, TrackId, itu_stereo_fold, main_channels, record,
    return_channel,
};

/// How the audio output is doing, for the UI.
#[derive(Default)]
pub(crate) struct Fault {
    /// A block panicked: output stays silent from then on.
    pub(crate) failed: AtomicBool,
    /// The device went away or the stream was invalidated: only a reconnect can restore audio.
    pub(crate) gone: AtomicBool,
    /// Blocks so far, to tell a device that stopped asking for audio.
    pub(crate) beats: std::sync::atomic::AtomicU64,
    /// The UI has been told.
    pub(crate) reported: AtomicBool,
}

impl Fault {
    /// What stopped the output, once.
    pub(crate) fn trouble(&self, stalled: bool) -> Option<OutputTrouble> {
        let trouble = if self.failed.load(Ordering::Relaxed) {
            OutputTrouble::Bug
        } else if self.gone.load(Ordering::Relaxed) {
            OutputTrouble::Gone
        } else if stalled {
            OutputTrouble::Stalled
        } else {
            return None;
        };
        (!self.reported.swap(true, Ordering::Relaxed)).then_some(trouble)
    }
}

/// How long a playing output may go without a block before it counts as stalled.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const STALL: std::time::Duration = std::time::Duration::from_secs(2);

/// True once the block count `beats` has stood still for `STALL` while playing; `seen` is the
/// count last seen and since when.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn stalled(seen: &mut Option<(u64, std::time::Instant)>, beats: u64, playing: bool) -> bool {
    match *seen {
        Some((b, since)) if playing && b == beats => since.elapsed() >= STALL,
        _ => {
            *seen = Some((beats, std::time::Instant::now()));
            false
        }
    }
}

/// Fill one block with `fill`. If it panics, the block and every later one are silent and the
/// transport stops, rather than the panic taking the app down.
pub(crate) fn guarded_block(fault: &Fault, shared: &Shared, out: &mut [f32], fill: impl FnOnce(&mut [f32])) {
    fault.beats.fetch_add(1, Ordering::Relaxed);
    if fault.failed.load(Ordering::Relaxed) || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fill(out))).is_err() {
        fault.failed.store(true, Ordering::Relaxed);
        shared.playing.store(false, Ordering::Relaxed);
        out.fill(0.0);
    }
}

/// Length of the output fade when playback starts or stops, so the waveform is not cut mid-cycle.
const FADE_SECONDS: f64 = 0.01;

/// State owned by the audio callback.
pub(crate) struct AudioState {
    pub(crate) rx: Receiver<Cmd>,
    /// A second session command is held until the next callback so only one structural sync runs
    /// during any single device block.
    pub(crate) deferred: Option<Cmd>,
    /// Replaced state that cannot fit in the return ring stays owned here. Further commands
    /// wait for retirement capacity so another update can never overwrite and destroy it.
    pub(crate) pending_old_session: Option<RetiredSession>,
    /// A retired plugin batch that could not fit in the bounded return queue stays owned here.
    pub(crate) pending_garbage: Option<Vec<Box<dyn Plugin>>>,
    pub(crate) shared: Arc<Shared>,
    pub(crate) session: Arc<Session>,
    pub(crate) mix: Box<MixEngine>,
    pub(crate) pos: Samples,
    pub(crate) playing: bool,
    end: Option<Samples>,
    looped: Option<Range>,
    /// Rendered frames waiting to be consumed (planar, main channels) and the fractional read index.
    pub(crate) pending: Vec<Vec<f32>>,
    /// Active channel count within the fixed-capacity `pending` workspace.
    pub(crate) pending_channels: usize,
    /// Stereo fold-down gains per main channel (used when the device has fewer channels).
    fold: [(f32, f32); MAX_CHANNELS],
    pub(crate) pending_len: usize,
    read: f64,
    pub(crate) speaker_guard: SpeakerGuard,
    /// Gain applied to the device output for start/stop fades (0..=1); 1 in steady state.
    pub(crate) out_gain: f32,
    /// A stop fading out: the position playback stopped at, restored once the fade ends.
    pub(crate) stopping: Option<Samples>,
    /// Stream shutdown also silences live input after the stop fade completes.
    output_shutdown: bool,
    /// A locate during playback fading out at the old position: the position to jump to once
    /// silent, after which the output fades back in. Jumping mid-waveform clicked.
    pub(crate) locating: Option<Samples>,
    /// Start/stop fades are on; tests comparing exact PCM from the first sample turn them off.
    pub(crate) fades: bool,
    pub(crate) feedback_guard: FeedbackGuard,
    pub(crate) device_rate: f64,
    pub(crate) input: Option<Arc<record::InputRing>>,
    pub(crate) recording: bool,
    /// Retired plugin instances go back to the UI thread to be dropped there.
    pub(crate) garbage: SyncSender<Vec<Box<dyn Plugin>>>,
    pub(crate) old_sessions: Option<ReturnSender<RetiredSession>>,
    /// Meters of blocks rendered while the UI held the shared snapshot (`try_lock` failed), so
    /// their peaks reach it with the next block instead of being lost. Shaped on the control
    /// thread and replaced with each session ([`Cmd::Session`]) so the callback never inserts
    /// entries.
    pub(crate) unsent: Box<MeterSnapshot>,
}

/// Callback-owned conservative detector for a loud, persistent near-sine input.
/// It ducks only the live monitor send; capture samples remain untouched.
#[derive(Default)]
pub(crate) struct FeedbackGuard {
    pub(crate) hot_blocks: u16,
    pub(crate) gain: f32,
    pub(crate) previous_rms: f32,
}

impl FeedbackGuard {
    pub(crate) fn update(&mut self, input: &[Vec<f32>], frames: usize, sample_rate: u32, active: bool) -> f32 {
        if active && frames >= 32 {
            let rms = input
                .iter()
                .filter_map(|channel| channel.get(..frames))
                .map(|samples| {
                    let energy = samples.iter().map(|sample| f64::from(*sample) * f64::from(*sample)).sum::<f64>();
                    (energy / frames as f64).sqrt() as f32
                })
                .fold(0.0f32, f32::max);
            let narrowband = input.iter().any(|channel| resonant_input(channel, frames, sample_rate));
            let rising_or_excessive = rms > 0.2 && (rms > self.previous_rms * 1.04 || rms > 0.8);
            self.previous_rms = rms;
            if narrowband && rising_or_excessive {
                self.hot_blocks = self.hot_blocks.saturating_add(1);
                if self.hot_blocks >= 8 {
                    self.gain = (self.gain * 0.72).max(0.125);
                }
            } else {
                self.hot_blocks = 0;
                self.gain = (self.gain + 0.02).min(1.0);
            }
        } else {
            self.hot_blocks = 0;
            self.previous_rms = 0.0;
            self.gain = (self.gain + 0.02).min(1.0);
        }
        self.gain
    }
}

/// Per-sample device-output ceiling with an immediate attack and a 50 ms release.
#[derive(Clone, Copy)]
pub(crate) struct SpeakerGuard {
    pub(crate) gain: f32,
}

impl SpeakerGuard {
    pub(crate) const CEILING: f32 = 0.891_250_9; // −1 dBFS

    pub(crate) fn process(&mut self, frame: &mut [f32], sample_rate: f64, enabled: bool) {
        if enabled {
            let peak = frame.iter().fold(0.0f32, |p, sample| p.max(sample.abs()));
            if peak > Self::CEILING {
                self.gain = (Self::CEILING / peak).min(self.gain);
            } else {
                let release = (1.0 / (sample_rate * 0.05).max(1.0)) as f32;
                self.gain = (self.gain + (1.0 - self.gain) * release).min(1.0);
            }
            for sample in frame {
                *sample = (*sample * self.gain).clamp(-Self::CEILING, Self::CEILING);
            }
        } else {
            for sample in frame {
                *sample = sample.clamp(-1.0, 1.0);
            }
        }
    }
}

/// Detect whether one narrow frequency bin dominates a loud monitor input. The bounded
/// Goertzel scan uses no scratch allocation and only checks roughly 200 Hz–4 kHz.
pub(crate) fn resonant_input(samples: &[f32], frames: usize, sample_rate: u32) -> bool {
    let Some(samples) = samples.get(..frames) else {
        return false;
    };
    let sum_sq = samples.iter().map(|sample| f64::from(*sample) * f64::from(*sample)).sum::<f64>();
    let rms = (sum_sq / frames as f64).sqrt();
    if rms < 0.2 {
        return false;
    }
    let mut dominant = 0.0f64;
    let rate = f64::from(sample_rate.max(1));
    let first_bin = ((200.0 * frames as f64 / rate).ceil() as usize).max(1);
    let last_bin = ((4_000.0 * frames as f64 / rate).floor() as usize).min(frames / 2);
    if first_bin > last_bin {
        return false;
    }
    for bin in first_bin..=last_bin {
        let omega = std::f64::consts::TAU * bin as f64 / frames as f64;
        let coeff = 2.0 * omega.cos();
        let (mut s1, mut s2) = (0.0f64, 0.0f64);
        for &sample in samples {
            let s0 = f64::from(sample) + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        let power = (s1 * s1 + s2 * s2 - coeff * s1 * s2).max(0.0);
        dominant = dominant.max(power / (frames as f64 * sum_sq.max(1.0e-12)));
    }
    dominant > 0.22
}

pub(crate) fn new_engine(sr: f32) -> MixEngine {
    let mut m = MixEngine::new(sr, BLOCK);
    m.set_external_instances(true);
    m.metronome = true;
    // Reserve the maximum supported input layout for this engine instance. The active channel
    // count is carried separately so input-channel changes do not resize per-block buffers.
    m.input = (0..MAX_CHANNELS).map(|_| vec![0.0; BLOCK]).collect();
    m
}

impl AudioState {
    pub(crate) fn new(
        rx: Receiver<Cmd>,
        shared: Arc<Shared>,
        session: Arc<Session>,
        device_rate: f64,
        garbage: SyncSender<Vec<Box<dyn Plugin>>>,
    ) -> Self {
        let sr = session.sample_rate.as_f64() as f32;
        let nch = main_channels(&session);
        let unsent = Box::new(MeterSnapshot::shaped(&session));
        // Construct the initial routing, strip buffers and meter storage before the device stream
        // starts invoking this state. The output workspace covers every supported main layout so
        // later format changes only alter its active channel count.
        let mut mix = Box::new(new_engine(sr));
        #[cfg(feature = "plugin-hosts")]
        mix.ensure_synced(&session);
        #[cfg(not(feature = "plugin-hosts"))]
        mix.prepare_realtime(&session, 0);
        AudioState {
            garbage,
            old_sessions: None,
            deferred: None,
            pending_old_session: None,
            pending_garbage: None,
            fold: fold_gains(&session),
            rx,
            shared,
            mix,
            session,
            pos: 0,
            playing: false,
            end: None,
            looped: None,
            pending: (0..MAX_CHANNELS).map(|_| vec![0.0; BLOCK]).collect(),
            pending_channels: nch,
            pending_len: 0,
            read: 0.0,
            speaker_guard: SpeakerGuard { gain: 1.0 },
            out_gain: 1.0,
            stopping: None,
            output_shutdown: false,
            locating: None,
            fades: true,
            feedback_guard: FeedbackGuard { hot_blocks: 0, gain: 1.0, previous_rms: 0.0 },
            device_rate,
            input: None,
            recording: false,
            unsent,
        }
    }

    pub(crate) fn drain(&mut self) {
        if !self.flush_garbage() {
            self.shared.playing.store(self.transport_playing(), Ordering::Relaxed);
            return;
        }
        if let Some(old) = self.pending_old_session.take() {
            if let Some(tx) = &mut self.old_sessions {
                match tx.try_send(old) {
                    Ok(()) => {}
                    Err(std::sync::mpsc::TrySendError::Full(old) | std::sync::mpsc::TrySendError::Disconnected(old)) => {
                        self.pending_old_session = Some(old);
                        self.shared.playing.store(self.transport_playing(), Ordering::Relaxed);
                        return;
                    }
                }
            } else {
                self.pending_old_session = Some(old);
                self.shared.playing.store(self.transport_playing(), Ordering::Relaxed);
                return;
            }
        }
        let mut changed = false;
        let mut drained = 0;
        let mut session_updates = 0;
        loop {
            if drained >= MAX_CONTROL_COMMANDS_PER_CALLBACK {
                break;
            }
            let command = self.deferred.take().or_else(|| self.rx.try_recv().ok());
            let Some(c) = command else { break };
            if session_updates > 0 && matches!(&c, Cmd::Session(..)) {
                self.deferred = Some(c);
                break;
            }
            drained += 1;
            match c {
                Cmd::Input(r) => {
                    if let Some(old) = self.input.replace(r) {
                        self.retire_state(RetiredSession { _session: None, _mix: None, _input: Some(old), _meters: None });
                    }
                }
                Cmd::Recording(r) => self.recording = r,
                Cmd::Session(s, instances, mut meters) => {
                    session_updates += 1;
                    changed = true;
                    #[cfg(feature = "plugin-hosts")]
                    let retired_mix = {
                        if (s.sample_rate.as_f64() - self.session.sample_rate.as_f64()).abs() > 0.5 {
                            let mut old = std::mem::replace(&mut self.mix, Box::new(new_engine(s.sample_rate.as_f64() as f32)));
                            self.queue_garbage(old.drain_plugins());
                        }
                        self.mix.adopt(instances);
                        None
                    };
                    #[cfg(not(feature = "plugin-hosts"))]
                    let retired_mix = {
                        let mut prepared = instances;
                        prepared.inherit_runtime(&mut self.mix);
                        Some(std::mem::replace(&mut self.mix, prepared))
                    };
                    // The main format changed (a document edit, not steady state): resize.
                    let nch = main_channels(&s);
                    if self.pending_channels != nch {
                        self.pending_channels = nch;
                        self.pending_len = 0;
                        self.read = 0.0;
                    }
                    self.fold = fold_gains(&s);
                    meters.carry_levels(&self.unsent);
                    let old_meters = std::mem::replace(&mut self.unsent, meters);
                    let old = RetiredSession {
                        _session: Some(std::mem::replace(&mut self.session, s)),
                        _mix: retired_mix,
                        _input: None,
                        _meters: Some(old_meters),
                    };
                    self.retire_state(old);
                }
                Cmd::OldSessions(tx) => self.old_sessions = Some(tx),
                Cmd::Play { from, end, looped } => {
                    self.output_shutdown = false;
                    // Start from silence and fade in, unless a monitored input is already sounding
                    // (fading the whole output would dip it).
                    self.out_gain = if self.monitoring() || !self.fades { 1.0 } else { 0.0 };
                    self.stopping = None;
                    self.locating = None;
                    self.pos = from;
                    self.end = end;
                    self.looped = looped.filter(|r| r.len() > 64);
                    self.playing = true;
                    self.pending_len = 0;
                    self.read = 0.0;
                    self.mix.reset();
                }
                Cmd::Stop => {
                    if self.playing && self.fades && !self.monitoring() {
                        // Fade out over the next frames; `fill` finishes the stop.
                        if self.stopping.is_none() {
                            // A pending locate's target is where the transport already reads.
                            self.stopping = Some(self.locating.take().unwrap_or(self.pos));
                        }
                    } else {
                        self.finish_stop();
                    }
                }
                Cmd::StopOutput => {
                    let sounding = self.playing || self.monitoring();
                    self.output_shutdown = true;
                    // Keep rendering through the fade even near a scheduled recording end.
                    self.end = None;
                    self.looped = None;
                    if sounding && self.fades {
                        if self.stopping.is_none() {
                            self.stopping = Some(self.locating.take().unwrap_or(self.pos));
                        }
                    } else {
                        self.finish_stop();
                    }
                }
                Cmd::Locate(at) => {
                    if let Some(stop_at) = &mut self.stopping {
                        *stop_at = at;
                    }
                    if self.playing && self.stopping.is_none() && self.fades && !self.monitoring() {
                        // Fade out here first; `fill` jumps once silent and fades back in.
                        self.locating = Some(at);
                    } else {
                        self.locating = None;
                        self.jump_to(at);
                    }
                }
            }
            if self.pending_old_session.is_some() {
                break;
            }
        }
        #[cfg(not(feature = "plugin-hosts"))]
        let _ = changed;
        #[cfg(feature = "plugin-hosts")]
        if changed {
            // Place adopted instances now (not only once playback starts), so editors see them;
            // `sync` creates no third-party plugins here.
            self.mix.ensure_synced(&self.session);
        }
        self.send_garbage();
        self.shared.playing.store(self.transport_playing(), Ordering::Relaxed);
    }

    pub(crate) fn retire_state(&mut self, old: RetiredSession) {
        if let Some(tx) = &mut self.old_sessions {
            match tx.try_send(old) {
                Ok(()) => {}
                Err(std::sync::mpsc::TrySendError::Full(old) | std::sync::mpsc::TrySendError::Disconnected(old)) => {
                    self.pending_old_session = Some(old);
                }
            }
        } else {
            self.pending_old_session = Some(old);
        }
    }

    fn send_garbage(&mut self) {
        if self.pending_garbage.is_some() {
            let _ = self.flush_garbage();
            return;
        }
        let g = self.mix.take_retired();
        if !g.is_empty() {
            self.queue_garbage(g);
        }
    }

    pub(crate) fn queue_garbage(&mut self, garbage: Vec<Box<dyn Plugin>>) {
        if garbage.is_empty() {
            return;
        }
        match self.garbage.try_send(garbage) {
            Ok(()) => {}
            Err(std::sync::mpsc::TrySendError::Full(garbage) | std::sync::mpsc::TrySendError::Disconnected(garbage)) => {
                self.pending_garbage = Some(garbage);
            }
        }
    }

    pub(crate) fn flush_garbage(&mut self) -> bool {
        let Some(garbage) = self.pending_garbage.take() else { return true };
        match self.garbage.try_send(garbage) {
            Ok(()) => true,
            Err(std::sync::mpsc::TrySendError::Full(garbage) | std::sync::mpsc::TrySendError::Disconnected(garbage)) => {
                self.pending_garbage = Some(garbage);
                false
            }
        }
    }

    /// Render the next block at the session rate into `pending`.
    pub(crate) fn render_next(&mut self, quantum: usize) {
        let mut n = quantum.clamp(1, BLOCK);
        if self.pending_garbage.is_some() {
            for channel in &mut self.pending[..self.pending_channels] {
                channel[..n].fill(0.0);
            }
            self.pending_len = n;
            return;
        }
        if let Some(l) = self.looped
            && self.pos >= l.end
        {
            self.pos = l.start;
        }
        if let Some(l) = self.looped {
            n = n.min(usize::try_from((l.end - self.pos).max(1)).unwrap_or(BLOCK));
        } else if let Some(e) = self.end {
            if self.pos >= e {
                self.playing = false;
                self.end = None;
                self.stopping = None;
                if let Some(at) = self.locating.take() {
                    self.pos = at;
                }
                self.out_gain = 1.0;
                self.shared.stop_fading.store(false, Ordering::Relaxed);
                self.shared.playing.store(false, Ordering::Relaxed);
                n = 0;
            } else {
                n = n.min(usize::try_from(e - self.pos).unwrap_or(BLOCK));
            }
        }
        if n == 0 {
            for c in &mut self.pending[..self.pending_channels] {
                c.iter_mut().for_each(|x| *x = 0.0);
            }
            self.pending_len = BLOCK;
            return;
        }
        let monitor_gain = f32::from_bits(self.shared.monitor_gain.load(Ordering::Relaxed));
        let monitor_enabled = self.shared.direct_monitor_enabled.load(Ordering::Relaxed) && monitor_gain > 0.0;
        if let Some(ring) = &self.input {
            let ch = ring.channels.load(Ordering::Relaxed).max(1).min(self.mix.input.len());
            self.mix.input_channels = ch;
            ring.pop_into(&mut self.mix.input[..ch], n, self.session.sample_rate.hz());
            let active = monitor_enabled
                && self.shared.feedback_protection.load(Ordering::Relaxed)
                && self.session.tracks.iter().any(|t| t.mixer.input_monitor);
            self.feedback_guard.update(&self.mix.input[..ch], n, self.session.sample_rate.hz(), active);
        } else {
            self.mix.input_channels = 0;
            self.feedback_guard.update(&[], 0, 1, false);
        }
        self.mix.monitor_gain = if !monitor_enabled {
            0.0
        } else if self.shared.feedback_protection.load(Ordering::Relaxed) {
            monitor_gain * self.feedback_guard.gain
        } else {
            monitor_gain
        };
        self.mix.monitor_use_rack = monitor_enabled && self.shared.monitor_use_rack.load(Ordering::Relaxed);
        self.mix.monitor_only = !self.playing;
        self.mix.recording = self.recording;
        self.mix.render(&self.session, self.pos, n, &mut self.pending[..self.pending_channels]);
        self.send_garbage();
        self.pending_len = n;
        self.publish_meters();
        if self.playing {
            self.pos += n as i64;
            // A fading stop or locate keeps rendering, but the transport already reads its target.
            if self.stopping.is_none() && self.locating.is_none() {
                self.shared.position.store(self.pos, Ordering::Relaxed);
            }
        }
    }

    /// Playing as the transport reports it: a stop that is still fading out counts as stopped.
    fn transport_playing(&self) -> bool {
        self.playing && self.stopping.is_none()
    }

    /// A monitored input is live, so output fades would dip it.
    fn monitoring(&self) -> bool {
        (!self.output_shutdown || self.stopping.is_some())
            && self.input.is_some()
            && self.session.tracks.iter().any(|t| t.mixer.input_monitor || t.mixer.record_arm)
    }

    /// Continue from `at`, dropping audio rendered for the old position and effect tails.
    fn jump_to(&mut self, at: Samples) {
        self.pos = at;
        self.pending_len = 0;
        self.read = 0.0;
        self.mix.reset();
    }

    /// End playback: what a stop does once its fade-out (if any) has played.
    fn finish_stop(&mut self) {
        self.playing = false;
        self.locating = None;
        self.end = None;
        self.looped = None;
        self.mix.reset();
        if let Some(at) = self.stopping.take() {
            self.pos = at;
        }
        self.out_gain = if self.output_shutdown { 0.0 } else { 1.0 };
        self.shared.stop_fading.store(false, Ordering::Relaxed);
    }

    /// Ramp the output gain one frame toward its target: down to silence while stopping (finishing
    /// the stop there) or locating, otherwise up to full level. Bounded per-frame work, no allocation.
    fn fade_frame(&mut self, frame: &mut [f32]) {
        if self.output_shutdown && self.stopping.is_none() {
            frame.fill(0.0);
            return;
        }
        let stopping = self.stopping.is_some();
        let down = stopping || self.locating.is_some();
        if !down && self.out_gain >= 1.0 {
            return;
        }
        let step = 1.0 / (self.device_rate * FADE_SECONDS).max(1.0) as f32;
        self.out_gain = if down { (self.out_gain - step).max(0.0) } else { (self.out_gain + step).min(1.0) };
        let gain = self.out_gain;
        frame.iter_mut().for_each(|x| *x *= gain);
        if stopping && gain <= 0.0 {
            self.finish_stop();
        }
    }

    /// Hand this block's meters to the UI without blocking: they are folded (max) into what the
    /// UI has not taken yet. If the UI holds the snapshot right now, they wait in `unsent`.
    pub(crate) fn publish_meters(&mut self) {
        self.unsent.accumulate(&self.mix.main_meter, &self.mix.meters);
        if let Ok(mut m) = self.shared.meters.try_lock() {
            m.accumulate(&self.unsent.main, &self.unsent.tracks);
            self.unsent.reset_levels();
        }
    }

    /// Fill an interleaved device buffer.
    pub(crate) fn fill(&mut self, out: &mut [f32], channels: usize) {
        #[cfg(not(feature = "plugin-hosts"))]
        if self.rx.overflowed() {
            self.playing = false;
            self.stopping = None;
            self.locating = None;
            self.shared.playing.store(false, Ordering::Relaxed);
            self.shared.stop_fading.store(false, Ordering::Relaxed);
            out.fill(0.0);
            return;
        }
        self.drain();
        let channels = channels.max(1);
        let monitoring = self.monitoring();
        if !self.playing && !monitoring {
            out.iter_mut().for_each(|x| *x = 0.0);
            self.unsent.silence();
            if let Ok(mut m) = self.shared.meters.try_lock() {
                m.silence();
            }
            return;
        }
        let speed = f64::from(self.shared.speed.load(Ordering::Relaxed)) / 1000.0;
        let step = self.session.sample_rate.as_f64() / self.device_rate.max(1.0) * speed.clamp(0.1, 4.0);
        let callback_frames = out.len() / channels;
        let quantum = ((callback_frames as f64 * step).ceil() as usize).clamp(1, BLOCK);
        for frame in out.chunks_mut(channels) {
            let monitoring = monitoring && (!self.output_shutdown || self.stopping.is_some());
            while self.read >= self.pending_len as f64 {
                self.read -= self.pending_len as f64;
                if !self.playing && !monitoring {
                    break;
                }
                self.render_next(quantum);
                if self.pending_len == 0 {
                    break;
                }
            }
            let i0 = self.read.floor() as usize;
            let fr = (self.read - i0 as f64) as f32;
            let sounding = self.playing || monitoring;
            let nch = self.pending_channels;
            let sample = |c: usize| {
                let ch = self.pending.get(c);
                let a = ch.and_then(|v| v.get(i0)).copied().unwrap_or(0.0);
                // `pending` is BLOCK long; only `pending_len` frames are valid. Hold the last one at the edge.
                let b = if i0 + 1 < self.pending_len { ch.and_then(|v| v.get(i0 + 1)).copied().unwrap_or(a) } else { a };
                a * (1.0 - fr) + b * fr
            };
            write_frame(frame, channels, nch, sounding, &self.fold, sample);
            self.speaker_guard.process(frame, self.device_rate, self.shared.speaker_protection.load(Ordering::Relaxed));
            self.fade_frame(frame);
            self.read += step;
            if self.out_gain <= 0.0
                && let Some(at) = self.locating.take()
            {
                // Silent at the old position: the next frame renders from the new one.
                self.jump_to(at);
            }
        }
    }
}

/// Write one device frame from `nch` rendered channels: fold surround to stereo (a mono device
/// averages the two sides), or copy the channels straight through. `sample(c)` is channel `c`.
fn write_frame(frame: &mut [f32], channels: usize, nch: usize, sounding: bool, fold: &[(f32, f32)], sample: impl Fn(usize) -> f32) {
    if channels < nch {
        // Fold surround to stereo first; a mono device averages the two sides.
        let (mut l, mut r) = (0.0f32, 0.0f32);
        if sounding {
            if nch == 2 {
                (l, r) = (sample(0), sample(1));
            } else {
                for (c, (gl, gr)) in fold.iter().enumerate().take(nch) {
                    let v = sample(c);
                    l += v * gl;
                    r += v * gr;
                }
            }
        }
        for (c, o) in frame.iter_mut().enumerate() {
            *o = match c {
                0 if channels == 1 => out_sample(0.5 * l + 0.5 * r),
                0 => out_sample(l),
                1 => out_sample(r),
                _ => 0.0,
            };
        }
    } else {
        for (c, o) in frame.iter_mut().enumerate() {
            *o = if c < nch.max(1) && sounding { out_sample(sample(c)) } else { 0.0 };
        }
    }
}

/// Clamp to the device range, mapping NaN and infinities to silence.
fn out_sample(v: f32) -> f32 {
    if v.is_finite() { v.clamp(-1.0, 1.0) } else { 0.0 }
}

/// UI-thread bookkeeping of the third-party instances the audio thread holds.
pub(crate) struct Side {
    /// What was last created for each slot: (plugin id, channels, created ok).
    pub(crate) shadow: HashMap<(TrackId, usize), (String, usize, bool)>,
    /// Session sample rate the instances were prepared for.
    rate: f32,
    /// Editor handles of the live instances.
    pub(crate) editors: HashMap<(TrackId, usize), Box<dyn PluginEditor>>,
    /// State readers of the live instances.
    pub(crate) states: BTreeMap<(TrackId, usize), Box<dyn PluginStateReader>>,
    /// Instances that refused their stored state: saves keep that state until the plugin's editor
    /// is opened.
    pub(crate) refused: HashSet<(TrackId, usize)>,
    garbage: MpscReceiver<Vec<Box<dyn Plugin>>>,
    old_sessions: Option<ReturnReceiver<RetiredSession>>,
}

impl Side {
    pub(crate) fn new(rate: f32, garbage: MpscReceiver<Vec<Box<dyn Plugin>>>) -> Side {
        Side { shadow: HashMap::new(), rate, editors: HashMap::new(), states: BTreeMap::new(), refused: HashSet::new(), garbage, old_sessions: None }
    }

    /// Drops retired instances (closing their editors first, inside the plugin's drop) and
    /// replaced session snapshots.
    pub(crate) fn collect_garbage(&mut self) {
        while let Ok(g) = self.garbage.try_recv() {
            for p in g {
                log::debug!("plugin instance released: {}", p.info().name);
                drop(p);
            }
        }
        if let Some(rx) = &mut self.old_sessions {
            while let Ok(old) = rx.try_recv() {
                drop(old);
            }
        }
    }

    pub(crate) fn return_old_sessions(&mut self, tx: &Sender<Cmd>) {
        let (otx, orx) = return_channel(OLD_SESSIONS);
        self.old_sessions = Some(orx);
        let _ = tx.send(Cmd::OldSessions(otx));
    }

    pub(crate) fn forget(&mut self, key: (TrackId, usize)) {
        self.shadow.remove(&key);
        self.states.remove(&key);
        self.refused.remove(&key);
        if let Some(mut e) = self.editors.remove(&key) {
            e.close();
        }
    }
}

pub(crate) fn take_meter_snapshot(source: &mut MeterSnapshot, out: &mut MeterSnapshot) {
    if !source.fresh {
        // No block since the last take: keep the readings, but not those of deleted tracks.
        out.tracks.retain(|id, _| source.tracks.contains_key(id));
        return;
    }
    copy_meter_snapshot(source, out);
    source.reset_levels();
}

pub(crate) fn copy_meter_snapshot(source: &MeterSnapshot, out: &mut MeterSnapshot) {
    out.tracks.retain(|id, _| source.tracks.contains_key(id));
    for (id, meter) in &source.tracks {
        out.tracks.entry(*id).or_default().copy_from(meter);
    }
    out.main.copy_from(&source.main);
}

/// Shape the shared meter snapshot for `session` on the control thread before the callback renders
/// it, so the audio thread never inserts or removes entries: deleted tracks are dropped here and new
/// ones added, while the levels not yet taken stay.
pub(crate) fn prepare_meter_layout(shared: &Shared, session: &Session) {
    shared.meters.lock().unwrap_or_else(PoisonError::into_inner).shape(session);
}

impl Side {
    pub(crate) fn capture_states(&mut self) -> PluginStates {
        self.states.iter_mut().filter(|(k, _)| !self.refused.contains(*k)).filter_map(|(&(t, slot), r)| Some((t, slot, r.save_state()?))).collect()
    }

    pub(crate) fn changed_states(&mut self) -> Vec<(TrackId, usize)> {
        self.editors.iter_mut().filter_map(|(k, e)| e.state_changed().then_some(*k)).collect()
    }

    pub(crate) fn capture_states_of(&mut self, slots: &[(TrackId, usize)]) -> PluginStates {
        self.states
            .iter_mut()
            .filter(|(k, _)| !self.refused.contains(*k) && slots.contains(k))
            .filter_map(|(&(t, slot), r)| Some((t, slot, r.save_state()?)))
            .collect()
    }

    /// A plugin that marks its state changed while it loads or restores it hasn't changed.
    pub(crate) fn add_editor(&mut self, key: (TrackId, usize), mut e: Box<dyn PluginEditor>) {
        let _ = e.state_changed();
        self.editors.insert(key, e);
    }

    pub(crate) fn open_editor(&mut self, key: (TrackId, usize)) -> Result<(), String> {
        let e = self.editors.get_mut(&key).ok_or_else(|| "this plugin has no editor (or it is not loaded)".to_string())?;
        e.open()?;
        self.refused.remove(&key);
        Ok(())
    }

    /// Creates the third-party instances `s` needs that the audio thread does not have yet.
    pub(crate) fn prepare(&mut self, s: &Session) -> Vec<PreparedInstance> {
        self.collect_garbage();
        let sr = s.sample_rate.as_f64() as f32;
        if (sr - self.rate).abs() > 0.5 {
            // The audio thread rebuilds its engine: everything is created anew.
            let keys: Vec<(TrackId, usize)> = self.shadow.keys().copied().collect();
            for k in keys {
                self.forget(k);
            }
            self.rate = sr;
        }
        let specs = soundcraft_mix::instance_specs(s);
        let stale: Vec<(TrackId, usize)> = self.shadow.keys().filter(|k| !specs.iter().any(|x| (x.track, x.slot) == **k)).copied().collect();
        for k in stale {
            self.forget(k);
        }
        let mut out = Vec::new();
        for spec in specs {
            let key = (spec.track, spec.slot);
            if self.shadow.get(&key).is_some_and(|(id, ch, _)| *id == spec.id && *ch == spec.ch) {
                continue;
            }
            self.forget(key);
            let made = soundcraft_mix::create_instance(&spec.id, spec.state.as_deref(), sr, BLOCK, spec.ch);
            log::debug!("plugin instance for {:?} slot {}: {} ({} ch, created: {})", spec.track, spec.slot, spec.id, spec.ch, made.is_some());
            self.shadow.insert(key, (spec.id.clone(), spec.ch, made.is_some()));
            if let Some(soundcraft_mix::NewInstance { mut plugin, refused_state }) = made {
                if let Some(e) = plugin.editor() {
                    self.add_editor(key, e);
                }
                if let Some(r) = plugin.state_reader() {
                    self.states.insert(key, r);
                }
                if refused_state {
                    self.refused.insert(key);
                }
                out.push(PreparedInstance { track: spec.track, slot: spec.slot, id: spec.id, ch: spec.ch, plugin });
            }
        }
        out
    }
}

/// ITU stereo fold-down gains for each main channel of a session.
fn fold_gains(s: &Session) -> [(f32, f32); MAX_CHANNELS] {
    let mut g = [(0.0, 0.0); MAX_CHANNELS];
    let f = s.main_format();
    let sp = f.speakers();
    if sp.len() == f.channels() {
        for (o, x) in g.iter_mut().zip(sp.iter()) {
            *o = itu_stereo_fold(*x);
        }
    } else {
        // Ambisonics: W only, at -3 dB per side.
        g[0] = (std::f32::consts::FRAC_1_SQRT_2, std::f32::consts::FRAC_1_SQRT_2);
    }
    g
}
