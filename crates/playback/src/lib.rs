//! Realtime playback through the system audio device (cpal: CoreAudio, WASAPI, ALSA/Pulse/
//! PipeWire via ALSA, JACK, WebAudio).
//!
//! The UI thread owns a [`Player`]; the audio callback owns the [`MixEngine`] and a snapshot of
//! the session (`Arc<Session>`) that the UI swaps whenever the document changes. Commands go to
//! the callback through a channel; position and meters come back through atomics and a
//! `try_lock`ed snapshot, so the callback never blocks. When no device is available the player
//! falls back to a silent clock thread so the transport, meters and automation still run.
//!
//! Surround: the device is opened with as many output channels as it offers, up to the width of
//! the session's main format. Main channels (SMPTE/WAV order) go to device channels in order; when
//! the device has fewer channels than the main mix, surround is folded down to stereo (ITU-R
//! BS.775). A mono device receives the average of the stereo channels, before output clipping.
//!
//! Plugins and threads: hosted third-party plugins (CLAP/VST3) are created, given their stored
//! state and prepared on the thread that calls [`Player::update_session`] (the UI thread), and
//! travel to the audio thread together with the session snapshot that needs them; the mix
//! engine adopts them (see `soundcraft_mix`'s crate docs). Instances the engine retires, and the
//! session snapshots the callback replaces, come back on return channels and are dropped on the UI
//! thread ([`Player::idle`]). Plugin editors are driven, and plugin states for saving read
//! ([`Player::capture_states`]), on the UI thread
//! through handles taken from each instance before it leaves ([`Player::open_editor`]).
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(not(feature = "plugin-hosts"))]
mod control;
pub mod devices;
pub mod record;

pub use devices::AudioConfig;

#[cfg(not(feature = "plugin-hosts"))]
use control::{Receiver, ReturnReceiver, ReturnSender, Sender, channel, return_channel};
use soundcraft_dsp::{Plugin, PluginEditor, PluginStateReader};
use soundcraft_mix::{MAX_CHANNELS, MixEngine, PreparedInstance, StripMeter, itu_stereo_fold, main_channels};
use soundcraft_model::{Session, TrackId, TrackKind};
use soundcraft_time::{Range, Samples};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, Ordering};
#[cfg(feature = "plugin-hosts")]
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::mpsc::{Receiver as MpscReceiver, SyncSender, sync_channel};
#[cfg(feature = "plugin-hosts")]
struct ReturnSender<T>(SyncSender<T>);
#[cfg(feature = "plugin-hosts")]
struct ReturnReceiver<T>(MpscReceiver<T>);
#[cfg(feature = "plugin-hosts")]
fn return_channel<T>(capacity: usize) -> (ReturnSender<T>, ReturnReceiver<T>) {
    let (tx, rx) = sync_channel(capacity);
    (ReturnSender(tx), ReturnReceiver(rx))
}
#[cfg(feature = "plugin-hosts")]
impl<T> ReturnSender<T> {
    fn try_send(&mut self, value: T) -> Result<(), std::sync::mpsc::TrySendError<T>> {
        self.0.try_send(value)
    }
}
#[cfg(feature = "plugin-hosts")]
impl<T> ReturnReceiver<T> {
    fn try_recv(&mut self) -> Result<T, std::sync::mpsc::TryRecvError> {
        self.0.try_recv()
    }
}
#[cfg(feature = "plugin-hosts")]
type SessionMixer = Vec<PreparedInstance>;
#[cfg(not(feature = "plugin-hosts"))]
type SessionMixer = Box<MixEngine>;

struct RetiredSession {
    _session: Option<Arc<Session>>,
    _mix: Option<Box<MixEngine>>,
    _input: Option<Arc<record::InputRing>>,
}

fn session_command(s: Arc<Session>, instances: Vec<PreparedInstance>, pos: Samples) -> Cmd {
    #[cfg(feature = "plugin-hosts")]
    {
        let _ = pos;
        Cmd::Session(s, instances)
    }
    #[cfg(not(feature = "plugin-hosts"))]
    {
        let _ = instances;
        let mut mix = Box::new(new_engine(s.sample_rate.as_f64() as f32));
        mix.prepare_realtime(&s, pos);
        Cmd::Session(s, mix)
    }
}
use std::sync::{Arc, Mutex, PoisonError};

const BLOCK: usize = 512;
/// Bound callback-side command draining so a burst of UI updates cannot monopolize a device block.
const MAX_CONTROL_COMMANDS_PER_CALLBACK: usize = 16;

/// Retired plugin batches are collected by the UI thread. A full queue makes the callback retain
/// the batch and temporarily render silence until the UI has room to collect it.
const RETIRED_PLUGINS: usize = 16;

/// Room for replaced snapshots/mixers/input rings; `update_session` collects them before sending.
#[cfg(feature = "plugin-hosts")]
const OLD_SESSIONS: usize = 16;
#[cfg(not(feature = "plugin-hosts"))]
const OLD_SESSIONS: usize = 4;

/// The realtime audio thread marker lives in the mix engine, which carries it into the strips it
/// renders on worker threads; the cpal callbacks here set it.
pub use soundcraft_mix::{mark_audio_thread, on_audio_thread};

/// Meter values for the UI.
#[derive(Debug, Clone, Default)]
pub struct MeterSnapshot {
    pub tracks: HashMap<TrackId, StripMeter>,
    pub main: StripMeter,
}

/// Plugin states read from the live instances: `(track, slot or INSTRUMENT_SLOT, state)`.
pub type PluginStates = Vec<(TrackId, usize, Vec<u8>)>;

/// A parameter edit made in a plugin's own editor: `(track, slot, param id, value)`.
pub type EditorEdit = (TrackId, usize, String, f32);

enum Cmd {
    /// A new snapshot plus the third-party instances it needs (created on the UI thread).
    Session(Arc<Session>, SessionMixer),
    /// Where the callback sends the snapshots it replaces (freeing one takes thousands of frees).
    OldSessions(ReturnSender<RetiredSession>),
    Input(Arc<record::InputRing>),
    Recording(bool),
    Play {
        from: Samples,
        end: Option<Samples>,
        looped: Option<Range>,
    },
    Stop,
    Locate(Samples),
}

struct Shared {
    position: AtomicI64,
    playing: AtomicBool,
    /// Speed × 1000.
    speed: AtomicU32,
    output_clip_events: AtomicU32,
    monitor_gain: AtomicU32,
    monitor_use_rack: AtomicBool,
    direct_monitor_enabled: AtomicBool,
    feedback_protection: AtomicBool,
    speaker_protection: AtomicBool,
    meters: Mutex<MeterSnapshot>,
}

/// What stopped the audio output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTrouble {
    /// A block panicked; the output stays silent.
    Bug,
    /// The device went away.
    Gone,
    /// The device stopped asking for audio while playing.
    Stalled,
    /// The bounded mobile command queue filled; output stays silent until reconnection.
    #[cfg(not(feature = "plugin-hosts"))]
    Overloaded,
}

/// How the audio output is doing, for the UI.
#[derive(Default)]
struct Fault {
    /// A block panicked: output stays silent from then on.
    failed: AtomicBool,
    gone: AtomicBool,
    /// Blocks so far, to tell a device that stopped asking for audio.
    beats: std::sync::atomic::AtomicU64,
    /// The UI has been told.
    reported: AtomicBool,
}

impl Fault {
    /// What stopped the output, once.
    fn trouble(&self, stalled: bool) -> Option<OutputTrouble> {
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
const STALL: std::time::Duration = std::time::Duration::from_secs(2);

/// True once the block count `beats` has stood still for `STALL` while playing; `seen` is the
/// count last seen and since when.
#[cfg(not(target_arch = "wasm32"))]
fn stalled(seen: &mut Option<(u64, std::time::Instant)>, beats: u64, playing: bool) -> bool {
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
fn guarded_block(fault: &Fault, shared: &Shared, out: &mut [f32], fill: impl FnOnce(&mut [f32])) {
    fault.beats.fetch_add(1, Ordering::Relaxed);
    if fault.failed.load(Ordering::Relaxed) || std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| fill(out))).is_err() {
        fault.failed.store(true, Ordering::Relaxed);
        shared.playing.store(false, Ordering::Relaxed);
        out.fill(0.0);
    }
}

/// State owned by the audio callback.
struct AudioState {
    rx: Receiver<Cmd>,
    /// A second session command is held until the next callback so only one structural sync runs
    /// during any single device block.
    deferred: Option<Cmd>,
    /// Replaced state that cannot fit in the return ring stays owned here. Further commands
    /// wait for retirement capacity so another update can never overwrite and destroy it.
    pending_old_session: Option<RetiredSession>,
    /// A retired plugin batch that could not fit in the bounded return queue stays owned here.
    pending_garbage: Option<Vec<Box<dyn Plugin>>>,
    shared: Arc<Shared>,
    session: Arc<Session>,
    mix: Box<MixEngine>,
    pos: Samples,
    playing: bool,
    end: Option<Samples>,
    looped: Option<Range>,
    /// Rendered frames waiting to be consumed (planar, main channels) and the fractional read index.
    pending: Vec<Vec<f32>>,
    /// Active channel count within the fixed-capacity `pending` workspace.
    pending_channels: usize,
    /// Stereo fold-down gains per main channel (used when the device has fewer channels).
    fold: [(f32, f32); MAX_CHANNELS],
    pending_len: usize,
    read: f64,
    speaker_guard: SpeakerGuard,
    feedback_guard: FeedbackGuard,
    device_rate: f64,
    input: Option<Arc<record::InputRing>>,
    recording: bool,
    /// Retired plugin instances go back to the UI thread to be dropped there.
    garbage: SyncSender<Vec<Box<dyn Plugin>>>,
    old_sessions: Option<ReturnSender<RetiredSession>>,
}

/// Callback-owned conservative detector for a loud, persistent near-sine input.
/// It ducks only the live monitor send; capture samples remain untouched.
#[derive(Default)]
struct FeedbackGuard {
    hot_blocks: u16,
    gain: f32,
    previous_rms: f32,
}

impl FeedbackGuard {
    fn update(&mut self, input: &[Vec<f32>], frames: usize, sample_rate: u32, active: bool) -> f32 {
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
struct SpeakerGuard {
    gain: f32,
}

impl SpeakerGuard {
    const CEILING: f32 = 0.891_250_9; // −1 dBFS

    fn process(&mut self, frame: &mut [f32], sample_rate: f64, enabled: bool) {
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
fn resonant_input(samples: &[f32], frames: usize, sample_rate: u32) -> bool {
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

fn new_engine(sr: f32) -> MixEngine {
    let mut m = MixEngine::new(sr, BLOCK);
    m.set_external_instances(true);
    m.metronome = true;
    // Reserve the maximum supported input layout for this engine instance. The active channel
    // count is carried separately so input-channel changes do not resize per-block buffers.
    m.input = (0..MAX_CHANNELS).map(|_| vec![0.0; BLOCK]).collect();
    m
}

impl AudioState {
    fn new(rx: Receiver<Cmd>, shared: Arc<Shared>, session: Arc<Session>, device_rate: f64, garbage: SyncSender<Vec<Box<dyn Plugin>>>) -> Self {
        let sr = session.sample_rate.as_f64() as f32;
        let nch = main_channels(&session);
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
            feedback_guard: FeedbackGuard { hot_blocks: 0, gain: 1.0, previous_rms: 0.0 },
            device_rate,
            input: None,
            recording: false,
        }
    }

    fn drain(&mut self) {
        if !self.flush_garbage() {
            self.shared.playing.store(self.playing, Ordering::Relaxed);
            return;
        }
        if let Some(old) = self.pending_old_session.take() {
            if let Some(tx) = &mut self.old_sessions {
                match tx.try_send(old) {
                    Ok(()) => {}
                    Err(std::sync::mpsc::TrySendError::Full(old) | std::sync::mpsc::TrySendError::Disconnected(old)) => {
                        self.pending_old_session = Some(old);
                        self.shared.playing.store(self.playing, Ordering::Relaxed);
                        return;
                    }
                }
            } else {
                self.pending_old_session = Some(old);
                self.shared.playing.store(self.playing, Ordering::Relaxed);
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
                        self.retire_state(RetiredSession { _session: None, _mix: None, _input: Some(old) });
                    }
                }
                Cmd::Recording(r) => self.recording = r,
                Cmd::Session(s, instances) => {
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
                    let old = RetiredSession { _session: Some(std::mem::replace(&mut self.session, s)), _mix: retired_mix, _input: None };
                    self.retire_state(old);
                }
                Cmd::OldSessions(tx) => self.old_sessions = Some(tx),
                Cmd::Play { from, end, looped } => {
                    self.pos = from;
                    self.end = end;
                    self.looped = looped.filter(|r| r.len() > 64);
                    self.playing = true;
                    self.pending_len = 0;
                    self.read = 0.0;
                    self.mix.reset();
                }
                Cmd::Stop => {
                    self.playing = false;
                    self.end = None;
                    self.looped = None;
                    self.mix.reset();
                }
                Cmd::Locate(at) => {
                    self.pos = at;
                    self.pending_len = 0;
                    self.read = 0.0;
                    self.mix.reset();
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
        self.shared.playing.store(self.playing, Ordering::Relaxed);
    }

    fn retire_state(&mut self, old: RetiredSession) {
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

    fn queue_garbage(&mut self, garbage: Vec<Box<dyn Plugin>>) {
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

    fn flush_garbage(&mut self) -> bool {
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
    fn render_next(&mut self, quantum: usize) {
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
        publish_mix_meters(&self.shared, &self.mix.main_meter, &self.mix.meters);
        if self.playing {
            self.pos += n as i64;
            self.shared.position.store(self.pos, Ordering::Relaxed);
        }
    }

    /// Fill an interleaved device buffer.
    fn fill(&mut self, out: &mut [f32], channels: usize) {
        #[cfg(not(feature = "plugin-hosts"))]
        if self.rx.overflowed() {
            self.playing = false;
            self.shared.playing.store(false, Ordering::Relaxed);
            out.fill(0.0);
            return;
        }
        self.drain();
        let channels = channels.max(1);
        let monitoring = self.input.is_some() && self.session.tracks.iter().any(|t| t.mixer.input_monitor || t.mixer.record_arm);
        if !self.playing && !monitoring {
            out.iter_mut().for_each(|x| *x = 0.0);
            if let Ok(mut m) = self.shared.meters.try_lock() {
                m.main.clear();
                for meter in m.tracks.values_mut() {
                    meter.clear();
                }
            }
            return;
        }
        let speed = f64::from(self.shared.speed.load(Ordering::Relaxed)) / 1000.0;
        let step = self.session.sample_rate.as_f64() / self.device_rate.max(1.0) * speed.clamp(0.1, 4.0);
        let callback_frames = out.len() / channels;
        let quantum = ((callback_frames as f64 * step).ceil() as usize).clamp(1, BLOCK);
        for frame in out.chunks_mut(channels) {
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
                let b = ch.and_then(|v| v.get(i0 + 1)).copied().unwrap_or(a);
                a + (b - a) * fr
            };
            if channels < nch {
                // Fold surround to stereo first; a mono device averages the two sides.
                let (mut l, mut r) = (0.0f32, 0.0f32);
                if sounding {
                    if nch == 2 {
                        (l, r) = (sample(0), sample(1));
                    } else {
                        for (c, (gl, gr)) in self.fold.iter().enumerate().take(nch) {
                            let v = sample(c);
                            l += v * gl;
                            r += v * gr;
                        }
                    }
                }
                for (c, o) in frame.iter_mut().enumerate() {
                    *o = match c {
                        0 if channels == 1 => {
                            let mono = (0.5 * l + 0.5 * r).clamp(-1.0, 1.0);
                            if mono.is_finite() { mono } else { 0.0 }
                        }
                        0 => l.clamp(-1.0, 1.0),
                        1 => r.clamp(-1.0, 1.0),
                        _ => 0.0,
                    };
                }
            } else {
                for (c, o) in frame.iter_mut().enumerate() {
                    *o = if c < nch.max(1) && sounding { sample(c).clamp(-1.0, 1.0) } else { 0.0 };
                }
            }
            self.speaker_guard.process(frame, self.device_rate, self.shared.speaker_protection.load(Ordering::Relaxed));
            self.read += step;
        }
    }
}

/// UI-thread bookkeeping of the third-party instances the audio thread holds.
struct Side {
    /// What was last created for each slot: (plugin id, channels, created ok).
    shadow: HashMap<(TrackId, usize), (String, usize, bool)>,
    /// Session sample rate the instances were prepared for.
    rate: f32,
    /// Editor handles of the live instances.
    editors: HashMap<(TrackId, usize), Box<dyn PluginEditor>>,
    /// State readers of the live instances.
    states: BTreeMap<(TrackId, usize), Box<dyn PluginStateReader>>,
    /// Instances that refused their stored state: saves keep that state until the plugin's editor
    /// is opened.
    refused: HashSet<(TrackId, usize)>,
    garbage: MpscReceiver<Vec<Box<dyn Plugin>>>,
    old_sessions: Option<ReturnReceiver<RetiredSession>>,
}

impl Side {
    fn new(rate: f32, garbage: MpscReceiver<Vec<Box<dyn Plugin>>>) -> Side {
        Side { shadow: HashMap::new(), rate, editors: HashMap::new(), states: BTreeMap::new(), refused: HashSet::new(), garbage, old_sessions: None }
    }

    /// Drops retired instances (closing their editors first, inside the plugin's drop) and
    /// replaced session snapshots.
    fn collect_garbage(&mut self) {
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

    fn return_old_sessions(&mut self, tx: &Sender<Cmd>) {
        let (otx, orx) = return_channel(OLD_SESSIONS);
        self.old_sessions = Some(orx);
        let _ = tx.send(Cmd::OldSessions(otx));
    }

    fn forget(&mut self, key: (TrackId, usize)) {
        self.shadow.remove(&key);
        self.states.remove(&key);
        self.refused.remove(&key);
        if let Some(mut e) = self.editors.remove(&key) {
            e.close();
        }
    }
}

/// The transport and audio device.
pub struct Player {
    shared: Arc<Shared>,
    fault: Arc<Fault>,
    /// The block count last seen while playing, and since when.
    #[cfg(not(target_arch = "wasm32"))]
    watch: Mutex<Option<(u64, std::time::Instant)>>,
    tx: Sender<Cmd>,
    side: Mutex<Side>,
    #[cfg(not(target_arch = "wasm32"))]
    _stream: Option<cpal::Stream>,
    #[cfg(target_arch = "wasm32")]
    _stream: Option<cpal::Stream>,
    pub device_name: String,
    pub device_rate: u32,
    /// Output channels the device was opened with (0 for the silent clock).
    pub device_channels: u16,
    /// The device buffer in frames when one was chosen (`None`: the device default).
    pub device_buffer: Option<u32>,
    /// True when no audio device could be opened (silent clock).
    pub silent: bool,
}

impl Player {
    /// Open the default output device. Never fails: without a device it runs a silent clock.
    pub fn new(session: Arc<Session>) -> Player {
        Player::with_config(session, &AudioConfig::default())
    }

    /// Open the output device `cfg` chooses (system, device, rate, buffer size), falling back to the
    /// defaults for whatever is unavailable. Never fails: without a device it runs a silent clock.
    pub fn with_config(session: Arc<Session>, cfg: &AudioConfig) -> Player {
        let p = Player::open(Arc::clone(&session), cfg);
        p.side().return_old_sessions(&p.tx);
        // Hand the initial session's third-party plugins over.
        p.update_session(session);
        p
    }

    /// What stopped the audio output, once: a bug in the engine, or the device went away or
    /// stopped asking for audio while playing.
    pub fn trouble(&self) -> Option<OutputTrouble> {
        #[cfg(not(feature = "plugin-hosts"))]
        if self.tx.overflowed() && !self.fault.reported.swap(true, Ordering::Relaxed) {
            return Some(OutputTrouble::Overloaded);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let stalled = {
            let playing = self.shared.playing.load(Ordering::Relaxed) && !self.silent;
            stalled(&mut self.watch.lock().unwrap_or_else(PoisonError::into_inner), self.fault.beats.load(Ordering::Relaxed), playing)
        };
        #[cfg(target_arch = "wasm32")]
        let stalled = false;
        self.fault.trouble(stalled)
    }

    fn side(&self) -> std::sync::MutexGuard<'_, Side> {
        self.side.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn open(session: Arc<Session>, cfg: &AudioConfig) -> Player {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        prepare_meter_layout(&shared, &session);
        let want = session.sample_rate.hz();
        let want_ch = u16::try_from(main_channels(&session)).unwrap_or(2);
        let side_rate = session.sample_rate.as_f64() as f32;
        let new_side = |garbage: MpscReceiver<Vec<Box<dyn Plugin>>>| Mutex::new(Side::new(side_rate, garbage));
        match open_device(want, want_ch, cfg) {
            Ok((device, config, format, name)) => {
                // The chosen buffer size first; if the device refuses it, its default.
                let mut attempts = vec![config.clone()];
                if config.buffer_size != cpal::BufferSize::Default {
                    attempts.push(cpal::StreamConfig { buffer_size: cpal::BufferSize::Default, ..config });
                }
                let opened = first_started(
                    attempts,
                    |config: &cpal::StreamConfig| -> Result<Opened, cpal::BuildStreamError> {
                        let rate = config.sample_rate.0;
                        let channels = usize::from(config.channels);
                        let (tx, rx) = channel();
                        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
                        let state = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&session), f64::from(rate), gtx);
                        let fault = Arc::new(Fault::default());
                        let stream = match format {
                            cpal::SampleFormat::I32 => build_output::<i32>(&device, config, state, channels, &fault, &shared),
                            cpal::SampleFormat::I16 => build_output::<i16>(&device, config, state, channels, &fault, &shared),
                            _ => build_output::<f32>(&device, config, state, channels, &fault, &shared),
                        }?;
                        Ok(Opened { stream, tx, grx, fault })
                    },
                    |opened| start_stream(&opened.stream),
                );
                match opened {
                    Some((config, Opened { stream, tx, grx, fault })) => {
                        let rate = config.sample_rate.0;
                        let channels = usize::from(config.channels);
                        let buffer = match config.buffer_size {
                            cpal::BufferSize::Fixed(n) => Some(n),
                            cpal::BufferSize::Default => None,
                        };
                        log::info!(
                            "audio device: {name} @ {rate} Hz, {channels} ch, {format}, buffer {}",
                            buffer.map_or("default".to_string(), |n| n.to_string())
                        );
                        Player {
                            shared,
                            fault,
                            #[cfg(not(target_arch = "wasm32"))]
                            watch: Mutex::new(None),
                            tx,
                            side: new_side(grx),
                            _stream: Some(stream),
                            device_name: name,
                            device_rate: rate,
                            device_channels: config.channels,
                            device_buffer: buffer,
                            silent: false,
                        }
                    }
                    None => {
                        log::warn!("no usable audio stream; using a silent clock");
                        let (tx, rx) = channel();
                        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
                        Player::silent_clock(shared, tx, rx, want, new_side(grx), gtx)
                    }
                }
            }
            Err(e) => {
                log::warn!("no audio output: {e}; using a silent clock");
                let (tx, rx) = channel();
                let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
                Player::silent_clock_with(shared, tx, rx, Arc::clone(&session), new_side(grx), gtx)
            }
        }
    }

    fn silent_clock(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        rx: Receiver<Cmd>,
        rate: u32,
        side: Mutex<Side>,
        gtx: SyncSender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        let s = Arc::new(Session::new("Untitled", soundcraft_time::SampleRate::new(rate).unwrap_or_default()));
        Player::silent_clock_with(shared, tx, rx, s, side, gtx)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn silent_clock_with(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        rx: Receiver<Cmd>,
        session: Arc<Session>,
        side: Mutex<Side>,
        gtx: SyncSender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        let rate = session.sample_rate.hz();
        let mut state = AudioState::new(rx, Arc::clone(&shared), session, f64::from(rate), gtx);
        let fault = Arc::new(Fault::default());
        let failed = Arc::clone(&fault);
        let spawn = std::thread::Builder::new().name("soundcraft-clock".into()).spawn(move || {
            let mut buf = vec![0.0f32; 4096];
            let mut last = std::time::Instant::now();
            loop {
                std::thread::sleep(std::time::Duration::from_millis(10));
                let now = std::time::Instant::now();
                let frames = ((now - last).as_secs_f64() * state.device_rate) as usize;
                last = now;
                let n = (frames * 2).min(buf.len());
                if let Some(b) = buf.get_mut(..n) {
                    let shared = Arc::clone(&state.shared);
                    guarded_block(&failed, &shared, b, |b| state.fill(b, 2));
                }
                if Arc::strong_count(&state.shared) <= 1 {
                    break;
                }
            }
        });
        if let Err(e) = spawn {
            log::warn!("clock thread failed: {e}");
        }
        Player {
            shared,
            fault,
            watch: Mutex::new(None),
            tx,
            side,
            _stream: None,
            device_name: "No audio device (silent)".into(),
            device_rate: rate,
            device_channels: 0,
            device_buffer: None,
            silent: true,
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn silent_clock_with(
        shared: Arc<Shared>,
        tx: Sender<Cmd>,
        _rx: Receiver<Cmd>,
        session: Arc<Session>,
        side: Mutex<Side>,
        _gtx: SyncSender<Vec<Box<dyn Plugin>>>,
    ) -> Player {
        Player {
            shared,
            fault: Arc::default(),
            tx,
            side,
            _stream: None,
            device_name: "No audio device".into(),
            device_rate: session.sample_rate.hz(),
            device_channels: 0,
            device_buffer: None,
            silent: true,
        }
    }

    /// Send a session from the control thread. Hosted builds prepare new third-party instances;
    /// built-in-only builds prepare the whole replacement mixer and its compensation storage.
    /// A full mobile command ring latches an output fault rather than silently losing edits.
    pub fn update_session(&self, s: Arc<Session>) {
        let instances = self.side().prepare(&s);
        prepare_meter_layout(&self.shared, &s);
        let _ = self.tx.send(session_command(s, instances, self.position()));
    }

    /// Reads the state of every live third-party plugin, for saving the session.
    pub fn capture_states(&self) -> PluginStates {
        self.side().capture_states()
    }

    /// Per-UI-frame housekeeping: drops retired instances and services plugin editors. Returns
    /// the parameter edits made in open editors since the last call.
    pub fn idle(&self) -> Vec<EditorEdit> {
        let mut side = self.side();
        side.collect_garbage();
        let mut out = Vec::new();
        for (&(t, slot), e) in side.editors.iter_mut() {
            for (id, v) in e.idle() {
                out.push((t, slot, id, v));
            }
        }
        out
    }

    /// True when the plugin in this slot has its own editor.
    pub fn has_editor(&self, track: TrackId, slot: usize) -> bool {
        self.side().editors.contains_key(&(track, slot))
    }

    /// Opens the plugin's own editor window. Main thread only.
    pub fn open_editor(&self, track: TrackId, slot: usize) -> Result<(), String> {
        self.side().open_editor((track, slot))
    }

    pub fn close_editor(&self, track: TrackId, slot: usize) {
        if let Some(e) = self.side().editors.get_mut(&(track, slot)) {
            e.close();
        }
    }

    /// Closes every plugin editor window (main thread). Call before replacing the player: its
    /// instances may be dropped on the audio thread, where editor windows cannot be closed.
    pub fn close_editors(&self) {
        for e in self.side().editors.values_mut() {
            e.close();
        }
    }

    pub fn editor_open(&self, track: TrackId, slot: usize) -> bool {
        self.side().editors.get(&(track, slot)).is_some_and(|e| e.is_open())
    }

    pub fn any_editor_open(&self) -> bool {
        self.side().editors.values().any(|e| e.is_open())
    }

    /// Shows a parameter change made elsewhere in the plugin's open editor.
    pub fn editor_set_param(&self, track: TrackId, slot: usize, id: &str, value: f32) {
        if let Some(e) = self.side().editors.get_mut(&(track, slot)) {
            e.set_param(id, value);
        }
    }

    /// The plugin's own text for a parameter value when SoundCraft's format would be wrong for it
    /// (see `PluginEditor::value_text`). Main thread only.
    pub fn editor_value_text(&self, track: TrackId, slot: usize, id: &str, value: f32) -> Option<String> {
        self.side().editors.get_mut(&(track, slot))?.value_text(id, value)
    }

    /// Start playback at `from`, stopping at `end`, or looping `looped`.
    pub fn play(&self, from: Samples, end: Option<Samples>, looped: Option<Range>) {
        self.shared.position.store(from, Ordering::Relaxed);
        self.shared.playing.store(true, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Play { from, end, looped });
    }

    /// Feed live input (from a `record::Recorder`) into the mix for input monitoring.
    pub fn set_input(&self, ring: Arc<record::InputRing>) {
        let _ = self.tx.send(Cmd::Input(ring));
    }

    /// Set post-insert listening gain without rebuilding DSP or queuing an audio command.
    pub fn set_monitor_gain(&self, gain: f32) {
        let gain = if gain.is_finite() { gain.clamp(0.0, 4.0) } else { 0.0 };
        self.shared.monitor_gain.store(gain.to_bits(), Ordering::Relaxed);
    }

    /// Include the track's inserts in live input monitoring (false is a dry listen).
    pub fn set_monitor_use_rack(&self, enabled: bool) {
        self.shared.monitor_use_rack.store(enabled, Ordering::Relaxed);
    }

    /// Enable or disable input monitoring while retaining its configured level.
    pub fn set_direct_monitor_enabled(&self, enabled: bool) {
        self.shared.direct_monitor_enabled.store(enabled, Ordering::Relaxed);
    }

    /// Enable gradual monitor-send suppression for sustained resonant input.
    pub fn set_feedback_protection(&self, enabled: bool) {
        self.shared.feedback_protection.store(enabled, Ordering::Relaxed);
    }

    /// Enable the output safety limiter at −1 dBFS.
    pub fn set_speaker_protection(&self, enabled: bool) {
        self.shared.speaker_protection.store(enabled, Ordering::Relaxed);
    }

    pub fn set_recording(&self, on: bool) {
        let _ = self.tx.send(Cmd::Recording(on));
    }

    pub fn stop(&self) {
        self.shared.playing.store(false, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Stop);
    }

    pub fn locate(&self, at: Samples) {
        self.shared.position.store(at, Ordering::Relaxed);
        let _ = self.tx.send(Cmd::Locate(at));
    }

    pub fn set_speed(&self, speed: f32) {
        let v = if speed.is_finite() { (speed.clamp(0.1, 4.0) * 1000.0) as u32 } else { 1000 };
        self.shared.speed.store(v, Ordering::Relaxed);
    }

    /// Playback rate last requested, 1.0 at full speed.
    pub fn speed(&self) -> f32 {
        self.shared.speed.load(Ordering::Relaxed) as f32 / 1000.0
    }

    pub fn position(&self) -> Samples {
        self.shared.position.load(Ordering::Relaxed)
    }

    pub fn is_playing(&self) -> bool {
        self.shared.playing.load(Ordering::Relaxed)
    }

    /// Counts master blocks reaching digital full scale. Updated before best-effort meter
    /// publishing, so UI polling or meter-lock contention cannot hide an overload.
    /// Wraps only after 2^32 overloaded blocks; a new Player starts at zero.
    pub fn output_clip_events(&self) -> u32 {
        self.shared.output_clip_events.load(Ordering::Relaxed)
    }

    pub fn meters(&self) -> MeterSnapshot {
        self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Copy meters into reusable UI storage. Once track/channel counts are stable this performs
    /// no allocation, unlike cloning a new snapshot every video frame.
    pub fn copy_meters(&self, out: &mut MeterSnapshot) {
        let source = self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner);
        copy_meter_snapshot(&source, out);
    }
}

fn copy_meter_snapshot(source: &MeterSnapshot, out: &mut MeterSnapshot) {
    out.tracks.retain(|id, _| source.tracks.contains_key(id));
    for (id, meter) in &source.tracks {
        out.tracks.entry(*id).or_default().copy_from(meter);
    }
    out.main.copy_from(&source.main);
}

/// Publish into the meter storage prepared by `prepare_meter_layout`. Tracks missing from this
/// snapshot can occur briefly while session updates cross the callback; skip them rather than
/// allocating a new map entry on the audio thread.
fn publish_mix_meters(shared: &Shared, main: &StripMeter, tracks: &HashMap<TrackId, StripMeter>) {
    if main.peaks.iter().any(|peak| peak.is_finite() && *peak >= 1.0) {
        shared.output_clip_events.fetch_add(1, Ordering::Relaxed);
    }
    if let Ok(mut meters) = shared.meters.try_lock() {
        meters.main.copy_from(main);
        for (id, source) in tracks {
            if let Some(destination) = meters.tracks.get_mut(id) {
                destination.copy_from(source);
            }
        }
    }
}

/// Allocate the meter map and channel vectors on the control thread before the callback renders a
/// changed session. Retain only live track ids here so the audio thread never inserts or removes
/// meter entries during normal playback or while stopped.
fn prepare_meter_layout(shared: &Shared, session: &Session) {
    let mut meters = shared.meters.lock().unwrap_or_else(PoisonError::into_inner);
    meters.main.peaks.resize(main_channels(session), 0.0);
    meters.main.clear();
    meters.tracks.retain(|id, _| session.track(*id).is_some());
    let missing_tracks = session.tracks.len().saturating_sub(meters.tracks.len());
    meters.tracks.reserve(missing_tracks);
    for track in &session.tracks {
        let channels = match track.kind {
            TrackKind::Master => main_channels(session),
            TrackKind::Instrument => track.channels().clamp(2, MAX_CHANNELS),
            _ => track.channels().clamp(1, MAX_CHANNELS),
        };
        let meter = meters.tracks.entry(track.id).or_default();
        meter.peaks.resize(channels, 0.0);
        meter.clear();
    }
}

impl Side {
    fn capture_states(&mut self) -> PluginStates {
        self.states.iter_mut().filter(|(k, _)| !self.refused.contains(*k)).filter_map(|(&(t, slot), r)| Some((t, slot, r.save_state()?))).collect()
    }

    fn open_editor(&mut self, key: (TrackId, usize)) -> Result<(), String> {
        let e = self.editors.get_mut(&key).ok_or_else(|| "this plugin has no editor (or it is not loaded)".to_string())?;
        e.open()?;
        self.refused.remove(&key);
        Ok(())
    }

    /// Creates the third-party instances `s` needs that the audio thread does not have yet.
    fn prepare(&mut self, s: &Session) -> Vec<PreparedInstance> {
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
                    self.editors.insert(key, e);
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

/// Start a freshly built output stream. `false` when the device refuses to start it, so the caller
/// treats it like a stream that failed to build and tries the next configuration.
fn start_stream(stream: &impl cpal::traits::StreamTrait) -> bool {
    match stream.play() {
        Ok(()) => true,
        Err(e) => {
            log::warn!("audio stream failed to start: {e}");
            false
        }
    }
}

/// A built output stream and the channels and fault flag its callback shares with the player.
struct Opened {
    stream: cpal::Stream,
    tx: Sender<Cmd>,
    grx: MpscReceiver<Vec<Box<dyn Plugin>>>,
    fault: Arc<Fault>,
}

/// Build and start the attempts in order; the first that builds and starts wins. A build error or
/// a stream that will not start is logged, and the next attempt is tried. `None` when none works.
fn first_started<A: std::fmt::Debug, S, E: std::fmt::Display>(
    attempts: impl IntoIterator<Item = A>,
    mut build: impl FnMut(&A) -> Result<S, E>,
    start: impl Fn(&S) -> bool,
) -> Option<(A, S)> {
    for attempt in attempts {
        match build(&attempt) {
            Ok(built) if start(&built) => return Some((attempt, built)),
            // `start` has already logged why the stream would not start.
            Ok(_) => {}
            Err(e) => log::warn!("cannot open audio stream ({attempt:?}): {e}"),
        }
    }
    None
}

/// The output stream: the mix renders f32 into a scratch buffer allocated here (so the callback
/// never allocates), converted to the device's sample type.
fn build_output<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    mut state: AudioState,
    channels: usize,
    fault: &Arc<Fault>,
    shared: &Arc<Shared>,
) -> Result<cpal::Stream, cpal::BuildStreamError>
where
    T: cpal::SizedSample + cpal::FromSample<f32>,
{
    use cpal::traits::DeviceTrait;
    let frame = channels.max(1);
    let mut scratch = vec![0.0f32; 8192 * frame];
    let chunk = scratch.len();
    let (failed, guard, gone) = (Arc::clone(fault), Arc::clone(shared), Arc::clone(fault));
    device.build_output_stream(
        config,
        move |data: &mut [T], _: &cpal::OutputCallbackInfo| {
            mark_audio_thread();
            for out in data.chunks_mut(chunk) {
                let Some(s) = scratch.get_mut(..out.len()) else { return };
                guarded_block(&failed, &guard, s, |d| state.fill(d, channels));
                for (o, x) in out.iter_mut().zip(s.iter()) {
                    *o = T::from_sample(*x);
                }
            }
        },
        // cpal may call this on the audio thread (on ALSA in a loop while the device stays
        // broken): marked, so the logger hands the record off instead of writing it.
        move |e: cpal::StreamError| {
            mark_audio_thread();
            if matches!(e, cpal::StreamError::DeviceNotAvailable) {
                gone.gone.store(true, Ordering::Relaxed);
            }
            log::warn!("audio stream error: {e}");
        },
        None,
    )
}

/// Open the output device `cfg` chooses (else the default) at the chosen rate (else the session
/// rate) if possible, in the best sample format it offers (f32, else 32- or 16-bit integers),
/// with as many channels as it offers up to `want_ch` (at least stereo when available), and the
/// chosen buffer size when the device allows it.
fn open_device(want_rate: u32, want_ch: u16, cfg: &AudioConfig) -> Result<(cpal::Device, cpal::StreamConfig, cpal::SampleFormat, String), String> {
    use cpal::traits::DeviceTrait;
    let host = devices::host(cfg);
    let device = devices::device(&host, cfg.output.as_deref(), true).ok_or_else(|| "no output device".to_string())?;
    #[allow(deprecated)]
    let name = device.name().unwrap_or_else(|_| "Audio device".into());
    // Prefer a config at the chosen (else session) rate; else the device default (we resample).
    let want_ch = want_ch.max(2);
    let rates: Vec<u32> = cfg.sample_rate.into_iter().chain(Some(want_rate)).collect();
    let mut chosen: Option<(cpal::StreamConfig, cpal::SupportedBufferSize, cpal::SampleFormat)> = None;
    // Better: a preferred format first, then more channels.
    let better = |c: &cpal::SupportedStreamConfigRange, x: &(cpal::StreamConfig, cpal::SupportedBufferSize, cpal::SampleFormat)| {
        let (rc, rx) = (devices::format_rank(c.sample_format()).unwrap_or(usize::MAX), devices::format_rank(x.2).unwrap_or(usize::MAX));
        rc < rx || (rc == rx && c.channels() > x.0.channels)
    };
    if let Ok(configs) = device.supported_output_configs() {
        let configs: Vec<cpal::SupportedStreamConfigRange> = configs.collect();
        for rate in rates {
            for c in &configs {
                if devices::format_rank(c.sample_format()).is_some()
                    && c.min_sample_rate().0 <= rate
                    && c.max_sample_rate().0 >= rate
                    && c.channels() >= 2
                    && c.channels() <= want_ch
                    && chosen.as_ref().is_none_or(|x| better(c, x))
                {
                    chosen = Some((c.with_sample_rate(cpal::SampleRate(rate)).config(), *c.buffer_size(), c.sample_format()));
                }
            }
            if chosen.is_some() {
                break;
            }
        }
    }
    let (config, format) = match chosen {
        Some((mut c, buffers, format)) => {
            c.buffer_size = devices::buffer_size(cfg.buffer_size, &buffers);
            (c, format)
        }
        None => {
            let d = device.default_output_config().map_err(|e| e.to_string())?;
            if devices::format_rank(d.sample_format()).is_none() {
                return Err(format!("unsupported device sample format {:?}", d.sample_format()));
            }
            let buffers = *d.buffer_size();
            let format = d.sample_format();
            let mut c = d.config();
            c.buffer_size = devices::buffer_size(cfg.buffer_size, &buffers);
            (c, format)
        }
    };
    Ok((device, config, format, name))
}

#[cfg(test)]
mod tests {
    fn test_channel<T>() -> (super::Sender<T>, super::Receiver<T>) {
        #[cfg(feature = "plugin-hosts")]
        {
            super::channel()
        }
        #[cfg(not(feature = "plugin-hosts"))]
        {
            super::control::channel_with_capacity(128)
        }
    }
    fn test_session(s: std::sync::Arc<soundcraft_model::Session>, instances: Vec<soundcraft_mix::PreparedInstance>) -> super::Cmd {
        super::session_command(s, instances, 0)
    }

    use super::*;

    /// A stream that builds; `starts` says whether the device lets it play (it can refuse after a
    /// reconnect).
    struct Fake {
        starts: bool,
    }

    impl cpal::traits::StreamTrait for Fake {
        fn play(&self) -> Result<(), cpal::PlayStreamError> {
            if self.starts { Ok(()) } else { Err(cpal::PlayStreamError::DeviceNotAvailable) }
        }
        fn pause(&self) -> Result<(), cpal::PauseStreamError> {
            Ok(())
        }
    }

    #[test]
    fn a_stream_that_will_not_start_is_reported_as_failed() {
        assert!(!start_stream(&Fake { starts: false }));
        assert!(start_stream(&Fake { starts: true }));
    }

    #[test]
    fn the_first_config_that_starts_wins_and_later_ones_are_not_tried() {
        let mut tried = Vec::new();
        let won = first_started(
            [1u32, 2, 3],
            |&n| {
                tried.push(n);
                Ok::<_, String>(Fake { starts: n != 1 })
            },
            start_stream,
        );
        assert_eq!(won.map(|(n, _)| n), Some(2));
        assert_eq!(tried, vec![1, 2]);
    }

    #[test]
    fn a_config_that_fails_to_build_falls_through_to_the_next() {
        let won =
            first_started([1u32, 2], |&n| if n == 1 { Err("no such buffer size".to_string()) } else { Ok(Fake { starts: true }) }, start_stream);
        assert_eq!(won.map(|(n, _)| n), Some(2));
    }

    #[test]
    fn no_config_that_starts_yields_none_so_the_caller_falls_back_to_a_silent_clock() {
        let mut tried = 0;
        let won = first_started(
            [1u32, 2],
            |_| {
                tried += 1;
                Ok::<_, String>(Fake { starts: false })
            },
            start_stream,
        );
        assert!(won.is_none());
        assert_eq!(tried, 2);
    }

    #[test]
    fn reusable_meter_snapshot_updates_and_drops_stale_tracks() {
        let keep = TrackId(1);
        let stale = TrackId(2);
        let mut source = MeterSnapshot::default();
        source.tracks.insert(keep, StripMeter { peak: [0.5, 0.25], peaks: vec![0.5, 0.25], gain_reduction_db: 3.0, ..StripMeter::default() });
        source.main.peaks = vec![0.75, 0.5];
        let mut out = MeterSnapshot::default();
        out.tracks.insert(stale, StripMeter::default());
        copy_meter_snapshot(&source, &mut out);
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.tracks[&keep], source.tracks[&keep]);
        assert_eq!(out.main, source.main);
    }

    #[test]
    fn callback_meter_publication_skips_unprepared_tracks_without_inserting() {
        let track = TrackId(7);
        let shared = Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        };
        let mut source = HashMap::new();
        source.insert(track, StripMeter { peak: [0.5, 0.25], peaks: vec![0.5, 0.25], ..StripMeter::default() });

        publish_mix_meters(&shared, &StripMeter::default(), &source);

        assert!(shared.meters.lock().unwrap().tracks.is_empty());
    }

    #[test]
    fn meter_layout_is_resized_and_pruned_on_the_control_thread() {
        let mut session = soundcraft_model::Session::default();
        let kept = session.add_track(TrackKind::Audio, soundcraft_model::ChannelFormat::Mono, None);
        let removed = session.add_track(TrackKind::Audio, soundcraft_model::ChannelFormat::Stereo, None);
        let shared = Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        };

        prepare_meter_layout(&shared, &session);
        let initial_capacity = shared.meters.lock().unwrap().tracks[&kept].peaks.capacity();
        assert_eq!(shared.meters.lock().unwrap().tracks[&kept].peaks.len(), 1);
        assert_eq!(shared.meters.lock().unwrap().tracks[&removed].peaks.len(), 2);

        session.tracks.retain(|track| track.id != removed);
        prepare_meter_layout(&shared, &session);
        let meters = shared.meters.lock().unwrap();
        assert!(meters.tracks.contains_key(&kept));
        assert!(!meters.tracks.contains_key(&removed));
        assert_eq!(meters.tracks[&kept].peaks.capacity(), initial_capacity);
    }

    #[test]
    fn a_block_that_panics_goes_silent_and_is_reported_once() {
        let shared = Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(true),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        };
        let fault = Fault::default();
        let mut out = [0.5f32; 8];
        guarded_block(&fault, &shared, &mut out, |o| o.fill(0.25));
        assert_eq!((out, fault.trouble(false)), ([0.25; 8], None));
        guarded_block(&fault, &shared, &mut out, |_| std::panic::panic_any("a bug in the mix"));
        guarded_block(&fault, &shared, &mut out, |o| o.fill(0.25));
        assert_eq!(out, [0.0; 8]);
        assert!(!shared.playing.load(Ordering::Relaxed));
        assert_eq!((fault.trouble(false), fault.trouble(false)), (Some(OutputTrouble::Bug), None));
        assert_eq!(fault.beats.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn a_lost_or_stalled_device_is_reported_once() {
        let ago = |seen: &mut Option<(u64, std::time::Instant)>| {
            if let Some(s) = seen {
                s.1 = std::time::Instant::now().checked_sub(STALL).unwrap();
            }
        };
        let mut seen = None;
        assert!(!stalled(&mut seen, 5, true));
        ago(&mut seen);
        assert!(stalled(&mut seen, 5, true));
        assert!(!stalled(&mut seen, 6, true), "it asked again");
        ago(&mut seen);
        assert!(!stalled(&mut seen, 6, false), "stopped, so nothing to ask for");
        assert_eq!(Fault::default().trouble(true), Some(OutputTrouble::Stalled));
        let fault = Fault::default();
        fault.gone.store(true, Ordering::Relaxed);
        assert_eq!((fault.trouble(false), fault.trouble(true)), (Some(OutputTrouble::Gone), None));
    }

    #[test]
    fn audio_state_plays_and_stops_at_end() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        tx.send(Cmd::Play { from: 0, end: Some(1000), looped: None }).unwrap();
        let mut out = vec![0.0f32; 4096];
        st.fill(&mut out, 2);
        assert!(shared.position.load(Ordering::Relaxed) >= 1000);
        assert!(!shared.playing.load(Ordering::Relaxed));
    }

    #[test]
    fn command_drain_is_bounded_and_keeps_commands_in_order() {
        let session = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let mut state = AudioState::new(rx, shared, session, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        for _ in 0..MAX_CONTROL_COMMANDS_PER_CALLBACK + 1 {
            tx.send(Cmd::Recording(true)).unwrap();
        }
        tx.send(Cmd::Recording(false)).unwrap();

        state.drain();
        assert!(state.recording, "the first drain handles only its bounded prefix");
        state.drain();
        assert!(!state.recording, "the later command remains queued in order");
    }

    #[test]
    fn output_overload_survives_meter_contention_and_later_quiet_blocks() {
        let shared = Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        };
        let guard = shared.meters.lock().unwrap();
        let mut main = StripMeter { peaks: vec![0.25, 1.1], ..Default::default() };
        publish_mix_meters(&shared, &main, &HashMap::new());
        assert_eq!(shared.output_clip_events.load(Ordering::Relaxed), 1);
        main.peaks.fill(0.0);
        publish_mix_meters(&shared, &main, &HashMap::new());
        assert_eq!(shared.output_clip_events.load(Ordering::Relaxed), 1);
        main.peaks[0] = 1.0;
        publish_mix_meters(&shared, &main, &HashMap::new());
        assert_eq!(shared.output_clip_events.load(Ordering::Relaxed), 2);
        drop(guard);
    }

    #[test]
    fn one_session_update_is_applied_per_callback_drain() {
        let initial = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let (session_tx, _session_rx) = return_channel(OLD_SESSIONS);
        tx.send(Cmd::OldSessions(session_tx)).unwrap();
        let mut state = AudioState::new(rx, shared, initial, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        for name in ["first", "second", "third"] {
            let session = soundcraft_model::Session { name: name.into(), ..Default::default() };
            tx.send(test_session(Arc::new(session), Vec::new())).unwrap();
        }

        for expected in ["first", "second", "third"] {
            state.drain();
            assert_eq!(state.session.name, expected);
        }
    }

    #[test]
    fn a_full_session_return_queue_never_drops_the_replaced_session_in_the_callback() {
        let initial = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let (mut retire_tx, mut retire_rx) = return_channel(OLD_SESSIONS);
        for _ in 0..OLD_SESSIONS {
            let _ = retire_tx.try_send(RetiredSession { _session: Some(Arc::new(soundcraft_model::Session::default())), _mix: None, _input: None });
        }
        tx.send(Cmd::OldSessions(retire_tx)).unwrap();
        let mut state = AudioState::new(rx, shared, Arc::clone(&initial), 48_000.0, sync_channel(RETIRED_PLUGINS).0);

        let first = Arc::new(soundcraft_model::Session { name: "first".into(), ..Default::default() });
        tx.send(test_session(Arc::clone(&first), Vec::new())).unwrap();
        state.drain();
        assert_eq!(state.session.name, "first");
        assert_eq!(Arc::strong_count(&initial), 2, "the pending retirement still owns the old snapshot");

        let second = Arc::new(soundcraft_model::Session { name: "second".into(), ..Default::default() });
        tx.send(test_session(Arc::clone(&second), Vec::new())).unwrap();
        state.drain();
        assert_eq!(state.session.name, "first", "a full retirement queue defers another session update");

        let _ = retire_rx.try_recv();
        state.drain();
        assert_eq!(state.session.name, "second");
        assert_eq!(Arc::strong_count(&first), 2, "the second full-queue retirement remains owned");
    }

    #[test]
    fn a_full_plugin_return_queue_retains_instances_and_stops_rendering_until_collected() {
        let session = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(true),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (_tx, rx) = test_channel();
        let (garbage_tx, garbage_rx) = sync_channel(1);
        let mut state = AudioState::new(rx, Arc::clone(&shared), session, 48_000.0, garbage_tx);
        // Occupy the sole return slot, then retire a real DSP instance.
        state.garbage.try_send(Vec::new()).unwrap();
        state.queue_garbage(vec![soundcraft_dsp::create("gain").unwrap()]);
        assert_eq!(state.pending_garbage.as_ref().map(Vec::len), Some(1));

        state.render_next(128);
        assert!(state.pending.iter().all(|channel| channel[..128].iter().all(|sample| *sample == 0.0)));
        assert_eq!(state.pos, 0, "transport must not advance while retired plugins are retained");
        assert_eq!(state.pending_garbage.as_ref().map(Vec::len), Some(1));

        garbage_rx.try_recv().unwrap();
        state.drain();
        assert!(state.pending_garbage.is_none());
        let returned = garbage_rx.try_recv().unwrap();
        assert_eq!(returned.len(), 1, "the plugin is returned for UI-thread destruction");
    }

    #[test]
    fn live_input_channel_changes_reuse_preallocated_mixer_buffers() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let mut state = AudioState::new(rx, shared, s, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        let lengths = state.mix.input.iter().map(Vec::len).collect::<Vec<_>>();
        let capacities = state.mix.input.iter().map(Vec::capacity).collect::<Vec<_>>();

        tx.send(Cmd::Input(record::InputRing::new(1, 48_000))).unwrap();
        state.drain();
        state.render_next(128);
        assert_eq!(state.mix.input_channels, 1);
        assert_eq!(state.mix.input.iter().map(Vec::len).collect::<Vec<_>>(), lengths);
        assert_eq!(state.mix.input.iter().map(Vec::capacity).collect::<Vec<_>>(), capacities);

        tx.send(Cmd::Input(record::InputRing::new(2, 48_000))).unwrap();
        state.drain();
        state.render_next(128);
        assert_eq!(state.mix.input_channels, 2);
        assert_eq!(state.mix.input.iter().map(Vec::len).collect::<Vec<_>>(), lengths);
        assert_eq!(state.mix.input.iter().map(Vec::capacity).collect::<Vec<_>>(), capacities);
    }

    #[test]
    fn initial_session_is_synchronized_before_the_callback_starts() {
        let mut session = soundcraft_model::Session::default();
        let track = session.add_track(soundcraft_model::TrackKind::Audio, soundcraft_model::ChannelFormat::Mono, None);
        let session = Arc::new(session);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (_tx, rx) = test_channel();
        let state = AudioState::new(rx, shared, Arc::clone(&session), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        assert!(state.mix.meters.contains_key(&track));
        assert_eq!(state.mix.meters[&track].peaks.len(), 1);
        assert_eq!(state.pending.len(), MAX_CHANNELS);
        assert_eq!(state.pending_channels, main_channels(&session));
        assert!(state.pending.iter().all(|channel| channel.len() == BLOCK));
    }

    #[test]
    fn short_device_callbacks_render_short_mix_blocks() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut out = vec![0.0f32; 2 * 64];
        st.fill(&mut out, 2);
        assert_eq!(shared.position.load(Ordering::Relaxed), 64);
    }

    fn audio_output_state(format: soundcraft_model::ChannelFormat, channels: Vec<Vec<f32>>) -> AudioState {
        use soundcraft_model::{Clip, SourceAudio, SourceId, TrackKind};
        let mut s = Session::default();
        s.edit.click = false;
        s.outputs[0].format = format;
        let t = s.add_track(TrackKind::Audio, format, None);
        let frames = channels[0].len();
        s.pool.insert(SourceId(5), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels })));
        let id = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "output", SourceId(5), 0, 0, frames as i64));
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let st = AudioState::new(rx, shared, Arc::new(s), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        st
    }

    #[test]
    fn mono_device_averages_stereo_before_clipping() {
        use soundcraft_model::ChannelFormat;
        // Right only, left only, identical, opposite polarity, and levels that must be averaged
        // before clipping either side to the device's range.
        let left = [0.0, 0.8, 0.8, 0.8, 2.0, -2.0, 2.0];
        let right = [0.8, 0.0, 0.8, -0.8, -0.5, 0.5, 2.0];
        let expected = [0.4, 0.4, 0.8, 0.0, 0.75, -0.75, 1.0];
        for callback in [1, 37, 512] {
            let channels = [left, right].map(|pattern| (0..2048).map(|i| pattern[i % pattern.len()]).collect());
            let mut st = audio_output_state(ChannelFormat::Stereo, channels.into());
            let mut out = vec![0.0; callback];
            for block in 0..3 {
                st.fill(&mut out, 1);
                for (i, actual) in out.iter().enumerate() {
                    let at = block * callback + i;
                    assert!((actual - expected[at % expected.len()]).abs() < 1e-6, "callback {callback}, sample {at}: {actual}");
                }
            }
        }
    }

    #[test]
    fn mono_device_preserves_centered_mono_level_and_includes_surround_right_channels() {
        use soundcraft_model::ChannelFormat;
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let mut mono = audio_output_state(ChannelFormat::Mono, vec![vec![0.8; 512]]);
        let mut out = [0.0; 128];
        mono.fill(&mut out, 1);
        let centered = 0.8 * soundcraft_dsp::db_to_gain(-3.0);
        assert!(out.iter().all(|x| (*x - centered).abs() < 1e-6), "a centered mono track keeps its rendered level under the -3 dB pan law");

        // Independently exercise each SMPTE 5.1 channel: L, R, C, LFE, Ls, Rs.
        let channels = (0..6).map(|c| (0..512).map(|i| if i % 6 == c { 0.8 } else { 0.0 }).collect()).collect();
        let mut surround = audio_output_state(ChannelFormat::Surround51, channels);
        surround.fill(&mut out, 1);
        let expected = [0.4, 0.4, 0.8 * h, 0.0, 0.4 * h, 0.4 * h];
        for (i, actual) in out.iter().enumerate() {
            assert!((actual - expected[i % expected.len()]).abs() < 1e-6, "surround sample {i}: {actual}");
        }
    }

    #[test]
    fn mono_device_contains_overflow_from_opposite_surround_sides() {
        use soundcraft_model::ChannelFormat;
        let channels = [f32::MAX, -f32::MAX, 0.0, 0.0, f32::MAX, -f32::MAX].map(|v| vec![v; 128]);
        let mut st = audio_output_state(ChannelFormat::Surround51, channels.into());
        let mut out = [0.5; 128];
        st.fill(&mut out, 1);
        assert!(out.iter().all(|x| *x == 0.0), "overflow in the stereo fold-down must not send NaN to a mono device");
    }

    #[test]
    fn surround_main_maps_to_device_channels_or_folds_down() {
        use soundcraft_model::{ChannelFormat, Clip, SourceAudio, SourceId, TrackKind};
        let mut s = soundcraft_model::Session::default();
        s.outputs[0].format = ChannelFormat::Surround51;
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Surround51, None);
        let chans: Vec<Vec<f32>> = (0..6).map(|c| vec![0.1 * (c + 1) as f32; 48_000]).collect();
        s.pool.insert(SourceId(5), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: chans })));
        let id = s.new_clip_id();
        s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "x", SourceId(5), 0, 0, 48_000));
        let s = Arc::new(s);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let initial = Arc::new(soundcraft_model::Session::default());
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&initial), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        let (old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
        tx.send(Cmd::OldSessions(old_tx)).unwrap();
        tx.send(test_session(Arc::clone(&s), Vec::new())).unwrap();
        assert_eq!(st.pending.len(), MAX_CHANNELS);
        let pending_capacity: Vec<usize> = st.pending.iter().map(Vec::capacity).collect();
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        // An 8-channel device: channels in order, the rest silent.
        let mut out = vec![0.0f32; 8 * 256];
        st.fill(&mut out, 8);
        assert_eq!(st.pending_channels, 6);
        assert_eq!(st.pending.iter().map(Vec::capacity).collect::<Vec<_>>(), pending_capacity);
        assert!(old_rx.try_recv().is_ok());
        let f = &out[8 * 100..8 * 101];
        for c in 0..6 {
            assert!((f[c] - 0.1 * (c + 1) as f32).abs() < 1e-5, "{f:?}");
        }
        assert!(f[6] == 0.0 && f[7] == 0.0);
        // A stereo device: ITU fold-down.
        let mut out = vec![0.0f32; 2 * 256];
        st.fill(&mut out, 2);
        let h = std::f32::consts::FRAC_1_SQRT_2;
        let l = (0.1 + h * 0.3 + h * 0.5f32).min(1.0);
        assert!((out[200] - l).abs() < 1e-4, "{} vs {l}", out[200]);
        assert_eq!(shared.meters.lock().unwrap().main.peaks.len(), 6);
    }

    #[cfg(feature = "plugin-hosts")]
    #[test]
    fn instances_arrive_with_the_session_and_leave_through_the_return_channel() {
        use soundcraft_model::{ChannelFormat, Insert, TrackKind};
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Stereo, None);
        s.track_mut(t).unwrap().mixer.inserts[1] = Some(Insert::new("clap:test.fake"));
        let s = Arc::new(s);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
        let (session_tx, _session_rx) = return_channel(OLD_SESSIONS);
        tx.send(Cmd::OldSessions(session_tx)).unwrap();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(soundcraft_model::Session::default()), 48_000.0, gtx);
        let plugin = soundcraft_dsp::create("gain").unwrap();
        tx.send(test_session(Arc::clone(&s), vec![PreparedInstance { track: t, slot: 1, id: "clap:test.fake".into(), ch: 2, plugin }])).unwrap();
        let mut out = vec![0.0f32; 256];
        st.fill(&mut out, 2);
        assert!(grx.try_recv().is_err(), "nothing retired yet");
        // A session without the insert retires the instance to the return channel.
        tx.send(test_session(Arc::new(soundcraft_model::Session::default()), Vec::new())).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(grx.try_recv().unwrap().len(), 1);
        // A sample-rate change rebuilds the engine without dropping plugins on this thread.
        tx.send(test_session(
            Arc::clone(&s),
            vec![PreparedInstance { track: t, slot: 1, id: "clap:test.fake".into(), ch: 2, plugin: soundcraft_dsp::create("gain").unwrap() }],
        ))
        .unwrap();
        st.fill(&mut out, 2);
        let mut s44 = (*s).clone();
        s44.sample_rate = soundcraft_time::SampleRate::new(44_100).unwrap();
        tx.send(test_session(Arc::new(s44), Vec::new())).unwrap();
        st.fill(&mut out, 2);
        assert_eq!(grx.try_recv().unwrap().len(), 1);
    }

    #[test]
    fn plugin_states_are_read_without_the_audio_thread() {
        let mut side = Side::new(48_000.0, std::sync::mpsc::channel().1);
        let (a, b) = (TrackId(1), TrackId(2));
        side.states.insert((b, 0), Box::new(|| Some(b"b0".to_vec())));
        side.states.insert((a, soundcraft_mix::INSTRUMENT_SLOT), Box::new(|| Some(b"synth".to_vec())));
        side.states.insert((a, 3), Box::new(|| Some(b"a3".to_vec())));
        side.states.insert((a, 4), Box::new(|| None));
        let want = vec![(a, 3, b"a3".to_vec()), (a, soundcraft_mix::INSTRUMENT_SLOT, b"synth".to_vec()), (b, 0, b"b0".to_vec())];
        assert_eq!(side.capture_states(), want);
        side.forget((a, 3));
        assert_eq!(side.capture_states(), want[1..]);
    }

    struct Window;

    impl PluginEditor for Window {
        fn open(&mut self) -> Result<(), String> {
            Ok(())
        }
        fn close(&mut self) {}
        fn is_open(&self) -> bool {
            false
        }
        fn idle(&mut self) -> Vec<(String, f32)> {
            Vec::new()
        }
    }

    #[test]
    fn a_refused_state_is_kept_until_the_editor_opens() {
        let mut side = Side::new(48_000.0, std::sync::mpsc::channel().1);
        let k = (TrackId(1), 0);
        side.states.insert(k, Box::new(|| Some(b"defaults".to_vec())));
        side.editors.insert(k, Box::new(Window));
        side.refused.insert(k);
        assert!(side.capture_states().is_empty());
        side.open_editor(k).unwrap();
        assert_eq!(side.capture_states(), vec![(TrackId(1), 0, b"defaults".to_vec())]);
    }

    #[cfg(feature = "plugin-hosts")]
    #[test]
    fn side_creates_each_missing_instance_once() {
        use soundcraft_model::{ChannelFormat, Insert, TrackKind};
        let mut side = Side::new(48_000.0, std::sync::mpsc::channel().1);
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        s.track_mut(t).unwrap().mixer.inserts[0] = Some(Insert::new("clap:does.not.exist"));
        s.track_mut(t).unwrap().mixer.inserts[1] = Some(Insert::new("gain"));
        assert!(side.prepare(&s).is_empty(), "the plugin is not installed; built-ins stay with the engine");
        assert_eq!(side.shadow.get(&(t, 0)), Some(&("clap:does.not.exist".to_string(), 1, false)));
        assert!(side.prepare(&s).is_empty());
        assert_eq!(side.shadow.len(), 1, "failures are remembered, not retried every update");
        s.track_mut(t).unwrap().mixer.inserts[0] = None;
        side.prepare(&s);
        assert!(side.shadow.is_empty());
    }

    #[test]
    fn replaced_sessions_are_freed_on_the_ui_thread() {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
        let mut st = AudioState::new(rx, shared, Arc::new(soundcraft_model::Session::default()), 48_000.0, gtx);
        let mut side = Side::new(48_000.0, grx);
        side.return_old_sessions(&tx);
        let mut out = vec![0.0f32; 256];
        let mut watched = Vec::new();
        for _ in 0..3 * OLD_SESSIONS {
            let s = Arc::new(soundcraft_model::Session::default());
            watched.push(Arc::downgrade(&s));
            side.collect_garbage();
            tx.send(test_session(s, Vec::new())).unwrap();
            st.fill(&mut out, 2);
            let replaced = watched.len().checked_sub(2).and_then(|i| watched.get(i));
            assert!(replaced.is_none_or(|w| w.strong_count() == 1), "freed on the audio thread");
        }
        side.collect_garbage();
        assert_eq!(watched.iter().filter(|w| w.strong_count() > 0).count(), 1, "only the live snapshot is left");
    }

    #[test]
    fn looping_wraps() {
        let s = Arc::new(soundcraft_model::Session::default());
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), s, 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        tx.send(Cmd::Play { from: 0, end: None, looped: Some(Range::new(0, 1000)) }).unwrap();
        let mut out = vec![0.0f32; 20_000];
        st.fill(&mut out, 2);
        let p = shared.position.load(Ordering::Relaxed);
        assert!(p <= 1000, "position {p} should stay inside the loop");
        assert!(shared.playing.load(Ordering::Relaxed));
    }

    #[test]
    fn locate_while_playing_drops_the_old_position_audio() {
        use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        for (plugin, params) in [("compressor", &[("lookahead", 5.0), ("ratio", 1.0)][..]), ("room_reverb", &[][..])] {
            let mut s = soundcraft_model::Session::default();
            s.edit.click = false;
            let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
            s.pool.insert(
                SourceId(5),
                Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![vec![0.5; 4800]] })),
            );
            let id = s.new_clip_id();
            let tr = s.track_mut(t).unwrap();
            tr.playlist_mut().unwrap().clips.push(Clip::audio(id, "x", SourceId(5), 0, 0, 4800));
            let mut ins = Insert::new(plugin);
            for (k, v) in params {
                ins.params.insert((*k).into(), *v);
            }
            tr.mixer.inserts[0] = Some(ins);
            let shared = Arc::new(Shared {
                position: AtomicI64::new(0),
                playing: AtomicBool::new(false),
                speed: AtomicU32::new(1000),
                output_clip_events: AtomicU32::new(0),
                monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
                monitor_use_rack: AtomicBool::new(true),
                direct_monitor_enabled: AtomicBool::new(true),
                feedback_protection: AtomicBool::new(false),
                speaker_protection: AtomicBool::new(false),
                meters: Mutex::new(MeterSnapshot::default()),
            });
            let (tx, rx) = test_channel();
            let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
            tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
            let mut out = vec![0.0f32; 2 * 512];
            let mut before = 0.0f32;
            for _ in 0..9 {
                st.fill(&mut out, 2);
                before = out.iter().fold(before, |m, x| m.max(x.abs()));
            }
            assert!(before > 0.1, "{plugin}: the clip plays before the locate");
            tx.send(Cmd::Locate(1_000_000)).unwrap();
            for _ in 0..8 {
                st.fill(&mut out, 2);
                assert!(out.iter().all(|x| x.abs() < 1e-6), "{plugin}: audio from before the locate");
            }
        }
    }

    #[test]
    fn armed_tracks_are_heard_and_metered_after_playback_stops() {
        use soundcraft_model::{ChannelFormat, TrackKind};
        let mut s = soundcraft_model::Session::default();
        let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        s.track_mut(t).unwrap().mixer.record_arm = true;
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        prepare_meter_layout(&shared, &s);
        let (tx, rx) = test_channel();
        let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        let ring = record::InputRing::new(1, 48_000);
        tx.send(Cmd::Input(Arc::clone(&ring))).unwrap();
        tx.send(Cmd::Play { from: 0, end: Some(1000), looped: None }).unwrap();
        let mut out = vec![0.0f32; 4096];
        st.fill(&mut out, 2);
        assert!(!shared.playing.load(Ordering::Relaxed));
        ring.push(&[0.5; 4096]);
        st.fill(&mut out, 2);
        assert!(out.iter().any(|x| x.abs() > 0.1), "the input is heard");
        assert!(shared.meters.lock().unwrap().tracks.get(&t).is_some_and(|m| m.peak[0] > 0.1), "and metered");
    }

    #[cfg(not(feature = "plugin-hosts"))]
    #[test]
    fn mobile_session_handoff_uses_the_prepared_box_and_preserves_the_live_tail() {
        use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        let mut session = Session::default();
        session.edit.click = false;
        let vocal = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let signal = (0..24_000).map(|n| if n < 512 { (n as f32 * 0.21).sin() * 0.4 } else { 0.0 }).collect();
        session
            .pool
            .insert(SourceId(1), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal] })));
        let clip = session.new_clip_id();
        session.track_mut(vocal).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(clip, "take", SourceId(1), 0, 0, 24_000));
        let mut reverb = Insert::new("room_reverb");
        reverb.params.insert("mix".into(), 60.0);
        session.track_mut(vocal).unwrap().mixer.inserts[0] = Some(reverb);
        let latent = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let mut comp = Insert::new("compressor");
        comp.params.insert("lookahead".into(), 5.0);
        session.track_mut(latent).unwrap().mixer.inserts[0] = Some(comp);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        prepare_meter_layout(&shared, &session);
        let fault = Fault::default();
        let (tx, rx) = channel();
        let (old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
        tx.send(Cmd::OldSessions(old_tx)).unwrap();
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        let mut reference = MixEngine::new(48_000.0, BLOCK);
        let mut expected = vec![vec![0.0; 128]; 2];
        let mut output = vec![0.0; 256];
        let mut tail = false;
        for block in 0..80 {
            let mut prepared_address = None;
            let mut old_address = None;
            if [12, 24, 36, 48, 60].contains(&block) {
                if block < 48 {
                    session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
                    if block == 24 {
                        session.track_mut(latent).unwrap().mixer.inserts[1] = Some(Insert::new("compressor"));
                    }
                } else if block == 48 {
                    session.track_mut(vocal).unwrap().mixer.inserts[0].as_mut().unwrap().params.insert("mix".into(), 30.0);
                } else {
                    session.tracks.retain(|track| track.id != latent);
                }
                prepare_meter_layout(&shared, &session);
                let command = session_command(Arc::new(session.clone()), Vec::new(), block * 128);
                if let Cmd::Session(_, mix) = &command {
                    prepared_address = Some(std::ptr::from_ref(mix.as_ref()));
                }
                old_address = Some(std::ptr::from_ref(state.mix.as_ref()));
                tx.send(command).unwrap();
            }
            let allocations = allocation_counter::measure(|| {
                guarded_block(&fault, &shared, &mut output, |out| state.fill(out, 2));
            });
            assert_eq!(allocations.count_total, 0, "callback allocated at block {block}: {allocations:?}");
            assert_eq!(allocations.count_current, 0, "callback freed objects at block {block}: {allocations:?}");
            reference.render(&session, block * 128, 128, &mut expected);
            for (i, frame) in output.chunks_exact(2).enumerate() {
                assert_eq!(frame, &[expected[0][i].clamp(-1.0, 1.0), expected[1][i].clamp(-1.0, 1.0)], "block {block}");
            }
            if let Some(prepared) = prepared_address {
                assert!(std::ptr::eq(prepared, std::ptr::from_ref(state.mix.as_ref())));
                let retired = old_rx.try_recv().unwrap();
                assert!(std::ptr::eq(old_address.unwrap(), std::ptr::from_ref(retired._mix.as_ref().unwrap().as_ref())));
                drop(retired); // Simulated control-thread collection, outside fill().
            }
            tail |= block > 12 && output.iter().any(|v| v.abs() > 1e-6);
        }
        assert!(tail, "handoff must preserve an actually sounding tail");
    }

    #[cfg(all(not(feature = "plugin-hosts"), not(target_arch = "wasm32")))]
    #[test]
    fn mobile_command_overload_stops_output_and_reports_once() {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(true),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = channel();
        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
        let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(Session::default()), 48_000.0, gtx);
        state.playing = true;
        let player = Player {
            shared: Arc::clone(&shared),
            fault: Arc::default(),
            watch: Mutex::new(None),
            tx,
            side: Mutex::new(Side::new(48_000.0, grx)),
            _stream: None,
            device_name: "test".into(),
            device_rate: 48_000,
            device_channels: 2,
            device_buffer: None,
            silent: true,
        };
        for _ in 0..9 {
            player.set_recording(false);
        }
        let mut output = vec![1.0; 256];
        state.fill(&mut output, 2);
        assert!(output.iter().all(|v| *v == 0.0));
        assert!(!state.playing);
        assert!(!shared.playing.load(Ordering::Relaxed));
        assert_eq!(player.trouble(), Some(OutputTrouble::Overloaded));
        assert_eq!(player.trouble(), None);
    }

    #[test]
    fn replaced_input_rings_are_retired_without_overwriting_pending_ownership() {
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        let (tx, rx) = test_channel();
        let (mut old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
        for _ in 0..OLD_SESSIONS {
            old_tx.try_send(RetiredSession { _session: None, _mix: None, _input: None }).ok();
        }
        tx.send(Cmd::OldSessions(old_tx)).unwrap();
        let mut state = AudioState::new(rx, shared, Arc::new(Session::default()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        let first = record::InputRing::new(1, 48_000);
        let second = record::InputRing::new(2, 48_000);
        let third = record::InputRing::new(1, 48_000);
        tx.send(Cmd::Input(Arc::clone(&first))).unwrap();
        tx.send(Cmd::Input(Arc::clone(&second))).unwrap();
        tx.send(Cmd::Input(Arc::clone(&third))).unwrap();
        state.drain();
        assert!(Arc::ptr_eq(state.input.as_ref().unwrap(), &second));
        assert_eq!(Arc::strong_count(&first), 2, "pending state must retain the replaced ring");
        state.drain();
        assert!(Arc::ptr_eq(state.input.as_ref().unwrap(), &second));
        old_rx.try_recv().ok();
        state.drain();
        assert!(Arc::ptr_eq(state.input.as_ref().unwrap(), &third));
        assert_eq!(Arc::strong_count(&first), 2);
        assert_eq!(Arc::strong_count(&second), 2);
        while let Ok(old) = old_rx.try_recv() {
            drop(old);
        }
        assert_eq!(Arc::strong_count(&first), 1);
        state.drain();
        while let Ok(old) = old_rx.try_recv() {
            drop(old);
        }
        assert_eq!(Arc::strong_count(&second), 1);
    }

    #[cfg(not(feature = "plugin-hosts"))]
    #[test]
    fn mobile_latency_automation_does_not_allocate_or_free_in_the_callback() {
        use soundcraft_model::{AutoParam, ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        let mut session = Session::default();
        session.edit.click = false;
        let source = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let signal = (0..4096).map(|n| (n as f32 * 0.13).sin() * 0.2).collect();
        session
            .pool
            .insert(SourceId(1), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal] })));
        let clip = session.new_clip_id();
        session.track_mut(source).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(clip, "signal", SourceId(1), 0, 0, 4096));
        let latent = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let track = session.track_mut(latent).unwrap();
        let mut compressor = Insert::new("compressor");
        compressor.params.insert("lookahead".into(), 0.0);
        track.mixer.inserts[0] = Some(compressor);
        let lane = track.lane_mut(&AutoParam::Plugin { slot: 0, param: "lookahead".into() });
        for (pos, value) in [(0, 0.0), (128, 10.0), (256, 0.0), (512, 10.0), (768, 0.0)] {
            lane.set_point(pos, value);
        }
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(false),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        prepare_meter_layout(&shared, &session);
        let (tx, rx) = channel();
        let (old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
        tx.send(Cmd::OldSessions(old_tx)).unwrap();
        let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
        let fault = Fault::default();
        let mut output = vec![0.0; 256];
        let mut expected = vec![vec![0.0; 128]; 2];
        let mut reference = MixEngine::new(48_000.0, BLOCK);
        let mut audible = false;
        for block in 0..8 {
            if block == 1 {
                let track = session.track_mut(latent).unwrap();
                let mut compressor = Insert::new("compressor");
                compressor.params.insert("lookahead".into(), 0.0);
                track.mixer.inserts[1] = Some(compressor);
                let lane = track.lane_mut(&AutoParam::Plugin { slot: 1, param: "lookahead".into() });
                for (pos, value) in [(0, 0.0), (128, 0.0), (256, 10.0), (512, 10.0), (768, 0.0)] {
                    lane.set_point(pos, value);
                }
                prepare_meter_layout(&shared, &session);
                // An ordered update can be prepared before the audio callback reaches it.
                // At position zero both latencies are zero, but their combined future maximum
                // exceeds the old graph's capacity. Handoff must keep the larger new storage.
                tx.send(session_command(Arc::new(session.clone()), Vec::new(), 0)).unwrap();
            }
            let allocations = allocation_counter::measure(|| guarded_block(&fault, &shared, &mut output, |out| state.fill(out, 2)));
            reference.render(&session, block as i64 * 128, 128, &mut expected);
            for (i, frame) in output.chunks_exact(2).enumerate() {
                assert_eq!(frame, &[expected[0][i], expected[1][i]], "automated delay PCM differs at block {block}");
            }
            audible |= output.iter().any(|v| v.abs() > 1e-6);
            assert_eq!(state.mix.latency, [0, 480, 480, 720, 960, 480, 0, 0][block]);
            assert_eq!(allocations.count_total, 0, "latency automation allocated at block {block}: {allocations:?}");
            assert_eq!(allocations.count_current, 0, "latency automation freed objects at block {block}: {allocations:?}");
            while let Ok(old) = old_rx.try_recv() {
                drop(old);
            }
        }
        assert!(audible, "automation regression must render actual source audio");
    }

    #[cfg(not(feature = "plugin-hosts"))]
    #[test]
    fn mobile_five_track_vocal_workload_has_no_callback_allocations_or_frees() {
        use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        const CALLBACKS: usize = 1024;
        for frames in [128, 256] {
            let mut session = Session::default();
            session.edit.click = false;
            let mut audio_tracks = Vec::new();
            for index in 0..5 {
                let format = if index == 0 { ChannelFormat::Stereo } else { ChannelFormat::Mono };
                let track_id = session.add_track(TrackKind::Audio, format, None);
                audio_tracks.push(track_id);
                let source = SourceId(session.alloc());
                let signal: Vec<f32> = (0..frames * CALLBACKS).map(|n| (n as f32 * (0.017 + index as f32 * 0.009)).sin() * 0.03).collect();
                session.pool.insert(
                    source,
                    Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal; format.channels()] })),
                );
                let clip_id = session.new_clip_id();
                let track = session.track_mut(track_id).unwrap();
                track.playlist_mut().unwrap().clips.push(Clip::audio(clip_id, "synthetic vocal workload", source, 0, 0, (frames * CALLBACKS) as i64));
                if index > 0 {
                    for (slot, plugin) in ["eq_7band", "compressor", "de_esser", "room_reverb"].iter().enumerate() {
                        let mut insert = Insert::new(*plugin);
                        let info = soundcraft_dsp::plugin_info(plugin).unwrap();
                        insert.params.extend(info.params.iter().map(|param| (param.id.into(), param.default)));
                        match slot {
                            0 => {
                                insert.params.insert("hpf_on".into(), 1.0);
                            }
                            1 => {
                                insert.params.insert("lookahead".into(), [0.0, 2.0, 5.0, 10.0][index - 1]);
                            }
                            3 => {
                                insert.params.insert("mix".into(), 20.0);
                            }
                            _ => {}
                        }
                        track.mixer.inserts[slot] = Some(insert);
                    }
                }
            }
            let master = session.add_track(TrackKind::Master, ChannelFormat::Stereo, None);
            session.track_mut(master).unwrap().mixer.volume_db = -3.0;
            let session = Arc::new(session);
            let shared = Arc::new(Shared {
                position: AtomicI64::new(0),
                playing: AtomicBool::new(false),
                speed: AtomicU32::new(1000),
                output_clip_events: AtomicU32::new(0),
                monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
                monitor_use_rack: AtomicBool::new(true),
                direct_monitor_enabled: AtomicBool::new(true),
                feedback_protection: AtomicBool::new(false),
                speaker_protection: AtomicBool::new(false),
                meters: Mutex::new(MeterSnapshot::default()),
            });
            prepare_meter_layout(&shared, &session);
            let (tx, rx) = channel();
            let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&session), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
            tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
            let fault = Fault::default();
            let mut output = vec![0.0; frames * 2];
            let mut reference = MixEngine::new(48_000.0, BLOCK);
            let mut expected = vec![vec![0.0; frames]; 2];
            let mut timings = Vec::with_capacity(CALLBACKS);
            let mut audible = false;
            for block in 0..CALLBACKS {
                let started = std::time::Instant::now();
                let allocations = allocation_counter::measure(|| guarded_block(&fault, &shared, &mut output, |out| state.fill(out, 2)));
                timings.push(started.elapsed().as_micros());
                assert_eq!(allocations.count_total, 0, "{frames}-frame vocal callback allocated in block {block}: {allocations:?}");
                assert_eq!(allocations.count_current, 0, "{frames}-frame vocal callback freed objects in block {block}: {allocations:?}");
                assert!(!fault.failed.load(Ordering::Relaxed));
                reference.render(&session, (block * frames) as i64, frames, &mut expected);
                for (index, frame) in output.chunks_exact(2).enumerate() {
                    assert_eq!(
                        frame,
                        &[expected[0][index].clamp(-1.0, 1.0), expected[1][index].clamp(-1.0, 1.0)],
                        "workload PCM differs at {frames} frames, block {block}"
                    );
                    audible |= frame.iter().any(|sample| sample.abs() > 1e-5);
                }
            }
            assert!(audible);
            let meters = shared.meters.lock().unwrap();
            for track in audio_tracks {
                assert!(
                    meters.tracks.get(&track).is_some_and(|meter| meter.peak.iter().any(|peak| *peak > 1e-5)),
                    "each workload track must actually produce metered audio"
                );
            }
            drop(meters);
            assert_eq!(shared.position.load(Ordering::Relaxed), (frames * CALLBACKS) as i64);
            timings.sort_unstable();
            eprintln!(
                "mobile_vocal_workload frames={frames} callbacks={CALLBACKS} rate=48000 p50_us={} p95_us={} max_us={} rust_allocations=0 rust_frees=0",
                timings[CALLBACKS / 2],
                timings[CALLBACKS * 95 / 100],
                timings[CALLBACKS - 1]
            );
        }
    }

    #[cfg(not(feature = "plugin-hosts"))]
    #[test]
    fn three_playing_tracks_with_dm_off_skip_monitor_rack_and_feedback_without_allocating() {
        use soundcraft_model::{ChannelFormat, Clip, Insert, SourceAudio, SourceId, TrackKind};
        const FRAMES: usize = 128;
        const BLOCKS: usize = 128;
        let mut session = Session::default();
        session.edit.click = false;
        for index in 0..3 {
            let track = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
            let source = SourceId(session.alloc());
            let signal = (0..FRAMES * BLOCKS).map(|i| (i as f32 * (0.03 + index as f32 * 0.01)).sin() * 0.1).collect();
            session.pool.insert(source, Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal] })));
            let clip_id = session.new_clip_id();
            session.track_mut(track).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(
                clip_id,
                "playing",
                source,
                0,
                0,
                (FRAMES * BLOCKS) as i64,
            ));
        }
        let capture = session.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
        let track = session.track_mut(capture).unwrap();
        track.mixer.input_monitor = true;
        let mut compressor = Insert::new("compressor");
        compressor.params.insert("lookahead".into(), 20.0);
        track.mixer.inserts[0] = Some(compressor);
        session.add_track(TrackKind::Master, ChannelFormat::Stereo, None);
        let shared = Arc::new(Shared {
            position: AtomicI64::new(0),
            playing: AtomicBool::new(true),
            speed: AtomicU32::new(1000),
            output_clip_events: AtomicU32::new(0),
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(false),
            feedback_protection: AtomicBool::new(true),
            speaker_protection: AtomicBool::new(false),
            meters: Mutex::new(MeterSnapshot::default()),
        });
        prepare_meter_layout(&shared, &session);
        let (_tx, rx) = channel();
        let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
        state.playing = true;
        state.recording = true;
        let input = record::InputRing::new(1, 48_000);
        state.input = Some(Arc::clone(&input));
        let mic: Vec<_> = (0..FRAMES).map(|i| (i as f32 * std::f32::consts::TAU * 4.0 / FRAMES as f32).sin() * 0.95).collect();
        let mut out = vec![0.0; FRAMES * 2];
        let mut expected = vec![vec![0.0; FRAMES]; 2];
        let mut reference = MixEngine::new(48_000.0, BLOCK);
        let mut expected_session = session.clone();
        expected_session.track_mut(capture).unwrap().mixer.inserts[0] = None;
        let mut max_us = 0;
        for block in 0..BLOCKS {
            input.push(&mic);
            let started = std::time::Instant::now();
            let allocations = allocation_counter::measure(|| state.fill(&mut out, 2));
            max_us = max_us.max(started.elapsed().as_micros());
            reference.render(&expected_session, (block * FRAMES) as i64, FRAMES, &mut expected);
            for (i, frame) in out.chunks_exact(2).enumerate() {
                assert_eq!(frame, &[expected[0][i], expected[1][i]], "muted monitor must not change backing playback");
            }
            assert_eq!(allocations.count_total, 0);
            assert_eq!(allocations.count_current, 0);
            assert!(!state.mix.monitor_use_rack);
            assert_eq!(state.feedback_guard.previous_rms, 0.0);
        }
        assert_eq!(input.input_peaks().len(), 1, "capture metering continues with DM off");
        shared.direct_monitor_enabled.store(true, Ordering::Relaxed);
        input.push(&mic);
        let on = allocation_counter::measure(|| state.fill(&mut out, 2));
        assert_eq!(on.count_total, 0);
        assert_eq!(on.count_current, 0);
        assert!(state.mix.monitor_use_rack, "DM can turn back on without rebuilding the player");
        assert!(state.feedback_guard.previous_rms > 0.2);
        shared.monitor_gain.store(0.0_f32.to_bits(), Ordering::Relaxed);
        input.push(&mic);
        state.fill(&mut out, 2);
        assert!(!state.mix.monitor_use_rack, "a zero monitor level also skips the rack");
        assert_eq!(state.feedback_guard.previous_rms, 0.0);
        eprintln!("three_track_dm_off callbacks={BLOCKS} frames={FRAMES} max_us={max_us} allocations=0 frees=0");
    }

    #[test]
    fn feedback_guard_ducks_sustained_resonance_and_ignores_broadband_input() {
        let frames = 512;
        let mut guard = FeedbackGuard { hot_blocks: 0, gain: 1.0, previous_rms: 0.0 };
        for block in 0..12 {
            let amplitude = 0.32 * 1.06_f32.powi(block);
            let tone: Vec<f32> = (0..frames).map(|i| (std::f32::consts::TAU * 10.0 * i as f32 / frames as f32).sin() * amplitude).collect();
            assert!(resonant_input(&tone, frames, 48_000));
            guard.update(std::slice::from_ref(&tone), frames, 48_000, true);
        }
        assert!(guard.gain < 1.0);
        guard.update(&[], 0, 48_000, false);
        assert!(guard.gain > 0.0);

        let mut state = 0x1234_5678u32;
        let noise: Vec<f32> = (0..frames)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state as i32 as f32 / i32::MAX as f32) * 0.5
            })
            .collect();
        assert!(!resonant_input(&noise, frames, 48_000));
        assert!(!resonant_input(&vec![0.01; frames], frames, 48_000));
    }

    #[test]
    fn speaker_guard_caps_peaks_and_recovers_smoothly() {
        let mut guard = SpeakerGuard { gain: 1.0 };
        let mut loud = [1.5, -1.25];
        guard.process(&mut loud, 48_000.0, true);
        assert!(loud.iter().all(|sample| sample.abs() <= SpeakerGuard::CEILING));
        let reduced = guard.gain;
        let mut silence = [0.0; 2];
        guard.process(&mut silence, 48_000.0, true);
        assert!(guard.gain > reduced && guard.gain < 1.0);
    }
}
