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
    _meters: Option<Box<MeterSnapshot>>,
}

fn session_command(s: Arc<Session>, instances: Vec<PreparedInstance>, pos: Samples) -> Cmd {
    let meters = Box::new(MeterSnapshot::shaped(&s));
    #[cfg(feature = "plugin-hosts")]
    {
        let _ = pos;
        Cmd::Session(s, instances, meters)
    }
    #[cfg(not(feature = "plugin-hosts"))]
    {
        let _ = instances;
        let mut mix = Box::new(new_engine(s.sample_rate.as_f64() as f32));
        mix.prepare_realtime(&s, pos);
        Cmd::Session(s, mix, meters)
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

mod callback;
mod stream;

#[cfg(not(feature = "plugin-hosts"))]
use callback::new_engine;
use callback::{AudioState, Fault, Side, prepare_meter_layout, take_meter_snapshot};
#[cfg(test)]
use callback::{FeedbackGuard, STALL, SpeakerGuard, copy_meter_snapshot, resonant_input};
#[cfg(not(target_arch = "wasm32"))]
use callback::{guarded_block, stalled};
#[cfg(test)]
use stream::pick_output_config;
use stream::{Opened, build_output, first_started, open_device, start_stream};

/// Meter values for the UI: the highest peak (and gain reduction) of every block rendered since
/// the UI last took them with [`Player::take_meters`]. The audio thread renders blocks faster than
/// the UI draws (about 94 blocks a second at 512 frames and 48 kHz against 20-60 polls), so
/// keeping only the latest block would miss most transients.
#[derive(Debug, Clone, Default)]
pub struct MeterSnapshot {
    pub tracks: HashMap<TrackId, StripMeter>,
    pub main: StripMeter,
    /// Something was written since the UI last took the values.
    fresh: bool,
}

impl MeterSnapshot {
    /// Entries for every track of `s` with its meter channel count, at zero. Allocates: built on
    /// the control thread so the callback only ever folds into entries that already exist.
    fn shaped(s: &Session) -> MeterSnapshot {
        let mut m = MeterSnapshot::default();
        m.shape(s);
        m
    }

    /// Give the snapshot exactly `s`'s tracks and channel counts, keeping the levels of entries
    /// that stay. Allocates; control thread only.
    fn shape(&mut self, s: &Session) {
        self.main.peaks.resize(main_channels(s), 0.0);
        self.tracks.retain(|id, _| s.track(*id).is_some());
        self.tracks.reserve(s.tracks.len().saturating_sub(self.tracks.len()));
        for track in &s.tracks {
            let channels = match track.kind {
                TrackKind::Master => main_channels(s),
                TrackKind::Instrument => track.channels().clamp(2, MAX_CHANNELS),
                _ => track.channels().clamp(1, MAX_CHANNELS),
            };
            self.tracks.entry(track.id).or_default().peaks.resize(channels, 0.0);
        }
    }

    /// Fold one rendered block's meters in (max). Tracks without an entry (a session update still
    /// crossing to the callback) are skipped, so this never allocates for an entry.
    fn accumulate(&mut self, main: &StripMeter, tracks: &HashMap<TrackId, StripMeter>) {
        self.main.accumulate(main);
        for (k, v) in tracks {
            if let Some(d) = self.tracks.get_mut(k) {
                d.accumulate(v);
            }
        }
        self.fresh = true;
    }

    /// Take over the levels `old` holds for the tracks this snapshot also has (same channel count),
    /// when a reshaped snapshot replaces it on the audio thread. Never allocates.
    fn carry_levels(&mut self, old: &MeterSnapshot) {
        if self.main.peaks.len() == old.main.peaks.len() {
            self.main.accumulate(&old.main);
        }
        for (k, d) in &mut self.tracks {
            if let Some(o) = old.tracks.get(k).filter(|o| o.peaks.len() == d.peaks.len()) {
                d.accumulate(o);
            }
        }
        self.fresh |= old.fresh;
    }

    /// Start a new accumulation period, keeping every entry and its storage.
    fn reset_levels(&mut self) {
        self.main.reset_levels();
        self.tracks.values_mut().for_each(StripMeter::reset_levels);
        self.fresh = false;
    }

    /// Nothing is playing or monitored: every meter reads silence. Levels are zeroed in place,
    /// so starting again does not allocate the tracks' entries anew.
    fn silence(&mut self) {
        self.reset_levels();
        self.fresh = true;
    }
}

/// Plugin states read from the live instances: `(track, slot or INSTRUMENT_SLOT, state)`.
pub type PluginStates = Vec<(TrackId, usize, Vec<u8>)>;

/// A parameter edit made in a plugin's own editor: `(track, slot, param id, value)`.
pub type EditorEdit = (TrackId, usize, String, f32);

enum Cmd {
    /// A new snapshot plus the third-party instances it needs (created on the UI thread) and the
    /// audio-side meter accumulator shaped for it.
    Session(Arc<Session>, SessionMixer, Box<MeterSnapshot>),
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
    /// The device went away, or the stream was invalidated and must be rebuilt.
    Gone,
    /// The device stopped asking for audio while playing.
    Stalled,
    /// The bounded mobile command queue filled; output stays silent until reconnection.
    #[cfg(not(feature = "plugin-hosts"))]
    Overloaded,
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

    /// Open the output device for `session`, trying the requested buffer size before the device default, then build
    /// the shared player state. If no configuration starts or the device cannot open, fall back to a silent clock at
    /// the session rate.
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
                let mut attempts = vec![config];
                if config.buffer_size != cpal::BufferSize::Default {
                    attempts.push(cpal::StreamConfig { buffer_size: cpal::BufferSize::Default, ..config });
                }
                let opened = first_started(
                    attempts,
                    |config: &cpal::StreamConfig| -> Result<Opened, cpal::Error> {
                        let rate = config.sample_rate;
                        let channels = usize::from(config.channels);
                        let (tx, rx) = channel();
                        let (gtx, grx) = sync_channel(RETIRED_PLUGINS);
                        let state = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&session), f64::from(rate), gtx);
                        let fault = Arc::new(Fault::default());
                        let stream = match format {
                            cpal::SampleFormat::I32 => build_output::<i32>(&device, *config, state, channels, &fault, &shared),
                            cpal::SampleFormat::I16 => build_output::<i16>(&device, *config, state, channels, &fault, &shared),
                            _ => build_output::<f32>(&device, *config, state, channels, &fault, &shared),
                        }?;
                        Ok(Opened { stream, tx, grx, fault })
                    },
                    |opened| start_stream(&opened.stream),
                );
                match opened {
                    Some((config, Opened { stream, tx, grx, fault })) => {
                        let rate = config.sample_rate;
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

    /// The plugins that said their state changed since the last call, so the session can keep
    /// changes made only in plugin editors.
    pub fn changed_states(&self) -> Vec<(TrackId, usize)> {
        self.side().changed_states()
    }

    /// Reads the state of the plugins in `slots`.
    pub fn capture_states_of(&self, slots: &[(TrackId, usize)]) -> PluginStates {
        self.side().capture_states_of(slots)
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

    /// A copy of the meters accumulated since the last [`Player::take_meters`], leaving them in
    /// place (for inspection; the UI takes them instead).
    pub fn meters(&self) -> MeterSnapshot {
        self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Take the meters accumulated since the last call into reusable UI storage and start a new
    /// period: `out` gets the highest peak of every block rendered in between. If no block was
    /// rendered since (a large device buffer renders several blocks per callback), `out` keeps
    /// the previous values rather than dropping to silence. Call it from one place only (two
    /// readers would split the peaks between them). Once track/channel counts are stable this
    /// performs no allocation. The audio thread only ever `try_lock`s, so holding the lock here
    /// never blocks it; blocks rendered meanwhile reach the next call.
    pub fn take_meters(&self, out: &mut MeterSnapshot) {
        let mut source = self.shared.meters.lock().unwrap_or_else(PoisonError::into_inner);
        take_meter_snapshot(&mut source, out);
    }
}

#[cfg(test)]
mod tests;
