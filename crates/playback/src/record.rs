//! Audio input capture for recording.
//!
//! A [`Recorder`] opens the default input device. While armed, the input callback copies each
//! block into a preallocated lock-free ring (it never locks or allocates), and a writer thread
//! streams the ring into one WAV file per take, rewriting the header every second so a crash leaves
//! a playable file.

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use soundcraft_model::TrackId;

/// What [`Recorder::finish`] returns: the takes that hold audio and the first problem met.
pub type Takes = (Vec<(TrackId, PathBuf)>, Option<String>);

/// Frames each block of a [`LiveTake`] sums up.
pub const LIVE_BLOCK: usize = 128;
/// Blocks a live take keeps per channel at most (three hours at 48 kHz); its file records on.
const LIVE_BLOCKS: usize = 1 << 22;

/// A take while it records, summed up for drawing: per channel, the (min, max) of each
/// [`LIVE_BLOCK`] frames so far.
pub type LiveTake = Vec<Vec<(f32, f32)>>;

const RING_SECONDS: usize = 4;
/// How long a recording input may go without a callback before it counts as gone.
const STALL: Duration = Duration::from_secs(2);
const GONE: u8 = 1;
const BROKEN: u8 = 2;
const WRITER_FAILED: &str = "The take writer stopped unexpectedly. Recorded files have been kept for recovery.";

/// Why the input can't go on.
#[derive(Debug, Clone, PartialEq)]
pub enum Trouble {
    /// The device went away or stopped sending audio, or its callback hit a bug.
    Device(String),
    /// Writing a take failed (a full disk, say).
    Disk(String),
}

/// Single-producer single-consumer FIFO of samples (input callback → writer thread). The slots
/// are atomics, so it needs no lock and no `unsafe`.
struct CaptureRing {
    slots: Box<[AtomicU32]>,
    mask: usize,
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl CaptureRing {
    fn new(capacity: usize) -> CaptureRing {
        let n = capacity.max(2).next_power_of_two();
        CaptureRing { slots: (0..n).map(|_| AtomicU32::new(0)).collect(), mask: n - 1, head: AtomicUsize::new(0), tail: AtomicUsize::new(0) }
    }

    /// Queue up to `n` samples (`sample(i)` is the i-th); returns how many fit.
    fn push(&self, n: usize, sample: impl Fn(usize) -> f32) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let n = n.min(self.slots.len().saturating_sub(head.wrapping_sub(self.tail.load(Ordering::Acquire))));
        for i in 0..n {
            if let Some(slot) = self.slots.get(head.wrapping_add(i) & self.mask) {
                slot.store(sample(i).to_bits(), Ordering::Relaxed);
            }
        }
        self.head.store(head.wrapping_add(n), Ordering::Release);
        n
    }

    fn pop_all(&self, out: &mut Vec<f32>) {
        let (tail, head) = (self.tail.load(Ordering::Relaxed), self.head.load(Ordering::Acquire));
        out.extend(
            (0..head.wrapping_sub(tail))
                .map(|i| self.slots.get(tail.wrapping_add(i) & self.mask).map_or(0.0, |s| f32::from_bits(s.load(Ordering::Relaxed)))),
        );
        self.tail.store(head, Ordering::Release);
    }
}

/// An input sample, or silence for NaN and infinities, which would poison the take and the mix.
fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { 0.0 }
}

struct Shared {
    armed: AtomicBool,
    ring: CaptureRing,
    /// Samples lost while the ring was full.
    dropped: AtomicUsize,
    /// Input callbacks so far, to tell a device that stopped sending.
    beats: AtomicU64,
    /// Set from the input stream when it can't go on: `GONE` or `BROKEN`.
    fault: AtomicU8,
    /// The first failure writing a take, until it's reported.
    failed: Mutex<Option<String>>,
}

impl Shared {
    fn new(ring: CaptureRing) -> Shared {
        let (beats, fault, failed) = (AtomicU64::new(0), AtomicU8::new(0), Mutex::new(None));
        Shared { armed: AtomicBool::new(false), ring, dropped: AtomicUsize::new(0), beats, fault, failed }
    }

    /// Run the input callback's `work`. After a panic in it, the input is broken and does
    /// nothing more.
    fn guarded(&self, work: impl FnOnce()) {
        self.beats.fetch_add(1, Ordering::Relaxed);
        if self.fault.load(Ordering::Relaxed) != BROKEN && std::panic::catch_unwind(std::panic::AssertUnwindSafe(work)).is_err() {
            self.fault.store(BROKEN, Ordering::Relaxed);
        }
    }

    /// The input callback's work while armed. Samples that don't fit are owed (`owed`) and queued
    /// as silence before the next block, so the take keeps its timing.
    fn capture(&self, data: &[f32], owed: &mut usize) {
        *owed -= self.ring.push(*owed, |_| 0.0);
        let kept = if *owed == 0 { self.ring.push(data.len(), |i| data.get(i).copied().map_or(0.0, finite)) } else { 0 };
        *owed = owed.saturating_add(data.len() - kept);
        self.dropped.fetch_add(data.len() - kept, Ordering::Relaxed);
    }
}

pub struct Recorder {
    shared: Arc<Shared>,
    /// Live input for monitoring (always fed while the input stream runs).
    pub monitor: Arc<InputRing>,
    #[cfg(not(target_os = "freebsd"))]
    _stream: Option<cpal::Stream>,
    pub device_name: String,
    pub sample_rate: u32,
    pub channels: usize,
    /// The take writer, and what each track's take holds so far.
    writer: Option<(std::thread::JoinHandle<Takes>, Vec<(TrackId, Arc<Mutex<LiveTake>>)>)>,
    /// The callback count last seen, and since when.
    seen: (u64, Instant),
}

/// The input stream: device samples are converted to f32 in a scratch buffer allocated here (so
/// the callback never allocates), then feed the monitor ring and, while armed, the capture.
#[cfg(not(target_os = "freebsd"))]
fn build_input<T>(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    shared: &Arc<Shared>,
    monitor: &Arc<InputRing>,
) -> Result<cpal::Stream, cpal::Error>
where
    T: cpal::SizedSample,
    f32: cpal::FromSample<T>,
{
    use cpal::Sample;
    use cpal::traits::DeviceTrait;
    let (sh, errors) = (Arc::clone(shared), Arc::clone(shared));
    let mon = Arc::clone(monitor);
    let mut scratch = vec![0.0f32; 8192 * usize::from(config.channels).max(1)];
    let chunk = scratch.len();
    let mut owed = 0;
    device.build_input_stream(
        config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            crate::mark_audio_thread();
            sh.guarded(|| {
                for part in data.chunks(chunk) {
                    let Some(s) = scratch.get_mut(..part.len()) else { return };
                    for (o, x) in s.iter_mut().zip(part) {
                        *o = f32::from_sample(*x);
                    }
                    mon.push(s);
                    if sh.armed.load(Ordering::Acquire) {
                        sh.capture(s, &mut owed);
                    } else {
                        owed = 0;
                    }
                }
            });
        },
        move |e: cpal::Error| {
            // Possibly the audio thread (see `crate::mark_audio_thread`).
            crate::mark_audio_thread();
            // cpal 0.18 reports every operation through one error type; a device that went away
            // or a stream that must be rebuilt (`StreamInvalidated`, not on mobile: see
            // `stream_lost`) stops capture.
            // `DeviceChanged` (a rerouted default device) and an xrun are not faults.
            if crate::devices::stream_lost(&e) {
                let _ = errors.fault.compare_exchange(0, GONE, Ordering::Relaxed, Ordering::Relaxed);
            }
            log::warn!("input stream error: {e}");
        },
        None,
    )
}

/// The input configuration at `rate` when the device offers one: the best sample format first,
/// then the most channels (so a stereo interface is preferred over a mono one). `None` lets the
/// caller use the device's default input configuration (the mix resamples when the rates differ).
fn pick_input_config(configs: &[cpal::SupportedStreamConfigRange], rate: u32) -> Option<cpal::SupportedStreamConfig> {
    configs
        .iter()
        .filter_map(|c| {
            let rank = crate::devices::format_rank(c.sample_format())?;
            if !c.contains_rate(rate) {
                return None;
            }
            c.try_with_sample_rate(rate).map(|config| (rank, std::cmp::Reverse(c.channels()), config))
        })
        .min_by_key(|(rank, channels, _)| (*rank, *channels))
        .map(|(_, _, config)| config)
}

impl Recorder {
    /// Open the default input device. Errors when there is none (the caller shows a message).
    pub fn open() -> Result<Recorder, String> {
        Recorder::open_with(&crate::AudioConfig::default())
    }

    /// Open the input device `audio` chooses (else the default) at its chosen rate and buffer
    /// size when the device offers them, else at the device's defaults. Integer devices (ASIO
    /// drivers usually are) are converted to f32.
    #[cfg(not(target_os = "freebsd"))]
    pub fn open_with(audio: &crate::AudioConfig) -> Result<Recorder, String> {
        use cpal::traits::{DeviceTrait, StreamTrait};
        let host = crate::devices::host(audio);
        let device = crate::devices::device(&host, audio.input.as_deref(), false).ok_or("no audio input device")?;
        let name = crate::devices::device_name(&device).unwrap_or_else(|| "Input".into());
        // At the chosen rate when the device offers it, so input and output run together: the
        // best format first, then the most channels.
        let configs: Vec<cpal::SupportedStreamConfigRange> = device.supported_input_configs().map(|configs| configs.collect()).unwrap_or_default();
        let at_rate = audio.sample_rate.and_then(|r| pick_input_config(&configs, r));
        let cfg = match at_rate {
            Some(c) => c,
            None => {
                let d = device.default_input_config().map_err(|e| e.to_string())?;
                // cpal 0.18's default can name a format these callbacks do not decode (I24, F64);
                // open the same rate and channels in a format they do.
                if crate::devices::format_rank(d.sample_format()).is_some() {
                    d
                } else {
                    crate::devices::supported_format_config(&configs, d.sample_rate(), d.channels())
                        .ok_or_else(|| format!("unsupported input sample format {:?}", d.sample_format()))?
                }
            }
        };
        let format = cfg.sample_format();
        if crate::devices::format_rank(format).is_none() {
            return Err(format!("unsupported input sample format {format:?}"));
        }
        let mut config = cfg.config();
        config.buffer_size = crate::devices::buffer_size(audio.buffer_size, cfg.buffer_size());
        let channels = usize::from(config.channels).max(1);
        let ring = CaptureRing::new((config.sample_rate as usize).saturating_mul(channels * RING_SECONDS).min(1 << 25));
        let shared = Arc::new(Shared::new(ring));
        let monitor = InputRing::new(usize::from(config.channels), config.sample_rate);
        let build = |config: cpal::StreamConfig| match format {
            cpal::SampleFormat::I32 => build_input::<i32>(&device, config, &shared, &monitor),
            cpal::SampleFormat::I16 => build_input::<i16>(&device, config, &shared, &monitor),
            _ => build_input::<f32>(&device, config, &shared, &monitor),
        };
        // The chosen buffer size first; if the device refuses it, its default.
        let stream = match build(config) {
            Ok(s) => s,
            Err(e) if config.buffer_size != cpal::BufferSize::Default => {
                log::warn!("cannot open the input with buffer {:?}: {e}; using the device default", config.buffer_size);
                config.buffer_size = cpal::BufferSize::Default;
                build(config).map_err(|e| e.to_string())?
            }
            Err(e) => return Err(e.to_string()),
        };
        stream.play().map_err(|e| e.to_string())?;
        let seen = (0, Instant::now());
        Ok(Recorder { monitor, shared, _stream: Some(stream), device_name: name, sample_rate: config.sample_rate, channels, writer: None, seen })
    }

    #[cfg(target_os = "freebsd")]
    pub fn open_with(_audio: &crate::AudioConfig) -> Result<Recorder, String> {
        Err("recording is not supported on this platform yet".into())
    }

    /// Create each take's file and start recording into them: a track's take records the given
    /// input channels into a new file.
    pub fn arm(&mut self, takes: Vec<(TrackId, Vec<usize>, PathBuf)>) -> Result<(), String> {
        if self.writer.is_some() {
            return Err("already recording".into());
        }
        let takes =
            takes.into_iter().map(|(id, inputs, path)| TakeWriter::create(id, inputs, path, self.sample_rate)).collect::<Result<Vec<_>, _>>()?;
        let live = takes.iter().map(|t| (t.id, Arc::clone(&t.live))).collect();
        let ring = &self.shared.ring;
        ring.tail.store(ring.head.load(Ordering::Acquire), Ordering::Release);
        *self.shared.failed.lock().unwrap_or_else(PoisonError::into_inner) = None;
        self.shared.armed.store(true, Ordering::Release);
        let (shared, channels) = (Arc::clone(&self.shared), self.channels);
        match std::thread::Builder::new().name("take writer".into()).spawn(move || write_takes(&shared, takes, channels)) {
            Ok(w) => self.writer = Some((w, live)),
            Err(e) => {
                self.shared.armed.store(false, Ordering::Release);
                return Err(e.to_string());
            }
        }
        Ok(())
    }

    /// What the take recording onto `track` holds so far.
    pub fn live_take(&self, track: TrackId) -> Option<MutexGuard<'_, LiveTake>> {
        let (_, take) = self.writer.as_ref()?.1.iter().find(|(id, _)| *id == track)?;
        Some(take.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Why the input can't go on, if it can't: the device went away or, while recording, stopped
    /// sending audio; its callback hit a bug; or writing a take failed (reported once).
    pub fn trouble(&mut self) -> Option<Trouble> {
        if let Some(e) = self.shared.failed.lock().unwrap_or_else(PoisonError::into_inner).take() {
            return Some(Trouble::Disk(e));
        }
        if self.shared.armed.load(Ordering::Acquire) && self.writer.as_ref().is_some_and(|(writer, _)| writer.is_finished()) {
            return Some(Trouble::Disk(WRITER_FAILED.into()));
        }
        let name = &self.device_name;
        match self.shared.fault.load(Ordering::Relaxed) {
            GONE => return Some(Trouble::Device(format!("“{name}” is unavailable"))),
            BROKEN => return Some(Trouble::Device("the audio input hit an internal error (see the log)".into())),
            _ => {}
        }
        let beats = self.shared.beats.load(Ordering::Relaxed);
        if beats != self.seen.0 || !self.shared.armed.load(Ordering::Relaxed) {
            self.seen = (beats, Instant::now());
        }
        (self.seen.1.elapsed() >= STALL).then(|| Trouble::Device(format!("“{name}” stopped sending audio")))
    }

    /// Interleaved input samples the current take lost to a full capture ring. Unlike
    /// `lost_ms`, this reports even a single lost sample, so a host can stop a damaged take
    /// immediately on its next control-thread poll.
    pub fn lost_samples(&self) -> usize {
        self.shared.dropped.load(Ordering::Relaxed)
    }

    /// Input the current take has lost to a slow disk so far, in whole milliseconds.
    pub fn lost_ms(&self) -> u64 {
        let frames = self.lost_samples() / self.channels.max(1);
        (frames as u64).saturating_mul(1000) / u64::from(self.sample_rate.max(1))
    }

    pub fn finish(&mut self) -> Takes {
        self.shared.armed.store(false, Ordering::Release);
        let (takes, problem) = match self.writer.take() {
            Some((writer, _)) => writer.join().unwrap_or_else(|_| (Vec::new(), Some(WRITER_FAILED.into()))),
            None => (Vec::new(), None),
        };
        let lost = self.shared.dropped.swap(0, Ordering::Relaxed);
        let slow = (lost > 0).then(|| {
            let ms = lost as f64 * 1000.0 / self.channels.max(1) as f64 / f64::from(self.sample_rate.max(1));
            format!("{ms:.3} ms of input lost ({lost} samples): the disk was too slow")
        });
        (takes, problem.or(slow))
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.finish();
    }
}

fn write_takes(shared: &Shared, mut takes: Vec<TakeWriter>, channels: usize) -> Takes {
    let (mut block, mut header_written, mut reported) = (Vec::new(), Instant::now(), false);
    loop {
        let stopping = !shared.armed.load(Ordering::Acquire);
        // The ring holds interleaved samples and a pop can end mid-frame: write whole frames and
        // keep the remainder for the next pop, so channels never shift.
        shared.ring.pop_all(&mut block);
        let whole = block.len() - block.len() % channels.max(1);
        takes.iter_mut().for_each(|t| t.write(&block[..whole], channels));
        block.drain(..whole);
        if !reported && let Some(e) = takes.iter().find_map(|t| t.error.clone()) {
            *shared.failed.lock().unwrap_or_else(PoisonError::into_inner) = Some(e);
            reported = true;
        }
        if stopping {
            break;
        }
        if header_written.elapsed() >= Duration::from_secs(1) {
            header_written = Instant::now();
            takes.iter_mut().for_each(TakeWriter::write_header);
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let (mut done, mut problem) = (Vec::new(), None);
    for mut t in takes {
        t.write_header();
        if t.frames == 0 {
            let _ = std::fs::remove_file(&t.path);
        } else {
            done.push((t.id, t.path));
        }
        problem = problem.or(t.error);
    }
    (done, problem)
}

/// One take's file. After an error it stops writing; its header keeps covering what was written.
struct TakeWriter {
    id: TrackId,
    inputs: Vec<usize>,
    path: PathBuf,
    file: File,
    rate: u32,
    frames: u64,
    error: Option<String>,
    live: Arc<Mutex<LiveTake>>,
}

impl TakeWriter {
    fn create(id: TrackId, inputs: Vec<usize>, path: PathBuf, rate: u32) -> Result<TakeWriter, String> {
        let file = path.parent().map_or(Ok(()), std::fs::create_dir_all).and_then(|()| File::create_new(&path));
        let file = file.map_err(|e| format!("{}: {e}", path.display()))?;
        let live = Arc::new(Mutex::new(vec![Vec::new(); inputs.len()]));
        let mut take = TakeWriter { id, inputs, path, file, rate, frames: 0, error: None, live };
        take.write_header();
        take.error.take().map_or(Ok(take), Err)
    }

    /// Append this take's channels of `block` (interleaved, `channels` wide).
    fn write(&mut self, block: &[f32], channels: usize) {
        let channels = channels.max(1);
        if self.error.is_some() {
            return;
        }
        let bytes: Vec<u8> = block
            .chunks_exact(channels)
            .flat_map(|f| self.inputs.iter().flat_map(move |&i| f.get(i).copied().unwrap_or(0.0).to_le_bytes()))
            .collect();
        match self.file.write_all(&bytes) {
            Ok(()) => {
                self.summarize(block, channels);
                self.frames += (block.len() / channels) as u64;
            }
            Err(e) => self.fail(e),
        }
    }

    /// Add this take's channels of `block` to its live take.
    fn summarize(&self, block: &[f32], channels: usize) {
        let mut live = self.live.lock().unwrap_or_else(PoisonError::into_inner);
        for (f, n) in block.chunks_exact(channels).zip(self.frames..) {
            let at = (n / LIVE_BLOCK as u64) as usize;
            for (blocks, &i) in live.iter_mut().zip(&self.inputs) {
                let v = f.get(i).copied().unwrap_or(0.0);
                match blocks.get_mut(at) {
                    Some(b) => *b = (b.0.min(v), b.1.max(v)),
                    None if at < LIVE_BLOCKS => blocks.push((v, v)),
                    None => {}
                }
            }
        }
    }

    fn write_header(&mut self) {
        let header = match soundcraft_audio_io::wav_float_header(self.inputs.len(), self.rate, self.frames) {
            Ok(h) => h,
            Err(e) => return self.fail(e),
        };
        let f = &mut self.file;
        if let Err(e) = f.seek(SeekFrom::Start(0)).and_then(|_| f.write_all(&header)).and_then(|()| f.seek(SeekFrom::End(0))) {
            self.fail(e);
        }
    }

    fn fail(&mut self, e: impl std::fmt::Display) {
        self.error.get_or_insert_with(|| format!("{}: {e}", self.path.display()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(path: &PathBuf) -> Vec<Vec<f32>> {
        soundcraft_audio_io::decode(&std::fs::read(path).unwrap(), None).unwrap().1.channels
    }

    #[test]
    fn a_full_ring_owes_the_lost_samples_as_silence() {
        let shared = Shared::new(CaptureRing::new(8));
        let (mut owed, mut out) = (0, Vec::new());
        shared.capture(&[1.0; 6], &mut owed);
        shared.capture(&[2.0; 4], &mut owed);
        shared.ring.pop_all(&mut out);
        shared.capture(&[3.0; 2], &mut owed);
        shared.ring.pop_all(&mut out);
        assert_eq!(out, [vec![1.0; 6], vec![2.0; 2], vec![0.0; 2], vec![3.0; 2]].concat());
        assert_eq!((owed, shared.dropped.load(Ordering::Relaxed)), (0, 2));
    }

    #[test]
    fn overflow_is_visible_below_one_millisecond_and_finish_keeps_the_usable_take() {
        for (channels, samples, kept_frames, lost) in [(1, 10, 8, 2), (3, 9, 2, 1)] {
            let dir = std::env::temp_dir().join(format!("soundcraft-overflow-take-{}-{channels}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            let take = TakeWriter::create(TrackId(1), vec![0], dir.join("Vox.wav"), 48_000).unwrap();
            let shared = Arc::new(Shared::new(CaptureRing::new(8)));
            shared.armed.store(true, Ordering::Release);
            shared.capture(&vec![0.25; samples], &mut 0);
            let mut r = recorder(&shared);
            r.channels = channels;
            assert_eq!(r.lost_ms(), 0, "whole milliseconds cannot detect this loss");
            assert_eq!(r.lost_samples(), lost, "even a fraction of an interleaved frame must be reported");
            let sh = Arc::clone(&shared);
            let writer = std::thread::spawn(move || write_takes(&sh, vec![take], channels));
            r.writer = Some((writer, Vec::new()));
            let (takes, warning) = r.finish();
            assert_eq!(takes.len(), 1, "overflow must not discard the valid prefix");
            assert_eq!(decode(&takes[0].1), vec![vec![0.25; kept_frames]]);
            assert!(warning.is_some_and(|text| text.contains(&format!("({lost} samples)"))), "finish reports the exact loss");
            assert_eq!(r.lost_samples(), 0);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }

    #[test]
    fn nan_and_infinite_input_records_as_silence() {
        let shared = Shared::new(CaptureRing::new(8));
        let (mut owed, mut out) = (0, Vec::new());
        shared.capture(&[0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.5], &mut owed);
        shared.ring.pop_all(&mut out);
        assert_eq!(out, [0.5, 0.0, 0.0, 0.0, -0.5]);
    }

    /// Build a cpal configuration range for tests with a 64..1024-frame buffer.
    fn range(channels: u16, min: u32, max: u32, format: cpal::SampleFormat) -> cpal::SupportedStreamConfigRange {
        cpal::SupportedStreamConfigRange::new(channels, min, max, cpal::SupportedBufferSize::Range { min: 64, max: 1024 }, format)
    }

    /// At the chosen rate, input selection prefers the best supported format, then more channels; a rate the device
    /// does not offer returns `None` so the caller can fall back.
    #[test]
    fn input_config_prefers_the_best_format_then_more_channels_at_the_chosen_rate() {
        let configs = [
            range(1, 44_100, 192_000, cpal::SampleFormat::I32),
            range(2, 44_100, 96_000, cpal::SampleFormat::I16),
            range(2, 48_000, 96_000, cpal::SampleFormat::F32),
            range(8, 44_100, 48_000, cpal::SampleFormat::F32),
        ];
        let c = pick_input_config(&configs, 96_000).unwrap();
        assert_eq!((c.sample_rate(), c.channels(), c.sample_format()), (96_000, 2, cpal::SampleFormat::F32));
        let c = pick_input_config(&configs, 48_000).unwrap();
        assert_eq!((c.sample_rate(), c.channels(), c.sample_format()), (48_000, 8, cpal::SampleFormat::F32), "more channels win in the same format");
        assert!(pick_input_config(&configs, 192_001).is_none(), "an unsupported rate falls back to the device default");
    }

    fn recorder(shared: &Arc<Shared>) -> Recorder {
        Recorder {
            shared: Arc::clone(shared),
            monitor: InputRing::new(1, 48_000),
            #[cfg(not(target_os = "freebsd"))]
            _stream: None,
            device_name: "Mic In".into(),
            sample_rate: 48_000,
            channels: 1,
            writer: None,
            seen: (0, Instant::now()),
        }
    }

    /// A stalled armed input, a panicking callback and a lost device all surface as `Trouble`; a quiet input that is
    /// not recording does not.
    #[test]
    fn a_bug_a_stall_or_a_lost_device_is_trouble() {
        let shared = Arc::new(Shared::new(CaptureRing::new(64)));
        let mut r = recorder(&shared);
        shared.armed.store(true, Ordering::Release);
        assert_eq!(r.trouble(), None);
        r.seen.1 = Instant::now().checked_sub(STALL).unwrap();
        assert_eq!(r.trouble(), Some(Trouble::Device("“Mic In” stopped sending audio".into())));
        shared.guarded(|| {});
        assert_eq!(r.trouble(), None, "it sent again");
        shared.armed.store(false, Ordering::Release);
        r.seen.1 = Instant::now().checked_sub(STALL).unwrap();
        assert_eq!(r.trouble(), None, "a quiet input that isn't recording is fine");
        shared.guarded(|| std::panic::panic_any("a bug in the input"));
        let mut ran = false;
        shared.guarded(|| ran = true);
        assert!(!ran);
        assert_eq!(r.trouble(), Some(Trouble::Device("the audio input hit an internal error (see the log)".into())));
        let gone = Arc::new(Shared::new(CaptureRing::new(64)));
        gone.fault.store(GONE, Ordering::Relaxed);
        assert_eq!(recorder(&gone).trouble(), Some(Trouble::Device("“Mic In” is unavailable".into())));
    }

    #[test]
    fn a_failed_write_is_trouble_while_still_recording() {
        let dir = std::env::temp_dir().join(format!("soundcraft-failed-take-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut take = TakeWriter::create(TrackId(1), vec![0], dir.join("Vox_01.wav"), 48_000).unwrap();
        take.file = File::open(&take.path).unwrap();
        let shared = Arc::new(Shared::new(CaptureRing::new(1 << 12)));
        shared.armed.store(true, Ordering::Release);
        let mut r = recorder(&shared);
        let sh = Arc::clone(&shared);
        let writer = std::thread::spawn(move || write_takes(&sh, vec![take], 1));
        shared.capture(&[0.5; 480], &mut 0);
        let trouble = (0..300).find_map(|_| {
            std::thread::sleep(Duration::from_millis(10));
            r.trouble()
        });
        assert!(matches!(&trouble, Some(Trouble::Disk(e)) if e.contains("Vox_01.wav")), "{trouble:?}");
        assert_eq!(r.trouble(), None, "reported once");
        shared.armed.store(false, Ordering::Release);
        assert!(writer.join().unwrap().1.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_crashed_writer_is_reported_and_its_partial_file_is_preserved() {
        let dir = std::env::temp_dir().join(format!("soundcraft-crashed-writer-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut take = TakeWriter::create(TrackId(1), vec![0], dir.join("Vox.wav"), 48_000).unwrap();
        take.write(&[0.25; 32], 1);
        take.write_header();
        let shared = Arc::new(Shared::new(CaptureRing::new(64)));
        shared.armed.store(true, Ordering::Release);
        let mut r = recorder(&shared);
        let writer = std::thread::spawn(move || {
            drop(take);
            std::panic::panic_any("simulated writer failure")
        });
        // Wait for the thread to terminate without joining it, as the live control poll does.
        for _ in 0..1_000 {
            if writer.is_finished() {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(writer.is_finished());
        r.writer = Some((writer, Vec::new()));
        assert_eq!(r.trouble(), Some(Trouble::Disk(WRITER_FAILED.into())));
        let (takes, problem) = r.finish();
        assert!(takes.is_empty(), "a crashed worker cannot return its completed-take list");
        assert_eq!(problem.as_deref(), Some(WRITER_FAILED));
        assert_eq!(decode(&dir.join("Vox.wav")), vec![vec![0.25; 32]], "journal recovery can still use the file");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn takes_stream_into_their_files() {
        let dir = std::env::temp_dir().join(format!("soundcraft-takes-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let gtr = TakeWriter::create(TrackId(1), vec![1, 2], dir.join("Gtr_01.wav"), 48_000).unwrap();
        let mut vox = TakeWriter::create(TrackId(2), vec![0], dir.join("Vox_01.wav"), 48_000).unwrap();
        assert!(TakeWriter::create(TrackId(3), vec![0], dir.join("Vox_01.wav"), 48_000).is_err(), "never overwrites a file");
        vox.write(&[0.5; 300], 3);
        vox.write_header();
        vox.write(&[0.25; 150], 3);
        assert_eq!(decode(&dir.join("Vox_01.wav"))[0].len(), 100, "the header covers what it has seen");

        let shared = Arc::new(Shared::new(CaptureRing::new(1 << 16)));
        shared.armed.store(true, Ordering::Release);
        let sh = Arc::clone(&shared);
        let writer = std::thread::spawn(move || write_takes(&sh, vec![gtr, vox], 3));
        let input: Vec<f32> = (0..3 * 4_800).map(|i| i as f32 / 20_000.0).collect();
        let mut owed = 0;
        input.chunks(3 * 480).for_each(|block| shared.capture(block, &mut owed));
        shared.armed.store(false, Ordering::Release);
        let (done, problem) = writer.join().unwrap();
        let channel = |c: usize| input.iter().skip(c).step_by(3).copied().collect::<Vec<f32>>();
        assert_eq!((done.len(), problem), (2, None));
        assert_eq!(decode(&done[0].1), vec![channel(1), channel(2)]);
        assert_eq!(decode(&done[1].1), vec![[vec![0.5; 100], vec![0.25; 50], channel(0)].concat()]);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_take_sums_itself_up_for_drawing_while_it_records() {
        let dir = std::env::temp_dir().join(format!("soundcraft-live-take-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let mut take = TakeWriter::create(TrackId(4), vec![2, 0], dir.join("Gtr_01.wav"), 48_000).unwrap();
        let input: Vec<f32> = (0..300).flat_map(|f| [f as f32, -1.0, -(f as f32)]).collect();
        take.write(&input[..3 * 100], 3);
        take.write(&input[3 * 100..], 3);
        let live = take.live.lock().unwrap().clone();
        assert_eq!(live, [[(-127.0, 0.0), (-255.0, -128.0), (-299.0, -256.0)], [(0.0, 127.0), (128.0, 255.0), (256.0, 299.0)]]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// Ring state behind the lock: interleaved input frames at the input device's rate, plus the
/// fractional read position (in input frames) of the next output sample.
struct RingState {
    data: std::collections::VecDeque<f32>,
    phase: f64,
}

/// A small interleaved FIFO from the input callback to the output callback, for input
/// monitoring. Both sides only `try_lock`; it holds at most ~0.5 s and drops the oldest frames.
/// The input device may run at a different rate than the session, so the reader converts
/// rates with linear interpolation (cheap, allocation-free, fine for monitoring).
pub struct InputRing {
    state: Mutex<RingState>,
    /// Callback-owned envelope, published without taking the monitor/overview lock.
    peaks: Box<[AtomicU32]>,
    pub channels: std::sync::atomic::AtomicUsize,
    /// Input device sample rate in Hz (0 when unknown, which reads as "same as the session").
    rate: u32,
    /// Most samples kept: half a second of input, or 24 000 frames when the rate is unknown.
    cap: usize,
    max_frames: AtomicUsize,
    latest_chunk: AtomicUsize,
}

impl InputRing {
    pub fn new(channels: usize, rate: u32) -> Arc<InputRing> {
        let ch = channels.max(1);
        let frames = if rate == 0 { 24_000 } else { rate as usize / 2 };
        Arc::new(InputRing {
            peaks: (0..ch).map(|_| AtomicU32::new(0)).collect(),
            state: Mutex::new(RingState { data: std::collections::VecDeque::with_capacity(rate.max(48_000) as usize * ch), phase: 0.0 }),
            channels: std::sync::atomic::AtomicUsize::new(ch),
            rate,
            cap: frames * ch,
            max_frames: AtomicUsize::new(frames),
            latest_chunk: AtomicUsize::new(0),
        })
    }

    /// Bound listening backlog in input-rate frames. Call from the control thread.
    /// Never retain less than one complete incoming callback or the reader's quantum.
    pub fn set_max_frames(&self, frames: usize) {
        let ch = self.channels.load(Ordering::Relaxed).max(1);
        self.max_frames.store(frames.clamp(2, (self.cap / ch).max(2)), Ordering::Relaxed);
    }

    /// Discard stale pre-roll on the control thread before attaching the output stream.
    pub fn clear(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.data.clear();
        state.phase = 0.0;
    }

    pub fn push(&self, samples: &[f32]) {
        let ch = self.channels.load(Ordering::Relaxed).max(1);
        let frames = samples.len() / ch;
        // Fast attack and 24 dB/s release at the actual input rate. A short
        // transient survives UI sampling, even when the monitor ring is busy.
        let rate = if self.rate == 0 { 48_000 } else { self.rate };
        let release = 10.0_f32.powf(-24.0 * frames as f32 / rate as f32 / 20.0);
        for (channel, peak) in self.peaks.iter().enumerate().take(ch) {
            let latest = samples.iter().skip(channel).step_by(ch).fold(0.0_f32, |max, sample| max.max(finite(*sample).abs()));
            let previous = f32::from_bits(peak.load(Ordering::Relaxed));
            peak.store(latest.max(previous * release).to_bits(), Ordering::Relaxed);
        }
        self.latest_chunk.store(frames, Ordering::Relaxed);
        if let Ok(mut s) = self.state.try_lock() {
            let cap_frames = self.max_frames.load(Ordering::Relaxed).max(frames.saturating_add(2)).min(self.cap / ch);
            let keep = frames.min(cap_frames).saturating_mul(ch);
            let end = frames.saturating_mul(ch);
            let Some(tail) = samples.get(end.saturating_sub(keep)..end) else { return };
            let discard = s.data.len().saturating_add(keep).saturating_sub(cap_frames * ch);
            if discard > 0 {
                // Drop whole old frames before extending: even oversized callbacks
                // cannot grow the prepared VecDeque or leave interleaved channels skewed.
                s.data.drain(..discard);
                s.phase = 0.0;
            }
            s.data.extend(tail.iter().copied().map(finite));
        }
    }

    /// Control-thread metering only; does not drain recording or monitoring audio.
    pub fn input_peaks(&self) -> Vec<f32> {
        self.peaks.iter().map(|peak| f32::from_bits(peak.load(Ordering::Relaxed))).collect()
    }

    /// Fill `out` (planar, at `session_rate`) with up to `frames` frames, converting from the
    /// input rate. Missing frames are silence.
    pub fn pop_into(&self, out: &mut [Vec<f32>], frames: usize, session_rate: u32) {
        let ch = self.channels.load(Ordering::Relaxed).max(1);
        for c in out.iter_mut() {
            c.iter_mut().take(frames).for_each(|x| *x = 0.0);
        }
        let Ok(mut s) = self.state.try_lock() else { return };
        // Input frames consumed per output frame.
        let step = if self.rate == 0 || session_rate == 0 { 1.0 } else { f64::from(self.rate) / f64::from(session_rate) };
        let needed = (frames as f64 * step).ceil() as usize + 2;
        let keep_frames = self.max_frames.load(Ordering::Relaxed).max(self.latest_chunk.load(Ordering::Relaxed)).max(needed);
        let discard_frames = (s.data.len() / ch).saturating_sub(keep_frames);
        if discard_frames > 0 {
            s.data.drain(..discard_frames * ch);
            s.phase = 0.0;
        }
        for f in 0..frames {
            // Drop input frames we have moved past; keep the one at `phase` and the next one.
            while s.phase >= 1.0 && s.data.len() >= 2 * ch {
                s.data.drain(..ch);
                s.phase -= 1.0;
            }
            if s.phase >= 1.0 || s.data.len() < 2 * ch {
                break;
            }
            let fr = s.phase as f32;
            for c in 0..ch {
                let a = s.data.get(c).copied().unwrap_or(0.0);
                let b = s.data.get(ch + c).copied().unwrap_or(a);
                if let Some(x) = out.get_mut(c).and_then(|o| o.get_mut(f)) {
                    *x = a + (b - a) * fr;
                }
            }
            s.phase += step;
        }
    }
}

#[cfg(test)]
mod ring_tests {
    use super::InputRing;

    #[test]
    fn capture_peak_envelope_survives_a_busy_monitor_and_releases_in_input_time() {
        let ring = InputRing::new(2, 48_000);
        let lock = ring.state.lock().unwrap();
        ring.push(&[0.8, -0.4, f32::NAN, f32::INFINITY]);
        assert_eq!(ring.input_peaks(), vec![0.8, 0.4]);
        drop(lock);
        ring.push(&vec![0.0; 48_000 * 2]);
        let peaks = ring.input_peaks();
        assert!((peaks[0] - 0.8 * 10.0_f32.powf(-24.0 / 20.0)).abs() < 1e-6);
        assert!((peaks[1] - peaks[0] / 2.0).abs() < 1e-6);
    }

    /// A 1 kHz sine recorded at 16 kHz must come out as a 1 kHz sine at 48 kHz, across calls.
    #[test]
    fn monitor_converts_input_rate_to_session_rate() {
        let ring = InputRing::new(1, 16_000);
        let input: Vec<f32> = (0..4_000).map(|i| (2.0 * std::f64::consts::PI * 1_000.0 * f64::from(i) / 16_000.0).sin() as f32).collect();
        ring.push(&input);
        let mut out = vec![vec![0.0f32; 48]];
        let mut k = 0usize;
        for _ in 0..10 {
            ring.pop_into(&mut out, 48, 48_000);
            for v in &out[0] {
                let want = (2.0 * std::f64::consts::PI * 1_000.0 * k as f64 / 48_000.0).sin() as f32;
                assert!((v - want).abs() < 0.05, "sample {k}: got {v}, want {want}");
                k += 1;
            }
        }
    }

    /// Input faster than the session (96 kHz into 48 kHz): every other frame is read, in order.
    #[test]
    fn monitor_decimates_when_input_is_faster() {
        let ring = InputRing::new(1, 96_000);
        let input: Vec<f32> = (0..1_000).map(|i| i as f32).collect();
        ring.push(&input);
        let mut out = vec![vec![0.0f32; 100]];
        ring.pop_into(&mut out, 100, 48_000);
        for (k, v) in out[0].iter().enumerate() {
            assert_eq!(*v, (2 * k) as f32, "output {k}");
        }
    }

    /// Same rate: samples pass through unchanged.
    #[test]
    fn monitor_passes_through_at_equal_rates() {
        let ring = InputRing::new(2, 48_000);
        let input: Vec<f32> = (0..200).map(|i| i as f32).collect();
        ring.push(&input);
        let mut out = vec![vec![0.0f32; 50], vec![0.0f32; 50]];
        ring.pop_into(&mut out, 50, 48_000);
        assert_eq!(out[0][0], 0.0);
        assert_eq!(out[1][0], 1.0);
        assert_eq!(out[0][49], 98.0);
    }

    /// Nothing buffered: silence, not stale data.
    #[test]
    fn monitor_underrun_is_silence() {
        let ring = InputRing::new(1, 16_000);
        let mut out = vec![vec![1.0f32; 16]];
        ring.pop_into(&mut out, 16, 48_000);
        assert!(out[0].iter().all(|v| *v == 0.0));
    }

    /// NaN and infinities from the device monitor as silence.
    #[test]
    fn monitor_silences_nan_and_infinite_input() {
        let ring = InputRing::new(1, 48_000);
        ring.push(&[0.5, f32::NAN, f32::INFINITY, -0.5, 0.0, 0.0, 0.0, 0.0]);
        let mut out = vec![vec![1.0f32; 4]];
        ring.pop_into(&mut out, 4, 48_000);
        assert_eq!(out[0], [0.5, 0.0, 0.0, -0.5]);
    }
    #[test]
    fn low_latency_ring_drops_old_frames_and_keeps_channel_alignment_without_allocating() {
        let ring = InputRing::new(2, 48000);
        ring.set_max_frames(4);
        ring.push(&[1.0, 11.0, 2.0, 12.0, 3.0, 13.0, 4.0, 14.0]);
        ring.push(&[5.0, 15.0, 6.0, 16.0, 7.0, 17.0, 8.0, 18.0]);
        let mut out = vec![vec![0.0; 2]; 2];
        ring.pop_into(&mut out, 2, 48000);
        assert_eq!(out[0], vec![5.0, 6.0]);
        assert_eq!(out[1], vec![15.0, 16.0]);
        let large = vec![0.25; 100_000];
        let measured = allocation_counter::measure(|| {
            ring.push(&large);
            ring.pop_into(&mut out, 2, 48000);
        });
        assert_eq!(measured.count_total, 0);
        assert_eq!(measured.count_current, 0);
        ring.clear();
        ring.pop_into(&mut out, 2, 48000);
        assert_eq!(out, vec![vec![0.0; 2]; 2]);
    }
}
