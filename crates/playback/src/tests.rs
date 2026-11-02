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
    /// Play reports the device's refusal when `starts` is false.
    fn play(&self) -> Result<(), cpal::Error> {
        if self.starts { Ok(()) } else { Err(cpal::Error::new(cpal::ErrorKind::DeviceNotAvailable)) }
    }
    /// Pause always succeeds on the fake stream.
    fn pause(&self) -> Result<(), cpal::Error> {
        Ok(())
    }
    /// The fake stream has no clock and no fixed buffer; neither is used by these tests.
    fn buffer_size(&self) -> Result<cpal::FrameCount, cpal::Error> {
        Ok(0)
    }
    /// The fake stream has no usable clock; these tests do not read it.
    fn now(&self) -> cpal::StreamInstant {
        cpal::StreamInstant::from_nanos(0)
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
    let won = first_started([1u32, 2], |&n| if n == 1 { Err("no such buffer size".to_string()) } else { Ok(Fake { starts: true }) }, start_stream);
    assert_eq!(won.map(|(n, _)| n), Some(2));
}

/// Build a cpal configuration range for tests with a 64..1024-frame buffer.
fn range(channels: u16, min: u32, max: u32, format: cpal::SampleFormat) -> cpal::SupportedStreamConfigRange {
    cpal::SupportedStreamConfigRange::new(channels, min, max, cpal::SupportedBufferSize::Range { min: 64, max: 1024 }, format)
}

/// Rates are tried in order, so a stale configured rate falls through to the session rate; when no requested rate
/// is offered the caller uses the device default and resamples.
#[test]
fn output_config_uses_the_first_requested_rate_the_device_offers() {
    let configs = [
        range(2, 44_100, 48_000, cpal::SampleFormat::I16),
        range(2, 44_100, 96_000, cpal::SampleFormat::F32),
        range(8, 44_100, 96_000, cpal::SampleFormat::F32),
    ];
    let (config, buffers, format) = pick_output_config(&configs, &[96_000, 48_000], 8).unwrap();
    assert_eq!((config.sample_rate, config.channels, config.buffer_size), (96_000, 8, cpal::BufferSize::Default));
    assert_eq!(format, cpal::SampleFormat::F32);
    assert_eq!(buffers, cpal::SupportedBufferSize::Range { min: 64, max: 1024 });
    // 192 kHz is not offered: the session's 48 kHz is tried next.
    let (config, _, format) = pick_output_config(&configs, &[192_000, 48_000], 8).unwrap();
    assert_eq!((config.sample_rate, config.channels, format), (48_000, 8, cpal::SampleFormat::F32));
    // No requested rate is offered: the caller uses the device's default config and resamples.
    assert!(pick_output_config(&configs, &[192_000], 8).is_none());
}

/// At a supported rate, a preferred sample format wins over more channels; output stays between stereo and the
/// requested channel cap, and mono-only or wider devices take the default path.
#[test]
fn output_config_prefers_format_over_channels_and_needs_stereo_within_the_cap() {
    let configs = [range(8, 44_100, 48_000, cpal::SampleFormat::I16), range(2, 44_100, 48_000, cpal::SampleFormat::F32)];
    let (config, _, format) = pick_output_config(&configs, &[48_000], 8).unwrap();
    assert_eq!((config.channels, format), (2, cpal::SampleFormat::F32), "a preferred format wins over more channels");
    let (config, _, _) = pick_output_config(&configs, &[48_000], 2).unwrap();
    assert_eq!(config.channels, 2, "the channel cap applies");
    let mono = [range(1, 44_100, 48_000, cpal::SampleFormat::F32)];
    assert!(pick_output_config(&mono, &[48_000], 8).is_none(), "a mono-only device takes the default config path");
    let surround = [range(6, 44_100, 48_000, cpal::SampleFormat::F32)];
    assert!(pick_output_config(&surround, &[48_000], 2).is_none(), "a wider device is not narrowed here");
    assert_eq!(pick_output_config(&surround, &[48_000], 6).map(|(c, _, _)| c.channels), Some(6));
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

/// A mono track playing a quiet signal with one short spike in the block starting at
/// `spike_at`, and an audio thread ready to play it.
fn spike_session(spike_at: usize) -> (Arc<Shared>, AudioState, super::Sender<Cmd>, TrackId) {
    use soundcraft_model::{ChannelFormat, Clip, SourceAudio, SourceId, TrackKind};
    let mut s = soundcraft_model::Session::default();
    let t = s.add_track(TrackKind::Audio, ChannelFormat::Mono, None);
    let mut x = vec![0.01f32; 48_000];
    x[spike_at..spike_at + 32].fill(0.9);
    s.pool.insert(SourceId(5), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![x] })));
    let id = s.new_clip_id();
    s.track_mut(t).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(id, "x", SourceId(5), 0, 0, 48_000));
    let shared = Arc::new(Shared {
        position: AtomicI64::new(0),
        playing: AtomicBool::new(false),
        speed: AtomicU32::new(1000),
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    // As Player does before the device starts: the control thread shapes the shared meters.
    prepare_meter_layout(&shared, &s);
    let (tx, rx) = test_channel();
    let st = AudioState::new(rx, Arc::clone(&shared), Arc::new(s), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
    (shared, st, tx, t)
}

#[test]
fn a_spike_in_one_block_among_quiet_ones_reaches_the_ui() {
    // Eight 512-frame blocks per callback, as between two UI polls; the spike is in the 3rd.
    let (shared, mut st, _tx, t) = spike_session(2 * 512 + 100);
    let mut out = vec![0.0f32; 2 * 8 * 512];
    st.fill(&mut out, 2);
    // The last block alone (what the UI used to see) is quiet.
    assert!(st.mix.meters[&t].peak[0] < 0.05);
    let mut ui = MeterSnapshot::default();
    take_meter_snapshot(&mut shared.meters.lock().unwrap(), &mut ui);
    assert!(ui.tracks[&t].peak[0] > 0.85, "{:?}", ui.tracks[&t]);
    assert!(ui.tracks[&t].peaks[0] > 0.85);
    assert!(ui.main.peak[0] > 0.5 && ui.main.peaks.iter().all(|p| *p > 0.5), "{:?}", ui.main);
    // Taking starts a new period: the next quiet blocks read quiet.
    st.fill(&mut out, 2);
    take_meter_snapshot(&mut shared.meters.lock().unwrap(), &mut ui);
    assert!(ui.tracks[&t].peak[0] < 0.05 && ui.main.peak[0] < 0.05);
    // No block since the last take: the reading stays rather than dropping to silence.
    take_meter_snapshot(&mut shared.meters.lock().unwrap(), &mut ui);
    assert!(ui.tracks[&t].peak[0] > 0.005);
}

#[test]
fn a_spike_rendered_while_the_ui_holds_the_meters_arrives_later() {
    let (shared, mut st, _tx, t) = spike_session(100);
    let mut out = vec![0.0f32; 2 * 512];
    {
        let _held = shared.meters.lock().unwrap();
        st.fill(&mut out, 2);
    }
    st.fill(&mut out, 2);
    let mut ui = MeterSnapshot::default();
    take_meter_snapshot(&mut shared.meters.lock().unwrap(), &mut ui);
    assert!(ui.tracks[&t].peak[0] > 0.85, "{:?}", ui.tracks[&t]);
}

#[test]
fn stopping_zeroes_meters_in_place_and_removed_tracks_are_pruned() {
    let (shared, mut st, tx, t) = spike_session(100);
    let mut out = vec![0.0f32; 2 * 512];
    st.fill(&mut out, 2);
    tx.send(Cmd::Stop).unwrap();
    // The first callback after the stop plays its fade-out; the next is silent.
    st.fill(&mut out, 2);
    st.fill(&mut out, 2);
    {
        let m = shared.meters.lock().unwrap();
        let e = m.tracks.get(&t).expect("kept while stopped, so playing again does not allocate it");
        assert!(e.peak == [0.0; 2] && e.peaks.iter().all(|p| *p == 0.0) && e.peaks.len() == 1, "{e:?}");
        assert!(m.fresh, "the UI sees the silence");
    }
    // The track is deleted: the control thread reshapes both meter snapshots for the new session.
    let empty = Arc::new(soundcraft_model::Session::default());
    prepare_meter_layout(&shared, &empty);
    tx.send(session_command(empty, Vec::new(), 0)).unwrap();
    tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
    st.fill(&mut out, 2);
    assert!(!shared.meters.lock().unwrap().tracks.contains_key(&t));
    assert!(!st.unsent.tracks.contains_key(&t) && !st.mix.meters.contains_key(&t));
}

#[test]
fn a_take_without_new_blocks_still_drops_deleted_tracks() {
    let mut session = soundcraft_model::Session::default();
    let gone = session.add_track(TrackKind::Audio, soundcraft_model::ChannelFormat::Mono, None);
    let mut source = MeterSnapshot::shaped(&session);
    source.accumulate(&StripMeter::default(), &HashMap::from([(gone, StripMeter::default())]));
    let mut ui = MeterSnapshot::default();
    take_meter_snapshot(&mut source, &mut ui);
    session.tracks.retain(|track| track.id != gone);
    source.shape(&session);
    take_meter_snapshot(&mut source, &mut ui);
    assert!(ui.tracks.is_empty());
}

#[test]
fn a_channel_count_change_takes_the_new_stereo_peaks() {
    let mut acc = StripMeter { peak: [0.9, 0.9], peaks: vec![0.9], ..StripMeter::default() };
    acc.accumulate(&StripMeter { peak: [0.9, 0.0], peaks: vec![0.9, 0.0], ..StripMeter::default() });
    assert_eq!(acc.peak, [0.9, 0.0]);
}

#[test]
fn gain_reduction_keeps_its_maximum_until_taken() {
    let mut acc = StripMeter::default();
    for gr in [1.0, 9.0, 2.0, f32::NAN] {
        acc.accumulate(&StripMeter { peak: [0.1, 0.1], peaks: vec![0.1], gain_reduction_db: gr, ..StripMeter::default() });
    }
    assert_eq!(acc.gain_reduction_db, 9.0);
    acc.reset_levels();
    assert_eq!((acc.gain_reduction_db, acc.peak, acc.peaks.len()), (0.0, [0.0; 2], 1));
}

#[test]
fn callback_meter_publication_skips_unprepared_tracks_without_inserting() {
    let track = TrackId(7);
    let shared = Shared {
        position: AtomicI64::new(0),
        playing: AtomicBool::new(false),
        speed: AtomicU32::new(1000),
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    };
    let mut source = HashMap::new();
    source.insert(track, StripMeter { peak: [0.5, 0.25], peaks: vec![0.5, 0.25], ..StripMeter::default() });

    shared.meters.lock().unwrap().accumulate(&StripMeter::default(), &source);
    let mut unsent = MeterSnapshot::default();
    unsent.accumulate(&StripMeter::default(), &source);

    assert!(shared.meters.lock().unwrap().tracks.is_empty());
    assert!(unsent.tracks.is_empty());
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
fn one_session_update_is_applied_per_callback_drain() {
    let initial = Arc::new(soundcraft_model::Session::default());
    let shared = Arc::new(Shared {
        position: AtomicI64::new(0),
        playing: AtomicBool::new(false),
        speed: AtomicU32::new(1000),
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    let (tx, rx) = test_channel();
    let (mut retire_tx, mut retire_rx) = return_channel(OLD_SESSIONS);
    for _ in 0..OLD_SESSIONS {
        let _ = retire_tx.try_send(RetiredSession {
            _session: Some(Arc::new(soundcraft_model::Session::default())),
            _mix: None,
            _input: None,
            _meters: None,
        });
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    let (tx, rx) = test_channel();
    let mut st = AudioState::new(rx, shared, Arc::new(s), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    // These tests compare exact output from the first sample.
    st.fades = false;
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
fn resampling_at_a_short_block_edge_does_not_mix_in_stale_frames() {
    use soundcraft_model::ChannelFormat;
    // DC for the first long block, then silence. The device runs slower than the session, so
    // reads land between frames; a short block leaves the long block's DC past `pending_len`.
    let mut st = audio_output_state(ChannelFormat::Stereo, vec![vec![0.8; BLOCK]; 2]);
    st.device_rate = 44_100.0;
    let mut long = vec![0.0; 2 * 470];
    st.fill(&mut long, 2);
    assert_eq!(st.pending_len, BLOCK);
    let mut short = [0.0; 2 * 8];
    for call in 0..20 {
        st.fill(&mut short, 2);
        assert!(st.pending_len < BLOCK);
        // The first frame of the first short call still reads the DC block's last frame.
        let from = if call == 0 { 2 } else { 0 };
        assert!(short[from..].iter().all(|x| x.abs() < 1e-6), "call {call}: {short:?}");
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
fn interpolation_overflow_between_huge_samples_never_reaches_the_device() {
    use soundcraft_model::ChannelFormat;
    let alternating: Vec<f32> = (0..512).map(|i| if i % 2 == 0 { -3.4e38 } else { 3.4e38 }).collect();
    for (format, nch, device) in [(ChannelFormat::Stereo, 2, 2), (ChannelFormat::Stereo, 2, 8), (ChannelFormat::Mono, 1, 1)] {
        let mut st = audio_output_state(format, vec![alternating.clone(); nch]);
        let mut out = vec![0.5; 64 * device];
        st.fill(&mut out, device);
        assert!(out.iter().all(|x| x.is_finite() && (-1.0..=1.0).contains(x)), "{format:?} on {device} channels: {out:?}");
    }
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    let (tx, rx) = test_channel();
    let initial = Arc::new(soundcraft_model::Session::default());
    let mut st = AudioState::new(rx, Arc::clone(&shared), Arc::clone(&initial), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    st.fades = false;
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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

/// An editor whose plugin said its state changed (CLAP `state.mark_dirty`).
struct Dirty(bool);

impl PluginEditor for Dirty {
    fn open(&mut self) -> Result<(), String> {
        Ok(())
    }
    fn close(&mut self) {}
    fn is_open(&self) -> bool {
        true
    }
    fn idle(&mut self) -> Vec<(String, f32)> {
        Vec::new()
    }
    fn state_changed(&mut self) -> bool {
        std::mem::take(&mut self.0)
    }
}

#[test]
fn changed_states_are_reported_once_and_read_for_the_session() {
    let mut side = Side::new(48_000.0, std::sync::mpsc::channel().1);
    let (a, b) = (TrackId(1), TrackId(2));
    let synth = (a, soundcraft_mix::INSTRUMENT_SLOT);
    side.states.insert((a, 0), Box::new(|| Some(b"a0".to_vec())));
    side.states.insert((a, 1), Box::new(|| Some(b"a1".to_vec())));
    side.states.insert(synth, Box::new(|| Some(b"synth".to_vec())));
    side.states.insert((b, 0), Box::new(|| Some(b"b0".to_vec())));
    side.editors.insert((a, 0), Box::new(Dirty(true)));
    side.editors.insert((a, 1), Box::new(Dirty(false)));
    side.editors.insert((b, 0), Box::new(Dirty(true)));
    side.refused.insert((b, 0));
    let mut changed = side.changed_states();
    changed.sort();
    assert_eq!(changed, vec![(a, 0), (b, 0)]);
    assert!(side.changed_states().is_empty(), "each change is reported once");
    let want = vec![(a, 0, b"a0".to_vec()), (a, soundcraft_mix::INSTRUMENT_SLOT, b"synth".to_vec())];
    assert_eq!(side.capture_states_of(&[(a, 0), synth, (b, 0)]), want);
}

#[test]
fn a_state_marked_changed_while_loading_is_not_a_change() {
    let mut side = Side::new(48_000.0, std::sync::mpsc::channel().1);
    let k = (TrackId(1), 0);
    side.states.insert(k, Box::new(|| Some(b"loaded".to_vec())));
    side.add_editor(k, Box::new(Dirty(true)));
    assert!(side.changed_states().is_empty());
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            stop_fading: AtomicBool::new(false),
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
        // The first callback fades the old position out (10 ms = 480 frames); then nothing of it.
        st.fill(&mut out, 2);
        assert!(out.chunks(2).skip(481).all(|f| f[0].abs() < 1e-6), "{plugin}: the fade-out ends in silence");
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
    session.pool.insert(SourceId(1), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal] })));
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    prepare_meter_layout(&shared, &session);
    let fault = Fault::default();
    let (tx, rx) = channel();
    let (old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
    tx.send(Cmd::OldSessions(old_tx)).unwrap();
    tx.send(Cmd::Play { from: 0, end: None, looped: None }).unwrap();
    let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    state.fades = false;
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
            if let Cmd::Session(_, mix, _) = &command {
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    let (tx, rx) = test_channel();
    let (mut old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
    for _ in 0..OLD_SESSIONS {
        old_tx.try_send(RetiredSession { _session: None, _mix: None, _input: None, _meters: None }).ok();
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
    session.pool.insert(SourceId(1), Arc::new(SourceAudio::new(soundcraft_audio_io::AudioBuffer { sample_rate: 48_000, channels: vec![signal] })));
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(true),
        feedback_protection: AtomicBool::new(false),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    prepare_meter_layout(&shared, &session);
    let (tx, rx) = channel();
    let (old_tx, mut old_rx) = return_channel(OLD_SESSIONS);
    tx.send(Cmd::OldSessions(old_tx)).unwrap();
    let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    state.fades = false;
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
            monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
            monitor_use_rack: AtomicBool::new(true),
            direct_monitor_enabled: AtomicBool::new(true),
            feedback_protection: AtomicBool::new(false),
            speaker_protection: AtomicBool::new(false),
            stop_fading: AtomicBool::new(false),
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
        session.track_mut(track).unwrap().playlist_mut().unwrap().clips.push(Clip::audio(clip_id, "playing", source, 0, 0, (FRAMES * BLOCKS) as i64));
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
        monitor_gain: AtomicU32::new(1.0_f32.to_bits()),
        monitor_use_rack: AtomicBool::new(true),
        direct_monitor_enabled: AtomicBool::new(false),
        feedback_protection: AtomicBool::new(true),
        speaker_protection: AtomicBool::new(false),
        stop_fading: AtomicBool::new(false),
        meters: Mutex::new(MeterSnapshot::default()),
    });
    prepare_meter_layout(&shared, &session);
    let (_tx, rx) = channel();
    let mut state = AudioState::new(rx, Arc::clone(&shared), Arc::new(session.clone()), 48_000.0, sync_channel(RETIRED_PLUGINS).0);
    state.fades = false;
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

#[test]
fn locate_while_playing_fades_out_jumps_and_fades_in() {
    use soundcraft_model::ChannelFormat;
    let mut st = audio_output_state(ChannelFormat::Stereo, vec![vec![0.8; 48_000]; 2]);
    st.fades = true;
    let fade = (48_000.0 * 0.01) as usize;
    let mut out = vec![0.0f32; 2 * 1024];
    st.fill(&mut out, 2);
    let steady = out[2 * 1000];
    assert!(steady > 0.5);

    let (tx, rx) = test_channel();
    st.rx = rx;
    st.shared.position.store(100, Ordering::Relaxed);
    tx.send(Cmd::Locate(100)).unwrap();
    let mut out = vec![0.0f32; 2 * 2048];
    st.fill(&mut out, 2);
    let left: Vec<f32> = out.chunks(2).map(|f| f[0]).collect();
    assert!(left[0] > steady * 0.9, "the locate fades rather than cutting");
    assert!(left[..fade].windows(2).all(|w| w[1] <= w[0]), "the old position ramps down");
    let silent = left.iter().position(|x| *x == 0.0).expect("the fade-out reaches silence");
    assert!(silent <= fade + 1);
    assert!(left[silent..silent + fade].windows(2).all(|w| w[1] >= w[0]), "the new position ramps up");
    assert!(left[silent + fade + 1..].iter().all(|x| (x - steady).abs() < 1e-6), "full level after the fade-in");
    assert!(st.playing && st.locating.is_none());
    let played = (2048 - silent) as i64;
    assert!(st.pos >= 100 + played - 1 && st.pos <= 100 + played + 512, "playback continued from the target: {}", st.pos);
}

#[test]
fn locate_then_stop_during_the_fade_stops_at_the_target() {
    use soundcraft_model::ChannelFormat;
    let mut st = audio_output_state(ChannelFormat::Stereo, vec![vec![0.8; 48_000]; 2]);
    st.fades = true;
    let mut out = vec![0.0f32; 2 * 1024];
    st.fill(&mut out, 2);
    let (tx, rx) = test_channel();
    st.rx = rx;
    tx.send(Cmd::Locate(7)).unwrap();
    tx.send(Cmd::Stop).unwrap();
    st.fill(&mut out, 2);
    assert!(!st.playing && st.locating.is_none());
    assert_eq!(st.pos, 7, "a stop during a locate's fade stops where the locate pointed");
}

#[test]
fn playback_fades_in_on_play_and_out_on_stop() {
    use soundcraft_model::ChannelFormat;
    let mut st = audio_output_state(ChannelFormat::Stereo, vec![vec![0.8; 48_000]; 2]);
    st.fades = true;
    let fade = (48_000.0 * 0.01) as usize;
    let mut out = vec![0.0f32; 2 * 1024];
    st.fill(&mut out, 2);
    let steady = out[2 * 1000];
    assert!(steady > 0.5);
    assert!(out[0] < steady * 0.01, "playback starts from silence: {}", out[0]);
    assert!(out.chunks(2).take(fade).collect::<Vec<_>>().windows(2).all(|w| w[1][0] >= w[0][0]), "the start ramps up");
    assert!(out.chunks(2).skip(fade + 1).all(|f| (f[0] - steady).abs() < 1e-6), "full level after the fade");

    let stopped_at = st.pos;
    st.shared.stop_fading.store(true, Ordering::Relaxed);
    st.shared.playing.store(false, Ordering::Relaxed);
    let (tx, rx) = test_channel();
    st.rx = rx;
    tx.send(Cmd::Stop).unwrap();
    st.fill(&mut out, 2);
    assert!(out[0] > steady * 0.9, "the stop fades rather than cutting");
    assert!(out.chunks(2).take(fade).collect::<Vec<_>>().windows(2).all(|w| w[1][0] <= w[0][0]), "the stop ramps down");
    assert!(out.chunks(2).skip(fade + 1).all(|f| f[0] == 0.0 && f[1] == 0.0), "silent once faded");
    assert!(!st.playing && st.stopping.is_none());
    assert!(!st.shared.stop_fading.load(Ordering::Relaxed), "the player can be dropped now");
    assert!(!st.shared.playing.load(Ordering::Relaxed), "the transport read stopped throughout");
    assert_eq!(st.pos, stopped_at, "the fade's extra frames do not move the stop position");
}
