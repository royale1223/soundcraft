# SoundCraft roadmap: milestones and what's next

> **Last reviewed:** 2026-10-15 · **Last updated:** 2026-10-18 · **Change:** minor (downstream 0.8.0 merge and Rust 1.95 minimum; mobile patches and parity unchanged) · **Target:** Avid Pro Tools Ultimate 2026.4.1

Forward-looking plan. Status and numbers: [`../ROADMAP.md`](../ROADMAP.md) and
[`target-app-parity.md`](target-app-parity.md); the ranked work list: [`gaps.md`](gaps.md).
Hours are Opus 5.5 agent-hours.

## Downstream input monitoring (2026-10-18)

Player monitor gain and rack choice are atomic control-side settings. The mixer
ramps monitor gain after inserts, independently of stored track gain, and offers
a dry path that skips inserts and compensation waits on live input strips.
InputRing can retain a small requested number of frames and discard stale
monitor backlog without changing the separate disk-capture queue. Its callback
uses try-locks and preallocated storage; no blocking mutex was introduced.
Host regressions cover channel alignment, bounded storage, monitor gain and
PDC bypass. Device round-trip latency, underruns and route behavior still need
physical-device measurements; hardware gap G-01 and parity metrics remain open.

## Downstream mobile preparation

The LiteDAW branch prepares first-render meter channels, parameter caches,
pre-fader/output buffers, and zero-delay channel storage during mixer sync.
Parameter-name edits invalidate caches; changing values retains plugin state.
Empty hosted-instance adoption no longer discards an already prepared layout.
Host regressions verify retained allocations and shared-path first-block PCM.
This remains partial hardening: structural edits still synchronize on callbacks,
nonzero delay changes may allocate, and physical-device deadlines are unmeasured.
The built-in-only mixer also exposes fresh-layout compensation preparation and
runtime ownership transfer. Exact host PCM regressions preserve reverb tails,
compensation histories, and parameter smoothing across structural edits. Replaced
instances remain in the retired engine. Built-in-only playback now connects that API through bounded command and
retirement rings. Structural session changes prepare on the control thread;
whole old mixers/snapshots and replaced input rings retire there. Full return
rings defer later commands without overwriting owned state. Command overflow
silences output and latches a diagnostic for control-thread recovery. Probe preparation
temporarily holds another mixer; its control-thread cost is unmeasured. Dynamic
latency automation is now covered by prepared processor bounds and retained delay
capacity for current built-ins. Guarded mobile callback regressions measure zero
Rust heap allocations/frees during track edits, reverb updates, and combined
lookahead automation, with sample equality against a persistent mixer. Larger
prepared rings retain capacity while inheriting old delay history. These are host
Rust allocation measurements; driver allocation behavior, physical deadlines,
clip-effect preparation/state, and other unmeasured paths remain open.

A target-workload regression also exercises one stereo backing plus four mono
vocal strips with EQ/compressor/de-esser/reverb and a stereo master. At 48 kHz,
both 128/256-frame callbacks produce exact reference PCM across 1,024 blocks,
zero Rust allocations/frees, and nonzero meters for every audio track. Printed
instrumented host timings are diagnostic only: phone deadlines, driver behavior,
drift, and long-duration physical recording remain unmeasured.

These changes do not alter the alpha gate or parity estimates below.

## Current focus

1. **Merge or close the open bug-fix PRs** (#123–#134, #146, #148, #75) and fix the rest of the
   correctness backlog (G-04). ≈ 15–25 h.
2. **Audio interface basics** (G-01): device selection, buffer size, duplex stream, ASIO. ≈ 25–40 h.
3. **MIDI devices** (G-02): input to armed tracks and instruments, output, thru, clock. ≈ 20–35 h.
4. **macOS plugin hosting** (G-05): library-validation entitlement now (#153), then AU editors. ≈ 15–25 h.
5. **A PR CI that runs `cargo xtask ci`** on Linux, macOS and Windows (G-14). ≈ 5–10 h.

## Alpha gate

The core workflows a Pro Tools professional does every day, checked end to end on macOS (the
main platform). Any **no** keeps SoundCraft **pre-alpha**, and those rows are the alpha checklist
(craftrules `standards/progress-docs.md`, Stages).

| Workflow | Works end to end? | Evidence | Hours to pass |
|---|---|---|---:|
| Record audio from a chosen interface at a chosen buffer size, monitor, punch, keep the takes | **no** | Default output and input devices only. Fixed 512-sample block. Separate input and output streams. `setup.playback_engine` params ignored by playback. Takes held in RAM until stop (PR #84). No input device choice (#42). | 35–55 |
| Record MIDI from a controller and play it back through an instrument | **no** | Playback of drawn and imported MIDI works. No MIDI device layer at all, so nothing can be recorded and instruments can't be played live (G-02). | 20–35 |
| Edit: select, trim, fade, comp playlists, Slip/Shuffle/Grid | yes (rough) | Every Edit menu command; property-tested with undo; open bugs #3, #5, #140 don't block the workflow | — |
| Mix with built-in and third-party plugins, sends, busses, automation | partial (passes on CLAP/AU) | 26 built-in processors; CLAP and AU host on the signed macOS build, but its missing entitlement blocks third-party VST3 (#153). No AU plugin windows yet; VST3 windows on macOS only | 1–2 (entitlement) |
| Bounce mix and stems; save and reopen the session | yes (rough) | WAV/BWF/AIFF/FLAC bounce, stems, `.scraft` save/reopen with plugin state; bugs #59, #103 in bus bounce and aux latency (PRs open) | — |
| Exchange a session with Pro Tools or a picture editor | **no** | No `.ptx`, AAF or OMF import or export (#40, G-03) | 50–90 |

**Result: pre-alpha.** Three workflows fail. Passing them is ~110–180 h, about 70 % of it
parallelizable (interfaces, MIDI and AAF are independent crates).

## Milestones

Status: ✅ done · 🟡 in progress · ⬜ not started.

| # | Milestone | Status | Remaining (h) |
|---|---|---|---:|
| M0–M4 | Workspace, gates, model, I/O, DSP, mix engine, command engine, Edit/Mix UI, playback, metering, plugins, bounce | ✅ | — |
| M5 | Editing depth (tools, modes, playlists, fades, comping, clip groups) | ✅ (bugs open, G-04) | in G-04 |
| M6 | Recording, punch, loop record, monitoring | 🟡 first cut; device choice and record-to-disk missing | 40–60 |
| M7 | MIDI and notation editing | 🟡 | 50–80 |
| M8 | CLI, control channel, MCP, agent acceptance test | ✅ | 3 |
| M9 | AudioSuite breadth ✅, Elastic Audio 🟡, Beat Detective 🟡 | 🟡 | 40–60 |
| M10 | Public releases on all platforms | ✅ v0.3.0 (2026-10-08) and v0.4.0 (2026-10-10) released for every platform | — |
| M11 | Third-party plugin hosting (CLAP, VST3, AU), video, surround | 🟡 mostly done | 40–60 |
| M12 | **Hardware**: audio interfaces, MIDI devices, sync, control surfaces | ⬜ | 110–180 |
| M13 | **Interchange**: AAF/OMF import/export, disk streaming, record to disk | ⬜ | 65–115 |
| M14 | **Fidelity harness** against Pro Tools (null tests, plugin responses) | ⬜ | 20–30 |
| M15 | **Localization**: catalog, fonts/RTL, the 8 Pro Tools languages first | ⬜ (PRs #141, #143 open) | 70–110 |
| M16 | Built-in plugin depth, sidechain, ARA | ⬜ | 110–170 |
| M17 | Immersive delivery (ADM BWF, object renderer, binaural) and video depth | ⬜ | 80–130 |

## To beta (~350–570 h, about 60–70 % parallelizable)

Beta needs ~75 % ready for real work and no blocking gap in session interchange.

| Step | Gaps | Est. (h) |
|---|---|---:|
| Correctness backlog and a PR CI | G-04, G-14 | 40–65 |
| Audio interfaces, MIDI devices, sync | G-01, G-02, G-11 | 60–100 |
| AAF/OMF interchange, disk streaming, record to disk | G-03, G-06 | 65–115 |
| Plugin hosting completeness on every OS, sidechain | G-05, G-13 | 35–60 |
| Fidelity harness and the fixes it finds | G-07 | 40–60 |
| Built-in plugin essentials (EQ/dynamics/reverb/delay depth, sampler) | G-08 (half) | 40–60 |
| UI feel pass: shortcuts, plugin windows, meters | G-19 (half) | 30–50 |
| Localization of the 8 Pro Tools languages (catalog, CJK fonts) | G-09 (part) | 40–60 |

Needs a human: hardware for G-01/G-02/G-10/G-11, native-speaker review, and owner decisions on
AAX, `.ptx` and Avid cloud features.

## After beta (to full parity, a further ~600–900 h)

Control surfaces (G-10), immersive delivery (G-12), the rest of the built-in plugin catalogue
(G-08), Elastic Audio depth (G-15), session templates and browser (G-16), video depth (G-17),
speech-to-text (G-18), notation and instrument depth (G-20), compressed export (G-21), and the
remaining languages.

There is no `cargo xtask scorecard` in this repo; the measured number is `cargo xtask parity`
([`parity-checklist.md`](parity-checklist.md)).

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-15 | minor | Add target five-track vocal callback allocation/PCM coverage |
| 2026-10-15 | minor | Add measured callback allocation regressions and dynamic-delay capacity bounds |
| 2026-10-15 | minor | Connect mobile handoff with bounded ordered rings and overload diagnostics |
| 2026-10-15 | minor | Add prepared mixer handoff evidence and outstanding integration gates |
| 2026-10-15 | minor | Record downstream first-render preparation and remaining callback gates |
| 2026-10-10 | minor | Added the alpha gate (3 of 6 core workflows fail → pre-alpha) |
| 2026-10-10 | major | Created from ROADMAP.md's milestones and alpha list; added M12–M17 and the beta plan with estimates |

## Downstream stereo balance (2026-10-17)

Optional serialized strip balance attenuates stereo output columns after inserts
and panning on the shared realtime/offline path. Default zero preserves existing
sessions and dual-panner behavior; LiteDAW uses hard L/R panners plus balance.
No callback allocation or parity metric changes.

## Downstream waveform RMS ranges (2026-10-17)

Added a bounded, cached RMS range query for integer sample intervals. LiteDAW
uses exact rational raster boundaries to avoid float-step inclusion of a sample
from the preceding bin. Existing floating-span callers retain their semantics.
Parity metrics unchanged.

## Downstream 0.8.0 integration (2026-10-18)

Merged upstream v0.8.0 while preserving all mobile callback preparation, bounded
queues/capture reporting, DSP-state handoffs, monitoring/metering and feature gates.
The import-video merge retains the mobile video feature gate and initializes the
new offline-reason field. Stored-path/ProRes fixes are retained for enabled paths.
Workspace MSRV is 1.95 to match mandatory craft-docpath. No parity metrics or
product-readiness gates changed. Validation is recorded in LiteDAW plan.md.
