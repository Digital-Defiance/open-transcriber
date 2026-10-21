# References for stage 0a

How the judge's reference notes are built (`refs.py`, `rwc_midi.py`, `ultrastar.py`). The rules were fixed in the stage 0a plan (D10) before any SheetSage2 output existed, and the reference files were built (2026-10-19 17:28 PDT) before SheetSage2's first output was written (17:30). SheetSage2 doesn't publish how it built its RWC-Pop references, so every choice is listed here.

The references themselves are derived from CC BY-NC material and stay in `.cache/judge/refs/`. The manifests record their SHA-256.

## Format

One TSV per item: `note_id onset offset midi midi_sung mask source`. Times in seconds; `midi` is the note as written; `midi_sung` is the octave-exact pitch; `mask` is empty or a reason (`freestyle`); `source` is `track:<n>`, `solo:<n>:<instrument>` or `chart:<kind>`. A provenance JSON per item (rows used, counts, solo choices) sits next to the references in `refs/provenance/`.

## RWC vocal melody, V1 (primary for P1 and P2)

- **Source:** RWC 2.0's aligned MIDI, `01_annotations_preprocessed/MIDI_aligned/RWC-<C>/RWC_<id>.mid` (rwc-annotations @ 0a1a6c3).
- **Tracks:** the piece's rows in `metadata_mapping_vocals.csv` (June 2026). `MIDITrackNo` is a 0-based index into the SMF track list; the track's name must equal `MIDITrackName` after stripping whitespace, or the build stops.
- **Rows left out:** a row whose comment matches `back ?vocals?|not vocal|not the main|nothing\?` (case-insensitive): 9 rows, in RWC-P031, P073, G006 (2), G012, G013 and G019 (3).
  - **Exception, found while building (before any SheetSage2 output):** the pattern also matches RWC-P026's only row, whose comment says the track follows the back vocals or an instrument in the chorus. Dropping it would leave P026 with no reference, which scores 0 for every system. The plan's list of affected pieces leaves P026 out, so its row is kept and the piece is flagged (`flagged: rwc-p026` in the build summary). Rule: rows are never all dropped; when the pattern matches every row of a piece, they're kept and the piece is flagged.
  - RWC-P066's two singers are both kept.
- **Notes:** every note-on with velocity above 0, paired with its note-off (or velocity-0 note-on), through the file's tempo map (every `set_tempo`; 120 BPM before the first). Pairing follows pretty_midi: a note-off closes every open note of that channel and pitch that started on an earlier tick, and a note starting on the same tick stays open (it's the next note). A note-off that finds only same-tick notes marks them zero-length: they're dropped and counted (0 in RWC 2.0). Exact duplicates (same track, onset tick and pitch) are collapsed: 768.
- **Octave-exact pitch:** `midi_sung = note + s·SemitoneOffset`. The sign is checked automatically: on all 100 RWC-P pieces, the median of (F0 annotation in MIDI − MIDI note) over voiced annotation frames inside V1's notes must be within 1 semitone of `s·offset` for every piece. **Result: s = −1 on all 100 pieces** (the aligned MIDI's melody tracks sit one or two octaves above the singing; the + sign fails on 85 pieces, the 85 with a nonzero offset). The same sign applies to G, R, C and J.
- RWC-P003 and RWC-P031 (comments report octave inconsistencies) are left out of octave-exact means only.
- No trimming, masking or time shift.

## RWC-Pop full melody, F1 (primary for the full-melody G0 check)

V1's notes plus the instrumental solos.

- **Solo segments:** each segment labelled `s:<instrument>` (or a bare `s`) in `AIST_RWC-MDB-P-2001_VOCA_INST` (rwc-annotations-archive @ 9944995). A line starts a segment that runs to the next line (the last one to the end of the audio). Labels are normalised: non-ASCII characters dropped, lower case, `s;` read as `s:`, and the typos `e;ectricguitar`, `chelo`, `malimba`, `okarina`, `synthesizers` read as `electricguitar`, `cello`, `marimba`, `ocarina`, `synthesizer`.
- **One track per segment,** from the first tier that has a track with an onset inside the segment. Vocal-mapping tracks (every row, kept or not) and drum tracks (channel 10, or DRUM or PERC in the name) are never chosen.
  1. tracks whose name contains `SOLO`;
  2. tracks whose first program is in the instrument's General MIDI family (table below);
  3. any other track.

  Within the tier: the track with the most onsets inside the segment; ties go to the higher median pitch, then the lower index.
- **Notes:** that track's notes with onsets inside the segment, as a skyline: of notes starting within 20 ms of a group's first onset, the highest is kept.
- **Counts:** 526 solo segments in 70 songs (49.7 min of solo); tier 1: 80, tier 2: 407, tier 3: 26; 13 segments had no playing track. 9,833 solo notes.

| VOCA_INST instrument | General MIDI programs (0-based) |
|---|---|
| electricguitar, acousticguitar, slideguitar | guitars 24–31 |
| guitarandbassriff | guitars 24–31, basses 32–39 |
| electricbass, bass, synthbass | basses 32–39 |
| synthesizer, synthlead | synth leads/pads/effects 80–103, pianos 0–7 |
| keyboard | pianos 0–7, synth 80–103, organs 16–23 |
| synthstrings | ensembles 48–55, synth 80–103 |
| strings, cello | strings 40–47 (strings also ensembles 48–55) |
| brass, trumpet | brass 56–63 (synthbrass also synth 80–103) |
| sax | saxes 64–67 (synthsax also synth 80–103) |
| oboe | reeds 64–71 |
| accordion, harmonica, organ | organs 16–23 |
| bell | chromatic percussion 8–15, tinkle bell 112 |
| marimba | chromatic percussion 8–15 |
| ocarina, pipe | pipes 72–79 |

## Diagnostic variants (scored only in G0, all reported)

- **V2:** every mapping row (no exclusions).
- **V3:** the 2006 AIST `MIDI_SYNC` files (`RM-P###.SMF_SYNC.MID`), V1's tracks found by name (V1's index when a name repeats).
- **V4:** V1 restricted to onsets inside VOCA_INST vocal segments (labels starting with m, f, g or r).
- **F2:** F1 without the skyline.
- **F3:** V4 plus F1's solos.
- **F4:** V1 plus every note of tracks named with `SOLO`, without VOCA_INST.

## UltraStar karaoke charts

- The chart is `song.txt` (every song has one). Header `#BPM` (a comma decimal is allowed) and `#GAP` in ms; relative-mode, `#VERSION` and duet charts are refused (none occur).
- `:` (normal) and `*` (golden) notes are pitched; `F` (freestyle) notes are masked; there are no rap notes.
- onset = GAP/1000 + start·15/BPM; offset = onset + max(length, 1)·15/BPM; MIDI = pitch + 60 (the octave isn't used: pitch class only).
- Chart QA (`QA.md`) masks further notes before any system is scored.

## G0: reproducing SheetSage2's RWC-Pop means

Targets (its `benchmark_results.json`): vocal 82.51, full 75.29, pitch class at 50 ms, each within 0.5 point. SheetSage2's official implementation (b83c3d7, MERT-v2 d8ba1c7), paper preset, RWC-Pop prompts, run once per song on the Mac's CPU in fp32 (its autocast is CUDA-only; the MPS probe died). Details and every number: `results/stage0a/g0.json`.

**Result: G0 fails.** Vocal **21.70** [20.14, 23.23] against 82.51; full **18.79** [17.53, 20.06] against 75.29. Every reference variant fails too (V2 21.70, V3 5.00, V4 21.37; F2 18.24, F3 18.51, F4 18.06).

**What the investigation found (plan section 6, in order):**

1. The scorer is right: SheetSage2's own environment's mir_eval gives identical per-song F1 on five songs, and the hand-made tests pass.
2. Parsing is right: `melody_vocal.lab` and `events.json` give identical F1 on all 100 songs.
3. SheetSage2 finds about as many notes as the references hold, and no run failed; 85 songs score below 0.30.
4. There is no common time offset (the median per-song best shift is −0.01 s). Instead the match rate is high for the first 20–30 s of each song and falls after it.
5. The references match the audio: the aligned MIDI agrees with RWC's F0 annotations steadily through each song, and RWC's beat annotations agree with librosa's beats on the same WAVs. SheetSage2's own beats agree with both for the first 30 s (98% within 50 ms on RWC-P014) and then drift (36%, 11%, then 0–30%).
6. It isn't numerics or the device: a rerun is identical, MPS (with the CPU fallback for unsupported ops) gives the identical output, and bf16 autocast emulated on the CPU doesn't help (RWC-P014 25.2 against 33.0).
7. It is position in the input window: excerpts restart the good stretch, and on consecutive 30-s excerpts (each its own `transcribe` call with the paper preset, labels stitched back) RWC-P014 scores 81.6 and RWC-P001 84.0, against 33.0 and 27.7 on the whole song.

**Diagnosis:** on this Mac the official implementation's decoder keeps accurate time for about the first 30 s of each 300-s window and drifts after it. The published CUDA runs can't have shown this; the cause in the code wasn't found. No comparison against SheetSage2 is reported until G0 is settled. A G0-investigation configuration, `sheetsage2-chunk30` (the official model and code on consecutive 30-s excerpts), is being scored on all 100 songs; it can stand in for SheetSage2 only if it passes G0 and the orchestrator agrees, and it would be labelled as such.

## Counts

| Set | Items | Notes | Notes |
|---|---|---|---|
| rwc-pop-vocal (V1) | 100 | 35,810 | no empty references |
| rwc-pop-full (F1) | 100 | 45,643 | 9,833 of them solo notes, in 70 songs |
| rwc-other-vocal (V1) | 75 | 20,028 | no empty references |
| usdx-32 | 32 | 11,072 | 153 freestyle (masked) |
| usdx-ht4 | 4 | 2,139 | 3 freestyle (masked) |
