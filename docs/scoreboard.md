# The transcriber scoreboard

One table of every transcriber on the frozen real dev sets, rebuilt every loop.

```
python -m open_transcriber.board            # score the systems, write the table
python -m open_transcriber.board --calibrate  # also add cross-fitted onset-shift variants
```

Evaluation only: RWC and the karaoke songs are CC BY-NC. Nothing here trains on them or leaves the Mac.

## Sets

Read from the frozen splits in `open_transcriber/bakeoff/splits`:

- **rwc-other-dev** (30 songs), scored on `pc_on50`, `oct_on50` (octave-exact against `midi_sung`, leaving out
  the score's `OCTAVE_EXCLUDED` songs) and `pc_on100`.
- **karaoke-dev** (12 songs), scored on `pc_on50`, `pc_on100` and `pc_onoff50`, with the QA masks applied
  (freestyle and QA-failed notes are don't-cares).

The first metric of each set (`pc_on50`) is the lead metric: the cross-fit maximises it and the markdown lead
column reports it.

## Systems

- `basic-pitch raw` and `basic-pitch one line` from `out/judge/basic-pitch-raw/<id>/notes.tsv`
  (one line is the prototype's amplitude-plus-leap-cost monophonic skyline).
- `open`/`neural` at `(grid)` and `(heard)` times from
  `out/judge/copyist-<engine>-audio/<id>/transcribe/events.json`.
- Every drop-in `out/judge/systems/<name>/<id>.tsv` (header `onset offset midi`), so a new engine plugs in
  without touching the board's code. GAME and later engines land here.

## Yardsticks (testing only)

Yardsticks are outside systems we score to see how far the field gets; they are never shipped and never run on
the owner's songs. Each has a runner under `scripts/` that lands its notes in `out/judge/systems/<name>/` and a
converter under `open_transcriber/board/`, so it reaches the board the same drop-in way as everything else.
Every runner has a `--smoke` CPU pre-run (resolve the inputs, round-trip the converter) that the GPU rules
require before a model run.

- **GAME** (`run_game.py`, `board/game.py`): six systems `game-<size>-<mix|stem>`. Code MIT; released weights
  state no licence, trained on private singing with non-commercial augmentation.
- **MuScriptor** (Kyutai and Mirelo; `run_muscriptor.py`, `board/muscriptor.py`): a general multi-instrument
  transcriber. We run the **non-gated GGUF mirror** `DamRsn/muscriptor-gguf` (CC BY-NC 4.0, the owner's choice,
  no Hugging Face account) through `muscriptor.cpp`'s benchmark. The bench carries a small `--notes-out <f.tsv>`
  flag (added locally) that writes each non-drum note as `onset<TAB>offset<TAB>pitch`; `board/muscriptor.py`
  reduces that polyphony to one melody line (a duration-weighted leap-cost skyline, like `basic-pitch one
  line`, since the bench has no velocity), giving six systems `muscriptor-<size>-<mix|stem>`. A standard-library
  SMF reader is also kept for the official `muscriptor transcribe --format midi` path. Code MIT; weights
  **CC BY-NC 4.0** — a yardstick only, never shipped, never on owner songs.

  Build `muscriptor.cpp` (in `.cache/muscriptor/muscriptor.cpp`, or set `$MUSCRIPTOR_CPP`):

  ```
  cmake -S cpp -B cpp/build -G Ninja -DMUSCRIPTOR_BUILD_TESTS=OFF -DMUSCRIPTOR_METAL_PRECOMPILED=OFF
  cmake --build cpp/build          # builds bench/muscriptor_bench with the --notes-out flag
  uv run msl-fetch --size small    # and medium, large -> testdata/weights/
  ```

  `--device gpu` runs on Metal; `--device cpu` for the pre-run. The runner resamples each song to 16 kHz mono
  with `ffmpeg -nostdin` (nohup ffmpeg otherwise blocks on stdin), transcribes, converts, and is resumable at
  the song level.
- **YourMT3+** (mimbres): code is **GPL-3.0** and the checkpoints' terms are unclear. Usable as a yardstick in
  a separate environment (testing only), but GPL-3.0 means it must never be linked into or derived from the
  shipped MIT product. Not wired up until it is actually run.

For the non-commercial test songs, only scores and credits are published — never their audio or our
transcriptions.

## Band parts on Copyist renders (`run_bandparts.py`, `board/bandparts.py`)

The band-parts baseline scores a transcriber against the parts Copyist's own band played, which are ground
truth by construction. `copyist build <song>` writes per-part audio (`out/<song>/stems/<part>.wav`) and the
exact MIDI (`out/<song>/stems/midi/<part>.mid`); the runner transcribes each part's stem and the full mix with
Basic Pitch (Apache-2.0, through the `.venv` boundary `bakeoff/bp_notes`) and scores it with the same strict
note scorer, writing `out/judge/bandparts.{md,json}`. A constant onset shift is cross-fitted (fitted on one
half of the song set, reported on the other) because the rendered performance sits a little off the quantised
grid — the same honesty rule as the vocal `--calibrate`.

This is clean-licence (Copyist's own renders), CPU-only, and the first brick of the data engine: render with
known labels, measure recovery. The pilot (bass on the 5 public-domain test songs; Silent Night is piano-vocal
so it has no bass part) baselines **bass from the stem at pc_on50 ≈ 31 (33 with the +20 ms cross-fit)** and
**bass from the mix ≈ 26**, stem above mix as everywhere. Run it:

```
python scripts/run_bandparts.py --smoke          # guard: .venv Basic Pitch + scorer round-trip
python scripts/run_bandparts.py                  # build, transcribe bass stem + mix, score
python scripts/run_bandparts.py --parts bass,piano,melody --songs when-the-saints,amazing-grace
```

## Data-engine pilot (`run_pilot.py`, `data/pilot.py`)

The pilot is the smallest honest test of "write music, then train a network to hear it": generate a few
validated sampler songs, render each with the band for an exact melody label and stem, run Basic Pitch on the
melody stem, build a per-candidate table (features + match labels), and train a tiny head
(`data.pilot.Head`, a small torch MLP, CPU) that decides which Basic Pitch notes to keep. **Success is judged
on real recordings, not our renders**: the head is applied to the RWC-other-dev and karaoke-dev Basic Pitch
outputs (already computed on the Demucs vocal stem) and scored with the strict scorer against the
`basic-pitch one line` baseline, into `out/judge/pilot.{md,json}`.

```
python scripts/run_pilot.py --smoke      # CPU plumbing only (no build/Basic Pitch)
python scripts/run_pilot.py --songs 6    # end to end on CPU; a full run is a separate GPU step
```

First pilot (seed 20261022, 6 render songs, 235 candidates): the head does **not** yet beat the baseline on
real dev — RWC `pc_on50` 18.5 vs baseline 22.3, karaoke 24.7 vs 25.4. That is the clean-render-vs-messy-real
gap the plan warns about: a head trained on a handful of clean sampled-instrument melodies underperforms the
plain skyline on sung vocal stems. The plumbing is proven; closing the gap needs more and more-realistic data
(SoulX-sung vocals, not sampled instruments) and is the GPU-scale step.

**yue2.cpp cannot write the vocal track alone** (checked 2026-10-22, `yue-synth --help`): it emits one mixed
audio (`--out`), with no stem/track flag. So the YuE2 singing-data path would need Demucs to isolate the vocal
after synthesis, which reintroduces separation — a reason the pilot uses the sfizz band render (each part is a
known stem by construction) for CPU labels, and keeps YuE2 singing as a later, melody-only, gated slice.

## Calibration (`--calibrate`)

A constant onset shift per system, searched over ±100 ms in 10 ms steps, **fitted on one set and reported on
the other** (cross-fit), so a timing correction is never fitted and reported on the same set. The fit maximises
the set's lead F1; ties (a uniform offset smaller than the onset tolerance leaves a plateau) are broken by the
smaller mean absolute onset error of the matched notes, so a system offset by +40 ms recovers −40 ms. Each
calibrated row is named `<system> (xfit <shift> ms)` and uses the shift fitted on the *other* set.

## The measured timing correction (`corrections.py`)

The cross-fit measures how far each system's onsets sit from the sung truth. For the heard-time outputs that
correction is wired in as a first-class system under a **stable name** — `open (heard, corrected)`,
`neural (heard, corrected)`, `basic-pitch one line (corrected)` — rather than only an `(xfit)` label, so the
engine's own heard-time output and the board agree on one measured shift. The shift applied to a set is still
the one fitted on the *other* dev set (the same cross-fit rule). `corrections.correct_heard(notes, shift)` is
the applier the engine uses; `corrections.fit_dev_shifts(board)` reads the per-set shift off a calibrated
board. Measured on the 2026-10-22 dev scoreboard: `open` heard about +20 ms, `neural` heard about +15 ms,
`basic-pitch one line` about −40 ms (so Basic Pitch one line on karaoke rises 25.4 → 36.7, `open` heard on
RWC 23.1 → 25.0). These are fitted on dev only, never on a test set.

## Output

- `out/judge/scoreboard.md` and `out/judge/scoreboard.json` (the current table), and
- `out/judge/scoreboard-history.jsonl` (one line appended per run: date, the Copyist and open-transcriber
  commits, and every row), all under the gitignored `out/judge`.

The test (`tests/test_board.py`) builds a tiny fake data folder where each system is its reference shifted
+40 ms with dense notes; the cross-fit recovers −40 ms and reports F1 1.0.
