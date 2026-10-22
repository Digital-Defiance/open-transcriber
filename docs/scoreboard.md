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

## Calibration (`--calibrate`)

A constant onset shift per system, searched over ±100 ms in 10 ms steps, **fitted on one set and reported on
the other** (cross-fit), so a timing correction is never fitted and reported on the same set. The fit maximises
the set's lead F1; ties (a uniform offset smaller than the onset tolerance leaves a plateau) are broken by the
smaller mean absolute onset error of the matched notes, so a system offset by +40 ms recovers −40 ms. Each
calibrated row is named `<system> (xfit <shift> ms)` and uses the shift fitted on the *other* set.

## Output

- `out/judge/scoreboard.md` and `out/judge/scoreboard.json` (the current table), and
- `out/judge/scoreboard-history.jsonl` (one line appended per run: date, the Copyist and open-transcriber
  commits, and every row), all under the gitignored `out/judge`.

The test (`tests/test_board.py`) builds a tiny fake data folder where each system is its reference shifted
+40 ms with dense notes; the cross-fit recovers −40 ms and reports F1 1.0.
