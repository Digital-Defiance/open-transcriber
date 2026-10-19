# Open Transcriber

A commercially usable **lead-sheet** automatic music transcription stack: melody, chords, key, meter and tempo from a recording — Apache/MIT only, no CC BY-NC weights.

Used by [Copyist](https://github.com/Digital-Defiance/copyist) as an optional engine (`copyist transcribe --engine neural` once it wins the gate). Copyist’s open DSP engine stays the default until then.

**Licence:** MIT (this repo). Dependencies must stay MIT or Apache-2.0. Do not train on or redistribute SheetSage / MuScriptor weights.

## Relationship to Copyist

| Repo | Role |
|---|---|
| **open-transcriber** (this) | Models, training, inference CLI, baselines (Basic Pitch / later MT3 · Omnizart / trained heads) |
| **copyist** | Product: song files, tidy path (`clean_notes`, lyric targets), `roundtrip.py` gate, UI |

Copyist vendors this tree as a git submodule at `open-transcriber/`. Product integration and the round-trip yardstick live in Copyist (`docs/open-transcriber.md`, `scripts/transcribe/roundtrip.py`).

## Gate

A candidate is good enough for Copyist only when it does not lose to the open engine on:

```bash
# from a Copyist checkout with this submodule present
python scripts/transcribe/roundtrip.py
```

Public-domain YuE2 truth takes only. No owner Broadway masters in training data.

## T0 — Basic Pitch baseline (ready)

[Spotify Basic Pitch](https://github.com/spotify/basic-pitch) (Apache-2.0) on Copyist vocal stems → note F1 via Copyist's `evaluate.py`.

```bash
# from Copyist
git submodule update --init open-transcriber
bash open-transcriber/scripts/setup_baseline.sh
python scripts/transcribe/neural_baseline.py
python scripts/transcribe/neural_baseline.py --only silent-night --write out/t0-basic-pitch.md
```

Uses the bundled CoreML model (no separate weight download). No chords yet — melody only. Full table: [docs/t0-results.md](docs/t0-results.md).

**Verdict:** raw Basic Pitch ≪ Copyist open engine on lyrical takes (over-segmentation); ≈ open on Major-General patter. Next is tidy-through-Copyist (T0b) and/or training (T1).

## Phases

1. **T0 — Baselines:** Basic Pitch first; add Omnizart chords / MT3 when useful. Report next to the open engine in Copyist's `docs/transcribing.md`.
2. **T1 — Train:** supervised lead-sheet head on PD band renders + gated YuE2 takes; tidy through Copyist’s note path.
3. **T2 — Ship:** Copyist `--engine neural` + Install extras weights pack when the full table wins.

## Layout

```
open_transcriber/   Python package (engines, case list, note types)
scripts/            t0_baseline.py
tests/
```

## Develop

```bash
git submodule update --init open-transcriber
cd open-transcriber
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,baseline]"
python -m unittest discover -s tests
```

Weights and large checkpoints stay out of git (see `.gitignore`).
