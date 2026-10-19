# Open Transcriber

A commercially usable **lead-sheet** automatic music transcription stack: melody, chords, key, meter and tempo from a recording — Apache/MIT only, no CC BY-NC weights.

Used by [Copyist](https://github.com/Digital-Defiance/copyist) as an optional engine (`copyist transcribe --engine neural` once it wins the gate). Copyist’s open DSP engine stays the default until then.

**Licence:** MIT (this repo). Dependencies must stay MIT or Apache-2.0. Do not train on or redistribute SheetSage / MuScriptor weights.

## Relationship to Copyist

| Repo | Role |
|---|---|
| **open-transcriber** (this) | Models, training, inference CLI, baselines (MT3 / Omnizart / trained heads) |
| **copyist** | Product: song files, tidy path (`clean_notes`, lyric targets), `roundtrip.py` gate, UI |

Copyist vendors this tree as a git submodule at `open-transcriber/`. Product integration and the round-trip yardstick live in Copyist (`docs/open-transcriber.md`, `scripts/transcribe/roundtrip.py`).

## Gate

A candidate is good enough for Copyist only when it does not lose to the open engine on:

```bash
# from a Copyist checkout with this submodule present
python scripts/transcribe/roundtrip.py
```

Public-domain YuE2 truth takes only. No owner Broadway masters in training data.

## Phases

1. **T0 — Baselines:** run off-the-shelf Apache/MIT AMT (MT3, Omnizart) on Copyist’s truth takes; report note/chord F1 next to the open engine.
2. **T1 — Train:** supervised lead-sheet head on PD band renders + gated YuE2 takes; tidy through Copyist’s note path.
3. **T2 — Ship:** Copyist `--engine neural` + Install extras weights pack when the full table wins.

## Layout (growing)

```
open_transcriber/   Python package
scripts/            T0 baselines, train, export
tests/
docs/               design notes that belong here (not Copyist product docs)
```

## Develop

```bash
# from Copyist
git submodule update --init open-transcriber
cd open-transcriber
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m unittest discover -s tests
```

Weights and large checkpoints stay out of git (see `.gitignore`). Pin downloads with checksums when T0 lands.
