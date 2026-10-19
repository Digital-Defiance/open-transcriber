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

**Verdict:** adaptive BP + Copyist `clean_notes` mean note F1 ~0.87 vs open 0.83 (see [docs/t0-results.md](docs/t0-results.md)). Magenta MT3 skipped for singing. Omnizart chords wrapped but blocked on Apple Silicon Vamp. Copyist `--engine neural` is wired; default stays open until every row wins.

## Phases

1. **T0 — Baselines:** Basic Pitch + adaptive tidy / `clean_notes` (done). Omnizart chords when arm64 chroma works.
2. **T1 — Train:** only if the remaining −0.01/−0.02 note gaps won't close without it.
3. **T2 — Ship:** Copyist `--engine neural` as default when the full round-trip table wins.

## Layout

```
open_transcriber/   Python package (engines, pipeline, tidy, bridges)
scripts/            t0_baseline.py, setup_baseline.sh, setup_omnizart.sh
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
