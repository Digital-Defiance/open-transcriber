# T0 results — Spotify Basic Pitch (Apache-2.0)

Measured 18–19 October 2026 on Copyist Demucs vocal stems. Melody note F1 only
(`evaluate.py` alignment, octave-free). No chords.

## Raw Basic Pitch (library defaults)

| Case | notes (vs true) | note F1 (P/R) | Open engine (ref) |
|---|---|---|---|
| saints | 107 vs 33 | 0.46 (0.30/0.97) | 0.85 |
| silent-night | 103 vs 46 | 0.58 (0.42/0.93) | 0.80 |
| amazing-grace | 137 vs 32 | 0.37 (0.23/0.97) | 0.88 |
| tyger | 62 vs 29 | 0.64 (0.47/1.00) | 0.78 |
| major-general | 128 vs 128 | 0.82 (0.82/0.82) | 0.84 |

## Adaptive tidy + Copyist clean_notes (default product path)

`melody_from_audio`: raised BP defaults → strict presets when flooded → loose when
starved → Copyist `clean_notes` (lyrical) / soft tidy (patter).

| Case | notes (vs true) | note F1 | vs open |
|---|---|---|---|
| saints | 34 vs 33 | 0.84 | −0.01 |
| silent-night | 46 vs 46 | **0.85** | **+0.05** |
| amazing-grace | 33 vs 32 | 0.86 | −0.02 |
| tyger | 29 vs 29 | **0.97** | **+0.19** |
| major-general | 121 vs 128 | 0.83 | −0.01 |

Mean note F1 **~0.87** vs open **0.83**. Beats open hard on Silent Night and Tyger;
within 0.02 on Saints / Amazing Grace / Major-General.

## Magenta MT3

Skipped for melody: official Colab notes MT3 is **not trained on singing**. YourMT3
has unclear licence. Adaptive BP + `clean_notes` is the commercial path.

## Omnizart chords

Wrapper in `open_transcriber/omnizart_chords.py` + `scripts/setup_omnizart.sh`.
On Apple Silicon, Omnizart's bundled `nnls-chroma` Vamp plugin is x86-only, so
chord inference fails until an arm64 chroma path exists. Open-engine chords stay.

## Copyist `--engine neural`

Wired: BP melody through open-transcriber, harmony from the open draft path.
Default remains `open` until every round-trip row is ≥ open (Saints/AG/MG still
a hair under on the note table; full draft path needs another pass on patter).

**Verdict:** T0b adaptive path is the contender. T1 training deferred until the
three −0.01/−0.02 gaps won't close without it.
