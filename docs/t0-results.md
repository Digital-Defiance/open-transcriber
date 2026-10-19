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
| saints | 33 vs 33 | **0.85** | **0.00** (open 0.85) |
| silent-night | 46 vs 46 | **0.85** | **+0.05** (open 0.80) |
| amazing-grace | 33 vs 32 | **0.89** | **+0.01** (open 0.88) |
| tyger | 29 vs 29 | **0.97** | **+0.19** (open 0.78) |
| major-general | 121 vs 128 | 0.83 | −0.01 (open 0.84) |

Mean note F1 **~0.88** vs open **0.83**. Beats or ties open engine on 4/5 cases;
within 0.01 on Major-General.

## Magenta MT3

Skipped for melody: official Colab notes MT3 is **not trained on singing**. YourMT3
has unclear licence. Adaptive BP + `clean_notes` is the commercial path.

## Omnizart chords

Wrapper in `open_transcriber/omnizart_chords.py` + `scripts/setup_omnizart.sh`.
Native arm64 `nnls-chroma.dylib` built via `vamp-plugin-sdk` + Boost on Apple Silicon;
Omnizart runs end-to-end natively on macOS. Raw ungridded chord F1 is ~0.67; Copyist's
tonal priors + beat-chroma integration achieve 0.89 on Saints and 0.91 on Amazing Grace.

## Copyist `--engine neural`

Wired: BP melody through open-transcriber, harmony from the open draft path.
Tested on full lead-sheet round-trip (`scripts/transcribe/roundtrip.py --engine neural`):
- Saints: 0.91 vs open 0.85 (+0.06)
- Tyger: 0.93 vs open 0.78 (+0.15)
- Amazing Grace: 0.89 vs open 0.88 (+0.01)
- Major-General: 0.84 vs open 0.84 (tied)
- Silent Night: 0.78 vs open 0.80 (−0.02)
Mean round-trip note F1: **0.87** vs open **0.83**.

**Verdict:** T0b adaptive path beats open engine on almost every metric. T1 training
is the plan to push into the >0.93 SheetSage tier.
