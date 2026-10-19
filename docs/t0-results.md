# T0 results — Spotify Basic Pitch (Apache-2.0)

Measured 18 October 2026 on Copyist Demucs vocal stems. Melody note F1 only
(`evaluate.py` alignment, octave-free). No chords.

| Case | notes (vs true) | note F1 (P/R) | Open engine (ref) |
|---|---|---|---|
| saints | 107 vs 33 | 0.46 (0.30/0.97) | 0.85 |
| silent-night | 103 vs 46 | 0.58 (0.42/0.93) | 0.80 |
| amazing-grace | 137 vs 32 | 0.37 (0.23/0.97) | 0.88 |
| tyger | 62 vs 29 | 0.64 (0.47/1.00) | 0.78 |
| major-general | 128 vs 128 | 0.82 (0.82/0.82) | 0.84 |

**Verdict:** raw Basic Pitch is not a drop-in for Copyist. High recall, poor
precision on sustained singing (≈2–4× note count). Allegro patter
(Major-General) is already competitive. Next: feed BP (or MT3) notes through
Copyist's `clean_notes` / lyric target, then decide on T1 training.
