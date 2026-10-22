# Pre-registration — the encoder ("ear") bake-off

Registered 2026-10-20, before any probe is trained or scored. **This file is immutable once tagged**
(`ear-prereg`). The decision below is applied by software (`decide.py`), not by any model's judgement.

Seed text for every stable selection: `copyist-ot2-20261020`.

## The question

Does a permissively licensed frozen audio encoder add enough real-music melody information over Basic Pitch
to justify building OT2 on it? A fixed GO/NO-GO rule decides.

## Encoders tested (free, licence-confirmed before download — see `licences.md`)

1. MusicFM (FMA checkpoint)
2. Dasheng **base** only
3. HeartCodec encoder
4. Whisper `base.en` encoder (vocal lyrics/onset auxiliary, tested as an add-on only)
5. Basic Pitch posteriors — the fixed melody side input for every probe **and the control** (encoder-free)

New probe/head weights are MIT; upstream encoders keep their own MIT/Apache terms.

## Identical probe for every encoder

Declared early/middle/final layers with a learned 3-way scalar mix; resample to 50 Hz; linear to 256;
2-layer BiGRU width 192, dropout 0.1; 12 s windows, fixed overlap. Heads: vocal pitch/silence, note onset,
note offset, beat, downbeat, chord root + maj/min/no-chord, 24-key. Basic Pitch contour/note/onset posteriors
are concatenated into every melody head's input. Same optimizer, schedule, early stopping.

Three seeds: **20261020, 20261021, 20261022**. Early-stop **only** on allowed synthetic / CC BY validation
(sampler val + VocalSet val singers). Score the frozen real-development sets once per trained probe.

## Metrics

- **Primary:** octave-exact vocal note F1 at 50 ms (`oct_on50`) and at 100 ms (`oct_on100`) on real dev.
- **Secondary:** onset+offset F1; repeated-pitch onset F1; beat/downbeat F1 at 70 ms; tempo-level error;
  duration-weighted major/minor chord score; key score.
- Synthetic and real are reported separately.

## GO / NO-GO rule (applied by `decide.py`)

**GO** if one music encoder, in **≥2 of the 3 seeds**:

1. real-dev octave-exact melody F1 ≥ the identical Basic-Pitch-only control **+ 3.0 absolute**;
2. synthetic/CC BY melody F1 ≥ control **+ 5.0**;
3. does **not** lower the mean of beat/downbeat/chord/key by more than **2.0** points versus the best
   candidate/control for those tasks;
4. fits the Mac shipping/throughput limit.

**Lexicographic tie-break among qualifying encoders:** highest real-dev melody F1; within 1.0 pt, highest
multi-task mean; within 0.5 pt, the smaller/faster model.

**Whisper add rule:** add Whisper to the winner only if it adds ≥ 2.0 pts on the repeated-pitch / lyric-onset
subset and loses ≤ 0.5 pt overall.

**NO-GO** if none qualifies: stop and report. Do **not** auto-escalate to a 30 h SoulX corpus or a larger
decoder. First inspect labels/probe capacity; then, under a new reviewed plan, test one larger Dasheng variant
or a different permissive ear.

## Frozen development splits (content hashes)

The bake-off scores probes only on these; the candidate-sealed TEST partitions (RWC-Pop, the RWC-other test
split, the USDX test items, COTB) are never touched. Each hash is the SHA-256 of the split's sorted member
ids (see `dev_splits.py`).

| Split | Items | Content hash (SHA-256) |
|---|---|---|
| rwc-other-dev (20 G / 6 R / 3 C / 1 J, real) | 30 | `05d375c53fef6092ef56f30a3b7ed52291257792ed194de936f3cf1db8d6bec1` |
| karaoke-dev (QA-surviving, diagnostic only) | 12 | `4f771646956c9a4e4c8c25b9754d0a62b9d751cd2eea54ca1f17c7de590d95aa` |
| vocalset-val (CC BY 4.0 validation singers) | 4 | `cdd21373883c41715f154f71117fd52114612c96e41ca79fd9f9998f676f5572` |
| sampler-dev (unseen sampler scores, by seed) | 80 | `f0ce96a6bdfd243882cfc34f98a0bec853f7cc55890df7cd7c33a734df30a2b7` |
| soulx-dev (unseen SoulX clips, by seed) | 40 | `33e260c13c6031363af3133e390491a4e546014ecac02cc6325e7b3a43867b7c` |
| yue2-truth (regression only, never a quality judge) | 5 | `98896f7d4e274862ad378477b0e3f4de9fa0df3362ccf17471772ba21a959e96` |

## Rights

$0 budget, free downloads only, stop below 150 GB free. RWC and karaoke are CC BY-NC: evaluation / probe-eval
only, **never** used to train a probe or model. VocalSet and the sampler/SoulX material are train/val. The
five YuE2 truth takes are regression only. SheetSage is never a label, teacher or tuning signal, and never
touches owner songs. Never train on owner songs. Never quote song lyrics.

---

## Amendment — 2026-10-22 (env provisioning + dropped-encoder plan)

Declared before the scored run, after the contest's first `--run` revealed two provisioning gaps that a
dry-run could not surface. This amendment records them for provenance. **It does not change the question, the
encoders tested, the identical probe, the metrics, the GO/NO-GO rule, its thresholds, the frozen splits/hashes,
or the three seeds (20261020/20261021/20261022).** `decide.py` remains the sole adjudicator, unchanged.

**1. Material/rendering libraries added to `.cache/ear/env`.** The pinned 2026-10-21 env was deliberately a
torch-only probe/extraction stack and lacked the libraries the real `--run` material path needs (the sampler
render pulls in `copyist`/`music21`; the SoulX slice and feature extraction use `librosa`/`nnAudio`). The first
`--run` therefore died at import with `ModuleNotFoundError: No module named 'music21'`, before any GPU work.
Fix: installed `music21>=9.1`, `verovio`, `PyYAML`, `pronouncing`, `pymupdf`, `mido`, `tinysoundfont`
(`--no-deps`), `librosa>=0.10` and `nnAudio` into `.cache/ear/env` under a constraints file pinning
`torch==2.5.1`, `torchaudio==2.5.1`, `numpy==2.1.3`, `transformers==4.46.3`, `scipy==1.18.1`,
`soundfile==0.12.1`, and `copyist` itself as an editable `--no-deps` install of this checkout. Verified after
install: all six protected pins unchanged, MPS available. The encoder/probe stack that produces the scored
features is untouched; only the (CPU) rendering/material layer changed. Basic Pitch still runs in Copyist's
`.venv` via subprocess (`bp_predict`), isolated from this env by design, so its TensorFlow/numpy are unaffected.
Re-frozen lock: `.agents/tasks/ear-bakeoff-2026-10-20/ear-env.lock.txt`.

**2. Dropped encoder excluded from the plan (not just at runtime).** HeartCodec was already dropped at runtime
by calibrate-and-drop (unsupported on MPS: its 24k→16k resample exceeds the 65536-channel conv1d limit), so it
was never trained and never a `decide()` candidate. The printed *plan*, however, still advertised
`extract-heartcodec` and three HeartCodec probes. To keep the plan and candidate set honest (no empty/zero row
from an encoder that never ran), `build_plan()` now excludes `DROPPED_ENCODERS = ("heartcodec",)` from the
extract and probe lists. HeartCodec remains one of the four declared encoders and is still recorded as a drop
in the calibrate ledger. No change to how any surviving encoder is extracted, trained, scored or adjudicated.
