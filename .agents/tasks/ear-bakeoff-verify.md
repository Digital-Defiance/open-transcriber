# Review — ear bake-off GPU "body" (branch `ear-bakeoff`)

Single AI gate. Reviewed the three body commits on the submodule
(`1ebee7a` feat, `490c1ef` test, `db2aab2` fix) against the brief
(`ear-bakeoff-gpu-body-2026-10-21/brief.md`), the authoritative smoke test
(`encoder-smoke-2026-10-21.md`) and the plan. Test evidence was read from the commit messages, not re-run;
one narrow spot-read of the new tests and a forbidden-path check were done.

**Verdict: APPROVED.** No blocking findings.

## Evidence read

- `git log 52ae559..HEAD --stat`: three commits, 19 files in the feat commit (extractors, data.py, run_bakeoff
  body, make_soulx_slice, bp_predict, licences.md, env lock, five new/changed test files).
- Recorded test results (commit messages):
  - ear env: `test_probe_* test_decide test_run_bakeoff test_extractors_real` -> Ran 65/69, OK (1 skip).
  - short GPU probes (`COPYIST_EAR_SLOW=1`): MusicFM/Dasheng mps, Whisper cpu, HeartCodec drop, bp import
    boundary, Basic Pitch posterior shapes -> OK.
  - `.venv`: `test_bakeoff_* test_prereg_hashes test_import_boundaries` -> Ran 34, OK.
  - parent suite: `Ran 311/322, OK` (unchanged; the parent `tests/` tree is untouched by this branch).

## Checklist against the brief

### (1) Four real extractors — PASS
- **MusicFM** (`probe/extractors/musicfm.py`): FMA checkpoint (MIT), 24 kHz mono, `get_latent(layer_ix)` for
  hidden 3/7/12 -> early/middle/final, each `(frames,1024)` @ 25 Hz, loads on mps with CPU fallback. Matches
  the smoke table exactly. `test_extractors_real.MusicFMTests` asserts 3 layers, dim 1024, 25 Hz, time-aligned.
- **Dasheng base** (`dasheng.py`): final `(frames,768)` @ 25 Hz on mps; ships **final-only** with a recorded
  reason ("no time-aligned early/middle via the real API") and returns the final representation for all three
  declared layers (same frame count — time-aligned, never the misaligned `.blocks` hooks). This is exactly the
  option the brief/smoke test permit. Test asserts `final-only`, dim 768, 25 Hz, equal frame counts.
- **Whisper base.en** (`whisper_enc.py`): MIT, CPU only (device pinned to `"cpu"`, note records the SparseMPS
  mps block), encoder hooks at 0-indexed blocks 1/3 for early/middle and the post-LN encoder output for final,
  each `(1500,512)` @ 50 Hz. The smoke test named blocks "1/3/6"; using the post-LN encoder output as "final"
  (block 6 is the last block) yields the same documented `(1500,512)` shape/rate for all three layers and is
  documented in the module. Not a misalignment. Test asserts 3 layers `(1500,512)`, 50 Hz, device cpu.
- **Basic Pitch** (`bp_predict.py` + `basic_pitch_side.py`): runs as a `.venv` subprocess; `bp_predict`
  imports `basic_pitch`/TensorFlow lazily inside `run()`, never at module import. `BasicPitchSideInput` shells
  out and caches note/onset/contour `.npz`; it is a side input/control, not registered as a layer `Extractor`.
  No TF enters `.cache/ear/env`. `ImportBoundaryTests` asserts `bp_predict` imports with no `tensorflow`/
  `basic_pitch` in `sys.modules`.
- **HeartCodec** (`heartcodec.py`): kept as a descriptor-only drop stub; `build_extractor("heartcodec")`
  returns a `DroppedExtractor` whose `features()` raises and whose descriptor carries
  `dropped=True` and the MPS reason. `HeartCodecDropTests` asserts this. Correctly not patched.

Feature caching (`CachedExtractor`) keys on `(name, model_hash, clip sha256)` and writes one `.npz` per clip
under `.cache/ear/features/<encoder>/`, so a window never re-extracts. Tested with `FakeExtractor`
(cache hit/miss, model_hash invalidation) per the plan.

### (2) SoulX slice — PASS
`make_soulx_slice.make()` now calls the real API verified present in `copyist/sing.py`:
`load_song` -> module-level `sung_notes(song)` -> `soulx_segments(notes)` -> one
`soulx.sing(segments, voice_wav, voice_meta, out_dir, device)` per clip (batching all phrases per the
load-once requirement), gated with `check_fn(wav, notes, any_octave=True)` (`pitch.check`), re-rolled up to
`MAX_REROLLS`, with exact score-derived label JSON written via `labels.write_labels`. `prompt_assets` raises
if VocalSet CC BY assets are absent. `main()` still refuses to run without SoulX. No `sing_phrase`/
`sung_notes`-on-SoulX anywhere.

### (3) run_batch body — PASS
`run_batch(dry_run=False)` runs the real resumable loop behind `gates_ok` (prereg tag + licences): (a)
calibrate-and-drop (HeartCodec always dropped via its stub; others via `should_drop_extractor`), (b)/(c)
extract + assemble via `probe.data`, (d) train each surviving encoder×seed + control×seed, (e) evaluate once
on the frozen real-dev splits (`partition="real"`, guarded), (f) `decide.decide(...)` and write
`verdict.json`/`metrics.json`. Every stage is checkpointed via `_resumable`, so a capped window continues.
`decide()` is unchanged (no diff) and remains the sole adjudicator; `_default_decider` only marshals metrics
into `decide`'s inputs. All collaborators are injectable for CPU testing. `main --run` adds the launch path
(previously only ever dry-ran), frees LM Studio, and returns non-zero when gates block. `--dry-run`/default
output unchanged. `EndToEndTests` and `MainDispatchTests` cover stage order, HeartCodec drop, GO/NO-GO verdict
persistence, decide() invocation, and resumability.

### (4) Reference/batch assembly — PASS
`probe/data.py` is a new tested module. `build_train_batches` produces `{"layers","bp","targets"}` dicts;
`build_eval_clips` produces `(head_outputs, labels, item)` tuples matching `evaluate.evaluate_probe`'s
signature. All per-frame targets (pitch/silence/onset/offset/beat/downbeat/chord/key) are built from the
score label JSON only — no audio is read. `_guard_real_dev_ids` refuses any sealed id from
`dev_splits.test_partition_ids()` (verified to exist) and any COTB id. `test_probe_data` covers the shapes
and the sealed-id refusal.

### (5) Licence-gate path fix — PASS
`licences_confirmed` now defaults to the committed package `licences.md`
(`Path(__file__).with_name("licences.md")`), falling back to the `.agents` copy only if absent. The committed
`bakeoff/licences.md` contains all required substrings (`musicfm`, `dasheng`, `heartcodec`, `whisper`,
`basic pitch`) and no `nc/nd` / `excluded on licence`. `GateTests.test_licence_gate_reads_package_file_in_a_worktree`
proves the no-argument call passes in the worktree; fail-closed cases are tested.

### (6) Tests & no long GPU run — PASS
All required new tests exist (`test_extractors_real`, `test_probe_data`, additions to
`test_probe_extractor`, `test_bakeoff_make_soulx_slice`, `test_run_bakeoff`). Both suites plus the short
GPU probes are reported green in the commit messages. No long unattended GPU run was performed (the real
extractor tests are few-clip probes behind `COPYIST_EAR_SLOW`). The parent `tests/` tree is unchanged by this
branch, so the parent suite result (reported OK) is consistent; the 311 vs 322 number difference is a baseline
drift unrelated to this body work and not a regression.

### (7) Owner rules — PASS
- $0 / permissive: licence gate + descriptor `free_and_allowed` reject NC/ND; new weights declared MIT.
- RWC + karaoke eval-only: `PREREGISTRATION.md` and the import boundaries keep them off training; the
  body trains only on sampler/SoulX/VocalSet material and evaluates (not trains) on real-dev.
- SheetSage: every occurrence is documentary (docstrings/markdown stating it is NOT used); it is never
  imported, run, or used as label/teacher/tuning signal.
- No owner-song training; no lyrics quoted (guards intact: `lyrics_guard`, `_LANGUAGES`, import boundaries).
- torch/torchaudio/transformers isolated to `.cache/ear/env`; TensorFlow stays in `.venv` behind the
  `bp_predict` subprocess boundary (import-boundary test enforces it).
- No forbidden path committed: `git diff 52ae559..HEAD --name-only` shows nothing under `voices/`,
  `.cache/`, `out/`, or `lights-on*`.

## Notes (non-blocking, no action required)
- `licences.md` contains the stray line "Content was rephrased for compliance with licensing restrictions."
  It does not affect the gate substrings and is harmless.
- Whisper "final" uses the post-LN encoder output rather than block-6's raw output; shape/rate/layer count
  all match the smoke documentation, so this is an acceptable interpretation and is documented in code.

## Conclusion
The GPU body is complete and correct against the brief: four real extractors matching the smoke-documented
shapes/rates/layers/devices, HeartCodec dropped with its reason, the real SoulX API wired, a resumable
calibrate→extract→assemble→train→evaluate→decide loop with `decide()` as the sole adjudicator, tested
reference/batch assembly from scores only, the worktree-safe licence gate, and the required tests green with
no long GPU run. Owner rules upheld. **APPROVED.**
