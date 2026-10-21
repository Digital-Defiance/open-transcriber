# Implementation Plan — the ear bake-off GPU "body"

Scope: write the real extractors, SoulX slice wiring, the `run_batch` execution body, real-dev
reference/batch assembly, and the worktree-safe licence-gate fix, so the next overnight window can run the
contest end-to-end and emit the pre-registered GO/NO-GO verdict. **Tests are the sole gate** (owner: no
multi-reviewer round, limited credits). One lean implementation pass. This task does **not** run the full
contest and does **not** grant a GPU window; GPU verification is short few-clip probes only.

All work is in the existing worktree `/Volumes/Code/copyist/.worktrees/ear-bakeoff` on branch `ear-bakeoff`,
submodule `/Volumes/Code/copyist/.worktrees/ear-bakeoff/open-transcriber` (detached at `52ae559`, parent
`ec19075`). Use `git -C /Volumes/Code/copyist/.worktrees/ear-bakeoff` for parent git and
`git -C /Volumes/Code/copyist/.worktrees/ear-bakeoff/open-transcriber` for submodule git. **Do not edit the
main checkout `/Volumes/Code/copyist` directly** — another session may commit to `main`.

Two Pythons:
- `/Volumes/Code/copyist/.venv/bin/python` — Copyist + Basic Pitch (TensorFlow). Runs the parent suite and the
  CPU-side bakeoff tests (`labels`, `make_soulx_slice`, `make_sampler_audio`, `dev_splits`, etc.).
- `/Volumes/Code/copyist/.cache/ear/env/bin/python` — the pinned torch env (torch 2.5.1). Runs the torch-side
  tests (`test_probe_*`, `test_decide`, `test_run_bakeoff`) and loads the real encoders. It has **no pytest**;
  use `python -m unittest`. Its pins must become torch 2.5.1 + torchaudio 2.5.1 + transformers 4.46.3.

Owner rules to honour throughout (encode in code + tests):
- $0; free/permissive downloads only (MIT/Apache/BSD/CC0/CC BY); model weights MIT. **RWC and karaoke are
  CC BY-NC: eval-only, never train a probe/model on them.** SheetSage is never a label/teacher/tuning signal
  and is never imported. Never train on owner songs. Never quote song lyrics. Never commit `voices/`,
  `.cache/`, `out/`, `lights-on*`.
- torch/torchaudio/transformers only in `.cache/ear/env`, never `.venv`. **TensorFlow/Basic Pitch stays in
  `.venv`**; the torch env calls it across a subprocess boundary (or reads cached posteriors). Never import
  TensorFlow into `.cache/ear/env`.
- Every long/GPU step goes through the existing `jobs.Job` runner (checkpoint `done.json`, output hashes,
  disk floor, `pmset -g batt` battery pause via `jobs.battery_watchdog`, `caffeinate -i`, the `DATA/gpu.lock`).
- Git: small commits saying what/why; stage by name; no force-push/`--no-verify`/`reset --hard`/`clean -fd`/
  `branch -D`; **submodule commit first, then the parent pointer**. Do not push.

## Verified baseline (confirmed during exploration)
- Torch-side tests green now: `test_probe_extractor`, `test_probe_model`, `test_probe_train`,
  `test_probe_evaluate`, `test_decide`, `test_run_bakeoff` = 43 tests pass on the ear env.
- CPU-side light tests green now: `test_bakeoff_make_soulx_slice`, `test_bakeoff_labels`,
  `test_bakeoff_dev_splits`, `test_bakeoff_vocalset_split`, `test_bakeoff_make_sampler_audio`,
  `test_prereg_hashes`, `test_import_boundaries` = 33 tests pass on `.venv`.
- `ear-prereg` tag exists at `2b606a4` and is an ancestor of HEAD (so `run_bakeoff.prereg_tag_ok()` passes).
- Confirmed real APIs: `copyist.sing.sung_notes(song, verse=1, bars=None, ...) -> list[SungNote]`;
  `copyist.sing.soulx_segments(notes, ...) -> list[dict]`; `copyist.sing.SoulX.find()` /
  `.sing(segments, voice_wav, voice_meta, out_dir, device="auto") -> Path`; `copyist.pitch.check(wav, notes,
  any_octave=True) -> list[NoteCheck]` (each has `.ok`). `scores.write_batch(out_dir, n, start_seed)` writes
  `song-<seed>.yaml`; `labels.labels_for_song(path, seed)` and `labels.write_labels(labels, path)` exist.
- The current `make_soulx_slice.make()` calls nonexistent `soulx.sing_phrase(...)` and `soulx.sung_notes(...)`
  (methods on SoulX). `run_bakeoff.run_batch()` only plans + gate-checks (the "wired there" stub). The licence
  gate reads `copyist_root()/.agents/.../licences.md`, which does not exist as a tracked file in a worktree.

## Design decisions made here (brief left these open)

- **Extractors live in a new `probe/extractors/` subpackage**, one module per encoder, each exposing a class
  that satisfies the existing `Extractor` Protocol and declares its `ExtractorDescriptor`. `extractor.py`
  keeps the Protocol/descriptor/`FakeExtractor` and gains a small registry (`build_extractor(name)` +
  `caching wrapper`). Rationale: keeps heavy torch/transformers imports lazy and per-encoder, leaves the
  CPU-testable Protocol module import-light, and matches the brief's "or a new `probe/extractors/` module".
- **Feature caching is a thin wrapper, not per-extractor code.** A `CachedExtractor` decorator computes a key
  from `(descriptor.name, model_hash, clip-file sha256)` and stores one `.npz` per clip under
  `.cache/ear/features/<encoder>/<clip-hash>.npz` (gitignored). Every extractor is wrapped once by the
  registry so caching/resume is uniform and testable with `FakeExtractor`. Rationale: single tested code path
  for "a window never re-extracts".
- **Basic Pitch runs as a subprocess from `.venv`, cached to disk.** A `BasicPitchSideInput` helper shells out
  to `/Volumes/Code/copyist/.venv/bin/python` running a tiny CLI that calls `basic_pitch.inference.predict`
  and writes the note/onset/contour posteriors to `.npz`; the torch env only reads those `.npz` files. It is
  **not** an `Extractor` (it is the fixed melody side input + the encoder-free control), so it is a separate
  helper, not in the `ENCODERS` extraction loop as a layer source. Rationale: the hard rule "never import
  TensorFlow into the ear env" and "Basic Pitch is the control + side input, not an encoder".
- **HeartCodec stays dropped.** Keep a descriptor-only stub recording "dropped: unsupported on MPS without a
  clear pilot lead (24k->16k resample exceeds the 65536-channel conv1d limit)"; it is listed in the drop
  ledger, never instantiated for extraction. The `run_batch` calibrate step records the drop reason.
- **Real-dev references + training batches live in a new tested module `probe/data.py`.** It turns cached
  features + by-construction `labels.py` JSON into the `{"layers","bp","targets"}` train dicts and the
  `(head_outputs, labels, item)` eval tuples that `train.py`/`evaluate.py` already consume. **Labels come from
  the score JSON, never audio.**
- **The licence gate finds `licences.md` inside the bakeoff package.** Add a committed
  `open_transcriber/bakeoff/licences.md` (the encoder licence table) and have `run_bakeoff.licences_confirmed`
  default to that package file, falling back to the `.agents` path only if present. Rationale: a worktree has
  no tracked `.agents/`, so the gate must not depend on it.
- **Dasheng ships final-layer-only if a time-aligned early/middle API is unavailable.** The smoke test showed
  naive `.blocks` hooks are not time-aligned. The extractor attempts Dasheng's intended hidden-states API; if
  it cannot return three time-aligned layers, it returns `final` three times (or a documented single-layer
  descriptor) and records "Dasheng: final-only (no time-aligned early/middle)" — never misaligned layers.
- **Whisper is CPU-only** (mps blocked by SparseMPS), extracted via encoder forward hooks on blocks 1/3/6.

---

## Phase 1 — Licence-gate path fix (worktree-safe) + licence table

- [ ] 1. Add the committed encoder licence table and make the gate find it in the package.
      Create `open_transcriber/bakeoff/licences.md` recording, for each tested encoder, the pinned
      source/revision, code licence, weight licence and a free/allowed verdict: MusicFM (FMA ckpt, MIT),
      Dasheng base (Apache-2.0), Whisper base.en (MIT), Basic Pitch (Apache-2.0), and **HeartCodec — DROPPED**
      (Apache-2.0 but unsupported on MPS). It must contain the substrings `run_bakeoff.licences_confirmed`
      already checks: `musicfm`, `dasheng`, `heartcodec`, `whisper`, `basic pitch`, and must **not** contain
      `nc/nd` or `excluded on licence`. Then edit `run_bakeoff.licences_confirmed(licences_md=None)` so the
      default path is `Path(__file__).with_name("licences.md")` (the package file), falling back to the old
      `.agents` path only if the package file is absent.
      Files: `open_transcriber/bakeoff/licences.md` (new), `open_transcriber/bakeoff/run_bakeoff.py`.
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_run_bakeoff` — existing
      gate tests still pass; `run_bakeoff.licences_confirmed()` (no arg) returns `(True, "ok")` from the
      worktree with no `.agents/licences.md`.

- [ ] 2. Add a worktree-safe licence-gate test.
      New test asserts `run_bakeoff.licences_confirmed()` with **no argument** returns ok in the worktree
      (proving it reads the package `licences.md`, not `.agents/`), and that a temp `licences.md` missing an
      encoder or containing `excluded on licence` still fails closed.
      Files: `tests/test_run_bakeoff.py` (add cases).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_run_bakeoff` passes.

## Phase 2 — Feature caching wrapper + extractor registry

- [ ] 3. Add the caching wrapper and registry in `extractor.py`.
      Add `CachedExtractor` (wraps any `Extractor`: key = sha256 of `(descriptor.name, descriptor model_hash,
      clip sha256)`, stores/loads one `.npz` per clip under `features_dir(encoder)` =
      `paths.copyist_root()/.cache/ear/features/<encoder>/`), and `build_extractor(name, device=None)` that
      lazily imports and constructs the right real extractor (Phase 3) or the HeartCodec drop-stub, wrapped in
      `CachedExtractor`. Keep `Extractor`, `ExtractorDescriptor`, `FakeExtractor`, `LAYER_NAMES` unchanged. Add
      `model_hash` to `ExtractorDescriptor` (default `""`) so the cache key is stable per weights.
      Files: `open_transcriber/bakeoff/probe/extractor.py`.
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_probe_extractor` — all
      existing tests still pass (adding a field with a default must not break them).

- [ ] 4. Test the caching wrapper with `FakeExtractor` (no model).
      New tests: wrapping `FakeExtractor`, the first `features()` call writes a `.npz`; the second call reads it
      without recomputing (patch the inner `.features` to count calls and assert it is called once across two
      outer calls); a different clip path produces a different cache file; changing `model_hash` invalidates
      the key. Use a temp features dir.
      Files: `tests/test_probe_extractor.py` (add a `CachedExtractorTests` class).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_probe_extractor` passes.

## Phase 3 — The four real extractors

- [ ] 5. MusicFM extractor (reference implementation).
      New `probe/extractors/musicfm.py`: load the FMA checkpoint + `fma_stats.json` (MIT, from
      `huggingface.co/minzwon/MusicFM`) into `.cache/ear/encoders/musicfm/` only, on `mps`; resample audio to
      24 kHz mono; `get_latent(wav, layer_ix)` for hidden layers 3/7/12 -> `early/middle/final`, each
      `(frames, 1024)` @ 25 Hz; needs `transformers` (non-flash Conformer). Declare the descriptor
      (`name="musicfm"`, code/weight licence MIT, `native_rate_hz=25.0`, `feature_dim=1024`,
      `model_hash`=checkpoint sha). Download via the existing `fetch`/`jobs` helpers, fail closed if licence
      not allowed.
      Files: `open_transcriber/bakeoff/probe/extractors/__init__.py` (new), `.../musicfm.py` (new),
      `open_transcriber/bakeoff/probe/extractor.py` (wire into `build_extractor`).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_extractors_real` — the
      MusicFM case asserts the 3 layers, `(frames,1024)` shape, 25 Hz, loads on mps, on 1–2 real sampler clips
      from `/Volumes/Code/copyist/.cache/ear/sampler/`. Mark the test `skipUnless` weights are downloadable /
      `COPYIST_EAR_SLOW=1` so it is the allowed short GPU probe but skippable in a weightless env.

- [ ] 6. Dasheng-base extractor (final; early/middle via the real hidden-states API or final-only).
      New `probe/extractors/dasheng.py`: `dasheng.dasheng_base()` pulls `dasheng_base.pt` (Apache, Zenodo) into
      `.cache/ear/encoders/dasheng/`, on `mps`; final `(frames, 768)` @ 25 Hz. Attempt Dasheng's intended
      intermediate-representation API for time-aligned early/middle; if three time-aligned layers are not
      available, return `final` for all declared layers (or a single-`final` descriptor) and set a descriptor
      flag/note `"final-only"`. **Never emit the misaligned `.blocks`-hook layers.** Declare the descriptor
      (`feature_dim=768`, Apache weight licence).
      Files: `open_transcriber/bakeoff/probe/extractors/dasheng.py` (new), `extractor.py` (wire in).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_extractors_real` — the
      Dasheng case asserts final `(frames,768)` @ 25 Hz on mps, and that any early/middle it returns has the
      **same frame count** as final (time-aligned) or is final-only; same `skipUnless` guard.

- [ ] 7. Whisper base.en extractor (CPU-only, auxiliary).
      New `probe/extractors/whisper_enc.py`: `whisper.load_model("base.en")` (MIT, cached under
      `.cache/whisper/`), **CPU only** (mps blocked by SparseMPS; do not force mps); forward hooks on encoder
      blocks 1/3/6 -> `early/middle/final`, each `(1500, 512)` @ 50 Hz (fixed 30 s window). Declare the
      descriptor (`native_rate_hz=50.0`, `feature_dim=512`, device `"cpu"`, MIT). Note in the descriptor that
      Whisper is an auxiliary ear per the prereg Whisper-add rule.
      Files: `open_transcriber/bakeoff/probe/extractors/whisper_enc.py` (new), `extractor.py` (wire in).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_extractors_real` — the
      Whisper case asserts 3 layers `(1500,512)` @ 50 Hz on cpu for a real clip; same `skipUnless` guard.

- [ ] 8. Basic Pitch side-input helper (subprocess to `.venv`, cached) + HeartCodec drop-stub.
      New `probe/extractors/basic_pitch_side.py`: `BasicPitchSideInput.posteriors(wav)` shells out to
      `/Volumes/Code/copyist/.venv/bin/python -m open_transcriber.bakeoff.bp_predict <wav> <out.npz>` (new tiny
      CLI module `bp_predict.py` that imports `basic_pitch` **only** when run under `.venv`), caches the
      note/onset/contour arrays to `.cache/ear/features/basic-pitch/<clip-hash>.npz`, and returns them. It is
      the fixed melody side input and the encoder-free control source — **not** registered as a layer
      `Extractor`. Also add the HeartCodec drop-stub: a descriptor-only entry
      (`name="heartcodec"`, note "dropped: unsupported on MPS ...") that `build_extractor("heartcodec")`
      returns and that `run_batch` records in the drop ledger; it raises if `features()` is called.
      Files: `open_transcriber/bakeoff/probe/extractors/basic_pitch_side.py` (new),
      `open_transcriber/bakeoff/bp_predict.py` (new), `extractor.py` (HeartCodec stub + wiring).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_extractors_real` — asserts
      `build_extractor("heartcodec").descriptor()` carries the drop note and `features()` raises; the Basic
      Pitch case (guarded by `COPYIST_EAR_SLOW`) runs the `.venv` subprocess on one real clip and asserts the
      note/onset/contour shapes from the smoke test (note/onset `(frames,88)`, contour `(frames,264)`), and
      asserts `bp_predict` does **not** import under the ear env (import-boundary check: run
      `/Volumes/Code/copyist/.cache/ear/env/bin/python -c "import open_transcriber.bakeoff.bp_predict"` must
      succeed only up to the lazy TF import — assert TF is not imported at module import time).

## Phase 4 — Real-dev reference + training-batch assembly (`probe/data.py`)

- [ ] 9. Create `probe/data.py` turning features + label JSON into the tensors train/eval consume.
      Add functions: `build_train_batches(clips, extractor, bp, config)` -> list of
      `{"layers": (B,n_layers,frames,dim), "bp": (B,frames,bp_dim), "targets": {head: tensor}}` dicts (resample
      each layer and the posteriors to the common 50 Hz grid via `model.resample_time`, stack via
      `model.stack_layers`, build per-frame targets from `labels.py` JSON: pitch class, silence, onset/offset
      pulses, beat/downbeat pulses, chord root/quality per frame, 24-key); and
      `build_eval_clips(clips, trained_model, extractor, bp)` -> list of `(head_outputs_numpy, label_dict,
      item_id)` tuples that `evaluate.evaluate_probe` consumes. **All labels come from the score JSON** emitted
      by `labels.labels_for_song` — nothing reads audio. Reuse `dev_splits` to resolve which clip ids belong to
      each real-dev split and refuse any sealed test id (call `dev_splits.test_partition_ids`).
      Files: `open_transcriber/bakeoff/probe/data.py` (new).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_probe_data` passes.

- [ ] 10. Test `probe/data.py` from a tiny synthetic stem + label to exact tensor shapes.
      New test builds a minimal `ScoreLabels`-shaped JSON (a couple of notes/beats/one chord/a key) + a
      `FakeExtractor` + a fake BP posterior array, runs `build_train_batches` and asserts the exact
      `{"layers","bp","targets"}` shapes and that every head key `train.LOSS_WEIGHTS` expects is present and
      frame-aligned at 50 Hz; runs `build_eval_clips` and asserts the `(out, labels, item)` tuple shape that
      `evaluate.score_one_clip` consumes; asserts a planted sealed test id is refused.
      Files: `tests/test_probe_data.py` (new).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_probe_data` passes.

## Phase 5 — SoulX slice wiring to the real API

- [ ] 11. Rewrite `make_soulx_slice.make()` against the real SoulX API.
      Replace the nonexistent `soulx.sing_phrase(...)` / `soulx.sung_notes(...)` calls with: load the sampled
      song via `copyist.song.load_song(song_path)`; build notes with `copyist.sing.sung_notes(song)`; build
      SoulX phrase segments with `copyist.sing.soulx_segments(notes)`; render with
      `soulx.sing(segments, voice_wav, voice_meta, out_dir=clip_dir, device="mps")` using the assigned CC BY
      VocalSet prompt's `voice_wav`/`voice_meta`; gate the returned `vocal.wav` note-by-note with
      `check_fn(wav, notes, any_octave=True)` (keep `gate_clip`); write `labels.labels_for_song(song_path,
      seed).write`... via existing `labels.write_labels`. **Batch many phrases per `sing()` call** (SoulX loads
      once per call) — restructure so one `sing()` renders all phrases for a clip/seed rather than per-attempt
      re-loads; re-roll a failed clip up to `MAX_REROLLS`. Add a `prompt_assets(prompt)` helper returning the
      `(voice_wav, voice_meta)` paths for a VocalSet prompt id (under `.cache/ear/vocalset/`), recording CC BY
      provenance; it must refuse/raise if the prompt assets are absent (tests inject a fake). Keep the
      "refuse to run without SoulX/GPU" behaviour and the `jobs.Job` runner wrapping.
      Files: `open_transcriber/bakeoff/make_soulx_slice.py`.
      Verify: `/Volumes/Code/copyist/.venv/bin/python -m unittest tests.test_bakeoff_make_soulx_slice` passes
      after the test is updated (next item).

- [ ] 12. Update the SoulX slice test to the real API with an injected fake SoulX + fake `pitch.check`.
      Change `FakeSoulX` to expose `sing(segments, voice_wav, voice_meta, out_dir, device)` (writing a
      placeholder `vocal.wav`, counting calls) instead of `sing_phrase`/`sung_notes`. Inject a fake
      `prompt_assets` and a real tiny sampler song (written via `scores.write_batch` into a temp dir, or a
      committed fixture song) so `sung_notes`/`soulx_segments` run on a real `Song`. Assert: segments are built
      from `sung_notes` (non-empty, correct SoulX `note_type`/`note_pitch` fields present), the pitch gate
      accepts/rejects per the fake `check`, re-roll then accept works, the run resumes (skips a finished clip),
      and it still refuses without SoulX. No model load.
      Files: `tests/test_bakeoff_make_soulx_slice.py`.
      Verify: `/Volumes/Code/copyist/.venv/bin/python -m unittest tests.test_bakeoff_make_soulx_slice` passes.

## Phase 6 — The `run_batch` execution body

- [ ] 13. Replace the `run_batch` stub with the real, resumable loop (behind the existing gates).
      Keep the gate checks (`gates_ok`: prereg tag + licences) and `build_plan`/`estimate`/`should_drop_extractor`
      unchanged. When `dry_run=False` and gates pass, execute in fixed order, each stage resumable via
      `jobs.Job` (checkpoints so a 10 h window that doesn't finish continues next window):
      (a) **calibrate-and-drop**: for each encoder in `ENCODERS`, calibrate on ~10 min of audio, call
      `should_drop_extractor(...)`, and skip + record dropped encoders (HeartCodec is always dropped via its
      stub); (b) **extract** features over dev+probe-train audio through `build_extractor(name)` (cached,
      resumable) and the Basic Pitch side input; (c) **assemble** train batches + real-dev eval clips via
      `probe.data` (labels from the score); (d) **train** the surviving `encoder×seed` probes + the 3 control
      seeds via `probe.train.train_probe` (per-epoch checkpoints, cap ~45 min each); (e) **evaluate** each
      trained probe once on the frozen real-dev splits via `probe.evaluate.evaluate_probe` (never touch sealed
      test partitions — `evaluate` already guards); (f) call `decide.decide(...)` and write the verdict JSON +
      metric table to `out_dir()/ear-bakeoff/`. Hold `caffeinate`/`gpu.lock` for the batch lifetime, check
      battery before each item (`jobs.battery_watchdog`), stop below the 150 GB floor, cap the whole batch at
      `BATCH_CAP_HOURS` and checkpoint cleanly at the cap/`--until`. Accept injected fakes
      (`extractor_factory`, `trainer`, `evaluator`, `decider`, audio lists) so the whole loop is CPU-testable
      without a model; `main()`'s `--dry-run` output stays unchanged.
      Files: `open_transcriber/bakeoff/run_bakeoff.py`.
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_run_bakeoff` — existing
      plan/drop/gate/dry-run tests still pass; `python -m open_transcriber.bakeoff.run_bakeoff --dry-run` prints
      the same ordered plan + gate PASS.

- [ ] 14. Add a fast end-to-end `run_batch` test with fakes exercising extract->assemble->train->eval->decide.
      New test drives `run_batch(dry_run=False, ...)` with injected fakes (a `FakeExtractor` factory, a tiny
      fake trainer returning a deterministic model, a fake evaluator returning a hand-built results table that
      makes `decide()` return both a clear GO and, in a second case, a NO-GO), a handful of fake clip ids, and
      patched `GpuLock`/`hold_caffeinate`/`disk_usage`. Assert: the fixed stage order runs; a dropped encoder
      (HeartCodec) is recorded and never trained; a re-run skips finished items (resumability); the verdict
      JSON + metric table are written; `decide()` is actually invoked and its verdict is persisted. No real
      model, no network, deterministic.
      Files: `tests/test_run_bakeoff.py` (add an `EndToEndTests` class).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_run_bakeoff` passes.

## Phase 7 — Full green + env pins + commits + report

- [ ] 15. Pin the ear env to torch 2.5.1 + torchaudio 2.5.1 + transformers 4.46.3 and record it.
      Ensure `.cache/ear/env` has `torchaudio==2.5.1` and `transformers==4.46.3` installed (the smoke test
      determined these; torchaudio was 2.11 — ABI mismatch; transformers 5.20 breaks the pinned torch). Write
      the exact `pip freeze` to `/Volumes/Code/copyist/.worktrees/ear-bakeoff/open-transcriber/.agents/tasks/ear-bakeoff-gpu-body-env.lock.txt`.
      Do not touch `.venv`.
      Files: `.agents/tasks/ear-bakeoff-gpu-body-env.lock.txt` (new, in the submodule's `.agents`).
      Verify: `/Volumes/Code/copyist/.cache/ear/env/bin/python -c "import torch,torchaudio,transformers;
      print(torch.__version__,torchaudio.__version__,transformers.__version__)"` prints `2.5.1 2.5.1 4.46.3`.

- [ ] 16. Run both test suites green.
      Torch-side: `/Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_probe_extractor
      tests.test_probe_model tests.test_probe_train tests.test_probe_evaluate tests.test_probe_data
      tests.test_decide tests.test_run_bakeoff tests.test_extractors_real` (real-extractor cases run as the
      allowed short GPU probes on this Mac, or skip without weights). CPU-side bakeoff:
      `/Volumes/Code/copyist/.venv/bin/python -m unittest tests.test_bakeoff_make_soulx_slice
      tests.test_bakeoff_labels tests.test_bakeoff_dev_splits tests.test_bakeoff_vocalset_split
      tests.test_bakeoff_make_sampler_audio tests.test_prereg_hashes tests.test_import_boundaries`. Parent
      suite: `cd /Volumes/Code/copyist/.worktrees/ear-bakeoff && /Volumes/Code/copyist/.venv/bin/python -m
      unittest discover -s tests` (must stay green).
      Files: none (verification only).
      Verify: all three commands report `OK`; the parent count is unchanged from the main baseline (the step
      prompt cites 311; confirm no parent test regressed).

- [ ] 17. Commit (submodule first, then the parent pointer) — do not push.
      In the submodule, stage by name the new/changed files: `open_transcriber/bakeoff/licences.md`,
      `open_transcriber/bakeoff/run_bakeoff.py`, `open_transcriber/bakeoff/make_soulx_slice.py`,
      `open_transcriber/bakeoff/bp_predict.py`, `open_transcriber/bakeoff/probe/extractor.py`,
      `open_transcriber/bakeoff/probe/data.py`, `open_transcriber/bakeoff/probe/extractors/**`,
      `tests/test_run_bakeoff.py`, `tests/test_bakeoff_make_soulx_slice.py`, `tests/test_probe_extractor.py`,
      `tests/test_probe_data.py`, `tests/test_extractors_real.py`, and the `.agents/.../ear-bakeoff-gpu-body-*`
      files. **Confirm nothing from `.cache/`, `out/`, `voices/`, `lights-on*` is staged.** Commit with a
      message saying what/why; then in the parent, stage the `open-transcriber` pointer and commit.
      Files: submodule commit + parent pointer commit.
      Verify: `git -C /Volumes/Code/copyist/.worktrees/ear-bakeoff status` and
      `git -C /Volumes/Code/copyist/.worktrees/ear-bakeoff/open-transcriber status` are clean with no ignored
      paths staged; `git -C .../open-transcriber log --oneline -1` shows the body commit.

- [ ] 18. Write the final report.
      Write `/Volumes/Code/copyist/.agents/tasks/ear-bakeoff-gpu-body-2026-10-21/report.md`: what was wired
      (extractors, SoulX slice, run_batch body, data.py, licence-gate fix), test results (both suites + the
      real-extractor probe results), the **surviving encoder list** (MusicFM, Dasheng-base, Whisper-CPU, Basic
      Pitch control/side-input) with **HeartCodec's recorded drop reason**, and the **exact command + estimated
      GPU hours** for the next overnight window (`/Volumes/Code/copyist/.cache/ear/env/bin/python -m
      open_transcriber.bakeoff.run_bakeoff` run on the Mac, ~10–20 GPU h over the window, capped ~10 h, 15–30 GB
      temp disk). Note the write report is under the **main checkout** `.agents` path per the brief (the report
      lives with the task folder, not committed into the worktree history).
      Files: `/Volumes/Code/copyist/.agents/tasks/ear-bakeoff-gpu-body-2026-10-21/report.md` (new).
      Verify: the report names every wired piece, the surviving encoders + HeartCodec drop, and the exact next
      window command with GPU-hour estimate.

## Final integration (merge)

- [ ] 19. Rebase, run both suites, fast-forward `main` only if clean.
      Rebase `ear-bakeoff` onto current `main` keeping both sides of any TASKS/DECISIONS/ROADMAP/AGENTS
      conflicts; re-run both suites in the worktree (Phase 7 item 16). Then **only if**
      `git -C /Volumes/Code/copyist status --porcelain` is empty with no git op in progress, fast-forward:
      `git -C /Volumes/Code/copyist merge --ff-only ear-bakeoff`. Otherwise stop and report. Do not push. Leave
      the worktree in place.
      Files: none (git + verification).
      Verify: both suites `OK` after the rebase; `main` fast-forwarded (or a clear report of why it was not).

## Notes / assumptions
- The real-extractor tests (`test_extractors_real.py`) are the brief's "allowed short GPU verification (a few
  clips)". They must pass on the owner's Mac but are guarded (`skipUnless COPYIST_EAR_SLOW=1` / weights
  present) so a weightless/cloud environment skips rather than fails. No long unattended GPU run happens in
  this task.
- VocalSet prompt assets (`voice_wav`/`voice_meta`) are assumed to live under `.cache/ear/vocalset/`
  (gitignored, CC BY 4.0); `prompt_assets` resolves them and tests inject fakes. If they are not yet on disk,
  the SoulX slice stays a GPU-window job that refuses to run without them — no behaviour change to the
  CPU-tested path.
- Keep `decide.py` the sole adjudicator; do not reinterpret the GO/NO-GO rule anywhere in `run_batch`.
