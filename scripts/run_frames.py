#!/usr/bin/env python3
"""Sung-data frame-level transcriber: generate sung training data, train the frame model, score on real dev.

The step up from ``run_pilot.py`` (the per-note keep/drop head that did NOT beat Basic Pitch on real sung
vocals). The model is now a small recurrent frame model over FROZEN Basic Pitch posteriors and a log-mel of a
Demucs vocal (:mod:`open_transcriber.data.frames`), trained on REALISTIC SUNG data: SoulX sings a Copyist
sampler score (exact labels by construction), mixed over the band, then Demucs-separated.

Stages (real pipeline, mostly GPU):
1. Generate ``--songs`` validated sampler songs (seeded; :mod:`open_transcriber.data.scores`).
2. ``copyist build`` each → its exact melody-label MIDI (``out/<slug>/stems/midi/melody.mid``) and the band mix.
3. SoulX sings the melody; mix the vocal over the band (realistic bleed); Demucs-separate the vocal.
4. Basic Pitch POSTERIORS on the Demucs vocal (``bp_predict`` → .npz note/onset/contour) + log-mel of the
   vocal → per-frame features; rasterise the exact MIDI to per-frame labels.
5. Train the frame model (one GPU window), save it.
6. Eval on REAL dev (RWC-other-dev, karaoke-dev) with the strict scorer and a fair cross-fit onset shift,
   against ``basic-pitch one line``. Judge NEVER on our own renders.

``--smoke`` proves the whole frame pipeline on CPU in seconds with NO GPU, NO SoulX, NO Basic Pitch: synthetic
posteriors/labels through the model, then DECODE the model's output and SCORE it with the real strict scorer
against the REAL frozen dev references, so the eval path is known to produce sane, non-empty metrics before a
GPU window is ever requested.

Everything is clean-licence and never touches owner songs. A fixed seed is recorded.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import notes_io, paths, score  # noqa: E402
from open_transcriber.board import board  # noqa: E402
from open_transcriber.data import frames  # noqa: E402


# --- environment boundary (TensorFlow Basic Pitch lives only in .venv) ----------------------------------

def venv_python(root: Path) -> Path:
    env = os.environ.get("COPYIST_VENV")
    if env:
        return Path(env)
    here = root / ".venv" / "bin" / "python"
    if here.exists():
        return here
    parts = root.parts
    if ".worktrees" in parts:
        main = Path(*parts[: parts.index(".worktrees")])
        shared = main / ".venv" / "bin" / "python"
        if shared.exists():
            return shared
    return here


def bp_posteriors(root: Path, wav: Path, out_npz: Path) -> Path:
    """Basic Pitch note/onset/contour posteriors for ``wav`` (cached). Shelled out to .venv (TensorFlow)."""
    if not out_npz.exists():
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(venv_python(root)), "-m", "open_transcriber.bakeoff.bp_predict",
                        str(wav), str(out_npz)],
                       cwd=str(root / "open-transcriber"), check=True, stdin=subprocess.DEVNULL)
    return out_npz


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a wav as mono float32 with its sample rate, with no heavy dependency (stdlib ``wave``)."""
    import wave

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    data = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    peak = float(np.iinfo(dtype).max)
    return data / peak, sr


def features_for_wav(root: Path, wav: Path, npz_dir: Path) -> np.ndarray:
    """Frozen features for one wav: BP note+onset posteriors stacked with a frame-aligned log-mel."""
    npz = bp_posteriors(root, wav, npz_dir / f"{wav.stem}.npz")
    blob = np.load(npz)
    note, onset = blob["note"], blob["onset"]
    wave_data, sr = load_wav(wav)
    mel = frames.log_mel(wave_data, sr, n_frames=min(len(note), len(onset)))
    return frames.frames_from_posteriors(note, onset, mel)


# --- real-dev eval (the honest success test) ------------------------------------------------------------

SHIFT_RANGE = np.round(np.arange(-0.06, 0.06 + 1e-9, 0.01), 3).tolist()


def dev_vocal_wav(item: str, out: Path, data: Path) -> Path | None:
    """The Demucs vocal stem of a real-dev item, if cached (preferred), else its source recording.

    The realistic case separates the vocal with Demucs; a cached stem is used when present. The eval only
    needs the wav to exist — producing the stem is a GPU step done before scoring.
    """
    stem = out / "dev-vocal" / f"{item}.wav"
    if stem.exists():
        return stem
    src = out / "copyist-open-audio" / item / "transcribe" / "source.wav"
    return src if src.exists() else None


def model_rows_at_shift(model: frames.FrameModel, item_feats: dict[str, np.ndarray], shift: float,
                        split: str, data: Path, dec: dict) -> list[dict]:
    kind = board.SET_METRICS[split][0]
    rows = []
    for item in board.set_members(split):
        feats = item_feats.get(item)
        if feats is None or len(feats) == 0:
            continue
        pn, po = model.predict(feats)
        est = frames.decode_notes(pn, po, onset_shift=shift, **dec)
        rows.append(board._score_item(split, kind, item, est, data))
    return rows


def fit_shift(model: frames.FrameModel, item_feats: dict[str, np.ndarray], split: str,
              data: Path, dec: dict) -> float:
    best_shift, best = 0.0, -1.0
    for shift in SHIFT_RANGE:
        m = score.mean_f1(model_rows_at_shift(model, item_feats, shift, split, data, dec), "pc_on50")
        if m > best + 1e-12:
            best, best_shift = m, shift
    return best_shift


def base_rows_at_shift(split: str, shift: float, data: Path, out: Path) -> list[dict]:
    kind = board.SET_METRICS[split][0]
    rows = []
    for item in board.set_members(split):
        bp_path = out / "basic-pitch-raw" / item / "notes.tsv"
        if not bp_path.exists():
            continue
        est = board.shift_notes(board.mono_bp(bp_path), shift)
        rows.append(board._score_item(split, kind, item, est, data))
    return rows


def fit_base_shift(split: str, data: Path, out: Path) -> float:
    best_shift, best = 0.0, -1.0
    for shift in SHIFT_RANGE:
        m = score.mean_f1(base_rows_at_shift(split, shift, data, out), "pc_on50")
        if m > best + 1e-12:
            best, best_shift = m, shift
    return best_shift


def eval_real_dev(model: frames.FrameModel, feats_by_split: dict[str, dict[str, np.ndarray]],
                  data: Path, out: Path, dec: dict, calibrate: bool = True) -> dict:
    """Score the frame model's decoded notes and the ``basic-pitch one line`` baseline on both dev sets.

    Both sides cross-fit the onset shift (fit on the OTHER set, report on this one), so the shift is never a
    free win for the model — the same honesty rule as ``run_pilot.py`` and the scoreboard.
    """
    splits = ("rwc-other-dev", "karaoke-dev")
    model_fit = {s: fit_shift(model, feats_by_split.get(s, {}), s, data, dec) for s in splits} if calibrate else {}
    base_fit = {s: fit_base_shift(s, data, out) for s in splits} if calibrate else {}
    result = {}
    for split in splits:
        other = next(s for s in splits if s != split)
        m_shift = model_fit.get(other, 0.0)
        b_shift = base_fit.get(other, 0.0)
        m_rows = model_rows_at_shift(model, feats_by_split.get(split, {}), m_shift, split, data, dec)
        b_rows = base_rows_at_shift(split, b_shift, data, out)
        result[split] = {
            "model_pc_on50": round(100 * score.mean_f1(m_rows, "pc_on50"), 3),
            "baseline_pc_on50": round(100 * score.mean_f1(b_rows, "pc_on50"), 3),
            "model_shift": m_shift,
            "baseline_shift": b_shift,
            "n": len(m_rows),
        }
    return result


def prep_dev_stems(root: Path, out: Path, splits=("rwc-other-dev", "karaoke-dev")) -> dict:
    """Pre-compute, once, the realistic separated vocal + BP posteriors for every real-dev item.

    For each dev item: Demucs two-stem separation of its source recording into ``out/dev-vocal/<item>.wav``
    (the realistic case the eval scores on), then cache that vocal's Basic Pitch posteriors. Both heavy
    stages run in the main ``.venv``; Demucs uses the GPU. Idempotent — an existing stem/posterior is reused,
    so this resumes cleanly inside a GPU window. Returns a small summary (counts) for the run log.
    """
    dev_dir = out / "dev-vocal"
    dev_dir.mkdir(parents=True, exist_ok=True)
    npz_dir = out / "frames-run" / "dev-bp"
    done = {"stems": 0, "posteriors": 0, "missing_source": []}
    for split in splits:
        for item in board.set_members(split):
            src = out / "copyist-open-audio" / item / "transcribe" / "source.wav"
            if not src.exists():
                done["missing_source"].append(item)
                continue
            stem = dev_dir / f"{item}.wav"
            if not stem.exists():
                vocals, _rest = _split_vocals(root, src, dev_dir / "_sep" / item)
                if vocals.exists():
                    import shutil

                    shutil.copyfile(vocals, stem)
                    done["stems"] += 1
            if stem.exists():
                npz = npz_dir / f"{stem.stem}.npz"
                if not npz.exists():
                    bp_posteriors(root, stem, npz)
                    done["posteriors"] += 1
    return done


def dev_features(root: Path, out: Path, data: Path, splits=("rwc-other-dev", "karaoke-dev")
                 ) -> dict[str, dict[str, np.ndarray]]:
    """Frozen features for every dev item that has a (vocal/source) wav. Cached posteriors are reused."""
    npz_dir = out / "frames-run" / "dev-bp"
    by_split: dict[str, dict[str, np.ndarray]] = {}
    for split in splits:
        feats: dict[str, np.ndarray] = {}
        for item in board.set_members(split):
            wav = dev_vocal_wav(item, out, data)
            if wav is None:
                continue
            feats[item] = features_for_wav(root, wav, npz_dir)
        by_split[split] = feats
    return by_split


# --- training-data generation (GPU: SoulX + Demucs) -----------------------------------------------------

def build_sung_dataset(root: Path, out: Path, n_songs: int, seed: int) -> frames.FrameDataset:
    """Generate sung clips and their frame features/labels. Needs SoulX + Demucs (a GPU step).

    This is the real data stage; the CPU smoke never calls it. Each song: generate → ``copyist build`` →
    SoulX sings → mix over the band → Demucs vocal → BP posteriors + log-mel → frame features; rasterise the
    exact melody MIDI to per-frame labels.

    Resumable: each song's (features, note-label, onset-label) is cached to ``frames-run/clips/<slug>.npz``
    the moment it is produced, and an existing clip is reused. A window cap or crash therefore never discards
    the singing/separation already done — a re-launch picks up where it stopped.
    """
    from open_transcriber.data import scores as score_gen

    run_dir = out / "frames-run"
    song_dir = run_dir / "scores"
    clip_dir = run_dir / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    songs = score_gen.write_batch(out_dir=song_dir, n=n_songs, start_seed=seed)
    seqs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    used: list[str] = []
    for yaml in songs:
        slug = yaml.stem
        clip = clip_dir / f"{slug}.npz"
        if clip.exists():  # resume: this song's features/labels are already cached
            blob = np.load(clip)
            seqs.append((blob["feats"], blob["yn"], blob["yo"]))
            used.append(slug)
            print(f"[data] {slug}: cached ({blob['feats'].shape[0]} frames)")
            continue
        subprocess.run([str(venv_python(root)), "-m", "copyist.cli", "build", str(yaml)],
                       cwd=str(root), check=True)
        truth_mid = root / "out" / slug / "stems" / "midi" / "melody.mid"
        vocal = _sing_and_separate(root, out, slug)
        if vocal is None or not truth_mid.exists():
            print(f"[skip] {slug}: no sung vocal or label")
            continue
        feats = features_for_wav(root, vocal, run_dir / "train-bp")
        truth = _midi_truth(truth_mid)
        yn, yo = frames.rasterise_labels(truth, n_frames=len(feats))
        np.savez_compressed(clip, feats=feats, yn=yn, yo=yo)
        seqs.append((feats, yn, yo))
        used.append(slug)
        print(f"[data] {slug}: {len(feats)} frames, {int(yo.sum())} onset labels")
    return frames.FrameDataset(seqs=seqs, songs=used, seed=seed)


def _sing_and_separate(root: Path, out: Path, slug: str) -> Path | None:
    """SoulX sings the song's melody, mix over the band, Demucs-separate the vocal. GPU. Returns the stem.

    Reuses Copyist's own tools so the vocal is the realistic case the field faces (a separated stem, bleed
    and all): ``copyist sing`` renders the band, sings the melody with SoulX and lays the vocal over the band
    (``sung/verse1/mix.wav``); then Demucs two-stem separation pulls the vocal back out
    (``copyist.stems.split_vocals``). Both heavy stages run in the main ``.venv`` (TensorFlow/torch there) and
    on the GPU; this is why the whole generation stage runs only in a granted GPU window.

    ``COPYIST_FRAMES_FAKE_SING=1`` short-circuits the GPU stages for the CPU plumbing proof: it fabricates a
    short silent-plus-tone "vocal" wav so the feature/label/dataset path downstream can be exercised end to
    end without SoulX, a band render or Demucs. It is NEVER used for a real result.
    """
    run_dir = out / "frames-run"
    sep_dir = run_dir / "sep" / slug
    sep_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("COPYIST_FRAMES_FAKE_SING") == "1":
        return _fake_vocal(root, slug, sep_dir)

    yaml = run_dir / "scores" / f"{slug}.yaml"
    # 1. Band render + SoulX sing + lay the vocal over the band → a realistic full mix.
    subprocess.run([str(venv_python(root)), "-m", "copyist.cli", "sing", str(yaml),
                    "--verse", "1", "--out", str(out), "--device", os.environ.get("COPYIST_SOULX_DEVICE", "auto")],
                   cwd=str(root), check=True)
    mix = out / slug / "sung" / "verse1" / "mix.wav"
    if not mix.exists():
        print(f"[skip] {slug}: no mix.wav from copyist sing")
        return None
    # 2. Demucs pulls the vocal back out of the mix (the realistic separated-vocal case).
    try:
        vocals, _rest = _split_vocals(root, mix, sep_dir)
    except Exception as e:  # pragma: no cover - exercised only in a GPU window
        print(f"[skip] {slug}: Demucs separation failed: {e}")
        return None
    return vocals if vocals.exists() else None


def _split_vocals(root: Path, mix: Path, out_dir: Path) -> tuple[Path, Path]:
    """Call ``copyist.stems.split_vocals`` in the main ``.venv`` (Demucs/torch live there), returning the two
    stem paths. Shelled out so this script's own (torch) env is never Demucs's."""
    helper = (
        "import json,sys;from pathlib import Path;from copyist.stems import split_vocals;"
        "v,r=split_vocals(Path(sys.argv[1]),Path(sys.argv[2]));"
        "print(json.dumps([str(v),str(r)]))"
    )
    res = subprocess.run([str(venv_python(root)), "-c", helper, str(mix), str(out_dir)],
                         cwd=str(root), check=True, capture_output=True, text=True)
    vocals, rest = json.loads(res.stdout.strip().splitlines()[-1])
    return Path(vocals), Path(rest)


def _fake_vocal(root: Path, slug: str, sep_dir: Path) -> Path:
    """A short deterministic tone-over-silence wav standing in for a separated vocal (CPU plumbing only)."""
    import wave

    truth_mid = root / "out" / slug / "stems" / "midi" / "melody.mid"
    sr = 22050
    seconds = 5.0
    t = np.arange(int(sr * seconds)) / sr
    # A few steady tones so Basic Pitch (when it runs) and the log-mel both see non-trivial signal.
    sig = np.zeros_like(t)
    for k, f in enumerate((220.0, 262.0, 330.0)):
        lo, hi = k / 3 * seconds, (k + 1) / 3 * seconds
        m = (t >= lo) & (t < hi)
        sig[m] = 0.3 * np.sin(2 * np.pi * f * t[m])
    out = sep_dir / "vocal.wav"
    pcm = (np.clip(sig, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    if not truth_mid.exists():
        print(f"[fake] {slug}: no melody.mid label (build may have been skipped)")
    return out


def _midi_truth(path: Path) -> notes_io.Notes:
    from open_transcriber.board.muscriptor import read_midi_notes

    ns = read_midi_notes(Path(path))
    ns.sort(key=lambda n: (n.onset, n.offset))
    return notes_io._validate(
        notes_io.Notes(np.array([n.onset for n in ns]), np.array([n.offset for n in ns]),
                       np.array([float(n.midi) for n in ns])), False, Path(path).name)


# --- smoke (CPU, no GPU/SoulX/Basic Pitch) --------------------------------------------------------------

def _synthetic_clip(rng, seconds: float = 6.0) -> tuple[np.ndarray, notes_io.Notes]:
    """A fake melody and matching noisy BP-style posteriors, so training has a learnable signal on CPU."""
    n_frames = int(seconds * frames.FPS)
    onsets, offsets, pitches = [], [], []
    t = 0.0
    while t < seconds - 0.4:
        dur = float(rng.uniform(0.25, 0.6))
        midi = int(rng.integers(55, 72))
        onsets.append(t)
        offsets.append(min(t + dur, seconds))
        pitches.append(float(midi))
        t += dur
    truth = notes_io._validate(notes_io.Notes(np.array(onsets), np.array(offsets), np.array(pitches)),
                               False, "synthetic")
    yn, yo = frames.rasterise_labels(truth, n_frames)
    # Posteriors = labels blurred and noised; the model should recover the clean line.
    note = np.clip(0.75 * yn + 0.08 * rng.standard_normal(yn.shape), 0, 1).astype(np.float32)
    onset = np.clip(0.8 * yo + 0.05 * rng.standard_normal(yo.shape), 0, 1).astype(np.float32)
    mel = (0.3 * rng.standard_normal((n_frames, frames.MEL_BANDS))).astype(np.float32)
    feats = frames.frames_from_posteriors(note, onset, mel)
    return feats, truth


def smoke(out: Path, data: Path) -> int:
    rng = np.random.default_rng(0)
    # 1. Train on synthetic clips with a learnable blur-and-denoise signal.
    seqs = []
    for _ in range(12):
        feats, truth = _synthetic_clip(rng)
        yn, yo = frames.rasterise_labels(truth, len(feats))
        seqs.append((feats, yn, yo))
    model = frames.FrameModel(hidden=32, seed=0)
    info = model.fit(seqs, epochs=40, window=256, batch=4)
    print(f"[smoke] trained frame model on CPU: {info}")
    assert info["loss"] is not None and info["loss"] < 1.5, f"loss did not fall: {info}"

    # 2. The model recovers notes from a held-out synthetic clip.
    feats, truth = _synthetic_clip(np.random.default_rng(99))
    pn, po = model.predict(feats)
    est = frames.decode_notes(pn, po)
    assert len(est) >= max(1, len(truth) // 2), f"decode kept too few notes: {len(est)} of {len(truth)}"
    print(f"[smoke] decode kept {len(est)} notes from a {len(truth)}-note clip")

    # 3. Prove the REAL eval path: decode a (fake) model output and score it with the strict scorer against
    #    the REAL frozen dev references, so the metric path is known to produce sane, non-empty numbers.
    checked = 0
    for split in ("rwc-other-dev", "karaoke-dev"):
        kind = board.SET_METRICS[split][0]
        for item in board.set_members(split)[:2]:
            try:
                row = board._score_item(split, kind, item, est, data)
            except FileNotFoundError:
                continue
            assert "pc_on50_f1" in row, f"scorer did not return pc_on50_f1 for {item}"
            checked += 1
    assert checked > 0, "no dev references found — cannot prove the eval path"
    print(f"[smoke] scored decoded notes against {checked} real dev reference(s); eval path OK")
    print("smoke OK: frame pipeline (features, rasterise, train, predict, decode) and the real-dev scorer "
          "run end to end on CPU with sane, non-empty output.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU plumbing check only; no GPU/SoulX/Basic Pitch")
    parser.add_argument("--songs", type=int, default=200, help="sung training songs (a GPU step)")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--frame-thresh", type=float, default=0.3)
    parser.add_argument("--onset-thresh", type=float, default=0.5)
    parser.add_argument("--prep-dev", action="store_true",
                        help="pre-compute Demucs vocal stems + BP posteriors for the real-dev sets, then exit")
    parser.add_argument("--eval-only", action="store_true", help="skip generation/training; score a saved model")
    parser.add_argument("--model", type=Path, default=None, help="path to a saved frame model (eval-only)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = paths.copyist_root()
    data = paths.data_dir()
    out = args.out or paths.out_dir()

    if args.smoke:
        return smoke(out, data)

    if args.prep_dev:
        print("[prep] Demucs vocal stems + BP posteriors for the real-dev sets (GPU; idempotent)")
        summary = prep_dev_stems(root, out)
        print(f"[prep] done: {summary}")
        return 0

    dec = {"frame_thresh": args.frame_thresh, "onset_thresh": args.onset_thresh}
    run_dir = out / "frames-run"
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        model = frames.FrameModel.load(args.model or run_dir / "model.pt")
    else:
        print(f"[gen ] {args.songs} sung songs (seed {args.seed}) — SoulX + Demucs (GPU)")
        ds = build_sung_dataset(root, out, args.songs, args.seed)
        ds.save(run_dir / "dataset.npz")
        print(f"[train] {len(ds.seqs)} clips over {len(ds.songs)} songs")
        model = frames.FrameModel(hidden=args.hidden, seed=args.seed)
        info = model.fit(ds.seqs, epochs=args.epochs)
        model.save(run_dir / "model.pt")
        print(f"[train] done: {info}")

    print("[eval] computing frozen dev features (cached BP posteriors reused)")
    feats_by_split = dev_features(root, out, data)
    print("[eval] on real dev (the honest success test)")
    ev = eval_real_dev(model, feats_by_split, data, out, dec, calibrate=True)
    payload = {"seed": args.seed, "songs": args.songs, "hidden": args.hidden,
               "frame_thresh": args.frame_thresh, "onset_thresh": args.onset_thresh, "real_dev": ev}
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    lines = ["# Sung frame-model pilot (real-dev success test)", "",
             "Frame model over frozen Basic Pitch posteriors + Demucs-vocal log-mel, trained on SoulX-sung",
             "sampler scores. Both sides cross-fit the onset shift on the other dev set.", "",
             "| Real dev set | model pc_on50 | baseline (BP one line) pc_on50 | model/base shift | n |",
             "|---|---|---|---|---|"]
    for split, r in ev.items():
        lines.append(f"| {split} | {r['model_pc_on50']} | {r['baseline_pc_on50']} | "
                     f"{r['model_shift']}/{r['baseline_shift']} | {r['n']} |")
    (out / "frames.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
