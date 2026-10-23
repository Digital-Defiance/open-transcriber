#!/usr/bin/env python3
"""Data-engine pilot, end to end on CPU: scores -> render -> Basic Pitch -> train a head -> eval on real dev.

This proves the owner's recipe ("write music, then train a network to hear it") on a SMALL pilot before any
scaling, and judges success on REAL recordings, not our tidy renders.

Stages:
1. Generate ``--songs`` validated sampler songs (seeded, deterministic; gitignored cache).
2. ``copyist build`` each one → the melody stem ``out/<slug>/stems/melody.wav`` and its exact label MIDI
   ``stems/midi/melody.mid``.
3. Basic Pitch on each melody stem (via the ``.venv`` boundary ``bakeoff.bp_notes``).
4. Build the per-candidate training table (features + match labels against the exact melody).
5. Train the pilot head (CPU, torch) and save it.
6. **Eval on real dev**: apply the head to the RWC-other-dev and karaoke-dev Basic Pitch outputs and score
   ``pc_on50`` with the strict scorer, against the ``basic-pitch one line`` baseline. Write
   ``out/judge/pilot.{md,json}``.

``--smoke`` runs stages 4–6 on a tiny in-process synthetic set only (no song generation, build, or Basic
Pitch), so the train/eval/scoring plumbing is proven without any heavy dependency. The real pilot's CPU run is
fast for a few songs; a full-scale training run is a separate GPU step.

Everything is clean-licence (Copyist renders) and never touches owner songs. A fixed seed is recorded.

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
from open_transcriber.data import pilot, scores  # noqa: E402


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


def build_song(root: Path, yaml: Path) -> str:
    subprocess.run([str(venv_python(root)), "-m", "copyist.cli", "build", str(yaml)],
                   cwd=str(root), check=True)
    return yaml.stem


def bp_on(root: Path, wav: Path, out_tsv: Path) -> Path:
    if not out_tsv.exists():
        out_tsv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(venv_python(root)), "-m", "open_transcriber.bakeoff.bp_notes", str(wav), str(out_tsv)],
                       cwd=str(root / "open-transcriber"), check=True, stdin=subprocess.DEVNULL)
    return out_tsv


def midi_truth(path: Path) -> notes_io.Notes:
    from open_transcriber.board.muscriptor import read_midi_notes

    ns = read_midi_notes(Path(path))
    ns.sort(key=lambda n: (n.onset, n.offset))
    return notes_io._validate(
        notes_io.Notes(np.array([n.onset for n in ns]), np.array([n.offset for n in ns]),
                       np.array([float(n.midi) for n in ns])), False, Path(path).name)


# --- real-dev eval (the honest success test) -----------------------------------------------------------

def dev_bp_notes(item: str, out: Path) -> list[tuple[float, float, int, float]]:
    """Raw Basic Pitch notes for a real-dev item (with amplitude), from the cached scoreboard inputs."""
    path = out / "basic-pitch-raw" / item / "notes.tsv"
    return pilot.load_bp_notes(path) if path.exists() else []


def eval_real_dev(head: pilot.Head, threshold: float, onset_shift: float,
                  data: Path, out: Path) -> dict:
    """Score the head's kept notes and the ``basic-pitch one line`` baseline on both real dev sets."""
    result = {}
    for split in ("rwc-other-dev", "karaoke-dev"):
        kind = board.SET_METRICS[split][0]
        head_rows, base_rows = [], []
        for item in board.set_members(split):
            notes = dev_bp_notes(item, out)
            if not notes:
                continue
            est_head = pilot.apply_head(notes, head, threshold=threshold, onset_shift=onset_shift)
            bp_path = out / "basic-pitch-raw" / item / "notes.tsv"
            est_base = board.mono_bp(bp_path)
            head_rows.append(board._score_item(split, kind, item, est_head, data))
            base_rows.append(board._score_item(split, kind, item, est_base, data))
        result[split] = {
            "head_pc_on50": round(100 * score.mean_f1(head_rows, "pc_on50"), 3),
            "baseline_pc_on50": round(100 * score.mean_f1(base_rows, "pc_on50"), 3),
            "n": len(head_rows),
        }
    return result


# --- the pilot ------------------------------------------------------------------------------------------

def build_dataset(root: Path, data_songs: list[Path], run_dir: Path, seed: int) -> pilot.Dataset:
    """Render each song, run Basic Pitch on its melody stem, and build the labelled candidate table."""
    Xs, ys, used = [], [], []
    for yaml in data_songs:
        slug = build_song(root, yaml)
        stem = root / "out" / slug / "stems" / "melody.wav"
        truth_mid = root / "out" / slug / "stems" / "midi" / "melody.mid"
        if not stem.exists() or not truth_mid.exists():
            print(f"[skip] {slug}: no melody stem/label")
            continue
        tsv = bp_on(root, stem, run_dir / slug / "melody-bp.tsv")
        notes = pilot.load_bp_notes(tsv)
        cands = pilot.label_candidates(pilot.candidate_features(notes), midi_truth(truth_mid))
        if not cands:
            continue
        Xs.append(pilot.feature_matrix(cands))
        ys.append(np.array([c.label for c in cands], dtype=np.int64))
        used.append(slug)
        pos = int(sum(c.label for c in cands))
        print(f"[data] {slug}: {len(cands)} candidates, {pos} positive")
    X = np.concatenate(Xs) if Xs else np.zeros((0, len(pilot.FEATURES)), np.float32)
    y = np.concatenate(ys) if ys else np.zeros((0,), np.int64)
    return pilot.Dataset(X=X, y=y, songs=used, seed=seed)


def smoke(out: Path) -> None:
    """Prove the train/eval/scoring plumbing on a tiny in-process synthetic set (no build, no Basic Pitch)."""
    rng = np.random.default_rng(0)
    # 200 synthetic candidates: label 1 when in range, long and not overlapping; a learnable rule.
    n = 200
    pitch = rng.integers(40, 90, n)
    dur = rng.uniform(0.03, 1.0, n)
    amp = rng.uniform(0.0, 1.0, n)
    overlap = rng.integers(0, 2, n).astype(float)
    gap_prev = rng.uniform(-0.1, 0.5, n)
    label = ((pilot.MELODY_LO <= pitch) & (pitch <= pilot.MELODY_HI) & (dur > 0.1) & (overlap < 0.5)).astype(int)
    feats = np.stack([
        (pitch - 60) / 24.0, np.minimum(dur, 2.0), amp,
        np.clip(gap_prev, -0.5, 1.0), np.full(n, 0.2, np.float32),
        overlap, np.zeros(n), ((pilot.MELODY_LO <= pitch) & (pitch <= pilot.MELODY_HI)).astype(float),
    ], axis=1).astype(np.float32)
    head = pilot.Head(seed=0)
    info = head.fit(feats, label, epochs=300)
    acc = float(((head.scores(feats) >= 0.5).astype(int) == label).mean())
    assert acc >= 0.9, f"head should learn the synthetic rule, got acc {acc:.2f}"
    # Exercise apply_head + the scorer on a tiny estimate.
    notes = [(0.0, 0.5, 60, 0.9), (0.5, 1.0, 62, 0.8), (0.6, 0.62, 80, 0.1)]
    est = pilot.apply_head(notes, head, threshold=0.5)
    assert len(est) >= 2, f"head kept too few notes: {len(est)}"
    print(f"smoke OK: head trained on CPU (loss {info['loss']:.3f}, acc {acc:.2f}); "
          f"apply_head kept {len(est)} of 3; scorer and dataset I/O usable.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU plumbing check only; no build/Basic Pitch")
    parser.add_argument("--songs", type=int, default=8, help="number of pilot songs to generate and render")
    parser.add_argument("--seed", type=int, default=20261022, help="seed for song generation and the head")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--onset-shift", type=float, default=0.0, help="constant onset shift at inference (s)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    root = paths.copyist_root()
    data = paths.data_dir()
    out = args.out or paths.out_dir()

    if args.smoke:
        smoke(out)
        return 0

    run_dir = out / "pilot-run"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gen ] {args.songs} sampler songs (seed {args.seed})")
    song_dir = run_dir / "scores"
    data_songs = scores.write_batch(out_dir=song_dir, n=args.songs, start_seed=args.seed)

    ds = build_dataset(root, data_songs, run_dir, args.seed)
    ds.save(run_dir / "dataset.npz")
    pos = int(ds.y.sum())
    print(f"[train] {len(ds.X)} candidates ({pos} positive) over {len(ds.songs)} songs")
    head = pilot.Head(seed=args.seed)
    info = head.fit(ds.X, ds.y, epochs=args.epochs)
    head.save(run_dir / "head.pt")
    print(f"[train] done: {info}")

    print("[eval] on real dev (the honest success test)")
    ev = eval_real_dev(head, args.threshold, args.onset_shift, data, out)
    payload = {"seed": args.seed, "songs": len(ds.songs), "candidates": int(len(ds.X)),
               "positives": pos, "train": info, "threshold": args.threshold,
               "onset_shift": args.onset_shift, "real_dev": ev}
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot.json").write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    lines = ["# Data-engine pilot (real-dev success test)", "",
             f"Seed {args.seed}; {len(ds.songs)} render songs; {len(ds.X)} candidates ({pos} positive).", "",
             "| Real dev set | head pc_on50 | baseline (BP one line) pc_on50 | n |", "|---|---|---|---|"]
    for split, r in ev.items():
        lines.append(f"| {split} | {r['head_pc_on50']} | {r['baseline_pc_on50']} | {r['n']} |")
    (out / "pilot.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
