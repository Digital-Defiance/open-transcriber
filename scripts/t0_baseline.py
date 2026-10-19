#!/usr/bin/env python3
"""Score open-transcriber melody on Copyist truth takes.

Modes:
  raw    — Basic Pitch library defaults (over-segments)
  tuned  — raised onset/frame + amp gate only
  tidy   — adaptive thresholds + pitch trim + merge / soft lyric fold (default)

Usage:

  bash scripts/setup_baseline.sh
  python scripts/t0_baseline.py --copyist-root ..
  python scripts/t0_baseline.py --copyist-root .. --mode tuned --only silent-night
  python scripts/t0_baseline.py --copyist-root .. --write ../out/t0-basic-pitch.md
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from open_transcriber.basic_pitch_engine import (  # noqa: E402
    DEFAULT_FRAME,
    DEFAULT_MIN_AMP,
    DEFAULT_ONSET,
    transcribe_melody,
)
from open_transcriber.copyist_cases import resolve_audio, select_cases  # noqa: E402
from open_transcriber.notes import to_eval_notes  # noqa: E402
from open_transcriber.pipeline import describe_defaults, lyric_note_target, melody_from_audio  # noqa: E402

# Open-engine note F1 from Copyist roundtrip (2026-10-18), for the printed delta.
OPEN_ENGINE_F1 = {
    "saints": 0.85,
    "silent-night": 0.80,
    "amazing-grace": 0.88,
    "tyger": 0.78,
    "major-general": 0.84,
}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def load_evaluate(copyist_root: Path):
    path = copyist_root / "scripts" / "transcribe" / "evaluate.py"
    if not path.is_file():
        raise SystemExit(f"Copyist evaluate.py not found at {path}")
    spec = importlib.util.spec_from_file_location("tr_eval", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def transcribe(mode: str, audio: Path, target: int | None):
    if mode == "raw":
        return transcribe_melody(
            audio, onset_threshold=0.5, frame_threshold=0.3, min_amp=0.0
        )
    if mode == "tuned":
        return transcribe_melody(audio)
    if mode == "tidy":
        return melody_from_audio(audio, target)
    if mode == "force-target":
        return melody_from_audio(audio, target, force_target=True)
    raise SystemExit(f"Unknown mode {mode!r}")


def score_case(ev, copyist_root: Path, case, audio: Path, mode: str) -> dict:
    song = copyist_root / case.song
    if not song.is_file():
        raise FileNotFoundError(song)
    target = lyric_note_target(copyist_root, case.song)
    t0 = time.perf_counter()
    events = transcribe(mode, audio, target)
    elapsed = time.perf_counter() - t0
    hyp = to_eval_notes(events)
    tn, _tc, spq = ev.truth(str(song))
    _f, shift, ratio = ev.best_alignment(tn, hyp, spq, True)
    f, p, r, hits, _offs = ev.match(tn, hyp, spq, True, shift, ratio)
    open_f1 = OPEN_ENGINE_F1.get(case.name)
    return {
        "name": case.name,
        "audio": _rel(audio, copyist_root),
        "mode": mode,
        "target": target,
        "n_hyp": len(hyp),
        "n_true": len(tn),
        "f1": f,
        "precision": p,
        "recall": r,
        "hits": hits,
        "shift": shift,
        "ratio": ratio,
        "seconds": elapsed,
        "open_f1": open_f1,
        "delta": (f - open_f1) if open_f1 is not None else None,
    }


def format_table(rows: list[dict]) -> str:
    lines = [
        "| Case | notes (vs true) | note F1 (P/R) | vs open | runtime |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        if r["delta"] is None:
            delta = "—"
        else:
            sign = "+" if r["delta"] >= 0 else ""
            delta = f"{sign}{r['delta']:.2f} (open {r['open_f1']:.2f})"
        lines.append(
            f"| {r['name']} | {r['n_hyp']} vs {r['n_true']} | "
            f"{r['f1']:.2f} ({r['precision']:.2f}/{r['recall']:.2f}) | "
            f"{delta} | {r['seconds']:.1f}s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copyist-root", type=Path, default=ROOT.parent)
    ap.add_argument("--only", default=None)
    ap.add_argument("--write", type=Path, default=None)
    ap.add_argument("--mix", action="store_true")
    ap.add_argument(
        "--mode",
        choices=("raw", "tuned", "tidy", "force-target"),
        default="tidy",
        help=f"default tidy = onset {DEFAULT_ONSET}/frame {DEFAULT_FRAME}/amp {DEFAULT_MIN_AMP} + merge",
    )
    args = ap.parse_args(argv)

    copyist_root = args.copyist_root.resolve()
    try:
        import basic_pitch  # noqa: F401
    except ImportError:
        print("Run: bash open-transcriber/scripts/setup_baseline.sh", file=sys.stderr)
        return 2

    ev = load_evaluate(copyist_root)
    cases = select_cases(args.only)
    rows = []
    print(f"open-transcriber [{args.mode}] on {copyist_root} ({len(cases)} case(s))")
    if args.mode == "tidy":
        print(f"  {describe_defaults()}\n")
    else:
        print()
    for case in cases:
        try:
            if args.mix:
                audio = copyist_root / case.mix
                if not audio.is_file():
                    raise FileNotFoundError(audio)
            else:
                audio = resolve_audio(copyist_root, case)
        except FileNotFoundError as e:
            print(f"… {case.name}\n  skip: {e}")
            continue
        print(f"… {case.name} ({audio.name})")
        row = score_case(ev, copyist_root, case, audio, args.mode)
        rows.append(row)
        delta = ""
        if row["delta"] is not None:
            sign = "+" if row["delta"] >= 0 else ""
            delta = f"  Δopen {sign}{row['delta']:.2f}"
        print(
            f"  note F1 {row['f1']:.2f}  {row['n_hyp']} vs {row['n_true']}  "
            f"({row['seconds']:.1f}s){delta}"
        )

    if not rows:
        print("No cases scored.", file=sys.stderr)
        return 1

    print()
    table = format_table(rows)
    print(table)
    wins = sum(1 for r in rows if r["delta"] is not None and r["delta"] >= 0)
    print(f"\nBeats or ties open engine on {wins}/{sum(1 for r in rows if r['delta'] is not None)} cases.")
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        body = (
            f"# open-transcriber melody (`--mode {args.mode}`)\n\n"
            "Melody note F1 vs Copyist written scores (octave-free). Vocal stem when present.\n\n"
            f"{table}\n"
        )
        args.write.write_text(body, encoding="utf-8")
        print(f"\nWrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
