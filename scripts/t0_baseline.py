#!/usr/bin/env python3
"""T0: score Spotify Basic Pitch (Apache-2.0) on Copyist truth takes.

Note F1 only (no chords). Compares against Copyist's open-engine table in
docs/transcribing.md. Weights download on first predict into the Basic Pitch
cache; never commit them.

Usage (from open-transcriber/, with Copyist checkout as parent):

  pip install -e ".[baseline]"
  python scripts/t0_baseline.py --copyist-root ..
  python scripts/t0_baseline.py --copyist-root .. --only silent-night,saints
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

from open_transcriber.basic_pitch_engine import transcribe_melody  # noqa: E402
from open_transcriber.copyist_cases import resolve_audio, select_cases  # noqa: E402
from open_transcriber.notes import to_eval_notes  # noqa: E402


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
    # evaluate.py inserts copyist on sys.path from its own parents
    spec.loader.exec_module(mod)
    return mod


def score_case(ev, copyist_root: Path, case, audio: Path) -> dict:
    song = copyist_root / case.song
    if not song.is_file():
        raise FileNotFoundError(song)
    t0 = time.perf_counter()
    events = transcribe_melody(audio)
    elapsed = time.perf_counter() - t0
    hyp = to_eval_notes(events)
    tn, _tc, spq = ev.truth(str(song))
    _f, shift, ratio = ev.best_alignment(tn, hyp, spq, True)
    f, p, r, hits, _offs = ev.match(tn, hyp, spq, True, shift, ratio)
    return {
        "name": case.name,
        "audio": _rel(audio, copyist_root),
        "n_hyp": len(hyp),
        "n_true": len(tn),
        "f1": f,
        "precision": p,
        "recall": r,
        "hits": hits,
        "shift": shift,
        "ratio": ratio,
        "seconds": elapsed,
    }


def format_table(rows: list[dict]) -> str:
    lines = [
        "| Case | audio | notes (vs true) | note F1 (P/R) | runtime |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | `{r['audio']}` | {r['n_hyp']} vs {r['n_true']} | "
            f"{r['f1']:.2f} ({r['precision']:.2f}/{r['recall']:.2f}) | {r['seconds']:.1f}s |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copyist-root", type=Path, default=ROOT.parent,
                    help="Copyist checkout (default: parent of this submodule)")
    ap.add_argument("--only", default=None, help="Comma-separated case names")
    ap.add_argument("--write", type=Path, default=None, help="Write a markdown table here")
    ap.add_argument("--mix", action="store_true", help="Use the mix instead of the vocal stem")
    args = ap.parse_args(argv)

    copyist_root = args.copyist_root.resolve()
    try:
        import basic_pitch  # noqa: F401
    except ImportError:
        print("Install the baseline extra: pip install -e \".[baseline]\"", file=sys.stderr)
        return 2

    ev = load_evaluate(copyist_root)
    cases = select_cases(args.only)
    rows = []
    print(f"T0 Basic Pitch on {copyist_root} ({len(cases)} case(s))\n")
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
        row = score_case(ev, copyist_root, case, audio)
        rows.append(row)
        print(
            f"  note F1 {row['f1']:.2f}  {row['n_hyp']} vs {row['n_true']}  "
            f"({row['seconds']:.1f}s)"
        )

    if not rows:
        print("No cases scored.", file=sys.stderr)
        return 1

    print()
    table = format_table(rows)
    print(table)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        body = (
            "# T0 baseline: Spotify Basic Pitch (Apache-2.0)\n\n"
            "Melody note F1 vs Copyist written scores (`evaluate.py --octave-free` alignment). "
            "No chords. Vocal stem when present.\n\n"
            f"{table}\n"
        )
        args.write.write_text(body, encoding="utf-8")
        print(f"\nWrote {args.write}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
