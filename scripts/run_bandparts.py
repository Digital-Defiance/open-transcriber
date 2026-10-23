#!/usr/bin/env python3
"""Band-parts baseline on Copyist renders: transcribe each part and score it against its known MIDI.

CPU-only, clean-licence (Copyist's own renders), never on owner songs. For each song this:

1. builds the song if its band render and per-part MIDI are missing (``copyist build``);
2. transcribes the isolated part stem (``out/<song>/stems/<part>.wav``) and the full band mix
   (``out/<song>/<song>.wav``) with Basic Pitch (Apache-2.0), through the ``.venv`` subprocess boundary
   (``-m open_transcriber.bakeoff.bp_notes``) that keeps TensorFlow out of the ear env;
3. scores each transcription against the part's ground-truth MIDI and writes ``out/judge/bandparts.{md,json}``.

Resumable: a transcription whose TSV already exists is not recomputed. This is the first brick of the data
engine — render with known labels, measure recovery — so the recipe is proven on the small test-song set
before any scaling.

``--smoke`` is the pre-run guard: it checks the build CLI and the ``.venv`` Basic Pitch boundary resolve, and
round-trips a synthetic part MIDI + transcription through the real scorer, without building or transcribing.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import paths  # noqa: E402
from open_transcriber.board import bandparts  # noqa: E402

#: The frozen test-song set for the pilot (public-domain; held out from training elsewhere).
PILOT_SONGS = ("when-the-saints", "amazing-grace", "silent-night", "the-tyger", "major-general")


def copyist_root() -> Path:
    return paths.copyist_root()


def venv_python(root: Path) -> Path:
    """Copyist's .venv Python, which has Basic Pitch (TensorFlow). ``$COPYIST_VENV`` overrides.

    A git worktree under ``.worktrees/<name>`` shares the main checkout's ``.venv``, so fall back to the main
    checkout when the worktree has no ``.venv`` of its own.
    """
    env = os.environ.get("COPYIST_VENV")
    if env:
        return Path(env)
    here = root / ".venv" / "bin" / "python"
    if here.exists():
        return here
    # <main>/.worktrees/<name> -> <main>/.venv
    parts = root.parts
    if ".worktrees" in parts:
        main = Path(*parts[: parts.index(".worktrees")])
        shared = main / ".venv" / "bin" / "python"
        if shared.exists():
            return shared
    return here


def ensure_built(root: Path, song: str, parts: tuple[str, ...]) -> bool:
    """Build ``song`` if its band mix or any part's ground-truth MIDI is missing. Return True if ready."""
    out_song = root / "out" / song
    mix = out_song / f"{song}.wav"
    midis = [out_song / "stems" / "midi" / f"{p}.mid" for p in parts]
    stems = [out_song / "stems" / f"{p}.wav" for p in parts]
    if mix.exists() and all(m.exists() for m in midis) and all(s.exists() for s in stems):
        return True
    yaml = root / "songs" / f"{song}.yaml"
    if not yaml.exists():
        print(f"[warn] {song}: no song file {yaml}")
        return False
    if not mix.exists():
        print(f"[build] {song}")
        subprocess.run([str(venv_python(root)), "-m", "copyist.cli", "build", str(yaml)],
                       cwd=str(root), check=True)
    if not mix.exists():
        return False
    # A part the arrangement simply doesn't have (e.g. Silent Night is piano-vocal, no bass) is not a failure.
    missing = [p for p, m in zip(parts, midis) if not m.exists()]
    if missing:
        print(f"[note] {song}: arrangement has no {', '.join(missing)} part; scoring the parts it has")
    return mix.exists()


def transcribe(root: Path, wav: Path, out_tsv: Path) -> None:
    """Run Basic Pitch on ``wav`` via the .venv boundary, writing an onset/offset/pitch TSV."""
    if out_tsv.exists():
        return
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(venv_python(root)), "-m", "open_transcriber.bakeoff.bp_notes", str(wav), str(out_tsv)],
                   cwd=str(root / "open-transcriber"), check=True, stdin=subprocess.DEVNULL)


def smoke(root: Path) -> None:
    venv = venv_python(root)
    if not venv.exists():
        raise FileNotFoundError(f".venv Python not found: {venv} (Basic Pitch lives there)")
    # Round-trip a synthetic part MIDI + a near-perfect transcription through the real scorer.
    import struct

    def vlq(n: int) -> bytes:
        out = bytearray([n & 0x7F]); n >>= 7
        while n:
            out.insert(0, (n & 0x7F) | 0x80); n >>= 7
        return bytes(out)

    tmp = paths.out_dir() / "bandparts-run" / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    ev = bytearray()
    ev += vlq(0) + bytes([0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")
    for tick_on, pitch in ((0, 48), (480, 50)):
        ev += vlq(tick_on if pitch == 48 else 0) + bytes([0x90, pitch, 90])
        ev += vlq(480) + bytes([0x80, pitch, 0])
    ev += vlq(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + struct.pack(">I", len(ev)) + bytes(ev)
    (tmp / "bass.mid").write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, 1, 480) + track)
    (tmp / "bass.tsv").write_text("0.000\t0.500\t48\n0.500\t1.000\t50\n", encoding="utf-8")
    ref = bandparts.midi_notes(tmp / "bass.mid")
    est = bandparts.tsv_notes(tmp / "bass.tsv")
    from open_transcriber.bakeoff import score
    row = score.score_pair(ref, est, bandparts.PART_METRICS, item="smoke")
    assert row["pc_on50_f1"] == 1.0, f"self-score should be perfect, got {row['pc_on50_f1']}"
    print(f"smoke OK: .venv Basic Pitch boundary present ({venv}); scorer round-trips a part MIDI to F1 1.0.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="pre-run guard only; no build or transcription")
    parser.add_argument("--songs", default=",".join(PILOT_SONGS), help="comma-separated song slugs")
    parser.add_argument("--parts", default="bass", help="comma-separated parts to score (default: bass)")
    parser.add_argument("--out", type=Path, default=None, help="judge output folder (default: out/judge)")
    args = parser.parse_args(argv)

    root = copyist_root()
    out = args.out or paths.out_dir()

    if args.smoke:
        smoke(root)
        return 0

    songs = [s for s in args.songs.split(",") if s]
    parts = tuple(p for p in args.parts.split(",") if p)
    run_dir = out / "bandparts-run"

    transcriptions: dict[tuple[str, str], dict[str, Path]] = {}
    ready: list[str] = []
    for song in songs:
        if not ensure_built(root, song, parts):
            print(f"[skip] {song}: not built")
            continue
        ready.append(song)
        out_song = root / "out" / song
        for part in parts:
            stem_wav = out_song / "stems" / f"{part}.wav"
            mix_wav = out_song / f"{song}.wav"
            if stem_wav.exists():
                tsv = run_dir / song / f"{part}-stem.tsv"
                transcribe(root, stem_wav, tsv)
                transcriptions.setdefault((part, "stem"), {})[song] = tsv
            if mix_wav.exists():
                tsv = run_dir / song / f"{part}-mix.tsv"
                transcribe(root, mix_wav, tsv)
                transcriptions.setdefault((part, "mix"), {})[song] = tsv

    board = bandparts.build(ready, transcriptions, parts=parts, calibrate=True)
    data = bandparts.write(board, out=out)
    print(bandparts.to_markdown(board))
    return 0 if data["systems"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
