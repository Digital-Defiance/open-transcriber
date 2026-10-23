"""Tiny CLI that runs Basic Pitch's note extraction and writes note events as a TSV.

Like :mod:`open_transcriber.bakeoff.bp_predict`, this is the subprocess boundary that keeps TensorFlow out of
the ear env: it imports ``basic_pitch`` only when run, and only ever under Copyist's ``.venv``. The ear env
shells out to ``.venv``'s Python running ``-m open_transcriber.bakeoff.bp_notes <wav> <out.tsv>`` and then
reads the TSV.

Basic Pitch's own note extractor returns ``(start_s, end_s, pitch_midi, amplitude, pitch_bends)`` per note;
this writes ``onset<TAB>offset<TAB>pitch`` (seconds, integer MIDI), the board's header-less drop-in shape.

Usage (under .venv):
    python -m open_transcriber.bakeoff.bp_notes INPUT.wav OUT.tsv
"""

from __future__ import annotations

import sys
from pathlib import Path


def run(wav_path: Path, out_path: Path) -> Path:
    """Run Basic Pitch on ``wav_path`` and write its note events to ``out_path`` (onset offset midi TSV)."""
    from basic_pitch.inference import predict  # TensorFlow import happens here, under .venv only

    _model_output, _midi, notes = predict(str(wav_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for note in notes:
        start, end, pitch = float(note[0]), float(note[1]), int(note[2])
        if end > start:
            lines.append(f"{start:.3f}\t{end:.3f}\t{pitch}")
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m open_transcriber.bakeoff.bp_notes INPUT.wav OUT.tsv", file=sys.stderr)
        return 2
    run(Path(argv[0]), Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
