"""Tiny CLI that runs Basic Pitch's posteriors and writes them to a ``.npz``.

This module is the subprocess boundary that keeps TensorFlow out of the ear env: it imports ``basic_pitch``
(TensorFlow) **only** when actually run, and only ever under Copyist's ``.venv``. The ear env never imports
this module's TF dependency — it shells out to ``.venv``'s Python running ``-m
open_transcriber.bakeoff.bp_predict <wav> <out.npz>`` and then reads the cached ``.npz``.

Usage (under .venv):
    python -m open_transcriber.bakeoff.bp_predict INPUT.wav OUT.npz

Writes arrays: ``note`` (frames, 88), ``onset`` (frames, 88), ``contour`` (frames, 264).
"""

from __future__ import annotations

import sys
from pathlib import Path


def run(wav_path: Path, out_path: Path) -> Path:
    """Run Basic Pitch on ``wav_path`` and save its note/onset/contour posteriors to ``out_path`` (.npz)."""
    import numpy as np
    from basic_pitch.inference import predict  # TensorFlow import happens here, under .venv only

    model_output, _midi, _notes = predict(str(wav_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(
            handle,
            note=np.asarray(model_output["note"], dtype=np.float32),
            onset=np.asarray(model_output["onset"], dtype=np.float32),
            contour=np.asarray(model_output["contour"], dtype=np.float32),
        )
    tmp.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m open_transcriber.bakeoff.bp_predict INPUT.wav OUT.npz", file=sys.stderr)
        return 2
    run(Path(argv[0]), Path(argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
