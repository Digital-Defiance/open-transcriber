"""Basic Pitch side input + encoder-free control source (run as a ``.venv`` subprocess, cached).

Basic Pitch (Apache-2.0) is the fixed melody side input for every probe and the only input to the
encoder-free control. It runs on TensorFlow, which must never enter the ear env, so this helper shells out
to Copyist's ``.venv`` running ``open_transcriber.bakeoff.bp_predict`` and reads the cached ``.npz``. It is
NOT an ``Extractor`` (it is not an encoder with layers); it is a separate helper the run loop uses alongside
the layer extractors.

Resolving the ``.venv``: ``$COPYIST_VENV_PYTHON`` wins (set it in a worktree, whose ``.venv`` is not checked
out), else ``copyist_root()/.venv/bin/python``.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import numpy as np

from ... import paths

POSTERIOR_KEYS = ("note", "onset", "contour")


def venv_python_default() -> Path:
    """Copyist's .venv Python (with basic_pitch/TensorFlow). ``$COPYIST_VENV_PYTHON`` overrides the default."""
    env = os.environ.get("COPYIST_VENV_PYTHON")
    if env:
        return Path(env)
    return paths.copyist_root() / ".venv" / "bin" / "python"


def side_cache_dir() -> Path:
    return paths.copyist_root() / ".cache" / "ear" / "features" / "basic-pitch"


def _clip_key(wav_path: Path) -> str:
    h = hashlib.sha256()
    with open(wav_path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class BasicPitchSideInput:
    """Compute (once, cached) Basic Pitch note/onset/contour posteriors for a clip, via a ``.venv`` subprocess."""

    def __init__(self, cache_dir: Path | None = None, venv_python: Path | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir is not None else side_cache_dir()
        self.venv_python = Path(venv_python) if venv_python is not None else venv_python_default()

    def cache_path(self, wav_path: Path) -> Path:
        return self.cache_dir / f"{_clip_key(Path(wav_path))}.npz"

    def posteriors(self, wav_path: Path) -> dict[str, np.ndarray]:
        """Return ``{"note","onset","contour"}`` posteriors, computing and caching them on first use."""
        wav_path = Path(wav_path)
        path = self.cache_path(wav_path)
        if not path.exists():
            if not self.venv_python.exists():
                raise FileNotFoundError(
                    f"Basic Pitch needs Copyist's .venv Python at {self.venv_python}; it was not found")
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [str(self.venv_python), "-m", "open_transcriber.bakeoff.bp_predict",
                 str(wav_path), str(path)],
                check=True, cwd=str(paths.copyist_root() / "open-transcriber"),
            )
        with np.load(path) as data:
            return {key: data[key] for key in POSTERIOR_KEYS}
