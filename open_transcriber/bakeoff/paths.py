"""Where the judge keeps its data and outputs, and the free-space floor.

- ``data_dir()``: test data, references, environments and models. ``$JUDGE_DATA``, else
  ``<copyist root>/.cache/judge`` (gitignored).
- ``out_dir()``: system outputs, scores, logs. ``$JUDGE_OUT``, else ``<copyist root>/out/judge`` (gitignored).

The Copyist root is ``$COPYIST_ROOT``, else the checkout this submodule sits in.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Stop before free space on the data disk would fall below this (owner, 2026-10-19).
FREE_SPACE_FLOOR = 150 * 10**9


class DiskFloorError(RuntimeError):
    """Raised when a job would leave less free space than the floor."""


def copyist_root() -> Path:
    """The Copyist checkout: ``$COPYIST_ROOT``, else the repo that holds this submodule."""
    env = os.environ.get("COPYIST_ROOT")
    if env:
        return Path(env)
    # .../<copyist>/open-transcriber/open_transcriber/bakeoff/paths.py
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    """The judge's data folder (test sets, references, environments, models)."""
    env = os.environ.get("JUDGE_DATA")
    return Path(env) if env else copyist_root() / ".cache" / "judge"


def out_dir() -> Path:
    """The judge's output folder (system outputs, scores, logs)."""
    env = os.environ.get("JUDGE_OUT")
    return Path(env) if env else copyist_root() / "out" / "judge"


def _existing(path: Path) -> Path:
    """The nearest existing ancestor of ``path`` (disk_usage needs a real path)."""
    path = Path(path).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def free_bytes(path: Path | None = None) -> int:
    """Free bytes on the disk that holds ``path`` (default: the data folder)."""
    return shutil.disk_usage(_existing(path or data_dir())).free


def require_free_space(need_bytes: int, floor: int = FREE_SPACE_FLOOR, path: Path | None = None) -> int:
    """Refuse a job that would leave less than ``floor`` bytes free; return the free bytes now.

    ``need_bytes`` is what the job will add (a download's size, a batch's outputs).
    """
    free = free_bytes(path)
    if free - int(need_bytes) < floor:
        raise DiskFloorError(
            f"Not enough disk: {free / 1e9:.1f} GB free, the job needs {need_bytes / 1e9:.1f} GB, "
            f"and at least {floor / 1e9:.0f} GB must stay free."
        )
    return free
