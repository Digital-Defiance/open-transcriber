#!/usr/bin/env python3
"""T0: run off-the-shelf Apache/MIT AMT on Copyist truth takes (not implemented yet).

When ready, this will:
  1. Load a vocal/mix from Copyist's out/<slug>/takes/truth/ (or stems).
  2. Run MT3 and/or Omnizart (weights in .cache/, never committed).
  3. Emit planner-dialect ABC (or MIDI) for Copyist's evaluate.py --octave-free.

Usage (planned):

  python scripts/t0_baseline.py --copyist-root ../.. --only silent-night

See README.md and Copyist's docs/open-transcriber.md.
"""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    print(
        "T0 baselines are not wired yet. Install plan: MT3 (Apache-2.0) and/or "
        "Omnizart (MIT); compare with Copyist roundtrip.py.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
