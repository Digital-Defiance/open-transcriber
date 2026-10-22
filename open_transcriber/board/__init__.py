"""Scoreboard v1: one table of every transcriber on the frozen real dev sets.

``python -m open_transcriber.board`` scores the built-in systems (Basic Pitch raw and one line, Copyist's
``open``/``neural`` at grid and heard times) and any drop-in system
(``out/judge/systems/<name>/<id>.tsv``, header ``onset offset midi``) against the frozen RWC-other dev and
karaoke dev references. ``--calibrate`` fits a constant onset shift per system on one set and reports it on
the other (cross-fit), so the timing correction is never fitted and reported on the same set. The heard-time
systems also carry that measured correction under stable ``… corrected`` names (``corrections``), the same
shift the engine applies to its own heard-time output.

Evaluation only: RWC and the karaoke songs are CC BY-NC. Nothing here trains on them or leaves the Mac.
"""

from .board import SETS, Board, System, build, discover_systems, main
from .corrections import CORRECTED_SYSTEMS, correct_heard, fit_dev_shifts

__all__ = ["SETS", "Board", "System", "build", "discover_systems", "main",
           "CORRECTED_SYSTEMS", "correct_heard", "fit_dev_shifts"]
