"""The measured timing correction for heard-time outputs, fitted on dev and never on a test set.

A transcriber's heard onset times sit a little before or after the sung truth: Basic Pitch fires early,
Copyist's ``open``/``neural`` heard times land a touch late. The scoreboard already measures the best constant
onset shift per system by a cross-fit (``board.build(calibrate=True)``: fit on one dev set, report on the
other). This module turns that measurement into a correction the engine can apply to its own heard-time output
so downstream consumers get already-aligned times, and names the corrected systems on the board.

The rule is the board's rule: the shift applied to a set is the one fitted on the *other* dev set, so no shift
is ever fitted and reported on the same data. :func:`fit_dev_shifts` returns, per system and set, that
cross-fitted shift (seconds). :func:`correct_heard` applies a shift to a :class:`notes_io.Notes`.

The headline measured corrections (from the 2026-10-22 dev scoreboard, both sets' cross-fit):
- ``open`` heard: about +20 ms (per-set fits +10 ms on RWC, +30 ms on karaoke).
- ``neural`` heard: about +15 ms (per-set fits +10 ms on RWC, +20 ms on karaoke).
- ``basic-pitch one line``: about -40 ms (per-set fits -50 ms on RWC, -30 ms on karaoke).

Evaluation only. The dev sets are CC BY-NC; this measures a correction, it does not train on them.
Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import numpy as np

from open_transcriber.bakeoff import notes_io
from open_transcriber.board import board

#: The heard-time (or one-line) systems a timing correction applies to, and the corrected name shown.
CORRECTED_SYSTEMS: dict[str, str] = {
    "open (heard)": "open (heard, corrected)",
    "neural (heard)": "neural (heard, corrected)",
    "basic-pitch one line": "basic-pitch one line (corrected)",
}


def fit_dev_shifts(b: board.Board) -> dict[str, dict[str, float]]:
    """Per system and set, the cross-fitted onset shift (seconds) recorded on a built, calibrated board.

    Reads ``b.shifts``: each corrected system (``… corrected``) carries the shift the board applied to each
    set (the one fitted on the *other* set), so the engine and the board use one measured correction. Returns
    ``{corrected_name: {set_name: shift}}`` for the systems in :data:`CORRECTED_SYSTEMS`.
    """
    wanted = set(CORRECTED_SYSTEMS.values())
    return {name: dict(per_set) for name, per_set in b.shifts.items() if name in wanted}


def correct_heard(notes: notes_io.Notes, shift_seconds: float) -> notes_io.Notes:
    """A copy of ``notes`` with the measured timing correction applied to every onset and offset.

    The correction is a constant shift in seconds (durations kept, times clipped at 0). A non-finite shift is
    rejected: a correction must be a real measured number, not NaN or infinity.
    """
    if not np.isfinite(shift_seconds):
        raise ValueError(f"timing correction must be finite, got {shift_seconds!r}")
    return board.shift_notes(notes, float(shift_seconds))

