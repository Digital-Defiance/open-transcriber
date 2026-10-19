"""Note events shared by baselines and (later) trained engines."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoteEvent:
    """One sung/played note: start and duration in seconds, MIDI pitch."""

    start: float
    duration: float
    midi: int

    def as_eval_row(self) -> list[float | int]:
        """Shape expected by Copyist's scripts/transcribe/evaluate.py match()."""
        return [self.start, max(0.05, self.duration), int(self.midi)]


def to_eval_notes(events: list[NoteEvent]) -> list[list[float | int]]:
    return [e.as_eval_row() for e in events]
