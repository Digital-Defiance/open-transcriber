"""Loading reference and estimated notes for the strict scorer. numpy only.

Ported from stage 0a ``judge/notes_io.py`` (source ``aa88f5a``); the SheetSage2 ``.lab``/``events.json``
loaders are dropped because the bake-off scores probe/engine outputs, not SheetSage.

Every loader returns ``Notes`` (onset and offset in seconds, MIDI pitch, a mask reason per note) and validates
it: times must be finite and not negative, pitches finite. References must have offset > onset (the reference
builders drop zero-length notes). Estimates with offset <= onset are kept with a 1 ms duration and counted
(``repaired``): mir_eval refuses zero-length intervals, and onset-only metrics don't look at offsets.

Formats:
- reference TSV: ``note_id onset offset midi midi_sung mask source`` (``refs.py``);
- Copyist: ``events.json`` in ``copyist-events/1``;
- raw Basic Pitch: ``notes.tsv`` (``onset offset midi amplitude``, with a header).

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MIN_DURATION = 0.001
COPYIST_FORMAT = "copyist-events/1"


@dataclass
class Notes:
    onset: np.ndarray
    offset: np.ndarray
    midi: np.ndarray
    mask: np.ndarray = None  # str per note: "" scored, else a reason
    repaired: int = 0
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.onset = np.asarray(self.onset, dtype=np.float64).reshape(-1)
        self.offset = np.asarray(self.offset, dtype=np.float64).reshape(-1)
        self.midi = np.asarray(self.midi, dtype=np.float64).reshape(-1)
        if self.mask is None:
            self.mask = np.array([""] * len(self.onset), dtype=object)
        else:
            self.mask = np.asarray(self.mask, dtype=object).reshape(-1)
        if not (len(self.onset) == len(self.offset) == len(self.midi) == len(self.mask)):
            raise ValueError("onset, offset, pitch and mask must have the same length")

    def __len__(self) -> int:
        return len(self.onset)

    @property
    def scored(self) -> np.ndarray:
        return self.mask == ""

    def select(self, keep: np.ndarray) -> "Notes":
        return Notes(self.onset[keep], self.offset[keep], self.midi[keep], self.mask[keep], 0, dict(self.meta))


def empty() -> Notes:
    return Notes([], [], [])


def _validate(notes: Notes, reference: bool, where: str) -> Notes:
    for name in ("onset", "offset", "midi"):
        values = getattr(notes, name)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{where}: non-finite {name}")
    if np.any(notes.onset < 0) or np.any(notes.offset < 0):
        raise ValueError(f"{where}: negative time")
    bad = notes.offset <= notes.onset
    if bad.any():
        if reference:
            raise ValueError(f"{where}: {int(bad.sum())} notes with offset <= onset")
        notes.offset = np.where(bad, notes.onset + MIN_DURATION, notes.offset)
        notes.repaired = int(bad.sum())
    if np.any(notes.midi <= 0):
        raise ValueError(f"{where}: non-positive pitch")
    return notes


def load_reference(path: Path, octave_exact: bool = False, masks: dict[int, str] | None = None) -> Notes:
    """A reference TSV; ``octave_exact`` takes ``midi_sung``; ``masks`` adds QA reasons by note id."""
    onsets, offsets, pitches, reasons, ids = [], [], [], [], set()
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            note_id = int(row["note_id"])
            if note_id in ids:
                raise ValueError(f"{Path(path).name}: duplicate note id {note_id}")
            ids.add(note_id)
            onsets.append(float(row["onset"]))
            offsets.append(float(row["offset"]))
            pitches.append(float(row["midi_sung" if octave_exact else "midi"]))
            reason = row["mask"] or ""
            if masks and note_id in masks and not reason:
                reason = masks[note_id]
            reasons.append(reason)
    return _validate(Notes(onsets, offsets, pitches, reasons), True, Path(path).name)


def load_copyist_events(path: Path, heard: bool = False) -> Notes:
    """Copyist's ``events.json`` (``copyist-events/1``): the written notes on the band's beat grid, or with
    ``heard`` the diagnostic heard times (the grid shift undone)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != COPYIST_FORMAT:
        raise ValueError(f"{Path(path).name}: not {COPYIST_FORMAT}")
    notes = payload.get("notes", [])
    on, off = ("heard_onset", "heard_offset") if heard else ("onset", "offset")
    meta = {k: payload.get(k) for k in ("engine", "melody_source", "octave_above", "shift_beats")}
    return _validate(Notes([n[on] for n in notes], [n[off] for n in notes], [n["midi"] for n in notes],
                           meta=meta), False, Path(path).name)


def load_basic_pitch(path: Path) -> Notes:
    """Raw Basic Pitch ``notes.tsv`` (header ``onset offset midi amplitude``)."""
    onsets, offsets, pitches = [], [], []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            onsets.append(float(row["onset"]))
            offsets.append(float(row["offset"]))
            pitches.append(float(row["midi"]))
    return _validate(Notes(onsets, offsets, pitches), False, Path(path).name)
