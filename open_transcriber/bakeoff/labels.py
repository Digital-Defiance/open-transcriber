"""Exact labels for a sampled score, built by construction (never from audio).

A sampled copyist-song/0 melody has a written score and a single tempo, so every label a probe head needs is
exact: note onsets/offsets in seconds and MIDI pitch (from the score's notes and tempo), a beat grid and
downbeats (from meter and tempo), a chord root + major/minor/no-chord per chord span, the 24-key class, and
the meter. Notes come through Copyist's own loader (``copyist.song.load_song`` and ``copyist.lyrics``), so tie
continuations, rests and grace notes are handled exactly as the engraver sees them.

The score is the source of truth; nothing here reads audio. Serialized to deterministic JSON so the same song
always yields byte-identical labels.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path

from copyist.abcplan import ACC_TEXT, NATURAL
from copyist.lyrics import lyric_slots
from copyist.song import ABC_BARLINE, Song, abc_bar_lengths, load_song

LABELS_FORMAT = "copyist-bakeoff-labels/1"

#: Chord-symbol suffixes we treat as minor; everything else diatonic in the sampler is major.
_MINOR_SUFFIXES = ("m", "min", "-")
#: A quoted ABC chord annotation and its root (letter + optional ^/_ accidental) and the suffix.
_CHORD_TOKEN = re.compile(r'"([A-G])([\^_=]?)([^"]*)"')
#: Accidental marks inside an ABC chord root to a semitone alteration.
_CHORD_ACC = {"^": 1, "_": -1, "=": 0, "": 0}


@dataclass
class NoteLabel:
    onset: float
    offset: float
    midi: int


@dataclass
class ChordLabel:
    onset: float
    offset: float
    root_pc: int          # 0..11, pitch class of the chord root
    quality: str          # "maj", "min" or "no-chord"


@dataclass
class ScoreLabels:
    """All exact labels for one sampled score."""
    seed: int
    key: str              # e.g. "D#m"
    key_class: int        # 0..23: tonic pitch class, + 12 if minor
    meter: str            # e.g. "3/4"
    tempo: int            # quarter-note BPM
    duration: float       # seconds
    notes: list[NoteLabel] = field(default_factory=list)
    beats: list[float] = field(default_factory=list)       # beat onsets in seconds
    downbeats: list[float] = field(default_factory=list)   # bar onsets in seconds
    chords: list[ChordLabel] = field(default_factory=list)

    def to_json(self) -> str:
        data = {"format": LABELS_FORMAT, **asdict(self)}
        return json.dumps(data, indent=1, sort_keys=True) + "\n"


def _seconds_per_quarter(tempo_bpm: float) -> float:
    return 60.0 / float(tempo_bpm)


def key_class(key: str) -> int:
    """24-key class: tonic pitch class, plus 12 when the key is minor."""
    m = re.match(r"\s*([A-Ga-g])([#b]?)\s*(m|min|minor)?", key)
    if not m:
        raise ValueError(f"can't read key {key!r}")
    pc = (NATURAL[m.group(1).upper()] + {"#": 1, "b": -1, "": 0}[m.group(2)]) % 12
    minor = bool(m.group(3))
    return pc + (12 if minor else 0)


def _note_labels(song: Song, spq: float) -> list[NoteLabel]:
    labels = []
    mel = song.melody
    for n in lyric_slots(mel):
        onset_q = float(n.getOffsetInHierarchy(mel))
        dur_q = float(n.quarterLength)
        labels.append(NoteLabel(onset=round(onset_q * spq, 6),
                                offset=round((onset_q + dur_q) * spq, 6),
                                midi=int(n.pitch.midi)))
    return labels


def _beat_grid(meter: str, total_q: float, spq: float) -> tuple[list[float], list[float]]:
    """Beat onsets and downbeat (bar) onsets in seconds, from the meter and tempo.

    The beat unit is the meter's denominator (a quarter is 1.0 in quarter units); the bar length is
    numerator beat-units. Beats march from 0 to the score's total quarter length.
    """
    num, den = (int(x) for x in meter.split("/"))
    beat_q = 4.0 / den                 # one beat in quarter notes
    bar_q = num * beat_q               # one bar in quarter notes
    beats, downbeats = [], []
    t = 0.0
    # a tiny epsilon so the final beat exactly at the end is included
    while t <= total_q + 1e-6:
        beats.append(round(t * spq, 6))
        t += beat_q
    t = 0.0
    while t <= total_q + 1e-6:
        downbeats.append(round(t * spq, 6))
        t += bar_q
    return beats, downbeats


def _chord_root_pc(letter: str, acc: str) -> int:
    return (NATURAL[letter] + _CHORD_ACC[acc]) % 12


def _chord_quality(suffix: str) -> str:
    low = suffix.strip().lower()
    if low.startswith(_MINOR_SUFFIXES):
        return "min"
    return "maj"


def _chord_labels(song: Song, spq: float, total_q: float) -> list[ChordLabel]:
    """Chord spans from the ABC chord annotations, placed at bar starts.

    Each bar's first ``"..."`` chord token sets the chord for that bar; a bar with no chord inherits the
    previous one, and any gap before the first chord is a ``no-chord`` span. Bar start offsets come from
    ``abc_bar_lengths`` (quarter notes), so placement is exact from the written score.
    """
    unit = song.unit
    bar_lengths = abc_bar_lengths(song.melody_abc, unit)
    # Split the ABC into bar chunks the same way abc_bar_lengths does, but keep chord tokens.
    text = ABC_BARLINE.split(song.melody_abc)
    chunks = [c for c in text if c.strip()]
    # Align chunk list to bar_lengths (both drop empty bars identically).
    starts_q, acc_q = [], 0.0
    for length in bar_lengths:
        starts_q.append(acc_q)
        acc_q += float(length)

    spans: list[ChordLabel] = []
    current: tuple[int, str] | None = None
    for i, length in enumerate(bar_lengths):
        chunk = chunks[i] if i < len(chunks) else ""
        token = _CHORD_TOKEN.search(chunk) if chunk else None
        onset = round(starts_q[i] * spq, 6)
        offset = round((starts_q[i] + float(length)) * spq, 6)
        if token:
            root_pc = _chord_root_pc(token.group(1), token.group(2))
            quality = _chord_quality(token.group(3))
            current = (root_pc, quality)
        if current is None:
            span = ChordLabel(onset=onset, offset=offset, root_pc=-1, quality="no-chord")
        else:
            span = ChordLabel(onset=onset, offset=offset, root_pc=current[0], quality=current[1])
        # merge consecutive identical spans
        if spans and spans[-1].root_pc == span.root_pc and spans[-1].quality == span.quality:
            spans[-1].offset = span.offset
        else:
            spans.append(span)
    return spans


def labels_for_song(path: Path | str, seed: int | None = None) -> ScoreLabels:
    """Build exact labels from a sampled song YAML (by construction, from the written score and tempo)."""
    song = load_song(path)
    tempo = int(str(song.tempo).split("=")[-1]) if "=" in str(song.tempo) else int(song.tempo)
    spq = _seconds_per_quarter(tempo)
    bar_lengths = abc_bar_lengths(song.melody_abc, song.unit)
    total_q = float(sum(bar_lengths)) if bar_lengths else 0.0
    notes = _note_labels(song, spq)
    beats, downbeats = _beat_grid(song.meter, total_q, spq)
    chords = _chord_labels(song, spq, total_q)
    return ScoreLabels(
        seed=seed if seed is not None else -1,
        key=song.key,
        key_class=key_class(song.key),
        meter=song.meter,
        tempo=tempo,
        duration=round(total_q * spq, 6),
        notes=notes,
        beats=beats,
        downbeats=downbeats,
        chords=chords,
    )


def write_labels(labels: ScoreLabels, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(labels.to_json(), encoding="utf-8")
    return path
