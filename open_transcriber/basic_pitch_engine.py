"""Spotify Basic Pitch (Apache-2.0) as a T0 melody baseline.

Polyphonic AMT aimed at one instrument; we run it on a Demucs vocal stem when
available, otherwise the mix. No chords — note F1 only for the round-trip gate.
"""
from __future__ import annotations

from pathlib import Path

from open_transcriber.notes import NoteEvent


def transcribe_melody(audio: Path, *, min_hz: float = 80.0, max_hz: float = 1000.0) -> list[NoteEvent]:
    """Return note events from Basic Pitch. Raises ImportError if not installed."""
    from basic_pitch.inference import predict

    _model_output, _midi, note_events = predict(
        str(audio),
        minimum_frequency=min_hz,
        maximum_frequency=max_hz,
        melodia_trick=True,
    )
    out: list[NoteEvent] = []
    for start, end, pitch, _amp, _bends in note_events:
        dur = float(end) - float(start)
        if dur <= 0:
            continue
        out.append(NoteEvent(float(start), dur, int(pitch)))
    out.sort(key=lambda n: n.start)
    return out
