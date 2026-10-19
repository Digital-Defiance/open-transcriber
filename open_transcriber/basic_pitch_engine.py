"""Spotify Basic Pitch (Apache-2.0) melody transcription.

Defaults are tuned on Copyist's YuE2 truth takes (2026-10-18): raised onset/frame
thresholds cut over-segmentation and beat the open engine on Silent Night and Tyger.
"""
from __future__ import annotations

from pathlib import Path

from open_transcriber.notes import NoteEvent

# Tuned on Saints / Silent Night / Amazing Grace / Tyger / Major-General truth vocals.
DEFAULT_ONSET = 0.75
DEFAULT_FRAME = 0.55
DEFAULT_MIN_HZ = 80.0
DEFAULT_MAX_HZ = 1000.0
# Drop ghost notes Basic Pitch emits at very low confidence.
DEFAULT_MIN_AMP = 0.3


def transcribe_melody(
    audio: Path,
    *,
    min_hz: float = DEFAULT_MIN_HZ,
    max_hz: float = DEFAULT_MAX_HZ,
    onset_threshold: float = DEFAULT_ONSET,
    frame_threshold: float = DEFAULT_FRAME,
    min_amp: float = DEFAULT_MIN_AMP,
    melodia_trick: bool = True,
) -> list[NoteEvent]:
    """Return note events from Basic Pitch. Raises ImportError if not installed."""
    from basic_pitch.inference import predict

    _model_output, _midi, note_events = predict(
        str(audio),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_frequency=min_hz,
        maximum_frequency=max_hz,
        melodia_trick=melodia_trick,
    )
    out: list[NoteEvent] = []
    for start, end, pitch, amp, _bends in note_events:
        dur = float(end) - float(start)
        if dur <= 0 or float(amp) < min_amp:
            continue
        out.append(NoteEvent(float(start), dur, int(pitch)))
    out.sort(key=lambda n: n.start)
    return out
