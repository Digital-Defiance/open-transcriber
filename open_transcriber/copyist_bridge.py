"""Optional bridge into Copyist's clean_notes (same tidy the open engine uses).

Keeps open-transcriber runnable without Copyist installed; when Copyist is on
sys.path (the normal submodule layout), BP notes get the lyric-aware tidy that
closed Amazing Grace and lifted Silent Night / Tyger on the note table.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from open_transcriber.notes import NoteEvent


def clean_with_copyist(
    events: list[NoteEvent],
    audio: Path,
    target: int | None,
) -> list[NoteEvent] | None:
    """Run Copyist clean_notes on BP events. Returns None if Copyist isn't importable."""
    try:
        import librosa
        from copyist.transcribe import HeardNote, clean_notes
    except ImportError:
        return None
    if not events:
        return []
    try:
        y, sr = librosa.load(str(audio), sr=22050, mono=True)
    except Exception:
        return None
    hop = 256
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    midi = np.full(len(times), np.nan)
    heard: list = []
    for e in events:
        a = int(np.searchsorted(times, e.start))
        b = int(np.searchsorted(times, e.start + e.duration))
        midi[a : max(a + 1, b)] = e.midi
        heard.append(HeardNote(e.start, e.start + e.duration, float(e.midi), attack=True, spread=0.05))
    cleaned = clean_notes(heard, times, midi, rms, target=target)
    return [NoteEvent(n.start, n.end - n.start, int(round(n.midi))) for n in cleaned]
