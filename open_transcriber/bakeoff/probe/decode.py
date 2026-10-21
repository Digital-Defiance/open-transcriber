"""Decode probe head logits into note/beat/chord/key events for scoring. Pure, CPU-only.

The probe emits per-frame logits at 50 Hz; these functions turn them into the event lists the strict scorer
and the beat/chord/key metrics consume. Frame index ``i`` maps to time ``i / rate`` seconds.
"""

from __future__ import annotations

import numpy as np

from .model import TARGET_RATE_HZ


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def frames_to_seconds(indices: np.ndarray, rate: float = TARGET_RATE_HZ) -> np.ndarray:
    return np.asarray(indices, dtype=np.float64) / rate


def decode_notes(pitch_logits: np.ndarray, silence_logits: np.ndarray, rate: float = TARGET_RATE_HZ,
                 silence_threshold: float = 0.5) -> list[tuple[float, float, int]]:
    """Per-frame pitch + silence logits -> note events [(onset_s, offset_s, midi)].

    A run of consecutive frames with the same argmax pitch and silence probability below the threshold is one
    note; its onset/offset are the run's first/last frame edges.
    """
    pitch = np.asarray(pitch_logits)
    silence = _sigmoid(np.asarray(silence_logits))
    voiced = silence < silence_threshold
    midi = pitch.argmax(axis=-1)
    notes = []
    i, n = 0, len(midi)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and voiced[j + 1] and midi[j + 1] == midi[i]:
            j += 1
        onset = i / rate
        offset = (j + 1) / rate
        notes.append((float(onset), float(offset), int(midi[i])))
        i = j + 1
    return notes


def decode_pulses(logits: np.ndarray, rate: float = TARGET_RATE_HZ, threshold: float = 0.5) -> list[float]:
    """Per-frame beat/downbeat logits -> onset times (seconds) at local maxima above the threshold."""
    p = _sigmoid(np.asarray(logits))
    times = []
    for i in range(len(p)):
        if p[i] < threshold:
            continue
        left = p[i - 1] if i > 0 else -1.0
        right = p[i + 1] if i + 1 < len(p) else -1.0
        if p[i] >= left and p[i] >= right:
            times.append(i / rate)
    return times


def decode_chords(root_logits: np.ndarray, quality_logits: np.ndarray,
                  rate: float = TARGET_RATE_HZ) -> list[tuple[float, float, int, str]]:
    """Per-frame chord root/quality logits -> spans [(onset_s, offset_s, root_pc, quality)].

    root argmax in 0..12 (12 = no-chord); quality argmax maps to maj/min/no-chord. Consecutive identical
    (root, quality) frames merge into one span.
    """
    root = np.asarray(root_logits).argmax(axis=-1)
    qual = np.asarray(quality_logits).argmax(axis=-1)
    quality_name = {0: "maj", 1: "min", 2: "no-chord"}
    spans = []
    i, n = 0, len(root)
    while i < n:
        j = i
        while j + 1 < n and root[j + 1] == root[i] and qual[j + 1] == qual[i]:
            j += 1
        root_pc = int(root[i]) if int(root[i]) < 12 else -1
        q = "no-chord" if root_pc == -1 else quality_name.get(int(qual[i]), "maj")
        spans.append((float(i / rate), float((j + 1) / rate), root_pc, q))
        i = j + 1
    return spans


def decode_key(key_logits: np.ndarray) -> int:
    """A clip's 24-key logits -> the 0..23 key class (argmax)."""
    return int(np.asarray(key_logits).argmax())
