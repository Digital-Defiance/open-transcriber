"""Post-process Basic Pitch (and later neural) note events toward a lead sheet.

Raised BP onset/frame thresholds cut over-segmentation; same-pitch merge cleans
the rest. Forcing the lyric count often hurts F1 — only fold when badly over.
"""
from __future__ import annotations

from open_transcriber.notes import NoteEvent


def drop_pitch_outliers(
    events: list[NoteEvent], *, near: int = 7, far: int = 12, long: float = 0.25
) -> list[NoteEvent]:
    """Drop stray partials / octave ghosts far from the sung line's median pitch.

    Same idea as Copyist's clean_notes: within ±near always; a longer note up to
    ±far can stay (Silent Night's high F). Helps Tyger and Major-General.
    """
    if len(events) < 3:
        return events
    med = sorted(e.midi for e in events)[len(events) // 2]
    return [
        e
        for e in events
        if abs(e.midi - med) <= near
        or (abs(e.midi - med) <= far and e.duration >= long)
    ]


def merge_same_pitch(events: list[NoteEvent], gap: float = 0.15) -> list[NoteEvent]:
    """Join same-pitch neighbours that look like splits, not allegro patter.

    Equal-length short notes on one pitch (Major-General eighths) stay separate —
    cascading merges were collapsing 128 notes to ~40.
    """
    if not events:
        return events
    out = [events[0]]
    for e in events[1:]:
        prev = out[-1]
        prev_end = prev.start + prev.duration
        if e.midi == prev.midi and e.start - prev_end <= gap:
            short_d, long_d = sorted((prev.duration, e.duration))
            # Patter: both syllable-length and similar → keep. Grace/echo → merge.
            if short_d >= 0.10 and short_d * 5 > long_d * 2:
                out.append(e)
            else:
                end = max(prev_end, e.start + e.duration)
                out[-1] = NoteEvent(prev.start, end - prev.start, prev.midi)
        else:
            out.append(e)
    return out


def fold_short(events: list[NoteEvent], short: float = 0.10) -> list[NoteEvent]:
    """Fold very short notes into the nearer-pitch neighbour."""
    if len(events) < 2:
        return events
    out = list(events)
    changed = True
    while changed:
        changed = False
        for i, n in enumerate(out):
            if n.duration >= short:
                continue
            cands: list[tuple[float, int]] = []
            if i > 0:
                cands.append((abs(n.midi - out[i - 1].midi), i - 1))
            if i + 1 < len(out):
                cands.append((abs(n.midi - out[i + 1].midi), i + 1))
            if not cands:
                continue
            cands.sort()
            j = cands[0][1]
            a, b = out[min(i, j)], out[max(i, j)]
            pitch = a.midi if a.duration >= b.duration else b.midi
            out[min(i, j)] = NoteEvent(a.start, (b.start + b.duration) - a.start, pitch)
            del out[max(i, j)]
            changed = True
            break
    return out


def fold_to_target(events: list[NoteEvent], target: int | None) -> list[NoteEvent]:
    """Merge the shortest note into its nearer-pitch neighbour until `target`."""
    if not target or len(events) <= target:
        return events
    evs = list(events)
    while len(evs) > target and len(evs) >= 2:
        i = min(range(len(evs)), key=lambda k: evs[k].duration)
        if i == 0:
            j = 1
        elif i == len(evs) - 1:
            j = i - 1
        else:
            j = i - 1 if abs(evs[i].midi - evs[i - 1].midi) <= abs(evs[i].midi - evs[i + 1].midi) else i + 1
        a, b = evs[min(i, j)], evs[max(i, j)]
        pitch = a.midi if a.duration >= b.duration else b.midi
        evs[min(i, j)] = NoteEvent(a.start, (b.start + b.duration) - a.start, pitch)
        del evs[max(i, j)]
    return evs


def tidy(events: list[NoteEvent], target: int | None = None, *,
         force_target: bool = False, overshoot: float = 1.6,
         short: float = 0.10, merge: bool = True,
         pitch_trim: bool = True) -> list[NoteEvent]:
    """Default tidy: pitch trim, same-pitch merge, short folds, soft lyric cap.

    `force_target` always folds to the lyric count (often hurts). Otherwise fold
    only when note count exceeds `overshoot * target` (1.6 keeps Amazing Grace's
    tuned extras; tighter caps destroyed F1).
    """
    out = list(events)
    if pitch_trim:
        out = drop_pitch_outliers(out)
    if merge:
        out = merge_same_pitch(out)
    if short > 0:
        out = fold_short(out, short=short)
    if not target:
        return out
    if force_target or len(out) > int(target * overshoot):
        out = fold_to_target(out, target)
    return out
