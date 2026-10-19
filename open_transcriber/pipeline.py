"""Lead-sheet melody pipeline: Basic Pitch → adaptive thresholds → tidy / clean_notes.

Copyist's truth takes taught three regimes:
  - Sustained singing near the lyric count: default raised onset/frame thresholds.
  - Over-segmentation (Saints, Amazing Grace): stricter thresholds + clean_notes.
  - Allegro patter (Major-General): loose thresholds; high ones starve; skip merge.

When a lyric note target is known:
  - under-detection (< 75% of target) re-runs loose and skips same-pitch merge;
  - over-detection (> 120% of target) re-runs stricter only if it clearly helps;
  - otherwise Copyist clean_notes (when available) beats soft tidy on lyrical takes.

Magenta MT3 is Apache-2.0 but not trained on singing (official Colab caveat), so
this path stays Basic Pitch + tidy. Gate: scripts/transcribe/roundtrip.py.
"""
from __future__ import annotations

from pathlib import Path

from open_transcriber.basic_pitch_engine import (
    DEFAULT_FRAME,
    DEFAULT_MIN_AMP,
    DEFAULT_ONSET,
    transcribe_melody,
)
from open_transcriber.copyist_bridge import clean_with_copyist
from open_transcriber.notes import NoteEvent
from open_transcriber.tidy import tidy

LOOSE_ONSET = 0.5
LOOSE_FRAME = 0.3
LOOSE_MIN_AMP = 0.05
UNDERDETECT = 0.75

# Saints prefers 0.82/0.62; Amazing Grace prefers 0.85/0.65 — try both when flooded.
STRICT_PRESETS = (
    (0.82, 0.62, 0.32),
    (0.85, 0.65, 0.35),
)
STRICT_ONSET, STRICT_FRAME, STRICT_MIN_AMP = STRICT_PRESETS[0]
OVERDETECT = 1.20


def _prefer_strict(default: list[NoteEvent], strict: list[NoteEvent], target: int) -> bool:
    """Take the strict pass only when it clearly lands on the lyric count."""
    d, s = len(default), len(strict)
    if s < target * 0.9:
        return False
    if abs(s - target) <= 1 and abs(d - target) > 3:
        return True
    return abs(s - target) + 5 < abs(d - target)


def _adaptive_raw(audio: Path, target: int | None) -> tuple[list[NoteEvent], bool]:
    """Return (raw notes, loose_patter)."""
    raw = transcribe_melody(audio)
    if not target:
        return raw, False
    if len(raw) < target * UNDERDETECT:
        raw = transcribe_melody(
            audio,
            onset_threshold=LOOSE_ONSET,
            frame_threshold=LOOSE_FRAME,
            min_amp=LOOSE_MIN_AMP,
        )
        return raw, True
    if len(raw) > target * OVERDETECT:
        best = raw
        for ot, ft, amp in STRICT_PRESETS:
            strict = transcribe_melody(
                audio, onset_threshold=ot, frame_threshold=ft, min_amp=amp
            )
            if _prefer_strict(best, strict, target) and abs(len(strict) - target) <= abs(
                len(best) - target
            ):
                best = strict
        raw = best
    return raw, False


def melody_from_audio(
    audio: Path,
    target: int | None = None,
    *,
    force_target: bool = False,
    adaptive: bool = True,
    use_copyist_clean: bool = True,
) -> list[NoteEvent]:
    """Transcribe a vocal (or mix) to tidied melody notes."""
    if adaptive:
        raw, loose = _adaptive_raw(audio, target)
    else:
        raw, loose = transcribe_melody(audio), False
    if loose:
        # Patter: leave Basic Pitch notes alone — same-pitch merge collapses eighths.
        return tidy(raw, target, force_target=force_target, overshoot=1.2, short=0.0, merge=False)
    if use_copyist_clean:
        cleaned = clean_with_copyist(raw, audio, target)
        if cleaned is not None:
            return cleaned
    return tidy(raw, target, force_target=force_target)


def lyric_note_target(copyist_root: Path, song_rel: str) -> int | None:
    """First-verse note target from a Copyist song YAML, or None if unavailable."""
    import sys

    song_path = copyist_root / song_rel
    if not song_path.is_file():
        return None
    root = str(copyist_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        import yaml
        from copyist import lyrics as lyr
    except ImportError:
        return None
    data = yaml.safe_load(song_path.read_text(encoding="utf-8"))
    verses = data.get("lyrics") or []
    if not verses:
        return None
    verse = verses[0]
    if isinstance(verse, list):
        verse = " ".join(verse)
    first_lines = [ln for ln in str(verse).strip().splitlines() if ln.strip()]
    if not first_lines:
        return None
    return lyr.note_target(" ".join(first_lines))


def describe_defaults() -> str:
    return (
        f"default onset={DEFAULT_ONSET}/frame={DEFAULT_FRAME}/amp={DEFAULT_MIN_AMP}; "
        f"strict onset={STRICT_ONSET}/frame={STRICT_FRAME} when notes > {OVERDETECT:.0%} of target; "
        f"loose onset={LOOSE_ONSET}/frame={LOOSE_FRAME} when notes < {UNDERDETECT:.0%} of target; "
        f"Copyist clean_notes when available"
    )
