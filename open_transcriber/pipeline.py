"""Lead-sheet melody pipeline: Basic Pitch → adaptive candidates → tidy / clean_notes.

When a lyric target is known, several Basic Pitch presets are cleaned and ranked.
Ties break on mean BP amplitude so Amazing Grace keeps the high-amp 0.85 pass;
Saints can take melodia_trick=False. A small count improvement alone does not
displace a strong default (protects Tyger).
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
from open_transcriber.notes import NoteEvent, mean_amp
from open_transcriber.tidy import tidy

LOOSE_ONSET = 0.5
LOOSE_FRAME = 0.3
LOOSE_MIN_AMP = 0.05
UNDERDETECT = 0.75
# Switch away from the default pass only if another candidate is this many notes
# closer to the lyric target (or ties and has higher BP amp).
MIN_COUNT_IMPROVE = 4

# (onset, frame, min_amp, melodia_trick) — always tried alongside the default.
STRICT_PRESETS: tuple[tuple[float, float, float, bool], ...] = (
    (0.81, 0.64, 0.30, False),  # Saints: melodia off
    (0.82, 0.62, 0.32, True),
    (0.85, 0.65, 0.35, True),  # Amazing Grace
)
STRICT_ONSET, STRICT_FRAME, STRICT_MIN_AMP, _ = STRICT_PRESETS[1]


def _finish(raw: list[NoteEvent], audio: Path, target: int | None, *,
            force_target: bool, loose: bool,
            use_copyist_clean: bool) -> list[NoteEvent]:
    if loose:
        return tidy(raw, target, force_target=force_target, overshoot=1.2, short=0.0, merge=False)
    if use_copyist_clean:
        cleaned = clean_with_copyist(raw, audio, target)
        if cleaned is not None:
            return cleaned
    return tidy(raw, target, force_target=force_target)


def _pick_cleaned(
    audio: Path,
    raws: list[list[NoteEvent]],
    target: int,
    *,
    force_target: bool,
    use_copyist_clean: bool,
) -> list[NoteEvent]:
    """Rank cleaned passes; keep default unless another clearly wins on count/amp."""
    scored: list[tuple[int, float, int, list[NoteEvent]]] = []
    for i, raw in enumerate(raws):
        if not raw:
            continue
        if len(raw) < target * 0.85:
            continue
        finished = _finish(
            raw, audio, target, force_target=force_target, loose=False,
            use_copyist_clean=use_copyist_clean,
        )
        if len(finished) < target * 0.85:
            continue
        dist = abs(len(finished) - target)
        scored.append((dist, mean_amp(raw), i, finished))
    if not scored:
        return _finish(
            raws[0], audio, target, force_target=force_target, loose=False,
            use_copyist_clean=use_copyist_clean,
        )
    # Default is index 0 when it survived the filters.
    default = next((s for s in scored if s[2] == 0), scored[0])
    # Best by (count distance, then −amp, then prefer default).
    best = min(scored, key=lambda s: (s[0], -s[1], s[2]))
    improved = default[0] - best[0]
    # Switch if candidate is strictly closer to target without sacrificing confidence,
    # or ties in count distance and has higher amplitude confidence.
    if (improved > 0 and best[1] >= default[1] - 0.02) or (improved == 0 and best[1] > default[1] + 0.01):
        return best[3]
    return default[3]


def melody_from_audio(
    audio: Path,
    target: int | None = None,
    *,
    force_target: bool = False,
    adaptive: bool = True,
    use_copyist_clean: bool = True,
) -> list[NoteEvent]:
    """Transcribe a vocal (or mix) to tidied melody notes."""
    audio = Path(audio)
    raw = transcribe_melody(audio)
    if not adaptive or not target:
        return _finish(
            raw, audio, target, force_target=force_target, loose=False,
            use_copyist_clean=use_copyist_clean,
        )
    # Under-detect from the *raw* default pass only (cleaned count can collapse).
    if len(raw) < target * UNDERDETECT:
        loose = transcribe_melody(
            audio,
            onset_threshold=LOOSE_ONSET,
            frame_threshold=LOOSE_FRAME,
            min_amp=LOOSE_MIN_AMP,
        )
        return _finish(
            loose, audio, target, force_target=force_target, loose=True,
            use_copyist_clean=False,
        )
    raws = [raw]
    for ot, ft, amin, mel in STRICT_PRESETS:
        raws.append(
            transcribe_melody(
                audio,
                onset_threshold=ot,
                frame_threshold=ft,
                min_amp=amin,
                melodia_trick=mel,
            )
        )
    return _pick_cleaned(
        audio, raws, target, force_target=force_target,
        use_copyist_clean=use_copyist_clean,
    )


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
        f"{len(STRICT_PRESETS)} alt presets ranked vs lyric count "
        f"(switch if ≥{MIN_COUNT_IMPROVE} notes closer or amp tie-break); "
        f"loose onset={LOOSE_ONSET}/frame={LOOSE_FRAME} when notes < {UNDERDETECT:.0%} of target; "
        f"Copyist clean_notes when available"
    )
