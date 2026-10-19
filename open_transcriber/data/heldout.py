"""Held-out guard: keep the benchmark songs out of training data.

The five round-trip tunes (and the two big songs added later) are held out so
the transcriber's scores stay honest. A candidate song is a *leak* when it is
one of those tunes in disguise. We catch three kinds of disguise:

- a transposed and/or re-rhythmed copy of a held-out melody, caught by a
  transposition- and tempo-invariant melodic fingerprint (n-grams of pitch
  intervals, plus coarse-binned inter-onset ratios);
- a copy under the same title;
- a copy carrying a held-out verse's lyric text.

Notes are read through Copyist's own loader (copyist.song.load_song and
copyist.lyrics.lyric_slots) so the ABC is never hand-parsed here.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from copyist.lyrics import lyric_slots
from copyist.song import Song, load_song

# open_transcriber/data/heldout.py -> open_transcriber/data -> open_transcriber -> <submodule> -> copyist root
COPYIST_ROOT = Path(os.environ.get("COPYIST_ROOT") or Path(__file__).resolve().parents[3])


def note_sequence(song: Song) -> tuple[list[int], list[float]]:
    """A song's melody as (MIDI pitches, onset offsets in quarter notes).

    Uses Copyist's own note-slot logic, so tie continuations, rests and grace
    notes are excluded exactly as the engraver sees them.
    """
    slots = lyric_slots(song.melody)
    pitches = [n.pitch.midi for n in slots]
    offsets = [float(n.getOffsetInHierarchy(song.melody)) for n in slots]
    return pitches, offsets


def pitch_intervals(pitches: list[int]) -> list[int]:
    """Consecutive pitch differences (semitones). Transposition-invariant."""
    return [b - a for a, b in zip(pitches, pitches[1:])]


def onset_contour(offsets: list[float], *, bin_to: float = 0.25) -> list[float]:
    """Ratios of successive inter-onset gaps, coarse-binned for tempo-invariance.

    The gap between onsets is scale-dependent; the ratio of one gap to the next
    is not, so a melody played twice as fast has the same contour. Each ratio is
    rounded to the nearest ``bin_to`` so small timing wobble does not change it.
    """
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    ratios = []
    for prev, cur in zip(gaps, gaps[1:]):
        if prev <= 0:
            continue
        ratio = cur / prev
        ratios.append(round(ratio / bin_to) * bin_to)
    return ratios


def ngrams(seq: list, n: int) -> set[tuple]:
    """The set of length-``n`` windows of ``seq`` (empty if too short)."""
    if n <= 0 or len(seq) < n:
        return set()
    return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}


_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")


def normalise_title(text: str) -> str:
    """Lowercase, strip punctuation/underscores/stars, collapse whitespace."""
    text = (text or "").lower().replace("_", " ").replace("*", " ")
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalise_lyrics(verses: list[str] | str) -> str:
    """One normalised lyric string for a song's verses (same cleaning as titles)."""
    if isinstance(verses, str):
        verses = [verses]
    joined = " ".join(str(v) for v in (verses or []))
    return normalise_title(joined)


@dataclass
class HeldoutTune:
    """A held-out song, fingerprinted once at load time."""
    name: str
    title: str
    path: Path
    pitches: list[int]
    intervals: list[int]
    onset_ratios: list[float]
    lyrics_norm: str
    title_norm: str


@dataclass
class Fingerprint:
    """Precomputed n-gram sets for one held-out tune at the current thresholds."""
    interval_8grams: set[tuple] = field(default_factory=set)
    interval_5grams: set[tuple] = field(default_factory=set)
    onset_8grams: set[tuple] = field(default_factory=set)
    title_norm: str = ""
    lyrics_norm: str = ""


@dataclass
class LeakResult:
    """Whether a candidate leaks a held-out tune, and why."""
    is_leak: bool
    reason: str | None = None
    matched_tune: str | None = None


def _tune_from_song(name: str, path: Path, song: Song) -> HeldoutTune:
    pitches, offsets = note_sequence(song)
    return HeldoutTune(
        name=name,
        title=song.title,
        path=path,
        pitches=pitches,
        intervals=pitch_intervals(pitches),
        onset_ratios=onset_contour(offsets),
        lyrics_norm=normalise_lyrics(song.verses),
        title_norm=normalise_title(song.title),
    )


def load_heldout(root: Path | str = COPYIST_ROOT) -> list[HeldoutTune]:
    """Build the held-out tune list from the runtime-derived song set.

    Reads copyist_cases.held_out_songs(root) so the two big songs added later
    are picked up automatically, then fingerprints each with load_song.
    """
    from open_transcriber.copyist_cases import held_out_songs

    root = Path(root)
    tunes = []
    for path in held_out_songs(root):
        song = load_song(path)
        tunes.append(_tune_from_song(path.stem, path, song))
    return tunes


def is_held_out(
    song: Song,
    heldout: list[HeldoutTune],
    *,
    gram_n: int = 8,
    share_n: int = 5,
    share_frac: float = 0.30,
) -> LeakResult:
    """Decide whether ``song`` is a disguised copy of a held-out tune.

    Rejects (is_leak True) when the candidate, against ANY held-out tune:
    - shares any ``gram_n``-gram of pitch intervals; or
    - shares more than ``share_frac`` of its ``share_n``-gram set; or
    - normalises to the same (or a contained) title; or
    - carries lyric text contained in (or containing) a held-out verse.

    All thresholds are keyword args so they can be tuned.
    """
    pitches, _ = note_sequence(song)
    intervals = pitch_intervals(pitches)
    cand_big = ngrams(intervals, gram_n)
    cand_small = ngrams(intervals, share_n)
    cand_title = normalise_title(song.title)
    cand_lyrics = normalise_lyrics(song.verses)

    for tune in heldout:
        if cand_title and tune.title_norm and (
            cand_title == tune.title_norm
            or cand_title in tune.title_norm
            or tune.title_norm in cand_title
        ):
            return LeakResult(True, f"title matches held-out tune {tune.name!r}", tune.name)

    for tune in heldout:
        if cand_lyrics and tune.lyrics_norm and (
            cand_lyrics in tune.lyrics_norm or tune.lyrics_norm in cand_lyrics
        ):
            return LeakResult(True, f"lyric text matches held-out tune {tune.name!r}", tune.name)

    for tune in heldout:
        tune_big = ngrams(tune.intervals, gram_n)
        if cand_big & tune_big:
            return LeakResult(
                True,
                f"shares a {gram_n}-gram melodic fingerprint with held-out tune {tune.name!r}",
                tune.name,
            )
        if cand_small:
            tune_small = ngrams(tune.intervals, share_n)
            shared = cand_small & tune_small
            if len(shared) > share_frac * len(cand_small):
                frac = len(shared) / len(cand_small)
                return LeakResult(
                    True,
                    f"shares {frac:.0%} of its {share_n}-gram melodic fingerprint "
                    f"with held-out tune {tune.name!r}",
                    tune.name,
                )

    return LeakResult(False)


def check_candidate(
    path: Path | str,
    heldout: list[HeldoutTune] | None = None,
    **kwargs,
) -> LeakResult:
    """Load the song at ``path`` and test it against the held-out set.

    Loads the held-out set itself when ``heldout`` is None. Extra keyword args
    are passed through to is_held_out (gram_n, share_n, share_frac).
    """
    if heldout is None:
        heldout = load_heldout()
    return is_held_out(load_song(path), heldout, **kwargs)
