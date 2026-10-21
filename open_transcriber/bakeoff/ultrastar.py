"""UltraStar Deluxe karaoke charts: header, timed notes, the reference, and the words (never printed).

TEST-ONLY. Karaoke charts are mostly CC BY-NC (evaluation/probe-eval only, never training). Not for
product/training import.

A chart (``song.txt``) is a header of ``#KEY:VALUE`` lines, then one line per event:

- ``: start length pitch syllable``: a normal note; ``*``: a golden note (both pitched);
- ``F``: freestyle, ``R``/``G``: rap and golden rap (no pitch: masked);
- ``- start``: a line break; ``E``: the end.

Times are in chart beats of ``15 / BPM`` seconds (a quarter of a beat at ``#BPM``). Per the stage 0a plan (D10):
onset = GAP/1000 + start·15/BPM, offset = onset + max(length, 1)·15/BPM, MIDI = pitch + 60 (the octave
isn't used: charts' octaves are unreliable). Relative-mode charts, ``#VERSION`` charts and duets are refused.

Syllables stay in memory (``repr=False``); nothing here prints them. ``lyrics_text`` writes the words to a
gitignored file and returns counts only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

PITCHED = (":", "*")
UNPITCHED = ("F", "R", "G")
_NOTE = re.compile(r"^([:*FRG])\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)(?:\s(.*))?$")
_BREAK = re.compile(r"^-\s*(-?\d+)(?:\s+(-?\d+))?\s*$")
#: Header fields read; every other tag (comments, sources, licences) is skipped unread.
HEADER_FIELDS = ("TITLE", "ARTIST", "MP3", "AUDIO", "BPM", "GAP", "LANGUAGE", "ENCODING")
_REFUSED = ("RELATIVE", "VERSION")


def decode(raw: bytes) -> str:
    """Chart text: UTF-8 (with or without a BOM), else Windows-1252."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252")


def chart_path(folder: Path) -> Path:
    """``song.txt`` when present, otherwise the folder's one non-licence ``.txt`` (ValueError if not one)."""
    folder = Path(folder)
    if (folder / "song.txt").is_file():
        return folder / "song.txt"
    candidates = [p for p in folder.iterdir() if p.suffix.lower() == ".txt"
                  and "licen" not in p.name.lower() and p.name.lower() != "instrumental.txt"]
    if len(candidates) != 1:
        raise ValueError(f"{folder.name}: expected one chart, found {len(candidates)}")
    return candidates[0]


def _number(text: str, tag: str) -> float:
    try:
        return float(text.strip().replace(",", "."))
    except ValueError:
        raise ValueError(f"#{tag} is not a number") from None


@dataclass(frozen=True)
class ChartNote:
    """One sung event of a chart. ``syllable`` is lyric text: never print it (excluded from repr)."""
    index: int
    kind: str
    start: int
    length: int
    pitch: int
    onset: float
    offset: float
    line: int
    syllable: str = field(default="", repr=False)

    @property
    def pitched(self) -> bool:
        return self.kind in PITCHED

    @property
    def midi(self) -> int:
        return self.pitch + 60


@dataclass
class Chart:
    header: dict[str, str]
    bpm: float
    gap: float  # seconds
    notes: list[ChartNote]
    lines: int

    @property
    def beat(self) -> float:
        """Seconds per chart beat."""
        return 15.0 / self.bpm


def read_header(path: Path) -> dict[str, str]:
    """The chart's whitelisted header fields (title, artist, audio file, BPM, GAP, language, encoding).

    Raises ``ValueError`` for relative-mode or ``#VERSION`` charts and duets.
    """
    header: dict[str, str] = {}
    for line in decode(Path(path).read_bytes()).splitlines():
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            key = key.strip().upper()
            if key in _REFUSED:
                raise ValueError(f"{Path(path).parent.name}: #{key} charts aren't supported")
            if key in HEADER_FIELDS:
                header[key] = value.strip()
        elif line.strip()[:2] in ("P1", "P2", "P3"):
            raise ValueError(f"{Path(path).parent.name}: duet charts aren't supported")
    return header


def parse(path: Path) -> Chart:
    """Parse a chart into timed notes (D10). ``ValueError`` on a malformed or unsupported chart."""
    path = Path(path)
    header = read_header(path)
    if "BPM" not in header:
        raise ValueError(f"{path.parent.name}: no #BPM")
    bpm = _number(header["BPM"], "BPM")
    if bpm <= 0:
        raise ValueError(f"{path.parent.name}: #BPM must be positive")
    gap = _number(header.get("GAP", "0"), "GAP") / 1000.0
    beat = 15.0 / bpm
    notes: list[ChartNote] = []
    line_no = 0
    ended = False
    for number, raw in enumerate(decode(path.read_bytes()).splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        if ended:
            break
        head = raw[0]
        if head == "E":
            ended = True
            continue
        if head == "-":
            if not _BREAK.match(raw.strip()):
                raise ValueError(f"{path.parent.name}: malformed line break on line {number}")
            line_no += 1
            continue
        match = _NOTE.match(raw.rstrip("\r\n"))
        if not match:
            raise ValueError(f"{path.parent.name}: malformed note on line {number}")
        kind, start, length, pitch, syllable = match.groups()
        start, length, pitch = int(start), int(length), int(pitch)
        onset = gap + start * beat
        notes.append(ChartNote(len(notes), kind, start, length, pitch, onset,
                               onset + max(length, 1) * beat, line_no, syllable or ""))
    for before, after in zip(notes, notes[1:]):
        if after.start < before.start:
            raise ValueError(f"{path.parent.name}: notes out of order at chart note {after.index}")
    return Chart(header, bpm, gap, notes, line_no + 1)


REFERENCE_COLUMNS = ("note_id", "onset", "offset", "midi", "midi_sung", "mask", "source")


def reference_rows(chart: Chart) -> list[dict]:
    """Reference rows: every chart note, unpitched ones masked as ``freestyle`` (their pitch is a placeholder)."""
    rows = []
    for note in chart.notes:
        rows.append({
            "note_id": note.index,
            "onset": note.onset,
            "offset": note.offset,
            "midi": note.midi,
            "midi_sung": note.midi,
            "mask": "" if note.pitched else "freestyle",
            "source": f"chart:{note.kind}",
        })
    return rows


def lyrics_text(chart: Chart) -> str:
    """The words as one paragraph, one line per chart line (D6). Never print the result.

    Syllables starting with a space begin a word; ``~`` continuations are dropped; freestyle syllables are kept.
    """
    lines: dict[int, str] = {}
    for note in chart.notes:
        syllable = note.syllable.replace("~", "")
        if not syllable.strip():
            continue
        lines[note.line] = lines.get(note.line, "") + syllable
    return "\n".join(" ".join(text.split()) for _, text in sorted(lines.items()) if text.strip()) + "\n"


def write_lyrics(chart: Chart, path: Path) -> dict:
    """Write ``lyrics_text`` to ``path`` (gitignored ``out/judge/lyrics/``); return counts only."""
    text = lyrics_text(chart)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return {"lines": text.count("\n"), "words": len(text.split())}
