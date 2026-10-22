"""Run GAME (openvpi) over the dev sets and drop its notes into the scoreboard.

GAME is a singing melody transcriber. Its ``infer.py extract`` call writes, per input ``<id>.wav``, a text
file ``<id>.txt`` whose lines are ``onset<TAB>offset<TAB>pitch`` (times in seconds). With
``--pitch-format number --round-pitch`` the pitch is an integer MIDI number, so a GAME ``.txt`` is a
header-less version of the board's drop-in system TSV (``onset offset midi``). :func:`convert_txt` reads one
and :func:`write_system_tsv` writes the board file.

We score GAME on both frozen dev sets, in all three model sizes, on the full mixture and on the Demucs vocal
stem. That is six systems, named ``game-<size>-<mix|stem>``. Each lands in
``out/judge/systems/game-<size>-<mix|stem>/<id>.tsv`` so the board picks it up with no code change.

GAME's released weights state no licence and were trained on private singing with non-commercial
augmentation, so this is testing only: never run on the owner's songs, never ship the weights.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from open_transcriber.bakeoff import paths

#: The three released GAME model sizes.
SIZES = ("small", "medium", "large")
#: The two audio sources we transcribe from.
SOURCES = ("mix", "stem")
#: The header the board's drop-in TSV loader expects.
TSV_HEADER = ("onset", "offset", "midi")


def config_name(size: str, source: str) -> str:
    """The board system name for a (size, source) run, e.g. ``game-small-stem``."""
    if size not in SIZES:
        raise ValueError(f"unknown GAME size {size!r}; expected one of {SIZES}")
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return f"game-{size}-{source}"


def convert_txt(txt_path: Path) -> list[tuple[float, float, int]]:
    """Read a GAME ``.txt`` (``onset<TAB>offset<TAB>midi``, no header) into sorted ``(onset, offset, midi)``.

    Blank lines are skipped. The pitch must be an integer MIDI number (run GAME with
    ``--pitch-format number --round-pitch``); a non-integer pitch means GAME wrote note names or cents and the
    run must be redone. Rows are returned sorted by onset so the board's loader is order-independent.
    """
    rows: list[tuple[float, float, int]] = []
    for lineno, line in enumerate(Path(txt_path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            raise ValueError(f"{txt_path}:{lineno}: expected onset<TAB>offset<TAB>midi, got {line!r}")
        onset, offset, pitch = parts
        midi = float(pitch)
        if midi != int(midi):
            raise ValueError(
                f"{txt_path}:{lineno}: pitch {pitch!r} is not an integer MIDI number; "
                "re-run GAME with --pitch-format number --round-pitch")
        rows.append((float(onset), float(offset), int(midi)))
    rows.sort()
    return rows


def write_system_tsv(rows: list[tuple[float, float, int]], dest: Path) -> None:
    """Write ``rows`` to ``dest`` as a board drop-in TSV with the ``onset offset midi`` header."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(TSV_HEADER)]
    for onset, offset, midi in rows:
        lines.append(f"{onset:.3f}\t{offset:.3f}\t{int(midi)}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_dir(raw_dir: Path, system_dir: Path, items: list[str]) -> list[str]:
    """Convert every ``<item>.txt`` in ``raw_dir`` to ``<item>.tsv`` in ``system_dir``; return items done."""
    done = []
    for item in items:
        txt = raw_dir / f"{item}.txt"
        if not txt.exists():
            continue
        write_system_tsv(convert_txt(txt), system_dir / f"{item}.tsv")
        done.append(item)
    return done


# --- audio layout (testing sets only) ---------------------------------------------------------------------

def rwc_mix_path(item: str, data: Path) -> Path:
    """The RWC mixture wav for ``rwc-g001`` -> ``.cache/judge/rwc/audio/RWC_G001.wav``."""
    code = item.split("-", 1)[1].upper()
    return data / "rwc" / "audio" / f"RWC_{code}.wav"


def usdx_mix_path(item: str, data: Path) -> Path:
    """The karaoke mixture wav: ``.cache/judge/usdx/audio/<item>.wav``."""
    return data / "usdx" / "audio" / f"{item}.wav"


def stem_path(item: str, out: Path) -> Path:
    """The Demucs vocal stem for any dev item: ``out/judge/stems/<item>/vocals.wav``."""
    return out / "stems" / item / "vocals.wav"


def source_path(item: str, source: str, data: Path, out: Path) -> Path:
    """The audio file to transcribe for ``item`` from ``source`` (``mix`` or ``stem``)."""
    if source == "stem":
        return stem_path(item, out)
    if item.startswith("rwc-"):
        return rwc_mix_path(item, data)
    return usdx_mix_path(item, data)


def model_path(size: str, data: Path) -> Path:
    """The GAME model file for a size: ``.cache/game/models/GAME-1.0-<size>/model.pt``."""
    return data.parent / "game" / "models" / f"GAME-1.0-{size}" / "model.pt"


@dataclass
class Timing:
    """Seconds GAME spent on a config, and how many songs that covered."""

    config: str
    songs: int
    seconds: float

    @property
    def seconds_per_song(self) -> float:
        return self.seconds / self.songs if self.songs else float("nan")


def load_timings(out: Path | None = None, name: str = "game-timings.json") -> dict[str, Timing]:
    """Read ``out/judge/systems/<name>`` into {config: Timing} (empty if absent).

    ``name`` lets a sibling runner (e.g. MuScriptor) keep its own timings file in the same layout.
    """
    out = out or paths.out_dir()
    path = out / "systems" / name
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: Timing(k, v["songs"], v["seconds"]) for k, v in payload.items()}


def save_timings(timings: dict[str, Timing], out: Path | None = None, name: str = "game-timings.json") -> None:
    """Write {config: Timing} to ``out/judge/systems/<name>`` (seconds and seconds/song)."""
    out = out or paths.out_dir()
    path = out / "systems" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        t.config: {"songs": t.songs, "seconds": round(t.seconds, 3),
                   "seconds_per_song": round(t.seconds_per_song, 3)}
        for t in timings.values()
    }
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
