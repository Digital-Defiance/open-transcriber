"""Truth-take cases shared with Copyist's scripts/transcribe/roundtrip.py.

Paths are relative to a Copyist checkout root. Kept here so T0 can run without
importing the copyist package (only evaluate.py is loaded for scoring).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    song: str          # relative to copyist root
    mix: str
    vocals: str
    accompaniment: str


# Keep in sync with copyist/scripts/transcribe/roundtrip.py CASES.
CASES: list[Case] = [
    Case(
        "saints",
        "songs/when-the-saints.yaml",
        "out/when-the-saints/takes/truth/production.wav",
        "out/when-the-saints-go-marching-in/transcribe/stems/htdemucs/source/vocals.wav",
        "out/when-the-saints-go-marching-in/transcribe/stems/htdemucs/source/no_vocals.wav",
    ),
    Case(
        "silent-night",
        "songs/silent-night.yaml",
        "out/silent-night/takes/truth/production.wav",
        "out/silent-night/transcribe/stems/htdemucs/source/vocals.wav",
        "out/silent-night/transcribe/stems/htdemucs/source/no_vocals.wav",
    ),
    Case(
        "amazing-grace",
        "songs/amazing-grace.yaml",
        "out/amazing-grace/production/production.wav",
        "out/amazing-grace/production/htdemucs/production/vocals.wav",
        "out/amazing-grace/production/htdemucs/production/no_vocals.wav",
    ),
    Case(
        "tyger",
        "songs/the-tyger.yaml",
        "out/the-tyger/takes/truth/production.wav",
        "out/the-tyger/takes/truth/htdemucs/production/vocals.wav",
        "out/the-tyger/takes/truth/htdemucs/production/no_vocals.wav",
    ),
    Case(
        "major-general",
        "songs/major-general.yaml",
        "out/major-general/takes/truth/production.wav",
        "out/major-general/takes/truth/htdemucs/production/vocals.wav",
        "out/major-general/takes/truth/htdemucs/production/no_vocals.wav",
    ),
]


def resolve_audio(root: Path, case: Case) -> Path:
    """Prefer the vocal stem; fall back to the mix."""
    voc = root / case.vocals
    if voc.is_file():
        return voc
    mix = root / case.mix
    if mix.is_file():
        return mix
    raise FileNotFoundError(
        f"{case.name}: missing vocals ({voc}) and mix ({mix}). "
        "Produce the truth take and/or run copyist transcribe once for stems."
    )


def select_cases(only: str | None) -> list[Case]:
    if not only:
        return list(CASES)
    want = {n.strip() for n in only.split(",") if n.strip()}
    chosen = [c for c in CASES if c.name in want]
    missing = want - {c.name for c in chosen}
    if missing:
        raise SystemExit(f"Unknown case(s): {', '.join(sorted(missing))}")
    return chosen
