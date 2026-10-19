"""Truth-take cases shared with Copyist's scripts/transcribe/roundtrip.py.

Paths are relative to a Copyist checkout root. Kept here so T0 can run without
importing the copyist package (only evaluate.py is loaded for scoring).
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    name: str
    song: str          # relative to copyist root
    mix: str
    vocals: str
    accompaniment: str


# The round-trip audio cases for T0 (mix/stem paths). The held-out *song* set is
# not this list: held_out_songs() derives it at runtime from songs/*.yaml plus the
# roundtrip CASES, so the two big songs added later are picked up automatically.
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


def _roundtrip_case_songs(root: Path) -> list[Path]:
    """Song paths from copyist's scripts/transcribe/roundtrip.py CASES.

    roundtrip.py inserts its ROOT on sys.path itself, so importing it by file
    path is enough. Returns [] if the import fails (e.g. copyist not importable).
    """
    rt_path = root / "scripts/transcribe/roundtrip.py"
    if not rt_path.is_file():
        return []
    mod_name = "copyist_roundtrip_cases"
    try:
        spec = importlib.util.spec_from_file_location(mod_name, rt_path)
        if spec is None or spec.loader is None:
            return []
        mod = importlib.util.module_from_spec(spec)
        # roundtrip.py defines its Case via @dataclass; dataclasses resolves the
        # defining module through sys.modules, so the module must be registered
        # there before exec_module or the decorator raises. Register, then clean up.
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(mod_name, None)
    except Exception:
        return []
    songs = []
    for case in getattr(mod, "CASES", []):
        song = getattr(case, "song", None)
        if song is not None:
            songs.append(Path(song))
    return songs


def held_out_songs(root: Path) -> list[Path]:
    """The held-out song YAML paths, derived at runtime (never a hard-coded list).

    De-duplicated union of songs/*.yaml and every song referenced by roundtrip.py's
    CASES. Falls back to the songs/ glob alone if the roundtrip import fails. This is
    why the two big songs added to songs/ (and CASES) later need no edit here.
    """
    seen: dict[str, Path] = {}
    for path in sorted((root / "songs").glob("*.yaml")):
        seen[str(path.resolve())] = path
    for path in _roundtrip_case_songs(root):
        key = str(path.resolve())
        if key not in seen:
            seen[key] = path
    return list(seen.values())


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
