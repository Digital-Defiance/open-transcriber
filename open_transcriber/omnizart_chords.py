"""Omnizart chord transcription (MIT).

Install: `bash scripts/setup_omnizart.sh` then `omnizart download-checkpoints`.
Returns (start_sec, end_sec, symbol) rows for Copyist / round-trip chord scoring.
"""
from __future__ import annotations

from pathlib import Path


def _normalize_symbol(sym: str) -> str:
    """e.g. C:maj -> C, E:min -> Em, G:7 -> G7."""
    if ":" in sym:
        root, qual = sym.split(":", 1)
        if qual == "maj":
            return root
        if qual == "min":
            return root + "m"
        return root + qual
    return sym


def chords_from_audio(audio: Path, out_dir: Path | None = None) -> list[tuple[float, float, str]]:
    """Transcribe chords; raises ImportError if Omnizart isn't installed."""
    from omnizart.chord import ChordTranscription

    audio = Path(audio)
    out = Path(out_dir) if out_dir else audio.parent / "omnizart-chords"
    out.mkdir(parents=True, exist_ok=True)
    ChordTranscription().transcribe(str(audio), model_path=None, output=str(out))
    # Omnizart writes <out>/<audio.stem> or <audio.stem>.csv along with .mid.
    cands = [p for p in out.glob("*") if p.is_file() and p.suffix != ".mid"]
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        return []
    rows: list[tuple[float, float, str]] = []
    lines = cands[0].read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].strip().lower().replace(";", ",")
    chord_first = header.startswith("chord")
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 3:
            continue
        try:
            if chord_first:
                sym, start, end = parts[0], float(parts[1]), float(parts[2])
            else:
                start, end, sym = float(parts[0]), float(parts[1]), parts[2]
        except ValueError:
            continue
        if sym and sym.upper() not in ("N", "NONE", "X"):
            rows.append((start, end, _normalize_symbol(sym)))
    return rows
