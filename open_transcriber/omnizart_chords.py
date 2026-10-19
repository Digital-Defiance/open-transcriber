"""Omnizart chord transcription (MIT).

Install: `bash scripts/setup_omnizart.sh` then `omnizart download-checkpoints`.
Returns (start_sec, end_sec, symbol) rows for Copyist / round-trip chord scoring.
"""
from __future__ import annotations

from pathlib import Path


def chords_from_audio(audio: Path, out_dir: Path | None = None) -> list[tuple[float, float, str]]:
    """Transcribe chords; raises ImportError if Omnizart isn't installed."""
    from omnizart.chord import ChordTranscription

    audio = Path(audio)
    out = Path(out_dir) if out_dir else audio.parent / "omnizart-chords"
    out.mkdir(parents=True, exist_ok=True)
    ChordTranscription().transcribe(str(audio), model_path=None, output=str(out))
    csvs = sorted(out.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return []
    rows: list[tuple[float, float, str]] = []
    for line in csvs[0].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.lower().startswith("time") or line.lower().startswith("start"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 3:
            continue
        try:
            start, end, sym = float(parts[0]), float(parts[1]), parts[2]
        except ValueError:
            continue
        if sym and sym.upper() not in ("N", "NONE", "X"):
            rows.append((start, end, sym))
    return rows
