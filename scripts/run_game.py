#!/usr/bin/env python3
"""Run GAME over the dev sets and land its notes on the scoreboard. Testing only, never on owner songs.

Six systems: the three model sizes on the full mixture and on the Demucs vocal stem. For each config this:

1. stages the dev audio as ``<id>.wav`` symlinks in ``out/judge/game-run/<config>/in`` (GAME names its output
   after the input stem, so the stem must be the item id);
2. runs ``infer.py extract`` once over that directory (one model load per config), writing ``<id>.txt``;
3. converts each ``<id>.txt`` to ``out/judge/systems/<config>/<id>.tsv`` (the board drop-in format);
4. records wall-clock seconds and seconds-per-song in ``out/judge/systems/game-timings.json``.

Resumable: a config whose every item TSV already exists is skipped. GAME's own ``extract`` reprocesses the
whole input directory each call, so a half-finished config re-runs from scratch (its model load dominates).

``--smoke`` is the CPU pre-run the GPU rules require: it stages the inputs, checks the audio and model files
resolve, converts a tiny synthetic GAME ``.txt`` through the real converter onto the board, and exits without
touching the model or the GPU.

Use ``.cache/game/env/bin/python`` as the interpreter (torch + MPS). The copyist checkout is found via
``$COPYIST_ROOT`` or the submodule's location.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Make ``open_transcriber`` importable when run as a loose script from the GAME env.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import paths  # noqa: E402
from open_transcriber.board import board, game  # noqa: E402


def dev_items() -> list[str]:
    """All 42 dev items: RWC-other-dev then karaoke-dev, in split order."""
    return board.set_members("rwc-other-dev") + board.set_members("karaoke-dev")


def stage_inputs(config: str, items: list[str], source: str, data: Path, out: Path) -> Path:
    """Symlink each item's source audio as ``<id>.wav`` in the config's input dir; return that dir."""
    in_dir = out / "game-run" / config / "in"
    in_dir.mkdir(parents=True, exist_ok=True)
    for item in items:
        audio = game.source_path(item, source, data, out)
        if not audio.exists():
            raise FileNotFoundError(f"missing {source} audio for {item}: {audio}")
        link = in_dir / f"{item}.wav"
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(audio.resolve())
    return in_dir


def config_done(config: str, items: list[str], out: Path) -> bool:
    """True if every item already has a converted TSV for this config."""
    system_dir = out / "systems" / config
    return all((system_dir / f"{item}.tsv").exists() for item in items)


def run_game(game_dir: Path, python: Path, in_dir: Path, model: Path, raw_dir: Path,
             device: str | None = None) -> None:
    """Call ``infer.py extract`` once over ``in_dir``, writing integer-MIDI ``.txt`` into ``raw_dir``."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python), "infer.py", "extract", str(in_dir),
        "-m", str(model),
        "--glob", "*.wav",
        "--output-formats", "txt",
        "--pitch-format", "number", "--round-pitch",
        "--output-dir", str(raw_dir),
    ]
    env = dict(os.environ)
    if device:
        env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    subprocess.run(cmd, cwd=str(game_dir), check=True, env=env)


def smoke(data: Path, out: Path) -> None:
    """CPU pre-run: resolve every input and model, then convert a synthetic GAME txt onto the board."""
    items = dev_items()
    for size in game.SIZES:
        model = game.model_path(size, data)
        if not model.exists():
            raise FileNotFoundError(f"GAME model not found: {model}")
    for source in game.SOURCES:
        for item in items:
            audio = game.source_path(item, source, data, out)
            if not audio.exists():
                raise FileNotFoundError(f"missing {source} audio for {item}: {audio}")
    # Convert a tiny synthetic GAME output through the real converter and score it against itself.
    tmp = out / "game-run" / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "probe.txt").write_text("0.100\t0.400\t60\n0.500\t0.700\t67\n", encoding="utf-8")
    rows = game.convert_txt(tmp / "probe.txt")
    dest = tmp / "probe.tsv"
    game.write_system_tsv(rows, dest)
    loaded = board._tsv_loader(dest)()
    assert len(loaded) == 2 and list(loaded.midi) == [60.0, 67.0], "converter round-trip failed"
    print(f"smoke OK: {len(items)} dev items resolve for mix and stem; "
          f"{len(game.SIZES)} GAME models present; converter round-trips.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU pre-run only; never loads the model")
    parser.add_argument("--sizes", default=",".join(game.SIZES),
                        help="comma-separated GAME sizes to run (default: all)")
    parser.add_argument("--sources", default=",".join(game.SOURCES),
                        help="comma-separated sources to run: mix,stem (default: both)")
    parser.add_argument("--device", default="mps", help="torch device hint passed through to GAME")
    parser.add_argument("--data", type=Path, default=None, help="judge data folder (default: .cache/judge)")
    parser.add_argument("--out", type=Path, default=None, help="judge output folder (default: out/judge)")
    args = parser.parse_args(argv)

    data = args.data or paths.data_dir()
    out = args.out or paths.out_dir()

    if args.smoke:
        smoke(data, out)
        return 0

    game_dir = data.parent / "game" / "GAME"
    python = data.parent / "game" / "env" / "bin" / "python"
    items = dev_items()
    sizes = [s for s in args.sizes.split(",") if s]
    sources = [s for s in args.sources.split(",") if s]

    timings = game.load_timings(out)
    for size in sizes:
        model = game.model_path(size, data)
        for source in sources:
            config = game.config_name(size, source)
            if config_done(config, items, out):
                print(f"[skip] {config}: all {len(items)} TSVs present")
                continue
            print(f"[run ] {config}: {len(items)} songs")
            in_dir = stage_inputs(config, items, source, data, out)
            raw_dir = out / "game-run" / config / "raw"
            t0 = time.monotonic()
            run_game(game_dir, python, in_dir, model, raw_dir, device=args.device)
            elapsed = time.monotonic() - t0
            done = game.convert_dir(raw_dir, out / "systems" / config, items)
            if len(done) != len(items):
                missing = sorted(set(items) - set(done))
                print(f"[warn] {config}: {len(done)}/{len(items)} converted; missing {missing[:5]}...")
            timings[config] = game.Timing(config, len(done), elapsed)
            game.save_timings(timings, out)
            print(f"[done] {config}: {len(done)} songs in {elapsed:.1f}s "
                  f"({timings[config].seconds_per_song:.2f}s/song)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
