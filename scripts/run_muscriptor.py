#!/usr/bin/env python3
"""Run MuScriptor over the dev sets and land its melody line on the scoreboard. Testing only.

Six systems: the three model sizes (small, medium, large) on the full mixture and on the Demucs vocal stem.
MuScriptor's CLI transcribes one file at a time, so for each config this:

1. for each dev item, runs ``muscriptor transcribe <audio> --output <raw>/<id>.mid`` (one call per song, the
   model is re-loaded each call -- MuScriptor has no batch mode), skipping songs already transcribed;
2. extracts the melody line from each MIDI and writes ``out/judge/systems/muscriptor-<size>-<mix|stem>/<id>.tsv``
   (the board drop-in format), so the scoreboard picks it up with no code change;
3. records wall-clock seconds and seconds-per-song in ``out/judge/systems/muscriptor-timings.json``.

Resumable at the song level: an item whose MIDI already exists is not re-transcribed, and a config whose every
TSV exists is skipped.

**Hugging Face access is required.** MuScriptor's official weights are gated under CC BY-NC 4.0: the owner must
accept the licence on the model page and authenticate (``hf auth login`` or ``HF_TOKEN``). Without a token the
run stops with that message rather than guessing. MuScriptor is a yardstick only: never on the owner's songs,
never shipped.

``--smoke`` is the CPU pre-run the GPU rules require: it resolves every dev input, checks the MuScriptor CLI is
importable, and converts a MIDI synthesised in-process through the real converter onto the board, without
loading the model, downloading weights, or touching the GPU.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

# Make ``open_transcriber`` importable when run as a loose script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from open_transcriber.bakeoff import paths  # noqa: E402
from open_transcriber.board import board, muscriptor  # noqa: E402


def dev_items() -> list[str]:
    """All 42 dev items: RWC-other-dev then karaoke-dev, in split order."""
    return board.set_members("rwc-other-dev") + board.set_members("karaoke-dev")


def config_done(config: str, items: list[str], out: Path) -> bool:
    """True if every item already has a converted TSV for this config."""
    system_dir = out / "systems" / config
    return all((system_dir / f"{item}.tsv").exists() for item in items)


def muscriptor_cmd(size: str) -> list[str]:
    """The base MuScriptor invocation: a local install if present, else ``uvx muscriptor``.

    ``$MUSCRIPTOR`` overrides the executable. On Apple Silicon ``uvx muscriptor`` runs on MPS automatically.
    """
    exe = os.environ.get("MUSCRIPTOR")
    if exe:
        base = exe.split()
    elif shutil.which("muscriptor"):
        base = ["muscriptor"]
    else:
        base = ["uvx", "muscriptor"]
    return base


def transcribe_one(size: str, audio: Path, dest_midi: Path) -> None:
    """Transcribe one file to a single MIDI with ``--fixed-tempo`` (we only need note times, not notation)."""
    dest_midi.parent.mkdir(parents=True, exist_ok=True)
    cmd = muscriptor_cmd(size) + [
        "transcribe", str(audio),
        "--model", size,
        "--format", "midi",
        "--output", str(dest_midi),
    ]
    subprocess.run(cmd, check=True)


def have_hf_token() -> bool:
    """Whether a Hugging Face token is available for the gated MuScriptor weights."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    token = Path.home() / ".cache" / "huggingface" / "token"
    return token.exists() and token.read_text(encoding="utf-8").strip() != ""


def _write_probe_midi(path: Path) -> None:
    """Write a tiny two-note type-0 SMF (480 tpb, 120 BPM) the real reader must parse: C4 then G4."""
    def vlq(n: int) -> bytes:
        out = bytearray([n & 0x7F])
        n >>= 7
        while n:
            out.insert(0, (n & 0x7F) | 0x80)
            n >>= 7
        return bytes(out)

    tpb = 480
    events = bytearray()
    events += vlq(0) + bytes([0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")  # 120 BPM
    events += vlq(0) + bytes([0x90, 60, 100])        # C4 on at tick 0
    events += vlq(tpb) + bytes([0x80, 60, 0])        # C4 off at tick 480 (0.5 s)
    events += vlq(0) + bytes([0x90, 67, 90])         # G4 on
    events += vlq(tpb) + bytes([0x80, 67, 0])        # G4 off (0.5 s)
    events += vlq(0) + bytes([0xFF, 0x2F, 0x00])     # end of track
    track = b"MTrk" + struct.pack(">I", len(events)) + bytes(events)
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpb)
    path.write_bytes(header + track)


def smoke(data: Path, out: Path) -> None:
    """CPU pre-run: resolve every dev input and round-trip a synthetic MIDI through the real converter."""
    items = dev_items()
    for source in muscriptor.SOURCES:
        for item in items:
            audio = muscriptor.source_path(item, source, data, out)
            if not audio.exists():
                raise FileNotFoundError(f"missing {source} audio for {item}: {audio}")
    tmp = out / "muscriptor-run" / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    probe = tmp / "probe.mid"
    _write_probe_midi(probe)
    rows = muscriptor.convert_midi(probe, tmp / "probe.tsv")
    assert [r[2] for r in rows] == [60, 67], f"converter round-trip failed: {rows}"
    assert abs(rows[0][1] - rows[0][0] - 0.5) < 1e-3, f"tempo map off: {rows}"
    loaded = board._tsv_loader(tmp / "probe.tsv")()
    assert len(loaded) == 2 and list(loaded.midi) == [60.0, 67.0], "board loader round-trip failed"
    token = "present" if have_hf_token() else "MISSING (owner must accept the CC BY-NC 4.0 licence)"
    print(f"smoke OK: {len(items)} dev items resolve for mix and stem; MIDI converter round-trips; "
          f"HF token {token}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU pre-run only; never loads the model")
    parser.add_argument("--sizes", default=",".join(muscriptor.SIZES),
                        help="comma-separated MuScriptor sizes to run (default: all)")
    parser.add_argument("--sources", default=",".join(muscriptor.SOURCES),
                        help="comma-separated sources to run: mix,stem (default: both)")
    parser.add_argument("--data", type=Path, default=None, help="judge data folder (default: .cache/judge)")
    parser.add_argument("--out", type=Path, default=None, help="judge output folder (default: out/judge)")
    args = parser.parse_args(argv)

    data = args.data or paths.data_dir()
    out = args.out or paths.out_dir()

    if args.smoke:
        smoke(data, out)
        return 0

    if not have_hf_token():
        print("MuScriptor's weights are gated (CC BY-NC 4.0). Accept the licence on the Hugging Face model "
              "page and set HF_TOKEN (or run `hf auth login`) before this run. Testing only.", file=sys.stderr)
        return 2

    items = dev_items()
    sizes = [s for s in args.sizes.split(",") if s]
    sources = [s for s in args.sources.split(",") if s]

    timings = muscriptor.load_timings(out, name="muscriptor-timings.json")
    for size in sizes:
        for source in sources:
            config = muscriptor.config_name(size, source)
            if config_done(config, items, out):
                print(f"[skip] {config}: all {len(items)} TSVs present")
                continue
            print(f"[run ] {config}: {len(items)} songs")
            raw_dir = out / "muscriptor-run" / config / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            for item in items:
                dest_midi = raw_dir / f"{item}.mid"
                if dest_midi.exists():
                    continue
                audio = muscriptor.source_path(item, source, data, out)
                if not audio.exists():
                    print(f"[warn] {config}: missing {source} audio for {item}: {audio}")
                    continue
                transcribe_one(size, audio, dest_midi)
            elapsed = time.monotonic() - t0
            done = muscriptor.convert_dir(raw_dir, out / "systems" / config, items)
            if len(done) != len(items):
                missing = sorted(set(items) - set(done))
                print(f"[warn] {config}: {len(done)}/{len(items)} converted; missing {missing[:5]}...")
            timings[config] = muscriptor.Timing(config, len(done), elapsed)
            muscriptor.save_timings(timings, out, name="muscriptor-timings.json")
            print(f"[done] {config}: {len(done)} songs in {elapsed:.1f}s "
                  f"({timings[config].seconds_per_song:.2f}s/song)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
