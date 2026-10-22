#!/usr/bin/env python3
"""Run MuScriptor over the dev sets and land its melody line on the scoreboard. Testing only.

The owner chose the non-gated GGUF mirror (``DamRsn/muscriptor-gguf``, CC BY-NC 4.0) run through
``muscriptor.cpp``'s benchmark, so this needs no Hugging Face account. Six systems: the three model sizes
(small, medium, large) on the full mixture and on the Demucs vocal stem. For each config this:

1. for each dev item, resamples its audio to 16 kHz mono float32 (``ffmpeg``) and runs
   ``muscriptor_bench --transcribe --weights <size>.gguf --device <dev> --audio <wav>
   --notes-out <raw>/<id>.notes.tsv`` (the bench loads the model each call; MuScriptor has no batch mode),
   skipping items already transcribed;
2. reduces each notes-TSV to one melody line and writes ``out/judge/systems/muscriptor-<size>-<mix|stem>/<id>.tsv``
   (the board drop-in format), so the scoreboard picks it up with no code change;
3. records wall-clock seconds and seconds-per-song in ``out/judge/systems/muscriptor-timings.json``.

Resumable at the song level: an item whose notes-TSV already exists is not re-transcribed, and a config whose
every board TSV exists is skipped.

``muscriptor.cpp`` is expected under ``.cache/muscriptor/muscriptor.cpp`` (built bench at
``cpp/build/bench/muscriptor_bench``, GGUF weights in ``testdata/weights/muscriptor-<size>-f16.gguf``), or set
``$MUSCRIPTOR_CPP`` to its root. MuScriptor is a yardstick only: never on the owner's songs, never shipped.

``--smoke`` is the CPU pre-run the GPU rules require: it resolves every dev input, checks the bench binary and
the GGUF weights exist, and round-trips a synthetic notes-TSV through the real melody reduction and board
loader, without loading the model or touching the GPU.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import argparse
import os
import shutil
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
    """True if every item already has a converted board TSV for this config."""
    system_dir = out / "systems" / config
    return all((system_dir / f"{item}.tsv").exists() for item in items)


def muscriptor_cpp_root() -> Path:
    """The muscriptor.cpp checkout: ``$MUSCRIPTOR_CPP`` else ``.cache/muscriptor/muscriptor.cpp``."""
    env = os.environ.get("MUSCRIPTOR_CPP")
    if env:
        return Path(env)
    return paths.copyist_root() / ".cache" / "muscriptor" / "muscriptor.cpp"


def bench_path(root: Path) -> Path:
    return root / "cpp" / "build" / "bench" / "muscriptor_bench"


def weights_path(root: Path, size: str) -> Path:
    return root / "testdata" / "weights" / f"muscriptor-{size}-f16.gguf"


def resample(audio: Path, dest_wav: Path) -> None:
    """Resample ``audio`` to 16 kHz mono float32 WAV with ffmpeg (the bench's required input)."""
    dest_wav.parent.mkdir(parents=True, exist_ok=True)
    # -nostdin: detached (nohup) ffmpeg otherwise reads stdin and blocks forever.
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(audio),
         "-ac", "1", "-ar", "16000", "-c:a", "pcm_f32le", str(dest_wav)],
        check=True, stdin=subprocess.DEVNULL)


def transcribe_one(bench: Path, weights: Path, wav: Path, notes_out: Path, device: str) -> None:
    """Run the bench on one wav, writing non-drum notes to ``notes_out``."""
    notes_out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(bench), "--transcribe", "--weights", str(weights),
           "--device", device, "--audio", str(wav), "--notes-out", str(notes_out)]
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def on_battery() -> bool:
    """True if the Mac is on battery (so a GPU config should wait for AC)."""
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=True).stdout
    except Exception:
        return False
    return "Battery Power" in out


def smoke(data: Path, out: Path) -> None:
    """CPU pre-run: resolve every dev input and round-trip a synthetic notes-TSV through the real reduction."""
    items = dev_items()
    for source in muscriptor.SOURCES:
        for item in items:
            audio = muscriptor.source_path(item, source, data, out)
            if not audio.exists():
                raise FileNotFoundError(f"missing {source} audio for {item}: {audio}")
    root = muscriptor_cpp_root()
    bench = bench_path(root)
    present = {size: weights_path(root, size).exists() for size in muscriptor.SIZES}
    tmp = out / "muscriptor-run" / "_smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    # Two sustained melody notes with a short overlapping accompaniment hit the reduction must drop.
    probe = tmp / "probe.notes.tsv"
    probe.write_text("0.000\t0.500\t60\n0.100\t0.180\t72\n0.600\t1.100\t67\n", encoding="utf-8")
    rows = muscriptor.convert_notes_tsv(probe, tmp / "probe.tsv")
    assert [r[2] for r in rows] == [60, 67], f"melody reduction failed: {rows}"
    loaded = board._tsv_loader(tmp / "probe.tsv")()
    assert len(loaded) == 2 and list(loaded.midi) == [60.0, 67.0], "board loader round-trip failed"
    bench_ok = "present" if bench.exists() else f"MISSING (build muscriptor.cpp: {bench})"
    weights_ok = ",".join(f"{s}:{'ok' if present[s] else 'MISSING'}" for s in muscriptor.SIZES)
    print(f"smoke OK: {len(items)} dev items resolve for mix and stem; melody reduction round-trips; "
          f"bench {bench_ok}; weights {weights_ok}.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true", help="CPU pre-run only; never loads the model")
    parser.add_argument("--sizes", default=",".join(muscriptor.SIZES),
                        help="comma-separated MuScriptor sizes to run (default: all)")
    parser.add_argument("--sources", default=",".join(muscriptor.SOURCES),
                        help="comma-separated sources to run: mix,stem (default: both)")
    parser.add_argument("--device", default="gpu", help="bench device: cpu, gpu, auto, or an index (default: gpu)")
    parser.add_argument("--data", type=Path, default=None, help="judge data folder (default: .cache/judge)")
    parser.add_argument("--out", type=Path, default=None, help="judge output folder (default: out/judge)")
    args = parser.parse_args(argv)

    data = args.data or paths.data_dir()
    out = args.out or paths.out_dir()

    if args.smoke:
        smoke(data, out)
        return 0

    root = muscriptor_cpp_root()
    bench = bench_path(root)
    if not bench.exists():
        print(f"muscriptor.cpp bench not built: {bench}. See .cache/muscriptor/muscriptor.cpp "
              "(cmake -S cpp -B cpp/build -G Ninja -DMUSCRIPTOR_METAL_PRECOMPILED=OFF && cmake --build cpp/build).",
              file=sys.stderr)
        return 2

    items = dev_items()
    sizes = [s for s in args.sizes.split(",") if s]
    sources = [s for s in args.sources.split(",") if s]

    timings = muscriptor.load_timings(out, name="muscriptor-timings.json")
    for size in sizes:
        weights = weights_path(root, size)
        if not weights.exists():
            print(f"[skip] size {size}: weights missing ({weights}); fetch with `uv run msl-fetch --size {size}`")
            continue
        for source in sources:
            config = muscriptor.config_name(size, source)
            if config_done(config, items, out):
                print(f"[skip] {config}: all {len(items)} TSVs present")
                continue
            # A GPU config waits for AC power (the grant is for mains only).
            if args.device not in ("cpu",):
                while on_battery():
                    print(f"[wait] {config}: on battery; waiting for AC before a GPU run", flush=True)
                    time.sleep(30)
            print(f"[run ] {config}: {len(items)} songs on device {args.device}")
            raw_dir = out / "muscriptor-run" / config / "raw"
            wav_dir = out / "muscriptor-run" / config / "wav"
            raw_dir.mkdir(parents=True, exist_ok=True)
            t0 = time.monotonic()
            for item in items:
                notes_out = raw_dir / f"{item}.notes.tsv"
                if notes_out.exists():
                    continue
                audio = muscriptor.source_path(item, source, data, out)
                if not audio.exists():
                    print(f"[warn] {config}: missing {source} audio for {item}: {audio}")
                    continue
                wav = wav_dir / f"{item}.wav"
                resample(audio, wav)
                transcribe_one(bench, weights, wav, notes_out, args.device)
                wav.unlink(missing_ok=True)  # the 16 kHz copy is disposable
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
