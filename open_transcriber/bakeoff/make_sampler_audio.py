"""Render sampler-audio probe-training material (CPU-only, deterministic, resumable).

For the bake-off we need audio with exact labels that is never RWC or NC karaoke. This generates unseen
sampler scores (new seed block, the existing held-out denylist, never a ``songs/`` tune), each 30-45 s, and
renders deterministic sfizz/mix variants of each with Copyist's own band renderer. Every clip writes its exact
label JSON (``labels.py``) alongside, built by construction from the score.

Output goes under ``.cache/ear/sampler/`` (gitignored). The job runs through the resumable
``jobs.Job`` runner so a long CPU render checkpoints, resumes and respects the disk floor. Nothing here is a
GPU job. The script prints an estimate and may run as an announced unattended CPU job (>15 min); this pass
only runs a tiny smoke batch.

    python -m open_transcriber.bakeoff.make_sampler_audio --dry-run         # plan + estimated hours
    python -m open_transcriber.bakeoff.make_sampler_audio --n 2 --variants 1 # a tiny real render
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from open_transcriber.data import scores

from . import jobs, labels, paths
from .dev_splits import SAMPLER_BASE_SEED as BASE_SEED  # single source of truth; chosen not to overlap
# the existing .cache/t1/scores seeds (those start at 0). Recorded here and in the report.
#: Deterministic render variants: (style, perform-seed). Styles are chosen to vary timbre/mix; a None style
#: lets the arranger pick its default. The perform seed makes the expressive performance deterministic.
VARIANTS: tuple[tuple[str | None, int], ...] = (
    (None, 20261020),
    ("folk", 20261021),
    ("waltz", 20261022),
)
#: Rough wall-time estimate per rendered variant on the M4 Max (CPU render, sfizz or SoundFont fallback).
SECONDS_PER_VARIANT = 25.0


def sampler_dir() -> Path:
    return paths.copyist_root() / ".cache" / "ear" / "sampler"


def unseen_seeds(n: int, base: int = BASE_SEED) -> list[int]:
    """``n`` consecutive seeds from the bake-off block."""
    return list(range(base, base + n))


@dataclass
class RenderResult:
    wav: Path
    labels_json: Path


def _render_one(song_path: Path, out_dir: Path, style: str | None, seed: int,
                cache: Path, render_fn: Callable | None = None) -> RenderResult:
    """Build the arrangement MIDI, perform it (seeded), render the band, and return the outputs.

    ``render_fn`` is injected in tests so the smoke path needs no instruments; by default it uses Copyist's
    real build path (arrange -> perform -> render_band).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / "audio.wav"
    if render_fn is not None:
        render_fn(song_path, wav, style, seed)
    else:  # pragma: no cover - the real render path needs instruments/SoundFont, exercised only unattended
        from copyist import arrange, engrave, perform, render
        from copyist.song import load_song
        song = load_song(song_path)
        band = arrange.arrange_band(song)
        midi = engrave.write_midi(band.performance(), out_dir / "band.mid")
        perform.perform(midi, midi, song.meter, float(perform.pickup_of(song)), seed=seed)
        render.render_band(midi, wav, band.style if style is None else style, cache)
    return RenderResult(wav=wav, labels_json=out_dir / "labels.json")


def make(n: int, variants: int, out: Path | None = None, cache: Path | None = None,
         base_seed: int = BASE_SEED, render_fn: Callable | None = None,
         until: str | None = None) -> dict:
    """Generate ``n`` unseen sampler scores and render ``variants`` deterministic variants of each.

    Returns the runner's counts. Resumable: a variant whose ``done.json`` exists is skipped.
    """
    out = Path(out or sampler_dir())
    cache = Path(cache or (paths.copyist_root() / ".cache"))
    chosen = VARIANTS[:variants]

    # Write the N unseen scores to a scores folder (write_batch enforces the held-out denylist and validators).
    scores_dir = out / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores.write_batch(scores_dir, n=n, start_seed=base_seed)

    items = []
    for seed in unseen_seeds(n, base_seed):
        song_path = scores_dir / f"song-{seed}.yaml"
        if not song_path.exists():
            continue  # a seed that failed validation/denylist was re-rolled to a different file name
        for vi, (style, perform_seed) in enumerate(chosen):
            items.append({"id": f"{seed}-v{vi}", "seed": seed, "song": str(song_path),
                          "style": style, "perform_seed": perform_seed})

    def run(item: dict) -> dict:
        item_dir = Path(out) / "clips" / item["id"]
        result = _render_one(Path(item["song"]), item_dir, item["style"], item["perform_seed"], cache, render_fn)
        lab = labels.labels_for_song(item["song"], seed=item["seed"])
        labels.write_labels(lab, result.labels_json)
        return {"outputs": {"wav": result.wav, "labels": result.labels_json},
                "n_notes": len(lab.notes), "duration": lab.duration}

    job = jobs.Job(name="sampler-audio", system="ear-sampler", run=run, gpu=False, device="cpu",
                   out=Path(out), until=until, versions=jobs.python_versions("copyist"))
    return job.run_queue(items)


def estimate_hours(n: int, variants: int) -> float:
    return n * min(variants, len(VARIANTS)) * SECONDS_PER_VARIANT / 3600.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=400, help="number of unseen sampler scores")
    parser.add_argument("--variants", type=int, default=len(VARIANTS), help="render variants per score (<=3)")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--until", type=str, default=None, help="stop cleanly at HH:MM local")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and estimated hours, render nothing")
    args = parser.parse_args(argv)

    n, variants = args.n, min(args.variants, len(VARIANTS))
    hours = estimate_hours(n, variants)
    print(f"plan: {n} unseen sampler scores (seeds {args.base_seed}..{args.base_seed + n - 1}), "
          f"{variants} variant(s) each = {n * variants} clips")
    print(f"      30-45 s each; estimated CPU wall time ~{hours:.1f} h; output under {args.out or sampler_dir()}")
    print(f"      variants: {', '.join(str(v) for v in VARIANTS[:variants])}")
    print("      held-out denylist enforced by scores.write_batch; no songs/ tune; no GPU.")
    if args.dry_run:
        return 0
    counts = make(n, variants, out=args.out, base_seed=args.base_seed, until=args.until)
    print(f"done: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
