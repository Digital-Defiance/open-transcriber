"""Plan and (in a GPU window) generate the SoulX mini-slice of sung probe-training material.

~120 short sung clips (1.5-2 h total) across >=4 CC BY VocalSet prompts, each phrase sung from an unseen
sampler score with SoulX-Singer, then gated note-by-note with ``copyist.pitch.check``: a clip whose sung
pitches don't match the written melody within tolerance is rejected and re-rolled. Each accepted clip writes
exact label JSON (``labels.py``) built from the score. SoulX is loaded once per invocation and many phrases
are sung in one call, so the ~8 s load cost is paid once.

This is a GPU sub-job: it runs inside the owner-approved bake-off window, not in the CPU/design pass. Without
SoulX/GPU it refuses to run and prints the plan. It holds the GPU lock, pauses on battery, is caffeinated and
resumes per clip through ``jobs.Job``.

    python -m open_transcriber.bakeoff.make_soulx_slice --dry-run   # clip count, prompts, estimated GPU hours
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from open_transcriber.data import scores

from . import jobs, labels, paths
from .dev_splits import SOULX_BASE_SEED as BASE_SEED  # single source of truth (separate block from sampler)
#: Target accepted-clip count and the estimated GPU time (SoulX ~0.5x real time + gating).
TARGET_CLIPS = 120
SECONDS_PER_CLIP_GPU = 45.0        # ~15-20 s sung audio at ~0.5x real time, plus gating overhead
#: CC BY VocalSet prompt ids used (>=4, singer-disjoint from the eval/audit splits). Provenance and the
#: CC BY 4.0 terms are recorded in the task folder; prompt audio is never committed.
VOCALSET_PROMPTS = ("vocalset-male1", "vocalset-female1", "vocalset-male2", "vocalset-female2")
#: Pitch-gate tolerance (cents) reuses copyist.pitch's default via pitch.check; a clip passes when every
#: scored note is within tolerance (octave allowed, as the transcriber scores octave separately).
MAX_REROLLS = 6


def soulx_dir() -> Path:
    return paths.copyist_root() / ".cache" / "ear" / "soulx"


def vocalset_dir() -> Path:
    """Where the CC BY 4.0 VocalSet prompt assets live (gitignored; never committed)."""
    return paths.copyist_root() / ".cache" / "ear" / "vocalset"


def prompt_assets(prompt: str, root: Path | None = None) -> tuple[Path, Path]:
    """The ``(voice_wav, voice_meta)`` for a CC BY VocalSet prompt id; raise if the assets are absent.

    The prompt audio is CC BY 4.0 and lives under ``.cache/ear/vocalset/<prompt>/`` (never committed). The
    SoulX slice is a GPU-window job; without the prompt assets it refuses to run (tests inject a fake).
    """
    base = Path(root or vocalset_dir()) / prompt
    voice_wav = base / "prompt.wav"
    voice_meta = base / "prompt.json"
    if not voice_wav.exists() or not voice_meta.exists():
        raise FileNotFoundError(
            f"VocalSet prompt {prompt!r} assets not found under {base} "
            "(CC BY 4.0; download into .cache/ear/vocalset/ for the GPU window)")
    return voice_wav, voice_meta


@dataclass
class ClipPlan:
    clip_id: str
    seed: int
    prompt: str


def plan_clips(n: int = TARGET_CLIPS, base_seed: int = BASE_SEED,
               prompts: tuple[str, ...] = VOCALSET_PROMPTS) -> list[ClipPlan]:
    """Round-robin the prompts across ``n`` unseen seeds so every prompt is used roughly equally."""
    plans = []
    for i in range(n):
        seed = base_seed + i
        prompt = prompts[i % len(prompts)]
        plans.append(ClipPlan(clip_id=f"{seed}-{prompt}", seed=seed, prompt=prompt))
    return plans


def soulx_available(find_fn: Callable | None = None) -> tuple[bool, str]:
    """Is SoulX-Singer installed and ready? Returns (ok, reason)."""
    try:
        from copyist.sing import SoulX
        finder = find_fn or SoulX.find
        finder()
        return True, "ready"
    except Exception as error:  # SongError when not configured, ImportError if deps missing
        return False, f"{type(error).__name__}: {error}"


def gate_clip(check_fn: Callable, wav: Path, notes) -> bool:
    """A clip passes the gate when every scored note's sung pitch matches the written note (octave allowed)."""
    results = check_fn(wav, notes, any_octave=True)
    return bool(results) and all(getattr(r, "ok", False) for r in results)


def estimate_hours(n: int = TARGET_CLIPS) -> float:
    return n * SECONDS_PER_CLIP_GPU / 3600.0


def make(n: int = TARGET_CLIPS, out: Path | None = None, base_seed: int = BASE_SEED,
         soulx=None, check_fn: Callable | None = None, until: str | None = None,
         assets_fn: Callable | None = None, device: str = "mps") -> dict:
    """Generate the sung slice. Requires SoulX unless ``soulx``/``check_fn`` are injected (tests).

    Each clip: load the sampled score, build sung notes with ``copyist.sing.sung_notes`` and SoulX phrase
    segments with ``copyist.sing.soulx_segments``, render **all** the clip's phrases in one ``soulx.sing``
    call (SoulX loads once per call), gate the returned ``vocal.wav`` note-by-note with ``pitch.check``,
    re-roll a failed clip up to ``MAX_REROLLS`` times, and write the accepted clip's exact label JSON.
    """
    out = Path(out or soulx_dir())
    from copyist.sing import soulx_segments, sung_notes
    from copyist.song import load_song
    if soulx is None or check_fn is None:
        from copyist import pitch
        from copyist.sing import SoulX
        if soulx is None:
            soulx = SoulX.find()
        if check_fn is None:
            check_fn = pitch.check
    assets_fn = assets_fn or prompt_assets

    scores_dir = out / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    scores.write_batch(scores_dir, n=n, start_seed=base_seed)
    plans = plan_clips(n, base_seed)

    def run(item: dict) -> dict:
        clip_dir = Path(out) / "clips" / item["clip_id"]
        clip_dir.mkdir(parents=True, exist_ok=True)
        song_path = scores_dir / f"song-{item['seed']}.yaml"
        song = load_song(song_path)
        notes = sung_notes(song)                       # SungNotes for the whole verse (from the score)
        segments = soulx_segments(notes)               # SoulX phrase segments; sung in one call
        voice_wav, voice_meta = assets_fn(item["prompt"])
        lab = labels.labels_for_song(song_path, seed=item["seed"])
        accepted = False
        for attempt in range(MAX_REROLLS):
            attempt_dir = clip_dir / f"attempt{attempt}"
            # One sing() call renders every phrase for this clip; the ~8 s model load is paid once per call.
            wav = soulx.sing(segments, voice_wav, voice_meta, out_dir=attempt_dir, device=device)
            if gate_clip(check_fn, wav, notes):
                accepted_wav = clip_dir / "vocal.wav"
                Path(wav).replace(accepted_wav)
                wav = accepted_wav
                accepted = True
                break
        if not accepted:
            raise RuntimeError(f"{item['clip_id']}: failed the pitch gate after {MAX_REROLLS} tries")
        labels.write_labels(lab, clip_dir / "labels.json")
        return {"outputs": {"vocal": wav, "labels": clip_dir / "labels.json"},
                "prompt": item["prompt"], "n_notes": len(lab.notes), "n_phrases": len(segments)}

    job = jobs.Job(name="soulx-slice", system="ear-soulx", run=run, gpu=True, device=device,
                   out=Path(out), until=until, versions=jobs.python_versions("copyist"))
    items = [{"id": p.clip_id, "clip_id": p.clip_id, "seed": p.seed, "prompt": p.prompt} for p in plans]
    return job.run_queue(items)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n", type=int, default=TARGET_CLIPS)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--until", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    hours = estimate_hours(args.n)
    print(f"plan: {args.n} sung clips over {len(VOCALSET_PROMPTS)} CC BY VocalSet prompts "
          f"({', '.join(VOCALSET_PROMPTS)})")
    print(f"      seeds {args.base_seed}..{args.base_seed + args.n - 1}; gated note-by-note with pitch.check "
          f"(re-roll up to {MAX_REROLLS}x)")
    print(f"      estimated GPU time ~{hours:.1f} h; output under {args.out or soulx_dir()}")
    ok, reason = soulx_available()
    print(f"      SoulX: {'ready' if ok else 'NOT available (' + reason + ')'}")
    if args.dry_run:
        return 0
    if not ok:
        print("Refusing to run: this is a GPU sub-job for the approved window; SoulX/GPU is not available here.",
              file=sys.stderr)
        return 1
    counts = make(args.n, out=args.out, base_seed=args.base_seed, until=args.until)
    print(f"done: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
