"""The unattended GPU bake-off batch: feature extraction, the SoulX mini-slice, then the 12 probe runs.

This is a GPU-window job. It is written and unit-tested on CPU (with fakes), but it runs only on the owner's
Mac inside an approved window. **Hard gates checked before any GPU work** (non-negotiable):

1. the ``ear-prereg`` tag exists and is an ancestor of HEAD (splits + rule frozen);
2. every encoder's licence is confirmed free/allowed in ``licences.md`` (fail closed).

Then, one GPU job at a time, in this fixed order:

1. per-encoder feature extraction over the dev+train audio, each extractor calibrated on ~10 min first and
   **dropped** if projected extraction > 3 h, > 48 GB, or unsupported on MPS without a clear pilot lead;
2. the SoulX mini-slice (~120 clips);
3. the 12 probe runs (4 encoders x 3 seeds), capped ~45 min each.

The whole batch: holds ``caffeinate -i`` and the ``DATA/gpu.lock`` for its lifetime; checks ``pmset -g batt``
before each item and pauses on battery; resumes from per-item/per-epoch checkpoints; caps the whole batch at
~10 h wall time and checkpoints cleanly at the cap or window end; stops below 150 GB free; downloads only
licence-confirmed encoders. It writes per-job progress to ``runs.jsonl`` and a final results JSON, then calls
``decide.py`` to compute the GO/NO-GO verdict from the pre-registered rule.

    python -m open_transcriber.bakeoff.run_bakeoff --dry-run   # ordered jobs, hours, disk, gate status
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import paths
from .probe.train import SEEDS

PREREG_TAG = "ear-prereg"
#: The music encoders, in a fixed order; "control" (Basic Pitch only) is always included.
ENCODERS = ("musicfm", "dasheng-base", "heartcodec", "whisper-base-en")
#: Per-extractor calibration limits (prereg compute rule).
MAX_EXTRACT_HOURS = 3.0
MAX_EXTRACT_GB = 48.0
#: Per-probe cap and the whole-batch cap.
PROBE_CAP_MINUTES = 45.0
BATCH_CAP_HOURS = 10.0
#: Rough estimates for the dry-run plan (GPU hours).
EST_EXTRACT_HOURS = 7.0
EST_SOULX_HOURS = 1.5
EST_PROBE_HOURS = 6.0
EST_TEMP_DISK_GB = (15.0, 30.0)


def _git(*args: str) -> subprocess.CompletedProcess:
    root = paths.copyist_root() / "open-transcriber"
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)


def prereg_tag_ok() -> tuple[bool, str]:
    """True when the ear-prereg tag exists and is an ancestor of HEAD."""
    if _git("rev-parse", "-q", "--verify", f"refs/tags/{PREREG_TAG}").returncode != 0:
        return False, f"no {PREREG_TAG} tag"
    if _git("merge-base", "--is-ancestor", PREREG_TAG, "HEAD").returncode != 0:
        return False, f"{PREREG_TAG} is not an ancestor of HEAD"
    return True, "ok"


def licences_confirmed(licences_md: Path | None = None) -> tuple[bool, str]:
    """True when the licence findings file marks every tested encoder free/allowed (fail closed)."""
    path = licences_md or (paths.copyist_root() / ".agents" / "tasks" / "ear-bakeoff-2026-10-20" / "licences.md")
    if not Path(path).exists():
        return False, "licences.md not found; confirm encoder licences before any download"
    text = Path(path).read_text(encoding="utf-8").lower()
    for enc in ("musicfm", "dasheng", "heartcodec", "whisper", "basic pitch"):
        if enc not in text:
            return False, f"licence for {enc} not recorded"
    if "nc/nd" in text or "excluded on licence" in text:
        return False, "a tested encoder is NC/ND or excluded; fail closed"
    return True, "ok"


def gates_ok(licences_md: Path | None = None) -> tuple[bool, list[str]]:
    reasons = []
    tag_ok, tag_why = prereg_tag_ok()
    lic_ok, lic_why = licences_confirmed(licences_md)
    if not tag_ok:
        reasons.append(f"prereg gate: {tag_why}")
    if not lic_ok:
        reasons.append(f"licence gate: {lic_why}")
    return (tag_ok and lic_ok), reasons


@dataclass
class BatchPlan:
    extract_jobs: list[str] = field(default_factory=list)
    soulx_job: str = "soulx-slice"
    probe_jobs: list[str] = field(default_factory=list)

    def ordered(self) -> list[str]:
        return list(self.extract_jobs) + [self.soulx_job] + list(self.probe_jobs)


def build_plan() -> BatchPlan:
    extract = [f"extract-{e}" for e in ENCODERS] + ["extract-basic-pitch"]
    probes = [f"probe-{e}-seed{s}" for e in (*ENCODERS, "control") for s in SEEDS]
    return BatchPlan(extract_jobs=extract, probe_jobs=probes)


def estimate(plan: BatchPlan) -> dict:
    return {
        "extract_hours": EST_EXTRACT_HOURS,
        "soulx_hours": EST_SOULX_HOURS,
        "probe_hours": EST_PROBE_HOURS,
        "total_hours": EST_EXTRACT_HOURS + EST_SOULX_HOURS + EST_PROBE_HOURS,
        "batch_cap_hours": BATCH_CAP_HOURS,
        "temp_disk_gb": EST_TEMP_DISK_GB,
        "n_extract": len(plan.extract_jobs),
        "n_probe": len(plan.probe_jobs),
    }


def should_drop_extractor(projected_hours: float, projected_gb: float, mps_ok: bool,
                          pilot_lead: bool) -> tuple[bool, str]:
    """Drop an extractor whose calibrated projection blows the limits (prereg compute rule)."""
    if projected_hours > MAX_EXTRACT_HOURS:
        return True, f"projected {projected_hours:.1f} h > {MAX_EXTRACT_HOURS} h"
    if projected_gb > MAX_EXTRACT_GB:
        return True, f"projected {projected_gb:.0f} GB > {MAX_EXTRACT_GB:.0f} GB"
    if not mps_ok and not pilot_lead:
        return True, "unsupported on MPS without a clear pilot lead"
    return False, "ok"


def run_batch(dry_run: bool = True, licences_md: Path | None = None) -> dict:
    """Plan (and, in a window, run) the batch. In this repository state the function only *plans* and
    enforces the gates; the actual GPU extraction/training is wired by the window agent on the Mac. Running
    with ``dry_run=False`` before the gates pass raises."""
    ok, reasons = gates_ok(licences_md)
    plan = build_plan()
    est = estimate(plan)
    if not dry_run and not ok:
        raise PermissionError("refusing to start the GPU batch: " + "; ".join(reasons))
    return {"gates_ok": ok, "gate_reasons": reasons, "order": plan.ordered(), "estimate": est}


def unload_lm_studio() -> None:
    """Free the LM Studio models (~36 GB) before the window; ignored if lms isn't installed."""
    try:
        subprocess.run(["lms", "unload", "--all"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="print the plan and gate status, run nothing")
    args = parser.parse_args(argv)

    result = run_batch(dry_run=True)
    est = result["estimate"]
    print("ear bake-off GPU batch plan (one GPU job at a time, in order):")
    for i, job in enumerate(result["order"], 1):
        print(f"  {i:2d}. {job}")
    print(f"estimate: extract ~{est['extract_hours']} h, soulx ~{est['soulx_hours']} h, "
          f"probes ~{est['probe_hours']} h = ~{est['total_hours']} h total "
          f"(batch cap {est['batch_cap_hours']} h)")
    print(f"temp disk: {est['temp_disk_gb'][0]:.0f}-{est['temp_disk_gb'][1]:.0f} GB; "
          f"stop below {paths.FREE_SPACE_FLOOR/1e9:.0f} GB free")
    print(f"gates: {'PASS' if result['gates_ok'] else 'BLOCKED'}"
          + ("" if result["gates_ok"] else " -> " + "; ".join(result["gate_reasons"])))
    if args.dry_run:
        return 0
    if not result["gates_ok"]:
        print("Refusing to run: gates not satisfied.", file=sys.stderr)
        return 1
    print("This script is run by the window agent on the Mac; GPU extraction/training is wired there.",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
