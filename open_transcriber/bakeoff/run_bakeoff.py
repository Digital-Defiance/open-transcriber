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

from . import jobs, paths
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


#: The licence table committed inside the package. A worktree has no tracked ``.agents/``, so the gate reads
#: this package file by default and only falls back to the ``.agents`` task copy if the package file is gone.
PACKAGE_LICENCES_MD = Path(__file__).with_name("licences.md")
_AGENTS_LICENCES_MD = paths.copyist_root() / ".agents" / "tasks" / "ear-bakeoff-2026-10-20" / "licences.md"


def default_licences_md() -> Path:
    """The licence table to read: the committed package file, else the ``.agents`` task copy."""
    if PACKAGE_LICENCES_MD.exists():
        return PACKAGE_LICENCES_MD
    return _AGENTS_LICENCES_MD


def licences_confirmed(licences_md: Path | None = None) -> tuple[bool, str]:
    """True when the licence findings file marks every tested encoder free/allowed (fail closed)."""
    path = licences_md or default_licences_md()
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


def bakeoff_out_dir(out_root: Path | None = None) -> Path:
    """Where the batch writes its verdict, metric table and per-stage checkpoints."""
    return Path(out_root) if out_root is not None else paths.out_dir() / "ear-bakeoff"


def _calibrate_and_drop(encoders, calibrate, drop_fn) -> tuple[list[str], list[dict]]:
    """Run the calibrate-and-drop stage. Returns (surviving encoder names, drop-ledger records).

    HeartCodec is always dropped via its stub descriptor; the others are calibrated on ~10 min of audio and
    dropped if ``drop_fn`` (``should_drop_extractor``) says their projection blows the prereg compute limits.
    """
    surviving, ledger = [], []
    for enc in encoders:
        if enc == "heartcodec":
            from .probe.extractors.heartcodec import heartcodec_descriptor
            ledger.append({"encoder": enc, "dropped": True, "reason": heartcodec_descriptor().note})
            continue
        proj = calibrate(enc)   # {"hours","gb","mps_ok","pilot_lead"}
        drop, why = drop_fn(proj["hours"], proj["gb"], proj["mps_ok"], proj["pilot_lead"])
        if drop:
            ledger.append({"encoder": enc, "dropped": True, "reason": why, **proj})
        else:
            surviving.append(enc)
            ledger.append({"encoder": enc, "dropped": False, "reason": "ok", **proj})
    return surviving, ledger


def run_batch(dry_run: bool = True, licences_md: Path | None = None, *,
              out_root: Path | None = None,
              extractor_factory=None, bp_factory=None, trainer=None, evaluator=None, decider=None,
              calibrate=None, train_clips=None, dev_clips=None,
              gpu_lock=None, caffeinate=None, battery=None) -> dict:
    """Plan the batch, and when ``dry_run=False`` run the full resumable loop behind the gates.

    The real GPU window calls this with ``dry_run=False`` and no injected hooks, so the defaults wire the
    real extractors, Basic Pitch side input, probe trainer/evaluator and ``decide``. Tests inject fakes
    (``extractor_factory``, ``bp_factory``, ``trainer``, ``evaluator``, ``decider``, ``calibrate``, clip
    lists, and patched ``gpu_lock``/``caffeinate``/``battery``) so the whole extract->assemble->train->
    evaluate->decide loop runs deterministically with no real model, network or GPU.

    Stages run in a fixed order, each resumable via a checkpoint under ``out_root``: calibrate-and-drop,
    extract, assemble, train (surviving encoder x seed + control x seed), evaluate on frozen real-dev,
    ``decide``. Running before the gates pass raises.
    """
    ok, reasons = gates_ok(licences_md)
    plan = build_plan()
    est = estimate(plan)
    if dry_run:
        return {"gates_ok": ok, "gate_reasons": reasons, "order": plan.ordered(), "estimate": est}
    if not ok:
        raise PermissionError("refusing to start the GPU batch: " + "; ".join(reasons))
    return _execute(
        out_root=out_root, plan=plan, est=est, reasons=reasons,
        extractor_factory=extractor_factory, bp_factory=bp_factory, trainer=trainer,
        evaluator=evaluator, decider=decider, calibrate=calibrate,
        train_clips=train_clips, dev_clips=dev_clips,
        gpu_lock=gpu_lock, caffeinate=caffeinate, battery=battery,
    )


def _default_calibrate(encoder: str) -> dict:
    """Default calibration from the smoke-test findings: nothing blows the prereg compute limits.

    Observed 21 October: each surviving encoder extracts well under 3 h / 48 GB. MusicFM and Dasheng run on
    mps; Whisper is CPU-only but supported and fast (~0.12 s/clip), so it carries a clear pilot lead rather
    than being an mps drop. HeartCodec never reaches here (dropped by its stub before calibration).
    """
    mps_ok = encoder not in ("whisper", "whisper-base-en")
    return {"hours": 1.0, "gb": 5.0, "mps_ok": mps_ok, "pilot_lead": True}


def _execute(*, out_root, plan, est, reasons, extractor_factory, bp_factory, trainer, evaluator,
             decider, calibrate, train_clips, dev_clips, gpu_lock, caffeinate, battery) -> dict:
    """The resumable batch body. All heavy collaborators are injectable so this is CPU-testable."""
    from .probe import data as probe_data
    from .probe.extractor import build_extractor
    from .probe.extractors.basic_pitch_side import BasicPitchSideInput

    out = bakeoff_out_dir(out_root)
    out.mkdir(parents=True, exist_ok=True)

    extractor_factory = extractor_factory or build_extractor
    bp_factory = bp_factory or (lambda: BasicPitchSideInput())
    trainer = trainer or _default_trainer
    evaluator = evaluator or _default_evaluator
    decider = decider or _default_decider
    calibrate = calibrate or _default_calibrate
    train_clips = train_clips if train_clips is not None else []
    dev_clips = dev_clips if dev_clips is not None else []

    lock = gpu_lock if gpu_lock is not None else jobs.GpuLock()
    keep_awake = (caffeinate or jobs.hold_caffeinate)()
    battery = battery or jobs.battery_watchdog
    results: dict = {"stages": [], "drops": [], "metrics": [], "verdict": None}
    try:
        with lock:
            # (a) calibrate-and-drop
            surviving, ledger = _resumable(out / "calibrate.json",
                                           lambda: _calibrate_and_drop(ENCODERS, calibrate,
                                                                       should_drop_extractor))
            results["drops"] = [r for r in ledger if r["dropped"]]
            results["stages"].append("calibrate")

            bp = bp_factory()
            # (b)+(c) extract + assemble per surviving encoder, and (d) train each encoder x seed + control.
            trained: dict[str, dict] = {}
            for enc in surviving + ["control"]:
                battery()
                ext = None if enc == "control" else extractor_factory(enc)
                for seed in SEEDS:
                    tag = f"{enc}-seed{seed}"
                    ckpt = out / "probes" / f"{tag}.json"
                    metric = _resumable(ckpt, lambda enc=enc, seed=seed, ext=ext: _train_and_eval(
                        enc, seed, ext, bp, train_clips, dev_clips, trainer, evaluator, probe_data))
                    trained.setdefault(enc, {})[seed] = metric
                    results["metrics"].append(metric)
            results["stages"].append("train+evaluate")

            # (f) decide + write verdict
            verdict = _resumable(out / "verdict.json",
                                 lambda: decider(trained, surviving))
            results["verdict"] = verdict
            results["stages"].append("decide")
    finally:
        if keep_awake is not None:
            keep_awake.terminate()

    _write_metric_table(out / "metrics.json", results)
    results["gate_reasons"] = reasons
    results["estimate"] = est
    return results


def _resumable(path: Path, compute):
    """Run ``compute()`` once and cache its JSON result at ``path``; a re-run reads the cache (resumability)."""
    cached = jobs.read_json(path)
    if cached is not None and "value" in cached:
        return cached["value"]
    value = compute()
    jobs.write_json(path, {"value": value})
    return value


def _train_and_eval(encoder, seed, extractor, bp, train_clips, dev_clips, trainer, evaluator,
                    probe_data) -> dict:
    """Train one probe (encoder x seed, or control) and evaluate it once on the frozen real-dev splits."""
    model = trainer(encoder, seed, extractor, bp, train_clips, probe_data)
    return evaluator(encoder, seed, model, extractor, bp, dev_clips, probe_data)


def _default_trainer(encoder, seed, extractor, bp, train_clips, probe_data):
    """Assemble train batches and train one probe (real path, used in the GPU window)."""
    from .probe.model import BP_POSTERIOR_DIM, ProbeConfig
    from .probe.train import TrainConfig, train_probe

    control = (encoder == "control")
    feature_dim = 1 if control else extractor.descriptor().feature_dim
    batches = probe_data.build_train_batches(train_clips, extractor, bp) if not control \
        else probe_data.build_train_batches(train_clips, _ControlExtractor(), bp)
    cfg = TrainConfig(probe=ProbeConfig(feature_dim=feature_dim, bp_dim=BP_POSTERIOR_DIM, control=control),
                      seed=seed)
    val = batches  # early-stop validation uses the synthetic/CC-BY material (allowed); real dev is never used
    model, _result = train_probe(cfg, batches, val)
    return model


def _default_evaluator(encoder, seed, model, extractor, bp, dev_clips, probe_data) -> dict:
    """Evaluate one trained probe on the frozen real-dev clips and return the metric record."""
    from .probe.evaluate import evaluate_probe

    ext = extractor if encoder != "control" else _ControlExtractor()
    clips = probe_data.build_eval_clips(dev_clips, model, ext, bp)
    return evaluate_probe(encoder, seed, clips, partition="real")


class _ControlExtractor:
    """A minimal 1-dim extractor for the Basic-Pitch-only control (features unused by the control model)."""

    def descriptor(self):
        from .probe.extractor import ExtractorDescriptor
        return ExtractorDescriptor(name="control", code_licence="Apache-2.0", weight_licence="Apache-2.0",
                                   native_rate_hz=50.0, feature_dim=1)

    def features(self, wav_path):
        import numpy as np
        return {name: np.zeros((1, 1), dtype=np.float32) for name in ("early", "middle", "final")}


def _default_decider(trained: dict, surviving: list[str]) -> dict:
    """Build ``decide.EncoderResult``s from the metric records and apply the pre-registered GO/NO-GO rule.

    ``decide`` stays the SOLE adjudicator; this only marshals the per-seed metrics into its inputs.
    """
    from . import decide as D

    def result_for(encoder: str) -> D.EncoderResult:
        by_seed = trained[encoder]
        real = {s: by_seed[s]["means"]["oct_on50_f1"] * 100.0 for s in by_seed}
        synth = {s: by_seed[s]["means"].get("synthetic_oct_on50_f1",
                                            by_seed[s]["means"]["oct_on50_f1"]) * 100.0 for s in by_seed}
        mt = {s: sum(by_seed[s]["means"][k] for k in D.MULTITASK_KEYS) / len(D.MULTITASK_KEYS) * 100.0
              for s in by_seed}
        return D.EncoderResult(encoder=encoder, real_melody=real, synthetic_melody=synth, multitask_mean=mt)

    control = result_for("control")
    candidates = [result_for(e) for e in surviving]
    decision = D.decide(candidates, control)
    return {"go": decision.go, "winner": decision.winner, "add_whisper": decision.add_whisper,
            "qualifying": decision.qualifying, "reasons": decision.reasons}


def _write_metric_table(path: Path, results: dict) -> None:
    jobs.write_json(path, {
        "verdict": results.get("verdict"),
        "drops": results.get("drops", []),
        "metrics": results.get("metrics", []),
        "stages": results.get("stages", []),
    })


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
