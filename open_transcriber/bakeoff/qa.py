"""Chart QA for the karaoke songs, by the committed thresholds (``qa_thresholds.json``; see ``QA.md``).

TEST-ONLY. Operates on CC BY-NC karaoke audio/charts for evaluation only. Not for product/training import.

For each item: a CREPE track of its seeded Demucs vocal (``crepe.py``), onset evidence (``onsets.py``), a verdict
per chart note (kept, or masked as unverifiable, pitch, onset or freestyle), the song-drop rule, and
diagnostics that are reported but never used to correct anything. Output: ``DATA/qa/<item>.json`` with the
thresholds' SHA-256; the CREPE track goes to ``DATA/crepe/<item>.npz``. Charts are never edited.

    python -m open_transcriber.bakeoff.qa [--items a,b] [--shard 1/4]     # evaluation environment
    python -m open_transcriber.bakeoff.qa --summary                        # results/stage0a/qa-summary.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

from . import crepe, fetch, jobs, notes_io, onsets, paths

JUDGE_DIR = Path(__file__).resolve().parent
THRESHOLDS = JUDGE_DIR / "qa_thresholds.json"
SUMMARY = JUDGE_DIR / "results" / "stage0a" / "qa-summary.json"


def load_thresholds() -> tuple[dict, str]:
    return json.loads(THRESHOLDS.read_text(encoding="utf-8")), fetch.sha256_file(THRESHOLDS)


def qa_thresholds_sha256() -> str:
    """SHA-256 of the committed QA thresholds (masks are only trusted when made with these)."""
    return fetch.sha256_file(THRESHOLDS)


def load_qa(item_id: str, data: Path | None = None) -> dict:
    """An item's QA verdicts, refusing masks made with other thresholds. Ported from the stage 0a scorer."""
    path = Path(data or paths.data_dir()) / "qa" / f"{item_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"{item_id}: no QA masks; run the chart QA first")
    verdicts = json.loads(path.read_text(encoding="utf-8"))
    if not THRESHOLDS.exists() or verdicts.get("thresholds_sha256") != qa_thresholds_sha256():
        raise ValueError(f"{item_id}: QA masks were made with different thresholds")
    return verdicts


def median3(values: np.ndarray) -> np.ndarray:
    """3-frame median; at each end, the lower of the two available values (as torchcrepe.filter.median)."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 3:
        return v.copy()
    out = np.median(np.stack([v[:-2], v[1:-1], v[2:]]), axis=0)
    return np.concatenate([[min(v[0], v[1])], out, [min(v[-2], v[-1])]])


def voiced_frames(track: dict, t: dict) -> np.ndarray:
    rule = t["voiced_frame"]
    periodic = median3(track["periodicity"]) >= rule["periodicity_min"]
    rms = np.asarray(track["rms"], dtype=np.float64)
    loud = rms >= np.percentile(rms, 95) * 10 ** (-rule["rms_floor_db_below_p95"] / 20.0)
    return periodic & loud


def pitch_window(onset: float, offset: float, t: dict) -> tuple[float, float]:
    rule = t["pitch_window"]
    d = offset - onset
    if d < rule["short_note_s"]:
        return onset, offset
    return onset + rule["head_fraction"] * d, offset - rule["tail_fraction"] * d


def pc_distance(a: float, b: float) -> float:
    """Circular pitch-class distance in semitones."""
    return abs(((a - b) + 6.0) % 12.0 - 6.0)


def note_verdicts(ref: notes_io.Notes, track: dict, evidence: dict[str, np.ndarray], t: dict) -> list[dict]:
    """One verdict per reference note: ``mask`` is "" (kept) or a reason."""
    voiced = voiced_frames(track, t)
    midi = np.asarray(track["midi"], dtype=np.float64)
    times = np.arange(len(midi)) * crepe.HOP / crepe.SAMPLE_RATE
    all_evidence = np.sort(np.concatenate([evidence[k] for k in ("flux", "voicing", "pitch")]))
    verdicts = []
    for i in range(len(ref)):
        if ref.mask[i] == "freestyle":
            verdicts.append({"id": i, "mask": "freestyle"})
            continue
        start, end = pitch_window(ref.onset[i], ref.offset[i], t)
        inside = (times >= start - 1e-9) & (times <= end + 1e-9)
        n_inside = int(inside.sum())
        n_voiced = int((inside & voiced).sum())
        verifiable = (n_voiced >= t["verifiable"]["min_voiced_frames"]
                      and n_inside > 0 and n_voiced / n_inside >= t["verifiable"]["min_voiced_share"])
        distance = None
        if verifiable:
            distance = pc_distance(float(np.median(midi[inside & voiced])), float(ref.midi[i]))
        nearest = onsets.nearest(all_evidence, float(ref.onset[i]))
        pitch_ok = verifiable and distance <= t["pitch_pass"]["max_semitones"] + 1e-9
        onset_ok = abs(nearest) <= t["onset_pass"]["tolerance_s"] + 1e-9
        if not verifiable:
            mask = "unverifiable"
        elif not pitch_ok:
            mask = "pitch"
        elif not onset_ok:
            mask = "onset"
        else:
            mask = ""
        verdicts.append({"id": i, "mask": mask, "offset_s": None if not np.isfinite(nearest) else round(nearest, 4)})
    return verdicts


def summarise(verdicts: list[dict], t: dict) -> dict:
    pitched = [v for v in verdicts if v["mask"] != "freestyle"]
    reasons = {r: sum(1 for v in pitched if v["mask"] == r) for r in t["note_verdict"]["reason_order"]}
    masked = sum(reasons.values())
    share = masked / len(pitched) if pitched else 1.0
    offsets = [v["offset_s"] for v in pitched if v.get("offset_s") is not None]
    return {
        "pitched": len(pitched), "freestyle": len(verdicts) - len(pitched), "masked": masked,
        "masked_share": round(share, 4),
        "masked_share_by_reason": {r: round(n / len(pitched), 4) if pitched else 0.0 for r, n in reasons.items()},
        "dropped": share > t["song_drop"]["max_masked_share"],
        "median_signed_offset_s": round(float(np.median(offsets)), 4) if offsets else None,
    }


def check_audio(audio16k: np.ndarray, ref: notes_io.Notes, model: crepe.Model, t: dict) -> tuple[list[dict], dict, dict]:
    """(verdicts, summary, CREPE track) for a 16 kHz vocal and a reference."""
    track = crepe.track(audio16k, model, t["pitch_tracker"]["fmin_hz"], t["pitch_tracker"]["fmax_hz"])
    voiced = voiced_frames(track, t)
    evidence = onsets.evidence(audio16k, crepe.SAMPLE_RATE, track["midi"], voiced)
    verdicts = note_verdicts(ref, track, evidence, t)
    return verdicts, summarise(verdicts, t), track


def run_item(item: dict, model: crepe.Model, data: Path, out: Path) -> dict:
    t, t_sha = load_thresholds()
    stem = Path(out) / "stems" / item["id"] / "vocals.wav"
    audio = crepe.load_16k(stem)
    ref = notes_io.load_reference(data / item["reference"]["path"])
    verdicts, summary, track = check_audio(audio, ref, model, t)
    crepe_path = data / "crepe" / f"{item['id']}.npz"
    provenance = {"item": item["id"], "stem_sha256": fetch.sha256_file(stem), "model_sha256": crepe.MODEL_SHA256,
                  "fmin": t["pitch_tracker"]["fmin_hz"], "fmax": t["pitch_tracker"]["fmax_hz"]}
    crepe.save_track(crepe_path, track, provenance)
    result = {"item": item["id"], "thresholds_version": t["version"], "thresholds_sha256": t_sha,
              "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
              "reference_sha256": item["reference"]["sha256"], **provenance, **summary,
              "notes": [{"id": v["id"], "mask": v["mask"]} for v in verdicts]}
    path = data / "qa" / f"{item['id']}.json"
    jobs.write_json(path, result)
    return {"outputs": {"qa": path, "crepe": crepe_path}, "dropped": summary["dropped"],
            "masked_share": summary["masked_share"]}


def write_summary(data: Path, items: list[dict]) -> dict:
    """Counts and ids only: per song, the masked share by reason, the drop verdict and the median offset."""
    _, t_sha = load_thresholds()
    songs = []
    for item in items:
        qa = json.loads((data / "qa" / f"{item['id']}.json").read_text(encoding="utf-8"))
        if qa["thresholds_sha256"] != t_sha:
            raise ValueError(f"{item['id']}: QA made with other thresholds")
        songs.append({k: qa[k] for k in ("item", "pitched", "freestyle", "masked", "masked_share",
                                         "masked_share_by_reason", "dropped", "median_signed_offset_s")}
                     | {"status": item["status"]})
    summary = {"thresholds_sha256": t_sha, "songs": songs,
               "dropped": [s["item"] for s in songs if s["dropped"]],
               "masked_notes": sum(s["masked"] for s in songs), "pitched_notes": sum(s["pitched"] for s in songs)}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def record_masks(data: Path) -> int:
    """Put each mask file's SHA-256 into the karaoke manifest (``item["qa"]``); return how many."""
    from . import manifests

    manifest = manifests.load("usdx")
    count = 0
    for item in manifest["items"]:
        path = data / "qa" / f"{item['id']}.json"
        if path.exists():
            item["qa"] = {"path": path.relative_to(data).as_posix(), "sha256": fetch.sha256_file(path)}
            count += 1
    manifests.write(manifest)
    return count


def main(argv: list[str] | None = None) -> int:
    from . import manifests

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--items", default=None)
    parser.add_argument("--shard", default=None, help="k/n: every n-th item from the k-th (1-based)")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)
    data, out = paths.data_dir(), paths.out_dir()
    items = jobs.select(manifests.load("usdx")["items"], args.items)
    if args.summary:
        summary = write_summary(data, items)
        recorded = record_masks(data)
        print(f"wrote {SUMMARY}: {len(summary['songs'])} songs, dropped {summary['dropped']}, "
              f"{summary['masked_notes']} of {summary['pitched_notes']} pitched notes masked; "
              f"{recorded} mask hashes in manifests/usdx.json")
        return 0
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        items = items[k - 1::n]
    model = crepe.Model(threads=args.threads)
    job = jobs.Job(name=f"qa{('-' + args.shard.replace('/', 'of')) if args.shard else ''}", system="qa",
                   run=lambda item: run_item(item, model, data, out), gpu=False, device="cpu",
                   versions=jobs.python_versions("numpy", "scipy", "onnxruntime"))
    counts = job.run_queue(items)
    return 0 if counts["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
