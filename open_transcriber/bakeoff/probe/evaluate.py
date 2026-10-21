"""Decode a trained probe's outputs and score them against the frozen real-development references.

Produces a results record keyed by encoder + seed with the primary metrics (octave-exact melody F1 at 50 ms
and 100 ms) and the secondary metrics, reporting synthetic and real separately. Reads the frozen C1 split
manifests and refuses any test-partition id (RWC-Pop, the RWC-other test split, USDX test, COTB).
"""

from __future__ import annotations

import numpy as np

from .. import dev_splits, notes_io, score
from . import decode

#: Real development melody metrics we report; the primary is octave-exact at 50 ms and 100 ms.
REAL_MELODY_METRICS = ("oct_on50", "pc_on50")  # "oct_on100" added below from the 100 ms variant
BEAT_TOLERANCE = 0.07


def guard_real_dev_ids(ids: list[str]) -> None:
    """Refuse to score if any id belongs to a sealed test partition."""
    sealed = dev_splits.test_partition_ids()
    leaked = sorted(set(ids) & sealed)
    if leaked:
        raise ValueError(f"refusing to score sealed test-partition ids: {leaked}")
    if any(str(i).lower().startswith("cotb") for i in ids):
        raise ValueError("refusing to score COTB ids during the bake-off")


def pulse_f1(pred_times: list[float], ref_times: list[float], tol: float = BEAT_TOLERANCE) -> float:
    """F1 of predicted pulses (beats/downbeats) against references within ``tol`` seconds (greedy match)."""
    if not pred_times and not ref_times:
        return 1.0
    if not pred_times or not ref_times:
        return 0.0
    ref = sorted(ref_times)
    used = [False] * len(ref)
    tp = 0
    for t in sorted(pred_times):
        for i, r in enumerate(ref):
            if not used[i] and abs(r - t) <= tol:
                used[i] = True
                tp += 1
                break
    precision = tp / len(pred_times)
    recall = tp / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def duration_weighted_chord(pred_spans, ref_spans) -> float:
    """Fraction of reference chord-span time whose root+maj/min matches the overlapping prediction."""
    total, correct = 0.0, 0.0
    for (r_on, r_off, r_root, r_q) in ref_spans:
        total += max(0.0, r_off - r_on)
        for (p_on, p_off, p_root, p_q) in pred_spans:
            overlap = max(0.0, min(r_off, p_off) - max(r_on, p_on))
            if overlap > 0 and p_root == r_root and p_q == r_q:
                correct += overlap
    return (correct / total) if total else 0.0


def score_one_clip(out: dict, labels: dict, metrics=("oct_on50", "oct_on100", "pc_on50"),
                   item: str = "") -> dict:
    """Score one decoded clip against its reference labels. ``out`` holds numpy head arrays; ``labels`` the
    exact label record (from ``labels.py`` JSON or an equivalent dict)."""
    notes = decode.decode_notes(out["pitch"], out["silence"])
    est = notes_io.Notes([n[0] for n in notes], [n[1] for n in notes], [n[2] for n in notes]) if notes \
        else notes_io.empty()
    ref_notes = labels["notes"]
    ref = notes_io.Notes([n["onset"] for n in ref_notes], [n["offset"] for n in ref_notes],
                         [n["midi"] for n in ref_notes]) if ref_notes else notes_io.empty()
    # octave-exact uses the written MIDI directly as truth (sampler labels are exact)
    sung = ref.midi
    row = {"item": item}
    for name in metrics:
        base = name.replace("oct_on100", "pc_on100")  # reuse pc_on100 matcher for the 100 ms grid
        metric_key = {"oct_on100": "pc_on100"}.get(name, name)
        result = score.score_metric(ref, est, score.METRICS[metric_key], sung if name.startswith("oct") else None)
        row[f"{name}_f1"] = round(result["f1"], 6)
    # beats / downbeats
    row["beat_f1"] = round(pulse_f1(decode.decode_pulses(out["beat"]), labels.get("beats", [])), 6)
    row["downbeat_f1"] = round(pulse_f1(decode.decode_pulses(out["downbeat"]), labels.get("downbeats", [])), 6)
    # chords
    pred_chords = decode.decode_chords(out["chord_root"], out["chord_quality"])
    ref_chords = [(c["onset"], c["offset"], c["root_pc"], c["quality"]) for c in labels.get("chords", [])]
    row["chord"] = round(duration_weighted_chord(pred_chords, ref_chords), 6)
    # key
    row["key"] = int(decode.decode_key(out["key"]) == labels.get("key_class", -1))
    return row


def aggregate(rows: list[dict], metric: str) -> float:
    vals = [r[metric] for r in rows if metric in r]
    return float(np.mean(vals)) if vals else 0.0


def evaluate_probe(encoder: str, seed: int, clips: list[tuple[dict, dict, str]],
                   partition: str = "real") -> dict:
    """Score every (head_outputs, labels, item) clip; return a results record keyed by encoder+seed.

    ``partition`` is "real" or "synthetic" so the caller reports them separately. Real partitions are guarded
    against sealed test ids.
    """
    ids = [item for (_out, _lab, item) in clips]
    if partition == "real":
        guard_real_dev_ids(ids)
    rows = [score_one_clip(out, lab, item=item) for (out, lab, item) in clips]
    means = {}
    for key in ("oct_on50_f1", "oct_on100_f1", "pc_on50_f1", "beat_f1", "downbeat_f1", "chord", "key"):
        means[key] = round(aggregate(rows, key), 6)
    return {"encoder": encoder, "seed": seed, "partition": partition, "n": len(rows), "means": means}
