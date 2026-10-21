"""The strict note scorer: a frozen melody metric and strict variants, with mir_eval parity.

Ported verbatim as the matching core (stage 0a ``judge/score.py``, source ``aa88f5a``). The SheetSage2 CLI
branches (``--system sheetsage2``, the pre-registration gate, ``endpoints.json`` and the committed-CSV verify)
are intentionally dropped here: the bake-off scores probe outputs against frozen development references, not
SheetSage. The matching core, metrics, masks and don't-care logic are unchanged, so mir_eval parity and the
randomized matching property tests still hold.

Per-recording note F1 with onsets within a tolerance, a pitch test (pitch class or octave-exact, ±50 cents),
an optional offset test, and the macro mean over recordings. Empty references score 0, no time alignment. It
runs on mir_eval's note matching (onset distances rounded to 4 decimals and compared with ``<=``; maximum
bipartite matching).

| metric | pitch | onset | offset |
|---|---|---|---|
| ``pc_on50`` | pitch class | 50 ms | none |
| ``oct_on50`` | exact (±50 cents) | 50 ms | none |
| ``oct_on50_shift`` | exact after the best whole-octave shift in [-2, 2] | 50 ms | none |
| ``pc_onoff50`` | pitch class | 50 ms | max(0.2·ref duration, 50 ms) |
| ``oct_onoff50`` | exact | 50 ms | max(0.2·ref duration, 50 ms) |
| ``pc_on100`` | pitch class | 100 ms | none |

Masked references (karaoke QA failures and freestyle notes) are don't-cares: estimates are matched to the
unmasked notes first; a leftover estimate that would match a masked note (pitch waived for freestyle) is dropped
from the estimate count. So precision = TP / (estimates - don't-cares) and recall = TP / unmasked references.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import notes_io

PITCH_TOLERANCE = 50.0
OFFSET_MIN = 0.05


@dataclass(frozen=True)
class Metric:
    name: str
    pitch: str  # "pc" or "exact"
    onset: float
    offset_ratio: float | None
    shift: bool = False


METRICS = {m.name: m for m in (
    Metric("pc_on50", "pc", 0.05, None),
    Metric("oct_on50", "exact", 0.05, None),
    Metric("oct_on50_shift", "exact", 0.05, None, shift=True),
    Metric("pc_onoff50", "pc", 0.05, 0.2),
    Metric("oct_onoff50", "exact", 0.05, 0.2),
    Metric("pc_on100", "pc", 0.10, None),
)}
RWC_METRICS = ("pc_on50", "oct_on50", "oct_on50_shift", "pc_onoff50", "oct_onoff50")
KARAOKE_METRICS = ("pc_on50", "pc_on100", "pc_onoff50")
#: Octave-exact metrics leave these RWC-Pop recordings out of their means (unreliable octave truth).
OCTAVE_EXCLUDED = ("rwc-p003", "rwc-p031")


def run_folder(system: str) -> str:
    """The output folder a system's runs live in (a diagnostic view reads its base system's runs)."""
    return system.split("@", 1)[0]


def to_hz(midi: np.ndarray, pitch: str) -> np.ndarray:
    """Hz for mir_eval: exact MIDI, or the pitch class folded into C4..B4 (MIDI mod 12, then +60)."""
    midi = np.asarray(midi, dtype=np.float64)
    folded = np.mod(midi, 12.0) + 60.0 if pitch == "pc" else midi
    return 440.0 * 2.0 ** ((folded - 69.0) / 12.0)


def _intervals(notes: notes_io.Notes) -> np.ndarray:
    return np.column_stack([notes.onset, notes.offset]) if len(notes) else np.zeros((0, 2))


def _hit_graph(ref_iv, ref_hz, est_iv, est_hz, metric: Metric, pitch_waived=None) -> dict:
    """mir_eval.transcription.match_notes's compatibility graph {est: [refs]} (same rounding and <=)."""
    from mir_eval import transcription

    onset = np.around(np.abs(np.subtract.outer(ref_iv[:, 0], est_iv[:, 0])), decimals=transcription.N_DECIMALS)
    hits = onset <= metric.onset
    pitch = np.abs(1200 * np.subtract.outer(np.log2(ref_hz), np.log2(est_hz))) <= PITCH_TOLERANCE
    if pitch_waived is not None:
        pitch = pitch | np.asarray(pitch_waived, dtype=bool)[:, None]
    hits = hits & pitch
    if metric.offset_ratio is not None:
        offset = np.around(np.abs(np.subtract.outer(ref_iv[:, 1], est_iv[:, 1])), decimals=transcription.N_DECIMALS)
        tolerance = np.maximum(metric.offset_ratio * (ref_iv[:, 1] - ref_iv[:, 0]), OFFSET_MIN)
        hits = hits & (offset <= tolerance.reshape(-1, 1))
    graph: dict = {}
    for ref_i, est_i in zip(*np.where(hits)):
        graph.setdefault(int(est_i), []).append(int(ref_i))
    return graph


def match(ref: notes_io.Notes, est: notes_io.Notes, metric: Metric, ref_pitch: np.ndarray,
          est_shift: int = 0) -> list[tuple[int, int]]:
    """Maximum matching between reference and estimated notes for one metric: [(ref, est)]."""
    from mir_eval import transcription

    if len(ref) == 0 or len(est) == 0:
        return []
    return transcription.match_notes(
        _intervals(ref), to_hz(ref_pitch, metric.pitch), _intervals(est), to_hz(est.midi + 12 * est_shift, metric.pitch),
        onset_tolerance=metric.onset, pitch_tolerance=PITCH_TOLERANCE, offset_ratio=metric.offset_ratio,
        offset_min_tolerance=OFFSET_MIN, strict=False)


def _prf(matched: int, n_ref: int, n_est: int) -> tuple[float, float, float]:
    if n_ref == 0 or n_est == 0:
        return 0.0, 0.0, 0.0
    precision, recall = matched / n_est, matched / n_ref
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_metric(ref: notes_io.Notes, est: notes_io.Notes, metric: Metric, ref_sung: np.ndarray | None = None) -> dict:
    """One recording, one metric: matched notes, don't-cares, precision, recall, F1 (and the octave shift)."""
    from mir_eval import util

    ref_pitch = ref.midi if metric.pitch == "pc" or ref_sung is None else ref_sung
    scored = ref.scored
    n_ref = int(scored.sum())
    shifts = range(-2, 3) if metric.shift else (0,)
    best = None
    for shift in shifts:
        unmasked = ref.select(scored)
        pairs = match(unmasked, est, metric, ref_pitch[scored], shift)
        dontcare = 0
        if (~scored).any() and len(est):
            used = {e for _, e in pairs}
            left = np.array([i for i in range(len(est)) if i not in used], dtype=int)
            if len(left):
                masked = ref.select(~scored)
                waived = masked.mask == "freestyle"
                graph = _hit_graph(_intervals(masked), to_hz(ref_pitch[~scored], metric.pitch),
                                   _intervals(est.select(left)), to_hz(est.midi[left] + 12 * shift, metric.pitch),
                                   metric, waived)
                dontcare = len(util._bipartite_match(graph))
        p, r, f = _prf(len(pairs), n_ref, len(est) - dontcare)
        result = {"matched": len(pairs), "dontcare": dontcare, "p": p, "r": r, "f1": f, "shift": shift}
        if best is None or f > best["f1"] + 1e-12 or (abs(f - best["f1"]) <= 1e-12 and abs(shift) < abs(best["shift"])):
            best = result
    return best


def mean_f1(rows: list[dict], metric: str) -> float:
    """The macro mean over recordings (octave-exact metrics leave out ``octave_excluded`` rows)."""
    values = [r[f"{metric}_f1"] for r in rows if not (metric.startswith("oct") and r.get("octave_excluded"))]
    return float(np.mean(values)) if values else 0.0


def score_pair(ref: notes_io.Notes, est: notes_io.Notes, metrics: tuple[str, ...],
               sung: np.ndarray | None = None, octave_excluded: bool = False, item: str = "") -> dict:
    """One (reference, estimate) pair scored on the named metrics: a flat row of matched/dontcare/p/r/f1.

    The bake-off builds references by construction (sampler/SoulX labels) and from the frozen real-dev
    references, then scores each against a probe's decoded estimate with this function.
    """
    row = {"item": item, "octave_excluded": int(octave_excluded), "n_ref": int(ref.scored.sum()),
           "n_ref_masked": int((~ref.scored).sum()), "n_est": len(est), "n_est_repaired": est.repaired}
    for name in metrics:
        result = score_metric(ref, est, METRICS[name], sung)
        row.update({f"{name}_matched": result["matched"], f"{name}_dontcare": result["dontcare"],
                    f"{name}_p": round(result["p"], 6), f"{name}_r": round(result["r"], 6),
                    f"{name}_f1": round(result["f1"], 6)})
        if METRICS[name].shift:
            row[f"{name}_shift"] = result["shift"]
    return row
