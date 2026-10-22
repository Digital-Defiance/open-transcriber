"""The scoreboard: load the dev sets, score every system, cross-fit the onset shift, write the table.

Sets (frozen splits in ``open_transcriber/bakeoff/splits``):
- ``rwc-other-dev`` (30 songs): metrics ``pc_on50``, ``oct_on50`` (octave-exact, using ``midi_sung`` and the
  score's ``OCTAVE_EXCLUDED`` list), ``pc_on100``.
- ``karaoke-dev`` (12 songs): metrics ``pc_on50``, ``pc_on100``, ``pc_onoff50`` with the QA masks applied.

Systems:
- ``basic-pitch raw`` and ``basic-pitch one line`` from ``out/judge/basic-pitch-raw/<id>/notes.tsv``;
- ``open``/``neural`` at ``(grid)`` and ``(heard)`` times from
  ``out/judge/copyist-<engine>-audio/<id>/transcribe/events.json``;
- every drop-in ``out/judge/systems/<name>/<id>.tsv`` (header ``onset offset midi``), so a new engine plugs in
  without code.

Calibration (``--calibrate``): a constant onset shift per system, ``SHIFT_RANGE`` (±100 ms in 10 ms steps),
fitted on one set and reported on the other. The shift that maximises a set's lead metric (the first metric in
``SET_METRICS``) mean becomes that set's fitted shift; a system is then reported on each set using the shift
fitted on the *other* set. Fit and report are never the same set.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from open_transcriber.bakeoff import notes_io, paths, score

#: Candidate constant onset shifts for the cross-fit: -100..+100 ms in 10 ms steps.
SHIFT_RANGE = np.round(np.arange(-0.10, 0.10 + 1e-9, 0.01), 3).tolist()

#: The two frozen real dev sets: split file, reference kind, and the metrics each is scored on. The first
#: metric is the lead metric the cross-fit maximises and the history/markdown lead column.
SET_METRICS: dict[str, tuple[str, tuple[str, ...]]] = {
    "rwc-other-dev": ("rwc", ("pc_on50", "oct_on50", "pc_on100")),
    "karaoke-dev": ("usdx", ("pc_on50", "pc_on100", "pc_onoff50")),
}
SETS = tuple(SET_METRICS)


def _splits_dir() -> Path:
    from open_transcriber.bakeoff import splits  # noqa: F401 — the splits live beside this package

    return Path(splits.__path__[0])


def set_members(split: str) -> list[str]:
    """The item ids of a frozen split, in file order."""
    payload = json.loads((_splits_dir() / f"{split}.json").read_text(encoding="utf-8"))
    return list(payload["members"])


# --- references -------------------------------------------------------------------------------------------

def load_rwc_reference(item: str, data: Path) -> tuple[notes_io.Notes, np.ndarray, bool]:
    """(notes, octave-exact sung pitches, octave-excluded) for an RWC-other dev item."""
    ref = notes_io.load_reference(data / "refs" / "rwc-other-vocal" / f"{item}.tsv")
    sung = notes_io.load_reference(data / "refs" / "rwc-other-vocal" / f"{item}.tsv", octave_exact=True).midi
    return ref, sung, item in score.OCTAVE_EXCLUDED


def load_karaoke_reference(item: str, data: Path) -> notes_io.Notes:
    """A karaoke dev reference with its QA masks applied (freestyle and QA-failed notes become don't-cares)."""
    qa = json.loads((data / "qa" / f"{item}.json").read_text(encoding="utf-8"))
    masks = {n["id"]: n["mask"] for n in qa["notes"] if n.get("mask")}
    return notes_io.load_reference(data / "refs" / "usdx" / f"{item}.tsv", masks=masks)


# --- systems ----------------------------------------------------------------------------------------------

@dataclass
class System:
    """A named transcriber and, per item, a loader that returns its estimated notes (or None if absent)."""

    name: str
    load: Callable[[str], notes_io.Notes | None]


def _tsv_loader(path: Path) -> Callable[[], notes_io.Notes]:
    def load() -> notes_io.Notes:
        rows = []
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                rows.append((float(row["onset"]), float(row["offset"]), float(row["midi"])))
        rows.sort()
        return notes_io._validate(
            notes_io.Notes([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]), False, path.name)

    return load


def mono_bp(path: Path, min_dur: float = 0.06, merge_gap: float = 0.03, jump: float = 0.04,
            lo: int = 45, hi: int = 84) -> notes_io.Notes:
    """One melody line from Basic Pitch's notes (the prototype's ``mono_bp``): within the singing range,
    overlaps resolved by amplitude with a leap cost, same-pitch fragments merged, short notes dropped."""
    raw = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            raw.append((float(row["onset"]), float(row["offset"]), int(float(row["midi"])), float(row["amplitude"])))
    notes = [n for n in raw if lo <= n[2] <= hi and n[1] - n[0] >= min_dur]
    notes.sort(key=lambda n: (n[0], -n[3]))
    line: list[tuple] = []
    for n in notes:
        if line and n[0] < line[-1][1] - 0.02:
            prev = line[-2][2] if len(line) > 1 else line[-1][2]
            cur = line[-1]
            score_new = n[3] - jump * abs(n[2] - prev)
            score_cur = cur[3] - jump * abs(cur[2] - prev)
            if score_new > score_cur and n[0] - cur[0] > 0.05:
                line[-1] = (cur[0], n[0], cur[2], cur[3])
                line.append(n)
            elif score_new > score_cur:
                line[-1] = n
            continue
        if line and n[2] == line[-1][2] and n[0] - line[-1][1] < merge_gap and n[3] <= line[-1][3] * 1.15:
            line[-1] = (line[-1][0], max(line[-1][1], n[1]), n[2], max(line[-1][3], n[3]))
            continue
        line.append(n)
    rows = sorted((a, b, p) for a, b, p, _ in line if b - a >= min_dur)
    return notes_io._validate(
        notes_io.Notes([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]), False, "one line")


def discover_systems(out: Path) -> list[System]:
    """Every system to score: the built-in ones, then each drop-in ``systems/<name>/`` folder (sorted)."""
    systems: list[System] = []

    def bp_raw(item: str) -> notes_io.Notes | None:
        path = out / "basic-pitch-raw" / item / "notes.tsv"
        return notes_io.load_basic_pitch(path) if path.exists() else None

    def bp_one(item: str) -> notes_io.Notes | None:
        path = out / "basic-pitch-raw" / item / "notes.tsv"
        return mono_bp(path) if path.exists() else None

    systems.append(System("basic-pitch raw", bp_raw))
    systems.append(System("basic-pitch one line", bp_one))

    for engine in ("open", "neural"):
        def events(item: str, engine: str = engine) -> Path:
            return out / f"copyist-{engine}-audio" / item / "transcribe" / "events.json"

        def grid(item: str, events=events) -> notes_io.Notes | None:
            path = events(item)
            return notes_io.load_copyist_events(path) if path.exists() else None

        def heard(item: str, events=events) -> notes_io.Notes | None:
            path = events(item)
            return notes_io.load_copyist_events(path, heard=True) if path.exists() else None

        systems.append(System(f"{engine} (grid)", grid))
        systems.append(System(f"{engine} (heard)", heard))

    systems_dir = out / "systems"
    if systems_dir.is_dir():
        for folder in sorted(p for p in systems_dir.iterdir() if p.is_dir()):
            def drop_in(item: str, folder=folder) -> notes_io.Notes | None:
                path = folder / f"{item}.tsv"
                return _tsv_loader(path)() if path.exists() else None

            systems.append(System(folder.name, drop_in))
    return systems


# --- scoring ----------------------------------------------------------------------------------------------

def shift_notes(est: notes_io.Notes, seconds: float) -> notes_io.Notes:
    """A copy of ``est`` with every onset and offset moved by ``seconds`` (durations kept), clipped at 0."""
    if seconds == 0:
        return est
    onset = np.maximum(est.onset + seconds, 0.0)
    offset = np.maximum(est.offset + seconds, onset + notes_io.MIN_DURATION)
    out = notes_io.Notes(onset, offset, est.midi.copy(), est.mask.copy(), 0, dict(est.meta))
    return out


@dataclass
class Board:
    """The scored board: per (set, system) a list of rows, one per item, keyed ``set|system``."""

    rows: dict[str, list[dict]] = field(default_factory=dict)
    shifts: dict[str, dict[str, float]] = field(default_factory=dict)  # system -> set -> fitted shift (s)

    def key(self, split: str, system: str) -> str:
        return f"{split}|{system}"


def _score_item(split: str, kind: str, item: str, est: notes_io.Notes, data: Path) -> dict:
    metrics = SET_METRICS[split][1]
    if kind == "rwc":
        ref, sung, excluded = load_rwc_reference(item, data)
        return score.score_pair(ref, est, metrics, sung=sung, octave_excluded=excluded, item=item)
    ref = load_karaoke_reference(item, data)
    return score.score_pair(ref, est, metrics, item=item)


def lead_metric(split: str) -> str:
    """The first metric of a set: the cross-fit objective and the markdown/history lead column."""
    return SET_METRICS[split][1][0]


def _mean_lead(rows: list[dict], split: str) -> float:
    return score.mean_f1(rows, lead_metric(split))


def _onset_residual(split: str, kind: str, items: list[str], est: dict, shift: float, data: Path) -> float:
    """Mean absolute onset error (seconds) of the matched notes at ``shift``: the cross-fit's tie-break.

    When several shifts tie on the lead F1 (a uniform offset smaller than the onset tolerance leaves a
    plateau), the one that best realigns the matched onsets wins, so a +40 ms system recovers -40 ms.
    """
    diffs = []
    for item in items:
        shifted = shift_notes(est[item], shift)
        metric = score.METRICS[lead_metric(split)]
        if kind == "rwc":
            ref, sung, _ = load_rwc_reference(item, data)
            ref_pitch = ref.midi if metric.pitch == "pc" else sung
        else:
            ref = load_karaoke_reference(item, data)
            ref_pitch = ref.midi
        scored = ref.scored
        unmasked = ref.select(scored)
        for ref_i, est_i in score.match(unmasked, shifted, metric, ref_pitch[scored]):
            diffs.append(abs(float(unmasked.onset[ref_i]) - float(shifted.onset[est_i])))
    return float(np.mean(diffs)) if diffs else float("inf")


def build(calibrate: bool = False, data: Path | None = None, out: Path | None = None) -> Board:
    """Score every system on both sets. With ``calibrate``, add cross-fitted onset-shifted variants."""
    data = data or paths.data_dir()
    out = out or paths.out_dir()
    systems = discover_systems(out)
    board = Board()

    # Base (unshifted) scores, and — for calibration — cached estimates per (set, system, item).
    estimates: dict[tuple[str, str, str], notes_io.Notes] = {}
    for split in SETS:
        kind = SET_METRICS[split][0]
        for item in set_members(split):
            for sys in systems:
                est = sys.load(item)
                if est is None:
                    continue
                estimates[(split, sys.name, item)] = est
                row = _score_item(split, kind, item, est, data)
                board.rows.setdefault(board.key(split, sys.name), []).append(row)

    if not calibrate:
        return board

    # For each system and set, the best constant shift by the set's own lead metric (the fit candidate).
    fitted: dict[str, dict[str, float]] = {}
    for sys in systems:
        fitted[sys.name] = {}
        for split in SETS:
            kind = SET_METRICS[split][0]
            items = [i for i in set_members(split) if (split, sys.name, i) in estimates]
            if not items:
                continue
            est = {i: estimates[(split, sys.name, i)] for i in items}
            best_shift, best_mean, best_resid = 0.0, -1.0, float("inf")
            for shift in SHIFT_RANGE:
                rows = [_score_item(split, kind, i, shift_notes(est[i], shift), data) for i in items]
                mean = _mean_lead(rows, split)
                if mean < best_mean - 1e-12:
                    continue
                resid = _onset_residual(split, kind, items, est, shift, data)
                better = mean > best_mean + 1e-12 or (abs(mean - best_mean) <= 1e-12 and resid < best_resid - 1e-12)
                if better:
                    best_mean, best_shift, best_resid = mean, shift, resid
            fitted[sys.name][split] = best_shift

    # Cross-fit: report each set using the shift fitted on the OTHER set.
    for sys in systems:
        for split in SETS:
            other = next(s for s in SETS if s != split)
            if other not in fitted[sys.name]:
                continue
            shift = fitted[sys.name][other]
            if shift == 0.0:
                continue
            kind = SET_METRICS[split][0]
            items = [i for i in set_members(split) if (split, sys.name, i) in estimates]
            if not items:
                continue
            name = f"{sys.name} (xfit {shift * 1000:+.0f} ms)"
            board.shifts.setdefault(name, {})[split] = shift
            for item in items:
                row = _score_item(split, kind, item, shift_notes(estimates[(split, sys.name, item)], shift), data)
                board.rows.setdefault(board.key(split, name), []).append(row)
    return board


# --- output -----------------------------------------------------------------------------------------------

def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def to_markdown(board: Board) -> str:
    """One table per set: system, n, then each of that set's metrics as a mean F1 percentage."""
    blocks = []
    for split in SETS:
        metrics = SET_METRICS[split][1]
        header = "| System | n | " + " | ".join(metrics) + " |"
        rule = "|---|---|" + "---|" * len(metrics)
        lines = [f"### {split}", "", header, rule]
        names = sorted(name for key in board.rows for (s, name) in [key.split("|", 1)] if s == split)
        for name in names:
            rows = board.rows[f"{split}|{name}"]
            cells = [f"{100 * score.mean_f1(rows, m):.1f}" for m in metrics]
            lines.append(f"| {name} | {len(rows)} | " + " | ".join(cells) + " |")
        blocks.append("\n".join(lines))
    return "# Transcriber scoreboard v1\n\n" + "\n\n".join(blocks) + "\n"


def to_json(board: Board) -> dict:
    data = {"sets": {}, "shifts": board.shifts}
    for split in SETS:
        metrics = SET_METRICS[split][1]
        systems = {}
        names = sorted(name for key in board.rows for (s, name) in [key.split("|", 1)] if s == split)
        for name in names:
            rows = board.rows[f"{split}|{name}"]
            systems[name] = {"n": len(rows), **{m: round(100 * score.mean_f1(rows, m), 3) for m in metrics}}
        data["sets"][split] = {"metrics": list(metrics), "systems": systems}
    return data


def _history_entry(board: Board, data_json: dict) -> dict:
    root = paths.copyist_root()
    submodule = root / "open-transcriber"
    rows = {}
    for split, info in data_json["sets"].items():
        rows[split] = {name: {m: s[m] for m in info["metrics"]} for name, s in info["systems"].items()}
    return {
        "date": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "copyist_commit": _git_commit(root),
        "open_transcriber_commit": _git_commit(submodule),
        "rows": rows,
    }


def write(board: Board, out: Path | None = None) -> dict:
    """Write ``scoreboard.md`` and ``.json`` and append to ``scoreboard-history.jsonl``; return the JSON."""
    out = out or paths.out_dir()
    out.mkdir(parents=True, exist_ok=True)
    data_json = to_json(board)
    (out / "scoreboard.md").write_text(to_markdown(board), encoding="utf-8")
    (out / "scoreboard.json").write_text(json.dumps(data_json, indent=1) + "\n", encoding="utf-8")
    with open(out / "scoreboard-history.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_history_entry(board, data_json)) + "\n")
    return data_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibrate", action="store_true",
                        help="add cross-fitted constant-onset-shift variants (fit on one set, report the other)")
    parser.add_argument("--data", type=Path, default=None, help="the judge data folder (default: .cache/judge)")
    parser.add_argument("--out", type=Path, default=None, help="the output folder (default: out/judge)")
    args = parser.parse_args(argv)
    board = build(calibrate=args.calibrate, data=args.data, out=args.out)
    data_json = write(board, out=args.out)
    print(to_markdown(board))
    return 0 if data_json["sets"] else 1
