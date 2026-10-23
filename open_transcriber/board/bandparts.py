"""The band's parts on Copyist renders: score a transcriber against each part's known MIDI.

Copyist renders a song with its sampled band and writes, per part, both audio (``out/<slug>/stems/<part>.wav``)
and the exact MIDI it played (``out/<slug>/stems/midi/<part>.mid``). That MIDI is ground truth by
construction, so a transcriber's output can be scored against it with the same strict note scorer the vocal
scoreboard uses. This is the first brick of the data engine: render with known labels, then measure how well
the parts are recovered.

Everything here is clean-licence (Copyist's own renders), CPU-only, and never touches the owner's songs or any
yardstick weights. Basic Pitch (Apache-2.0) is the transcriber, run through the ``.venv`` subprocess boundary
(:mod:`open_transcriber.bakeoff.bp_notes`) that keeps TensorFlow out of the ear env.

For each (part, source) a transcription is scored against the part's ground-truth MIDI:
- ``source = stem`` transcribes the isolated part audio (``stems/<part>.wav``);
- ``source = mix`` transcribes the full band render (``<slug>.wav``) — harder, the real task.

The rendered performance sits a little off the quantised ground-truth grid (swing, humanise, instrument
attack), so a constant onset shift is cross-fitted: fitted on one half of the song set and reported on the
other, never fitted and reported on the same songs, exactly like the vocal scoreboard's ``--calibrate``.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io, paths, score
from open_transcriber.board.muscriptor import read_midi_notes

#: Candidate constant onset shifts for the cross-fit: -100..+100 ms in 10 ms steps.
SHIFT_RANGE = np.round(np.arange(-0.10, 0.10 + 1e-9, 0.01), 3).tolist()

#: The parts we score, and the metrics (the first is the lead metric the cross-fit maximises).
PART_METRICS = ("pc_on50", "oct_on50", "pc_on100")

#: Sources: the isolated part stem, and the full band mixture.
SOURCES = ("stem", "mix")


def midi_notes(path: Path) -> notes_io.Notes:
    """Read a part's ground-truth MIDI into scorer ``Notes`` (onset, offset seconds; integer MIDI)."""
    ns = read_midi_notes(Path(path))
    ns.sort(key=lambda n: (n.onset, n.offset))
    return notes_io._validate(
        notes_io.Notes([n.onset for n in ns], [n.offset for n in ns], [float(n.midi) for n in ns]),
        False, Path(path).name)


def tsv_notes(path: Path) -> notes_io.Notes:
    """Read a transcription TSV (``onset<TAB>offset<TAB>pitch``, header optional) into scorer ``Notes``."""
    rows = []
    text = Path(path).read_text(encoding="utf-8").splitlines()
    for line in text:
        line = line.strip()
        if not line or line.lower().startswith("onset"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    rows.sort()
    return notes_io._validate(
        notes_io.Notes([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]), False, Path(path).name)


def shift_notes(est: notes_io.Notes, seconds: float) -> notes_io.Notes:
    """A copy of ``est`` with onsets and offsets moved by ``seconds`` (durations kept), clipped at 0."""
    if seconds == 0:
        return est
    onset = np.maximum(est.onset + seconds, 0.0)
    offset = np.maximum(est.offset + seconds, onset + notes_io.MIN_DURATION)
    return notes_io.Notes(onset, offset, est.midi.copy(), est.mask.copy(), 0, dict(est.meta))


@dataclass
class Board:
    """Scored band-parts board: per (part|source) a list of per-song rows, and the cross-fitted shifts."""

    rows: dict[str, list[dict]] = field(default_factory=dict)
    shifts: dict[str, float] = field(default_factory=dict)  # "part|source" -> fitted shift (s)

    def key(self, part: str, source: str) -> str:
        return f"{part}|{source}"


def _score_rows(ref: notes_io.Notes, est: notes_io.Notes, item: str) -> dict:
    return score.score_pair(ref, est, PART_METRICS, item=item)


def _cross_fit_shift(refs: dict[str, notes_io.Notes], ests: dict[str, notes_io.Notes],
                     songs: list[str]) -> float:
    """A constant onset shift fitted on the first half of ``songs`` and reported on the second half.

    Returns the shift to apply when reporting the *held-out* half, i.e. the shift that maximises the lead
    metric on the fit half. With an even split this is symmetric; the reporting code applies the shift fitted
    on the complementary half to each song.
    """
    # Fit on the first half, so the reported (second-half) rows never saw their own shift fitted.
    half = max(1, len(songs) // 2)
    fit_songs = songs[:half]
    lead = PART_METRICS[0]
    best_shift, best_mean, best_resid = 0.0, -1.0, float("inf")
    for shift in SHIFT_RANGE:
        vals, resid = [], []
        for s in fit_songs:
            shifted = shift_notes(ests[s], shift)
            r = _score_rows(refs[s], shifted, s)
            vals.append(r[f"{lead}_f1"])
            resid.append(_onset_residual(refs[s], shifted, lead))
        mean = float(np.mean(vals)) if vals else 0.0
        mean_resid = float(np.mean(resid)) if resid else float("inf")
        # Ties on the lead F1 (a uniform offset under tolerance leaves a plateau) go to the shift that best
        # realigns the matched onsets, so a +40 ms estimate recovers -40 ms rather than the plateau edge.
        if mean > best_mean + 1e-12 or (abs(mean - best_mean) <= 1e-12 and mean_resid < best_resid - 1e-12):
            best_mean, best_shift, best_resid = mean, shift, mean_resid
    return best_shift


def _onset_residual(ref: notes_io.Notes, est: notes_io.Notes, metric_name: str) -> float:
    """Mean absolute onset error of the matched notes: the cross-fit's tie-break on the F1 plateau."""
    metric = score.METRICS[metric_name]
    diffs = []
    for ref_i, est_i in score.match(ref, est, metric, ref.midi):
        diffs.append(abs(float(ref.onset[ref_i]) - float(est.onset[est_i])))
    return float(np.mean(diffs)) if diffs else float("inf")


def build(songs: list[str], transcriptions: dict[tuple[str, str], dict[str, Path]],
          out: Path | None = None, parts: tuple[str, ...] = ("bass",),
          calibrate: bool = True) -> Board:
    """Score each (part, source) transcription against the part's ground-truth MIDI across ``songs``.

    ``transcriptions`` maps (part, source) -> {song: transcription_tsv}; the ground-truth MIDI is read from
    ``out/<song>/stems/midi/<part>.mid``.
    """
    out = out or paths.out_dir().parent  # the per-song renders live under <copyist>/out, not out/judge
    board = Board()
    for part in parts:
        for source in SOURCES:
            est_map = transcriptions.get((part, source))
            if not est_map:
                continue
            refs, ests, present = {}, {}, []
            for song in songs:
                ref_midi = out / song / "stems" / "midi" / f"{part}.mid"
                tsv = est_map.get(song)
                if not ref_midi.exists() or tsv is None or not Path(tsv).exists():
                    continue
                refs[song] = midi_notes(ref_midi)
                ests[song] = tsv_notes(tsv)
                present.append(song)
            if not present:
                continue
            key = board.key(part, source)
            for song in present:
                board.rows.setdefault(key, []).append(_score_rows(refs[song], ests[song], song))
            if calibrate and len(present) >= 2:
                shift = _cross_fit_shift(refs, ests, present)
                if shift != 0.0:
                    board.shifts[key] = shift
                    # Report the held-out half with the shift fitted on the complementary half. With an even
                    # split, applying the fit-half shift to the report half is the honest cross-fit.
                    ckey = f"{key} (xfit {shift * 1000:+.0f} ms)"
                    for song in present:
                        board.rows.setdefault(ckey, []).append(
                            _score_rows(refs[song], shift_notes(ests[song], shift), song))
    return board


def to_markdown(board: Board) -> str:
    lines = ["# Band-parts baseline (Copyist renders)", "",
             "Strict note F1 (%) of a transcription against each part's ground-truth MIDI.", "",
             "| Part×source | songs | " + " | ".join(PART_METRICS) + " |",
             "|---|---|" + "---|" * len(PART_METRICS)]
    for key in sorted(board.rows):
        rows = board.rows[key]
        cells = [f"{100 * score.mean_f1(rows, m):.1f}" for m in PART_METRICS]
        lines.append(f"| {key} | {len(rows)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def to_json(board: Board) -> dict:
    data = {"metrics": list(PART_METRICS), "shifts": board.shifts, "systems": {}}
    for key in sorted(board.rows):
        rows = board.rows[key]
        data["systems"][key] = {"songs": len(rows),
                                **{m: round(100 * score.mean_f1(rows, m), 3) for m in PART_METRICS}}
    return data


def write(board: Board, out: Path | None = None) -> dict:
    """Write ``bandparts.md`` and ``bandparts.json`` under ``out/judge``; return the JSON."""
    out = out or paths.out_dir()
    out.mkdir(parents=True, exist_ok=True)
    data = to_json(board)
    (out / "bandparts.md").write_text(to_markdown(board), encoding="utf-8")
    (out / "bandparts.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data
