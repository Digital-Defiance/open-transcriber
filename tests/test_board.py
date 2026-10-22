"""The scoreboard on a tiny fake data folder, and the cross-fit property.

A fake system is each set's reference shifted +40 ms. With dense reference notes the only constant shift that
realigns every note is -40 ms, so the cross-fit (fit on one set, report on the other) recovers -40 ms and the
reported F1 is 1.0. Fit and report are always different sets.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_transcriber.board import board


# Two real members from each frozen split, so ``set_members`` resolves.
RWC_ITEMS = ["rwc-c045", "rwc-c048"]
KARAOKE_ITEMS = ["usdx-jonathan-coulton-big-bad-world-one", "usdx-jonathan-coulton-chiron-beta-prime"]
INDUCED = 0.040  # the fake system shifts every onset/offset this much later


def _dense_notes(n=24, step=0.12, dur=0.10, base_midi=60):
    """A melody with notes close enough (120 ms apart) that only the right shift realigns them all."""
    rows = []
    for i in range(n):
        onset = 1.0 + i * step
        rows.append((onset, onset + dur, base_midi + (i % 7)))
    return rows


def _write_ref(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(("note_id", "onset", "offset", "midi", "midi_sung", "mask", "source"))]
    for i, (on, off, midi) in enumerate(rows):
        lines.append(f"{i}\t{on:.6f}\t{off:.6f}\t{midi}\t{midi}\t\tfake")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_system(path: Path, rows, shift):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(("onset", "offset", "midi"))]
    for on, off, midi in rows:
        lines.append(f"{on + shift:.6f}\t{off + shift:.6f}\t{midi}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fake_qa(path: Path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"item": path.stem, "notes": [{"id": i, "mask": ""} for i in range(n)]}),
                    encoding="utf-8")


class BoardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data = root / "data"
        self.out = root / "out"
        self.rows = {}
        for item in RWC_ITEMS:
            rows = _dense_notes()
            self.rows[item] = rows
            _write_ref(self.data / "refs" / "rwc-other-vocal" / f"{item}.tsv", rows)
            _write_system(self.out / "systems" / "fake" / f"{item}.tsv", rows, INDUCED)
        for item in KARAOKE_ITEMS:
            rows = _dense_notes()
            self.rows[item] = rows
            _write_ref(self.data / "refs" / "usdx" / f"{item}.tsv", rows)
            _fake_qa(self.data / "qa" / f"{item}.json", len(rows))
            _write_system(self.out / "systems" / "fake" / f"{item}.tsv", rows, INDUCED)

    def tearDown(self):
        self._tmp.cleanup()

    def test_base_system_is_offset(self):
        b = board.build(calibrate=False, data=self.data, out=self.out)
        # +40 ms is within the 50 ms tolerance, so the uncalibrated fake still matches every note.
        for split in board.SETS:
            rows = b.rows[b.key(split, "fake")]
            self.assertEqual(len(rows), 2)
            self.assertAlmostEqual(board.score.mean_f1(rows, "pc_on50"), 1.0)

    def test_cross_fit_finds_minus_40ms(self):
        # Fit on one set, report on the other; the onset-residual tie-break recovers -40 ms on each set.
        est = {item: board._tsv_loader(self.out / "systems" / "fake" / f"{item}.tsv")()
               for item in RWC_ITEMS + KARAOKE_ITEMS}
        for split, items in (("rwc-other-dev", RWC_ITEMS), ("karaoke-dev", KARAOKE_ITEMS)):
            kind = board.SET_METRICS[split][0]
            best_shift, best_mean, best_resid = 0.0, -1.0, float("inf")
            for shift in board.SHIFT_RANGE:
                scored = [board._score_item(split, kind, i, board.shift_notes(est[i], shift), self.data)
                          for i in items]
                mean = board._mean_lead(scored, split)
                if mean < best_mean - 1e-12:
                    continue
                resid = board._onset_residual(split, kind, items, est, shift, self.data)
                if mean > best_mean + 1e-12 or (abs(mean - best_mean) <= 1e-12 and resid < best_resid - 1e-12):
                    best_mean, best_shift, best_resid = mean, shift, resid
            self.assertAlmostEqual(best_shift, -INDUCED, places=6, msg=f"{split} fit")
            self.assertAlmostEqual(best_mean, 1.0, places=6)

    def test_calibrated_cross_fit_reports_one(self):
        b = board.build(calibrate=True, data=self.data, out=self.out)
        name = f"fake (xfit {-INDUCED * 1000:+.0f} ms)"
        self.assertIn(name, {k.split("|", 1)[1] for k in b.rows})
        for split in board.SETS:
            rows = b.rows[b.key(split, name)]
            self.assertAlmostEqual(board.score.mean_f1(rows, "pc_on50"), 1.0)
            self.assertAlmostEqual(b.shifts[name][split], -INDUCED, places=6)

    def test_write_outputs(self):
        b = board.build(calibrate=True, data=self.data, out=self.out)
        board.write(b, out=self.out)
        self.assertTrue((self.out / "scoreboard.md").exists())
        data = json.loads((self.out / "scoreboard.json").read_text())
        self.assertIn("rwc-other-dev", data["sets"])
        self.assertIn("fake", data["sets"]["rwc-other-dev"]["systems"])
        history = (self.out / "scoreboard-history.jsonl").read_text().splitlines()
        self.assertEqual(len(history), 1)
        entry = json.loads(history[0])
        self.assertIn("rows", entry)
        self.assertIn("date", entry)


if __name__ == "__main__":
    unittest.main()
