"""The measured timing correction: applied to heard-time notes, and the failing input it must reject.

CPU-only. ``correct_heard`` shifts every onset/offset by a constant measured on dev; the board applies the
same cross-fitted shift under a stable ``… corrected`` name. A fake heard-time system shifted +40 ms is
re-aligned to F1 1.0 by a -40 ms correction (the number the cross-fit recovers), and a corrected system shows
up on the board. A non-finite shift is rejected.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io
from open_transcriber.board import board, corrections


class CorrectHeardTests(unittest.TestCase):
    def test_shifts_onsets_and_offsets(self):
        notes = notes_io.Notes([1.0, 2.0], [1.2, 2.2], [60, 62])
        out = corrections.correct_heard(notes, -0.040)
        self.assertTrue(np.allclose(out.onset, [0.96, 1.96]))
        self.assertTrue(np.allclose(out.offset, [1.16, 2.16]))
        # Durations are kept.
        self.assertTrue(np.allclose(out.offset - out.onset, [0.2, 0.2]))

    def test_zero_shift_is_identity(self):
        notes = notes_io.Notes([1.0], [1.2], [60])
        out = corrections.correct_heard(notes, 0.0)
        self.assertTrue(np.allclose(out.onset, notes.onset))

    def test_non_finite_shift_is_rejected(self):
        notes = notes_io.Notes([1.0], [1.2], [60])
        with self.assertRaises(ValueError):
            corrections.correct_heard(notes, float("nan"))
        with self.assertRaises(ValueError):
            corrections.correct_heard(notes, float("inf"))


# Two real members per split so ``set_members`` resolves and the board builds against a fake data folder.
RWC_ITEMS = ["rwc-c045", "rwc-c048"]
KARAOKE_ITEMS = ["usdx-jonathan-coulton-big-bad-world-one", "usdx-jonathan-coulton-chiron-beta-prime"]
INDUCED = 0.040


def _dense(n=24, step=0.12, dur=0.10, base=60):
    return [(1.0 + i * step, 1.0 + i * step + dur, base + (i % 7)) for i in range(n)]


def _write_ref(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(("note_id", "onset", "offset", "midi", "midi_sung", "mask", "source"))]
    for i, (on, off, midi) in enumerate(rows):
        lines.append(f"{i}\t{on:.6f}\t{off:.6f}\t{midi}\t{midi}\t\tfake")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_events(path: Path, rows, shift):
    """A copyist ``events.json`` whose heard times are the reference shifted by ``shift``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    notes = [{"midi": m, "onset": on, "offset": off,
              "heard_onset": on + shift, "heard_offset": off + shift} for on, off, m in rows]
    path.write_text(json.dumps({"format": notes_io.COPYIST_FORMAT, "engine": "open", "notes": notes}),
                    encoding="utf-8")


def _fake_qa(path: Path, n):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"item": path.stem, "notes": [{"id": i, "mask": ""} for i in range(n)]}),
                    encoding="utf-8")


class CorrectedBoardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.data = root / "data"
        self.out = root / "out"
        for item in RWC_ITEMS:
            rows = _dense()
            _write_ref(self.data / "refs" / "rwc-other-vocal" / f"{item}.tsv", rows)
            _write_events(self.out / "copyist-open-audio" / item / "transcribe" / "events.json", rows, INDUCED)
        for item in KARAOKE_ITEMS:
            rows = _dense()
            _write_ref(self.data / "refs" / "usdx" / f"{item}.tsv", rows)
            _fake_qa(self.data / "qa" / f"{item}.json", len(rows))
            _write_events(self.out / "copyist-open-audio" / item / "transcribe" / "events.json", rows, INDUCED)

    def tearDown(self):
        self._tmp.cleanup()

    def test_corrected_system_reaches_one(self):
        b = board.build(calibrate=True, data=self.data, out=self.out)
        name = "open (heard, corrected)"
        self.assertIn(name, {k.split("|", 1)[1] for k in b.rows})
        for split in board.SETS:
            rows = b.rows[b.key(split, name)]
            # The +40 ms heard times are pulled back to the reference, so F1 is 1.0.
            self.assertAlmostEqual(board.score.mean_f1(rows, "pc_on50"), 1.0)
            # The applied shift is the one fitted on the OTHER set: -40 ms here.
            self.assertAlmostEqual(b.shifts[name][split], -INDUCED, places=6)

    def test_fit_dev_shifts_reads_corrected_only(self):
        b = board.build(calibrate=True, data=self.data, out=self.out)
        shifts = corrections.fit_dev_shifts(b)
        self.assertIn("open (heard, corrected)", shifts)
        self.assertNotIn("open (heard)", shifts)
        for split in board.SETS:
            self.assertAlmostEqual(shifts["open (heard, corrected)"][split], -INDUCED, places=6)


if __name__ == "__main__":
    unittest.main()
