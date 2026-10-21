"""Tests for the UltraStar chart parser, reference rows and the words file (made-up syllables only)."""

import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import ultrastar

# A made-up chart: syllables are "la", "li" and "lo"; BPM with a comma decimal, a fractional GAP.
CHART = """#TITLE:Test Tune
#ARTIST:Nobody
#MP3:audio.mp3
#BPM:300,5
#GAP:1234.5
: 0 4 5 la
* 4 2 7 li
F 8 2 0 lo
- 12
: 12 0 9  la
: 14 2 9 ~
: 16 3 -3  lo
E
: 99 1 1 never
"""

BEAT = 15.0 / 300.5
GAP = 1.2345


def write_chart(text: str) -> Path:
    folder = Path(tempfile.mkdtemp())
    (folder / "song.txt").write_text(text, encoding="utf-8")
    (folder / "license.txt").write_text("License: none\n", encoding="utf-8")
    return folder / "song.txt"


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.chart = ultrastar.parse(write_chart(CHART))

    def test_header_and_counts(self):
        self.assertEqual(self.chart.header["TITLE"], "Test Tune")
        self.assertAlmostEqual(self.chart.bpm, 300.5)
        self.assertAlmostEqual(self.chart.gap, GAP)
        self.assertEqual(len(self.chart.notes), 6)  # nothing after E
        self.assertEqual(self.chart.lines, 2)

    def test_exact_times(self):
        first, golden, free, held, tilde, low = self.chart.notes
        self.assertAlmostEqual(first.onset, GAP)
        self.assertAlmostEqual(first.offset, GAP + 4 * BEAT)
        self.assertAlmostEqual(golden.onset, GAP + 4 * BEAT)
        self.assertAlmostEqual(golden.offset, GAP + 6 * BEAT)
        self.assertAlmostEqual(low.onset, GAP + 16 * BEAT)
        self.assertAlmostEqual(low.offset, GAP + 19 * BEAT)
        self.assertEqual(first.midi, 65)
        self.assertEqual(low.midi, 57)

    def test_zero_length_is_one_grid_step(self):
        held = self.chart.notes[3]
        self.assertEqual(held.length, 0)
        self.assertAlmostEqual(held.offset - held.onset, BEAT)

    def test_freestyle_is_masked(self):
        rows = ultrastar.reference_rows(self.chart)
        self.assertEqual([r["mask"] for r in rows], ["", "", "freestyle", "", "", ""])
        self.assertEqual(rows[1]["source"], "chart:*")

    def test_syllables_stay_out_of_repr(self):
        text = repr(self.chart.notes[0])
        self.assertNotIn("syllable", text)
        self.assertNotIn("'la'", text)

    def test_words_file(self):
        text = ultrastar.lyrics_text(self.chart)
        self.assertEqual(text, "lalilo\nla lo\n")
        with tempfile.TemporaryDirectory() as tmp:
            counts = ultrastar.write_lyrics(self.chart, Path(tmp) / "x.txt")
            self.assertEqual(counts, {"lines": 2, "words": 3})

    def test_chart_path(self):
        path = write_chart(CHART)
        self.assertEqual(ultrastar.chart_path(path.parent), path)


class RefusalTests(unittest.TestCase):
    def test_relative_mode_raises(self):
        # Should fail: relative-mode charts aren't supported.
        with self.assertRaises(ValueError):
            ultrastar.parse(write_chart("#RELATIVE:yes\n" + CHART))

    def test_missing_bpm_raises(self):
        # Should fail: no #BPM, no time base.
        with self.assertRaises(ValueError):
            ultrastar.parse(write_chart(CHART.replace("#BPM:300,5\n", "")))

    def test_duet_raises(self):
        with self.assertRaises(ValueError):
            ultrastar.parse(write_chart(CHART.replace(": 0 4 5 la", "P1\n: 0 4 5 la")))

    def test_malformed_note_raises(self):
        with self.assertRaises(ValueError):
            ultrastar.parse(write_chart(CHART.replace(": 0 4 5 la", ": 0 four 5 la")))


if __name__ == "__main__":
    unittest.main()
