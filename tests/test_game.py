"""GAME's text output converted to a board drop-in TSV, and the failing inputs that must be rejected.

These are CPU-only: they exercise the conversion/scoring plumbing (the same code a GPU run uses to land
GAME's notes on the scoreboard) without running the model. A GAME ``.txt`` is ``onset<TAB>offset<TAB>midi``
with no header; the board TSV is the same rows under an ``onset offset midi`` header, so a converted file must
load through the board's own drop-in loader and score.
"""

import tempfile
import unittest
from pathlib import Path

from open_transcriber.board import board, game


class GameConvertTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_txt(self, name: str, text: str) -> Path:
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_convert_txt_reads_rows_sorted(self):
        txt = self._write_txt("a.txt", "1.500\t1.800\t62\n0.100\t0.400\t60\n\n0.900\t1.000\t64\n")
        rows = game.convert_txt(txt)
        self.assertEqual(rows, [(0.1, 0.4, 60), (0.9, 1.0, 64), (1.5, 1.8, 62)])

    def test_converted_tsv_loads_through_board(self):
        txt = self._write_txt("song.txt", "0.100\t0.400\t60\n0.500\t0.700\t67\n")
        dest = self.root / "systems" / "game-small-stem" / "song.tsv"
        game.write_system_tsv(game.convert_txt(txt), dest)
        # The board's own drop-in loader must read what we wrote, with the header it expects.
        notes = board._tsv_loader(dest)()
        self.assertEqual(len(notes), 2)
        self.assertEqual(list(notes.midi), [60.0, 67.0])
        self.assertAlmostEqual(float(notes.onset[0]), 0.1)
        self.assertAlmostEqual(float(notes.offset[1]), 0.7)

    def test_convert_dir_skips_missing(self):
        self._write_txt("one.txt", "0.000\t0.200\t60\n")
        done = game.convert_dir(self.root, self.root / "sys", ["one", "two"])
        self.assertEqual(done, ["one"])
        self.assertTrue((self.root / "sys" / "one.tsv").exists())
        self.assertFalse((self.root / "sys" / "two.tsv").exists())

    def test_config_name(self):
        self.assertEqual(game.config_name("small", "stem"), "game-small-stem")
        self.assertEqual(game.config_name("large", "mix"), "game-large-mix")

    def test_config_name_rejects_unknown(self):
        with self.assertRaises(ValueError):
            game.config_name("huge", "mix")
        with self.assertRaises(ValueError):
            game.config_name("small", "demucs")

    def test_non_integer_pitch_is_rejected(self):
        # The failing input: GAME run without --round-pitch writes a note name or cents, not an integer MIDI.
        txt = self._write_txt("bad.txt", "0.100\t0.400\tC4+10\n")
        with self.assertRaises(ValueError):
            game.convert_txt(txt)
        frac = self._write_txt("frac.txt", "0.100\t0.400\t60.3\n")
        with self.assertRaises(ValueError):
            game.convert_txt(frac)

    def test_wrong_column_count_is_rejected(self):
        txt = self._write_txt("cols.txt", "0.100\t0.400\n")
        with self.assertRaises(ValueError):
            game.convert_txt(txt)


if __name__ == "__main__":
    unittest.main()
