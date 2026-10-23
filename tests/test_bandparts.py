"""Band-parts baseline scoring: ground-truth part MIDI vs a transcription, and the input that must fail.

CPU-only: these exercise the MIDI/TSV readers, the strict scoring, and the cross-fit, without building a song
or running Basic Pitch. A part's ground truth is the MIDI Copyist's band played; a transcription is an
``onset offset pitch`` TSV. A self-scored part must reach F1 1.0, and a transcription shifted +40 ms must be
recovered by the cross-fit's -40 ms.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import score
from open_transcriber.board import bandparts


def _vlq(n: int) -> bytes:
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _part_midi(path: Path, pitches_at: list[tuple[int, int]], tpb: int = 480) -> None:
    """Write a one-track SMF with notes at (onset_tick, pitch), each one beat long, 120 BPM."""
    ev = bytearray()
    ev += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")
    last = 0
    for tick, pitch in pitches_at:
        ev += _vlq(tick - last) + bytes([0x90, pitch, 90])
        ev += _vlq(tpb) + bytes([0x80, pitch, 0])
        last = tick + tpb
    ev += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + struct.pack(">I", len(ev)) + bytes(ev)
    path.write_bytes(b"MThd" + struct.pack(">IHHH", 6, 1, 1, tpb) + track)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_midi_and_tsv_read_the_same_notes(self):
        _part_midi(self.root / "bass.mid", [(0, 48), (480, 50)])
        ref = bandparts.midi_notes(self.root / "bass.mid")
        self.assertEqual(list(ref.midi), [48.0, 50.0])
        (self.root / "bass.tsv").write_text("0.000\t0.500\t48\n0.500\t1.000\t50\n", encoding="utf-8")
        est = bandparts.tsv_notes(self.root / "bass.tsv")
        self.assertEqual(list(est.midi), [48.0, 50.0])

    def test_tsv_reader_skips_header(self):
        (self.root / "h.tsv").write_text("onset\toffset\tmidi\n0.0\t0.5\t48\n", encoding="utf-8")
        est = bandparts.tsv_notes(self.root / "h.tsv")
        self.assertEqual(list(est.midi), [48.0])

    def test_not_a_midi_file_rejected(self):
        # The failing input: the render's part MIDI is missing/corrupt.
        (self.root / "bad.mid").write_bytes(b"not a midi")
        with self.assertRaises(ValueError):
            bandparts.midi_notes(self.root / "bad.mid")


class ScoringTests(unittest.TestCase):
    def test_self_score_is_perfect(self):
        import numpy as np
        onset = np.array([0.0, 0.5, 1.0])
        from open_transcriber.bakeoff import notes_io
        ref = notes_io._validate(notes_io.Notes(onset, onset + 0.4, np.array([48.0, 50.0, 52.0])), False, "r")
        row = score.score_pair(ref, ref, bandparts.PART_METRICS, item="self")
        self.assertEqual(row["pc_on50_f1"], 1.0)

    def test_cross_fit_recovers_a_uniform_shift(self):
        import numpy as np
        from open_transcriber.bakeoff import notes_io
        onset = np.arange(0.0, 4.0, 0.5)
        midi = np.array([48.0 + (i % 5) for i in range(len(onset))])
        ref = notes_io._validate(notes_io.Notes(onset, onset + 0.4, midi), False, "r")
        # Estimate is the reference pushed +40 ms late, for four "songs".
        shifted = bandparts.shift_notes(ref, 0.04)
        songs = ["s1", "s2", "s3", "s4"]
        refs = {s: ref for s in songs}
        ests = {s: shifted for s in songs}
        shift = bandparts._cross_fit_shift(refs, ests, songs)
        self.assertAlmostEqual(shift, -0.04, places=2)


class BuildTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # Lay out <out>/<song>/stems/midi/bass.mid as the render would.
        song = self.root / "out-root" / "song1"
        (song / "stems" / "midi").mkdir(parents=True)
        _part_midi(song / "stems" / "midi" / "bass.mid", [(0, 48), (480, 50), (960, 52)])
        self.song_root = self.root / "out-root"
        tsv = self.root / "song1-bass.tsv"
        tsv.write_text("0.000\t0.400\t48\n0.500\t0.900\t50\n1.000\t1.400\t52\n", encoding="utf-8")
        self.tsv = tsv

    def tearDown(self):
        self._tmp.cleanup()

    def test_build_scores_a_part_against_ground_truth(self):
        tx = {("bass", "stem"): {"song1": self.tsv}}
        board = bandparts.build(["song1"], tx, out=self.song_root, parts=("bass",), calibrate=False)
        key = "bass|stem"
        self.assertIn(key, board.rows)
        self.assertEqual(len(board.rows[key]), 1)
        # A near-perfect transcription of three notes scores high at pc_on50.
        self.assertGreaterEqual(score.mean_f1(board.rows[key], "pc_on50"), 0.9)


if __name__ == "__main__":
    unittest.main()
