"""MuScriptor's MIDI output converted to a board drop-in melody TSV, and the inputs that must be rejected.

CPU-only: these exercise the MIDI reader, the melody reduction, and the board round-trip (the same plumbing a
GPU run would use to land MuScriptor's notes on the scoreboard) without running or downloading the model.
MuScriptor writes a Standard MIDI File per input; the board needs a monophonic ``onset offset midi`` line, so
a converted file must load through the board's own drop-in loader and score.
"""

import struct
import tempfile
import unittest
from pathlib import Path

from open_transcriber.board import board, muscriptor


def _vlq(n: int) -> bytes:
    """A MIDI variable-length quantity."""
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def _smf(track_events: bytes, *, fmt: int = 1, tpb: int = 480) -> bytes:
    """Wrap one track's event bytes in a minimal SMF (header + one MTrk with an end-of-track)."""
    events = bytearray(track_events)
    events += _vlq(0) + bytes([0xFF, 0x2F, 0x00])  # end of track
    track = b"MTrk" + struct.pack(">I", len(events)) + bytes(events)
    header = b"MThd" + struct.pack(">IHHH", 6, fmt, 1, tpb)
    return header + track


def _note(delta: int, on: bool, pitch: int, vel: int, channel: int = 0) -> bytes:
    status = (0x90 if on else 0x80) | channel
    return _vlq(delta) + bytes([status, pitch, vel if on else 0])


class MidiReaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def test_tempo_map_turns_ticks_into_seconds(self):
        # 120 BPM (500000 us/beat), 480 tpb: one beat = 0.5 s. C4 for a beat, then G4 for a beat.
        ev = bytearray()
        ev += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")
        ev += _note(0, True, 60, 100)
        ev += _note(480, False, 60, 0)
        ev += _note(0, True, 67, 90)
        ev += _note(480, False, 67, 0)
        path = self._write("a.mid", _smf(bytes(ev)))
        notes = muscriptor.read_midi_notes(path)
        self.assertEqual([n.midi for n in notes], [60, 67])
        self.assertAlmostEqual(notes[0].onset, 0.0, places=3)
        self.assertAlmostEqual(notes[0].offset, 0.5, places=3)
        self.assertAlmostEqual(notes[1].onset, 0.5, places=3)
        self.assertAlmostEqual(notes[1].offset, 1.0, places=3)

    def test_percussion_channel_is_skipped(self):
        ev = bytearray()
        ev += _note(0, True, 60, 100, channel=0)      # a melody note
        ev += _note(0, True, 38, 100, channel=9)      # a snare on the drum channel
        ev += _note(480, False, 38, 0, channel=9)
        ev += _note(0, False, 60, 0, channel=0)
        path = self._write("drums.mid", _smf(bytes(ev)))
        notes = muscriptor.read_midi_notes(path)
        self.assertEqual([n.midi for n in notes], [60])

    def test_not_a_midi_file_is_rejected(self):
        # The failing input: MuScriptor didn't write a MIDI (empty or wrong-format file).
        path = self._write("bad.mid", b"this is not a midi file")
        with self.assertRaises(ValueError):
            muscriptor.read_midi_notes(path)


class MelodyReductionTests(unittest.TestCase):
    def test_midi_overlap_resolved_by_velocity_and_short_notes_dropped(self):
        notes = [
            muscriptor.MidiNote(0.0, 0.5, 60, 100),   # kept
            muscriptor.MidiNote(0.1, 0.4, 72, 40),     # overlaps, quieter -> loses
            muscriptor.MidiNote(0.6, 0.62, 64, 90),    # 20 ms, too short -> dropped
            muscriptor.MidiNote(0.7, 1.2, 67, 95),     # kept
        ]
        rows = muscriptor.melody_line(notes)
        pitches = [p for _, _, p in rows]
        self.assertIn(60, pitches)
        self.assertIn(67, pitches)
        self.assertNotIn(64, pitches)  # the 20 ms note is gone

    def test_rows_overlap_resolved_by_duration(self):
        # The bench's notes-TSV carries no velocity, so the longer note wins an overlap.
        rows_in = [
            (0.0, 0.5, 60),    # long -> kept
            (0.1, 0.18, 72),   # short accompaniment hit -> loses / dropped
            (0.6, 1.2, 67),    # long -> kept
        ]
        rows = muscriptor.melody_line_rows(rows_in)
        self.assertEqual([p for _, _, p in rows], [60, 67])

    def test_out_of_range_notes_dropped(self):
        rows_in = [
            (0.0, 0.5, 30),    # below the singing range
            (0.6, 1.0, 60),    # in range
            (1.1, 1.5, 100),   # above the singing range
        ]
        rows = muscriptor.melody_line_rows(rows_in)
        self.assertEqual([p for _, _, p in rows], [60])


class NotesTsvTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_read_notes_tsv_sorts_and_reads(self):
        path = self.root / "a.notes.tsv"
        path.write_text("0.600\t1.200\t67\n0.000\t0.500\t60\n\n", encoding="utf-8")
        rows = muscriptor.read_notes_tsv(path)
        self.assertEqual(rows, [(0.0, 0.5, 60), (0.6, 1.2, 67)])

    def test_non_integer_pitch_rejected(self):
        path = self.root / "bad.notes.tsv"
        path.write_text("0.000\t0.500\t60.5\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            muscriptor.read_notes_tsv(path)

    def test_wrong_column_count_rejected(self):
        path = self.root / "cols.notes.tsv"
        path.write_text("0.000\t0.500\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            muscriptor.read_notes_tsv(path)

    def test_convert_notes_tsv_loads_through_board(self):
        path = self.root / "song.notes.tsv"
        path.write_text("0.000\t0.500\t60\n0.100\t0.180\t72\n0.600\t1.200\t67\n", encoding="utf-8")
        dest = self.root / "systems" / "muscriptor-small-stem" / "song.tsv"
        muscriptor.convert_notes_tsv(path, dest)
        notes = board._tsv_loader(dest)()
        self.assertEqual(list(notes.midi), [60.0, 67.0])


class ConvertTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _probe_midi(self, name: str) -> Path:
        ev = bytearray()
        ev += _vlq(0) + bytes([0xFF, 0x51, 0x03]) + (500000).to_bytes(3, "big")
        ev += _note(0, True, 60, 100)
        ev += _note(480, False, 60, 0)
        ev += _note(0, True, 67, 90)
        ev += _note(480, False, 67, 0)
        path = self.root / name
        path.write_bytes(_smf(bytes(ev)))
        return path

    def test_converted_tsv_loads_through_board(self):
        midi = self._probe_midi("song.mid")
        dest = self.root / "systems" / "muscriptor-small-stem" / "song.tsv"
        muscriptor.convert_midi(midi, dest)
        notes = board._tsv_loader(dest)()
        self.assertEqual(len(notes), 2)
        self.assertEqual(list(notes.midi), [60.0, 67.0])
        self.assertAlmostEqual(float(notes.offset[1]), 1.0, places=3)

    def test_convert_dir_prefers_notes_tsv_then_midi_and_skips_missing(self):
        # item "one": the bench notes-TSV (the primary GGUF path)
        (self.root / "one.notes.tsv").write_text("0.000\t0.500\t60\n0.600\t1.200\t67\n", encoding="utf-8")
        # item "two": only a MIDI (the official --format midi fallback)
        self._probe_midi("two.mid")
        # item "three": nothing -> skipped
        done = muscriptor.convert_dir(self.root, self.root / "sys", ["one", "two", "three"])
        self.assertEqual(done, ["one", "two"])
        self.assertTrue((self.root / "sys" / "one.tsv").exists())
        self.assertTrue((self.root / "sys" / "two.tsv").exists())
        self.assertFalse((self.root / "sys" / "three.tsv").exists())

    def test_config_name(self):
        self.assertEqual(muscriptor.config_name("medium", "mix"), "muscriptor-medium-mix")

    def test_config_name_rejects_unknown(self):
        with self.assertRaises(ValueError):
            muscriptor.config_name("huge", "mix")
        with self.assertRaises(ValueError):
            muscriptor.config_name("small", "demucs")


if __name__ == "__main__":
    unittest.main()
