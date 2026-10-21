"""Tests for RWC MIDI reading and the reference rules (synthetic MIDI and made-up rows only)."""

import tempfile
import unittest
from pathlib import Path

import mido

from open_transcriber.bakeoff import refs, rwc_midi


def build_midi(tracks, tempos=((0, 500000), (960, 1000000)), tpb=480) -> Path:
    """tracks: list of (name, program, channel, [(on_tick, off_tick, pitch, off_as_velocity_zero)])."""
    mf = mido.MidiFile(type=1, ticks_per_beat=tpb)
    conductor = mido.MidiTrack()
    last = 0
    for tick, tempo in tempos:
        conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - last))
        last = tick
    mf.tracks.append(conductor)
    for name, program, channel, notes in tracks:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        track.append(mido.Message("program_change", program=program, channel=channel, time=0))
        events = []
        for on, off, pitch, zero in notes:
            # at a shared tick, note-offs come before note-ons, except a zero-length note's own on and off
            events.append((on, -1 if on == off else 1,
                           mido.Message("note_on", note=pitch, velocity=90, channel=channel)))
            off_msg = (mido.Message("note_on", note=pitch, velocity=0, channel=channel) if zero
                       else mido.Message("note_off", note=pitch, velocity=0, channel=channel))
            events.append((off, 0, off_msg))
        now = 0
        for tick, _order, msg in sorted(events, key=lambda e: (e[0], e[1])):
            track.append(msg.copy(time=tick - now))
            now = tick
        mf.tracks.append(track)
    path = Path(tempfile.mkdtemp()) / "test.mid"
    mf.save(str(path))
    return path


class MidiTests(unittest.TestCase):
    def test_tempo_change_gives_exact_seconds(self):
        # 120 BPM for the first 960 ticks (1.0 s), then 60 BPM: tick 1440 is 1.0 + 0.5 = 1.5 s... at 60 BPM
        # 480 ticks last 1 s, so tick 1440 is 2.0 s.
        path = build_midi([("MELODY", 73, 0, [(0, 480, 60, False), (720, 1200, 62, True), (1440, 1920, 64, False)])])
        midi = rwc_midi.read(path)
        notes = midi.tracks[1].notes
        self.assertEqual([n.pitch for n in notes], [60, 62, 64])
        self.assertAlmostEqual(notes[0].onset, 0.0)
        self.assertAlmostEqual(notes[0].offset, 0.5)
        self.assertAlmostEqual(notes[1].onset, 0.75)     # inside the 120 BPM part
        self.assertAlmostEqual(notes[1].offset, 1.5)     # across the change: 1.0 + 240/480 * 1.0
        self.assertAlmostEqual(notes[2].onset, 2.0)
        self.assertAlmostEqual(notes[2].offset, 3.0)

    def test_zero_length_dropped_overlap_kept_duplicates_collapsed(self):
        path = build_midi([("MELODY", 73, 0, [(0, 0, 60, False), (480, 960, 62, False), (960, 1200, 62, False),
                                             (1200, 1300, 65, False), (1200, 1300, 65, False)])])
        midi = rwc_midi.read(path)
        notes = midi.tracks[1].notes
        self.assertEqual(midi.counts["zero_length_dropped"], 1)
        self.assertEqual(midi.counts["duplicates_collapsed"], 1)
        # the note-off at 960 ends the first 62; the 62 that starts at 960 stays open until 1200
        self.assertEqual([(n.onset_tick, n.offset_tick, n.pitch) for n in notes],
                         [(480, 960, 62), (960, 1200, 62), (1200, 1300, 65)])

    def test_drums_are_recognised(self):
        path = build_midi([("KICK", 0, 9, [(0, 10, 36, False)]), ("PERC.", 0, 3, [(0, 10, 40, False)]),
                           ("BASS", 33, 2, [(0, 10, 40, False)])])
        tracks = rwc_midi.read(path).tracks
        self.assertEqual([t.is_drums for t in tracks[1:]], [True, True, False])


def row(track, name, offset=12, comment=""):
    return refs.Row("RWC_X001", track, name, offset, bool(refs.EXCLUDE_ROW.search(comment)), comment)


class RowTests(unittest.TestCase):
    def test_exclusion_pattern_on_made_up_comments(self):
        for comment, excluded in [("Back Vocal", True), ("backvocals in the bridge", True), ("not vocal", True),
                                  ("this is not the main melody", True), ("Nothing?", True), ("", False),
                                  ("octave is wrong in verse 2", False), ("Main melody", False)]:
            self.assertEqual(row(1, "X", comment=comment).excluded, excluded, comment)

    def test_a_piece_is_never_left_without_rows(self):
        kept, flagged = refs.included([row(3, "MELODY", comment="in the chorus it is the back vocals")])
        self.assertEqual(len(kept), 1)
        self.assertTrue(flagged)
        kept, flagged = refs.included([row(4, "Melo"), row(3, "Sub", comment="back vocals")])
        self.assertEqual([r.track for r in kept], [4])
        self.assertFalse(flagged)

    def test_track_name_mismatch_raises(self):
        # Should fail: the mapping names a track that isn't there.
        midi = rwc_midi.read(build_midi([("MELODY", 73, 0, [(0, 480, 60, False)])]))
        self.assertEqual(len(refs.track_notes(midi, row(1, "MELODY"))), 1)
        with self.assertRaises(ValueError):
            refs.track_notes(midi, row(1, "VOCAL"))

    def test_octave_exact_pitch_uses_the_sign(self):
        midi = rwc_midi.read(build_midi([("MELODY", 73, 0, [(0, 480, 72, False)])]))
        notes = refs.vocal_notes(midi, [row(1, "MELODY", offset=12)], sign=-1)
        self.assertEqual((notes[0].midi, notes[0].midi_sung), (72, 60))


class SignCheckTests(unittest.TestCase):
    def test_f0_an_octave_above_picks_plus(self):
        note = rwc_midi.MidiNote(1, 0, 60, 90, 0.0, 1.0, 0, 480)
        f0 = [(t / 100, 523.25 if t % 7 else 0.0) for t in range(100)]  # C5 over the C4 note, some unvoiced
        median = refs.f0_offset(f0, [note])
        self.assertAlmostEqual(median, 12.0, places=1)
        self.assertEqual(refs.sign_check({"a": (median, 12), "b": (0.2, 0)}), 1)
        self.assertEqual(refs.sign_check({"a": (-11.8, 12), "b": (-24.3, 24)}), -1)

    def test_inconsistent_signs_raise(self):
        with self.assertRaises(ValueError):
            refs.sign_check({"a": (12.0, 12), "b": (-12.0, 12)})


class SoloTests(unittest.TestCase):
    def test_skyline_keeps_the_highest_of_near_simultaneous_notes(self):
        notes = [rwc_midi.MidiNote(1, 0, p, 90, t, t + 0.2, 0, 0) for t, p in
                 [(1.000, 60), (1.010, 67), (1.019, 64), (1.300, 62), (1.330, 59)]]
        self.assertEqual([(n.onset, n.pitch) for n in refs.skyline(notes)], [(1.010, 67), (1.300, 62), (1.330, 59)])

    def test_solo_tier_order(self):
        path = build_midi([
            ("MELODY", 73, 0, [(0, 4000, 70, False)]),                       # vocal track: never a solo
            ("E.G", 29, 1, [(t, t + 100, 64, False) for t in range(960, 1900, 120)]),   # guitar family
            ("PAD", 89, 2, [(t, t + 100, 50, False) for t in range(960, 1900, 60)]),    # busiest, not guitar
            ("DRUMS", 0, 9, [(t, t + 10, 36, False) for t in range(960, 1900, 30)]),
        ], tempos=((0, 500000),))
        midi = rwc_midi.read(path)
        # 120 BPM: ticks 960..1900 are 1.0..1.98 s
        self.assertEqual(refs.solo_track(midi, "electricguitar", 1.0, 2.0, {1}), (2, 2))
        self.assertEqual(refs.solo_track(midi, "oboe", 1.0, 2.0, {1}), (3, 3))  # no oboe: busiest non-drum
        self.assertIsNone(refs.solo_track(midi, "oboe", 5.0, 6.0, {1}))
        named = build_midi([("MELODY", 73, 0, []), ("GT SOLO", 81, 1, [(1000, 1100, 76, False)]),
                            ("E.G", 29, 2, [(t, t + 50, 64, False) for t in range(960, 1900, 60)])],
                           tempos=((0, 500000),))
        self.assertEqual(refs.solo_track(rwc_midi.read(named), "electricguitar", 1.0, 2.0, {1}), (1, 2))

    def test_voca_inst_segments_and_label_fixes(self):
        text = "#header\n0.0 b\n10.0 m\n20.5 s:e;ectricguitar\n30.0 s;bass\n40.0 f extra\n"
        path = Path(tempfile.mkdtemp()) / "x.TXT"
        path.write_bytes(text.encode("cp932"))
        segments = refs.voca_inst(path, end=50.0)
        self.assertEqual(segments, [(0.0, 10.0, "b"), (10.0, 20.5, "m"), (20.5, 30.0, "s:electricguitar"),
                                    (30.0, 40.0, "s:bass"), (40.0, 50.0, "f")])


if __name__ == "__main__":
    unittest.main()
