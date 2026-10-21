"""The label emitter builds exact note/beat/chord/key/meter labels from a sampled score (not from audio)."""

import json
import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import labels

# A tiny hand-built song: C major, 4/4, 120 BPM (0.5 s per quarter), two bars.
#   bar 1: C major chord, C4(q) D4(q) E4(h)   -> onsets 0.0, 0.5, 1.0 s
#   bar 2: G major chord, G4(h) C4(h)         -> onsets 2.0, 3.0 s
FIXTURE = """\
format: copyist-song/0
title: Fixture
key: C
meter: 4/4
unit: 1/4
tempo: 120
voice:
  name: Soprano
  range: [C4, C6]
melody: |
  "C"C D E2 | "G"G2 c2 |]
lyrics:
  - la la la la la
"""


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.path = self.tmp / "fixture.yaml"
        self.path.write_text(FIXTURE, encoding="utf-8")

    def test_key_meter_tempo(self):
        L = labels.labels_for_song(self.path, seed=1)
        self.assertEqual(L.key, "C")
        self.assertEqual(L.key_class, 0)         # C major
        self.assertEqual(L.meter, "4/4")
        self.assertEqual(L.tempo, 120)
        self.assertAlmostEqual(L.duration, 4.0)  # two 4/4 bars at 0.5 s/quarter

    def test_minor_key_class(self):
        self.assertEqual(labels.key_class("A"), 9)      # A major
        self.assertEqual(labels.key_class("Am"), 21)    # A minor = 9 + 12
        self.assertEqual(labels.key_class("D#m"), 15)   # D# = 3, minor
        self.assertEqual(labels.key_class("Bb"), 10)    # Bb major

    def test_note_onsets_offsets_pitches(self):
        L = labels.labels_for_song(self.path, seed=1)
        got = [(n.onset, n.offset, n.midi) for n in L.notes]
        self.assertEqual(got, [
            (0.0, 0.5, 60),   # C4 quarter
            (0.5, 1.0, 62),   # D4 quarter
            (1.0, 2.0, 64),   # E4 half
            (2.0, 3.0, 67),   # G4 half
            (3.0, 4.0, 72),   # C5 half
        ])

    def test_beats_and_downbeats(self):
        L = labels.labels_for_song(self.path, seed=1)
        # 4/4 at 0.5 s/quarter: beats every 0.5 s, bars every 2.0 s, out to 4.0 s inclusive.
        self.assertEqual(L.beats, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
        self.assertEqual(L.downbeats, [0.0, 2.0, 4.0])

    def test_chords(self):
        L = labels.labels_for_song(self.path, seed=1)
        got = [(c.onset, c.offset, c.root_pc, c.quality) for c in L.chords]
        self.assertEqual(got, [
            (0.0, 2.0, 0, "maj"),   # C major, bar 1
            (2.0, 4.0, 7, "maj"),   # G major, bar 2
        ])

    def test_deterministic_json(self):
        a = labels.labels_for_song(self.path, seed=1).to_json()
        b = labels.labels_for_song(self.path, seed=1).to_json()
        self.assertEqual(a, b)
        payload = json.loads(a)
        self.assertEqual(payload["format"], labels.LABELS_FORMAT)

    def test_write_labels_roundtrips(self):
        L = labels.labels_for_song(self.path, seed=7)
        out = labels.write_labels(L, self.tmp / "out" / "fixture.labels.json")
        self.assertTrue(out.exists())
        self.assertEqual(json.loads(out.read_text())["seed"], 7)


if __name__ == "__main__":
    unittest.main()
