"""Tests for chart QA on synthetic singing: harmonic tones at made-up chart pitches."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from open_transcriber.bakeoff import crepe, notes_io, qa

RATE = 16000
PITCHES = [64, 67, 69, 67, 64, 62]
START, NOTE, GAP = 0.5, 0.5, 0.1


def sung(pitches=PITCHES, silent=()):
    """Harmonic tones with soft attacks and releases, one per chart note, silence where asked."""
    total = START + len(pitches) * (NOTE + GAP) + 0.5
    audio = np.zeros(int(total * RATE))
    t = np.arange(int(NOTE * RATE)) / RATE
    envelope = np.minimum(1.0, np.minimum(t / 0.01, (NOTE - t) / 0.05))
    for i, p in enumerate(pitches):
        if i in silent:
            continue
        f = 440.0 * 2 ** ((p - 69) / 12)
        tone = sum(0.6 ** k * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(5)) * envelope
        start = int((START + i * (NOTE + GAP)) * RATE)
        audio[start:start + len(tone)] += 0.25 * tone
    return audio.astype(np.float32)


def chart(pitches=PITCHES, shift=0.0, freestyle=()):
    onset = [START + i * (NOTE + GAP) + shift for i in range(len(pitches))]
    mask = ["freestyle" if i in freestyle else "" for i in range(len(pitches))]
    return notes_io.Notes(onset, [o + NOTE for o in onset], pitches, mask)


@unittest.skipUnless(crepe.model_path().exists(), "crepe-full.onnx not exported on this machine")
class QaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = crepe.Model()
        cls.t, _ = qa.load_thresholds()

    def masks(self, audio, ref):
        verdicts, summary, _ = qa.check_audio(audio, ref, self.model, self.t)
        return [v["mask"] for v in verdicts], summary

    def test_a_correct_chart_keeps_every_note(self):
        masks, summary = self.masks(sung(), chart())
        self.assertEqual(masks, [""] * 6)
        self.assertFalse(summary["dropped"])
        self.assertLessEqual(abs(summary["median_signed_offset_s"]), 0.05)

    def test_octave_is_ignored_but_a_wrong_pitch_is_masked(self):
        wrong = list(PITCHES)
        wrong[2] += 2
        wrong[4] += 12  # an octave off: same pitch class, kept
        masks, _ = self.masks(sung(), chart(wrong))
        self.assertEqual(masks, ["", "", "pitch", "", "", ""])

    def test_notes_over_silence_are_unverifiable_and_freestyle_is_masked(self):
        masks, _ = self.masks(sung(silent=(1, 4)), chart(freestyle=(5,)))
        self.assertEqual(masks, ["", "unverifiable", "", "", "unverifiable", "freestyle"])

    def test_a_late_chart_is_dropped(self):
        # Should fail: a chart 250 ms late fails its onsets and most of its notes.
        masks, summary = self.masks(sung(), chart(shift=0.25))
        self.assertGreater(masks.count("onset"), 3)
        self.assertTrue(summary["dropped"])

    def test_median3_edges(self):
        np.testing.assert_allclose(qa.median3([0.9, 0.1, 0.8, 0.7]), [0.1, 0.8, 0.7, 0.7])


class GuardTests(unittest.TestCase):
    def test_the_scorer_refuses_masks_from_other_thresholds(self):
        # Should fail: masks made with different thresholds.
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "qa").mkdir()
            good = {"thresholds_sha256": qa.qa_thresholds_sha256(), "notes": [], "dropped": False}
            bad = dict(good, thresholds_sha256="0" * 64)
            (Path(tmp) / "qa" / "good.json").write_text(json.dumps(good))
            (Path(tmp) / "qa" / "bad.json").write_text(json.dumps(bad))
            self.assertEqual(qa.load_qa("good", Path(tmp))["dropped"], False)
            with self.assertRaises(ValueError):
                qa.load_qa("bad", Path(tmp))
            with self.assertRaises(FileNotFoundError):
                qa.load_qa("missing", Path(tmp))

    def test_pc_distance_is_circular(self):
        self.assertAlmostEqual(qa.pc_distance(71.8, 60.0), 0.2)
        self.assertAlmostEqual(qa.pc_distance(66.0, 60.0), 6.0)
        self.assertAlmostEqual(qa.pc_distance(48.3, 60.0), 0.3)


if __name__ == "__main__":
    unittest.main()
