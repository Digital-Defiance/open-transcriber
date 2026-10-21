"""Smoke tests for the sampler-audio render script: label JSON written, resumable, deterministic plan."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from collections import namedtuple

from open_transcriber.bakeoff import make_sampler_audio as msa

Usage = namedtuple("Usage", "total used free")


def fake_render(song_path, wav_path, style, seed):
    """Write a tiny placeholder WAV header so the runner records a real output file; no instruments needed."""
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    Path(wav_path).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")


class SamplerAudioTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        # keep the runner's disk-floor check happy in a sandbox
        self.disk = mock.patch("shutil.disk_usage", return_value=Usage(1000e9, 100e9, 900e9))
        self.disk.start()
        self.caffeinate = mock.patch.object(msa.jobs, "hold_caffeinate", return_value=None)
        self.caffeinate.start()

    def tearDown(self):
        self.caffeinate.stop()
        self.disk.stop()

    def test_estimate_and_seeds(self):
        self.assertEqual(msa.unseen_seeds(3, 90_000_000), [90_000_000, 90_000_001, 90_000_002])
        self.assertAlmostEqual(msa.estimate_hours(400, 3), 400 * 3 * msa.SECONDS_PER_VARIANT / 3600.0)

    def test_dry_run_renders_nothing(self):
        rc = msa.main(["--dry-run", "--n", "2", "--variants", "1", "--out", str(self.tmp / "o")])
        self.assertEqual(rc, 0)
        self.assertFalse((self.tmp / "o" / "clips").exists())

    def test_smoke_render_writes_labels_and_resumes(self):
        out = self.tmp / "ear"
        counts = msa.make(n=1, variants=1, out=out, base_seed=90_000_000, render_fn=fake_render)
        self.assertEqual(counts["ok"], 1)
        clip = out / "clips" / "90000000-v0"
        self.assertTrue((clip / "audio.wav").exists())
        lab = json.loads((clip / "labels.json").read_text())
        self.assertEqual(lab["format"], "copyist-bakeoff-labels/1")
        self.assertEqual(lab["seed"], 90_000_000)
        self.assertGreater(len(lab["notes"]), 0)
        self.assertGreater(len(lab["chords"]), 0)
        # Re-run: the done item is skipped, nothing re-rendered.
        calls = []
        counts2 = msa.make(n=1, variants=1, out=out, base_seed=90_000_000,
                           render_fn=lambda *a: calls.append(a))
        self.assertEqual(counts2["skipped"], 1)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
