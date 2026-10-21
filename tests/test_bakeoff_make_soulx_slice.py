"""Unit tests for the SoulX mini-slice planner: gating accepts/rejects, resumes, and refuses without SoulX.

No model is loaded: a fake SoulX and a fake pitch.check drive the gating and resume logic.
"""

import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from open_transcriber.bakeoff import make_soulx_slice as mss

Usage = namedtuple("Usage", "total used free")
Check = namedtuple("Check", "ok")


class FakeSoulX:
    """A stand-in for SoulX-Singer: writes a placeholder wav and reports notes; counts sing calls."""

    def __init__(self, fail_first=0):
        self.sing_calls = 0
        self.fail_first = fail_first  # how many sing attempts "sing out of tune" before passing

    def sing_phrase(self, song_path, prompt, wav, attempt=0):
        self.sing_calls += 1
        Path(wav).parent.mkdir(parents=True, exist_ok=True)
        Path(wav).write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")

    def sung_notes(self, song_path):
        return [object()]  # opaque; the fake check decides pass/fail


class SoulXSliceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.disk = mock.patch("shutil.disk_usage", return_value=Usage(1000e9, 100e9, 900e9))
        self.disk.start()
        self.caffeinate = mock.patch.object(mss.jobs, "hold_caffeinate", return_value=None)
        self.caffeinate.start()

    def tearDown(self):
        self.caffeinate.stop()
        self.disk.stop()

    def test_plan_round_robins_prompts(self):
        plans = mss.plan_clips(5, base_seed=91_000_000, prompts=("a", "b"))
        self.assertEqual([p.prompt for p in plans], ["a", "b", "a", "b", "a"])
        self.assertEqual(plans[0].seed, 91_000_000)
        self.assertEqual(plans[0].clip_id, "91000000-a")

    def test_estimate_hours(self):
        self.assertAlmostEqual(mss.estimate_hours(120), 120 * mss.SECONDS_PER_CLIP_GPU / 3600.0)

    def test_gate_accepts_only_when_every_note_passes(self):
        passing = lambda wav, notes, any_octave=True: [Check(True), Check(True)]
        failing = lambda wav, notes, any_octave=True: [Check(True), Check(False)]
        self.assertTrue(mss.gate_clip(passing, Path("x"), [1, 2]))
        self.assertFalse(mss.gate_clip(failing, Path("x"), [1, 2]))
        self.assertFalse(mss.gate_clip(lambda *a, **k: [], Path("x"), []))  # no notes -> reject

    def test_make_gates_rerolls_and_resumes(self):
        out = self.tmp / "ear-soulx"
        # The fake check fails the first two sing attempts, then passes.
        state = {"n": 0}

        def check(wav, notes, any_octave=True):
            state["n"] += 1
            ok = state["n"] >= 3  # first two attempts fail the gate
            return [Check(ok)]

        soulx = FakeSoulX()
        with mock.patch.object(mss.jobs, "GpuLock"):  # no real GPU lock in the sandbox
            counts = mss.make(n=1, out=out, base_seed=91_000_000, soulx=soulx, check_fn=check)
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(soulx.sing_calls, 3)  # two re-rolls then an accept
        self.assertTrue((out / "clips" / "91000000-vocalset-male1" / "labels.json").exists())
        # Re-run skips the finished clip.
        with mock.patch.object(mss.jobs, "GpuLock"):
            counts2 = mss.make(n=1, out=out, base_seed=91_000_000, soulx=FakeSoulX(), check_fn=check)
        self.assertEqual(counts2["skipped"], 1)

    def test_dry_run_exits_without_model(self):
        rc = mss.main(["--dry-run", "--n", "4"])
        self.assertEqual(rc, 0)

    def test_soulx_available_reports_reason_when_missing(self):
        def missing():
            raise RuntimeError("SoulX not configured")
        ok, reason = mss.soulx_available(find_fn=missing)
        self.assertFalse(ok)
        self.assertIn("SoulX not configured", reason)


if __name__ == "__main__":
    unittest.main()
