"""Unit tests for the SoulX mini-slice planner against the REAL SoulX API (no model loaded).

A fake SoulX exposing ``sing(segments, voice_wav, voice_meta, out_dir, device)`` and a fake ``pitch.check``
drive the gating and resume logic. The segments are built from a real sampler ``Song`` through
``copyist.sing.sung_notes``/``soulx_segments``, so the wiring is exercised end to end without a GPU.
"""

import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from open_transcriber.bakeoff import make_soulx_slice as mss

Usage = namedtuple("Usage", "total used free")
Check = namedtuple("Check", "ok")


class FakeSoulX:
    """A stand-in for SoulX-Singer: ``sing`` writes a placeholder vocal.wav and counts calls."""

    def __init__(self):
        self.sing_calls = 0
        self.last_segments = None

    def sing(self, segments, voice_wav, voice_meta, out_dir, device="mps"):
        self.sing_calls += 1
        self.last_segments = segments
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        wav = out_dir / "vocal.wav"
        wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfake")
        return wav


def fake_assets(prompt):
    return Path(f"/tmp/{prompt}/prompt.wav"), Path(f"/tmp/{prompt}/prompt.json")


#: A tiny singable fixture (real CMU-pronounceable words) so sung_notes/soulx_segments run on a real Song.
FIXTURE_SONG = """title: Test Tune
composer: test
key: C
meter: 4/4
tempo: 120
unit: 1/4
melody: |
  C D E F | G A G F | E D C2 |
lyrics:
  - la la la la love you so my dear friend now
"""


def write_fixture_scores(out_dir, n=1, start_seed=91_000_000, **kwargs):
    """Stand in for scores.write_batch: write a real, singable fixture song for each seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n):
        path = out_dir / f"song-{start_seed + i}.yaml"
        path.write_text(FIXTURE_SONG, encoding="utf-8")
        written.append(path)
    return written


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

    def test_prompt_assets_refuses_when_absent(self):
        with self.assertRaises(FileNotFoundError):
            mss.prompt_assets("vocalset-male1", root=self.tmp / "nope")

    def test_make_builds_segments_from_sung_notes_gates_rerolls_and_resumes(self):
        out = self.tmp / "ear-soulx"
        # The fake check fails the first two sing attempts, then passes.
        state = {"n": 0}

        def check(wav, notes, any_octave=True):
            state["n"] += 1
            ok = state["n"] >= 3  # first two attempts fail the gate
            return [Check(ok) for _ in range(max(1, len(notes)))]

        soulx = FakeSoulX()
        with mock.patch.object(mss.jobs, "GpuLock"), \
             mock.patch.object(mss.scores, "write_batch", side_effect=write_fixture_scores):
            counts = mss.make(n=1, out=out, base_seed=91_000_000, soulx=soulx, check_fn=check,
                              assets_fn=fake_assets)
        self.assertEqual(counts["ok"], 1)
        self.assertEqual(soulx.sing_calls, 3)  # two re-rolls then an accept
        # segments were built from the real score (sung_notes -> soulx_segments), with SoulX fields present.
        self.assertTrue(soulx.last_segments)
        self.assertIn("index", soulx.last_segments[0])
        self.assertIn("note_pitch", soulx.last_segments[0])
        self.assertIn("note_type", soulx.last_segments[0])
        clip = out / "clips" / "91000000-vocalset-male1"
        self.assertTrue((clip / "labels.json").exists())
        self.assertTrue((clip / "vocal.wav").exists())
        # Re-run skips the finished clip (resumability).
        with mock.patch.object(mss.jobs, "GpuLock"), \
             mock.patch.object(mss.scores, "write_batch", side_effect=write_fixture_scores):
            counts2 = mss.make(n=1, out=out, base_seed=91_000_000, soulx=FakeSoulX(), check_fn=check,
                               assets_fn=fake_assets)
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
