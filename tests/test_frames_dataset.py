"""build_sung_dataset must be resilient: one unsingable/unbuildable song is SKIPPED (logged, counted) and
the batch continues to its target, instead of aborting on the first failure (the bug that killed the first
GPU pilot). No GPU, SoulX, Basic Pitch or Demucs — the heavy stages are monkeypatched with light stand-ins.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_frames  # noqa: E402
from open_transcriber.data import frames  # noqa: E402


class BuildSungDatasetResilienceTests(unittest.TestCase):
    def test_one_unsingable_song_is_skipped_and_batch_reaches_target(self):
        root = run_frames.paths.copyist_root()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            # Decide which generated song will be "unsingable" without touching real singing: pre-list the
            # candidate slugs write_batch will produce (deterministic by seed).
            from open_transcriber.data import scores as score_gen
            cand = score_gen.write_batch(out_dir=out / "frames-run" / "scores", n=10, start_seed=1)
            fail_slug = cand[0].stem

            orig_run = run_frames.subprocess.run
            orig_sing = run_frames._sing_and_separate
            orig_feats = run_frames.features_for_wav
            orig_truth = run_frames._midi_truth

            def fake_build(cmd, **kw):
                # Stand in for `copyist build`: create the melody-label MIDI the builder would write, so the
                # truth_mid existence check passes for every candidate (the label stage is not under test).
                slug = Path(cmd[-1]).stem
                mid = root / "out" / slug / "stems" / "midi" / "melody.mid"
                mid.parent.mkdir(parents=True, exist_ok=True)
                mid.write_bytes(b"")  # presence is all build_sung_dataset checks
                return subprocess.CompletedProcess(cmd, 0)

            def fake_sing(_root, _out, slug):
                return None if slug == fail_slug else out / f"{slug}.wav"

            def fake_feats(_root, _wav, _npz_dir):
                return np.zeros((20, frames.FEATURE_DIM), np.float32)

            def fake_truth(_path):
                return run_frames.notes_io._validate(
                    run_frames.notes_io.Notes(np.array([0.0]), np.array([0.4]), np.array([60.0])),
                    False, "truth")

            run_frames.subprocess.run = fake_build
            run_frames._sing_and_separate = fake_sing
            run_frames.features_for_wav = fake_feats
            run_frames._midi_truth = fake_truth
            made_mids = []
            try:
                ds = run_frames.build_sung_dataset(root, out, n_songs=3, seed=1)
                self.assertEqual(len(ds.seqs), 3, "batch did not reach its target despite a skip")
                self.assertEqual(len(ds.songs), 3)
                self.assertNotIn(fail_slug, ds.songs, "the unsingable song was not skipped")
            finally:
                run_frames.subprocess.run = orig_run
                run_frames._sing_and_separate = orig_sing
                run_frames.features_for_wav = orig_feats
                run_frames._midi_truth = orig_truth
                # Clean the stand-in melody.mid files the fake build wrote under the real out/ tree.
                for slug in [c.stem for c in cand]:
                    d2 = root / "out" / slug
                    if d2.exists():
                        import shutil
                        shutil.rmtree(d2, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
