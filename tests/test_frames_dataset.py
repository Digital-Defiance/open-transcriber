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

            def fake_feats(_root, _wav, _npz_dir, key=None):
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


class CacheKeyTests(unittest.TestCase):
    """Two different songs whose vocals are BOTH named ``vocals.wav`` (Demucs convention) must NOT share a BP
    posterior cache file. The collision bug trained the whole pilot on one song's features (~1% result)."""

    def test_cache_key_is_unique_per_source_even_with_same_filename(self):
        a = Path("/x/song-A/htdemucs/mix/vocals.wav")
        b = Path("/x/song-B/htdemucs/mix/vocals.wav")
        self.assertNotEqual(run_frames._cache_key(a), run_frames._cache_key(b),
                            "same-filename vocals from different folders collided")

    def test_features_for_wav_uses_distinct_cache_files(self):
        # Two different wavs named vocals.wav in different dirs must cache to different npz paths. We stub the
        # heavy BP call and just assert the out_npz paths differ.
        seen = []

        def fake_bp(root, wav, out_npz):
            out_npz.parent.mkdir(parents=True, exist_ok=True)
            seen.append(out_npz.name)
            np.savez(out_npz, note=np.zeros((5, frames.N_BINS), np.float32),
                     onset=np.zeros((5, frames.N_BINS), np.float32))
            return out_npz

        import wave as wavemod

        with tempfile.TemporaryDirectory() as d:
            npz_dir = Path(d) / "bp"
            # make two tiny real wavs so load_wav/log_mel run
            wavs = []
            for name in ("song-A", "song-B"):
                wp = Path(d) / name / "vocals.wav"
                wp.parent.mkdir(parents=True)
                with wavemod.open(str(wp), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(22050)
                    w.writeframes((np.zeros(2205, np.int16)).tobytes())
                wavs.append(wp)
            orig = run_frames.bp_posteriors
            run_frames.bp_posteriors = fake_bp
            try:
                for wp in wavs:
                    run_frames.features_for_wav(Path("/root"), wp, npz_dir)
            finally:
                run_frames.bp_posteriors = orig
        self.assertEqual(len(set(seen)), 2, f"cache files collided: {seen}")


if __name__ == "__main__":
    unittest.main()
