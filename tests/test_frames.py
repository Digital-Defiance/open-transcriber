"""The sung frame-model's CPU plumbing: features, labels, decode, the model, and an input that must fail.

These exercise the frozen-feature stack, the label rasteriser, Basic-Pitch-style decoding, the frame model's
train/predict round-trip, and the dataset save/load, without any GPU, SoulX, Basic Pitch or Demucs. A model
trained on a learnable synthetic blur-and-denoise signal must recover notes; a mis-shaped feature stack must
be rejected.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io
from open_transcriber.data import frames


def _truth(onsets, offsets, pitches):
    return notes_io._validate(
        notes_io.Notes(np.array(onsets, float), np.array(offsets, float), np.array(pitches, float)),
        False, "truth")


class FeatureTests(unittest.TestCase):
    def test_frames_from_posteriors_shape_and_trim(self):
        note = np.zeros((100, frames.N_BINS), np.float32)
        onset = np.zeros((95, frames.N_BINS), np.float32)
        mel = np.zeros((90, frames.MEL_BANDS), np.float32)
        feats = frames.frames_from_posteriors(note, onset, mel)
        self.assertEqual(feats.shape, (90, frames.FEATURE_DIM))

    def test_frames_from_posteriors_allows_no_mel(self):
        note = np.zeros((40, frames.N_BINS), np.float32)
        onset = np.zeros((40, frames.N_BINS), np.float32)
        feats = frames.frames_from_posteriors(note, onset, None)
        self.assertEqual(feats.shape, (40, frames.FEATURE_DIM))

    def test_wrong_bin_count_is_rejected(self):
        note = np.zeros((10, 87), np.float32)  # not 88
        onset = np.zeros((10, frames.N_BINS), np.float32)
        with self.assertRaises(ValueError):
            frames.frames_from_posteriors(note, onset, None)

    def test_log_mel_frame_count_and_finite(self):
        sr = 22050
        wav = (0.1 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
        mel = frames.log_mel(wav, sr, n_frames=86)
        self.assertEqual(mel.shape, (86, frames.MEL_BANDS))
        self.assertTrue(np.isfinite(mel).all())

    def test_rasterise_labels_marks_onset_and_span(self):
        truth = _truth([0.0, 1.0], [0.5, 1.5], [60, 62])
        n = int(2.0 * frames.FPS)
        note, onset = frames.rasterise_labels(truth, n)
        b60, b62 = 60 - frames.MIDI_LO, 62 - frames.MIDI_LO
        self.assertEqual(onset[0, b60], 1.0)
        self.assertEqual(note[0, b60], 1.0)
        self.assertEqual(int(onset[:, b60].sum()), 1)  # one onset frame per note
        mid = int(0.25 * frames.FPS)
        self.assertEqual(note[mid, b60], 1.0)  # active mid-note
        self.assertEqual(int(onset[:, b62].sum()), 1)


class AlignTests(unittest.TestCase):
    """Label time-alignment: the exact MIDI is in SCORE time, but SoulX sings at a different tempo with a
    lead-in, so labels must be mapped onto the sung-audio (BP-voiced) span before rasterising. (Diagnosed:
    BP passthrough ~1.6% at score time vs ~13% time-aligned.)"""

    def test_align_rescales_labels_onto_bp_voiced_span(self):
        # Score-time notes span [0, 2]s; the BP note posterior is voiced only in frames ~[2.0, 4.0]s (a slower
        # sung tempo with a lead-in). Alignment must move the first onset from ~0 to ~2 s.
        truth = _truth([0.0, 1.0, 2.0], [0.9, 1.9, 2.9], [60, 62, 64])
        n = int(5.0 * frames.FPS)
        bp = np.zeros((n, frames.N_BINS), np.float32)
        lo, hi = int(2.0 * frames.FPS), int(4.0 * frames.FPS)
        bp[lo:hi, 60 - frames.MIDI_LO] = 0.9  # voiced only in [2,4]s
        aligned = frames.align_truth_to_audio(truth, bp)
        self.assertAlmostEqual(aligned.onset[0], 2.0, delta=0.1)
        self.assertAlmostEqual(aligned.offset[-1], 4.0, delta=0.2)
        # pitches are untouched
        self.assertTrue(np.array_equal(aligned.midi, truth.midi))

    def test_align_is_noop_without_voiced_bp(self):
        truth = _truth([0.0, 1.0], [0.9, 1.9], [60, 62])
        bp = np.zeros((int(3.0 * frames.FPS), frames.N_BINS), np.float32)  # silent
        aligned = frames.align_truth_to_audio(truth, bp)
        self.assertTrue(np.array_equal(aligned.onset, truth.onset))

    def test_dtw_follows_nonlinear_warp(self):
        # Score: three notes at 0,1,2 s (pitches 60,62,64). Sung audio holds the FIRST note long (a melisma),
        # so the BP evidence is pitch 60 for [0,3)s, 62 for [3,4)s, 64 for [4,5)s — a non-linear warp a single
        # rescale cannot follow. DTW must stretch note 1 and push notes 2,3 later.
        truth = _truth([0.0, 1.0, 2.0], [0.9, 1.9, 2.9], [60, 62, 64])
        n = int(5.0 * frames.FPS)
        bp = np.zeros((n, frames.N_BINS), np.float32)
        seg = [(0, 3, 60), (3, 4, 62), (4, 5, 64)]
        for t0, t1, m in seg:
            bp[int(t0 * frames.FPS):int(t1 * frames.FPS), m - frames.MIDI_LO] = 0.9
        aligned = frames.align_truth_dtw(truth, bp)
        self.assertEqual(len(aligned.onset), 3)
        self.assertTrue(np.array_equal(aligned.midi, truth.midi))      # pitches untouched
        # note 1 should now extend to ~3 s; note 2 onset should be pushed to ~3 s (not stay at 1 s)
        self.assertGreater(aligned.offset[0], 2.0)
        self.assertGreater(aligned.onset[1], 2.0)

    def test_dtw_falls_back_when_degenerate(self):
        truth = _truth([0.0, 1.0], [0.9, 1.9], [60, 62])
        bp = np.zeros((3, frames.N_BINS), np.float32)  # far too few frames
        aligned = frames.align_truth_dtw(truth, bp)
        self.assertEqual(len(aligned.onset), 2)


class DecodeTests(unittest.TestCase):
    def test_decode_recovers_clean_notes(self):
        truth = _truth([0.0, 0.6, 1.2], [0.5, 1.1, 1.7], [60, 64, 67])
        n = int(2.0 * frames.FPS)
        note, onset = frames.rasterise_labels(truth, n)
        est = frames.decode_notes(note, onset)
        self.assertEqual(len(est), 3)
        self.assertEqual(sorted(int(p) for p in est.midi), [60, 64, 67])

    def test_decode_empty_on_silence(self):
        n = int(1.0 * frames.FPS)
        est = frames.decode_notes(np.zeros((n, frames.N_BINS), np.float32),
                                  np.zeros((n, frames.N_BINS), np.float32))
        self.assertEqual(len(est), 0)

    def test_decode_applies_onset_shift_clipped(self):
        truth = _truth([0.0], [0.5], [60])
        n = int(1.0 * frames.FPS)
        note, onset = frames.rasterise_labels(truth, n)
        est = frames.decode_notes(note, onset, onset_shift=-5.0)
        self.assertGreaterEqual(float(est.onset[0]), 0.0)


class ModelTests(unittest.TestCase):
    def _signal(self, rng, seconds=5.0):
        n = int(seconds * frames.FPS)
        onsets, offsets, pitches, t = [], [], [], 0.0
        while t < seconds - 0.4:
            dur = float(rng.uniform(0.3, 0.6))
            onsets.append(t); offsets.append(min(t + dur, seconds)); pitches.append(float(int(rng.integers(58, 70))))
            t += dur
        truth = _truth(onsets, offsets, pitches)
        yn, yo = frames.rasterise_labels(truth, n)
        note = np.clip(0.8 * yn + 0.05 * rng.standard_normal(yn.shape), 0, 1).astype(np.float32)
        on = np.clip(0.85 * yo + 0.03 * rng.standard_normal(yo.shape), 0, 1).astype(np.float32)
        mel = (0.2 * rng.standard_normal((n, frames.MEL_BANDS))).astype(np.float32)
        return frames.frames_from_posteriors(note, on, mel), yn, yo, truth

    def test_model_learns_and_decodes(self):
        rng = np.random.default_rng(0)
        seqs = [self._signal(rng)[:3] for _ in range(8)]
        model = frames.FrameModel(hidden=24, seed=0)
        info = model.fit(seqs, epochs=30, window=256, batch=4)
        self.assertIsNotNone(info["loss"])
        feats, _, _, truth = self._signal(np.random.default_rng(42))
        pn, po = model.predict(feats)
        est = frames.decode_notes(pn, po)
        self.assertGreater(len(est), 0)

    def test_model_is_deterministic(self):
        rng = np.random.default_rng(1)
        seqs = [self._signal(rng)[:3] for _ in range(4)]
        a = frames.FrameModel(hidden=16, seed=7); a.fit(seqs, epochs=10, window=128, batch=2)
        b = frames.FrameModel(hidden=16, seed=7); b.fit(seqs, epochs=10, window=128, batch=2)
        feats = seqs[0][0]
        self.assertTrue(np.allclose(a.predict(feats)[0], b.predict(feats)[0], atol=1e-5))

    def test_empty_fit(self):
        model = frames.FrameModel(seed=0)
        self.assertEqual(model.fit([])["n"], 0)

    def test_dataset_round_trip(self):
        rng = np.random.default_rng(2)
        feats, yn, yo, _ = self._signal(rng)
        ds = frames.FrameDataset(seqs=[(feats, yn, yo)], songs=["s1"], seed=5)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ds.npz"
            ds.save(p)
            back = frames.FrameDataset.load(p)
        self.assertEqual(back.songs, ["s1"])
        self.assertEqual(back.seed, 5)
        self.assertTrue(np.allclose(back.seqs[0][0], feats))

    def test_save_load_model(self):
        model = frames.FrameModel(hidden=16, seed=3)
        feats = np.random.default_rng(0).standard_normal((50, frames.FEATURE_DIM)).astype(np.float32)
        before = model.predict(feats)[0]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.pt"
            model.save(p)
            back = frames.FrameModel.load(p)
        self.assertTrue(np.allclose(back.predict(feats)[0], before, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
