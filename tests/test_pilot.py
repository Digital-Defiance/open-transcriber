"""The data-engine pilot's CPU plumbing: features, labels, the head, and an input that must fail.

These exercise the Basic Pitch candidate features, the match labelling, the tiny head's train/apply, and the
dataset round-trip, without generating songs, building, or running Basic Pitch. A head trained on a learnable
synthetic rule must fit it; a mismatched feature matrix must be rejected.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff import notes_io
from open_transcriber.data import pilot


class FeatureTests(unittest.TestCase):
    def test_load_bp_notes_with_and_without_amplitude(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.tsv"
            p.write_text("onset\toffset\tmidi\tamplitude\n0.0\t0.5\t60\t0.9\n0.5\t1.0\t62\t0.7\n", encoding="utf-8")
            rows = pilot.load_bp_notes(p)
            self.assertEqual([r[2] for r in rows], [60, 62])
            self.assertAlmostEqual(rows[0][3], 0.9)
            q = Path(d) / "b.tsv"
            q.write_text("0.0\t0.5\t60\n", encoding="utf-8")  # no amplitude -> default 1.0
            rows2 = pilot.load_bp_notes(q)
            self.assertAlmostEqual(rows2[0][3], 1.0)

    def test_candidate_features_shape(self):
        notes = [(0.0, 0.5, 60, 0.9), (0.6, 1.0, 62, 0.8)]
        cands = pilot.candidate_features(notes)
        X = pilot.feature_matrix(cands)
        self.assertEqual(X.shape, (2, len(pilot.FEATURES)))

    def test_label_candidates_marks_matches(self):
        notes = [(0.0, 0.5, 60, 0.9), (0.5, 1.0, 62, 0.8), (0.6, 0.62, 80, 0.1)]
        cands = pilot.candidate_features(notes)
        truth = notes_io._validate(
            notes_io.Notes(np.array([0.0, 0.5]), np.array([0.5, 1.0]), np.array([60.0, 62.0])), False, "t")
        pilot.label_candidates(cands, truth)
        self.assertEqual(cands[0].label, 1)   # matches true C
        self.assertEqual(cands[1].label, 1)   # matches true D
        self.assertEqual(cands[2].label, 0)   # the stray high note matches nothing


class HeadTests(unittest.TestCase):
    def _synthetic(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        pitch = rng.integers(40, 90, n)
        dur = rng.uniform(0.03, 1.0, n)
        label = ((pilot.MELODY_LO <= pitch) & (pitch <= pilot.MELODY_HI) & (dur > 0.1)).astype(int)
        X = np.stack([
            (pitch - 60) / 24.0, np.minimum(dur, 2.0), rng.uniform(0, 1, n),
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
            ((pilot.MELODY_LO <= pitch) & (pitch <= pilot.MELODY_HI)).astype(float),
        ], axis=1).astype(np.float32)
        return X, label

    def test_head_learns_a_separable_rule(self):
        X, y = self._synthetic()
        head = pilot.Head(seed=0)
        head.fit(X, y, epochs=300)
        acc = float(((head.scores(X) >= 0.5).astype(int) == y).mean())
        self.assertGreaterEqual(acc, 0.9)

    def test_head_is_deterministic(self):
        X, y = self._synthetic()
        a = pilot.Head(seed=7); a.fit(X, y, epochs=50)
        b = pilot.Head(seed=7); b.fit(X, y, epochs=50)
        np.testing.assert_allclose(a.scores(X), b.scores(X), rtol=0, atol=1e-6)

    def test_apply_head_keeps_accepted_notes(self):
        X, y = self._synthetic()
        head = pilot.Head(seed=0); head.fit(X, y, epochs=300)
        notes = [(0.0, 0.5, 60, 0.9), (0.5, 1.0, 62, 0.8), (0.6, 0.62, 80, 0.1)]
        est = pilot.apply_head(notes, head, threshold=0.5, onset_shift=0.01)
        self.assertGreaterEqual(len(est), 1)
        self.assertAlmostEqual(float(est.onset[0]), 0.01, places=3)  # shift applied

    def test_apply_head_clips_negative_shift_at_zero(self):
        # A negative onset shift must not push a note before t=0 (the scorer rejects negative time).
        X, y = self._synthetic()
        head = pilot.Head(seed=0); head.fit(X, y, epochs=50)
        notes = [(0.0, 0.5, 60, 0.9), (0.02, 0.5, 62, 0.8)]
        est = pilot.apply_head(notes, head, threshold=0.0, onset_shift=-0.06)  # threshold 0 keeps all
        self.assertTrue(all(o >= 0.0 for o in est.onset.tolist()))

    def test_mismatched_feature_width_rejected(self):
        # The failing input: a feature matrix whose width is not len(FEATURES).
        head = pilot.Head(seed=0)
        bad = np.zeros((4, len(pilot.FEATURES) + 1), np.float32)
        with self.assertRaises(Exception):
            head.scores(bad)

    def test_dataset_round_trip(self):
        X, y = self._synthetic(n=10)
        ds = pilot.Dataset(X=X, y=y, songs=["s1", "s2"], seed=3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ds.npz"
            ds.save(p)
            back = pilot.Dataset.load(p)
            np.testing.assert_array_equal(back.X, X)
            np.testing.assert_array_equal(back.y, y)
            self.assertEqual(back.songs, ["s1", "s2"])
            self.assertEqual(back.seed, 3)


if __name__ == "__main__":
    unittest.main()
