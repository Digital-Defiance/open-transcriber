"""probe/data.py: features + score label JSON -> the exact train/eval tensor shapes (CPU, no model)."""

import unittest
from pathlib import Path

import numpy as np
import torch

from open_transcriber.bakeoff import dev_splits
from open_transcriber.bakeoff.probe import data as D
from open_transcriber.bakeoff.probe import model as M
from open_transcriber.bakeoff.probe.extractor import ExtractorDescriptor, FakeExtractor
from open_transcriber.bakeoff.probe.train import LOSS_WEIGHTS

#: A tiny by-construction label record (the ScoreLabels JSON shape the scorer consumes).
LABELS = {
    "duration": 2.0,
    "key_class": 3,
    "meter": "4/4",
    "tempo": 120,
    "notes": [{"onset": 0.0, "offset": 0.5, "midi": 60}, {"onset": 1.0, "offset": 2.0, "midi": 64}],
    "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
    "downbeats": [0.0, 2.0],
    "chords": [{"onset": 0.0, "offset": 2.0, "root_pc": 0, "quality": "maj"}],
}


class FakeBP:
    """A fake Basic Pitch side input: fixed-shape note/onset/contour posteriors (not read from audio)."""
    bp_dim = M.BP_POSTERIOR_DIM

    def posteriors(self, wav_path):
        frames = 40
        return {
            "note": np.random.default_rng(0).random((frames, 88)).astype(np.float32),
            "onset": np.random.default_rng(1).random((frames, 88)).astype(np.float32),
            "contour": np.random.default_rng(2).random((frames, 264)).astype(np.float32),
        }


class BuildTargetsTests(unittest.TestCase):
    def test_target_frames_and_head_keys(self):
        frames = D.frame_count(LABELS["duration"])
        self.assertEqual(frames, int(round(2.0 * M.TARGET_RATE_HZ)))
        targets = D.build_targets(LABELS, frames)
        # every head the loss expects is present and frame-aligned at 50 Hz
        for head in LOSS_WEIGHTS:
            self.assertIn(head, targets)
            if head == "key":
                self.assertEqual(targets[head].shape, torch.Size([]))
            else:
                self.assertEqual(targets[head].shape[0], frames)
        # pitch/silence reflect the two notes
        self.assertEqual(int(targets["pitch"][0]), 60)
        self.assertEqual(float(targets["silence"][0]), 0.0)
        self.assertEqual(float(targets["silence"][frames // 2 - 1]), 1.0)  # the gap between notes is silent
        self.assertEqual(int(targets["key"]), 3)

    def test_chord_root_and_quality_classes(self):
        frames = D.frame_count(LABELS["duration"])
        targets = D.build_targets(LABELS, frames)
        self.assertEqual(int(targets["chord_root"][0]), 0)     # C
        self.assertEqual(int(targets["chord_quality"][0]), 0)  # maj


class BuildBatchesTests(unittest.TestCase):
    def test_build_train_batches_exact_shapes(self):
        ext = FakeExtractor(feature_dim=16, frames=50)  # native 25 Hz; will be resampled to 50 Hz
        clips = [{"wav": "a.wav", "labels": LABELS}, {"wav": "b.wav", "labels": LABELS}]
        batches = D.build_train_batches(clips, ext, FakeBP())
        self.assertEqual(len(batches), 1)  # same frame count -> one batch
        b = batches[0]
        frames = D.frame_count(LABELS["duration"])
        self.assertEqual(b["layers"].shape, (2, 3, frames, 16))
        self.assertEqual(b["bp"].shape, (2, frames, M.BP_POSTERIOR_DIM))
        for head in LOSS_WEIGHTS:
            self.assertIn(head, b["targets"])
        self.assertEqual(b["targets"]["pitch"].shape, (2, frames))
        self.assertEqual(b["targets"]["key"].shape, (2,))

    def test_build_train_batches_feed_the_real_train_loop(self):
        from open_transcriber.bakeoff.probe.train import TrainConfig, train_probe
        ext = FakeExtractor(feature_dim=16, frames=50)
        clips = [{"wav": "a.wav", "labels": LABELS}]
        batches = D.build_train_batches(clips, ext, FakeBP())
        cfg = TrainConfig(probe=M.ProbeConfig(feature_dim=16), max_epochs=1, patience=1)
        net, result = train_probe(cfg, batches, batches)
        self.assertTrue(result.best_val == result.best_val)  # finite loss -> shapes line up end to end

    def test_build_eval_clips_tuple_shape_and_scorer_consumes_it(self):
        from open_transcriber.bakeoff.probe.evaluate import score_one_clip
        ext = FakeExtractor(feature_dim=16, frames=50)
        cfg_model = M.ProbeModel(M.ProbeConfig(feature_dim=16))
        clips = [{"id": "sampler-1", "wav": "a.wav", "labels": LABELS}]
        out_clips = D.build_eval_clips(clips, cfg_model, ext, FakeBP())
        self.assertEqual(len(out_clips), 1)
        out, labels, item = out_clips[0]
        self.assertEqual(item, "sampler-1")
        self.assertIn("pitch", out)
        row = score_one_clip(out, labels, item=item)   # the scorer consumes it without error
        self.assertIn("oct_on50_f1", row)

    def test_eval_refuses_sealed_test_id(self):
        ext = FakeExtractor(feature_dim=8, frames=50)
        cfg_model = M.ProbeModel(M.ProbeConfig(feature_dim=8))
        sealed = sorted(dev_splits.rwc_pop_test())[0]
        clips = [{"id": sealed, "wav": "a.wav", "labels": LABELS}]
        with self.assertRaises(ValueError):
            D.build_eval_clips(clips, cfg_model, ext, FakeBP())


if __name__ == "__main__":
    unittest.main()
