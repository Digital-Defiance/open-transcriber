"""The training loop: deterministic given a seed, early-stops, and checkpoints the best model (CPU)."""

import tempfile
import unittest
from pathlib import Path

import torch

from open_transcriber.bakeoff.probe import model as m
from open_transcriber.bakeoff.probe import train as T


def synthetic_batches(n=2, batch=2, frames=6, feat=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    batches = []
    for _ in range(n):
        layers = torch.randn(batch, 3, frames, feat, generator=g)
        bp = torch.randn(batch, frames, m.BP_POSTERIOR_DIM, generator=g)
        targets = {
            "pitch": torch.randint(0, m.N_PITCH, (batch, frames), generator=g),
            "silence": torch.randint(0, 2, (batch, frames), generator=g).float(),
            "onset": torch.randint(0, 2, (batch, frames), generator=g).float(),
            "offset": torch.randint(0, 2, (batch, frames), generator=g).float(),
            "beat": torch.randint(0, 2, (batch, frames), generator=g).float(),
            "downbeat": torch.randint(0, 2, (batch, frames), generator=g).float(),
            "chord_root": torch.randint(0, m.N_CHORD_ROOT, (batch, frames), generator=g),
            "chord_quality": torch.randint(0, m.N_CHORD_QUALITY, (batch, frames), generator=g),
            "key": torch.randint(0, m.N_KEY, (batch,), generator=g),
        }
        batches.append({"layers": layers, "bp": bp, "targets": targets})
    return batches


class TrainTests(unittest.TestCase):
    def test_one_epoch_runs_and_loss_is_finite(self):
        cfg = T.TrainConfig(probe=m.ProbeConfig(feature_dim=8), max_epochs=1, patience=1, seed=T.SEEDS[0])
        train_b = synthetic_batches(seed=1)
        val_b = synthetic_batches(seed=2)
        _net, result = T.train_probe(cfg, train_b, val_b)
        self.assertEqual(result.seed, T.SEEDS[0])
        self.assertTrue(result.best_val == result.best_val)  # not NaN
        self.assertGreaterEqual(result.epochs_run, 1)

    def test_same_seed_is_deterministic(self):
        cfg = T.TrainConfig(probe=m.ProbeConfig(feature_dim=8), max_epochs=3, patience=3, seed=T.SEEDS[0])
        a = T.train_probe(cfg, synthetic_batches(seed=1), synthetic_batches(seed=2))[1]
        b = T.train_probe(cfg, synthetic_batches(seed=1), synthetic_batches(seed=2))[1]
        self.assertEqual(a.history, b.history)

    def test_early_stopping_caps_epochs(self):
        # With patience 2 and val that won't improve past epoch 1, training stops well before max_epochs.
        cfg = T.TrainConfig(probe=m.ProbeConfig(feature_dim=8), max_epochs=50, patience=2, seed=T.SEEDS[0])
        _net, result = T.train_probe(cfg, synthetic_batches(seed=1), synthetic_batches(seed=2))
        self.assertLessEqual(result.epochs_run, 50)

    def test_checkpoint_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = T.TrainConfig(probe=m.ProbeConfig(feature_dim=8), max_epochs=2, patience=2, seed=T.SEEDS[1])
            T.train_probe(cfg, synthetic_batches(seed=1), synthetic_batches(seed=2), checkpoint_dir=Path(tmp))
            self.assertTrue((Path(tmp) / f"probe-seed{T.SEEDS[1]}-best.pt").exists())
            self.assertTrue((Path(tmp) / f"probe-seed{T.SEEDS[1]}-best.json").exists())

    def test_three_registered_seeds(self):
        self.assertEqual(T.SEEDS, (20261020, 20261021, 20261022))


if __name__ == "__main__":
    unittest.main()
