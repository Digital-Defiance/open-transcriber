"""The probe body + heads: forward pass and shapes on a tiny CPU batch; control ignores encoder features."""

import unittest

import torch

from open_transcriber.bakeoff.probe import model as m


def tiny_batch(batch=2, frames=6, feat=8, bp=m.BP_POSTERIOR_DIM, n_layers=3):
    layers = torch.randn(batch, n_layers, frames, feat)
    posteriors = torch.randn(batch, frames, bp)
    return layers, posteriors


class ProbeModelTests(unittest.TestCase):
    def test_forward_head_shapes(self):
        cfg = m.ProbeConfig(feature_dim=8)
        net = m.ProbeModel(cfg)
        layers, bp = tiny_batch()
        out = net(layers, bp)
        b, t = 2, 6
        self.assertEqual(out["pitch"].shape, (b, t, m.N_PITCH))
        self.assertEqual(out["silence"].shape, (b, t))
        self.assertEqual(out["onset"].shape, (b, t))
        self.assertEqual(out["offset"].shape, (b, t))
        self.assertEqual(out["beat"].shape, (b, t))
        self.assertEqual(out["downbeat"].shape, (b, t))
        self.assertEqual(out["chord_root"].shape, (b, t, m.N_CHORD_ROOT))
        self.assertEqual(out["chord_quality"].shape, (b, t, m.N_CHORD_QUALITY))
        self.assertEqual(out["key"].shape, (b, m.N_KEY))

    def test_control_ignores_encoder_features(self):
        cfg = m.ProbeConfig(feature_dim=8, control=True)
        net = m.ProbeModel(cfg).eval()   # eval() so dropout is off and the comparison is deterministic
        layers, bp = tiny_batch()
        with torch.no_grad():
            out_a = net(layers, bp)
            out_b = net(layers + 100.0, bp)   # change encoder features only
        # control mode never looks at the encoder layers, so outputs are identical
        for k in out_a:
            torch.testing.assert_close(out_a[k], out_b[k])

    def test_non_control_uses_encoder_features(self):
        cfg = m.ProbeConfig(feature_dim=8, control=False)
        net = m.ProbeModel(cfg).eval()
        layers, bp = tiny_batch()
        with torch.no_grad():
            out_a = net(layers, bp)
            out_b = net(layers + 100.0, bp)
        # at least one head must change when the encoder features change
        changed = any(not torch.allclose(out_a[k], out_b[k]) for k in out_a)
        self.assertTrue(changed)

    def test_layer_mix_weights_are_a_softmax(self):
        mix = m.LayerMix(3)
        w = torch.softmax(mix.weights, dim=0)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=5)

    def test_resample_changes_frame_count(self):
        x = torch.randn(1, 10, 4)
        y = m.resample_time(x, native_rate=25.0, target_rate=50.0)
        self.assertEqual(y.shape[1], 20)
        same = m.resample_time(x, native_rate=50.0, target_rate=50.0)
        self.assertEqual(same.shape[1], 10)

    def test_stack_layers(self):
        import numpy as np
        feats = {n: np.zeros((5, 8), dtype=np.float32) for n in ("early", "middle", "final")}
        t = m.stack_layers(feats)
        self.assertEqual(tuple(t.shape), (1, 3, 5, 8))


if __name__ == "__main__":
    unittest.main()
