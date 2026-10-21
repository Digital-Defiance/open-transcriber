"""The Extractor protocol and the deterministic FakeExtractor (CPU, no model)."""

import unittest
from pathlib import Path

import numpy as np

from open_transcriber.bakeoff.probe import extractor as ex


class ExtractorTests(unittest.TestCase):
    def test_fake_declares_layers_and_shapes(self):
        fake = ex.FakeExtractor(feature_dim=8, frames=20)
        desc = fake.descriptor()
        self.assertEqual(desc.layer_names, ex.LAYER_NAMES)
        feats = fake.features(Path("clip.wav"))
        self.assertEqual(set(feats), set(ex.LAYER_NAMES))
        for layer in ex.LAYER_NAMES:
            self.assertEqual(feats[layer].shape, (20, 8))
            self.assertEqual(feats[layer].dtype, np.float32)

    def test_fake_is_deterministic_per_path(self):
        a = ex.FakeExtractor().features(Path("x.wav"))
        b = ex.FakeExtractor().features(Path("x.wav"))
        for layer in ex.LAYER_NAMES:
            np.testing.assert_array_equal(a[layer], b[layer])
        c = ex.FakeExtractor().features(Path("y.wav"))
        self.assertFalse(np.array_equal(a["early"], c["early"]))

    def test_fake_satisfies_the_protocol(self):
        self.assertIsInstance(ex.FakeExtractor(), ex.Extractor)

    def test_licence_gate(self):
        good = ex.ExtractorDescriptor("g", "MIT", "Apache-2.0", 25.0, 8)
        self.assertTrue(good.free_and_allowed)
        bad = ex.ExtractorDescriptor("b", "MIT", "CC BY-NC 4.0", 25.0, 8)
        self.assertFalse(bad.free_and_allowed)


if __name__ == "__main__":
    unittest.main()
