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


class _CountingExtractor:
    """A FakeExtractor whose inner ``features`` calls are counted, to prove the cache avoids recompute."""

    def __init__(self, model_hash="h1", frames=4, feature_dim=8):
        self.calls = 0
        self._inner = ex.FakeExtractor(feature_dim=feature_dim, frames=frames)
        self._desc = ex.ExtractorDescriptor("counting", "MIT", "MIT", 25.0, feature_dim,
                                            model_hash=model_hash)

    def descriptor(self):
        return self._desc

    def features(self, wav_path):
        self.calls += 1
        return self._inner.features(wav_path)


class CachedExtractorTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())

    def _clip(self, name="clip.wav", data=b"RIFFfake"):
        path = self.tmp / name
        path.write_bytes(data)
        return path

    def test_first_call_writes_npz_second_call_reads_without_recompute(self):
        inner = _CountingExtractor()
        cached = ex.CachedExtractor(inner, cache_dir=self.tmp / "cache")
        clip = self._clip()
        a = cached.features(clip)
        self.assertTrue(cached.cache_path(clip).exists())
        b = cached.features(clip)
        self.assertEqual(inner.calls, 1)  # computed once across two outer calls
        for layer in ex.LAYER_NAMES:
            np.testing.assert_array_equal(a[layer], b[layer])

    def test_different_clip_different_cache_file(self):
        inner = _CountingExtractor()
        cached = ex.CachedExtractor(inner, cache_dir=self.tmp / "cache")
        p1 = cached.cache_path(self._clip("a.wav", b"AAAA"))
        p2 = cached.cache_path(self._clip("b.wav", b"BBBB"))
        self.assertNotEqual(p1, p2)

    def test_model_hash_change_invalidates_key(self):
        clip = self._clip()
        k1 = ex.CachedExtractor(_CountingExtractor(model_hash="h1"), cache_dir=self.tmp).cache_path(clip)
        k2 = ex.CachedExtractor(_CountingExtractor(model_hash="h2"), cache_dir=self.tmp).cache_path(clip)
        self.assertNotEqual(k1, k2)


class RegistryAndDropTests(unittest.TestCase):
    def test_heartcodec_is_a_dropped_stub_that_refuses_to_extract(self):
        stub = ex.build_extractor("heartcodec")
        desc = stub.descriptor()
        self.assertTrue(desc.dropped)
        self.assertIn("dropped", desc.note.lower())
        self.assertIn("mps", desc.note.lower())
        with self.assertRaises(RuntimeError):
            stub.features(Path("x.wav"))

    def test_unknown_extractor_raises(self):
        with self.assertRaises(ValueError):
            ex.build_extractor("not-a-real-encoder")


if __name__ == "__main__":
    unittest.main()
