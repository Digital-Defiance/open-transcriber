"""The VocalSet split is disjoint by singer, deterministic, and 12/4/4."""

import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import vocalset_split as vs


class VocalSetSplitTests(unittest.TestCase):
    def test_counts(self):
        split = vs.build_split()
        self.assertEqual((len(split.train), len(split.val), len(split.audit)), (12, 4, 4))

    def test_singer_disjoint(self):
        split = vs.build_split()
        self.assertEqual(len(set(split.all_singers())), 20)  # all distinct
        self.assertFalse(set(split.train) & set(split.val))
        self.assertFalse(set(split.train) & set(split.audit))
        self.assertFalse(set(split.val) & set(split.audit))

    def test_deterministic(self):
        a = vs.build_split().to_dict()
        b = vs.build_split().to_dict()
        self.assertEqual(a, b)

    def test_refuses_too_few_singers(self):
        with self.assertRaises(ValueError):
            vs.build_split(singers=("one", "two", "three"))

    def test_write_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = vs.write_manifest(vs.build_split(), Path(tmp) / "vocalset.json")
            self.assertTrue(out.exists())
            import json
            data = json.loads(out.read_text())
            self.assertEqual(data["licence"], "CC BY 4.0")
            self.assertEqual(len(data["train"]), 12)


if __name__ == "__main__":
    unittest.main()
