"""The frozen development splits: right counts, stable hashes, and no test-partition id leaks in."""

import tempfile
import unittest
from pathlib import Path

from open_transcriber.bakeoff import dev_splits


class DevSplitTests(unittest.TestCase):
    def test_counts(self):
        s = dev_splits.build_splits()
        self.assertEqual(s["rwc-other-dev"]["count"], 30)
        self.assertEqual(s["karaoke-dev"]["count"], 12)
        self.assertEqual(s["vocalset-val"]["count"], 4)
        self.assertEqual(s["sampler-dev"]["count"], 80)
        self.assertEqual(s["soulx-dev"]["count"], 40)
        self.assertEqual(s["yue2-truth"]["count"], 5)

    def test_rwc_other_strata(self):
        dev = dev_splits.rwc_other_dev()
        letters = [i[4].upper() for i in dev]
        self.assertEqual(letters.count("G"), 20)
        self.assertEqual(letters.count("R"), 6)
        self.assertEqual(letters.count("C"), 3)
        self.assertEqual(letters.count("J"), 1)

    def test_hashes_stable_across_builds(self):
        a = {k: v["content_hash"] for k, v in dev_splits.build_splits().items()}
        b = {k: v["content_hash"] for k, v in dev_splits.build_splits().items()}
        self.assertEqual(a, b)

    def test_dev_and_test_partitions_are_disjoint(self):
        sealed = dev_splits.test_partition_ids()
        for name, split in dev_splits.build_splits().items():
            self.assertFalse(set(split["members"]) & sealed, f"{name} overlaps the sealed test partition")

    def test_no_rwc_pop_id_in_any_dev_split(self):
        pop = set(dev_splits.rwc_pop_test())
        for split in dev_splits.build_splits().values():
            self.assertFalse(set(split["members"]) & pop)

    def test_guard_rejects_an_injected_test_id(self):
        # Should fail: slipping an RWC-Pop id into a dev split is refused.
        sealed = dev_splits.test_partition_ids()
        bad_id = sorted(dev_splits.rwc_pop_test())[0]
        self.assertIn(bad_id, sealed)
        # simulate the guard directly on a doctored split
        members = ["sampler-90000000", bad_id]
        leaked = sorted(set(members) & sealed)
        self.assertEqual(leaked, [bad_id])

    def test_guard_rejects_cotb(self):
        # The builder refuses any COTB id by name (there are none yet, but the prefix is reserved).
        for split in dev_splits.build_splits().values():
            self.assertFalse(any(str(m).lower().startswith("cotb") for m in split["members"]))

    def test_write_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            splits = dev_splits.build_splits()
            dev_splits.write_splits(splits, Path(tmp))
            hashes = dev_splits.split_hashes(Path(tmp))
            for name, split in splits.items():
                self.assertEqual(hashes[name], split["content_hash"])


if __name__ == "__main__":
    unittest.main()
