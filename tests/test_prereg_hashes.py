"""The pre-registration file's frozen split hashes must match the committed split manifests exactly."""

import re
import unittest
from pathlib import Path

from open_transcriber.bakeoff import dev_splits

PREREG = Path(dev_splits.__file__).with_name("PREREGISTRATION.md")
# A table row like: | rwc-other-dev (...) | 30 | `<64 hex>` |
_ROW = re.compile(r"^\|\s*([a-z0-9-]+)[^|]*\|\s*\d+\s*\|\s*`([0-9a-f]{64})`\s*\|", re.MULTILINE)


class PreregHashTests(unittest.TestCase):
    def test_prereg_hashes_match_committed_manifests(self):
        text = PREREG.read_text(encoding="utf-8")
        registered = {name: h for name, h in _ROW.findall(text)}
        committed = dev_splits.split_hashes()
        self.assertTrue(registered, "no split-hash rows found in PREREGISTRATION.md")
        # every registered split exists and matches
        for name, h in registered.items():
            self.assertIn(name, committed, f"{name} registered but not committed")
            self.assertEqual(h, committed[name], f"{name} hash drifted from the committed manifest")
        # every committed split is registered
        for name in committed:
            self.assertIn(name, registered, f"{name} committed but not registered in PREREGISTRATION.md")


if __name__ == "__main__":
    unittest.main()
