"""Import-boundary guard: NC/SheetSage evaluation code must not be reachable from product/training code.

Encodes the process-reset rules (report §4.4 item 6, §9.2): training/product modules consume a read-only
denylist and the strict scorer, never the NC dataset builders; and no SheetSage runner/env/output lives in
the bake-off package.
"""

import subprocess
import sys
import unittest
from pathlib import Path

BAKEOFF_DIR = Path(__file__).resolve().parents[1] / "open_transcriber" / "bakeoff"

#: NC/eval-only modules that training and product code must never pull in transitively.
NC_MODULES = (
    "open_transcriber.bakeoff.manifests",
    "open_transcriber.bakeoff.refs",
    "open_transcriber.bakeoff.rwc_midi",
    "open_transcriber.bakeoff.ultrastar",
    "open_transcriber.bakeoff.lyrics_guard",
    "open_transcriber.bakeoff.qa",
    "open_transcriber.bakeoff.crepe",
    "open_transcriber.bakeoff.onsets",
)


def _modules_after_import(target: str) -> set[str]:
    """The set of imported module names after a fresh import of ``target`` in a clean subprocess."""
    code = (
        "import sys, json, importlib;"
        f"importlib.import_module({target!r});"
        "print(json.dumps(sorted(sys.modules)))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    import json
    return set(json.loads(out.splitlines()[-1]))


class ImportBoundaryTests(unittest.TestCase):
    def test_training_data_modules_do_not_pull_in_nc_builders(self):
        # The seeded sampler and the held-out denylist are training-side; they must not drag in NC datasets.
        for target in ("open_transcriber.data.scores", "open_transcriber.data.heldout"):
            loaded = _modules_after_import(target)
            leaked = sorted(m for m in NC_MODULES if m in loaded)
            self.assertEqual(leaked, [], f"{target} transitively imports NC/eval modules: {leaked}")

    def test_no_sheetsage_file_in_the_bakeoff_package(self):
        offenders = [p.name for p in BAKEOFF_DIR.rglob("*") if "sheetsage" in p.name.lower()]
        self.assertEqual(offenders, [], f"SheetSage files must not live in the bake-off package: {offenders}")

    def test_scorer_does_not_import_nc_dataset_builders(self):
        # The strict scorer is shared infrastructure; importing it must not load NC dataset builders.
        loaded = _modules_after_import("open_transcriber.bakeoff.score")
        leaked = sorted(m for m in NC_MODULES if m in loaded)
        self.assertEqual(leaked, [], f"bakeoff.score transitively imports NC/eval modules: {leaked}")


if __name__ == "__main__":
    unittest.main()
