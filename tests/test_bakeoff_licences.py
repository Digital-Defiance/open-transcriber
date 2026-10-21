"""Tests for the evaluation environment's licence audit."""

import sys
import unittest
from pathlib import Path

from open_transcriber.bakeoff import licences, paths


class ClassifyTests(unittest.TestCase):
    def family(self, expression=None, text=None, classifiers=()):
        return licences.classify("pkg", "1.0", expression, text, list(classifiers)).family

    def test_spdx_expressions(self):
        self.assertEqual(self.family("MIT"), "MIT")
        self.assertEqual(self.family("Apache-2.0 OR BSD-2-Clause"), "Apache-2.0")
        self.assertEqual(self.family("(MIT AND BSD-3-Clause)"), "MIT")
        self.assertEqual(self.family("LGPL-2.1-or-later OR MIT"), "MIT")

    def test_classifiers(self):
        self.assertEqual(self.family(classifiers=["License :: OSI Approved :: BSD License"]), "BSD")
        self.assertEqual(self.family(classifiers=["License :: OSI Approved :: Apache Software License"]),
                         "Apache-2.0")

    def test_free_text(self):
        for text, family in [("MIT", "MIT"), ("MIT License", "MIT"), ("BSD-2-Clause", "BSD"),
                             ("3-Clause BSD License", "BSD"), ("Apache 2.0", "Apache-2.0"),
                             ("Apache License, Version 2.0", "Apache-2.0")]:
            with self.subTest(text=text):
                self.assertEqual(self.family(text=text), family)

    def test_copyleft_and_unknown_fail(self):
        # Should fail: copyleft or unplaceable licences are never allowed.
        self.assertIsNone(self.family("LGPL-2.1-or-later"))
        self.assertIsNone(self.family("MPL-2.0"))
        self.assertIsNone(self.family(classifiers=[
            "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)"]))
        self.assertIsNone(self.family(classifiers=[
            "License :: OSI Approved :: MIT License",
            "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)"]))
        self.assertIsNone(self.family(text="GPL-3.0"))
        self.assertIsNone(self.family(text="Proprietary"))
        self.assertIsNone(self.family())

    def test_the_evaluation_environment_passes(self):
        if Path(sys.prefix).resolve() != (paths.data_dir() / "env").resolve():
            self.skipTest("not running in the judge's evaluation environment")
        bad = [v.name for v in licences.audit() if v.family is None]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
