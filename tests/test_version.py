import unittest

import open_transcriber


class VersionTests(unittest.TestCase):
    def test_version_is_set(self):
        self.assertTrue(open_transcriber.__version__)


if __name__ == "__main__":
    unittest.main()
