import unittest

from open_transcriber.copyist_cases import CASES, select_cases
from open_transcriber.notes import NoteEvent, to_eval_notes


class NoteTests(unittest.TestCase):
    def test_eval_row_shape(self):
        rows = to_eval_notes([NoteEvent(1.0, 0.5, 60), NoteEvent(2.0, 0.0, 62)])
        self.assertEqual(rows[0], [1.0, 0.5, 60])
        self.assertGreaterEqual(rows[1][1], 0.05)


class CaseTests(unittest.TestCase):
    def test_known_cases(self):
        names = {c.name for c in CASES}
        self.assertIn("silent-night", names)
        self.assertIn("major-general", names)

    def test_select_only(self):
        chosen = select_cases("saints,tyger")
        self.assertEqual([c.name for c in chosen], ["saints", "tyger"])


if __name__ == "__main__":
    unittest.main()
