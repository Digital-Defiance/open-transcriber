import unittest

from open_transcriber.notes import NoteEvent
from open_transcriber.tidy import fold_to_target, merge_same_pitch, tidy


class TidyTests(unittest.TestCase):
    def test_merge_same_pitch_joins_grace_echo(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.52, 0.08, 60),  # short echo
            NoteEvent(1.0, 0.4, 62),
        ]
        out = merge_same_pitch(notes, gap=0.15)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].midi, 60)

    def test_merge_same_pitch_keeps_patter_eighths(self):
        notes = [NoteEvent(i * 0.2, 0.18, 70) for i in range(8)]
        out = merge_same_pitch(notes, gap=0.15)
        self.assertEqual(len(out), 8)

    def test_drop_pitch_outliers_keeps_body_line(self):
        from open_transcriber.tidy import drop_pitch_outliers

        body = [NoteEvent(float(i), 0.3, 60 + (i % 3)) for i in range(8)]
        ghost = NoteEvent(4.0, 0.08, 84)  # short octave+ above median
        out = drop_pitch_outliers(body + [ghost])
        self.assertEqual(len(out), 8)
        self.assertTrue(all(e.midi < 70 for e in out))

    def test_fold_to_target(self):
        notes = [
            NoteEvent(0.0, 0.5, 60),
            NoteEvent(0.5, 0.08, 61),
            NoteEvent(0.6, 0.5, 62),
            NoteEvent(1.2, 0.5, 64),
        ]
        out = fold_to_target(notes, 3)
        self.assertEqual(len(out), 3)

    def test_tidy_soft_overshoot_keeps_mild_extras(self):
        # 5 notes, target 4 — under 1.6× so soft tidy does not force the count.
        notes = [NoteEvent(i * 0.5, 0.4, 60 + i) for i in range(5)]
        out = tidy(notes, target=4)
        self.assertEqual(len(out), 5)
        forced = tidy(notes, target=4, force_target=True)
        self.assertEqual(len(forced), 4)


if __name__ == "__main__":
    unittest.main()
