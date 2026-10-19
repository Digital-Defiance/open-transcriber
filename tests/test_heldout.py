import tempfile
import unittest
from pathlib import Path

from copyist.song import load_song

from open_transcriber.data import heldout


def _song_from_yaml(text: str):
    """Write a copyist-song/0 YAML string to a tmp file and load it."""
    path = Path(tempfile.mktemp(suffix=".yaml"))
    path.write_text(text, encoding="utf-8")
    try:
        return load_song(path)
    finally:
        path.unlink(missing_ok=True)


# Amazing Grace ("New Britain"), transposed up a whole tone into A and given a
# different note-length pattern. The pitch intervals are unchanged, so the
# melodic fingerprint must still catch it.
NEW_BRITAIN_DISGUISED = """format: copyist-song/0
title: A Quiet Hymn
key: A
meter: 4/4
tempo: 100
melody: |
  E | A3 c/A/ | c3 B | A2 F2 | E4 E2 | A2 c/A/ | c3 B | e4- | e2 c2 |
  e3 c/A/ | c2 B2 | A4 F | E2 E2 | A3 c/A/ | c2 B | A6- | A2 |]
"""

# A fresh, unrelated diatonic tune: a plain C-major scale run with a distinct
# title and lyric. Its interval set should not touch any held-out tune.
FRESH_TUNE = """format: copyist-song/0
title: Counting Sheep Lullaby
key: C
meter: 4/4
tempo: 96
melody: |
  "C"C D E F | "G"G A G2 | "F"F E D C | "C"C4 |
lyrics:
  - One lit-tle lamb goes walk-ing far a-way to-night
"""

# Same melody as the fresh tune but titled so it normalises to "Amazing Grace".
TITLE_CLASH = """format: copyist-song/0
title: "Amazing, Grace!"
key: C
meter: 4/4
tempo: 96
melody: |
  "C"C D E F | "G"G A G2 | "F"F E D C | "C"C4 |
"""

# Fresh melody but carrying an Amazing Grace verse's lyric text.
LYRIC_CLASH = """format: copyist-song/0
title: Borrowed Words
key: C
meter: 4/4
tempo: 96
melody: |
  "C"C D E F | "G"G A G2 | "F"F E D C | "C"C4 |
lyrics:
  - A-maz-ing_ grace, how sweet the sound
"""


class HeldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.heldout = heldout.load_heldout()

    def test_load_heldout_returns_five_named_tunes(self):
        names = sorted(t.name for t in self.heldout)
        self.assertEqual(
            names,
            ["amazing-grace", "major-general", "silent-night", "the-tyger", "when-the-saints"],
        )

    def test_transposed_rerhythmed_amazing_grace_is_rejected(self):
        song = _song_from_yaml(NEW_BRITAIN_DISGUISED)
        result = heldout.is_held_out(song, self.heldout)
        self.assertTrue(result.is_leak)
        self.assertEqual(result.matched_tune, "amazing-grace")
        self.assertIn("gram", result.reason)

    def test_fresh_unrelated_tune_is_accepted(self):
        song = _song_from_yaml(FRESH_TUNE)
        result = heldout.is_held_out(song, self.heldout)
        self.assertFalse(result.is_leak, result.reason)

    def test_title_match_is_rejected(self):
        song = _song_from_yaml(TITLE_CLASH)
        result = heldout.is_held_out(song, self.heldout)
        self.assertTrue(result.is_leak)
        self.assertEqual(result.matched_tune, "amazing-grace")
        self.assertIn("title", result.reason)

    def test_lyric_match_is_rejected(self):
        song = _song_from_yaml(LYRIC_CLASH)
        result = heldout.is_held_out(song, self.heldout)
        self.assertTrue(result.is_leak)
        self.assertEqual(result.matched_tune, "amazing-grace")
        self.assertIn("lyric", result.reason)

    def test_thresholds_are_tunable(self):
        # The disguised Amazing Grace leaks at the defaults; an impossibly large
        # gram_n (longer than the melody) with a share_frac above 1.0 lets it
        # through, proving the thresholds are honoured.
        song = _song_from_yaml(NEW_BRITAIN_DISGUISED)
        loose = heldout.is_held_out(song, self.heldout, gram_n=999, share_frac=1.01)
        self.assertFalse(loose.is_leak, loose.reason)

    def test_check_candidate_on_a_held_out_song_file(self):
        # A held-out song checked against itself is obviously a leak.
        path = self.heldout[0].path
        result = heldout.check_candidate(path, self.heldout)
        self.assertTrue(result.is_leak)


if __name__ == "__main__":
    unittest.main()
