import tempfile
import unittest
from collections import Counter
from pathlib import Path

from copyist.song import load_song
from copyist.validate import is_finished, validate

from open_transcriber.data import heldout as heldout_mod
from open_transcriber.data import scores


# A hand-built copyist-song/0 YAML whose lyric line has more syllables than the
# melody has notes: check_lyrics_fit must FAIL on it (the required should-fail
# input).
BROKEN_YAML = """format: copyist-song/0
title: Too Many Words
key: C
meter: 4/4
tempo: 100
melody: |
  "C"C D E F | "G"G A G2 |
lyrics:
  - one two three four five six sev en eight nine ten
"""


class DeterminismTests(unittest.TestCase):
    def test_same_seed_is_byte_identical(self):
        for seed in (0, 1, 7, 42, 99):
            self.assertEqual(
                scores.sample_song(seed).yaml_text,
                scores.sample_song(seed).yaml_text,
                f"seed {seed} not deterministic",
            )

    def test_different_seeds_generally_differ(self):
        texts = {scores.sample_song(seed).yaml_text for seed in range(12)}
        # They should not collapse to one or two identical songs.
        self.assertGreater(len(texts), 8)


class ValidatorTests(unittest.TestCase):
    def test_sampled_batch_all_finish(self):
        for seed in range(30):
            song = scores.sample_song(seed)
            findings = scores.validate_song_text(song.yaml_text)
            self.assertTrue(
                is_finished(findings),
                msg=f"seed {seed} failed: "
                + "; ".join(f.message for f in findings if f.severity == "fail"),
            )

    def test_broken_input_fails_validation(self):
        # The required should-fail input: validators must report a FAIL.
        findings = scores.validate_song_text(BROKEN_YAML)
        self.assertFalse(is_finished(findings))
        self.assertTrue(any(f.check == "lyrics-fit" and f.severity == "fail" for f in findings))

    def test_validate_song_text_matches_direct_validate(self):
        text = scores.sample_song(3).yaml_text
        path = Path(tempfile.mktemp(suffix=".yaml"))
        path.write_text(text, encoding="utf-8")
        try:
            direct = validate(load_song(path))
        finally:
            path.unlink(missing_ok=True)
        helper = scores.validate_song_text(text)
        self.assertEqual(
            [(f.check, f.severity) for f in direct],
            [(f.check, f.severity) for f in helper],
        )


class _FakeHeldout:
    """Leaks the first distinct song text it sees, then accepts everything."""

    def __init__(self):
        self.first_text = None

    def is_held_out(self, song, heldout, **kwargs):
        text = song.melody_abc
        if self.first_text is None:
            self.first_text = text
            return heldout_mod.LeakResult(True, "fake leak", "fake-tune")
        if text == self.first_text:
            return heldout_mod.LeakResult(True, "fake leak", "fake-tune")
        return heldout_mod.LeakResult(False)


class HeldoutFilterTests(unittest.TestCase):
    def test_write_batch_rejects_real_held_out_leaks(self):
        heldout = heldout_mod.load_heldout()
        with tempfile.TemporaryDirectory() as d:
            paths = scores.write_batch(Path(d), n=8, start_seed=100, heldout=heldout)
            self.assertEqual(len(paths), 8)
            for p in paths:
                result = heldout_mod.is_held_out(load_song(p), heldout)
                self.assertFalse(result.is_leak, f"{p.name} leaks: {result.reason}")

    def test_write_batch_rerolls_on_a_leak(self):
        # Inject a fake held-out guard that leaks the first candidate, forcing a
        # deterministic re-roll; the batch must still return n files, and the
        # kept song must differ from the first (rejected) sampling.
        fake = _FakeHeldout()
        original = heldout_mod.is_held_out
        heldout_mod.is_held_out = fake.is_held_out
        try:
            with tempfile.TemporaryDirectory() as d:
                paths = scores.write_batch(Path(d), n=1, start_seed=0, heldout=fake)
                self.assertEqual(len(paths), 1)
                kept = load_song(paths[0])
                self.assertNotEqual(kept.melody_abc, fake.first_text)
        finally:
            heldout_mod.is_held_out = original


class CoverageTests(unittest.TestCase):
    """Prove the widened musical range across a batch, read from SampledSong.meta."""

    @classmethod
    def setUpClass(cls):
        cls.metas = [scores.sample_song(seed).meta for seed in range(60)]

    def test_multiple_keys_including_minor(self):
        keys = {m["key"] for m in self.metas}
        self.assertGreaterEqual(len(keys), 8)
        self.assertTrue(any(m["minor"] for m in self.metas))

    def test_most_meters_appear(self):
        meters = {m["meter"] for m in self.metas}
        self.assertGreaterEqual(len(meters & set(scores.METERS)), 4)

    def test_wide_tempo_span(self):
        tempos = [m["tempo"] for m in self.metas]
        self.assertGreaterEqual(max(tempos) - min(tempos), 60)
        self.assertTrue(all(50 <= t <= 180 for t in tempos))

    def test_each_rhythmic_feature_appears(self):
        counts = Counter()
        for m in self.metas:
            for feat in ("sixteenths", "dotted", "syncopation", "ties", "melisma", "patter"):
                if m.get(feat):
                    counts[feat] += 1
        for feat in ("sixteenths", "dotted", "syncopation", "ties", "melisma", "patter"):
            self.assertGreater(counts[feat], 0, f"{feat} never appeared across the batch")


if __name__ == "__main__":
    unittest.main()
