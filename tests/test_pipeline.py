import unittest
from unittest.mock import patch

from open_transcriber.notes import NoteEvent
from open_transcriber import pipeline as pl


class PipelineTests(unittest.TestCase):
    def test_adaptive_uses_loose_when_underdetecting(self):
        tight = [NoteEvent(float(i), 0.2, 60) for i in range(20)]   # 20 << 128
        loose = [NoteEvent(float(i) * 0.15, 0.12, 70) for i in range(120)]

        def fake_transcribe(audio, **kw):
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) <= 0.55:
                return list(loose)
            return list(tight)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe), \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=128)
        # Loose path keeps patter-length notes (not collapsed by same-pitch merge).
        self.assertGreater(len(out), 80)

    def test_adaptive_keeps_default_when_near_target(self):
        near = [NoteEvent(float(i) * 0.3, 0.25, 60 + (i % 5), 0.7) for i in range(50)]

        with patch.object(pl, "transcribe_melody", return_value=list(near)) as mock, \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=46)
        # Default + alt presets (same notes → default wins on index).
        self.assertEqual(mock.call_count, 1 + len(pl.STRICT_PRESETS))
        self.assertGreaterEqual(len(out), 40)

    def test_adaptive_uses_strict_when_it_hits_target(self):
        flooded = [NoteEvent(float(i) * 0.2, 0.15, 60 + (i % 7), 0.4) for i in range(41)]
        strict = [NoteEvent(float(i) * 0.35, 0.3, 62 + (i % 4), 0.6) for i in range(33)]

        def fake_transcribe(audio, **kw):
            # Any raised-threshold / melodia-off preset returns the tight pass.
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) >= 0.80:
                return list(strict)
            return list(flooded)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe) as mock, \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=33)
        self.assertEqual(mock.call_count, 1 + len(pl.STRICT_PRESETS))
        self.assertEqual(len(out), 33)

    def test_adaptive_ranks_by_amp_when_counts_tie(self):
        """Same cleaned count → keep the higher-confidence BP pass (AG 0.85)."""
        low = [NoteEvent(float(i) * 0.3, 0.25, 60 + (i % 3), 0.4) for i in range(40)]
        high = [NoteEvent(float(i) * 0.3, 0.25, 62 + (i % 3), 0.8) for i in range(40)]

        def fake_transcribe(audio, **kw):
            ot = kw.get("onset_threshold", pl.DEFAULT_ONSET)
            if ot >= 0.84:
                return list(high)
            return list(low)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe), \
             patch.object(pl, "clean_with_copyist", side_effect=lambda evs, *a, **k: list(evs)[:32]):
            out = pl.melody_from_audio("dummy.wav", target=32)
        self.assertEqual(len(out), 32)
        self.assertEqual(out[0].midi, 62)  # from the high-amp pass

    def test_adaptive_keeps_default_on_mild_overdetect(self):
        """Tyger-like: a slightly closer strict pass must not displace a good default."""
        default = [NoteEvent(float(i) * 0.25, 0.2, 60 + (i % 5), 0.7) for i in range(37)]
        strict = [NoteEvent(float(i) * 0.3, 0.25, 61 + (i % 4), 0.5) for i in range(35)]

        def fake_transcribe(audio, **kw):
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) >= 0.80:
                return list(strict)
            return list(default)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe), \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=29)
        # improve = 2 notes < MIN_COUNT_IMPROVE → keep default.
        self.assertEqual(len(out), 37)
        self.assertEqual(out[0].midi, 60)

    def test_adaptive_switches_when_count_clearly_better(self):
        default = [NoteEvent(float(i) * 0.2, 0.15, 60, 0.5) for i in range(48)]
        tight = [NoteEvent(float(i) * 0.3, 0.25, 62, 0.5) for i in range(33)]

        def fake_transcribe(audio, **kw):
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) >= 0.80:
                return list(tight)
            return list(default)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe), \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=33)
        self.assertEqual(len(out), 33)
        self.assertEqual(out[0].midi, 62)

    def test_copyist_clean_used_when_available(self):
        raw = [NoteEvent(float(i) * 0.3, 0.25, 60 + (i % 4)) for i in range(40)]
        cleaned = [NoteEvent(float(i) * 0.35, 0.3, 62) for i in range(32)]
        with patch.object(pl, "transcribe_melody", return_value=list(raw)), \
             patch.object(pl, "clean_with_copyist", return_value=list(cleaned)) as mock:
            out = pl.melody_from_audio("dummy.wav", target=32)
        # One clean call per candidate (default + presets).
        self.assertEqual(mock.call_count, 1 + len(pl.STRICT_PRESETS))
        self.assertEqual(len(out), 32)


if __name__ == "__main__":
    unittest.main()
