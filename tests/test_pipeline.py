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
        near = [NoteEvent(float(i) * 0.3, 0.25, 60 + (i % 5)) for i in range(50)]

        with patch.object(pl, "transcribe_melody", return_value=list(near)) as mock, \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=46)
        self.assertEqual(mock.call_count, 1)
        self.assertGreaterEqual(len(out), 40)

    def test_adaptive_uses_strict_when_it_hits_target(self):
        flooded = [NoteEvent(float(i) * 0.2, 0.15, 60 + (i % 7)) for i in range(41)]
        strict = [NoteEvent(float(i) * 0.35, 0.3, 62 + (i % 4)) for i in range(33)]

        def fake_transcribe(audio, **kw):
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) >= pl.STRICT_ONSET - 1e-9:
                return list(strict)
            return list(flooded)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe) as mock, \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=33)
        # Default pass + both STRICT_PRESETS.
        self.assertEqual(mock.call_count, 1 + len(pl.STRICT_PRESETS))
        self.assertEqual(len(out), 33)

    def test_adaptive_keeps_default_on_mild_overdetect(self):
        """Tyger-like: strict is closer but not enough to justify the switch."""
        default = [NoteEvent(float(i) * 0.25, 0.2, 60 + (i % 5)) for i in range(39)]
        strict = [NoteEvent(float(i) * 0.3, 0.25, 61 + (i % 4)) for i in range(35)]

        def fake_transcribe(audio, **kw):
            if kw.get("onset_threshold", pl.DEFAULT_ONSET) >= pl.STRICT_ONSET - 1e-9:
                return list(strict)
            return list(default)

        with patch.object(pl, "transcribe_melody", side_effect=fake_transcribe), \
             patch.object(pl, "clean_with_copyist", return_value=None):
            out = pl.melody_from_audio("dummy.wav", target=29)
        # Default tidied count stays (soft overshoot may fold a little).
        self.assertGreaterEqual(len(out), 35)

    def test_copyist_clean_used_when_available(self):
        raw = [NoteEvent(float(i) * 0.3, 0.25, 60 + (i % 4)) for i in range(40)]
        cleaned = [NoteEvent(float(i) * 0.35, 0.3, 62) for i in range(32)]
        with patch.object(pl, "transcribe_melody", return_value=list(raw)), \
             patch.object(pl, "clean_with_copyist", return_value=list(cleaned)) as mock:
            out = pl.melody_from_audio("dummy.wav", target=32)
        mock.assert_called_once()
        self.assertEqual(len(out), 32)


if __name__ == "__main__":
    unittest.main()
