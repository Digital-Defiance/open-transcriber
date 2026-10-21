"""Tests for chart QA's onset evidence."""

import unittest

import numpy as np

from open_transcriber.bakeoff import onsets

RATE = 16000


def tone_bursts(starts, seconds=3.0, length=0.25, freq=330.0):
    """Harmonic tone bursts that start abruptly at ``starts`` (seconds), over faint noise."""
    rng = np.random.default_rng(3)
    audio = 1e-4 * rng.standard_normal(int(seconds * RATE))
    t = np.arange(int(length * RATE)) / RATE
    fade = np.clip((length - t) / 0.08, 0.0, 1.0)  # gentle 80 ms release: no offset transient
    burst = sum(0.5 ** k * np.sin(2 * np.pi * freq * (k + 1) * t) for k in range(4)) * np.exp(-t * 6) * fade
    for start in starts:
        i = int(round(start * RATE))
        audio[i:i + len(burst)] += 0.4 * burst
    return audio.astype(np.float32)


class FluxTests(unittest.TestCase):
    def test_bursts_are_found_near_their_onsets(self):
        # Centre-stamped, log-compressed flux peaks about 20 ms before an abrupt onset (see onsets.py).
        starts = [0.5, 1.0, 1.7, 2.4]
        peaks = onsets.flux_peaks(onsets.spectral_flux(tone_bursts(starts), RATE))
        for start in starts:
            self.assertLessEqual(abs(onsets.nearest(peaks, start)), 0.025, f"burst at {start}")
        self.assertEqual(len(peaks), len(starts))

    def test_clicks_are_found_within_25_ms(self):
        audio = np.zeros(int(2.0 * RATE), dtype=np.float32)
        clicks = [0.3, 0.9, 1.45]
        for c in clicks:
            audio[int(c * RATE)] = 0.9
        peaks = onsets.flux_peaks(onsets.spectral_flux(audio, RATE))
        for c in clicks:
            self.assertLessEqual(abs(onsets.nearest(peaks, c)), 0.025)

    def test_silence_has_no_onsets(self):
        silence = np.zeros(RATE * 2, dtype=np.float32)
        self.assertEqual(len(onsets.flux_peaks(onsets.spectral_flux(silence, RATE))), 0)
        self.assertEqual(len(onsets.voicing_onsets(np.zeros(200, dtype=bool))), 0)

    def test_wrong_rate_raises(self):
        # Should fail: 44.1 kHz audio passed without resampling.
        with self.assertRaises(ValueError):
            onsets.spectral_flux(np.zeros(44100, dtype=np.float32), 44100)


class CrepeEvidenceTests(unittest.TestCase):
    def test_voicing_onsets_need_three_unvoiced_frames(self):
        voiced = np.zeros(40, dtype=bool)
        voiced[0:5] = True     # starts voiced: an onset at 0
        voiced[7:12] = True    # only 2 unvoiced frames before: no onset
        voiced[20:30] = True   # 8 unvoiced before: onset at 0.20 s
        np.testing.assert_allclose(onsets.voicing_onsets(voiced), [0.0, 0.20])

    def test_pitch_steps(self):
        midi = np.array([60.0] * 20 + [62.0] * 20 + [62.4] * 20)
        voiced = np.ones(len(midi), dtype=bool)
        np.testing.assert_allclose(onsets.pitch_change_onsets(midi, voiced), [0.20])

    def test_nearest_is_signed(self):
        self.assertAlmostEqual(onsets.nearest(np.array([1.0, 2.0]), 1.05), 0.05)
        self.assertAlmostEqual(onsets.nearest(np.array([1.0, 2.0]), 1.97), -0.03)
        self.assertEqual(onsets.nearest(np.array([]), 1.0), float("inf"))


if __name__ == "__main__":
    unittest.main()
