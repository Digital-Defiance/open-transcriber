"""Tests for the numpy CREPE pipeline: Viterbi, framing, bins and the ONNX model."""

import itertools
import unittest

import numpy as np

from open_transcriber.bakeoff import crepe


def librosa_viterbi_reference(probs, transition):
    """librosa.sequence._viterbi, transcribed literally (loops), for parity on ties."""
    epsilon = np.finfo(probs.dtype).tiny
    n_steps, n_states = probs.shape
    log_trans = np.log(transition + epsilon)
    log_prob = np.log(probs + epsilon)
    log_p_init = np.log(np.full(n_states, 1.0 / n_states) + epsilon)
    value = np.zeros((n_steps, n_states))
    ptr = np.zeros((n_steps, n_states), dtype=np.uint16)
    value[0] = log_prob[0] + log_p_init
    for t in range(1, n_steps):
        trans_out = value[t - 1] + log_trans.T
        for j in range(n_states):
            ptr[t, j] = np.argmax(trans_out[j])
            value[t, j] = log_prob[t, j] + trans_out[j, ptr[t][j]]
    state = np.zeros(n_steps, dtype=np.int64)
    state[-1] = np.argmax(value[-1])
    for t in range(n_steps - 2, -1, -1):
        state[t] = ptr[t + 1, state[t + 1]]
    return state


def path_score(path, probs, transition):
    epsilon = np.finfo(probs.dtype).tiny
    log_trans = np.log(transition + epsilon)
    log_prob = np.log(probs + epsilon)
    score = np.log(1.0 / probs.shape[1] + epsilon) + log_prob[0, path[0]]
    for t in range(1, len(path)):
        score += log_trans[path[t - 1], path[t]] + log_prob[t, path[t]]
    return score


class ViterbiTests(unittest.TestCase):
    def test_matches_brute_force_and_librosa_on_random_cases(self):
        rng = np.random.default_rng(7)
        for case in range(50):
            n_states, n_steps = 4, 6
            if case % 5 == 0:
                # ties: equal observation rows and a symmetric transition
                probs = np.full((n_steps, n_states), 0.25, dtype=np.float32)
            else:
                probs = rng.dirichlet(np.ones(n_states), size=n_steps).astype(np.float32)
            transition = rng.dirichlet(np.ones(n_states), size=n_states)
            if case % 10 == 0:
                transition = np.full((n_states, n_states), 0.25)
            ours = crepe.viterbi(probs, transition)
            np.testing.assert_array_equal(ours, librosa_viterbi_reference(probs, transition))
            best = max(path_score(p, probs, transition) for p in itertools.product(range(n_states), repeat=n_steps))
            self.assertAlmostEqual(path_score(ours, probs, transition), best, places=9)

    def test_ties_take_the_first_state(self):
        probs = np.full((3, 4), 0.25, dtype=np.float32)
        np.testing.assert_array_equal(crepe.viterbi(probs, np.full((4, 4), 0.25)), [0, 0, 0])


class FramingTests(unittest.TestCase):
    def test_frame_count_and_padding(self):
        audio = np.arange(1600, dtype=np.float32)
        frames = crepe.raw_frames(audio)
        self.assertEqual(frames.shape, (11, 1024))
        self.assertEqual(frames[0, 512], 0.0)  # frame 0 is centred on sample 0
        self.assertEqual(frames[1, 512], 160.0)
        self.assertEqual(frames[0, 0], 0.0)  # zero padding

    def test_normalise_uses_the_unbiased_std(self):
        frames = np.tile(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), (1, 256))
        out = crepe.normalise(frames)
        expected = (frames - frames.mean()) / frames.std(ddof=1)
        np.testing.assert_allclose(out, expected, rtol=1e-6)
        silent = crepe.normalise(np.zeros((1, 1024), dtype=np.float32))
        self.assertTrue(np.all(silent == 0))

    def test_bin_limits_match_torchcrepe(self):
        self.assertEqual(crepe.frequency_to_bin(50.0), 39)
        self.assertEqual(crepe.frequency_to_bin(1100.0, ceil=True), 308)
        self.assertAlmostEqual(float(crepe.bins_to_midi(np.array([0]))[0]),
                               69 + 12 * np.log2(10 * 2 ** (1997.3794084376191 / 1200) / 440))

    def test_resampling_to_16k(self):
        tone = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100).astype(np.float32)
        out = crepe.to_16k_mono(np.stack([tone, tone], axis=1), 44100)
        self.assertEqual(len(out), 16000)
        self.assertEqual(out.dtype, np.float32)


@unittest.skipUnless(crepe.model_path().exists(), "crepe-full.onnx not exported on this machine")
class ModelTests(unittest.TestCase):
    def test_model_hash_matches(self):
        from open_transcriber.bakeoff.fetch import sha256_file

        self.assertEqual(sha256_file(crepe.model_path()), crepe.MODEL_SHA256)

    def test_sweep_is_tracked_within_25_cents(self):
        seconds, rate = 2.0, 16000
        t = np.arange(int(seconds * rate)) / rate
        f = 200.0 * 2.0 ** (t / seconds)
        phase = 2 * np.pi * np.cumsum(f) / rate
        audio = (0.3 * (np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)) / 1.75).astype(np.float32)
        result = crepe.track(audio, crepe.Model())
        times = np.arange(len(result["midi"])) * crepe.HOP / rate
        truth = 69 + 12 * np.log2(200.0 * 2.0 ** (np.clip(times, 0, seconds) / seconds) / 440.0)
        voiced = result["periodicity"] >= 0.5
        voiced[:5] = voiced[-5:] = False
        self.assertGreater(voiced.mean(), 0.8)
        within = np.abs(result["midi"][voiced] - truth[voiced]) * 100 <= 25
        self.assertGreaterEqual(within.mean(), 0.95)

    def test_a_bad_model_file_is_refused(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "crepe-full.onnx"
            fake.write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                crepe.Model(fake)


if __name__ == "__main__":
    unittest.main()
