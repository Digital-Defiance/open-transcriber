"""Real-encoder shape/rate/layer probes on 1-2 real sampler clips (the allowed short GPU verification).

These load the real encoders and must pass on the owner's Mac, but are guarded so a weightless/cloud env
skips rather than fails: set ``COPYIST_EAR_SLOW=1`` and have the weights on disk. They assert exactly the
shapes/rates/layers the 21 October smoke test documented. Run on the torch env (.cache/ear/env); Basic Pitch
runs through a .venv subprocess.

Run on the Mac:
    COPYIST_EAR_SLOW=1 /Volumes/Code/copyist/.cache/ear/env/bin/python -m unittest tests.test_extractors_real
"""

import os
import unittest
from pathlib import Path

from open_transcriber.bakeoff import paths
from open_transcriber.bakeoff.probe import extractor as ex

SLOW = os.environ.get("COPYIST_EAR_SLOW") == "1"
SAMPLER = paths.copyist_root() / ".cache" / "ear" / "sampler" / "clips"


def _clips(n=1):
    """Up to ``n`` real sampler clip wavs, if the sampler material is on disk."""
    if not SAMPLER.exists():
        return []
    wavs = sorted(SAMPLER.glob("*/audio.wav"))
    return wavs[:n]


def _skip_unless_ready(weights_present: bool, reason: str):
    if not SLOW:
        raise unittest.SkipTest("set COPYIST_EAR_SLOW=1 to run the real-encoder probes")
    if not _clips():
        raise unittest.SkipTest("no real sampler clips on disk")
    if not weights_present:
        raise unittest.SkipTest(reason)


class HeartCodecDropTests(unittest.TestCase):
    """The drop-stub carries its reason and refuses to extract (no weights, always runs)."""

    def test_heartcodec_stub(self):
        stub = ex.build_extractor("heartcodec")
        self.assertTrue(stub.descriptor().dropped)
        self.assertIn("mps", stub.descriptor().note.lower())
        with self.assertRaises(RuntimeError):
            stub.features(Path("x.wav"))


class ImportBoundaryTests(unittest.TestCase):
    """bp_predict must not import TensorFlow/Basic Pitch at module import time (TF stays lazy, .venv-only)."""

    def test_bp_predict_does_not_import_tensorflow_at_module_load(self):
        import subprocess
        import sys
        code = ("import sys, json, importlib;"
                "importlib.import_module('open_transcriber.bakeoff.bp_predict');"
                "print(json.dumps({'tf': 'tensorflow' in sys.modules, 'bp': 'basic_pitch' in sys.modules}))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
        import json
        flags = json.loads(out.splitlines()[-1])
        self.assertFalse(flags["tf"], "bp_predict must not import tensorflow at module import")
        self.assertFalse(flags["bp"], "bp_predict must not import basic_pitch at module import")


class MusicFMTests(unittest.TestCase):
    def test_three_layers_1024_at_25hz_on_mps(self):
        from open_transcriber.bakeoff.probe.extractors.musicfm import musicfm_data_dir
        ckpt = musicfm_data_dir() / "pretrained_fma.pt"
        _skip_unless_ready(ckpt.exists(), "MusicFM FMA checkpoint not downloaded")
        extr = ex.build_extractor("musicfm")
        self.assertEqual(extr.descriptor().native_rate_hz, 25.0)
        self.assertEqual(extr.descriptor().feature_dim, 1024)
        feats = extr.features(_clips()[0])
        self.assertEqual(set(feats), set(ex.LAYER_NAMES))
        lengths = {name: feats[name].shape for name in ex.LAYER_NAMES}
        for name in ex.LAYER_NAMES:
            self.assertEqual(feats[name].shape[1], 1024)
        # all three layers time-aligned (equal frame count)
        self.assertEqual(len({s[0] for s in lengths.values()}), 1)


class DashengTests(unittest.TestCase):
    def test_final_768_at_25hz_and_layers_time_aligned(self):
        try:
            import dasheng  # noqa: F401
            have = True
        except Exception:
            have = False
        _skip_unless_ready(have, "dasheng package/weights not available")
        extr = ex.build_extractor("dasheng-base")
        self.assertEqual(extr.descriptor().native_rate_hz, 25.0)
        self.assertEqual(extr.descriptor().feature_dim, 768)
        self.assertIn("final-only", extr.descriptor().note)
        feats = extr.features(_clips()[0])
        for name in ex.LAYER_NAMES:
            self.assertEqual(feats[name].shape[1], 768)
        # final-only: every declared layer is the same time-aligned final representation
        self.assertEqual(len({feats[name].shape[0] for name in ex.LAYER_NAMES}), 1)


class WhisperTests(unittest.TestCase):
    def test_three_layers_512_at_50hz_on_cpu(self):
        try:
            import whisper  # noqa: F401
            have = True
        except Exception:
            have = False
        _skip_unless_ready(have, "whisper package not available")
        extr = ex.build_extractor("whisper-base-en")
        self.assertEqual(extr.descriptor().native_rate_hz, 50.0)
        self.assertEqual(extr.descriptor().feature_dim, 512)
        self.assertEqual(extr.descriptor().device, "cpu")
        feats = extr.features(_clips()[0])
        for name in ex.LAYER_NAMES:
            self.assertEqual(feats[name].shape, (1500, 512))  # fixed 30 s window -> 1500 frames @ 50 Hz


class BasicPitchSideTests(unittest.TestCase):
    def test_posterior_shapes_from_venv_subprocess(self):
        from open_transcriber.bakeoff.probe.extractors.basic_pitch_side import (
            BasicPitchSideInput, venv_python_default)
        venv = venv_python_default()
        _skip_unless_ready(venv.exists(), f"Copyist .venv (Basic Pitch/TensorFlow) not at {venv}")
        import tempfile
        bp = BasicPitchSideInput(cache_dir=Path(tempfile.mkdtemp()))
        post = bp.posteriors(_clips()[0])
        self.assertEqual(post["note"].shape[1], 88)
        self.assertEqual(post["onset"].shape[1], 88)
        self.assertEqual(post["contour"].shape[1], 264)


if __name__ == "__main__":
    unittest.main()
