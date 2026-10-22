"""clips.py: resolve frozen-split ids into clip dicts, refuse sealed ids, keep the adapter decision open.

CPU only; no music21, no audio, no model. The material-script imports stay lazy so this runs in the
torch-only probe-eval env.
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from open_transcriber.bakeoff import clips, dev_splits


def _write_clip(clips_dir: Path, system_dir: Path, clip_id: str, wav_name: str, status: str = "ok") -> None:
    """Make a fake generated clip: a wav, a labels.json and a jobs.Job done.json with the given status."""
    clip_dir = clips_dir / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    (clip_dir / wav_name).write_bytes(b"RIFF")                       # content irrelevant: no audio is read
    (clip_dir / "labels.json").write_text(json.dumps({"duration": 1.0, "notes": []}), encoding="utf-8")
    done_dir = system_dir / clip_id
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "done.json").write_text(json.dumps({"status": status}), encoding="utf-8")


class SamplerResolverTests(unittest.TestCase):
    def test_resolves_every_accepted_variant_and_skips_failed(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            seed = dev_splits.SAMPLER_BASE_SEED
            _write_clip(out / "clips", out / "ear-sampler", f"{seed}-v0", "audio.wav")
            _write_clip(out / "clips", out / "ear-sampler", f"{seed}-v1", "audio.wav")
            _write_clip(out / "clips", out / "ear-sampler", f"{seed}-v2", "audio.wav", status="failed")
            found = clips.sampler_clips_for(seed, out=out)
            self.assertEqual(len(found), 2)                          # the failed variant is skipped
            self.assertTrue(all("wav" in c and "labels" in c and "id" in c for c in found))
            self.assertTrue(all(Path(c["wav"]).name == "audio.wav" for c in found))

    def test_missing_dir_yields_nothing(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(clips.sampler_clips_for(dev_splits.SAMPLER_BASE_SEED, out=Path(tmp)), [])


class SoulxResolverTests(unittest.TestCase):
    def test_resolves_accepted_soulx_clip(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            seed = dev_splits.SOULX_BASE_SEED
            _write_clip(out / "clips", out / "ear-soulx", f"{seed}-vocalset-male1", "vocal.wav")
            found = clips.soulx_clips_for(seed, out=out)
            self.assertEqual(len(found), 1)
            self.assertEqual(Path(found[0]["wav"]).name, "vocal.wav")


class AdapterIsStubbedTests(unittest.TestCase):
    def test_reference_adapter_raises_until_the_design_decision_is_made(self):
        with self.assertRaises(NotImplementedError):
            clips.reference_to_eval_labels(Path("ref.tsv"), {"id": "rwc-g001"})


class SealedGuardTests(unittest.TestCase):
    def test_assemble_refuses_a_sealed_test_id(self):
        sealed = sorted(dev_splits.rwc_pop_test())[0]
        with self.assertRaises(ValueError):
            clips._guard_no_sealed_ids([sealed])

    def test_assemble_refuses_a_cotb_id(self):
        with self.assertRaises(ValueError):
            clips._guard_no_sealed_ids(["cotb-001"])

    def test_guard_passes_dev_ids(self):
        clips._guard_no_sealed_ids(["rwc-g001", "sampler-90000000"])  # no raise


if __name__ == "__main__":
    unittest.main()
