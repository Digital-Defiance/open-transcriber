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


def _write_reference_tsv(path: Path, rows: list[tuple]) -> None:
    """Write a note-only reference TSV (``note_id onset offset midi midi_sung mask source``)."""
    header = "note_id\tonset\toffset\tmidi\tmidi_sung\tmask\tsource"
    lines = [header]
    for i, (onset, offset, midi, midi_sung, mask, source) in enumerate(rows):
        lines.append(f"{i}\t{onset:.6f}\t{offset:.6f}\t{midi}\t{midi_sung}\t{mask}\t{source}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class ReferenceToEvalLabelsTests(unittest.TestCase):
    """Option 1: note-only reference TSV -> melody-only eval labels; freestyle mask honored; secondaries N/A."""

    def test_melody_only_labels_with_mask_and_duration(self):
        with TemporaryDirectory() as tmp:
            ref = Path(tmp) / "rwc-g001.tsv"
            _write_reference_tsv(ref, [
                (0.0, 0.5, 60, 60, "", "track:1"),
                (0.5, 1.2, 62, 62, "", "track:1"),
                (1.2, 2.0, 64, 64, "freestyle", "chart"),   # masked -> don't-care
            ])
            labels = clips.reference_to_eval_labels(ref, {"id": "rwc-g001"})
            # melody notes only, onset/offset/midi/mask preserved
            self.assertEqual(len(labels["notes"]), 3)
            self.assertEqual([n["midi"] for n in labels["notes"]], [60, 62, 64])
            self.assertEqual(labels["notes"][0]["mask"], "")
            self.assertEqual(labels["notes"][2]["mask"], "freestyle")   # freestyle row keeps its mask
            self.assertTrue(all({"onset", "offset", "midi"} <= set(n) for n in labels["notes"]))
            # duration = max offset + tail
            self.assertAlmostEqual(labels["duration"], 2.0 + clips.TAIL)
            # secondaries are the NOT-LABELLED sentinel None (not [], not 0)
            for head in ("beats", "downbeats", "chords", "key_class", "meter", "tempo"):
                self.assertIsNone(labels[head], head)

    def test_empty_reference_duration_is_tail(self):
        with TemporaryDirectory() as tmp:
            ref = Path(tmp) / "empty.tsv"
            _write_reference_tsv(ref, [])
            labels = clips.reference_to_eval_labels(ref, {"id": "rwc-g002"})
            self.assertEqual(labels["notes"], [])
            self.assertAlmostEqual(labels["duration"], clips.TAIL)
            self.assertIsNone(labels["beats"])


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
