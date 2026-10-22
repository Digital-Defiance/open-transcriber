"""Decode + eval: metric keys present, a perfect decode scores high, sealed test ids are refused (CPU)."""

import unittest

import numpy as np

from open_transcriber.bakeoff import dev_splits
from open_transcriber.bakeoff.probe import decode, evaluate
from open_transcriber.bakeoff.probe.model import N_CHORD_QUALITY, N_CHORD_ROOT, N_KEY, N_PITCH, TARGET_RATE_HZ


def perfect_outputs_for(labels):
    """Build head logits that decode exactly to the label note/beat/chord/key (one clip)."""
    frames = int(round(labels["duration"] * TARGET_RATE_HZ)) + 2
    pitch = np.full((frames, N_PITCH), -10.0, dtype=np.float32)
    silence = np.full(frames, 10.0, dtype=np.float32)   # silent everywhere unless a note says otherwise
    for n in labels["notes"]:
        a = int(round(n["onset"] * TARGET_RATE_HZ))
        b = int(round(n["offset"] * TARGET_RATE_HZ))
        for i in range(a, min(b, frames)):
            pitch[i, :] = -10.0
            pitch[i, n["midi"]] = 10.0
            silence[i] = -10.0
    beat = np.full(frames, -10.0, dtype=np.float32)
    for t in labels["beats"]:
        beat[min(int(round(t * TARGET_RATE_HZ)), frames - 1)] = 10.0
    downbeat = np.full(frames, -10.0, dtype=np.float32)
    for t in labels["downbeats"]:
        downbeat[min(int(round(t * TARGET_RATE_HZ)), frames - 1)] = 10.0
    root = np.full((frames, N_CHORD_ROOT), -10.0, dtype=np.float32)
    qual = np.full((frames, N_CHORD_QUALITY), -10.0, dtype=np.float32)
    qmap = {"maj": 0, "min": 1, "no-chord": 2}
    for c in labels["chords"]:
        a = int(round(c["onset"] * TARGET_RATE_HZ))
        b = int(round(c["offset"] * TARGET_RATE_HZ))
        rpc = c["root_pc"] if c["root_pc"] >= 0 else 12
        for i in range(a, min(b, frames)):
            root[i, :] = -10.0
            root[i, rpc] = 10.0
            qual[i, qmap[c["quality"]]] = 10.0
    key = np.full(N_KEY, -10.0, dtype=np.float32)
    key[labels["key_class"]] = 10.0
    return {"pitch": pitch, "silence": silence, "onset": beat, "offset": beat,
            "beat": beat, "downbeat": downbeat, "chord_root": root, "chord_quality": qual, "key": key}


LABELS = {
    "duration": 2.0,
    "notes": [{"onset": 0.0, "offset": 0.5, "midi": 60}, {"onset": 0.5, "offset": 1.0, "midi": 62},
              {"onset": 1.0, "offset": 2.0, "midi": 64}],
    "beats": [0.0, 0.5, 1.0, 1.5, 2.0],
    "downbeats": [0.0, 2.0],
    "chords": [{"onset": 0.0, "offset": 2.0, "root_pc": 0, "quality": "maj"}],
    "key_class": 0,
}


class EvaluateTests(unittest.TestCase):
    def test_metric_keys_present(self):
        out = perfect_outputs_for(LABELS)
        row = evaluate.score_one_clip(out, LABELS, item="sampler-1")
        for key in ("oct_on50_f1", "oct_on100_f1", "pc_on50_f1", "beat_f1", "downbeat_f1", "chord", "key"):
            self.assertIn(key, row)

    def test_perfect_decode_scores_high(self):
        out = perfect_outputs_for(LABELS)
        row = evaluate.score_one_clip(out, LABELS, item="sampler-1")
        self.assertGreaterEqual(row["oct_on50_f1"], 0.6)   # exact pitches, aligned onsets
        self.assertEqual(row["key"], 1)
        self.assertGreaterEqual(row["chord"], 0.9)
        self.assertGreaterEqual(row["beat_f1"], 0.8)

    def test_decode_notes_merges_runs(self):
        out = perfect_outputs_for(LABELS)
        notes = decode.decode_notes(out["pitch"], out["silence"])
        self.assertEqual([n[2] for n in notes], [60, 62, 64])

    def test_evaluate_record_shape(self):
        out = perfect_outputs_for(LABELS)
        rec = evaluate.evaluate_probe("musicfm", 20261020, [(out, LABELS, "sampler-1")], partition="synthetic")
        self.assertEqual(rec["encoder"], "musicfm")
        self.assertEqual(rec["seed"], 20261020)
        self.assertIn("oct_on50_f1", rec["means"])

    def test_real_partition_refuses_sealed_test_id(self):
        out = perfect_outputs_for(LABELS)
        bad = sorted(dev_splits.rwc_pop_test())[0]
        with self.assertRaises(ValueError):
            evaluate.evaluate_probe("musicfm", 20261020, [(out, LABELS, bad)], partition="real")

    def test_real_partition_allows_dev_id(self):
        out = perfect_outputs_for(LABELS)
        dev_id = dev_splits.rwc_other_dev()[0]
        rec = evaluate.evaluate_probe("musicfm", 20261020, [(out, LABELS, dev_id)], partition="real")
        self.assertEqual(rec["n"], 1)


# A real-dev-shaped melody-only labels dict (Option 1): notes present (one freestyle-masked), secondaries None.
MELODY_ONLY_LABELS = {
    "format": "ear-eval-labels/melody-only-1",
    "duration": 2.5,
    "notes": [{"onset": 0.0, "offset": 0.5, "midi": 60, "mask": ""},
              {"onset": 0.5, "offset": 1.0, "midi": 62, "mask": ""},
              {"onset": 1.0, "offset": 2.0, "midi": 64, "mask": "freestyle"}],  # don't-care
    "beats": None,
    "downbeats": None,
    "chords": None,
    "key_class": None,
    "meter": None,
    "tempo": None,
}


class MelodyOnlyMaskingTests(unittest.TestCase):
    """Option 1: NOT-LABELLED secondary heads are EXCLUDED (not zeroed); freestyle notes are don't-cares."""

    def test_melody_only_clip_omits_secondary_head_keys(self):
        out = perfect_outputs_for(LABELS)   # decoded output shape is independent of the reference labels
        row = evaluate.score_one_clip(out, MELODY_ONLY_LABELS, item="rwc-g001")
        for key in ("oct_on50_f1", "oct_on100_f1", "pc_on50_f1"):
            self.assertIn(key, row)                 # melody heads always scored
        for key in ("beat_f1", "downbeat_f1", "chord", "key"):
            self.assertNotIn(key, row)              # excluded, not scored as all-wrong

    def test_freestyle_note_is_a_dontcare_not_a_miss(self):
        # A decode that matches only the two scored notes (and not the freestyle one) still scores perfectly:
        # the masked note is a don't-care, so recall is over the two unmasked references.
        scored_only = {"duration": MELODY_ONLY_LABELS["duration"],
                       "notes": MELODY_ONLY_LABELS["notes"][:2], "beats": [], "downbeats": [],
                       "chords": [], "key_class": 0}
        out = perfect_outputs_for(scored_only)
        row = evaluate.score_one_clip(out, MELODY_ONLY_LABELS, item="rwc-g001")
        self.assertGreaterEqual(row["oct_on50_f1"], 0.99)   # the freestyle note is not counted as a miss

    def test_full_clip_still_scores_all_seven_heads(self):
        out = perfect_outputs_for(LABELS)
        row = evaluate.score_one_clip(out, LABELS, item="sampler-1")
        for key in ("oct_on50_f1", "oct_on100_f1", "pc_on50_f1", "beat_f1", "downbeat_f1", "chord", "key"):
            self.assertIn(key, row)

    def test_evaluate_probe_excludes_melody_only_clip_from_secondary_means(self):
        full_out = perfect_outputs_for(LABELS)
        melody_out = perfect_outputs_for(LABELS)
        rwc_id = dev_splits.rwc_other_dev()[0]
        karaoke_like = dev_splits.rwc_other_dev()[1]
        rec = evaluate.evaluate_probe(
            "musicfm", 20261020,
            [(full_out, LABELS, rwc_id), (melody_out, MELODY_ONLY_LABELS, karaoke_like)],
            partition="real")
        self.assertEqual(rec["n"], 2)
        # Melody means average BOTH clips.
        self.assertGreater(rec["means"]["oct_on50_f1"], 0.0)
        # Beat mean reflects only the one fully-labelled clip (the melody-only clip is excluded, not a zero).
        # With one perfect beat clip and the other excluded, the mean equals that single clip's beat F1 > 0.
        full_row = evaluate.score_one_clip(full_out, LABELS, item=rwc_id)
        self.assertAlmostEqual(rec["means"]["beat_f1"], round(full_row["beat_f1"], 6))
        self.assertGreater(rec["means"]["beat_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
