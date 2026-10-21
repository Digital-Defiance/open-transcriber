"""Hand-made cases with known answers for the strict scorer and the mir_eval parity property test.

Ported from stage 0a ``tests_judge/test_score.py`` (source ``aa88f5a``); the SheetSage ``.lab``/events
loader cases are dropped with their loaders. The matching core, metrics, masks and don't-care logic are
unchanged, so the ``ParityTests`` 300 randomized cases still prove mir_eval parity.
"""

import itertools
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from mir_eval import transcription

from open_transcriber.bakeoff import notes_io, score

M = score.METRICS


def notes(rows, mask=None):
    rows = list(rows)
    if not rows:
        return notes_io.empty()
    onset, offset, midi = zip(*rows)
    return notes_io.Notes(onset, offset, midi, mask)


def f1(ref, est, metric, sung=None):
    return score.score_metric(ref, est, M[metric], sung)["f1"]


TUNE = [(1.0, 1.5, 60), (1.5, 2.0, 62), (2.0, 3.0, 64), (3.0, 3.25, 65)]


class MetricTests(unittest.TestCase):
    def test_identical_notes_score_one_everywhere(self):
        ref, est = notes(TUNE), notes(TUNE)
        for name in M:
            self.assertEqual(f1(ref, est, name), 1.0, name)

    def test_onset_boundary_after_rounding(self):
        ref = notes([(1.0, 2.0, 60)])
        self.assertEqual(f1(ref, notes([(1.05, 2.0, 60)]), "pc_on50"), 1.0)     # 50.00 ms: a hit
        self.assertEqual(f1(ref, notes([(1.0501, 2.0, 60)]), "pc_on50"), 0.0)   # 50.1 ms: a miss
        self.assertEqual(f1(ref, notes([(0.95, 2.0, 60)]), "pc_on50"), 1.0)     # early by 50 ms: a hit

    def test_octave_errors(self):
        ref, up = notes(TUNE), notes([(a, b, p + 12) for a, b, p in TUNE])
        self.assertEqual(f1(ref, up, "pc_on50"), 1.0)
        self.assertEqual(f1(ref, up, "oct_on50"), 0.0)
        shifted = score.score_metric(ref, up, M["oct_on50_shift"])
        self.assertEqual((shifted["f1"], shifted["shift"]), (1.0, -1))

    def test_octave_exact_uses_the_sung_pitch(self):
        ref = notes([(1.0, 2.0, 72)])
        sung = np.array([60.0])
        self.assertEqual(f1(ref, notes([(1.0, 2.0, 60)]), "oct_on50", sung), 1.0)
        self.assertEqual(f1(ref, notes([(1.0, 2.0, 72)]), "oct_on50", sung), 0.0)

    def test_a_semitone_off_misses_everywhere(self):
        ref, est = notes(TUNE), notes([(a, b, p + 1) for a, b, p in TUNE])
        for name in M:
            self.assertEqual(f1(ref, est, name), 0.0, name)

    def test_offsets(self):
        ref = notes([(1.0, 2.0, 60)])
        self.assertEqual(f1(ref, notes([(1.0, 9.0, 60)]), "pc_on50"), 1.0)      # no offset matching
        self.assertEqual(f1(ref, notes([(1.0, 2.19, 60)]), "pc_onoff50"), 1.0)  # within 0.2 * 1.0 s
        self.assertEqual(f1(ref, notes([(1.0, 2.21, 60)]), "pc_onoff50"), 0.0)
        short = notes([(1.0, 1.1, 60)])                                         # the 50 ms floor
        self.assertEqual(f1(short, notes([(1.0, 1.149, 60)]), "pc_onoff50"), 1.0)
        self.assertEqual(f1(short, notes([(1.0, 1.151, 60)]), "pc_onoff50"), 0.0)

    def test_one_to_one(self):
        result = score.score_metric(notes([(1.0, 2.0, 60)]), notes([(1.0, 2.0, 60), (1.01, 2.0, 60)]), M["pc_on50"])
        self.assertEqual((result["p"], result["r"]), (0.5, 1.0))
        self.assertAlmostEqual(result["f1"], 2 / 3)

    def test_maximum_matching_beats_greedy(self):
        ref = notes([(1.00, 1.5, 60), (1.08, 1.5, 60)])
        est = notes([(1.03, 1.5, 60), (0.955, 1.5, 60)])
        self.assertEqual(score.score_metric(ref, est, M["pc_on50"])["matched"], 2)

    def test_empty_reference_or_estimate_scores_zero(self):
        self.assertEqual(f1(notes([]), notes(TUNE), "pc_on50"), 0.0)
        self.assertEqual(f1(notes(TUNE), notes([]), "pc_on50"), 0.0)
        self.assertEqual(f1(notes([]), notes([]), "pc_on50"), 0.0)

    def test_macro_not_micro(self):
        rows = [{"pc_on50_f1": 1.0, "octave_excluded": 0}, {"pc_on50_f1": 0.0, "octave_excluded": 0}]
        self.assertEqual(score.mean_f1(rows, "pc_on50"), 0.5)
        rows = [{"oct_on50_f1": 1.0, "octave_excluded": 0}, {"oct_on50_f1": 0.0, "octave_excluded": 1}]
        self.assertEqual(score.mean_f1(rows, "oct_on50"), 1.0)  # octave-excluded rows left out of octave means

    def test_global_shift_should_fail_at_50_ms(self):
        # Should fail: a system 60 ms late everywhere scores 0 at 50 ms and 1 at 100 ms.
        ref, est = notes(TUNE), notes([(a + 0.06, b + 0.06, p) for a, b, p in TUNE])
        self.assertEqual(f1(ref, est, "pc_on50"), 0.0)
        self.assertEqual(f1(ref, est, "pc_on100"), 1.0)


class MaskTests(unittest.TestCase):
    def test_estimate_on_a_masked_note_is_neither_hit_nor_false_alarm(self):
        ref = notes([(1.0, 1.5, 60), (2.0, 2.5, 62)], mask=["", "pitch"])
        result = score.score_metric(ref, notes([(1.0, 1.5, 60), (2.0, 2.5, 62)]), M["pc_on50"])
        self.assertEqual((result["matched"], result["dontcare"], result["p"], result["r"]), (1, 1, 1.0, 1.0))

    def test_unmasked_references_have_priority(self):
        ref = notes([(1.00, 1.5, 60), (1.02, 1.5, 60)], mask=["pitch", ""])
        result = score.score_metric(ref, notes([(1.01, 1.5, 60)]), M["pc_on50"])
        self.assertEqual((result["matched"], result["dontcare"], result["f1"]), (1, 0, 1.0))

    def test_freestyle_waives_the_pitch_test(self):
        ref = notes([(1.0, 1.5, 60), (2.0, 2.5, 60)], mask=["", "freestyle"])
        result = score.score_metric(ref, notes([(1.0, 1.5, 60), (2.0, 2.5, 67)]), M["pc_on50"])
        self.assertEqual((result["dontcare"], result["p"]), (1, 1.0))
        qa_masked = notes([(1.0, 1.5, 60), (2.0, 2.5, 60)], mask=["", "onset"])
        result = score.score_metric(qa_masked, notes([(1.0, 1.5, 60), (2.0, 2.5, 67)]), M["pc_on50"])
        self.assertEqual((result["dontcare"], result["p"]), (0, 0.5))  # QA masks keep the pitch test

    def test_no_masks_equals_the_unmasked_path(self):
        ref, est = notes(TUNE), notes(TUNE[:3])
        result = score.score_metric(ref, est, M["pc_on50"])
        p, r, f, _ = transcription.precision_recall_f1_overlap(
            score._intervals(ref), score.to_hz(ref.midi, "pc"), score._intervals(est), score.to_hz(est.midi, "pc"),
            offset_ratio=None)
        self.assertEqual((result["p"], result["r"], result["f1"]), (p, r, f))


def brute_force_matching(ref, est, metric):
    """Exhaustive maximum matching size on a tiny case, from the same compatibility rule."""
    graph = score._hit_graph(score._intervals(ref), score.to_hz(ref.midi, metric.pitch), score._intervals(est),
                             score.to_hz(est.midi, metric.pitch), metric)
    edges = {(r, e) for e, refs in graph.items() for r in refs}
    best = 0
    for k in range(min(len(ref), len(est)), 0, -1):
        for refs in itertools.combinations(range(len(ref)), k):
            for ests in itertools.permutations(range(len(est)), k):
                if all((r, e) in edges for r, e in zip(refs, ests)):
                    return k
    return best


class ParityTests(unittest.TestCase):
    def test_brute_force_and_mir_eval_parity_on_random_cases(self):
        rng = np.random.default_rng(20261019)
        for case in range(300):
            n_ref, n_est = rng.integers(0, 8), rng.integers(0, 8)
            ref = notes([(t, t + rng.uniform(0.05, 0.6), rng.integers(55, 70))
                         for t in np.sort(rng.uniform(0, 1.5, n_ref))])
            est = notes([(t, t + rng.uniform(0.05, 0.6), rng.integers(55, 70))
                         for t in np.sort(rng.uniform(0, 1.5, n_est))])
            for name in ("pc_on50", "oct_on50", "pc_onoff50", "pc_on100"):
                metric = M[name]
                result = score.score_metric(ref, est, metric)
                if len(ref) and len(est):
                    self.assertEqual(result["matched"], brute_force_matching(ref, est, metric), (case, name))
                    p, r, f, _ = transcription.precision_recall_f1_overlap(
                        score._intervals(ref), score.to_hz(ref.midi, metric.pitch), score._intervals(est),
                        score.to_hz(est.midi, metric.pitch), onset_tolerance=metric.onset,
                        offset_ratio=metric.offset_ratio, offset_min_tolerance=0.05)
                    self.assertAlmostEqual(result["f1"], f, places=12, msg=(case, name))
                else:
                    self.assertEqual(result["f1"], 0.0)


class ScorePairTests(unittest.TestCase):
    def test_score_pair_builds_a_flat_row(self):
        ref, est = notes(TUNE), notes(TUNE)
        row = score.score_pair(ref, est, score.RWC_METRICS, item="x")
        self.assertEqual(row["item"], "x")
        self.assertEqual(row["pc_on50_f1"], 1.0)
        self.assertEqual(row["oct_on50_f1"], 1.0)
        self.assertIn("oct_on50_shift_shift", row)  # shift metric records its shift
        self.assertEqual(row["n_ref"], 4)

    def test_run_folder_strips_diagnostic_suffix(self):
        self.assertEqual(score.run_folder("probe-musicfm@heard"), "probe-musicfm")


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def write(self, name, text):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_reference_tsv(self):
        path = self.write("ref.tsv", "note_id\tonset\toffset\tmidi\tmidi_sung\tmask\tsource\n"
                                     "0\t1.000000\t1.500000\t72\t60\t\ttrack:4\n"
                                     "1\t2.000000\t2.500000\t74\t62\tfreestyle\tchart:F\n")
        ref = notes_io.load_reference(path)
        self.assertEqual(list(ref.midi), [72, 74])
        self.assertEqual(list(notes_io.load_reference(path, octave_exact=True).midi), [60, 62])
        self.assertEqual(list(ref.mask), ["", "freestyle"])
        masked = notes_io.load_reference(path, masks={0: "pitch"})
        self.assertEqual(list(masked.mask), ["pitch", "freestyle"])

    def test_copyist_events(self):
        payload = {"format": "copyist-events/1", "engine": "open", "melody_source": "pyin", "octave_above": 0,
                   "notes": [{"onset": 1.0, "offset": 1.4, "midi": 62, "heard_onset": 0.9, "heard_offset": 1.3},
                             {"onset": 1.4, "offset": 1.4, "midi": 64, "heard_onset": 1.3, "heard_offset": 1.5}]}
        path = self.write("events.json", json.dumps(payload))
        est = notes_io.load_copyist_events(path)
        self.assertEqual(list(est.midi), [62, 64])
        self.assertEqual(est.repaired, 1)  # a zero-length estimate keeps 1 ms
        self.assertAlmostEqual(est.offset[1], 1.401)
        heard = notes_io.load_copyist_events(path, heard=True)
        self.assertEqual(list(heard.onset), [0.9, 1.3])
        with self.assertRaises(ValueError):
            notes_io.load_copyist_events(self.write("bad.json", json.dumps({"format": "other", "notes": []})))

    def test_basic_pitch_tsv(self):
        est = notes_io.load_basic_pitch(self.write("notes.tsv", "onset\toffset\tmidi\tamplitude\n1.0\t1.3\t60\t0.5\n"))
        self.assertEqual((len(est), est.midi[0]), (1, 60))

    def test_invalid_inputs_raise(self):
        # Should fail: negative times, offset <= onset in a reference, NaN pitch, duplicate ids.
        header = "note_id\tonset\toffset\tmidi\tmidi_sung\tmask\tsource\n"
        for body in ("0\t-1.0\t1.0\t60\t60\t\tx\n", "0\t1.0\t1.0\t60\t60\t\tx\n", "0\t1.0\t2.0\tnan\t60\t\tx\n",
                     "0\t1.0\t2.0\t60\t60\t\tx\n0\t2.0\t3.0\t60\t60\t\tx\n"):
            with self.assertRaises(ValueError, msg=body):
                notes_io.load_reference(self.write("bad.tsv", header + body))
        with self.assertRaises(KeyError):
            score.METRICS["pc_on75"]


if __name__ == "__main__":
    unittest.main()
