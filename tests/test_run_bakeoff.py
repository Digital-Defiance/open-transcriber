"""The GPU batch planner: fixed order, calibration drop rule, gates, and refusal to start before freeze/licence.

CPU only; no GPU, no model. Exercises the plan, the drop rule and the gates with fakes.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from open_transcriber.bakeoff import run_bakeoff as rb
from open_transcriber.bakeoff.probe.train import SEEDS


class PlanTests(unittest.TestCase):
    def test_fixed_order_extract_then_soulx_then_probes(self):
        order = rb.build_plan().ordered()
        # all extraction jobs come first, then soulx-slice, then the probe runs
        soulx_idx = order.index("soulx-slice")
        self.assertTrue(all(o.startswith("extract-") for o in order[:soulx_idx]))
        self.assertTrue(all(o.startswith("probe-") for o in order[soulx_idx + 1:]))

    def test_twelve_music_probes_plus_control(self):
        probes = rb.build_plan().probe_jobs
        # 4 encoders x 3 seeds = 12, plus control x 3 = 15
        self.assertEqual(len(probes), (len(rb.ENCODERS) + 1) * len(SEEDS))
        self.assertEqual(len(rb.ENCODERS) * len(SEEDS), 12)

    def test_estimate_total_and_cap(self):
        est = rb.estimate(rb.build_plan())
        self.assertAlmostEqual(est["total_hours"],
                               rb.EST_EXTRACT_HOURS + rb.EST_SOULX_HOURS + rb.EST_PROBE_HOURS)
        self.assertEqual(est["batch_cap_hours"], rb.BATCH_CAP_HOURS)


class DropRuleTests(unittest.TestCase):
    def test_drop_when_too_slow(self):
        drop, why = rb.should_drop_extractor(3.5, 10, mps_ok=True, pilot_lead=False)
        self.assertTrue(drop)
        self.assertIn("h", why)

    def test_drop_when_too_big(self):
        drop, _ = rb.should_drop_extractor(1.0, 60, mps_ok=True, pilot_lead=False)
        self.assertTrue(drop)

    def test_drop_when_no_mps_and_no_pilot_lead(self):
        self.assertTrue(rb.should_drop_extractor(1.0, 10, mps_ok=False, pilot_lead=False)[0])
        self.assertFalse(rb.should_drop_extractor(1.0, 10, mps_ok=False, pilot_lead=True)[0])

    def test_keep_within_limits(self):
        self.assertFalse(rb.should_drop_extractor(2.0, 20, mps_ok=True, pilot_lead=False)[0])


class GateTests(unittest.TestCase):
    def test_refuses_to_run_before_gates(self):
        # With the tag check forced false, a non-dry run must raise.
        with mock.patch.object(rb, "prereg_tag_ok", return_value=(False, "no tag")), \
             mock.patch.object(rb, "licences_confirmed", return_value=(True, "ok")):
            with self.assertRaises(PermissionError):
                rb.run_batch(dry_run=False)

    def test_licence_gate_fails_closed_when_file_missing(self):
        ok, why = rb.licences_confirmed(Path(tempfile.mkdtemp()) / "nope.md")
        self.assertFalse(ok)

    def test_licence_gate_fails_when_encoder_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "licences.md"
            p.write_text("musicfm dasheng whisper basic pitch\n")  # heartcodec missing
            ok, why = rb.licences_confirmed(p)
            self.assertFalse(ok)
            self.assertIn("heartcodec", why)

    def test_licence_gate_passes_with_all_encoders(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "licences.md"
            p.write_text("musicfm dasheng heartcodec whisper basic pitch all free/allowed\n")
            self.assertTrue(rb.licences_confirmed(p)[0])

    def test_dry_run_reports_plan_without_running(self):
        result = rb.run_batch(dry_run=True)
        self.assertIn("order", result)
        self.assertIn("estimate", result)

    def test_real_prereg_tag_is_present(self):
        # The ear-prereg tag was created in Phase C, so this gate passes in the worktree.
        ok, why = rb.prereg_tag_ok()
        self.assertTrue(ok, why)


if __name__ == "__main__":
    unittest.main()
