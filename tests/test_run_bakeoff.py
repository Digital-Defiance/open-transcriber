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

    def test_licence_gate_reads_package_file_in_a_worktree(self):
        # No argument: the gate must read the committed package licences.md, not .agents/ (worktree-safe).
        self.assertTrue(rb.PACKAGE_LICENCES_MD.exists())
        self.assertEqual(rb.default_licences_md(), rb.PACKAGE_LICENCES_MD)
        ok, why = rb.licences_confirmed()
        self.assertTrue(ok, why)

    def test_licence_gate_fails_closed_on_nc_in_package_shaped_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "licences.md"
            p.write_text("musicfm dasheng heartcodec whisper basic pitch; one is CC BY-NC/ND\n")
            ok, _ = rb.licences_confirmed(p)
            self.assertFalse(ok)

    def test_dry_run_reports_plan_without_running(self):
        result = rb.run_batch(dry_run=True)
        self.assertIn("order", result)
        self.assertIn("estimate", result)

    def test_real_prereg_tag_is_present(self):
        # The ear-prereg tag was created in Phase C, so this gate passes in the worktree.
        ok, why = rb.prereg_tag_ok()
        self.assertTrue(ok, why)


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class EndToEndTests(unittest.TestCase):
    """The full extract->assemble->train->evaluate->decide loop with fakes: no real model, network or GPU."""

    def _run(self, tmp, metric_fn):
        order = []

        def calibrate(enc):
            order.append(("calibrate", enc))
            return {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True}

        def extractor_factory(name):
            order.append(("extract", name))
            return object()

        def trainer(encoder, seed, extractor, bp, train_clips, probe_data):
            order.append(("train", encoder, seed))
            return ("model", encoder, seed)

        def evaluator(encoder, seed, model, extractor, bp, dev_clips, probe_data):
            order.append(("evaluate", encoder, seed))
            return {"encoder": encoder, "seed": seed, "means": metric_fn(encoder)}

        def decider(trained, surviving):
            order.append(("decide",))
            from open_transcriber.bakeoff import decide as D

            def result(enc):
                by = trained[enc]
                real = {s: by[s]["means"]["oct_on50_f1"] * 100 for s in by}
                synth = {s: by[s]["means"]["synthetic_oct_on50_f1"] * 100 for s in by}
                mt = {s: by[s]["means"]["mt"] * 100 for s in by}
                return D.EncoderResult(enc, real, synth, mt)

            d = D.decide([result(e) for e in surviving], result("control"))
            return {"go": d.go, "winner": d.winner, "qualifying": d.qualifying}

        # This exercises the execution body, not the gates (those are tested separately), so force the
        # gates open regardless of where the submodule's ear-prereg tag lives.
        with mock.patch.object(rb, "gates_ok", return_value=(True, [])):
            return rb.run_batch(
                dry_run=False, out_root=Path(tmp),
                extractor_factory=extractor_factory, bp_factory=lambda: object(),
                trainer=trainer, evaluator=evaluator, decider=decider, calibrate=calibrate,
                train_clips=[{"wav": "a.wav", "labels": {}}],
                dev_clips=[{"id": "sampler-1", "wav": "a.wav", "labels": {}}],
                gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
            ), order

    def test_go_path_runs_all_stages_drops_heartcodec_and_persists_verdict(self):
        def metric(encoder):
            base = {"beat_f1": 0.5, "downbeat_f1": 0.5, "chord": 0.5, "key": 0.5, "mt": 0.5}
            if encoder == "control":
                return {"oct_on50_f1": 0.50, "synthetic_oct_on50_f1": 0.50, **base}
            return {"oct_on50_f1": 0.55, "synthetic_oct_on50_f1": 0.57, **base}  # +5 real, +7 synth -> GO

        with tempfile.TemporaryDirectory() as tmp:
            result, order = self._run(tmp, metric)
            self.assertEqual(result["stages"], ["calibrate", "train+evaluate", "decide"])
            # HeartCodec is dropped and never trained.
            self.assertTrue(any(d["encoder"] == "heartcodec" and d["dropped"] for d in result["drops"]))
            self.assertFalse(any(step[0] == "train" and step[1] == "heartcodec" for step in order))
            # decide() ran and a GO verdict is persisted.
            self.assertTrue(result["verdict"]["go"])
            self.assertTrue((Path(tmp) / "verdict.json").exists())
            self.assertTrue((Path(tmp) / "metrics.json").exists())

    def test_no_go_path_persists_no_go_verdict(self):
        def metric(encoder):
            base = {"beat_f1": 0.5, "downbeat_f1": 0.5, "chord": 0.5, "key": 0.5, "mt": 0.5}
            # Only +1 real -> below the +3 threshold -> NO-GO.
            if encoder == "control":
                return {"oct_on50_f1": 0.50, "synthetic_oct_on50_f1": 0.50, **base}
            return {"oct_on50_f1": 0.51, "synthetic_oct_on50_f1": 0.57, **base}

        with tempfile.TemporaryDirectory() as tmp:
            result, _ = self._run(tmp, metric)
            self.assertFalse(result["verdict"]["go"])
            self.assertIsNone(result["verdict"]["winner"])

    def test_rerun_resumes_and_skips_finished_stages(self):
        calls = {"train": 0}

        def metric(encoder):
            base = {"beat_f1": 0.5, "downbeat_f1": 0.5, "chord": 0.5, "key": 0.5, "mt": 0.5}
            if encoder == "control":
                return {"oct_on50_f1": 0.50, "synthetic_oct_on50_f1": 0.50, **base}
            return {"oct_on50_f1": 0.55, "synthetic_oct_on50_f1": 0.57, **base}

        with tempfile.TemporaryDirectory() as tmp:
            self._run(tmp, metric)
            # A second run must read every checkpoint and not recompute (resumability).
            def trainer(*a, **k):
                calls["train"] += 1
                return ("model",)
            with mock.patch.object(rb, "gates_ok", return_value=(True, [])):
                rb.run_batch(
                    dry_run=False, out_root=Path(tmp),
                    extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                    trainer=trainer,
                    evaluator=lambda *a, **k: {"means": {}},
                    decider=lambda trained, surviving: {"go": False, "winner": None},
                    calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                    train_clips=[], dev_clips=[],
                    gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
                )
            self.assertEqual(calls["train"], 0)  # all probe checkpoints were reused


if __name__ == "__main__":
    unittest.main()
