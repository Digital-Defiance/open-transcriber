"""The GPU batch planner: fixed order, calibration drop rule, gates, and refusal to start before freeze/licence.

CPU only; no GPU, no model. Exercises the plan, the drop rule and the gates with fakes.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from open_transcriber.bakeoff import dev_splits
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

    def test_real_run_generates_material_and_assembles_nonempty_train_and_dev(self):
        # The hole the first review missed: with no injected clips, --run must GENERATE material and ASSEMBLE
        # a non-empty train set AND a non-empty real-dev set before any probe trains.
        calls = {"generate": 0, "assemble": 0}
        seen_clips = {}

        def generate_material():
            calls["generate"] += 1

        def assemble_clips():
            calls["assemble"] += 1
            return ([{"wav": "t.wav", "labels": {}}],
                    [{"id": "rwc-g001", "wav": "d.wav", "labels": {}}])

        def trainer(encoder, seed, extractor, bp, train_clips, probe_data):
            seen_clips["train"] = train_clips
            return ("model",)

        def evaluator(encoder, seed, model, extractor, bp, dev_clips, probe_data):
            seen_clips["dev"] = dev_clips
            return {"encoder": encoder, "seed": seed,
                    "means": {"oct_on50_f1": 0.5, "synthetic_oct_on50_f1": 0.5, "beat_f1": 0.5,
                              "downbeat_f1": 0.5, "chord": 0.5, "key": 0.5, "mt": 0.5}}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rb, "gates_ok", return_value=(True, [])):
            result = rb.run_batch(
                dry_run=False, out_root=Path(tmp),
                extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                trainer=trainer, evaluator=evaluator,
                decider=lambda trained, surviving: {"go": False, "winner": None},
                calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                generate_material=generate_material, assemble_clips=assemble_clips,
                gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
            )
        self.assertEqual(calls["generate"], 1)
        self.assertEqual(calls["assemble"], 1)
        self.assertIn("generate-material", result["stages"])
        self.assertIn("assemble-clips", result["stages"])
        self.assertTrue(seen_clips["train"])   # non-empty train
        self.assertTrue(seen_clips["dev"])     # non-empty real-dev

    def test_real_run_refuses_empty_assembled_clip_sets(self):
        # If generation/assembly yields nothing, the batch must fail loudly, never produce an n=0 verdict.
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rb, "gates_ok", return_value=(True, [])):
            with self.assertRaises(RuntimeError):
                rb.run_batch(
                    dry_run=False, out_root=Path(tmp),
                    extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                    trainer=lambda *a, **k: ("model",), evaluator=lambda *a, **k: {"means": {}},
                    decider=lambda trained, surviving: {"go": False, "winner": None},
                    calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                    generate_material=lambda: None, assemble_clips=lambda: ([], []),
                    gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
                )

    def test_injected_clips_skip_generation_and_assembly(self):
        # Backwards-compatible: injecting clip lists (the other end-to-end tests) must NOT generate material.
        called = {"generate": 0}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rb, "gates_ok", return_value=(True, [])):
            result = rb.run_batch(
                dry_run=False, out_root=Path(tmp),
                extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                trainer=lambda *a, **k: ("model",),
                evaluator=lambda encoder, *a, **k: {"means": {}},
                decider=lambda trained, surviving: {"go": False, "winner": None},
                calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                generate_material=lambda: called.__setitem__("generate", called["generate"] + 1),
                train_clips=[{"wav": "a.wav", "labels": {}}],
                dev_clips=[{"id": "sampler-1", "wav": "a.wav", "labels": {}}],
                gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
            )
        self.assertEqual(called["generate"], 0)
        self.assertNotIn("generate-material", result["stages"])

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


class MelodyOnlyRealDevEndToEndTests(unittest.TestCase):
    """Fake tiny extractors but the REAL label plumbing: prove decide() gets non-empty real-dev melody
    metrics and that NOT-LABELLED (melody-only) heads are EXCLUDED from the means, not scored as all-wrong.

    This is the check that would have caught the empty-NO-GO and the empty-data blocker: the evaluator here
    is the real ``evaluate.evaluate_probe`` running the Option-1 masking path over melody-only real-dev
    labels, and the decider is the real ``_default_decider`` + ``decide``.
    """

    def _melody_only_labels(self, control: bool):
        """Real-dev-shaped melody-only labels: notes present (one freestyle), secondaries NOT-LABELLED."""
        # A music encoder decodes more notes right than the Basic-Pitch-only control, so it clears the +3.0
        # real-dev gate; both decode the two scored notes, but the control misses the second.
        notes = [{"onset": 0.0, "offset": 0.5, "midi": 60, "mask": ""},
                 {"onset": 0.5, "offset": 1.0, "midi": 62, "mask": ""},
                 {"onset": 1.0, "offset": 2.0, "midi": 64, "mask": "freestyle"}]
        return {"format": "ear-eval-labels/melody-only-1", "duration": 2.5, "notes": notes,
                "beats": None, "downbeats": None, "chords": None, "key_class": None,
                "meter": None, "tempo": None}

    def _synthetic_labels(self):
        return {"duration": 2.0,
                "notes": [{"onset": 0.0, "offset": 0.5, "midi": 60}, {"onset": 0.5, "offset": 1.0, "midi": 62},
                          {"onset": 1.0, "offset": 2.0, "midi": 64}],
                "beats": [0.0, 0.5, 1.0, 1.5, 2.0], "downbeats": [0.0, 2.0],
                "chords": [{"onset": 0.0, "offset": 2.0, "root_pc": 0, "quality": "maj"}], "key_class": 0}

    def _run(self):
        from open_transcriber.bakeoff.probe import evaluate
        from test_probe_evaluate import perfect_outputs_for

        rwc_ids = dev_splits.rwc_other_dev()[:2]
        melody_labels = self._melody_only_labels(control=False)
        dev_clips = [{"id": rwc_ids[0], "wav": "d0.wav", "labels": melody_labels},
                     {"id": rwc_ids[1], "wav": "d1.wav", "labels": melody_labels}]
        captured = {}

        def evaluator(encoder, seed, model, extractor, bp, clips_in, probe_data):
            # Fake tiny extractor path: decode each real-dev clip directly from its melody-only labels, then
            # run the REAL evaluate_probe so the Option-1 masking path (not a stub) produces the means.
            if encoder == "control":
                # The control decodes only the first scored note right (misses the second) -> lower melody F1.
                partial = {"duration": melody_labels["duration"], "notes": melody_labels["notes"][:1],
                           "beats": [], "downbeats": [], "chords": [], "key_class": 0}
                scored = [(perfect_outputs_for(partial), lab, cid) for (_o, lab, cid)
                          in [(None, c["labels"], c["id"]) for c in clips_in]]
            else:
                full = {"duration": melody_labels["duration"], "notes": melody_labels["notes"][:2],
                        "beats": [], "downbeats": [], "chords": [], "key_class": 0}
                scored = [(perfect_outputs_for(full), lab, cid) for (_o, lab, cid)
                          in [(None, c["labels"], c["id"]) for c in clips_in]]
            rec = evaluate.evaluate_probe(encoder, seed, scored, partition="real")
            captured.setdefault(encoder, rec)
            return rec

        def decider(trained, surviving):
            captured["trained"] = trained
            return rb._default_decider(trained, surviving)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rb, "gates_ok", return_value=(True, [])):
            result = rb.run_batch(
                dry_run=False, out_root=Path(tmp),
                extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                trainer=lambda *a, **k: ("model",), evaluator=evaluator, decider=decider,
                calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                train_clips=[{"wav": "t.wav", "labels": self._synthetic_labels()}],
                dev_clips=dev_clips,
                gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
            )
        return result, captured

    def test_decide_gets_nonempty_real_dev_melody_metrics_and_masked_heads_excluded(self):
        result, captured = self._run()
        # Every metric record scored two real-dev clips (non-empty, non-vacuous).
        for rec in result["metrics"]:
            self.assertEqual(rec["n"], 2)
            self.assertGreater(rec["means"]["oct_on50_f1"], 0.0)   # real melody metric is real, not 0
            # The NOT-LABELLED secondary heads are EXCLUDED: their means are 0.0 only because no clip labelled
            # them (mean over zero rows), never because a labelled head scored all-wrong.
            self.assertEqual(rec["means"]["beat_f1"], 0.0)
            self.assertEqual(rec["means"]["chord"], 0.0)
        # The per-clip rows for a melody-only clip omit the secondary keys entirely (excluded, not zeroed).
        from open_transcriber.bakeoff.probe import evaluate
        from test_probe_evaluate import perfect_outputs_for
        row = evaluate.score_one_clip(perfect_outputs_for(self._synthetic_labels()),
                                      self._melody_only_labels(control=False), item="rwc-test")
        for key in ("beat_f1", "downbeat_f1", "chord", "key"):
            self.assertNotIn(key, row)
        # A verdict was produced (non-vacuous): the music encoder beats the control on real melody -> GO.
        self.assertIsNotNone(result["verdict"])
        self.assertTrue(result["verdict"]["go"])
        self.assertIn(result["verdict"]["winner"], rb.ENCODERS)


class MainDispatchTests(unittest.TestCase):
    """The CLI entrypoint: --dry-run/default plan only; --run launches the real contest and reports it."""

    def test_default_and_dry_run_do_not_launch_the_contest(self):
        called = {"n": 0}

        def runner(dry_run=True, **kwargs):
            if not dry_run:
                called["n"] += 1
            return rb.run_batch(dry_run=True)

        with mock.patch.object(rb, "unload_lm_studio") as unload:
            self.assertEqual(rb.main([], batch_runner=runner), 0)
            self.assertEqual(rb.main(["--dry-run"], batch_runner=runner), 0)
        self.assertEqual(called["n"], 0)  # no contest launched
        unload.assert_not_called()

    def test_run_flag_launches_contest_and_reports_verdict(self):
        seen = {"dry_run": None}

        def runner(dry_run=True, **kwargs):
            seen["dry_run"] = dry_run
            return {"verdict": {"go": True, "winner": "musicfm"}}

        with mock.patch.object(rb, "gates_ok", return_value=(True, [])), \
             mock.patch.object(rb, "unload_lm_studio") as unload:
            rc = rb.main(["--run"], batch_runner=runner)
        self.assertEqual(rc, 0)
        self.assertEqual(seen["dry_run"], False)  # the CLI reached run_batch(dry_run=False)
        unload.assert_called_once()

    def test_run_flag_returns_nonzero_when_gates_block(self):
        called = {"n": 0}

        def runner(dry_run=True, **kwargs):
            if not dry_run:
                called["n"] += 1
            return {"verdict": {"go": False}}

        with mock.patch.object(rb, "gates_ok", return_value=(False, ["prereg gate: no tag"])):
            rc = rb.main(["--run"], batch_runner=runner)
        self.assertEqual(rc, 1)
        self.assertEqual(called["n"], 0)  # the real loop was never launched

    def test_run_flag_persists_verdict_through_the_real_run_batch(self):
        # End-to-end-ish: --run drives the real run_batch(dry_run=False) with injected fakes and the verdict
        # JSON lands on disk (proving the CLI path writes the verdict, not just prints a plan).
        with tempfile.TemporaryDirectory() as tmp:
            from open_transcriber.bakeoff import decide as D

            def runner(dry_run=True, **kwargs):
                if dry_run:
                    return rb.run_batch(dry_run=True)
                return rb.run_batch(
                    dry_run=False, out_root=Path(tmp),
                    extractor_factory=lambda name: object(), bp_factory=lambda: object(),
                    trainer=lambda *a, **k: ("model",),
                    evaluator=lambda encoder, *a, **k: {
                        "encoder": encoder,
                        "means": {"oct_on50_f1": 0.55 if encoder != "control" else 0.50,
                                  "synthetic_oct_on50_f1": 0.57 if encoder != "control" else 0.50,
                                  "mt": 0.5}},
                    decider=lambda trained, surviving: {"go": True, "winner": surviving[0]},
                    calibrate=lambda enc: {"hours": 1.0, "gb": 5.0, "mps_ok": True, "pilot_lead": True},
                    train_clips=[], dev_clips=[],
                    gpu_lock=_NullLock(), caffeinate=lambda: None, battery=lambda *a, **k: 0.0,
                )

            with mock.patch.object(rb, "gates_ok", return_value=(True, [])), \
                 mock.patch.object(rb, "bakeoff_out_dir", return_value=Path(tmp)), \
                 mock.patch.object(rb, "unload_lm_studio"):
                rc = rb.main(["--run"], batch_runner=runner)
            self.assertEqual(rc, 0)
            self.assertTrue((Path(tmp) / "verdict.json").exists())


if __name__ == "__main__":
    unittest.main()
