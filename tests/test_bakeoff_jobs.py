"""Tests for the bake-off job runner: power parsing, checkpoints, the GPU lock, the disk floor and the
mid-item battery watchdog. Ported from stage 0a ``tests_judge/test_jobs.py`` (source ``aa88f5a``)."""

import datetime as dt
import json
import os
import tempfile
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from open_transcriber.bakeoff import jobs

AC = """Now drawing from 'AC Power'
 -InternalBattery-0 (id=22610019)	100%; charged; 0:00 remaining present: true
"""
BATTERY = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=22610019)	87%; discharging; 5:12 remaining present: true
"""
Usage = namedtuple("Usage", "total used free")


class PowerTests(unittest.TestCase):
    def test_pmset_parsing(self):
        self.assertEqual(jobs.parse_pmset(AC), "ac")
        self.assertEqual(jobs.parse_pmset(BATTERY), "battery")
        self.assertEqual(jobs.parse_pmset(""), "unknown")

    def test_wait_for_ac_pauses_on_battery(self):
        states = iter(["battery", "battery", "battery", "ac"])
        slept = []
        waited = jobs.wait_for_ac(lambda m: None, poll=60, source=lambda: next(states), sleep=slept.append)
        self.assertEqual(waited, 120)
        self.assertEqual(slept, [60, 60])

    def test_battery_watchdog_pauses_mid_item(self):
        # A long job calls this between steps; it pauses on battery and resumes on AC.
        states = iter(["battery", "battery", "ac"])  # if-check, one loop poll, then AC
        slept = []
        waited = jobs.battery_watchdog(poll=30, source=lambda: next(states), sleep=slept.append)
        self.assertEqual(waited, 30)
        self.assertEqual(slept, [30])
        # On AC it returns at once, waiting nothing.
        self.assertEqual(jobs.battery_watchdog(source=lambda: "ac", sleep=slept.append), 0.0)

    def test_until(self):
        start = dt.datetime(2026, 10, 19, 23, 30)
        self.assertFalse(jobs.past("07:00", start, dt.datetime(2026, 10, 20, 6, 59)))
        self.assertTrue(jobs.past("07:00", start, dt.datetime(2026, 10, 20, 7, 0)))
        self.assertFalse(jobs.past(None, start))


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.env = mock.patch.dict(os.environ, {"JUDGE_OUT": str(self.tmp / "out"), "JUDGE_DATA": str(self.tmp / "data")})
        self.env.start()
        self.caffeinate = mock.patch.object(jobs, "hold_caffeinate", return_value=None)
        self.caffeinate.start()
        self.disk = mock.patch("shutil.disk_usage", return_value=Usage(1000e9, 100e9, 900e9))
        self.disk.start()

    def tearDown(self):
        self.disk.stop()
        self.caffeinate.stop()
        self.env.stop()

    def make_job(self, calls, **kw):
        def run(item):
            calls.append(item["id"])
            if item["id"] == "bad":
                raise RuntimeError("model crashed")
            out = self.tmp / "out" / "sys" / item["id"] / "x.txt"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(item["id"])
            return {"outputs": {"x": out}, "notes": 3}
        return jobs.Job(name="test", system="sys", run=run, **kw)

    def test_checkpoints_skip_done_items_and_record_failures(self):
        calls = []
        items = [{"id": "a"}, {"id": "bad"}, {"id": "b"}]
        counts = self.make_job(calls).run_queue(items, power=lambda: "ac")
        self.assertEqual(calls, ["a", "bad", "b"])
        self.assertEqual((counts["ok"], counts["failed"]), (2, 1))
        done = json.loads((self.tmp / "out/sys/a/done.json").read_text())
        self.assertEqual(done["status"], "ok")
        self.assertEqual(done["notes"], 3)
        self.assertEqual(len(done["outputs"]["x"]["sha256"]), 64)
        failed = json.loads((self.tmp / "out/sys/bad/done.json").read_text())
        self.assertIn("model crashed", failed["error"])
        calls.clear()
        counts = self.make_job(calls).run_queue(items, power=lambda: "ac")
        self.assertEqual(calls, [])
        self.assertEqual(counts["skipped"], 3)
        counts = self.make_job(calls, retry_failed=True).run_queue(items, power=lambda: "ac")
        self.assertEqual(calls, ["bad"])
        self.assertTrue((self.tmp / "out/sys/bad/done.attempt1.json").exists())
        ledger = (self.tmp / "out/runs.jsonl").read_text().splitlines()
        self.assertEqual(len(ledger), 4)

    def test_disk_floor_stops_the_queue(self):
        # Should fail: below the floor, nothing runs.
        calls = []
        with mock.patch("shutil.disk_usage", return_value=Usage(1000e9, 860e9, 140e9)):
            counts = self.make_job(calls).run_queue([{"id": "a"}], power=lambda: "ac")
        self.assertEqual(calls, [])
        self.assertTrue(counts["stopped"])

    def test_second_gpu_lock_is_refused(self):
        # Should fail: one GPU job at a time.
        lock_path = self.tmp / "gpu.lock"
        with jobs.GpuLock(lock_path, check_programs=False):
            with self.assertRaises(jobs.LockError):
                with jobs.GpuLock(lock_path, check_programs=False):
                    pass
        with jobs.GpuLock(lock_path, check_programs=False):
            pass  # free again afterwards

    def test_select(self):
        items = [{"id": "a", "status": "scored"}, {"id": "b", "status": "excluded"}, {"id": "c"}]
        self.assertEqual([i["id"] for i in jobs.select(items)], ["a", "c"])
        self.assertEqual([i["id"] for i in jobs.select(items, "c,a")], ["c", "a"])
        with self.assertRaises(ValueError):
            jobs.select(items, "b")


if __name__ == "__main__":
    unittest.main()
