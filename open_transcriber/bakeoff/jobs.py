"""Resumable, battery-aware, caffeinated job runner for the bake-off's long jobs. Standard library only.

Ported from stage 0a ``judge/jobs.py`` (source ``aa88f5a``). The SheetSage environment/runner is not part of
this package. Added here (report §4.4 item 5): a mid-item battery watchdog (``battery_watchdog``) a long
extraction or probe loop can call between steps, so an hours-long GPU job pauses on battery and resumes on
AC without finishing the item first.

A job runs a queue of items in-process. Before each item the runner:

1. stops cleanly at ``--until HH:MM`` (local time), or when free space would fall below the floor;
2. waits while the Mac runs on battery (``pmset -g batt``), and carries on once it's back on AC;
3. skips items whose checkpoint (``OUT/<system>/<item>/done.json``) already exists: each system runs once per
   item (a failed item is retried only with ``retry_failed``, after a harness fix, and both runs are kept);
4. runs the item and writes ``done.json`` atomically: status, outputs with hashes, elapsed time, device and
   versions. A failure is recorded, not raised, so the queue goes on.

The process holds ``caffeinate -i`` for its lifetime. GPU jobs also hold an exclusive lock
(``DATA/gpu.lock``) and refuse to start while Demucs, yue2.cpp or SoulX-Singer are running. Every item
appends a line to ``OUT/runs.jsonl``; the log is ``OUT/logs/<job>.log``.

Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import fetch, paths

#: Command-line fragments of GPU programs that must not run alongside a GPU job.
GPU_PROGRAMS = ("demucs.separate", "bin/demucs", "yue-synth", "yue-plan", "yue-transcribe", "yue-tokenize",
                "SoulX", "soulx")


class LockError(RuntimeError):
    """Another GPU job holds the lock, or another GPU program is running."""


def parse_pmset(text: str) -> str:
    """'ac', 'battery' or 'unknown' from ``pmset -g batt`` output."""
    first = text.splitlines()[0] if text else ""
    if "AC Power" in first:
        return "ac"
    if "Battery Power" in first:
        return "battery"
    return "unknown"


def power_source() -> str:
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return parse_pmset(out)


def wait_for_ac(log: Callable[[str], None], poll: float = 60.0, source: Callable[[], str] = power_source,
                sleep: Callable[[float], None] = time.sleep) -> float:
    """Block while on battery; return the seconds waited."""
    waited = 0.0
    if source() == "battery":
        log("on battery: pausing until the Mac is back on AC")
        while source() == "battery":
            sleep(poll)
            waited += poll
        log(f"back on AC after {waited / 60:.1f} min")
    return waited


def battery_watchdog(log: Callable[[str], None] = lambda _m: None, poll: float = 60.0,
                     source: Callable[[], str] = power_source,
                     sleep: Callable[[float], None] = time.sleep) -> float:
    """Call this between the steps of a long job: pause while on battery, resume on AC; return seconds waited.

    The queue runner checks power only between items. A feature-extraction or probe loop that runs for hours
    per item should call this between checkpoints so it, too, pauses on battery and resumes on AC rather than
    draining the Mac. It is a thin alias of ``wait_for_ac`` with a safe default logger so a long job can call
    it without wiring one up.
    """
    return wait_for_ac(log, poll=poll, source=source, sleep=sleep)


def hold_caffeinate() -> subprocess.Popen | None:
    """``caffeinate -i -w <our pid>``: keeps the Mac awake while this process lives."""
    try:
        return subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())])
    except OSError:
        return None


def other_gpu_programs(patterns: Iterable[str] = GPU_PROGRAMS) -> list[str]:
    """Running processes (not ours) whose command line names a GPU program."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
    except OSError:
        return []
    mine = {os.getpid(), os.getppid()}
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, command = int(parts[0]), int(parts[1]), parts[2]
        if pid in mine or ppid == os.getpid():
            continue
        if any(p in command for p in patterns) and "bakeoff.jobs" not in command:
            found.append(f"{pid} {command[:80]}")
    return found


class GpuLock:
    """An exclusive ``flock`` on ``DATA/gpu.lock``; one GPU job at a time."""

    def __init__(self, path: Path | None = None, check_programs: bool = True):
        self.path = Path(path or paths.data_dir() / "gpu.lock")
        self.check_programs = check_programs
        self.handle = None

    def __enter__(self) -> "GpuLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise LockError(f"another GPU job holds {self.path}") from None
        if self.check_programs:
            busy = other_gpu_programs()
            if busy:
                self.__exit__()
                raise LockError("other GPU programs are running: " + "; ".join(busy))
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"{os.getpid()} {dt.datetime.now().isoformat(timespec='seconds')}\n")
        self.handle.flush()
        return self

    def __exit__(self, *exc) -> None:
        if self.handle and not self.handle.closed:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def write_json(path: Path, data: dict) -> None:
    """Atomic JSON write (temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def output_record(path: Path) -> dict:
    path = Path(path)
    return {"path": str(path), "sha256": fetch.sha256_file(path), "bytes": path.stat().st_size}


def past(until: str | None, started: dt.datetime, now: dt.datetime | None = None) -> bool:
    """True once local time reaches ``until`` (HH:MM); an ``until`` earlier than the start means tomorrow."""
    if not until:
        return False
    now = now or dt.datetime.now()
    hour, minute = (int(x) for x in until.split(":"))
    stop = started.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop <= started:
        stop += dt.timedelta(days=1)
    return now >= stop


@dataclass
class Job:
    """One queue: ``run(item) -> dict`` returns ``{"outputs": {name: path}, ...extra fields}`` or raises."""
    name: str
    system: str
    run: Callable[[dict], dict]
    gpu: bool = False
    device: str = "cpu"
    versions: dict = field(default_factory=dict)
    until: str | None = None
    need_bytes: int = 200 * 10**6
    floor: int = paths.FREE_SPACE_FLOOR
    retry_failed: bool = False
    out: Path | None = None
    poll: float = 60.0

    def item_dir(self, item_id: str) -> Path:
        return Path(self.out or paths.out_dir()) / self.system / item_id

    def log(self, message: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} [{self.name}] {message}"
        print(line, flush=True)
        log_dir = Path(self.out or paths.out_dir()) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / f"{self.name}.log", "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _ledger(self, record: dict) -> None:
        path = Path(self.out or paths.out_dir()) / "runs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def run_queue(self, items: list[dict], power: Callable[[], str] = power_source) -> dict:
        """Run every item not yet done; return counts. Holds caffeinate (and the GPU lock for GPU jobs)."""
        started = dt.datetime.now()
        keep_awake = hold_caffeinate()
        counts = {"ok": 0, "failed": 0, "skipped": 0, "stopped": False}
        lock = GpuLock() if self.gpu else None
        try:
            if lock:
                lock.__enter__()
            self.log(f"start: {len(items)} items, device {self.device}")
            for item in items:
                ident = item["id"]
                done_path = self.item_dir(ident) / "done.json"
                previous = read_json(done_path)
                if previous and (previous.get("status") == "ok" or not self.retry_failed):
                    counts["skipped"] += 1
                    continue
                if past(self.until, started):
                    self.log(f"reached {self.until}: stopping before {ident}")
                    counts["stopped"] = True
                    break
                try:
                    paths.require_free_space(self.need_bytes, floor=self.floor, path=self.item_dir(ident).parent)
                except paths.DiskFloorError as error:
                    self.log(str(error))
                    counts["stopped"] = True
                    break
                wait_for_ac(self.log, self.poll, power)
                attempt = 1
                if previous:
                    attempt = int(previous.get("attempt", 1)) + 1
                    write_json(done_path.with_name(f"done.attempt{attempt - 1}.json"), previous)
                record = self._run_one(item, attempt)
                write_json(done_path, record)
                self._ledger({k: record[k] for k in ("job", "system", "item", "status", "attempt", "started",
                                                     "elapsed", "device", "error")})
                counts[record["status"]] += 1
                self.log(f"{ident}: {record['status']} in {record['elapsed']:.1f} s"
                         + (f" ({record['error']})" if record["error"] else ""))
        finally:
            if lock:
                lock.__exit__()
            if keep_awake:
                keep_awake.terminate()
        self.log(f"end: {counts}")
        return counts

    def _run_one(self, item: dict, attempt: int) -> dict:
        t0 = time.time()
        started = dt.datetime.now().isoformat(timespec="seconds")
        record = {"job": self.name, "system": self.system, "item": item["id"], "attempt": attempt,
                  "started": started, "device": self.device, "versions": self.versions, "error": None,
                  "outputs": {}}
        try:
            result = self.run(item) or {}
            outputs = result.pop("outputs", {})
            record["outputs"] = {name: output_record(p) for name, p in outputs.items()}
            record.update(result)
            record["status"] = "ok"
        except Exception as error:  # a system failure is recorded and scored as an empty estimate
            record["status"] = "failed"
            record["error"] = f"{type(error).__name__}: {error}"[:500]
        record["elapsed"] = round(time.time() - t0, 3)
        return record


_ITEM = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def select(items: list[dict], ids: str | None = None, limit: int | None = None) -> list[dict]:
    """Items to run: all scored ones, or a comma-separated id list, optionally the first ``limit``."""
    chosen = [i for i in items if not i.get("status", "scored").startswith("excluded")]
    if ids:
        wanted = [x.strip() for x in ids.split(",") if x.strip()]
        for ident in wanted:
            if not _ITEM.match(ident):
                raise ValueError(f"bad item id {ident!r}")
        by_id = {i["id"]: i for i in chosen}
        missing = [x for x in wanted if x not in by_id]
        if missing:
            raise ValueError("unknown items: " + ", ".join(missing))
        chosen = [by_id[x] for x in wanted]
    return chosen[:limit] if limit else chosen


def python_versions(*modules: str) -> dict:
    """Versions of the running Python and the named modules (those that import)."""
    versions = {"python": sys.version.split()[0]}
    for name in modules:
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "?")
        except Exception:
            pass
    return versions
