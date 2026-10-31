"""Bounded configuration subprocess queue; numerical work stays in the training CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Callable, Mapping, Sequence

import psutil

from runtime_resources import available_cpu_count, resource_snapshot

GIB = 1024 ** 3


def default_cpu_budget() -> int:
    available = available_cpu_count()
    return max(1, available - min(4, available // 8))


@dataclass(frozen=True)
class ConfigJob:
    key: str
    family: str
    command: Sequence[str]
    env: Mapping[str, str]
    cwd: Path
    log_path: Path
    cpu_reservation: int
    memory_reservation_bytes: int = 8 * GIB
    priority: tuple = ()


@dataclass
class _Running:
    job: ConfigJob
    process: subprocess.Popen
    log: object
    started: float
    record: dict
    profile_file: object | None = None
    counters: dict = field(default_factory=dict)
    last_cpu: float = 0.0
    last_sample: float = 0.0

    def sample(self, now: float, memory_available: int | None) -> None:
        processes = []
        try:
            parent = psutil.Process(self.process.pid)
            processes = [parent, *parent.children(recursive=True)]
        except psutil.Error:
            pass
        rss = 0
        for process in processes:
            try:
                with process.oneshot():
                    identity = (process.pid, process.create_time())
                    cpu = process.cpu_times()
                    rss += process.memory_info().rss
                    io = process.io_counters()
                    counters = self.counters.setdefault(identity, [0.0, 0, 0])
                    counters[0] = max(counters[0], cpu.user + cpu.system)
                    counters[1] = max(counters[1], io.read_bytes)
                    counters[2] = max(counters[2], io.write_bytes)
            except psutil.Error:
                continue
        cpu_seconds = sum(value[0] for value in self.counters.values())
        interval = now - self.last_sample
        row = {
            "elapsed_seconds": now - self.started,
            "sampled_cpu_seconds": cpu_seconds,
            "cpu_percent_one_core": (100 * (cpu_seconds - self.last_cpu) / interval) if interval > 0 else 0,
            "summed_rss_bytes": rss,
            "sampled_read_bytes": sum(value[1] for value in self.counters.values()),
            "sampled_write_bytes": sum(value[2] for value in self.counters.values()),
            "memory_available_bytes": memory_available,
        }
        self.record.update({key: row[key] for key in ("sampled_cpu_seconds", "sampled_read_bytes", "sampled_write_bytes")})
        self.record["peak_summed_rss_bytes"] = max(self.record.get("peak_summed_rss_bytes", 0), rss)
        if self.profile_file is not None:
            self.profile_file.write(json.dumps(row) + "\n")
            self.profile_file.flush()
        self.last_cpu, self.last_sample = cpu_seconds, now


def _stop_group(process: subprocess.Popen, *, wait: bool = True) -> None:
    # Descendants (including native-library pools) share this new process group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    if wait:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _write_summary(output_dir: Path, records: list[dict], cpu_budget: int, memory_budget: int) -> None:
    payload = {
        "cpu_budget": cpu_budget,
        "memory_budget_bytes": memory_budget,
        "profiling_note": "Sampled process-tree CPU/I/O may miss short-lived processes; summed RSS can double-count shared pages.",
        "configurations": records,
    }
    temporary = output_dir / "scheduler_summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_dir / "scheduler_summary.json")


def run_configurations(
    jobs: Sequence[ConfigJob], *, output_dir: Path, cpu_budget: int | None = None,
    max_concurrent_configs: int | None = None, profile: bool = True,
    legacy: bool = False, on_started: Callable[[dict], None] | None = None,
    on_finished: Callable[[dict], None] | None = None,
    poll_seconds: float = 0.2, sample_seconds: float = 1.0,
) -> list[dict]:
    """Run independent jobs with backfilling; caller owns study lock and manifests.

    Nonzero subprocess exits are returned as failed records; other admitted jobs
    continue. Exceptions/signals stop all owned process groups and are re-raised.
    Admission never changes model settings or training seeds. ``priority`` sorts
    within each family; families alternate when more than one can fit.
    """
    cpu_budget = default_cpu_budget() if cpu_budget is None else cpu_budget
    if not 1 <= cpu_budget <= available_cpu_count():
        raise ValueError("CPU budget must be positive and fit affinity/quota")
    if max_concurrent_configs is not None and max_concurrent_configs < 1:
        raise ValueError("max_concurrent_configs must be positive")
    if poll_seconds <= 0 or sample_seconds <= 0:
        raise ValueError("Polling/sample intervals must be positive")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("Duplicate configuration keys")
    if len({Path(job.log_path).resolve() for job in jobs}) != len(jobs):
        raise ValueError("Each configuration needs a unique log path")
    initial = resource_snapshot(output_dir)
    available_memory = initial["memory_available_bytes"]
    if available_memory is None:
        raise ValueError("Cannot establish available memory for safe admission")
    memory_budget = int(available_memory * 0.8)
    memory_floor = max(8 * GIB, int((initial["memory_total_bytes"] or available_memory) * 0.1))
    for job in jobs:
        if not job.command or not job.key:
            raise ValueError("Configurations need a key and command")
        if not 1 <= job.cpu_reservation <= cpu_budget:
            raise ValueError(f"Configuration {job.key} exceeds CPU budget")
        if not 1 <= job.memory_reservation_bytes <= memory_budget:
            raise ValueError(f"Configuration {job.key} exceeds memory budget")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = list(jobs) if legacy else sorted(jobs, key=lambda job: (job.priority, job.key))
    queued = time.monotonic()
    records = []
    running: list[_Running] = []
    family_dispatches = {job.family: 0 for job in jobs}
    old_handlers = {}

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Scheduler received signal {signum}")

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, interrupted)
        while pending or running:
            now = time.monotonic()
            snapshot = resource_snapshot(output_dir)
            free_memory = snapshot["memory_available_bytes"]
            for active in list(running):
                if profile and now - active.last_sample >= sample_seconds:
                    active.sample(now, free_memory)
                exit_code = active.process.poll()
                if exit_code is None:
                    continue
                if profile:
                    active.sample(now, free_memory)
                _stop_group(active.process)
                active.record.update(status="complete" if exit_code == 0 else "failed", exit_code=exit_code, elapsed_seconds=now - active.started)
                active.log.close()
                if active.profile_file:
                    active.profile_file.close()
                running.remove(active)
                if on_finished:
                    on_finished(active.record)
                _write_summary(output_dir, records, cpu_budget, memory_budget)
            limit = 1 if legacy else (max_concurrent_configs or len(jobs))
            while pending and len(running) < limit and free_memory is not None and free_memory >= memory_floor:
                used_cpu = sum(active.job.cpu_reservation for active in running)
                used_memory = sum(active.job.memory_reservation_bytes for active in running)
                candidates = [job for job in pending if (
                    job.cpu_reservation + used_cpu <= cpu_budget
                    and job.memory_reservation_bytes + used_memory <= memory_budget
                    and not (job.family == "ebm" and any(active.job.family == "ebm" for active in running))
                )]
                if not candidates:
                    break
                if not legacy:
                    candidates.sort(key=lambda job: (family_dispatches[job.family], job.priority, job.key))
                job = candidates[0]
                pending.remove(job)
                family_dispatches[job.family] += 1
                job.log_path.parent.mkdir(parents=True, exist_ok=True)
                record = {"key": job.key, "family": job.family, "status": "starting", "queue_wait_seconds": time.monotonic() - queued,
                          "cpu_reservation": job.cpu_reservation, "memory_reservation_bytes": job.memory_reservation_bytes,
                          "log_path": str(job.log_path.resolve())}
                records.append(record)
                log = job.log_path.open("a")
                try:
                    process = subprocess.Popen(list(job.command), cwd=job.cwd, env=dict(job.env), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    log.close()
                    record.update(status="failed", exit_code=None, error=str(error), elapsed_seconds=0.0)
                    if on_finished:
                        on_finished(record)
                    _write_summary(output_dir, records, cpu_budget, memory_budget)
                    continue
                started = time.monotonic()
                active = _Running(job, process, log, started, record, last_sample=started)
                running.append(active)
                record.update(status="running", pid=process.pid)
                if profile:
                    profile_path = output_dir / (hashlib.sha256(job.key.encode()).hexdigest()[:20] + ".resources.jsonl")
                    active.profile_file = profile_path.open("a")
                    record["profile_path"] = str(profile_path.resolve())
                    active.sample(started, free_memory)
                if on_started:
                    on_started(record.copy())
                _write_summary(output_dir, records, cpu_budget, memory_budget)
            if pending or running:
                time.sleep(poll_seconds)
    except BaseException:
        for active in running:
            _stop_group(active.process, wait=False)
        for active in running:
            _stop_group(active.process)
            active.log.close()
            if active.profile_file:
                active.profile_file.close()
            active.record.update(status="interrupted", exit_code=active.process.returncode, elapsed_seconds=time.monotonic() - active.started)
        for job in pending:
            records.append({"key": job.key, "family": job.family, "status": "pending"})
        _write_summary(output_dir, records, cpu_budget, memory_budget)
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    _write_summary(output_dir, records, cpu_budget, memory_budget)
    return records
