"""Adaptive admission for independent, single-threaded feature tasks only."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import psutil

from model_creation.config_scheduler import _machine_busy_cores, _PressureWindow
from runtime_resources import available_cpu_count, resource_snapshot

GIB = 1024 ** 3
MEMORY_FLOOR = 32 * GIB
TASK_MEMORY = 2 * GIB
SAMPLE_SECONDS = 5.0
WINDOW_SECONDS = 60.0


class PilotDrained(Exception):
    """Admitted tasks finished; unsubmitted tasks require a later resume."""

    def __init__(self, completed, total):
        self.completed, self.total, self.pending = completed, total, total - completed
        super().__init__(f"Pilot drained: {completed}/{total} tasks complete; {self.pending} pending")


def admission_limit(*, max_workers, initial_workers, cpu_budget,
                    other_busy_cores, memory_available_bytes, inflight=0):
    """Pure policy: sustained other load reserves CPU; memory bounds new tasks."""
    initial = min(max_workers, initial_workers, cpu_budget)
    if other_busy_cores is None or not math.isfinite(other_busy_cores) or other_busy_cores < 0:
        desired = initial
    else:
        desired = max(1, min(max_workers, math.floor(cpu_budget - other_busy_cores)))
    slots = (max(0, (memory_available_bytes - MEMORY_FLOOR) // TASK_MEMORY)
             if memory_available_bytes is not None else 0)
    return int(min(desired, inflight + slots))


def growth_limit(current, desired, *, now, last_growth, telemetry_ready, initial_limit=0):
    """Reduce promptly; grow by at most eight after a full minute of telemetry."""
    if desired <= current:
        return desired, last_growth
    if current < initial_limit:
        return min(desired, initial_limit), last_growth
    if telemetry_ready and now - last_growth >= WINDOW_SECONDS:
        return min(desired, current + 8), now
    return current, last_growth


def _lower_priority():
    os.nice(5)


def _tree_cpu_seconds(counters):
    """Retain CPU counters of exited descendants so totals cannot go backwards."""
    try:
        parent = psutil.Process()
        processes = [parent, *parent.children(recursive=True)]
    except psutil.Error:
        return None
    for process in processes:
        try:
            identity = (process.pid, process.create_time())
            cpu = process.cpu_times()
            counters[identity] = max(counters.get(identity, 0.0), cpu.user + cpu.system)
        except psutil.Error:
            continue
    return sum(counters.values())


def adaptive_map(function, jobs, *, max_workers, initial_workers=8,
                 cpu_budget=None, volume=Path("/media/volume/catch22"),
                 progress_path=None, stop_admission_path=None):
    """Yield completed results, submitting only the currently admitted workload.

    Caller retains numerical/thread settings and checkpoint ownership. Spawn
    permits actual process growth on later submissions. Falling limits stop
    replenishment without killing work or restarting workers.
    """
    if max_workers < 1 or initial_workers < 1:
        raise ValueError("Worker limits must be positive")
    available = available_cpu_count()
    usable = max(1, available - min(4, available // 8))
    cap = min(max_workers, usable)
    budget = usable if cpu_budget is None else cpu_budget
    if not 1 <= budget <= usable:
        raise ValueError("CPU budget must fit available CPUs after the system reserve")
    initial = min(initial_workers, cap, budget)
    pending = deque(jobs)
    total, completed = len(pending), 0
    started = sampled = last_growth = time.monotonic()
    target, admitted_since_sample = initial, 0
    global_pressure = _PressureWindow(WINDOW_SECONDS, SAMPLE_SECONDS, started)
    other_pressure = _PressureWindow(WINDOW_SECONDS, SAMPLE_SECONDS, started)
    counters = {}
    own_previous = _tree_cpu_seconds(counters)
    _machine_busy_cores()  # Prime psutil; its first nonblocking reading is discarded.
    snapshot = resource_snapshot(Path(volume))
    global_mean = other_mean = own_cores = global_cores = None
    inflight, draining, peak_inflight = set(), False, 0

    def write_progress(state="running", error=None):
        if progress_path is None:
            return
        path = Path(progress_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"status": state, "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                   "elapsed_seconds": time.monotonic() - started,
                   "target_limit": target, "inflight": len(inflight), "completed": completed,
                   "peak_inflight": peak_inflight,
                   "total": total, "pending": len(pending), "cpu_budget": budget,
                   "max_workers": cap, "available_cpus_at_start": available,
                   "sample_global_busy_cores": global_cores, "sample_own_busy_cores": own_cores,
                   "rolling_global_busy_cores": global_mean, "rolling_other_busy_cores": other_mean,
                   "memory_floor_bytes": MEMORY_FLOOR, "new_task_memory_bytes": TASK_MEMORY,
                   "snapshot": snapshot, "error": error}
        partial = path.with_suffix(path.suffix + ".partial")
        partial.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        partial.replace(path)

    write_progress()
    try:
        with ProcessPoolExecutor(max_workers=cap, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_lower_priority) as pool:
            while pending or inflight:
                now = time.monotonic()
                if stop_admission_path is not None and Path(stop_admission_path).exists():
                    draining = True
                if now - sampled >= SAMPLE_SECONDS:
                    snapshot = resource_snapshot(Path(volume))
                    own_total = _tree_cpu_seconds(counters)
                    own_cores = ((own_total - own_previous) / (now - sampled)
                                 if own_total is not None and own_previous is not None else None)
                    own_previous = own_total
                    global_cores = _machine_busy_cores()
                    other = (max(0.0, global_cores - own_cores)
                             if global_cores is not None and own_cores is not None else None)
                    global_pressure.add(now, global_cores)
                    other_pressure.add(now, other)
                    global_mean, other_mean = global_pressure.mean(now), other_pressure.mean(now)
                    desired = admission_limit(
                        max_workers=cap, initial_workers=initial, cpu_budget=budget,
                        other_busy_cores=other_mean, memory_available_bytes=snapshot.get("memory_available_bytes"),
                        inflight=len(inflight))
                    target, last_growth = growth_limit(
                        target, desired, now=now, last_growth=last_growth,
                        telemetry_ready=global_mean is not None and other_mean is not None,
                        initial_limit=initial)
                    sampled, admitted_since_sample = now, 0
                    write_progress("draining" if draining else "running")
                free_memory = snapshot.get("memory_available_bytes")
                memory_slots = (max(0, (free_memory - MEMORY_FLOOR) // TASK_MEMORY - admitted_since_sample)
                                if free_memory is not None else 0)
                while pending and not draining and len(inflight) < target and memory_slots > 0:
                    inflight.add(pool.submit(function, pending.popleft()))
                    peak_inflight = max(peak_inflight, len(inflight))
                    admitted_since_sample += 1
                    memory_slots -= 1
                if draining and not inflight:
                    break
                delay = max(0.01, SAMPLE_SECONDS - (time.monotonic() - sampled))
                if inflight:
                    finished, _ = wait(inflight, timeout=delay, return_when=FIRST_COMPLETED)
                    for future in finished:
                        inflight.remove(future)
                        result = future.result()
                        completed += 1
                        yield result
                elif pending:
                    time.sleep(delay)
        if pending:
            write_progress("drained")
            raise PilotDrained(completed, total)
        write_progress("complete")
    except PilotDrained:
        raise
    except BaseException as exc:
        write_progress("failed", error=f"{type(exc).__name__}: {exc}")
        raise
