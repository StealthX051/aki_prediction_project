"""Admission policy tests avoid native estimators and research data."""
import json
import os

import pytest

from scripts.window_duration_resources import (
    GIB, PilotDrained, adaptive_map, admission_limit, growth_limit,
)
from model_creation.config_scheduler import _PressureWindow


def _worker_probe(value):
    return abs(value), os.getpid(), os.getpriority(os.PRIO_PROCESS, 0)


def _limit(other, memory=200 * GIB, inflight=8):
    return admission_limit(max_workers=60, initial_workers=8, cpu_budget=60,
                           other_busy_cores=other, memory_available_bytes=memory, inflight=inflight)


def test_other_workload_reserves_cpu_and_missing_telemetry_never_expands():
    assert _limit(52) == 8
    assert _limit(0) == 60
    assert _limit(60) == 1
    assert _limit(None, inflight=60) == 8
    assert _limit(float("nan")) == 8
    assert _limit(0, memory=34 * GIB) == 9
    assert _limit(0, memory=31 * GIB, inflight=0) == 0
    assert _limit(0, memory=None, inflight=0) == 0


def test_growth_is_bounded_and_reductions_do_not_wait_for_cooldown():
    assert growth_limit(8, 60, now=59, last_growth=0, telemetry_ready=True) == (8, 0)
    assert growth_limit(8, 60, now=60, last_growth=0, telemetry_ready=True) == (16, 60)
    assert growth_limit(16, 60, now=61, last_growth=60, telemetry_ready=True) == (16, 60)
    assert growth_limit(60, 8, now=61, last_growth=60, telemetry_ready=False) == (8, 60)
    assert growth_limit(8, 60, now=600, last_growth=0, telemetry_ready=False) == (8, 0)
    assert growth_limit(0, 8, now=5, last_growth=0, telemetry_ready=False, initial_limit=8) == (8, 0)


def test_full_pressure_window_rejects_short_dips_and_missing_samples():
    pressure = _PressureWindow(60, 5, 0)
    for now in range(5, 60, 5):
        pressure.add(now, 52)
    assert pressure.mean(55) is None
    pressure.add(60, 0)
    assert pressure.mean(60) == pytest.approx(52 * 55 / 60)
    pressure.add(65, None)
    assert pressure.mean(65) is None


def test_existing_drain_request_admits_no_work_and_records_pending(tmp_path):
    stop, progress = tmp_path / "stop", tmp_path / "progress.json"
    stop.touch()
    with pytest.raises(PilotDrained) as caught:
        list(adaptive_map(abs, [-1, -2], max_workers=2, volume=tmp_path,
                          progress_path=progress, stop_admission_path=stop))
    assert caught.value.completed == 0
    assert caught.value.pending == 2
    record = json.loads(progress.read_text())
    assert record["status"] == "drained"
    assert record["completed"] == 0 and record["pending"] == 2


def test_spawn_pool_returns_results_and_stays_bounded(tmp_path):
    progress = tmp_path / "progress.json"
    parent_priority = os.getpriority(os.PRIO_PROCESS, 0)
    results = list(adaptive_map(_worker_probe, [-1, -2, -3, -4], max_workers=2,
                                initial_workers=2, volume=tmp_path, progress_path=progress))
    assert sorted(row[0] for row in results) == [1, 2, 3, 4]
    assert 1 <= len({row[1] for row in results}) <= 2
    assert all(row[2] == parent_priority + 5 for row in results)
    record = json.loads(progress.read_text())
    assert record["status"] == "complete"
    assert record["completed"] == 4 and record["inflight"] == 0
    assert record["max_workers"] <= 2 and record["target_limit"] <= 2
    assert record["peak_inflight"] == 2
