"""Observable scheduling, failure and process-lifecycle contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from model_creation import config_scheduler as scheduler


@pytest.fixture(autouse=True)
def resources(monkeypatch):
    monkeypatch.setattr(scheduler, "available_cpu_count", lambda: 64)
    monkeypatch.setattr(scheduler, "resource_snapshot", lambda path: {
        "memory_available_bytes": 64 * scheduler.GIB,
        "memory_total_bytes": 80 * scheduler.GIB,
    })


def job(tmp_path, key, *, family="xgboost", cpus=1, seconds=0.1, exit_code=0, priority=()):
    code = f"import time,sys; print({key!r},flush=True); time.sleep({seconds}); sys.exit({exit_code})"
    return scheduler.ConfigJob(key, family, [sys.executable, "-c", code], os.environ.copy(), tmp_path,
                               tmp_path / f"{key}.log", cpus, priority=priority)


def test_backfills_within_resources_and_serializes_ebm(tmp_path):
    # xgb cannot fit alongside EBM, but LR can. A second EBM must wait.
    jobs = [job(tmp_path, "a-ebm", family="ebm", cpus=3),
            job(tmp_path, "b-xgb", cpus=2), job(tmp_path, "c-lr", family="logistic_regression"),
            job(tmp_path, "d-ebm", family="ebm", cpus=1)]
    active = {}
    events = []

    def started(record):
        active[record["key"]] = record
        events.append(("start", record["key"]))
        assert sum(r["cpu_reservation"] for r in active.values()) <= 4
        assert sum(r["family"] == "ebm" for r in active.values()) <= 1

    def finished(record):
        active.pop(record["key"])
        events.append(("finish", record["key"]))

    records = scheduler.run_configurations(jobs, output_dir=tmp_path / "profile", cpu_budget=4,
                                           admission_policy="reserved",
                                           on_started=started, on_finished=finished, poll_seconds=0.01, sample_seconds=0.02)
    assert events[:2] == [("start", "a-ebm"), ("start", "c-lr")]
    assert all(record["status"] == "complete" for record in records)
    assert not active
    summary = json.loads((tmp_path / "profile/scheduler_summary.json").read_text())
    assert len(summary["configurations"]) == 4
    for record in records:
        samples = [json.loads(line) for line in Path(record["profile_path"]).read_text().splitlines()]
        assert samples[-1]["sampled_cpu_seconds"] >= samples[0]["sampled_cpu_seconds"]
        assert record["elapsed_seconds"] > 0
        assert Path(record["log_path"]).read_text().strip() == record["key"]


def test_legacy_order_and_failure_do_not_lose_other_jobs(tmp_path):
    jobs = [job(tmp_path, "z", exit_code=7), job(tmp_path, "a", priority=(-1,))]
    events = []
    records = scheduler.run_configurations(jobs, output_dir=tmp_path / "profile", cpu_budget=2,
                                           legacy=True, profile=False, poll_seconds=0.01,
                                           on_started=lambda r: events.append(("start", r["key"])),
                                           on_finished=lambda r: events.append(("finish", r["key"])))
    assert events == [("start", "z"), ("finish", "z"), ("start", "a"), ("finish", "a")]
    assert [(r["key"], r["status"], r["exit_code"]) for r in records] == [("z", "failed", 7), ("a", "complete", 0)]
    assert not list((tmp_path / "profile").glob("*.resources.jsonl"))
    assert json.loads((tmp_path / "profile/scheduler_summary.json").read_text())["status"] == "failed"


def test_duplicate_or_oversize_requests_fail_before_outputs(tmp_path):
    candidate = job(tmp_path, "a", cpus=5)
    output = tmp_path / "not-created"
    with pytest.raises(ValueError, match="Duplicate"):
        scheduler.run_configurations([candidate, candidate], output_dir=output, cpu_budget=4)
    with pytest.raises(ValueError, match="exceeds CPU"):
        scheduler.run_configurations([candidate], output_dir=output, cpu_budget=4)
    huge = job(tmp_path, "huge")
    huge = scheduler.ConfigJob(**{**huge.__dict__, "memory_reservation_bytes": 100 * scheduler.GIB})
    with pytest.raises(ValueError, match="exceeds memory"):
        scheduler.run_configurations([huge], output_dir=output, cpu_budget=4)
    assert not output.exists()


def test_memory_pressure_pauses_admission(tmp_path, monkeypatch):
    calls = 0

    def snapshot(path):
        nonlocal calls
        calls += 1
        return {"memory_total_bytes": 80 * scheduler.GIB,
                "memory_available_bytes": (4 if 2 <= calls <= 4 else 64) * scheduler.GIB}

    monkeypatch.setattr(scheduler, "resource_snapshot", snapshot)
    started_after = []
    scheduler.run_configurations([job(tmp_path, "a", seconds=0)], output_dir=tmp_path / "profile",
                                  cpu_budget=2, poll_seconds=0.01, profile=False,
                                  on_started=lambda record: started_after.append(calls))
    assert started_after[0] >= 5


def test_family_fairness_and_concurrency_cap(tmp_path):
    jobs = [job(tmp_path, f"a{i}", family="a", priority=(0,), seconds=0) for i in range(3)]
    jobs += [job(tmp_path, f"b{i}", family="b", priority=(1,), seconds=0) for i in range(3)]
    jobs += [job(tmp_path, "c0", family="c", priority=(2,), seconds=0)]
    starts = []
    scheduler.run_configurations(jobs, output_dir=tmp_path / "profile", cpu_budget=4,
                                  max_concurrent_configs=1, profile=False, poll_seconds=0.01,
                                  on_started=lambda r: starts.append(r["family"]))
    assert starts[:3] == ["a", "b", "c"]


def test_spawn_failure_and_callback_error_cleanup(tmp_path):
    invalid = job(tmp_path, "bad")
    invalid = scheduler.ConfigJob(**{**invalid.__dict__, "command": ["/nonexistent-executable"]})
    records = scheduler.run_configurations([invalid, job(tmp_path, "good", seconds=0)],
                                           output_dir=tmp_path / "profile", cpu_budget=2, profile=False, poll_seconds=0.01)
    assert {r["key"]: r["status"] for r in records} == {"bad": "failed", "good": "complete"}
    child_pid = []

    def broken(record):
        child_pid.append(record["pid"])
        raise RuntimeError("manifest failure")

    with pytest.raises(RuntimeError, match="manifest failure"):
        scheduler.run_configurations([job(tmp_path, "long", seconds=60)], output_dir=tmp_path / "failed",
                                      cpu_budget=2, on_started=broken, profile=False)
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid[0], 0)
    status = json.loads((tmp_path / "failed/scheduler_summary.json").read_text())
    assert status["configurations"][0]["status"] == "interrupted"


def test_sigterm_reaps_owned_process_group(tmp_path):
    # Run the controller as a real process so SIGTERM exercises its installed handler.
    script = tmp_path / "controller.py"
    root = Path(__file__).resolve().parents[1]
    script.write_text('''import os, sys
from pathlib import Path
from model_creation.config_scheduler import ConfigJob, run_configurations
root = Path(sys.argv[1])
code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(p.pid,flush=True); time.sleep(120)"
job=ConfigJob('signal', 'xgboost', [sys.executable, '-c', code], os.environ.copy(), root, root/'child.log', 1, 1)
run_configurations([job], output_dir=root/'profile', cpu_budget=1, profile=False, poll_seconds=.02)
''')
    env = dict(os.environ, PYTHONPATH=str(root))
    controller = subprocess.Popen([sys.executable, str(script), str(tmp_path)], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owned_children = []
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "child.log").exists() or not (tmp_path / "child.log").read_text().strip():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        grandchild = int((tmp_path / "child.log").read_text().strip())
        owned_children = scheduler.psutil.Process(controller.pid).children(recursive=True)
        controller.send_signal(signal.SIGTERM)
        assert controller.wait(timeout=10) != 0
        # A reparented zombie may await host init; it must no longer execute.
        try:
            process = scheduler.psutil.Process(grandchild)
            assert process.status() == scheduler.psutil.STATUS_ZOMBIE
        except scheduler.psutil.NoSuchProcess:
            pass
        summary = json.loads((tmp_path / "profile/scheduler_summary.json").read_text())
        assert summary["configurations"][0]["status"] == "interrupted"
    finally:
        if controller.poll() is None:
            if not owned_children:
                try:
                    owned_children = scheduler.psutil.Process(controller.pid).children(recursive=True)
                except scheduler.psutil.NoSuchProcess:
                    pass
            controller.terminate()
            try:
                controller.wait(timeout=10)
            except subprocess.TimeoutExpired:
                controller.kill()
                controller.wait()
        # A failed assertion must not leave independently grouped test workers.
        for child in owned_children:
            try:
                child.kill()
            except scheduler.psutil.NoSuchProcess:
                pass


def test_pressure_window_weights_time_and_rejects_missing_or_stale_samples():
    window = scheduler._PressureWindow(4, 1, 0)
    for second, busy in enumerate([20, 40, 60, 80], start=1):
        window.add(second, busy)
        if second < 4:
            assert window.mean(second) is None
    assert window.mean(4) == pytest.approx(50)
    window.add(5, 100)
    assert window.mean(5) == pytest.approx(70)
    assert window.mean(8) is None
    window.add(9, 0)  # The missing interval cannot count as idle CPU.
    assert window.mean(9) is None
    for second in range(10, 14):
        window.add(second, 0)
    assert window.mean(13) == 0
    window.add(14, None)
    assert window.mean(14) is None


def test_machine_pressure_includes_other_work_on_allowed_cpus(monkeypatch):
    monkeypatch.setattr(scheduler.psutil, "cpu_percent", lambda **kwargs: [100, 50, 25, 0])
    monkeypatch.setattr(scheduler.os, "sched_getaffinity", lambda pid: {0, 2})
    assert scheduler._machine_busy_cores() == 1.25


@pytest.mark.parametrize("busy,overlaps", [(0.0, True), (2.0, False), (None, False)])
def test_adaptive_admission_requires_idle_window_and_cooldown(tmp_path, busy, overlaps):
    active, events = set(), []

    def started(record):
        events.append(("start", record["key"], time.monotonic(), bool(active), record))
        active.add(record["key"])

    def finished(record):
        active.remove(record["key"])

    scheduler.run_configurations(
        [job(tmp_path, "a", cpus=2, seconds=.5), job(tmp_path, "b", cpus=2, seconds=.1)],
        output_dir=tmp_path / "profile", cpu_budget=2, profile=False,
        on_started=started, on_finished=finished, poll_seconds=.005,
        adaptive_window_seconds=.1, admission_cooldown_seconds=.15,
        pressure_sample_seconds=.02, pressure_provider=lambda: busy,
    )
    assert events[1][3] is overlaps
    if overlaps:
        assert events[1][2] - events[0][2] >= .14
        assert events[1][4]["admission_reason"] == "sustained_spare_cpu"
    else:
        assert events[1][4]["admission_reason"] == "reserved_capacity"
    # Admission telemetry is independent of expensive per-process profiling.
    rows = [json.loads(row) for row in (tmp_path / "profile/scheduler_pressure.jsonl").read_text().splitlines()]
    assert rows and not list((tmp_path / "profile").glob("*.resources.jsonl"))


def test_adaptive_allows_concurrent_ebm_and_defaults_to_twelve_jobs(tmp_path):
    active, peak, first_finished = set(), [], []

    def started(record):
        active.add(record["key"])
        peak.append(len(active))
        if record["key"] == "ebm12":
            assert first_finished

    def finished(record):
        active.remove(record["key"])
        first_finished.append(record["key"])

    jobs = [job(tmp_path, f"ebm{i:02d}", family="ebm", seconds=.4) for i in range(13)]
    jobs = [scheduler.ConfigJob(**{**j.__dict__, "memory_reservation_bytes": 1}) for j in jobs]
    scheduler.run_configurations(jobs, output_dir=tmp_path / "profile", cpu_budget=64,
                                 profile=False, poll_seconds=.005,
                                 on_started=started, on_finished=finished)
    assert max(peak) == 12
    assert not active


def test_native_drain_finishes_active_jobs_and_preserves_pending_request(tmp_path):
    request = tmp_path / "drain"
    events = []

    def started(record):
        events.append(("start", record["key"]))
        request.write_text(json.dumps({"pid": os.getpid(), "create_time": scheduler.psutil.Process().create_time()}))

    with pytest.raises(scheduler.SchedulerDrained) as error:
        scheduler.run_configurations(
            [job(tmp_path, "a"), job(tmp_path, "b")], output_dir=tmp_path / "profile",
            cpu_budget=2, drain_file=request, profile=False, poll_seconds=.005,
            on_started=started, on_finished=lambda r: events.append(("finish", r["key"])),
        )
    assert events == [("start", "a"), ("finish", "a")]
    assert error.value.pending_keys == ["b"]
    assert (tmp_path / "a.log").read_text().strip() == "a"
    assert not (tmp_path / "b.log").exists()
    summary = json.loads((tmp_path / "profile/scheduler_summary.json").read_text())
    assert summary["status"] == "drained"
    assert summary["configurations"][-1]["status"] == "pending"


def test_existing_drain_request_schedules_zero_fits(tmp_path):
    request = tmp_path / "drain"
    request.write_text(json.dumps({"pid": os.getpid(), "create_time": scheduler.psutil.Process().create_time()}))
    with pytest.raises(scheduler.SchedulerDrained) as error:
        scheduler.run_configurations([job(tmp_path, "untouched")], output_dir=tmp_path / "profile",
                                     drain_file=request, cpu_budget=1)
    assert error.value.pending_keys == ["untouched"]
    assert not (tmp_path / "untouched.log").exists()


def test_stale_or_invalid_drain_request_cannot_stop_a_different_controller(tmp_path):
    request = tmp_path / "drain"
    current = {"pid": os.getpid(), "create_time": scheduler.psutil.Process().create_time()}
    for payload in ["{", "[]", "{}", json.dumps({**current, "pid": -1}),
                    json.dumps({**current, "create_time": current["create_time"] - 1})]:
        request.write_text(payload)
        assert not scheduler._drain_requested(request)
    # A stale file also must not change the actual queue behavior.
    records = scheduler.run_configurations([job(tmp_path, "valid")], output_dir=tmp_path / "profile",
                                           drain_file=request, cpu_budget=1, profile=False)
    assert records[0]["status"] == "complete"
    request.write_text(json.dumps(current))
    assert scheduler._drain_requested(request)


def test_adaptive_oversubscription_still_obeys_aggregate_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "resource_snapshot", lambda path: {
        "memory_total_bytes": 80 * scheduler.GIB, "memory_available_bytes": 20 * scheduler.GIB,
    })
    active, peaks = set(), []

    def started(record):
        active.add(record["key"])
        peaks.append(len(active))

    scheduler.run_configurations(
        [job(tmp_path, f"job{i}", seconds=.4) for i in range(3)],
        output_dir=tmp_path / "profile", cpu_budget=1, profile=False,
        on_started=started, on_finished=lambda r: active.remove(r["key"]), poll_seconds=.005,
        adaptive_window_seconds=.1, admission_cooldown_seconds=.1,
        pressure_sample_seconds=.02, pressure_provider=lambda: 0.,
    )
    assert max(peaks) == 2  # Two 8-GiB reservations fit; three exceed 80% of 20 GiB.
