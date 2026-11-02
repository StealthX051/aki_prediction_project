#!/usr/bin/env python3
"""Supervise the approved full-budget canary gate, then ICU/mortality core runs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
STUDIES = Path("/media/volume/catch22/data/aki_prediction_project/studies")
SCOPES = ("icu_waveform_sensitivity", "mortality_waveform_sensitivity")
ORCHESTRATION_OVERLAY_FILES = frozenset({
    "run_catch22.py", "model_creation/config_scheduler.py",
    "model_creation/experiment_queue.py", "scripts/cpu/run-optimized-core.py",
})


class DrainedRun(RuntimeError):
    """A requested drain finished active work but intentionally left a queue."""


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _execution_snapshot() -> dict:
    from run_catch22 import _study_identity
    names = ("run_catch22.py", "model_creation/config_scheduler.py", "model_creation/experiment_queue.py",
             "model_creation/scheduler_canary.py", "model_creation/run_inventory.py", "model_creation/split_plan.py",
             "scripts/cpu/run-optimized-core.py")
    return {"recorded_at": _now(), "study_identity": _study_identity(), "python": sys.version,
            "python_executable": sys.executable, "platform": platform.platform(),
            "numerical_runtime_versions": {name: importlib.metadata.version(name)
                                           for name in ("scipy", "joblib", "threadpoolctl")},
            "source_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in names}}


def _validate_execution_snapshot(control: Path, snapshot: dict) -> dict | None:
    """Accept only exact original sources or an explicitly committed scheduler overlay."""
    path = control / "execution-snapshot.json"
    if not path.exists():
        _write(path, snapshot)
        return None
    original_bytes = path.read_bytes()
    original = json.loads(original_bytes)
    fixed = ("study_identity", "python", "python_executable", "numerical_runtime_versions", "platform")
    if any(snapshot.get(key) != original.get(key) for key in fixed):
        raise ValueError("Supervisor source/environment changed since its frozen execution snapshot")
    if snapshot["source_sha256"] == original["source_sha256"]:
        return None
    attestation_path = control / "orchestration-overlay.json"
    if not attestation_path.exists():
        raise ValueError("Supervisor source/environment changed without a reviewed orchestration overlay")
    attestation = json.loads(attestation_path.read_text())
    before, after = original["source_sha256"], snapshot["source_sha256"]
    changed = {name for name in set(before) | set(after) if before.get(name) != after.get(name)}
    if (not changed or not changed <= ORCHESTRATION_OVERLAY_FILES
            or attestation.get("original_snapshot_sha256") != hashlib.sha256(original_bytes).hexdigest()
            or attestation.get("before_source_sha256") != before
            or attestation.get("after_source_sha256") != after
            or attestation.get("study_identity") != snapshot["study_identity"]):
        raise ValueError("Orchestration overlay differs from its reviewed transition")
    source = Path(attestation["source_checkout"])
    revision = attestation["source_commit"]
    if not isinstance(revision, str) or len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Overlay must name an exact source commit")
    for name in changed:
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=source)
        if hashlib.sha256(committed).hexdigest() != after[name]:
            raise ValueError(f"Overlay file does not match committed source: {name}")
    return attestation


def _verify_saved_canary(args, agreement: dict) -> None:
    """Recheck saved evidence and bundles without fitting or resampling again."""
    from data_preparation.artifact_metadata import file_sha256
    from model_creation.run_inventory import verify_completed_configuration
    dataset = args.baseline_root / "data/processed/aki_features_master_wide.csv"
    identity = json.loads((args.study_root / "control/execution-snapshot.json").read_text())["study_identity"]
    if any(agreement["bootstrap"].get(k) != v for k, v in {"replicates": 5000, "seed": 0, "stratified": False}.items()):
        raise ValueError("Saved canary statistical settings changed")
    for side in ("legacy", "optimized"):
        directory = Path(agreement[f"{side}_config"]).resolve()
        expected_root = args.study_root / "validation/legacy-canary" if side == "legacy" else args.study_root
        expected_directory = expected_root / "results/catch22/experiments" / SCOPES[0] / "nested_cv/models/xgboost/icu_admission/non_windowed/preop_and_all_waveforms"
        expected_fp = {**identity, "outer_folds": 5, "inner_folds": 3, "calibration_folds": 5,
                       "repeats": 1, "n_trials": 50, "diagnostic": False,
                       "threads_per_model": 12 if side == "legacy" else 4}
        fingerprint = agreement[f"{side}_fingerprint"]
        if directory != expected_directory.resolve() or any(fingerprint.get(k) != v for k, v in expected_fp.items()):
            raise ValueError("Saved canary identity or validation settings changed")
        for relative, digest in agreement["artifact_sha256"][side].items():
            artifact = (directory / relative).resolve()
            if not artifact.is_relative_to(directory) or file_sha256(artifact) != digest:
                raise ValueError(f"Saved canary artifact changed: {artifact}")
        verify_completed_configuration(directory, dataset_path=dataset,
                                       expected_fingerprint=fingerprint)


def inventory(root: Path) -> dict:
    """Hash files and record symlinks without following linked directories."""
    if not root.is_dir():
        raise ValueError(f"Protected study does not exist: {root}")
    files, links = {}, {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in sorted([*dirs, *names]):
            path = Path(directory) / name
            relative = str(path.relative_to(root))
            if path.is_symlink():
                links[relative] = os.readlink(path)
            elif path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                files[relative] = digest.hexdigest()
    return {"root": str(root.resolve()), "files": files, "symlinks": links}


def _environment(args, root: Path) -> dict:
    env = dict(os.environ)
    # Remove inherited output/input overrides; this command always uses the sealed
    # processed source and fresh scope-specific outputs on the research volume.
    for key in ("DATA_DIR", "RESULTS_DIR", "PAPER_DIR", "LOG_FILE", "SMOKE_ROOT", "INPUT_FILE", "LAB_DATA_FILE",
                "FULL_FEATURES_FILE", "WINDOWED_FEATURES_FILE", "COHORT_FILE", "PREOP_FILE", "PROCESSED_PREOP",
                "FIGURES_DIR", "TABLES_DIR", "REPORTS_DIR", "XAI_DIR", "AKI_SPLIT_CACHE_DIR",
                "N_TRIALS", "THREADS_PER_MODEL", "MAX_WORKERS", "OUTER_FOLDS", "INNER_FOLDS",
                "CALIBRATION_FOLDS", "REPEATS", "COHORT_SCOPE", "VALIDATION_SCHEME", "AKI_TRIAL_HISTORY_DIR",
                "AKI_CONFIG_CPU_RESERVATION", "AKI_EXPECTED_FOLD_HASH", "AKI_REPORT_CPU_BUDGET"):
        env.pop(key, None)
    env.update(AKI_ARTIFACT_ROOT=str(root), PROCESSED_DIR=str(args.baseline_root / "data/processed"),
               RAW_DIR=str(args.raw_dir), AKI_STORAGE_POLICY="enforce", SMOKE_TEST_FLAG="False",
               SAVE_FINAL_REFIT_FLAG="False", RESUME_FLAG="True", PYTHONUNBUFFERED="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    return env


def commands(args) -> list[dict]:
    common = [sys.executable, "-m", "run_catch22", "experiments", "--preset", "core", "--prep", "skip",
              "--validation-scheme", "nested_cv", "--outer-folds", "5", "--inner-folds", "3",
              "--calibration-folds", "5", "--repeats", "1", "--max-workers", "5", "--resume",
              "--bootstrap", "5000", "--bootstrap-seed", "0", "--no-bootstrap-stratified",
              "--parallel-backend", "processes"]
    if args.cpu_budget is not None:
        common += ["--cpu-budget", str(args.cpu_budget)]
    extensions = [*common, "--cohort-scopes", *SCOPES, "--scheduler", "optimized", "--reuse-from", str(args.baseline_root)]
    extensions += ["--admission-policy", getattr(args, "admission_policy", "adaptive")]
    if getattr(args, "max_concurrent_configs", None) is not None:
        extensions += ["--max-concurrent-configs", str(args.max_concurrent_configs)]
    canary = ["--cohort-scope", SCOPES[0], "--model-types", "xgboost", "--branches", "non_windowed",
              "--feature-sets", "preop_and_all_waveforms", "--n-trials", "50", "--blas-threads", "1", "--skip-report"]
    legacy_root = args.study_root / "validation/legacy-canary"
    def config(root):
        return root / "results/catch22/experiments" / SCOPES[0] / "nested_cv/models/xgboost/icu_admission/non_windowed/preop_and_all_waveforms"
    return [
        {"stage": "preflight", "command": [*extensions, "--dry-run"], "env": _environment(args, args.study_root)},
        {"stage": "legacy_canary", "command": [*common, *canary, "--scheduler", "legacy", "--admission-policy", "reserved", "--threads-per-model", "12"], "env": _environment(args, legacy_root)},
        {"stage": "optimized_canary", "command": [*common, *canary, "--scheduler", "optimized", "--admission-policy", "reserved", "--threads-per-model", "4"], "env": _environment(args, args.study_root)},
        {"stage": "equivalence_gate", "command": [sys.executable, "-m", "model_creation.scheduler_canary",
             "--legacy-config", str(config(legacy_root)), "--optimized-config", str(config(args.study_root)),
             "--output", str(args.study_root / "control/canary-agreement.json"), "--bootstrap", "5000", "--bootstrap-seed", "0"],
         "env": _environment(args, args.study_root)},
        {"stage": "extensions", "command": extensions, "env": _environment(args, args.study_root)},
    ]


def _stop(process):
    descendants = []
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        pass
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    # Configuration children have their own sessions. The launcher normally
    # cleans them; retain a bounded fallback if it failed during interruption.
    for child in descendants:
        try:
            if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                child.terminate()
        except psutil.Error:
            pass
    _, remaining = psutil.wait_procs(descendants, timeout=5)
    for child in remaining:
        try:
            child.kill()
        except psutil.Error:
            pass


def _run(command, *, env, cwd, capture=False):
    print(json.dumps({"started_at": _now(), "command": command}), flush=True)
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE if capture else None,
                               text=True, bufsize=1, start_new_session=True)
    output = []
    try:
        if capture:
            for line in process.stdout:
                print(line, end="", flush=True)
                output.append(line)
        code = process.wait()
    except BaseException:
        _stop(process)
        raise
    finally:
        if process.stdout:
            process.stdout.close()
    if code == 75:
        raise DrainedRun("Configuration queue drained with unfinished work")
    if code:
        raise RuntimeError(f"Stage command exited with status {code}")
    return "".join(output)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("An optimized-core supervisor already owns this study") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _executing(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _resource_tracker(command):
    return (len(command) == 3 and command[1] == "-c"
            and command[2].startswith("from multiprocessing.resource_tracker import main;main(")
            and command[2].endswith(")"))


def _wait_terminal(process, *, poll_seconds, timeout=None):
    deadline = None if timeout is None else time.monotonic() + timeout
    while _executing(process):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(f"Process {process.pid} did not finish")
        time.sleep(poll_seconds)


def _drain_legacy_queue(queue, *, capture, verify, progress, poll_seconds=1.0):
    """Suspend admission only; workers remain in their independent sessions."""
    suspended = False
    old_term = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Drain helper interrupted; resuming original queue")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        queue.suspend()
        suspended = True
        deadline = time.monotonic() + 10
        while queue.status() != psutil.STATUS_STOPPED:
            if time.monotonic() >= deadline:
                raise TimeoutError("Queue did not suspend")
            time.sleep(poll_seconds)
        workers = capture(queue)  # Capture AFTER suspension: Popen cannot race us.
        progress("draining", workers)
        while any(_executing(worker) for worker, _ in workers):
            time.sleep(poll_seconds)
        verified = verify(workers)
        progress("workers_verified", workers)
        # Do not stop the supervisor first: its timeout would kill this queue.
        queue.send_signal(signal.SIGTERM)
        queue.resume()
        suspended = False
        _wait_terminal(queue, poll_seconds=poll_seconds, timeout=30)
        return verified
    finally:
        try:
            if suspended and _executing(queue):
                queue.resume()
        finally:
            signal.signal(signal.SIGTERM, old_term)


def drain_current(args, *, poll_seconds=1.0):
    """Finish active configurations, then retire the verified study controller."""
    control = args.study_root / "control"
    status_file = control / "supervisor_status.json"
    record_file = control / "drain-status.json"
    with _lock(control / ".drain-helper.lock"):
        status = json.loads(status_file.read_text())
        if status.get("status") != "running" or status.get("stage") != "extensions":
            raise ValueError("Drain requires a running extensions stage")
        code = (args.study_root / "code").resolve()
        supervisor = psutil.Process(status["pid"])
        started = datetime.fromisoformat(status["started_at"]).timestamp()
        if (not _executing(supervisor) or Path(supervisor.cwd()).resolve() != code
                or not any("run-optimized-core.py" in arg for arg in supervisor.cmdline())
                or not 0 <= started - supervisor.create_time() < 10):
            raise ValueError("Supervisor process identity does not match this study")
        queues = [child for child in supervisor.children()
                  if "run_catch22" in child.cmdline() and "experiments" in child.cmdline()]
        if len(queues) != 1 or Path(queues[0].cwd()).resolve() != code:
            raise ValueError("Expected exactly one study-owned experiment queue")
        queue = queues[0]
        record = {"status": "validating", "started_at": _now(), "helper_pid": os.getpid(),
                  "supervisor_pid": supervisor.pid, "supervisor_created": supervisor.create_time(),
                  "queue_pid": queue.pid, "queue_created": queue.create_time()}
        pointer = control / "active_scheduler.json"
        active = json.loads(pointer.read_text()) if pointer.exists() else {}
        native = active.get("native_drain") and active.get("pid") == queue.pid
        if native and active.get("create_time") != queue.create_time():
            raise ValueError("Native drain queue identity changed")
        if not native:
            snapshot = json.loads((control / "execution-snapshot.json").read_text())
            for name, digest in snapshot["source_sha256"].items():
                if hashlib.sha256((code / name).read_bytes()).hexdigest() != digest:
                    raise ValueError(f"Legacy drain source differs from frozen snapshot: {name}")
        _write(record_file, record)

        def capture(parent):
            workers = []
            for child in parent.children():
                try:
                    command = child.cmdline()
                except psutil.ZombieProcess:
                    saved = json.loads((control / "scheduler/scheduler_summary.json").read_text())
                    previous = next((j for j in saved["configurations"] if j.get("pid") == child.pid), None)
                    if previous is None:
                        raise ValueError(f"Unknown completed queue child {child.pid}")
                    fields = dict(zip(("cohort-scope", "model_type", "outcome", "branch", "feature_set"), previous["key"].split("/")))
                    command = ["model_creation.step_07_train_evaluate", *[v for k, value in fields.items() for v in ("--" + k, value)]]
                if _resource_tracker(command):
                    continue  # Split-cache multiprocessing helper; not a training job.
                if "model_creation.step_07_train_evaluate" not in command:
                    raise ValueError(f"Unexpected queue child {child.pid}; refusing drain")
                workers.append((child, {"pid": child.pid, "created": child.create_time(), "command": command}))
            return workers

        def progress(phase, workers):
            record.update(status=phase, updated_at=_now(), workers=[row for _, row in workers])
            _write(record_file, record)
            print(json.dumps({"drain": phase, "workers": [p.pid for p, _ in workers]}), flush=True)

        def verify(workers):
            from model_creation.run_inventory import verify_completed_configuration
            snapshot = json.loads((control / "execution-snapshot.json").read_text())
            verified = []
            for _, row in workers:
                command = row["command"]
                fields = {name: command[command.index("--" + name) + 1]
                          for name in ("cohort-scope", "model_type", "outcome", "branch", "feature_set")}
                if fields["cohort-scope"] not in SCOPES:
                    raise ValueError("Drain worker is outside the approved scopes")
                directory = (args.study_root / "results/catch22/experiments" / fields["cohort-scope"] / "nested_cv/models"
                             / fields["model_type"] / fields["outcome"] / fields["branch"] / fields["feature_set"])
                suffix = "_windowed" if fields["branch"] == "windowed" else ""
                result = verify_completed_configuration(directory, dataset_path=args.baseline_root / "data/processed" / f"aki_features_master_wide{suffix}.csv")
                if any(result["validation_fingerprint"].get(k) != v for k, v in snapshot["study_identity"].items()):
                    raise ValueError("Drained worker scientific identity changed")
                verified.append({"key": result["key"], "fold_count": result["fold_count"], "configuration_dir": str(directory)})
            return verified

        try:
            if native:
                _write(control / "drain-request.json", {"pid": queue.pid, "create_time": queue.create_time(), "requested_at": _now()})
                record.update(status="draining", method="native")
                _write(record_file, record)
                _wait_terminal(queue, poll_seconds=poll_seconds)
                record["verified_configurations"] = []  # Native completion callbacks verify each job.
            else:
                record.update(method="legacy-suspend", profiling_note="Queue process-tree sampling pauses during drain; exclude this gap from continuous utilization estimates")
                record["verified_configurations"] = _drain_legacy_queue(queue, capture=capture, verify=verify,
                                                                        progress=progress, poll_seconds=poll_seconds)
            _wait_terminal(supervisor, poll_seconds=poll_seconds, timeout=120)
            for name in (".controller.lock", ".supervisor.lock"):
                path = control / name
                if path.exists():
                    with path.open() as handle:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        fcntl.flock(handle, fcntl.LOCK_UN)
            record.update(status="drained", finished_at=_now())
            _write(record_file, record)
            print(json.dumps({"drain": "drained", "record": str(record_file)}), flush=True)
            return 0
        except BaseException as error:
            record.update(status="failed", error=str(error), finished_at=_now())
            _write(record_file, record)
            raise


def run_pipeline(args, *, runner=_run):
    baseline, study = args.baseline_root.resolve(), args.study_root.resolve()
    if baseline == study or baseline in study.parents or study in baseline.parents:
        raise ValueError("New study and protected baseline must be separate directory trees")
    control = study / "control"
    status_file = control / "supervisor_status.json"
    agreement_file = control / "canary-agreement.json"
    status = {"status": "running", "stage": "initializing", "pid": os.getpid(), "started_at": _now()}
    old_handlers = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Supervisor received signal {signum}")
    with _lock(control / ".supervisor.lock"):
        if agreement_file.exists() and json.loads(agreement_file.read_text()).get("status") != "passed":
            raise ValueError("A previous canary gate failed; investigate before any further full-budget execution")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                # nohup deliberately ignores HUP; foreground runs instead need
                # the same owned-process cleanup as an explicit interruption.
                if signum == signal.SIGHUP and signal.getsignal(signum) == signal.SIG_IGN:
                    continue
                old_handlers[signum] = signal.signal(signum, interrupted)
            snapshot = _execution_snapshot()
            overlay = _validate_execution_snapshot(control, snapshot)
            epoch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            epoch_file = control / "execution-epochs" / f"{epoch}.json"
            _write(epoch_file, {**snapshot, "orchestration_overlay": overlay,
                                "admission_policy": getattr(args, "admission_policy", "adaptive"),
                                "max_concurrent_configs": getattr(args, "max_concurrent_configs", None) or 12})
            _write(control / "active_execution_epoch.json", {"path": str(epoch_file), "epoch": epoch})
            saved_canary = agreement_file.exists()
            if saved_canary:
                _verify_saved_canary(args, json.loads(agreement_file.read_text()))
            status.update(stage="baseline_inventory", updated_at=_now())
            _write(status_file, status)
            observed = inventory(baseline)
            inventory_file = control / "baseline-inventory.json"
            before = json.loads(inventory_file.read_text()) if inventory_file.exists() else observed
            if observed != before:
                raise ValueError("Protected study differs from the previously sealed inventory")
            _write(inventory_file, before)
            for stage in commands(args):
                if saved_canary and stage["stage"] in {"legacy_canary", "optimized_canary", "equivalence_gate"}:
                    continue
                status.update(stage=stage["stage"], updated_at=_now())
                _write(status_file, status)
                output = runner(stage["command"], env=stage["env"], cwd=PROJECT_ROOT, capture=stage["stage"] == "preflight")
                if stage["stage"] == "preflight":
                    plan = json.loads(output)
                    # A resumed supervisor may already have the accepted canary
                    # or more extension configurations. Never retrain them.
                    completed = plan["requested_local_completed_configurations"]
                    if (plan["protected_completed_configurations"] != 36 or len(plan["configurations"]) != 36
                            or plan["pending_configurations"] + completed != 36
                            or plan["requested_reused_configurations"] != 0
                            or set(plan["cohort_scopes"]) != set(SCOPES)
                            or any(item["feature_set"] not in {"preop_only", "all_waveforms", "preop_and_all_waveforms"}
                                   for item in plan["configurations"])):
                        raise ValueError("Preflight does not match the approved 36 protected /36 new core plan")
                    _write(control / "preflight.json", plan)
                elif stage["stage"] == "equivalence_gate":
                    if not agreement_file.exists() or json.loads(agreement_file.read_text()).get("status") != "passed":
                        raise ValueError("Canary did not produce a passed agreement gate")
            status.update(stage="zero_fit_relaunch", updated_at=_now())
            _write(status_file, status)
            extension = commands(args)[-1]
            relaunch = json.loads(runner([*extension["command"], "--dry-run"], env=extension["env"], cwd=PROJECT_ROOT, capture=True))
            _write(control / "zero-fit-relaunch.json", relaunch)
            if (relaunch.get("pending_configurations") != 0
                    or relaunch.get("requested_local_completed_configurations") != 36
                    or relaunch.get("protected_completed_configurations") != 36
                    or len(relaunch.get("configurations", [])) != 36
                    or set(relaunch.get("cohort_scopes", [])) != set(SCOPES)):
                raise ValueError("Completed extensions would schedule fits on relaunch")
            status.update(stage="final_audit", updated_at=_now())
            _write(status_file, status)
            for scope in SCOPES:
                path = study / "results/catch22/experiments" / scope / "nested_cv/run_manifest.json"
                manifest = json.loads(path.read_text())
                if manifest.get("status") != "validated" or len(manifest["configurations"]) != 18:
                    raise ValueError(f"New scope is not complete and report-validated: {scope}")
                bundles = list(path.parent.glob("models/**/artifacts/folds/repeat_*/outer_*/metadata.json"))
                if len(bundles) != 90:
                    raise ValueError(f"New scope does not contain exactly 90 outer-fold bundles: {scope}")
            after = inventory(baseline)
            unchanged = before == after
            _write(control / "artifact-preservation.json", {"unchanged": unchanged, "baseline_files": len(before["files"]),
                                                           "checked_at": _now()})
            if not unchanged:
                raise ValueError("Protected study changed during execution")
            status.update(status="completed", stage="completed", finished_at=_now(), configurations=36, outer_fold_bundles=180)
            _write(status_file, status)
        except DrainedRun as error:
            status.update(status="drained", error=str(error), finished_at=_now())
            _write(status_file, status)
            return 75
        except BaseException as error:
            status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error), finished_at=_now())
            _write(status_file, status)
            # This read-only check also runs on failed canaries and interruptions.
            if "before" in locals():
                after = inventory(baseline)
                _write(control / "artifact-preservation.json", {"unchanged": before == after, "baseline_files": len(before["files"]), "checked_at": _now()})
            raise
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=STUDIES / "manuscript-core-v1")
    parser.add_argument("--study-root", type=Path, default=STUDIES / "manuscript-optimized-v1")
    parser.add_argument("--raw-dir", type=Path, default=Path("/media/volume/catch22/projects/aki_prediction_project/data/raw"))
    parser.add_argument("--cpu-budget", type=int)
    parser.add_argument("--admission-policy", choices=("adaptive", "reserved"), default="adaptive")
    parser.add_argument("--max-concurrent-configs", type=int)
    parser.add_argument("--drain-current", action="store_true", help="Finish active configurations and stop this study's queue")
    parser.add_argument("--print-commands", action="store_true", help="Read-only command inspection; does not create directories")
    args = parser.parse_args(argv)
    for name in ("baseline_root", "study_root", "raw_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.drain_current and args.print_commands:
        parser.error("--drain-current and --print-commands are mutually exclusive")
    if args.drain_current:
        return drain_current(args)
    if args.print_commands:
        print(json.dumps([{key: value for key, value in stage.items() if key != "env"} | {
            "paths": {name: stage["env"][name] for name in ("AKI_ARTIFACT_ROOT", "PROCESSED_DIR", "RAW_DIR")}}
            for stage in commands(args)], indent=2))
        return 0
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
