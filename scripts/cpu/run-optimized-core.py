#!/usr/bin/env python3
"""Supervise the approved full-budget canary gate, then ICU/mortality core runs."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
import platform
from pathlib import Path
import signal
import subprocess
import sys

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDIES = Path("/media/volume/catch22/data/aki_prediction_project/studies")
SCOPES = ("icu_waveform_sensitivity", "mortality_waveform_sensitivity")


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
            "source_sha256": {name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() for name in names}}


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
    common = [args.python, "-m", "run_catch22", "experiments", "--preset", "core", "--prep", "skip",
              "--validation-scheme", "nested_cv", "--outer-folds", "5", "--inner-folds", "3",
              "--calibration-folds", "5", "--repeats", "1", "--max-workers", "5", "--resume",
              "--bootstrap", "5000", "--bootstrap-seed", "0", "--no-bootstrap-stratified"]
    if args.cpu_budget is not None:
        common += ["--cpu-budget", str(args.cpu_budget)]
    extensions = [*common, "--cohort-scopes", *SCOPES, "--scheduler", "optimized", "--reuse-from", str(args.baseline_root)]
    canary = ["--cohort-scope", SCOPES[0], "--model-types", "xgboost", "--branches", "non_windowed",
              "--feature-sets", "preop_and_all_waveforms", "--n-trials", "50", "--blas-threads", "1", "--skip-report"]
    legacy_root = args.study_root / "validation/legacy-canary"
    def config(root):
        return root / "results/catch22/experiments" / SCOPES[0] / "nested_cv/models/xgboost/icu_admission/non_windowed/preop_and_all_waveforms"
    return [
        {"stage": "preflight", "command": [*extensions, "--dry-run"], "env": _environment(args, args.study_root)},
        {"stage": "legacy_canary", "command": [*common, *canary, "--scheduler", "legacy", "--threads-per-model", "12"], "env": _environment(args, legacy_root)},
        {"stage": "optimized_canary", "command": [*common, *canary, "--scheduler", "optimized", "--threads-per-model", "4"], "env": _environment(args, args.study_root)},
        {"stage": "equivalence_gate", "command": [args.python, "-m", "model_creation.scheduler_canary",
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
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, interrupted)
            snapshot = _execution_snapshot()
            snapshot_file = control / "execution-snapshot.json"
            if snapshot_file.exists():
                previous_snapshot = json.loads(snapshot_file.read_text())
                if any(snapshot[key] != previous_snapshot.get(key) for key in ("study_identity", "source_sha256", "python")):
                    raise ValueError("Supervisor source/environment changed since its frozen execution snapshot")
            else:
                _write(snapshot_file, snapshot)
            status.update(stage="baseline_inventory", updated_at=_now())
            _write(status_file, status)
            observed = inventory(baseline)
            inventory_file = control / "baseline-inventory.json"
            before = json.loads(inventory_file.read_text()) if inventory_file.exists() else observed
            if observed != before:
                raise ValueError("Protected study differs from the previously sealed inventory")
            _write(inventory_file, before)
            for stage in commands(args):
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cpu-budget", type=int)
    parser.add_argument("--print-commands", action="store_true", help="Read-only command inspection; does not create directories")
    args = parser.parse_args(argv)
    for name in ("baseline_root", "study_root", "raw_dir"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.print_commands:
        print(json.dumps([{key: value for key, value in stage.items() if key != "env"} | {
            "paths": {name: stage["env"][name] for name in ("AKI_ARTIFACT_ROOT", "PROCESSED_DIR", "RAW_DIR")}}
            for stage in commands(args)], indent=2))
        return 0
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
