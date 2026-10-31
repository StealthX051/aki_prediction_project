"""Supervisor plan gates and protected-study integrity, without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import pytest

SUPERVISOR = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/cpu/run-optimized-core.py"))


def arguments(tmp_path):
    baseline = tmp_path / "old"
    baseline.mkdir()
    (baseline / "original.csv").write_text("original data")
    return argparse.Namespace(baseline_root=baseline, study_root=tmp_path / "new", raw_dir=tmp_path / "raw",
                              python=sys.executable, cpu_budget=60)


def preflight():
    return {"protected_completed_configurations": 36, "requested_local_completed_configurations": 0,
            "pending_configurations": 36, "requested_reused_configurations": 0,
            "cohort_scopes": list(SUPERVISOR["SCOPES"]),
            "configurations": [{"feature_set": feature} for feature in ["preop_only", "all_waveforms", "preop_and_all_waveforms"] * 12]}


def finalize(args):
    root = args.study_root / "results/catch22/experiments"
    for scope in SUPERVISOR["SCOPES"]:
        result = root / scope / "nested_cv"
        result.mkdir(parents=True, exist_ok=True)
        (result / "run_manifest.json").write_text(json.dumps({"status": "validated", "configurations": [{}] * 18}))
        for config in range(18):
            for fold in range(5):
                bundle = result / "models" / str(config) / "artifacts/folds/repeat_00" / f"outer_{fold:02d}"
                bundle.mkdir(parents=True)
                (bundle / "metadata.json").write_text("{}")


def test_commands_pin_scientific_budget_and_isolate_outputs(tmp_path, monkeypatch):
    args = arguments(tmp_path)
    monkeypatch.setenv("SAVE_FINAL_REFIT_FLAG", "True")
    monkeypatch.setenv("SMOKE_TEST_FLAG", "True")
    monkeypatch.setenv("RESULTS_DIR", str(args.baseline_root))
    monkeypatch.setenv("N_TRIALS", "1")
    stages = SUPERVISOR["commands"](args)
    assert [s["stage"] for s in stages] == ["preflight", "legacy_canary", "optimized_canary", "equivalence_gate", "extensions"]
    for stage in stages:
        env = stage["env"]
        assert env["PROCESSED_DIR"] == str(args.baseline_root / "data/processed")
        assert env["RAW_DIR"] == str(args.raw_dir)
        assert env["SAVE_FINAL_REFIT_FLAG"] == "False"
        assert env["SMOKE_TEST_FLAG"] == "False"
        assert "RESULTS_DIR" not in env and "N_TRIALS" not in env
        if stage["stage"] != "equivalence_gate":
            command = stage["command"]
            assert command[command.index("--outer-folds") + 1] == "5"
            assert command[command.index("--inner-folds") + 1] == "3"
            assert command[command.index("--calibration-folds") + 1] == "5"
            assert "--diagnostic" not in command and "--save-final-refit" not in command
    assert stages[1]["env"]["AKI_ARTIFACT_ROOT"] == str(args.study_root / "validation/legacy-canary")
    assert stages[2]["env"]["AKI_ARTIFACT_ROOT"] == str(args.study_root)
    assert stages[1]["command"][stages[1]["command"].index("--threads-per-model") + 1] == "12"
    assert stages[2]["command"][stages[2]["command"].index("--threads-per-model") + 1] == "4"
    assert "--skip-report" not in stages[-1]["command"]
    assert "--preset" in stages[-1]["command"] and "full" not in stages[-1]["command"]


def test_sequence_gate_and_final_integrity(tmp_path):
    args = arguments(tmp_path)
    visited = []

    def runner(command, **kwargs):
        visited.append(command)
        if "--dry-run" in command:
            plan = preflight()
            if len(visited) > 1:
                plan.update(pending_configurations=0, requested_local_completed_configurations=36)
            return json.dumps(plan)
        if "model_creation.scheduler_canary" in command:
            SUPERVISOR["_write"](args.study_root / "control/canary-agreement.json", {"status": "passed"})
        elif "--cohort-scopes" in command:
            finalize(args)
        return ""

    assert SUPERVISOR["run_pipeline"](args, runner=runner) == 0
    assert len(visited) == 6
    relaunch = json.loads((args.study_root / "control/zero-fit-relaunch.json").read_text())
    assert relaunch["pending_configurations"] == 0 and relaunch["requested_local_completed_configurations"] == 36
    snapshot = json.loads((args.study_root / "control/execution-snapshot.json").read_text())
    assert snapshot["study_identity"]["code_revision"]
    assert "model_creation/config_scheduler.py" in snapshot["source_sha256"]
    status = json.loads((args.study_root / "control/supervisor_status.json").read_text())
    assert status["status"] == "completed" and status["outer_fold_bundles"] == 180
    assert json.loads((args.study_root / "control/artifact-preservation.json").read_text())["unchanged"]
    assert (args.baseline_root / "original.csv").read_text() == "original data"


def test_preflight_mismatch_stops_before_any_training(tmp_path):
    args = arguments(tmp_path)
    commands = []
    def runner(command, **kwargs):
        commands.append(command)
        plan = preflight()
        plan["protected_completed_configurations"] = 35
        return json.dumps(plan)
    with pytest.raises(ValueError, match="Preflight"):
        SUPERVISOR["run_pipeline"](args, runner=runner)
    assert len(commands) == 1 and "--dry-run" in commands[0]


def test_failed_canary_is_sticky_and_never_launches_extensions(tmp_path):
    args = arguments(tmp_path)
    visited = []
    def runner(command, **kwargs):
        visited.append(command)
        if "--dry-run" in command:
            return json.dumps(preflight())
        if "model_creation.scheduler_canary" in command:
            SUPERVISOR["_write"](args.study_root / "control/canary-agreement.json", {"status": "failed"})
            raise RuntimeError("gate mismatch")
        return ""
    with pytest.raises(RuntimeError, match="gate mismatch"):
        SUPERVISOR["run_pipeline"](args, runner=runner)
    assert len(visited) == 4
    status = json.loads((args.study_root / "control/supervisor_status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "equivalence_gate"
    with pytest.raises(ValueError, match="previous canary gate failed"):
        SUPERVISOR["run_pipeline"](args, runner=lambda *a, **kw: pytest.fail("No further commands allowed"))


def test_protected_changes_are_detected_even_when_stage_fails(tmp_path):
    args = arguments(tmp_path)
    def runner(command, **kwargs):
        (args.baseline_root / "original.csv").write_text("changed")
        raise RuntimeError("unexpected writer")
    with pytest.raises(RuntimeError, match="unexpected writer"):
        SUPERVISOR["run_pipeline"](args, runner=runner)
    assert not json.loads((args.study_root / "control/artifact-preservation.json").read_text())["unchanged"]
    with pytest.raises(ValueError, match="previously sealed inventory"):
        SUPERVISOR["run_pipeline"](args, runner=lambda *a, **kw: pytest.fail("No launch"))
    assert not json.loads((args.study_root / "control/artifact-preservation.json").read_text())["unchanged"]


def test_print_commands_is_read_only_and_supervisor_lock_is_exclusive(tmp_path, capsys):
    args = arguments(tmp_path)
    SUPERVISOR["main"](["--baseline-root", str(args.baseline_root), "--study-root", str(args.study_root), "--print-commands"])
    assert not args.study_root.exists()
    assert len(json.loads(capsys.readouterr().out)) == 5
    with SUPERVISOR["_lock"](args.study_root / "control/.supervisor.lock"):
        with pytest.raises(ValueError, match="already owns"):
            SUPERVISOR["run_pipeline"](args, runner=lambda *a, **kw: pytest.fail("Duplicate launch"))


def test_relaunch_that_would_refit_is_rejected(tmp_path):
    args = arguments(tmp_path)
    visited = []
    def runner(command, **kwargs):
        visited.append(command)
        if "--dry-run" in command:
            # Deliberately claim one pending configuration after all reports.
            plan = preflight()
            if len(visited) > 1:
                plan.update(pending_configurations=1, requested_local_completed_configurations=35)
            return json.dumps(plan)
        if "model_creation.scheduler_canary" in command:
            SUPERVISOR["_write"](args.study_root / "control/canary-agreement.json", {"status": "passed"})
        elif "--cohort-scopes" in command:
            finalize(args)
        return ""
    with pytest.raises(ValueError, match="would schedule fits"):
        SUPERVISOR["run_pipeline"](args, runner=runner)
    assert len(visited) == 6 and "--dry-run" in visited[-1]
    status = json.loads((args.study_root / "control/supervisor_status.json").read_text())
    assert status["status"] == "failed" and status["stage"] == "zero_fit_relaunch"
