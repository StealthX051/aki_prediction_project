from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run_catch22


def test_experiments_fail_fast_when_generated_paths_are_off_media(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "experiment.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "enforce")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_catch22,
        "run_module",
        lambda *args, **kwargs: pytest.fail("run_module should not be called when storage policy fails"),
    )

    result = run_catch22.main(["experiments", "--prep", "skip", "--only-xgboost", "--no-resume"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Generated artifact paths must live under /media/volume/catch22" in captured.err
    assert "Starting Experiments" not in captured.out


def test_experiments_skip_reporting_on_validation_failures(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "experiment.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)

    calls: list[tuple[str, list[str]]] = []

    def fake_run_module(module: str, args: list[str], *, env, log_handle) -> int:
        calls.append((module, list(args)))
        if module == "data_preparation.validate_processed_artifacts":
            return 0
        if module == "model_creation.step_07_train_evaluate":
            return 1
        pytest.fail(f"Unexpected module call: {module}")

    monkeypatch.setattr(run_catch22, "run_module", fake_run_module)

    result = run_catch22.main(["experiments", "--prep", "skip", "--only-xgboost", "--no-resume"])

    captured = capsys.readouterr()
    assert result == 1
    assert any(module == "data_preparation.validate_processed_artifacts" for module, _ in calls)
    assert any(module == "model_creation.step_07_train_evaluate" for module, _ in calls)
    assert "skipping metrics/report generation" in captured.out
    assert "FAILED: model=xgboost" in captured.out
    assert "Running evaluation (results -> results directory)" not in captured.out


def test_smoke_uses_isolated_root_and_selected_models(tmp_path: Path, monkeypatch):
    smoke_root = tmp_path / "smoke"
    monkeypatch.setenv("SMOKE_ROOT", str(smoke_root))
    monkeypatch.setenv("LOG_FILE", str(smoke_root / "smoke_test.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)

    calls: list[tuple[str, list[str], str, str]] = []

    def fake_run_module(module: str, args: list[str], *, env, log_handle) -> int:
        calls.append((module, list(args), env["PROCESSED_DIR"], env["RESULTS_DIR"]))
        return 0

    monkeypatch.setattr(run_catch22, "run_module", fake_run_module)

    result = run_catch22.main(
        [
            "smoke",
            "--outcomes",
            "any_aki",
            "--model-types",
            "xgboost,ebm",
            "--case-limit",
            "6",
            "--hpo-trials",
            "1",
        ]
    )

    assert result == 0
    trim_calls = [entry for entry in calls if entry[0] == "data_preparation.smoke_trim_cohort"]
    assert len(trim_calls) == 1
    trim_module, trim_args, processed_dir, results_dir = trim_calls[0]
    assert trim_module == "data_preparation.smoke_trim_cohort"
    assert str(smoke_root / "data" / "processed" / "aki_pleth_ecg_co2_awp.csv") in trim_args
    assert processed_dir == str(smoke_root / "data" / "processed")
    assert results_dir == str(smoke_root / "results" / "catch22" / "experiments")

    validation_calls = [entry for entry in calls if entry[0] == "model_creation.step_07_train_evaluate"]
    assert len(validation_calls) == 2
    assert {call[1][call[1].index("--model_type") + 1] for call in validation_calls} == {"xgboost", "ebm"}
    assert all(call[2] == str(smoke_root / "data" / "processed") for call in validation_calls)
    assert all(call[3] == str(smoke_root / "results" / "catch22" / "experiments") for call in validation_calls)


def test_descriptive_fails_clearly_on_missing_inputs(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "data" / "processed"))
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results" / "catch22" / "experiments"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "results" / "catch22" / "paper"))
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "descriptive.log"))
    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    monkeypatch.setattr(run_catch22, "refresh_repo_convenience_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        run_catch22,
        "run_module",
        lambda *args, **kwargs: pytest.fail("run_module should not be called when required inputs are missing"),
    )

    result = run_catch22.main(["descriptive"])

    captured = capsys.readouterr()
    assert result == 1
    assert "Required file not found" in captured.out
