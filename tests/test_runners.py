import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_run_experiments_exits_nonzero_and_skips_reporting_on_validation_failures(tmp_path: Path):
    results_dir = tmp_path / "results" / "catch22" / "experiments"
    paper_dir = tmp_path / "results" / "catch22" / "paper"
    stale_predictions_dir = (
        results_dir
        / "models"
        / "xgboost"
        / "any_aki"
        / "windowed"
        / "preop_only"
        / "predictions"
    )
    stale_predictions_dir.mkdir(parents=True, exist_ok=True)
    (stale_predictions_dir / "test.csv").write_text("caseid,y_true,y_prob_calibrated\n1,0,0.5\n")

    proxy = tmp_path / "python_proxy.sh"
    proxy.write_text(
        "#!/bin/bash\n"
        "if [[ \"$1\" == \"-m\" && \"$2\" == \"data_preparation.validate_processed_artifacts\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    proxy.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "RESULTS_DIR": str(results_dir),
            "PAPER_DIR": str(paper_dir),
            "LOG_FILE": str(tmp_path / "experiment.log"),
            "AKI_STORAGE_POLICY": "off",
            "PYTHON_BIN": str(proxy),
        }
    )

    proc = subprocess.run(
        ["bash", "run_experiments.sh", "--prep", "skip", "--only-xgboost", "--no-resume"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 1
    assert f"Python runner: {proxy}" in combined_output
    assert "skipping metrics/report generation" in combined_output
    assert "FAILED: model=xgboost" in combined_output
    assert "Running evaluation (results -> results directory)" not in combined_output
    assert "command not found" not in combined_output
    assert not (paper_dir / "tables" / "metrics_summary.csv").exists()


def test_run_smoke_test_accepts_multiword_python_bin_prefix(tmp_path: Path):
    smoke_root = tmp_path / "smoke"
    env = os.environ.copy()
    env.update(
        {
            "SMOKE_ROOT": str(smoke_root),
            "AKI_STORAGE_POLICY": "off",
            "PYTHON_BIN": "env PYTHONUNBUFFERED=1 false",
        }
    )

    proc = subprocess.run(
        ["bash", "run_smoke_test.sh"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode != 0
    assert "Python runner: env PYTHONUNBUFFERED=1 false" in combined_output
    assert "command not found" not in combined_output
    assert "--- Step 01: Cohort construction ---" in combined_output


def test_run_experiments_fails_fast_when_generated_paths_are_off_media(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "RESULTS_DIR": str(tmp_path / "results" / "catch22" / "experiments"),
            "PAPER_DIR": str(tmp_path / "results" / "catch22" / "paper"),
            "PROCESSED_DIR": str(tmp_path / "data" / "processed"),
            "LOG_FILE": str(tmp_path / "experiment.log"),
            "PYTHON_BIN": "env PYTHONUNBUFFERED=1 false",
        }
    )

    proc = subprocess.run(
        ["bash", "run_experiments.sh", "--prep", "skip", "--only-xgboost", "--no-resume"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 1
    assert "Generated artifact path must live under /media/volume/catch22" in combined_output
    assert "Starting Experiments" not in combined_output


def test_run_experiments_fails_fast_on_stale_processed_artifacts(tmp_path: Path):
    processed_dir = tmp_path / "data" / "processed"
    results_dir = tmp_path / "results" / "catch22" / "experiments"
    paper_dir = tmp_path / "results" / "catch22" / "paper"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name in [
        "aki_preop_processed.csv",
        "aki_features_master_wide.csv",
        "aki_features_master_wide_windowed.csv",
    ]:
        (processed_dir / name).write_text("caseid\n1\n")

    env = os.environ.copy()
    env.update(
        {
            "PROCESSED_DIR": str(processed_dir),
            "RESULTS_DIR": str(results_dir),
            "PAPER_DIR": str(paper_dir),
            "LOG_FILE": str(tmp_path / "experiment.log"),
            "AKI_STORAGE_POLICY": "off",
            "PYTHON_BIN": sys.executable,
        }
    )

    proc = subprocess.run(
        ["bash", "run_experiments.sh", "--prep", "skip", "--only-xgboost", "--no-resume"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 1
    assert "failed schema validation" in combined_output
    assert "metadata sidecar is missing" in combined_output
    assert "Validation run complete" not in combined_output


def test_smoke_trim_module_runs_under_conda_invocation_when_available(tmp_path: Path):
    conda_bin = shutil.which("conda")
    if conda_bin is None:
        return

    cohort_path = tmp_path / "cohort.csv"
    pd.DataFrame(
        {
            "caseid": list(range(1, 13)),
            "subjectid": list(range(101, 113)),
            "aki_label": [0, 1] * 6,
            "y_icu_admit": [1, 0] * 6,
            "eligible_any_aki": [1] * 12,
            "eligible_icu_admission": [1] * 12,
        }
    ).to_csv(cohort_path, index=False)

    proc = subprocess.run(
        [
            conda_bin,
            "run",
            "-n",
            "aki_prediction_project",
            "python",
            "-m",
            "data_preparation.smoke_trim_cohort",
            "--cohort-path",
            str(cohort_path),
            "--limit",
            "6",
            "--outcomes",
            "any_aki,icu_admission",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )

    combined_output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0 and (
        "EnvironmentLocationNotFound" in combined_output
        or "Could not find conda environment" in combined_output
        or "Not a conda environment" in combined_output
    ):
        return

    assert proc.returncode == 0
    trimmed = pd.read_csv(cohort_path)
    assert len(trimmed) == 8
    assert "Cohort trimmed to 8 rows" in combined_output
