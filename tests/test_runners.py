import os
import subprocess
from pathlib import Path


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

    env = os.environ.copy()
    env.update(
        {
            "RESULTS_DIR": str(results_dir),
            "PAPER_DIR": str(paper_dir),
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
    assert "Python runner: env PYTHONUNBUFFERED=1 false" in combined_output
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
