from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifact_paths import (
    DEFAULT_ARTIFACT_ROOT,
    StoragePolicyError,
    build_catch22_layout,
    enforce_storage_policy,
    refresh_repo_convenience_paths,
    resolve_log_file,
)


def test_build_catch22_layout_defaults_to_media_root(monkeypatch):
    for key in (
        "AKI_ARTIFACT_ROOT",
        "AKI_STORAGE_POLICY",
        "DATA_DIR",
        "RAW_DIR",
        "PROCESSED_DIR",
        "RESULTS_DIR",
        "PAPER_DIR",
        "SMOKE_ROOT",
        "LOG_FILE",
    ):
        monkeypatch.delenv(key, raising=False)

    project_root = Path("/repo/aki_prediction_project")
    layout = build_catch22_layout(project_root, default_log_name="experiment_log.txt")

    assert layout.artifact_root == DEFAULT_ARTIFACT_ROOT
    assert layout.processed_dir == DEFAULT_ARTIFACT_ROOT / "data" / "processed"
    assert layout.results_dir == DEFAULT_ARTIFACT_ROOT / "results" / "catch22" / "experiments"
    assert layout.paper_dir == DEFAULT_ARTIFACT_ROOT / "results" / "catch22" / "paper"
    assert layout.smoke_root == DEFAULT_ARTIFACT_ROOT / "smoke_test_outputs"
    assert layout.log_file == DEFAULT_ARTIFACT_ROOT / "logs" / "experiment_log.txt"
    assert layout.raw_dir == project_root / "data" / "raw"


def test_enforce_storage_policy_modes(tmp_path: Path, monkeypatch):
    candidate = {"results_dir": tmp_path / "results"}

    monkeypatch.setenv("AKI_STORAGE_POLICY", "enforce")
    with pytest.raises(StoragePolicyError):
        enforce_storage_policy(candidate)

    monkeypatch.setenv("AKI_STORAGE_POLICY", "warn")
    with pytest.warns(RuntimeWarning):
        enforce_storage_policy(candidate)

    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    enforce_storage_policy(candidate)


def test_resolve_log_file_can_target_custom_default_dir(monkeypatch):
    monkeypatch.delenv("LOG_FILE", raising=False)

    resolved = resolve_log_file("smoke_test.log", default_dir=Path("/tmp/aki_smoke"))

    assert resolved == Path("/tmp/aki_smoke/smoke_test.log")


def test_refresh_repo_convenience_paths_creates_symlinks(tmp_path: Path):
    project_root = tmp_path / "repo"
    processed_dir = tmp_path / "artifacts" / "data" / "processed"
    results_dir = tmp_path / "artifacts" / "results" / "catch22" / "experiments"
    paper_dir = tmp_path / "artifacts" / "results" / "catch22" / "paper"

    refreshed = refresh_repo_convenience_paths(
        project_root,
        processed_dir=processed_dir,
        results_dir=results_dir,
        paper_dir=paper_dir,
    )

    processed_link = project_root / "data" / "processed"
    results_link = project_root / "results" / "catch22" / "experiments"
    paper_link = project_root / "results" / "catch22" / "paper"

    assert refreshed["processed_dir"] == processed_dir.resolve(strict=False)
    assert processed_link.is_symlink()
    assert results_link.is_symlink()
    assert paper_link.is_symlink()
    assert processed_link.resolve(strict=False) == processed_dir.resolve(strict=False)
    assert results_link.resolve(strict=False) == results_dir.resolve(strict=False)
    assert paper_link.resolve(strict=False) == paper_dir.resolve(strict=False)
