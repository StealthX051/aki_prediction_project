from pathlib import Path

import pytest

from artifact_paths import (
    DEFAULT_ARTIFACT_ROOT,
    StoragePolicyError,
    build_catch22_layout,
    enforce_storage_policy,
    refresh_repo_convenience_paths,
    resolve_log_file,
    get_scoped_results_dir,
    get_scoped_paper_dir,
    ensure_repo_symlink,
)


@pytest.mark.parametrize("old_target_kind", ["empty", "nonempty", "missing"])
def test_refresh_existing_symlink_preserves_old_target(tmp_path, old_target_kind):
    old_target, new_target, link = tmp_path / "old", tmp_path / "new", tmp_path / "link"
    if old_target_kind != "missing":
        old_target.mkdir()
    if old_target_kind == "nonempty":
        (old_target / "keep.txt").write_text("keep")
    link.symlink_to(old_target)
    returned = ensure_repo_symlink(link, new_target)
    assert returned == link
    assert link.is_symlink() and link.resolve(strict=False) == new_target
    assert not old_target.is_symlink()
    assert old_target.exists() == (old_target_kind != "missing")
    if old_target_kind == "nonempty":
        assert (old_target / "keep.txt").read_text() == "keep"


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
        enforce_storage_policy(candidate, allowed_root=tmp_path / "allowed-volume")

    monkeypatch.setenv("AKI_STORAGE_POLICY", "warn")
    with pytest.warns(RuntimeWarning):
        enforce_storage_policy(candidate, allowed_root=tmp_path / "allowed-volume")

    monkeypatch.setenv("AKI_STORAGE_POLICY", "off")
    enforce_storage_policy(candidate, allowed_root=tmp_path / "allowed-volume")


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


def test_scopes_and_validation_roots_are_isolated_and_idempotent(tmp_path, monkeypatch):
    project = tmp_path / "repo"
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("PAPER_DIR", str(tmp_path / "paper"))
    main = get_scoped_results_dir(project, "main_shared", "nested_cv")
    sensitivity = get_scoped_results_dir(project, "icu_waveform_sensitivity", "nested_cv")
    holdout = get_scoped_results_dir(project, "main_shared", "holdout")
    assert len({main, sensitivity, holdout}) == 3
    monkeypatch.setenv("RESULTS_DIR", str(main))
    assert get_scoped_results_dir(project, "main_shared", "nested_cv") == main
    with pytest.raises(ValueError, match="another cohort"):
        get_scoped_results_dir(project, "icu_waveform_sensitivity", "nested_cv")
    assert get_scoped_paper_dir(project, "main_shared", "nested_cv") == tmp_path / "paper/main_shared/nested_cv"
