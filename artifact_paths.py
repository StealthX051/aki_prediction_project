from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

DEFAULT_MEDIA_ROOT = Path("/media/volume/catch22")
DEFAULT_ARTIFACT_ROOT = DEFAULT_MEDIA_ROOT / "data" / "aki_prediction_project"
DEFAULT_STORAGE_POLICY = "enforce"
VALID_STORAGE_POLICIES = {"enforce", "warn", "off"}


class StoragePolicyError(RuntimeError):
    """Raised when a generated artifact path violates the storage policy."""


@dataclass(frozen=True)
class Catch22ArtifactLayout:
    """Resolved Catch22 artifact layout after env/default handling."""

    project_root: Path
    artifact_root: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    results_dir: Path
    paper_dir: Path
    smoke_root: Path
    logs_dir: Path
    log_file: Path


def _normalize(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _path_from_env(name: str) -> Optional[Path]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return _normalize(raw)


def path_is_within(path: Path | str, root: Path | str = DEFAULT_MEDIA_ROOT) -> bool:
    """Return True when ``path`` resolves under ``root``."""

    candidate = _normalize(path)
    allowed_root = _normalize(root)
    try:
        candidate.relative_to(allowed_root)
        return True
    except ValueError:
        return False


def get_storage_policy() -> str:
    """Return the validated storage policy."""

    policy = os.getenv("AKI_STORAGE_POLICY", DEFAULT_STORAGE_POLICY).strip().lower()
    if policy not in VALID_STORAGE_POLICIES:
        raise ValueError(
            f"Invalid AKI_STORAGE_POLICY={policy!r}. Expected one of: {', '.join(sorted(VALID_STORAGE_POLICIES))}."
        )
    return policy


def get_artifact_root() -> Path:
    """Return the canonical artifact root."""

    return _path_from_env("AKI_ARTIFACT_ROOT") or DEFAULT_ARTIFACT_ROOT


def get_data_dir(project_root: Path) -> Path:
    """Return the parent data directory used by shell runners and preprocessing."""

    explicit = _path_from_env("DATA_DIR")
    if explicit is not None:
        return explicit
    return get_artifact_root() / "data"


def get_raw_dir(project_root: Path) -> Path:
    """Return the raw input directory.

    Raw data is intentionally left at the repository default unless the user
    overrides ``RAW_DIR`` or ``DATA_DIR``.
    """

    explicit = _path_from_env("RAW_DIR")
    if explicit is not None:
        return explicit
    data_dir = _path_from_env("DATA_DIR")
    if data_dir is not None:
        return data_dir / "raw"
    return _normalize(project_root / "data" / "raw")


def get_processed_dir(project_root: Path) -> Path:
    """Return the processed-data directory."""

    explicit = _path_from_env("PROCESSED_DIR")
    if explicit is not None:
        return explicit
    data_dir = _path_from_env("DATA_DIR")
    if data_dir is not None:
        return data_dir / "processed"
    return get_artifact_root() / "data" / "processed"


def get_results_dir(project_root: Path) -> Path:
    """Return the experiments root."""

    explicit = _path_from_env("RESULTS_DIR")
    if explicit is not None:
        return explicit
    return get_artifact_root() / "results" / "catch22" / "experiments"


def get_paper_dir(project_root: Path) -> Path:
    """Return the Catch22 paper surface root."""

    explicit = _path_from_env("PAPER_DIR")
    if explicit is not None:
        return explicit
    results_dir = _path_from_env("RESULTS_DIR")
    if results_dir is not None:
        return results_dir.parent / "paper"
    return get_artifact_root() / "results" / "catch22" / "paper"


def get_smoke_root(project_root: Path) -> Path:
    """Return the default smoke-test root."""

    explicit = _path_from_env("SMOKE_ROOT")
    if explicit is not None:
        return explicit
    return get_artifact_root() / "smoke_test_outputs"


def get_logs_dir() -> Path:
    """Return the canonical logs directory."""

    return get_artifact_root() / "logs"


def get_log_file(default_name: str) -> Path:
    """Return the resolved log-file path."""

    explicit = _path_from_env("LOG_FILE")
    if explicit is not None:
        return explicit
    return get_logs_dir() / default_name


def build_catch22_layout(project_root: Path, *, default_log_name: str) -> Catch22ArtifactLayout:
    """Resolve the Catch22 artifact layout from env/defaults."""

    artifact_root = get_artifact_root()
    return Catch22ArtifactLayout(
        project_root=_normalize(project_root),
        artifact_root=artifact_root,
        data_dir=get_data_dir(project_root),
        raw_dir=get_raw_dir(project_root),
        processed_dir=get_processed_dir(project_root),
        results_dir=get_results_dir(project_root),
        paper_dir=get_paper_dir(project_root),
        smoke_root=get_smoke_root(project_root),
        logs_dir=get_logs_dir(),
        log_file=get_log_file(default_log_name),
    )


def enforce_storage_policy(
    paths: Mapping[str, Path | str] | Iterable[tuple[str, Path | str]],
    *,
    policy: Optional[str] = None,
    allowed_root: Path | str = DEFAULT_MEDIA_ROOT,
) -> Dict[str, Path]:
    """Validate generated artifact paths against the configured storage policy."""

    resolved_policy = get_storage_policy() if policy is None else policy
    if resolved_policy not in VALID_STORAGE_POLICIES:
        raise ValueError(
            f"Invalid storage policy {resolved_policy!r}. Expected one of: {', '.join(sorted(VALID_STORAGE_POLICIES))}."
        )

    items = dict(paths)
    normalized = {label: _normalize(path) for label, path in items.items()}
    if resolved_policy == "off":
        return normalized

    violations = {
        label: path
        for label, path in normalized.items()
        if not path_is_within(path, allowed_root)
    }
    if not violations:
        return normalized

    message = (
        "Generated artifact paths must live under "
        f"{_normalize(allowed_root)} when AKI_STORAGE_POLICY={resolved_policy!r}. "
        "Override the specific path env vars or set AKI_STORAGE_POLICY=warn/off to bypass this check. "
        "Violations: "
        + ", ".join(f"{label}={path}" for label, path in violations.items())
    )
    if resolved_policy == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return normalized

    raise StoragePolicyError(message)
