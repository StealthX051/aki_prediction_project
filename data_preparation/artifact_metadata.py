from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

STEP_01_COHORT_ARTIFACT = "step_01_cohort"
STEP_03_PREOP_ARTIFACT = "step_03_preop"
STEP_05_MERGED_ARTIFACT = "step_05_merged"

ARTIFACT_SCHEMA_VERSIONS = {
    STEP_01_COHORT_ARTIFACT: 1,
    STEP_03_PREOP_ARTIFACT: 1,
    STEP_05_MERGED_ARTIFACT: 1,
}


class ArtifactCompatibilityError(RuntimeError):
    """Raised when a processed artifact is stale or missing required metadata."""


def metadata_path_for(data_path: Path) -> Path:
    return data_path.with_suffix(f"{data_path.suffix}.metadata.json")


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_rebuild_message(data_path: Path, *, artifact_role: str, detail: str) -> str:
    return (
        f"Processed artifact '{data_path}' is not compatible with the current {artifact_role} schema: "
        f"{detail}. Rebuild the data preparation pipeline from Step 01 through Step 05 "
        "(or run the launcher with '--prep force')."
    )


def write_artifact_metadata(
    data_path: Path,
    *,
    artifact_role: str,
    available_columns: Sequence[str],
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    meta_path = metadata_path_for(data_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "artifact_role": artifact_role,
        "schema_version": ARTIFACT_SCHEMA_VERSIONS[artifact_role],
        "data_path": str(data_path),
        "available_columns": _dedupe_preserve_order([str(col) for col in available_columns]),
    }
    if extra_metadata:
        payload.update(extra_metadata)

    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return meta_path


def load_artifact_metadata(data_path: Path) -> dict[str, Any]:
    meta_path = metadata_path_for(data_path)
    if not meta_path.exists():
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role="processed artifact",
                detail=f"metadata sidecar is missing at {meta_path}",
            )
        )

    with meta_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role="processed artifact",
                detail="metadata sidecar does not contain a JSON object",
            )
        )

    return payload


def validate_artifact_metadata(
    data_path: Path,
    *,
    artifact_role: str,
    required_columns: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(metadata or load_artifact_metadata(data_path))
    expected_version = ARTIFACT_SCHEMA_VERSIONS[artifact_role]

    actual_role = payload.get("artifact_role")
    if actual_role != artifact_role:
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role=artifact_role,
                detail=f"expected artifact_role '{artifact_role}' but found '{actual_role}'",
            )
        )

    actual_version = payload.get("schema_version")
    if actual_version != expected_version:
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role=artifact_role,
                detail=f"expected schema_version {expected_version} but found {actual_version!r}",
            )
        )

    available_columns = set(payload.get("available_columns") or [])
    missing_columns = [col for col in _dedupe_preserve_order(required_columns) if col not in available_columns]
    if missing_columns:
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role=artifact_role,
                detail=f"metadata is missing required columns {missing_columns}",
            )
        )

    return payload


def validate_dataframe_columns(
    df: pd.DataFrame,
    *,
    data_path: Path,
    artifact_role: str,
    required_columns: Sequence[str],
) -> None:
    missing_columns = [col for col in _dedupe_preserve_order(required_columns) if col not in df.columns]
    if missing_columns:
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role=artifact_role,
                detail=f"CSV is missing required columns {missing_columns}",
            )
        )


def read_versioned_csv(
    data_path: Path,
    *,
    artifact_role: str,
    required_columns: Sequence[str],
    **read_csv_kwargs: Any,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = validate_artifact_metadata(
        data_path,
        artifact_role=artifact_role,
        required_columns=required_columns,
    )

    try:
        df = pd.read_csv(data_path, **read_csv_kwargs)
    except FileNotFoundError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArtifactCompatibilityError(
            build_rebuild_message(
                data_path,
                artifact_role=artifact_role,
                detail=f"CSV could not be loaded ({exc})",
            )
        ) from exc

    validate_dataframe_columns(
        df,
        data_path=data_path,
        artifact_role=artifact_role,
        required_columns=required_columns,
    )
    df.attrs["artifact_metadata"] = metadata
    return df, metadata
