from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from data_preparation.artifact_metadata import (
    STEP_01_COHORT_ARTIFACT,
    STEP_03_PREOP_ARTIFACT,
    STEP_05_MERGED_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
    ARTIFACT_SCHEMA_VERSIONS,
    file_sha256,
)
from data_preparation.cohort_scopes import COHORT_SCOPES, LEGACY_STORED_SCOPES, build_cohort_manifest
from data_preparation.inputs import (
    COHORT_FILE,
    PREOP_PROCESSED_FILE,
    WIDE_FEATURES_FILE,
    WIDE_FEATURES_WINDOWED_FILE,
    INPUT_FILE,
    LAB_DATA_FILE,
)
from data_preparation.step_02_catch_22 import COHORT_REQUIRED_COLUMNS, build_extraction_fingerprint
from data_preparation.outcome_registry import (
    ALL_ELIGIBILITY_COLUMNS,
    ALL_OUTCOME_COLUMNS,
    ALL_SPLIT_COLUMNS,
)

PREOP_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]

MERGED_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]
COHORT_IDENTITY_COLUMNS = ["subjectid", *ALL_OUTCOME_COLUMNS, *ALL_ELIGIBILITY_COLUMNS]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed Catch22 artifacts before reuse.")
    parser.add_argument(
        "--skip-preop",
        action="store_true",
        help="Skip validation of the Step 03 preoperative artifact.",
    )
    return parser.parse_args()


def validate_prepared_artifacts(
    preop_path: Path | None,
    branch_paths: dict[str, Path],
    *,
    cohort_path: Path,
    source_paths: dict[str, Path] | None = None,
) -> dict[str, dict[str, object]]:
    """Verify shared source identity and actual branch rosters before reuse.

    Raw files are optional; when present their content must agree with the
    source hashes carried by every prepared artifact. The saved extraction
    cohort and current extraction protocol/settings are required for reuse.
    """
    cohort, _ = read_versioned_csv(
        cohort_path, artifact_role=STEP_01_COHORT_ARTIFACT,
        required_columns=[*COHORT_REQUIRED_COLUMNS, *COHORT_IDENTITY_COLUMNS],
    )
    metadata_by_path = {}
    derived_scope_manifests = {}
    if preop_path is not None:
        frame, metadata = read_versioned_csv(
            preop_path, artifact_role=STEP_03_PREOP_ARTIFACT, required_columns=PREOP_REQUIRED_COLUMNS
        )
        if metadata.get("upstream_schema_version") != ARTIFACT_SCHEMA_VERSIONS["step_01_cohort"]:
            raise ArtifactCompatibilityError(f"Stale upstream cohort schema in {preop_path}.")
        metadata_by_path[preop_path] = (frame, metadata)

    for branch, path in branch_paths.items():
        frame, metadata = read_versioned_csv(
            path, artifact_role=STEP_05_MERGED_ARTIFACT, required_columns=MERGED_REQUIRED_COLUMNS
        )
        if metadata.get("upstream_schema_version") != ARTIFACT_SCHEMA_VERSIONS[STEP_03_PREOP_ARTIFACT]:
            raise ArtifactCompatibilityError(f"Stale upstream preoperative schema in {path}.")
        if metadata.get("branch") != branch:
            raise ArtifactCompatibilityError(f"Processing branch identity mismatch in {path}.")
        manifests = metadata.get("cohort_scopes", {})
        derived_scope_manifests[branch] = validate_scope_manifests(frame, manifests, branch, path)
        extraction_mode = {"non_windowed": "full", "windowed": "windowed"}.get(branch)
        if extraction_mode is None:
            raise ArtifactCompatibilityError(f"Unsupported processing branch {branch!r} in {path}.")
        expected_fingerprint = build_extraction_fingerprint(cohort, extraction_mode)
        if metadata.get("extraction_fingerprint") != expected_fingerprint:
            raise ArtifactCompatibilityError(
                f"Waveform extraction fingerprint mismatch in {path}; use a fresh processed root."
            )
        metadata_by_path[path] = (frame, metadata)

    required_hashes = {"clinical_data_sha256", "lab_data_sha256"}
    reference_hashes = None
    for path, (frame, metadata) in metadata_by_path.items():
        if metadata.get("data_sha256") != file_sha256(path):
            raise ArtifactCompatibilityError(f"Prepared CSV content hash mismatch in {path}.")
        try:
            saved = frame.set_index("caseid", verify_integrity=True)[COHORT_IDENTITY_COLUMNS]
            source = cohort.set_index("caseid", verify_integrity=True)[COHORT_IDENTITY_COLUMNS]
            if not saved.index.isin(source.index).all():
                raise AssertionError("Prepared operation is absent from the saved extraction cohort.")
            pd.testing.assert_frame_equal(
                saved, source.reindex(saved.index), check_dtype=False, check_index_type=False, check_exact=True
            )
        except (AssertionError, ValueError) as exc:
            raise ArtifactCompatibilityError(
                f"Patient/outcome/eligibility values disagree with the saved cohort in {path}."
            ) from exc
        hashes = metadata.get("source_hashes", {})
        if not required_hashes.issubset(hashes) or not all(hashes[key] for key in required_hashes):
            raise ArtifactCompatibilityError(f"Missing raw source identities in {path}.")
        if reference_hashes is None:
            reference_hashes = hashes
        elif hashes != reference_hashes:
            raise ArtifactCompatibilityError(f"Prepared artifacts disagree on raw source hashes: {path}.")
    for key, source_path in (source_paths or {}).items():
        if Path(source_path).is_file() and (reference_hashes or {}).get(key) != file_sha256(source_path):
            raise ArtifactCompatibilityError(f"Raw source changed since preprocessing: {source_path}.")

    return derived_scope_manifests


def validate_scope_manifests(frame, manifests, branch, path):
    """Validate stored scope contracts and derive new scope overlays read-only.

    Only the newly introduced mortality entry may be absent from schema-v5
    sidecars. If present, it must match just as strictly as original entries.
    Callers may persist the returned overlay under their new study root.
    """
    current = {scope: build_cohort_manifest(frame, scope, branch) for scope in COHORT_SCOPES}
    for scope in COHORT_SCOPES:
        if scope in LEGACY_STORED_SCOPES or scope in manifests:
            if manifests.get(scope) != current[scope]:
                raise ArtifactCompatibilityError(f"Cohort roster/count metadata mismatch for {scope} in {path}.")
    return current


def main() -> None:
    args = _parse_args()

    try:
        validate_prepared_artifacts(
            None if args.skip_preop else PREOP_PROCESSED_FILE,
            {"non_windowed": WIDE_FEATURES_FILE, "windowed": WIDE_FEATURES_WINDOWED_FILE},
            cohort_path=COHORT_FILE,
            source_paths={"clinical_data_sha256": INPUT_FILE, "lab_data_sha256": LAB_DATA_FILE},
        )
    except (ArtifactCompatibilityError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print("Processed artifacts passed schema validation.")


if __name__ == "__main__":
    main()
