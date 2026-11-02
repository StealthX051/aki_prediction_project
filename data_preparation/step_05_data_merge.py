import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.artifact_metadata import (
    ARTIFACT_SCHEMA_VERSIONS,
    STEP_03_PREOP_ARTIFACT,
    STEP_05_MERGED_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
    write_artifact_metadata,
    file_sha256,
    metadata_path_for,
)
from data_preparation.cohort_scopes import (
    COHORT_SCOPES, build_cohort_manifest, cohort_counts, select_cohort, selected_case_hash,
)
from data_preparation.inputs import (
    PREOP_PROCESSED_FILE,
    INTRAOP_WIDE_FILE,
    INTRAOP_WIDE_WINDOWED_FILE,
    WIDE_FEATURES_FILE,
    WIDE_FEATURES_WINDOWED_FILE,
    IMPUTE_MISSING,
)
from data_preparation.outcome_registry import (
    ALL_ELIGIBILITY_COLUMNS,
    ALL_OUTCOME_COLUMNS,
    ALL_SPLIT_COLUMNS,
)
from data_preparation.step_02_catch_22 import EXTRACTION_PROTOCOL_VERSION

PREOP_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]


def write_branch_roster_manifest(branch_files: dict[str, Path], output_path: Path) -> None:
    """Record branch-specific rosters and the intersection for branch contrasts."""
    branches = {
        name: read_versioned_csv(
            path, artifact_role=STEP_05_MERGED_ARTIFACT,
            required_columns=["caseid", "subjectid", "eligible_any_aki", "aki_label",
                              "eligible_icu_admission", "y_icu_admit"],
        )[0]
        for name, path in branch_files.items()
    }
    intersections = {}
    for scope in COHORT_SCOPES:
        selected = {name: select_cohort(frame, scope) for name, frame in branches.items()}
        common_caseids = set.intersection(*(set(frame["caseid"]) for frame in selected.values()))
        common = next(iter(selected.values())).loc[lambda frame: frame["caseid"].isin(common_caseids)]
        # A shared operation must retain the same patient identity in every branch.
        for frame in selected.values():
            matched = frame.loc[frame["caseid"].isin(common_caseids)]
            if selected_case_hash(matched) != selected_case_hash(common):
                raise ValueError("Patient identity differs between waveform-processing branches.")
        intersections[scope] = {
            "branches": {name: build_cohort_manifest(frame, scope, name) for name, frame in selected.items()},
            "intersection": {**cohort_counts(common), "selected_case_hash": selected_case_hash(common),
                             "caseids": sorted(common_caseids)},
        }
    output_path.write_text(json.dumps(intersections, indent=2, sort_keys=True))


def merge_and_save(
    preop_df,
    intraop_path,
    output_path,
    mode_name,
    impute_missing: bool,
    *,
    preop_metadata: Optional[dict] = None,
    require_intraop_metadata: bool = False,
):
    print(f"\n--- Merging {mode_name} Data ---")
    print(f"Intraop Input: {intraop_path}")
    
    if not intraop_path.exists():
        print(f"Skipping {mode_name}: Intraop file not found.")
        return

    # 1. Load Intraop Data
    try:
        intraop_df = pd.read_csv(intraop_path)
        print(f"Loaded intraop data. Shape: {intraop_df.shape}")
    except Exception as e:
        print(f"Error loading {intraop_path}: {e}")
        return

    intraop_metadata = {}
    if require_intraop_metadata:
        meta_path = metadata_path_for(intraop_path)
        if not meta_path.exists():
            raise ArtifactCompatibilityError(f"Intraoperative artifact {intraop_path} lacks extraction provenance.")
        intraop_metadata = json.loads(meta_path.read_text())
        if intraop_metadata.get("assembly_protocol_version") != 1:
            raise ArtifactCompatibilityError(f"Unsupported waveform assembly protocol for {intraop_path}.")
        if intraop_metadata.get("data_sha256") != file_sha256(intraop_path):
            raise ArtifactCompatibilityError(f"Waveform artifact hash mismatch for {intraop_path}.")
        fingerprint = intraop_metadata.get("extraction_fingerprint")
        if not isinstance(fingerprint, dict) or fingerprint.get("extraction_protocol_version") != EXTRACTION_PROTOCOL_VERSION:
            raise ArtifactCompatibilityError(f"Stale waveform extraction protocol for {intraop_path}.")

    # 2. Merge
    print("Merging with preoperative data...")
    # Retain the branch's waveform-qualified operations and require one
    # matching clinical record for each operation.
    
    master_df = pd.merge(
        intraop_df,
        preop_df,
        on=['caseid'],
        how='left',
        validate='one_to_one',
        indicator=True,
    )

    unmatched_rows = int((master_df["_merge"] != "both").sum())
    if unmatched_rows:
        raise ValueError(
            f"Merge failed for {unmatched_rows} rows in {mode_name}; missing preop metadata for some caseids."
        )
    master_df = master_df.drop(columns=["_merge"])
    branch = "windowed" if "windowed" in mode_name.lower() else "non_windowed"
    cohort_manifests = {}
    # Small programmatic utility tests may supply only a subset of outcome/ID
    # columns; production main() always requires the full versioned schema.
    if {"subjectid", "eligible_any_aki", "aki_label", "eligible_icu_admission", "y_icu_admit"}.issubset(master_df):
        cohort_manifests = {
            scope: build_cohort_manifest(master_df, scope, branch) for scope in COHORT_SCOPES
        }
    
    # 3. Check for Merge Issues
    merge_nan_count = master_df.isnull().sum().sum()
    if merge_nan_count > 0:
        print(f"Merged output contains {merge_nan_count} NaN values; preserving missing outcomes and features.")
    if impute_missing:
        print("NOTE: --impute-missing is retained for CLI compatibility only and does not modify Step 05 outputs.")

    print(f"Final merged shape: {master_df.shape}")

    # 5. Save
    print(f"Saving to {output_path}...")
    master_df.to_csv(output_path, index=False)
    write_artifact_metadata(
        output_path,
        artifact_role=STEP_05_MERGED_ARTIFACT,
        available_columns=master_df.columns,
        extra_metadata={
            "mode_name": mode_name,
            "split_status": dict((preop_metadata or {}).get("split_status") or {}),
            "upstream_artifact": str(PREOP_PROCESSED_FILE),
            "upstream_schema_version": (preop_metadata or {}).get(
                "schema_version", ARTIFACT_SCHEMA_VERSIONS[STEP_03_PREOP_ARTIFACT]
            ),
            "source_hashes": (preop_metadata or {}).get("source_hashes", {}),
            "outer_flow_counts": (preop_metadata or {}).get("outer_flow_counts", {}),
            "diagnostic_trim": (preop_metadata or {}).get("diagnostic_trim"),
            "data_sha256": file_sha256(output_path),
            "branch": branch,
            "cohort_scopes": cohort_manifests,
            "source_cohort_scopes": (preop_metadata or {}).get("cohort_scopes", {}),
            "extraction_fingerprint": intraop_metadata.get("extraction_fingerprint"),
            "extraction_diagnostics": intraop_metadata.get("extraction_diagnostics", {}),
        },
    )
    print(f"Done with {mode_name}.")

def main(impute_missing: Optional[bool] = None):
    parser = argparse.ArgumentParser(description="Merge preop and intraop datasets.")
    parser.add_argument(
        "--impute-missing",
        action="store_true",
        default=IMPUTE_MISSING,
        help=(
            "Deprecated compatibility option; missing outcomes and features remain NaN."
        ),
    )

    if impute_missing is None:
        args = parser.parse_args()
        impute_missing = args.impute_missing

    print("--- Step 05: Data Merge ---")
    
    # 1. Load Preop Data
    print(f"Loading Preop Data from {PREOP_PROCESSED_FILE}...")
    if not PREOP_PROCESSED_FILE.exists():
        print("ERROR: Preop processed file not found. Run step_03 first.")
        sys.exit(1)
        
    try:
        preop_df, preop_metadata = read_versioned_csv(
            PREOP_PROCESSED_FILE,
            artifact_role=STEP_03_PREOP_ARTIFACT,
            required_columns=PREOP_REQUIRED_COLUMNS,
        )
    except ArtifactCompatibilityError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
    print(f"Preop Data Shape: {preop_df.shape}")
    
    # Define tasks
    tasks = [
        (INTRAOP_WIDE_FILE, WIDE_FEATURES_FILE, "Full Features"),
        (INTRAOP_WIDE_WINDOWED_FILE, WIDE_FEATURES_WINDOWED_FILE, "Windowed Features")
    ]
    
    for intraop_p, output_p, name in tasks:
        merge_and_save(
            preop_df,
            intraop_p,
            output_p,
            name,
            impute_missing,
            preop_metadata=preop_metadata,
            require_intraop_metadata=True,
        )

    if WIDE_FEATURES_FILE.exists() and WIDE_FEATURES_WINDOWED_FILE.exists():
        manifest_path = WIDE_FEATURES_FILE.parent / "cohort_rosters.json"
        write_branch_roster_manifest(
            {"non_windowed": WIDE_FEATURES_FILE, "windowed": WIDE_FEATURES_WINDOWED_FILE}, manifest_path
        )

if __name__ == "__main__":
    main()
