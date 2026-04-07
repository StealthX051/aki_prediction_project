import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.artifact_metadata import (
    STEP_03_PREOP_ARTIFACT,
    STEP_05_MERGED_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
    write_artifact_metadata,
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

PREOP_REQUIRED_COLUMNS = [
    "caseid",
    "subjectid",
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,
    *ALL_SPLIT_COLUMNS,
]


def merge_and_save(
    preop_df,
    intraop_path,
    output_path,
    mode_name,
    impute_missing: bool,
    *,
    preop_metadata: Optional[dict] = None,
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

    # 2. Merge
    print("Merging with preoperative data...")
    # Use a left merge to ensure we keep all patients from the intraop cohort
    # (or intersection? The original script used left merge on waveform data)
    # Original: master_df = pd.merge(waveform_wide_df, preop_final_df, on=..., how='left')
    # This implies we prioritize the waveform cohort.
    
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
    
    # 3. Check for Merge Issues
    merge_nan_count = master_df.isnull().sum().sum()
    if merge_nan_count > 0:
        print(f"WARNING: Merge introduced {merge_nan_count} NaN values.")
        if impute_missing:
            print("Imputing these with -99 (assuming they are missing preop features)...")
            master_df.fillna(-99, inplace=True)
        else:
            print("Leaving NaNs in merged output (imputation disabled).")

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
            "upstream_schema_version": 1,
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
            "Fill merged NaNs with -99 (previous behavior). Default preserves NaNs"
            " introduced during merging."
        ),
    )

    if impute_missing is None:
        args = parser.parse_args()
        impute_missing = args.impute_missing
    else:
        _ = parser.parse_args([])

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
        )

if __name__ == "__main__":
    main()
