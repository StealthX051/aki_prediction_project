import pandas as pd
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import (
    CATCH_22_FILE,
    CATCH_22_WINDOWED_FILE,
    INTRAOP_WIDE_FILE,
    INTRAOP_WIDE_WINDOWED_FILE,
    MANDATORY_WAVEFORMS,
)
from data_preparation.artifact_metadata import ArtifactCompatibilityError, file_sha256, metadata_path_for
from data_preparation.step_02_catch_22 import EXTRACTION_PROTOCOL_VERSION

def process_waveform_file(input_path, output_path, mode_name):
    print(f"\n--- Processing {mode_name} Waveform Data ---")
    print(f"Input: {input_path}")
    
    if not input_path.exists():
        print(f"Skipping {mode_name}: Input file not found.")
        return

    # 1. Load Data
    try:
        long_df = pd.read_csv(input_path)
        print(f"Loaded data. Shape: {long_df.shape}")
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        return

    # 2. Pivot to Wide Format
    print("Pivoting to wide format...")
    id_cols = ['caseid']
    pivot_col = 'waveform'
    
    # Identify feature columns (all cols that are not ID or pivot cols)
    feature_cols = [col for col in long_df.columns if col not in id_cols + [pivot_col]]
    
    if not feature_cols:
        print("ERROR: No feature columns found to pivot.")
        return

    if long_df[["caseid", "waveform"]].isna().any().any():
        raise ValueError("Waveform extraction contains missing case/channel identities.")
    if long_df.duplicated(["caseid", "waveform"]).any():
        raise ValueError("Duplicate case/channel pairs cannot be averaged during waveform assembly.")
    unknown_channels = set(long_df["waveform"]) - set(MANDATORY_WAVEFORMS)
    if unknown_channels:
        raise ValueError(f"Unexpected waveform channels: {sorted(unknown_channels)}")
    source_meta_path = metadata_path_for(input_path)
    if not source_meta_path.exists():
        raise ArtifactCompatibilityError(f"Waveform source {input_path} lacks extraction provenance.")
    source_metadata = json.loads(source_meta_path.read_text())
    if source_metadata.get("extraction_protocol_version") != EXTRACTION_PROTOCOL_VERSION or not source_metadata.get("extraction_fingerprint"):
        raise ArtifactCompatibilityError(f"Waveform source {input_path} lacks a supported extraction fingerprint.")

    waveform_wide_df = long_df.pivot(
        index=id_cols,
        columns=pivot_col,
        values=feature_cols
    )
    # Retain whole missing channels as NaN instead of silently changing feature
    # set definitions when every extraction of a channel failed.
    waveform_wide_df = waveform_wide_df.reindex(
        columns=pd.MultiIndex.from_product([feature_cols, MANDATORY_WAVEFORMS])
    )
    print(f"Pivot complete. Shape before flattening: {waveform_wide_df.shape}")

    # 3. Flatten Column Names
    print("Flattening column names...")
    new_cols = []
    for feature_name, waveform_name in waveform_wide_df.columns:
        # Replace slashes (e.g., in 'SNUADC/PLETH') with underscores
        clean_waveform = waveform_name.replace('/', '_')
        new_cols.append(f"{clean_waveform}_{feature_name}")

    waveform_wide_df.columns = new_cols
    waveform_wide_df = waveform_wide_df.reset_index()
    print(f"Column names flattened. Shape: {waveform_wide_df.shape}")

    # 4. Save
    print(f"Saving to {output_path}...")
    waveform_wide_df.to_csv(output_path, index=False)
    channels_by_case = long_df.groupby("caseid")["waveform"].agg(set)
    missing_channels = {
        str(caseid): sorted(set(MANDATORY_WAVEFORMS) - channels)
        for caseid, channels in channels_by_case.items()
        if len(channels) < len(MANDATORY_WAVEFORMS)
    }
    error_pairs = set()
    error_path = Path(source_metadata.get("error_file", "__no_error_file__"))
    if error_path.is_file():
        errors = pd.read_csv(error_path, usecols=["caseid", "waveform"])
        error_pairs = set(zip(errors["caseid"], errors["waveform"]))
    successful_pairs = set(zip(long_df["caseid"], long_df["waveform"]))
    requested_caseids = set(source_metadata.get("requested_caseids") or long_df["caseid"].tolist())
    unavailable_caseids = requested_caseids - set(long_df["caseid"])
    metadata = {
        "artifact_role": "step_04_waveforms",
        "assembly_protocol_version": 1,
        "source_long_file_sha256": file_sha256(input_path),
        "data_sha256": file_sha256(output_path),
        "extraction_fingerprint": source_metadata["extraction_fingerprint"],
        "extraction_diagnostics": {
            "successful_pairs": len(successful_pairs),
            "failed_pairs": len(error_pairs - successful_pairs),
            "operations": len(waveform_wide_df),
            "requested_operations": len(requested_caseids),
            "operations_without_successful_channels": sorted(unavailable_caseids),
            "operations_with_partial_channels": len(missing_channels),
            "missing_channels_by_caseid": missing_channels,
        },
    }
    metadata_path_for(output_path).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"Done with {mode_name}.")

def main():
    print("--- Step 04: Intraoperative Data Preparation ---")
    
    # Define tasks: (Input Path, Output Path, Mode Name)
    tasks = [
        (CATCH_22_FILE, INTRAOP_WIDE_FILE, "Full Features"),
        (CATCH_22_WINDOWED_FILE, INTRAOP_WIDE_WINDOWED_FILE, "Windowed Features")
    ]
    
    for input_p, output_p, name in tasks:
        process_waveform_file(input_p, output_p, name)

if __name__ == "__main__":
    main()
