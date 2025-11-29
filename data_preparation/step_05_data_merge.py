import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import (
    PREOP_PROCESSED_FILE,
    INTRAOP_WIDE_FILE,
    INTRAOP_WIDE_WINDOWED_FILE,
    WIDE_FEATURES_FILE,
    WIDE_FEATURES_WINDOWED_FILE,
    OUTCOME
)

def merge_and_save(preop_df, intraop_path, output_path, mode_name):
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
        on=['caseid', OUTCOME],
        how='left'
    )
    
    # 3. Check for Merge Issues
    merge_nan_count = master_df.isnull().sum().sum()
    if merge_nan_count > 0:
        print(f"WARNING: Merge introduced {merge_nan_count} NaN values.")
        print("Imputing these with -99 (assuming they are missing preop features)...")
        master_df.fillna(-99, inplace=True)

    # 4. Verify Split Group
    if 'split_group' not in master_df.columns:
        print("ERROR: 'split_group' column missing after merge! This is critical.")
        # Attempt to recover if possible, or fail?
        # If preop_df has it, it should be there.
        # If intraop_df has rows not in preop_df, they will have NaN split_group.
        # We should check for that.
        pass
    else:
        missing_split = master_df['split_group'].isnull().sum()
        if missing_split > 0:
            print(f"WARNING: {missing_split} rows have missing 'split_group'.")
            # These are likely patients in waveform file but not in preop file.
            # We should probably drop them or assign them to 'train'/'test' randomly?
            # For safety, let's drop them as they lack preop data.
            print("Dropping rows with missing split_group/preop data...")
            master_df = master_df.dropna(subset=['split_group'])

    print(f"Final merged shape: {master_df.shape}")

    # 5. Save
    print(f"Saving to {output_path}...")
    master_df.to_csv(output_path, index=False)
    print(f"Done with {mode_name}.")

def main():
    print("--- Step 05: Data Merge ---")
    
    # 1. Load Preop Data
    print(f"Loading Preop Data from {PREOP_PROCESSED_FILE}...")
    if not PREOP_PROCESSED_FILE.exists():
        print("ERROR: Preop processed file not found. Run step_03 first.")
        sys.exit(1)
        
    preop_df = pd.read_csv(PREOP_PROCESSED_FILE)
    print(f"Preop Data Shape: {preop_df.shape}")
    
    # Define tasks
    tasks = [
        (INTRAOP_WIDE_FILE, WIDE_FEATURES_FILE, "Full Features"),
        (INTRAOP_WIDE_WINDOWED_FILE, WIDE_FEATURES_WINDOWED_FILE, "Windowed Features")
    ]
    
    for intraop_p, output_p, name in tasks:
        merge_and_save(preop_df, intraop_p, output_p, name)

if __name__ == "__main__":
    main()
