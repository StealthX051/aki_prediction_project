import pandas as pd
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
    OUTCOME
)

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
    id_cols = ['caseid', OUTCOME]
    pivot_col = 'waveform'
    
    # Identify feature columns (all cols that are not ID or pivot cols)
    feature_cols = [col for col in long_df.columns if col not in id_cols + [pivot_col]]
    
    if not feature_cols:
        print("ERROR: No feature columns found to pivot.")
        return

    waveform_wide_df = long_df.pivot_table(
        index=id_cols,
        columns=pivot_col,
        values=feature_cols
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
