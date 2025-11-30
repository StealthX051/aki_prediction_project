# Test to verify the preop variable extraction functionality in step_03_preop_prep.py is working
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import PREOP_PROCESSED_FILE

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import PREOP_PROCESSED_FILE

def verify_columns():
    print(f"Checking file: {PREOP_PROCESSED_FILE}")
    
    if not PREOP_PROCESSED_FILE.exists():
        print("ERROR: File does not exist.")
        return

    df = pd.read_csv(PREOP_PROCESSED_FILE)
    print(f"Loaded data shape: {df.shape}")
    
    # --- Define Expected Variables ---
    
    # Continuous variables (should exist as is)
    continuous_vars = [
        # Demographics / Body
        'age', 'height', 'weight', 'bmi',
        
        # Labs (Clinical)
        'preop_hb', 'preop_plt', 'preop_pt', 'preop_aptt',
        'preop_na', 'preop_k', 'preop_gluc', 'preop_alb', 
        'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr', 'preop_hco3',
        
        # ABG
        'preop_ph', 'preop_be', 'preop_pao2', 'preop_paco2', 'preop_sao2',
        
        # New Labs
        'preop_wbc', 'preop_gfr', 'preop_crp', 'preop_lac',
        
        # Derived Continuous
        'preop_los_days', 'preop_egfr_ckdepi'
    ]
    
    # Categorical variables (Original names - expected to be One-Hot Encoded)
    categorical_vars = [
        'sex', 'emop', 'department', 'approach', 'asa', 'optype', 
        'ane_type', 'position', 'preop_htn', 'preop_dm', 'preop_ecg', 'preop_pft'
    ]
    
    # Derived Binary Flags (Should exist as is)
    binary_flags = [
        'inpatient_preop', 'bun_high', 'hypoalbuminemia', 'preop_anemia', 
        'hyponatremia', 'metabolic_acidosis', 'hypercapnia', 'hypoxemia'
    ]
    
    # --- Verification Logic ---
    
    missing_cols = []
    empty_cols = []
    
    print("\n--- 1. Checking Continuous Variables ---")
    for col in continuous_vars:
        if col not in df.columns:
            missing_cols.append(col)
        else:
            # Check for content (not all NaN and not all -99 imputed)
            # Note: -99 is the imputation value used in step_03
            valid_data = df[col][(df[col].notna()) & (df[col] != -99)]
            if valid_data.empty:
                empty_cols.append(f"{col} (All NaN or -99)")
            else:
                # Optional: Print stats for confirmation
                # print(f"  OK: {col} (Valid: {len(valid_data)}/{len(df)})")
                pass

    print("\n--- 2. Checking Binary Flags ---")
    for col in binary_flags:
        if col not in df.columns:
            missing_cols.append(col)
        else:
            if df[col].nunique() <= 1 and df[col].iloc[0] == -99:
                 empty_cols.append(f"{col} (All -99)")
    
    print("\n--- 3. Checking Categorical Variables (One-Hot Encoded) ---")
    # For categorical, we check if there is at least one column starting with the prefix
    # Exception: 'sex' becomes 'sex_M' etc.
    for cat in categorical_vars:
        # Look for columns starting with cat_
        # Note: pandas get_dummies usually does prefix_val
        # But here we passed columns to get_dummies, so it uses colname_val
        
        # We need to be careful with overlapping names if any
        encoded_cols = [c for c in df.columns if c.startswith(f"{cat}_")]
        
        if not encoded_cols:
             # Check if it exists raw (it shouldn't)
             if cat in df.columns:
                 print(f"  WARNING: {cat} found as raw column, expected encoded.")
             else:
                 missing_cols.append(f"{cat} (Encoded versions)")
        else:
             # Check if they have data (they are 0/1, so usually fine)
             pass

    # --- Report ---
    if missing_cols:
        print(f"\nFAILURE: Missing Columns:\n" + "\n".join([f"  - {c}" for c in missing_cols]))
    else:
        print("\nSUCCESS: All expected variables (continuous, binary, encoded categorical) are present.")
        
    if empty_cols:
        print(f"\nWARNING: Columns with no valid information (All NaN or -99):\n" + "\n".join([f"  - {c}" for c in empty_cols]))
    else:
        print("\nSUCCESS: All present columns contain valid information (not just NaN/-99).")

    # Check specifically for 'adm' (should be dropped)
    if 'adm' in df.columns:
        print("\nNOTE: 'adm' column is present (unexpected, supposed to be dropped).")
    else:
        print("\nNOTE: 'adm' column correctly dropped (used for derivation).")

if __name__ == "__main__":
    verify_columns()
