import os
import sys
import pandas as pd
import shutil
from unittest.mock import patch
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.inputs import COHORT_FILE

def run_real_smoke_test():
    print("=== Starting 100-Case Real Smoke Test ===")
    
    # 1. Setup Temporary Paths
    temp_cohort = 'tests/temp_real_cohort.csv'
    temp_features = 'tests/temp_real_features.csv'
    temp_features_win = 'tests/temp_real_features_windowed.csv'
    
    # Clean up any previous run
    if os.path.exists(temp_cohort): os.remove(temp_cohort)
    if os.path.exists(temp_features): os.remove(temp_features)
    if os.path.exists(temp_features_win): os.remove(temp_features_win)
    if os.path.exists('tests/temp_real_errors.csv'): os.remove('tests/temp_real_errors.csv')
    if os.path.exists('tests/temp_real_errors_windowed.csv'): os.remove('tests/temp_real_errors_windowed.csv')
    
    # 2. Create Sample Cohort from Real Data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=100, help='Number of cases to run')
    args, unknown = parser.parse_known_args()
    
    limit = args.limit
    print(f"Loading original cohort from {COHORT_FILE}...")
    try:
        full_cohort = pd.read_csv(COHORT_FILE)
        if len(full_cohort) > limit:
            sample_cohort = full_cohort.head(limit).copy()
        else:
            sample_cohort = full_cohort.copy()
            
        sample_cohort.to_csv(temp_cohort, index=False)
        print(f"Created sample cohort with {len(sample_cohort)} cases at {temp_cohort}")
    except Exception as e:
        print(f"FAILED to load/create sample cohort: {e}")
        return

    # 3. Import Module
    import data_preparation.step_02_catch_22 as step_02_catch_22
    
    # 4. Run Pipeline with Patched Config
    print("Running pipeline on real data (subset)...")
    
    # We only patch the file paths. We DO NOT patch VitalDB or Pool, so it runs for real.
    with patch('data_preparation.step_02_catch_22.COHORT_FILE', temp_cohort), \
         patch('data_preparation.step_02_catch_22.CATCH_22_FILE', temp_features), \
         patch('data_preparation.step_02_catch_22.CATCH_22_ERROR_FILE', 'tests/temp_real_errors.csv'):
         
         try:
             step_02_catch_22.main()
         except Exception as e:
             print(f"Pipeline execution FAILED: {e}")
             import traceback
             traceback.print_exc()
             return

    # 5. Verify Outputs
    print("\n=== Verifying Outputs ===")
    success = True
    
    # Check Full Features
    if os.path.exists(temp_features):
        df = pd.read_csv(temp_features)
        print(f"[PASS] Full features CSV exists. Shape: {df.shape}")
        if df.empty: 
            print("[WARN] Full features CSV is empty.")
    else:
        print("[FAIL] Full features CSV missing.")
        success = False

    # Check Windowed Features
    if os.path.exists(temp_features_win):
        df = pd.read_csv(temp_features_win)
        print(f"[PASS] Windowed features CSV exists. Shape: {df.shape}")
        if df.empty: 
            print("[WARN] Windowed features CSV is empty.")
    else:
        print("[FAIL] Windowed features CSV missing.")
        success = False

    # 6. Cleanup (Optional - maybe keep for inspection?)
    # if success:
    #     print("\n=== Cleanup ===")
    #     if os.path.exists(temp_cohort): os.remove(temp_cohort)
    #     if os.path.exists(temp_features): os.remove(temp_features)
    #     if os.path.exists(temp_features_win): os.remove(temp_features_win)
    #     if os.path.exists('tests/temp_real_errors.csv'): os.remove('tests/temp_real_errors.csv')
    #     if os.path.exists('tests/temp_real_errors_windowed.csv'): os.remove('tests/temp_real_errors_windowed.csv')
    #     print("Temporary files removed.")
    
    if success:
        print("\n>>> REAL SMOKE TEST PASSED <<<")
    else:
        print("\n>>> REAL SMOKE TEST FAILED <<<")

if __name__ == "__main__":
    run_real_smoke_test()
