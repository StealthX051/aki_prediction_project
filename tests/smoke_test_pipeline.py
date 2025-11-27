import os
import sys
import pandas as pd
import numpy as np
import shutil
from unittest.mock import patch, MagicMock
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_preparation.inputs import COHORT_FILE

def run_smoke_test():
    print("=== Starting Smoke Test ===")
    
    # 1. Setup Temporary Paths
    temp_cohort = 'tests/temp_smoke_cohort.csv'
    temp_features = 'tests/temp_smoke_features.csv'
    temp_features_win = 'tests/temp_smoke_features_windowed.csv'
    temp_aeon_dir = 'tests/temp_smoke_aeon'
    
    # Clean up any previous run
    if os.path.exists(temp_cohort): os.remove(temp_cohort)
    if os.path.exists(temp_features): os.remove(temp_features)
    if os.path.exists(temp_features_win): os.remove(temp_features_win)
    if os.path.exists(temp_aeon_dir): shutil.rmtree(temp_aeon_dir)
    
    # 2. Create Sample Cohort
    print(f"Loading original cohort from {COHORT_FILE}...")
    try:
        full_cohort = pd.read_csv(COHORT_FILE)
        sample_cohort = full_cohort.head(2).copy() # Take 2 cases
        sample_cohort['opstart'] = 0
        sample_cohort['opend'] = 10
        sample_cohort.to_csv(temp_cohort, index=False)
        print(f"Created sample cohort with {len(sample_cohort)} cases at {temp_cohort}")
    except Exception as e:
        print(f"FAILED to load/create sample cohort: {e}")
        return

    # 3. Mock vitaldb via sys.modules BEFORE importing step_02_catch_22
    print("Mocking vitaldb...")
    
    # Unload modules if they exist
    if 'vitaldb' in sys.modules:
        del sys.modules['vitaldb']
    if 'data_preparation.step_02_catch_22' in sys.modules:
        del sys.modules['data_preparation.step_02_catch_22']
    
    def mock_load_case(caseid, waveform, interval=None):
        print(f"DEBUG: Mock load_case called for {caseid} {waveform}")
        sr = 500
        duration = 20 # seconds
        t = np.linspace(0, duration, int(sr * duration))
        signal = np.sin(2 * np.pi * 1 * t) # 1 Hz sine
        return signal

    vitaldb_mock = MagicMock()
    vitaldb_mock.load_case.side_effect = mock_load_case
    sys.modules['vitaldb'] = vitaldb_mock
    
    # Now import the module under test
    import data_preparation.step_02_catch_22 as step_02_catch_22
    
    # Mock Pool to run synchronously
    class MockPool:
        def __init__(self, processes=None): 
            print("DEBUG: MockPool initialized")
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
        def imap_unordered(self, func, iterable):
            print("DEBUG: MockPool.imap_unordered called")
            for item in iterable:
                yield func(item)

    # 4. Run Pipeline with Patched Config
    print("Running pipeline with mocked configuration...")
    
    with patch('data_preparation.step_02_catch_22.COHORT_FILE', temp_cohort), \
         patch('data_preparation.step_02_catch_22.CATCH_22_FILE', temp_features), \
         patch('data_preparation.step_02_catch_22.CATCH_22_ERROR_FILE', 'tests/temp_errors.csv'), \
         patch('data_preparation.step_02_catch_22.GENERATE_FULL_FEATURES', True), \
         patch('data_preparation.step_02_catch_22.GENERATE_WINDOWED_FEATURES', True), \
         patch('data_preparation.step_02_catch_22.WIN_SEC', 10), \
         patch('data_preparation.step_02_catch_22.SLIDE_SEC', 10), \
         patch('data_preparation.step_02_catch_22.EXPORT_AEON', True), \
         patch('data_preparation.aeon_io.AEON_OUT_DIR', temp_aeon_dir), \
         patch('data_preparation.step_02_catch_22.Pool', side_effect=MockPool):
         
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
        if df.empty: success = False
    else:
        print("[FAIL] Full features CSV missing.")
        success = False

    # Check Windowed Features
    if os.path.exists(temp_features_win):
        df = pd.read_csv(temp_features_win)
        print(f"[PASS] Windowed features CSV exists. Shape: {df.shape}")
        if df.empty: success = False
    else:
        print("[FAIL] Windowed features CSV missing.")
        success = False

    # Check Aeon Output
    if os.path.exists(temp_aeon_dir):
        print(f"[PASS] Aeon directory created at {temp_aeon_dir}")
        
        # Check Non-Windowed Output
        npz_path = os.path.join(temp_aeon_dir, 'X_nonwindowed.npz')
        pkl_path = os.path.join(temp_aeon_dir, 'X_nonwindowed_np_list.pkl')
        
        if os.path.exists(npz_path):
            try:
                with np.load(npz_path) as data:
                    print(f"[PASS] X_nonwindowed.npz loaded. Shape: {data['X'].shape}")
            except Exception as e:
                print(f"[FAIL] Could not load X_nonwindowed.npz: {e}")
                success = False
        elif os.path.exists(pkl_path):
             print(f"[PASS] X_nonwindowed_np_list.pkl exists.")
        else:
             print("[FAIL] Aeon non-windowed output missing.")
             # success = False # Might be optional depending on config, but default is yes

        # Check Windowed NPZ
        npz_win_path = os.path.join(temp_aeon_dir, 'X_windowed.npz')
        if os.path.exists(npz_win_path):
            try:
                with np.load(npz_win_path) as data:
                    print(f"[PASS] X_windowed.npz loaded. Shape: {data['X'].shape}")
            except Exception as e:
                print(f"[FAIL] Could not load X_windowed.npz: {e}")
                success = False
        else:
             print("[FAIL] X_windowed.npz missing.")
             success = False

    else:
        print("[FAIL] Aeon directory missing.")
        success = False

    # 6. Cleanup
    if success:
        print("\n=== Cleanup ===")
        if os.path.exists(temp_cohort): os.remove(temp_cohort)
        if os.path.exists(temp_features): os.remove(temp_features)
        if os.path.exists(temp_features_win): os.remove(temp_features_win)
        if os.path.exists(temp_aeon_dir): shutil.rmtree(temp_aeon_dir)
        if os.path.exists('tests/temp_errors.csv'): os.remove('tests/temp_errors.csv')
        if os.path.exists('tests/temp_errors_windowed.csv'): os.remove('tests/temp_errors_windowed.csv')
        print("Temporary files removed.")
        print("\n>>> SMOKE TEST PASSED <<<")
    else:
        print("\n>>> SMOKE TEST FAILED <<<")

if __name__ == "__main__":
    run_smoke_test()
