# =============================================================================
# 11_unit_test_from_user_script.py
#
# Description:
# A focused, self-contained unit test to verify that the core feature
# extraction logic from the user-provided '7_..._fast.py' script
# works correctly for a single case.
#
# How to Run:
# 1. Run from the command line: python 11_unit_test_from_user_script.py
# =============================================================================

# --- Standard Library Imports ---
import os
import sys
import logging
import unittest
import pandas as pd
import numpy as np

# --- Third-Party Imports for the function being tested ---
import vitaldb
import pycatch22

# =============================================================================
# FUNCTION TO BE TESTED
# (Copied EXACTLY from the user's '7_..._fast.py' script)
# =============================================================================

# --- Parameters for the function ---
WAVEFORM = ['SNUADC/PLETH']
TARGET_SR = 100
WIN_SEC = 10 * 60
SLIDE_SEC = 5 * 60
WIN_SAMP = WIN_SEC * TARGET_SR
SLIDE_SAMP = SLIDE_SEC * TARGET_SR

def _process_case(case):
    """
    Worker function to process a single case.
    This is the core logic from the user-provided script.
    """
    caseid, opstart, opend, death = case
    try:
        # Using vitaldb.load_case as seen in the user's script
        wave = vitaldb.load_case(caseid, WAVEFORM, 1 / TARGET_SR)
        if wave is None or wave.size == 0:
            return {'caseid': caseid, 'error': 'empty'}

        seg = wave[int(opstart * TARGET_SR):int(opend * TARGET_SR)]
        if seg.size < WIN_SAMP:
            return {'caseid': caseid, 'error': 'too_short'}

        feats_list = []
        for i in range(0, seg.size - WIN_SAMP + 1, SLIDE_SAMP):
            win = seg[i:i + WIN_SAMP]
            if np.isnan(win).all() or np.nanstd(win) < 1e-6:
                continue
            all_feature_results = pycatch22.catch22_all(win, catch24=False)
            feats_list.append(all_feature_results)

        if not feats_list:
            return {'caseid': caseid, 'error': 'no_valid_window'}

        feature_names = feats_list[0]['names']
        feature_values = [res['values'] for res in feats_list]

        m = np.asarray(feature_values, dtype=np.float32)
        out = {
            **{f'{n}_mean': v for n, v in zip(feature_names, m.mean(0))},
            **{f'{n}_std': v for n, v in zip(feature_names, m.std(0))},
            **{f'{n}_min': v for n, v in zip(feature_names, m.min(0))},
            **{f'{n}_max': v for n, v in zip(feature_names, m.max(0))}
        }
        out['caseid'] = caseid
        out['death_label'] = death
        return out

    except Exception as exc:
        return {'caseid': caseid, 'error': str(exc)}


# =============================================================================
# 1. TEST CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- Select a Test Case ---
TEST_CASE_ID = 1
TEST_OPSTART = 1668
TEST_OPEND = 10368
TEST_DEATH_LABEL = 0 # From final_cohort_with_death_label.csv for caseid 1

# --- Paths ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
except NameError:
    PROJECT_ROOT = os.getcwd()
    if os.path.basename(PROJECT_ROOT) == 'scripts':
        PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

VITAL_FILES_DIR = os.path.join(PROJECT_ROOT, "data", "raw", 'vital_files')

# Expected columns: 22 features * 4 stats + caseid + death_label
EXPECTED_NUM_COLUMNS = (22 * 4) + 2


# =============================================================================
# 2. UNIT TEST CLASS
# =============================================================================
class TestCatch22FeatureExtraction(unittest.TestCase):

    def test_single_case_processing(self):
        """
        Tests the _process_case function on a single, known-good case.
        """
        logging.info(f"--- Running Unit Test for caseid: {TEST_CASE_ID} ---")
        
        # --- 1. Arrange ---
        self.assertTrue(os.path.exists(VITAL_FILES_DIR),
                        f"FATAL: Vital files directory not found at: {VITAL_FILES_DIR}")
        
        # The user's script passes a tuple of (caseid, opstart, opend, death_label)
        case_info = (TEST_CASE_ID, TEST_OPSTART, TEST_OPEND, TEST_DEATH_LABEL)

        # --- 2. Act ---
        result = _process_case(case_info)
        
        # --- 3. Assert ---
        logging.info(f"Function returned a dictionary with {len(result)} keys.")

        self.assertNotIn('error', result, f"The function returned an error: {result.get('error')}")
        self.assertIsInstance(result, dict, "The function did not return a dictionary.")
        self.assertEqual(len(result.keys()), EXPECTED_NUM_COLUMNS,
                         f"Expected {EXPECTED_NUM_COLUMNS} columns but got {len(result.keys())}")
        self.assertEqual(result['caseid'], TEST_CASE_ID,
                         "The caseid in the result does not match the input caseid.")
        
        for key, value in result.items():
            self.assertFalse(pd.isna(value), f"Found a NaN value for feature '{key}'")

        logging.info("--- Unit Test Passed Successfully! ---")
        logging.info("The core '_process_case' function from your script is working correctly.")


# =============================================================================
# 3. SCRIPT EXECUTION
# =============================================================================
if __name__ == '__main__':
    # Set the current working directory, vitaldb needs this to find the data folder
    os.chdir(PROJECT_ROOT)
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
