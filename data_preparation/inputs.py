from pathlib import Path

# --- Project Paths ---
# Resolve project root relative to this file (data_preparation/inputs.py)
# This file is in <root>/data_preparation, so we go up two levels.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# --- User Parameters ---
INPUT_FILE = RAW_DIR / 'clinical_data.csv'
LAB_DATA_FILE = RAW_DIR / 'lab_data.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}.csv
COHORT_FILE = PROCESSED_DIR / 'aki_pleth_ecg_co2_awp.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}.csv
CATCH_22_FILE = PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}_errors.csv
CATCH_22_ERROR_FILE = PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf_errors.csv'
# List of departments you want to include (lowercase), or None to include all
# list should look like: ['general surgery', 'thoracic surgery']
DEPARTMENTS = None 
# Add required columns to this list (ie your outcome variables)
MANDATORY_COLUMNS = ['preop_cr', 'opend']
# Add required waveforms to this list. needs to be the name of the track in VitalDB
MANDATORY_WAVEFORMS = ['SNUADC/PLETH', 'SNUADC/ECG_II', 'Primus/CO2', 'Primus/AWP']
# If a required waveform is missing, but another waveform in the substitution list is present, use that instead
# Set acceptable substitutions here
WAVEFORM_SUBSTITUTIONS = {'SNUADC/ECG_II': ['SNUADC/ECG_V5'],}
# custom filters to apply (import from custom_filters folder)
from data_preparation.custom_filters.preop_cr import filter_preop_cr
from data_preparation.custom_filters.sampling_independence import ensure_sample_independence
from data_preparation.custom_filters.postop_cr import filter_postop_cr
from data_preparation.custom_filters.aki_label import add_aki_label
CUSTOM_FILTERS = [filter_preop_cr, ensure_sample_independence, filter_postop_cr, add_aki_label]
# window length and slide length for catch-22 feature extraction (in seconds). None is entire waveform
WIN_SEC = 10
SLIDE_SEC = 5
# Configuration for feature extraction modes
# You can enable both to generate both full-segment and windowed features in one run.
GENERATE_FULL_FEATURES = True
GENERATE_WINDOWED_FEATURES = False
# target sampling frequency for resampling (in Hz)
TARGET_SR = 10
# Sampling rate for full-case feature extraction (Non-Windowed)
# Reduced to 10 Hz to manage computational complexity of catch22 on long sequences.
FULL_FEATURE_TARGET_SR = 10
# name of column for outcome variable
OUTCOME = 'aki_label'

import os

# === NEW: AEON export controls ===
EXPORT_AEON = False
AEON_OUT_DIR = os.getenv("AEON_OUT_DIR", "outputs/aeon")
AEON_SAVE_FORMATS = ["nested_pkl", "numpy3d_npz", "np_list_pkl"]
AEON_PAD_POLICY = "aeon_padding_transformer" # 'aeon_padding_transformer','in_memory_pad','none'
AEON_PAD_LENGTH = None
AEON_PAD_FILL = 0
AEON_STRICT_CHANNELS = True
AEON_COMMON_SR = 100.0  # Kept for reference, but fixed length overrides this in step_02_aeon.
AEON_WINDOW_POLICY = "intersection"
AEON_FIXED_LENGTH = 8000  # Target length for fixed-length resampling (approx 1 Hz for long ops)

# --- Step 03 Output Paths ---
PREOP_PROCESSED_FILE = PROCESSED_DIR / 'aki_preop_processed.csv'
CATCH_22_WINDOWED_FILE = PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf_windowed.csv'

# --- Step 04 Output Paths ---
INTRAOP_WIDE_FILE = PROCESSED_DIR / 'aki_intraop_wide.csv'
INTRAOP_WIDE_WINDOWED_FILE = PROCESSED_DIR / 'aki_intraop_wide_windowed.csv'

# --- Step 05 Output Paths ---
WIDE_FEATURES_FILE = PROCESSED_DIR / 'aki_features_master_wide.csv'
WIDE_FEATURES_WINDOWED_FILE = PROCESSED_DIR / 'aki_features_master_wide_windowed.csv'

RESULTS_DIR = PROJECT_ROOT / 'results'
