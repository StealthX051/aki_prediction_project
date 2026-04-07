import os
from pathlib import Path

from artifact_paths import get_data_dir, get_processed_dir, get_raw_dir, get_results_dir
from data_preparation.outcome_registry import DEFAULT_OUTCOME_COLUMN

# --- Project Paths ---
# Resolve project root relative to this file (data_preparation/inputs.py)
# This file is in <root>/data_preparation, so we go up two levels.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = get_data_dir(PROJECT_ROOT)
RAW_DIR = get_raw_dir(PROJECT_ROOT)
PROCESSED_DIR = get_processed_dir(PROJECT_ROOT)

# --- User Parameters ---
INPUT_FILE = Path(os.getenv("INPUT_FILE", RAW_DIR / 'clinical_data.csv'))
LAB_DATA_FILE = Path(os.getenv("LAB_DATA_FILE", RAW_DIR / 'lab_data.csv'))
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}.csv
COHORT_FILE = Path(os.getenv("COHORT_FILE", PROCESSED_DIR / 'aki_pleth_ecg_co2_awp.csv'))
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}.csv
CATCH_22_FILE = Path(os.getenv("CATCH_22_FILE", PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf.csv'))
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}_errors.csv
CATCH_22_ERROR_FILE = Path(os.getenv("CATCH_22_ERROR_FILE", PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf_errors.csv'))
# List of departments you want to include (lowercase), or None to include all
# list should look like: ['general surgery', 'thoracic surgery']
DEPARTMENTS = None 
# Add required columns to this list (ie your outcome variables)
OUTER_COHORT_MANDATORY_COLUMNS = ['opend']
AKI_COHORT_FLOW_MANDATORY_COLUMNS = ['preop_cr', 'opend']
MANDATORY_COLUMNS = AKI_COHORT_FLOW_MANDATORY_COLUMNS
# Add required waveforms to this list. needs to be the name of the track in VitalDB
MANDATORY_WAVEFORMS = ['SNUADC/PLETH', 'SNUADC/ECG_II', 'Primus/CO2', 'Primus/AWP']
# If a required waveform is missing, but another waveform in the substitution list is present, use that instead
# Set acceptable substitutions here
WAVEFORM_SUBSTITUTIONS = {'SNUADC/ECG_II': ['SNUADC/ECG_V5'],}
# window length and slide length for catch-22 feature extraction (in seconds). None is entire waveform
WIN_SEC = 10
SLIDE_SEC = 5
# Configuration for feature extraction modes
# You can enable both to generate both full-segment and windowed features in one run.
GENERATE_FULL_FEATURES = os.getenv("GENERATE_FULL_FEATURES", "True").lower() == "true"
GENERATE_WINDOWED_FEATURES = os.getenv("GENERATE_WINDOWED_FEATURES", "False").lower() == "true"
# target sampling frequency for resampling (in Hz)
TARGET_SR = 10
# Sampling rate for full-case feature extraction (Non-Windowed)
# Reduced to 10 Hz to manage computational complexity of catch22 on long sequences.
FULL_FEATURE_TARGET_SR = 10
# name of column for outcome variable
OUTCOME = DEFAULT_OUTCOME_COLUMN

# === Experimental Aeon controls ===
AEON_OUT_DIR = os.getenv("AEON_OUT_DIR", "outputs/aeon")
AEON_SAVE_FORMATS = ["nested_pkl", "numpy3d_npz", "np_list_pkl"]
AEON_PAD_POLICY = "in_memory_pad" # 'aeon_padding_transformer','in_memory_pad','none'
AEON_PAD_LENGTH = 57600
AEON_PAD_FILL = 0
AEON_STRICT_CHANNELS = True
AEON_WINDOW_POLICY = "intersection"
AEON_FIXED_LENGTH = 57600  # Target length (16 hours @ 1 Hz)

# --- Imputation Controls ---
# Controls whether preprocessing steps should impute missing values.
# By default, preprocessing will leave NaNs in place so downstream steps can
# decide how to handle them. Set to True to restore the previous -99/imputed
# behavior in step_03/step_04/step_05.
IMPUTE_MISSING = False

# --- Step 03 Output Paths ---
PREOP_PROCESSED_FILE = Path(os.getenv("PREOP_PROCESSED_FILE", PROCESSED_DIR / 'aki_preop_processed.csv'))
CATCH_22_WINDOWED_FILE = Path(os.getenv("CATCH_22_WINDOWED_FILE", PROCESSED_DIR / 'aki_pleth_ecg_co2_awp_inf_windowed.csv'))

# --- Step 04 Output Paths ---
INTRAOP_WIDE_FILE = Path(os.getenv("INTRAOP_WIDE_FILE", PROCESSED_DIR / 'aki_intraop_wide.csv'))
INTRAOP_WIDE_WINDOWED_FILE = Path(os.getenv("INTRAOP_WIDE_WINDOWED_FILE", PROCESSED_DIR / 'aki_intraop_wide_windowed.csv'))

# --- Step 05 Output Paths ---
WIDE_FEATURES_FILE = Path(os.getenv("WIDE_FEATURES_FILE", PROCESSED_DIR / 'aki_features_master_wide.csv'))
WIDE_FEATURES_WINDOWED_FILE = Path(os.getenv("WIDE_FEATURES_WINDOWED_FILE", PROCESSED_DIR / 'aki_features_master_wide_windowed.csv'))

RESULTS_DIR = get_results_dir(PROJECT_ROOT)
