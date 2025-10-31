# --- User Parameters ---
INPUT_FILE = './data/raw/clinical_data.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}.csv
COHORT_FILE = './data/processed/aki_pleth_ecg_co2_awp.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}.csv
CATCH_22_FILE = './data/processed/aki_pleth_ecg_co2_awp_inf.csv'
# {outcome}_{waveform1}_...{waveformN}_{dept1}---{dept_N}_{window_length}_errors.csv
CATCH_22_ERROR_FILE = './data/processed/aki_pleth_ecg_co2_awp_inf_errors.csv'
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
WIN_SEC = None
SLIDE_SEC = None
# target sampling frequency for resampling (in Hz)
TARGET_SR = 10
# name of column for outcome variable
OUTCOME = 'aki_label'
