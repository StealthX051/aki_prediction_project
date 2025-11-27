import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass
import numpy as np
import pandas as pd
import vitaldb
import pycatch22
from tqdm import tqdm
import time
from collections import defaultdict
from scipy.signal import butter, filtfilt, resample_poly
import math
import joblib

"""
CATCH-22 & AEON WAVEFORM PREPROCESSING PIPELINE

DESCRIPTION:
This script runs a parallelized data processing pipeline on the VitalDB cohort.
Its primary functions are twofold, controlled by settings in 'inputs.py':

1. CATCH-22 FEATURE EXTRACTION:
   - Loads a cohort from COHORT_FILE.
   - For each case and each MANDATORY_WAVEFORM:
     - Loads the native signal from VitalDB.
     - Applies modality-specific, morphology-preserving filters (e.g., BPF, LPF).
     - Anti-aliases and downsamples to a target sample rate (e.g., 500Hz -> 100Hz).
     - Extracts catch22 features, either for the full segment or in sliding windows.
   - Saves results to a CSV (CATCH_22_FILE).
   - Saves errors to a separate CSV (CATCH_22_ERROR_FILE).
   - This process is fully checkpointed and will append to existing files.

2. AEON-COMPATIBLE EXPORT (if EXPORT_AEON = True):
   - In parallel, it saves the *processed waveforms* (post-filter/resample)
     from step 1.
   - Collates and saves these waveforms in formats suitable for `aeon`
     (e.g., MultiRocket).
   - Supports non-windowed (full segment) and windowed exports.
   - Handles padding, channel alignment (intersection), and sample rate
     harmonization (AEON_COMMON_SR).

INPUTS (from data_preparation.inputs.py):
- COHORT_FILE: CSV of cases with 'caseid', 'opstart', 'opend', and OUTCOME.
- MANDATORY_WAVEFORMS: List of waveform keys to process (e.g., ['ART', 'PLETH']).
- WAVEFORM_SUBSTITUTIONS: Dict for fallback (e.g., {'ART': ['ABP', 'FEM']}).
- WIN_SEC / SLIDE_SEC: Defines windowing for both pipelines. If None,
  processes the full segment.
- EXPORT_AEON: Master switch to enable/disable the Aeon export.
- AEON_...: Configuration for Aeon export (paths, padding, formats).

OUTPUTS:
- CATCH_22_FILE: CSV containing extracted catch22 features.
  (e.g., "outputs/catch22.csv" or "outputs/catch22_windowed.csv")
- CATCH_22_ERROR_FILE: CSV logging any errors (e.g., missing signal).
- AEON_OUT_DIR: Directory (if EXPORT_AEON=True) containing:
  - `X_...pkl / .npz`: Serialized waveform data.
  - `y_...csv`: Corresponding labels.
  - `README.md` & example training scripts.
"""

# Assuming 'data_preparation.inputs' exists in the same environment
# We define dummy values here for demonstration if inputs.py is not present
try:
    from data_preparation.inputs import (
        COHORT_FILE, 
        CATCH_22_FILE,
        CATCH_22_ERROR_FILE,
        MANDATORY_WAVEFORMS, 
        WAVEFORM_SUBSTITUTIONS, 
        WIN_SEC,
        SLIDE_SEC,
        TARGET_SR, # Note: TARGET_SR is no longer used for loading, only for legacy check
        OUTCOME,
        
        # === NEW: AEON export controls ===
        EXPORT_AEON,
        AEON_OUT_DIR,
        AEON_SAVE_FORMATS,
        AEON_PAD_POLICY,
        AEON_PAD_LENGTH,
        AEON_PAD_FILL,
        AEON_STRICT_CHANNELS,
        AEON_COMMON_SR,
        AEON_WINDOW_POLICY
    )
except ImportError:
    logging.warning("Could not import from data_preparation.inputs. Using dummy values.")
    logging.warning("THIS SCRIPT WILL NOT RUN without a valid inputs file.")
    COHORT_FILE = 'dummy_cohort.csv'
    CATCH_22_FILE = 'dummy_catch22.csv'
    CATCH_22_ERROR_FILE = 'dummy_catch22_errors.csv'
    MANDATORY_WAVEFORMS = ['ART', 'PLETH', 'ECG_II']
    WAVEFORM_SUBSTITUTIONS = {'ART': ['ABP', 'FEM']} # Added FEM as per spec
    WIN_SEC = 60 # or e.g., 60
    SLIDE_SEC = 30 # or e.g., 30
    TARGET_SR = 100 # This is now unused, but we keep it to show inputs.py is loaded
    OUTCOME = 'outcome'
    
    # === NEW: AEON dummy values ===
    EXPORT_AEON = True
    AEON_OUT_DIR = "outputs/aeon"
    AEON_SAVE_FORMATS = ["nested_pkl", "numpy3d_npz", "np_list_pkl"]
    AEON_PAD_POLICY = "aeon_padding_transformer" # 'aeon_padding_transformer','in_memory_pad','none'
    AEON_PAD_LENGTH = None
    AEON_PAD_FILL = 0
    AEON_STRICT_CHANNELS = True
    AEON_COMMON_SR = 100.0
    AEON_WINDOW_POLICY = "intersection"


# === NEW: Signal Processing Helpers ===
# Note: Using Butterworth, order 4 (unless specified), zero-phase with filtfilt
# to preserve morphology, as recommended in analysis.
def bpf(x, fs, lo, hi, order=4):
    """Zero-phase band-pass filter"""
    nyquist = fs / 2
    b, a = butter(order, [lo / nyquist, hi / nyquist], btype='band')
    return filtfilt(b, a, x)

def lpf(x, fs, fc, order=4):
    """Zero-phase low-pass filter"""
    nyquist = fs / 2
    b, a = butter(order, fc / nyquist, btype='low')
    return filtfilt(b, a, x)

def hpf(x, fs, fc, order=2):
    """Zero-phase high-pass filter"""
    nyquist = fs / 2
    b, a = butter(order, fc / nyquist, btype='high')
    return filtfilt(b, a, x)

def aa_downsample(x, up, down):
    """Anti-aliased resampling"""
    return resample_poly(x, up, down)

# === NEW: Aeon Export Helpers ===
@dataclass
class AeonSeriesPayload:
    caseid: int
    waveform: str
    target_sr: float
    seg_full: Optional[np.ndarray] = None               # non-windowed
    win_mat: Optional[np.ndarray] = None                # shape (n_windows, win_samp) if windowed, else None
    valid_window_mask: Optional[np.ndarray] = None      # boolean mask length = n_windows
    length: Optional[int] = None                        # seg_full length or win_samp

def harmonize_sr(x: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    """Resample x from src_sr to dst_sr using polyphase (resample_poly)."""
    if src_sr == dst_sr:
        return x
    
    # Calculate rational fraction for resampling
    g = math.gcd(int(src_sr * 100), int(dst_sr * 100))
    up = int(dst_sr * 100) // g
    down = int(src_sr * 100) // g
    
    return aa_downsample(x, up, down)

def pad_1d(x: np.ndarray, pad_len: int, fill_value) -> np.ndarray:
    """Right-pad 1D array to pad_len."""
    if x.shape[0] >= pad_len:
        return x[:pad_len]
    
    # np.pad can't use 'mean', 'median' etc. directly.
    # We'll handle fill_value logic during collation
    if not isinstance(fill_value, (int, float)):
        # Default to 0 if a string like 'mean' is passed here
        fill_value = 0 
        
    return np.pad(x, (0, pad_len - x.shape[0]), mode='constant', constant_values=fill_value)

# === NEW: Waveform Processing Specifications ===
# Maps keys (used in inputs.py) to their VitalDB track IDs, native SR,
# target SR (after resampling), and filtering parameters.
# IDs & sample rates from VitalDB docs / dataset (Lee 2022, Sci Data; Lee & Jung 2018, Sci Rep).
# Use anti-aliased resampling (scipy.signal.resample_poly; SciPy Docs [v1.16]).
WAVEFORM_SPECS = {
    # 500 Hz signals -> 100 Hz
    'ECG_II': {'id': 'SNUADC/ECG_II', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.5, 40)},  # 0.5–40 Hz morphology for ML; cf. diagnostic 0.05–150 (Kligfield 2007, Circulation); QRS 5–15 (Pan & Tompkins 1985, IEEE TBME)
    'ECG_V5': {'id': 'SNUADC/ECG_V5', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.5, 40)},  # same (Kligfield 2007; Pan & Tompkins 1985)

    'PLETH':  {'id': 'SNUADC/PLETH', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.1, 10)},  # 0.1–10 Hz preserves notch/phase; <10 dampens notch (Lapitan 2024, Sci Rep; Park 2022, Front Physiol)

    'ART':    {'id': 'SNUADC/ART', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.3, 30)},  # ABP morphology incl. dicrotic notch; arterial content to ~20–25 Hz (Watanabe 2020, J Anesth; Pal 2024, CMPB)
    'FEM':    {'id': 'SNUADC/FEM', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.3, 30)},  # same as ART (Watanabe 2020; Pal 2024)
    'CVP':    {'id': 'SNUADC/CVP', 'native_sr': 500, 'target_sr': 100, 'up': 1, 'down': 5,
               'filter': ('bpf', 0.05, 10)}, # retain respiratory (~0.2–0.3 Hz) + cardiac (~1–2 Hz) components (Magder 2015, Curr Opin Crit Care)

    # 62.5 Hz signals -> 62.5 Hz (no resampling)
    'AWP':    {'id': 'Primus/AWP', 'native_sr': 62.5, 'target_sr': 62.5, 'up': 1, 'down': 1,
               'filter': ('lpf', 12)},       # LPF ~10–15 Hz keeps breath shape/PEEP (de Haro 2024, Crit Care; Thome 1998, J Appl Physiol)
    'CO2':    {'id': 'Primus/CO2', 'native_sr': 62.5, 'target_sr': 62.5, 'up': 1, 'down': 1,
               'filter': ('lpf', 8)},        # Morphology LPF 5–10 Hz; for ventilation detection use ≤1–2 Hz (Gutiérrez 2018, PLoS One; Leturiondo 2017, CinC)

    # 128 Hz signals -> 128 Hz (no resampling)
    'EEG1':   {'id': 'BIS/EEG1_WAV', 'native_sr': 128, 'target_sr': 128, 'up': 1, 'down': 1,
               'filter': ('bpf', 0.5, 30)},  # anesthesia EEG preproc 0.5–30/40 Hz is common (Schmidlin 2001, Br J Anaesth; Nagaraj 2018, PLoS One)
    'EEG2':   {'id': 'BIS/EEG2_WAV', 'native_sr': 128, 'target_sr': 128, 'up': 1, 'down': 1,
               'filter': ('bpf', 0.5, 30)},  # same (Schmidlin 2001; Nagaraj 2018)

    # 180 Hz signals -> 90 Hz
    'ABP':    {'id': 'CardioQ/ABP', 'native_sr': 180, 'target_sr': 90, 'up': 1, 'down': 2,
               'filter': ('bpf', 0.3, 30)},  # as invasive ABP; retain notch/harmonics (Watanabe 2020, J Anesth; Lam 2021, Cureus)
    'FLOW':   {'id': 'CardioQ/FLOW', 'native_sr': 180, 'target_sr': 90, 'up': 1, 'down': 2,
               'filter': ('bpf', 0.5, 20)},  # esophageal Doppler flow: suppress drift, keep systolic accel/FTc features (Deltex 2018, Operating Handbook)
}
# === END NEW ===


# WIN_SAMP and SLIDE_SAMP are now calculated *inside* _process_case
# based on the waveform's specific target_sr.
assert (WIN_SEC is None and SLIDE_SEC is None) or (WIN_SEC is not None and SLIDE_SEC is not None), "Invalid window/slide settings"

def _process_case(case: Tuple[int, float, float, int, str]) -> Tuple[Dict[str, Any], Dict[str, float], Optional[AeonSeriesPayload]]:
    """
    Processes a single case/waveform pair to extract catch22 features
    and optionally, the processed waveform for Aeon export.
    """
    caseid, opstart, opend, outcome, waveform_key = case # Renamed for clarity
    timings = {}
    aeon_payload: Optional[AeonSeriesPayload] = None
    start = time.perf_counter()

    try:
        wave = None
        spec_key_loaded = None # This will be 'ART', 'ABP', 'FEM', etc.
        spec = None

        # 1. Try loading the primary waveform key
        if waveform_key in WAVEFORM_SPECS:
            try:
                spec = WAVEFORM_SPECS[waveform_key]
                # Load native signal, no interval
                wave = vitaldb.load_case(caseid, spec['id'], interval=None) 
                if wave is not None and wave.size > 0:
                    spec_key_loaded = waveform_key
            except Exception as e:
                logging.debug(f"Case {caseid}: Primary load {waveform_key} ({spec['id']}) failed. {e}")
                pass # Will try substitutions

        # 2. If primary failed, try substitutions
        if wave is None or wave.size == 0:
            for sub_key in WAVEFORM_SUBSTITUTIONS.get(waveform_key, []):
                if sub_key in WAVEFORM_SPECS:
                    try:
                        spec = WAVEFORM_SPECS[sub_key]
                        # Load native signal, no interval
                        wave = vitaldb.load_case(caseid, spec['id'], interval=None)
                        if wave is not None and wave.size > 0:
                            spec_key_loaded = sub_key
                            logging.debug(f"Case {caseid}: Used substitute {sub_key} for {waveform_key}")
                            break # Found a valid substitute
                    except Exception as e:
                        logging.debug(f"Case {caseid}: Substitute load {sub_key} ({spec['id']}) failed. {e}")
                        continue # Try next substitute
                else:
                    logging.warning(f"Case {caseid}: Unknown substitution key {sub_key}")

        # 3. Check if loading failed entirely
        if wave is None or wave.size == 0:
            return {'caseid': caseid, 'waveform': waveform_key, 'error': 'empty_signal_or_missing'}, timings, None
        
        # 4. We have a wave and its spec
        spec = WAVEFORM_SPECS[spec_key_loaded]
        end = time.perf_counter()
        timings['load_case'] = end - start
        start = end

        # 5. Slicing at NATIVE sample rate
        native_sr = spec['native_sr']
        seg = wave[int(opstart * native_sr):int(opend * native_sr)]
        seg = seg.squeeze()
        
        # 6. NaN checks (on native signal)
        nan_mask = np.isnan(seg)
        nan_pct = np.mean(nan_mask)
        if nan_pct > 0.05:
            return {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_gt_5_pct_nan'}, timings, None
        
        if nan_pct > 0: # If < 5% NaNs, interpolate
            x = np.arange(seg.size)
            seg = np.interp(
                x=x,
                xp=x[~nan_mask],
                fp=seg[~nan_mask]
            )
        
        # 7. NEW: Filtering (at native SR)
        try:
            filt_type, *params = spec['filter']
            if filt_type == 'bpf':
                seg = bpf(seg, fs=native_sr, lo=params[0], hi=params[1])
            elif filt_type == 'lpf':
                seg = lpf(seg, fs=native_sr, fc=params[0])
            elif filt_type == 'hpf':
                seg = hpf(seg, fs=native_sr, fc=params[0])
        except ValueError as e:
            # Handle signals too short for filtering
            logging.warning(f"Case {caseid} {waveform_key}: Filtering error (signal likely too short). {e}")
            return {'caseid': caseid, 'waveform': waveform_key, 'error': f'filtering_error_signal_too_short'}, timings, None
        except Exception as e:
            logging.error(f"Case {caseid} {waveform_key}: Unknown filtering error. {e}")
            return {'caseid': caseid, 'waveform': waveform_key, 'error': f'filtering_error_unknown'}, timings, None

        # 8. NEW: Anti-alias downsampling
        target_sr = spec['target_sr']
        if spec['up'] != 1 or spec['down'] != 1:
            try:
                seg = aa_downsample(seg, spec['up'], spec['down'])
            except Exception as e:
                logging.error(f"Case {caseid} {waveform_key}: Resampling error. {e}")
                return {'caseid': caseid, 'waveform': waveform_key, 'error': f'resampling_error_{e}'}, timings, None

        # 9. NEW: Calculate windowing params LOCALLY
        win_samp = int(WIN_SEC * target_sr) if WIN_SEC else None
        slide_samp = int(SLIDE_SEC * target_sr) if SLIDE_SEC else None

        # === NEW: 10. Create Aeon Payload ===
        if EXPORT_AEON:
            try:
                if win_samp is None:
                    # Non-windowed: Export the full segment
                    seg_out = seg.astype(np.float32, copy=False)
                    current_sr = target_sr
                    if AEON_COMMON_SR:
                        seg_out = harmonize_sr(seg_out, target_sr, AEON_COMMON_SR)
                        current_sr = AEON_COMMON_SR
                    
                    aeon_payload = AeonSeriesPayload(
                        caseid=caseid, waveform=waveform_key,
                        target_sr=current_sr, seg_full=seg_out, length=seg_out.size
                    )
                else:
                    # Windowed: Build window matrix and valid mask
                    win_vals: List[np.ndarray] = []
                    valid_mask: List[bool] = []
                    
                    n_potential_windows = (seg.size - win_samp + slide_samp) // slide_samp if seg.size >= win_samp else 0
                    
                    current_sr = target_sr
                    final_win_samp = win_samp

                    # Pre-calculate window indices
                    window_starts = range(0, seg.size - win_samp + 1, slide_samp)
                    
                    # Store valid windows
                    for i in window_starts:
                        win = seg[i:i+win_samp]
                        is_valid = (not np.isnan(win).all()) and (np.nanstd(win) >= 1e-6)
                        valid_mask.append(is_valid)
                        if is_valid:
                            win_vals.append(win.astype(np.float32, copy=False))
                    
                    # Fill remainder of mask if any
                    valid_mask.extend([False] * (n_potential_windows - len(valid_mask)))

                    win_mat: np.ndarray
                    if win_vals:
                        if AEON_COMMON_SR and target_sr != AEON_COMMON_SR:
                            # Resample each valid window
                            win_vals = [harmonize_sr(w, target_sr, AEON_COMMON_SR) for w in win_vals]
                            current_sr = AEON_COMMON_SR
                            final_win_samp = win_vals[0].shape[0] # Update win_samp to new length

                        win_mat = np.vstack(win_vals)
                    else:
                        # No valid windows, create empty matrix with correct second dim
                        if AEON_COMMON_SR:
                            final_win_samp = int(win_samp * (AEON_COMMON_SR / target_sr))
                        win_mat = np.empty((0, final_win_samp), dtype=np.float32)

                    aeon_payload = AeonSeriesPayload(
                        caseid=caseid, waveform=waveform_key,
                        target_sr=current_sr, win_mat=win_mat,
                        valid_window_mask=np.array(valid_mask, dtype=bool),
                        length=final_win_samp
                    )

            except Exception as e:
                logging.error(f"Case {caseid} {waveform_key}: Failed to create Aeon payload. {e}")
                # Do not fail the whole process, just skip aeon export for this case
                aeon_payload = None 

        # 11. Run feature extraction (this logic is now on the processed signal)

        # Mode 1: Process entire segment as one window
        if win_samp is None and slide_samp is None:
            if np.isnan(seg).all() or np.nanstd(seg) < 1e-6: 
                end = time.perf_counter()
                timings['signal_empty'] = end - start
                return {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_flatline_or_all_nan'}, timings, None
            
            all_feature_results = pycatch22.catch22_all(seg, catch24=True)
            out = {n: v for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
            out['caseid'] = caseid
            out['waveform'] = waveform_key # Use original key for consistency
            out[OUTCOME] = outcome 
            end = time.perf_counter()
            timings['process_signal_full'] = end - start
            return out, timings, aeon_payload
        
        # Mode 2: Process using sliding windows
        else:
            assert win_samp is not None and slide_samp is not None # From local calculation
            if seg.size < win_samp:
                return {'caseid': caseid, 'waveform': waveform_key, 'error': 'segment_too_short_for_window'}, timings, None

            feats_list = []
            for i in range(0, seg.size - win_samp + 1, slide_samp):
                win = seg[i:i + win_samp]
                if np.isnan(win).all() or np.nanstd(win) < 1e-6:
                    continue
                all_feature_results = pycatch22.catch22_all(win, catch24=True)
                feats_list.append(all_feature_results)

            if not feats_list:
                return {'caseid': caseid, 'waveform': waveform_key, 'error': 'no_valid_windows_found'}, timings, None
            
            feature_names = feats_list[0]['names']
            feature_values = [res['values'] for res in feats_list]

            m = np.asarray(feature_values, dtype=np.float32)
            out = {
                **{f'{n}_mean': v for n, v in zip(feature_names, m.mean(0))},
                **{f'{n}_std':  v for n, v in zip(feature_names, m.std(0, ddof=1))}, # Use ddof=1 for sample std
                **{f'{n}_min':  v for n, v in zip(feature_names, m.min(0))},
                **{f'{n}_max':  v for n, v in zip(feature_names, m.max(0))}
            }
            out['caseid'] = caseid
            out['waveform'] = waveform_key # Use original key
            out[OUTCOME] = outcome
            end = time.perf_counter()
            timings['process_signal_windowed'] = end - start
            return out, timings, aeon_payload

    except Exception as e:
        logging.error(f"Error processing case {caseid} waveform {waveform_key}: {e}")
        return {'caseid': caseid, 'waveform': waveform_key, 'error': str(e)}, timings, None

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting catch22 feature extraction...")

    # === EDIT: Dynamic file name logic ===
    # Use local variables for file paths, modified based on window settings
    catch_22_file = CATCH_22_FILE
    catch_22_error_file = CATCH_22_ERROR_FILE

    # Windowed file name logic is now based on WIN_SEC
    if WIN_SEC is not None:
        logging.info("Windowed feature extraction is enabled. Appending '_windowed' to output files.")
        
        def _add_suffix(filepath, suffix):
            """Helper to add suffix before file extension"""
            name, ext = os.path.splitext(filepath)
            return f"{name}{suffix}{ext}"

        catch_22_file = _add_suffix(CATCH_22_FILE, '_windowed')
        catch_22_error_file = _add_suffix(CATCH_22_ERROR_FILE, '_windowed')
    
    logging.info(f"Output file set to: {catch_22_file}")
    logging.info(f"Error file set to: {catch_22_error_file}")
    # === END EDIT ===

    try:
        cohort_df = pd.read_csv(COHORT_FILE)
    except FileNotFoundError:
        logging.error(f"Cohort file not found at {COHORT_FILE}. Exiting.")
        return
    logging.info(f"Loaded {len(cohort_df)} cases from {COHORT_FILE}")

    processed_pairs = set()

    # Use the (potentially modified) local file path variable
    if os.path.exists(catch_22_file):
        logging.info(f"Checkpoint file found at {catch_22_file}. Loading processed cases.")
        try:
            # Only load columns needed for checkpointing to save memory
            processed_df = pd.read_csv(catch_22_file, usecols=['caseid', 'waveform'])
            processed_pairs = set(zip(processed_df['caseid'], processed_df['waveform']))
            logging.info(f"Found {len(processed_pairs)} previously processed case/waveform pairs.")
            del processed_df # Free memory
        except Exception as e:
            logging.warning(f"Could not read checkpoint file {catch_22_file}. Processing all cases. Error: {e}")
            processed_pairs = set()
    else:
        logging.info("No checkpoint file found. Starting from scratch.")


    # Expand cohort x waveforms
    wf_df = pd.DataFrame({'waveform': MANDATORY_WAVEFORMS})
    cohort_df['__k'] = 1
    wf_df['__k'] = 1
    cohort_expanded = (
        cohort_df.merge(wf_df, on='__k', how='left')
             .drop(columns='__k')
    )
    del cohort_df # Free memory

    original_case_count = len(pd.unique(cohort_expanded['caseid']))
    original_pair_count = len(cohort_expanded)

    # Anti-join to remove already processed pairs
    if processed_pairs:
        processed_pairs_df = pd.DataFrame(list(processed_pairs), columns=['caseid', 'waveform'])
        cohort_to_process = (
            cohort_expanded.merge(processed_pairs_df, on=['caseid', 'waveform'], how='left', indicator=True)
                           .loc[lambda df: df['_merge'] == 'left_only']
                           .drop(columns=['_merge'])
        )
        del processed_pairs_df
    else:
        cohort_to_process = cohort_expanded
    
    del cohort_expanded # Free memory

    if cohort_to_process.empty:
        logging.info("All case/waveform pairs have already been processed. Nothing to do.")
        cases_to_process = []
    else:
        # Build the worklist
        cases_to_process = list(
            cohort_to_process[['caseid', 'opstart', 'opend', OUTCOME, 'waveform']]
            .itertuples(index=False, name=None)
        )
    
    # === NEW: Create outcome map for Aeon export ===
    caseid_to_outcome = {}
    if not cohort_to_process.empty:
        # Use drop_duplicates to create a clean map
        caseid_to_outcome = pd.Series(
            cohort_to_process[OUTCOME].values, 
            index=cohort_to_process['caseid']
        ).drop_duplicates().to_dict()
    # === END NEW ===

    logging.info(
        f"{original_case_count} cases expanded to {original_pair_count} pairs; "
        f"{len(cohort_to_process)} pairs remain to process."
    )

    if not cases_to_process:
        logging.info("Exiting.")
        return

    num_processes = max(1, cpu_count() - 2) # Leave 2 cores free
    logging.info(f"Using {num_processes} processes for extraction.")

    all_features = []
    timings = []
    # === NEW: Aeon buffers ===
    case_buffers = defaultdict(dict)                  # non-windowed: caseid -> {waveform: 1D array}
    window_buffers = defaultdict(lambda: defaultdict(dict)) # windowed: (caseid, window_idx) -> {waveform: 1D array}
    # === END NEW ===

    start_total = time.perf_counter()
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Extracting Features") as pbar:
            # === NEW: Unpack 3-tuple ===
            for result, timing, payload in pool.imap_unordered(_process_case, cases_to_process):
                all_features.append(result)
                timings.append(timing)
                
                # === NEW: Accumulate Aeon payloads ===
                if EXPORT_AEON and payload is not None and 'error' not in result:
                    if WIN_SEC is None: # Non-windowed
                        case_buffers[payload.caseid][payload.waveform] = payload.seg_full
                    else: # Windowed
                        # Align windows by index
                        valid_idx = np.flatnonzero(payload.valid_window_mask)
                        for local_wi, global_wi in enumerate(valid_idx):
                            window_buffers[(payload.caseid, global_wi)][payload.waveform] = payload.win_mat[local_wi]
                # === END NEW ===

                pbar.update()
    
    end_total = time.perf_counter()
    logging.info(f"Feature extraction completed in {end_total - start_total:.2f} seconds.")

    if not all_features:
        logging.warning("Extraction ran but produced no results. Check input data.")
        return

    features_df = pd.DataFrame(all_features)

    if 'error' in features_df.columns:
        error_df = features_df[features_df['error'].notna()]
        success_df = features_df[features_df['error'].isna()].drop(columns=['error'])
    else:
        error_df = pd.DataFrame()
        success_df = features_df

    # === EDIT: Append-aware file writing ===
    if not error_df.empty:
        logging.info(f"Encountered {len(error_df)} errors during processing.")
        logging.info(f"Error examples:\n{error_df.head()}")
        
        # Check if error file exists to decide on mode and header
        error_file_exists = os.path.exists(catch_22_error_file)
        logging.info(f"Writing errors to {catch_22_error_file} (append: {error_file_exists})")
        
        error_df.to_csv(
            catch_22_error_file, 
            index=False, 
            mode='a' if error_file_exists else 'w',
            header=not error_file_exists
        )
        logging.info(f"Error log saved.")

    if not success_df.empty:
        # Fill potential NaN values from std dev calculation with 0
        # (happens if only one valid window is found)
        success_df.fillna(0, inplace=True)

        # Reorder columns for readability
        cols = ['caseid', 'waveform'] + [col for col in success_df.columns if col not in ['caseid', 'waveform']]
        success_df = success_df[cols]
        logging.info(f"Successfully extracted features for {len(success_df)} pairs.")
        logging.info(f"Success examples:\n{success_df.head()}")
        
        # Check if success file exists to decide on mode and header
        success_file_exists = os.path.exists(catch_22_file)
        logging.info(f"Writing features to {catch_22_file} (append: {success_file_exists})")

        success_df.to_csv(
            catch_22_file, 
            index=False,
            mode='a' if success_file_exists else 'w',
            header=not success_file_exists
        )
        logging.info(f"Features saved.")
        
    elif success_df.empty and not error_df.empty:
        logging.info("No new features were successfully extracted. Check error log.")
    elif success_df.empty and error_df.empty:
         logging.info("Processing complete, but no new success or error data was generated.")
    # === END EDIT ===
         
    logging.info("Feature extraction complete.")
    
    # Timing summary
    all_timings = defaultdict(lambda: {'count': 0, 'total': 0.0})
    for timing in timings:
        for k, v in timing.items():
            all_timings[k]['count'] += 1
            all_timings[k]['total'] += v
    
    logging.info("Timing summary (averages over processed cases):")
    for k, v in all_timings.items():
        if v['count'] > 0:
            logging.info(f"  {k}: {v['total']:.4f} sec total over {v['count']} cases (avg {v['total']/v['count']:.4f} sec)")

    # === NEW: Aeon Export Collation and Saving ===
    if EXPORT_AEON:
        logging.info("Starting Aeon export collation...")
        os.makedirs(AEON_OUT_DIR, exist_ok=True)
        channel_order = MANDATORY_WAVEFORMS # Define channel order for 3D numpy
        
        if WIN_SEC is None:
            # --- Non-Windowed Export ---
            logging.info(f"Collating non-windowed data for {len(case_buffers)} cases.")
            
            final_cases = []
            final_labels = []
            
            # 1. Filter cases based on strict channel policy
            for caseid, wave_dict in case_buffers.items():
                if AEON_STRICT_CHANNELS:
                    if all(c in wave_dict for c in channel_order):
                        final_cases.append((caseid, wave_dict))
                        final_labels.append({'caseid': caseid, OUTCOME: caseid_to_outcome.get(caseid)})
                else:
                    final_cases.append((caseid, wave_dict))
                    final_labels.append({'caseid': caseid, OUTCOME: caseid_to_outcome.get(caseid)})
            
            logging.info(f"Retained {len(final_cases)} cases after applying AEON_STRICT_CHANNELS={AEON_STRICT_CHANNELS}")

            if final_cases:
                # Save labels
                y_df = pd.DataFrame(final_labels)
                y_path = os.path.join(AEON_OUT_DIR, 'y_nonwindowed.csv')
                y_df.to_csv(y_path, index=False)
                logging.info(f"Saved non-windowed labels to {y_path}")

                # 2. Handle padding and save formats
                pad_len = AEON_PAD_LENGTH
                if AEON_PAD_POLICY == 'in_memory_pad' and pad_len is None:
                    pad_len = max(len(seg) for _, wave_dict in final_cases for seg in wave_dict.values())
                
                # --- Save np_list_pkl (unequal length) ---
                if "np_list_pkl" in AEON_SAVE_FORMATS or AEON_PAD_POLICY == "aeon_padding_transformer":
                    X_list = []
                    for _, wave_dict in final_cases:
                        # Shape (n_channels, n_timepoints_i)
                        arr = np.stack([wave_dict[c] for c in channel_order if c in wave_dict])
                        X_list.append(arr)
                    
                    list_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed_np_list.pkl')
                    joblib.dump(X_list, list_path)
                    logging.info(f"Saved unequal length list (for PaddingTransformer) to {list_path}")

                # --- Save padded formats ---
                if AEON_PAD_POLICY == 'in_memory_pad':
                    logging.info(f"Padding to length {pad_len} with fill '{AEON_PAD_FILL}'")
                    padded_dicts = [] # List of {channel: 1D_array}
                    
                    for caseid, wave_dict in final_cases:
                        padded_wave_dict = {}
                        for c in channel_order:
                            if c in wave_dict:
                                padded_wave_dict[c] = pad_1d(wave_dict[c], pad_len, AEON_PAD_FILL)
                            else:
                                padded_wave_dict[c] = np.full(pad_len, fill_value=AEON_PAD_FILL, dtype=np.float32)
                        padded_dicts.append(padded_wave_dict)

                    if "nested_pkl" in AEON_SAVE_FORMATS:
                        # Build nested DataFrame (n_cases x n_channels)
                        index = pd.Index([caseid for caseid, _ in final_cases], name='caseid')
                        X_nested = pd.DataFrame(columns=channel_order, index=index)
                        for i, (caseid, _) in enumerate(final_cases):
                            for c in channel_order:
                                X_nested.loc[caseid, c] = padded_dicts[i][c]
                        
                        nested_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed_nested.pkl')
                        joblib.dump(X_nested, nested_path)
                        logging.info(f"Saved padded nested DataFrame to {nested_path}")

                    if "numpy3d_npz" in AEON_SAVE_FORMATS:
                        # Stack to (n_cases, n_channels, n_timepoints)
                        X_3d = np.stack([
                            np.stack([d[c] for c in channel_order]) for d in padded_dicts
                        ])
                        npz_path = os.path.join(AEON_OUT_DIR, 'X_nonwindowed.npz')
                        np.savez_compressed(npz_path, X=X_3d, caseids=y_df['caseid'].values, channels=channel_order)
                        logging.info(f"Saved padded 3D NumPy to {npz_path} with shape {X_3d.shape}")
            
        else:
            # --- Windowed Export ---
            logging.info(f"Collating windowed data for {len(window_buffers)} valid windows.")
            
            final_windows = [] # List of ( (caseid, win_idx), {wave_dict} )
            
            # 1. Filter windows based on policy
            if AEON_WINDOW_POLICY == 'intersection':
                for (caseid, win_idx), wave_dict in window_buffers.items():
                    if all(c in wave_dict for c in channel_order):
                        final_windows.append(((caseid, win_idx), wave_dict))
            else: # 'union' (or default)
                final_windows = list(window_buffers.items()) # Keep all
            
            logging.info(f"Retained {len(final_windows)} windows after applying {AEON_WINDOW_POLICY} policy.")

            if final_windows:
                # Get final window length (should be consistent)
                win_len = final_windows[0][1][list(final_windows[0][1].keys())[0]].shape[0]
                
                # Build labels
                y_data = []
                for (caseid, win_idx), _ in final_windows:
                    y_data.append({
                        'caseid': caseid, 
                        'window_idx': win_idx, 
                        OUTCOME: caseid_to_outcome.get(caseid),
                        'bag_id': caseid # for aeon
                    })
                y_df = pd.DataFrame(y_data)
                y_path = os.path.join(AEON_OUT_DIR, 'y_windowed.csv')
                y_df.to_csv(y_path, index=False)
                logging.info(f"Saved windowed labels to {y_path}")

                # Prepare data for formats
                window_data_dicts = [] # List of {channel: 1D_array}
                multi_index = []
                
                for (caseid, win_idx), wave_dict in final_windows:
                    multi_index.append((caseid, win_idx))
                    processed_wave_dict = {}
                    for c in channel_order:
                        if c in wave_dict:
                            processed_wave_dict[c] = wave_dict[c]
                        else:
                            # Pad missing channels (if union)
                            processed_wave_dict[c] = np.full(win_len, fill_value=AEON_PAD_FILL, dtype=np.float32)
                    window_data_dicts.append(processed_wave_dict)

                if "nested_pkl" in AEON_SAVE_FORMATS:
                    index = pd.MultiIndex.from_tuples(multi_index, names=['caseid', 'window_idx'])
                    X_nested = pd.DataFrame(columns=channel_order, index=index)
                    
                    # This is faster than .loc
                    for c in channel_order:
                        X_nested[c] = [d[c] for d in window_data_dicts]
                        
                    nested_path = os.path.join(AEON_OUT_DIR, 'X_windowed_nested.pkl')
                    joblib.dump(X_nested, nested_path)
                    logging.info(f"Saved windowed nested DataFrame to {nested_path}")
                
                if "numpy3d_npz" in AEON_SAVE_FORMATS:
                    # Stack to (n_windows, n_channels, n_timepoints)
                    X_3d = np.stack([
                        np.stack([d[c] for c in channel_order]) for d in window_data_dicts
                    ])
                    npz_path = os.path.join(AEON_OUT_DIR, 'X_windowed.npz')
                    np.savez_compressed(
                        npz_path, X=X_3d, 
                        caseids=y_df['caseid'].values, 
                        window_idx=y_df['window_idx'].values,
                        channels=channel_order
                    )
                    logging.info(f"Saved windowed 3D NumPy to {npz_path} with shape {X_3d.shape}")

        # --- Write README and Example Scripts ---
        _write_aeon_helpers(AEON_OUT_DIR, WIN_SEC)
        logging.info(f"Wrote helper README and example training scripts to {AEON_OUT_DIR}")
    # === END AEON EXPORT ===


def _write_aeon_helpers(out_dir, win_sec):
    """Writes README and example training scripts to the aeon output directory."""
    
    readme_content = f"""
# Aeon Export Directory

This directory contains data exported from `catch22_extraction.py` in an `aeon`-compatible format.

- **Mode:** {"Windowed" if win_sec is not None else "Non-Windowed"}
- **Data (X):** Serialized in one or more formats (e.g., `X_...pkl`, `X_...npz`).
- **Labels (y):** `y_{"windowed" if win_sec is not None else "nonwindowed"}.csv`

See the `train_...py` example scripts for how to load and use this data with `aeon` and `MultiRocket`.
    """
    with open(os.path.join(out_dir, 'README.md'), 'w') as f:
        f.write(readme_content)

    nonwindowed_script = """
# train_nonwindowed_example.py
import joblib, numpy as np, pandas as pd
from aeon.transformations.collection.pad import PaddingTransformer
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

print("Loading non-windowed data...")
# Load unequal-length export (recommended if using PaddingTransformer)
try:
    X_list = joblib.load("X_nonwindowed_np_list.pkl")  # list of (C, T_i)
    y = pd.read_csv("y_nonwindowed.csv")["outcome"].to_numpy()
except FileNotFoundError:
    print("Could not find 'X_nonwindowed_np_list.pkl'. This requires AEON_PAD_POLICY='aeon_padding_transformer'")
    exit()

print(f"Loaded {len(X_list)} cases.")

pipe = make_pipeline(
    PaddingTransformer(pad_length=None, fill_value=0),  # pads to global max, or set a cap
    MultiRocket(n_kernels=20_000, n_jobs=-1, random_state=42),
    RidgeClassifierCV(alphas=np.logspace(-3,3,9))
)

print("Fitting pipeline (MultiRocket + Ridge)...")
pipe.fit(X_list, y)

print("Fit complete.")
print(f"Pipeline score: {pipe.score(X_list, y)}")
    """
    
    windowed_script = """
# train_windowed_example.py
import joblib, numpy as np, pandas as pd
from aeon.transformations.collection.convolution_based import MultiRocket
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from aeon.utils.validation.collection import convert_collection

print("Loading windowed data...")
# Load nested DataFrame (or 3D NPZ)
try:
    # X_nested = joblib.load("X_windowed_nested.pkl")  # nested DF
    # X_np = convert_collection(X_nested, output_type="numpy3D")
    
    # Or load 3D NPZ directly if available
    X_data = np.load("X_windowed.npz")
    X_np = X_data['X']
    
    y = pd.read_csv("y_windowed.csv")["outcome"].to_numpy()
except FileNotFoundError:
    print("Could not find 'X_windowed.npz' or 'y_windowed.csv'.")
    exit()

print(f"Loaded {X_np.shape[0]} windows, {X_np.shape[1]} channels, {X_np.shape[2]} timepoints.")

pipe = make_pipeline(
    MultiRocket(n_kernels=20_000, n_jobs=-1, random_state=42),
    RidgeClassifierCV(alphas=np.logspace(-3,3,9))
)

print("Fitting pipeline (MultiRocket + Ridge)...")
pipe.fit(X_np, y)

print("Fit complete.")
print(f"Pipeline score: {pipe.score(X_np, y)}")
    """

    if win_sec is None:
        with open(os.path.join(out_dir, 'train_nonwindowed_example.py'), 'w') as f:
            f.write(nonwindowed_script)
    else:
        with open(os.path.join(out_dir, 'train_windowed_example.py'), 'w') as f:
            f.write(windowed_script)


if __name__ == "__main__":
    main()