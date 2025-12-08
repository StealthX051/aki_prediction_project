import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import time
from collections import defaultdict
from tqdm import tqdm

from data_preparation.inputs import (
    COHORT_FILE, 
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    OUTCOME,
    AEON_FIXED_LENGTH,
    AEON_OUT_DIR # Ensure this is used
)
from data_preparation.waveform_processing import process_signal, load_and_validate_case, WAVEFORM_SPECS
from data_preparation.aeon_io import collate_and_save_aeon, AeonExportConfig

def resample_fixed_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resamples a 1D signal to a fixed length using linear interpolation.
    """
    if sig.size == target_len:
        return sig
    if sig.size == 0:
        return np.zeros(target_len, dtype=np.float32)
        
    x_old = np.linspace(0, 1, sig.size)
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, sig).astype(np.float32)

def _process_case_aeon(case: Tuple[int, float, float, int, str]) -> Tuple[Dict[str, Any], Optional[Dict[str, float]]]:
    """
    Processes a single case/waveform pair for Aeon export.
    Features:
    - Load & Validate
    - Filter (via process_signal)
    - Resample to Fixed Length
    """
    caseid, opstart, opend, outcome, waveform_key = case
    timings = {}
    start = time.perf_counter()
    
    # 1. Load
    wave, spec_key_loaded, error_msg = load_and_validate_case(caseid, waveform_key, WAVEFORM_SUBSTITUTIONS)
    timings['load'] = time.perf_counter() - start
    
    if error_msg:
        return {'error': {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}}, timings

    spec = WAVEFORM_SPECS[spec_key_loaded]
    
    # 2. Slice at native SR
    native_sr = spec['native_sr']
    # Ensure indices are within bounds
    idx_start = max(0, int(opstart * native_sr))
    idx_end = min(wave.size, int(opend * native_sr))
    
    seg = wave[idx_start:idx_end].squeeze()
    
    # 3. NaN Check
    nan_mask = np.isnan(seg)
    if np.mean(nan_mask) > 0.05:
         return {'error': {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_gt_5_pct_nan'}}, timings
         
    if np.any(nan_mask):
        x = np.arange(seg.size)
        seg = np.interp(x=x, xp=x[~nan_mask], fp=seg[~nan_mask])

    # 4. Filter (process_signal handles BPF/LPF and optional downsampling to target_sr)
    # We allow it to downsample to target_sr (100Hz) first as an anti-aliasing step before our final fixed resampling
    seg, error_msg = process_signal(seg, spec, caseid, waveform_key)
    if error_msg:
         return {'error': {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}}, timings

    # 5. Resample to 10 Hz (Fixed Frequency)
    start_res = time.perf_counter()
    # Calculate target length for 10 Hz
    duration_sec = (idx_end - idx_start) / native_sr
    target_hz = 10.0
    target_len = max(1, int(np.ceil(duration_sec * target_hz)))
    
    # Resample to the dynamic 10 Hz length (padding happens in collate step)
    resampled_seg = resample_fixed_length(seg, target_len)
    timings['resample'] = time.perf_counter() - start_res
    
    # Return success payload
    return {
        'caseid': caseid,
        'waveform': waveform_key,
        'seg': resampled_seg,
        'original_dur_min': duration_sec / 60.0
    }, timings

import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Limit number of cases for testing")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Starting Aeon Export (Freq=10Hz, Padding to {AEON_FIXED_LENGTH})...")

    # Load Cohort
    if not os.path.exists(COHORT_FILE):
        logging.error(f"Cohort file not found: {COHORT_FILE}")
        return
    cohort_df = pd.read_csv(COHORT_FILE)
    logging.info(f"Loaded {len(cohort_df)} cases.")

    # Expand Cohort (Case x Waveform)
    wf_df = pd.DataFrame({'waveform': MANDATORY_WAVEFORMS})
    cohort_df['__k'] = 1
    wf_df['__k'] = 1
    cohort_expanded = (cohort_df.merge(wf_df, on='__k', how='left').drop(columns='__k'))
    
    cases_to_process = list(
        cohort_expanded[['caseid', 'opstart', 'opend', OUTCOME, 'waveform']]
        .itertuples(index=False, name=None)
    )
    
    if args.limit:
        logging.info(f"Limiting to first {args.limit} cases (approx {args.limit * len(MANDATORY_WAVEFORMS)} tasks).")
        # Ensure balanced sampling for smoke tests
        if OUTCOME in cohort_df.columns:
            positives = cohort_df[cohort_df[OUTCOME] == 1]['caseid'].unique()
            negatives = cohort_df[cohort_df[OUTCOME] == 0]['caseid'].unique()
            
            n_pos = min(len(positives), args.limit // 2)
            n_neg = min(len(negatives), args.limit - n_pos)
            
            # If we requested a limit but have very few positives, ensure we at least get SOME negatives to fill
            # If we have 0 positives, we just take negatives (results in 1 class, but can't help that)
            if n_pos < 2:
                 logging.warning(f"Only found {n_pos} positive cases. Training might fail with <2 classes.")

            selected_cases = np.concatenate([
                positives[:n_pos],
                negatives[:n_neg]
            ])
            unique_cases = selected_cases
            logging.info(f"Selected {len(unique_cases)} cases ({n_pos} pos, {n_neg} neg) for export.")
        else:
            # Fallback if outcome not in cohort (unlikely)
            unique_cases = cohort_df['caseid'].head(args.limit).values
            
        cases_to_process = [c for c in cases_to_process if c[0] in unique_cases]
    
    caseid_to_outcome = pd.Series(
        cohort_df[OUTCOME].values, 
        index=cohort_df['caseid']
    ).to_dict()

    # Processing Loop
    num_processes = max(1, len(os.sched_getaffinity(0)) - 2)
    logging.info(f"Using {num_processes} processes.")
    
    aeon_buffers = defaultdict(dict)
    errors = []
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Exporting Waveforms") as pbar:
            for res, timings in pool.imap_unordered(_process_case_aeon, cases_to_process):
                if 'error' in res:
                    errors.append(res['error'])
                else:
                    aeon_buffers[res['caseid']][res['waveform']] = res['seg']
                pbar.update()

    if errors:
        err_df = pd.DataFrame(errors)
        err_path = os.path.join(AEON_OUT_DIR, 'export_errors.csv')
        os.makedirs(AEON_OUT_DIR, exist_ok=True)
        err_df.to_csv(err_path, index=False)
        logging.info(f"Saved {len(errors)} errors to {err_path}")

    # Collate and Save
    if aeon_buffers:
        config = AeonExportConfig(
            pad_policy='in_memory_pad', # Required to trigger numpy3d_npz saving in aeon_io.py
            pad_length=AEON_FIXED_LENGTH,
            save_formats=frozenset(["numpy3d_npz", "y"]) # We mostly need npz for fixed length
        )
        
        collate_and_save_aeon(
            case_buffers=aeon_buffers,
            window_buffers={}, # No windows
            caseid_to_outcome=caseid_to_outcome,
            channel_order=MANDATORY_WAVEFORMS,
            is_windowed=False,
            config=config
        )
    else:
        logging.warning("No valid data to save!")

if __name__ == "__main__":
    main()
