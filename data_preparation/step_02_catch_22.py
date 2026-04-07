import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import pycatch22
from tqdm import tqdm
import time

from data_preparation.artifact_metadata import (
    STEP_01_COHORT_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
)
from data_preparation.inputs import (
    COHORT_FILE, 
    CATCH_22_FILE,
    CATCH_22_ERROR_FILE,
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    WIN_SEC,
    SLIDE_SEC,
    GENERATE_FULL_FEATURES,
    GENERATE_WINDOWED_FEATURES,
    FULL_FEATURE_TARGET_SR
)
from data_preparation.waveform_processing import WAVEFORM_SPECS, process_signal, harmonize_sr, load_and_validate_case

COHORT_REQUIRED_COLUMNS = ["caseid", "opstart", "opend"]

# Validation
if GENERATE_WINDOWED_FEATURES:
    assert WIN_SEC is not None and SLIDE_SEC is not None, "WIN_SEC and SLIDE_SEC must be set for windowed extraction."

def _process_case(case: Tuple[int, float, float, str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Processes a single case/waveform pair.
    Returns:
        - results: Dict with keys 'full' and/or 'windowed' containing feature dicts.
        - timings: Dict of timing metrics.
    """
    caseid, opstart, opend, waveform_key = case
    timings = {}
    results = {}
    
    start = time.perf_counter()

    try:
        # 1. Load waveform with substitutions logic
        wave, spec_key_loaded, error_msg = load_and_validate_case(caseid, waveform_key, WAVEFORM_SUBSTITUTIONS)
        
        if error_msg:
             err = {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}
             return {'error': err}, timings
        
        spec = WAVEFORM_SPECS[spec_key_loaded]
        end = time.perf_counter()
        timings['load_case'] = end - start
        start = end

        # 5. Slicing at NATIVE sample rate
        native_sr = spec['native_sr']
        seg = wave[int(opstart * native_sr):int(opend * native_sr)]
        seg = seg.squeeze()
        
        # 6. NaN checks
        nan_mask = np.isnan(seg)
        nan_pct = np.mean(nan_mask)
        if nan_pct > 0.05:
            err = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_gt_5_pct_nan'}
            return {'error': err}, timings
        
        if nan_pct > 0:
            x = np.arange(seg.size)
            seg = np.interp(x=x, xp=x[~nan_mask], fp=seg[~nan_mask])
        
        # 7. & 8. Filtering and Resampling
        seg, error_msg = process_signal(seg, spec, caseid, waveform_key)
        if error_msg:
             err = {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}
             return {'error': err}, timings

        target_sr = spec['target_sr']

        # === FULL FEATURE EXTRACTION ===
        if GENERATE_FULL_FEATURES:
            start_full = time.perf_counter()
            if np.isnan(seg).all() or np.nanstd(seg) < 1e-6: 
                # Log error but don't fail the whole case if windowed might work (unlikely but possible)
                results['full'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_flatline_or_all_nan'}
            else:
                # Downsample to 10 Hz for efficient full-series extraction
                # This matches the "Old Code" effective resolution for this step
                seg_full = harmonize_sr(seg, target_sr, FULL_FEATURE_TARGET_SR)
                all_feature_results = pycatch22.catch22_all(seg_full, catch24=True)
                out = {n: v for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
                out['caseid'] = caseid
                out['waveform'] = waveform_key 
                results['full'] = out
                
            timings['process_full'] = time.perf_counter() - start_full

        # === WINDOWED FEATURE EXTRACTION ===
        if GENERATE_WINDOWED_FEATURES:
            start_win = time.perf_counter()
            win_samp = int(WIN_SEC * target_sr)
            slide_samp = int(SLIDE_SEC * target_sr)
            
            if seg.size < win_samp:
                results['windowed'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'segment_too_short_for_window'}
            else:
                feats_list = []
                
                # Pre-calculate window indices
                window_starts = range(0, seg.size - win_samp + 1, slide_samp)
                
                for i in window_starts:
                    win = seg[i:i + win_samp]
                    is_valid = (not np.isnan(win).all()) and (np.nanstd(win) >= 1e-6)
                    
                    if is_valid:
                        all_feature_results = pycatch22.catch22_all(win, catch24=True)
                        feats_list.append(all_feature_results)

                if not feats_list:
                    results['windowed'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'no_valid_windows_found'}
                else:
                    feature_names = feats_list[0]['names']
                    feature_values = [res['values'] for res in feats_list]
                    m = np.asarray(feature_values, dtype=np.float32)
                    out = {
                        **{f'{n}_mean': v for n, v in zip(feature_names, m.mean(0))},
                        **{f'{n}_std':  v for n, v in zip(feature_names, m.std(0, ddof=1))},
                        **{f'{n}_min':  v for n, v in zip(feature_names, m.min(0))},
                        **{f'{n}_max':  v for n, v in zip(feature_names, m.max(0))}
                    }
                    out['caseid'] = caseid
                    out['waveform'] = waveform_key
                    results['windowed'] = out

            timings['process_windowed'] = time.perf_counter() - start_win

        return results, timings

    except Exception as e:
        logging.error(f"Error processing case {caseid} waveform {waveform_key}: {e}")
        err = {'caseid': caseid, 'waveform': waveform_key, 'error': str(e)}
        return {'error': err}, timings

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting catch22 feature extraction...")

    # Define outputs based on configuration
    outputs_config = {}
    if GENERATE_FULL_FEATURES:
        outputs_config['full'] = {
            'file': CATCH_22_FILE,
            'error_file': CATCH_22_ERROR_FILE,
            'features': [],
            'errors': []
        }
    if GENERATE_WINDOWED_FEATURES:
        def _add_suffix(filepath, suffix):
            name, ext = os.path.splitext(filepath)
            return f"{name}{suffix}{ext}"
        
        outputs_config['windowed'] = {
            'file': _add_suffix(CATCH_22_FILE, '_windowed'),
            'error_file': _add_suffix(CATCH_22_ERROR_FILE, '_windowed'),
            'features': [],
            'errors': []
        }

    for mode, conf in outputs_config.items():
        logging.info(f"Mode '{mode}': Output={conf['file']}, Error={conf['error_file']}")

    # Load Cohort
    try:
        cohort_df, _ = read_versioned_csv(
            COHORT_FILE,
            artifact_role=STEP_01_COHORT_ARTIFACT,
            required_columns=COHORT_REQUIRED_COLUMNS,
        )
    except FileNotFoundError:
        logging.error(f"Cohort file not found at {COHORT_FILE}. Exiting.")
        return
    except ArtifactCompatibilityError as exc:
        logging.error("%s", exc)
        return
    logging.info(f"Loaded {len(cohort_df)} cases from {COHORT_FILE}")

    # Checkpoint logic: We need to process a pair if it is missing from ANY enabled output
    processed_pairs_by_mode = {}
    for mode, conf in outputs_config.items():
        processed_pairs = set()
        if os.path.exists(conf['file']):
            try:
                df = pd.read_csv(conf['file'], usecols=['caseid', 'waveform'])
                processed_pairs = set(zip(df['caseid'], df['waveform']))
            except Exception:
                pass
        processed_pairs_by_mode[mode] = processed_pairs
        logging.info(f"Mode '{mode}': Found {len(processed_pairs)} previously processed pairs.")

    # Expand cohort
    wf_df = pd.DataFrame({'waveform': MANDATORY_WAVEFORMS})
    cohort_df['__k'] = 1
    wf_df['__k'] = 1
    cohort_expanded = (cohort_df.merge(wf_df, on='__k', how='left').drop(columns='__k'))
    del cohort_df

    # Determine worklist
    # A pair needs processing if it's missing in ANY enabled mode
    # However, to avoid complexity, we can just process everything that isn't in ALL enabled modes?
    # Or simpler: process if missing in ANY. The writer will append.
    # But if we process a case that is present in 'full' but missing in 'windowed', we re-compute 'full'.
    # That's acceptable overhead to keep logic simple.
    
    # Actually, let's find the intersection of processed pairs across all enabled modes
    # If a pair is processed in ALL enabled modes, skip it.
    if not processed_pairs_by_mode:
        universal_processed = set()
    else:
        universal_processed = set.intersection(*processed_pairs_by_mode.values())
    
    if universal_processed:
        processed_pairs_df = pd.DataFrame(list(universal_processed), columns=['caseid', 'waveform'])
        cohort_to_process = (
            cohort_expanded.merge(processed_pairs_df, on=['caseid', 'waveform'], how='left', indicator=True)
                           .loc[lambda df: df['_merge'] == 'left_only']
                           .drop(columns=['_merge'])
        )
    else:
        cohort_to_process = cohort_expanded
    
    del cohort_expanded

    if cohort_to_process.empty:
        logging.info("All pairs processed in all enabled modes. Nothing to do.")
        return

    cases_to_process = list(
        cohort_to_process[['caseid', 'opstart', 'opend', 'waveform']]
        .itertuples(index=False, name=None)
    )
    
    logging.info(f"{len(cohort_to_process)} pairs to process.")

    # Determine number of processes
    try:
        # On Linux, this respects cgroups/quotas (important for VMs/Containers)
        num_processes = max(1, len(os.sched_getaffinity(0)) - 2)
    except AttributeError:
        # Fallback for Windows/MacOS
        num_processes = max(1, cpu_count() - 2)
    
    logging.info(f"Using {num_processes} worker processes.")
    
    timings = []
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Extracting Features") as pbar:
            for result_dict, timing in pool.imap_unordered(_process_case, cases_to_process):
                timings.append(timing)
                
                # Handle global error (loading failure)
                if 'error' in result_dict:
                    # This error applies to ALL modes
                    for mode in outputs_config:
                        outputs_config[mode]['errors'].append(result_dict['error'])
                else:
                    # Distribute results to respective modes
                    for mode, res in result_dict.items():
                        if 'error' in res:
                            outputs_config[mode]['errors'].append(res)
                        else:
                            outputs_config[mode]['features'].append(res)
                pbar.update()

    # Save results for each mode
    for mode, conf in outputs_config.items():
        features_df = pd.DataFrame(conf['features'])
        error_df = pd.DataFrame(conf['errors'])
        
        # Save Errors
        if not error_df.empty:
            exists = os.path.exists(conf['error_file'])
            error_df.to_csv(conf['error_file'], index=False, mode='a' if exists else 'w', header=not exists)
            logging.info(f"[{mode}] Saved {len(error_df)} errors.")

        # Save Features
        if not features_df.empty:
            features_df.fillna(0, inplace=True)
            cols = ['caseid', 'waveform'] + [c for c in features_df.columns if c not in ['caseid', 'waveform']]
            features_df = features_df[cols]
            
            exists = os.path.exists(conf['file'])
            features_df.to_csv(conf['file'], index=False, mode='a' if exists else 'w', header=not exists)
            logging.info(f"[{mode}] Saved {len(features_df)} features.")
        else:
            logging.info(f"[{mode}] No new features to save.")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
