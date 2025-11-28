import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import vitaldb
import pycatch22
from tqdm import tqdm
import time
from collections import defaultdict

from data_preparation.inputs import (
    COHORT_FILE, 
    CATCH_22_FILE,
    CATCH_22_ERROR_FILE,
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    WIN_SEC,
    SLIDE_SEC,
    OUTCOME,
    EXPORT_AEON,
    AEON_COMMON_SR,
    GENERATE_FULL_FEATURES,
    GENERATE_WINDOWED_FEATURES
)
from data_preparation.waveform_processing import WAVEFORM_SPECS, process_signal, harmonize_sr
from data_preparation.aeon_io import AeonSeriesPayload, collate_and_save_aeon

# Create mapping from VitalDB ID to Spec Key
ID_TO_SPEC_KEY = {spec['id']: key for key, spec in WAVEFORM_SPECS.items()}

# Validation
if GENERATE_WINDOWED_FEATURES:
    assert WIN_SEC is not None and SLIDE_SEC is not None, "WIN_SEC and SLIDE_SEC must be set for windowed extraction."

def _process_case(case: Tuple[int, float, float, int, str]) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, Optional[AeonSeriesPayload]]]:
    """
    Processes a single case/waveform pair.
    Returns:
        - results: Dict with keys 'full' and/or 'windowed' containing feature dicts.
        - timings: Dict of timing metrics.
        - payloads: Dict with keys 'full' and/or 'windowed' containing Aeon payloads.
    """
    caseid, opstart, opend, outcome, waveform_key = case
    timings = {}
    results = {}
    payloads = {}
    
    start = time.perf_counter()

    try:
        wave = None
        spec_key_loaded = None
        spec = None
        
        # Resolve the primary spec key from the input waveform string (which might be an ID or a Key)
        primary_spec_key = ID_TO_SPEC_KEY.get(waveform_key)
        if not primary_spec_key and waveform_key in WAVEFORM_SPECS:
            primary_spec_key = waveform_key

        # 1. Try loading the primary waveform key
        if primary_spec_key:
            try:
                spec = WAVEFORM_SPECS[primary_spec_key]
                wave = vitaldb.load_case(caseid, spec['id'], interval=1/spec['native_sr']) 
                if wave is not None and wave.size > 0:
                    spec_key_loaded = primary_spec_key
            except Exception:
                pass 

        # 2. If primary failed, try substitutions
        if wave is None or wave.size == 0:
            for sub_id in WAVEFORM_SUBSTITUTIONS.get(waveform_key, []):
                # Resolve substitute spec key
                sub_spec_key = ID_TO_SPEC_KEY.get(sub_id)
                if not sub_spec_key and sub_id in WAVEFORM_SPECS:
                    sub_spec_key = sub_id
                
                if sub_spec_key:
                    try:
                        spec = WAVEFORM_SPECS[sub_spec_key]
                        wave = vitaldb.load_case(caseid, spec['id'], interval=1/spec['native_sr'])
                        if wave is not None and wave.size > 0:
                            spec_key_loaded = sub_spec_key
                            break 
                    except Exception:
                        continue 

        # 3. Check if loading failed entirely
        if wave is None or wave.size == 0:
            err = {'caseid': caseid, 'waveform': waveform_key, 'error': 'empty_signal_or_missing'}
            return {'error': err}, timings, {}
        
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
            return {'error': err}, timings, {}
        
        if nan_pct > 0:
            x = np.arange(seg.size)
            seg = np.interp(x=x, xp=x[~nan_mask], fp=seg[~nan_mask])
        
        # 7. & 8. Filtering and Resampling
        seg, error_msg = process_signal(seg, spec, caseid, waveform_key)
        if error_msg:
             err = {'caseid': caseid, 'waveform': waveform_key, 'error': error_msg}
             return {'error': err}, timings, {}

        target_sr = spec['target_sr']

        # === FULL FEATURE EXTRACTION ===
        if GENERATE_FULL_FEATURES:
            start_full = time.perf_counter()
            if np.isnan(seg).all() or np.nanstd(seg) < 1e-6: 
                # Log error but don't fail the whole case if windowed might work (unlikely but possible)
                results['full'] = {'caseid': caseid, 'waveform': waveform_key, 'error': 'invalid_signal_flatline_or_all_nan'}
            else:
                all_feature_results = pycatch22.catch22_all(seg, catch24=True)
                out = {n: v for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
                out['caseid'] = caseid
                out['waveform'] = waveform_key 
                out[OUTCOME] = outcome 
                results['full'] = out
                
                if EXPORT_AEON:
                    seg_out = seg.astype(np.float32, copy=False)
                    current_sr = target_sr
                    if AEON_COMMON_SR:
                        seg_out = harmonize_sr(seg_out, target_sr, AEON_COMMON_SR)
                        current_sr = AEON_COMMON_SR
                    
                    payloads['full'] = AeonSeriesPayload(
                        caseid=caseid, waveform=waveform_key,
                        target_sr=current_sr, seg_full=seg_out, length=seg_out.size
                    )
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
                win_vals = []
                valid_mask = []
                
                # Pre-calculate window indices
                window_starts = range(0, seg.size - win_samp + 1, slide_samp)
                n_potential_windows = (seg.size - win_samp + slide_samp) // slide_samp
                
                for i in window_starts:
                    win = seg[i:i + win_samp]
                    is_valid = (not np.isnan(win).all()) and (np.nanstd(win) >= 1e-6)
                    
                    if EXPORT_AEON:
                        valid_mask.append(is_valid)
                        if is_valid:
                            win_vals.append(win.astype(np.float32, copy=False))
                    
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
                    out[OUTCOME] = outcome
                    results['windowed'] = out

                    if EXPORT_AEON:
                        # Fill remainder of mask
                        valid_mask.extend([False] * (n_potential_windows - len(valid_mask)))
                        
                        current_sr = target_sr
                        final_win_samp = win_samp
                        win_mat = np.empty((0, final_win_samp), dtype=np.float32)

                        if win_vals:
                            if AEON_COMMON_SR and target_sr != AEON_COMMON_SR:
                                win_vals = [harmonize_sr(w, target_sr, AEON_COMMON_SR) for w in win_vals]
                                current_sr = AEON_COMMON_SR
                                final_win_samp = win_vals[0].shape[0]

                            win_mat = np.vstack(win_vals)
                        elif AEON_COMMON_SR:
                             final_win_samp = int(win_samp * (AEON_COMMON_SR / target_sr))

                        payloads['windowed'] = AeonSeriesPayload(
                            caseid=caseid, waveform=waveform_key,
                            target_sr=current_sr, win_mat=win_mat,
                            valid_window_mask=np.array(valid_mask, dtype=bool),
                            length=final_win_samp
                        )
            timings['process_windowed'] = time.perf_counter() - start_win

        return results, timings, payloads

    except Exception as e:
        logging.error(f"Error processing case {caseid} waveform {waveform_key}: {e}")
        err = {'caseid': caseid, 'waveform': waveform_key, 'error': str(e)}
        return {'error': err}, timings, {}

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
        cohort_df = pd.read_csv(COHORT_FILE)
    except FileNotFoundError:
        logging.error(f"Cohort file not found at {COHORT_FILE}. Exiting.")
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
        cohort_to_process[['caseid', 'opstart', 'opend', OUTCOME, 'waveform']]
        .itertuples(index=False, name=None)
    )
    
    # Outcome map for Aeon
    caseid_to_outcome = {}
    if not cohort_to_process.empty:
        caseid_to_outcome = pd.Series(
            cohort_to_process[OUTCOME].values, 
            index=cohort_to_process['caseid']
        ).drop_duplicates().to_dict()

    logging.info(f"{len(cohort_to_process)} pairs to process.")

    # Determine number of processes
    try:
        # On Linux, this respects cgroups/quotas (important for VMs/Containers)
        num_processes = max(1, len(os.sched_getaffinity(0)) - 2)
    except AttributeError:
        # Fallback for Windows/MacOS
        num_processes = max(1, cpu_count() - 2)
    
    logging.info(f"Using {num_processes} worker processes.")
    
    # Aeon buffers
    aeon_buffers = {
        'full': defaultdict(dict),
        'windowed': defaultdict(lambda: defaultdict(dict))
    }

    timings = []
    
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Extracting Features") as pbar:
            for result_dict, timing, payload_dict in pool.imap_unordered(_process_case, cases_to_process):
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
                    
                    # Distribute Aeon payloads
                    if EXPORT_AEON:
                        if 'full' in payload_dict:
                             p = payload_dict['full']
                             aeon_buffers['full'][p.caseid][p.waveform] = p.seg_full
                        
                        if 'windowed' in payload_dict:
                            p = payload_dict['windowed']
                            valid_idx = np.flatnonzero(p.valid_window_mask)
                            for local_wi, global_wi in enumerate(valid_idx):
                                aeon_buffers['windowed'][(p.caseid, global_wi)][p.waveform] = p.win_mat[local_wi]

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

    # Aeon Export
    if EXPORT_AEON:
        if GENERATE_FULL_FEATURES and aeon_buffers['full']:
            collate_and_save_aeon(
                case_buffers=aeon_buffers['full'],
                window_buffers={},
                caseid_to_outcome=caseid_to_outcome,
                channel_order=MANDATORY_WAVEFORMS,
                is_windowed=False
            )
        
        if GENERATE_WINDOWED_FEATURES and aeon_buffers['windowed']:
            # We need to temporarily override AEON_OUT_DIR or file names to avoid collision?
            # aeon_io.py saves to fixed filenames like 'X_nonwindowed.npz' and 'X_windowed.npz'.
            # So they won't collide.
            collate_and_save_aeon(
                case_buffers={},
                window_buffers=aeon_buffers['windowed'],
                caseid_to_outcome=caseid_to_outcome,
                channel_order=MANDATORY_WAVEFORMS,
                is_windowed=True
            )

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
