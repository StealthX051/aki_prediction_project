import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple
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
    TARGET_SR,
    OUTCOME
)


WIN_SAMP = int(WIN_SEC * TARGET_SR) if WIN_SEC else None
SLIDE_SAMP = int(SLIDE_SEC * TARGET_SR) if SLIDE_SEC else None
assert (WIN_SAMP is None and SLIDE_SAMP is None) or (WIN_SAMP is not None and SLIDE_SAMP is not None), "Invalid window/slide settings"

def _process_case(case: Tuple[int, float, float, int, str]) -> Tuple[Dict[str, Any], Dict[str, float]]:
    caseid, opstart, opend, outcome, waveform = case
    timings = {}
    start = time.perf_counter()
    wave = vitaldb.load_case(caseid, waveform, 1 / TARGET_SR)
    for sub_waveform in WAVEFORM_SUBSTITUTIONS.get(waveform, []):
        if wave is None or wave.size == 0:
            wave = vitaldb.load_case(caseid, sub_waveform, 1 / TARGET_SR)
        else:
            break
    if wave is None or wave.size == 0:
        return {'caseid': caseid, 'waveform':waveform, 'error': 'empty'}, timings
    end = time.perf_counter()
    timings['load_case'] =  end - start
    start = end

    seg = wave[int(opstart*TARGET_SR):int(opend*TARGET_SR)]
    seg = seg.squeeze()
    nan_mask = np.isnan(seg)
    # If more than 5% NaNs, consider invalid
    if np.mean(nan_mask) > 0.05:
        return {'caseid': caseid, 'waveform':waveform, 'error': 'invalid_signal'}, timings
    # If < 5% NaNs, interpolate
    if np.any(nan_mask):
        x = np.arange(seg.size)
        seg = np.interp(
            x=x,
            xp=x[~nan_mask],
            fp=seg[~nan_mask]
        )
    if WIN_SAMP is None and SLIDE_SAMP is None:
        if np.isnan(seg).all() or np.nanstd(seg) < 1e-6: 
            end = time.perf_counter()
            timings['signal_empty'] = end - start
            return {'caseid': caseid, 'waveform':waveform, 'error': 'invalid_signal'}, timings
        all_feature_results = pycatch22.catch22_all(seg, catch24=True)
        out = {n: v for n, v in zip(all_feature_results['names'], all_feature_results['values'])}
        out['caseid'] = caseid
        out['waveform'] = waveform
        out[OUTCOME] = outcome 
        end = time.perf_counter()
        timings['process_signal'] =  end - start
        return out, timings
    else:
        assert WIN_SAMP is not None and SLIDE_SAMP is not None
        if seg.size < WIN_SAMP:
            return {'caseid': caseid, 'waveform':waveform, 'error': 'too_short'}, timings

        feats_list = []
        for i in range(0, seg.size - WIN_SAMP + 1, SLIDE_SAMP):
            win = seg[i:i+WIN_SAMP]
            if np.isnan(win).all() or np.nanstd(win) < 1e-6:
                continue
            all_feature_results = pycatch22.catch22_all(win, catch24=True)
            feats_list.append(all_feature_results)

        if not feats_list:
            return {'caseid': caseid, 'waveform':waveform, 'error': 'no_valid_window'}, timings
        
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
        out['waveform'] = waveform
        out[OUTCOME] = outcome
        return out, timings

def main():
    print("Starting catch22 feature extraction...")

    cohort_df = pd.read_csv(COHORT_FILE)
    print(f"Loaded {len(cohort_df)} cases from {COHORT_FILE}")

    processed_pairs = set()

    if os.path.exists(CATCH_22_FILE):
        print(f"Checkpoint file found at {CATCH_22_FILE}. Loading processed cases.")
        processed_df = pd.read_csv(CATCH_22_FILE)
        processed_pairs = set(zip(processed_df['caseid'], processed_df['waveform']))
        print(f"Found {len(processed_pairs)} previously processed case/waveform pairs.")

    logging.info(f"Reading cohort file: {COHORT_FILE}")
    cohort = pd.read_csv(COHORT_FILE)

    # Expand cohort x waveforms via a small helper DataFrame (more memory-friendly than large Python loops)
    wf_df = pd.DataFrame({'waveform': MANDATORY_WAVEFORMS})
    cohort['__k'] = 1
    wf_df['__k'] = 1
    cohort_expanded = (
        cohort.merge(wf_df, on='__k', how='left')
            .drop(columns='__k')
    )

    original_case_count = len(cohort)
    original_pair_count = len(cohort_expanded)

    # If we have processed pairs, anti-join to remove them
    if processed_pairs:
        processed_pairs_df = pd.DataFrame(list(processed_pairs), columns=['caseid', 'waveform'])
        cohort_to_process = (
            cohort_expanded.merge(processed_pairs_df, on=['caseid', 'waveform'], how='left', indicator=True)
                        .loc[lambda df: df['_merge'] == 'left_only']
                        .drop(columns=['_merge'])
        )
    else:
        cohort_to_process = cohort_expanded

    if cohort_to_process.empty:
        print("All case/waveform pairs have already been processed. Nothing to do.")
        cases_to_process = []
    else:
        # Build the worklist: (caseid, opstart, opend, outcome, waveform)
        cases_to_process = list(
            cohort_to_process[['caseid', 'opstart', 'opend', OUTCOME, 'waveform']]
            .itertuples(index=False, name=None)
        )

    print(
        f"{original_case_count} cases expanded to {original_pair_count} pairs; "
        f"{len(cohort_to_process)} pairs remain to process."
    )

    num_processes = max(1, cpu_count() - 1)
    print(f"Using {num_processes} processes for extraction.")

    all_features = []
    timings = []
    start = time.perf_counter()
    cases_to_process = cases_to_process
    with Pool(processes=num_processes) as pool:
        with tqdm(total=len(cases_to_process), desc="Extracting Features") as pbar:
            for result, timing in pool.imap_unordered(_process_case, cases_to_process):
                all_features.append(result)
                timings.append(timing)
                pbar.update()
    end = time.perf_counter()
    print(f"Feature extraction completed in {end - start:.2f} seconds.")

    features_df = pd.DataFrame(all_features)

    if 'error' in features_df.columns:
        error_df = features_df[features_df['error'].notna()]
        success_df = features_df[features_df['error'].isna()].drop(columns=['error'])
    else:
        error_df = pd.DataFrame()
        success_df = features_df

    if not error_df.empty:
        print(f"Encountered {len(error_df)} errors during processing.")
        print(error_df.head())
        error_df.to_csv(CATCH_22_ERROR_FILE, index=False)
        print(f"Error log saved to {CATCH_22_ERROR_FILE}")

    if not success_df.empty:
        # Fill potential NaN values from std dev calculation with 0
        success_df.fillna(0, inplace=True)

        # Reorder columns to have caseid first for readability
        cols = ['caseid'] + [col for col in success_df.columns if col != 'caseid']
        success_df = success_df[cols]
        print(success_df.head())
        success_df.to_csv(CATCH_22_FILE, index=False)
        
        print(f"Successfully extracted features for {len(success_df)} cases.")
        print(f"Features saved to {CATCH_22_FILE}")
    else:
        print("No features were successfully extracted. Check error log.")
        print(success_df.head())
        
    print("Feature extraction complete.")
    all_timings = defaultdict(lambda: {'count': 0, 'total': 0.0})
    for timing in timings:
        for k, v in timing.items():
            all_timings[k]['count'] += 1
            all_timings[k]['total'] += v
    print("Timing summary (averages over processed cases):")
    for k, v in all_timings.items():
        print(f"  {k}: {v['total']:.4f} sec total over {v['count']} cases (avg {v['total']/v['count']:.4f} sec)")

if __name__ == "__main__":
    main()
