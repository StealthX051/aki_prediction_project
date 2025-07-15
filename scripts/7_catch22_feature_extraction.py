#!/usr/bin/env python
# 7_generate_catch22_features_sliding_window_fast.py
#
# Accelerated sliding-window catch22 feature extraction
# – 100 Hz load from VitalDB, pure-NumPy aggregation, sane multiprocessing.
#
# **FIXED**: Corrected the final file-saving logic to be more robust and
# prevent silent failures after processing is complete.
# -----------------------------------------------------------------------------

# --- Top-level print statement for debugging silent exits ---
print("--- Script starting execution ---")

import os
import logging
from multiprocessing import Pool, cpu_count
from typing import Dict, Any, Tuple
import traceback

# This outer try/except block is a safety net to catch ANY error during startup.
try:
    import numpy as np
    import pandas as pd
    import vitaldb
    import pycatch22
    from tqdm import tqdm
    print("--- All imports successful ---")

    # ------------------------------------------------------------------------------
    # CONFIGURATION
    # ------------------------------------------------------------------------------
    print("--- Entering configuration block ---")
    
    # ** Using a more robust pathing method that works in more environments **
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    except NameError:
        PROJECT_ROOT = os.getcwd()
        if os.path.basename(PROJECT_ROOT) == 'scripts':
            PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)
            
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    print(f"--- Project Root detected as: {PROJECT_ROOT}")
    print(f"--- Processed Data Directory set to: {PROCESSED_DIR}")

    COHORT_FILE   = 'final_cohort_with_death_label.csv'
    OUT_FILE      = 'waveform_catch22_features_sliding_window.csv'
    ERROR_FILE    = 'catch22_sliding_window_errors.csv'

    WAVEFORM      = ['SNUADC/PLETH']
    TARGET_SR     = 100
    WIN_SEC       = 10 * 60
    SLIDE_SEC     = 5  * 60
    WIN_SAMP      = WIN_SEC   * TARGET_SR
    SLIDE_SAMP    = SLIDE_SEC * TARGET_SR

    print("--- Basic configuration loaded ---")
    print("--- Configuration block loaded successfully ---")

    # ------------------------------------------------------------------------------
    # WORKER
    # ------------------------------------------------------------------------------
    def _process_case(case: Tuple[int, float, float, int]) -> Dict[str, Any]:
        caseid, opstart, opend, death = case
        try:
            wave = vitaldb.load_case(caseid, WAVEFORM, 1 / TARGET_SR)
            if wave is None or wave.size == 0:
                return {'caseid': caseid, 'error': 'empty'}

            seg = wave[int(opstart*TARGET_SR):int(opend*TARGET_SR)]
            if seg.size < WIN_SAMP:
                return {'caseid': caseid, 'error': 'too_short'}

            feats_list = []
            for i in range(0, seg.size - WIN_SAMP + 1, SLIDE_SAMP):
                win = seg[i:i+WIN_SAMP]
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
                **{f'{n}_std':  v for n, v in zip(feature_names, m.std(0, ddof=1))}, # Use ddof=1 for sample std
                **{f'{n}_min':  v for n, v in zip(feature_names, m.min(0))},
                **{f'{n}_max':  v for n, v in zip(feature_names, m.max(0))}
            }
            out['caseid'] = caseid
            out['death_label'] = death
            return out

        except Exception as exc:
            return {'caseid': caseid, 'error': str(exc)}

    # ------------------------------------------------------------------------------
    # MAIN
    # ------------------------------------------------------------------------------
    def main() -> None:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(levelname)-8s  %(message)s')
        
        output_filepath = os.path.join(PROCESSED_DIR, OUT_FILE)

        processed_cases = set()
        if os.path.exists(output_filepath):
            logging.info(f"Checkpoint file found at {output_filepath}. Loading processed cases.")
            try:
                processed_df = pd.read_csv(output_filepath)
                processed_cases = set(processed_df['caseid'])
                logging.info(f"Found {len(processed_cases)} previously processed cases.")
            except Exception as e:
                logging.error(f"Could not read checkpoint file. It might be corrupted. Error: {e}")
                return

        logging.info(f"Reading cohort file: {os.path.join(PROCESSED_DIR, COHORT_FILE)}")
        cohort = pd.read_csv(os.path.join(PROCESSED_DIR, COHORT_FILE))
        
        original_case_count = len(cohort)
        cohort_to_process = cohort[~cohort['caseid'].isin(processed_cases)]
        
        if cohort_to_process.empty:
            logging.info("All cases have already been processed. Nothing to do.")
            return
            
        logging.info(f"Skipping {original_case_count - len(cohort_to_process)} cases. Processing {len(cohort_to_process)} new cases.")
        jobs = cohort_to_process[['caseid', 'opstart', 'opend', 'death_label']].itertuples(index=False, name=None)

        n_workers = max(1, cpu_count() - 1)
        logging.info("Starting catch22 extraction with %d workers @ %d Hz…", n_workers, TARGET_SR)

        results = []
        with Pool(n_workers) as pool, tqdm(total=len(cohort_to_process), desc='catch22->CSV') as bar:
            for res in pool.imap_unordered(_process_case, jobs, chunksize=8):
                results.append(res)
                bar.update()

        df = pd.DataFrame(results)

        logging.info("Saving diagnostics and results...")
        errs = df[df.error.notna()]
        if not errs.empty:
            errs.to_csv(os.path.join(PROCESSED_DIR, ERROR_FILE), index=False)
            logging.warning("Errors in %d / %d cases (see %s).", len(errs), len(df), ERROR_FILE)

        ok = df[df.error.isna()].drop(columns='error')
        if ok.empty:
            logging.info("No new features were successfully extracted in this run.")
            return

        # ** ROBUST SAVING LOGIC **
        # This simplified logic ensures the file saves correctly.
        ok.fillna(0, inplace=True)
        
        # Define a consistent column order
        feature_cols = sorted([col for col in ok.columns if col not in ['caseid', 'death_label']])
        final_cols = ['caseid', 'death_label'] + feature_cols
        ok = ok[final_cols]

        if os.path.exists(output_filepath):
            logging.info(f"Appending {len(ok)} new rows to {OUT_FILE}")
            ok.to_csv(output_filepath, mode='a', header=False, index=False)
        else:
            logging.info(f"Creating new file and saving {len(ok)} rows to {OUT_FILE}")
            ok.to_csv(output_filepath, index=False)
            
        logging.info("Save operation complete.")

    # ------------------------------------------------------------------------------
    # SCRIPT ENTRY POINT
    # ------------------------------------------------------------------------------
    if __name__ == '__main__':
        print("--- Calling main() ---")
        os.chdir(PROJECT_ROOT) # Set working directory for vitaldb
        main()
        print("--- main() finished ---")

except Exception as e:
    print(f"\n--- A FATAL ERROR OCCURRED DURING SCRIPT INITIALIZATION ---\n")
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Details: {e}")
    print("\n--- Traceback ---")
    traceback.print_exc()
