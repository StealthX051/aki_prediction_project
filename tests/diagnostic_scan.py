import pandas as pd
import numpy as np
import vitaldb
import os
import logging
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import Counter
import time

from data_preparation.inputs import (
    COHORT_FILE,
    MANDATORY_WAVEFORMS,
    WAVEFORM_SUBSTITUTIONS,
    WIN_SEC,
    TARGET_SR
)
from data_preparation.waveform_processing import WAVEFORM_SPECS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create mapping from VitalDB ID to Spec Key
ID_TO_SPEC_KEY = {spec['id']: key for key, spec in WAVEFORM_SPECS.items()}

def check_case(case_info):
    """
    Checks a single case for data quality issues for a specific waveform.
    """
    caseid, opstart, opend, waveform_key = case_info
    
    try:
        # Resolve spec key
        spec_key = ID_TO_SPEC_KEY.get(waveform_key)
        if not spec_key and waveform_key in WAVEFORM_SPECS:
            spec_key = waveform_key
            
        if not spec_key:
             return {'caseid': caseid, 'waveform': waveform_key, 'status': 'config_error_unknown_waveform'}

        spec = WAVEFORM_SPECS[spec_key]
        native_sr = spec['native_sr']
        
        # 1. Attempt to load primary
        wave = vitaldb.load_case(caseid, spec['id'], interval=1/native_sr)
        
        # 2. Attempt substitutions if primary missing
        if wave is None or wave.size == 0:
            for sub_id in WAVEFORM_SUBSTITUTIONS.get(waveform_key, []):
                sub_spec_key = ID_TO_SPEC_KEY.get(sub_id)
                if not sub_spec_key and sub_id in WAVEFORM_SPECS:
                    sub_spec_key = sub_id
                
                if sub_spec_key:
                    try:
                        sub_spec = WAVEFORM_SPECS[sub_spec_key]
                        wave = vitaldb.load_case(caseid, sub_spec['id'], interval=1/sub_spec['native_sr'])
                        if wave is not None and wave.size > 0:
                            spec = sub_spec # Use substitute spec
                            native_sr = spec['native_sr']
                            break
                    except:
                        continue
        
        if wave is None or wave.size == 0:
            return {'caseid': caseid, 'waveform': waveform_key, 'status': 'missing_data_empty_load'}

        # 3. Slice to operation time
        start_idx = int(opstart * native_sr)
        end_idx = int(opend * native_sr)
        
        if start_idx >= wave.size:
             return {'caseid': caseid, 'waveform': waveform_key, 'status': 'missing_data_outside_window'}
        
        # Handle end index clamping but note if it was significantly short? 
        # For now just slice.
        seg = wave[start_idx:end_idx]
        seg = seg.squeeze()
        
        if seg.size == 0:
             return {'caseid': caseid, 'waveform': waveform_key, 'status': 'missing_data_empty_segment'}

        # 4. Check Length (for windowing)
        # We need at least one window if WIN_SEC is defined
        if WIN_SEC:
             # We need to account for resampling. 
             # The check in step_02 is: seg.size < win_samp (at target_sr)
             # So we should approximate this check at native_sr or just check duration
             duration = seg.size / native_sr
             if duration < WIN_SEC:
                  return {'caseid': caseid, 'waveform': waveform_key, 'status': f'too_short_<{WIN_SEC}s'}

        # 5. NaN Check
        nan_mask = np.isnan(seg)
        nan_pct = np.mean(nan_mask)
        if nan_pct > 0.05:
             return {'caseid': caseid, 'waveform': waveform_key, 'status': 'invalid_>5%_nan'}
        
        if nan_pct > 0:
             # Interpolate for flatline check
             x = np.arange(seg.size)
             seg = np.interp(x=x, xp=x[~nan_mask], fp=seg[~nan_mask])

        # 6. Flatline Check
        if np.nanstd(seg) < 1e-6:
             return {'caseid': caseid, 'waveform': waveform_key, 'status': 'invalid_flatline'}

        return {'caseid': caseid, 'waveform': waveform_key, 'status': 'valid'}

    except Exception as e:
        return {'caseid': caseid, 'waveform': waveform_key, 'status': f'error_{str(e)}'}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Diagnostic scan of cohort data.')
    parser.add_argument('--limit', type=int, help='Limit the number of cases to scan (for testing).')
    args = parser.parse_args()

    print("Starting Diagnostic Scan...")
    
    if not os.path.exists(COHORT_FILE):
        print(f"Cohort file not found: {COHORT_FILE}")
        return

    cohort_df = pd.read_csv(COHORT_FILE)
    print(f"Loaded {len(cohort_df)} cases from cohort.")
    
    if args.limit:
        cohort_df = cohort_df.head(args.limit)
        print(f"Limiting to first {args.limit} cases.")

    # Prepare worklist
    # We want to check EVERY mandatory waveform for EVERY case
    worklist = []
    for row in cohort_df.itertuples():
        for wf in MANDATORY_WAVEFORMS:
            worklist.append((row.caseid, row.opstart, row.opend, wf))
    
    print(f"Scanning {len(worklist)} case-waveform pairs...")
    
    results = []
    with Pool(processes=max(1, cpu_count() - 2)) as pool:
        for res in tqdm(pool.imap_unordered(check_case, worklist), total=len(worklist)):
            results.append(res)
            
    # Save results
    results_df = pd.DataFrame(results)
    out_file = './data/processed/diagnostic_report.csv'
    results_df.to_csv(out_file, index=False)
    print(f"\nDetailed results saved to {out_file}")
    
    # Generate Summary
    print("\n=== Diagnostic Summary ===")
    if not results_df.empty:
        summary = results_df.groupby(['waveform', 'status']).size().unstack(fill_value=0)
        
        # Add percentages
        summary_pct = summary.div(summary.sum(axis=1), axis=0) * 100
        
        print(summary)
        print("\n--- Percentages ---")
        print(summary_pct.round(1))
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
