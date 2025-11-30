import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# --- Engineer the AKI Outcome Label (KDIGO Criteria) ---
def add_aki_label(cohort_df: pd.DataFrame) -> pd.DataFrame:
    # Load the lab results data
    print("Loading lab data...")
    lab_df = pd.read_csv('./data/raw/lab_data.csv')

    # Filter for only creatinine labs to make the search faster
    cr_labs = lab_df[lab_df['name'] == 'cr']
    print("Lab data loaded and filtered for creatinine.")

    def get_aki_stage(row, cr_labs_df):
        """
        Applies KDIGO criteria to determine the AKI Stage (0-3) for a single patient row.
        
        Stage 1: Increase >= 0.3 mg/dL (48h) OR Increase >= 1.5-1.9x baseline (7d)
        Stage 2: Increase >= 2.0-2.9x baseline
        Stage 3: Increase >= 3.0x baseline OR Cr >= 4.0 mg/dL
        """
        # Find all creatinine labs for the specific caseid
        case_labs = cr_labs_df[cr_labs_df['caseid'] == row['caseid']]
        
        # Filter for labs that occurred after the operation ended
        postop_labs = case_labs[case_labs['dt'] > row['opend']]
        
        if postop_labs.empty:
            return 0 # No postop labs -> No AKI detected
            
        baseline_cr = row['preop_cr']
        
        # --- Calculate Max Increases ---
        
        # 48h Window (for absolute increase)
        labs_48h = postop_labs[postop_labs['dt'] <= row['opend'] + (48 * 3600)]
        max_cr_48h = labs_48h['result'].max() if not labs_48h.empty else 0
        
        # 7 Day Window (for relative increase)
        labs_7d = postop_labs[postop_labs['dt'] <= row['opend'] + (7 * 24 * 3600)]
        max_cr_7d = labs_7d['result'].max() if not labs_7d.empty else 0
        
        # --- Determine Stage (Check highest stage first) ---
        
        # Stage 3
        # - 3.0x baseline
        # - Increase to >= 4.0 mg/dL (Note: KDIGO also requires acute increase >= 0.3 if initiating RRT, 
        #   but we simplify to >= 4.0 as per common electronic definitions if RRT unknown)
        if (max_cr_7d / baseline_cr >= 3.0) or (max_cr_7d >= 4.0):
            return 3
            
        # Stage 2
        # - 2.0-2.9x baseline
        if (max_cr_7d / baseline_cr >= 2.0):
            return 2
            
        # Stage 1
        # - 1.5-1.9x baseline
        # - Increase >= 0.3 mg/dL within 48h
        if (max_cr_7d / baseline_cr >= 1.5) or ((max_cr_48h - baseline_cr) >= 0.3):
            return 1
            
        return 0

    # Use tqdm to show progress as this can take a moment
    tqdm.pandas(desc="Calculating AKI Stages")

    # Apply the function to each row of the dataframe
    cohort_df['aki_stage'] = cohort_df.progress_apply(lambda row: get_aki_stage(row, cr_labs), axis=1)

    # Derive Binary Outcomes
    # aki_label: Stage >= 1
    cohort_df['aki_label'] = (cohort_df['aki_stage'] >= 1).astype(int)
    
    # y_severe_aki: Stage >= 2
    cohort_df['y_severe_aki'] = (cohort_df['aki_stage'] >= 2).astype("Int64")

    # Report the distribution of the outcome variable
    print("\nAKI Labeling Complete.")
    print("Distribution of AKI Stages:")
    print(cohort_df['aki_stage'].value_counts(dropna=False).sort_index())
    
    print("\nDistribution of AKI Binary (Stage >= 1):")
    print(cohort_df['aki_label'].value_counts(dropna=False))
    
    print("\nDistribution of Severe AKI (Stage >= 2):")
    print(cohort_df['y_severe_aki'].value_counts(dropna=False))
    
    return cohort_df
