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

    # This is the function definition from your project plan
    def get_aki_label(row, cr_labs_df):
        """
        Applies KDIGO criteria to determine the AKI label for a single patient row.
        """
        # Find all creatinine labs for the specific caseid
        case_labs = cr_labs_df[cr_labs_df['caseid'] == row['caseid']]
        
        # Filter for labs that occurred after the operation ended
        postop_labs = case_labs[case_labs['dt'] > row['opend']]
        
        if postop_labs.empty:
            return None # This case should have already been excluded, but as a safeguard.
            
        baseline_cr = row['preop_cr']
        
        # Condition A: Increase of >= 0.3 mg/dL within 48 hours 
        labs_48h = postop_labs[postop_labs['dt'] <= row['opend'] + (48 * 3600)]
        condition_A = False
        if not labs_48h.empty:
            max_cr_48h = labs_48h['result'].max()
            if (max_cr_48h - baseline_cr) >= 0.3:
                condition_A = True
                
        # Condition B: Increase of >= 1.5x baseline within 7 days 
        labs_7d = postop_labs[postop_labs['dt'] <= row['opend'] + (7 * 24 * 3600)]
        condition_B = False
        if not labs_7d.empty:
            max_cr_7d = labs_7d['result'].max()
            if (max_cr_7d / baseline_cr) >= 1.5:
                condition_B = True
                
        return 1 if condition_A or condition_B else 0 # 

    # Use tqdm to show progress as this can take a moment
    tqdm.pandas(desc="Applying AKI criteria")

    # Apply the function to each row of the dataframe
    cohort_df['aki_label'] = cohort_df.progress_apply(lambda row: get_aki_label(row, cr_labs), axis=1)

    # Report the distribution of the outcome variable
    print("\nAKI Labeling Complete.")
    print("Distribution of AKI labels:")
    print(cohort_df['aki_label'].value_counts(dropna=False))
    return cohort_df
