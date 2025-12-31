import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import (
    COHORT_FILE,
    PREOP_PROCESSED_FILE,
    OUTCOME,
    LAB_DATA_FILE
)

# --- Configuration ---
RANDOM_STATE = 42

# All preoperative features to be selected from the cohort file
# All preoperative features to be selected from the cohort file
PREOP_FEATURES_TO_SELECT = [
    'caseid',
    OUTCOME,
    'y_severe_aki',
    'y_inhosp_mortality',
    'y_icu_admit',
    'y_prolonged_los_postop',

    # Demographics & body size
    'age',
    'sex',
    'height',
    'weight',
    'bmi',

    # Surgical / anesthetic context
    'emop',
    'department',
    'approach',
    'asa',
    'optype',
    'ane_type',
    'adm',              # used to derive LOS and inpatient_preop

    # Comorbidities / preop tests
    'preop_htn',
    'preop_dm',
    'preop_ecg',
    'preop_pft',

    # Hematology / coagulation
    'preop_hb',
    'preop_plt',
    'preop_pt',
    'preop_aptt',

    # Electrolytes / metabolic / liver
    'preop_na',
    'preop_k',
    'preop_gluc',
    'preop_alb',
    'preop_ast',
    'preop_alt',
    'preop_bun',
    'preop_cr',
    'preop_hco3',

    # ABG components from clinical_data
    'preop_ph',
    'preop_be',
    'preop_pao2',
    'preop_paco2',
    'preop_sao2',

    # Extra preop labs from lab_data
    'preop_wbc',
    'preop_gfr',
    'preop_crp',
    'preop_lac',
]

# Continuous features for outlier handling
# Continuous features for outlier handling
CONTINUOUS_COLS = [
    # Demographics / body size
    'age',
    'height',
    'weight',
    'bmi',

    # Hematology / coagulation
    'preop_hb',
    'preop_plt',
    'preop_pt',
    'preop_aptt',

    # Electrolytes / metabolic / liver
    'preop_na',
    'preop_k',
    'preop_gluc',
    'preop_alb',
    'preop_ast',
    'preop_alt',
    'preop_bun',
    'preop_cr',
    'preop_hco3',

    # ABG
    'preop_ph',
    'preop_be',
    'preop_pao2',
    'preop_paco2',
    'preop_sao2',

    # Labs from lab_data
    'preop_wbc',
    'preop_gfr',
    'preop_crp',
    'preop_lac',

    # Derived continuous features
    'preop_los_days',
    'preop_egfr_ckdepi',
]

# Categorical features for merging and one-hot encoding
# Categorical features for merging and one-hot encoding
CATEGORICAL_COLS = [
    'sex',
    'emop',
    'department',
    'approach',
    'asa',
    'optype',
    'ane_type',
    'preop_htn',
    'preop_dm',
    'preop_ecg',
    'preop_pft',
]

# Labs we want to derive preop values for.
# Names are exactly as in VitalDB lab_data. 
PREOP_LABS_FROM_LABDATA = [
    'wbc',   # white blood cells
    'gfr',   # eGFR from lab table
    'crp',   # C-reactive protein
    'lac',   # lactate
]

def build_preop_labs_from_labdata(lab_df: pd.DataFrame,
                                  lab_names=PREOP_LABS_FROM_LABDATA,
                                  preop_window_days: int = 30) -> pd.DataFrame:
    """
    Compute one 'last preop' value per lab per case from VitalDB lab_data.csv.

    lab_df columns (VitalDB): caseid, dt, name, result. 
      - dt: seconds from casestart; preop labs have dt < 0.
      - name: lab name (e.g., 'wbc', 'gfr', 'crp', 'lac').
      - result: lab value as string.

    We:
      * keep rows with dt < 0 (preop),
      * restrict to a window of N days before surgery (default 30),
      * take the row with the maximum dt (closest to 0) per (caseid, name),
      * return wide DF with columns: caseid, preop_<lab_name>.
    """
    # Preoperative only
    preop = lab_df[lab_df['dt'] < 0].copy()

    # Restrict to e.g. last 30 days before surgery
    window_secs = preop_window_days * 24 * 3600
    preop = preop[preop['dt'] >= -window_secs]

    # Numeric results
    preop['result'] = pd.to_numeric(preop['result'], errors='coerce')

    agg_frames = []
    for lab_name in lab_names:
        tmp = preop[preop['name'] == lab_name].copy()
        if tmp.empty:
            continue

        # Sort so last row is closest to surgery (dt nearest to 0)
        tmp.sort_values(['caseid', 'dt'], inplace=True)

        last = (
            tmp.groupby('caseid')['result']
               .last()
               .rename(f'preop_{lab_name}')
        )
        agg_frames.append(last)

    if not agg_frames:
        return pd.DataFrame(columns=['caseid'])

    preop_labs = pd.concat(agg_frames, axis=1).reset_index()
    return preop_labs

def add_derived_preop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived preop features using existing columns:
      - adm: admission time from casestart (sec) from clinical_data.csv
      - preop_bun, preop_cr, preop_alb, preop_hb, preop_na, preop_hco3, preop_be,
        preop_pao2, preop_paco2, preop_sao2, etc.
    """

    # --- Preop hospital LOS in days and inpatient flag ---
    # adm: admission time relative to casestart (sec); preop admissions have adm < 0.
    df['adm'] = pd.to_numeric(df['adm'], errors='coerce')

    df['preop_los_days'] = 0.0
    admitted_mask = df['adm'] < 0
    df.loc[admitted_mask, 'preop_los_days'] = (
        -df.loc[admitted_mask, 'adm'] / (24 * 3600.0)
    )
    df['inpatient_preop'] = (df['adm'] < 0).astype(int)

    # Optionally drop raw adm if you don't want it as a feature
    # (we'll still keep it in PREOP_FEATURES_TO_SELECT so we can derive these)
    df.drop(columns=['adm'], inplace=True)

    # --- Creatinine-based eGFR (CKD-EPI 2009, race-free) ---
    # Uses sex, age, preop_cr. Units: Scr mg/dL, age years.
    sex = df['sex'].astype(str)
    age = pd.to_numeric(df['age'], errors='coerce')
    scr = pd.to_numeric(df['preop_cr'], errors='coerce')

    kappa = np.where(sex == 'F', 0.7, 0.9)
    alpha = np.where(sex == 'F', -0.329, -0.411)

    scr_k = scr / kappa
    egfr = 141 * np.minimum(scr_k, 1.0) ** alpha * np.maximum(scr_k, 1.0) ** (-1.209) * (0.993 ** age)
    egfr *= np.where(sex == 'F', 1.018, 1.0)

    df['preop_egfr_ckdepi'] = egfr

    # --- Simple binary clinical flags (0/1) ---
    # BUN high
    df['bun_high'] = (pd.to_numeric(df['preop_bun'], errors='coerce') > 27).astype(int)

    # Hypoalbuminemia
    df['hypoalbuminemia'] = (pd.to_numeric(df['preop_alb'], errors='coerce') < 3.5).astype(int)

    # Sex-specific anemia
    preop_hb = pd.to_numeric(df['preop_hb'], errors='coerce')
    male = (df['sex'] == 'M')
    female = (df['sex'] == 'F')
    anemia = pd.Series(0, index=df.index, dtype=int)
    anemia[male & (preop_hb < 13.0)] = 1
    anemia[female & (preop_hb < 12.0)] = 1
    df['preop_anemia'] = anemia

    # Hyponatremia
    df['hyponatremia'] = (pd.to_numeric(df['preop_na'], errors='coerce') < 135).astype(int)

    # Metabolic acidosis
    preop_hco3 = pd.to_numeric(df['preop_hco3'], errors='coerce')
    preop_be = pd.to_numeric(df['preop_be'], errors='coerce')
    df['metabolic_acidosis'] = ((preop_hco3 < 22) | (preop_be < -2)).astype(int)

    # Hypercapnia
    preop_paco2 = pd.to_numeric(df['preop_paco2'], errors='coerce')
    df['hypercapnia'] = (preop_paco2 > 45).astype(int)

    # Hypoxemia
    preop_pao2 = pd.to_numeric(df['preop_pao2'], errors='coerce')
    preop_sao2 = pd.to_numeric(df['preop_sao2'], errors='coerce')
    df['hypoxemia'] = ((preop_pao2 < 80) | (preop_sao2 < 95)).astype(int)

    return df

def handle_outliers(df, train_df, continuous_cols):
    """
    Handles outliers based on training set percentiles.
    Replaces outliers with random values from a plausible range.
    """
    df_processed = df.copy()
    for col in continuous_cols:
        if col in df_processed.columns:
            
            # 1. Create a clean, numeric version of the training data column
            train_col_numeric = pd.to_numeric(train_df[col], errors='coerce')
            
            # 2. Force the column in the dataframe being processed to also be numeric
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

            # Calculate percentile thresholds ONLY from the numeric training column
            train_col_numeric.dropna(inplace=True)
            if train_col_numeric.empty:
                print(f"Warning: No valid data for '{col}' in training set. Skipping outlier handling for this column.")
                continue

            low_p_0_5, low_p_1, low_p_5 = np.percentile(train_col_numeric, [0.5, 1, 5])
            high_p_95, high_p_99_5 = np.percentile(train_col_numeric, [95, 99.5])
            
            # Identify outlier indices in the processed dataframe
            low_outlier_indices = df_processed[df_processed[col] < low_p_1].index
            high_outlier_indices = df_processed[df_processed[col] > high_p_99_5].index
            
            # Replace with random values from the specified plausible range
            low_replacements = np.random.uniform(low_p_0_5, low_p_5, size=len(low_outlier_indices))
            high_replacements = np.random.uniform(high_p_95, high_p_99_5, size=len(high_outlier_indices))
            
            df_processed.loc[low_outlier_indices, col] = low_replacements
            df_processed.loc[high_outlier_indices, col] = high_replacements
    return df_processed

def main():
    print("--- Step 03: Preoperative Data Preparation ---")
    print(f"Input: {COHORT_FILE}")
    print(f"Output: {PREOP_PROCESSED_FILE}")

    # 1. Load Data
    try:
        cohort_df = pd.read_csv(COHORT_FILE)
        print(f"Loaded cohort data. Shape: {cohort_df.shape}")

        # 1b. Merge extra preop labs from lab_data.csv using dt < 0
        try:
            lab_df = pd.read_csv(LAB_DATA_FILE)
            lab_df = lab_df[lab_df['caseid'].isin(cohort_df['caseid'])].copy()
            preop_labs = build_preop_labs_from_labdata(lab_df)
            print(f"Merging preop labs from lab_data.csv. Shape: {preop_labs.shape}")
            cohort_df = cohort_df.merge(preop_labs, on='caseid', how='left')
        except FileNotFoundError:
            print(f"WARNING: lab_data file not found at {LAB_DATA_FILE}. "
                  "Skipping additional preop labs from lab_data.csv.")
    except FileNotFoundError:
        print(f"ERROR: Cohort file not found at {COHORT_FILE}")
        sys.exit(1)

    # 2. Select Columns
    try:
        preop_df = cohort_df[PREOP_FEATURES_TO_SELECT].copy()

        # Add derived preop features (LOS, eGFR, binary flags)
        preop_df = add_derived_preop_features(preop_df)

        # Drop derived features not used for modeling
        cols_to_drop = ['preop_los_days', 'inpatient_preop', 'position']
        existing_drop_cols = [col for col in cols_to_drop if col in preop_df.columns]
        if existing_drop_cols:
            preop_df = preop_df.drop(columns=existing_drop_cols)
    except KeyError as e:
        print(f"ERROR: Missing columns in cohort file: {e}")
        sys.exit(1)

    # 3. Train/Test Split
    print("Performing stratified train/test split (80/20)...")
    X = preop_df.drop(columns=[OUTCOME])
    y = preop_df[OUTCOME]
    
    # We split indices to keep track of rows
    indices = np.arange(len(preop_df))
    
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.2, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    
    # Assign split group
    preop_df['split_group'] = 'train' # Default
    preop_df.loc[test_idx, 'split_group'] = 'test'
    
    # Create views for processing (we process the whole DF but use train stats)
    train_mask = preop_df['split_group'] == 'train'
    test_mask = preop_df['split_group'] == 'test'
    
    X_train = preop_df.loc[train_mask].copy()
    X_test = preop_df.loc[test_mask].copy()
    
    print(f"Train set: {len(X_train)}, Test set: {len(X_test)}")

    # 4. Categorical Processing
    print("Processing categorical variables...")
    
    # Merge small departments (using train stats)
    dept_counts = X_train['department'].value_counts()
    depts_to_merge = dept_counts[dept_counts < 30].index.tolist()
    
    if depts_to_merge:
        print(f"Merging {len(depts_to_merge)} departments into 'other'")
        preop_df['department'] = preop_df['department'].replace(depts_to_merge, 'other')
        # Update views
        X_train = preop_df.loc[train_mask] 

    # One-Hot Encoding
    print("Applying one-hot encoding...")
    # We encode the whole dataframe to ensure consistent columns, 
    # BUT we must be careful not to introduce leakage if new categories appear in test.
    # Ideally, we fit on train and transform test. 
    # Using pandas get_dummies on the whole set is common but technically leaks the *existence* of categories.
    # Given the 'department' merge above used only train stats, we are mostly safe.
    # A stricter approach would be to fit a OneHotEncoder on train.
    # For now, we'll stick to the pandas approach but align columns based on train.
    
    X_train_dummies = pd.get_dummies(X_train[CATEGORICAL_COLS], drop_first=True, dtype=int)
    X_test_dummies = pd.get_dummies(X_test[CATEGORICAL_COLS], drop_first=True, dtype=int)
    
    # Align columns to Train
    X_train_aligned, X_test_aligned = X_train_dummies.align(
        X_test_dummies, join='left', axis=1, fill_value=0
    )
    
    # Reconstruct the full dataframe with encoded columns
    # We do this by concatenating the aligned parts
    encoded_cols = pd.concat([X_train_aligned, X_test_aligned]).sort_index()
    
    # Drop original categorical and add encoded
    preop_df_encoded = preop_df.drop(columns=CATEGORICAL_COLS)
    preop_df_encoded = pd.concat([preop_df_encoded, encoded_cols], axis=1)

    # 5. Outlier Handling (Continuous)
    print("Handling outliers...")
    # We pass the full dataframe to be processed, but pass the TRAIN subset for stats calculation
    train_subset = preop_df_encoded.loc[train_mask]
    preop_df_cleaned = handle_outliers(preop_df_encoded, train_subset, CONTINUOUS_COLS)

    # 6. Imputation
    print("Imputing missing values with -99...")
    preop_df_final = preop_df_cleaned.fillna(-99)
    
    # 7. Save
    print(f"Saving processed data to {PREOP_PROCESSED_FILE}...")
    preop_df_final.to_csv(PREOP_PROCESSED_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
