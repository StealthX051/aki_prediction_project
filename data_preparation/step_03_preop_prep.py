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
    OUTCOME
)

# --- Configuration ---
RANDOM_STATE = 42

# All preoperative features to be selected from the cohort file
PREOP_FEATURES_TO_SELECT = [
    'caseid', OUTCOME, 'age', 'sex', 'emop', 'department', 'bmi', 'approach',
    'preop_htn', 'preop_dm', 'preop_ecg', 'preop_pft', 'preop_hb', 'preop_plt',
    'preop_pt', 'preop_aptt', 'preop_na', 'preop_k', 'preop_gluc', 'preop_alb',
    'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr', 'preop_hco3'
]

# Continuous features for outlier handling
CONTINUOUS_COLS = [
    'age', 'bmi', 'preop_hb', 'preop_plt', 'preop_pt', 'preop_aptt', 'preop_na',
    'preop_k', 'preop_gluc', 'preop_alb', 'preop_ast', 'preop_alt',
    'preop_bun', 'preop_cr', 'preop_hco3'
]

# Categorical features for merging and one-hot encoding
CATEGORICAL_COLS = [
    'sex', 'emop', 'department', 'approach', 'preop_htn', 'preop_dm',
    'preop_ecg', 'preop_pft'
]

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
    except FileNotFoundError:
        print(f"ERROR: Cohort file not found at {COHORT_FILE}")
        sys.exit(1)

    # 2. Select Columns
    try:
        preop_df = cohort_df[PREOP_FEATURES_TO_SELECT].copy()
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
