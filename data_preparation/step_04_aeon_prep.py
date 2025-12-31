import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import (
    PREOP_PROCESSED_FILE,
    AEON_OUT_DIR,
    OUTCOME,
    COHORT_FILE,
    IMPUTE_MISSING,
)

def main(impute_missing: Optional[bool] = None):
    parser = argparse.ArgumentParser(
        description="Prepare Aeon-ready preop dataset with optional imputation."
    )
    parser.add_argument(
        "--impute-missing",
        action="store_true",
        default=IMPUTE_MISSING,
        help=(
            "Impute missing values using median strategy with indicators. By"
            " default, NaNs are preserved in the output."
        ),
    )

    if impute_missing is None:
        args = parser.parse_args()
        impute_missing = args.impute_missing
    else:
        _ = parser.parse_args([])

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting Aeon Preop Preparation...")

    # Define paths
    aeon_y_path = Path(AEON_OUT_DIR) / 'y_nonwindowed.csv'
    out_file = Path(AEON_OUT_DIR) / 'aki_preop_aeon.csv'

    # 1. Load Data
    if not aeon_y_path.exists():
        logging.error(f"Aeon labels file not found: {aeon_y_path}. Run step_02_aeon_export.py first.")
        return

    if not Path(PREOP_PROCESSED_FILE).exists():
        logging.error(f"Preop processed file not found: {PREOP_PROCESSED_FILE}. Run step_03_preop_prep.py first.")
        return

    logging.info(f"Loading preop data from {PREOP_PROCESSED_FILE}")
    preop_df = pd.read_csv(PREOP_PROCESSED_FILE)
    
    logging.info(f"Loading aeon labels from {aeon_y_path}")
    y_aeon = pd.read_csv(aeon_y_path)
    
    # 2. Filter Preop to match Aeon cases
    # Aeon export might have dropped cases due to strict channel checks
    valid_caseids = set(y_aeon['caseid'])
    preop_df = preop_df[preop_df['caseid'].isin(valid_caseids)].copy()
    logging.info(f"Filtered preop data to {len(preop_df)} cases matching Aeon export.")

    # 3. Add Anesthesia Duration
    # We need opstart/opend from cohort file or original source
    # Ideally, step_03 should have kept them, but let's check input columns.
    # step_03 output usually drops raw times. 
    # We should reload cohort file to get opstart/opend for these cases.
    logging.info("Deriving anesthesia_duration_minutes...")
    cohort_df = pd.read_csv(COHORT_FILE, usecols=['caseid', 'opstart', 'opend'])
    
    preop_df = preop_df.merge(cohort_df, on='caseid', how='left')
    preop_df['anesthesia_duration_minutes'] = (preop_df['opend'] - preop_df['opstart']) / 60.0
    
    # Check for negative or missing durations
    invalid_dur = preop_df['anesthesia_duration_minutes'] <= 0
    if invalid_dur.any():
        logging.warning(f"Found {invalid_dur.sum()} cases with invalid duration. Filling with median.")
        med_dur = preop_df.loc[~invalid_dur, 'anesthesia_duration_minutes'].median()
        preop_df.loc[invalid_dur, 'anesthesia_duration_minutes'] = med_dur

    preop_df.drop(columns=['opstart', 'opend'], inplace=True)

    # 4. Imputation Strategy
    # Exclude non-feature columns from imputation
    meta_cols = ['caseid', OUTCOME, 'split_group', 'y_severe_aki', 'y_inhosp_mortality', 'y_icu_admit', 'y_prolonged_los_postop']
    feature_cols = [c for c in preop_df.columns if c not in meta_cols]

    if impute_missing:
        logging.info("Replacing -99 with NaN before imputation...")
        preop_df[feature_cols] = preop_df[feature_cols].replace(-99, np.nan)

        # Split Train/Test for fitting imputer
        train_mask = preop_df['split_group'] == 'train'
        X_train = preop_df.loc[train_mask, feature_cols]

        if X_train.empty:
            logging.error("No training data found! Check split_group column.")
            return

        logging.info("Fitting SimpleImputer(median, add_indicator=True) on training set feature columns...")
        imputer = SimpleImputer(strategy='median', add_indicator=True)
        imputer.fit(X_train)

        # Transform whole dataset
        X_imputed = imputer.transform(preop_df[feature_cols])

        # Get new column names (features + indicators)
        # get_feature_names_out available in sklearn > 1.0
        try:
            new_cols = imputer.get_feature_names_out(feature_cols)
        except AttributeError:
            # Fallback for older sklearn
            new_cols = feature_cols + [
                f"{c}_missing" for c, idx in zip(feature_cols, imputer.indicator_.features_)
                if idx < len(feature_cols)
            ]
            pass

        df_imputed = pd.DataFrame(X_imputed, columns=new_cols, index=preop_df.index)

        # Reassemble
        final_df = pd.concat([preop_df[meta_cols], df_imputed], axis=1)

        # Save imputer for inference reference
        joblib.dump(imputer, Path(AEON_OUT_DIR) / 'preop_imputer.joblib')
        logging.info("Saved imputer to outputs for reference.")
    else:
        logging.info("Skipping imputation; preserving NaNs in feature columns.")
        final_df = preop_df

    # 5. Save
    logging.info(f"Saving Aeon-ready preop data to {out_file}...")
    final_df.to_csv(out_file, index=False)
    logging.info("Complete.")

if __name__ == "__main__":
    main()
