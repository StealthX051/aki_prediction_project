import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.artifact_metadata import (
    STEP_01_COHORT_ARTIFACT,
    STEP_03_PREOP_ARTIFACT,
    ArtifactCompatibilityError,
    read_versioned_csv,
    write_artifact_metadata,
    file_sha256,
)
from data_preparation.inputs import (
    COHORT_FILE,
    PREOP_PROCESSED_FILE,
    LAB_DATA_FILE,
    IMPUTE_MISSING,
)
from data_preparation.outcome_registry import (
    ALL_ELIGIBILITY_COLUMNS,
    ALL_OUTCOME_COLUMNS,
    ALL_SPLIT_COLUMNS,
    DEFAULT_OUTCOME_SPEC,
    LEGACY_SPLIT_ALIAS,
    TRAINABLE_OUTCOME_SPECS,
)

# --- Configuration ---
RANDOM_STATE = 42

# All preoperative features to be selected from the cohort file
PREOP_FEATURES_TO_SELECT = [
    'caseid',
    'subjectid',
    *ALL_OUTCOME_COLUMNS,
    *ALL_ELIGIBILITY_COLUMNS,

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
    'adm',              # used to derive inpatient_preop

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
    'preop_crp',
    'preop_lac',
]

# Continuous preoperative features retained for reporting/model-time preprocessing.
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
    'preop_crp',
    'preop_lac',

    # Derived continuous features
    'preop_egfr_ckdepi_2021',
]

# Raw categorical preoperative features retained for reporting/model-time preprocessing.
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
    'crp',   # C-reactive protein
    'lac',   # lactate
]

COHORT_REQUIRED_COLUMNS = [
    *[col for col in PREOP_FEATURES_TO_SELECT if col not in {"preop_wbc", "preop_crp", "preop_lac"}],
    'opstart',
    'opend',
]

def _convert_creatinine_to_mg_dl(scr_series: pd.Series) -> pd.Series:
    """Validate source mg/dL values without inferring units from magnitude."""
    scr_numeric = pd.to_numeric(scr_series, errors='coerce')

    # Treat nonpositive values as invalid before conversion/derivation
    scr_numeric = scr_numeric.where(np.isfinite(scr_numeric) & scr_numeric.gt(0))

    return scr_numeric

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
      * restrict to N days before recording start (default 30),
      * take the finite numeric result with maximum valid dt (closest to 0)
        per (caseid, name),
      * return wide DF with columns: caseid, preop_<lab_name>.
    """
    # Select the last usable result; invalid timestamps/results must not
    # displace an older valid measurement.
    preop = lab_df.copy()
    preop['dt'] = pd.to_numeric(preop['dt'], errors='coerce')
    preop['result'] = pd.to_numeric(preop['result'], errors='coerce')
    preop = preop[np.isfinite(preop['dt']) & np.isfinite(preop['result'])]
    preop = preop[preop['dt'] < 0]

    # Restrict to the configured lookback before recording start.
    window_secs = preop_window_days * 24 * 3600
    preop = preop[preop['dt'] >= -window_secs]

    agg_frames = []
    for lab_name in lab_names:
        tmp = preop[preop['name'] == lab_name].copy()
        if tmp.empty:
            continue

        # Sort so the last result is closest to recording start (dt nearest to 0).
        tmp.sort_values(['caseid', 'dt'], inplace=True)

        last = (
            tmp.groupby('caseid')['result']
               .last()
               .rename(f'preop_{lab_name}')
        )
        agg_frames.append(last)

    if not agg_frames:
        preop_labs = pd.DataFrame(columns=['caseid'])
    else:
        preop_labs = pd.concat(agg_frames, axis=1).reset_index()

    # Ensure all requested labs are present (fill with NaN if missing)
    for lab_name in lab_names:
        col_name = f'preop_{lab_name}'
        if col_name not in preop_labs.columns:
            preop_labs[col_name] = np.nan

    return preop_labs

def add_derived_preop_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive inpatient status and CKD-EPI eGFR from admission, sex, age, and creatinine."""

    # --- Preop inpatient flag ---
    # adm: admission time relative to casestart (sec); preop admissions have adm < 0.
    df['adm'] = pd.to_numeric(df['adm'], errors='coerce')
    df['inpatient_preop'] = (df['adm'] < 0).astype(float).where(np.isfinite(df['adm']))

    # Only the derived admission indicator is a model predictor.
    df.drop(columns=['adm'], inplace=True)

    # --- Creatinine-based eGFR (CKD-EPI 2021, creatinine-only, race-free) ---
    # Normalize only the derivation inputs, preserving source categories.
    sex = df['sex'].astype('string').str.strip().str.upper()
    female = sex.eq('F').fillna(False).to_numpy(dtype=bool)
    age = pd.to_numeric(df['age'], errors='coerce')
    age = age.where(np.isfinite(age))
    scr_mg_dl = _convert_creatinine_to_mg_dl(df['preop_cr'])
    df['preop_cr'] = scr_mg_dl

    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)

    scr_k = scr_mg_dl / kappa
    egfr = 142 * np.minimum(scr_k, 1.0) ** alpha * np.maximum(scr_k, 1.0) ** (-1.200) * (0.9938 ** age)
    egfr *= np.where(female, 1.012, 1.0)

    df['preop_egfr_ckdepi_2021'] = egfr.where(sex.isin(['F', 'M']) & np.isfinite(egfr))

    return df

def select_patient_level_holdout_indices(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    group_col: str = "subjectid",
    target_test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    max_splits: int = 5,
) -> tuple[pd.Index, pd.Index]:
    """Return train/test index labels with patient-disjoint stratified groups."""
    if group_col not in df.columns:
        raise ValueError(f"Required patient group column '{group_col}' not found.")

    groups = df[group_col]
    if groups.isna().any():
        raise ValueError(f"Patient group column '{group_col}' contains missing values.")

    y = df[outcome_col]
    if y.isna().any():
        raise ValueError(f"Outcome column '{outcome_col}' contains missing values.")

    class_counts = y.value_counts()
    if class_counts.empty:
        raise ValueError(f"Outcome column '{outcome_col}' is empty after filtering.")

    unique_groups_per_class = [
        int(groups[y == class_value].nunique()) for class_value in sorted(y.dropna().unique())
    ]
    if not unique_groups_per_class:
        raise ValueError(f"Unable to derive patient-level class counts from '{outcome_col}'.")

    n_splits = min(
        max_splits,
        int(class_counts.min()),
        int(groups.nunique()),
        min(unique_groups_per_class),
    )
    if n_splits < 2:
        raise ValueError(
            "Need at least two patient groups in each class to create a grouped holdout split."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    target_test_rows = max(1, int(round(len(df) * target_test_size)))

    best_train_pos = None
    best_test_pos = None
    best_gap = None

    for train_pos, test_pos in splitter.split(df, y, groups):
        if y.iloc[train_pos].nunique() < 2 or y.iloc[test_pos].nunique() < 2:
            continue
        gap = abs(len(test_pos) - target_test_rows)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_train_pos = train_pos
            best_test_pos = test_pos

    if best_train_pos is None or best_test_pos is None:
        raise ValueError("Failed to construct a patient-level holdout split with both classes in train and test.")

    train_groups = set(groups.iloc[best_train_pos])
    test_groups = set(groups.iloc[best_test_pos])
    overlap = train_groups & test_groups
    if overlap:
        raise AssertionError(f"Patient overlap detected across holdout split: {sorted(overlap)[:5]}")

    return df.index[best_train_pos], df.index[best_test_pos]


def assign_outcome_specific_splits(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    split_df = df.copy()
    split_status: dict[str, dict[str, object]] = {}

    for spec in TRAINABLE_OUTCOME_SPECS:
        split_df[spec.split_col] = pd.Series(pd.NA, index=split_df.index, dtype="object")

    for spec in TRAINABLE_OUTCOME_SPECS:
        eligible_mask = split_df[spec.eligibility_col].eq(1)
        outcome_ready_df = split_df.loc[eligible_mask].dropna(subset=[spec.target_col]).copy()
        status_payload: dict[str, object] = {
            "status": "unsupported_in_artifact",
            "eligible_rows": int(eligible_mask.sum()),
            "labeled_rows": int(len(outcome_ready_df)),
            "split_column": spec.split_col,
        }

        if outcome_ready_df.empty:
            status_payload["reason"] = "no eligible labeled rows available"
            split_status[spec.name] = status_payload
            print(
                f"{spec.name} holdout split unavailable in this artifact: "
                f"{status_payload['reason']}."
            )
            continue

        try:
            train_idx, test_idx = select_patient_level_holdout_indices(
                outcome_ready_df,
                outcome_col=spec.target_col,
                group_col='subjectid',
                target_test_size=0.2,
                random_state=RANDOM_STATE,
            )
        except ValueError as exc:
            status_payload["reason"] = str(exc)
            split_status[spec.name] = status_payload
            print(f"{spec.name} holdout split unavailable in this artifact: {exc}")
            continue

        split_df.loc[train_idx, spec.split_col] = 'train'
        split_df.loc[test_idx, spec.split_col] = 'test'

        assigned = split_df.loc[outcome_ready_df.index, spec.split_col]
        missing_assignments = int(assigned.isna().sum())
        if missing_assignments:
            status_payload["status"] = "invalid"
            status_payload["reason"] = (
                f"{missing_assignments} eligible labeled rows were not assigned to train/test"
            )
        else:
            train_subjects = set(split_df.loc[split_df[spec.split_col] == 'train', 'subjectid'])
            test_subjects = set(split_df.loc[split_df[spec.split_col] == 'test', 'subjectid'])
            overlap = sorted(train_subjects & test_subjects)
            if overlap:
                status_payload["status"] = "invalid"
                status_payload["reason"] = (
                    f"patient overlap detected across train/test split: {overlap[:5]}"
                )
            else:
                status_payload["status"] = "ready"
                status_payload["train_rows"] = int((split_df[spec.split_col] == 'train').sum())
                status_payload["test_rows"] = int((split_df[spec.split_col] == 'test').sum())

        split_status[spec.name] = status_payload
        print(
            f"{spec.name} holdout split status: {status_payload['status']} "
            f"(eligible={status_payload['eligible_rows']}, labeled={status_payload['labeled_rows']})"
        )

    split_df[LEGACY_SPLIT_ALIAS] = split_df[DEFAULT_OUTCOME_SPEC.split_col]
    return split_df, split_status

def main(impute_missing: Optional[bool] = None):
    parser = argparse.ArgumentParser(
        description="Prepare the preoperative dataset and assign the patient-grouped holdout split."
    )
    parser.add_argument(
        "--impute-missing",
        action="store_true",
        default=IMPUTE_MISSING,
        help=(
            "Deprecated compatibility flag. Step 03 no longer performs train-fitted "
            "imputation; missing-value handling now occurs inside model-time folds only."
        ),
    )

    if impute_missing is None:
        args = parser.parse_args()
        impute_missing = args.impute_missing

    print("--- Step 03: Preoperative Data Preparation ---")
    print(f"Input: {COHORT_FILE}")
    print(f"Output: {PREOP_PROCESSED_FILE}")

    # 1. Load Data
    try:
        cohort_df, cohort_metadata = read_versioned_csv(
            COHORT_FILE,
            artifact_role=STEP_01_COHORT_ARTIFACT,
            required_columns=COHORT_REQUIRED_COLUMNS,
        )
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
    except ArtifactCompatibilityError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # 2. Select Columns
    try:
        preop_df = cohort_df[PREOP_FEATURES_TO_SELECT].copy()

        # Add the two derived predictors retained for modeling.
        preop_df = add_derived_preop_features(preop_df)
    except KeyError as e:
        print(f"ERROR: Missing columns in cohort file: {e}")
        sys.exit(1)

    # 3. Train/Test Split
    print("Performing patient-level grouped train/test split (~80/20) per trainable outcome...")
    preop_df, split_status = assign_outcome_specific_splits(preop_df)

    train_mask = preop_df[LEGACY_SPLIT_ALIAS] == 'train'
    test_mask = preop_df[LEGACY_SPLIT_ALIAS] == 'test'

    print(f"Default AKI train set: {int(train_mask.sum())}, Test set: {int(test_mask.sum())}")
    for split_col in ALL_SPLIT_COLUMNS:
        split_counts = preop_df[split_col].value_counts(dropna=False)
        print(f"{split_col} distribution:\n{split_counts}")

    print("Skipping train-fitted preprocessing in Step 03.")
    print("Categorical encoding, outlier handling, and optional imputation now occur inside model-time folds only.")
    if impute_missing:
        print("NOTE: --impute-missing is retained for CLI compatibility only and does not modify Step 03 outputs.")

    # 4. Save
    print(f"Saving processed data to {PREOP_PROCESSED_FILE}...")
    preop_df.to_csv(PREOP_PROCESSED_FILE, index=False)
    write_artifact_metadata(
        PREOP_PROCESSED_FILE,
        artifact_role=STEP_03_PREOP_ARTIFACT,
        available_columns=preop_df.columns,
        extra_metadata={
            "split_status": split_status,
            "upstream_artifact": str(COHORT_FILE),
            "upstream_schema_version": cohort_metadata["schema_version"],
            "source_hashes": cohort_metadata.get("source_hashes", {}),
            "cohort_scopes": cohort_metadata.get("cohort_scopes", {}),
            "outer_flow_counts": cohort_metadata.get("outer_flow_counts", {}),
            "diagnostic_trim": cohort_metadata.get("diagnostic_trim"),
            "data_sha256": file_sha256(PREOP_PROCESSED_FILE),
        },
    )
    print("Done.")

if __name__ == "__main__":
    main()
