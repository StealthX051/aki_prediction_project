import os
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("PROCESSED_DIR", PROJECT_ROOT / 'data' / 'processed'))

# Results layout (defaults to the reorganized Catch22 tree)
RESULTS_ROOT = Path(os.getenv("RESULTS_DIR", PROJECT_ROOT / "results" / "catch22" / "experiments"))
PAPER_ROOT = Path(os.getenv("PAPER_DIR", RESULTS_ROOT.parent / "paper"))
XAI_ROOT = Path(os.getenv("XAI_DIR", RESULTS_ROOT.parent / "xai"))

# Backwards-compatible alias used throughout the codebase for experiment outputs
RESULTS_DIR = RESULTS_ROOT

# Input Files
FULL_FEATURES_FILE = DATA_DIR / 'aki_features_master_wide.csv'
WINDOWED_FEATURES_FILE = DATA_DIR / 'aki_features_master_wide_windowed.csv'

# Outcomes
OUTCOMES = {
    'any_aki': 'aki_label',
    'severe_aki': 'y_severe_aki',
    'mortality': 'y_inhosp_mortality',
    'extended_los': 'y_prolonged_los_postop',
    'icu_admission': 'y_icu_admit'
}

# Waveform Prefixes
WAVEFORM_PREFIXES = {
    'pleth': 'SNUADC_PLETH',
    'ecg': 'SNUADC_ECG_II',
    'co2': 'Primus_CO2',
    'awp': 'Primus_AWP'
}

# Continuous Preop Columns (from 04_hpo_xgboost.ipynb)
CONTINUOUS_PREOP_COLS = [
    'age', 'bmi', 'preop_hb', 'preop_plt', 'preop_pt', 'preop_aptt', 'preop_na',
    'preop_k', 'preop_gluc', 'preop_alb', 'preop_ast', 'preop_alt',
    'preop_bun', 'preop_cr', 'preop_hco3'
]

PATIENT_GROUP_COLUMNS = ("subjectid", "subject_id")

def load_data(branch: str) -> pd.DataFrame:
    """
    Loads the dataset based on the branch (windowed or non-windowed).
    """
    if branch == 'non_windowed':
        file_path = FULL_FEATURES_FILE
    elif branch == 'windowed':
        file_path = WINDOWED_FEATURES_FILE
    else:
        raise ValueError(f"Invalid branch: {branch}. Must be 'windowed' or 'non_windowed'.")

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df

def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Returns a dictionary of available feature sets based on columns in the dataframe.
    """
    all_cols = df.columns.tolist()
    feature_sets = {}

    # 1. Pre-op features alone
    # We define preop features as everything that is NOT a waveform feature, NOT an ID, and NOT an outcome.
    # Note: 'split_group' is also excluded.
    
    # Identify waveform columns
    waveform_cols = []
    for prefix in WAVEFORM_PREFIXES.values():
        waveform_cols.extend([c for c in all_cols if c.startswith(prefix)])
        
    # Identify excluded columns (IDs, outcomes, split_group)
    excluded_cols = set(['caseid', 'subjectid', 'subject_id', 'hadm_id', 'split_group'])
    excluded_cols.update(OUTCOMES.values())
    
    # Preop cols = All cols - Waveform cols - Excluded cols
    preop_cols = [c for c in all_cols if c not in waveform_cols and c not in excluded_cols]
    
    feature_sets['preop_only'] = preop_cols

    # 2. Waveform A alone
    for name, prefix in WAVEFORM_PREFIXES.items():
        cols = [c for c in all_cols if c.startswith(prefix)]
        feature_sets[f'{name}_only'] = cols

    # 3. All Waveforms combined
    feature_sets['all_waveforms'] = waveform_cols

    # 4. Pre-op + All Waveforms
    feature_sets['preop_and_all_waveforms'] = preop_cols + waveform_cols

    # 5. Ventilator Only (etco2 and AWP)
    vent_cols = feature_sets['co2_only'] + feature_sets['awp_only']
    feature_sets['ventilator_only'] = vent_cols

    # 6. Monitors only (ECG & SPo2/Pleth)
    monitor_cols = feature_sets['ecg_only'] + feature_sets['pleth_only']
    feature_sets['monitors_only'] = monitor_cols

    # Ablation Models: Pre-op + [Waveform A]
    for name, prefix in WAVEFORM_PREFIXES.items():
        feature_sets[f'preop_and_{name}'] = preop_cols + feature_sets[f'{name}_only']

    # Ablation Models: Pre-op + All Waveforms minus [Waveform A]
    for name, prefix in WAVEFORM_PREFIXES.items():
        cols = [c for c in waveform_cols if not c.startswith(prefix)]
        feature_sets[f'preop_and_all_minus_{name}'] = preop_cols + cols

    return feature_sets


def get_patient_group_column(df: pd.DataFrame) -> Optional[str]:
    """Return the first available patient-group column name, if present."""
    for candidate in PATIENT_GROUP_COLUMNS:
        if candidate in df.columns:
            return candidate
    return None


def get_patient_groups(df: pd.DataFrame, indices: Optional[pd.Index] = None) -> Optional[pd.Series]:
    """Return patient-group labels aligned to ``indices`` when available."""
    group_col = get_patient_group_column(df)
    if group_col is None:
        return None

    groups = df[group_col] if indices is None else df.loc[indices, group_col]
    if groups.isna().any():
        raise ValueError(f"Patient group column '{group_col}' contains missing values.")
    return groups


def select_patient_level_holdout_positions(
    y: pd.Series,
    groups: pd.Series,
    *,
    random_state: int = 42,
    target_test_size: float = 0.2,
    max_splits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return patient-disjoint train/test positional indices for a grouped holdout split."""
    class_counts = y.value_counts()
    if class_counts.empty:
        raise ValueError("Cannot split an empty target vector.")

    unique_groups_per_class = [
        int(groups[y == class_value].nunique()) for class_value in sorted(y.dropna().unique())
    ]
    if not unique_groups_per_class:
        raise ValueError("Unable to derive patient-level class counts.")

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
    target_test_rows = max(1, int(round(len(y) * target_test_size)))

    best_train = None
    best_test = None
    best_gap = None

    dummy = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in splitter.split(dummy, y.to_numpy(), groups.to_numpy()):
        gap = abs(len(test_idx) - target_test_rows)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_train = train_idx
            best_test = test_idx

    if best_train is None or best_test is None:
        raise ValueError("Failed to construct a patient-level holdout split.")

    train_groups = set(groups.iloc[best_train])
    test_groups = set(groups.iloc[best_test])
    overlap = train_groups & test_groups
    if overlap:
        raise ValueError(f"Patient overlap detected across grouped holdout split: {sorted(overlap)[:5]}")

    return best_train, best_test

def prepare_data(
    df: pd.DataFrame,
    outcome_name: str,
    feature_set_name: str,
    random_state: int = 42,
    preserve_nan: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    """
    Prepares data for training: selects features, handles leakage, splits data.
    Returns X_train, X_test, y_train, y_test, scale_pos_weight.

    Args:
        df: Input dataframe containing features and outcomes.
        outcome_name: Key for the outcome to predict.
        feature_set_name: Name of the feature set to use.
        random_state: Seed for the train/test split when applicable.
        preserve_nan: If True, leave NaN values in place for models that can
            handle them; if False, apply legacy imputation (-99 for pre-op,
            0 for waveform features).
    """
    target_col = OUTCOMES.get(outcome_name)
    if not target_col:
        raise ValueError(f"Invalid outcome: {outcome_name}. Available: {list(OUTCOMES.keys())}")

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe.")

    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    
    # Get feature columns
    feature_sets = get_feature_sets(df)
    if feature_set_name not in feature_sets:
        raise ValueError(f"Invalid feature set: {feature_set_name}. Available: {list(feature_sets.keys())}")
    
    selected_features = feature_sets[feature_set_name]
    
    # Select X and y
    X = df[selected_features].copy()
    y = df[target_col].copy()

    # Handle missing values (simple imputation as per original notebook logic)
    # Preop features -> -99, Waveform features -> 0
    if not preserve_nan:
        logger.info("Applying legacy imputation for missing values.")
        for col in X.columns:
            if col in CONTINUOUS_PREOP_COLS:
                X[col] = X[col].fillna(-99)
            else:
                X[col] = X[col].fillna(0)

    # Split data
    # Use 'split_group' if available to respect the original split
    group_col = get_patient_group_column(df)
    if 'split_group' in df.columns:
        logger.info("Using existing 'split_group' column for train/test split.")
        train_mask = df['split_group'] == 'train'
        test_mask = df['split_group'] == 'test'

        if group_col is not None:
            train_groups = set(df.loc[train_mask, group_col])
            test_groups = set(df.loc[test_mask, group_col])
            overlap = train_groups & test_groups
            if overlap:
                raise ValueError(
                    f"Existing split_group leaks patients across train/test via '{group_col}': {sorted(overlap)[:5]}"
                )
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
    else:
        if group_col is not None:
            logger.warning(
                "'split_group' column not found. Performing patient-level grouped holdout split using '%s'.",
                group_col,
            )
            train_idx, test_idx = select_patient_level_holdout_positions(
                y,
                df[group_col],
                random_state=random_state,
            )
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
        else:
            logger.warning("'split_group' column not found. Performing random stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state, stratify=y
            )

    # Calculate scale_pos_weight
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    
    logger.info(f"Data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Positive samples in train: {pos}, Negative: {neg}, Scale Pos Weight: {scale_pos_weight:.4f}")

    return X_train, X_test, y_train, y_test, scale_pos_weight

def compute_quantile_cuts_per_feature(X: np.ndarray, max_bins: int = 256) -> List:
    """
    Returns feature_types list where each entry is a list of float cutpoints
    (InterpretML EBM accepts [List[float]] as explicit continuous cut values).
    
    This allows 'freezing' the bins so they are not re-computed for every HPO trial.
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    # We want at most (max_bins - 2) internal cutpoints:
    # (one bin reserved for missing, plus one for values below first cut, etc.)
    n_cuts = max(0, max_bins - 2)
    
    # Fallback to uniform if only 2 bins (basically binary) or less requested
    if n_cuts == 0:
        return ["uniform"] * n_features

    qs = np.linspace(0, 1, n_cuts + 2, endpoint=True)[1:-1]  # drop 0 and 1

    feature_types = []
    for j in range(n_features):
        col = X[:, j].astype(np.float64, copy=False)
        # Drop NaNs before computing quantiles
        col = col[~np.isnan(col)]
        
        if col.size == 0:
            feature_types.append("continuous")  # nothing to cut on
            continue

        cuts = np.quantile(col, qs)
        cuts = np.unique(cuts)  # remove duplicates from ties
        feature_types.append(cuts.tolist())

    return feature_types
