import logging
import sys
import ast
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_preparation.inputs import (
    PROCESSED_DIR,
    RESULTS_DIR,
    WIDE_FEATURES_FILE,
    WIDE_FEATURES_WINDOWED_FILE,
    OUTCOME
)

# --- Configuration ---
N_TRIALS = 100
N_SPLITS = 5
RANDOM_STATE = 42
WAVEFORM_PREFIXES = ['SNUADC_PLETH', 'SNUADC_ECG_II', 'Primus_CO2', 'Primus_AWP']

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
optuna.logging.set_verbosity(optuna.logging.WARNING)

def get_data_path(mode):
    if mode == 'full':
        return WIDE_FEATURES_FILE
    elif mode == 'windowed':
        return WIDE_FEATURES_WINDOWED_FILE
    else:
        raise ValueError(f"Unknown dataset mode: {mode}")

def load_and_prep_data(dataset_mode, feature_mode):
    data_path = get_data_path(dataset_mode)
    logging.info(f"Loading data from {data_path}...")
    
    if not data_path.exists():
        logging.error(f"File not found: {data_path}")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    
    # Feature Selection
    all_cols = df.columns.tolist()
    waveform_cols = [c for c in all_cols if any(c.startswith(p) for p in WAVEFORM_PREFIXES)]
    metadata_cols = ['caseid', OUTCOME, 'split_group']
    preop_cols = [c for c in all_cols if c not in waveform_cols and c not in metadata_cols]
    
    selected_features = []
    
    # Logic for feature modes (simplified from notebook)
    # feature_mode options: 'preop', 'waveforms', 'all'
    # You can expand this to match the granular modes if needed.
    
    if feature_mode in ['preop', 'all']:
        selected_features.extend(preop_cols)
        
    if feature_mode in ['waveforms', 'all']:
        selected_features.extend(waveform_cols)
        
    logging.info(f"Selected {len(selected_features)} features ({feature_mode}).")
    
    # Train/Test Split using split_group
    if 'split_group' not in df.columns:
        logging.error("CRITICAL: 'split_group' column missing.")
        sys.exit(1)
        
    train_mask = df['split_group'] == 'train'
    test_mask = df['split_group'] == 'test'
    
    X_train = df.loc[train_mask, selected_features]
    y_train = df.loc[train_mask, OUTCOME]
    X_test = df.loc[test_mask, selected_features]
    y_test = df.loc[test_mask, OUTCOME]
    
    return X_train, y_train, X_test, y_test

def objective(trial, X, y):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'n_jobs': -1,
        'random_state': RANDOM_STATE
    }
    
    # Simple Stratified K-Fold for HPO
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores.mean()

def train_model(dataset_mode='full', feature_mode='all'):
    logging.info(f"--- Starting Training: Dataset={dataset_mode}, Features={feature_mode} ---")
    
    X_train, y_train, X_test, y_test = load_and_prep_data(dataset_mode, feature_mode)
    
    # HPO
    logging.info("Starting HPO...")
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=20) # Reduced trials for speed
    
    best_params = study.best_params
    logging.info(f"Best Params: {best_params}")
    
    # Final Train
    logging.info("Training final model...")
    model = xgb.XGBClassifier(**best_params, n_jobs=-1, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    
    # Evaluate
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    logging.info(f"Test AUROC: {auc:.4f}")
    
    # Save Results (Simplified)
    out_dir = RESULTS_DIR / f"xgboost_{dataset_mode}_{feature_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"AUROC: {auc:.4f}\n")
        f.write(f"Best Params: {best_params}\n")
        
    logging.info(f"Results saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='full', choices=['full', 'windowed'])
    parser.add_argument('--features', type=str, default='all', choices=['preop', 'waveforms', 'all'])
    args = parser.parse_args()
    
    train_model(args.dataset, args.features)
