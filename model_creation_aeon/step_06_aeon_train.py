import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from typing import Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import joblib
import numpy as np
import pandas as pd

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Imports for HPO
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from aeon.transformations.collection.convolution_based import MiniRocket, MultiRocket

from data_preparation.inputs import AEON_OUT_DIR, OUTCOME
from model_creation_aeon.classifiers import FreshPrinceFused # Keep FreshPrince as is
from model_creation.postprocessing import (
    apply_logistic_recalibration,
    fit_logistic_recalibration,
    find_youden_j_threshold,
)

PIPELINE_NAME = "step_06_aeon_train"

# Outcome Mapping (Friendly Name -> Column Name)
OUTCOME_MAPPING = {
    'any_aki': 'aki_label',
    'severe_aki': 'y_severe_aki',
    'mortality': 'y_inhosp_mortality',
    'extended_los': 'y_prolonged_los_postop',
    'icu_admission': 'y_icu_admit',
    # Fallback for direct column usage or if already mapped
    'aki_label': 'aki_label',
    'y_severe_aki': 'y_severe_aki',
    'y_inhosp_mortality': 'y_inhosp_mortality',
    'y_prolonged_los_postop': 'y_prolonged_los_postop',
    'y_icu_admit': 'y_icu_admit'
}

def compute_file_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file without loading it fully into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_aeon_data(outcome_name, channels, include_preop):
    """
    Loads waveform and preop data for the given configuration.
    Returns: X_wave_train, X_preop_train, y_train, X_wave_test, X_preop_test, y_test,
    caseids_train, caseids_test, train_df, test_df
    """
    logging.info(f"Loading data from {AEON_OUT_DIR}...")
    
    # 1. Load Waveform Data (NPZ usually)
    npz_path = Path(AEON_OUT_DIR) / 'X_nonwindowed.npz'
    if not npz_path.exists():
        raise FileNotFoundError(f"Waveform data not found at {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    X_all = data['X'] # (N, C, L)
    caseids = data['caseids']
    available_channels = data['channels']
    
    # 2. Select Channels
    if channels == ['all']:
        target_indices = range(len(available_channels))
        logging.info(f"Using ALL channels: {available_channels}")
    else:
        target_indices = []
        for c in channels:
            if c in available_channels:
                target_indices.append(list(available_channels).index(c))
            else:
                # Try various slash replacements for mapping
                c_slash_first = c.replace('_', '/', 1) # SNUADC_ECG_II -> SNUADC/ECG_II
                c_slash_all = c.replace('_', '/')      # SNUADC_PLETH -> SNUADC/PLETH

                if c_slash_first in available_channels:
                    target_indices.append(list(available_channels).index(c_slash_first))
                    logging.info(f"Mapped {c} -> {c_slash_first}")
                elif c_slash_all in available_channels:
                    target_indices.append(list(available_channels).index(c_slash_all))
                    logging.info(f"Mapped {c} -> {c_slash_all}")
                else:
                    raise ValueError(f"Requested channel '{c}' not found in available: {available_channels}")

        logging.info(f"Using subset channels: {channels}")
    
    X_wave = X_all[:, target_indices, :]

    # 3. Load Labels (y_nonwindowed.csv) is typically redundant if we merge with preop
    # logic handled below.
    
    # 4. Load Preop (contains split_group) to align splits
    preop_path = Path(AEON_OUT_DIR) / 'aki_preop_aeon.csv'
    if not preop_path.exists():
        raise FileNotFoundError(f"Preop data not found at {preop_path}. Run step_04.")
    
    preop_df = pd.read_csv(preop_path)
    
    # Merge to align X_wave with preop
    # X_wave is numpy array aligned with 'caseids' variable.
    wave_map = pd.DataFrame({'caseid': caseids, 'idx': range(len(caseids))})
    
    # Master DF: Preop (has splits) + Wave Map
    full_df = preop_df.merge(wave_map, on='caseid', how='inner')
    
    # Sort to ensure alignment
    X_wave_aligned = X_wave[full_df['idx'].values]
    
    # Preop Features
    if include_preop:
        # Determine actual column names to exclude
        mapped_outcome = OUTCOME_MAPPING.get(outcome_name, outcome_name)
        
        # Safe exclusion list
        ignore = ['caseid', 'split_group', 'idx', mapped_outcome] + list(OUTCOME_MAPPING.values())
        feat_cols = [c for c in preop_df.columns if c not in ignore and c in full_df.columns]
        X_preop_aligned = full_df[feat_cols].values
        logging.info(f"Using {len(feat_cols)} preop features.")
    else:
        X_preop_aligned = None
        logging.info("Preop features excluded.")

    # Resolve Outcome Column
    target_col = OUTCOME_MAPPING.get(outcome_name, outcome_name)
    if target_col not in full_df.columns:
        raise KeyError(f"Target column '{target_col}' not found in data (mapped from '{outcome_name}').")

    y_aligned = full_df[target_col].values

    # 5. Split
    train_mask = full_df['split_group'] == 'train'
    test_mask = full_df['split_group'] == 'test'
    
    X_wave_train = X_wave_aligned[train_mask]
    X_wave_test = X_wave_aligned[test_mask]

    if include_preop:
        X_preop_train = X_preop_aligned[train_mask]
        X_preop_test = X_preop_aligned[test_mask]
    else:
        X_preop_train = None
        X_preop_test = None

    y_train = y_aligned[train_mask]
    y_test = y_aligned[test_mask]

    caseids_train = full_df.loc[train_mask, 'caseid'].values
    caseids_test = full_df.loc[test_mask, 'caseid'].values

    return (
        X_wave_train,
        X_preop_train,
        y_train,
        X_wave_test,
        X_preop_test,
        y_test,
        caseids_train,
        caseids_test,
        full_df.loc[train_mask],
        full_df.loc[test_mask],
    )

def optimize_aeon_model(X_wave_tr, X_preop_tr, y_tr, model_type, n_trials=100, n_jobs=-1):
    """
    Runs Optuna HPO for generic Aeon models (MiniRocket/MultiRocket).
    1. Precompute Rocket features.
    2. Optimize Linear Head (LogisticRegression).
    """
    
    logging.info(f"Starting HPO for {model_type}...")
    
    # 1. Transform Waveforms Once
    if model_type == 'minirocket':
        transformer = MiniRocket(n_kernels=10000, n_jobs=n_jobs, random_state=42)
    elif model_type == 'multirocket':
        transformer = MultiRocket(n_kernels=10000, n_jobs=n_jobs, random_state=42)
    else:
        raise ValueError(f"HPO not implemented for {model_type}")
        
    logging.info(f"Transforming training data with {model_type}...")
    X_rocket_tr = transformer.fit_transform(X_wave_tr)
    logging.info(f"Transformer output shape: {X_rocket_tr.shape}")
    
    # 2. Define Objective
    def objective(trial):
        # Hyperparameters
        C = trial.suggest_float("C", 1e-3, 10.0, log=True)
        # class_weight = trial.suggest_categorical("class_weight", [None, "balanced"]) 
        # User requested to fix class_weight to 'balanced'
        class_weight = 'balanced'
        
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auprcs = []
        
        for train_idx, val_idx in kf.split(X_rocket_tr, y_tr):
            # Split
            X_r_train, X_r_val = X_rocket_tr[train_idx], X_rocket_tr[val_idx]
            y_train_fold, y_val_fold = y_tr[train_idx], y_tr[val_idx]
            
            # Fuse Preop (if exists)
            if X_preop_tr is not None:
                X_p_train, X_p_val = X_preop_tr[train_idx], X_preop_tr[val_idx]
                X_train_fold = np.hstack([X_r_train, X_p_train])
                X_val_fold = np.hstack([X_r_val, X_p_val])
            else:
                X_train_fold = X_r_train
                X_val_fold = X_r_val
            
            # Scale (StandardScaler fitted on fold training data)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # Train Logistic Regression
            try:
                clf = LogisticRegression(
                    C=C, 
                    class_weight=class_weight,
                    solver='lbfgs', 
                    max_iter=5000,
                    n_jobs=1 # Use 1 core since trials are parallel
                )
                clf.fit(X_train_scaled, y_train_fold)
                
                # Predict
                probs = clf.predict_proba(X_val_scaled)[:, 1]
                ap = average_precision_score(y_val_fold, probs)
                auprcs.append(ap)
            except ValueError as e:
                # This catches cases where a fold has only 1 class
                logging.warning(f"Fold failed: {e}")
                auprcs.append(0.0) # Penalty for failure
            
        return np.mean(auprcs)

    # 3. Optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs) # trials parallel
    
    logging.info(f"Best Trial: {study.best_trial.value} | Params: {study.best_params}")
    return study.best_params, transformer, X_rocket_tr


def _stack_features(rocket_features: np.ndarray, preop: np.ndarray) -> np.ndarray:
    preop_np = np.array(preop, dtype=np.float32)
    return np.hstack([rocket_features, preop_np])


def generate_oof_predictions(
    model_name: str,
    best_params: dict,
    X_wave: np.ndarray,
    X_preop: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate out-of-fold predictions for Aeon models without test leakage."""

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.empty_like(y, dtype=float)
    fold_indices = np.empty_like(y, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_wave, y)):
        if model_name == 'freshprince':
            fold_model = FreshPrinceFused(n_estimators=200, random_state=random_state, n_jobs=-1)
            fold_model.fit(X_wave[train_idx], None if X_preop is None else X_preop[train_idx], y[train_idx])
            probs = fold_model.predict_proba(
                X_wave[val_idx], None if X_preop is None else X_preop[val_idx]
            )[:, 1]
        else:
            if model_name == 'minirocket':
                transformer = MiniRocket(n_kernels=10000, n_jobs=-1, random_state=random_state)
            elif model_name == 'multirocket':
                transformer = MultiRocket(n_kernels=10000, n_jobs=-1, random_state=random_state)
            else:
                raise ValueError(f"Unsupported model for OOF generation: {model_name}")

            X_wave_train = transformer.fit_transform(X_wave[train_idx])
            X_wave_val = transformer.transform(X_wave[val_idx])

            if X_preop is not None:
                X_train_combined = _stack_features(X_wave_train, X_preop[train_idx])
                X_val_combined = _stack_features(X_wave_val, X_preop[val_idx])
            else:
                X_train_combined = X_wave_train
                X_val_combined = X_wave_val

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_combined)
            X_val_scaled = scaler.transform(X_val_combined)

            clf = LogisticRegression(
                C=best_params.get('C', 1.0),
                class_weight='balanced',
                solver='lbfgs',
                max_iter=5000,
                n_jobs=-1,
            )
            clf.fit(X_train_scaled, y[train_idx])
            probs = clf.predict_proba(X_val_scaled)[:, 1]

        oof_predictions[val_idx] = probs
        fold_indices[val_idx] = fold

    if np.isnan(oof_predictions).any():
        raise ValueError("OOF predictions contain NaN values.")

    return oof_predictions, fold_indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['multirocket', 'minirocket', 'freshprince'])
    parser.add_argument("--channels", type=str, nargs='+', default=['all'], help="List of channels or 'all'")
    parser.add_argument("--include_preop", action='store_true')
    parser.add_argument("--outcome", type=str, default='any_aki')
    parser.add_argument("--limit", type=int, help="Limit number of train/test samples for debugging")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of HPO trials")
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / 'results' / 'aeon'), help="Base results directory")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds for OOF predictions")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for CV")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Name Experiment
    chan_str = "all" if args.channels == ['all'] else "_".join(args.channels)
    preop_str = "fused" if args.include_preop else "waveonly"
    exp_name = f"{args.model}_{chan_str}_{preop_str}_{args.outcome}"

    out_dir = Path(args.results_dir) / 'models' / args.model / exp_name
    predictions_dir = out_dir / 'predictions'
    artifacts_dir = out_dir / 'artifacts'
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting Experiment: {exp_name}")
    
    # 1. Load Data
    data = load_aeon_data(args.outcome, args.channels, args.include_preop)
    Xw_tr, Xp_tr, y_tr, Xw_te, Xp_te, y_te, caseids_tr, caseids_te, df_tr, df_te = data

    if args.limit:
        logging.info(f"Limiting usage to {args.limit} samples per split.")
        Xw_tr, Xp_tr = Xw_tr[:args.limit], (Xp_tr[:args.limit] if Xp_tr is not None else None)
        y_tr = y_tr[:args.limit]
        caseids_tr = caseids_tr[:args.limit]
        df_tr = df_tr.iloc[:args.limit]

        Xw_te, Xp_te = Xw_te[:args.limit], (Xp_te[:args.limit] if Xp_te is not None else None)
        y_te = y_te[:args.limit]
        caseids_te = caseids_te[:args.limit]
        df_te = df_te.iloc[:args.limit]

    logging.info(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")

    positive_count = int((y_tr == 1).sum())
    negative_count = int((y_tr == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both classes for calibration and evaluation.")

    n_splits = args.n_splits
    min_class = min(positive_count, negative_count)
    if n_splits > min_class:
        logging.warning("Reducing n_splits from %s to %s to match minority class count.", n_splits, min_class)
        n_splits = min_class

    dataset_hash = compute_file_hash(Path(AEON_OUT_DIR) / 'aki_preop_aeon.csv')

    # 2. HPO & Training
    model = None
    best_params = {}

    if args.model == 'freshprince':
        # FreshPrince HPO is too expensive; use default/existing wrapper
        logging.info("Using FreshPrinceFused (No HPO Loop)...")
        model = FreshPrinceFused(n_estimators=200, random_state=args.random_state, n_jobs=-1)
        model.fit(Xw_tr, Xp_tr, y_tr)

        final_prob_raw = model.predict_proba(Xw_te, Xp_te)[:, 1]
        
    else:
        # MiniRocket / MultiRocket with Optuna HPO
        best_params, transformer, X_rocket_tr = optimize_aeon_model(
            Xw_tr, Xp_tr, y_tr, 
            model_type=args.model, 
            n_trials=args.n_trials
        )
        
        # Refit Final Model on Full Train
        logging.info("Refitting final model with best params on full training set...")
        
        # 1. Transform Train (Already done, using returned X_rocket_tr)
        # X_rocket_tr = transformer.transform(Xw_tr) # Removed redundant transform
        
        # 2. Fuse
        if Xp_tr is not None:
            X_combined_tr = np.hstack([X_rocket_tr, np.array(Xp_tr, dtype=np.float32)])
        else:
            X_combined_tr = X_rocket_tr
            
        # 3. Scale
        scaler = StandardScaler()
        X_scaled_tr = scaler.fit_transform(X_combined_tr)
        
        # 4. Train Classifier
        final_clf = LogisticRegression(
            C=best_params['C'],
            class_weight='balanced', # Fixed to balanced
            solver='lbfgs',
            max_iter=5000,
            n_jobs=-1
        )
        final_clf.fit(X_scaled_tr, y_tr)
        
        # Predictions on Test
        logging.info("Predicting on Test Set...")
        X_rocket_te = transformer.transform(Xw_te)
        
        if Xp_te is not None:
             X_combined_te = np.hstack([X_rocket_te, np.array(Xp_te, dtype=np.float32)])
        else:
             X_combined_te = X_rocket_te

        X_scaled_te = scaler.transform(X_combined_te)
        final_prob_raw = final_clf.predict_proba(X_scaled_te)[:, 1]

        # Save Model Components manually since we aren't using the FusedClassifier wrapper
        # We can construct a pipeline or save simple dict
        joblib.dump(transformer, out_dir / 'transformer.pkl')
        joblib.dump(scaler, out_dir / 'scaler.pkl')
        joblib.dump(final_clf, out_dir / 'classifier.pkl')
        model = "Optuna_Pipeline" # Sentinel

    logging.info("Generating out-of-fold predictions for calibration...")
    oof_predictions, fold_indices = generate_oof_predictions(
        args.model,
        best_params,
        Xw_tr,
        Xp_tr,
        y_tr,
        n_splits=n_splits,
        random_state=args.random_state,
    )

    if len(oof_predictions) != len(y_tr):
        raise ValueError("OOF predictions length mismatch with training data.")

    recalibration_model = fit_logistic_recalibration(y_tr, oof_predictions)
    calibrated_oof = apply_logistic_recalibration(oof_predictions, recalibration_model)

    threshold, youden_j, sensitivity, specificity = find_youden_j_threshold(y_tr, calibrated_oof)

    if set(caseids_tr) & set(caseids_te):
        raise AssertionError("Train and test caseids overlap; calibration would leak test data.")

    calibrated_test = apply_logistic_recalibration(final_prob_raw, recalibration_model)

    train_predictions = pd.DataFrame({
        'caseid': caseids_tr,
        'y_true': y_tr,
        'y_prob_raw': oof_predictions,
        'y_prob_calibrated': calibrated_oof,
        'threshold': threshold,
        'y_pred_label': (calibrated_oof >= threshold).astype(int),
        'fold': fold_indices,
        'is_oof': True,
        'outcome': args.outcome,
        'branch': 'aeon',
        'feature_set': chan_str + ("_fused" if args.include_preop else "_waveonly"),
        'model_name': args.model,
        'pipeline': PIPELINE_NAME,
    })

    test_predictions = pd.DataFrame({
        'caseid': caseids_te,
        'y_true': y_te,
        'y_prob_raw': final_prob_raw,
        'y_prob_calibrated': calibrated_test,
        'threshold': threshold,
        'y_pred_label': (calibrated_test >= threshold).astype(int),
        'fold': -1,
        'is_oof': False,
        'outcome': args.outcome,
        'branch': 'aeon',
        'feature_set': chan_str + ("_fused" if args.include_preop else "_waveonly"),
        'model_name': args.model,
        'pipeline': PIPELINE_NAME,
    })

    train_predictions.to_csv(predictions_dir / 'train_oof.csv', index=False)
    test_predictions.to_csv(predictions_dir / 'test.csv', index=False)
    logging.info(f"Saved train OOF predictions to {predictions_dir / 'train_oof.csv'}")
    logging.info(f"Saved test predictions to {predictions_dir / 'test.csv'}")

    calibration_payload = {
        'intercept': recalibration_model.intercept,
        'slope': recalibration_model.slope,
        'eps': recalibration_model.eps,
    }
    threshold_payload = {
        'threshold': threshold,
        'youden_j': youden_j,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    metadata_payload = {
        'seed': args.random_state,
        'folds': n_splits,
        'dataset_hash': dataset_hash,
        'counts': {
            'train': {
                'total': int(len(y_tr)),
                'positive': positive_count,
                'negative': negative_count,
            },
            'test': {
                'total': int(len(y_te)),
                'positive': int((y_te == 1).sum()),
                'negative': int((y_te == 0).sum()),
            },
        },
    }

    with open(artifacts_dir / 'calibration.json', 'w') as f:
        json.dump(calibration_payload, f, indent=2)
    with open(artifacts_dir / 'threshold.json', 'w') as f:
        json.dump(threshold_payload, f, indent=2)
    with open(artifacts_dir / 'metadata.json', 'w') as f:
        json.dump(metadata_payload, f, indent=2)

    logging.info(f"Saved calibration and threshold artifacts to {artifacts_dir}")

    # Save Config/Best Params
    config = vars(args)
    config.update(best_params)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)

    logging.info("Experiment Complete.")

if __name__ == "__main__":
    main()
