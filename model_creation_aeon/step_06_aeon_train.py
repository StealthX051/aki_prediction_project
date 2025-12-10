import argparse
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

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

def load_aeon_data(outcome_name, channels, include_preop):
    """
    Loads waveform and preop data for the given configuration.
    Returns: X_wave_train, X_preop_train, y_train, X_wave_test, X_preop_test, y_test
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
    
    caseids_test = full_df.loc[test_mask, 'caseid'].values

    return (X_wave_train, X_preop_train, y_train, 
            X_wave_test, X_preop_test, y_test, caseids_test, full_df.loc[test_mask])

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
                    max_iter=1000,
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['multirocket', 'minirocket', 'freshprince'])
    parser.add_argument("--channels", type=str, nargs='+', default=['all'], help="List of channels or 'all'")
    parser.add_argument("--include_preop", action='store_true')
    parser.add_argument("--outcome", type=str, default='any_aki')
    parser.add_argument("--limit", type=int, help="Limit number of train/test samples for debugging")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of HPO trials")
    parser.add_argument("--results_dir", type=str, default=str(PROJECT_ROOT / 'results' / 'aeon'), help="Base results directory")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Name Experiment
    chan_str = "all" if args.channels == ['all'] else "_".join(args.channels)
    preop_str = "fused" if args.include_preop else "waveonly"
    exp_name = f"{args.model}_{chan_str}_{preop_str}_{args.outcome}"
    
    out_dir = Path(args.results_dir) / 'models' / args.model / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting Experiment: {exp_name}")
    
    # 1. Load Data
    data = load_aeon_data(args.outcome, args.channels, args.include_preop)
    Xw_tr, Xp_tr, y_tr, Xw_te, Xp_te, y_te, caseids_te, df_te = data
    
    if args.limit:
        logging.info(f"Limiting usage to {args.limit} samples per split.")
        Xw_tr, Xp_tr = Xw_tr[:args.limit], (Xp_tr[:args.limit] if Xp_tr is not None else None)
        y_tr = y_tr[:args.limit]
        
        Xw_te, Xp_te = Xw_te[:args.limit], (Xp_te[:args.limit] if Xp_te is not None else None)
        y_te = y_te[:args.limit]
        caseids_te = caseids_te[:args.limit]
    
    logging.info(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")
    
    # 2. HPO & Training
    model = None
    best_params = {}
    
    if args.model == 'freshprince':
        # FreshPrince HPO is too expensive; use default/existing wrapper
        logging.info("Using FreshPrinceFused (No HPO Loop)...")
        model = FreshPrinceFused(n_estimators=200, n_jobs=-1)
        model.fit(Xw_tr, Xp_tr, y_tr)
        
        # Predict
        probs = model.predict_proba(Xw_te, Xp_te)[:, 1]
        
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
            max_iter=1000,
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
        probs = final_clf.predict_proba(X_scaled_te)[:, 1]
        
        # Save Model Components manually since we aren't using the FusedClassifier wrapper
        # We can construct a pipeline or save simple dict
        joblib.dump(transformer, out_dir / 'transformer.pkl')
        joblib.dump(scaler, out_dir / 'scaler.pkl')
        joblib.dump(final_clf, out_dir / 'classifier.pkl')
        model = "Optuna_Pipeline" # Sentinel

    # 3. Save Predictions
    logging.info(f"Saving artifacts to {out_dir}...")
    pred_df = pd.DataFrame({
        'caseid': caseids_te,
        'y_true': y_te,
        'y_pred_proba': probs,
        'split_group': 'test',
        'model_name': args.model,
        'feature_set': chan_str + ("_fused" if args.include_preop else "_waveonly"),
        'outcome': args.outcome,
        'branch': 'aeon' 
    })
    
    pred_path = out_dir / 'predictions.csv'
    pred_df.to_csv(pred_path, index=False)
    logging.info(f"Saved predictions to {pred_path}")
    
    # Save Config/Best Params
    config = vars(args)
    config.update(best_params)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
        
    logging.info("Experiment Complete.")

if __name__ == "__main__":
    main()
