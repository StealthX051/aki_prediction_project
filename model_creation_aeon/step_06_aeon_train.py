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

from data_preparation.inputs import AEON_OUT_DIR, OUTCOME
from model_creation_aeon.classifiers import RocketFused, FreshPrinceFused

def load_aeon_data(outcome_name, channels, include_preop):
    """
    Loads waveform and preop data for the given configuration.
    Returns: X_wave_train, X_preop_train, y_train, X_wave_test, X_preop_test, y_test
    """
    logging.info(f"Loading data from {AEON_OUT_DIR}...")
    
    # 1. Load Waveform Data (NPZ usually)
    # We used 'numpy3d_npz' format in aeon_io.py -> 'X_nonwindowed.npz'
    # Wait, we only made non-windowed 8000 len in step_02_aeon_export.
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
                # Try replacing underscore with slash (common shell safety pattern)
                c_slash = c.replace('_', '/')
                if c_slash in available_channels:
                    target_indices.append(list(available_channels).index(c_slash))
                    logging.info(f"Mapped {c} -> {c_slash}")
                else:
                    raise ValueError(f"Requested channel '{c}' (or '{c_slash}') not found in available: {available_channels}")

        logging.info(f"Using subset channels: {channels}")
    
    X_wave = X_all[:, target_indices, :]

    # 3. Load Labels (y_nonwindowed.csv)
    y_path = Path(AEON_OUT_DIR) / 'y_nonwindowed.csv'
    y_df = pd.read_csv(y_path)
    
    # Align caseids just in case (though npz and csv should match order if generated together)
    # Better to create a dataframe mapping to split_group?
    # We need split_group. Where is it?
    # step_02_aeon_export didn't save split_group in y_nonwindowed.csv, only outcome.
    # However, step_04_aeon_prep.py SAVES split_group in 'aki_preop_aeon.csv'.
    
    # 4. Load Preop (contains split_group) to align splits
    preop_path = Path(AEON_OUT_DIR) / 'aki_preop_aeon.csv'
    if not preop_path.exists():
        raise FileNotFoundError(f"Preop data not found at {preop_path}. Run step_04.")
    
    preop_df = pd.read_csv(preop_path)
    # preop_df has 'caseid', 'split_group', 'outcome', features...
    
    # Merge to align X_wave with preop
    # X_wave is numpy array aligned with 'caseids' variable.
    # Let's create a DataFrame for mapping
    wave_map = pd.DataFrame({'caseid': caseids, 'idx': range(len(caseids))})
    
    # Master DF: Preop (has splits) + Wave Map
    # We treat preop_df as the "cohort with valid preop features"
    full_df = preop_df.merge(wave_map, on='caseid', how='inner')
    
    if len(full_df) < len(preop_df):
        logging.warning("Some cases in preop data missing from waveform export.")
    if len(full_df) < len(caseids):
        logging.warning("Some cases in waveform export missing from preop data.")

    # Sort to ensure alignment if convenient, or just use indices
    X_wave_aligned = X_wave[full_df['idx'].values]
    
    # Preop Features
    if include_preop:
        meta_cols = ['caseid', 'split_group', 'idx'] + list(OUTCOME) if isinstance(OUTCOME, str) else [OUTCOME] # OUTCOME might be str
        # Safely determine non-feature cols.
        # OUTCOME is typically 'aki_label' string.
        # step_04 saved: caseid, outcome, split_group, other outcomes...
        # Look at preop_df columns
        ignore = ['caseid', 'split_group', 'idx', OUTCOME, 'y_severe_aki', 'y_inhosp_mortality', 'y_icu_admit', 'y_prolonged_los_postop']
        feat_cols = [c for c in preop_df.columns if c not in ignore and c in full_df.columns]
        X_preop_aligned = full_df[feat_cols].values
        logging.info(f"Using {len(feat_cols)} preop features.")
    else:
        X_preop_aligned = None
        logging.info("Preop features excluded.")

    y_aligned = full_df[OUTCOME if isinstance(OUTCOME, str) else OUTCOME[0]].values # Logic check
    if isinstance(OUTCOME, dict): # inputs.py defines OUTCOME as string 'aki_label'?
        # inputs.py says: OUTCOME = 'aki_label'
        pass

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=['multirocket', 'minirocket', 'freshprince'])
    parser.add_argument("--channels", type=str, nargs='+', default=['all'], help="List of channels or 'all'")
    parser.add_argument("--include_preop", action='store_true')
    parser.add_argument("--outcome", type=str, default='aki_label')
    parser.add_argument("--n_kernels", type=int, default=10000)
    parser.add_argument("--limit", type=int, help="Limit number of train/test samples for debugging")
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
    
    logging.info(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")
    
    # 3b. Initialize Model
    # Use actual sklearn objects for head estimators to be compatible with Aeon/Sklearn cloning
    from sklearn.linear_model import LogisticRegressionCV
    linear_head = LogisticRegressionCV(
        Cs=10, penalty='l2', solver='lbfgs', max_iter=2000, n_jobs=-1, random_state=42, cv=5
    )
    
    logging.info(f"Initializing {args.model} model...")
    if args.model == 'multirocket':
        clf = RocketFused(
            variant='multi', 
            n_kernels=args.n_kernels, 
            n_jobs=-1, 
            estimator=linear_head
        )
    elif args.model == 'minirocket':
        # MiniRocket is smaller/faster
        clf = RocketFused(
            variant='mini', 
            n_kernels=args.n_kernels, 
            n_jobs=-1, 
            estimator=linear_head
        )
    elif args.model == 'freshprince':
        clf = FreshPrinceFused(n_estimators=200, n_jobs=-1)
        # raise ValueError(f"Unknown model: {args.model}") # Removed buggy line
    else:
        raise ValueError(f"Unknown model: {args.model}")
        
    # 3. Train
    logging.info(f"Starting training on {len(y_tr)} samples...")
    clf.fit(Xw_tr, Xp_tr, y_tr)
    logging.info("Training complete.")
    
    # 4. Predict (Test Set)
    logging.info(f"Predicting on {len(y_te)} test samples...")
    if hasattr(clf, 'predict_proba'):
        # Get probs for positive class
        probs = clf.predict_proba(Xw_te, Xp_te)[:, 1]
    else:
        # Fallback if no predict_proba (shouldn't happen with our wrappers)
        probs = clf.predict(Xw_te, Xp_te)
    logging.info("Prediction complete.")
    
    # 5. Save Predictions (Standard Format for Step 07)
    # Format: caseid, y_true, y_pred_proba, split_group, model_name, feature_set, outcome...
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
    
    # 6. Save Model
    # Joblib pickle
    # Note: Rocket models can be large
    joblib.dump(clf, out_dir / 'model.pkl')
    logging.info("Saved model.")
    
    # 7. Save Config
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

if __name__ == "__main__":
    main()
