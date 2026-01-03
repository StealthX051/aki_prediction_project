import argparse
import json
import logging
from pathlib import Path
import sys

import optuna
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from xgboost.callback import EarlyStopping

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgboost(trial, X, y, scale_pos_weight, sample_weights=None):
    """
    Optuna objective function for XGBoost HPO.
    Optimizes for AUPRC using Stratified K-Fold CV.
    """
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "aucpr",  # Optimize for AUPRC
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
        "verbosity": 0,
        "scale_pos_weight": scale_pos_weight,
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
    }

    # Dynamic n_splits for small data
    min_class =  min(y.value_counts())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return 0.5

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auprc_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        train_sample_weights = None if sample_weights is None else sample_weights[train_idx]

        model = xgb.XGBClassifier(**params, callbacks=[EarlyStopping(rounds=50)])
        model.fit(X_tr, y_tr, sample_weight=train_sample_weights, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_pred_proba)
        auprc_scores.append(score)

    return np.mean(auprc_scores)


def objective_ebm(trial, X, y, sample_weights, feature_types=None):
    """
    Optuna objective function for EBM HPO.
    Optimizes for AUPRC using Stratified K-Fold CV.
    """
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as e:
        raise ImportError("interpret library is required for EBM model_type") from e

    params = {
        "interactions": 0,
        "missing": "gain",
        "inner_bags": 0,  # lightweight bagging for faster HPO
        "outer_bags": 1,
        "max_bins": trial.suggest_categorical("max_bins", [32, 64, 128, 256]),
        "random_state": 42,
        "n_jobs": -2,
        "max_leaves": trial.suggest_categorical("max_leaves", [2, 3]),
        "smoothing_rounds": trial.suggest_categorical(
            "smoothing_rounds", [0, 25, 50, 75, 100, 150, 200, 350, 500]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
        ),
        "validation_size": trial.suggest_categorical(
            "validation_size", [0.1, 0.15, 0.2]
        ),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [100, 200]),
        "early_stopping_tolerance": trial.suggest_categorical(
            "early_stopping_tolerance", [0.0, 1e-5]
        ),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 3, 4, 5, 10]),
        "min_hessian": trial.suggest_categorical("min_hessian", [0.0, 1e-6, 1e-4, 1e-2]),
        "greedy_ratio": trial.suggest_categorical("greedy_ratio", [0.0, 5.0, 10.0]),
        "cyclic_progress": trial.suggest_categorical("cyclic_progress", [0.0, 1.0]),
    }

    # Dynamic n_splits for small data
    min_class =  min(y.value_counts())
    n_splits = min(5, min_class)
    
    # Preventing hangs on very small datasets
    if len(y) < 50:
        # logger.info(f"Dataset size {len(y)} < 50; forcing n_jobs=1 to prevent hangs.")
        params['n_jobs'] = 1
        params['inner_bags'] = 0
        params['outer_bags'] = 1
        params['validation_size'] = 0.2
        
    if n_splits < 2:
        return 0.5 # Return dummy score if cross-validation is impossible
        
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    auprc_scores = []

    # If reusing frozen feature_types, we must ensure X is contiguous numpy array 
    # and that we don't accidentally pass DataFrames if the cuts were computed on them differently.
    # However, utils.compute_quantile_cuts_per_feature handles numpy/pandas by converting to array.
    # Here we assume X is compatible.
    
    # IMPORTANT: When using frozen bins, we should set feature_types in the params.
    if feature_types is not None:
        params['feature_types'] = feature_types

    for train_idx, val_idx in cv.split(X, y):
        # We need to be careful: splitting the dataframe is fine, but if we pass feature_types, 
        # EBM expects the input X to match the columns of feature_types.
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        sample_weight_tr = sample_weights[train_idx]
        
        # Optimization: convert to contiguous float32 array to match frozen bins assumptions
        # and avoid internal overhead
        X_tr_np = np.ascontiguousarray(X_tr.values, dtype=np.float32)
        X_val_np = np.ascontiguousarray(X_val.values, dtype=np.float32)

        model = ExplainableBoostingClassifier(**params)
        model.fit(X_tr_np, y_tr, sample_weight=sample_weight_tr)

        y_pred_proba = model.predict_proba(X_val_np)[:, 1]
        score = average_precision_score(y_val, y_pred_proba)
        auprc_scores.append(score)

    return np.mean(auprc_scores)

def run_hpo(outcome, branch, feature_set, n_trials=100, smoke_test=False, model_type="xgboost"):
    logger.info(
        f"Starting HPO for Outcome: {outcome}, Branch: {branch}, Feature Set: {feature_set}, Model: {model_type}"
    )
    
    if smoke_test:
        logger.info("SMOKE TEST MODE: Running with reduced trials.")
        n_trials = 2

    # Load and prepare data
    try:
        df = utils.load_data(branch)
        X_train, _, y_train, _, scale_pos_weight = utils.prepare_data(
            df, outcome, feature_set
        )
    except Exception as e:
        logger.exception(f"Failed to prepare data: {e}")
        return

    sample_weights = np.where(y_train == 1, scale_pos_weight, 1.0)

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    if model_type == "xgboost":
        objective_fn = lambda trial: objective_xgboost(trial, X_train, y_train, scale_pos_weight, sample_weights)
    elif model_type == "ebm":
        logger.info("Pre-computing quantile cuts to freeze bins for EBM HPO...")
        # Convert to contiguous array for binning
        X_train_np = np.ascontiguousarray(X_train.values, dtype=np.float32)
        feature_types_frozen = utils.compute_quantile_cuts_per_feature(X_train_np, max_bins=1024)
        logger.info("Bins computed. Starting HPO with frozen feature_types.")
        
        objective_fn = lambda trial: objective_ebm(trial, X_train, y_train, sample_weights, feature_types=feature_types_frozen)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    study.optimize(
        objective_fn,
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info(f"HPO Complete. Best AUPRC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    # Save best params
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight # Save this too as it's computed
    
    output_dir = utils.RESULTS_DIR / 'params' / model_type / outcome / branch
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{feature_set}.json"

    with open(output_file, 'w') as f:
        json.dump(best_params, f, indent=4)
    
    logger.info(f"Best parameters saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HPO for AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument("--branch", type=str, required=True, choices=['windowed', 'non_windowed'], help="Data branch")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--n_trials", type=int, default=100, help="Number of Optuna trials")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "ebm"],
        default="xgboost",
        help="Model type to optimize",
    )

    args = parser.parse_args()

    run_hpo(args.outcome, args.branch, args.feature_set, args.n_trials, args.smoke_test, args.model_type)
