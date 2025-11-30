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

def objective(trial, X, y, scale_pos_weight):
    """
    Optuna objective function for XGBoost HPO.
    Optimizes for AUPRC using Stratified K-Fold CV.
    """
    params = {
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "aucpr", # Optimize for AUPRC
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auprc_scores = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params, callbacks=[EarlyStopping(rounds=50)])
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        y_pred_proba = model.predict_proba(X_val)[:, 1]
        score = average_precision_score(y_val, y_pred_proba)
        auprc_scores.append(score)

    return np.mean(auprc_scores)

def run_hpo(outcome, branch, feature_set, n_trials=50, smoke_test=False):
    logger.info(f"Starting HPO for Outcome: {outcome}, Branch: {branch}, Feature Set: {feature_set}")
    
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
        logger.error(f"Failed to prepare data: {e}")
        return

    # Run Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=True
    )

    logger.info(f"HPO Complete. Best AUPRC: {study.best_value:.4f}")
    logger.info(f"Best Params: {study.best_params}")

    # Save best params
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight # Save this too as it's computed
    
    output_dir = utils.RESULTS_DIR / 'params' / outcome / branch
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

    args = parser.parse_args()

    run_hpo(args.outcome, args.branch, args.feature_set, args.n_trials, args.smoke_test)
