# =============================================================================
# 9_hpo_xgboost.py   –   compatible with XGBoost ≥ 2.1
#
# Performs Optuna hyper-parameter optimisation for an XGBClassifier using
# callback-style early stopping *passed through the constructor* (required
# since XGBoost 2.1 removed callbacks/early_stopping_rounds from .fit()).
# =============================================================================

# --- Standard Library Imports ------------------------------------------------
import os
import time
import logging
from pathlib import Path

# --- Third-Party Imports -----------------------------------------------------
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

# =============================================================================
# 1. CONFIG & PATHS
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)

N_TRIALS      = 50
RANDOM_STATE  = 42
INPUT_CSV     = "waveform_catch22_features_sliding_window.csv"
TARGET_COLUMN = "death_label"

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
PROCESSED_DIR      = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR        = PROJECT_ROOT / "results" / "xgboost_sliding_window_bootstrap"
HPO_RESULTS_FILE   = RESULTS_DIR / "hpo_best_params.txt"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. OPTUNA OBJECTIVE
# =============================================================================
def objective(trial, X_tr, y_tr, X_val, y_val, spw):
    """Optuna objective: return validation AUC."""
    params = {
        # Core
        "objective": "binary:logistic",
        "booster": "gbtree",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
        # Search space
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "scale_pos_weight": spw,
        "verbosity": 0,
        # --- early stopping now lives **here** (constructor) ---------------
        "early_stopping_rounds": 50,
        "callbacks": [EarlyStopping(rounds=50, save_best=True)],
    }

    model = xgb.XGBClassifier(**params)

    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
    )

    y_hat = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, y_hat)

# =============================================================================
# 3. MAIN
# =============================================================================
def main() -> None:
    logging.info("🚀  Starting XGBoost HPO...")

    # ---------- load data ----------------------------------------------------
    csv_path = PROCESSED_DIR / INPUT_CSV
    try:
        df = pd.read_csv(csv_path).dropna()
    except FileNotFoundError:
        logging.error(f"Data file not found: {csv_path}")
        return

    try:
        X = df.drop(columns=["caseid", TARGET_COLUMN])
        y = df[TARGET_COLUMN]
    except KeyError as e:
        logging.error(f"Missing column in CSV: {e}")
        return

    # ---------- train/val split & scaling ------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)

    neg, pos = np.bincount(y_tr, minlength=2)
    spw = neg / pos if pos else 1.0
    logging.info(f"scale_pos_weight set to {spw:.3f}")

    # ---------- Optuna study --------------------------------------------------
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda t: objective(t, X_tr_s, y_tr, X_val_s, y_val, spw),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = dict(study.best_params)
    best_params["scale_pos_weight"] = spw  # keep for reproducibility
    best_auc = study.best_value

    logging.info(f"🏆 Best val-AUC: {best_auc:.4f}")
    for k, v in best_params.items():
        logging.info(f"  {k}: {v}")

    # ---------- persist best params ------------------------------------------
    with open(HPO_RESULTS_FILE, "w") as f:
        f.write(f"# Saved {time.ctime()}\n")
        f.write("hpo_params_xgboost = {\n")
        for k, v in best_params.items():
            f.write(f"    '{k}': {v},\n")
        f.write("}\n")
    logging.info(f"Parameters written to {HPO_RESULTS_FILE}")

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    main()
