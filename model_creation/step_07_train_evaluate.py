import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils
from model_creation.postprocessing import (
    apply_logistic_recalibration,
    fit_logistic_recalibration,
    find_youden_j_threshold,
    generate_stratified_oof_predictions,
    write_json,
)
from model_creation.prediction_io import write_prediction_files

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PIPELINE_NAME = "step_07_train_evaluate"


def compute_file_hash(path: Path) -> str:
    """Compute the SHA256 hash of a file without loading it into memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_shap_plots(model, X_test, output_dir):
    """
    Calculates and saves SHAP summary plots.
    """
    logger.info("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Bar plot
    plt.figure(figsize=(10, max(10, len(X_test.columns) * 0.5)))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=20)
    plt.title("SHAP Feature Importance (Bar)")
    plt.savefig(output_dir / "shap_summary_bar.png", bbox_inches='tight')
    plt.close()

    # Dot plot
    plt.figure(figsize=(10, max(10, len(X_test.columns) * 0.5)))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False, max_display=20)
    plt.title("SHAP Summary Plot (Dot)")
    plt.savefig(output_dir / "shap_summary_dot.png", bbox_inches='tight')
    plt.close()
    
    logger.info("SHAP plots saved.")


def train_evaluate(outcome, branch, feature_set, smoke_test=False, model_type="xgboost", legacy_imputation=False):
    logger.info(
        "Starting Training/Evaluation for Outcome: %s, Branch: %s, Feature Set: %s, Model: %s",
        outcome,
        branch,
        feature_set,
        model_type,
    )

    # Load params
    params_file = utils.RESULTS_DIR / 'params' / model_type / outcome / branch / f"{feature_set}.json"
    if not params_file.exists():
        logger.error(f"Parameters file not found: {params_file}. Run HPO first.")
        return

    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Load and prepare data
    try:
        random_state = params.get('random_state', 42)
        params['random_state'] = random_state
        df = utils.load_data(branch)
        data_file = utils.FULL_FEATURES_FILE if branch == 'non_windowed' else utils.WINDOWED_FEATURES_FILE
        dataset_hash = compute_file_hash(data_file)
        preserve_nan = not legacy_imputation
        X_train, X_test, y_train, y_test, scale_pos_weight = utils.prepare_data(
            df, outcome, feature_set, random_state=random_state, preserve_nan=preserve_nan
        )
    except Exception as e:
        logger.error(f"Failed to prepare data: {e}")
        return

    if smoke_test:
        logger.info("SMOKE TEST: Reducing data size.")
        X_train = X_train.head(100)
        y_train = y_train.head(100)
        X_test = X_test.head(50)
        y_test = y_test.head(50)
        if model_type == "xgboost":
            params['n_estimators'] = 10

    # Set reproducibility controls
    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training data must contain both classes for calibration and evaluation.")

    n_splits = params.get('n_splits', 5)
    min_class = min(positive_count, negative_count)
    if n_splits > min_class:
        logger.warning(
            "Reducing n_splits from %s to %s to match minority class count.",
            n_splits,
            min_class,
        )
        n_splits = min_class

    if 'scale_pos_weight' not in params:
        params['scale_pos_weight'] = scale_pos_weight

    model_label = "XGBoost" if model_type == "xgboost" else "EBM"
    sample_weight = None

    if model_type == "xgboost":
        model_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": -1,
            **params,
        }
        oof_fit_params = None
    elif model_type == "ebm":
        try:
            from interpret.glassbox import ExplainableBoostingClassifier
        except ImportError as exc:
            logger.error("interpret library is required for EBM training: %s", exc)
            return

        ebm_defaults = {
            "interactions": 0,
            "missing": "gain",
            "inner_bags": 20,
            "outer_bags": 14,
            "max_bins": 1024,
            "random_state": random_state,
            "n_jobs": -2,
        }

        ebm_params = {**ebm_defaults, **params}
        ebm_params.pop("scale_pos_weight", None)
        model_params = ebm_params
        pos_weight = params.get("scale_pos_weight", scale_pos_weight)
        sample_weight = np.where(y_train.values == 1, pos_weight, 1.0)
        oof_fit_params = {"sample_weight": sample_weight}
    else:
        logger.error("Unsupported model_type: %s", model_type)
        return

    logger.info("Generating out-of-fold predictions for calibration...")
    if model_type == "xgboost":
        oof_model = xgb.XGBClassifier(**model_params)
    else:
        oof_model = ExplainableBoostingClassifier(**model_params)
    oof_predictions, fold_indices = generate_stratified_oof_predictions(
        oof_model,
        X_train.values,
        y_train.values,
        n_splits=n_splits,
        random_state=random_state,
        sample_weight=oof_fit_params.get("sample_weight") if oof_fit_params else None,
    )

    if len(oof_predictions) != len(X_train):
        raise ValueError("OOF predictions length mismatch with training data.")

    recalibration_model = fit_logistic_recalibration(y_train.values, oof_predictions)
    calibrated_oof = apply_logistic_recalibration(oof_predictions, recalibration_model)

    threshold, youden_j, sensitivity, specificity = find_youden_j_threshold(
        y_train.values, calibrated_oof
    )

    if set(X_train.index) & set(X_test.index):
        raise AssertionError("Train and test indices overlap; calibration would leak test data.")

    logger.info("Training final model on full training data...")
    if model_type == "xgboost":
        final_model = xgb.XGBClassifier(**model_params)
        final_model.fit(X_train, y_train)
    else:
        final_model = ExplainableBoostingClassifier(**model_params)
        final_model.fit(X_train, y_train, sample_weight=sample_weight)

    logger.info("Generating predictions on test set...")
    test_pred_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred_calibrated = apply_logistic_recalibration(test_pred_proba, recalibration_model)

    output_dir = utils.RESULTS_DIR / 'models' / model_type / outcome / branch / feature_set
    predictions_dir = output_dir / 'predictions'
    artifacts_dir = output_dir / 'artifacts'
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    train_predictions = pd.DataFrame({
        'caseid': df.loc[X_train.index, 'caseid'],
        'y_true': y_train.values,
        'y_prob_raw': oof_predictions,
        'y_prob_calibrated': calibrated_oof,
        'threshold': threshold,
        'y_pred_label': (calibrated_oof >= threshold).astype(int),
        'fold': fold_indices,
        'is_oof': True,
        'outcome': outcome,
        'branch': branch,
        'feature_set': feature_set,
        'model_name': model_label,
        'pipeline': PIPELINE_NAME,
    })

    test_predictions = pd.DataFrame({
        'caseid': df.loc[X_test.index, 'caseid'],
        'y_true': y_test.values,
        'y_prob_raw': test_pred_proba,
        'y_prob_calibrated': test_pred_calibrated,
        'threshold': threshold,
        'y_pred_label': (test_pred_calibrated >= threshold).astype(int),
        'fold': -1,
        'is_oof': False,
        'outcome': outcome,
        'branch': branch,
        'feature_set': feature_set,
        'model_name': model_label,
        'pipeline': PIPELINE_NAME,
    })

    write_prediction_files(predictions_dir, train_predictions, test_predictions, logger)

    calibration_payload: Dict[str, float] = {
        'intercept': recalibration_model.intercept,
        'slope': recalibration_model.slope,
        'eps': recalibration_model.eps,
    }
    threshold_payload: Dict[str, float] = {
        'threshold': threshold,
        'youden_j': youden_j,
        'sensitivity': sensitivity,
        'specificity': specificity,
    }
    metadata_payload = {
        'seed': random_state,
        'folds': n_splits,
        'dataset_hash': dataset_hash,
        'model_type': model_type,
        'counts': {
            'train': {
                'total': int(len(y_train)),
                'positive': positive_count,
                'negative': negative_count,
            },
            'test': {
                'total': int(len(y_test)),
                'positive': int((y_test == 1).sum()),
                'negative': int((y_test == 0).sum()),
            },
        },
    }

    write_json(artifacts_dir / 'calibration.json', calibration_payload)
    write_json(artifacts_dir / 'threshold.json', threshold_payload)
    write_json(artifacts_dir / 'metadata.json', metadata_payload)
    logger.info("Saved calibration and threshold artifacts to %s", artifacts_dir)

    # Save model
    if model_type == "xgboost":
        model_path = output_dir / "model.json"
        final_model.save_model(model_path)
    else:
        model_path = output_dir / "model.ebm"
        final_model.save(str(model_path))
    logger.info("Saved model to %s", model_path)

    # SHAP Feature Importance (XGBoost only)
    if not smoke_test and model_type == "xgboost":
        save_shap_plots(final_model, X_test, output_dir)
        logger.info("Saved SHAP plots.")
    elif model_type != "xgboost":
        logger.info("Skipping SHAP generation for model type %s", model_type)

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument("--branch", type=str, required=True, choices=['windowed', 'non_windowed'], help="Data branch")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["xgboost", "ebm"],
        default="xgboost",
        help="Model type to train and evaluate",
    )
    parser.add_argument(
        "--legacy_imputation",
        action="store_true",
        help="Apply legacy imputation instead of preserving NaNs",
    )

    args = parser.parse_args()

    train_evaluate(
        args.outcome,
        args.branch,
        args.feature_set,
        args.smoke_test,
        args.model_type,
        args.legacy_imputation,
    )
