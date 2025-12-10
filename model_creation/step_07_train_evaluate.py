import argparse
import json
import logging
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix, brier_score_loss
)

# Add project root to path to import utils
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from model_creation import utils

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def train_evaluate(outcome, branch, feature_set, smoke_test=False):
    logger.info(f"Starting Training/Evaluation for Outcome: {outcome}, Branch: {branch}, Feature Set: {feature_set}")

    # Load params
    params_file = utils.RESULTS_DIR / 'params' / outcome / branch / f"{feature_set}.json"
    if not params_file.exists():
        logger.error(f"Parameters file not found: {params_file}. Run HPO first.")
        return

    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Load and prepare data
    try:
        df = utils.load_data(branch)
        # We need to keep the original df to retrieve caseids later
        X_train, X_test, y_train, y_test, _ = utils.prepare_data(
            df, outcome, feature_set
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
        params['n_estimators'] = 10

    # Train model
    logger.info("Training XGBoost model...")
    # Ensure random state for reproducibility
    if 'random_state' not in params:
        params['random_state'] = 42
        
    model = xgb.XGBClassifier(**params, n_jobs=-1)
    model.fit(X_train, y_train)

    # Generate Predictions on Test Set
    logger.info("Generating predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Construct Tidy Output DataFrame
    # Critical: Retrieve caseids using the index of y_test which matches the original df
    logger.info("Constructing predictions dataframe...")
    try:
        caseids = df.loc[y_test.index, 'caseid']
        split_group = df.loc[y_test.index, 'split_group'] # Should be 'test' but good to be explicit
            
        predictions_df = pd.DataFrame({
            'caseid': caseids,
            'y_true': y_test,
            'y_pred_proba': y_pred_proba,
            'split_group': split_group,
            'model_name': 'XGBoost',
            'feature_set': feature_set,
            'outcome': outcome,
            'branch': branch
        })
    except KeyError as e:
        logger.error(f"Failed to retrieve metadata (caseid/split_group): {e}")
        return

    # Define Output Directory
    output_dir = utils.RESULTS_DIR / 'models' / outcome / branch / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save Predictions
    pred_path = output_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"Saved predictions to {pred_path}")
    
    # 2. Save Model
    model.save_model(output_dir / "model.json")
    logger.info(f"Saved model to {output_dir / 'model.json'}")

    # 3. SHAP Feature Importance (Skip for smoke test/speed if needed, but usually good to have)
    if not smoke_test:
        save_shap_plots(model, X_test, output_dir)
        logger.info("Saved SHAP plots.")

    logger.info("Training and evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument("--branch", type=str, required=True, choices=['windowed', 'non_windowed'], help="Data branch")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")

    args = parser.parse_args()

    train_evaluate(args.outcome, args.branch, args.feature_set, args.smoke_test)
