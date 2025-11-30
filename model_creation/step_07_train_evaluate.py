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

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and returns a dictionary of metrics.
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold (maximizing F1)
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_test, y_pred_proba >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred_class = (y_pred_proba >= best_threshold).astype(int)

    metrics = {
        'AUROC': roc_auc_score(y_test, y_pred_proba),
        'AUPRC': average_precision_score(y_test, y_pred_proba),
        'Brier Score': brier_score_loss(y_test, y_pred_proba),
        'Accuracy': accuracy_score(y_test, y_pred_class),
        'Precision': precision_score(y_test, y_pred_class, zero_division=0),
        'Sensitivity (Recall)': recall_score(y_test, y_pred_class),
        'F1-Score': f1_score(y_test, y_pred_class),
        'Best Threshold': best_threshold
    }
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics

def bootstrap_evaluate(model, X_test, y_test, n_bootstraps=25, random_state=42):
    """
    Performs bootstrapping to calculate metrics and 95% CIs.
    """
    rng = np.random.RandomState(random_state)
    bootstrapped_metrics = []
    
    # Original evaluation
    original_metrics = evaluate_model(model, X_test, y_test)
    y_pred_proba_original = model.predict_proba(X_test)[:, 1]

    # Bootstrapping
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_test), len(y_test))
        if len(np.unique(y_test.iloc[indices])) < 2:
            continue
            
        X_boot = X_test.iloc[indices]
        y_boot = y_test.iloc[indices]
        
        metrics = evaluate_model(model, X_boot, y_boot)
        bootstrapped_metrics.append(metrics)
        
    # Calculate CIs
    results = {}
    for metric in original_metrics:
        values = [m[metric] for m in bootstrapped_metrics]
        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        results[metric] = original_metrics[metric]
        results[f'{metric}_lower'] = lower
        results[f'{metric}_upper'] = upper
        
    return results, y_pred_proba_original

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
    logger.info("Training model...")
    model = xgb.XGBClassifier(**params, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate with bootstrapping
    logger.info("Evaluating model with bootstrapping...")
    metrics, _ = bootstrap_evaluate(model, X_test, y_test, n_bootstraps=25 if not smoke_test else 2)
    
    logger.info("--- Final Metrics (with 95% CIs) ---")
    for k, v in metrics.items():
        if not k.endswith('_lower') and not k.endswith('_upper'):
            logger.info(f"{k}: {v:.4f} ({metrics[f'{k}_lower']:.4f} - {metrics[f'{k}_upper']:.4f})")

    # Save results
    output_dir = utils.RESULTS_DIR / 'models' / outcome / branch / feature_set
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    
    # Save model
    model.save_model(output_dir / "model.json")

    # SHAP
    if not smoke_test:
        save_shap_plots(model, X_test, output_dir)

    logger.info(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate AKI Prediction Models")
    parser.add_argument("--outcome", type=str, required=True, help="Target outcome name")
    parser.add_argument("--branch", type=str, required=True, choices=['windowed', 'non_windowed'], help="Data branch")
    parser.add_argument("--feature_set", type=str, required=True, help="Feature set name")
    parser.add_argument("--smoke_test", action="store_true", help="Run a quick smoke test")

    args = parser.parse_args()

    train_evaluate(args.outcome, args.branch, args.feature_set, args.smoke_test)
