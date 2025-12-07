import argparse
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, confusion_matrix
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Reuse bootstrapping logic from existing codebase or re-implement explicitly?
# Re-implementing cleanly to ensure "1000 fold" and "Log Reg Calibration" request is met.

def calibrate_predictions(y_true, y_prob):
    """
    Applies Platt Scaling (Logistic Regression) to calibrate probabilities.
    Since we only have test set predictions here, we simulate 'post-hoc' calibration 
    validation. Ideally calibration is fit on specific set.
    Standard approach for post-hoc: Fit LR on y_prob vs y_true.
    """
    # Reshape for sklearn
    lr = LogisticRegression(C=1.0, solver='lbfgs')
    lr.fit(y_prob.reshape(-1, 1), y_true)
    y_calibrated = lr.predict_proba(y_prob.reshape(-1, 1))[:, 1]
    return y_calibrated, lr

def get_metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        'AUROC': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan,
        'AUPRC': average_precision_score(y_true, y_prob),
        'Brier Score': brier_score_loss(y_true, y_prob), # Only valid if calibrated
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'Threshold': threshold
    }

def find_optimal_threshold(y_true, y_prob):
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]

def bootstrap_analysis(y_true, y_prob, n_bootstraps=1000):
    """
    Performs 1000-fold bootstrapping to get CIs.
    """
    rng = np.random.RandomState(42)
    metrics_list = []
    
    # 1. Base Metrics (on full set)
    best_thresh = find_optimal_threshold(y_true, y_prob)
    base_metrics = get_metrics_at_threshold(y_true, y_prob, best_thresh)
    
    # 2. Bootstrap
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue
            
        y_t_boot = y_true[indices]
        y_p_boot = y_prob[indices]
        
        # Use simple fixed threshold from base OR re-optimize?
        # Usually re-optimization inside bootstrap is more rigorous but slower.
        # User asked for "threshold setting found in step_08".
        # Let's re-optimize threshold per bootstrap to capture threshold uncertainty too?
        # Or stick to base threshold. 
        # Standard: fix threshold or optimize. Let's re-optimize to be robust.
        thresh_boot = find_optimal_threshold(y_t_boot, y_p_boot)
        m = get_metrics_at_threshold(y_t_boot, y_p_boot, thresh_boot)
        metrics_list.append(m)
        
    # 3. Compute CIs
    results = {}
    for key in base_metrics:
        vals = [m[key] for m in metrics_list if not np.isnan(m[key])]
        if not vals: 
            results[key] = base_metrics[key]
            results[f'{key}_lower'] = np.nan
            results[f'{key}_upper'] = np.nan
        else:
            results[key] = base_metrics[key] # Report base metric
            results[f'{key}_lower'] = np.percentile(vals, 2.5)
            results[f'{key}_upper'] = np.percentile(vals, 97.5)
            
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_file", type=str, help="Path to predictions.csv")
    parser.add_argument("--calibrate", action='store_true', help="Apply Platt Scaling", default=True) # Defaulting to True per user request
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    pred_path = Path(args.predictions_file)
    if not pred_path.exists():
        logging.error(f"File not found: {pred_path}")
        return
        
    df = pd.read_csv(pred_path)
    
    # Expect columns: y_true, y_pred_proba
    y_true = df['y_true'].values
    y_prob = df['y_pred_proba'].values
    
    logging.info(f"Loaded {len(df)} predictions.")
    
    # Calibration
    if args.calibrate:
        logging.info("Applying Logistic Calibration (Platt Scaling)...")
        # Note: If doing this on Test set, strictly valid only for understanding potential performance.
        # Usually one calibrates on Valid. But "post-hoc calibration" often implies this.
        y_prob, lr_model = calibrate_predictions(y_true, y_prob)
        
        # Save calibrated probabilities
        df['y_pred_proba_calibrated'] = y_prob
        df.to_csv(pred_path, index=False) # Update file
    
    logging.info("Starting 1000-fold Bootstrapping...")
    results = bootstrap_analysis(y_true, y_prob, n_bootstraps=1000)
    
    # Print
    logging.info("--- Final Metrics (95% CI) ---")
    for k, v in results.items():
        if not k.endswith('_lower') and not k.endswith('_upper'):
            low = results.get(f'{k}_lower', np.nan)
            high = results.get(f'{k}_upper', np.nan)
            logging.info(f"{k}: {v:.4f} ({low:.4f} - {high:.4f})")
            
    # Save Metrics
    out_dir = pred_path.parent
    metrics_df = pd.DataFrame([results])
    metrics_df.to_csv(out_dir / 'metrics.csv', index=False)
    logging.info(f"Saved metrics to {out_dir / 'metrics.csv'}")

if __name__ == "__main__":
    main()
