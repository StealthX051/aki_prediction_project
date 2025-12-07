import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import seaborn as sns
from typing import Dict, List
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement
from docx.oxml.ns import qn
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed

# Constants for Aeon Analysis
# Point to 'results/aeon' instead of 'results'
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results' / 'aeon'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Update Mappings for Aeon Channels
# We can dynamically map or just pass through names if they are descriptive enough
FEATURE_SET_MAPPING = {
    'all_fused': 'All Channels (Fused)',
    'all_waveonly': 'All Channels (Waveform Only)',
    # Add common patterns if needed
}

NEURIPS_CSS = """
<style>
    body { font-family: 'Times New Roman', Times, serif; margin: 40px; }
    h1 { text-align: center; }
    table { border-collapse: collapse; width: 100%; margin-bottom: 40px; border-top: 2px solid #000; border-bottom: 2px solid #000; }
    th { text-align: left; padding: 8px; border-bottom: 1px solid #000; }
    td { padding: 8px; border-bottom: 1px solid #e0e0e0; }
    .footer { font-size: 10pt; color: #666; text-align: center; margin-top: 20px; }
</style>
"""

def load_aeon_predictions() -> pd.DataFrame:
    """
    Crawls results/aeon/models to find predictions.csv files.
    """
    # Look for predictions.csv in subdirectories of results/aeon/models
    pred_files = list(RESULTS_DIR.glob('models/**/predictions.csv'))
    if not pred_files:
        print(f"No predictions.csv files found in {RESULTS_DIR}/models!")
        return pd.DataFrame()
    
    dfs = []
    print(f"Found {len(pred_files)} prediction files.")
    for f in pred_files:
        try:
            df = pd.read_csv(f)
            # Ensure branch/outcome/model cols exist (step_06 saves them)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    return full_df

# Reuse calibration and metric calculation logic
# Since this script is distinct, we can just copy/import the functions 
# or implement them locally to ensure self-containment.
# I'll implement locally for robustness (avoid import issues if original changes).

def calibrate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    print("Calibrating predictions (Aeon)...")
    calibrated_dfs = []
    # Group by key identifiers
    groups = df.groupby(['outcome', 'branch', 'feature_set', 'model_name'])
    
    for name, group in groups:
        if len(group) < 10:
            group['y_prob_calibrated'] = group['y_pred_proba']
            calibrated_dfs.append(group)
            continue
            
        y_true = group['y_true'].values
        y_prob = group['y_pred_proba'].values
        
        lr = LogisticRegression(C=1.0, solver='lbfgs')
        try:
            X = y_prob.reshape(-1, 1)
            lr.fit(X, y_true)
            y_calib = lr.predict_proba(X)[:, 1]
        except:
            y_calib = y_prob
            
        group['y_prob_calibrated'] = y_calib
        calibrated_dfs.append(group)
        
    return pd.concat(calibrated_dfs, ignore_index=True)

def find_optimal_threshold_f2(y_true, y_prob, min_spec=0.6):
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f2 = -1.0
    best_thresh = 0.5
    from sklearn.metrics import fbeta_score
    
    # 1. Constrained search
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if y_pred.sum() == 0: continue
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        
        if spec >= min_spec:
            score = fbeta_score(y_true, y_pred, beta=2)
            if score > best_f2:
                best_f2 = score
                best_thresh = t
                
    # 2. Fallback
    if best_f2 < 0:
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0: continue
            score = fbeta_score(y_true, y_pred, beta=2)
            if score > best_f2:
                best_f2 = score
                best_thresh = t
                
    return best_thresh

def calc_single_pass(y_t, y_p):
    try:
        t = find_optimal_threshold_f2(y_t, y_p)
        y_pred = (y_p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        return {
            'AUROC': roc_auc_score(y_t, y_p),
            'AUPRC': average_precision_score(y_t, y_p),
            'Brier Score': brier_score_loss(y_t, y_p),
            'Sensitivity': recall_score(y_t, y_pred),
            'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
            'F1 Score': f1_score(y_t, y_pred),
            'Precision': precision_score(y_t, y_pred, zero_division=0)
        }
    except:
        return {}

def process_group(outcome, branch, feature_set, model_name, group, n_bootstrap):
    y_true = group['y_true'].values
    y_prob = group['y_prob_calibrated'].values
    
    point = calc_single_pass(y_true, y_prob)
    if not point: return None
    
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    results = {k:[] for k in point}
    
    for _ in range(n_bootstrap):
        idx = rng.choice(indices, len(indices), replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2: continue
        m = calc_single_pass(yt, yp)
        for k,v in m.items(): results[k].append(v)
        
    final = {'Outcome': outcome, 'Branch': branch, 'Feature Set': feature_set, 'Model': model_name}
    for k, v in point.items():
        boots = results[k]
        if boots:
            p2_5, p97_5 = np.percentile(boots, [2.5, 97.5])
            final[k] = f"{v:.3f} ({p2_5:.3f}-{p97_5:.3f})"
            final[f'{k}_val'] = v
        else:
            final[k] = f"{v:.3f}"
            final[f'{k}_val'] = v
            
    return final

def calculate_metrics(df, n_bootstrap=1000):
    groups = list(df.groupby(['outcome', 'branch', 'feature_set', 'model_name']))
    print(f"Bootstrapping {len(groups)} configurations...")
    results = Parallel(n_jobs=-1)(
        delayed(process_group)(o,b,f,m,g,n_bootstrap) for (o,b,f,m),g in groups
    )
    return pd.DataFrame([r for r in results if r])

def plot_curves_aeon(df):
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    
    for (outcome, branch), group in df.groupby(['outcome', 'branch']):
        # ROC
        plt.figure(figsize=(8,6))
        for key, g in group.groupby(['feature_set', 'model_name']):
            yt, yp = g['y_true'], g['y_pred_proba']
            fpr, tpr, _ = roc_curve(yt, yp)
            auc_val = roc_auc_score(yt, yp)
            label = f"{key[1]} - {key[0]} (AUC={auc_val:.3f})"
            plt.plot(fpr, tpr, label=label)
        plt.plot([0,1],[0,1],'k--')
        plt.legend()
        plt.title(f"ROC - {outcome}")
        plt.savefig(FIGURES_DIR / f"roc_{outcome}.png")
        plt.close()

def main():
    print("Starting Aeon Results Analysis...")
    df = load_aeon_predictions()
    if df.empty:
        return
    
    df = calibrate_predictions(df)
    metrics = calculate_metrics(df)
    
    metrics.to_csv(TABLES_DIR / 'measurements_aeon.csv', index=False)
    print(f"Saved metrics to {TABLES_DIR}")
    
    plot_curves_aeon(df)
    print("Plots generated.")

if __name__ == "__main__":
    main()
