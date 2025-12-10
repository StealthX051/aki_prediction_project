import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ColorConverter, to_hex
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.shared import OxmlElement
from docx.oxml.ns import qn
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed

# Constants
RESULTS_DIR = Path(__file__).resolve().parent.parent / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET_MAPPING = {
    'preop_only': 'Preoperative Only',
    'all_waveforms': 'All Waveforms',
    'preop_and_all_waveforms': 'Preop + All Waveforms',
    'awp_only': 'Airway Pressure Only',
    'co2_only': 'CO2 Only',
    'ecg_only': 'ECG Only',
    'pleth_only': 'Plethysmography Only',
    'monitors_only': 'Standard Monitors Only',
    'ventilator_only': 'Ventilator Only',
    'preop_and_awp': 'Preop + Airway Pressure',
    'preop_and_co2': 'Preop + CO2',
    'preop_and_ecg': 'Preop + ECG',
    'preop_and_pleth': 'Preop + Plethysmography',
    'preop_and_all_minus_awp': 'Preop + All (No AWP)',
    'preop_and_all_minus_co2': 'Preop + All (No CO2)',
    'preop_and_all_minus_ecg': 'Preop + All (No ECG)',
    'preop_and_all_minus_pleth': 'Preop + All (No Pleth)',
    # Aeon / Multirocket Mappings
    'all_fused': 'All Channels (Fused)',
    'all_waveonly': 'All Channels (Waveform Only)',
}

NEURIPS_CSS = """
<style>
    body {
        font-family: 'Times New Roman', Times, serif;
        background-color: #ffffff;
        color: #000000;
        margin: 40px;
    }
    
    h1 {
        font-weight: bold;
        font-size: 24pt;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .subtitle {
        font-size: 14pt;
        text-align: center;
        margin-bottom: 30px;
        font-style: italic;
    }
    
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 40px;
        font-size: 11pt;
        border-top: 2px solid #000000;
        border-bottom: 2px solid #000000;
    }
    
    th {
        background-color: #ffffff;
        font-weight: bold;
        text-align: left;
        padding: 8px;
        border-bottom: 1px solid #000000;
    }
    
    td {
        padding: 8px;
        border-bottom: 1px solid #e0e0e0;
    }
    
    tr:last-child td {
        border-bottom: none;
    }
    
    /* Tasteful highlighting */
    .highlight-high {
        background-color: #f0f7f0; /* Very light green */
        font-weight: bold;
    }
    
    .footer {
        font-size: 10pt;
        color: #666666;
        margin-top: 20px;
        text-align: center;
        border-top: 1px solid #000000;
        padding-top: 10px;
    }
</style>
"""

def load_all_predictions() -> pd.DataFrame:
    """
    Crawls results/models AND results/aeon/models to find and concat predictions.csv files.
    """
    all_dfs = []
    
    # 1. Standard (Catch22) Models
    try:
        catch22_path = RESULTS_DIR / 'models'
        if catch22_path.exists():
            pred_files = list(catch22_path.glob('**/predictions.csv'))
            print(f"Found {len(pred_files)} Catch22 prediction files.")
            for f in pred_files:
                try:
                    df = pd.read_csv(f)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        else:
            print(f"Warning: {catch22_path} does not exist.")
    except Exception as e:
        print(f"Error searching standard models: {e}")

    # 2. Aeon (Multirocket) Models
    try:
        aeon_path = RESULTS_DIR / 'aeon' / 'models'
        if aeon_path.exists():
            aeon_files = list(aeon_path.glob('**/predictions.csv'))
            print(f"Found {len(aeon_files)} Aeon prediction files.")
            for f in aeon_files:
                try:
                    df = pd.read_csv(f)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error loading {f}: {e}")
        else:
            print(f"Warning: {aeon_path} does not exist.")
    except Exception as e:
        print(f"Error searching Aeon models: {e}")

    if not all_dfs:
        print("No prediction files found in any location!")
        return pd.DataFrame()
        
    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df

def calibrate_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Logistic Regression calibration (Platt scaling) to predictions.
    Returns dataframe with 'y_prob_calibrated'.
    """
    print("Calibrating predictions...")
    calibrated_dfs = []
    
    groups = df.groupby(['outcome', 'branch', 'feature_set', 'model_name'])
    # We calibrate each model instance independently
    groups = df.groupby(['outcome', 'branch', 'feature_set', 'model_name'])
    
    for name, group in groups:
        y_true = group['y_true'].values
        y_prob = group['y_pred_proba'].values
        
        # If too few samples or only one class, skip calibration
        if len(y_true) < 10 or len(np.unique(y_true)) < 2:
            group['y_prob_calibrated'] = y_prob
            calibrated_dfs.append(group)
            continue
            
        # Fit a single logistic calibration model on the entire group
        # This preserves ranking (monotone transformation) unlike fold-wise calibration
        lr = LogisticRegression(C=1.0, solver='lbfgs')
        try:
            # Reshape for sklearn
            X = y_prob.reshape(-1, 1)
            lr.fit(X, y_true)
            y_prob_calibrated = lr.predict_proba(X)[:, 1]
        except Exception:
            # Fall back to raw probabilities if calibration fails
            y_prob_calibrated = y_prob
            
        group['y_prob_calibrated'] = y_prob_calibrated
        calibrated_dfs.append(group)
        
    return pd.concat(calibrated_dfs, ignore_index=True)

def find_optimal_threshold_f2(y_true, y_prob, min_spec=0.6):
    """
    Finds threshold maximizing F2 score with a minimum specificity constraint.
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_f2 = -1.0
    best_thresh = 0.5
    
    from sklearn.metrics import fbeta_score, confusion_matrix

    # Pass 1: With Constraint
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Need at least one positive prediction to have meaningful F2
        if y_pred.sum() == 0:
            continue
            
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        if spec < min_spec:
            continue

        score = fbeta_score(y_true, y_pred, beta=2)
        if score > best_f2:
            best_f2 = score
            best_thresh = t

    # Pass 2: Fallback if no threshold satisfies min_spec
    if best_f2 < 0:
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            if y_pred.sum() == 0:
                continue
            score = fbeta_score(y_true, y_pred, beta=2)
            if score > best_f2:
                best_f2 = score
                best_thresh = t
            
    return best_thresh

def calc_single_pass(y_t, y_p):
    try:
        # Thresholding
        best_thresh = find_optimal_threshold_f2(y_t, y_p)
        y_pred = (y_p >= best_thresh).astype(int)
        
        auroc = roc_auc_score(y_t, y_p)
        auprc = average_precision_score(y_t, y_p)
        brier = brier_score_loss(y_t, y_p)
        acc = accuracy_score(y_t, y_pred)
        prec = precision_score(y_t, y_pred, zero_division=0)
        sens = recall_score(y_t, y_pred)
        f1 = f1_score(y_t, y_pred)
        
        tn, fp, fn, tp = confusion_matrix(y_t, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'AUROC': auroc, 'AUPRC': auprc, 'Brier Score': brier,
            'Accuracy': acc, 'Sensitivity': sens, 'Specificity': spec,
            'Precision': prec, 'F1 Score': f1
        }
    except:
        return {}

def process_group(outcome, branch, feature_set, model_name, group, n_bootstrap):
    y_true = group['y_true'].values
    y_prob = group['y_prob_calibrated'].values
    
    # 1. Point Estimates (on full data)
    point_metrics = calc_single_pass(y_true, y_prob)
    if not point_metrics:
        return None
        
    # 2. Bootstrapping
    bootstrap_results = {k: [] for k in point_metrics.keys()}
    rng = np.random.RandomState(42)
    indices = np.arange(len(y_true))
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        boot_idx = rng.choice(indices, len(indices), replace=True)
        y_t_boot = y_true[boot_idx]
        y_p_boot = y_prob[boot_idx]
        
        # Skip degenerate samples (only one class)
        if len(np.unique(y_t_boot)) < 2:
            continue
            
        m = calc_single_pass(y_t_boot, y_p_boot)
        for k, v in m.items():
            bootstrap_results[k].append(v)
            
    # 3. Calculate CIs
    final_metrics = {
        'Outcome': outcome,
        'Branch': branch,
        'Feature Set': feature_set,
        'Model': model_name
    }
    
    for k, v in point_metrics.items():
        boot_vals = bootstrap_results[k]
        if boot_vals:
            lower = np.percentile(boot_vals, 2.5)
            upper = np.percentile(boot_vals, 97.5)
            # Store as formatted string: "0.123 (0.100-0.150)"
            final_metrics[k] = f"{v:.3f} ({lower:.3f}-{upper:.3f})"
            # Also store raw value for sorting/highlighting
            final_metrics[f"{k}_val"] = v
        else:
            final_metrics[k] = f"{v:.3f}"
            final_metrics[f"{k}_val"] = v
            
    return final_metrics

def calculate_metrics(df: pd.DataFrame, n_bootstrap: int = 1000) -> pd.DataFrame:
    """
    Calculates performance metrics with 95% CIs using bootstrapping (Parallelized).
    """
    groups = list(df.groupby(['outcome', 'branch', 'feature_set', 'model_name']))
    print(f"Starting parallel bootstrap analysis for {len(groups)} model configurations...")
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_group)(outcome, branch, feature_set, model_name, group, n_bootstrap)
        for (outcome, branch, feature_set, model_name), group in groups
    )
    
    # Filter out None results
    metrics_list = [r for r in results if r is not None]
            
    return pd.DataFrame(metrics_list)

def generate_html_tables(metrics_df: pd.DataFrame):
    """
    Generates styled HTML tables for each Outcome/Branch.
    """
    # Apply Naming Convention
    metrics_df['Feature Set'] = metrics_df['Feature Set'].map(FEATURE_SET_MAPPING).fillna(metrics_df['Feature Set'])
    
    # Group by Outcome and Branch
    for (outcome, branch, model_name), group in metrics_df.groupby(['Outcome', 'Branch', 'Model']):
        # Sort by AUROC Value (Hidden column)
        table_df = group.sort_values('AUROC_val', ascending=False)
        
        # Display Columns
        display_cols = ['Feature Set', 'AUROC', 'AUPRC', 'Brier Score', 'Sensitivity', 'Specificity', 'F1 Score']
        
        # Filter columns
        table_df_display = table_df[display_cols].copy()
        
        # Style: Academic + Modern Subtle Gradient
        cmap = LinearSegmentedColormap.from_list("soft_teal", ["#ffffff", "#d1e7dd"]) 
        cmap_r = LinearSegmentedColormap.from_list("soft_teal_r", ["#d1e7dd", "#ffffff"]) 
        
        # We need to apply styles based on the _val columns but display the formatted strings
        # Pandas Styler is tricky with this.
        # Strategy: Create a Styler on the DISPLAY dataframe, but use the _val dataframe for logic.
        
        styler = table_df_display.style
        
        # Helper to apply background gradient based on hidden values
        def apply_gradient(s, val_col, cmap):
            # Get values from the original sorted dataframe
            # We must ensure alignment. table_df is sorted, table_df_display is a copy of it.
            # s.index should match table_df.index
            vals = table_df.loc[s.index, val_col]
            
            # Normalize
            min_v, max_v = vals.min(), vals.max()
            if max_v == min_v:
                return ['' for _ in s]
                
            norm = plt.Normalize(min_v, max_v)
            colors = [cmap(norm(v)) for v in vals]
            
            # Convert to hex/css
            from matplotlib.colors import to_hex
            return [f'background-color: {to_hex(c)}' for c in colors]

        # Helper for bolding best
        def highlight_best(s, val_col, mode='max'):
            vals = table_df.loc[s.index, val_col]
            if mode == 'max':
                is_best = vals == vals.max()
            else:
                is_best = vals == vals.min()
            return ['font-weight: bold' if v else '' for v in is_best]

        # Apply styles column by column
        # AUROC
        styler.apply(apply_gradient, val_col='AUROC_val', cmap=cmap, subset=['AUROC'])
        styler.apply(highlight_best, val_col='AUROC_val', mode='max', subset=['AUROC'])
        
        # AUPRC
        styler.apply(apply_gradient, val_col='AUPRC_val', cmap=cmap, subset=['AUPRC'])
        styler.apply(highlight_best, val_col='AUPRC_val', mode='max', subset=['AUPRC'])
        
        # Brier (Lower Better)
        styler.apply(apply_gradient, val_col='Brier Score_val', cmap=cmap_r, subset=['Brier Score'])
        styler.apply(highlight_best, val_col='Brier Score_val', mode='min', subset=['Brier Score'])
        
        # Sens/Spec/F1
        for col in ['Sensitivity', 'Specificity', 'F1 Score']:
            styler.apply(apply_gradient, val_col=f'{col}_val', cmap=cmap, subset=[col])
            styler.apply(highlight_best, val_col=f'{col}_val', mode='max', subset=[col])

        styler.hide(axis="index")
        
        # Format for HTML: Replace " (" with "<br>("
        # We need to apply this to the display values
        def format_cell(v):
            if isinstance(v, str) and ' (' in v:
                return v.replace(' (', '<br>(')
            return v
            
        styler.format(format_cell)
        
        # Center align all cells
        styler.set_properties(**{'text-align': 'center'})
                               
        html_table = styler.to_html(table_id=f"results_{outcome}_{branch}_{model_name}", escape=False)
        
        # Construct Full HTML
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            {NEURIPS_CSS}
            <title>Results: {outcome} ({branch}) - {model_name}</title>
        </head>
        <body>
            <h1>Model Performance Analysis</h1>
            <div class="subtitle">Outcome: {outcome} | Branch: {branch} | Model: {model_name}</div>
            
            {html_table}
            
            <div class="footer">
                Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | AKI Prediction Project
            </div>
        </body>
        </html>
        """
        
        # Save
        with open(TABLES_DIR / f'results_{outcome}_{branch}_{model_name}.html', 'w') as f:
            f.write(full_html)
            
    print("Tables generated.")

def plot_curves(df: pd.DataFrame):
    """
    Generates ROC, PR, and Calibration curves.
    """
    # Set Style for NeurIPS (Serif, Academic)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.9
    plt.rcParams['legend.edgecolor'] = 'white'
    
    # Group by Outcome and Branch
    for (outcome, branch, model_name), group in df.groupby(['outcome', 'branch', 'model_name']):
        
        # 1. ROC Curve (Raw)
        plt.figure(figsize=(8, 6)) # Standard academic figure size
        
        # Pre-calculate AUCs to sort legend
        auc_scores = []
        for feature_set, model_group in group.groupby('feature_set'):
            y_true = model_group['y_true']
            y_prob = model_group['y_pred_proba']
            roc_auc = roc_auc_score(y_true, y_prob)
            auc_scores.append((feature_set, roc_auc))
        
        # Sort by AUC descending
        auc_scores.sort(key=lambda x: x[1], reverse=True)
        
        for feature_set, roc_auc in auc_scores:
            model_group = group[group['feature_set'] == feature_set]
            y_true = model_group['y_true']
            y_prob = model_group['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            
            label_name = FEATURE_SET_MAPPING.get(feature_set, feature_set)
            plt.plot(fpr, tpr, label=f'{label_name} (AUC = {roc_auc:.3f})', linewidth=1.5)
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curves - {outcome} ({branch}) - {model_name}')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'roc_{outcome}_{branch}_{model_name}.png', dpi=300)
        plt.close()
        
        # 2. PR Curve (Raw)
        plt.figure(figsize=(8, 6))
        
        # Pre-calculate AUPRCs to sort legend
        auprc_scores = []
        for feature_set, model_group in group.groupby('feature_set'):
            y_true = model_group['y_true']
            y_prob = model_group['y_pred_proba']
            pr_auc = average_precision_score(y_true, y_prob)
            auprc_scores.append((feature_set, pr_auc))
            
        # Sort by AUPRC descending
        auprc_scores.sort(key=lambda x: x[1], reverse=True)
        
        for feature_set, pr_auc in auprc_scores:
            model_group = group[group['feature_set'] == feature_set]
            y_true = model_group['y_true']
            y_prob = model_group['y_pred_proba']
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            
            label_name = FEATURE_SET_MAPPING.get(feature_set, feature_set)
            plt.plot(rec, prec, label=f'{label_name} (AUPRC = {pr_auc:.3f})', linewidth=1.5)
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'PR Curves - {outcome} ({branch}) - {model_name}')
        plt.legend(loc='lower left')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'pr_{outcome}_{branch}_{model_name}.png', dpi=300)
        plt.close()
        
        # 3. Calibration Curve
        plt.figure(figsize=(8, 6))
        for feature_set, model_group in group.groupby('feature_set'):
            y_true = model_group['y_true']
            y_prob = model_group['y_prob_calibrated']
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            
            label_name = FEATURE_SET_MAPPING.get(feature_set, feature_set)
            plt.plot(prob_pred, prob_true, marker='o', label=label_name, linewidth=1, markersize=4, alpha=0.7)
            
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curves - {outcome} ({branch}) - {model_name}')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / f'calibration_{outcome}_{branch}_{model_name}.png', dpi=300)
        plt.close()

def set_cell_background(cell, color_hex):
    """
    Set background color of a table cell in DOCX.
    color_hex: string like "#FFFFFF" or "FFFFFF"
    """
    if color_hex.startswith('#'):
        color_hex = color_hex[1:]
    
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)
    tcPr.append(shd)

def get_style_info(df: pd.DataFrame) -> Dict:
    """
    Calculates style information (bold, background color) for each cell.
    Returns a nested dict: {row_idx: {col_name: {'bold': bool, 'bg': hex_str}}}
    """
    style_info = {}
    
    # Colormaps
    cmap = LinearSegmentedColormap.from_list("soft_teal", ["#ffffff", "#d1e7dd"]) 
    cmap_r = LinearSegmentedColormap.from_list("soft_teal_r", ["#d1e7dd", "#ffffff"]) 
    
    # Initialize
    for idx in df.index:
        style_info[idx] = {}
        
    # Process each metric column
    metrics = ['AUROC', 'AUPRC', 'Brier Score', 'Sensitivity', 'Specificity', 'F1 Score']
    
    for col in metrics:
        val_col = f'{col}_val'
        if val_col not in df.columns:
            continue
            
        vals = df[val_col]
        min_v, max_v = vals.min(), vals.max()
        
        # Determine best
        if col == 'Brier Score':
            is_best = vals == min_v
            norm = plt.Normalize(min_v, max_v)
            current_cmap = cmap_r
        else:
            is_best = vals == max_v
            norm = plt.Normalize(min_v, max_v)
            current_cmap = cmap
            
        # Calculate styles
        for idx, val in vals.items():
            # Bold
            bold = is_best[idx]
            
            # Background
            if max_v == min_v:
                bg_color = "#ffffff"
            else:
                bg_color = to_hex(current_cmap(norm(val)))
                
            style_info[idx][col] = {'bold': bold, 'bg': bg_color}
            
    return style_info

def generate_docx_report(metrics_df: pd.DataFrame):
    """
    Generates a DOCX report with all tables.
    """
    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Times New Roman'
    style.font.size = Pt(11)
    
    doc.add_heading('AKI Prediction Project - Results Report', 0)
    
    # Ensure mapping is applied (idempotent)
    metrics_df['Feature Set'] = metrics_df['Feature Set'].map(FEATURE_SET_MAPPING).fillna(metrics_df['Feature Set'])
    
    for (outcome, branch, model_name), group in metrics_df.groupby(['Outcome', 'Branch', 'Model']):
        doc.add_heading(f'Outcome: {outcome} | Branch: {branch} | Model: {model_name}', level=1)
        
        # Sort
        table_df = group.sort_values('AUROC_val', ascending=False)
        display_cols = ['Feature Set', 'AUROC', 'AUPRC', 'Brier Score', 'Sensitivity', 'Specificity', 'F1 Score']
        
        # Get Styles
        styles = get_style_info(table_df)
        
        table_df = table_df[display_cols]
        
        # Add Table
        table = doc.add_table(rows=1, cols=len(display_cols))
        table.style = 'Table Grid'
        
        # Header
        hdr_cells = table.rows[0].cells
        for i, col in enumerate(display_cols):
            hdr_cells[i].text = col
            hdr_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            hdr_cells[i].paragraphs[0].runs[0].bold = True
            
        # Rows
        for idx, row in table_df.iterrows():
            row_cells = table.add_row().cells
            for i, col in enumerate(display_cols):
                val = str(row[col])
                # Format: Value\n(CI)
                if ' (' in val:
                    val = val.replace(' (', '\n(')
                
                cell = row_cells[i]
                cell.text = val
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                
                # Apply Styles
                if col in styles[idx]:
                    s = styles[idx][col]
                    if s['bold']:
                        for run in cell.paragraphs[0].runs:
                            run.bold = True
                    set_cell_background(cell, s['bg'])
                
        doc.add_paragraph() # Spacer
        
    doc.save(RESULTS_DIR / 'report.docx')
    print("DOCX report generated.")

def generate_pdf_report(metrics_df: pd.DataFrame):
    """
    Generates an aggregated PDF report using Matplotlib.
    """
    # Ensure mapping is applied (idempotent)
    metrics_df['Feature Set'] = metrics_df['Feature Set'].map(FEATURE_SET_MAPPING).fillna(metrics_df['Feature Set'])
    
    with PdfPages(RESULTS_DIR / 'report.pdf') as pdf:
        for (outcome, branch, model_name), group in metrics_df.groupby(['Outcome', 'Branch', 'Model']):
            # Sort
            table_df = group.sort_values('AUROC_val', ascending=False)
            display_cols = ['Feature Set', 'AUROC', 'AUPRC', 'Brier Score', 'Sensitivity', 'Specificity', 'F1 Score']
            
            # Get Styles
            styles = get_style_info(table_df)
            
            table_df_display = table_df[display_cols]
            
            # Format values for display
            display_data = []
            for _, row in table_df_display.iterrows():
                new_row = []
                for col in display_cols:
                    val = str(row[col])
                    if ' (' in val:
                        val = val.replace(' (', '\n(')
                    new_row.append(val)
                display_data.append(new_row)
            
            # Create Figure
            # Estimate height: Header + Rows * Height per row
            row_height = 0.6
            fig_height = len(table_df) * row_height + 2
            
            fig, ax = plt.subplots(figsize=(12, fig_height))
            ax.axis('off')
            ax.set_title(f"Outcome: {outcome} | Branch: {branch} | Model: {model_name}", fontsize=14, fontweight='bold', pad=20)
            
            # Create Table
            table = ax.table(cellText=display_data, colLabels=display_cols, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2.5) # Scale height to accommodate newlines
            
            # Style headers
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold')
                    cell.set_facecolor('#f0f0f0')
                else:
                    # Data rows (1-indexed in table, 0-indexed in df)
                    # We need to map back to df index
                    df_idx = table_df.index[row - 1]
                    col_name = display_cols[col]
                    
                    if col_name in styles[df_idx]:
                        s = styles[df_idx][col_name]
                        cell.set_facecolor(s['bg'])
                        if s['bold']:
                            cell.set_text_props(weight='bold')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
    print("PDF report generated.")

def main():
    print("Script started. Loading predictions... (this may take a moment)", flush=True)
    df = load_all_predictions()
    if df.empty:
        print("No data found.")
        return
        
    print("Calibrating...")
    df_calib = calibrate_predictions(df)
    
    print("Calculating metrics...")
    metrics_df = calculate_metrics(df_calib)
    
    # Save metrics to CSV for backup
    metrics_df.to_csv(TABLES_DIR / 'metrics_summary.csv', index=False)
    print("Metrics saved to CSV.")
    
    print("Generating tables...")
    generate_html_tables(metrics_df)
    generate_docx_report(metrics_df)
    generate_pdf_report(metrics_df)
    
    print("Generating plots...")
    plot_curves(df_calib)
    
    print("Done!")

if __name__ == "__main__":
    main()
