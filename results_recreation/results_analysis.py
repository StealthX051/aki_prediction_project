import logging
import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml.shared import OxmlElement
from docx.shared import Pt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, to_hex
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reporting.display_dictionary import load_display_dictionary

# Constants
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / 'results'))
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'

# Ensure directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_SET_MAPPING = {
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

try:
    DISPLAY_DICTIONARY = load_display_dictionary()
    FEATURE_SET_MAPPING = DISPLAY_DICTIONARY.feature_set_labels(DEFAULT_FEATURE_SET_MAPPING)
except FileNotFoundError:
    logger.warning(
        "Display dictionary not found; using default feature set labels only. "
        "Create metadata/display_dictionary.json to override."
    )
    DISPLAY_DICTIONARY = None
    FEATURE_SET_MAPPING = DEFAULT_FEATURE_SET_MAPPING

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

METRIC_DISPLAY_MAP = {
    'auroc': 'AUROC',
    'auprc': 'AUPRC',
    'brier': 'Brier Score',
    'sensitivity': 'Sensitivity',
    'specificity': 'Specificity',
    'f1': 'F1 Score',
}


def prepare_metrics_for_display(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare metrics_summary output for styling and reporting."""

    metrics_df = summary_df.copy()

    rename_map = {
        'outcome': 'Outcome',
        'branch': 'Branch',
        'feature_set': 'Feature Set',
        'model_name': 'Model',
    }
    metrics_df.rename(columns=rename_map, inplace=True)

    display_pairs = list(METRIC_DISPLAY_MAP.items())
    delta_pairs = [(f"delta_{k}", f"Δ {v}") for k, v in METRIC_DISPLAY_MAP.items()]

    for metric_key, display_name in display_pairs + delta_pairs:
        if metric_key not in metrics_df.columns:
            continue

        lower_col = f"{metric_key}_lower"
        upper_col = f"{metric_key}_upper"
        metrics_df[f"{display_name}_val"] = metrics_df[metric_key]

        if lower_col in metrics_df.columns and upper_col in metrics_df.columns:
            metrics_df[display_name] = metrics_df.apply(
                lambda row: f"{row[metric_key]:.3f} ({row[lower_col]:.3f}-{row[upper_col]:.3f})",
                axis=1,
            )
        else:
            metrics_df[display_name] = metrics_df[metric_key].apply(lambda v: f"{v:.3f}")

    return metrics_df

def generate_html_tables(metrics_df: pd.DataFrame):
    """
    Generates styled HTML tables for each Outcome/Branch.
    """
    metrics_df['Feature Set'] = metrics_df['Feature Set'].map(FEATURE_SET_MAPPING).fillna(metrics_df['Feature Set'])

    for (outcome, branch, model_name), group in metrics_df.groupby(['Outcome', 'Branch', 'Model']):
        table_df = group.sort_values('AUROC_val', ascending=False)

        display_cols = [
            'Feature Set',
            'AUROC',
            'AUPRC',
            'Brier Score',
            'Sensitivity',
            'Specificity',
            'F1 Score',
        ]
        table_df_display = table_df[display_cols].copy()

        cmap = LinearSegmentedColormap.from_list("soft_teal", ["#ffffff", "#d1e7dd"])
        cmap_r = LinearSegmentedColormap.from_list("soft_teal_r", ["#d1e7dd", "#ffffff"])

        styler = table_df_display.style

        def apply_gradient(s, val_col, cmap):
            vals = table_df.loc[s.index, val_col]
            min_v, max_v = vals.min(), vals.max()
            if max_v == min_v:
                return ['' for _ in s]
            norm = plt.Normalize(min_v, max_v)
            colors = [cmap(norm(v)) for v in vals]
            from matplotlib.colors import to_hex
            return [f'background-color: {to_hex(c)}' for c in colors]

        def highlight_best(s, val_col, mode='max'):
            vals = table_df.loc[s.index, val_col]
            is_best = vals == (vals.max() if mode == 'max' else vals.min())
            return ['font-weight: bold' if v else '' for v in is_best]

        if 'AUROC_val' in table_df.columns:
            styler.apply(apply_gradient, val_col='AUROC_val', cmap=cmap, subset=['AUROC'])
            styler.apply(highlight_best, val_col='AUROC_val', mode='max', subset=['AUROC'])
        if 'AUPRC_val' in table_df.columns:
            styler.apply(apply_gradient, val_col='AUPRC_val', cmap=cmap, subset=['AUPRC'])
            styler.apply(highlight_best, val_col='AUPRC_val', mode='max', subset=['AUPRC'])
        if 'Brier Score_val' in table_df.columns:
            styler.apply(apply_gradient, val_col='Brier Score_val', cmap=cmap_r, subset=['Brier Score'])
            styler.apply(highlight_best, val_col='Brier Score_val', mode='min', subset=['Brier Score'])
        for col in ['Sensitivity', 'Specificity', 'F1 Score']:
            val_col = f'{col}_val'
            if val_col in table_df.columns:
                styler.apply(apply_gradient, val_col=val_col, cmap=cmap, subset=[col])
                styler.apply(highlight_best, val_col=val_col, mode='max', subset=[col])

        styler.hide(axis="index")

        def format_cell(v):
            if isinstance(v, str) and ' (' in v:
                return v.replace(' (', '<br>(')
            return v

        styler.format(format_cell)
        styler.set_properties(**{'text-align': 'center'})

        html_table = styler.to_html(table_id=f"results_{outcome}_{branch}_{model_name}", escape=False)

        # Optional delta table
        delta_candidates = [
            'Δ AUROC',
            'Δ AUPRC',
            'Δ Brier Score',
            'Δ Sensitivity',
            'Δ Specificity',
            'Δ F1 Score',
        ]
        delta_cols = ['Feature Set'] + [c for c in delta_candidates if c in table_df.columns]
        delta_html = ""
        if len(delta_cols) > 1:
            delta_df = table_df[delta_cols].copy()
            delta_styler = delta_df.style.hide(axis="index")
            delta_styler.set_properties(**{'text-align': 'center'})
            delta_styler.format(format_cell)

            # Heatmap only if CI excludes 0
            display_to_key = {col: f"delta_{col.split('Δ ')[-1].lower().replace(' ', '_')}" for col in delta_candidates}
            delta_styles = get_delta_style_info(table_df, display_to_key)

            def apply_delta_background(s, col_name):
                styles = []
                for idx in s.index:
                    bg = delta_styles.get(idx, {}).get(col_name, {}).get('bg')
                    styles.append(f'background-color: {bg}' if bg else '')
                return styles

            for col in delta_candidates:
                if col not in delta_df.columns:
                    continue
                delta_styler.apply(apply_delta_background, col_name=col, subset=[col])

            delta_html = delta_styler.to_html(
                table_id=f"delta_{outcome}_{branch}_{model_name}", escape=False
            )

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
            {('<h2>Δ vs Reference</h2>' + delta_html) if delta_html else ''}
            <div class="footer">
                Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | AKI Prediction Project
            </div>
        </body>
        </html>
        """

        with open(TABLES_DIR / f'results_{outcome}_{branch}_{model_name}.html', 'w') as f:
            f.write(full_html)

    print("Tables generated.")

def plot_curves(df: pd.DataFrame) -> None:
    """Generate ROC, PR, and calibration plots for each model grouping.

    Groups lacking both positive and negative examples are skipped to avoid
    metric calculation errors.
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
        if group['y_true'].nunique() < 2:
            logger.warning(
                "Skipping plots for %s/%s/%s because only one class is present.",
                outcome,
                branch,
                model_name,
            )
            continue

        valid_groups = {}
        for feature_set, model_group in group.groupby('feature_set'):
            if model_group['y_true'].nunique() < 2:
                logger.warning(
                    "Skipping feature set %s for %s/%s/%s because only one class is present.",
                    feature_set,
                    outcome,
                    branch,
                    model_name,
                )
                continue
            valid_groups[feature_set] = model_group

        if not valid_groups:
            logger.warning(
                "No feature sets with positive and negative cases for %s/%s/%s; skipping plots.",
                outcome,
                branch,
                model_name,
            )
            continue

        # 1. ROC Curve (Raw)
        plt.figure(figsize=(8, 6)) # Standard academic figure size

        auc_scores = []
        for feature_set, model_group in valid_groups.items():
            y_true = model_group['y_true']
            prob_col = 'y_prob_calibrated' if 'y_prob_calibrated' in model_group.columns else 'y_prob_raw'
            y_prob = model_group[prob_col]
            roc_auc = roc_auc_score(y_true, y_prob)
            auc_scores.append((feature_set, roc_auc))

        auc_scores.sort(key=lambda x: x[1], reverse=True)

        for feature_set, roc_auc in auc_scores:
            model_group = valid_groups[feature_set]
            y_true = model_group['y_true']
            prob_col = 'y_prob_calibrated' if 'y_prob_calibrated' in model_group.columns else 'y_prob_raw'
            y_prob = model_group[prob_col]
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

        auprc_scores = []
        for feature_set, model_group in valid_groups.items():
            y_true = model_group['y_true']
            prob_col = 'y_prob_calibrated' if 'y_prob_calibrated' in model_group.columns else 'y_prob_raw'
            y_prob = model_group[prob_col]
            pr_auc = average_precision_score(y_true, y_prob)
            auprc_scores.append((feature_set, pr_auc))

        auprc_scores.sort(key=lambda x: x[1], reverse=True)

        for feature_set, pr_auc in auprc_scores:
            model_group = valid_groups[feature_set]
            y_true = model_group['y_true']
            prob_col = 'y_prob_calibrated' if 'y_prob_calibrated' in model_group.columns else 'y_prob_raw'
            y_prob = model_group[prob_col]
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
        for feature_set, model_group in valid_groups.items():
            y_true = model_group['y_true']
            prob_col = 'y_prob_calibrated' if 'y_prob_calibrated' in model_group.columns else 'y_prob_raw'
            y_prob = model_group[prob_col]
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


def get_delta_style_info(df: pd.DataFrame, display_to_key: Dict[str, str]) -> Dict:
    """Background styling for delta tables; no color if CI crosses 0."""

    style_info: Dict[int, Dict[str, Dict[str, str]]] = {}
    cmap_pos = LinearSegmentedColormap.from_list("delta_pos", ["#ffffff", "#d1e7dd"])
    cmap_neg = LinearSegmentedColormap.from_list("delta_neg", ["#ffffff", "#f8d7da"])

    for idx in df.index:
        style_info[idx] = {}

    for display_col, metric_key in display_to_key.items():
        val_col = f"delta_{metric_key.split('delta_')[-1]}"
        lower_col = f"{val_col}_lower"
        upper_col = f"{val_col}_upper"
        if val_col not in df.columns:
            continue

        vals = df[val_col]
        lowers = df[lower_col] if lower_col in df.columns else pd.Series(index=df.index, data=np.nan)
        uppers = df[upper_col] if upper_col in df.columns else pd.Series(index=df.index, data=np.nan)

        mask = (~vals.isna()) & (~lowers.isna()) & (~uppers.isna()) & ~((lowers <= 0) & (uppers >= 0))
        if not mask.any():
            continue

        abs_vals = vals.abs()[mask]
        max_abs = abs_vals.max()
        if max_abs == 0:
            continue
        norm = plt.Normalize(0, max_abs)

        for idx in df.index:
            if not mask.get(idx, False):
                continue
            val = vals.loc[idx]
            color = cmap_pos(norm(abs(val))) if val >= 0 else cmap_neg(norm(abs(val)))
            style_info[idx][display_col] = {'bg': to_hex(color)}

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

        # Delta table (if available)
        delta_candidates = [
            'Δ AUROC',
            'Δ AUPRC',
            'Δ Brier Score',
            'Δ Sensitivity',
            'Δ Specificity',
            'Δ F1 Score',
        ]
        delta_cols = ['Feature Set'] + [c for c in delta_candidates if c in table_df.columns]
        if len(delta_cols) > 1:
            doc.add_paragraph('Δ vs Reference', style='Heading 2')
            delta_table_df = table_df[delta_cols]

            display_to_key = {col: f"delta_{col.split('Δ ')[-1].lower().replace(' ', '_')}" for col in delta_candidates}
            delta_styles = get_delta_style_info(table_df, display_to_key)

            delta_table = doc.add_table(rows=1, cols=len(delta_cols))
            delta_table.style = 'Table Grid'

            hdr = delta_table.rows[0].cells
            for i, col in enumerate(delta_cols):
                hdr[i].text = col
                hdr[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                hdr[i].paragraphs[0].runs[0].bold = True

            for _, row in delta_table_df.iterrows():
                row_cells = delta_table.add_row().cells
                for i, col in enumerate(delta_cols):
                    val = str(row[col])
                    if ' (' in val:
                        val = val.replace(' (', '\n(')
                    row_cells[i].text = val
                    row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                    bg = delta_styles.get(row.name, {}).get(col, {}).get('bg')
                    if bg:
                        set_cell_background(row_cells[i], bg)
            doc.add_paragraph()  # spacer
        
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

            # Delta table as separate page if available
            delta_candidates = [
                'Δ AUROC',
                'Δ AUPRC',
                'Δ Brier Score',
                'Δ Sensitivity',
                'Δ Specificity',
                'Δ F1 Score',
            ]
            delta_cols = ['Feature Set'] + [c for c in delta_candidates if c in table_df.columns]
            if len(delta_cols) > 1:
                delta_df = table_df[delta_cols]
                display_data = []
                for _, row in delta_df.iterrows():
                    new_row = []
                    for col in delta_cols:
                        val = str(row[col])
                        if ' (' in val:
                            val = val.replace(' (', '\n(')
                        new_row.append(val)
                    display_data.append(new_row)

                row_height = 0.6
                fig_height = len(delta_df) * row_height + 2

                fig, ax = plt.subplots(figsize=(12, fig_height))
                ax.axis('off')
                ax.set_title(
                    f"Δ vs Reference - Outcome: {outcome} | Branch: {branch} | Model: {model_name}",
                    fontsize=14,
                    fontweight='bold',
                    pad=20,
                )

                delta_table = ax.table(cellText=display_data, colLabels=delta_cols, loc='center', cellLoc='center')
                delta_table.auto_set_font_size(False)
                delta_table.set_fontsize(10)
                delta_table.scale(1, 2.5)

                display_to_key = {col: f"delta_{col.split('Δ ')[-1].lower().replace(' ', '_')}" for col in delta_candidates}
                delta_styles = get_delta_style_info(table_df, display_to_key)

                for (row, col), cell in delta_table.get_celld().items():
                    if row == 0:
                        cell.set_text_props(weight='bold')
                        cell.set_facecolor('#f0f0f0')
                    else:
                        df_idx = delta_df.index[row - 1]
                        col_name = delta_cols[col]
                        bg = delta_styles.get(df_idx, {}).get(col_name, {}).get('bg')
                        if bg:
                            cell.set_facecolor(bg)

                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
            
    print("PDF report generated.")

def main():
    """Deprecated entrypoint maintained for backwards compatibility."""

    raise SystemExit(
        "results_analysis now expects precomputed artifacts. "
        "Run reporting/make_report.py after generating metrics_summary.csv."
    )


if __name__ == "__main__":
    main()
