import time
import logging
import sys
import ast
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from xgboost.callback import EarlyStopping
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Import inputs from the project
from data_preparation.inputs import (
    PROJECT_ROOT,
    COHORT_FILE,
    CATCH_22_FILE,
    WIDE_FEATURES_FILE,
    RESULTS_DIR,
    OUTCOME as TARGET_COLUMN
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.float_format', lambda x: '%.4f' % x)

# --- Configuration ---
RANDOM_STATE = 42
N_TRIALS = 100
N_SPLITS = 5
USE_HPO_CACHING = True
FORCE_HPO_RERUN = False

# --- Feature Definitions (from Notebook) ---
PREOP_FEATURES_TO_SELECT = [
    'caseid', TARGET_COLUMN, 'age', 'sex', 'emop', 'department', 'bmi', 'approach',
    'preop_htn', 'preop_dm', 'preop_ecg', 'preop_pft', 'preop_hb', 'preop_plt',
    'preop_pt', 'preop_aptt', 'preop_na', 'preop_k', 'preop_gluc', 'preop_alb',
    'preop_ast', 'preop_alt', 'preop_bun', 'preop_cr', 'preop_hco3'
]

CONTINUOUS_COLS = [
    'age', 'bmi', 'preop_hb', 'preop_plt', 'preop_pt', 'preop_aptt', 'preop_na',
    'preop_k', 'preop_gluc', 'preop_alb', 'preop_ast', 'preop_alt',
    'preop_bun', 'preop_cr', 'preop_hco3'
]

CATEGORICAL_COLS = [
    'sex', 'emop', 'department', 'approach', 'preop_htn', 'preop_dm',
    'preop_ecg', 'preop_pft'
]

WAVEFORM_PREFIXES = ['SNUADC_PLETH', 'SNUADC_ECG_II', 'Primus_CO2', 'Primus_AWP']

TRAINING_MODE = {
    'pleth_only': {'waveforms': ['SNUADC_PLETH'], 'preop_features': False},
    'ecg_only': {'waveforms': ['SNUADC_ECG_II'], 'preop_features': False},
    'co2_only': {'waveforms': ['Primus_CO2'], 'preop_features': False},
    'awp_only': {'waveforms': ['Primus_AWP'], 'preop_features': False},
    'all_waveforms': {'waveforms': None, 'preop_features': False},
    'baseline_only': {'waveforms': [], 'preop_features': True},
    'baseline_and_pleth': {'waveforms': ['SNUADC_PLETH'], 'preop_features': True},
    'baseline_and_all_waveforms': {'waveforms': None, 'preop_features': True}
}

# --- Part 1: Data Wrangling ---

def handle_outliers(df, train_df, continuous_cols):
    df_processed = df.copy()
    for col in continuous_cols:
        if col in df_processed.columns:
            train_col_numeric = pd.to_numeric(train_df[col], errors='coerce')
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            train_col_numeric.dropna(inplace=True)
            if train_col_numeric.empty:
                continue

            low_p_0_5, low_p_1, low_p_5 = np.percentile(train_col_numeric, [0.5, 1, 5])
            high_p_95, high_p_99_5 = np.percentile(train_col_numeric, [95, 99.5])
            
            low_outlier_indices = df_processed[df_processed[col] < low_p_1].index
            high_outlier_indices = df_processed[df_processed[col] > high_p_99_5].index
            
            low_replacements = np.random.uniform(low_p_0_5, low_p_5, size=len(low_outlier_indices))
            high_replacements = np.random.uniform(high_p_95, high_p_99_5, size=len(high_outlier_indices))
            
            df_processed.loc[low_outlier_indices, col] = low_replacements
            df_processed.loc[high_outlier_indices, col] = high_replacements
    return df_processed

def wrangle_data():
    logging.info("--- Starting Data Wrangling ---")
    
    # Load Cohort
    if not COHORT_FILE.exists():
        logging.error(f"Cohort file not found: {COHORT_FILE}")
        return False
    cohort_df = pd.read_csv(COHORT_FILE)
    
    # Select Preop
    try:
        preop_df = cohort_df[PREOP_FEATURES_TO_SELECT].copy()
    except KeyError as e:
        logging.error(f"Missing columns in cohort file: {e}")
        return False

    # Split for processing
    X_preop = preop_df.drop(columns=[TARGET_COLUMN, 'caseid'])
    y_preop = preop_df[TARGET_COLUMN]
    caseid_series = preop_df['caseid']

    X_train, X_test, y_train, y_test, caseid_train, caseid_test = train_test_split(
        X_preop, y_preop, caseid_series, test_size=0.2, random_state=RANDOM_STATE, stratify=y_preop
    )

    # Categorical Processing
    dept_counts = X_train['department'].value_counts()
    depts_to_merge = dept_counts[dept_counts < 30].index.tolist()
    if depts_to_merge:
        X_train['department'] = X_train['department'].replace(depts_to_merge, 'other')
        X_test['department'] = X_test['department'].replace(depts_to_merge, 'other')

    X_train_dummies = pd.get_dummies(X_train[CATEGORICAL_COLS], drop_first=True, dtype=int)
    X_test_dummies = pd.get_dummies(X_test[CATEGORICAL_COLS], drop_first=True, dtype=int)
    
    X_train_aligned, X_test_aligned = X_train_dummies.align(X_test_dummies, join='left', axis=1, fill_value=0)
    
    X_train = X_train.drop(columns=CATEGORICAL_COLS)
    X_test = X_test.drop(columns=CATEGORICAL_COLS)
    X_train = pd.concat([X_train, X_train_aligned], axis=1)
    X_test = pd.concat([X_test, X_test_aligned], axis=1)

    # Outliers & Imputation
    X_train_cleaned = handle_outliers(X_train, X_train, CONTINUOUS_COLS)
    X_test_cleaned = handle_outliers(X_test, X_train, CONTINUOUS_COLS)
    
    X_train_imputed = X_train_cleaned.fillna(-99)
    X_test_imputed = X_test_cleaned.fillna(-99)

    # Recombine
    X_preop_processed = pd.concat([X_train_imputed, X_test_imputed])
    preop_final_df = pd.DataFrame({
        'caseid': pd.concat([caseid_train, caseid_test]),
        TARGET_COLUMN: pd.concat([y_train, y_test])
    }).reset_index(drop=True)
    X_preop_processed = X_preop_processed.reset_index(drop=True)
    preop_final_df = pd.concat([preop_final_df, X_preop_processed], axis=1)

    # Waveforms
    if not CATCH_22_FILE.exists():
        logging.error(f"Waveform file not found: {CATCH_22_FILE}")
        return False
    long_df = pd.read_csv(CATCH_22_FILE)
    
    id_cols = ['caseid', TARGET_COLUMN]
    pivot_col = 'waveform'
    feature_cols = [c for c in long_df.columns if c not in id_cols + [pivot_col]]
    
    waveform_wide_df = long_df.pivot_table(index=id_cols, columns=pivot_col, values=feature_cols)
    new_cols = []
    for feature_name, waveform_name in waveform_wide_df.columns:
        clean_waveform = waveform_name.replace('/', '_')
        new_cols.append(f"{clean_waveform}_{feature_name}")
    waveform_wide_df.columns = new_cols
    waveform_wide_df = waveform_wide_df.reset_index()
    waveform_wide_df.fillna(0, inplace=True)

    # Merge
    master_df = pd.merge(waveform_wide_df, preop_final_df, on=['caseid', TARGET_COLUMN], how='left')
    master_df.fillna(-99, inplace=True)
    
    WIDE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_csv(WIDE_FEATURES_FILE, index=False)
    logging.info(f"Wrangling complete. Saved to {WIDE_FEATURES_FILE}")
    return True

# --- Part 2: Training & HPO ---

def parse_hpo_params(filepath: Path) -> dict:
    with open(filepath, 'r') as f:
        content = f.read()
    dict_start = content.find('{')
    dict_end = content.rfind('}') + 1
    if dict_start == -1 or dict_end == -1:
        raise ValueError(f"Could not find dictionary in {filepath}")
    return ast.literal_eval(content[dict_start:dict_end])

def write_hpo_params(filepath: Path, params: dict, best_value: float):
    output_string_list = ["hpo_params_xgboost = {"]
    for k, v in params.items():
        output_string_list.append(f"    '{k}': {v!r},")
    output_string_list.append("}")
    with open(filepath, "w") as f:
        f.write(f"# Hyperparameters saved on {time.ctime()}\n")
        f.write(f"# Best CV AUROC from HPO: {best_value:.4f}\n")
        f.write("\n".join(output_string_list))

def objective(trial, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    auroc_scores = []
    neg, pos = np.bincount(y)
    spw = np.sqrt(neg / pos) if pos > 0 else 1.0

    params = {
        "objective": "binary:logistic", "booster": "gbtree", "eval_metric": "auc",
        "tree_method": "hist", "n_jobs": -1, "random_state": RANDOM_STATE, "verbosity": 0,
        "scale_pos_weight": spw,
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "callbacks": [EarlyStopping(rounds=50)],
    }

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        y_hat = model.predict_proba(X_val_s)[:, 1]
        auroc_scores.append(roc_auc_score(y_val, y_hat))
        
    return np.mean(auroc_scores)

def run_hpo_and_evaluate(mode_to_run: str):
    config = TRAINING_MODE[mode_to_run]
    mode_results_dir = RESULTS_DIR / f"xgboost_{mode_to_run}"
    mode_results_dir.mkdir(parents=True, exist_ok=True)
    hpo_file = mode_results_dir / "hpo_best_params.txt"
    
    logging.info(f"--- 🚀 Starting Run For Mode: {mode_to_run} ---")
    
    df = pd.read_csv(WIDE_FEATURES_FILE)
    
    # Feature Selection
    all_available_cols = [c for c in df.columns if c not in ['caseid', TARGET_COLUMN]]
    all_preop_cols = [c for c in all_available_cols if c in CONTINUOUS_COLS]
    all_waveform_cols = [c for c in all_available_cols if any(c.startswith(p) for p in WAVEFORM_PREFIXES)]
    
    selected_cols = []
    if config.get('preop_features', False):
        selected_cols.extend(all_preop_cols)
    
    selected_waveforms = config.get('waveforms')
    if selected_waveforms is None:
        selected_cols.extend(all_waveform_cols)
    elif selected_waveforms:
        for prefix in selected_waveforms:
            selected_cols.extend([c for c in all_waveform_cols if c.startswith(prefix)])
            
    if not selected_cols:
        logging.error(f"No features selected for mode {mode_to_run}")
        return

    X = df[selected_cols]
    y = df[TARGET_COLUMN]
    
    # Split
    X_train_main, X_test, y_train_main, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    # HPO
    best_params = None
    if USE_HPO_CACHING and not FORCE_HPO_RERUN and hpo_file.exists():
        try:
            best_params = parse_hpo_params(hpo_file)
            logging.info("Loaded cached HPO params")
        except Exception:
            pass
            
    if best_params is None:
        logging.info("Starting HPO...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, X_train_main, y_train_main), n_trials=N_TRIALS)
        best_params = dict(study.best_params)
        write_hpo_params(hpo_file, best_params, study.best_value)
        
    # Final Training
    logging.info("Training final model...")
    neg, pos = np.bincount(y_train_main)
    best_params["scale_pos_weight"] = np.sqrt(neg / pos) if pos > 0 else 1.0
    
    final_params = {
        "objective": "binary:logistic", "booster": "gbtree", "tree_method": "hist",
        "n_jobs": -1, "random_state": RANDOM_STATE, **best_params
    }
    
    scaler = StandardScaler()
    X_train_main_s = scaler.fit_transform(X_train_main)
    X_test_s = scaler.transform(X_test)
    
    final_model = xgb.XGBClassifier(**final_params)
    final_model.fit(X_train_main_s, y_train_main)
    
    y_test_pred_proba = final_model.predict_proba(X_test_s)[:, 1]
    
    # Evaluation
    thresholds = np.arange(0.01, 1.0, 0.01)
    f1_scores = [f1_score(y_test, y_test_pred_proba >= t) for t in thresholds]
    best_threshold = thresholds[np.argmax(f1_scores)]
    y_test_pred_class = (y_test_pred_proba >= best_threshold).astype(int)
    
    metrics = {
        'AUROC': roc_auc_score(y_test, y_test_pred_proba),
        'AUPRC': average_precision_score(y_test, y_test_pred_proba),
        'Accuracy': accuracy_score(y_test, y_test_pred_class),
        'Precision': precision_score(y_test, y_test_pred_class, zero_division=0),
        'Sensitivity (Recall)': recall_score(y_test, y_test_pred_class),
        'F1-Score (Maximized)': f1_score(y_test, y_test_pred_class),
        'Specificity': confusion_matrix(y_test, y_test_pred_class).ravel()[0] / (confusion_matrix(y_test, y_test_pred_class).ravel()[0] + confusion_matrix(y_test, y_test_pred_class).ravel()[1])
    }
    
    pd.DataFrame(metrics.items(), columns=['Metric', 'Score']).to_csv(mode_results_dir / "final_metrics.csv", index=False)
    
    # SHAP
    logging.info("Calculating SHAP...")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_s)
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_s, feature_names=X.columns, plot_type="bar", show=False, max_display=50)
    plt.savefig(mode_results_dir / "shap_summary_bar.png", bbox_inches='tight')
    plt.close()
    
    plt.figure()
    shap.summary_plot(shap_values, X_test_s, feature_names=X.columns, plot_type="dot", show=False, max_display=50)
    plt.savefig(mode_results_dir / "shap_summary_dot.png", bbox_inches='tight')
    plt.close()
    
    logging.info(f"Completed mode: {mode_to_run}")

def main():
    if not WIDE_FEATURES_FILE.exists():
        logging.info("Wide features file not found. Running wrangler...")
        if not wrangle_data():
            logging.error("Wrangling failed.")
            return

    logging.info("Starting batch run for all modes...")
    for mode in TRAINING_MODE:
        try:
            run_hpo_and_evaluate(mode)
        except Exception as e:
            logging.error(f"Failed mode {mode}: {e}")

if __name__ == "__main__":
    main()
