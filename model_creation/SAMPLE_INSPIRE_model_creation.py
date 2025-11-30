#!/usr/bin/env python3
import os
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    balanced_accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix
)
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
TARGET = 'aki_boolean'
RANDOM_STATE = 42
USE_BOOTSTRAPPING = True  # Set to True to run the full 25-iteration cross-validation
N_BOOTSTRAP_ITERATIONS = 25

# --- I/O Configuration ---
BASE_DATA_DIR = '/home/server/Projects/data/AKI/'
RESULTS_DIR = '/home/server/Projects/data/AKI/results/'

# --- Dataset Configurations ---
datasets_to_run = [
    {'name': 'preop', 'path': os.path.join(BASE_DATA_DIR, 'tabular_preop.csv')},
    {'name': 'intraop', 'path': os.path.join(BASE_DATA_DIR, 'tabular_intraop.csv')},
    {'name': 'combined', 'path': os.path.join(BASE_DATA_DIR, 'tabular_combined.csv')},
]

# --- Dataset & Model Toggles ---
dataset_configs = {'preop': False, 'intraop': True, 'combined': True}
model_configs = {
    'log_reg': False, 'autogluon': True, 'xgb': False, 'rf': False,
    'svm': False, 'mlp': False, 'knn': False, 'asa_rule': False,
}

# --- Optimized Hyperparameters for Each Dataset ---
hpo_params_preop = {
    'log_reg': {'C': 0.19863},
    'xgb': {'n_estimators': 515, 'learning_rate': 0.01148, 'max_depth': 9, 'subsample': 0.7182,
            'colsample_bytree': 0.6793, 'gamma': 0.01291},
    'rf': {'n_estimators': 778, 'max_depth': 14, 'min_samples_split': 7, 'min_samples_leaf': 9, 'max_features': 'log2'},
    'svm': {'C': 0.002052},
    'knn': {'n_neighbors': 315},
    'mlp': {'lr': 0.001828, 'n_layers': 2, 'n_units': 69, 'dropout_rate': 0.3610832}
}
hpo_params_intraop = {
    'log_reg': {'C': 141.3782},
    'xgb': {'n_estimators': 675, 'learning_rate': 0.0105, 'max_depth': 6, 'subsample': 0.8110,
            'colsample_bytree': 0.6934, 'gamma': 2.3038},
    'rf': {'n_estimators': 309, 'max_depth': 15, 'min_samples_split': 5, 'min_samples_leaf': 10, 'max_features': 'sqrt'},
    'svm': {'C': 0.2005},
    'mlp': {'lr': 0.0027, 'n_layers': 1, 'n_units': 106, 'dropout_rate': 0.6151},
    'knn': {'n_neighbors': 324},
}
hpo_params_combined = {
    'log_reg': {'C': 0.0006},
    'xgb': {'n_estimators': 1110, 'learning_rate': 0.0172, 'max_depth': 9, 'subsample': 0.8281,
            'colsample_bytree': 0.7827, 'gamma': 1.7456},
    'rf': {'n_estimators': 597, 'max_depth': 17, 'min_samples_split': 10, 'min_samples_leaf': 7,
           'max_features': 'sqrt'},
    'svm': {'C': 0.0001},
    'mlp': {'lr': 0.0004, 'n_layers': 4, 'n_units': 114, 'dropout_rate': 0.4839},
    'knn': {'n_neighbors': 484},
}
all_hpo_params = {'preop': hpo_params_preop, 'intraop': hpo_params_intraop, 'combined': hpo_params_combined}

# Set seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def performance_dict(y_true, y_pred_binary, y_prob):
    """Calculates a comprehensive dictionary of performance metrics."""
    if len(np.unique(y_true)) < 2:
        tn, fp, fn, tp = (len(y_true), 0, 0, 0) if np.all(y_true == 0) else (0, 0, 0, len(y_true))
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

    return {
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'y_true': y_true,
        'y_pred_binary': y_pred_binary,
        'y_prob': y_prob,
    }


def log_performance(model_name, dataset_name, iteration, perf_dict):
    """Prints a concise log for the current run."""
    print(
        f"    Run {iteration}: "
        f"AUC: {perf_dict['roc_auc']:.4f}, "
        f"Bal. Acc: {perf_dict['balanced_accuracy']:.4f}, "
        f"Recall: {perf_dict['recall']:.4f}"
    )


def save_results(model_name, df_results, output_pkl):
    """Saves the results DataFrame to a pickle file, aggregating multiple runs."""
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df_collapsed = pd.DataFrame({col: [df_results[col].values] for col in df_results.columns})
    df_collapsed['model_name'] = model_name

    if os.path.exists(output_pkl):
        try:
            df_output = pd.read_pickle(output_pkl)
            if not df_output.empty:
                df_output = df_output[df_output['model_name'] != model_name]
            df_output = pd.concat([df_output, df_collapsed], ignore_index=True)
        except (EOFError, FileNotFoundError):
            df_output = df_collapsed
    else:
        df_output = df_collapsed
    df_output.to_pickle(output_pkl)
    print(f"Results for {model_name} saved to {output_pkl}")


# =============================================================================
# BOOTSTRAP SPLITTER CLASS
# =============================================================================
class BootstrapSplitter:
    def __init__(self, df, use_bootstrapping=False, n_iterations=25):
        self.df = df
        self.use_bootstrapping = use_bootstrapping
        self.n_iterations = n_iterations if use_bootstrapping else 1
        self.i = 0
        self.i_df = 0
        self.df_fifths = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n_iterations:
            raise StopIteration

        if self.use_bootstrapping:
            if self.i % 5 == 0:
                self.i_df = 0
                self.df_fifths = []
                df_remainder = self.df.copy()
                for remaining_fifths in range(5, 1, -1):
                    rest_df, fold_df = train_test_split(
                        df_remainder,
                        test_size=(1.0 / remaining_fifths),
                        random_state=RANDOM_STATE + (self.i // 5),
                        stratify=df_remainder[TARGET]
                    )
                    self.df_fifths.append(fold_df)
                    df_remainder = rest_df
                self.df_fifths.append(df_remainder)

            test_df = self.df_fifths[self.i_df]
            train_dfs = [df for j, df in enumerate(self.df_fifths) if j != self.i_df]
            train_df = pd.concat(train_dfs)
            self.i_df += 1

        else:  # Single run mode
            train_df, test_df = train_test_split(
                self.df, test_size=0.2, random_state=RANDOM_STATE, stratify=self.df[TARGET]
            )

        feature_cols = [col for col in self.df.columns if col not in ['op_id', TARGET]] # instead of dropping 'op_id' remember to exclude all outcome columns
        X_train, y_train = train_df[feature_cols].values, train_df[TARGET].values
        X_test, y_test = test_df[feature_cols].values, test_df[TARGET].values

        self.i += 1
        return (X_test, y_test, feature_cols), (X_train, y_train)


# =============================================================================
# PYTORCH MLP MODEL DEFINITION
# =============================================================================
class PyTorchMLP(nn.Module):
    def __init__(self, input_size, n_layers, n_units, dropout_rate):
        super(PyTorchMLP, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    for dataset_info in datasets_to_run:
        dataset_name = dataset_info['name']
        if not dataset_configs.get(dataset_name, False):
            print(f"\n{'=' * 25} SKIPPING DATASET: {dataset_name.upper()} {'=' * 25}")
            continue

        print(f"\n{'=' * 25} RUNNING MODELS FOR: {dataset_name.upper()} DATA {'=' * 25}")
        print(f"Current time: {time.ctime()}")

        try:
            df = pd.read_csv(dataset_info['path'])
        except FileNotFoundError:
            print(f"--> WARNING: Data file not found: {dataset_info['path']}. Skipping.")
            continue

        output_pkl = os.path.join(RESULTS_DIR, f"tabular_{dataset_name}_test.pkl")
        current_hpo_params = all_hpo_params.get(dataset_name, {})
        base_results_saved = False

        for model_key, should_run in model_configs.items():
            if not should_run:
                print(f"\nSkipping {model_key.upper()} for {dataset_name} data.")
                continue

            splitter = BootstrapSplitter(df, use_bootstrapping=USE_BOOTSTRAPPING, n_iterations=N_BOOTSTRAP_ITERATIONS)

            print(f"\n--- Running {model_key.upper()} for {dataset_name.upper()} data ---")
            df_results = pd.DataFrame()

            for i, (test_data, train_data) in enumerate(splitter, 1):
                X_test, y_test, feature_names = test_data
                X_train, y_train = train_data

                # Note: Scaling is done here for models that require it.
                # AutoGluon will use the raw (unscaled) X_train and X_test.
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = None
                y_pred, y_prob = None, None

                if model_key == 'log_reg':
                    params = current_hpo_params.get('log_reg', {})
                    model = LogisticRegression(
                        **params, class_weight='balanced', max_iter=10000, tol=0.01, random_state=RANDOM_STATE + i
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]

                elif model_key == 'autogluon':
                    try:
                        from autogluon.tabular import TabularPredictor
                        # Using un-scaled data as recommended for AutoGluon
                        train_df = pd.DataFrame(X_train, columns=feature_names)
                        train_df[TARGET] = y_train
                        test_df = pd.DataFrame(X_test, columns=feature_names)
                        
                        # MODIFIED: Moved sample_weight to the constructor
                        predictor = TabularPredictor(
                            label=TARGET,
                            eval_metric='balanced_accuracy',
                            sample_weight='balance_weight'
                        )
                        
                        # MODIFIED: Removed sample_weight from fit()
                        predictor.fit(
                            train_data=train_df,
                            time_limit=600,
                            presets='best_quality',
                            num_cpus=8
                        )
                        y_prob = predictor.predict_proba(test_df, as_pandas=False)[:, 1]
                        y_pred = predictor.predict(test_df).values
                    except ImportError:
                        print("AutoGluon not installed, skipping.")
                        continue

                elif model_key == 'rf':
                    params = current_hpo_params.get('rf', {})
                    model = RandomForestClassifier(
                        **params, class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE + i
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]

                elif model_key == 'xgb':
                    params = current_hpo_params.get('xgb', {})
                    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
                    model = xgb.XGBClassifier(**params, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE + i)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]

                elif model_key == 'svm':
                    params = current_hpo_params.get('svm', {})
                    model = LinearSVC(
                        **params, class_weight='balanced', dual='auto', random_state=RANDOM_STATE + i, max_iter=5000
                    )
                    model.fit(X_train_scaled, y_train)
                    y_scores = model.decision_function(X_test_scaled)
                    y_prob = 1 / (1 + np.exp(-y_scores))
                    y_pred = (y_scores > 0).astype(int)

                elif model_key == 'knn':
                    params = current_hpo_params.get('knn', {})
                    model = KNeighborsClassifier(**params, weights='distance', n_jobs=-1)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]

                elif model_key == 'mlp':
                    params = current_hpo_params.get('mlp', {})
                    lr = params.get('lr', 0.001)
                    n_layers = params.get('n_layers', 2)
                    n_units = params.get('n_units', 32)
                    dropout_rate = params.get('dropout_rate', 0.5)
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                    X_train_part, X_val_part, y_train_part, y_val_part = train_test_split(
                        X_train_scaled, y_train, test_size=0.2, random_state=RANDOM_STATE + i, stratify=y_train
                    )

                    X_train_tensor = torch.FloatTensor(X_train_part).to(device)
                    y_train_tensor = torch.FloatTensor(y_train_part).reshape(-1, 1).to(device)
                    X_val_tensor = torch.FloatTensor(X_val_part).to(device)

                    model = PyTorchMLP(X_train.shape[1], n_layers, n_units, dropout_rate).to(device)
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    pos_weight_val = np.sum(y_train_part == 0) / np.sum(y_train_part == 1) if np.sum(y_train_part == 1) > 0 else 1.0
                    pos_weight = torch.tensor([pos_weight_val], device=device)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                    best_val_auc, patience_counter, best_model_state = 0, 0, None
                    for epoch in range(5000):
                        model.train()
                        optimizer.zero_grad()
                        outputs = model(X_train_tensor)
                        loss = criterion(outputs, y_train_tensor)
                        loss.backward()
                        optimizer.step()

                        if (epoch + 1) % 10 == 0:
                            model.eval()
                            with torch.no_grad():
                                val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy().flatten()
                            current_val_auc = roc_auc_score(y_val_part, val_probs) if len(np.unique(y_val_part)) > 1 else 0.5
                            if current_val_auc > best_val_auc:
                                best_val_auc = current_val_auc
                                patience_counter = 0
                                best_model_state = copy.deepcopy(model.state_dict())
                            else:
                                patience_counter += 1
                            if patience_counter >= 20:
                                break

                    if best_model_state:
                        model.load_state_dict(best_model_state)
                    model.eval()
                    with torch.no_grad():
                        X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
                        y_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
                    y_pred = (y_prob >= 0.5).astype(int)

                elif model_key == 'asa_rule':
                    try:
                        asa_idx = feature_names.index('asa')
                        y_prob = (X_test[:, asa_idx] / 6.0).astype(float)
                        y_pred = (X_test[:, asa_idx] >= 4).astype(int)
                    except (ValueError, IndexError):
                        print("ASA feature not found, skipping ASA rule.")
                        continue

                perf = performance_dict(y_test, y_pred, y_prob)
                log_performance(model_key, dataset_name, i, perf)
                df_results = pd.concat([df_results, pd.DataFrame([perf])], ignore_index=True)

            if not df_results.empty:
                save_results(model_key, df_results, output_pkl)

                if not base_results_saved:
                    base_df = pd.DataFrame({'y_pred_binary': df_results['y_true']})
                    save_results('base', base_df, output_pkl)
                    base_results_saved = True

    print("\n\nScript finished for all configured datasets.")


if __name__ == "__main__":
    main()