#!/usr/bin/env python3
import os
import time
import copy
import numpy as np
import pandas as pd
import optuna

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
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
TARGET_IS_BOOLEAN = True
RANDOM_STATE = 42
N_TRIALS = 50 # Number of HPO trials to run for each model
RESULTS_FILE_PATH = '/home/server/Projects/data/AKI/results/tabular_hpo_results.txt'

# --- Dataset Configurations ---
datasets_to_run = [
    {'name': 'preop', 'path': '/home/server/Projects/data/AKI/tabular_preop.csv'},
    {'name': 'intraop', 'path': '/home/server/Projects/data/AKI/tabular_intraop.csv'},
    {'name': 'combined', 'path': '/home/server/Projects/data/AKI/tabular_combined.csv'},
]

# --- Dataset HPO Toggles ---
dataset_hpo_configs = {
    'preop': True,
    'intraop': True,
    'combined': True,
}

# --- Universal Model HPO Toggles ---
hpo_configs = {
    'log_reg': True,
    'xgb': True,
    'rf': True,
    'svm': True,
    'pytorch_mlp': True,
    'knn': True,
}

# =============================================================================
# HPO HELPER FUNCTIONS & MODEL DEFINITIONS
# =============================================================================

def check_search_space_boundaries(study, search_space):
    """Checks if the best parameters are at the boundaries of the search space."""
    warnings = []
    best_params = study.best_params
    for param, value in best_params.items():
        if param not in search_space:
            continue
        min_val, max_val = search_space[param]
        if isinstance(value, (int, float)) and (np.isclose(value, min_val) or np.isclose(value, max_val)):
            warnings.append(
                f"  - WARNING for {study.study_name}: Best value for '{param}' ({value}) is at the boundary "
                f"of its search space [{min_val}, {max_val}]. Consider expanding the range."
            )
    if warnings:
        print("\n" + "="*20 + " BOUNDARY WARNINGS " + "="*20)
        for warning in warnings:
            print(warning)
        print("="*63 + "\n")

search_spaces = {
    'log_reg': {'C': (1e-4, 1e4)},
    'xgb': {'n_estimators': (200, 2000), 'learning_rate': (0.01, 0.3), 'max_depth': (3, 10), 'subsample': (0.6, 1.0), 'colsample_bytree': (0.6, 1.0), 'gamma': (0, 5)},
    'rf': {'n_estimators': (100, 1000), 'max_depth': (5, 50), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 20)},
    'svm': {'C': (1e-4, 1e4)},
    'knn': {'n_neighbors': (5, 500)},
    'pytorch_mlp': {'lr': (1e-5, 1e-1), 'n_layers': (1, 4), 'n_units': (4, 128), 'dropout_rate': (0.2, 0.7)}
}

class PyTorchMLP(nn.Module):
    def __init__(self, input_size, n_layers, n_units, dropout_rate):
        super(PyTorchMLP, self).__init__()
        layers = []
        in_features = input_size
        for i in range(n_layers):
            layers.append(nn.Linear(in_features, n_units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = n_units
        layers.append(nn.Linear(in_features, 1))
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

def save_hpo_results_to_file(filepath, results_dict):
    """
    Formats and saves the final HPO results to a text file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        f.write("# Hyperparameter Optimization Results\n")
        f.write(f"# Generated on: {time.ctime()}\n\n")
        f.write("# --- COPY-PASTE THE DICTIONARIES BELOW INTO THE MAIN SCRIPT ---\n")
        
        for dataset_name, best_params_dict in results_dict.items():
            if not best_params_dict: continue
            f.write(f"\nhpo_params_{dataset_name} = {{\n")
            for model_name, params in best_params_dict.items():
                formatted_params = {k: f"{v:.6f}" if isinstance(v, float) else v for k, v in params.items()}
                f.write(f"    '{model_name}': {formatted_params},\n")
            f.write("}\n")
    print(f"\nFinal HPO results saved to: {filepath}")

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    """
    Main function to run the HPO process.
    """
    # Set seed for reproducibility
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    all_hpo_results = {}

    for dataset_info in datasets_to_run:
        dataset_name = dataset_info['name']
        
        if not dataset_hpo_configs.get(dataset_name, False):
            print(f"\n{'='*25} SKIPPING HPO FOR: {dataset_name.upper()} DATA {'='*25}")
            continue
            
        file_path = dataset_info['path']
        
        print(f"\n{'='*25} STARTING HPO FOR: {dataset_name.upper()} DATA {'='*25}")
        print(f"Current time: {time.ctime()}")

        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"--> WARNING: Data file not found at: {file_path}. Skipping this dataset.")
            continue

        train_val_df, _ = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET])
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=RANDOM_STATE, stratify=train_val_df[TARGET])
        
        feature_cols = [col for col in df.columns if col not in ['op_id', TARGET, f"{TARGET}_boolean", f"{TARGET}_positive"]]
        X_train, y_train = train_df[feature_cols].values, train_df[TARGET].values
        X_val, y_val = val_df[feature_cols].values, val_df[TARGET].values
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        dataset_best_params = {}

        def get_objective_function(model_name):
            def objective_log_reg(trial):
                log_reg_c = trial.suggest_float("C", *search_spaces['log_reg']['C'], log=True)
                model = LogisticRegression(C=log_reg_c, penalty='l2', solver='lbfgs', tol=0.001, max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')
                model.fit(X_train_scaled, y_train)
                return roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])

            def objective_xgb(trial):
                params = {
                    'objective': 'binary:logistic', 'eval_metric': 'logloss', 'device': 'cuda', 'random_state': RANDOM_STATE,
                    'n_estimators': trial.suggest_int('n_estimators', *search_spaces['xgb']['n_estimators']),
                    'learning_rate': trial.suggest_float('learning_rate', *search_spaces['xgb']['learning_rate']),
                    'max_depth': trial.suggest_int('max_depth', *search_spaces['xgb']['max_depth']),
                    'subsample': trial.suggest_float('subsample', *search_spaces['xgb']['subsample']),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', *search_spaces['xgb']['colsample_bytree']),
                    'gamma': trial.suggest_float('gamma', *search_spaces['xgb']['gamma']),
                    'scale_pos_weight': np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train_scaled, y_train, verbose=False)
                return roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])
            
            def objective_rf(trial):
                params = {
                    'n_jobs': -1, 'random_state': RANDOM_STATE, 'class_weight': 'balanced',
                    'n_estimators': trial.suggest_int('n_estimators', *search_spaces['rf']['n_estimators']),
                    'max_depth': trial.suggest_int('max_depth', *search_spaces['rf']['max_depth'], log=True),
                    'min_samples_split': trial.suggest_int('min_samples_split', *search_spaces['rf']['min_samples_split']),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', *search_spaces['rf']['min_samples_leaf']),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                }
                model = RandomForestClassifier(**params)
                model.fit(X_train_scaled, y_train)
                return roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])

            def objective_svm(trial):
                svm_c = trial.suggest_float("C", *search_spaces['svm']['C'], log=True)
                model = LinearSVC(C=svm_c, class_weight='balanced', random_state=RANDOM_STATE, dual='auto', max_iter=5000)
                model.fit(X_train_scaled, y_train)
                return roc_auc_score(y_val, model.decision_function(X_val_scaled))

            def objective_knn(trial):
                n_neighbors = trial.suggest_int('n_neighbors', *search_spaces['knn']['n_neighbors'])
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', n_jobs=-1)
                model.fit(X_train_scaled, y_train)
                return roc_auc_score(y_val, model.predict_proba(X_val_scaled)[:, 1])

            def objective_pytorch_mlp(trial):
                lr = trial.suggest_float('lr', *search_spaces['pytorch_mlp']['lr'], log=True)
                n_layers = trial.suggest_int('n_layers', *search_spaces['pytorch_mlp']['n_layers'])
                n_units = trial.suggest_int('n_units', *search_spaces['pytorch_mlp']['n_units'])
                dropout_rate = trial.suggest_float('dropout_rate', *search_spaces['pytorch_mlp']['dropout_rate'])
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
                y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
                X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
                
                model = PyTorchMLP(X_train_scaled.shape[1], n_layers, n_units, dropout_rate).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                pos_weight = torch.tensor([np.sum(y_train == 0) / np.sum(y_train == 1)], device=device)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                for epoch in range(100):
                    model.train()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                return roc_auc_score(y_val, val_probs)

            return locals().get(f"objective_{model_name}")

        for model_name, should_run in hpo_configs.items():
            if should_run:
                print(f"\nOptimizing {model_name.upper()} for {dataset_name.upper()} data...")
                objective_func = get_objective_function(model_name)
                if objective_func:
                    study = optuna.create_study(direction='maximize', study_name=f"{dataset_name}_{model_name}")
                    study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)
                    dataset_best_params[model_name] = study.best_params
                    print(f"Best trial for {model_name.upper()}: AUC = {study.best_value:.4f}")
                    check_search_space_boundaries(study, search_spaces.get(model_name, {}))
                else:
                    print(f"Objective function for {model_name} not found.")
        
        all_hpo_results[dataset_name] = dataset_best_params
    
    # Save results to file after all datasets and models are processed
    save_hpo_results_to_file(RESULTS_FILE_PATH, all_hpo_results)

if __name__ == "__main__":
    main()