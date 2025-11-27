# AKI Prediction Project

This project implements a machine learning pipeline to predict Postoperative Acute Kidney Injury (AKI) using intraoperative physiological waveforms (ECG, PPG, EtCO2) from the VitalDB dataset.

## 📋 Project Overview
The pipeline consists of three main stages:
1.  **Cohort Selection**: Filtering patients based on clinical criteria and waveform availability.
2.  **Feature Extraction**: Extracting time-series features (Catch22) from high-frequency waveforms.
3.  **Modeling**: Training XGBoost classifiers to predict AKI.

## 🛠️ Prerequisites & Setup

### 1. Environment Setup
The project uses a Conda environment to manage dependencies.

```bash
# Create the environment from the provided file
conda env create -f environment.yml

# Activate the environment
conda activate aki_prediction_project
```

### 2. VitalDB Access
Ensure you have access to the VitalDB API. The `vitaldb` Python package is used to download waveforms on-demand.

## 🚀 Execution Guide

Run the pipeline steps in the following order. All commands should be run from the **project root directory** (`d:\Projects\aki_prediction_project`).

### Step 1: Configuration
**File**: `data_preparation/inputs.py`

This file contains all global settings. Open it and verify:
*   `INPUT_FILE`: Path to your raw clinical data CSV.
*   `MANDATORY_WAVEFORMS`: List of required VitalDB tracks (e.g., `['SNUADC/PLETH', 'SNUADC/ECG_II']`).
*   `TARGET_SR`: Sampling rate for analysis (default: 10 Hz).

### Step 2: Cohort Construction
**File**: `data_preparation/step_01_cohort_construction.py`

This script filters the raw clinical data and verifies waveform availability for each patient.

**Command:**
```bash
python -m data_preparation.step_01_cohort_construction
```

**Output:**
*   `data/processed/aki_pleth_ecg_co2_awp.csv`: The final cohort of patients who meet all clinical and data quality criteria.

### Step 3: Feature Extraction
**File**: `data_preparation/step_02_catch_22.py`

This script downloads waveforms for the cohort and extracts 22 canonical time-series features (Catch22) for each waveform.

**Command:**
```bash
python -m data_preparation.step_02_catch_22
```

**Output:**
*   `data/processed/aki_pleth_ecg_co2_awp_inf.csv`: A "long-format" CSV containing features for every patient-waveform pair.
*   *Note: This step can take a significant amount of time depending on your internet connection and CPU cores.*

### Step 4: Model Training & HPO
**File**: `notebooks/04_hpo_xgboost.ipynb`

Open this notebook to train the model. It performs the following:
1.  **Data Wrangling**: Pivots the long-format feature file into a wide format (one row per patient).
2.  **Preprocessing**: Imputes missing values and handles outliers.
3.  **Training**: Runs Hyperparameter Optimization (HPO) using Optuna to find the best XGBoost model.
4.  **Evaluation**: Reports AUC, Accuracy, and SHAP feature importance.

**To run:**
```bash
jupyter notebook notebooks/04_hpo_xgboost.ipynb
```

## 📂 Directory Structure

*   `data_preparation/`: Python scripts for the data pipeline.
    *   `inputs.py`: Global configuration.
    *   `step_01_cohort_construction.py`: Cohort filtering.
    *   `step_02_catch_22.py`: Feature extraction.
*   `notebooks/`: Jupyter notebooks for analysis and modeling.
    *   `04_hpo_xgboost.ipynb`: Main modeling notebook.
*   `data/`: Data storage (raw and processed).
*   `archive/`: Deprecated or legacy files.

## ⚠️ Common Issues
*   **ModuleNotFoundError**: Ensure you run scripts as modules (`python -m ...`) from the project root, not by running the file path directly (`python data_preparation/...`).
*   **VitalDB Connection**: If feature extraction fails, check your internet connection and VitalDB API access.
