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

## 🚀 Execution Guide (Refactored Pipeline)

**Recommended**: This new pipeline uses modular scripts for better reproducibility, data leakage prevention, and support for both full and windowed datasets.

### Advantages over Legacy Pipeline
*   **Reproducibility**: Modular Python scripts replace monolithic notebooks, making execution deterministic and easier to automate.
*   **Leakage Prevention**: The train/test split is performed **once** at the beginning of the preprocessing stage (`step_03`) and propagated to the final datasets. This guarantees that no patient from the test set is used to calculate training statistics (e.g., outlier thresholds).
*   **Dual Mode Support**: Automatically handles both **Full** (entire case) and **Windowed** (segmented) feature sets.

### Step 1: Configuration
**File**: `data_preparation/inputs.py`
Central configuration file. Defines input/output paths, mandatory waveforms/columns, and **AEON export settings** (e.g., padding policy, save formats). Verify `INPUT_FILE`, `MANDATORY_WAVEFORMS`, and `TARGET_SR`.

### Step 2: Cohort Construction
**File**: `data_preparation/step_01_cohort_construction.py`
Filters the raw dataset to create a valid cohort.
*   **Logic**: Selects patients who have all `MANDATORY_WAVEFORMS` and `MANDATORY_COLUMNS`.
*   **Output**: A cohort CSV containing valid `caseid`s.
```bash
python -m data_preparation.step_01_cohort_construction
```

### Step 3: Feature Extraction
**File**: `data_preparation/step_02_catch_22.py`
Extracts 22 time-series features (Catch22) from each waveform channel.
*   **Technical Details**:
    *   Resamples waveforms to `TARGET_SR` (default 10Hz).
    *   **Full Mode**: Extracts features from the entire case duration.
    *   **Windowed Mode**: Segments waveforms into windows (defined by `WIN_SEC`, `SLIDE_SEC`) and extracts features per window.
    *   Handles multiprocessing for efficiency.
```bash
python -m data_preparation.step_02_catch_22
```

### Step 4: Preoperative Data Prep & Split
**File**: `data_preparation/step_03_preop_prep.py`
Processes clinical data and defines the **stratified train/test split**.
*   **Technical Details**:
    *   **Splitting**: Performs an 80/20 stratified split based on the outcome. This `split_group` is saved and used downstream to prevent leakage.
    *   **Outlier Handling**: Calculates percentiles (0.5%, 99.5%) **only on the training set** and applies them to clip/impute outliers in both train and test sets.
    *   **Imputation**: Fills missing values with -99.
```bash
python data_preparation/step_03_preop_prep.py
```

### Step 5: Intraoperative Data Prep
**File**: `data_preparation/step_04_intraop_prep.py`
Processes waveform features (pivoting, imputation) for both full and windowed modes.
*   **Technical Details**:
    *   **Pivoting**: Converts long-format Catch22 results into a wide format (one row per case/window).
    *   **Flattening**: Renames columns to `{waveform}_{feature}` (e.g., `SNUADC_PLETH_DN_HistogramMode_5`).
    *   **Imputation**: Fills missing waveform features with 0.
```bash
python data_preparation/step_04_intraop_prep.py
```

### Step 6: Data Merge
**File**: `data_preparation/step_05_data_merge.py`
Combines preop and intraop data into master datasets.
*   **Technical Details**:
    *   Merges intraop features with processed preop data on `caseid`.
    *   **Integrity Check**: Ensures every row has a valid `split_group` from Step 3.
    *   Outputs final wide CSVs ready for training.
```bash
python data_preparation/step_05_data_merge.py
```

### Step 7: Model Training
**File**: `notebooks/EXPERIMENTAL_train_model.py`
Train XGBoost models using the master datasets.
*   **Technical Details**:
    *   **XGBoost**: Uses histogram-based tree method for speed.
    *   **HPO**: Integrates **Optuna** for Bayesian hyperparameter optimization.
    *   **Explainability**: Generates SHAP summary plots (bar and dot) to interpret model feature importance.
    *   **Metrics**: Reports AUROC, AUPRC, F1, Sensitivity, and Specificity.

```bash
# Train on Full Features
python notebooks/EXPERIMENTAL_train_model.py --dataset full --features all

# Train on Windowed Features
python notebooks/EXPERIMENTAL_train_model.py --dataset windowed --features all
```

---

## 🕰️ Legacy Execution Guide (Notebook Based)

Use this approach if you need to replicate older results. Note that this method has potential data leakage risks due to inconsistent splitting between notebooks.

### Step 1-3: Same as above (Cohort & Feature Extraction)

### Step 3: Data Wrangling, Preoperative data extraction and preprocessing
**File**: `notebooks/03_tabular_data_prep.ipynb`

Open this notebook to prepare data for the next step.
*   **Technical Details**:
    *   Performs data transformation (pivots from long to wide format) for the intraoperative data.
    *   Extracts and cleans preoperative clinical data.
    *   **Note**: This legacy method may not enforce the strict train/test split boundaries used in the new pipeline. 

### Step 4: Model Training & HPO
**File**: `notebooks/04_hpo_xgboost.ipynb`

Open this notebook to train the model.
*   **Technical Details**:
    *   **Monolithic Workflow**: Performs data wrangling, preprocessing, HPO, and training in a single notebook execution.
    *   **HPO**: Uses Optuna for hyperparameter tuning.
    *   **Visualization**: Includes code for plotting ROC curves and SHAP values inline.

**To run:**
```bash
jupyter notebook notebooks/04_hpo_xgboost.ipynb
```

## 📂 Directory Structure

*   `data_preparation/`: Python scripts for the data pipeline.
    *   `inputs.py`: Global configuration and AEON export settings.
    *   `step_01_cohort_construction.py`: Cohort filtering.
    *   `step_02_catch_22.py`: Feature extraction.
    *   `step_03_preop_prep.py`: Preop processing & splitting (New).
    *   `step_04_intraop_prep.py`: Intraop processing (New).
    *   `step_05_data_merge.py`: Data merging (New).
    *   `aeon_io.py`: **(New)** Helper for exporting data to AEON formats (Nested, 3D NumPy) with padding support.
    *   `waveform_processing.py`: Helper functions for signal processing.
    *   `EXPERIMENTAL_xgboost_hpo.py`: Alternative script for XGBoost HPO experiments.
*   `notebooks/`: Jupyter notebooks and training scripts.
    *   `EXPERIMENTAL_train_model.py`: Modular training script (New).
    *   `03_tabular_data_prep.ipynb`: Legacy data transformation + preoperative data extraction notebook.
    *   `04_hpo_xgboost.ipynb`: Legacy modeling notebook.
*   `data/`: Data storage (raw and processed).
