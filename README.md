# AKI Prediction Project

This project implements a machine learning pipeline to predict Postoperative Acute Kidney Injury (AKI) using intraoperative physiological waveforms (ECG, PPG, EtCO2) from the VitalDB dataset.

## 📋 Project Overview
The pipeline consists of three main stages:
1.  **Cohort Selection**: Filtering patients based on clinical criteria and waveform availability.
2.  **Feature Extraction**: Extracting time-series features (Catch22) from high-frequency waveforms.
3.  **Modeling**: Training XGBoost classifiers to predict AKI (Primary) and secondary outcomes (Mortality, ICU Admission, Prolonged LOS).

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
*   **Output**: A cohort CSV containing valid `caseid`s and the following derived outcomes:
    *   `aki_label`: Primary outcome (KDIGO AKI).
    *   `y_inhosp_mortality`: In-hospital mortality.
    *   `y_icu_admit`: ICU admission (>0 days).
    *   `y_prolonged_los_postop`: Prolonged postoperative LOS (>= 75th percentile).
    *   `y_severe_aki`: Severe AKI (KDIGO Stage 2 or 3).
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

#### 1. Variable Selection & Derivation
We extract a comprehensive set of preoperative variables from `clinical_data.csv` and `lab_data.csv`:

*   **Demographics**: Age, Sex, Height, Weight, BMI.
*   **Surgical Context**: Emergency Operation (`emop`), Department, Approach, ASA Class, Operation Type (`optype`), Anesthesia Type (`ane_type`), Position.
*   **Comorbidities**: Hypertension (`preop_htn`), Diabetes (`preop_dm`), ECG abnormalities (`preop_ecg`), Pulmonary Function Test (`preop_pft`).
*   **Labs (Clinical Table)**: Hb, Platelets, PT (INR), aPTT, Na, K, Glucose, Albumin, AST, ALT, BUN, Creatinine, Bicarbonate.
*   **ABG (Clinical Table)**: pH, Base Excess (`preop_be`), PaO2, PaCO2, SaO2.
*   **Labs (Lab Table)**: Last preoperative value (within 30 days) for:
    *   `preop_wbc`: White Blood Cell count.
    *   `preop_gfr`: eGFR (from lab system).
    *   `preop_crp`: C-Reactive Protein.
    *   `preop_lac`: Lactate.
*   **Derived Features**:
    *   `preop_los_days`: Preoperative Length of Stay (days).
    *   `inpatient_preop`: Binary flag for inpatient admission prior to surgery.
    *   `preop_egfr_ckdepi`: eGFR calculated using CKD-EPI 2009 equation (race-free).
    *   **Clinical Flags** (Binary): `bun_high` (>27), `hypoalbuminemia` (<3.5), `preop_anemia` (Sex-specific Hb thresholds), `hyponatremia` (<135), `metabolic_acidosis` (HCO3 < 22 or BE < -2), `hypercapnia` (PaCO2 > 45), `hypoxemia` (PaO2 < 80 or SaO2 < 95).

#### 2. Processing Steps
*   **Splitting**: Performs an 80/20 stratified split based on the outcome. This `split_group` is saved and used downstream to prevent leakage.
*   **Outlier Handling**: Calculates percentiles (0.5%, 99.5%) **only on the training set** and applies them to clip/impute outliers in both train and test sets.
*   **Imputation**: Fills missing values with -99.
*   **Encoding**: Categorical variables are one-hot encoded. Rare categories (<30 occurrences) in `department` are merged into 'other'.

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

### Step 7: Model Training & Evaluation
**Directory**: `model_creation/`

We have refactored the modeling pipeline into two robust scripts: `run_hpo.py` and `train_evaluate.py`.

#### 1. Hyperparameter Optimization (HPO)
**File**: `model_creation/run_hpo.py`
### 6. Hyperparameter Optimization (HPO)
Run the HPO script to find the best hyperparameters for a specific configuration.
*   **Optimization Metric**: AUPRC (Area Under the Precision-Recall Curve).
*   **Trials**: Defaults to 100 trials.
*   **Output**: Saves best parameters to `results/params/{outcome}/{branch}/{feature_set}.json`.

```bash
python model_creation/step_06_run_hpo.py --outcome any_aki --branch windowed --feature_set all_waveforms
```

### 7. Model Training & Evaluation
Train the final model using the best hyperparameters and evaluate it on the test set.
*   **Bootstrapping**: Performs 25 bootstrap iterations to calculate **95% Confidence Intervals** for all metrics.
*   **Explainability**: Generates SHAP summary plots.
*   **Output**: Saves metrics (`metrics.csv`), model artifacts (`model.json`), and plots to `results/models/{outcome}/{branch}/{feature_set}/`.

```bash
python model_creation/step_07_train_evaluate.py --outcome any_aki --branch windowed --feature_set all_waveforms
```

#### Available Options
*   **Outcomes**: `any_aki`, `severe_aki`, `mortality`, `icu_admission`, `extended_los`.
*   **Branches**: `non_windowed` (Full Case), `windowed` (Segmented).
*   **Feature Sets**: `preop_only`, `all_waveforms`, `preop_and_all_waveforms`, `pleth_only`, `ecg_only`, etc.

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
*   `model_creation/`: **(New)** Modular modeling pipeline.
    *   `utils.py`: Shared logic for data loading and preprocessing.
    *   `run_hpo.py`: Hyperparameter optimization script.
    *   `train_evaluate.py`: Model training and evaluation script.
*   `notebooks/`: Jupyter notebooks.
    *   `03_tabular_data_prep.ipynb`: Legacy data transformation + preoperative data extraction notebook.
    *   `04_hpo_xgboost.ipynb`: Legacy modeling notebook.
*   `data/`: Data storage (raw and processed).
*   `results/`: Model outputs (params, metrics, plots).
