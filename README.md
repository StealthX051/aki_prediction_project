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
*   **Surgical Context**: Emergency Operation (`emop`), Department, Approach, ASA Class, Operation Type (`optype`), Anesthesia Type (`ane_type`).
*   **Comorbidities**: Hypertension (`preop_htn`), Diabetes (`preop_dm`), ECG abnormalities (`preop_ecg`), Pulmonary Function Test (`preop_pft`).
*   **Labs (Clinical Table)**: Hb, Platelets, PT (INR), aPTT, Na, K, Glucose, Albumin, AST, ALT, BUN, Creatinine, Bicarbonate.
*   **ABG (Clinical Table)**: pH, Base Excess (`preop_be`), PaO2, PaCO2, SaO2.
*   **Labs (Lab Table)**: Last preoperative value (within 30 days) for:
    *   `preop_wbc`: White Blood Cell count.
    *   `preop_crp`: C-Reactive Protein.
    *   `preop_lac`: Lactate.
*   **Derived Features**:
    *   `inpatient_preop`: Binary flag for inpatient admission prior to surgery.
    *   `preop_egfr_ckdepi_2021`: eGFR calculated using the CKD-EPI 2021 creatinine-only, race-free equation (per the National Kidney Foundation). The dataset-supplied `preop_gfr` column is intentionally excluded in favor of this derived value.
    *   Clinical flag helpers (e.g., `bun_high`, `hypoalbuminemia`) are computed for intermediate use but removed before one-hot encoding/imputation in the saved `preop_processed.csv`.

#### 2. Processing Steps
*   **Splitting**: Performs an 80/20 stratified split based on the outcome. This `split_group` is saved and used downstream to prevent leakage.
*   **Outlier Handling**: Calculates percentiles (0.5%, 99.5%) **only on the training set** and applies them to clip/impute outliers in both train and test sets.
*   **Imputation**: **Disabled by default**; missing values remain as `NaN` for downstream handling. Pass `--impute-missing` (or set `IMPUTE_MISSING=True` in `data_preparation/inputs.py`) to restore the previous `-99` sentinel behavior.
*   **Encoding**: Categorical variables are one-hot encoded. Rare categories (<30 occurrences) in `department` are merged into 'other'.

```bash
python data_preparation/step_03_preop_prep.py
# Optional: restore -99 imputation
python data_preparation/step_03_preop_prep.py --impute-missing
```

### Step 5: Intraoperative Data Prep
**File**: `data_preparation/step_04_intraop_prep.py`
Processes waveform features (pivoting, imputation) for both full and windowed modes.
*   **Technical Details**:
    *   **Pivoting**: Converts long-format Catch22 results into a wide format (one row per case/window).
    *   **Flattening**: Renames columns to `{waveform}_{feature}` (e.g., `SNUADC_PLETH_DN_HistogramMode_5`).
    *   **Missing Data**: Keeps `NaN` values for wholly missing waveform segments instead of zero-filling. Partial gaps remain
        linearly interpolated by upstream Catch22 extraction, and empty 10s windows are silently dropped as before.
```bash
python data_preparation/step_04_intraop_prep.py
```

### Step 6: Data Merge
**File**: `data_preparation/step_05_data_merge.py`
Combines preop and intraop data into master datasets.
*   **Technical Details**:
    *   Merges intraop features with processed preop data on `caseid`.
    *   **Integrity Check**: Ensures every row has a valid `split_group` from Step 3.
    *   **Missing Data**: Leaves NaNs in place by default; pass `--impute-missing` (or set `IMPUTE_MISSING=True`) to fill merge-introduced NaNs with `-99`.
    *   Outputs final wide CSVs ready for training.
```bash
python data_preparation/step_05_data_merge.py
# Optional: restore -99 imputation during merge
python data_preparation/step_05_data_merge.py --impute-missing
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
*   **Evaluation**: Generates predictions on the test set (`predictions.csv`) for unified analysis.
*   **Explainability**: Generates SHAP summary plots.
*   **Output**: Saves model (`model.json`), predictions (`predictions.csv`), and plots to `results/models/{outcome}/{branch}/{feature_set}/`.

```bash
python model_creation/step_07_train_evaluate.py --outcome any_aki --branch windowed --feature_set all_waveforms
```

#### Available Options
*   **Outcomes**: `any_aki`, `icu_admission` (default experiment script focus; other outcomes can be run manually if needed).
*   **Branches**: `non_windowed` (Full Case), `windowed` (Segmented).
*   **Feature Sets**: `preop_only`, `all_waveforms`, `preop_and_all_waveforms`, `pleth_only`, `ecg_only`, etc.
*   **Default Grid (`run_experiments.sh`)**: Primary runs cover preop-only, single-waveform models (`pleth_only`, `ecg_only`, `co2_only`, `awp_only`), all waveforms, and fused preop + all waveforms. Ablations pair preop with each single waveform (`preop_and_<waveform>`) and with all waveforms minus one (`preop_and_all_minus_<waveform>`). Two-channel waveform-only combinations (e.g., **AWP+CO2** or **ECG+PLETH**) are intentionally excluded from default sweeps and should be launched manually if needed.

### Step 8: Post-hoc Analysis & Visualization
**File**: `results_recreation/results_analysis.py`
Generates publication-ready tables and figures for both the **Primary Pipeline** (Catch22/XGBoost) and the **Experimental Pipeline** (Aeon/Multirocket).
*   **Unified Reporting**: Automatically detects and aggregates results from both pipelines into a single report.
*   **Global Calibration**: Applies Logistic Regression calibration to raw probabilities to ensure accurate risk estimates while preserving ranking.
*   **Constrained Thresholding**: Selects optimal thresholds maximizing F2-score with a minimum specificity constraint (0.6) to ensure balanced performance.
*   **Bootstrapping**: Calculates 95% Confidence Intervals using 1000 bootstrap iterations (parallelized).
*   **Output**:
    *   **Reports**:
        *   `results/report.docx`: A formatted Word document containing all results tables with selective bolding and background gradients.
        *   `results/report.pdf`: An aggregated PDF report of all tables.
        *   `results/tables/*.html`: Individual HTML tables for each outcome/branch.
    *   **Figures**: High-quality ROC, PR, and Calibration curves saved in `results/figures/`.
    *   **Data**: `results/tables/metrics_summary.csv` containing all calculated metrics and CIs.
```bash
python results_recreation/results_analysis.py
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
*   `model_creation/`: **(New)** Modular modeling pipeline.
    *   `utils.py`: Shared logic for data loading and preprocessing.
    *   `run_hpo.py`: Hyperparameter optimization script.
    *   `train_evaluate.py`: Model training and evaluation script (consolidated).
*   `model_creation_aeon/`: **(New)** Experimental Aeon pipeline.
    *   `classifiers.py`: Custom `FusedClassifier`, `RocketFused`, and `FreshPrinceFused` classes implementing early fusion.
    *   `step_06_aeon_train.py`: CLI-driven script for training Aeon models.
*   `notebooks/`: Jupyter notebooks.
    *   `03_tabular_data_prep.ipynb`: Legacy data transformation + preoperative data extraction notebook.
    *   `04_hpo_xgboost.ipynb`: Legacy modeling notebook.
*   `data/`: Data storage (raw and processed).
*   `results/`: Model outputs (params, metrics, plots).

## 🧪 Experimental Pipeline: Aeon (Time Series Classification)

> **Note**: This pipeline operates separately from the main Catch22/XGBoost pipeline.

This branch implements an end-to-end Deep Learning/State-of-the-Art Time Series Classification pipeline using the **Aeon** library. It bypasses manual feature engineering (Catch22) in favor of direct waveform processing and fusion.

### Overview
*   **Resampling**: All waveforms are resampled to **1 Hz** (1 data point per second) to preserve temporal structure while managing dimensionality.
*   **Padding**: Variable-length sequences are right-padded with zeros to a fixed maximum length of **16 hours** (57,600 timepoints).
*   **Fusion**: Early fusion of these 1 Hz waveforms (transformed via Rocket) with 115 static preoperative features. FreshPRINCE is currently disabled due to computational constraints.
*   **Goal**: Compare manual feature engineering (Catch22+XGB) against SOTA Time Series Classifiers (Rocket, FreshPRINCE).
*   **Fusion Strategy**: **Early Fusion**. Preoperative tabular features are concatenated with waveform embeddings before the final classifier head.

### Pipeline Steps
### Pipeline Components

| Component | Script | Description |
| :--- | :--- | :--- |
| **Export** | `data_preparation/step_02_aeon_export.py` | Loads waveforms, resamples to `L=8000` (fixed length), exports to `.npz`. <br> **Args**: `--limit` (debug) |
| **Preop Prep** | `data_preparation/step_04_aeon_prep.py` | Prepares tabular data: preserves `NaN` by default; add `--impute-missing` to apply median imputation with missingness indicators. |
| **Training** | `model_creation_aeon/step_06_aeon_train.py` | Trains separate or fused models. <br> **Models**: `multirocket` (default `n_kernels=10000`), `minirocket`, `freshprince`. <br> **HPO**: Optuna optimization (100 trials) for linear head (`C`) maximizing AUPRC. Class weight is fixed to 'balanced'. <br> **Fusion**: Concatenates preop features with Rocket embeddings. <br> **Outputs**: Saves `predictions.csv` for unified analysis. |
| **Reference** | `model_creation_aeon/classifiers.py` | Contains `RocketFused` and `FreshPrinceFused` class definitions. |
| **Analysis** | `results_recreation/results_analysis.py` | Unified 1000-fold bootstrapping and report generation for both pipelines. |

### Technical Specifications
*   **Library**: `aeon` (Sktime fork), `tsfresh`.
*   **Input Shape**: `(N_samples, N_channels=4, Length=8000)`.
*   **Fusion Type**: Early Fusion (Preop features concatenated to transform embeddings).
*   **Evaluation**: 
    *   **Outcomes**: Primary (AKI) + Secondary (Mortality, ICU, Extended LOS, Severe AKI).
    *   **Metrics**: 1000-fold bootstrapped CIs for AUROC, AUPRC, etc.
    *   **Ablations**: Single Channel, Leave-One-Out, and Fusion impact analysis.

### 5. Troubleshooting & Learnings
*   **Aeon v1.1.0 Compatibility**:
    *   The pipeline explicitly supports `aeon >= 1.1.0`. `MultiRocket` and `MiniRocket` are imported from `aeon.transformations.collection.convolution_based`.
    *   The correct parameter for kernel count is `n_kernels` (not `num_kernels`).
    *   Estimators must be passed as instantiated Objects
    *   **Head**: A **Logistic Regression** classifier is trained on the fused, standardized feature set.
    *   **Optimization (HPO)**: We optimize the linear head using **Optuna** with 5-fold stratified cross-validation on the training set.
        *   **Objective**: Maximize **AUPRC**.
        *   **Search Space**:
            *   Regularization Strength (`C`): Log-uniform distribution [1e-3, 10.0].
            *   Class Weight (`class_weight`): Fixed to 'balanced'.
        *   **Scaling**: A `StandardScaler` is fitted within each CV fold to prevent data leakage.
        *   **Trials**: 100 trials.
*   **Class Balance in Testing**:
    *   When running with `--limit` (e.g., smoke testing), the export script enforces a balanced selection of positive and negative cases. This is critical because `LogisticRegression` will crash if the training fold contains only a single class.
*   **Performance**:
    *   Crucial: Ensure `n_jobs=-1` is passed to the Aeon transformer. For HPO, we utilize **parallel Optuna trials** (`n_jobs=-1`) with **sequential** scikit-learn fits (`n_jobs=1`).
    *   **Threading**: We explicitly set `OMP_NUM_THREADS=1` (and related BLAS variables) to prevent thread contention. This ensures that the 32 parallel trials (processes) each use a single core efficiently, rather than each trial attempting to spawn multiple threads.
*   **Convergence**:
    *   **Max Iterations**: The default `lbfgs` solver in `LogisticRegression` may fail to converge on the high-dimensional MiniRocket features (10k+). We increased `max_iter` to **5000** to resolve `ConvergenceWarning`s.

### Execution
Run the full experimental suite:
```bash
bash run_full_pipeline_aeon.sh
```
Or individual experiments:
```bash
bash run_experiments_aeon.sh
```
The Aeon experiment script currently evaluates the `any_aki` and `icu_admission` outcomes.
