# 🗄️ Archive Documentation: AKI & Mortality Prediction Experiments

This archive contains the complete history of experiments, data processing pipelines, and model training scripts developed for the AKI (Acute Kidney Injury) and Mortality prediction project.

This documentation is designed to be **extremely expansive**, providing a detailed breakdown of every file, its purpose, the logic it implements, and why it was archived. This serves as a knowledge base for future iterations.

---

## 📂 Directory Structure

The archive is organized into two main directories:

*   **`notebooks/`**: Jupyter notebooks used for initial exploration (EDA), data cleaning, cohort selection, and prototyping of feature extraction pipelines.
*   **`scripts/`**: Production-ready Python scripts for large-scale data processing (spectrogram generation, feature extraction) and model training.

---

## 1. 📊 Data Preparation & EDA (Notebooks)

These notebooks represent the foundational work of the project: defining the cohort, cleaning clinical data, and engineering the target labels.

### `notebooks/1_eda.ipynb` (Exploratory Data Analysis)
*   **Purpose**: Initial assessment of the `vitaldb` dataset feasibility.
*   **Key Logic**:
    *   **Cohort Selection**: Filters `clinical_data.csv` for "General Surgery" cases.
    *   **Data Quality**: Checks for missing values in preoperative variables (e.g., `preop_cr`, `preop_hb`).
    *   **Waveform Availability**: Uses `vitaldb.find_cases()` to cross-reference clinical cases with available `SNUADC/PLETH` (plethysmograph) waveforms.
    *   **Result**: Established that ~96% of the general surgery cohort had usable waveform data.

### `notebooks/2a_aki_data_preparation.ipynb` (AKI Pipeline)
*   **Purpose**: End-to-end data cleaning and label engineering for the **Acute Kidney Injury (AKI)** target.
*   **Key Logic**:
    *   **Exclusions**: Removes patients with pre-existing severe kidney disease (Creatinine > 4.0 mg/dL).
    *   **Label Engineering (KDIGO)**: Implements the KDIGO criteria for AKI:
        *   Increase in serum creatinine by $\ge 0.3$ mg/dL within 48 hours.
        *   Increase in serum creatinine to $\ge 1.5$ times baseline within 7 days.
    *   **Imputation**: Fills missing preoperative values with `-99` (for tree-based models).
    *   **Outlier Handling**: Clamps continuous variables (e.g., BMI, Age) to the 1st and 99th percentiles to remove physiological impossibilities.
    *   **Output**: `preop_train_cleaned.csv` and `preop_test_cleaned.csv`.

### `notebooks/2b_death_data_preparation.ipynb` (Mortality Pipeline)
*   **Purpose**: Adapted version of the 2a pipeline for the **In-Hospital Mortality** target.
*   **Key Logic**:
    *   **Target**: Uses the `death_inhosp` column directly.
    *   **Class Imbalance**: Notes the severe class imbalance (approx. 0.9% mortality rate), which necessitated different modeling strategies (like Focal Loss) later on.
    *   **Output**: `final_cohort_with_death_label.csv`.

---

## 2. 🌊 Waveform Processing & Feature Engineering

This project experimented with two distinct approaches to handling high-frequency waveform data: **Spectrograms (Deep Learning)** and **Catch22 Features (Statistical/Machine Learning)**.

### Approach A: Spectrograms (Image-Based Deep Learning)

This approach converts 1D waveform signals into 2D time-frequency images (spectrograms) to leverage computer vision models.

#### `scripts/5b_death_generate_spectrograms.py`
*   **Purpose**: High-performance pipeline to generate RGB spectrogram images from `SNUADC/PLETH` waveforms.
*   **Key Logic**:
    1.  **Load Waveform**: Streams data from VitalDB at 500Hz.
    2.  **Downsample**: Decimates signal to 100Hz to reduce noise and size.
    3.  **Segmentation**: Slices signal into 10-second windows.
    4.  **STFT**: Applies Short-Time Fourier Transform using `scipy.signal.spectrogram`.
    5.  **Image Conversion**:
        *   Converts power to Decibels (dB).
        *   Normalizes to [0, 1].
        *   Applies 'viridis' colormap.
        *   Saves as 256x256 PNG.
*   **Optimization**: Uses `concurrent.futures.ThreadPoolExecutor` for parallel processing and a custom numpy-based image saving function (bypassing Matplotlib) for speed.

#### `notebooks/3_spectrogram_test.ipynb`
*   **Purpose**: Visual verification and unit testing of the spectrogram generation code.
*   **Key Logic**: Generates plots comparing the raw signal to the generated spectrogram to ensure alignment and artifact-free conversion.

### Approach B: Catch22 (Time-Series Feature Extraction)

This approach extracts 22 canonical time-series features (Catch22) from sliding windows to feed into tabular models like XGBoost.

#### `notebooks/5_catch22_feature_generation_test.ipynb`
*   **Purpose**: Prototyping the sliding window statistical aggregation method.
*   **Key Logic**:
    *   **Sliding Window**: Moves a 10-minute window with a 5-minute step over the waveform.
    *   **Feature Extraction**: Calculates 22 features (e.g., DN_HistogramMode_5, CO_f1ecac) for *each* window.
    *   **Aggregation**: Computes `mean`, `std`, `min`, and `max` for each feature across all windows, resulting in a single vector per patient.

#### `scripts/8b_catch22_test.py`
*   **Purpose**: A strict unit test for the Catch22 worker function.
*   **Key Logic**: Verifies that the function handles edge cases (empty waveforms, NaNs, short signals) gracefully without crashing the pipeline.

---

## 3. 🤖 Model Training & Optimization

### Deep Learning (AutoGluon MultiModal)

#### `scripts/6b_death_train_multi_model.py`
*   **Purpose**: Trains a Convolutional Neural Network (CNN) on the generated spectrograms.
*   **Framework**: AutoGluon MultiModal (automating PyTorch Lightning).
*   **Key Techniques**:
    *   **Backbone**: `mobilenetv3_large_100` (efficient, mobile-friendly architecture).
    *   **Loss Function**: **Focal Loss** ($\gamma=2.0$) to handle the 0.9% mortality class imbalance.
    *   **Augmentation**: **SpecAugment** (Time masking and Frequency masking) to prevent overfitting on the spectrograms.
    *   **Metric**: Optimized for **AUPRC (Average Precision)** rather than Accuracy.

#### `scripts/7_continue_training.py`
*   **Purpose**: Utility to resume an interrupted AutoGluon training session.
*   **Key Logic**: Loads a specific `.ckpt` checkpoint file and calls `.fit()` again with a new time limit.

#### `notebooks/4_model_training_test.ipynb`
*   **Purpose**: A "Smoke Test" for the training rig.
*   **Key Logic**: Runs the full training loop on a tiny subset (N=400) of data for just 10 minutes. This verifies that CUDA is active, memory is sufficient, and paths are correct before committing to a multi-day run.

### Machine Learning (XGBoost)

#### `scripts/9_hpo_xgboost.py`
*   **Purpose**: Hyperparameter Optimization (HPO) for an XGBoost classifier using the Catch22 features.
*   **Framework**: Optuna + XGBoost.
*   **Key Logic**:
    *   **Objective**: Maximize ROC-AUC on a validation set.
    *   **Search Space**: Tunes `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `reg_alpha`, and `reg_lambda`.
    *   **Imbalance Handling**: Uses `scale_pos_weight` calculated as `neg_samples / pos_samples`.
    *   **Early Stopping**: Implements callback-based early stopping to prevent overfitting.

---

## 4. � File Manifest

| File | Type | Description |
| :--- | :--- | :--- |
| `1_eda.ipynb` | Notebook | Initial cohort selection and data quality checks. |
| `2a_aki_data_preparation.ipynb` | Notebook | KDIGO label engineering for AKI. |
| `2b_death_data_preparation.ipynb` | Notebook | Label engineering for Mortality. |
| `3_spectrogram_test.ipynb` | Notebook | Visual QA for spectrogram generation. |
| `4_model_training_test.ipynb` | Notebook | Smoke test for AutoGluon training. |
| `5_catch22_feature_generation_test.ipynb` | Notebook | Prototype for sliding window feature extraction. |
| `5b_death_generate_spectrograms.py` | Script | Production pipeline for RGB spectrograms. |
| `6b_death_train_multi_model.py` | Script | AutoGluon training script with Focal Loss. |
| `7_continue_training.py` | Script | Resume training from checkpoint. |
| `8b_catch22_test.py` | Script | Unit test for Catch22 logic. |
| `9_hpo_xgboost.py` | Script | Optuna HPO for XGBoost. |
| `catch22.ipynb` | Notebook | (Empty/Deprecated) Likely a scratchpad. |

---

## 5. 💡 Lessons Learned & Archival Reasons

1.  **Spectrogram Size**: Generating 256x256 PNGs for every 10-second window resulted in a massive dataset (millions of files). While effective for DL, storage and I/O became bottlenecks.
2.  **Class Imbalance**: The mortality dataset (0.9% positive) proved extremely difficult for standard CrossEntropy loss. The switch to Focal Loss and AUPRC metric in `6b_death_train_multi_model.py` was a direct response to this.
3.  **Feature Engineering**: The Catch22 approach (`9_hpo_xgboost.py`) was explored as a computationally cheaper alternative to the deep learning approach, offering interpretability via feature importance.
