# Methods

## Study Design and Data Source
This retrospective observational study utilized high-resolution intraoperative physiological data from the **VitalDB** (Vital Signs DataBase), an open-access public dataset containing multi-parameter monitoring data from 6,388 surgical patients undergoing non-cardiac surgery at Seoul National University Hospital.

## Cohort Selection
Patients were included in the study if they met the following criteria:
1.  **Data Availability**: Complete preoperative clinical data, specifically preoperative creatinine (`preop_cr`) and operation end time (`opend`).
2.  **Waveform Availability**: Presence of at least one of the following high-frequency intraoperative waveforms:
    *   **Photoplethysmography (PPG)**: `SNUADC/PLETH`
    *   **Electrocardiogram (ECG)**: `SNUADC/ECG_II` (or `SNUADC/ECG_V5` as a substitute)
    *   **Capnography (CO2)**: `Primus/CO2`
    *   **Airway Pressure (AWP)**: `Primus/AWP`
3.  **Clinical Criteria**:
    *   Availability of postoperative creatinine measurements for AKI adjudication.
    *   Exclusion of patients with pre-existing end-stage renal disease (baseline creatinine > 4.0 mg/dL).
    *   Exclusion of cases with insufficient waveform duration or quality (>5% missing data in the signal segment).


**Outcome Definition**:
*   **Primary Outcome**: Postoperative Acute Kidney Injury (AKI) was defined according to the **KDIGO** (Kidney Disease: Improving Global Outcomes) criteria. The outcome was binary, comparing patients who developed AKI against those who did not.
*   **AKI Positive**: Any increase in serum creatinine $\ge 0.3$ mg/dL within 48 hours of surgery OR $\ge 1.5$ times baseline within 3 days (72 hours) of surgery, emphasizing perioperative attribution for this anesthesiology-focused cohort.
*   **AKI Negative**: Patients not meeting the above criteria.
*   **Secondary Outcomes**:
    *   **Severe AKI**: Defined as KDIGO Stage 2 or 3 AKI (`y_severe_aki`).
        *   **Stage 2**: Serum creatinine $\ge 2.0$ times baseline.
        *   **Stage 3**: Serum creatinine $\ge 3.0$ times baseline OR $\ge 4.0$ mg/dL.
    *   **In-hospital Mortality**: Defined as death occurring before discharge from the index hospitalization (`y_inhosp_mortality`).
    *   **ICU Admission**: Defined as any postoperative ICU stay > 0 days (`y_icu_admit`).
    *   **Prolonged Postoperative Length of Stay (LOS)**: Defined as postoperative hospital stay $\ge$ 75th percentile of the cohort (`y_prolonged_los_postop`).

## Data Preprocessing

### Clinical Data
A comprehensive set of preoperative variables was extracted from the VitalDB clinical and laboratory tables.

#### 1. Variable Definitions
*   **Demographics**: Age (years), Sex (Male/Female), Height (cm), Weight (kg), Body Mass Index (BMI, kg/m²).
*   **Surgical Context**:
    *   **Emergency Operation (`emop`)**: Binary flag.
    *   **Department**: Surgical department (e.g., General Surgery, Thoracic). Rare departments (<30 cases) were merged into 'other'.
    *   **Approach**: Surgical approach (e.g., Open, Laparoscopic).
    *   **ASA Class**: American Society of Anesthesiologists physical status classification.
    *   **Operation Type (`optype`)**: Specific type of surgery.
    *   **Anesthesia Type (`ane_type`)**: Type of anesthesia used.
*   **Comorbidities**:
    *   **Hypertension (`preop_htn`)**: Binary.
    *   **Diabetes Mellitus (`preop_dm`)**: Binary.
    *   **ECG Abnormalities (`preop_ecg`)**: Categorical (Normal vs. various abnormalities).
    *   **Pulmonary Function Test (`preop_pft`)**: Categorical results.
*   **Laboratory Values**:
    *   **Source**: Most recent value within 30 days prior to surgery.
    *   **Hematology**: Hemoglobin (`preop_hb`), Platelets (`preop_plt`), White Blood Cells (`preop_wbc`).
    *   **Coagulation**: Prothrombin Time (`preop_pt`, INR), aPTT (`preop_aptt`).
    *   **Electrolytes/Metabolic**: Sodium (`preop_na`), Potassium (`preop_k`), Glucose (`preop_gluc`), Albumin (`preop_alb`), Bicarbonate (`preop_hco3`).
    *   **Liver/Kidney**: AST (`preop_ast`), ALT (`preop_alt`), BUN (`preop_bun`), Creatinine (`preop_cr`).
    *   **Inflammatory/Other**: C-Reactive Protein (`preop_crp`), Lactate (`preop_lac`).
*   **Arterial Blood Gas (ABG)**: pH (`preop_ph`), Base Excess (`preop_be`), PaO2 (`preop_pao2`), PaCO2 (`preop_paco2`), SaO2 (`preop_sao2`).

#### 2. Derived Features
We computed several derived features to capture clinical status more effectively:
*   **Inpatient Status (`inpatient_preop`)**: Binary indicator that the patient was admitted prior to surgery (admission time < 0 seconds relative to case start).
*   **Estimated GFR (`preop_egfr_ckdepi`)**: Calculated using the **CKD-EPI 2009** creatinine equation (race-free version):
    $$ eGFR = 141 \times \min(S_{cr}/\kappa, 1)^\alpha \times \max(S_{cr}/\kappa, 1)^{-1.209} \times 0.993^{Age} \times 1.018 [if Female] $$
    Where $S_{cr}$ is serum creatinine, $\kappa$ is 0.7 (F) or 0.9 (M), and $\alpha$ is -0.329 (F) or -0.411 (M).
*   **Clinical Flags** (Binary indicators of abnormal physiology):
    *   **High BUN**: `preop_bun > 27` mg/dL.
    *   **Hypoalbuminemia**: `preop_alb < 3.5` g/dL.
    *   **Anemia**: `preop_hb < 13.0` (Male) or `< 12.0` (Female) g/dL.
    *   **Hyponatremia**: `preop_na < 135` mmol/L.
    *   **Metabolic Acidosis**: `preop_hco3 < 22` mmol/L OR `preop_be < -2` mEq/L.
    *   **Hypercapnia**: `preop_paco2 > 45` mmHg.
    *   **Hypoxemia**: `preop_pao2 < 80` mmHg OR `preop_sao2 < 95` %.

#### 3. Data Preprocessing
*   **Train/Test Split**: An 80/20 stratified split was performed based on the primary outcome (`aki_label`) *before* any further processing to ensure strict separation.
*   **Outlier Handling**: Continuous variables were checked for outliers. Percentiles (0.5% and 99.5%) were calculated **using only the training set**. Values outside this range in both training and test sets were replaced with random values drawn from the [0.5%, 5%] range (for low outliers) or [95%, 99.5%] range (for high outliers) to preserve distribution shape while capping extremes.
*   **Imputation**: Missing values in continuous and categorical features were imputed with a constant value (**-99**) to allow tree-based models (XGBoost) to learn missingness patterns explicitly.
*   **Encoding**:
    *   Categorical variables (e.g., `department`, `approach`) were **One-Hot Encoded**.
    *   Binary variables (e.g., `sex`, `emop`, clinical flags) were encoded as 0/1.
    *   Rare categories in `department` (<30 occurrences in training set) were merged into an 'other' category.

### Waveform Signal Processing
Raw waveforms were processed to remove artifacts and standardize sampling rates using the following specifications:
*   **Photoplethysmography (PPG)**:
    *   Downsampled from 500 Hz to **100 Hz**.
    *   Band-pass filtered (**0.1–10 Hz**, Butterworth 4th order) to preserve pulse wave morphology while removing baseline wander and high-frequency noise (Lapitan 2024, Sci Rep; Park 2022, Front Physiol).
*   **Electrocardiogram (ECG)**:
    *   Downsampled from 500 Hz to **100 Hz**.
    *   Band-pass filtered (**0.5–40 Hz**, Butterworth 4th order) (Kligfield 2007, Circulation; Pan & Tompkins 1985, IEEE TBME).
*   **Capnography (CO2)**:
    *   Maintained at native **62.5 Hz**.
    *   Low-pass filtered (**8 Hz**, Butterworth 4th order) (Gutiérrez 2018, PLoS One; Leturiondo 2017, CinC).
*   **Airway Pressure (AWP)**:
    *   Maintained at native **62.5 Hz**.
    *   Low-pass filtered (**12 Hz**, Butterworth 4th order) (de Haro 2024, Crit Care; Thome 1998, J Appl Physiol).

**Quality Control**: Segments with >5% missing values were excluded. Shorter gaps were filled using linear interpolation.

## Feature Engineering
We utilized the **Catch24** feature set, which consists of the standard 22 **Catch22** (CAnonical Time-series CHaracteristics) features plus the **Mean** and **Standard Deviation** of the signal. This results in 24 features per waveform channel.

Two feature extraction strategies were employed:
1.  **Full-Case Features**: Features were extracted from the entire duration of the surgery (resampled to 10 Hz for computational efficiency).
2.  **Windowed Features**: Waveforms were segmented into **10-second windows** with a **5-second overlap** (50% overlap). Catch24 features were calculated for each window. To generate a fixed-size input for the tabular model, these windowed features were aggregated across the entire case, calculating the **Mean**, **Standard Deviation**, **Minimum**, and **Maximum** for each of the 24 features.

### 3. Model Development
Feature selection and hyperparameter optimization were performed using the training set.

*   **Algorithm**: XGBoost (Extreme Gradient Boosting).
*   **Hyperparameter Optimization (HPO)**: Performed using Optuna with 5-fold stratified cross-validation on the training set to maximize the Area Under the Precision-Recall Curve (AUPRC).
*   **Final Model Training**: The optimal hyperparameters identified were used to train the final model on the full training set.
*   **Evaluation**: The final model was evaluated on the held-out test set. We generated 95% confidence intervals (CIs) for all performance metrics using 1000-fold bootstrapping of the test set predictions. Platt scaling was applied to calibrate predicted probabilities.

## Statistical Analysis
Model performance was evaluated on the independent hold-out test set.
*   **Metrics**: The primary performance metric was **AUPRC**, given the class imbalance. Secondary metrics included AUROC, F1-score, Sensitivity, Specificity, Accuracy, and Brier Score.
*   **Interpretability**: SHapley Additive exPlanations (**SHAP**) values were calculated to quantify the contribution of each feature to the model's predictions, providing global and local interpretability.

## Post-hoc Analysis
To ensure robust and clinically applicable performance estimates, a rigorous post-hoc analysis pipeline was implemented:

1.  **Global Calibration**: Raw model probabilities were calibrated using **Logistic Regression (Platt Scaling)**. A single calibrator was fit for each model configuration (Outcome × Branch × Feature Set) on the entire prediction set. This approach ensures that the predicted probabilities reflect true risk levels while strictly preserving the global ranking of patients (monotone transformation), avoiding the rank distortions often caused by fold-wise calibration methods.

2.  **Constrained Thresholding**: Optimal decision thresholds were selected to maximize the **F2-score** (which prioritizes recall over precision, reflecting the clinical importance of missing AKI cases). To prevent degenerate solutions in highly imbalanced scenarios (e.g., predicting all positives), we imposed a **minimum specificity constraint of 0.6**. If no threshold met this constraint, the unconstrained F2-optimal threshold was used as a fallback.

3.  **Statistical Inference**: We computed **95% Confidence Intervals (CIs)** for all reported metrics (AUROC, AUPRC, Brier Score, Sensitivity, Specificity, F1) using **non-parametric bootstrapping** with **1000 iterations**. This provides a reliable measure of the uncertainty associated with our performance estimates.

## Experimental Pipeline: Time Series Classification (Aeon)

> **Note**: This section describes the methodology for the experimental parallel pipeline designed to benchmark State-of-the-Art (SOTA) time series classifiers against the primary Catch22/XGBoost approach.

### 1. Data Preprocessing (Aeon Branch)
To accommodate fixed-input classifiers (e.g., Rocket), a distinct preprocessing strategy was employed:
*   **1 Hz Resampling**: All intraoperative waveforms (`SNUADC/PLETH`, `SNUADC/ECG_II`, `Primus/CO2`, `Primus/AWP`) were resampled to a uniform frequency of 1 Hz.
*   **Fixed-Length Padding**: To facilitate batch processing and compatibility with certain classifiers, all series were right-padded with zeros to a fixed maximum duration of 16 hours (57,600 timepoints). Sequences longer than 16 hours were truncated.
*   **Anesthesia Duration**: To compensate for the loss of absolute time information due to resampling, `anesthesia_duration_minutes` was explicitly added as a tabular feature.
*   **Imputation**:
    *   **Waveforms**: Cases with >5% missing data were dropped. Remaining gaps were linearly interpolated to prevent zero-padding artifacts which can distort convolution kernels.
    *   **Tabular Data**: Unlike the tree-based pipeline (which uses -99), tabular features for linear heads (Ridge/Logistic) were imputed using **Median Imputation** and augmented with **Binary Missingness Indicators**.

### 2. Modeling Strategy (Early Fusion)
We implemented an **Early Fusion** architecture where waveform features and preoperative tabular features are concatenated into a single vector before being passed to the final classifier.

*   **Convolutional Models (MultiRocket / MiniRocket)**:
    *   **Transform**: We utilized `MultiRocket` and `MiniRocket` (Aeon implementations) to extract features. `MultiRocket` applies **10,000** random convolutional kernels (dilated, padded) to the raw 3D input `(N, Channels, 8000)`.
    *   **Fusion**: The resulting feature map (approx. 50k features for MultiRocket) is concatenated with the processed preoperative vector (Median imputed).
    *   **Scaling**: A global `StandardScaler` is applied to the combined feature matrix to normalize scales between waveform embeddings and clinical variables.
    *   **Head**: A **Logistic Regression** classifier is trained on the fused, standardized feature set.
    *   **Optimization (HPO)**: We optimize the linear head using **Optuna** with 5-fold stratified cross-validation on the training set.
        *   **Objective**: Maximize **AUPRC**.
        *   **Search Space**:
            *   Regularization Strength (`C`): Log-uniform distribution [1e-3, 10.0].
            *   Class Weight (`class_weight`): Fixed to 'balanced'.
        *   **Scaling**: A `StandardScaler` is fitted within each CV fold to prevent data leakage.
        *   **Trials**: 100 trials.
*   **FreshPRINCE** (FreshPRince Is Not a Clustered Ensemble):
    *   **Transform**: Extracts comprehensive time-series features using **TSFresh** (relevant features selected via FDR control).
    *   **Head**: A **Rotation Forest** classifier (an ensemble of **200** PCA-based decision trees) is trained on the fused features.

### 3. Software and Libraries
The experimental pipeline relies on the **`aeon`** toolkit (v1.1.0+) for time series learning, **`tsfresh`** for feature extraction, and **`scikit-learn`** for underlying classifier implementations (Logistic Regression).

### 4. Experimental Design and Evaluation
The Aeon pipeline mirrors the rigorous design of the primary pipeline:
*   **Outcomes**: We train separate models for the primary outcome (`aki_label`) and each secondary outcome (`y_severe_aki`, `y_inhosp_mortality`, `y_icu_admit`, `y_prolonged_los_postop`). This ensures that the training labels correctly correspond to the intended prediction target.
*   **Ablation Studies**: To quantify feature importance, we conduct systematic ablations:
    *   **Single Channel**: Performance of each waveform individually (ECG, PLETH, CO2, AWP).
    *   **Leave-One-Out**: Performance impact of removing a single waveform from the full set.
    *   **Fusion Impact**: Comparison of "Waveform Only" vs. "Early Fusion" (Waveform + Preop) performance.
*   **Bootstrapping**: 1000-fold bootstrapped metrics (AUROC, AUPRC) with 95% Confidence Intervals.
*   **Calibration**: Logistic calibration (Platt Scaling) applied to the decision function outputs.
