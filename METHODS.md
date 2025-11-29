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

**Outcome Definition**: Postoperative Acute Kidney Injury (AKI) was defined according to the **KDIGO** (Kidney Disease: Improving Global Outcomes) criteria. The outcome was binary, comparing patients who developed AKI against those who did not.
*   **AKI Positive**: Any increase in serum creatinine $\ge 0.3$ mg/dL within 48 hours of surgery OR $\ge 1.5$ times baseline within 7 days of surgery.
*   **AKI Negative**: Patients not meeting the above criteria.

## Data Preprocessing

### Clinical Data
A comprehensive set of preoperative variables was extracted:
*   **Demographics**: Age, Sex, Body Mass Index (BMI).
*   **Surgical Details**: Emergency Operation status (`emop`), Department, Surgical Approach.
*   **Comorbidities**: Hypertension (`preop_htn`), Diabetes Mellitus (`preop_dm`), ECG abnormalities (`preop_ecg`), Pulmonary Function Test results (`preop_pft`).
*   **Laboratory Values**: Hemoglobin (`preop_hb`), Platelets (`preop_plt`), Prothrombin Time (`preop_pt`), aPTT (`preop_aptt`), Sodium (`preop_na`), Potassium (`preop_k`), Glucose (`preop_gluc`), Albumin (`preop_alb`), AST (`preop_ast`), ALT (`preop_alt`), BUN (`preop_bun`), Creatinine (`preop_cr`), Bicarbonate (`preop_hco3`).

**Preprocessing Steps**:
*   **Outlier Handling**: To prevent data leakage, outlier thresholds (0.5th and 99.5th percentiles) were calculated strictly on the **training set**. Values outside this range in both training and test sets were clipped to these thresholds or replaced with random values within the plausible range.
*   **Imputation**: Missing clinical values were imputed with a constant value (-99) to allow tree-based models to handle missingness explicitly.
*   **Encoding**: Categorical variables were one-hot encoded. Rare categories (<30 occurrences) were merged into an 'other' category based on training set statistics.

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

## Model Development
We developed Gradient Boosted Decision Tree models using **XGBoost** (Extreme Gradient Boosting).

*   **Data Splitting**: The dataset was split into a training set (80%) and a hold-out test set (20%) using stratified sampling to maintain the prevalence of AKI in both sets. This split was performed prior to any preprocessing to ensure no data leakage.
*   **Hyperparameter Optimization (HPO)**: We used **Optuna**, a Bayesian optimization framework, to tune hyperparameters (e.g., learning rate, max depth, subsample ratio). The optimization objective was to maximize the Area Under the Receiver Operating Characteristic Curve (AUROC) using **3-fold stratified cross-validation** on the training set.
*   **Final Model**: The final model was trained on the entire training set using the best hyperparameters found during HPO.

## Statistical Analysis
Model performance was evaluated on the independent hold-out test set.
*   **Metrics**: The primary performance metric was **AUROC**. Secondary metrics included Area Under the Precision-Recall Curve (AUPRC), F1-score, Sensitivity, Specificity, and Accuracy.
*   **Interpretability**: SHapley Additive exPlanations (**SHAP**) values were calculated to quantify the contribution of each feature to the model's predictions, providing global and local interpretability.
