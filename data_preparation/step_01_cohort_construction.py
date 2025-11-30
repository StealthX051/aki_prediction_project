import pandas as pd
import vitaldb
import numpy as np
from data_preparation.inputs import (
    INPUT_FILE, 
    COHORT_FILE, 
    DEPARTMENTS, 
    MANDATORY_COLUMNS, 
    MANDATORY_WAVEFORMS, 
    WAVEFORM_SUBSTITUTIONS, 
    CUSTOM_FILTERS
)


# Load the clinical data
clinical_df = pd.read_csv(INPUT_FILE)

# --- STROBE Tracking ---
# Track 1: Total cases in the database is 6,388 
print(f"Total cases in VitalDB: {len(clinical_df)}")

# Track 2: Filter for DEPARTMENT_FILTER cases (using .str.lower() for robustness)
if DEPARTMENTS is not None:
    gs_df = clinical_df[
        clinical_df['department'].str.lower().isin([d.lower() for d in DEPARTMENTS])
    ].copy()
    print(f"{DEPARTMENTS} cases: {len(gs_df)}")
else:
    gs_df = clinical_df.copy()
    print(f"All department cases: {len(gs_df)}")

# Track 3: Filter for cases with essential data
initial_cohort_df = gs_df.dropna(subset=MANDATORY_COLUMNS)
print(f"Cases after removing missing essential data: {len(initial_cohort_df)}")

# Display the first few rows of your initial cohort
print("\nInitial Cohort DataFrame Head:")
print(initial_cohort_df.head())


# --- Assess Preoperative Variable Completeness ---
# Calculate the percentage of missing values for each column
missing_percentages = initial_cohort_df.isnull().sum() / len(initial_cohort_df) * 100

# Filter to show only columns that have missing values, and sort them
# This makes the output easier to read.
print("Missing value percentages for preoperative variables (>0%):")
print(missing_percentages[missing_percentages > 0].sort_values(ascending=False))


# --- Filter out cases without required waveforms ---
print(f"Using vitaldb.find_cases() to find all cases with the '{MANDATORY_WAVEFORMS}' waveforms...")
# Get the case IDs from your cohort
cohort_case_ids = initial_cohort_df['caseid'].tolist()
valid_cases_in_cohort = cohort_case_ids  # Start with all cases in the cohort
for track_name in MANDATORY_WAVEFORMS:
    # 1. Find all cases in the entire dataset that have the track_name (or substitute) waveform 
    cases_with_wave = vitaldb.find_cases([track_name])
    if track_name in WAVEFORM_SUBSTITUTIONS:
        for substitute in WAVEFORM_SUBSTITUTIONS[track_name]:
            substitute_cases = vitaldb.find_cases([substitute])
            cases_with_wave = np.union1d(cases_with_wave, substitute_cases)
    print(f"Total cases in VitalDB with '{track_name}' or substitute '{WAVEFORM_SUBSTITUTIONS.get(track_name)}': {len(cases_with_wave)}")

    # 2. Find the intersection between the two lists to see which of YOUR cases have the waveform
    # Using sets is a very fast way to do this
    valid_cases_in_cohort = np.intersect1d(cases_with_wave, valid_cases_in_cohort)

    # 4. Report the results
    print(f"Found {len(valid_cases_in_cohort)} cases with '{track_name}' out of {len(cohort_case_ids)} cases left in your cohort.")
    availability_percent = (len(valid_cases_in_cohort) / len(cohort_case_ids)) * 100
    print(f"Waveform Availability in Cohort: {availability_percent:.2f}%")

# You can now create a final cohort dataframe that only includes cases with the required waveform
final_cohort_with_waveform = initial_cohort_df[initial_cohort_df['caseid'].isin(valid_cases_in_cohort)].copy()

print(f"\nYour final cohort with available waveform data now has {len(final_cohort_with_waveform)} cases.")


# --- Apply any custom filters ---
cohort_df = final_cohort_with_waveform.copy()
for filter_func in CUSTOM_FILTERS:
    cohort_df = filter_func(cohort_df)

print(f"\nYour final cohort after all custom filters has {len(cohort_df)} cases.")
print(cohort_df.head())


# --- Derive Additional Outcomes ---
print("\n--- Deriving Additional Outcomes ---")

# Ensure source columns exist
# Map icu_days -> los_icu if needed
if 'icu_days' in cohort_df.columns and 'los_icu' not in cohort_df.columns:
    print("Mapping 'icu_days' to 'los_icu'...")
    cohort_df['los_icu'] = cohort_df['icu_days']

# Calculate los_postop if needed
if 'los_postop' not in cohort_df.columns:
    print("Calculating 'los_postop' from 'dis' and 'opend'...")
    # dis: Discharge time from casestart (sec)
    # opend: Operation end time from casestart (sec)
    # los_postop = (dis - opend) / (24 * 3600)
    cohort_df['los_postop'] = (cohort_df['dis'] - cohort_df['opend']) / 86400.0

# 1. In-hospital mortality
# Source: death_inhosp (0/1/NaN)
cohort_df["y_inhosp_mortality"] = cohort_df["death_inhosp"].astype("Int64")
print(f"In-hospital mortality (y_inhosp_mortality) counts:\n{cohort_df['y_inhosp_mortality'].value_counts(dropna=False)}")

# 2. ICU admission
# Source: los_icu (days). 1 if > 0, 0 if == 0, NaN if missing.
cohort_df["icu_los_days"] = cohort_df["los_icu"] # Keep raw continuous value
icu_admit = np.where(
    cohort_df["los_icu"].notna(),
    (cohort_df["los_icu"] > 0).astype("int8"),
    np.nan
)
cohort_df["y_icu_admit"] = pd.Series(icu_admit, index=cohort_df.index).astype("Int64")
print(f"ICU admission (y_icu_admit) counts:\n{cohort_df['y_icu_admit'].value_counts(dropna=False)}")

# 3. Prolonged postoperative hospital LOS (>= 75th percentile)
# Source: los_postop (days).
cohort_df["los_postop_days"] = cohort_df["los_postop"] # Keep raw continuous value

# Compute 75th percentile on non-missing LOS
los_q75 = cohort_df["los_postop"].dropna().quantile(0.75)
LOS_PROLONGED_THRESHOLD = los_q75
print(f"Prolonged LOS Threshold (75th percentile): {LOS_PROLONGED_THRESHOLD:.2f} days")

# Binary prolonged LOS label
prolonged_los = np.where(
    cohort_df["los_postop"].notna(),
    (cohort_df["los_postop"] >= LOS_PROLONGED_THRESHOLD).astype("int8"),
    np.nan
)
cohort_df["y_prolonged_los_postop"] = pd.Series(
    prolonged_los, index=cohort_df.index
).astype("Int64")
print(f"Prolonged LOS (y_prolonged_los_postop) counts:\n{cohort_df['y_prolonged_los_postop'].value_counts(dropna=False)}")

# 4. Severe AKI (Stage 2/3)
# Source: y_severe_aki (derived in add_aki_label)
if 'y_severe_aki' in cohort_df.columns:
    print(f"Severe AKI (y_severe_aki) counts:\n{cohort_df['y_severe_aki'].value_counts(dropna=False)}")

# --- Save the final cohort to a CSV file ---
cohort_df.to_csv(COHORT_FILE, index=False)
print(f"\nFinal cohort saved to {COHORT_FILE}")
