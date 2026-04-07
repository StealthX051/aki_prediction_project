import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

from data_preparation.aki_cohort import (
    annotate_aki_eligibility,
    annotate_aki_labels,
    load_creatinine_labs,
)
from data_preparation.artifact_metadata import (
    STEP_01_COHORT_ARTIFACT,
    write_artifact_metadata,
)
from data_preparation.inputs import (
    AKI_COHORT_FLOW_MANDATORY_COLUMNS,
    COHORT_FILE,
    DEPARTMENTS,
    INPUT_FILE,
    LAB_DATA_FILE,
    MANDATORY_WAVEFORMS,
    OUTER_COHORT_MANDATORY_COLUMNS,
    RESULTS_DIR,
    WAVEFORM_SUBSTITUTIONS,
)

PAPER_DIR = Path(os.getenv("PAPER_DIR", RESULTS_DIR.parent / "paper"))
COHORT_FLOW_COUNTS_FILE = Path(
    os.getenv("COHORT_FLOW_COUNTS_FILE", PAPER_DIR / "metadata" / "cohort_flow_counts.json")
)
ASA_EXCLUDED_VALUES = {"5", "5.0", "6", "6.0", "V", "VI"}


def _filter_departments(clinical_df: pd.DataFrame) -> pd.DataFrame:
    if DEPARTMENTS is None:
        print(f"All department cases: {len(clinical_df)}")
        return clinical_df.copy()

    filtered_df = clinical_df[
        clinical_df["department"].str.lower().isin([d.lower() for d in DEPARTMENTS])
    ].copy()
    print(f"{DEPARTMENTS} cases: {len(filtered_df)}")
    return filtered_df


def _exclude_asa_v_vi(df: pd.DataFrame) -> pd.DataFrame:
    asa_mask = ~df["asa"].astype(str).isin(ASA_EXCLUDED_VALUES)
    return df[asa_mask].copy()


def _print_missing_percentages(df: pd.DataFrame) -> None:
    missing_percentages = df.isnull().sum() / len(df) * 100
    print("Missing value percentages for preoperative variables (>0%):")
    print(missing_percentages[missing_percentages > 0].sort_values(ascending=False))


def _restrict_to_required_waveforms(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    print(
        "Using vitaldb.find_cases() to find all cases with the "
        f"'{MANDATORY_WAVEFORMS}' waveforms..."
    )

    cohort_case_ids = df["caseid"].tolist()
    valid_cases_in_cohort = cohort_case_ids
    waveform_counts: list[dict[str, object]] = []

    for track_name in MANDATORY_WAVEFORMS:
        cases_with_wave = vitaldb.find_cases([track_name])
        if track_name in WAVEFORM_SUBSTITUTIONS:
            for substitute in WAVEFORM_SUBSTITUTIONS[track_name]:
                substitute_cases = vitaldb.find_cases([substitute])
                cases_with_wave = np.union1d(cases_with_wave, substitute_cases)

        print(
            "Total cases in VitalDB with "
            f"'{track_name}' or substitute '{WAVEFORM_SUBSTITUTIONS.get(track_name)}': "
            f"{len(cases_with_wave)}"
        )

        valid_cases_in_cohort = np.intersect1d(cases_with_wave, valid_cases_in_cohort)
        remaining_cases = len(valid_cases_in_cohort)
        print(
            f"Found {remaining_cases} cases with '{track_name}' "
            f"out of {len(cohort_case_ids)} cases left in your cohort."
        )
        availability_percent = (remaining_cases / len(cohort_case_ids)) * 100
        print(f"Waveform Availability in Cohort: {availability_percent:.2f}%")
        waveform_counts.append({"channel": track_name, "count": remaining_cases})

    waveform_df = df[df["caseid"].isin(valid_cases_in_cohort)].copy()
    return waveform_df, waveform_counts


def _derive_shared_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    cohort_df = df.copy()

    print("\n--- Deriving Shared Outcomes ---")

    if "icu_days" in cohort_df.columns and "los_icu" not in cohort_df.columns:
        print("Mapping 'icu_days' to 'los_icu'...")
        cohort_df["los_icu"] = cohort_df["icu_days"]

    if "los_postop" not in cohort_df.columns:
        print("Calculating 'los_postop' from 'dis' and 'opend'...")
        cohort_df["los_postop"] = (cohort_df["dis"] - cohort_df["opend"]) / 86400.0

    cohort_df["y_inhosp_mortality"] = cohort_df["death_inhosp"].astype("Int64")
    print(
        "In-hospital mortality (y_inhosp_mortality) counts:\n"
        f"{cohort_df['y_inhosp_mortality'].value_counts(dropna=False)}"
    )

    cohort_df["icu_los_days"] = cohort_df["los_icu"]
    icu_admit = np.where(
        cohort_df["los_icu"].notna(),
        (cohort_df["los_icu"] > 0).astype("int8"),
        np.nan,
    )
    cohort_df["y_icu_admit"] = pd.Series(icu_admit, index=cohort_df.index).astype("Int64")
    print(f"ICU admission (y_icu_admit) counts:\n{cohort_df['y_icu_admit'].value_counts(dropna=False)}")

    cohort_df["los_postop_days"] = cohort_df["los_postop"]
    los_q75 = cohort_df["los_postop"].dropna().quantile(0.75)
    print(f"Prolonged LOS Threshold (75th percentile): {los_q75:.2f} days")
    prolonged_los = np.where(
        cohort_df["los_postop"].notna(),
        (cohort_df["los_postop"] >= los_q75).astype("int8"),
        np.nan,
    )
    cohort_df["y_prolonged_los_postop"] = pd.Series(
        prolonged_los,
        index=cohort_df.index,
    ).astype("Int64")
    print(
        "Prolonged LOS (y_prolonged_los_postop) counts:\n"
        f"{cohort_df['y_prolonged_los_postop'].value_counts(dropna=False)}"
    )

    return cohort_df


def _annotate_non_aki_eligibility(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    annotated["eligible_icu_admission"] = annotated["y_icu_admit"].notna().astype("Int64")
    annotated["eligible_mortality"] = annotated["y_inhosp_mortality"].notna().astype("Int64")
    annotated["eligible_extended_los"] = annotated["y_prolonged_los_postop"].notna().astype("Int64")
    return annotated


def _build_aki_flow_counts(
    department_df: pd.DataFrame,
    cr_labs: pd.DataFrame,
) -> dict[str, object]:
    aki_flow_counts: dict[str, object] = {"total_cases": len(pd.read_csv(INPUT_FILE))}

    if DEPARTMENTS is not None:
        aki_flow_counts["departments"] = {"count": len(department_df), "departments": DEPARTMENTS}
    else:
        aki_flow_counts["departments"] = {"count": len(department_df)}

    mandatory_df = department_df.dropna(subset=AKI_COHORT_FLOW_MANDATORY_COLUMNS).copy()
    aki_flow_counts["mandatory_columns"] = {
        "count": len(mandatory_df),
        "columns": AKI_COHORT_FLOW_MANDATORY_COLUMNS,
    }

    custom_filter_counts: list[dict[str, object]] = []
    asa_before = len(mandatory_df)
    mandatory_df = _exclude_asa_v_vi(mandatory_df)
    custom_filter_counts.append(
        {
            "name": "Exclude ASA V/VI",
            "label": "Excluded ASA V/VI",
            "count_before": asa_before,
            "count": len(mandatory_df),
        }
    )

    waveform_df, waveform_counts = _restrict_to_required_waveforms(mandatory_df)
    annotated_df = annotate_aki_eligibility(waveform_df, cr_labs)

    preop_before = len(annotated_df)
    annotated_df = annotated_df[annotated_df["aki_baseline_not_esrd"].eq(1)].copy()
    custom_filter_counts.append(
        {
            "name": "filter_preop_cr",
            "count_before": preop_before,
            "count": len(annotated_df),
        }
    )

    postop_before = len(annotated_df)
    annotated_df = annotated_df[annotated_df["aki_has_postop_cr_7d"].eq(1)].copy()
    custom_filter_counts.append(
        {
            "name": "filter_postop_cr",
            "count_before": postop_before,
            "count": len(annotated_df),
        }
    )

    label_before = len(annotated_df)
    annotated_df = annotate_aki_labels(annotated_df, cr_labs)
    custom_filter_counts.append(
        {
            "name": "add_aki_label",
            "count_before": label_before,
            "count": len(annotated_df),
        }
    )

    aki_flow_counts["waveforms"] = waveform_counts
    aki_flow_counts["custom_filters"] = custom_filter_counts
    aki_flow_counts["final_cohort"] = {"count": len(annotated_df)}
    aki_flow_counts["label_split"] = {
        "label": "AKI label",
        "true": int((annotated_df["aki_label"] == 1).sum()),
        "false": int((annotated_df["aki_label"] == 0).sum()),
    }
    return aki_flow_counts


def main() -> None:
    clinical_df = pd.read_csv(INPUT_FILE)
    print(f"Total cases in VitalDB: {len(clinical_df)}")

    department_df = _filter_departments(clinical_df)

    broad_df = department_df.dropna(subset=OUTER_COHORT_MANDATORY_COLUMNS).copy()
    print(f"Cases after removing missing essential data: {len(broad_df)}")

    asa_before = len(broad_df)
    broad_df = _exclude_asa_v_vi(broad_df)
    print(
        f"Removed ASA V/VI cases: {asa_before - len(broad_df)} excluded; "
        f"{len(broad_df)} remain."
    )

    print("\nInitial Shared Cohort DataFrame Head:")
    print(broad_df.head())
    _print_missing_percentages(broad_df)

    broad_df, _ = _restrict_to_required_waveforms(broad_df)
    print(
        "\nYour shared outer cohort with available waveform data now has "
        f"{len(broad_df)} cases."
    )

    cr_labs = load_creatinine_labs(LAB_DATA_FILE)

    cohort_df = _derive_shared_outcomes(broad_df)
    cohort_df = annotate_aki_eligibility(cohort_df, cr_labs)
    cohort_df = annotate_aki_labels(cohort_df, cr_labs)
    cohort_df = _annotate_non_aki_eligibility(cohort_df)

    print("\nAKI eligibility counts:")
    print(cohort_df["eligible_any_aki"].value_counts(dropna=False))
    print("\nSevere AKI eligibility counts:")
    print(cohort_df["eligible_severe_aki"].value_counts(dropna=False))
    print("\nMortality eligibility counts:")
    print(cohort_df["eligible_mortality"].value_counts(dropna=False))
    print("\nICU admission eligibility counts:")
    print(cohort_df["eligible_icu_admission"].value_counts(dropna=False))

    print("\nAKI stage counts (including missing for ineligible rows):")
    print(cohort_df["aki_stage"].value_counts(dropna=False).sort_index())
    print("\nAKI label counts (including missing for ineligible rows):")
    print(cohort_df["aki_label"].value_counts(dropna=False))
    print("\nSevere AKI label counts (including missing for ineligible rows):")
    print(cohort_df["y_severe_aki"].value_counts(dropna=False))

    cohort_df.to_csv(COHORT_FILE, index=False)
    write_artifact_metadata(
        COHORT_FILE,
        artifact_role=STEP_01_COHORT_ARTIFACT,
        available_columns=cohort_df.columns,
        extra_metadata={"cohort_profile": "shared_outer"},
    )
    print(f"\nShared cohort saved to {COHORT_FILE}")

    aki_flow_counts = _build_aki_flow_counts(department_df, cr_labs)
    COHORT_FLOW_COUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with COHORT_FLOW_COUNTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(aki_flow_counts, f, indent=2)
    print(f"AKI cohort flow counts saved to {COHORT_FLOW_COUNTS_FILE}")


if __name__ == "__main__":
    main()
