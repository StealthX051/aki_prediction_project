from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


POSTOP_CR_WINDOW_SECONDS = 7 * 24 * 3600
AKI_RELATIVE_WINDOW_SECONDS = 3 * 24 * 3600
AKI_ABSOLUTE_WINDOW_SECONDS = 48 * 3600


def load_creatinine_labs(lab_data_file: Path) -> pd.DataFrame:
    lab_df = pd.read_csv(lab_data_file)
    cr_labs = lab_df[lab_df["name"] == "cr"].copy()
    cr_labs["result"] = pd.to_numeric(cr_labs["result"], errors="coerce")
    return cr_labs


def get_caseids_with_postop_creatinine(
    cohort_df: pd.DataFrame,
    cr_labs: pd.DataFrame,
) -> set[int]:
    merged_labs = cohort_df[["caseid", "opend"]].merge(cr_labs[["caseid", "dt"]], on="caseid", how="left")
    postop_labs = merged_labs[
        (merged_labs["dt"] > merged_labs["opend"])
        & (merged_labs["dt"] <= merged_labs["opend"] + POSTOP_CR_WINDOW_SECONDS)
    ]
    return set(postop_labs["caseid"].dropna().astype(int))


def annotate_aki_eligibility(
    cohort_df: pd.DataFrame,
    cr_labs: pd.DataFrame,
) -> pd.DataFrame:
    annotated = cohort_df.copy()
    baseline_cr = pd.to_numeric(annotated["preop_cr"], errors="coerce")
    postop_caseids = get_caseids_with_postop_creatinine(annotated, cr_labs)

    annotated["aki_has_baseline_cr"] = baseline_cr.notna().astype("Int64")
    annotated["aki_baseline_not_esrd"] = ((baseline_cr <= 4.0) & baseline_cr.notna()).astype("Int64")
    annotated["aki_has_postop_cr_7d"] = annotated["caseid"].isin(postop_caseids).astype("Int64")

    eligible = (
        annotated["aki_has_baseline_cr"].eq(1)
        & annotated["aki_baseline_not_esrd"].eq(1)
        & annotated["aki_has_postop_cr_7d"].eq(1)
    )
    annotated["eligible_any_aki"] = eligible.astype("Int64")
    annotated["eligible_severe_aki"] = eligible.astype("Int64")
    return annotated


def _get_aki_stage(row: pd.Series, cr_labs_df: pd.DataFrame) -> int:
    case_labs = cr_labs_df[cr_labs_df["caseid"] == row["caseid"]]
    postop_labs = case_labs[case_labs["dt"] > row["opend"]]

    if postop_labs.empty:
        return 0

    baseline_cr = pd.to_numeric(row["preop_cr"], errors="coerce")
    if pd.isna(baseline_cr) or baseline_cr <= 0:
        return 0

    labs_48h = postop_labs[postop_labs["dt"] <= row["opend"] + AKI_ABSOLUTE_WINDOW_SECONDS]
    max_cr_48h = labs_48h["result"].max() if not labs_48h.empty else 0

    labs_3d = postop_labs[postop_labs["dt"] <= row["opend"] + AKI_RELATIVE_WINDOW_SECONDS]
    max_cr_3d = labs_3d["result"].max() if not labs_3d.empty else 0

    if (max_cr_3d / baseline_cr >= 3.0) or (max_cr_3d >= 4.0):
        return 3

    if max_cr_3d / baseline_cr >= 2.0:
        return 2

    if (max_cr_3d / baseline_cr >= 1.5) or ((max_cr_48h - baseline_cr) >= 0.3):
        return 1

    return 0


def annotate_aki_labels(
    cohort_df: pd.DataFrame,
    cr_labs: pd.DataFrame,
    *,
    eligibility_col: str = "eligible_any_aki",
) -> pd.DataFrame:
    annotated = cohort_df.copy()
    aki_stage = pd.Series(pd.NA, index=annotated.index, dtype="Int64")

    eligible_mask = annotated[eligibility_col].eq(1)
    eligible_df = annotated.loc[eligible_mask]

    if not eligible_df.empty:
        tqdm.pandas(desc="Calculating AKI Stages")
        eligible_stages = eligible_df.progress_apply(
            lambda row: _get_aki_stage(row, cr_labs),
            axis=1,
        ).astype("Int64")
        aki_stage.loc[eligible_stages.index] = eligible_stages

    annotated["aki_stage"] = aki_stage
    annotated["aki_label"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["y_severe_aki"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")

    valid_stage_mask = annotated["aki_stage"].notna()
    annotated.loc[valid_stage_mask, "aki_label"] = (
        annotated.loc[valid_stage_mask, "aki_stage"] >= 1
    ).astype("Int64")
    annotated.loc[valid_stage_mask, "y_severe_aki"] = (
        annotated.loc[valid_stage_mask, "aki_stage"] >= 2
    ).astype("Int64")

    return annotated
