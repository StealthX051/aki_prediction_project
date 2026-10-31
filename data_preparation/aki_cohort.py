from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


AKI_RELATIVE_WINDOW_SECONDS = 3 * 24 * 3600
AKI_ABSOLUTE_WINDOW_SECONDS = 48 * 3600
POSTOP_CR_WINDOW_SECONDS = AKI_RELATIVE_WINDOW_SECONDS


def _valid_creatinine_labs(cr_labs: pd.DataFrame) -> pd.DataFrame:
    """Keep usable timestamps and finite, positive creatinine results in mg/dL."""
    numeric = cr_labs.copy()
    numeric["dt"] = pd.to_numeric(numeric["dt"], errors="coerce")
    numeric["result"] = pd.to_numeric(numeric["result"], errors="coerce")
    valid = (
        np.isfinite(numeric["dt"])
        & np.isfinite(numeric["result"])
        & numeric["result"].gt(0)
    )
    return numeric.loc[valid].copy()


def _at_least(value: float, threshold: float) -> bool:
    # Account only for floating-point roundoff at inclusive clinical thresholds.
    return bool(value >= threshold or np.isclose(value, threshold, rtol=0, atol=1e-12))


def load_creatinine_labs(lab_data_file: Path) -> pd.DataFrame:
    lab_df = pd.read_csv(lab_data_file)
    cr_labs = lab_df[lab_df["name"] == "cr"].copy()
    cr_labs["result"] = pd.to_numeric(cr_labs["result"], errors="coerce")
    return cr_labs


def get_caseids_with_postop_creatinine(
    cohort_df: pd.DataFrame,
    cr_labs: pd.DataFrame,
) -> set[int]:
    """Return operations with a valid result strictly after surgery and by 72h."""
    operations = cohort_df[["caseid", "opend"]].copy()
    operations["opend"] = pd.to_numeric(operations["opend"], errors="coerce")
    operations = operations.loc[np.isfinite(operations["opend"])]
    merged_labs = operations.merge(
        _valid_creatinine_labs(cr_labs)[["caseid", "dt"]], on="caseid", how="inner"
    )
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

    valid_baseline = np.isfinite(baseline_cr) & baseline_cr.gt(0)
    annotated["aki_has_baseline_cr"] = valid_baseline.fillna(False).astype("Int64")
    annotated["aki_baseline_not_esrd"] = (
        valid_baseline & baseline_cr.le(4.0)
    ).fillna(False).astype("Int64")
    # Never retain an obsolete diagnostic if the caller supplied an older table.
    annotated = annotated.drop(columns=["aki_has_postop_cr_7d"], errors="ignore")
    annotated["aki_has_postop_cr_72h"] = annotated["caseid"].isin(postop_caseids).astype("Int64")

    eligible = (
        annotated["aki_has_baseline_cr"].eq(1)
        & annotated["aki_baseline_not_esrd"].eq(1)
        & annotated["aki_has_postop_cr_72h"].eq(1)
    )
    annotated["eligible_any_aki"] = eligible.astype("Int64")
    annotated["eligible_severe_aki"] = eligible.astype("Int64")
    return annotated


def _get_aki_stage(row: pd.Series, cr_labs_df: pd.DataFrame) -> int | None:
    baseline_cr = pd.to_numeric(row["preop_cr"], errors="coerce")
    opend = pd.to_numeric(row["opend"], errors="coerce")
    if (
        pd.isna(baseline_cr)
        or not np.isfinite(baseline_cr)
        or not 0 < baseline_cr <= 4.0
        or pd.isna(opend)
        or not np.isfinite(opend)
    ):
        return None

    case_labs = _valid_creatinine_labs(cr_labs_df[cr_labs_df["caseid"] == row["caseid"]])
    postop_labs = case_labs[
        (case_labs["dt"] > opend)
        & (case_labs["dt"] <= opend + AKI_RELATIVE_WINDOW_SECONDS)
    ]
    if postop_labs.empty:
        return None

    labs_48h = postop_labs[postop_labs["dt"] <= opend + AKI_ABSOLUTE_WINDOW_SECONDS]
    max_cr_72h = postop_labs["result"].max()
    relative_rise = max_cr_72h / baseline_cr

    has_absolute_rise = not labs_48h.empty and _at_least(labs_48h["result"].max() - baseline_cr, 0.3)
    has_aki = _at_least(relative_rise, 1.5) or has_absolute_rise
    if not has_aki:
        return 0

    if _at_least(relative_rise, 3.0) or _at_least(max_cr_72h, 4.0):
        return 3

    if _at_least(relative_rise, 2.0):
        return 2

    return 1


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
