"""Shared manuscript populations and operation-level roster identities.

These selectors run before feature selection, so clinical and waveform feature
sets use the same operation roster within each processing branch.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


COHORT_SCOPES = ("main_shared", "icu_waveform_sensitivity")


def _validate_identities(df: pd.DataFrame) -> None:
    missing = [column for column in ("caseid", "subjectid") if column not in df]
    if missing:
        raise ValueError(f"Cohort is missing identity columns: {missing}")
    if df[["caseid", "subjectid"]].isna().any().any():
        raise ValueError("Cohort requires nonmissing caseid and subjectid.")
    if df["caseid"].duplicated().any():
        raise ValueError("Cohort requires one row per operation; duplicate caseid found.")


def select_cohort(df: pd.DataFrame, scope: str, outcome: str | None = None) -> pd.DataFrame:
    """Select a manuscript scope without discarding repeat operations or NaNs."""
    if scope not in COHORT_SCOPES:
        raise ValueError(f"Unknown cohort scope {scope!r}; expected one of {COHORT_SCOPES}.")
    if scope == "icu_waveform_sensitivity" and outcome not in (None, "icu_admission"):
        raise ValueError("icu_waveform_sensitivity only supports icu_admission.")
    if outcome not in (None, "any_aki", "icu_admission"):
        raise ValueError(f"Outcome {outcome!r} is outside the manuscript comparison grid.")
    _validate_identities(df)
    required = ["eligible_icu_admission", "y_icu_admit"]
    if scope == "main_shared":
        required += ["eligible_any_aki", "aki_label"]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Cohort scope {scope!r} is missing required columns: {missing}")

    mask = df["eligible_icu_admission"].eq(1) & df["y_icu_admit"].isin([0, 1])
    if scope == "main_shared":
        mask &= df["eligible_any_aki"].eq(1) & df["aki_label"].isin([0, 1])
    selected = df.loc[mask].copy()
    # Step 01 enforces anesthesia eligibility; enforce it again when a raw
    # cohort table is supplied directly to reporting or a model entrypoint.
    if "ane_type" in selected:
        anesthesia = selected["ane_type"].astype("string").str.strip().str.lower()
        selected = selected.loc[anesthesia.eq("general").fillna(False)].copy()
    return selected


def _canonical_id(value: object) -> str:
    # CSV replay may convert an integer identity to an integral float.
    try:
        numeric = float(value)
        if numeric.is_integer():
            return str(int(numeric))
    except (ValueError, TypeError, OverflowError):
        pass
    return str(value)


def selected_case_hash(df: pd.DataFrame) -> str:
    """Hash the operation/patient mapping independently of row order."""
    _validate_identities(df)
    identities = sorted(
        (_canonical_id(caseid), _canonical_id(subjectid))
        for caseid, subjectid in df[["caseid", "subjectid"]].itertuples(index=False, name=None)
    )
    payload = json.dumps(identities, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cohort_counts(df: pd.DataFrame) -> dict[str, object]:
    """Count operations and patients, plus observed outcome denominators/events."""
    _validate_identities(df)
    outcomes = {}
    for outcome, column in (("any_aki", "aki_label"), ("icu_admission", "y_icu_admit")):
        if column in df:
            valid = df[column].isin([0, 1])
            outcomes[outcome] = {
                "operations": int(valid.sum()),
                "events": int(df.loc[valid, column].eq(1).sum()),
                "missing": int((~valid).sum()),
            }
    return {
        "operations": int(len(df)),
        "patients": int(df["subjectid"].nunique()),
        "outcomes": outcomes,
    }


def build_cohort_manifest(df: pd.DataFrame, scope: str, branch: str | None = None) -> dict[str, object]:
    """Describe the selected population; never add audit fields to predictors."""
    selected = select_cohort(df, scope)
    return {
        "cohort_scope": scope,
        "branch": branch,
        "selected_case_hash": selected_case_hash(selected),
        "caseids": sorted(selected["caseid"].tolist()),
        **cohort_counts(selected),
    }
