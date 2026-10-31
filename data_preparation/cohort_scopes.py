"""Shared manuscript populations and operation-level roster identities.

These selectors run before feature selection, so clinical and waveform feature
sets use the same operation roster within each processing branch.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd


SCOPE_OUTCOMES = {
    "main_shared": ("any_aki", "icu_admission"),
    "icu_waveform_sensitivity": ("icu_admission",),
    "mortality_waveform_sensitivity": ("mortality",),
}
COHORT_SCOPES = tuple(SCOPE_OUTCOMES)
# Existing schema-v5 sidecars predate the mortality scope. Their two stored
# populations remain mandatory and unchanged; mortality is a derived overlay.
LEGACY_STORED_SCOPES = ("main_shared", "icu_waveform_sensitivity")


def scope_required_columns(scope: str) -> tuple[str, ...]:
    if scope not in SCOPE_OUTCOMES:
        raise ValueError(f"Unknown cohort scope {scope!r}; expected one of {COHORT_SCOPES}.")
    if scope == "mortality_waveform_sensitivity":
        return ("eligible_mortality", "y_inhosp_mortality")
    columns = ("eligible_icu_admission", "y_icu_admit")
    return columns + (("eligible_any_aki", "aki_label") if scope == "main_shared" else ())


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
    required = scope_required_columns(scope)
    if outcome is not None and outcome not in SCOPE_OUTCOMES[scope]:
        raise ValueError(f"{scope} only supports {', '.join(SCOPE_OUTCOMES[scope])}.")
    _validate_identities(df)
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Cohort scope {scope!r} is missing required columns: {missing}")

    if scope == "mortality_waveform_sensitivity":
        mask = df["eligible_mortality"].eq(1) & df["y_inhosp_mortality"].isin([0, 1])
    else:
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


def cohort_counts(df: pd.DataFrame, *, mortality: bool = False) -> dict[str, object]:
    """Count operations and patients, plus observed outcome denominators/events."""
    _validate_identities(df)
    outcomes = {}
    columns = (("mortality", "y_inhosp_mortality"),) if mortality else (
        ("any_aki", "aki_label"), ("icu_admission", "y_icu_admit"))
    for outcome, column in columns:
        if column in df:
            valid = df[column].isin([0, 1])
            outcomes[outcome] = {
                "operations": int(valid.sum()),
                "events": int(df.loc[valid, column].eq(1).sum()),
                "missing": int((~valid).sum()),
            }
            if mortality:
                outcomes[outcome]["patients"] = int(df.loc[valid, "subjectid"].nunique())
                outcomes[outcome]["event_patients"] = int(df.loc[valid & df[column].eq(1), "subjectid"].nunique())
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
        **cohort_counts(selected, mortality=scope == "mortality_waveform_sensitivity"),
    }
