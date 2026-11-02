"""Audit INSPIRE admission linkage without deriving labels or changing study files.

Run with the project Conda Python. Explicit input paths avoid importing pipeline
configuration. Output defaults to aggregate JSON. The optional case crosswalk
is private audit metadata and must be kept outside Git and study predictors.
INSPIRE times are minutes from the patient's initial admission, whereas VitalDB
times are seconds from each case's recording start.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_or_none(value):
    """Represent undefined aggregate statistics as JSON null, never NaN/Inf."""
    return value if pd.notna(value) and np.isfinite(value) else None


def link_cases(clinical: pd.DataFrame, operations: pd.DataFrame) -> pd.DataFrame:
    """Validate case cardinality and the cross-release patient crosswalk."""
    linked = operations.loc[operations.case_id.notna()]
    if clinical.caseid.isna().any() or not clinical.caseid.is_unique:
        raise ValueError("VitalDB case IDs must be present and unique")
    if not operations.op_id.is_unique or operations.op_id.isna().any():
        raise ValueError("INSPIRE operation IDs must be present and unique")
    if not linked.case_id.is_unique:
        raise ValueError("INSPIRE contains duplicate linked case IDs")
    merged = clinical.merge(linked, left_on="caseid", right_on="case_id",
                            how="left", validate="one_to_one", indicator=True,
                            suffixes=("", "_inspire"))
    matched = merged.loc[merged._merge.eq("both")]
    if matched[["subjectid", "subject_id", "hadm_id"]].isna().any().any():
        raise ValueError("Matched case has a missing patient or admission ID")
    for left, right in [("subjectid", "subject_id"), ("subject_id", "subjectid"),
                        ("hadm_id", "subject_id")]:
        if matched.groupby(left)[right].nunique().gt(1).any():
            raise ValueError(f"Inconsistent crosswalk: {left} maps to multiple {right}")
    return merged


def timing_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Return audit categories, never replacement binary outcome labels.

    A recorded death alone is insufficient: require admission membership and
    chronology. Missing or outside-admission death times do not establish survival.
    """
    result = pd.DataFrame(index=frame.index)
    death = frame.inhosp_death_time
    valid = (np.isfinite(frame.admission_time) & np.isfinite(frame.discharge_time)
             & np.isfinite(frame.opend_time) & frame.admission_time.le(frame.opend_time)
             & frame.opend_time.le(frame.discharge_time))
    within = valid & np.isfinite(death) & death.between(frame.admission_time, frame.discharge_time)
    result["mortality_timing"] = "no_recorded_time"
    result.loc[death.notna(), "mortality_timing"] = "outside_operation_admission"
    result.loc[death.notna() & ~np.isfinite(death), "mortality_timing"] = "invalid_recorded_time"
    result.loc[within & death.le(frame.opend_time), "mortality_timing"] = "at_or_before_operation_end"
    result.loc[within & death.gt(frame.opend_time), "mortality_timing"] = "after_operation_within_admission"
    result.loc[~valid, "mortality_timing"] = "invalid_or_missing_stay_chronology"
    result["icu_timing"] = "no_recorded_time"
    lag = frame.icuin_time - frame.opend_time
    result.loc[lag.notna(), "icu_timing"] = "outside_operation_admission"
    result.loc[frame.icuin_time.notna() & ~np.isfinite(frame.icuin_time), "icu_timing"] = "invalid_recorded_time"
    in_stay = valid & np.isfinite(frame.icuin_time) & frame.icuin_time.between(frame.admission_time, frame.discharge_time)
    result.loc[in_stay & lag.lt(0), "icu_timing"] = "before_operation_end"
    result.loc[in_stay & lag.between(0, 1440), "icu_timing"] = "within_24h_after_operation_end"
    result.loc[in_stay & lag.gt(1440), "icu_timing"] = "over_24h_after_operation_end"
    result.loc[~valid, "icu_timing"] = "invalid_or_missing_stay_chronology"
    return result


def source_icu_status(frame: pd.DataFrame) -> pd.Series:
    status = pd.Series("unascertained", index=frame.index)
    finite = np.isfinite(frame.icu_days)
    status.loc[finite & frame.icu_days.eq(0)] = "negative"
    status.loc[finite & frame.icu_days.gt(0)] = "positive"
    return status


def summarize(frame: pd.DataFrame) -> dict:
    matched = frame.loc[frame._merge.eq("both")].copy()
    unmatched = frame.loc[~frame._merge.eq("both")]
    audit = timing_audit(matched)
    mixed = frame.groupby("subjectid").death_inhosp.nunique().gt(1)
    mixed_matched = matched.loc[matched.subjectid.isin(mixed[mixed].index)]
    repeats = frame.loc[frame.death_inhosp.eq(1)].groupby("subjectid").size()
    linked_positive = matched.loc[matched.death_inhosp.eq(1)]
    concordance = {}
    for field in ["sex", "asa"]:
        left, right = matched[field], matched[f"{field}_inspire"]
        valid = left.notna() & right.notna()
        concordance[field] = {"comparable": int(valid.sum()),
                              "equal": int((valid & left.eq(right)).sum())}
    age = pd.to_numeric(matched.age, errors="coerce")
    other_age = pd.to_numeric(matched.age_inspire, errors="coerce")
    valid_age = age.notna() & other_age.notna()
    concordance["age"] = {"comparable": int(valid_age.sum()),
                          "difference_counts_vitaldb_minus_inspire": (age - other_age).value_counts().sort_index().to_dict(),
                          "caution": "INSPIRE ages are binned; exact age equality is not required."}
    def profile(part):
        age = pd.to_numeric(part.age, errors="coerce")
        return {"operations": len(part), "patients": part.subjectid.nunique(),
                "mortality_positive_operations": int(part.death_inhosp.eq(1).sum()),
                "mortality_positive_patients": part.loc[part.death_inhosp.eq(1), "subjectid"].nunique(),
                "icu_positive_operations": int(source_icu_status(part).eq("positive").sum()),
                "icu_unascertained_operations": int(source_icu_status(part).eq("unascertained").sum()),
                "median_numeric_age": _finite_or_none(age.median()) if age.notna().any() else None,
                "nonnumeric_or_missing_age": int(age.isna().sum())}
    return {
        "all": profile(frame), "linked": profile(matched), "unmatched": profile(unmatched),
        "cross_source_demographic_concordance": concordance,
        "linked_patients_with_unmatched_operations": int(unmatched.subjectid.isin(matched.subjectid).groupby(unmatched.subjectid).any().sum()),
        "mortality_timing_by_source_flag": pd.crosstab(
            audit.mortality_timing, matched.death_inhosp).to_dict(),
        "icu_timing_by_source_status": pd.crosstab(
            audit.icu_timing, source_icu_status(matched)).to_dict(),
        "mixed_flag_patients_total": int(mixed.sum()),
        "mixed_flag_patients_with_any_link": mixed_matched.subjectid.nunique(),
        "mixed_flag_patients_with_all_operations_linked": int(sum(
            frame.loc[frame.subjectid.eq(patient), "_merge"].eq("both").all()
            for patient in mixed[mixed].index)),
        "linked_mixed_flag_admissions": int(mixed_matched.groupby("hadm_id").death_inhosp.nunique().gt(1).sum()),
        "repeat_positive_patients_total": int(repeats.gt(1).sum()),
        "repeat_positive_patients_with_all_positive_operations_linked": int(sum(
            linked_positive.subjectid.eq(patient).sum() == count
            for patient, count in repeats[repeats.gt(1)].items())),
        "linked_positive_operations_per_admission": linked_positive.groupby("hadm_id").size().value_counts().sort_index().to_dict(),
        "admissions_with_inconsistent_discharge_times": int(matched.groupby("hadm_id").discharge_time.nunique().gt(1).sum()),
        "admissions_with_inconsistent_admission_times": int(matched.groupby("hadm_id").admission_time.nunique().gt(1).sum()),
        "admissions_with_multiple_recorded_death_times": int(matched.groupby("hadm_id").inhosp_death_time.nunique().gt(1).sum()),
        "median_absolute_operation_duration_difference_minutes": _finite_or_none((
            (matched.opend_time - matched.opstart_time)
            - (matched.opend - matched.opstart) / 60).abs().median()),
    }


def creatinine_surveillance(frame: pd.DataFrame, labs_path: Path) -> dict:
    """Count observed tests inside each operation's stay, without AKI labels."""
    matched = frame.loc[frame._merge.eq("both")]
    pieces = []
    for chunk in pd.read_csv(labs_path, chunksize=250_000):
        keep = chunk.item_name.eq("creatinine") & chunk.subject_id.isin(matched.subject_id)
        pieces.append(chunk.loc[keep, ["subject_id", "chart_time", "value"]])
    labs = pd.concat(pieces, ignore_index=True)
    valid = np.isfinite(labs.value) & labs.value.gt(0) & np.isfinite(labs.chart_time)
    times = {patient: np.sort(group.chart_time.to_numpy())
             for patient, group in labs.loc[valid].groupby("subject_id")}
    counts = {hours: [] for hours in [48, 72, 168]}
    for row in matched.itertuples():
        if not (np.isfinite([row.admission_time, row.opend_time, row.discharge_time]).all()
                and row.admission_time <= row.opend_time <= row.discharge_time):
            continue
        observed = times.get(row.subject_id, np.array([]))
        start = np.searchsorted(observed, row.opend_time, side="right")
        for hours, values in counts.items():
            end = min(row.opend_time + hours * 60, row.discharge_time)
            values.append(int(np.searchsorted(observed, end, side="right") - start))
    return {"linked_operations_with_valid_stay": len(counts[48]),
            "observed_postoperative_tests_within_stay": {
                str(hours): {"operations_with_any": sum(value > 0 for value in values),
                             "operations_with_multiple": sum(value > 1 for value in values),
                             "operation_test_pairs": sum(values)}
                for hours, values in counts.items()},
            "caution": "Counts describe recorded, binned/clipped creatinine; no AKI labels derived. Repeated operations can share tests."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", required=True, type=Path)
    parser.add_argument("--operations", required=True, type=Path)
    parser.add_argument("--labs", type=Path, help="Optional creatinine surveillance audit")
    parser.add_argument("--crosswalk", type=Path, help="Optional fresh private case-to-admission CSV")
    parser.add_argument("--roster", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--output", required=True, type=Path,
                        help="Fresh aggregate JSON path outside study results/inputs")
    args = parser.parse_args()
    paths = [args.clinical, args.operations]
    if args.labs:
        paths.append(args.labs)
    rosters = {}
    for spec in args.roster:
        name, path = spec.split("=", 1)
        if name == "source" or name in rosters:
            parser.error("Roster names must be unique and cannot be 'source'")
        rosters[name] = Path(path)
        paths.append(Path(path))
    if args.output.exists() or args.output.resolve() in {p.resolve() for p in paths}:
        parser.error("Output must be a fresh file distinct from every input")
    if args.crosswalk and (args.crosswalk.exists() or args.crosswalk.resolve() in
                           {args.output.resolve(), *(p.resolve() for p in paths)}):
        parser.error("Crosswalk must be a fresh file distinct from inputs and summary")
    before = {str(p): sha256(p) for p in paths}
    # Require the released checksum for this credentialed input before joining.
    checksums = args.operations.parent / "SHA256SUMS.txt"
    expected = dict((name.lstrip("*"), digest) for digest, name in
                    (line.split(maxsplit=1) for line in checksums.read_text().splitlines() if line.strip()))
    for path in [args.operations] + ([args.labs] if args.labs else []):
        if before[str(path)] != expected[path.name]:
            raise ValueError(f"INSPIRE checksum mismatch: {path.name}")
    clinical = pd.read_csv(args.clinical)
    operations = pd.read_csv(args.operations, low_memory=False)
    merged = link_cases(clinical, operations)
    case_ids = operations.loc[operations.case_id.notna(), "case_id"]
    identifier_inventory = {
        "operations": len(operations), "patients": operations.subject_id.nunique(),
        "admissions": operations.hadm_id.nunique(),
        "nonmissing_case_ids": len(case_ids), "unique_case_ids": case_ids.nunique(),
        "negative_case_ids": int(case_ids.lt(0).sum()),
        "zero_case_ids": int(case_ids.eq(0).sum()),
        "minimum_case_id": _finite_or_none(case_ids.min()),
        "maximum_case_id": _finite_or_none(case_ids.max()),
        "case_ids_matching_source": int(case_ids.isin(clinical.caseid).sum()),
        "case_ids_outside_source": int((~case_ids.isin(clinical.caseid)).sum()),
        "caution": "Nonmissing case_id is not proof of membership in the 6,388-case VitalDB release; signed IDs are preserved without remapping.",
    }
    summaries = {"source": summarize(merged)}
    surveillance = creatinine_surveillance(merged, args.labs) if args.labs else None
    for name, path in rosters.items():
        roster = pd.read_csv(path, usecols=["caseid", "subjectid"])
        if roster.caseid.isna().any() or not roster.caseid.is_unique:
            raise ValueError(f"Invalid case IDs in roster {name}")
        checked = roster.merge(clinical[["caseid", "subjectid"]], on="caseid",
                               how="left", validate="one_to_one", suffixes=("", "_source"))
        if not checked.subjectid.eq(checked.subjectid_source).all():
            raise ValueError(f"Case/patient identity differs in roster {name}")
        summaries[name] = summarize(merged.loc[merged.caseid.isin(roster.caseid)])
    if before != {str(p): sha256(p) for p in paths}:
        raise RuntimeError("An audit input changed during execution")
    crosswalk = None
    if args.crosswalk:
        args.crosswalk.parent.mkdir(parents=True, exist_ok=True)
        mapping = merged[["caseid", "subjectid", "op_id", "subject_id", "hadm_id", "_merge"]].copy()
        mapping = mapping.rename(columns={"_merge": "link_status"})
        for field in ["caseid", "subjectid", "op_id", "subject_id", "hadm_id"]:
            mapping[field] = mapping[field].astype("Int64")
        with args.crosswalk.open("x") as stream:
            mapping.to_csv(stream, index=False)
        crosswalk = {"path": str(args.crosswalk), "sha256": sha256(args.crosswalk),
                     "use": "Private audit metadata; not model predictors or replacement outcomes."}
    result = {"audited_at_utc": datetime.now(timezone.utc).isoformat(),
              "input_sha256": before, "operations_checksum_verified": True,
              "script_sha256": sha256(Path(__file__)),
              "inspire_identifier_inventory": identifier_inventory,
              "private_case_crosswalk": crosswalk,
              "no_study_labels_or_inputs_changed": True,
              "interpretation": "Timing categories are audit evidence, not adjudicated outcomes; missing time is not survival or no ICU admission.",
              "time_units": "INSPIRE patient-relative minutes; VitalDB case-relative seconds. Only within-source durations compared.",
              "creatinine_surveillance_source_linked_cases": surveillance,
              "summaries": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False, default=lambda value: value.item())
        stream.write("\n")
    print(f"Aggregate linkage evidence: {args.output}")


if __name__ == "__main__":
    main()
