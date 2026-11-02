"""Admission auditing must not infer negatives or merge unrelated patients."""
import json

import pandas as pd
import pytest

from scripts.audit_inspire_linkage import (
    creatinine_surveillance, link_cases, source_icu_status, summarize, timing_audit,
)


def test_death_timing_is_scoped_to_the_operation_admission():
    frame = pd.DataFrame({
        "admission_time": [0, 100, 0, 0, 200, 0],
        "discharge_time": [50, 200, 50, 50, 100, 50],
        "opend_time": [10, 120, 10, 10, 210, 10],
        "inhosp_death_time": [150, 150, None, 10, 220, 50],
        "icuin_time": [20, None, None, 5, None, 70],
    })
    audit = timing_audit(frame)
    assert audit.mortality_timing.tolist() == [
        "outside_operation_admission", "after_operation_within_admission",
        "no_recorded_time", "at_or_before_operation_end",
        "invalid_or_missing_stay_chronology", "after_operation_within_admission",
    ]
    assert audit.icu_timing.iloc[[0, 2, 3, 5]].tolist() == [
        "within_24h_after_operation_end", "no_recorded_time",
        "before_operation_end", "outside_operation_admission",
    ]


def test_unlinked_operations_are_excluded_before_case_cardinality_check():
    clinical = pd.DataFrame({"caseid": [1, 2], "subjectid": [7, 8]})
    operations = pd.DataFrame({"op_id": [41, 42, 43], "case_id": [1, None, None],
                               "subject_id": [8, 99, 98], "hadm_id": [21, 22, 23]})
    result = link_cases(clinical, operations)
    assert len(result) == 2
    assert result._merge.tolist() == ["both", "left_only"]
    # Numeric patient ID equality across releases has no linking significance.
    assert result.subject_id.iloc[0] == 8
    operations.loc[1, "case_id"] = 1
    with pytest.raises(ValueError, match="duplicate linked case"):
        link_cases(clinical, operations)


def test_crosswalk_rejects_patient_or_admission_identity_conflicts():
    clinical = pd.DataFrame({"caseid": [1, 2], "subjectid": [7, 8]})
    operations = pd.DataFrame({"op_id": [41, 42], "case_id": [1, 2],
                               "subject_id": [100, 100], "hadm_id": [21, 22]})
    with pytest.raises(ValueError, match="Inconsistent crosswalk"):
        link_cases(clinical, operations)
    operations.subject_id = [100, 101]
    operations.hadm_id = [21, 21]
    with pytest.raises(ValueError, match="Inconsistent crosswalk"):
        link_cases(clinical, operations)


@pytest.mark.parametrize("linked", [False, True])
def test_summary_undefined_statistics_are_strict_json_null(linked):
    clinical = pd.DataFrame({
        "caseid": [1], "subjectid": [7], "sex": ["M"], "asa": [2],
        "age": [None], "death_inhosp": [0], "icu_days": [0.],
        "opstart": [0.], "opend": [60.],
    })
    operations = pd.DataFrame({
        "op_id": [41], "case_id": [1 if linked else 999],
        "subject_id": [100], "hadm_id": [21], "sex": ["M"], "asa": [2],
        "age": [40.], "admission_time": [0.], "discharge_time": [100.],
        "opstart_time": [0.], "opend_time": [1.],
        "inhosp_death_time": [float("nan")], "icuin_time": [float("nan")],
    })
    result = summarize(link_cases(clinical, operations))
    assert result["all"]["median_numeric_age"] is None
    assert result["linked"]["operations"] == int(linked)
    assert result["median_absolute_operation_duration_difference_minutes"] == (0. if linked else None)
    # A missing age or absent linkage is legitimate audit evidence, not an
    # invalid JSON NaN token. NumPy integer counts retain their numeric values.
    saved = json.loads(json.dumps(result, allow_nan=False, default=lambda value: value.item()))
    assert saved["all"]["operations"] == 1
    assert saved["all"]["median_numeric_age"] is None


def test_nonfinite_times_and_invalid_icu_los_remain_unascertained():
    frame = pd.DataFrame({"admission_time": [0, 0], "opend_time": [10, 10],
                          "discharge_time": [float("inf"), 30],
                          "inhosp_death_time": [20, float("inf")], "icuin_time": [20, float("inf")]})
    audit = timing_audit(frame)
    assert audit.mortality_timing.tolist() == ["invalid_or_missing_stay_chronology", "invalid_recorded_time"]
    assert audit.icu_timing.tolist() == ["invalid_or_missing_stay_chronology", "invalid_recorded_time"]
    assert source_icu_status(pd.DataFrame({"icu_days": [0, 1, -1, None, float("inf")]})).tolist() == [
        "negative", "positive", "unascertained", "unascertained", "unascertained"]


def test_creatinine_windows_exclude_operation_end_and_stop_at_discharge(tmp_path):
    path = tmp_path / "labs.csv"
    pd.DataFrame({"subject_id": [100] * 8, "item_name": ["creatinine"] * 8,
                  "chart_time": [10, 11, 2890, 2891, 3000, 3001, 12, 13],
                  "value": [1, 1, 1, 1, 1, 1, -1, float("inf")]}).to_csv(path, index=False)
    frame = pd.DataFrame({"_merge": ["both"], "subject_id": [100],
                          "admission_time": [0], "opend_time": [10], "discharge_time": [3000]})
    result = creatinine_surveillance(frame, path)
    windows = result["observed_postoperative_tests_within_stay"]
    assert windows["48"] == {"operations_with_any": 1, "operations_with_multiple": 1, "operation_test_pairs": 2}
    assert windows["72"]["operation_test_pairs"] == 4
    # A second operation ending after the tests has no observed follow-up.
    frame.opend_time = 3000
    result = creatinine_surveillance(frame, path)
    assert result["observed_postoperative_tests_within_stay"]["168"]["operations_with_any"] == 0
