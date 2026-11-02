from pathlib import Path

import pandas as pd
import pytest

from data_preparation.aki_cohort import annotate_aki_eligibility, annotate_aki_labels
from data_preparation.artifact_metadata import (
    ARTIFACT_SCHEMA_VERSIONS,
    STEP_03_PREOP_ARTIFACT,
    ArtifactCompatibilityError,
    load_artifact_metadata,
    metadata_path_for,
    read_versioned_csv,
    write_artifact_metadata,
)
from data_preparation.step_03_preop_prep import PREOP_FEATURES_TO_SELECT, assign_outcome_specific_splits
from data_preparation.step_01_cohort_construction import _derive_shared_outcomes, _annotate_non_aki_eligibility
from data_preparation.step_05_data_merge import merge_and_save
from model_creation import utils
from model_creation.validation import select_modeling_dataset


@pytest.mark.parametrize("duration", [None, pd.NA, "unknown", -1, float("inf"), float("-inf")])
def test_invalid_icu_duration_preserves_unascertainable_outcome(duration):
    cohort = pd.DataFrame({"los_icu": [duration], "death_inhosp": [0], "los_postop": [2]}, index=[7])
    result = _annotate_non_aki_eligibility(_derive_shared_outcomes(cohort))
    assert pd.isna(result.loc[7, "icu_los_days"])
    assert pd.isna(result.loc[7, "y_icu_admit"])
    assert result.loc[7, "eligible_icu_admission"] == 0


def test_finite_nonnegative_icu_durations_preserve_binary_outcome():
    cohort = pd.DataFrame({"icu_days": [0, 0.5, "2"], "death_inhosp": [0] * 3, "los_postop": [2] * 3})
    result = _annotate_non_aki_eligibility(_derive_shared_outcomes(cohort))
    assert result["y_icu_admit"].tolist() == [0, 1, 1]
    assert result["eligible_icu_admission"].tolist() == [1, 1, 1]
    assert result["icu_los_days"].tolist() == [0, 0.5, 2]


@pytest.mark.parametrize(
    "baseline,hours,result,expected_stage",
    [
        (1.0, 0, 3.0, None),
        (1.0, -1, 3.0, None),
        (1.0, 48, 1.3, 1),
        (0.9, 48, 1.2, 1),  # 0.3 rise despite binary floating-point roundoff
        (1.1, 72, 1.65, 1),
        (1.0, 48 + 1 / 3600, 1.3, 0),
        (1.0, 72, 1.5, 1),
        (1.0, 72 + 1 / 3600, 3.0, None),
        (1.0, 120, 3.0, None),
        (1.0, 72, 2.0, 2),
        (1.0, 72, 3.0, 3),
        (3.7, 48, 4.0, 3),
        (3.7, 49, 4.0, 0),  # no acute-rise criterion is met
        (4.0, 24, 4.0, 0),
        (1.0, 24, 1.299999, 0),
        (1.0, 72, 1.499999, 0),
        (1.0, 24, "1.6", 1),
        (1.0, 24, None, None),
        (1.0, 24, "not measured", None),
        (1.0, 24, "<0.30", None),
        (1.0, 24, float("inf"), None),
        (1.0, 24, float("-inf"), None),
        (1.0, 24, 0, None),
        (1.0, 24, -0.1, None),
        (None, 24, 1.6, None),
        (pd.NA, 24, 1.6, None),
        ("invalid", 24, 1.6, None),
        (float("inf"), 24, 1.6, None),
        (float("-inf"), 24, 1.6, None),
        (0, 24, 1.6, None),
        (-1.0, 24, 1.6, None),
        (4.01, 24, 6.0, None),
    ],
)
def test_aki_72h_contract(baseline, hours, result, expected_stage):
    # Use a nonzero operation end and non-default index to check both alignments.
    cohort = pd.DataFrame({"caseid": [1], "opend": [100], "preop_cr": [baseline]}, index=[7])
    labs = pd.DataFrame({"caseid": [1], "dt": [100 + hours * 3600], "result": [result]})
    annotated = annotate_aki_labels(annotate_aki_eligibility(cohort, labs), labs)

    expected_eligible = int(expected_stage is not None)
    assert annotated.loc[7, "eligible_any_aki"] == expected_eligible
    assert annotated.loc[7, "eligible_severe_aki"] == expected_eligible
    assert "aki_has_postop_cr_72h" in annotated
    assert "aki_has_postop_cr_7d" not in annotated
    if expected_stage is None:
        assert annotated.loc[7, ["aki_stage", "aki_label", "y_severe_aki"]].isna().all()
    else:
        assert annotated.loc[7, "aki_stage"] == expected_stage
        assert annotated.loc[7, "aki_label"] == int(expected_stage >= 1)
        assert annotated.loc[7, "y_severe_aki"] == int(expected_stage >= 2)


@pytest.mark.parametrize("invalid_time", [None, "invalid", float("inf"), float("-inf")])
@pytest.mark.parametrize("time_column", ["dt", "opend"])
def test_aki_rejects_invalid_times(invalid_time, time_column):
    cohort = pd.DataFrame({"caseid": [1], "opend": [100], "preop_cr": [1.0]})
    labs = pd.DataFrame({"caseid": [1], "dt": [200], "result": [1.6]})
    if time_column == "opend":
        cohort[time_column] = invalid_time
    else:
        labs[time_column] = invalid_time

    annotated = annotate_aki_labels(annotate_aki_eligibility(cohort, labs), labs)
    assert annotated.loc[0, "eligible_any_aki"] == 0
    assert annotated.loc[0, ["aki_stage", "aki_label", "y_severe_aki"]].isna().all()


def test_aki_uses_only_valid_in_window_results_and_retains_repeat_operations():
    cohort = pd.DataFrame(
        {"caseid": [1, 2, 3], "subjectid": [101, 101, 102], "opend": [100] * 3,
         "preop_cr": [1.0] * 3, "aki_has_postop_cr_7d": [1] * 3}
    )
    labs = pd.DataFrame(
        {"caseid": [1, 1, 1, 2, 2, 3],
         "dt": [200, 300, 100 + 120 * 3600, 100 + 72 * 3600, 200, 100 + 120 * 3600],
         "result": [1.0, float("inf"), 5.0, 2.0, "invalid", 3.0]}
    )
    original_cohort = cohort.copy(deep=True)
    original_labs = labs.copy(deep=True)
    annotated = annotate_aki_labels(annotate_aki_eligibility(cohort, labs), labs)

    assert annotated["caseid"].tolist() == [1, 2, 3]
    assert annotated["subjectid"].tolist() == [101, 101, 102]
    assert annotated["aki_stage"].iloc[:2].tolist() == [0, 2]
    assert pd.isna(annotated.loc[2, "aki_stage"])
    assert "aki_has_postop_cr_7d" not in annotated
    pd.testing.assert_frame_equal(cohort, original_cohort)
    pd.testing.assert_frame_equal(labs, original_labs)

    diagnostic_columns = {col for col in annotated if col.startswith("aki_")} - {"aki_label"}
    assert diagnostic_columns.isdisjoint(PREOP_FEATURES_TO_SELECT)


def test_aki_labels_do_not_turn_stale_eligibility_into_negative_outcomes():
    cohort = pd.DataFrame(
        {"caseid": [1, 2], "opend": [100, 100], "preop_cr": [1.0, 0],
         "eligible_any_aki": [1, 1]}
    )
    labs = pd.DataFrame({"caseid": [1, 2], "dt": [100 + 120 * 3600, 200], "result": [2.0, 1.0]})
    annotated = annotate_aki_labels(cohort, labs)
    assert annotated[["aki_stage", "aki_label", "y_severe_aki"]].isna().all().all()


def test_aki_without_creatinine_records_remains_unascertained():
    cohort = pd.DataFrame({"caseid": [1], "opend": [100], "preop_cr": [1.0]})
    labs = pd.DataFrame(columns=["caseid", "dt", "result"])
    annotated = annotate_aki_labels(annotate_aki_eligibility(cohort, labs), labs)
    assert annotated.loc[0, "eligible_any_aki"] == 0
    assert annotated.loc[0, ["aki_stage", "aki_label", "y_severe_aki"]].isna().all()


@pytest.mark.parametrize("dtype", ["Float64", "Int64"])
def test_aki_day_three_only_nullable_results_allow_an_observed_negative(dtype):
    cohort = pd.DataFrame({"caseid": [1], "opend": [100], "preop_cr": [1.0]})
    labs = pd.DataFrame(
        {"caseid": [1, 1], "dt": [200, 100 + 60 * 3600],
         "result": pd.Series([pd.NA, 1], dtype=dtype)}
    )
    annotated = annotate_aki_labels(annotate_aki_eligibility(cohort, labs), labs)
    assert annotated.loc[0, "eligible_any_aki"] == 1
    assert annotated.loc[0, ["aki_stage", "aki_label", "y_severe_aki"]].tolist() == [0, 0, 0]


@pytest.mark.parametrize("artifact_role", list(ARTIFACT_SCHEMA_VERSIONS))
def test_72h_contract_rejects_version_one_artifacts(artifact_role, tmp_path):
    import json

    artifact_path = tmp_path / "artifact.csv"
    pd.DataFrame({"caseid": [1]}).to_csv(artifact_path, index=False)
    metadata_path_for(artifact_path).write_text(
        json.dumps({"artifact_role": artifact_role, "schema_version": 1,
                    "available_columns": ["caseid"]}), encoding="utf-8"
    )
    expected_version = ARTIFACT_SCHEMA_VERSIONS[artifact_role]
    with pytest.raises(ArtifactCompatibilityError, match=f"expected schema_version {expected_version} but found 1"):
        read_versioned_csv(artifact_path, artifact_role=artifact_role, required_columns=["caseid"])

    write_artifact_metadata(artifact_path, artifact_role=artifact_role, available_columns=["caseid"])
    frame, metadata = read_versioned_csv(
        artifact_path, artifact_role=artifact_role, required_columns=["caseid"]
    )
    assert metadata["schema_version"] == expected_version
    assert frame["caseid"].tolist() == [1]


def test_aki_flow_counts_and_labels_use_valid_72h_followup(tmp_path, monkeypatch):
    from data_preparation import step_01_cohort_construction as step_01
    from reporting.cohort_flow import normalize_counts

    cohort = pd.DataFrame(
        {"caseid": [1, 2, 3, 4, 5], "subjectid": [101, 101, 102, 103, 104], "opend": [100] * 5,
         "preop_cr": [1.0, 1.0, 0, 1.0, None], "asa": [2.0, None, 2.0, 2.0, 2.0],
         "ane_type": ["General"] * 5}
    )
    input_path = tmp_path / "clinical.csv"
    cohort.to_csv(input_path, index=False)
    monkeypatch.setattr(step_01, "INPUT_FILE", input_path)
    monkeypatch.setattr(step_01, "DEPARTMENTS", None)
    monkeypatch.setattr(step_01, "_restrict_to_required_waveforms", lambda df: (df, []))
    labs = pd.DataFrame(
        {"caseid": [1, 2, 3, 4], "dt": [200, 100 + 120 * 3600, 200, 200],
         "result": [1.6, 1.0, 1.6, "invalid"]}
    )

    counts = step_01._build_aki_flow_counts(cohort, labs)
    filters = {entry["name"]: entry for entry in counts["custom_filters"]}
    assert filters["filter_valid_baseline_cr"]["count"] == 3
    assert filters["filter_postop_cr_72h"]["count"] == 1
    assert counts["final_cohort"]["count"] == 1
    assert counts["label_split"] == {"label": "AKI label", "true": 1, "false": 0}
    stages = normalize_counts(counts).stages
    assert any("Documented operation end time" in (stage.detail or "") for stage in stages)
    stage = next(stage for stage in stages if "72 Hours" in stage.title)
    assert stage.removal_reason == "No valid postoperative creatinine within 72 hours"


def test_assign_outcome_specific_splits_only_populates_eligible_rows():
    df = pd.DataFrame(
        {
            "caseid": list(range(1, 11)),
            "subjectid": list(range(101, 111)),
            "aki_label": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, None, None], dtype="Int64"),
            "y_severe_aki": pd.Series([0, 1, 0, 1, 0, 1, 0, 1, None, None], dtype="Int64"),
            "y_inhosp_mortality": pd.Series([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype="Int64"),
            "y_icu_admit": pd.Series([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype="Int64"),
            "eligible_any_aki": pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype="Int64"),
            "eligible_severe_aki": pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 0, 0], dtype="Int64"),
            "eligible_mortality": pd.Series([1] * 10, dtype="Int64"),
            "eligible_icu_admission": pd.Series([1] * 10, dtype="Int64"),
            "eligible_extended_los": pd.Series([1] * 10, dtype="Int64"),
        }
    )

    split_df, split_status = assign_outcome_specific_splits(df)

    aki_ineligible = split_df["eligible_any_aki"] == 0
    assert split_df.loc[aki_ineligible, "split_group_any_aki"].isna().all()
    assert split_df["split_group_icu_admission"].notna().all()
    assert split_status["mortality"]["status"] == "unsupported_in_artifact"
    assert split_df["split_group_mortality"].isna().all()

    train_subjects = set(
        split_df.loc[split_df["split_group_any_aki"] == "train", "subjectid"]
    )
    test_subjects = set(
        split_df.loc[split_df["split_group_any_aki"] == "test", "subjectid"]
    )
    assert not (train_subjects & test_subjects)


def test_select_modeling_dataset_uses_outcome_specific_split_and_excludes_metadata():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4],
            "subjectid": [101, 102, 103, 104],
            "feature_real": [0.1, 0.2, 0.3, 0.4],
            "aki_label": pd.Series([1, 0, 1, None], dtype="Int64"),
            "y_icu_admit": pd.Series([0, 1, 0, 1], dtype="Int64"),
            "eligible_any_aki": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_severe_aki": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_mortality": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_icu_admission": pd.Series([1, 1, 1, 1], dtype="Int64"),
            "eligible_extended_los": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "split_group_any_aki": ["test", "test", "train", None],
            "split_group_severe_aki": ["test", "test", "train", None],
            "split_group_mortality": ["train", "train", "test", None],
            "split_group_icu_admission": ["train", "train", "test", "test"],
        }
    )

    working_df, X, y, caseids, _ = select_modeling_dataset(
        df,
        "icu_admission",
        "preop_only",
        require_holdout_split=True,
    )

    assert list(caseids) == [1, 2, 3, 4]
    assert list(working_df["split_group"]) == ["train", "train", "test", "test"]
    assert list(X.columns) == ["feature_real"]
    assert list(y) == [0, 1, 0, 1]


def test_merge_and_save_keeps_non_aki_rows_with_missing_aki_split_metadata(tmp_path: Path):
    preop_df = pd.DataFrame(
        {
            "caseid": [1, 2],
            "feature_preop": [10.0, 20.0],
            "aki_label": pd.Series([0, None], dtype="Int64"),
            "y_icu_admit": pd.Series([0, 1], dtype="Int64"),
            "eligible_any_aki": pd.Series([1, 0], dtype="Int64"),
            "eligible_severe_aki": pd.Series([1, 0], dtype="Int64"),
            "eligible_mortality": pd.Series([1, 1], dtype="Int64"),
            "eligible_icu_admission": pd.Series([1, 1], dtype="Int64"),
            "eligible_extended_los": pd.Series([1, 1], dtype="Int64"),
            "split_group": ["train", None],
            "split_group_any_aki": ["train", None],
            "split_group_severe_aki": ["train", None],
            "split_group_mortality": ["train", "test"],
            "split_group_icu_admission": ["train", "test"],
        }
    )
    intraop_path = tmp_path / "intraop.csv"
    output_path = tmp_path / "merged.csv"
    pd.DataFrame({"caseid": [1, 2], "waveform_feature": [0.1, 0.2]}).to_csv(intraop_path, index=False)

    merge_and_save(preop_df, intraop_path, output_path, "Test Features", impute_missing=False)

    merged_df = pd.read_csv(output_path)
    assert list(merged_df["caseid"]) == [1, 2]
    assert pd.isna(merged_df.loc[merged_df["caseid"] == 2, "split_group_any_aki"]).all()
    assert merged_df.loc[merged_df["caseid"] == 2, "split_group_icu_admission"].iloc[0] == "test"
    assert load_artifact_metadata(output_path)["upstream_schema_version"] == ARTIFACT_SCHEMA_VERSIONS[
        STEP_03_PREOP_ARTIFACT
    ]


def test_load_data_rejects_stale_merged_artifact_without_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    stale_path = tmp_path / "stale.csv"
    # Missing provenance must fail before any clinical columns are trusted.
    pd.DataFrame({"caseid": [1]}).to_csv(stale_path, index=False)

    monkeypatch.setattr(utils, "FULL_FEATURES_FILE", stale_path)

    with pytest.raises(ArtifactCompatibilityError, match="metadata sidecar is missing"):
        utils.load_data("non_windowed")


@pytest.mark.parametrize("branch,unexpected", [
    pytest.param("non_windowed", "icu_los_days", id="icu-outcome-metadata"),
    pytest.param("windowed", "los_postop_days", id="postoperative-duration"),
    pytest.param("non_windowed", "future_lab_result", id="unregistered-clinical-field"),
])
def test_production_loader_rejects_unregistered_predictors(monkeypatch, tmp_path, branch, unexpected):
    path = tmp_path / "merged.csv"
    path.touch()
    frame = pd.DataFrame({
        "caseid": [1], "subjectid": [101], "age": [50], "sex": ["F"],
        "preop_cr": [1.0], "inpatient_preop": [1], "SNUADC_PLETH_feature": [0.1],
        "aki_has_postop_cr_72h": [1], "postop_measurement_count": [2],
        "aki_label": [0], "eligible_any_aki": [1], "split_group_any_aki": ["train"],
    })
    monkeypatch.setattr(utils, "FULL_FEATURES_FILE", path)
    monkeypatch.setattr(utils, "WINDOWED_FEATURES_FILE", path)
    monkeypatch.setattr(utils, "read_versioned_csv", lambda *args, **kwargs: (frame, {"schema_version": 3}))
    accepted = utils.load_data(branch)
    assert utils.get_feature_sets(accepted)["preop_only"] == ["age", "sex", "preop_cr", "inpatient_preop"]
    frame[unexpected] = 1
    with pytest.raises(ArtifactCompatibilityError, match=f"Unapproved preoperative predictor columns.*{unexpected}"):
        utils.load_data(branch)


def test_select_modeling_dataset_rejects_unsupported_holdout_outcome_from_metadata():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "subjectid": [101, 102, 103],
            "feature_real": [0.1, 0.2, 0.3],
            "y_inhosp_mortality": pd.Series([0, 1, 0], dtype="Int64"),
            "eligible_mortality": pd.Series([1, 1, 1], dtype="Int64"),
            "split_group_mortality": [None, None, None],
        }
    )
    df.attrs["artifact_metadata"] = {
        "split_status": {
            "mortality": {
                "status": "unsupported_in_artifact",
                "reason": "Need at least two patient groups in each class to create a grouped holdout split.",
            }
        }
    }

    with pytest.raises(ValueError, match="unsupported in this processed artifact"):
        select_modeling_dataset(
            df,
            "mortality",
            "preop_only",
            require_holdout_split=True,
        )
