import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preparation.aki_cohort import annotate_aki_eligibility, annotate_aki_labels
from data_preparation.step_03_preop_prep import assign_outcome_specific_splits
from data_preparation.step_05_data_merge import merge_and_save
from model_creation import utils
from model_creation.validation import select_modeling_dataset


def test_aki_eligibility_and_labels_leave_ineligible_rows_missing():
    cohort_df = pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "opend": [100, 100, 100],
            "preop_cr": [1.0, None, 5.0],
        }
    )
    cr_labs = pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "dt": [200, 200, 200],
            "result": [1.6, 2.0, 7.0],
        }
    )

    annotated = annotate_aki_eligibility(cohort_df, cr_labs)
    annotated = annotate_aki_labels(annotated, cr_labs)

    assert annotated.loc[0, "eligible_any_aki"] == 1
    assert annotated.loc[1, "eligible_any_aki"] == 0
    assert annotated.loc[2, "eligible_any_aki"] == 0

    assert annotated.loc[0, "aki_label"] == 1
    assert annotated.loc[0, "y_severe_aki"] == 0
    assert pd.isna(annotated.loc[1, "aki_label"])
    assert pd.isna(annotated.loc[2, "aki_label"])


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
            "y_icu_admit": pd.Series([0, 1, 0, None], dtype="Int64"),
            "eligible_any_aki": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_severe_aki": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_mortality": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_icu_admission": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "eligible_extended_los": pd.Series([1, 1, 1, 0], dtype="Int64"),
            "split_group_any_aki": ["test", "test", "train", None],
            "split_group_severe_aki": ["test", "test", "train", None],
            "split_group_mortality": ["train", "train", "test", None],
            "split_group_icu_admission": ["train", "train", "test", None],
        }
    )

    working_df, X, y, caseids, _ = select_modeling_dataset(
        df,
        "icu_admission",
        "preop_only",
        require_holdout_split=True,
    )

    assert list(caseids) == [1, 2, 3]
    assert list(working_df["split_group"]) == ["train", "train", "test"]
    assert list(X.columns) == ["feature_real"]
    assert list(y) == [0, 1, 0]


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


def test_load_data_rejects_stale_merged_artifact_without_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    stale_path = tmp_path / "stale.csv"
    pd.DataFrame(
        {
            "caseid": [1],
            "subjectid": [101],
            "aki_label": [0],
            "y_severe_aki": [0],
            "y_inhosp_mortality": [0],
            "y_icu_admit": [0],
            "y_prolonged_los_postop": [0],
            "eligible_any_aki": [1],
            "eligible_severe_aki": [1],
            "eligible_mortality": [1],
            "eligible_icu_admission": [1],
            "eligible_extended_los": [1],
            "split_group_any_aki": ["train"],
            "split_group_severe_aki": ["train"],
            "split_group_mortality": ["train"],
            "split_group_icu_admission": ["train"],
        }
    ).to_csv(stale_path, index=False)

    monkeypatch.setattr(utils, "FULL_FEATURES_FILE", stale_path)

    with pytest.raises(Exception, match="metadata sidecar is missing"):
        utils.load_data("non_windowed")


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
