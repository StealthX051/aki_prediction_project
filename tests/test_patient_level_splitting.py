import numpy as np
import pandas as pd
import pytest

from data_preparation.step_03_preop_prep import select_patient_level_holdout_indices
from model_creation import utils


def _multi_operation_df() -> pd.DataFrame:
    subject_ids = np.repeat(np.arange(100, 110), 2)
    labels_by_subject = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    aki_label = np.repeat(labels_by_subject, 2)

    return pd.DataFrame(
        {
            "caseid": np.arange(1, len(subject_ids) + 1),
            "subjectid": subject_ids,
            "feature1": np.linspace(0.0, 1.0, len(subject_ids)),
            "feature2": np.linspace(1.0, 2.0, len(subject_ids)),
            "aki_label": aki_label,
            "eligible_any_aki": np.ones(len(subject_ids), dtype=int),
            "split_group_any_aki": ["train"] * 16 + ["test"] * 4,
        },
        index=np.arange(200, 220) * 3,
    )


def test_select_patient_level_holdout_indices_keeps_patients_disjoint():
    df = _multi_operation_df()

    train_idx, test_idx = select_patient_level_holdout_indices(df, outcome_col="aki_label")

    train_subjects = set(df.loc[train_idx, "subjectid"])
    test_subjects = set(df.loc[test_idx, "subjectid"])

    assert not (train_subjects & test_subjects)
    assert len(test_idx) == 4
    assert set(train_idx).isdisjoint(test_idx)
    assert set(train_idx) | set(test_idx) == set(df.index)
    assert df.loc[train_idx, "aki_label"].nunique() == df.loc[test_idx, "aki_label"].nunique() == 2


def test_prepare_data_uses_persisted_outcome_specific_holdout_split():
    df = _multi_operation_df()

    X_train, X_test, y_train, y_test, _ = utils.prepare_data(
        df,
        outcome_name="any_aki",
        feature_set_name="preop_only",
    )

    train_subjects = set(df.loc[X_train.index, "subjectid"])
    test_subjects = set(df.loc[X_test.index, "subjectid"])

    assert not (train_subjects & test_subjects)
    assert len(X_train) + len(X_test) == len(df)
    assert len(y_train) + len(y_test) == len(df)
    assert set(X_train.index) == set(df.index[df.split_group_any_aki.eq("train")])
    assert set(X_test.index) == set(df.index[df.split_group_any_aki.eq("test")])
    assert set(X_train.columns) == set(X_test.columns) == {"feature1", "feature2"}
    pd.testing.assert_series_equal(y_train, df.loc[X_train.index, "aki_label"])
    pd.testing.assert_series_equal(y_test, df.loc[X_test.index, "aki_label"])


def test_prepare_data_rejects_stale_legacy_split_group_for_non_aki_outcome():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4],
            "subjectid": [100, 100, 200, 200],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "y_icu_admit": [0, 1, 0, 1],
            "eligible_icu_admission": [1, 1, 1, 1],
            "split_group": ["train", "test", "train", "test"],
        }
    )

    with pytest.raises(ValueError, match="requires persisted split column 'split_group_icu_admission'"):
        utils.prepare_data(df, outcome_name="icu_admission", feature_set_name="preop_only")


def test_prepare_data_rejects_existing_patient_overlap_in_persisted_split():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4],
            "subjectid": [100, 100, 200, 200],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "aki_label": [0, 1, 0, 1],
            "eligible_any_aki": [1, 1, 1, 1],
            "split_group_any_aki": ["train", "test", "train", "test"],
        }
    )

    with pytest.raises(ValueError, match="leaks patients"):
        utils.prepare_data(df, outcome_name="any_aki", feature_set_name="preop_only")


def test_select_patient_level_holdout_requires_multiple_patients_per_class():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4, 5, 6],
            "subjectid": [100, 100, 100, 200, 300, 400],
            "feature1": np.linspace(0.0, 1.0, 6),
            "aki_label": [1, 1, 1, 0, 0, 0],
            "eligible_any_aki": [1, 1, 1, 1, 1, 1],
        }
    )

    with pytest.raises(ValueError, match="patient groups in each class"):
        select_patient_level_holdout_indices(df, outcome_col="aki_label")
