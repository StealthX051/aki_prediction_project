import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

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
        }
    )


def test_select_patient_level_holdout_indices_keeps_patients_disjoint():
    df = _multi_operation_df()

    train_idx, test_idx = select_patient_level_holdout_indices(df, outcome_col="aki_label")

    train_subjects = set(df.loc[train_idx, "subjectid"])
    test_subjects = set(df.loc[test_idx, "subjectid"])

    assert not (train_subjects & test_subjects)
    assert len(test_idx) == 4


def test_prepare_data_uses_patient_grouped_holdout_when_split_group_missing():
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


def test_prepare_data_rejects_existing_patient_overlap_in_split_group():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4],
            "subjectid": [100, 100, 200, 200],
            "feature1": [0.1, 0.2, 0.3, 0.4],
            "aki_label": [0, 1, 0, 1],
            "split_group": ["train", "test", "train", "test"],
        }
    )

    with pytest.raises(ValueError, match="split_group leaks patients"):
        utils.prepare_data(df, outcome_name="any_aki", feature_set_name="preop_only")


def test_select_patient_level_holdout_requires_multiple_patients_per_class():
    df = pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4, 5, 6],
            "subjectid": [100, 100, 100, 200, 300, 400],
            "feature1": np.linspace(0.0, 1.0, 6),
            "aki_label": [1, 1, 1, 0, 0, 0],
        }
    )

    with pytest.raises(ValueError, match="patient groups in each class"):
        select_patient_level_holdout_indices(df, outcome_col="aki_label")
