from pathlib import Path

import pandas as pd
import pytest

from model_creation.prediction_io import (
    REQUIRED_PREDICTION_COLUMNS,
    validate_prediction_dataframe,
    write_prediction_files,
)


def _base_predictions(threshold: float = 0.5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1, 2, 3],
            "y_true": [0, 1, 0],
            "y_prob_raw": [0.2, 0.8, 0.4],
            "y_prob_calibrated": [0.25, 0.85, 0.35],
            "threshold": [threshold] * 3,
            "y_pred_label": [0, 1, 0],
            "fold": [0, 1, 2],
            "is_oof": [True, True, True],
            "outcome": ["o"] * 3,
            "branch": ["b"] * 3,
            "feature_set": ["f"] * 3,
            "model_name": ["m"] * 3,
            "pipeline": ["p"] * 3,
        }
    )


def test_validate_prediction_dataframe_catches_duplicates(tmp_path: Path):
    df = _base_predictions()
    df.loc[2, "caseid"] = 1

    with pytest.raises(ValueError, match="duplicate caseids"):
        validate_prediction_dataframe(df, tmp_path / "train_oof.csv")


def test_write_prediction_files_enforces_shared_threshold(tmp_path: Path):
    train_df = _base_predictions(threshold=0.5)
    test_df = _base_predictions(threshold=0.6)

    with pytest.raises(ValueError, match="Train/test thresholds differ"):
        write_prediction_files(tmp_path, train_df, test_df)

    # Ensure files are not created on failure
    assert not (tmp_path / "train_oof.csv").exists()
    assert not (tmp_path / "test.csv").exists()


@pytest.mark.parametrize("include_patients", [False, True])
def test_write_prediction_files_roundtrips_train_and_test_values(tmp_path: Path, include_patients):
    train_df = _base_predictions(threshold=0.7)
    test_df = _base_predictions(threshold=0.7)
    if include_patients:
        train_df["subjectid"] = test_df["subjectid"] = [10, 20, 30]
    test_df["is_oof"] = False

    write_prediction_files(tmp_path, train_df, test_df)

    for filename, expected in (("train_oof.csv", train_df), ("test.csv", test_df)):
        written = pd.read_csv(tmp_path / filename)
        pd.testing.assert_frame_equal(written, expected)
        assert REQUIRED_PREDICTION_COLUMNS <= set(written)
        assert validate_prediction_dataframe(written, tmp_path / filename) == pytest.approx(.7)


def test_validate_prediction_dataframe_rejects_null_optional_subjectid(tmp_path: Path):
    df = _base_predictions()
    df["subjectid"] = [10, None, 30]

    with pytest.raises(ValueError, match="optional metadata column 'subjectid'"):
        validate_prediction_dataframe(df, tmp_path / "test.csv")


def test_validate_prediction_dataframe_allows_row_varying_thresholds_with_nested_metadata(tmp_path: Path):
    df = pd.DataFrame(
        {
            "caseid": [1, 1, 2, 2],
            "y_true": [1, 1, 0, 0],
            "y_prob_raw": [0.6, 0.6, 0.4, 0.4],
            "y_prob_calibrated": [0.6, 0.6, 0.4, 0.4],
            "threshold": [0.5, 0.7, 0.3, 0.9],
            "y_pred_label": [1, 0, 1, 0],
            "fold": [0, 1, 0, 1],
            "is_oof": [False, False, False, False],
            "outcome": ["o"] * 4,
            "branch": ["b"] * 4,
            "feature_set": ["f"] * 4,
            "model_name": ["m"] * 4,
            "pipeline": ["p"] * 4,
            "repeat_id": [0, 1, 0, 1],
            "outer_fold_id": [0, 1, 2, 3],
        }
    )

    threshold = validate_prediction_dataframe(
        df,
        tmp_path / "nested.csv",
        require_unique_caseids=False,
        require_single_threshold=False,
    )

    assert threshold == pytest.approx(0.6)


def test_validate_prediction_dataframe_rejects_duplicate_caseids_without_nested_metadata(tmp_path: Path):
    df = _base_predictions()
    df.loc[1, "caseid"] = 1

    with pytest.raises(ValueError, match="duplicate caseids without repeat_id/outer_fold_id metadata"):
        validate_prediction_dataframe(
            df,
            tmp_path / "nested.csv",
            require_unique_caseids=False,
            require_single_threshold=False,
        )


def test_write_prediction_files_allows_duplicate_train_caseids_but_not_test_caseids(tmp_path: Path):
    train_df = pd.concat([_base_predictions(), _base_predictions()], ignore_index=True)
    train_df["outer_fold_id"] = [0, 0, 0, 1, 1, 1]
    train_df["repeat_id"] = [0] * len(train_df)
    train_df["threshold"] = [0.5] * len(train_df)
    train_df["y_pred_label"] = (train_df["y_prob_calibrated"] >= train_df["threshold"]).astype(int)

    test_df = _base_predictions()
    test_df.loc[2, "caseid"] = 1

    with pytest.raises(ValueError, match="duplicate caseids"):
        write_prediction_files(
            tmp_path,
            train_df,
            test_df,
            train_allow_duplicate_caseids=True,
            test_allow_duplicate_caseids=False,
        )
    assert not list(tmp_path.glob("*.csv"))
    test_df = _base_predictions()

    write_prediction_files(
        tmp_path,
        train_df,
        test_df,
        train_allow_duplicate_caseids=True,
        test_allow_duplicate_caseids=False,
    )

    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "train_oof.csv"), train_df)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "test.csv"), test_df)
