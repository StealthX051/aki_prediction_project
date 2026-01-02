import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

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


def test_validate_prediction_dataframe_happy_path(tmp_path: Path):
    df = _base_predictions()
    threshold = validate_prediction_dataframe(df, tmp_path / "train_oof.csv")

    assert threshold == pytest.approx(0.5)


def test_validate_prediction_dataframe_catches_duplicates(tmp_path: Path):
    df = _base_predictions()
    df.loc[2, "caseid"] = 1

    with pytest.raises(ValueError):
        validate_prediction_dataframe(df, tmp_path / "train_oof.csv")


def test_write_prediction_files_enforces_shared_threshold(tmp_path: Path):
    train_df = _base_predictions(threshold=0.5)
    test_df = _base_predictions(threshold=0.6)

    with pytest.raises(ValueError):
        write_prediction_files(tmp_path, train_df, test_df)

    # Ensure files are not created on failure
    assert not (tmp_path / "train_oof.csv").exists()
    assert not (tmp_path / "test.csv").exists()


def test_write_prediction_files_succeeds(tmp_path: Path):
    train_df = _base_predictions(threshold=0.7)
    test_df = _base_predictions(threshold=0.7)

    write_prediction_files(tmp_path, train_df, test_df)

    written_train = pd.read_csv(tmp_path / "train_oof.csv")
    assert set(written_train.columns) == REQUIRED_PREDICTION_COLUMNS
