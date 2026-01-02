"""Shared helpers for writing and validating prediction artifacts."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

import numpy as np
import pandas as pd

PREDICTION_COLUMNS: Sequence[str] = (
    "caseid",
    "y_true",
    "y_prob_raw",
    "y_prob_calibrated",
    "threshold",
    "y_pred_label",
    "fold",
    "is_oof",
    "outcome",
    "branch",
    "feature_set",
    "model_name",
    "pipeline",
)

REQUIRED_PREDICTION_COLUMNS: Set[str] = set(PREDICTION_COLUMNS)
_NON_NULL_COLUMNS: Sequence[str] = (
    "caseid",
    "y_true",
    "y_prob_calibrated",
    "y_pred_label",
    "threshold",
)


def validate_prediction_dataframe(
    df: pd.DataFrame,
    path: Path,
    required_columns: Iterable[str] = REQUIRED_PREDICTION_COLUMNS,
    require_unique_caseids: bool = True,
    require_single_threshold: bool = True,
) -> float:
    """Validate a predictions dataframe before writing or evaluation.

    Args:
        df: Predictions dataframe to validate.
        path: Target path (used for error messages).
        required_columns: Columns that must be present in ``df``.
        require_unique_caseids: Whether to enforce unique ``caseid`` values.
        require_single_threshold: Whether to require a single global threshold.

    Returns:
        The single threshold value found in the dataframe.

    Raises:
        ValueError: If validation fails.
    """

    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")

    for col in _NON_NULL_COLUMNS:
        if df[col].isna().any():
            raise ValueError(f"{path} contains null values in column '{col}'")

    if require_unique_caseids and df["caseid"].duplicated().any():
        dupes = df[df["caseid"].duplicated()]["caseid"].unique()
        raise ValueError(f"{path} has duplicate caseids: {dupes}")

    thresholds = df["threshold"].dropna().unique()
    if require_single_threshold and len(thresholds) != 1:
        raise ValueError(f"{path} must contain a single threshold; found {thresholds}")

    threshold = float(thresholds[0])

    y_true_values = set(df["y_true"].dropna().unique())
    if not y_true_values.issubset({0, 1}):
        raise ValueError(f"{path} has non-binary y_true values: {sorted(y_true_values)}")

    if not ((0 <= df["y_prob_calibrated"]).all() and (df["y_prob_calibrated"] <= 1).all()):
        raise ValueError(f"{path} contains calibrated probabilities outside [0, 1]")

    predicted = (df["y_prob_calibrated"] >= threshold).astype(int)
    if not np.array_equal(predicted.values, df["y_pred_label"].astype(int).values):
        raise ValueError(
            f"{path} predicted labels do not match applying threshold {threshold} to y_prob_calibrated"
        )

    return threshold


def write_prediction_files(
    predictions_dir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Validate and persist train/test prediction files with a shared schema."""

    log = logger or logging.getLogger(__name__)

    predictions_dir.mkdir(parents=True, exist_ok=True)
    train_path = predictions_dir / "train_oof.csv"
    test_path = predictions_dir / "test.csv"

    train_threshold = validate_prediction_dataframe(train_df, train_path)
    test_threshold = validate_prediction_dataframe(test_df, test_path)

    if not math.isclose(train_threshold, test_threshold, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(
            f"Train/test thresholds differ (train={train_threshold}, test={test_threshold}); "
            "artifacts must share a single threshold."
        )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    log.info("Saved train OOF predictions to %s", train_path)
    log.info("Saved test predictions to %s", test_path)
