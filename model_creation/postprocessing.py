"""Postprocessing utilities for model evaluation and calibration.

This module provides helpers for generating out-of-fold predictions,
logistic recalibration, threshold selection, and JSON persistence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold


def _validate_array(name: str, array: ArrayLike, allow_nan: bool = False) -> np.ndarray:
    """Convert input to ``np.ndarray`` and optionally validate it for NaNs.

    Args:
        name: Name of the array for error messages.
        array: Array-like input to validate.

    Returns:
        The validated array as ``np.ndarray``.

    Raises:
        ValueError: If the array contains NaN values and ``allow_nan`` is False.
    """
    arr = np.asarray(array)
    if not allow_nan and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN values.")
    return arr


@dataclass
class LogisticRecalibrationModel:
    """Parameters describing a fitted logistic recalibration model."""

    intercept: float
    slope: float
    eps: float = 1e-15


def generate_stratified_oof_predictions(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: ArrayLike,
    n_splits: int = 5,
    random_state: Optional[int] = None,
    shuffle: bool = True,
    fit_params: Optional[Dict[str, Any]] = None,
    proba: bool = True,
    sample_weight: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate stratified K-fold out-of-fold predictions.

    Args:
        model: Estimator implementing ``fit`` and ``predict``/``predict_proba``.
        X: Feature matrix.
        y: Binary target labels.
        n_splits: Number of folds.
        random_state: Random seed for reproducibility.
        shuffle: Whether to shuffle before splitting.
        fit_params: Optional dictionary of parameters passed to ``fit``.
        proba: If True, use ``predict_proba`` and return positive class
            probabilities; otherwise, use ``predict``.
        sample_weight: Optional sample weights aligned with ``X``/``y`` to use
            during training folds.

    Returns:
        A tuple of ``(oof_predictions, fold_indices)`` where ``fold_indices``
        indicates the fold assignment for each sample.

    Raises:
        ValueError: If ``y`` or ``sample_weight`` contain NaNs or if any fold
            is missing.
    """
    X_validated = _validate_array("X", X, allow_nan=True)
    y_validated = _validate_array("y", y)

    if X_validated.shape[0] != y_validated.shape[0]:
        raise ValueError("X and y must have the same number of rows.")

    sample_weight_arr: Optional[np.ndarray] = None
    if sample_weight is not None:
        sample_weight_arr = _validate_array("sample_weight", sample_weight)
        if sample_weight_arr.shape[0] != y_validated.shape[0]:
            raise ValueError("sample_weight must match number of rows in y.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    oof_predictions = np.empty_like(y_validated, dtype=float)
    fold_indices = np.empty_like(y_validated, dtype=int)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_validated, y_validated)):
        model_clone = clone(model)
        fit_kwargs = dict(fit_params or {})
        if sample_weight_arr is not None:
            fit_kwargs["sample_weight"] = sample_weight_arr[train_idx]

        model_clone.fit(X_validated[train_idx], y_validated[train_idx], **fit_kwargs)

        if proba:
            preds = model_clone.predict_proba(X_validated[val_idx])[:, 1]
        else:
            preds = model_clone.predict(X_validated[val_idx])

        oof_predictions[val_idx] = preds
        fold_indices[val_idx] = fold

    if np.isnan(oof_predictions).any():
        raise ValueError("OOF predictions contain NaN values.")

    expected_folds = set(range(n_splits))
    observed_folds = set(np.unique(fold_indices))
    if expected_folds != observed_folds:
        missing = expected_folds - observed_folds
        raise ValueError(f"Missing folds in OOF predictions: {sorted(missing)}")

    return oof_predictions, fold_indices


def fit_logistic_recalibration(
    y_true: ArrayLike,
    pred_probs: ArrayLike,
    eps: float = 1e-15,
) -> LogisticRecalibrationModel:
    """Fit a logistic recalibration model on predicted probabilities.

    Args:
        y_true: True binary labels.
        pred_probs: Predicted probabilities for the positive class.
        eps: Small value to avoid taking the logit of 0 or 1.

    Returns:
        A ``LogisticRecalibrationModel`` with fitted intercept and slope.

    Raises:
        ValueError: If inputs contain NaNs, probabilities are outside [0, 1],
            or ``y_true`` lacks both classes.
    """
    y_arr = _validate_array("y_true", y_true)
    preds_arr = _validate_array("pred_probs", pred_probs)

    if preds_arr.min() < 0 or preds_arr.max() > 1:
        raise ValueError("pred_probs must be within [0, 1].")

    clipped = np.clip(preds_arr, eps, 1 - eps)
    logit_preds = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    if len(np.unique(y_arr)) < 2:
        raise ValueError("y_true must contain both classes for recalibration.")

    lr = LogisticRegression(solver="lbfgs")
    lr.fit(logit_preds, y_arr)

    return LogisticRecalibrationModel(intercept=float(lr.intercept_[0]), slope=float(lr.coef_[0][0]), eps=eps)


def apply_logistic_recalibration(
    pred_probs: ArrayLike,
    model: LogisticRecalibrationModel,
) -> np.ndarray:
    """Apply a fitted logistic recalibration model to probabilities.

    Args:
        pred_probs: Predicted probabilities to recalibrate.
        model: Fitted logistic recalibration parameters.

    Returns:
        Recalibrated probabilities.

    Raises:
        ValueError: If predictions contain NaNs or values outside [0, 1].
    """
    preds_arr = _validate_array("pred_probs", pred_probs)

    if preds_arr.min() < 0 or preds_arr.max() > 1:
        raise ValueError("pred_probs must be within [0, 1].")

    clipped = np.clip(preds_arr, model.eps, 1 - model.eps)
    logit_preds = np.log(clipped / (1 - clipped))
    recalibrated = 1 / (1 + np.exp(-(model.intercept + model.slope * logit_preds)))
    return recalibrated


def find_youden_j_threshold(
    y_true: ArrayLike, pred_probs: ArrayLike, pos_label: int = 1
) -> Tuple[float, float, float, float]:
    """Find the threshold maximizing Youden's J statistic.

    Args:
        y_true: True binary labels.
        pred_probs: Predicted probabilities for the positive class.
        pos_label: Label considered positive.

    Returns:
        Tuple of ``(threshold, youden_j, sensitivity, specificity)``.

    Raises:
        ValueError: If inputs contain NaNs or probabilities are outside [0, 1].
    """
    y_arr = _validate_array("y_true", y_true)
    preds_arr = _validate_array("pred_probs", pred_probs)

    if preds_arr.min() < 0 or preds_arr.max() > 1:
        raise ValueError("pred_probs must be within [0, 1].")

    thresholds = np.unique(preds_arr)
    best_threshold = 0.5
    best_j = -np.inf
    best_sensitivity = 0.0
    best_specificity = 0.0

    for thr in thresholds:
        preds_bin = (preds_arr >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, preds_bin, labels=[1 - pos_label, pos_label]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j_stat = sensitivity + specificity - 1
        if j_stat > best_j:
            best_threshold = float(thr)
            best_j = float(j_stat)
            best_sensitivity = float(sensitivity)
            best_specificity = float(specificity)

    return best_threshold, best_j, best_sensitivity, best_specificity


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write a dictionary to disk as JSON with UTF-8 encoding."""
    with Path(path).open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file from disk."""
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)
