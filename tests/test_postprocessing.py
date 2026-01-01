import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_creation.postprocessing import (
    LogisticRecalibrationModel,
    apply_logistic_recalibration,
    find_youden_j_threshold,
    fit_logistic_recalibration,
    generate_stratified_oof_predictions,
    read_json,
    write_json,
)


def test_generate_stratified_oof_predictions_validations_and_output(tmp_path):
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=0,
    )
    model = LogisticRegression(max_iter=100)
    preds, folds = generate_stratified_oof_predictions(model, X, y, n_splits=4, random_state=0)

    assert preds.shape[0] == y.shape[0]
    assert set(folds) == {0, 1, 2, 3}
    assert np.all((preds >= 0) & (preds <= 1))

    with pytest.raises(ValueError):
        generate_stratified_oof_predictions(model, X, np.concatenate([y[:50], [np.nan] * 50]))


def test_logistic_recalibration_and_application():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    pred_probs = np.array([0.1, 0.2, 0.8, 0.7, 0.9, 0.3, 0.6, 0.4])

    model = fit_logistic_recalibration(y_true, pred_probs, eps=1e-3)
    recalibrated = apply_logistic_recalibration(pred_probs, model)

    assert isinstance(model, LogisticRecalibrationModel)
    assert model.eps == 1e-3
    assert recalibrated.shape == pred_probs.shape
    assert np.all((recalibrated > 0) & (recalibrated < 1))

    with pytest.raises(ValueError):
        fit_logistic_recalibration(np.ones_like(y_true), pred_probs)


def test_youden_j_threshold():
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.array([0.1, 0.4, 0.6, 0.9])

    threshold, j_stat, sensitivity, specificity = find_youden_j_threshold(y_true, pred_probs)

    assert 0 <= threshold <= 1
    assert j_stat >= 0
    assert 0 <= sensitivity <= 1
    assert 0 <= specificity <= 1

    with pytest.raises(ValueError):
        find_youden_j_threshold(y_true, np.array([0.1, 1.2, 0.3, 0.4]))


def test_json_helpers(tmp_path: Path):
    data = {"intercept": 0.1, "slope": 0.9, "eps": 1e-3}
    path = tmp_path / "params.json"

    write_json(path, data)
    assert path.exists()

    loaded = read_json(path)
    assert loaded == json.loads(json.dumps(data))
