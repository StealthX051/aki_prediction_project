import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from model_creation.postprocessing import (
    LogisticRecalibrationModel,
    apply_logistic_recalibration,
    find_youden_j_threshold,
    fit_logistic_recalibration,
    generate_stratified_oof_predictions,
)


def test_generate_stratified_oof_predictions_validations_and_output():
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


def test_generate_stratified_oof_predictions_respects_groups():
    X, y = make_classification(
        n_samples=24,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
    )
    groups = np.repeat(np.arange(12), 2)
    fit_patients = []

    class RecordingLogisticRegression(LogisticRegression):
        def fit(self, X, y, **kwargs):
            # A unique operation marker makes the fitting boundary inspectable.
            self.fit_operations_ = set(X[:, 0])
            fit_patients.append(set((X[:, 0] // 2).astype(int)))
            return super().fit(X, y, **kwargs)

        def predict_proba(self, X):
            assert not self.fit_operations_ & set(X[:, 0])
            assert not set((X[:, 0] // 2).astype(int)) & fit_patients[-1]
            return super().predict_proba(X)

    X[:, 0] = np.arange(len(X))
    model = RecordingLogisticRegression(max_iter=1000)

    preds, folds = generate_stratified_oof_predictions(
        model,
        X,
        y,
        n_splits=4,
        random_state=0,
        groups=groups,
    )

    assert preds.shape[0] == y.shape[0]
    assert set(folds) == {0, 1, 2, 3}
    assert len(fit_patients) == 4
    assert not hasattr(model, "coef_")  # Each fold fits a clone, never the caller's estimator.
    assert np.isfinite(preds).all()

    for group_value in np.unique(groups):
        group_folds = set(folds[groups == group_value])
        assert len(group_folds) == 1


def test_generate_stratified_oof_predictions_rejects_insufficient_groups_per_class():
    X = np.arange(12, dtype=float).reshape(-1, 1)
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    groups = np.array([10, 10, 10, 10, 20, 21, 22, 23, 24, 25, 26, 27])
    model = LogisticRegression(max_iter=100)

    with pytest.raises(ValueError, match="unique patient groups available in each class"):
        generate_stratified_oof_predictions(model, X, y, n_splits=2, random_state=0, groups=groups)


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


def test_recalibration_matches_hand_calculated_odds_and_clips_boundary_probabilities():
    model = LogisticRecalibrationModel(intercept=np.log(2), slope=2, eps=1e-3)
    # Calibrated odds are 2 * original_odds**2.
    np.testing.assert_allclose(apply_logistic_recalibration([.2, .5, .8], model), [1 / 9, 2 / 3, 32 / 33])
    np.testing.assert_allclose(apply_logistic_recalibration([0., 1.], model),
                               apply_logistic_recalibration([model.eps, 1 - model.eps], model))


@pytest.mark.parametrize("probabilities,expected", [
    ([.1, .4, .6, .9], (.6, 1., 1., 1.)),
    ([.1, .7, .6, .9], (.6, .5, 1., .5)),
])
def test_youden_j_threshold_matches_hand_calculated_confusion_counts(probabilities, expected):
    y_true = np.array([0, 0, 1, 1])
    pred_probs = np.asarray(probabilities)

    threshold, j_stat, sensitivity, specificity = find_youden_j_threshold(y_true, pred_probs)

    assert (threshold, j_stat, sensitivity, specificity) == pytest.approx(expected)

    with pytest.raises(ValueError):
        find_youden_j_threshold(y_true, np.array([0.1, 1.2, 0.3, 0.4]))
