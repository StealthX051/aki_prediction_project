"""Fold-local modeling, grouped EBM bags, and persisted nested-model regressions."""
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from model_creation import validation as validation
from model_creation import step_07_train_evaluate as train_module


def test_lr_preprocessing_is_fitted_only_on_training_numeric_values_and_categories():
    train = pd.DataFrame({
        "signal": [1.0, 2.0, np.nan, 4.0], "empty": [np.nan] * 4,
        "constant": [7.0] * 4, "department": ["A", "B", "A", None],
    })
    heldout = pd.DataFrame({
        "signal": [1000.0, np.nan], "empty": [np.nan, 99.0],
        "constant": [7.0, 7.0], "department": ["UNSEEN", None],
    })
    prep, transformed = validation.fit_preprocessor(train, preserve_nan=True, model_type="logistic_regression")
    assert prep.numeric_impute_values_ == {"signal": 2.0, "empty": 0.0, "constant": 7.0}
    np.testing.assert_allclose(transformed[prep.numeric_columns_].mean(), 0, atol=1e-12)
    result = prep.transform(heldout)
    assert np.isfinite(result.to_numpy()).all()
    assert result["signal__missing"].tolist() == [0, 1]
    assert result["empty__missing"].tolist() == [1, 0]
    assert "department_UNSEEN" not in result
    assert result.columns.tolist() == transformed.columns.tolist()
    assert result.loc[0, "signal"] > transformed["signal"].max()
    assert prep.numeric_impute_values_["signal"] == 2.0


@pytest.mark.parametrize("model_type", ["xgboost", "ebm"])
def test_tree_preprocessing_still_preserves_native_missing_values(model_type):
    prep, transformed = validation.fit_preprocessor(
        pd.DataFrame({"signal": [1.0, np.nan], "department": ["A", "A"]}),
        preserve_nan=True, model_type=model_type,
    )
    assert pd.isna(transformed.loc[1, "signal"])
    assert not prep.scale_numeric and not prep.add_missing_indicators
    heldout = prep.transform(pd.DataFrame({"signal": [np.nan], "department": ["UNSEEN"]}))
    assert pd.isna(heldout.loc[0, "signal"])
    assert "department_UNSEEN" not in heldout
    assert heldout.columns.tolist() == transformed.columns.tolist()


@pytest.fixture(scope="module", params=["logistic_regression", "xgboost", "ebm"])
def fitted_outer_pipeline(request):
    """Share the unchanged real fit across the two independent held-out attacks."""
    family = request.param
    n = 80
    rng = np.random.default_rng(17)
    groups = pd.Series(np.repeat(np.arange(n // 2), 2))
    outcome = (groups % 2).astype(int)
    X = pd.DataFrame({"signal": outcome * 0.5 + rng.normal(size=n),
                      "sex": np.where(groups % 3, "F", "M")})
    X.loc[::11, "signal"] = np.nan
    frame = pd.DataFrame({"caseid": np.arange(n), "subjectid": groups, "aki_label": outcome})
    train_idx, test_idx = np.arange(64), np.arange(64, n)
    common = dict(
        df=frame, X=X, caseids=frame.caseid, groups=groups, train_idx=train_idx, test_idx=test_idx,
        outcome="any_aki", branch="non_windowed", feature_set="preop_only", model_type=family,
        model_label=family, pipeline_name="outer_label_independence", validation_scheme="nested_cv",
        n_trials=1, inner_folds=2, random_state=42, preserve_nan=True, threads_per_model=1,
        repeat_id=0, outer_fold_id=0, return_fitted_model=True, cohort_scope="main_shared", diagnostic=True,
    )
    original = validation.run_outer_split(y=outcome, **common)
    native_module = {"logistic_regression": "sklearn", "xgboost": "xgboost", "ebm": "interpret"}[family]
    assert type(original.final_model).__module__.startswith(native_module)
    return common, outcome, original


@pytest.mark.parametrize("perturbation", ["labels", "features"])
def test_outer_test_data_cannot_change_any_fitted_pipeline_component(fitted_outer_pipeline, perturbation):
    common, outcome, original = fitted_outer_pipeline
    X, train_idx, test_idx = common["X"], common["train_idx"], common["test_idx"]
    perturbed = outcome.copy()
    changed_X = X.copy()
    if perturbation == "labels":
        perturbed.iloc[test_idx] = 1 - perturbed.iloc[test_idx]
    else:
        changed_X.loc[test_idx, "signal"] = 1e9
        changed_X.loc[test_idx, "sex"] = "HELDOUT_ONLY_CATEGORY"
    changed = validation.run_outer_split(y=perturbed, **{**common, "X": changed_X})
    assert original.best_params == changed.best_params
    assert original.best_score == changed.best_score
    pd.testing.assert_frame_equal(original.train_predictions, changed.train_predictions)
    assert original.calibration_model == changed.calibration_model
    assert original.threshold_payload == changed.threshold_payload
    if perturbation == "labels":
        pd.testing.assert_frame_equal(
            original.test_predictions.drop(columns="y_true"), changed.test_predictions.drop(columns="y_true"))
        np.testing.assert_array_equal(changed.test_predictions.y_true, 1 - original.test_predictions.y_true)
    else:
        assert not original.transformed_test_features.equals(changed.transformed_test_features)
        np.testing.assert_array_equal(changed.test_predictions.y_true, original.test_predictions.y_true)
    left, right = original.final_preprocessor, changed.final_preprocessor
    assert left.clip_bounds_ == right.clip_bounds_
    assert left.numeric_impute_values_ == right.numeric_impute_values_
    assert left.rare_mappings_ == right.rare_mappings_
    if left.numeric_scaler_ is not None:
        np.testing.assert_array_equal(left.numeric_scaler_.mean_, right.numeric_scaler_.mean_)
        np.testing.assert_array_equal(left.numeric_scaler_.scale_, right.numeric_scaler_.scale_)
    for left_categories, right_categories in zip(left.encoder_.categories_, right.encoder_.categories_):
        np.testing.assert_array_equal(left_categories, right_categories)
    np.testing.assert_allclose(
        original.final_model.predict_proba(left.transform(X.iloc[train_idx])),
        changed.final_model.predict_proba(right.transform(X.iloc[train_idx])),
        rtol=0, atol=0,
    )


def test_ebm_bags_are_reproducible_and_group_every_repeat_operation():
    groups = pd.Series(np.repeat(np.arange(16), 2))
    y = pd.Series(np.repeat(np.arange(16) % 2, 2))
    bags = validation.make_ebm_bags(y, groups, n_bags=14, validation_size=0.15, random_state=42)
    replay = validation.make_ebm_bags(y, groups, n_bags=14, validation_size=0.15, random_state=42)
    np.testing.assert_array_equal(bags, replay)
    assert bags.shape == (32, 14)
    for column in bags.T:
        assert set(column) == {-1, 1}
        assert not set(groups[column == 1]) & set(groups[column == -1])
        assert y[column == 1].nunique() == 2
        assert y[column == -1].nunique() == 2


def test_ebm_bags_allow_patients_with_mixed_operation_outcomes():
    groups = pd.Series(np.repeat(np.arange(4), 2))
    y = pd.Series([0, 1] * 4)
    bags = validation.make_ebm_bags(y, groups, n_bags=2, validation_size=0.15, random_state=42)
    for column in bags.T:
        assert not set(groups[column == 1]) & set(groups[column == -1])
        assert y[column == 1].nunique() == y[column == -1].nunique() == 2


def test_grouped_nested_partitions_balance_class_counts_before_ebm_internal_bags():
    # A feasible 40-operation/10-event cohort: the first SGKF seed yields
    # outer-training event counts 7/3 and can strand one event in an inner fit.
    y = pd.Series(np.random.default_rng(0).permutation([0] * 30 + [1] * 10))
    groups = pd.Series(np.arange(40))
    X = pd.DataFrame({"signal": np.arange(40)})
    outer, count = validation.build_cv_splits(X, y, groups, 2, 42)
    assert count == 2
    assert [int(y.iloc[train].sum()) for train, _ in outer] == [5, 5]
    for outer_id, (train, test) in enumerate(outer):
        assert not set(groups.iloc[train]) & set(groups.iloc[test])
        inner, _ = validation.build_cv_splits(
            X.iloc[train], y.iloc[train], groups.iloc[train], 2, 42 + outer_id,
        )
        for inner_train, inner_val in inner:
            fit_y, fit_groups = y.iloc[train].iloc[inner_train], groups.iloc[train].iloc[inner_train]
            assert int(fit_y.sum()) >= 2
            assert not set(fit_groups) & set(groups.iloc[train].iloc[inner_val])
            bags = validation.make_ebm_bags(
                fit_y, fit_groups, n_bags=14, validation_size=0.15, random_state=42 + outer_id,
            )
            assert bags.shape == (len(fit_y), 14)


def test_grouped_fold_cache_reuses_assignments_without_exposing_mutable_arrays():
    validation._balanced_grouped_splits.cache_clear()
    y = pd.Series([0, 1] * 8)
    groups = pd.Series(np.arange(16))
    X = pd.DataFrame({"signal": np.arange(16)})
    first, _ = validation.build_cv_splits(X, y, groups, 2, 42)
    original = first[0][0].copy()
    first[0][0][:] = -1
    second, _ = validation.build_cv_splits(X, y, groups, 2, 42)
    np.testing.assert_array_equal(second[0][0], original)
    assert validation._balanced_grouped_splits.cache_info().hits >= 1


@pytest.mark.parametrize("diagnostic", [False, True])
def test_ebm_diagnostic_profile_is_explicit_and_never_changes_manuscript_fits(diagnostic, monkeypatch):
    captured = []

    class CaptureEbm:
        def __init__(self, **params):
            self.params = params
            captured.append(params)

        def fit(self, X, y, sample_weight=None, bags=None):
            assert bags.shape == (len(y), self.params["outer_bags"])
            np.testing.assert_array_equal(sample_weight, np.where(y == 1, (y == 0).sum() / (y == 1).sum(), 1.))
            fit_groups = groups.loc[X.index].to_numpy()
            for bag in bags.T:
                assert not set(fit_groups[bag > 0]) & set(fit_groups[bag < 0])
            return self

        def predict_proba(self, X):
            return np.tile([0.6, 0.4], (len(X), 1))

    params = {"max_rounds": 2000, "smoothing_rounds": 500, "max_bins": 16,
              "outer_bags": 1, "inner_bags": 0, "validation_size": 0.15}
    monkeypatch.setattr(validation, "_ebm_class", lambda: CaptureEbm)
    monkeypatch.setattr(validation, "suggest_ebm_params", lambda trial: dict(params))
    X = pd.DataFrame({"signal": np.arange(32)})
    y = pd.Series(np.repeat((np.arange(16) % 4 == 0).astype(int), 2))
    groups = pd.Series(np.repeat(np.arange(16), 2))
    common = dict(model_type="ebm", random_state=42, preserve_nan=True,
                  threads_per_model=1, diagnostic=diagnostic)
    validation.tune_model(X, y, groups, n_trials=1, requested_splits=2, **common)
    hpo_fits = len(captured)
    validation.generate_cross_fitted_predictions(X, y, groups, best_params=params, requested_splits=2, **common)
    validation.fit_final_model(X, y, groups=groups, best_params=params, **common)
    assert all(p["outer_bags"] == 1 and p["inner_bags"] == 0 for p in captured[:hpo_fits])
    assert all(p["outer_bags"] == (2 if diagnostic else 14) and p["inner_bags"] == (0 if diagnostic else 20)
               for p in captured[hpo_fits:])
    assert all(p["max_rounds"] == (200 if diagnostic else 2000) for p in captured)
    assert all(p["smoothing_rounds"] == (50 if diagnostic else 500) for p in captured)
    assert all(p["max_bins"] == 16 for p in captured)


def test_diagnostic_compute_and_thread_environment_are_fingerprinted(monkeypatch):
    kwargs = dict(dataset_hash="data", validation_scheme="nested_cv", model_type="ebm",
                  outcome="any_aki", branch="non_windowed", feature_set="preop_only",
                  outer_folds=5, inner_folds=3, calibration_folds=5, repeats=1, n_trials=1, threads_per_model=2,
                  preserve_nan=True, base_random_state=42)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    manuscript = validation.build_validation_fingerprint(**kwargs)
    diagnostic = validation.build_validation_fingerprint(**kwargs, diagnostic=True)
    assert manuscript["compute_profile"] == "manuscript"
    assert diagnostic["compute_profile"] == "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"
    assert not validation.compare_validation_fingerprints(manuscript, diagnostic)[0]
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    changed_threads = validation.build_validation_fingerprint(**kwargs)
    assert not validation.compare_validation_fingerprints(manuscript, changed_threads)[0]


def test_native_ebm_accepts_grouped_bags_and_preserves_nan_handling():
    X = pd.DataFrame({"signal": np.linspace(0.0, 1.0, 32)})
    X.loc[0, "signal"] = np.nan
    groups = pd.Series(np.repeat(np.arange(16), 2))
    y = pd.Series(np.repeat(np.arange(16) % 2, 2))
    prep, model, transformed = validation.fit_final_model(
        X, y, groups=groups, model_type="ebm",
        best_params={"max_rounds": 3, "max_bins": 16, "min_samples_leaf": 2},
        random_state=42, preserve_nan=True, threads_per_model=1,
    )
    assert np.isnan(transformed.loc[0, "signal"])
    assert not prep.impute_missing
    probabilities = model.predict_proba(transformed)[:, 1]
    assert np.isfinite(probabilities).all()
    assert model.bagged_scores_[0].shape[0] == 14


@pytest.mark.parametrize("pathway", ["tune", "crossfit", "final"])
@pytest.mark.parametrize("family", ["logistic_regression", "xgboost"])
def test_family_weighting_contract_in_every_fit_path(family, pathway, monkeypatch):
    seen = []

    class CaptureModel:
        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, **kwargs):
            assert "sample_weight" not in kwargs
            if family == "xgboost":
                assert self.params["scale_pos_weight"] == (y == 0).sum() / (y == 1).sum()
            else:
                assert self.params["class_weight"] is None
            seen.append(len(y))
            return self

        def set_params(self, **params):
            self.params.update(params)
            return self

        def predict_proba(self, X):
            return np.tile([0.7, 0.3], (len(X), 1))

    if family == "xgboost":
        monkeypatch.setattr(validation.xgb, "XGBClassifier", CaptureModel)
    else:
        monkeypatch.setattr(validation, "LogisticRegression", CaptureModel)
    monkeypatch.setattr(validation, "suggest_xgboost_params", lambda trial: {"n_estimators": 2})
    monkeypatch.setattr(validation, "suggest_logistic_regression_params", lambda trial: {"C": .5})
    X = pd.DataFrame({"signal": np.arange(24)})
    y = pd.Series([0, 0, 1] * 8)
    groups = pd.Series(np.arange(24))
    kwargs = dict(model_type=family, random_state=42, preserve_nan=True, threads_per_model=1)
    if pathway == "tune":
        validation.tune_model(X, y, groups, n_trials=1, requested_splits=2, **kwargs)
    elif pathway == "crossfit":
        validation.generate_cross_fitted_predictions(X, y, groups, best_params={}, requested_splits=2, **kwargs)
    else:
        validation.fit_final_model(X, y, groups=groups, best_params={}, **kwargs)
    assert seen


def test_logistic_convergence_retry_and_failure_are_visible():
    class Nonconvergent:
        def __init__(self, always=False):
            self.iterations = []
            self.always = always

        def set_params(self, max_iter):
            self.iterations.append(max_iter)

        def fit(self, X, y):
            if self.always or len(self.iterations) == 1:
                warnings.warn("failed convergence", ConvergenceWarning)

    X = pd.DataFrame({"signal": [0.0, 1.0]})
    y = pd.Series([0, 1])
    model = Nonconvergent()
    validation.fit_estimator(model, X, y, model_type="logistic_regression", groups=None, params={}, random_state=42)
    assert model.iterations == [5000, 10000]
    with pytest.raises(RuntimeError, match="failed to converge"):
        validation.fit_estimator(Nonconvergent(always=True), X, y, model_type="logistic_regression", groups=None, params={}, random_state=42)


def test_native_model_bundle_roundtrip_preserves_heldout_transforms_and_probabilities(fitted_outer_pipeline, tmp_path):
    common, _, original = fitted_outer_pipeline
    family = common["model_type"]
    prep, model = original.final_preprocessor, original.final_model
    X = common["X"].iloc[common["test_idx"]]
    transformed = prep.transform(X)
    validation.save_model_bundle(tmp_path, model_type=family, model=model, preprocessor=prep)
    with (tmp_path / "preprocessor.pkl").open("rb") as stream:
        reloaded_prep = pickle.load(stream)
    if family == "xgboost":
        reloaded = validation.xgb.XGBClassifier()
        reloaded.load_model(tmp_path / "model.json")
    else:
        with (tmp_path / "model.pkl").open("rb") as stream:
            reloaded = pickle.load(stream)
    pd.testing.assert_frame_equal(reloaded_prep.transform(X), transformed)
    np.testing.assert_array_equal(reloaded.predict_proba(reloaded_prep.transform(X)), model.predict_proba(transformed))
    if family == "logistic_regression":
        assert model.class_weight is None


def test_real_grouped_nested_lr_saves_calibrated_predictions_and_complete_resumable_bundles(tmp_path, monkeypatch):
    n = 48
    frame = pd.DataFrame({
        "caseid": np.arange(n), "subjectid": np.repeat(np.arange(n // 2), 2),
        "signal": np.linspace(0, 1, n), "sex": ["M", "F"] * (n // 2),
        "aki_label": np.repeat(np.arange(n // 2) % 2, 2),
        "y_icu_admit": np.repeat(np.arange(n // 2) % 2, 2),
        "eligible_any_aki": 1, "eligible_icu_admission": 1,
    })
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: frame)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "synthetic-data")
    kwargs = dict(
        outcome="any_aki", branch="non_windowed", feature_set="preop_only",
        model_type="logistic_regression", outer_folds=2, inner_folds=2, calibration_folds=3,
        max_workers=1, threads_per_model=1, n_trials=1, diagnostic=True,
    )
    train_module.train_evaluate(**kwargs)
    root = tmp_path / "results/main_shared/nested_cv/models/logistic_regression/any_aki/non_windowed/preop_only"
    predictions = pd.read_csv(root / "predictions/test.csv")
    assert len(predictions) == n and predictions["caseid"].is_unique
    assert predictions["subjectid"].nunique() == n // 2
    assert predictions.groupby("subjectid")["outer_fold_id"].nunique().eq(1).all()
    assert np.isfinite(predictions["y_prob_calibrated"]).all()
    assert predictions["cohort_scope"].eq("main_shared").all()
    assert predictions["diagnostic"].all()
    fold_dirs = sorted((root / "artifacts/folds/repeat_00").glob("outer_*"))
    assert len(fold_dirs) == 2
    for fold_dir in fold_dirs:
        assert validation.is_outer_fold_checkpoint_complete(fold_dir)
        train_ids = set(pd.read_csv(fold_dir / "train_oof.csv")["subjectid"])
        test_ids = set(pd.read_csv(fold_dir / "test_predictions.csv")["subjectid"])
        assert not train_ids & test_ids
        metadata = json.loads((fold_dir / "metadata.json").read_text())
        assert metadata["model_type"] == "logistic_regression"
        assert metadata["requested_inner_folds"] == metadata["actual_inner_folds"] == 2
        assert metadata["requested_calibration_folds"] == metadata["actual_calibration_folds"] == 3
        assert metadata["validation_fingerprint"]["calibration_folds"] == 3
    root_metadata = json.loads((root / "artifacts/metadata.json").read_text())
    assert root_metadata["inner_folds"] == 2 and root_metadata["calibration_folds"] == 3
    monkeypatch.setattr(train_module, "run_outer_split", lambda **kwargs: pytest.fail("completed folds must resume"))
    train_module.train_evaluate(**kwargs, resume=True)
    model_path = fold_dirs[0] / "model.pkl"
    model_path.write_bytes(b"truncated")
    assert not validation.is_outer_fold_checkpoint_complete(fold_dirs[0])


def test_definitive_nested_validation_rejects_a_reduced_design(tmp_path, monkeypatch):
    frame = pd.DataFrame({"caseid": np.arange(12), "subjectid": np.arange(12), "signal": np.arange(12),
                          "aki_label": [0, 1] * 6, "y_icu_admit": [0, 1] * 6,
                          "eligible_any_aki": 1, "eligible_icu_admission": 1})
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: frame)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "synthetic-data")
    with pytest.raises(ValueError, match="Definitive validation requires 5 outer, 3 inner HPO, and 5 calibration"):
        train_module.train_evaluate("any_aki", "non_windowed", "preop_only", outer_folds=2, inner_folds=2)


@pytest.mark.parametrize("n,expected", [(8, "outer-fold count"), (12, "calibration-fold count")])
def test_definitive_nested_validation_rejects_silent_fold_reduction(n, expected, tmp_path, monkeypatch):
    frame = pd.DataFrame({"caseid": np.arange(n), "subjectid": np.arange(n), "signal": np.arange(n),
                          "aki_label": [0, 1] * (n // 2), "y_icu_admit": [0, 1] * (n // 2),
                          "eligible_any_aki": 1, "eligible_icu_admission": 1})
    monkeypatch.setenv("RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: frame)
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "synthetic-data")
    with pytest.raises(ValueError, match=expected):
        train_module.train_evaluate("any_aki", "non_windowed", "preop_only")
