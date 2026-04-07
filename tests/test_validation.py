import sys
import concurrent.futures
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import model_creation.step_07_train_evaluate as train_module
import model_creation.validation as validation_module
from model_creation.postprocessing import LogisticRecalibrationModel
from model_creation.preprocessing import FoldPreprocessor
from model_creation.validation import (
    OuterFoldRunResult,
    build_cached_folds,
    build_validation_fingerprint,
    get_cached_ebm_feature_types,
    save_outer_fold_checkpoint,
)


def _mock_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "caseid": [1, 2, 3, 4, 5, 6],
            "subjectid": [101, 102, 103, 104, 105, 106],
            "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature2": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "aki_label": [0, 1, 0, 1, 0, 1],
        }
    )


def _make_nested_outer_result(df: pd.DataFrame, *, outer_fold_id: int, test_idx: np.ndarray) -> OuterFoldRunResult:
    threshold = 0.5
    train_idx = np.setdiff1d(np.arange(len(df)), test_idx)
    train_predictions = pd.DataFrame(
        {
            "caseid": df.iloc[train_idx]["caseid"].values,
            "subjectid": df.iloc[train_idx]["subjectid"].values,
            "y_true": df.iloc[train_idx]["aki_label"].values,
            "y_prob_raw": np.linspace(0.2, 0.8, len(train_idx)),
            "y_prob_calibrated": np.linspace(0.25, 0.85, len(train_idx)),
            "threshold": threshold,
            "y_pred_label": (np.linspace(0.25, 0.85, len(train_idx)) >= threshold).astype(int),
            "fold": np.zeros(len(train_idx), dtype=int),
            "outer_fold_id": [outer_fold_id] * len(train_idx),
            "repeat_id": [0] * len(train_idx),
            "validation_scheme": ["nested_cv"] * len(train_idx),
            "is_oof": [True] * len(train_idx),
            "outcome": ["any_aki"] * len(train_idx),
            "branch": ["non_windowed"] * len(train_idx),
            "feature_set": ["preop_only"] * len(train_idx),
            "model_name": ["XGBoost"] * len(train_idx),
            "pipeline": ["step_07_train_evaluate_nested_cv"] * len(train_idx),
        }
    )
    probs = np.linspace(0.3, 0.7, len(test_idx))
    test_predictions = pd.DataFrame(
        {
            "caseid": df.iloc[test_idx]["caseid"].values,
            "subjectid": df.iloc[test_idx]["subjectid"].values,
            "y_true": df.iloc[test_idx]["aki_label"].values,
            "y_prob_raw": probs,
            "y_prob_calibrated": probs,
            "threshold": threshold,
            "y_pred_label": (probs >= threshold).astype(int),
            "fold": [outer_fold_id] * len(test_idx),
            "outer_fold_id": [outer_fold_id] * len(test_idx),
            "repeat_id": [0] * len(test_idx),
            "validation_scheme": ["nested_cv"] * len(test_idx),
            "is_oof": [True] * len(test_idx),
            "outcome": ["any_aki"] * len(test_idx),
            "branch": ["non_windowed"] * len(test_idx),
            "feature_set": ["preop_only"] * len(test_idx),
            "model_name": ["XGBoost"] * len(test_idx),
            "pipeline": ["step_07_train_evaluate_nested_cv"] * len(test_idx),
        }
    )
    return OuterFoldRunResult(
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        best_params={"learning_rate": 0.1},
        best_score=0.5,
        calibration_model=LogisticRecalibrationModel(intercept=0.0, slope=1.0, eps=1e-15),
        threshold_payload={"threshold": threshold, "youden_j": 0.1, "sensitivity": 1.0, "specificity": 0.5},
        metadata_payload={"actual_inner_folds": 2},
    )


def _validation_fingerprint(
    *,
    dataset_hash: str = "dummy-hash",
    validation_scheme: str = "nested_cv",
    n_trials: int = 1,
    inner_folds: int = 2,
    outer_folds: int = 2,
    repeats: int = 1,
    threads_per_model: int = 1,
) -> dict:
    return build_validation_fingerprint(
        dataset_hash=dataset_hash,
        validation_scheme=validation_scheme,
        model_type="xgboost",
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        outer_folds=outer_folds,
        inner_folds=inner_folds,
        repeats=repeats,
        n_trials=n_trials,
        threads_per_model=threads_per_model,
        preserve_nan=True,
        base_random_state=42,
    )


def test_fold_preprocessor_does_not_learn_validation_only_category():
    train_df = pd.DataFrame(
        {
            "department": ["A", "A", "B", "B"],
            "feature1": [0.1, 0.2, 0.3, 0.4],
        }
    )
    val_df = pd.DataFrame(
        {
            "department": ["C"],
            "feature1": [0.5],
        }
    )

    preprocessor = FoldPreprocessor(impute_missing=False)
    preprocessor.fit(train_df)
    transformed_val = preprocessor.transform(val_df)

    assert "department_C" not in transformed_val.columns
    assert transformed_val.shape[1] == len(preprocessor.feature_names_)


def test_cached_ebm_feature_types_use_requested_max_bins_and_cache_per_fold(monkeypatch: pytest.MonkeyPatch):
    X = pd.DataFrame({"feature1": np.linspace(0.0, 1.0, 6), "feature2": np.linspace(1.0, 2.0, 6)})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    groups = pd.Series([10, 20, 30, 40, 50, 60])
    seen_calls = []

    monkeypatch.setattr(
        train_module.utils,
        "compute_quantile_cuts_per_feature",
        lambda arr, max_bins=1024: seen_calls.append((arr.shape[0], max_bins))
        or [f"bins_{max_bins}"] * arr.shape[1],
    )

    cached_folds, _ = build_cached_folds(
        X,
        y,
        groups,
        requested_splits=2,
        random_state=42,
        preserve_nan=True,
        model_type="ebm",
    )

    first_fold = cached_folds[0]
    second_fold = cached_folds[1]

    ft_32_first = get_cached_ebm_feature_types(first_fold, 32)
    ft_32_first_again = get_cached_ebm_feature_types(first_fold, 32)
    ft_128_first = get_cached_ebm_feature_types(first_fold, 128)
    ft_32_second = get_cached_ebm_feature_types(second_fold, 32)

    assert ft_32_first is ft_32_first_again
    assert ft_32_first is not ft_128_first
    assert ft_32_first is not ft_32_second
    assert seen_calls == [(3, 32), (3, 128), (3, 32)]


def test_ebm_uses_light_hpo_bagging_and_heavy_refit_bagging(monkeypatch: pytest.MonkeyPatch):
    X = pd.DataFrame({"feature1": np.linspace(0.0, 1.0, 8), "feature2": np.linspace(1.0, 2.0, 8)})
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
    groups = pd.Series([10, 20, 30, 40, 50, 60, 70, 80])
    seen_params = []
    best_params = {
        "interactions": 0,
        "missing": "gain",
        "inner_bags": 0,
        "outer_bags": 1,
        "max_bins": 64,
        "max_leaves": 2,
        "smoothing_rounds": 25,
        "learning_rate": 0.01,
        "validation_size": 0.15,
        "early_stopping_rounds": 100,
        "early_stopping_tolerance": 0.0,
        "min_samples_leaf": 2,
        "min_hessian": 0.0,
        "greedy_ratio": 0.0,
        "cyclic_progress": 0.0,
    }

    class CaptureEbm:
        def __init__(self, **kwargs):
            self.kwargs = dict(kwargs)
            seen_params.append(self.kwargs)

        def fit(self, X, y, sample_weight=None):
            return self

        def predict_proba(self, X):
            probs = np.full(len(X), 0.6, dtype=float)
            return np.column_stack([1 - probs, probs])

    monkeypatch.setattr(validation_module, "_ebm_class", lambda: CaptureEbm)
    monkeypatch.setattr(
        validation_module,
        "suggest_ebm_params",
        lambda trial: dict(best_params),
    )

    validation_module.tune_model(
        X,
        y,
        groups,
        model_type="ebm",
        n_trials=1,
        requested_splits=2,
        random_state=42,
        preserve_nan=True,
        threads_per_model=1,
    )

    assert seen_params
    assert all(call["inner_bags"] == 0 for call in seen_params)
    assert all(call["outer_bags"] == 1 for call in seen_params)

    seen_params.clear()
    validation_module.generate_cross_fitted_predictions(
        X,
        y,
        groups,
        model_type="ebm",
        best_params=best_params,
        requested_splits=2,
        random_state=42,
        preserve_nan=True,
        threads_per_model=1,
    )
    assert seen_params
    assert all(call["inner_bags"] == 20 for call in seen_params)
    assert all(call["outer_bags"] == 14 for call in seen_params)
    assert all(call["max_bins"] == 64 for call in seen_params)

    seen_params.clear()
    validation_module.fit_final_model(
        X,
        y,
        model_type="ebm",
        best_params=best_params,
        random_state=42,
        preserve_nan=True,
        threads_per_model=1,
    )
    assert len(seen_params) == 1
    assert seen_params[0]["inner_bags"] == 20
    assert seen_params[0]["outer_bags"] == 14
    assert seen_params[0]["max_bins"] == 64


def test_train_evaluate_nested_cv_resume_uses_completed_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: _mock_df())
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    split_list = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]
    monkeypatch.setattr(train_module, "build_cv_splits", lambda *args, **kwargs: (split_list, 2))
    monkeypatch.setattr(train_module, "run_outer_split", lambda **kwargs: (_ for _ in ()).throw(AssertionError("should not run")))

    output_dir = results_dir / "models" / "xgboost" / "any_aki" / "non_windowed" / "preop_only"
    folds_dir = output_dir / "artifacts" / "folds"
    df = _mock_df()
    for outer_fold_id, (_, test_idx) in enumerate(split_list):
        fold_dir = folds_dir / "repeat_00" / f"outer_{outer_fold_id:02d}"
        save_outer_fold_checkpoint(
            fold_dir,
            _make_nested_outer_result(df, outer_fold_id=outer_fold_id, test_idx=test_idx),
            validation_fingerprint=_validation_fingerprint(),
        )

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        model_type="xgboost",
        validation_scheme="nested_cv",
        outer_folds=2,
        inner_folds=2,
        repeats=1,
        max_workers=1,
        threads_per_model=1,
        resume=True,
        n_trials=1,
    )

    test_predictions = pd.read_csv(output_dir / "predictions" / "test.csv")
    assert len(test_predictions) == len(df)
    assert set(test_predictions["caseid"]) == set(df["caseid"])
    assert set(test_predictions["validation_scheme"]) == {"nested_cv"}


@pytest.mark.parametrize(
    ("fingerprint_overrides", "expected_message"),
    [
        ({"dataset_hash": "stale-hash"}, "dataset_hash"),
        ({"n_trials": 99}, "n_trials"),
        ({"inner_folds": 99}, "inner_folds"),
        ({"threads_per_model": 99}, "threads_per_model"),
        ({"validation_protocol_version": -1}, "validation_protocol_version"),
    ],
)
def test_train_evaluate_nested_cv_resume_skips_mismatched_fingerprints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fingerprint_overrides: dict,
    expected_message: str,
    caplog: pytest.LogCaptureFixture,
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: _mock_df())
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    split_list = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]
    monkeypatch.setattr(train_module, "build_cv_splits", lambda *args, **kwargs: (split_list, 2))

    output_dir = results_dir / "models" / "xgboost" / "any_aki" / "non_windowed" / "preop_only"
    folds_dir = output_dir / "artifacts" / "folds"
    df = _mock_df()
    for outer_fold_id, (_, test_idx) in enumerate(split_list):
        fold_dir = folds_dir / "repeat_00" / f"outer_{outer_fold_id:02d}"
        fingerprint = _validation_fingerprint()
        fingerprint.update(fingerprint_overrides)
        save_outer_fold_checkpoint(
            fold_dir,
            _make_nested_outer_result(df, outer_fold_id=outer_fold_id, test_idx=test_idx),
            validation_fingerprint=fingerprint,
        )

    calls = []

    def _fake_run_outer_split(**kwargs):
        calls.append((kwargs["repeat_id"], kwargs["outer_fold_id"]))
        return _make_nested_outer_result(df, outer_fold_id=kwargs["outer_fold_id"], test_idx=kwargs["test_idx"])

    monkeypatch.setattr(train_module, "run_outer_split", _fake_run_outer_split)

    with caplog.at_level("WARNING"):
        train_module.train_evaluate(
            outcome="any_aki",
            branch="non_windowed",
            feature_set="preop_only",
            model_type="xgboost",
            validation_scheme="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            max_workers=1,
            threads_per_model=1,
            resume=True,
            n_trials=1,
        )

    assert sorted(calls) == [(0, 0), (0, 1)]
    assert expected_message in caplog.text


def test_train_evaluate_rejects_repeats_greater_than_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: _mock_df())
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    with pytest.raises(ValueError, match="repeats=1"):
        train_module.train_evaluate(
            outcome="any_aki",
            branch="non_windowed",
            feature_set="preop_only",
            model_type="xgboost",
            validation_scheme="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=2,
            max_workers=1,
            threads_per_model=1,
            n_trials=1,
        )


def test_train_evaluate_nested_cv_rejects_duplicate_test_caseids(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: _mock_df())
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")

    split_list = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]
    monkeypatch.setattr(train_module, "build_cv_splits", lambda *args, **kwargs: (split_list, 2))

    def _bad_run_outer_split(**kwargs):
        result = _make_nested_outer_result(
            _mock_df(),
            outer_fold_id=kwargs["outer_fold_id"],
            test_idx=kwargs["test_idx"],
        )
        if kwargs["outer_fold_id"] == 1:
            result.test_predictions.loc[result.test_predictions.index[0], "caseid"] = 4
        return result

    monkeypatch.setattr(train_module, "run_outer_split", _bad_run_outer_split)

    with pytest.raises(ValueError, match="duplicate caseids"):
        train_module.train_evaluate(
            outcome="any_aki",
            branch="non_windowed",
            feature_set="preop_only",
            model_type="xgboost",
            validation_scheme="nested_cv",
            outer_folds=2,
            inner_folds=2,
            repeats=1,
            max_workers=1,
            threads_per_model=1,
            resume=False,
            n_trials=1,
        )


def test_train_evaluate_caps_nested_workers_to_cpu_budget(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    results_dir = tmp_path / "results"
    monkeypatch.setattr(train_module.utils, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(train_module.utils, "load_data", lambda branch: _mock_df())
    monkeypatch.setattr(train_module, "compute_file_hash", lambda path: "dummy-hash")
    monkeypatch.setattr(train_module.os, "cpu_count", lambda: 16)

    split_list = [
        (np.array([0, 1, 2]), np.array([3, 4, 5])),
        (np.array([3, 4, 5]), np.array([0, 1, 2])),
    ]
    monkeypatch.setattr(train_module, "build_cv_splits", lambda *args, **kwargs: (split_list, 2))

    captured_workers = []

    class FakeExecutor:
        def __init__(self, max_workers):
            captured_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, **kwargs):
            future = concurrent.futures.Future()
            future.set_result(fn(**kwargs))
            return future

    monkeypatch.setattr(train_module.concurrent.futures, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(
        train_module,
        "run_outer_split",
        lambda **kwargs: _make_nested_outer_result(
            _mock_df(),
            outer_fold_id=kwargs["outer_fold_id"],
            test_idx=kwargs["test_idx"],
        ),
    )

    train_module.train_evaluate(
        outcome="any_aki",
        branch="non_windowed",
        feature_set="preop_only",
        model_type="xgboost",
        validation_scheme="nested_cv",
        outer_folds=2,
        inner_folds=2,
        repeats=1,
        max_workers=8,
        threads_per_model=8,
        resume=False,
        n_trials=1,
    )

    assert captured_workers == [2]
