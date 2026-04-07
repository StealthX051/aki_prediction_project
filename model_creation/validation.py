"""Shared validation engine for holdout and nested CV workflows."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from model_creation import utils
from model_creation.postprocessing import (
    LogisticRecalibrationModel,
    apply_logistic_recalibration,
    find_youden_j_threshold,
    fit_logistic_recalibration,
    write_json,
)
from model_creation.preprocessing import FoldPreprocessor

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
VALIDATION_PROTOCOL_VERSION = 2
EBM_HPO_BAGGING = {"inner_bags": 0, "outer_bags": 1}
EBM_REFIT_BAGGING = {"inner_bags": 20, "outer_bags": 14}


@dataclass
class CachedFold:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    sample_weight: np.ndarray
    ebm_feature_types_cache: Optional[Dict[int, List]] = None


@dataclass
class OuterFoldRunResult:
    train_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    best_params: Dict[str, Any]
    best_score: float
    calibration_model: LogisticRecalibrationModel
    threshold_payload: Dict[str, float]
    metadata_payload: Dict[str, Any]
    final_model: Optional[Any] = None
    final_preprocessor: Optional[FoldPreprocessor] = None
    transformed_test_features: Optional[pd.DataFrame] = None
    raw_test_logits: Optional[np.ndarray] = None


def compute_scale_pos_weight(y: pd.Series) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / pos) if pos > 0 else 1.0


def make_sample_weight(y: pd.Series) -> np.ndarray:
    scale_pos_weight = compute_scale_pos_weight(y)
    return np.where(y.to_numpy() == 1, scale_pos_weight, 1.0).astype(float)


def select_modeling_dataset(
    df: pd.DataFrame,
    outcome_name: str,
    feature_set_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    target_col = utils.OUTCOMES.get(outcome_name)
    if target_col is None:
        raise ValueError(f"Invalid outcome: {outcome_name}. Available: {list(utils.OUTCOMES.keys())}")

    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in dataframe.")

    working_df = df.dropna(subset=[target_col]).copy()
    feature_sets = utils.get_feature_sets(working_df)
    if feature_set_name not in feature_sets:
        raise ValueError(
            f"Invalid feature set: {feature_set_name}. Available: {list(feature_sets.keys())}"
        )

    X = working_df[feature_sets[feature_set_name]].copy()
    y = working_df[target_col].astype(int).copy()
    caseids = working_df["caseid"].copy()
    groups = utils.get_patient_groups(working_df)
    return working_df, X, y, caseids, groups


def build_validation_fingerprint(
    *,
    dataset_hash: str,
    validation_scheme: str,
    model_type: str,
    outcome: str,
    branch: str,
    feature_set: str,
    outer_folds: int,
    inner_folds: int,
    repeats: int,
    n_trials: int,
    threads_per_model: int,
    preserve_nan: bool,
    base_random_state: int,
) -> Dict[str, Any]:
    return {
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "dataset_hash": dataset_hash,
        "validation_scheme": validation_scheme,
        "model_type": model_type,
        "outcome": outcome,
        "branch": branch,
        "feature_set": feature_set,
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "repeats": int(repeats),
        "n_trials": int(n_trials),
        "threads_per_model": int(threads_per_model),
        "preserve_nan": bool(preserve_nan),
        "legacy_imputation": not bool(preserve_nan),
        "base_random_state": int(base_random_state),
    }


def compare_validation_fingerprints(
    expected: Dict[str, Any],
    actual: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if actual is None:
        return False, "checkpoint is missing validation_fingerprint metadata"

    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    if missing:
        return False, f"checkpoint fingerprint is missing keys: {missing}"

    for key in sorted(expected_keys):
        if actual.get(key) != expected.get(key):
            return (
                False,
                f"fingerprint mismatch for '{key}': expected {expected.get(key)!r}, "
                f"found {actual.get(key)!r}",
            )

    extra = sorted(actual_keys - expected_keys)
    if extra:
        return False, f"checkpoint fingerprint has unexpected keys: {extra}"

    return True, "fingerprint matches"


def resolve_n_splits(
    y: pd.Series,
    groups: Optional[pd.Series],
    requested_splits: int,
) -> int:
    min_class_count = int(y.value_counts().min())
    n_splits = min(requested_splits, min_class_count)
    if groups is not None:
        unique_groups_per_class = [
            int(groups[y == class_value].nunique()) for class_value in sorted(y.dropna().unique())
        ]
        if not unique_groups_per_class:
            raise ValueError("Unable to derive patient-level class counts.")
        n_splits = min(n_splits, int(groups.nunique()), min(unique_groups_per_class))
    if n_splits < 2:
        raise ValueError("Need at least two folds with both classes represented.")
    return n_splits


def build_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    requested_splits: int,
    random_state: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    n_splits = resolve_n_splits(y, groups, requested_splits)
    if groups is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        split_iter = splitter.split(X, y)
    return list(split_iter), n_splits


def _ebm_class():
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError as exc:  # pragma: no cover - covered indirectly
        raise ImportError("interpret library is required for EBM model_type") from exc
    return ExplainableBoostingClassifier


def get_ebm_max_bins(params: Dict[str, Any], default: int = 1024) -> int:
    try:
        return int(params.get("max_bins", default))
    except (TypeError, ValueError):
        return int(default)


def get_effective_refit_params(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    effective = dict(params)
    if model_type == "ebm":
        effective.update(EBM_REFIT_BAGGING)
    return effective


def get_refit_param_overrides(model_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
    if model_type != "ebm":
        return {}

    effective = get_effective_refit_params(model_type, params)
    return {
        key: effective[key]
        for key in sorted(EBM_REFIT_BAGGING)
        if params.get(key) != effective.get(key)
    }


def compute_ebm_feature_types(X_train: pd.DataFrame, max_bins: int) -> List:
    return utils.compute_quantile_cuts_per_feature(
        np.ascontiguousarray(X_train.values, dtype=np.float32),
        max_bins=max_bins,
    )


def get_cached_ebm_feature_types(fold: CachedFold, max_bins: int) -> List:
    if fold.ebm_feature_types_cache is None:
        fold.ebm_feature_types_cache = {}

    if max_bins not in fold.ebm_feature_types_cache:
        fold.ebm_feature_types_cache[max_bins] = compute_ebm_feature_types(fold.X_train, max_bins)

    return fold.ebm_feature_types_cache[max_bins]


def suggest_xgboost_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
    }


def suggest_ebm_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        "interactions": 0,
        "missing": "gain",
        **EBM_HPO_BAGGING,
        "max_bins": trial.suggest_categorical("max_bins", [32, 64, 128, 256]),
        "max_leaves": trial.suggest_categorical("max_leaves", [2, 3]),
        "smoothing_rounds": trial.suggest_categorical(
            "smoothing_rounds", [0, 25, 50, 75, 100, 150, 200, 350, 500]
        ),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", [0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04]
        ),
        "validation_size": trial.suggest_categorical("validation_size", [0.1, 0.15, 0.2]),
        "early_stopping_rounds": trial.suggest_categorical("early_stopping_rounds", [100, 200]),
        "early_stopping_tolerance": trial.suggest_categorical(
            "early_stopping_tolerance", [0.0, 1e-5]
        ),
        "min_samples_leaf": trial.suggest_categorical("min_samples_leaf", [2, 3, 4, 5, 10]),
        "min_hessian": trial.suggest_categorical("min_hessian", [0.0, 1e-6, 1e-4, 1e-2]),
        "greedy_ratio": trial.suggest_categorical("greedy_ratio", [0.0, 5.0, 10.0]),
        "cyclic_progress": trial.suggest_categorical("cyclic_progress", [0.0, 1.0]),
    }


def build_model(
    model_type: str,
    params: Dict[str, Any],
    *,
    random_state: int,
    threads_per_model: int,
    scale_pos_weight: float,
    feature_types: Optional[List] = None,
) -> Any:
    if model_type == "xgboost":
        model_params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "n_jobs": max(1, threads_per_model),
            "random_state": random_state,
            "verbosity": 0,
            **params,
        }
        model_params["scale_pos_weight"] = scale_pos_weight
        return xgb.XGBClassifier(**model_params)

    if model_type == "ebm":
        ebm_cls = _ebm_class()
        model_params = {
            "interactions": 0,
            "missing": "gain",
            **EBM_REFIT_BAGGING,
            "max_bins": 1024,
            "random_state": random_state,
            "n_jobs": max(1, threads_per_model),
            **params,
        }
        model_params.pop("scale_pos_weight", None)
        if feature_types is not None:
            model_params["feature_types"] = feature_types
        return ebm_cls(**model_params)

    raise ValueError(f"Unsupported model_type: {model_type}")


def fit_preprocessor(
    X_train: pd.DataFrame,
    *,
    preserve_nan: bool,
) -> Tuple[FoldPreprocessor, pd.DataFrame]:
    preprocessor = FoldPreprocessor(impute_missing=not preserve_nan)
    X_train_transformed = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_transformed


def build_cached_folds(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    *,
    requested_splits: int,
    random_state: int,
    preserve_nan: bool,
    model_type: str,
) -> Tuple[List[CachedFold], int]:
    split_list, actual_splits = build_cv_splits(X, y, groups, requested_splits, random_state)
    cached_folds: List[CachedFold] = []
    for train_idx, val_idx in split_list:
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]
        preprocessor, X_train_transformed = fit_preprocessor(X_train_raw, preserve_nan=preserve_nan)
        X_val_transformed = preprocessor.transform(X_val_raw)
        sample_weight = make_sample_weight(y_train)
        cached_folds.append(
            CachedFold(
                X_train=X_train_transformed,
                X_val=X_val_transformed,
                y_train=y_train,
                y_val=y_val,
                sample_weight=sample_weight,
                ebm_feature_types_cache={} if model_type == "ebm" else None,
            )
        )
    return cached_folds, actual_splits


def tune_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    *,
    model_type: str,
    n_trials: int,
    requested_splits: int,
    random_state: int,
    preserve_nan: bool,
    threads_per_model: int,
) -> Tuple[Dict[str, Any], float, int]:
    cached_folds, actual_inner_folds = build_cached_folds(
        X,
        y,
        groups,
        requested_splits=requested_splits,
        random_state=random_state,
        preserve_nan=preserve_nan,
        model_type=model_type,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_xgboost_params(trial) if model_type == "xgboost" else suggest_ebm_params(trial)
        scores: List[float] = []
        for fold in cached_folds:
            scale_pos_weight = compute_scale_pos_weight(fold.y_train)
            feature_types = None
            if model_type == "ebm":
                feature_types = get_cached_ebm_feature_types(fold, get_ebm_max_bins(params))
            model = build_model(
                model_type,
                params,
                random_state=random_state,
                threads_per_model=threads_per_model,
                scale_pos_weight=scale_pos_weight,
                feature_types=feature_types,
            )
            if model_type == "xgboost":
                model.fit(fold.X_train, fold.y_train, sample_weight=fold.sample_weight, verbose=False)
            else:
                model.fit(fold.X_train, fold.y_train, sample_weight=fold.sample_weight)
            y_pred_proba = model.predict_proba(fold.X_val)[:, 1]
            scores.append(float(average_precision_score(fold.y_val, y_pred_proba)))
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=random_state)
    try:
        study = optuna.create_study(direction="maximize", sampler=sampler)
    except TypeError:
        study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)
    return study.best_params, float(study.best_value), actual_inner_folds


def generate_cross_fitted_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    *,
    model_type: str,
    best_params: Dict[str, Any],
    requested_splits: int,
    random_state: int,
    preserve_nan: bool,
    threads_per_model: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    split_list, actual_splits = build_cv_splits(X, y, groups, requested_splits, random_state)
    predictions = np.empty(len(X), dtype=float)
    fold_indices = np.empty(len(X), dtype=int)
    effective_params = get_effective_refit_params(model_type, best_params)

    for fold_id, (train_idx, val_idx) in enumerate(split_list):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        preprocessor, X_train_transformed = fit_preprocessor(X_train_raw, preserve_nan=preserve_nan)
        X_val_transformed = preprocessor.transform(X_val_raw)
        scale_pos_weight = compute_scale_pos_weight(y_train)
        sample_weight = make_sample_weight(y_train)
        feature_types = None
        if model_type == "ebm":
            feature_types = compute_ebm_feature_types(
                X_train_transformed,
                get_ebm_max_bins(effective_params),
            )
        model = build_model(
            model_type,
            effective_params,
            random_state=random_state,
            threads_per_model=threads_per_model,
            scale_pos_weight=scale_pos_weight,
            feature_types=feature_types,
        )
        if model_type == "xgboost":
            model.fit(X_train_transformed, y_train, sample_weight=sample_weight, verbose=False)
        else:
            model.fit(X_train_transformed, y_train, sample_weight=sample_weight)
        predictions[val_idx] = model.predict_proba(X_val_transformed)[:, 1]
        fold_indices[val_idx] = fold_id

    return predictions, fold_indices, actual_splits


def fit_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    model_type: str,
    best_params: Dict[str, Any],
    random_state: int,
    preserve_nan: bool,
    threads_per_model: int,
) -> Tuple[FoldPreprocessor, Any, pd.DataFrame]:
    preprocessor, X_train_transformed = fit_preprocessor(X_train, preserve_nan=preserve_nan)
    scale_pos_weight = compute_scale_pos_weight(y_train)
    sample_weight = make_sample_weight(y_train)
    effective_params = get_effective_refit_params(model_type, best_params)
    feature_types = None
    if model_type == "ebm":
        feature_types = compute_ebm_feature_types(
            X_train_transformed,
            get_ebm_max_bins(effective_params),
        )
    model = build_model(
        model_type,
        effective_params,
        random_state=random_state,
        threads_per_model=threads_per_model,
        scale_pos_weight=scale_pos_weight,
        feature_types=feature_types,
    )
    if model_type == "xgboost":
        model.fit(X_train_transformed, y_train, sample_weight=sample_weight, verbose=False)
    else:
        model.fit(X_train_transformed, y_train, sample_weight=sample_weight)
    return preprocessor, model, X_train_transformed


def save_model_bundle(
    output_dir: Path,
    *,
    model_type: str,
    model: Any,
    preprocessor: FoldPreprocessor,
) -> Path:
    preprocessor_path = output_dir / "preprocessor.pkl"
    with preprocessor_path.open("wb") as fp:
        pickle.dump(preprocessor, fp)

    if model_type == "xgboost":
        model_path = output_dir / "model.json"
        model.save_model(model_path)
    else:
        model_path = output_dir / "model.pkl"
        with model_path.open("wb") as fp:
            pickle.dump(model, fp)
    return model_path


def _safe_logit(probabilities: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    clipped = np.clip(probabilities, eps, 1 - eps)
    return np.log(clipped / (1 - clipped))


def run_outer_split(
    *,
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    caseids: pd.Series,
    groups: Optional[pd.Series],
    train_idx: Sequence[int],
    test_idx: Sequence[int],
    outcome: str,
    branch: str,
    feature_set: str,
    model_type: str,
    model_label: str,
    pipeline_name: str,
    validation_scheme: str,
    n_trials: int,
    inner_folds: int,
    random_state: int,
    preserve_nan: bool,
    threads_per_model: int,
    repeat_id: int,
    outer_fold_id: int,
    return_fitted_model: bool = False,
) -> OuterFoldRunResult:
    X_train_raw = X.iloc[list(train_idx)]
    X_test_raw = X.iloc[list(test_idx)]
    y_train = y.iloc[list(train_idx)]
    y_test = y.iloc[list(test_idx)]
    train_groups = groups.iloc[list(train_idx)] if groups is not None else None
    test_groups = groups.iloc[list(test_idx)] if groups is not None else None

    best_params, best_score, actual_inner_folds = tune_model(
        X_train_raw,
        y_train,
        train_groups,
        model_type=model_type,
        n_trials=n_trials,
        requested_splits=inner_folds,
        random_state=random_state,
        preserve_nan=preserve_nan,
        threads_per_model=threads_per_model,
    )

    oof_predictions, fold_indices, actual_calibration_folds = generate_cross_fitted_predictions(
        X_train_raw,
        y_train,
        train_groups,
        model_type=model_type,
        best_params=best_params,
        requested_splits=inner_folds,
        random_state=random_state,
        preserve_nan=preserve_nan,
        threads_per_model=threads_per_model,
    )
    recalibration_model = fit_logistic_recalibration(y_train.values, oof_predictions)
    calibrated_oof = apply_logistic_recalibration(oof_predictions, recalibration_model)
    threshold, youden_j, sensitivity, specificity = find_youden_j_threshold(
        y_train.values, calibrated_oof
    )

    final_preprocessor, final_model, _ = fit_final_model(
        X_train_raw,
        y_train,
        model_type=model_type,
        best_params=best_params,
        random_state=random_state,
        preserve_nan=preserve_nan,
        threads_per_model=threads_per_model,
    )
    X_test_transformed = final_preprocessor.transform(X_test_raw)
    test_pred_proba = final_model.predict_proba(X_test_transformed)[:, 1]
    test_pred_calibrated = apply_logistic_recalibration(test_pred_proba, recalibration_model)
    if hasattr(final_model, "decision_function"):
        try:
            raw_test_logits = final_model.decision_function(X_test_transformed)
        except Exception:
            raw_test_logits = _safe_logit(test_pred_proba)
    else:
        raw_test_logits = _safe_logit(test_pred_proba)

    if set(X_train_raw.index) & set(X_test_raw.index):
        raise AssertionError("Train and test indices overlap within outer split.")
    if train_groups is not None and test_groups is not None:
        overlap = set(train_groups) & set(test_groups)
        if overlap:
            raise AssertionError(
                f"Patient overlap detected across outer split: {sorted(overlap)[:5]}"
            )

    effective_refit_params = get_effective_refit_params(model_type, best_params)
    metadata_base = {
        "validation_scheme": validation_scheme,
        "repeat_id": int(repeat_id),
        "outer_fold_id": int(outer_fold_id),
        "random_state": int(random_state),
        "n_trials": int(n_trials),
        "requested_inner_folds": int(inner_folds),
        "actual_inner_folds": int(actual_inner_folds),
        "actual_calibration_folds": int(actual_calibration_folds),
        "best_score": float(best_score),
        "best_params": json.loads(json.dumps(best_params)),
        "effective_refit_params": json.loads(json.dumps(effective_refit_params)),
        "refit_param_overrides": json.loads(
            json.dumps(get_refit_param_overrides(model_type, best_params))
        ),
    }

    train_predictions = pd.DataFrame(
        {
            "caseid": caseids.iloc[list(train_idx)].values,
            "y_true": y_train.values,
            "y_prob_raw": oof_predictions,
            "y_prob_calibrated": calibrated_oof,
            "threshold": threshold,
            "y_pred_label": (calibrated_oof >= threshold).astype(int),
            "fold": fold_indices,
            "outer_fold_id": outer_fold_id,
            "repeat_id": repeat_id,
            "validation_scheme": validation_scheme,
            "is_oof": True,
            "outcome": outcome,
            "branch": branch,
            "feature_set": feature_set,
            "model_name": model_label,
            "pipeline": pipeline_name,
        },
        index=X_train_raw.index,
    )
    if train_groups is not None:
        train_predictions["subjectid"] = train_groups.values

    test_predictions = pd.DataFrame(
        {
            "caseid": caseids.iloc[list(test_idx)].values,
            "y_true": y_test.values,
            "y_prob_raw": test_pred_proba,
            "y_prob_calibrated": test_pred_calibrated,
            "threshold": threshold,
            "y_pred_label": (test_pred_calibrated >= threshold).astype(int),
            "fold": outer_fold_id,
            "outer_fold_id": outer_fold_id,
            "repeat_id": repeat_id,
            "validation_scheme": validation_scheme,
            "is_oof": validation_scheme == "nested_cv",
            "outcome": outcome,
            "branch": branch,
            "feature_set": feature_set,
            "model_name": model_label,
            "pipeline": pipeline_name,
        },
        index=X_test_raw.index,
    )
    if test_groups is not None:
        test_predictions["subjectid"] = test_groups.values

    threshold_payload = {
        "threshold": float(threshold),
        "youden_j": float(youden_j),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }
    metadata_payload = {
        **metadata_base,
        "counts": {
            "train": {
                "total": int(len(y_train)),
                "positive": int((y_train == 1).sum()),
                "negative": int((y_train == 0).sum()),
            },
            "test": {
                "total": int(len(y_test)),
                "positive": int((y_test == 1).sum()),
                "negative": int((y_test == 0).sum()),
            },
        },
    }

    return OuterFoldRunResult(
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        best_params=best_params,
        best_score=best_score,
        calibration_model=recalibration_model,
        threshold_payload=threshold_payload,
        metadata_payload=metadata_payload,
        final_model=final_model if return_fitted_model else None,
        final_preprocessor=final_preprocessor if return_fitted_model else None,
        transformed_test_features=X_test_transformed if return_fitted_model else None,
        raw_test_logits=raw_test_logits if return_fitted_model else None,
    )


def save_outer_fold_checkpoint(
    fold_dir: Path,
    result: OuterFoldRunResult,
    *,
    validation_fingerprint: Dict[str, Any],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    result.train_predictions.to_csv(fold_dir / "train_oof.csv", index=False)
    result.test_predictions.to_csv(fold_dir / "test_predictions.csv", index=False)
    write_json(
        fold_dir / "calibration.json",
        {
            "intercept": result.calibration_model.intercept,
            "slope": result.calibration_model.slope,
            "eps": result.calibration_model.eps,
        },
    )
    write_json(fold_dir / "threshold.json", result.threshold_payload)
    write_json(
        fold_dir / "metadata.json",
        {
            **result.metadata_payload,
            "dataset_hash": validation_fingerprint["dataset_hash"],
            "validation_protocol_version": validation_fingerprint["validation_protocol_version"],
            "validation_fingerprint": validation_fingerprint,
            "status": "completed",
        },
    )
    write_json(fold_dir / "best_params.json", result.best_params)


def load_outer_fold_checkpoint(fold_dir: Path) -> OuterFoldRunResult:
    train_predictions = pd.read_csv(fold_dir / "train_oof.csv")
    test_predictions = pd.read_csv(fold_dir / "test_predictions.csv")
    calibration_payload = json.loads((fold_dir / "calibration.json").read_text())
    threshold_payload = json.loads((fold_dir / "threshold.json").read_text())
    metadata_payload = json.loads((fold_dir / "metadata.json").read_text())
    best_params = json.loads((fold_dir / "best_params.json").read_text())
    calibration_model = LogisticRecalibrationModel(**calibration_payload)
    return OuterFoldRunResult(
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        best_params=best_params,
        best_score=float(metadata_payload.get("best_score", 0.0)),
        calibration_model=calibration_model,
        threshold_payload=threshold_payload,
        metadata_payload=metadata_payload,
    )


def is_outer_fold_checkpoint_complete(fold_dir: Path) -> bool:
    required = (
        fold_dir / "train_oof.csv",
        fold_dir / "test_predictions.csv",
        fold_dir / "calibration.json",
        fold_dir / "threshold.json",
        fold_dir / "metadata.json",
        fold_dir / "best_params.json",
    )
    return all(path.exists() for path in required)


def checkpoint_matches_validation_fingerprint(
    fold_dir: Path,
    validation_fingerprint: Dict[str, Any],
) -> Tuple[bool, str]:
    metadata_path = fold_dir / "metadata.json"
    if not metadata_path.exists():
        return False, "metadata.json is missing"

    try:
        metadata = json.loads(metadata_path.read_text())
    except Exception as exc:  # noqa: BLE001
        return False, f"failed to parse metadata.json: {exc}"

    if metadata.get("status") != "completed":
        return False, f"checkpoint status is {metadata.get('status')!r}, not 'completed'"

    return compare_validation_fingerprints(
        validation_fingerprint,
        metadata.get("validation_fingerprint"),
    )


def write_validation_manifest(
    artifacts_dir: Path,
    *,
    validation_fingerprint: Dict[str, Any],
    summary_metadata: Dict[str, Any],
) -> None:
    write_json(
        artifacts_dir / "validation.json",
        {
            "validation_protocol_version": validation_fingerprint["validation_protocol_version"],
            "validation_scheme": validation_fingerprint["validation_scheme"],
            "dataset_hash": validation_fingerprint["dataset_hash"],
            "validation_fingerprint": validation_fingerprint,
            **summary_metadata,
        },
    )
