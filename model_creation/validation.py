"""Shared validation engine for holdout and nested CV workflows."""
from __future__ import annotations

import json
import hashlib
import inspect
import importlib.metadata
import logging
import os
import pickle
import subprocess
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from model_creation import utils
from data_preparation.artifact_metadata import file_sha256
from model_creation.postprocessing import (
    LogisticRecalibrationModel,
    apply_logistic_recalibration,
    find_youden_j_threshold,
    fit_logistic_recalibration,
    write_json,
)
from model_creation.preprocessing import FoldPreprocessor
from model_creation.split_cache import cached_grouped_splits

logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
VALIDATION_PROTOCOL_VERSION = 4
CHECKPOINT_BUNDLE_PROTOCOL_VERSION = 2
EBM_HPO_BAGGING = {"inner_bags": 0, "outer_bags": 1}
EBM_REFIT_BAGGING = {"inner_bags": 20, "outer_bags": 14}
EBM_DIAGNOSTIC_REFIT_BAGGING = {"inner_bags": 0, "outer_bags": 2}
EBM_DIAGNOSTIC_MAX_ROUNDS = 200
GROUPED_SPLIT_STRATEGY = "minimum_class_count_imbalance_of_64_seeded_candidates"


@dataclass
class CachedFold:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    ebm_feature_types_cache: Optional[Dict[int, List]] = None
    groups_train: Optional[pd.Series] = None
    ebm_bags_cache: Optional[Dict[tuple, np.ndarray]] = None


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
    *,
    require_holdout_split: bool = False,
    cohort_scope: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Optional[pd.Series]]:
    working_df, spec = utils.prepare_working_dataset_for_outcome(
        df,
        outcome_name,
        require_holdout_split=require_holdout_split,
        cohort_scope=cohort_scope,
    )
    target_col = spec.target_col
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
    cohort_scope: Optional[str] = None,
    selected_case_hash: str = "",
    fold_assignment_hash: str = "",
    diagnostic: bool = False,
    calibration_folds: int = 5,
) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    code_digest = hashlib.sha256()
    for relative_path in (
        "model_creation/validation.py", "model_creation/preprocessing.py",
        "model_creation/split_cache.py",
        "model_creation/utils.py", "model_creation/postprocessing.py",
        "model_creation/step_07_train_evaluate.py", "data_preparation/cohort_scopes.py",
    ):
        path = project_root / relative_path
        if path.exists():
            code_digest.update(relative_path.encode())
            code_digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    environment = {}
    for package in ("scikit-learn", "numpy", "pandas", "xgboost-cpu", "interpret-core", "optuna"):
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = "unavailable"
    return {
        "validation_protocol_version": VALIDATION_PROTOCOL_VERSION,
        "checkpoint_bundle_protocol_version": CHECKPOINT_BUNDLE_PROTOCOL_VERSION,
        "dataset_hash": dataset_hash,
        "validation_scheme": validation_scheme,
        "model_type": model_type,
        "outcome": outcome,
        "branch": branch,
        "feature_set": feature_set,
        "outer_folds": int(outer_folds),
        "inner_folds": int(inner_folds),
        "calibration_folds": int(calibration_folds),
        "repeats": int(repeats),
        "n_trials": int(n_trials),
        "threads_per_model": int(threads_per_model),
        "preserve_nan": bool(preserve_nan),
        "legacy_imputation": not bool(preserve_nan),
        "base_random_state": int(base_random_state),
        "cohort_scope": cohort_scope,
        "selected_case_hash": selected_case_hash,
        "fold_assignment_hash": fold_assignment_hash,
        "diagnostic": bool(diagnostic),
        "compute_profile": (
            "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"
            if model_type == "ebm" and diagnostic else "manuscript"
        ),
        "cv_split_strategy": GROUPED_SPLIT_STRATEGY,
        "thread_environment": {name: os.getenv(name) for name in (
            "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS"
        )},
        "preprocessing_protocol": "fold-local-v4-finite-numeric-schema",
        "numeric_missing_policy": (
            "training_median_and_missingness_indicators" if model_type == "logistic_regression"
            else "native_nan" if preserve_nan else "training_median"
        ),
        "numeric_scaling": "training_standard_deviation" if model_type == "logistic_regression" else "none",
        "class_weighting": {
            "xgboost": "fold_training_scale_pos_weight_only",
            "ebm": "fold_training_sample_weight_only",
            "logistic_regression": "none",
        }.get(model_type, "unknown"),
        "code_revision": revision,
        "code_hash": code_digest.hexdigest(),
        "environment": environment,
    }


def compare_validation_fingerprints(
    expected: Dict[str, Any],
    actual: Optional[Dict[str, Any]],
) -> Tuple[bool, str]:
    if not isinstance(actual, dict):
        return False, "checkpoint validation_fingerprint metadata must be a JSON object"

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


@lru_cache(maxsize=64)
def _balanced_grouped_splits(
    outcomes: tuple, patient_ids: tuple, n_splits: int, random_state: int,
    cache_dir: Optional[str] = None,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], int, float]:
    policy = {"strategy": GROUPED_SPLIT_STRATEGY, "selector_sha256": hashlib.sha256(
        inspect.getsource(_select_balanced_grouped_splits).encode()).hexdigest()}
    return cached_grouped_splits(outcomes, patient_ids, n_splits, random_state, policy=policy,
                                 compute=lambda: _select_balanced_grouped_splits(outcomes, patient_ids, n_splits, random_state),
                                 cache_dir=cache_dir)


def _select_balanced_grouped_splits(
    outcomes: tuple, patient_ids: tuple, n_splits: int, random_state: int,
) -> tuple[tuple[tuple[np.ndarray, np.ndarray], ...], int, float]:
    """Select partitions using only class counts, never fitted-model scores."""
    y = np.asarray(outcomes)
    groups = np.asarray(patient_ids)
    _, encoded = np.unique(y, return_inverse=True)
    target = np.bincount(encoded) / n_splits
    best_splits, best_seed, best_score = None, random_state, float("inf")
    for attempt in range(64):
        seed = random_state + attempt
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros((len(y), 1)), y, groups))
        if not all(np.unique(y[train]).size == 2 and np.unique(y[val]).size == 2 for train, val in splits):
            continue
        score = float(sum(np.square((np.bincount(encoded[val], minlength=len(target)) - target) / target).sum()
                          for _, val in splits))
        if score < best_score:
            best_splits, best_seed, best_score = tuple(splits), seed, score
        if score == 0:
            break
    if best_splits is None:
        raise ValueError("Cannot construct patient-grouped folds with both classes in every partition.")
    logger.info("Grouped CV strategy=%s base_seed=%s selected_seed=%s imbalance=%.6g",
                GROUPED_SPLIT_STRATEGY, random_state, best_seed, best_score)
    return best_splits, best_seed, best_score


def describe_cv_split(y, groups, requested_splits: int, random_state: int) -> dict:
    n_splits = resolve_n_splits(y, groups, requested_splits)
    if groups is None:
        return {"strategy": "stratified_kfold", "base_seed": random_state, "selected_seed": random_state,
                "actual_folds": n_splits}
    _, seed, score = _balanced_grouped_splits(tuple(y.tolist()), tuple(groups.tolist()), n_splits, random_state,
                                             os.getenv("AKI_SPLIT_CACHE_DIR", ""))
    return {"strategy": GROUPED_SPLIT_STRATEGY, "base_seed": random_state, "selected_seed": seed,
            "class_count_imbalance": score, "actual_folds": n_splits}


def build_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Optional[pd.Series],
    requested_splits: int,
    random_state: int,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], int]:
    n_splits = resolve_n_splits(y, groups, requested_splits)
    if groups is not None:
        if len(X) != len(y) or len(groups) != len(y):
            raise ValueError("Features, outcomes, and patient groups must have equal lengths.")
        splits, _, _ = _balanced_grouped_splits(tuple(y.tolist()), tuple(groups.tolist()), n_splits, random_state,
                                               os.getenv("AKI_SPLIT_CACHE_DIR", ""))
        # Do not expose cached mutable arrays to callers.
        return [(train.copy(), val.copy()) for train, val in splits], n_splits
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


def get_effective_refit_params(model_type: str, params: Dict[str, Any], diagnostic: bool = False) -> Dict[str, Any]:
    effective = dict(params)
    if model_type == "ebm":
        effective.update(EBM_DIAGNOSTIC_REFIT_BAGGING if diagnostic else EBM_REFIT_BAGGING)
        if diagnostic:
            effective["max_rounds"] = min(int(effective.get("max_rounds", EBM_DIAGNOSTIC_MAX_ROUNDS)), EBM_DIAGNOSTIC_MAX_ROUNDS)
            effective["smoothing_rounds"] = min(int(effective.get("smoothing_rounds", 50)), 50)
    return effective


def get_refit_param_overrides(model_type: str, params: Dict[str, Any], diagnostic: bool = False) -> Dict[str, Any]:
    if model_type != "ebm":
        return {}

    effective = get_effective_refit_params(model_type, params, diagnostic)
    return {
        key: effective[key]
        for key in sorted(effective)
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


def suggest_logistic_regression_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-4, 1e4, log=True)}


def build_model(
    model_type: str,
    params: Dict[str, Any],
    *,
    random_state: int,
    threads_per_model: int,
    scale_pos_weight: float,
    feature_types: Optional[List] = None,
) -> Any:
    if model_type == "logistic_regression":
        return LogisticRegression(
            penalty="l2", solver="lbfgs", class_weight=None,
            max_iter=5000, random_state=random_state, **params,
        )
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
    model_type: str = "xgboost",
) -> Tuple[FoldPreprocessor, pd.DataFrame]:
    is_logistic = model_type == "logistic_regression"
    preprocessor = FoldPreprocessor(
        impute_missing=is_logistic or not preserve_nan,
        scale_numeric=is_logistic, add_missing_indicators=is_logistic,
    )
    X_train_transformed = preprocessor.fit_transform(X_train)
    return preprocessor, X_train_transformed


def make_ebm_bags(
    y: pd.Series, groups: Optional[pd.Series], *, n_bags: int,
    validation_size: float, random_state: int,
) -> np.ndarray:
    """Assign whole patients to each EBM early-stopping validation bag."""
    if groups is None or groups.isna().any():
        raise ValueError("Patient identifiers are required for EBM internal validation bags.")
    patients = pd.DataFrame({"group": groups.to_numpy(), "y": y.to_numpy()}).groupby(
        "group", sort=False
    )["y"].max()
    bags = np.ones((len(y), n_bags), dtype=np.int8)
    if validation_size <= 0:
        return bags
    if len(patients) < 2:
        raise ValueError("EBM internal validation needs at least two patient groups.")
    can_stratify = patients.nunique() == 2 and patients.value_counts().min() >= 2
    n_val = int(np.ceil(len(patients) * validation_size)) if validation_size < 1 else int(validation_size)
    minimum = 2 if can_stratify else 1
    n_val = min(max(minimum, n_val), len(patients) - minimum)
    for bag_id in range(n_bags):
        for attempt in range(64):
            seed = random_state + bag_id * 64 + attempt
            if can_stratify:
                negative = patients.index[patients.eq(0)].to_numpy()
                positive = patients.index[patients.eq(1)].to_numpy()
                n_positive = min(max(1, round(n_val * len(positive) / len(patients))), len(positive) - 1)
                n_negative = min(max(1, n_val - n_positive), len(negative) - 1)
                n_positive = n_val - n_negative
                rng = np.random.default_rng(seed)
                val_groups = np.concatenate([
                    rng.choice(negative, n_negative, replace=False),
                    rng.choice(positive, n_positive, replace=False),
                ])
            else:
                _, val_groups = train_test_split(
                    patients.index.to_numpy(), test_size=n_val, random_state=seed,
                )
            val_mask = groups.isin(val_groups).to_numpy()
            if y.iloc[np.flatnonzero(val_mask)].nunique() == 2 and y.iloc[np.flatnonzero(~val_mask)].nunique() == 2:
                bags[val_mask, bag_id] = -1
                break
        else:
            raise ValueError("Cannot make patient-grouped EBM internal partitions with both classes.")
    return bags


def fit_estimator(
    model: Any, X: pd.DataFrame, y: pd.Series, *, model_type: str,
    groups: Optional[pd.Series], params: Dict[str, Any], random_state: int,
    bags_cache: Optional[Dict[tuple, np.ndarray]] = None,
) -> Any:
    """Keep family-specific weighting, grouped bags, and convergence in one path."""
    if model_type == "xgboost":
        model.fit(X, y, verbose=False)
    elif model_type == "ebm":
        key = (int(params.get("outer_bags", 14)), float(params.get("validation_size", 0.15)), random_state)
        if bags_cache is None:
            bags_cache = {}
        if key not in bags_cache:
            bags_cache[key] = make_ebm_bags(
                y, groups, n_bags=key[0], validation_size=key[1], random_state=key[2]
            )
        model.fit(X, y, sample_weight=make_sample_weight(y), bags=bags_cache[key])
    elif model_type == "logistic_regression":
        for max_iter in (5000, 10000):
            model.set_params(max_iter=max_iter)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X, y)
            convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
            for item in caught:
                if not issubclass(item.category, ConvergenceWarning):
                    warnings.warn(str(item.message), item.category, stacklevel=2)
            reached_limit = np.asarray(getattr(model, "n_iter_", [])).size and np.max(model.n_iter_) >= max_iter
            if not convergence and not reached_limit:
                return model
            logger.warning("Logistic regression did not converge at max_iter=%s.", max_iter)
        raise RuntimeError("Logistic regression failed to converge after 10000 iterations.")
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model


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
        preprocessor, X_train_transformed = fit_preprocessor(
            X_train_raw, preserve_nan=preserve_nan, model_type=model_type
        )
        X_val_transformed = preprocessor.transform(X_val_raw)
        cached_folds.append(
            CachedFold(
                X_train=X_train_transformed,
                X_val=X_val_transformed,
                y_train=y_train,
                y_val=y_val,
                ebm_feature_types_cache={} if model_type == "ebm" else None,
                groups_train=groups.iloc[train_idx] if groups is not None else None,
                ebm_bags_cache={} if model_type == "ebm" else None,
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
    diagnostic: bool = False,
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
        suggest = {
            "xgboost": suggest_xgboost_params, "ebm": suggest_ebm_params,
            "logistic_regression": suggest_logistic_regression_params,
        }
        params = suggest[model_type](trial)
        if model_type == "ebm" and diagnostic:
            params = get_effective_refit_params(model_type, params, diagnostic=True)
            params.update(EBM_HPO_BAGGING)
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
            fit_estimator(
                model, fold.X_train, fold.y_train, model_type=model_type,
                groups=fold.groups_train, params=params, random_state=random_state,
                bags_cache=fold.ebm_bags_cache,
            )
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
    diagnostic: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    split_list, actual_splits = build_cv_splits(X, y, groups, requested_splits, random_state)
    predictions = np.empty(len(X), dtype=float)
    fold_indices = np.empty(len(X), dtype=int)
    effective_params = get_effective_refit_params(model_type, best_params, diagnostic)

    for fold_id, (train_idx, val_idx) in enumerate(split_list):
        X_train_raw = X.iloc[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_train = y.iloc[train_idx]
        preprocessor, X_train_transformed = fit_preprocessor(
            X_train_raw, preserve_nan=preserve_nan, model_type=model_type
        )
        X_val_transformed = preprocessor.transform(X_val_raw)
        scale_pos_weight = compute_scale_pos_weight(y_train)
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
        fit_estimator(
            model, X_train_transformed, y_train, model_type=model_type,
            groups=groups.iloc[train_idx] if groups is not None else None,
            params=effective_params, random_state=random_state,
        )
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
    groups: Optional[pd.Series] = None,
    diagnostic: bool = False,
) -> Tuple[FoldPreprocessor, Any, pd.DataFrame]:
    preprocessor, X_train_transformed = fit_preprocessor(
        X_train, preserve_nan=preserve_nan, model_type=model_type
    )
    scale_pos_weight = compute_scale_pos_weight(y_train)
    effective_params = get_effective_refit_params(model_type, best_params, diagnostic)
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
    fit_estimator(
        model, X_train_transformed, y_train, model_type=model_type,
        groups=groups, params=effective_params, random_state=random_state,
    )
    return preprocessor, model, X_train_transformed


def save_model_bundle(
    output_dir: Path,
    *,
    model_type: str,
    model: Any,
    preprocessor: FoldPreprocessor,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = output_dir / "preprocessor.pkl"
    preprocessor_tmp = output_dir / "preprocessor.pkl.tmp"
    with preprocessor_tmp.open("wb") as fp:
        pickle.dump(preprocessor, fp)
    preprocessor_tmp.replace(preprocessor_path)

    if model_type == "xgboost":
        model_path = output_dir / "model.json"
        model_tmp = output_dir / "model.tmp.json"
        model.save_model(model_tmp)
        model_tmp.replace(model_path)
    else:
        model_path = output_dir / "model.pkl"
        model_tmp = output_dir / "model.pkl.tmp"
        with model_tmp.open("wb") as fp:
            pickle.dump(model, fp)
        model_tmp.replace(model_path)
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
    cohort_scope: Optional[str] = None,
    diagnostic: bool = False,
    calibration_folds: Optional[int] = None,
) -> OuterFoldRunResult:
    # Preserve explicit reduced low-level diagnostics; manuscript callers pass five.
    calibration_folds = inner_folds if calibration_folds is None else calibration_folds
    train_idx, test_idx = list(train_idx), list(test_idx)
    X_train_raw = X.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    train_groups = groups.iloc[train_idx] if groups is not None else None
    test_groups = groups.iloc[test_idx] if groups is not None else None

    if set(train_idx) & set(test_idx) or set(X_train_raw.index) & set(X_test_raw.index):
        raise AssertionError("Train and test indices overlap within outer split.")
    if groups is not None and set(train_groups) & set(test_groups):
        raise AssertionError("Patient overlap detected across outer split.")

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
        diagnostic=diagnostic,
    )

    oof_predictions, fold_indices, actual_calibration_folds = generate_cross_fitted_predictions(
        X_train_raw,
        y_train,
        train_groups,
        model_type=model_type,
        best_params=best_params,
        requested_splits=calibration_folds,
        random_state=random_state,
        preserve_nan=preserve_nan,
        threads_per_model=threads_per_model,
        diagnostic=diagnostic,
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
        groups=train_groups,
        diagnostic=diagnostic,
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

    effective_refit_params = get_effective_refit_params(model_type, best_params, diagnostic)
    metadata_base = {
        "validation_scheme": validation_scheme,
        "cohort_scope": cohort_scope,
        "model_type": model_type,
        "diagnostic": bool(diagnostic),
        "compute_profile": (
            "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"
            if model_type == "ebm" and diagnostic else "manuscript"
        ),
        "inner_split": describe_cv_split(y_train, train_groups, inner_folds, random_state),
        "calibration_split": describe_cv_split(y_train, train_groups, calibration_folds, random_state),
        "repeat_id": int(repeat_id),
        "outer_fold_id": int(outer_fold_id),
        "random_state": int(random_state),
        "n_trials": int(n_trials),
        "requested_inner_folds": int(inner_folds),
        "requested_calibration_folds": int(calibration_folds),
        "actual_inner_folds": int(actual_inner_folds),
        "actual_calibration_folds": int(actual_calibration_folds),
        "best_score": float(best_score),
        "best_params": json.loads(json.dumps(best_params)),
        "effective_refit_params": json.loads(json.dumps(effective_refit_params)),
        "refit_param_overrides": json.loads(
            json.dumps(get_refit_param_overrides(model_type, best_params, diagnostic))
        ),
    }

    train_predictions = pd.DataFrame(
        {
            "caseid": caseids.iloc[train_idx].values,
            "y_true": y_train.values,
            "y_prob_raw": oof_predictions,
            "y_prob_calibrated": calibrated_oof,
            "threshold": threshold,
            "y_pred_label": (calibrated_oof >= threshold).astype(int),
            "fold": fold_indices,
            "outer_fold_id": outer_fold_id,
            "repeat_id": repeat_id,
            "validation_scheme": validation_scheme,
            "cohort_scope": cohort_scope,
            "model_type": model_type,
            "diagnostic": bool(diagnostic),
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
            "caseid": caseids.iloc[test_idx].values,
            "y_true": y_test.values,
            "y_prob_raw": test_pred_proba,
            "y_prob_calibrated": test_pred_calibrated,
            "threshold": threshold,
            "y_pred_label": (test_pred_calibrated >= threshold).astype(int),
            "fold": outer_fold_id,
            "outer_fold_id": outer_fold_id,
            "repeat_id": repeat_id,
            "validation_scheme": validation_scheme,
            "cohort_scope": cohort_scope,
            "model_type": model_type,
            "diagnostic": bool(diagnostic),
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
                "patients": int(train_groups.nunique()) if train_groups is not None else None,
                "positive": int((y_train == 1).sum()),
                "negative": int((y_train == 0).sum()),
            },
            "test": {
                "total": int(len(y_test)),
                "patients": int(test_groups.nunique()) if test_groups is not None else None,
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
    if result.final_model is None or result.final_preprocessor is None:
        raise ValueError("A completed outer-fold checkpoint requires its fitted model and preprocessor.")
    if validation_fingerprint.get("checkpoint_bundle_protocol_version") != CHECKPOINT_BUNDLE_PROTOCOL_VERSION:
        raise ValueError("A new checkpoint requires the current bundle protocol fingerprint.")
    write_json(fold_dir / "metadata.json", {"status": "writing"})
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
    model_path = save_model_bundle(
        fold_dir, model_type=validation_fingerprint["model_type"],
        model=result.final_model, preprocessor=result.final_preprocessor,
    )
    write_json(fold_dir / "best_params.json", result.best_params)
    artifact_hashes = {
        name: file_sha256(fold_dir / name)
        for name in (model_path.name, "preprocessor.pkl", "train_oof.csv", "test_predictions.csv",
                     "calibration.json", "threshold.json", "best_params.json")
    }
    write_json(
        fold_dir / "metadata.json",
        {
            **result.metadata_payload,
            "dataset_hash": validation_fingerprint["dataset_hash"],
            "validation_protocol_version": validation_fingerprint["validation_protocol_version"],
            "validation_fingerprint": validation_fingerprint,
            "model_type": validation_fingerprint["model_type"],
            "cohort_scope": validation_fingerprint.get("cohort_scope"),
            "model_path": model_path.name,
            "model_sha256": artifact_hashes[model_path.name],
            "preprocessor_sha256": artifact_hashes["preprocessor.pkl"],
            "checkpoint_bundle_protocol_version": CHECKPOINT_BUNDLE_PROTOCOL_VERSION,
            "artifact_sha256": artifact_hashes,
            "status": "completed",
        },
    )


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
    if not all(path.exists() for path in required):
        return False
    try:
        metadata = json.loads((fold_dir / "metadata.json").read_text())
        if not isinstance(metadata, dict):
            return False
        if metadata.get("checkpoint_bundle_protocol_version") != CHECKPOINT_BUNDLE_PROTOCOL_VERSION:
            return False
        fingerprint = metadata.get("validation_fingerprint")
        if not isinstance(fingerprint, dict) or fingerprint.get("checkpoint_bundle_protocol_version") != CHECKPOINT_BUNDLE_PROTOCOL_VERSION:
            return False
        model_path = fold_dir / metadata["model_path"]
        expected_model = "model.json" if metadata.get("model_type") == "xgboost" else "model.pkl"
        if metadata["model_path"] != expected_model:
            return False
        preprocessor_path = fold_dir / "preprocessor.pkl"
        artifact_hashes = metadata.get("artifact_sha256")
        expected_files = {model_path.name, "preprocessor.pkl", "train_oof.csv", "test_predictions.csv",
                          "calibration.json", "threshold.json", "best_params.json"}
        if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_files:
            return False
        return (
            metadata.get("status") == "completed"
            and artifact_hashes[model_path.name] == metadata["model_sha256"]
            and artifact_hashes[preprocessor_path.name] == metadata["preprocessor_sha256"]
            and all(file_sha256(fold_dir / name) == digest
                    for name, digest in artifact_hashes.items())
        )
    except (OSError, KeyError, ValueError, TypeError):
        return False


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

    if not isinstance(metadata, dict):
        return False, "metadata.json must contain a JSON object"
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
            "cohort_scope": validation_fingerprint.get("cohort_scope"),
            "selected_case_hash": validation_fingerprint.get("selected_case_hash"),
            "diagnostic": validation_fingerprint.get("diagnostic", False),
            **summary_metadata,
        },
    )
