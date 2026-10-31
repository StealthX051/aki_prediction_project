"""Aggregate prediction metrics for held-out test sets.

This script crawls `<results_dir>/models/**/predictions/test.csv`, validates
required artifacts and columns, computes metrics using stored thresholds, and
writes consolidated outputs to `<results_dir>/tables/metrics_summary.csv`.
Optional bootstrap samples can also be saved to Parquet for further analysis.
When patient identifiers are present in the prediction files, bootstrap
resampling is clustered at the patient level.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from functools import lru_cache
import os
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from multiprocessing import TimeoutError as ProcessTimeoutError
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from joblib import Parallel, delayed, effective_n_jobs
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from artifact_paths import enforce_storage_policy, get_scoped_results_dir, get_scoped_paper_dir
from data_preparation.inputs import COHORT_FILE, INPUT_FILE
from model_creation.prediction_io import REQUIRED_PREDICTION_COLUMNS, validate_prediction_dataframe
from reporting.manuscript_assets import export_dataframe_bundle, refresh_paper_bundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESULTS_DIR = get_scoped_results_dir(PROJECT_ROOT, os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv"))
PAPER_DIR = get_scoped_paper_dir(PROJECT_ROOT, os.getenv("COHORT_SCOPE", "main_shared"), os.getenv("VALIDATION_SCHEME", "nested_cv"))
TABLES_DIR = RESULTS_DIR / "tables"
PAPER_TABLES_DIR = PAPER_DIR / "tables"

ARTIFACT_FILES = ("calibration.json", "threshold.json")
UNCERTAINTY_PROTOCOL_VERSION = 3
PATIENT_ID_COLUMNS = ("subjectid", "subject_id")


@dataclass
class PredictionSet:
    """Container for a single model's test predictions and metadata."""

    path: Path
    df: pd.DataFrame
    threshold: float
    outcome: str
    branch: str
    feature_set: str
    model_name: str
    pipeline: Optional[str]
    cohort_scope: str = "legacy"
    validation_scheme: str = "legacy"
    analysis_status: str = "legacy"


class ValidationError(Exception):
    """Raised when a predictions file fails validation."""


def _validate_artifacts(pred_path: Path) -> None:
    """Ensure calibration/threshold artifacts exist alongside predictions.

    The expected layout is `<model_dir>/predictions/test.csv` with artifacts in
    `<model_dir>/artifacts/`.
    """

    model_dir = pred_path.parent.parent
    artifact_dir = model_dir / "artifacts"
    for name in ARTIFACT_FILES:
        artifact_path = artifact_dir / name
        if not artifact_path.exists():
            raise ValidationError(f"Missing artifact {artifact_path} for {pred_path}")


def _validate_dataframe(df: pd.DataFrame, path: Path) -> float:
    """Validate required columns and return the unique threshold value."""

    try:
        return validate_prediction_dataframe(
            df,
            path,
            REQUIRED_PREDICTION_COLUMNS,
            require_unique_caseids=True,
            require_single_threshold=False,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _assert_single_value(df: pd.DataFrame, col: str, path: Path) -> str:
    values = df[col].dropna().unique()
    if len(values) != 1:
        raise ValidationError(f"{path} expected a single value for '{col}', found {values}")
    return str(values[0])


def _summarize_threshold(values: Union[pd.Series, np.ndarray]) -> float:
    thresholds = np.asarray(values, dtype=float)
    if thresholds.size == 0:
        return float("nan")
    return float(np.median(thresholds))


@lru_cache(maxsize=1)
def _load_case_subject_lookup() -> Optional[pd.Series]:
    lookup_frames: List[pd.DataFrame] = []
    for source_path, source_name in ((COHORT_FILE, "cohort"), (INPUT_FILE, "clinical")):
        if not source_path.exists():
            logger.warning(
                "%s file %s not found; cannot use it for patient-ID recovery during clustered bootstrap.",
                source_name.capitalize(),
                source_path,
            )
            continue

        try:
            source_df = pd.read_csv(source_path, usecols=["caseid", "subjectid"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to load case-to-subject mapping from %s: %s", source_path, exc)
            continue

        if source_df["caseid"].duplicated().any():
            logger.warning("%s contains duplicate caseids; skipping it for patient-ID recovery.", source_path)
            continue

        lookup_frames.append(source_df)

    if not lookup_frames:
        return None

    combined_lookup = pd.concat(lookup_frames, ignore_index=True)
    combined_lookup = combined_lookup.drop_duplicates(subset="caseid", keep="first")
    return combined_lookup.set_index("caseid")["subjectid"]


def _attach_patient_ids(df: pd.DataFrame, pred_path: Path) -> pd.DataFrame:
    if any(column in df.columns for column in PATIENT_ID_COLUMNS):
        return df

    case_subject_lookup = _load_case_subject_lookup()
    if case_subject_lookup is None:
        return df

    mapped_subjectids = df["caseid"].map(case_subject_lookup)
    if mapped_subjectids.isna().any():
        missing = int(mapped_subjectids.isna().sum())
        logger.warning(
            "Could not recover patient IDs for %s/%s predictions from %s missing caseids in %s or %s; falling back to case-level bootstrap.",
            pred_path.parent.parent.parent.name,
            pred_path.name,
            missing,
            COHORT_FILE,
            INPUT_FILE,
        )
        return df

    enriched_df = df.copy()
    enriched_df["subjectid"] = mapped_subjectids.to_numpy()
    logger.info("Recovered subject IDs for %s using %s and %s.", pred_path, COHORT_FILE, INPUT_FILE)
    return enriched_df


def _load_predictions(pred_path: Path, *, strict: bool = False) -> PredictionSet:
    df = pd.read_csv(pred_path)
    threshold = _validate_dataframe(df, pred_path)
    _validate_artifacts(pred_path)
    if strict:
        for column in ("subjectid", "cohort_scope", "validation_scheme"):
            if column not in df or df[column].isna().any():
                raise ValidationError(f"{pred_path} requires nonmissing {column} for definitive reporting")
    else:
        df = _attach_patient_ids(df, pred_path)

    outcome = _assert_single_value(df, "outcome", pred_path)
    branch = _assert_single_value(df, "branch", pred_path)
    feature_set = _assert_single_value(df, "feature_set", pred_path)
    model_name = _assert_single_value(df, "model_name", pred_path)
    pipeline = (
        _assert_single_value(df, "pipeline", pred_path)
        if "pipeline" in df.columns and not df["pipeline"].dropna().empty
        else None
    )

    return PredictionSet(
        path=pred_path,
        df=df,
        threshold=threshold,
        outcome=outcome,
        branch=branch,
        feature_set=feature_set,
        model_name=model_name,
        pipeline=pipeline,
        cohort_scope=_assert_single_value(df, "cohort_scope", pred_path) if "cohort_scope" in df else "legacy",
        validation_scheme=_assert_single_value(df, "validation_scheme", pred_path) if "validation_scheme" in df else "legacy",
    )


def crawl_predictions(results_dir: Path, *, strict: bool = False, run_manifest: Optional[Path] = None, require_reference: bool = False, reference_feature_set: str = "preop_only", reference_model_name: Optional[str] = None) -> List[PredictionSet]:
    """Discover and load all test prediction files under the results directory."""

    patterns = [
        results_dir / "models" / "**" / "predictions" / "test.csv",
    ]

    pred_files: List[Path] = []
    for pattern in patterns:
        if not results_dir.exists():
            matched: List[Path] = []
        else:
            matched = list(results_dir.glob(str(pattern.relative_to(results_dir))))
        if matched:
            pred_files.extend(matched)
            logger.info("Found %d prediction files for pattern %s", len(matched), pattern)

    manifest_path = run_manifest or results_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for config in manifest.get("configurations", []):
            if "artifact_path" in config:
                if not config.get("reuse_reference"):
                    raise ValidationError("External artifacts require a verified reuse reference")
                pred_files.append(Path(config["artifact_path"]) / "predictions" / "test.csv")
    pred_files = list(dict.fromkeys(path.resolve() for path in pred_files))

    if not pred_files:
        raise FileNotFoundError(
            "No test prediction files found. Expected them under 'results/models/**/predictions/test.csv' "
            "or 'results/aeon/models/**/predictions/test.csv'."
        )

    prediction_sets: List[PredictionSet] = []
    for pred_file in pred_files:
        logger.info("Loading predictions from %s", pred_file)
        try:
            prediction_sets.append(_load_predictions(pred_file, strict=strict))
        except ValidationError as exc:
            if strict:
                raise
            logger.warning("Skipping invalid prediction file %s: %s", pred_file, exc)

    if not prediction_sets:
        raise FileNotFoundError(
            "No valid test prediction files found. Expected them under 'results/models/**/predictions/test.csv' "
            "or 'results/aeon/models/**/predictions/test.csv'."
        )

    if strict:
        prediction_sets = _validate_run_manifest(prediction_sets, run_manifest or results_dir / "run_manifest.json", require_reference=require_reference, reference_feature_set=reference_feature_set, reference_model_name=reference_model_name)
    return prediction_sets


def _model_id(name: str) -> str:
    return {"XGBoost": "xgboost", "EBM": "ebm", "Logistic regression": "logistic_regression"}.get(name, name)


def _validate_run_manifest(prediction_sets: List[PredictionSet], path: Path, *, require_reference: bool = False, reference_feature_set: str = "preop_only", reference_model_name: Optional[str] = None) -> List[PredictionSet]:
    if not path.exists():
        raise ValidationError(f"Definitive reporting requires run manifest: {path}")
    manifest = json.loads(path.read_text())
    expected = {(c["model_type"], c["outcome"], c["branch"], c["feature_set"]) for c in manifest["configurations"]}
    if not expected:
        raise ValidationError("Run manifest contains no requested configurations")
    scope, scheme = manifest["cohort_scope"], manifest["validation_scheme"]
    found = {}
    for ps in prediction_sets:
        if (ps.cohort_scope, ps.validation_scheme) != (scope, scheme):
            raise ValidationError(f"Mixed cohort or validation scheme in {ps.path}")
        key = (_model_id(ps.model_name), ps.outcome, ps.branch, ps.feature_set)
        if key in found:
            raise ValidationError(f"Duplicate prediction configuration: {key}")
        found[key] = ps
    missing = expected - found.keys()
    if missing:
        raise ValidationError(f"Missing required prediction configurations: {sorted(missing)}")
    requested = set(expected)
    for family, outcome, branch, feature in requested:
        if require_reference and reference_model_name and _model_id(reference_model_name) != family:
            raise ValidationError("Strict reporting requires a same-family reference; cross-family overrides require --legacy")
        reference = (_model_id(reference_model_name) if reference_model_name else family, outcome, branch, reference_feature_set)
        if reference in found:
            expected.add(reference)
        elif require_reference:
            raise ValidationError(f"Missing same-family reference dependency: {reference}")
    from data_preparation.cohort_scopes import selected_case_hash
    protocol_keys = ("validation_protocol_version", "code_revision", "code_hash", "environment", "preprocessing_protocol")
    expected_protocol = manifest.get("study_identity")
    config_by_key = {(c["model_type"], c["outcome"], c["branch"], c["feature_set"]): c
                     for c in manifest["configurations"]}
    diagnostic = bool(manifest.get("diagnostic", False))
    if not diagnostic and not expected_protocol:
        raise ValidationError("Definitive reporting requires study_identity in the run manifest")
    for key in expected:
        ps = found[key]
        roster = manifest.get("branch_rosters", {}).get(ps.branch)
        if not roster:
            raise ValidationError(f"Missing frozen branch roster for {ps.branch}")
        roster_frame = ps.df
        if scheme == "holdout":
            train_path = ps.path.parent / "train_oof.csv"
            if not train_path.exists():
                raise ValidationError(f"Holdout roster verification requires {train_path}")
            training = pd.read_csv(train_path)
            if (set(training["caseid"]) & set(ps.df["caseid"])
                    or set(training["subjectid"]) & set(ps.df["subjectid"])):
                raise ValidationError(f"Holdout training/evaluation operation or patient overlap: {ps.path}")
            roster_frame = pd.concat([training[["caseid", "subjectid"]], ps.df[["caseid", "subjectid"]]], ignore_index=True)
        actual_hash = selected_case_hash(roster_frame)
        if actual_hash != roster["selected_case_hash"]:
            raise ValidationError(f"Prediction cases/patients do not match frozen branch roster: {ps.path}")
        if "selected_case_hash" in ps.df:
            if ps.df["selected_case_hash"].isna().any():
                raise ValidationError(f"Missing prediction selected_case_hash: {ps.path}")
            declared = _assert_single_value(ps.df, "selected_case_hash", ps.path)
            if declared != actual_hash:
                raise ValidationError(f"Prediction selected_case_hash does not match saved rows: {ps.path}")
        artifact_dir = ps.path.parent.parent / "artifacts"
        metadata_path = artifact_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValidationError(f"Missing saved validation metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        fingerprint = metadata.get("validation_fingerprint", {})
        dataset_hash = roster.get("dataset_sha256")
        if (not diagnostic and not dataset_hash) or (dataset_hash and fingerprint.get("dataset_hash") != dataset_hash):
            raise ValidationError(f"Saved modeling dataset differs from the frozen branch dataset hash: {metadata_path}")
        protocol = {name: fingerprint.get(name) for name in protocol_keys}
        if any(value is None for value in protocol.values()):
            raise ValidationError(f"Incomplete saved protocol identity: {metadata_path}")
        if expected_protocol is None:
            expected_protocol = protocol
        config = config_by_key.get(key, {})
        if config.get("reuse_reference"):
            try:
                from model_creation.run_inventory import verify_legacy_reference
                reference = verify_legacy_reference(config, manifest.get("compatibility_attestation", {}), expected_protocol)
            except (ValueError, OSError, KeyError) as exc:
                raise ValidationError(f"Invalid protected reference: {metadata_path}: {exc}") from exc
            if ps.path.resolve() != Path(reference["prediction_path"]):
                raise ValidationError("Protected reference prediction path differs from imported inventory")
        elif protocol != expected_protocol:
            raise ValidationError(f"Mixed or stale model code/environment protocol: {metadata_path}")
        declared_diagnostic = ps.df.get("diagnostic", pd.Series(dtype=object)).astype(str).str.lower().map(
            {"true": True, "false": False, "1": True, "0": False})
        if (len(declared_diagnostic) != len(ps.df) or declared_diagnostic.isna().any()
                or not declared_diagnostic.eq(diagnostic).all()
                or metadata.get("diagnostic") != diagnostic
                or fingerprint.get("diagnostic") != diagnostic):
            raise ValidationError(f"Diagnostic status conflicts with saved validation: {metadata_path}")
        if fingerprint.get("cohort_scope") != scope or fingerprint.get("validation_scheme") != scheme:
            raise ValidationError(f"Saved protocol cohort/validation scope mismatch: {metadata_path}")
        compute_profile = fingerprint.get("compute_profile")
        if compute_profile not in {"manuscript", "diagnostic_ebm_200_rounds_2_outer_bags_no_inner_bags"}:
            raise ValidationError(f"Missing or unknown model compute profile: {metadata_path}")
        if metadata.get("compute_profile") != compute_profile:
            raise ValidationError(f"Model compute profile conflicts with saved fingerprint: {metadata_path}")
        if not diagnostic and compute_profile != "manuscript":
            raise ValidationError(f"Diagnostic compute profile cannot be promoted to definitive results: {metadata_path}")
        if not diagnostic:
            if (scheme != "nested_cv" or metadata.get("outer_folds") != 5 or metadata.get("inner_folds") != 3
                    or metadata.get("calibration_folds") != 5
                    or metadata.get("repeats") != 1 or fingerprint.get("outer_folds") != 5
                    or fingerprint.get("inner_folds") != 3 or fingerprint.get("calibration_folds") != 5
                    or fingerprint.get("repeats") != 1
                    or "outer_fold_id" not in ps.df or ps.df.outer_fold_id.nunique() != 5
                    or "repeat_id" not in ps.df or ps.df.repeat_id.nunique() != 1):
                raise ValidationError(f"Definitive report requires 5 outer / 3 HPO / 5 calibration folds and one repeat: {metadata_path}")
            fold_paths = list(artifact_dir.glob("folds/repeat_*/outer_*/metadata.json"))
            if len(fold_paths) != 5:
                raise ValidationError(f"Definitive report requires five completed fold bundles: {artifact_dir}")
            for fold_path in fold_paths:
                fold = json.loads(fold_path.read_text())
                from model_creation.validation import is_outer_fold_checkpoint_complete
                if (fold.get("status") != "completed" or fold.get("actual_inner_folds") != 3
                        or fold.get("requested_inner_folds") != 3
                        or fold.get("requested_calibration_folds") != 5
                        or fold.get("actual_calibration_folds") != 5
                        or fold.get("validation_fingerprint") != fingerprint
                        or not is_outer_fold_checkpoint_complete(fold_path.parent)):
                    raise ValidationError(f"Incomplete or reduced definitive validation fold: {fold_path}")
        if scheme == "nested_cv" and (not diagnostic or fingerprint.get("checkpoint_bundle_protocol_version") == 2):
            _validate_nested_prediction_union(ps, artifact_dir, fingerprint)
    status = "diagnostic" if diagnostic else "definitive"
    return [replace(found[key], analysis_status=status) for key in sorted(expected)]


def _validate_nested_prediction_union(ps: PredictionSet, artifact_dir: Path, fingerprint: dict) -> None:
    """Require pooled predictions to be the saved, verified outer-test union."""
    from model_creation.validation import is_outer_fold_checkpoint_complete
    fold_paths = sorted(artifact_dir.glob("folds/repeat_*/outer_*/metadata.json"))
    if not fold_paths:
        raise ValidationError(f"Missing saved outer-fold prediction bundles: {artifact_dir}")
    frames = []
    for path in fold_paths:
        metadata = json.loads(path.read_text())
        if metadata.get("validation_fingerprint") != fingerprint or not is_outer_fold_checkpoint_complete(path.parent):
            raise ValidationError(f"Invalid saved outer-fold bundle: {path.parent}")
        frame = pd.read_csv(path.parent / "test_predictions.csv")
        for column in ("repeat_id", "outer_fold_id"):
            if column not in frame or frame[column].isna().any() or not frame[column].eq(metadata.get(column)).all():
                raise ValidationError(f"Saved fold prediction identity mismatch: {path}")
        frames.append(frame)
    saved = pd.concat(frames, ignore_index=True)
    keys = ["caseid", "repeat_id", "outer_fold_id"]
    values = ["subjectid", "y_true", "y_prob_raw", "y_prob_calibrated", "threshold", "y_pred_label"]
    columns = keys + values
    for frame in (ps.df, saved):
        if any(column not in frame for column in columns) or frame[columns].isna().any().any() or frame.duplicated(keys).any():
            raise ValidationError(f"Incomplete or duplicate root/fold prediction identity: {ps.path}")
    root = ps.df.sort_values(keys).reset_index(drop=True)
    saved = saved.sort_values(keys).reset_index(drop=True)
    exact = keys + ["subjectid", "y_true", "y_pred_label"]
    if (len(root) != len(saved) or not root[exact].equals(saved[exact])
            or not np.allclose(root[["y_prob_raw", "y_prob_calibrated", "threshold"]],
                               saved[["y_prob_raw", "y_prob_calibrated", "threshold"]], rtol=0, atol=1e-12)):
        raise ValidationError(f"Root predictions differ from verified outer-fold prediction union: {ps.path}")


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:
        return float("nan")


def compute_point_metrics(pred_set: PredictionSet) -> Dict[str, float]:
    """Compute threshold-based metrics on test predictions."""

    y_true = pred_set.df["y_true"].values
    y_prob = pred_set.df["y_prob_calibrated"].values
    y_pred = pred_set.df["y_pred_label"].astype(int).values
    threshold = _summarize_threshold(pred_set.df["threshold"].values)

    return _compute_metrics_arrays(y_true, y_prob, y_pred, threshold)


def _compute_metrics_arrays(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Fast path to compute metrics without constructing a PredictionSet."""

    has_positive = bool(np.any(y_true == 1))
    has_negative = bool(np.any(y_true == 0))
    metrics = {
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "threshold": threshold,
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob) if has_positive and has_negative else float("nan"),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob) if has_positive else float("nan"),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "sensitivity": _safe_metric(recall_score, y_true, y_pred) if has_positive else float("nan"),
    }

    tn, fp, fn, tp = (0, 0, 0, 0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # Invalid labels or shapes. Leave defaults; single-class binary draws
        # have a valid 2x2 matrix because labels are explicit above.
        pass

    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["f1"] = _safe_metric(f1_score, y_true, y_pred, zero_division=0)

    return metrics


def _get_cluster_column(df: pd.DataFrame) -> str:
    """Return the patient-cluster column when available, else fall back to case-level rows."""
    for column in PATIENT_ID_COLUMNS:
        if column in df.columns:
            return column
    return "caseid"


def _get_cluster_ids(df: pd.DataFrame) -> np.ndarray:
    cluster_col = _get_cluster_column(df)
    return df[cluster_col].to_numpy()


def _prediction_frames_are_aligned(reference_df: pd.DataFrame, candidate_df: pd.DataFrame) -> bool:
    if len(reference_df) != len(candidate_df):
        return False
    if not np.array_equal(reference_df["caseid"].to_numpy(), candidate_df["caseid"].to_numpy()):
        return False
    if not np.array_equal(reference_df["y_true"].to_numpy(), candidate_df["y_true"].to_numpy()):
        return False
    for column in ("repeat_id", "outer_fold_id"):
        if column in reference_df.columns or column in candidate_df.columns:
            if column not in reference_df.columns or column not in candidate_df.columns:
                return False
            if not np.array_equal(reference_df[column].to_numpy(), candidate_df[column].to_numpy()):
                return False
    return np.array_equal(_get_cluster_ids(reference_df), _get_cluster_ids(candidate_df))


def _generate_bootstrap_indices(
    df: pd.DataFrame, n_bootstrap: int, stratified: bool, seed: int
) -> List[np.ndarray]:
    """Create clustered bootstrap index sets, optionally stratified by patient label."""

    rng = np.random.default_rng(seed)
    cluster_col = _get_cluster_column(df)

    if cluster_col == "caseid":
        logger.warning(
            "Predictions at %s do not include a patient identifier; falling back to case-level bootstrap.",
            cluster_col,
        )

    cluster_to_indices = {
        cluster_id: np.asarray(cluster_rows, dtype=int)
        for cluster_id, cluster_rows in df.groupby(cluster_col, sort=False).indices.items()
    }
    cluster_ids = np.array(list(cluster_to_indices.keys()), dtype=object)

    if not stratified:
        samples = []
        for _ in range(n_bootstrap):
            sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sampled_rows = np.concatenate([cluster_to_indices[cluster_id] for cluster_id in sampled_clusters])
            samples.append(sampled_rows)
        return samples

    cluster_labels = df.groupby(cluster_col, sort=False)["y_true"].max()
    pos_clusters = cluster_labels[cluster_labels == 1].index.to_numpy(dtype=object)
    neg_clusters = cluster_labels[cluster_labels == 0].index.to_numpy(dtype=object)

    if cluster_labels.nunique() > 1:
        mixed_clusters = df.groupby(cluster_col, sort=False)["y_true"].nunique()
        if (mixed_clusters > 1).any():
            logger.warning(
                "Mixed operation-level outcomes detected within patient clusters; stratified clustered bootstrap uses patient label=max(y_true)."
            )

    # If a class is missing, fall back to unstratified to avoid degenerate samples.
    if len(pos_clusters) == 0 or len(neg_clusters) == 0:
        logger.warning("Stratified clustered bootstrap requested but one patient class is missing; falling back to unstratified.")
        samples = []
        for _ in range(n_bootstrap):
            sampled_clusters = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sampled_rows = np.concatenate([cluster_to_indices[cluster_id] for cluster_id in sampled_clusters])
            samples.append(sampled_rows)
        return samples

    samples: List[np.ndarray] = []
    for _ in range(n_bootstrap):
        pos_sample = rng.choice(pos_clusters, size=len(pos_clusters), replace=True)
        neg_sample = rng.choice(neg_clusters, size=len(neg_clusters), replace=True)
        sampled_clusters = np.concatenate([pos_sample, neg_sample])
        samples.append(np.concatenate([cluster_to_indices[cluster_id] for cluster_id in sampled_clusters]))
    return samples


def _bootstrap_metrics(
    pred_set: PredictionSet,
    n_bootstrap: int,
    seed: int = 0,
    stratified: bool = False,
    indices_list: Optional[Sequence[np.ndarray]] = None,
) -> pd.DataFrame:
    df = pred_set.df
    y_true = df["y_true"].values
    y_prob = df["y_prob_calibrated"].values
    y_pred = df["y_pred_label"].astype(int).values

    if indices_list is None:
        indices_list = _generate_bootstrap_indices(df, n_bootstrap, stratified=stratified, seed=seed)

    records = []
    for i, sample_idx in enumerate(indices_list):
        y_true_sample = y_true[sample_idx]
        y_prob_sample = y_prob[sample_idx]
        y_pred_sample = y_pred[sample_idx]

        threshold_sample = _summarize_threshold(df.iloc[sample_idx]["threshold"].values)
        metrics = _compute_metrics_arrays(y_true_sample, y_prob_sample, y_pred_sample, threshold_sample)
        metrics.update(
            {
                "outcome": pred_set.outcome,
                "branch": pred_set.branch,
                "feature_set": pred_set.feature_set,
                "model_name": pred_set.model_name,
                "pipeline": pred_set.pipeline,
                "bootstrap_id": i,
            }
        )
        records.append(metrics)

    return pd.DataFrame(records)


def _run_bootstrap_jobs(
    tasks: List,
    *,
    n_jobs: int,
    parallel_backend: str,
    timeout: Optional[float],
    max_retries: int,
) -> List[pd.DataFrame]:
    """Retry ordered tasks; enforce deadlines only on cancellable process workers."""

    backend_plan: List[str] = []
    backend_plan.append(parallel_backend)
    if parallel_backend == "processes":
        backend_plan.append("threads")
    elif parallel_backend == "threads":
        backend_plan.append("processes")
    backend_plan.append("sequential")

    last_error: Optional[BaseException] = None

    for backend in backend_plan:
        jobs = 1 if backend == "sequential" else n_jobs
        deadline_requested = timeout is not None and timeout > 0
        deadline = timeout if deadline_requested and backend == "processes" and effective_n_jobs(jobs) > 1 else None
        if deadline_requested and deadline is None:
            logger.warning(
                "Bootstrap backend=%s with n_jobs=%s cannot enforce a hard task deadline; running without one.",
                backend, jobs,
            )
        attempt_limit = 1 if backend == "sequential" else max_retries
        for attempt in range(1, attempt_limit + 1):
            try:
                logger.info(
                    "Launching bootstrap jobs (backend=%s, attempt=%d/%d, n_jobs=%s, timeout=%s)",
                    backend,
                    attempt,
                    attempt_limit,
                    jobs,
                    deadline,
                )
                if backend == "sequential":
                    return Parallel(n_jobs=1)(tasks)
                backend_name = "loky" if backend == "processes" else "threading"
                return Parallel(n_jobs=jobs, backend=backend_name, timeout=deadline, batch_size=1)(tasks)
            except (TimeoutError, ProcessTimeoutError) as exc:
                last_error = exc
                logger.error(
                    "Bootstrap attempt %d/%d (backend=%s) exceeded its %s-second task deadline.",
                    attempt,
                    attempt_limit,
                    backend,
                    deadline,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.exception(
                    "Bootstrap attempt %d/%d (backend=%s) failed: %s.",
                    attempt,
                    attempt_limit,
                    backend,
                    exc,
                )
        if deadline is not None:
            # A thread/serial fallback cannot preserve a configured hard
            # deadline. Stop visibly after bounded process retries instead.
            break

    raise RuntimeError("Bootstrapping failed after allowed retries/backends.") from last_error


def _evaluation_hash(df: pd.DataFrame) -> str:
    columns = ["caseid", "y_true", _get_cluster_column(df)]
    columns.extend(c for c in ("repeat_id", "outer_fold_id") if c in df)
    return hashlib.sha256(pd.util.hash_pandas_object(df[columns], index=False).values.tobytes()).hexdigest()


def _prediction_content_hash(ps: PredictionSet) -> str:
    """Bind summary uncertainty to the exact saved predictions and model protocol."""
    frame = ps.df.sort_values("caseid", kind="stable").sort_index(axis=1).reset_index(drop=True)
    metadata_path = ps.path.parent.parent / "artifacts" / "metadata.json"
    fingerprint = json.loads(metadata_path.read_text()).get("validation_fingerprint", {}) if metadata_path.exists() else {}
    digest = hashlib.sha256(json.dumps({"columns": frame.columns.tolist(), "validation_fingerprint": fingerprint},
                                      sort_keys=True, separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=False).values.tobytes())
    return digest.hexdigest()


def _identity(ps: PredictionSet) -> dict:
    cluster_column = _get_cluster_column(ps.df)
    return {"n_patients": int(ps.df[cluster_column].nunique()) if cluster_column != "caseid" else float("nan"),
            "uncertainty_protocol_version": UNCERTAINTY_PROTOCOL_VERSION,
            "analysis_status": ps.analysis_status, "cohort_scope": ps.cohort_scope, "validation_scheme": ps.validation_scheme,
            "outcome": ps.outcome, "branch": ps.branch, "feature_set": ps.feature_set,
            "model_name": ps.model_name, "pipeline": ps.pipeline or "",
            "evaluation_case_hash": _evaluation_hash(ps.df),
            "prediction_content_hash": _prediction_content_hash(ps)}


def _bootstrap_interval(values, requested: int, *, label: str, warn_missing: bool = True):
    """Percentile interval and effective count, excluding only undefined metrics."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if warn_missing and (count < requested or count < 2):
        logger.warning(
            "Bootstrap interval %s has %d/%d finite replicates; undefined draws are "
            "excluded per metric and at least two are required to estimate an interval.",
            label, count, requested,
        )
    if count < 2:
        return float("nan"), float("nan"), count
    lower, upper = np.percentile(values, [2.5, 97.5])
    return float(lower), float(upper), count


def bootstrap_draw_hash(indices_list) -> str:
    """Identify the actual row draws shared by metrics and decision curves."""
    digest = hashlib.sha256()
    for indices in indices_list:
        values = np.asarray(indices, dtype="<i8")
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def summarize(
    prediction_sets: Iterable[PredictionSet], n_bootstrap: int = 5000,
    stratified_bootstrap: bool = False, bootstrap_seed: int = 0, n_jobs: int = -1,
    parallel_backend: str = "processes", delta_mode: str = "none",
    reference_feature_set: str = "preop_only", reference_model_name: Optional[str] = None,
    bootstrap_timeout: Optional[float] = 1800.0, bootstrap_max_retries: int = 2,
    strict: bool = False,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if n_bootstrap < 0:
        raise ValidationError("Bootstrap replicate count must be nonnegative")
    # Canonical case order is required before sharing patient-cluster draws.
    sets = [replace(ps, df=ps.df.sort_values("caseid", kind="stable").reset_index(drop=True))
            for ps in prediction_sets]
    if not sets:
        raise ValidationError("No prediction sets to summarize")
    for ps in sets:
        if ps.df["caseid"].duplicated().any():
            raise ValidationError(f"Duplicate caseids in {ps.path}")
        if strict and ("subjectid" not in ps.df or ps.df["subjectid"].isna().any()):
            raise ValidationError(f"Missing patient IDs in {ps.path}")
        if strict and delta_mode == "reference":
            if reference_model_name and _model_id(reference_model_name) != _model_id(ps.model_name):
                raise ValidationError("Strict reporting requires a same-family reference; cross-family overrides require --legacy")
            reference = [r for r in sets if (r.cohort_scope, r.validation_scheme, r.outcome, r.branch,
                         r.pipeline, r.model_name, r.feature_set) ==
                        (ps.cohort_scope, ps.validation_scheme, ps.outcome, ps.branch, ps.pipeline,
                         reference_model_name or ps.model_name, reference_feature_set)]
            if len(reference) != 1:
                raise ValidationError(f"Requires one same-family {reference_feature_set} reference for {ps.path}")
            if not _prediction_frames_are_aligned(reference[0].df, ps.df):
                raise ValidationError(f"Case/patient/truth/fold mismatch against reference for {ps.path}")
    identities = [{**_identity(ps), "bootstrap_replicates": int(n_bootstrap),
                   "bootstrap_seed": int(bootstrap_seed), "bootstrap_stratified": bool(stratified_bootstrap),
                   "bootstrap_draw_hash": "none",
                   "delta_mode": delta_mode,
                   "reference_feature_set": reference_feature_set if delta_mode == "reference" else "none",
                   "reference_model_name": (reference_model_name or ps.model_name) if delta_mode == "reference" else "none"}
                  for ps in sets]
    pairing_columns = ("cohort_scope", "validation_scheme", "outcome", "branch", "pipeline", "evaluation_case_hash")
    reference_hashes = {tuple(identity[c] for c in pairing_columns) + (identity["model_name"], identity["feature_set"]):
                        identity["prediction_content_hash"] for identity in identities}
    for identity in identities:
        reference_key = tuple(identity[c] for c in pairing_columns) + (identity["reference_model_name"], identity["reference_feature_set"])
        identity["reference_prediction_content_hash"] = reference_hashes.get(reference_key, "unavailable") if delta_mode == "reference" else "none"
    rows = [{**identity, **compute_point_metrics(ps)} for ps, identity in zip(sets, identities)]
    summary_df = pd.DataFrame(rows)
    if delta_mode == "reference":
        context = ["cohort_scope", "validation_scheme", "outcome", "branch", "pipeline", "evaluation_case_hash"]
        for idx, target in summary_df.iterrows():
            mask = pd.Series(True, index=summary_df.index)
            for col in context:
                mask &= summary_df[col].eq(target[col])
            reference = summary_df.loc[mask & summary_df.feature_set.eq(reference_feature_set)
                & summary_df.model_name.eq(reference_model_name or target.model_name)]
            if len(reference) == 1:
                for col in ("auprc", "auroc", "brier", "sensitivity", "specificity", "precision", "f1", "accuracy"):
                    summary_df.loc[idx, f"delta_{col}"] = target[col] - reference.iloc[0][col]
    if n_bootstrap <= 0:
        return summary_df, None
    # Process one population at a time instead of retaining every draw matrix.
    populations = defaultdict(list)
    for idx, (ps, identity) in enumerate(zip(sets, identities)):
        key = tuple(identity[c] for c in ("cohort_scope", "validation_scheme", "outcome", "branch", "pipeline", "evaluation_case_hash"))
        populations[key].append((idx, ps))
    frames = [None] * len(sets)
    for members in populations.values():
        indices = _generate_bootstrap_indices(members[0][1].df, n_bootstrap, stratified_bootstrap, bootstrap_seed)
        draw_hash = bootstrap_draw_hash(indices)
        tasks = []
        for idx, ps in members:
            identities[idx]["bootstrap_draw_hash"] = draw_hash
            summary_df.loc[idx, "bootstrap_draw_hash"] = draw_hash
            tasks.append(delayed(_bootstrap_metrics)(ps, n_bootstrap, seed=bootstrap_seed,
                         stratified=stratified_bootstrap, indices_list=indices))
        group_frames = _run_bootstrap_jobs(tasks, n_jobs=n_jobs, parallel_backend=parallel_backend,
                                          timeout=bootstrap_timeout, max_retries=max(1, bootstrap_max_retries))
        for (idx, _), frame in zip(members, group_frames):
            frames[idx] = frame
    for frame, identity in zip(frames, identities):
        if not frame.empty:
            for col, value in identity.items():
                frame[col] = value
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        logger.warning("No bootstrap replicates were produced; confidence intervals cannot be estimated.")
        for metric in (c for c in rows[0] if c not in identities[0]):
            summary_df[f"{metric}_bootstrap_valid"] = 0
            summary_df[f"{metric}_lower"] = float("nan")
            summary_df[f"{metric}_upper"] = float("nan")
        return summary_df, None
    bootstrap_df = pd.concat(frames, ignore_index=True)
    metric_cols = [c for c in rows[0] if c not in identities[0]]
    identity_cols = list(identities[0])
    for idx, row in summary_df.iterrows():
        mask = pd.Series(True, index=bootstrap_df.index)
        for col in identity_cols:
            mask &= bootstrap_df[col].isna() if pd.isna(row[col]) else bootstrap_df[col].eq(row[col])
        samples = bootstrap_df.loc[mask]
        for metric in metric_cols:
            lower, upper, count = _bootstrap_interval(
                samples[metric], n_bootstrap,
                label=f"{row.model_name}/{row.feature_set}/{metric}", warn_missing=metric != "threshold",
            )
            summary_df.loc[idx, f"{metric}_bootstrap_valid"] = count
            summary_df.loc[idx, f"{metric}_lower"] = lower
            summary_df.loc[idx, f"{metric}_upper"] = upper
    if delta_mode == "reference":
        _attach_delta_cis(summary_df, bootstrap_df, metric_cols,
                          reference_feature_set=reference_feature_set,
                          reference_model_name=reference_model_name)
    return summary_df, bootstrap_df


def _attach_delta_cis(summary_df, bootstrap_df, metric_cols, *,
                      reference_feature_set, reference_model_name, paired_groups=None):
    """Paired target-minus-reference differences within one aligned population."""
    context = ["cohort_scope", "validation_scheme", "outcome", "branch", "pipeline", "evaluation_case_hash"]
    for idx, target in summary_df.iterrows():
        mask = pd.Series(True, index=summary_df.index)
        boot_mask = pd.Series(True, index=bootstrap_df.index)
        for col in context:
            mask &= summary_df[col].eq(target[col])
            boot_mask &= bootstrap_df[col].eq(target[col])
        reference_name = reference_model_name or target["model_name"]
        ref_rows = summary_df.loc[mask & summary_df["feature_set"].eq(reference_feature_set)
                                  & summary_df["model_name"].eq(reference_name)]
        if len(ref_rows) != 1:
            logger.warning("No aligned %s/%s reference for %s", reference_name, reference_feature_set, target["feature_set"])
            continue
        reference = ref_rows.iloc[0]
        pool = bootstrap_df.loc[boot_mask]
        target_boot = pool.loc[pool["feature_set"].eq(target["feature_set"]) & pool["model_name"].eq(target["model_name"])].set_index("bootstrap_id")
        ref_boot = pool.loc[pool["feature_set"].eq(reference_feature_set) & pool["model_name"].eq(reference_name)].set_index("bootstrap_id")
        for metric in metric_cols:
            lower, upper, count = _bootstrap_interval(
                target_boot[metric] - ref_boot[metric], int(target["bootstrap_replicates"]),
                label=f"{target.model_name}/{target.feature_set}/delta_{metric}", warn_missing=metric != "threshold",
            )
            summary_df.loc[idx, f"delta_{metric}_bootstrap_valid"] = count
            summary_df.loc[idx, f"delta_{metric}"] = target[metric] - reference[metric]
            summary_df.loc[idx, f"delta_{metric}_lower"] = lower
            summary_df.loc[idx, f"delta_{metric}_upper"] = upper


def write_outputs(summary_df: pd.DataFrame, bootstrap_df: Optional[pd.DataFrame], save_bootstrap: Optional[Path]) -> None:
    enforce_storage_policy(
        {
            "results_tables_dir": TABLES_DIR,
            "paper_tables_dir": PAPER_TABLES_DIR,
            **({"bootstrap_output": save_bootstrap} if save_bootstrap is not None else {}),
        }
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLES_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote summary metrics to %s", summary_path)

    # Surface the publication-ready copy under the paper directory
    try:
        PAPER_TABLES_DIR.mkdir(parents=True, exist_ok=True)
        paper_summary = PAPER_TABLES_DIR / "metrics_summary.csv"
        shutil.copy2(summary_path, paper_summary)
        logger.info("Copied summary metrics to %s", paper_summary)
    except Exception as exc:
        logger.warning("Unable to copy metrics summary to paper directory: %s", exc)
    else:
        from results_recreation.results_analysis import prepare_metrics_for_display

        display_df = prepare_metrics_for_display(summary_df)
        export_dataframe_bundle(
            PAPER_TABLES_DIR / "metrics_summary",
            title="Metrics Summary",
            dataframe=display_df,
            include_csv=False,
        )
        refresh_paper_bundle(PAPER_DIR)

    if save_bootstrap and bootstrap_df is not None:
        bootstrap_path = (
            save_bootstrap if save_bootstrap.suffix else save_bootstrap / "metrics_bootstrap.parquet"
        )
        bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_df.to_parquet(bootstrap_path, index=False)
        logger.info("Wrote bootstrap samples to %s", bootstrap_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics from prediction files.")
    parser.set_defaults(strict=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--strict", dest="strict", action="store_true", help="Require complete scoped patient-grouped run artifacts (default).")
    mode.add_argument("--legacy", dest="strict", action="store_false", help="Explicitly read historical unscoped prediction artifacts.")
    parser.add_argument("--run-manifest", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root results directory (default: results)")
    parser.add_argument(
        "--bootstrap", type=int, default=5000, help="Number of bootstrap iterations (default: 5000; 0 to skip)"
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0, help="Random seed for bootstrap resampling (default: 0)"
    )
    parser.add_argument(
        "--bootstrap-timeout",
        type=float,
        default=1800.0,
        help="Per-task deadline (seconds) for process bootstrap workers with n_jobs > 1 (default: 1800). "
        "Process retries fail visibly rather than falling back to uncancellable threads. "
        "Threads/single-worker execution has no hard deadline; set to 0 to disable deadlines and permit backend fallbacks.",
    )
    parser.add_argument(
        "--bootstrap-max-retries",
        type=int,
        default=2,
        help="Attempts per backend (default: 2); configured process deadlines stop after these attempts.",
    )
    parser.add_argument(
        "--bootstrap-stratified",
        action="store_true",
        default=False,
        help="Enable outcome-stratified bootstrap sampling (default: off)",
    )
    parser.add_argument(
        "--no-bootstrap-stratified",
        action="store_false",
        dest="bootstrap_stratified",
        help="Disable outcome-stratified bootstrap sampling.",
    )
    parser.add_argument(
        "--bootstrap-output",
        type=Path,
        default=None,
        help=(
            "Optional path to save bootstrap samples (parquet). If a directory, saves metrics_bootstrap.parquet inside."
        ),
    )
    parser.add_argument(
        "--delta-mode",
        choices=["none", "reference"],
        default="none",
        help="Delta CI mode: none (default) or reference-based comparisons.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for bootstrap computation (joblib, default: -1 for all cores).",
    )
    parser.add_argument(
        "--parallel-backend",
        choices=["threads", "processes"],
        default="processes",
        help="Joblib backend to use for bootstrapping (threads or processes; default: processes).",
    )
    parser.add_argument(
        "--reference-feature-set",
        type=str,
        default="preop_only",
        help="Feature set name to use as the reference model when computing delta CIs (default: preop_only).",
    )
    parser.add_argument(
        "--reference-model-name",
        type=str,
        default=None,
        help="Optional model_name to disambiguate the reference model when computing delta CIs.",
    )
    return parser.parse_args()


def main() -> None:
    global RESULTS_DIR, TABLES_DIR, PAPER_DIR, PAPER_TABLES_DIR
    args = parse_args()
    if args.results_dir != RESULTS_DIR:
        RESULTS_DIR = args.results_dir
        TABLES_DIR = RESULTS_DIR / "tables"
        if not os.getenv("PAPER_DIR"):
            if RESULTS_DIR.name in {"nested_cv", "holdout"}:
                PAPER_DIR = RESULTS_DIR.parents[2] / "paper" / RESULTS_DIR.parent.name / RESULTS_DIR.name
            else:
                PAPER_DIR = RESULTS_DIR.parent / "paper"
            PAPER_TABLES_DIR = PAPER_DIR / "tables"
    prediction_sets = crawl_predictions(args.results_dir, strict=args.strict, run_manifest=args.run_manifest, require_reference=args.delta_mode == "reference", reference_feature_set=args.reference_feature_set, reference_model_name=args.reference_model_name)
    summary_df, bootstrap_df = summarize(
        prediction_sets,
        n_bootstrap=args.bootstrap,
        stratified_bootstrap=args.bootstrap_stratified,
        bootstrap_seed=args.bootstrap_seed,
        delta_mode=args.delta_mode,
        reference_feature_set=args.reference_feature_set,
        reference_model_name=args.reference_model_name,
        bootstrap_timeout=args.bootstrap_timeout,
        bootstrap_max_retries=args.bootstrap_max_retries,
        n_jobs=args.n_jobs,
        parallel_backend=args.parallel_backend,
        strict=args.strict,
    )
    comparisons = None
    if args.strict:
        from results_recreation.comparisons import build_comparison_artifacts
        manifest_path = args.run_manifest or args.results_dir / "run_manifest.json"
        comparisons = build_comparison_artifacts(
            prediction_sets, summary_df, bootstrap_df, n_bootstrap=args.bootstrap,
            stratified_bootstrap=args.bootstrap_stratified, bootstrap_seed=args.bootstrap_seed,
            run_manifest=json.loads(manifest_path.read_text()), strict=True,
        )
    write_outputs(summary_df, bootstrap_df, args.bootstrap_output)
    if comparisons is not None:
        from results_recreation.comparisons import write_comparison_artifacts
        write_comparison_artifacts(comparisons, TABLES_DIR, paper_tables_dir=PAPER_TABLES_DIR)


if __name__ == "__main__":
    main()
