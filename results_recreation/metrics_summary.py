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
import concurrent.futures
import logging
import sys
from collections import defaultdict
from functools import lru_cache
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from joblib import Parallel, delayed
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

from artifact_paths import enforce_storage_policy, get_paper_dir, get_results_dir
from data_preparation.inputs import COHORT_FILE, INPUT_FILE
from model_creation.prediction_io import REQUIRED_PREDICTION_COLUMNS, validate_prediction_dataframe
from reporting.manuscript_assets import export_dataframe_bundle, refresh_paper_bundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = get_results_dir(PROJECT_ROOT)
PAPER_DIR = get_paper_dir(PROJECT_ROOT)
TABLES_DIR = RESULTS_DIR / "tables"
PAPER_TABLES_DIR = PAPER_DIR / "tables"

ARTIFACT_FILES = ("calibration.json", "threshold.json")
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


def _load_predictions(pred_path: Path) -> PredictionSet:
    df = pd.read_csv(pred_path)
    threshold = _validate_dataframe(df, pred_path)
    _validate_artifacts(pred_path)
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
    )


def crawl_predictions(results_dir: Path) -> List[PredictionSet]:
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

    if not pred_files:
        raise FileNotFoundError(
            "No test prediction files found. Expected them under 'results/models/**/predictions/test.csv' "
            "or 'results/aeon/models/**/predictions/test.csv'."
        )

    prediction_sets: List[PredictionSet] = []
    for pred_file in pred_files:
        logger.info("Loading predictions from %s", pred_file)
        try:
            prediction_sets.append(_load_predictions(pred_file))
        except ValidationError as exc:
            logger.warning("Skipping invalid prediction file %s: %s", pred_file, exc)

    if not prediction_sets:
        raise FileNotFoundError(
            "No valid test prediction files found. Expected them under 'results/models/**/predictions/test.csv' "
            "or 'results/aeon/models/**/predictions/test.csv'."
        )

    return prediction_sets


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

    metrics = {
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "threshold": threshold,
        "auroc": _safe_metric(roc_auc_score, y_true, y_prob),
        "auprc": _safe_metric(average_precision_score, y_true, y_prob),
        "brier": _safe_metric(brier_score_loss, y_true, y_prob),
        "accuracy": _safe_metric(accuracy_score, y_true, y_pred),
        "precision": _safe_metric(precision_score, y_true, y_pred, zero_division=0),
        "sensitivity": _safe_metric(recall_score, y_true, y_pred),
    }

    tn, fp, fn, tp = (0, 0, 0, 0)
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        # Confusion matrix not defined (e.g., single class). Leave defaults.
        pass

    metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    metrics["f1"] = _safe_metric(f1_score, y_true, y_pred)

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
    stratified: bool = True,
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

        # Skip replicate if only one class is present after sampling.
        if len(np.unique(y_true_sample)) < 2:
            continue

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


def _run_with_timeout(func, timeout: Optional[float], desc: str):
    """Run a callable with an optional wall-clock timeout."""
    if timeout is None or timeout <= 0:
        return func()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            raise TimeoutError(f"{desc} timed out after {timeout} seconds.") from exc


def _run_bootstrap_jobs(
    tasks: List,
    *,
    n_jobs: int,
    parallel_backend: str,
    timeout: Optional[float],
    max_retries: int,
) -> List[pd.DataFrame]:
    """Execute bootstrap tasks with retries, backend fallbacks, and timeout safeguards."""

    backend_plan: List[str] = []
    backend_plan.append(parallel_backend)
    if parallel_backend == "processes":
        backend_plan.append("threads")
    elif parallel_backend == "threads":
        backend_plan.append("processes")
    backend_plan.append("sequential")

    last_error: Optional[BaseException] = None

    for backend in backend_plan:
        attempt_limit = 1 if backend == "sequential" else max_retries
        for attempt in range(1, attempt_limit + 1):
            try:
                logger.info(
                    "Launching bootstrap jobs (backend=%s, attempt=%d/%d, n_jobs=%s, timeout=%s)",
                    backend,
                    attempt,
                    attempt_limit,
                    n_jobs if backend != "sequential" else 1,
                    timeout if timeout and timeout > 0 else None,
                )

                def runner():
                    if backend == "sequential":
                        return [task() for task in tasks]
                    backend_name = "loky" if backend == "processes" else "threading"
                    return Parallel(n_jobs=n_jobs, backend=backend_name)(tasks)

                effective_timeout = timeout if timeout and timeout > 0 else None
                return _run_with_timeout(runner, effective_timeout, f"Bootstrap ({backend})")
            except TimeoutError as exc:
                last_error = exc
                logger.error(
                    "Bootstrap attempt %d/%d (backend=%s) timed out after %s seconds. Falling back if available.",
                    attempt,
                    attempt_limit,
                    backend,
                    timeout if timeout and timeout > 0 else "unspecified",
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.exception(
                    "Bootstrap attempt %d/%d (backend=%s) failed: %s. Falling back if available.",
                    attempt,
                    attempt_limit,
                    backend,
                    exc,
                )

    raise RuntimeError("Bootstrapping failed after retries and fallbacks.") from last_error


def summarize(
    prediction_sets: Iterable[PredictionSet],
    n_bootstrap: int = 1000,
    stratified_bootstrap: bool = True,
    bootstrap_seed: int = 0,
    n_jobs: int = -1,
    parallel_backend: str = "processes",
    delta_mode: str = "none",
    reference_feature_set: str = "preop_only",
    reference_model_name: Optional[str] = None,
    bootstrap_timeout: Optional[float] = 1800.0,
    bootstrap_max_retries: int = 2,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    summary_rows: List[Dict[str, object]] = []
    bootstrap_frames: List[pd.DataFrame] = []

    prediction_sets = list(prediction_sets)
    logger.info(
        "Summarizing %d prediction files (bootstrap=%d, stratified=%s, n_jobs=%s)",
        len(prediction_sets),
        n_bootstrap,
        stratified_bootstrap,
        n_jobs,
    )

    pairing_groups: Dict[Tuple[str, str, Optional[str]], List[PredictionSet]] = defaultdict(list)
    for pred_set in prediction_sets:
        grouping_key = (pred_set.outcome, pred_set.branch, pred_set.pipeline)
        pairing_groups[grouping_key].append(pred_set)

    shared_indices: Dict[Tuple[str, str, Optional[str]], Optional[List[np.ndarray]]] = {}
    paired_groups: set = set()
    if n_bootstrap > 0:
        for key, group_sets in pairing_groups.items():
            first = group_sets[0]
            base_len = len(first.df)
            if any(len(ps.df) != base_len for ps in group_sets):
                logger.warning(
                    "Grouping %s has mismatched prediction lengths; skipping paired bootstrap for this group.",
                    key,
                )
                shared_indices[key] = None
                continue

            if any(not _prediction_frames_are_aligned(first.df, ps.df) for ps in group_sets[1:]):
                logger.warning(
                    "Grouping %s has mismatched case order, outcomes, or patient IDs; skipping paired bootstrap for this group.",
                    key,
                )
                shared_indices[key] = None
                continue

            shared_indices[key] = _generate_bootstrap_indices(
                first.df,
                n_bootstrap,
                stratified=stratified_bootstrap,
                seed=bootstrap_seed,
            )
            paired_groups.add(key)
            logger.info(
                "Prepared bootstrap indices for group %s (models=%d, cases=%d, reps=%d)",
                key,
                len(group_sets),
                base_len,
                n_bootstrap,
            )

    for pred_set in prediction_sets:
        metrics = compute_point_metrics(pred_set)
        row: Dict[str, object] = {
            "outcome": pred_set.outcome,
            "branch": pred_set.branch,
            "feature_set": pred_set.feature_set,
            "model_name": pred_set.model_name,
            "pipeline": pred_set.pipeline,
        }
        for key, value in metrics.items():
            row[key] = value
        summary_rows.append(row)

    if n_bootstrap > 0:
        tasks = []
        for pred_set in prediction_sets:
            grouping_key = (pred_set.outcome, pred_set.branch, pred_set.pipeline)
            indices_list = shared_indices.get(grouping_key)
            tasks.append(
                delayed(_bootstrap_metrics)(
                    pred_set,
                    n_bootstrap,
                    seed=bootstrap_seed,
                    stratified=stratified_bootstrap,
                    indices_list=indices_list,
                )
            )

        logger.info("Prepared %d bootstrap tasks (backend preference=%s)", len(tasks), parallel_backend)
        bootstrap_frames = _run_bootstrap_jobs(
            tasks,
            n_jobs=n_jobs,
            parallel_backend=parallel_backend,
            timeout=bootstrap_timeout,
            max_retries=max(1, bootstrap_max_retries),
        )

    summary_df = pd.DataFrame(summary_rows)

    bootstrap_df = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else None
    if bootstrap_df is not None:
        ci_cols = [
            c
            for c in bootstrap_df.columns
            if c
            not in {
                "bootstrap_id",
                "outcome",
                "branch",
                "feature_set",
                "model_name",
                "pipeline",
            }
        ]
        ci_map: Dict[
            str, Dict[Tuple[str, str, str, str, Optional[str]], Tuple[float, float]]
        ] = {}
        grouped = bootstrap_df.groupby(
            ["outcome", "branch", "feature_set", "model_name", "pipeline"], dropna=False
        )
        for key, group in grouped:
            for col in ci_cols:
                vals = group[col].dropna()
                if vals.empty:
                    continue
                lower, upper = np.percentile(vals, [2.5, 97.5])
                ci_map.setdefault(col, {})[key] = (float(lower), float(upper))

        # attach CIs to summary
        for idx, row in summary_df.iterrows():
            key = (
                row["outcome"],
                row["branch"],
                row["feature_set"],
                row["model_name"],
                row.get("pipeline"),
            )
            for col, mapping in ci_map.items():
                if key in mapping:
                    lower, upper = mapping[key]
                    summary_df.loc[idx, f"{col}_lower"] = lower
                    summary_df.loc[idx, f"{col}_upper"] = upper

        if delta_mode == "reference":
            _attach_delta_cis(
                summary_df,
                bootstrap_df,
                ci_cols,
                paired_groups=paired_groups,
                reference_feature_set=reference_feature_set,
                reference_model_name=reference_model_name,
            )

    return summary_df, bootstrap_df


def _attach_delta_cis(
    summary_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    paired_groups: set,
    reference_feature_set: str,
    reference_model_name: Optional[str],
) -> None:
    """Compute delta CIs (target - reference) within each outcome/branch/pipeline group."""

    grouping_cols = ["outcome", "branch", "pipeline"]
    bootstrap_grouped = bootstrap_df.groupby(grouping_cols, dropna=False)

    for group_key, group_df in bootstrap_grouped:
        if group_key not in paired_groups:
            logger.warning("Skipping delta CIs for unpaired group %s.", group_key)
            continue

        outcome, branch, pipeline = group_key
        # Locate reference rows for this group.
        ref_candidates = group_df[group_df["feature_set"] == reference_feature_set]
        if ref_candidates.empty:
            logger.warning(
                "No reference rows found for group %s with feature_set=%s%s; skipping delta CIs.",
                group_key,
                reference_feature_set,
                f", model_name={reference_model_name}" if reference_model_name else "",
            )
            continue

        def select_reference(target_model_name: str):
            """Pick the reference matching the target model when possible."""
            preferred_models = []
            unique_models = ref_candidates["model_name"].dropna().unique()
            if target_model_name in unique_models:
                preferred_models.append(target_model_name)
            if reference_model_name and reference_model_name not in preferred_models:
                preferred_models.append(reference_model_name)
            # Deterministic fallback order for any remaining candidates.
            for name in sorted(unique_models):
                if name not in preferred_models:
                    preferred_models.append(name)

            for model_name in preferred_models:
                ref_bootstrap = ref_candidates[ref_candidates["model_name"] == model_name]
                if ref_bootstrap.empty:
                    continue
                ref_summary_row = summary_df[
                    (summary_df["outcome"] == outcome)
                    & (summary_df["branch"] == branch)
                    & (summary_df["pipeline"] == pipeline)
                    & (summary_df["feature_set"] == reference_feature_set)
                    & (summary_df["model_name"] == model_name)
                ]
                ref_point = ref_summary_row.iloc[0] if not ref_summary_row.empty else None
                return model_name, ref_bootstrap.set_index("bootstrap_id"), ref_point

            logger.warning(
                "Reference selection failed for group %s (target model=%s, feature_set=%s).",
                group_key,
                target_model_name,
                reference_feature_set,
            )
            return None

        # Iterate target models within the same grouping (including other feature sets).
        target_rows = summary_df[
            (summary_df["outcome"] == outcome)
            & (summary_df["branch"] == branch)
            & (summary_df["pipeline"] == pipeline)
        ]

        for idx, target_row in target_rows.iterrows():
            ref_selection = select_reference(target_row["model_name"])
            if ref_selection is None:
                continue

            ref_model_name, ref_lookup, ref_point = ref_selection

            if target_row["feature_set"] == reference_feature_set and target_row["model_name"] == ref_model_name:
                # Reference row: deltas are zero by definition.
                for col in metric_cols:
                    summary_df.loc[idx, f"delta_{col}"] = 0.0
                    summary_df.loc[idx, f"delta_{col}_lower"] = 0.0
                    summary_df.loc[idx, f"delta_{col}_upper"] = 0.0
                continue

            target_bootstrap = group_df[
                (group_df["feature_set"] == target_row["feature_set"])
                & (group_df["model_name"] == target_row["model_name"])
            ].set_index("bootstrap_id")

            # Align on bootstrap_id; require overlapping indices.
            joined_ids = target_bootstrap.index.intersection(ref_lookup.index)
            if joined_ids.empty:
                logger.warning(
                    "No overlapping bootstrap samples for delta comparison in group %s between %s/%s and reference %s/%s.",
                    group_key,
                    target_row["feature_set"],
                    target_row["model_name"],
                    reference_feature_set,
                    ref_model_name,
                )
                continue

            for col in metric_cols:
                if col not in target_bootstrap.columns or col not in ref_lookup.columns:
                    continue
                deltas = target_bootstrap.loc[joined_ids, col] - ref_lookup.loc[joined_ids, col]
                deltas = deltas.dropna()
                if deltas.empty:
                    continue
                lower, upper = np.percentile(deltas, [2.5, 97.5])
                ref_point_val = ref_point.get(col, float("nan")) if ref_point is not None else float("nan")
                point_delta = target_row.get(col, float("nan")) - ref_point_val
                summary_df.loc[idx, f"delta_{col}"] = point_delta
                summary_df.loc[idx, f"delta_{col}_lower"] = float(lower)
                summary_df.loc[idx, f"delta_{col}_upper"] = float(upper)


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
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Root results directory (default: results)")
    parser.add_argument(
        "--bootstrap", type=int, default=1000, help="Number of bootstrap iterations (default: 1000; 0 to skip)"
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=0, help="Random seed for bootstrap resampling (default: 0)"
    )
    parser.add_argument(
        "--bootstrap-timeout",
        type=float,
        default=1800.0,
        help="Wall-clock timeout (seconds) for executing all bootstrap tasks per backend (default: 1800). "
        "Set to 0 to disable timeouts.",
    )
    parser.add_argument(
        "--bootstrap-max-retries",
        type=int,
        default=2,
        help="Number of retry attempts per backend before falling back (default: 2).",
    )
    parser.add_argument(
        "--bootstrap-stratified",
        action="store_true",
        default=True,
        help="Enable outcome-stratified bootstrap sampling (default: on)",
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
    args = parse_args()
    prediction_sets = crawl_predictions(args.results_dir)
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
    )
    write_outputs(summary_df, bootstrap_df, args.bootstrap_output)


if __name__ == "__main__":
    main()
