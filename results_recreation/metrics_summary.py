"""Aggregate prediction metrics for held-out test sets.

This script crawls `results/models/**/predictions/test.csv` and
`results/aeon/models/**/predictions/test.csv`, validates required artifacts and
columns, computes metrics using stored thresholds, and writes consolidated
outputs to `results/tables/metrics_summary.csv`. Optional bootstrap samples can
also be saved to Parquet for further analysis.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

from model_creation.prediction_io import REQUIRED_PREDICTION_COLUMNS, validate_prediction_dataframe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

RESULTS_DIR = Path("results")
TABLES_DIR = RESULTS_DIR / "tables"

ARTIFACT_FILES = ("calibration.json", "threshold.json")


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
        return validate_prediction_dataframe(df, path, REQUIRED_PREDICTION_COLUMNS)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc


def _assert_single_value(df: pd.DataFrame, col: str, path: Path) -> str:
    values = df[col].dropna().unique()
    if len(values) != 1:
        raise ValidationError(f"{path} expected a single value for '{col}', found {values}")
    return str(values[0])


def _load_predictions(pred_path: Path) -> PredictionSet:
    df = pd.read_csv(pred_path)
    threshold = _validate_dataframe(df, pred_path)
    _validate_artifacts(pred_path)

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
        results_dir / "aeon" / "models" / "**" / "predictions" / "test.csv",
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
    y_pred = (y_prob >= pred_set.threshold).astype(int)

    metrics = {
        "n": len(y_true),
        "prevalence": float(np.mean(y_true)),
        "threshold": pred_set.threshold,
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


def _bootstrap_metrics(pred_set: PredictionSet, n_bootstrap: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pred_set.df
    y_true = df["y_true"].values
    y_prob = df["y_prob_calibrated"].values
    indices = np.arange(len(df))

    records = []
    for i in range(n_bootstrap):
        sample_idx = rng.choice(indices, size=len(indices), replace=True)
        sample_df = pd.DataFrame(
            {
                "y_true": y_true[sample_idx],
                "y_prob_calibrated": y_prob[sample_idx],
            }
        )
        sample_set = PredictionSet(
            path=pred_set.path,
            df=sample_df,
            threshold=pred_set.threshold,
            outcome=pred_set.outcome,
            branch=pred_set.branch,
            feature_set=pred_set.feature_set,
            model_name=pred_set.model_name,
            pipeline=pred_set.pipeline,
        )
        metrics = compute_point_metrics(sample_set)
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


def summarize(
    prediction_sets: Iterable[PredictionSet], n_bootstrap: int = 1000
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    summary_rows: List[Dict[str, object]] = []
    bootstrap_frames: List[pd.DataFrame] = []

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
            bootstrap_frames.append(_bootstrap_metrics(pred_set, n_bootstrap))

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

    return summary_df, bootstrap_df


def write_outputs(summary_df: pd.DataFrame, bootstrap_df: Optional[pd.DataFrame], save_bootstrap: Optional[Path]) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = TABLES_DIR / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info("Wrote summary metrics to %s", summary_path)

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
        "--bootstrap-output",
        type=Path,
        default=None,
        help=(
            "Optional path to save bootstrap samples (parquet). If a directory, saves metrics_bootstrap.parquet inside."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prediction_sets = crawl_predictions(args.results_dir)
    summary_df, bootstrap_df = summarize(prediction_sets, n_bootstrap=args.bootstrap)
    write_outputs(summary_df, bootstrap_df, args.bootstrap_output)


if __name__ == "__main__":
    main()
